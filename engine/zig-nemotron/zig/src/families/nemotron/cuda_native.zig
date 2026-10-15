//! Nemotron's CUDA engine, MTP head and lane backend behind the native family interface.
const std = @import("std");
const cuda = @import("cuda");
const lanes = @import("lanes");
const engine = @import("cuda_engine.zig");
const state = @import("cuda_state.zig");
const Head = @import("cuda_mtp.zig").Head;
const Lanes = @import("cuda_lanes.zig").Cuda;
const lone = @import("cuda_lone.zig");
const reuse = @import("cuda_reuse.zig");
const vstore = @import("vision_store.zig");
const mirror = @import("cuda_mirror.zig");

pub const snap_bytes = reuse.bytes;
pub const snap_save = reuse.save;
pub const snap_restore = reuse.restore;
pub const snap_drop = reuse.drop;
pub const snap_ready = reuse.ready; // the spare Mamba buffers it sets aside come out of the cache's budget
const core = @import("core");

/// Sliding Weights' learner (--slide): lessons change the weights this engine's lanes decode with.
pub const slide = @import("cuda_slide.zig");

pub const model_type = "nemotron_h";
pub const formats: []const []const u8 = &.{ "mlx-q4g64", "modelopt-nvfp4" };
pub const default_context: i64 = engine.default_context;
pub const max_segments: u32 = @import("cuda_segments.zig").MAX;
pub const prompt_rows: u32 = state.pass_rows;

/// What the server calls to turn an image into the rows its virtual ids stand for (see vision_store.zig).
pub const VisionHook = struct {
    ctx: *anyopaque,
    embed: *const fn (ctx: *anyopaque, img: *const vstore.HeldImage) anyerror!void,
};

pub const Options = struct { context: usize, drafts: bool, segments: usize = 1, slide: bool = false };

/// Thinking budgets the lone driver takes (cuda_lone.zig): above it, the stream goes to the lane core before its cut.
pub const lone_min_budget = @import("cuda_lone.zig").min_budget;

/// Nanoseconds a reply's background pass waits for the next request first: its first chunk holds the GPU.
pub const warm_quiet: u64 = 150 * std.time.ns_per_ms;

/// A lone drafted stream's own driver: decodes it until it finishes (false) or `yield` hands it over (true).
pub const LoneRun = *const fn (ctx: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool;

/// What the native server drives: the lane backend, the facts its round loop reads, and how to free it.
pub const Loaded = struct {
    backend: lanes.backend.Backend,
    facts: lanes.Model,
    rows: u32,
    /// Device bytes each admitted stream allocates for its own sequence (caches, state, head caches).
    stream_bytes: usize,
    /// Streams that allocate nothing: the first takes the engine's own sequence, whose buffers the graphs hold.
    free_streams: u32 = 1,
    ctx: *anyopaque,
    deinit: *const fn (*anyopaque) void,
    lone: ?LoneRun = null, // called with `ctx`; null: every stream in the lane core
    target: *reuse.Target, // the prompt cache copies this engine (cuda_reuse.zig)
    vision: ?VisionHook = null, // a multimodal checkpoint's image path (TF_NEMOTRON_VISION=0 turns it off)
    learns: ?*engine.Engine = null, // --slide: the engine its learner trains, whose sites it attached at load
    learn_bytes: usize = 0, // --slide: what its trainer allocates at the first lesson, kept out of the streams' room
    tp: bool = false, // tensor parallelism's rank 0: no prompt cache (a restored state would exist on this rank alone)
};

/// The first shard, index or config.json in `dir` that a write in place would reach through a link, or null.
pub fn linked(arena: std.mem.Allocator, io: std.Io, dir: []const u8) !?[]const u8 {
    return core.shard_edit.linked(arena, io, dir);
}

/// A shared expert's projection config.json keeps unquantized, as an earlier release's lessons left it, or null.
pub fn plainShared(arena: std.mem.Allocator, io: std.Io, dir: []const u8) !?[]const u8 {
    return core.shard_edit.plainShared(arena, io, dir);
}

const Owned = struct { gpa: std.mem.Allocator, e: *engine.Engine, head: ?*Head, lanes: Lanes, target: reuse.Target, carve: ?*cuda.Carveout, living: ?@import("cuda_sites.zig").Sites = null, store: ?*vstore.Store = null, vision: ?*vstore.Vision = null, mirror: ?*mirror.Mirror = null };

fn visionEmbed(p: *anyopaque, img: *const vstore.HeldImage) anyerror!void {
    const own: *Owned = @ptrCast(@alignCast(p));
    return own.vision.?.embed(img);
}

/// Own-sequence graphs support both draw modes; drafting also captures the head.
pub fn open(gpa: std.mem.Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, kernels: ?[]const u8, o: Options) !Loaded {
    const carve = try openCarveout(gpa, ctx);
    errdefer if (carve) |c| closeCarveout(gpa, c);
    // the image-row table first: the engine's graphs capture its address
    const vision_on = !std.mem.eql(u8, env("TF_NEMOTRON_VISION") orelse "1", "0") and vstore.Vision.present(gpa, io, dir);
    const store: ?*vstore.Store = if (vision_on) try vstore.Store.init(gpa, io, ctx.d, (std.fmt.parseInt(u64, env("TF_NEMOTRON_VISION_STORE_MB") orelse "1024", 10) catch 1024) << 20) else null;
    errdefer if (store) |st| st.deinit();
    const e = try engine.Engine.init(gpa, io, ctx, dir, kernels, .{ .context = o.context, .mtp = o.drafts, .graphs = true, .sampling = null, .segments = o.segments, .carveout = carve, .slide = o.slide });
    errdefer e.deinit();
    const head: ?*Head = if (o.drafts) try Head.init(e) else null;
    errdefer if (head) |h| h.deinit();
    if (head) |h| try h.capture();
    if (!o.drafts) try e.captureWindows(); // the one-row window: Engine.init captures windows only when drafting
    // the sampled mode's sets: a placeholder rule now, each stream's own uploaded at its prefill
    try e.setSampling(.{ .seed = 0, .temperature = 1.0 });
    try e.captureWindows();
    if (head) |h| try h.capture();
    try e.setSampling(null);
    const own = try gpa.create(Owned);
    errdefer gpa.destroy(own);
    own.* = .{ .gpa = gpa, .e = e, .head = head, .lanes = try Lanes.init(gpa, e, head), .target = .{ .e = e, .head = head }, .carve = carve, .store = store };
    errdefer own.lanes.deinit();
    if (store) |st| own.vision = try vstore.Vision.init(gpa, io, ctx, dir, st);
    errdefer if (own.vision) |v| v.deinit();
    own.living = try slide.loadLiving(gpa, io, e, dir); // before the lanes' costs are measured: they decode through it
    errdefer if (own.living) |*s| s.deinit();
    try own.lanes.measure(io, dir);
    if (e.tp.on()) return tensorParallel(gpa, own, o, head != null);
    return .{
        .backend = own.lanes.backend(),
        .facts = own.lanes.facts(),
        .rows = if (head != null) state.max_rows else 1,
        .stream_bytes = e.seqBytes(),
        .ctx = own,
        .deinit = release,
        .target = &own.target,
        .vision = if (own.vision != null) .{ .ctx = own, .embed = visionEmbed } else null,
        .lone = if (head != null) loneRun else null,
        .learns = if (o.slide) e else null,
        .learn_bytes = if (o.slide) @import("cuda_train.zig").Trainer.memory(e.c) else 0,
    };
}

/// Tensor parallelism: rank 0 serves through the mirror (cuda_mirror.zig); every other rank follows its calls here
/// until rank 0 stops, then exits (it never serves). Images, lone drivers and the prompt cache stay off for now.
fn tensorParallel(gpa: std.mem.Allocator, own: *Owned, o: Options, drafting: bool) !Loaded {
    const e = own.e;
    const link = &e.tp.session.?.plan;
    if (e.tp.rank != 0) {
        std.log.info("nemotron tp: rank {d} follows rank 0's calls", .{e.tp.rank});
        const result = mirror.follow(gpa, own.lanes.backend(), link);
        release(own);
        result catch |err| {
            std.log.err("nemotron tp: rank following failed: {t}", .{err});
            std.process.exit(70);
        };
        std.process.exit(0);
    }
    const m = try gpa.create(mirror.Mirror);
    m.* = mirror.Mirror.init(gpa, own.lanes.backend(), link);
    own.mirror = m;
    _ = o;
    return .{
        .backend = m.backend(),
        .facts = own.lanes.facts(),
        .rows = if (drafting) state.max_rows else 1,
        .stream_bytes = e.seqBytes(),
        .ctx = own,
        .deinit = release,
        .target = &own.target,
        .vision = null,
        .lone = null,
        .tp = true,
    };
}

/// What a learner over the loaded engine `p` learns into; `dir` must outlive it.
pub fn learnModel(p: *anyopaque, dir: []const u8) slide.Model {
    const own: *Owned = @ptrCast(@alignCast(p));
    return .{ .e = own.e, .dir = dir, .living = &own.living };
}

fn loneRun(p: *anyopaque, s: *lanes.Stream, hooks: *anyopaque, committed: *const fn (*anyopaque) void, yield: *const fn (*anyopaque) bool, resumed: bool) anyerror!bool {
    const own: *Owned = @ptrCast(@alignCast(p));
    return lone.run(own.gpa, &own.lanes, s, .{ .ctx = hooks, .committed = committed, .yield = yield, .resumed = resumed });
}

/// A request this engine refuses, in words; null: none of its own.
pub fn explain(_: ?*anyopaque, err: anyerror) ?[]const u8 {
    return switch (err) {
        error.PromptTooLong => "the prompt and its reply exceed this server's context window: shorten it or lower max_tokens",
        error.OutOfDeviceMemory => "the GPU had no memory left for this request's caches: retry once another request ends",
        else => null,
    };
}

fn release(p: *anyopaque) void {
    const own: *Owned = @ptrCast(@alignCast(p));
    if (own.mirror) |m| {
        m.deinit(); // the other ranks stop following, then close with this one
        own.gpa.destroy(m);
    }
    own.lanes.deinit();
    if (own.living) |*s| s.deinit();
    if (own.head) |h| h.deinit();
    own.e.deinit();
    if (own.vision) |v| v.deinit();
    if (own.store) |st| st.deinit();
    if (own.carve) |c| closeCarveout(own.gpa, c);
    own.gpa.destroy(own);
}

/// TF_CUDA_CARVEOUT=1: the display carveout KV planes go to first (cuda/carveout.zig); null when it is off.
fn openCarveout(gpa: std.mem.Allocator, ctx: *const cuda.Context) !?*cuda.Carveout {
    const bytes = (try cuda.carveout.requested(false, env("TF_CUDA_CARVEOUT"), env("TF_CUDA_CARVEOUT_MIB"))) orelse return null;
    const card: [*:0]const u8 = std.c.getenv("TF_DRM_CARD") orelse cuda.carveout.default_card;
    const c = try gpa.create(cuda.Carveout);
    errdefer gpa.destroy(c);
    c.* = cuda.Carveout.open(ctx.d, card, bytes) catch |err| {
        std.log.err("display carveout of {d} MiB from {s}: {t} (needs nvidia_drm modeset=1, the card's device node and no display in use)", .{ bytes >> 20, card, err });
        return err;
    };
    std.log.info("display carveout: {d} MiB from {s}; KV planes go there first", .{ bytes >> 20, card });
    return c;
}

fn closeCarveout(gpa: std.mem.Allocator, c: *cuda.Carveout) void {
    c.close();
    gpa.destroy(c);
}

fn env(name: [:0]const u8) ?[]const u8 {
    return std.mem.span(std.c.getenv(name) orelse return null);
}
