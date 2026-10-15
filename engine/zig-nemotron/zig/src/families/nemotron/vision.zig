//! Nemotron 3.5 Super VL's vision path on CUDA (M3): the C-RADIO tower (CPE patch embedding + 10 cls / register rows,
//! 32 pre-norm blocks: LayerNorm, full attention hdim 80, GELU MLP, bias everywhere, layer scale = identity), the
//! projector's final LayerNorm, the 2 x 2 pixel shuffle, and mlp1 (RMSNorm, linear, ReLU^2, linear): an image's
//! normalized bf16 pixels [3, H, W] -> its embedding rows [H / 32 * W / 32, 4096], one row per `<image>` token.
//! Every product is cuBLASLt (fp32 accumulate, bias epilogue); the glue is kernels/cuda/nemotron_vision.cu.
//! The numerics gate is tools/nemotron/vision_ref.py (the checkpoint's own remote code on CPU).
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const lt_mod = cuda.cublaslt;
const launch = cuda.launch;

const log = std.log.scoped(.nemotron_vision);
const Allocator = std.mem.Allocator;
const Stream = cuda.Stream;

pub const Shape = struct {
    layers: u32 = 32,
    heads: u32 = 16,
    dim: u32 = 1280,
    inter: u32 = 5120,
    patch: u32 = 16,
    prefix: u32 = 10, // 3 cls + 7 register rows
    table: u32 = 128, // the CPE position table's side (cpe_max_size 2048 / 16)
    shuffle: u32 = 2,
    proj_hidden: u32 = 20480,
    out: u32 = 4096,
    ln_eps: f32 = 1e-6,
    rms_eps: f32 = 1e-5,
};

const R = "vision_model.radio_model.model.";
const epilogue_attr: c_int = 7; // CUBLASLT_MATMUL_DESC_EPILOGUE
const bias_pointer_attr: c_int = 8; // CUBLASLT_MATMUL_DESC_BIAS_POINTER
const epilogue_bias: u32 = 4; // CUBLASLT_EPILOGUE_BIAS
const pref_min_alignment_a: c_int = 5; // ..._MIN_ALIGNMENT_A_BYTES (B 6, C 7, D 8)
const workspace_bytes: usize = 32 << 20;

/// Largest power of two (up to 256) an address is a multiple of, as cuBLASLt's alignment preference wants.
fn alignment(p: u64) u32 {
    var a: u32 = 256;
    while (a > 1 and p % a != 0) a >>= 1;
    return a;
}

/// A stage's output for the gates: `name`, its device pointer and byte count (the stream is synchronized first).
pub const Tap = struct { ctx: *anyopaque, at: *const fn (ctx: *anyopaque, name: []const u8, ptr: u64, bytes: usize) anyerror!void };

pub const Tower = struct {
    gpa: Allocator,
    d: *const cuda.Driver,
    shape: Shape = .{},
    mod: cuda.Module,
    f_ln: cuda.Function,
    f_patchify: cuda.Function,
    f_embed: cuda.Function,
    f_attn: cuda.Function,
    f_gelu: cuda.Function,
    f_relu2: cuda.Function,
    f_add: cuda.Function,
    f_shuffle: cuda.Function,
    lt: lt_mod.Library,
    handle: lt_mod.Handle,
    workspace: cuda.DeviceBuffer,
    weights: std.StringHashMapUnmanaged(cuda.DeviceBuffer) = .empty,
    bytes: u64 = 0,
    scratch: ?cuda.DeviceBuffer = null,

    /// The tower, projector and position-table tensors of `dir` uploaded (bf16, as stored), kernels and cuBLASLt opened.
    pub fn load(gpa: Allocator, io: std.Io, d: *const cuda.Driver, dir: []const u8) !*Tower {
        if (!cuda.kernels.available) return error.BuiltWithoutKernels;
        const t = try gpa.create(Tower);
        errdefer gpa.destroy(t);
        var mod = try cuda.Module.load(d, cuda.kernels.nemotron_vision);
        errdefer mod.unload();
        var lt = try lt_mod.Library.open();
        errdefer lt.close();
        var handle: lt_mod.Handle = null;
        try lt.check(lt.api.cublasLtCreate(&handle), "cublasLtCreate");
        errdefer _ = lt.api.cublasLtDestroy(handle);
        var ws = try cuda.DeviceBuffer.alloc(d, workspace_bytes);
        errdefer ws.free();
        t.* = .{
            .gpa = gpa, .d = d, .mod = mod, .lt = lt, .handle = handle, .workspace = ws,
            .f_ln = try mod.function("tf_vis_layernorm"),
            .f_patchify = try mod.function("tf_vis_patchify"),
            .f_embed = try mod.function("tf_vis_embed"),
            .f_attn = try mod.function("tf_vis_attn"),
            .f_gelu = try mod.function("tf_vis_gelu"),
            .f_relu2 = try mod.function("tf_vis_relu2"),
            .f_add = try mod.function("tf_vis_add"),
            .f_shuffle = try mod.function("tf_vis_shuffle_rms"),
        };
        errdefer t.deinit();
        var ck = try core.Checkpoint.openModel(gpa, io, dir);
        defer ck.close();
        const sh = t.shape;
        try t.put(&ck, R ++ "patch_generator.embedder.weight", &.{ sh.dim, 3 * sh.patch * sh.patch });
        try t.put(&ck, R ++ "patch_generator.pos_embed", &.{ 1, sh.table * sh.table, sh.dim });
        try t.put(&ck, R ++ "patch_generator.cls_token.token", &.{ sh.prefix, sh.dim });
        var nb: [128]u8 = undefined;
        for (0..sh.layers) |i| {
            inline for (.{
                .{ "norm1.weight", 1, 0 }, .{ "norm1.bias", 1, 0 }, .{ "norm2.weight", 1, 0 }, .{ "norm2.bias", 1, 0 },
            }) |e| try t.put(&ck, try std.fmt.bufPrint(&nb, R ++ "blocks.{d}." ++ e[0], .{i}), &.{sh.dim});
            try t.put(&ck, try std.fmt.bufPrint(&nb, R ++ "blocks.{d}.attn.qkv.weight", .{i}), &.{ 3 * sh.dim, sh.dim });
            try t.put(&ck, try std.fmt.bufPrint(&nb, R ++ "blocks.{d}.attn.qkv.bias", .{i}), &.{3 * sh.dim});
            try t.put(&ck, try std.fmt.bufPrint(&nb, R ++ "blocks.{d}.attn.proj.weight", .{i}), &.{ sh.dim, sh.dim });
            try t.put(&ck, try std.fmt.bufPrint(&nb, R ++ "blocks.{d}.attn.proj.bias", .{i}), &.{sh.dim});
            try t.put(&ck, try std.fmt.bufPrint(&nb, R ++ "blocks.{d}.mlp.fc1.weight", .{i}), &.{ sh.inter, sh.dim });
            try t.put(&ck, try std.fmt.bufPrint(&nb, R ++ "blocks.{d}.mlp.fc1.bias", .{i}), &.{sh.inter});
            try t.put(&ck, try std.fmt.bufPrint(&nb, R ++ "blocks.{d}.mlp.fc2.weight", .{i}), &.{ sh.dim, sh.inter });
            try t.put(&ck, try std.fmt.bufPrint(&nb, R ++ "blocks.{d}.mlp.fc2.bias", .{i}), &.{sh.dim});
        }
        try t.put(&ck, "vision_projector.vision_final_layernorm.weight", &.{sh.dim});
        try t.put(&ck, "vision_projector.vision_final_layernorm.bias", &.{sh.dim});
        const d4 = 4 * sh.dim;
        try t.put(&ck, "mlp1.0.weight", &.{d4});
        try t.put(&ck, "mlp1.1.weight", &.{ sh.proj_hidden, d4 });
        try t.put(&ck, "mlp1.3.weight", &.{ sh.out, sh.proj_hidden });
        log.info("vision tower: {d} tensors, {d:.2} GB", .{ t.weights.count(), @as(f64, @floatFromInt(t.bytes)) / 1e9 });
        return t;
    }

    fn put(t: *Tower, ck: *core.Checkpoint, name: []const u8, shape: []const usize) !void {
        const tensor = try ck.expect(name, .bf16, shape);
        const dev = try cuda.DeviceBuffer.fromHost(t.d, tensor.bytes);
        errdefer {
            var dd = dev;
            dd.free();
        }
        try t.weights.put(t.gpa, try t.gpa.dupe(u8, name), dev);
        t.bytes += tensor.bytes.len;
    }

    pub fn deinit(t: *Tower) void {
        var it = t.weights.iterator();
        while (it.next()) |e| {
            t.gpa.free(e.key_ptr.*);
            e.value_ptr.free();
        }
        t.weights.deinit(t.gpa);
        if (t.scratch) |*s| s.free();
        t.workspace.free();
        _ = t.lt.api.cublasLtDestroy(t.handle);
        t.lt.close();
        t.mod.unload();
        t.gpa.destroy(t);
    }

    fn w(t: *const Tower, comptime fmt: []const u8, args: anytype) !u64 {
        var nb: [128]u8 = undefined;
        const name = try std.fmt.bufPrint(&nb, fmt, args);
        return (t.weights.get(name) orelse {
            log.err("missing vision weight {s}", .{name});
            return error.MissingVisionWeight;
        }).ptr;
    }

    fn grow(t: *Tower, bytes: usize) !u64 {
        if (t.scratch == null or t.scratch.?.len < bytes) {
            if (t.scratch) |*b| b.free();
            t.scratch = null;
            t.scratch = try cuda.DeviceBuffer.alloc(t.d, bytes);
        }
        return t.scratch.?.ptr;
    }

    /// F.linear(x [m, k], W [n, k], b) -> y [m, n] bf16 (cuBLASLt, fp32 accumulate, the bias in the epilogue).
    fn linear(t: *Tower, s: Stream, x: u64, m: usize, k: usize, wt: u64, bias: ?u64, n: usize, y: u64) !void {
        const api = t.lt.api;
        var desc: lt_mod.MatmulDesc = null;
        try t.lt.check(api.cublasLtMatmulDescCreate(&desc, lt_mod.compute_32f, .f32), "desc");
        defer _ = api.cublasLtMatmulDescDestroy(desc);
        const ta: c_int = @backingInt(lt_mod.Op.t);
        const tb: c_int = @backingInt(lt_mod.Op.n);
        try t.lt.check(api.cublasLtMatmulDescSetAttribute(desc, lt_mod.desc_transa, &ta, @sizeOf(c_int)), "transa");
        try t.lt.check(api.cublasLtMatmulDescSetAttribute(desc, lt_mod.desc_transb, &tb, @sizeOf(c_int)), "transb");
        if (bias) |b| {
            try t.lt.check(api.cublasLtMatmulDescSetAttribute(desc, epilogue_attr, &epilogue_bias, @sizeOf(u32)), "epilogue");
            try t.lt.check(api.cublasLtMatmulDescSetAttribute(desc, bias_pointer_attr, &b, @sizeOf(u64)), "bias");
        }
        var la: lt_mod.Layout = null;
        var lb: lt_mod.Layout = null;
        var lc: lt_mod.Layout = null;
        try t.lt.check(api.cublasLtMatrixLayoutCreate(&la, .bf16, k, n, @intCast(k)), "layout W");
        defer _ = api.cublasLtMatrixLayoutDestroy(la);
        try t.lt.check(api.cublasLtMatrixLayoutCreate(&lb, .bf16, k, m, @intCast(k)), "layout X");
        defer _ = api.cublasLtMatrixLayoutDestroy(lb);
        try t.lt.check(api.cublasLtMatrixLayoutCreate(&lc, .bf16, n, m, @intCast(n)), "layout Y");
        defer _ = api.cublasLtMatrixLayoutDestroy(lc);
        var pref: lt_mod.Preference = null;
        try t.lt.check(api.cublasLtMatmulPreferenceCreate(&pref), "pref");
        defer _ = api.cublasLtMatmulPreferenceDestroy(pref);
        const limit: u64 = workspace_bytes;
        try t.lt.check(api.cublasLtMatmulPreferenceSetAttribute(pref, lt_mod.pref_max_workspace_bytes, &limit, @sizeOf(u64)), "workspace");
        for ([_]u64{ wt, x, y, bias orelse y }, 0..) |p, i| {
            const al: u32 = alignment(p);
            try t.lt.check(api.cublasLtMatmulPreferenceSetAttribute(pref, pref_min_alignment_a + @as(c_int, @intCast(i)), &al, @sizeOf(u32)), "alignment");
        }
        var found: [1]lt_mod.Heuristic = undefined;
        var count: c_int = 0;
        try t.lt.check(api.cublasLtMatmulAlgoGetHeuristic(t.handle, desc, la, lb, lc, lc, pref, 1, &found, &count), "heuristic");
        if (count < 1 or found[0].state != 0) return error.NoAlgorithm;
        const one: f32 = 1;
        const zero: f32 = 0;
        try t.lt.check(api.cublasLtMatmul(t.handle, desc, &one, wt, la, x, lb, &zero, y, lc, y, lc, &found[0].algo, t.workspace.ptr, workspace_bytes, s.handle), "matmul");
    }

    fn ln(t: *Tower, s: Stream, x: u64, wt: u64, b: u64, out: u64, rows: usize, dim: usize) !void {
        var a: launch.Args = .{};
        for ([_]u64{ x, wt, b, out }) |v| a.add(v);
        a.add(@as(i32, @intCast(dim)));
        a.add(t.shape.ln_eps);
        try launch.launch(t.f_ln, .{ .grid = .{ .x = @intCast(rows), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }

    fn pointwise(t: *Tower, f: cuda.Function, s: Stream, x: u64, y: ?u64, n: usize) !void {
        _ = t;
        var a: launch.Args = .{};
        a.add(x);
        if (y) |v| a.add(v);
        a.add(@as(i64, @intCast(n)));
        try launch.launch(f, .{ .grid = .{ .x = @intCast((n + 255) / 256), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
    }

    /// Patch rows an image of `rows x cols` 16-pixel patches puts through the tower (before the 2 x 2 shuffle).
    pub fn tokens(rows: usize, cols: usize) usize {
        return rows * cols / 4;
    }

    /// `pixels`: bf16 [3, rows * 16, cols * 16] on the device, already normalized; `out`: bf16 [rows * cols / 4, 4096].
    pub fn encode(t: *Tower, s: Stream, pixels: u64, rows: usize, cols: usize, out: u64, tap: ?Tap) !void {
        const sh = t.shape;
        if (rows % 2 != 0 or cols % 2 != 0 or rows == 0 or cols == 0) return error.VisionGrid;
        const P = rows * cols;
        const n = P + sh.prefix;
        const D: usize = sh.dim;
        const I: usize = sh.inter;
        const T = P / 4;
        // scratch: x, h, o [n, D]; qkv [n, 3D]; m [n, I]; patches [P, 768] (reused for the projector's rows)
        const sizes = [_]usize{ n * D, n * D, n * D, n * 3 * D, n * I, P * 768 };
        var offs: [sizes.len]usize = undefined;
        var total: usize = 0;
        for (sizes, &offs) |z, *o| {
            o.* = total;
            total += std.mem.alignForward(usize, z * 2, 256);
        }
        const base = try t.grow(total);
        const x = base + offs[0];
        const h = base + offs[1];
        const o = base + offs[2];
        const qkv = base + offs[3];
        const m = base + offs[4];
        const patches = base + offs[5];
        // patch embedding: patchify -> GEMM (no bias) -> + position table, prefix rows first
        {
            var a: launch.Args = .{};
            a.add(pixels);
            a.add(patches);
            a.add(@as(i32, @intCast(rows)));
            a.add(@as(i32, @intCast(cols)));
            try launch.launch(t.f_patchify, .{ .grid = .{ .x = @intCast(P), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
        }
        try t.linear(s, patches, P, 768, try t.w(R ++ "patch_generator.embedder.weight", .{}), null, D, h);
        {
            var a: launch.Args = .{};
            a.add(h);
            a.add(try t.w(R ++ "patch_generator.pos_embed", .{}));
            a.add(try t.w(R ++ "patch_generator.cls_token.token", .{}));
            a.add(x);
            for ([_]usize{ rows, cols, D, sh.prefix, sh.table }) |v| a.add(@as(i32, @intCast(v)));
            try launch.launch(t.f_embed, .{ .grid = .{ .x = @intCast(n), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
        }
        if (tap) |tp| try tapped(tp, s, "embed", x, n * D * 2);
        const scale: f32 = 1.0 / @sqrt(@as(f32, @floatFromInt(D / sh.heads)));
        const scale_log2: f32 = @floatCast(@as(f64, scale) * std.math.log2e);
        for (0..sh.layers) |i| {
            try t.ln(s, x, try t.w(R ++ "blocks.{d}.norm1.weight", .{i}), try t.w(R ++ "blocks.{d}.norm1.bias", .{i}), h, n, D);
            try t.linear(s, h, n, D, try t.w(R ++ "blocks.{d}.attn.qkv.weight", .{i}), try t.w(R ++ "blocks.{d}.attn.qkv.bias", .{i}), 3 * D, qkv);
            {
                var a: launch.Args = .{};
                a.add(qkv);
                a.add(o);
                a.add(@as(i32, @intCast(n)));
                a.add(@as(i32, @intCast(sh.heads)));
                a.add(scale_log2);
                try launch.launch(t.f_attn, .{ .grid = .{ .x = @intCast((n + 127) / 128), .y = sh.heads, .z = 1 }, .block = .{ .x = 128 } }, s, &a);
            }
            try t.linear(s, o, n, D, try t.w(R ++ "blocks.{d}.attn.proj.weight", .{i}), try t.w(R ++ "blocks.{d}.attn.proj.bias", .{i}), D, h);
            try t.pointwise(t.f_add, s, x, h, n * D);
            try t.ln(s, x, try t.w(R ++ "blocks.{d}.norm2.weight", .{i}), try t.w(R ++ "blocks.{d}.norm2.bias", .{i}), h, n, D);
            try t.linear(s, h, n, D, try t.w(R ++ "blocks.{d}.mlp.fc1.weight", .{i}), try t.w(R ++ "blocks.{d}.mlp.fc1.bias", .{i}), I, m);
            try t.pointwise(t.f_gelu, s, m, null, n * I);
            try t.linear(s, m, n, I, try t.w(R ++ "blocks.{d}.mlp.fc2.weight", .{i}), try t.w(R ++ "blocks.{d}.mlp.fc2.bias", .{i}), D, h);
            try t.pointwise(t.f_add, s, x, h, n * D);
            if (tap) |tp| {
                var nb: [32]u8 = undefined;
                try tapped(tp, s, try std.fmt.bufPrint(&nb, "block{d:0>2}", .{i}), x, n * D * 2);
            }
        }
        // the features (past the prefix rows): final LayerNorm, shuffle + RMSNorm, mlp1
        const feats = x + sh.prefix * D * 2;
        try t.ln(s, feats, try t.w("vision_projector.vision_final_layernorm.weight", .{}), try t.w("vision_projector.vision_final_layernorm.bias", .{}), h, P, D);
        if (tap) |tp| try tapped(tp, s, "feat_ln", h, P * D * 2);
        {
            var a: launch.Args = .{};
            a.add(h);
            a.add(try t.w("mlp1.0.weight", .{}));
            a.add(o);
            a.add(@as(i32, @intCast(rows)));
            a.add(@as(i32, @intCast(cols)));
            a.add(@as(i32, @intCast(D)));
            a.add(sh.rms_eps);
            try launch.launch(t.f_shuffle, .{ .grid = .{ .x = @intCast(T), .y = 1, .z = 1 }, .block = .{ .x = 256 } }, s, &a);
        }
        if (tap) |tp| try tapped(tp, s, "shuffled", o, T * 4 * D * 2);
        try t.linear(s, o, T, 4 * D, try t.w("mlp1.1.weight", .{}), null, sh.proj_hidden, m);
        try t.pointwise(t.f_relu2, s, m, null, T * sh.proj_hidden);
        try t.linear(s, m, T, sh.proj_hidden, try t.w("mlp1.3.weight", .{}), null, sh.out, out);
    }
};

fn tapped(tp: Tap, s: Stream, name: []const u8, ptr: u64, bytes: usize) !void {
    try s.synchronize();
    try tp.at(tp.ctx, name, ptr, bytes);
}

fn ms(io: std.Io, since: std.Io.Timestamp) i64 {
    return @intCast(@divTrunc(std.Io.Clock.awake.now(io).toNanoseconds() - since.toNanoseconds(), std.time.ns_per_ms));
}

// ---- the `tensorfold vision` gate: a reference's pixels in, the stages out ---------------------------------------------

const Dumper = struct {
    gpa: Allocator,
    io: std.Io,
    dir: []const u8,
    d: *const cuda.Driver,

    fn at(ctx: *anyopaque, name: []const u8, ptr: u64, bytes: usize) anyerror!void {
        const self: *Dumper = @ptrCast(@alignCast(ctx));
        const host = try self.gpa.alloc(u8, bytes);
        defer self.gpa.free(host);
        try self.d.check(self.d.api.cuMemcpyDtoH_v2(host.ptr, ptr, bytes), "cuMemcpyDtoH");
        const path = try std.fmt.allocPrint(self.gpa, "{s}/{s}.bin", .{ self.dir, name });
        defer self.gpa.free(path);
        try std.Io.Dir.cwd().writeFile(self.io, .{ .sub_path = path, .data = host });
    }
};

/// `tensorfold vision MODEL PIXELS_BF16 ROWS COLS DUMP_DIR [--repeat N]`: runs the tower on a reference's bf16 pixels
/// [3, 16 ROWS, 16 COLS], dumps embed / blockNN / feat_ln / shuffled stages and `zig_out.bin`, and times a run.
pub fn check(gpa: Allocator, io: std.Io, ctx: *const cuda.Context, d: *const cuda.Driver, dir: []const u8, pixels_path: []const u8, rows: usize, cols: usize, dump: []const u8, repeat: usize) !u8 {
    _ = ctx;
    var t0 = std.Io.Clock.awake.now(io);
    const t = try Tower.load(gpa, io, d, dir);
    defer t.deinit();
    std.debug.print("loaded vision tower ({d:.2} GB) in {d} ms\n", .{ @as(f64, @floatFromInt(t.bytes)) / 1e9, ms(io, t0) });
    const raw = try std.Io.Dir.cwd().readFileAlloc(io, pixels_path, gpa, .limited(1 << 30));
    defer gpa.free(raw);
    if (raw.len != 3 * 256 * rows * cols * 2) return error.PixelsSize;
    var px = try cuda.DeviceBuffer.fromHost(d, raw);
    defer px.free();
    const T = Tower.tokens(rows, cols);
    var out = try cuda.DeviceBuffer.alloc(d, T * 4096 * 2);
    defer out.free();
    var s = try Stream.init(d, true);
    defer s.deinit();
    try std.Io.Dir.cwd().createDirPath(io, dump);
    var dumper: Dumper = .{ .gpa = gpa, .io = io, .dir = dump, .d = d };
    const tap: Tap = .{ .ctx = &dumper, .at = Dumper.at };
    // the first run dumps the stages for the gate (blocks 0, 15, 31 only: the files are big)
    var first = try TapFilter.init(tap);
    try t.encode(s, px.ptr, rows, cols, out.ptr, first.tap());
    try s.synchronize();
    try Dumper.at(&dumper, "zig_out", out.ptr, out.len);
    std.debug.print("wrote {d} stages to {s}\n", .{ first.kept, dump });
    for (0..repeat) |_| {
        t0 = std.Io.Clock.awake.now(io);
        try t.encode(s, px.ptr, rows, cols, out.ptr, null);
        try s.synchronize();
        std.debug.print("encode {d}x{d} patches ({d} tokens): {d} ms\n", .{ rows, cols, T, ms(io, t0) });
    }
    return 0;
}

/// Keeps only the stages worth a file: embed, blocks 0 / 15 / 31, feat_ln, shuffled.
const TapFilter = struct {
    inner: Tap,
    kept: usize = 0,

    fn init(inner: Tap) !TapFilter {
        return .{ .inner = inner };
    }

    fn tap(f: *TapFilter) Tap {
        return .{ .ctx = f, .at = at };
    }

    fn at(ctx: *anyopaque, name: []const u8, ptr: u64, bytes: usize) anyerror!void {
        const f: *TapFilter = @ptrCast(@alignCast(ctx));
        const skip = std.mem.startsWith(u8, name, "block") and !(std.mem.eql(u8, name, "block00") or std.mem.eql(u8, name, "block15") or std.mem.eql(u8, name, "block31"));
        if (skip) return;
        f.kept += 1;
        try f.inner.at(f.inner.ctx, name, ptr, bytes);
    }
};
