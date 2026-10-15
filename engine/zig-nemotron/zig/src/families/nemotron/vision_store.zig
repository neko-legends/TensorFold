//! The image rows a served model reads in place of token embeddings, and the vision engine that makes them.
//!
//! An image reaches the engine as a run of *virtual ids* (>= 2^24, derived from the image's digest: every token-keyed
//! cache then tells images apart). `tf_nemo_embed16` looks such an id up in `table`, an open-addressing hash on the
//! device (a fixed address: CUDA graphs capture it) whose slots hold the address of the id's row in a stored image's
//! embeddings. `Store` owns the rows (least recently used out first, up to a byte cap) and keeps the table in step;
//! `Vision` runs the tower for an image on its own stream (the server's request thread calls it) and stores the result.
const std = @import("std");
const cuda = @import("cuda");
const glue = @import("cuda_glue.zig");
const vision = @import("vision.zig");

const log = std.log.scoped(.nemotron_vision);
const Allocator = std.mem.Allocator;

/// A request's prepared image as the server hands it over (engine_api.HeldImage, the same bytes).
pub const HeldImage = extern struct {
    /// bf16 normalized pixels [3, 16 rows, 16 cols]
    pixels: [*]const u16,
    rows: u32,
    cols: u32,
    digest: [32]u8,
    /// the image's virtual ids, rows * cols / 4 of them
    vids: [*]const u32,
    n_vids: u64,
};

pub const Slot = extern struct { key: u32, pad: u32 = 0, row: u64 };
comptime {
    std.debug.assert(@sizeOf(Slot) == 16);
}

pub const slot_count: u32 = 1 << 19; // 8 MiB: half full at 262,144 rows (8 KiB each: 2 GiB of images)
const empty: u32 = 0xFFFFFFFF;
const row_bytes: usize = 4096 * 2;

const Entry = struct { buf: cuda.DeviceBuffer, vids: []u32, tick: u64 };

pub const Store = struct {
    gpa: Allocator,
    io: std.Io,
    d: *const cuda.Driver,
    mutex: std.Io.Mutex = .init,
    table: cuda.DeviceBuffer,
    mirror: []Slot,
    entries: std.AutoArrayHashMapUnmanaged([32]u8, Entry) = .empty,
    bytes: u64 = 0,
    cap: u64,
    tick: u64 = 0,

    /// Allocates the table and publishes it to the embedding kernel: call before the engine captures a graph.
    pub fn init(gpa: Allocator, io: std.Io, d: *const cuda.Driver, cap_bytes: u64) !*Store {
        const s = try gpa.create(Store);
        errdefer gpa.destroy(s);
        const mirror = try gpa.alloc(Slot, slot_count);
        errdefer gpa.free(mirror);
        @memset(mirror, .{ .key = empty, .row = 0 });
        var table = try cuda.DeviceBuffer.alloc(d, slot_count * @sizeOf(Slot));
        errdefer table.free();
        try table.upload(0, std.mem.sliceAsBytes(mirror));
        s.* = .{ .gpa = gpa, .io = io, .d = d, .table = table, .mirror = mirror, .cap = cap_bytes };
        glue.vid_table = table.ptr;
        glue.vid_mask = slot_count - 1;
        return s;
    }

    pub fn deinit(s: *Store) void {
        glue.vid_table = 0;
        glue.vid_mask = 0;
        for (s.entries.values()) |*e| {
            e.buf.free();
            s.gpa.free(e.vids);
        }
        s.entries.deinit(s.gpa);
        s.table.free();
        s.gpa.free(s.mirror);
        s.gpa.destroy(s);
    }

    /// Whether the image's rows are stored (and now the most recently used).
    pub fn has(s: *Store, digest: *const [32]u8) bool {
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        const e = s.entries.getPtr(digest.*) orelse return false;
        s.tick += 1;
        e.tick = s.tick;
        return true;
    }

    fn put(s: *Store, vid: u32, row: u64) void {
        var h = (vid *% 0x9E3779B1) & (slot_count - 1);
        while (s.mirror[h].key != empty and s.mirror[h].key != vid) h = (h + 1) & (slot_count - 1);
        s.mirror[h] = .{ .key = vid, .row = row };
    }

    fn rebuild(s: *Store) void {
        @memset(s.mirror, .{ .key = empty, .row = 0 });
        for (s.entries.values()) |e| for (e.vids, 0..) |v, k| s.put(v, e.buf.ptr + k * row_bytes);
    }

    /// Takes the image's rows (`buf`: [vids.len, 4096] bf16) and its ids; older images leave when the cap is passed.
    pub fn insert(s: *Store, digest: *const [32]u8, vids: []const u32, buf: cuda.DeviceBuffer) !void {
        if (buf.len != vids.len * row_bytes) return error.VisionRows;
        s.mutex.lockUncancelable(s.io);
        defer s.mutex.unlock(s.io);
        var mine = buf;
        const copy = try s.gpa.dupe(u32, vids);
        errdefer s.gpa.free(copy);
        s.tick += 1;
        const got = try s.entries.getOrPut(s.gpa, digest.*);
        if (got.found_existing) { // a racing request stored it first
            mine.free();
            s.gpa.free(copy);
            return;
        }
        got.value_ptr.* = .{ .buf = buf, .vids = copy, .tick = s.tick };
        s.bytes += buf.len;
        var evicted = false;
        while (s.bytes > s.cap and s.entries.count() > 1) {
            var oldest: usize = 0;
            var best: u64 = std.math.maxInt(u64);
            for (s.entries.values(), 0..) |e, i| if (e.tick < best and !std.mem.eql(u32, e.vids, copy)) {
                best = e.tick;
                oldest = i;
            };
            var e = s.entries.values()[oldest];
            s.entries.orderedRemoveAt(oldest);
            s.bytes -= e.buf.len;
            e.buf.free();
            s.gpa.free(e.vids);
            evicted = true;
        }
        if (evicted) s.rebuild() else for (copy, 0..) |v, k| s.put(v, buf.ptr + k * row_bytes);
        try s.table.upload(0, std.mem.sliceAsBytes(s.mirror));
    }
};

/// The tower, a stream of its own and the store: `embed` makes one image's rows.
pub const Vision = struct {
    gpa: Allocator,
    io: std.Io,
    ctx: *const cuda.Context,
    tower: *vision.Tower,
    stream: cuda.Stream,
    store: *Store,
    mutex: std.Io.Mutex = .init,

    pub fn init(gpa: Allocator, io: std.Io, ctx: *const cuda.Context, dir: []const u8, store: *Store) !*Vision {
        const v = try gpa.create(Vision);
        errdefer gpa.destroy(v);
        const tower = try vision.Tower.load(gpa, io, ctx.d, dir);
        errdefer tower.deinit();
        v.* = .{ .gpa = gpa, .io = io, .ctx = ctx, .tower = tower, .stream = try cuda.Stream.init(ctx.d, true), .store = store };
        return v;
    }

    pub fn deinit(v: *Vision) void {
        v.stream.deinit();
        v.tower.deinit();
        v.gpa.destroy(v);
    }

    /// Whether `dir` holds a vision tower (a multimodal checkpoint): the tower's first tensor is the probe.
    pub fn present(gpa: Allocator, io: std.Io, dir: []const u8) bool {
        const core = @import("core");
        var ck = core.Checkpoint.openModel(gpa, io, dir) catch return false;
        defer ck.close();
        return ck.has("vision_model.radio_model.model.patch_generator.embedder.weight");
    }

    /// The image's rows, stored under its virtual ids (a no-op for an image already held).
    pub fn embed(v: *Vision, img: *const HeldImage) !void {
        if (v.store.has(&img.digest)) return;
        v.mutex.lockUncancelable(v.io);
        defer v.mutex.unlock(v.io);
        if (v.store.has(&img.digest)) return; // another request embedded it while this one waited
        try v.ctx.makeCurrent();
        const rows: usize = img.rows;
        const cols: usize = img.cols;
        const tokens = vision.Tower.tokens(rows, cols);
        if (img.n_vids != tokens) return error.VisionRows;
        var px = try cuda.DeviceBuffer.fromHost(v.ctx.d, std.mem.sliceAsBytes(img.pixels[0 .. 3 * 256 * rows * cols]));
        defer px.free();
        var out = try cuda.DeviceBuffer.alloc(v.ctx.d, tokens * row_bytes);
        errdefer out.free();
        try v.tower.encode(v.stream, px.ptr, rows, cols, out.ptr, null);
        try v.stream.synchronize();
        try v.store.insert(&img.digest, img.vids[0..tokens], out);
    }
};
