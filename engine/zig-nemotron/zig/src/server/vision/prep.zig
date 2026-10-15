//! Nemotron 3.5 Super VL's image processor (image_processing_nemotron_h_omni.py, the image path) on an RGB image: a
//! dynamic-resolution tile of 16-pixel patches chosen from the aspect ratio and the patch budget, torch's antialiased
//! bicubic resize (``F.interpolate(mode="bicubic", antialias=True)``: separable, width first, float32), the CLIP
//! normalisation, bf16, and the digest every virtual id derives from. The processor's tokens: one per 2 x 2 patches.
const std = @import("std");
const pixels = @import("pixels.zig");
const Allocator = std.mem.Allocator;

pub const VERSION = "tensorfold-nemotron-vision/1";

pub const Error = error{ BadSettings, OutOfMemory };

/// preprocessor_config.json's numbers and the server's limits (``TF_NEMOTRON_VISION_*``).
pub const Settings = struct {
    patch: u32 = 16,
    shuffle: u32 = 2,
    min_patches: u32 = 1024,
    max_patches: u32 = 13312,
    mean: [3]f32 = .{ 0.48145466, 0.4578275, 0.40821073 },
    std: [3]f32 = .{ 0.26862954, 0.26130258, 0.27577711 },
    max_images: u32 = 8,
    fetch: bool = true,
    fetch_timeout_s: f64 = 10,
    fetch_total_s: f64 = 30,
    max_bytes: usize = 20 * 1024 * 1024,
    max_pixels: u64 = 64 * 1024 * 1024,
    fetch_http: bool = false,
    fetch_private: bool = false,
    prep_slots: u32 = 4,
    prep_mb: u64 = 512,

    pub fn read(env: ?*const std.process.Environ.Map, problem: *[]const u8) Error!Settings {
        var s: Settings = .{};
        const m = env orelse return s;
        s.max_patches = try envInt(m, "TF_NEMOTRON_VISION_MAX_PATCHES", s.max_patches, problem);
        s.max_images = try envInt(m, "TF_NEMOTRON_VISION_MAX_IMAGES", s.max_images, problem);
        s.fetch = try envFlag(m, "TF_NEMOTRON_VISION_FETCH", true, problem);
        s.fetch_timeout_s = try envFloat(m, "TF_NEMOTRON_VISION_FETCH_TIMEOUT", 10, problem);
        s.fetch_total_s = try envFloat(m, "TF_NEMOTRON_VISION_FETCH_TOTAL_S", 30, problem);
        s.max_bytes = try envInt(m, "TF_NEMOTRON_VISION_MAX_BYTES", s.max_bytes, problem);
        s.max_pixels = try envInt(m, "TF_NEMOTRON_VISION_MAX_PIXELS", s.max_pixels, problem);
        s.fetch_http = try envFlag(m, "TF_NEMOTRON_VISION_FETCH_HTTP", false, problem);
        s.fetch_private = try envFlag(m, "TF_NEMOTRON_VISION_FETCH_PRIVATE", false, problem);
        s.prep_slots = try envInt(m, "TF_NEMOTRON_VISION_PREP_SLOTS", s.prep_slots, problem);
        s.prep_mb = try envInt(m, "TF_NEMOTRON_VISION_PREP_MB", s.prep_mb, problem);
        if (s.max_images < 1 or s.prep_slots < 1 or s.max_patches < s.shuffle * s.shuffle) return bad(problem, "TF_NEMOTRON_VISION_MAX_IMAGES >= 1, _PREP_SLOTS >= 1, _MAX_PATCHES >= 4");
        if (s.min_patches > s.max_patches) s.min_patches = s.max_patches;
        return s;
    }

    /// Most patches an image may take (the per-image budget of the processor).
    pub fn budget(s: *const Settings) u32 {
        return s.max_patches;
    }
};

fn bad(problem: *[]const u8, msg: []const u8) error{BadSettings} {
    problem.* = msg;
    return error.BadSettings;
}

fn envInt(m: *const std.process.Environ.Map, name: []const u8, default: anytype, problem: *[]const u8) error{BadSettings}!@TypeOf(default) {
    const raw = std.mem.trim(u8, m.get(name) orelse "", " \t\r\n");
    if (raw.len == 0) return default;
    const f = std.fmt.parseFloat(f64, raw) catch return bad(problem, name);
    if (!(f >= 0) or f > 1e15) return bad(problem, name);
    return std.math.cast(@TypeOf(default), @as(u64, @intFromFloat(@trunc(f)))) orelse bad(problem, name);
}

fn envFloat(m: *const std.process.Environ.Map, name: []const u8, default: f64, problem: *[]const u8) error{BadSettings}!f64 {
    const raw = std.mem.trim(u8, m.get(name) orelse "", " \t\r\n");
    if (raw.len == 0) return default;
    return std.fmt.parseFloat(f64, raw) catch bad(problem, name);
}

fn envFlag(m: *const std.process.Environ.Map, name: []const u8, default: bool, problem: *[]const u8) error{BadSettings}!bool {
    const raw = std.mem.trim(u8, m.get(name) orelse "", " \t\r\n");
    if (raw.len == 0) return default;
    if (std.mem.eql(u8, raw, "1")) return true;
    if (std.mem.eql(u8, raw, "0")) return false;
    return bad(problem, name);
}

/// Python's ``round()`` of a float: halves to even.
fn pyRound(x: f64) i64 {
    const f = @floor(x);
    const d = x - f;
    if (d < 0.5) return @intFromFloat(f);
    if (d > 0.5) return @intFromFloat(f + 1);
    const lo: i64 = @intFromFloat(f);
    return if (@mod(lo, 2) == 0) lo else lo + 1;
}

/// ``_compute_target_patches``: the (width, height) in patches that best fit `avail` patches.
pub fn plan(orig_w: u32, orig_h: u32, s: *const Settings) [2]u32 {
    const ps: f64 = @floatFromInt(s.patch);
    const avail: f64 = @floatFromInt(s.max_patches);
    const min_p: f64 = @floatFromInt(s.min_patches);
    const closest_h = pyRound(@as(f64, @floatFromInt(orig_h)) / ps + 0.5);
    const closest_w = pyRound(@as(f64, @floatFromInt(orig_w)) / ps + 0.5);
    const patches: f64 = @floatFromInt(closest_h * closest_w);
    const factor = @min(@sqrt(avail / patches), 1.0);
    var th: i64 = @intFromFloat(@floor(factor * @as(f64, @floatFromInt(closest_h))));
    var tw: i64 = @intFromFloat(@floor(factor * @as(f64, @floatFromInt(closest_w))));
    if (avail > min_p and @as(f64, @floatFromInt(th * tw)) < min_p) {
        const up = @sqrt(min_p / @as(f64, @floatFromInt(th * tw)));
        th = @intFromFloat(@ceil(up * @as(f64, @floatFromInt(th))));
        tw = @intFromFloat(@ceil(up * @as(f64, @floatFromInt(tw))));
    }
    const div: i64 = s.shuffle;
    const max_i: i64 = @intFromFloat(avail);
    const rem_h = @mod(th, div);
    if (rem_h != 0) {
        const inc = div - rem_h;
        if ((th + inc) * tw <= max_i) th += inc else th = @max(div, th - rem_h);
    }
    const rem_w = @mod(tw, div);
    if (rem_w != 0) {
        const inc = div - rem_w;
        if (th * (tw + inc) <= max_i) tw += inc else tw = @max(div, tw - rem_w);
    }
    return .{ @intCast(tw), @intCast(th) };
}

/// One preprocessed image: bf16 [3, 16 rows, 16 cols] normalized pixels, the grid, its digest and virtual ids.
pub const Prepared = struct {
    patches: []u16, // the pixels
    rows: u32, // patch rows
    cols: u32,
    digest: [32]u8,
    size: [2]u32, // the original (width, height)
    vids: []u32 = &.{},

    pub fn tokens(p: *const Prepared) u32 {
        return p.rows * p.cols / 4;
    }

    pub fn deinit(p: *Prepared, gpa: Allocator) void {
        gpa.free(p.patches);
        gpa.free(p.vids);
    }
};

pub fn bf16(x: f32) u16 {
    const b: u32 = @bitCast(x);
    if (std.math.isNan(x)) return @intCast((b >> 16) | 0x40);
    return @intCast((b + 0x7fff + ((b >> 16) & 1)) >> 16);
}

/// torch's ``_compute_indices_min_size_weights_aa`` for the bicubic filter (a = -0.5), float32 throughout.
const Taps = struct {
    min: []u32,
    len: []u32,
    w: []f32, // [out, stride]
    stride: usize,

    fn init(gpa: Allocator, inn: u32, out: u32) !Taps {
        const scale: f32 = @as(f32, @floatFromInt(inn)) / @as(f32, @floatFromInt(out));
        const support: f32 = if (scale >= 1.0) 2.0 * scale else 2.0;
        const invscale: f32 = if (scale >= 1.0) 1.0 / scale else 1.0;
        const stride: usize = @as(usize, @intFromFloat(@ceil(support))) * 2 + 2;
        const mins = try gpa.alloc(u32, out);
        errdefer gpa.free(mins);
        const lens = try gpa.alloc(u32, out);
        errdefer gpa.free(lens);
        const w = try gpa.alloc(f32, @as(usize, out) * stride);
        errdefer gpa.free(w);
        @memset(w, 0);
        for (0..out) |i| {
            const center: f32 = scale * (@as(f32, @floatFromInt(i)) + 0.5);
            const lo: i64 = @max(@as(i64, @intFromFloat(center - support + 0.5)), 0);
            const hi: i64 = @min(@as(i64, @intFromFloat(center + support + 0.5)), inn);
            const size: usize = @intCast(hi - lo);
            mins[i] = @intCast(lo);
            lens[i] = @intCast(size);
            var total: f32 = 0;
            for (0..size) |j| {
                const x = (@as(f32, @floatFromInt(@as(i64, @intCast(j)) + lo)) - center + 0.5) * invscale;
                const wt = filter(x);
                w[i * stride + j] = wt;
                total += wt;
            }
            if (total != 0) for (0..size) |j| {
                w[i * stride + j] /= total;
            };
        }
        return .{ .min = mins, .len = lens, .w = w, .stride = stride };
    }

    fn deinit(t: Taps, gpa: Allocator) void {
        gpa.free(t.min);
        gpa.free(t.len);
        gpa.free(t.w);
    }
};

fn filter(x0: f32) f32 {
    const a: f32 = -0.5;
    const x = @abs(x0);
    if (x < 1.0) return ((a + 2.0) * x - (a + 3.0)) * x * x + 1.0;
    if (x < 2.0) return ((a * x - 5.0 * a) * x + 8.0 * a) * x - 4.0 * a;
    return 0.0;
}

/// ``preprocess``: an RGB image -> its normalized bf16 pixels and digest (the virtual ids come from the registry).
pub fn preprocess(gpa: Allocator, image: pixels.Rgb, s: *const Settings) Error!Prepared {
    const g = plan(image.w, image.h, s);
    const ow = g[0] * s.patch;
    const oh = g[1] * s.patch;
    const hw: usize = @as(usize, oh) * ow;
    const out = try gpa.alloc(u16, 3 * hw);
    errdefer gpa.free(out);
    if (image.w == ow and image.h == oh) {
        for (0..3) |c| for (0..hw) |i| {
            out[c * hw + i] = norm(@floatFromInt(image.data[i * 3 + c]), s, c);
        };
    } else {
        const tw = try Taps.init(gpa, image.w, ow);
        defer tw.deinit(gpa);
        const th = try Taps.init(gpa, image.h, oh);
        defer th.deinit(gpa);
        // width first: [3, h, ow] float32, then height
        const mid = try gpa.alloc(f32, 3 * @as(usize, image.h) * ow);
        defer gpa.free(mid);
        for (0..3) |c| for (0..image.h) |y| {
            const src = image.data[y * @as(usize, image.w) * 3 ..];
            const dst = mid[(c * @as(usize, image.h) + y) * ow ..][0..ow];
            for (0..ow) |i| {
                var acc: f32 = 0;
                const base = tw.min[i];
                for (0..tw.len[i]) |j| acc += tw.w[i * tw.stride + j] * @as(f32, @floatFromInt(src[(base + j) * 3 + c]));
                dst[i] = acc;
            }
        };
        const row = try gpa.alloc(f32, ow);
        defer gpa.free(row);
        for (0..3) |c| for (0..oh) |i| {
            @memset(row, 0);
            const base = th.min[i];
            for (0..th.len[i]) |j| {
                const wt = th.w[i * th.stride + j];
                const src = mid[(c * @as(usize, image.h) + base + j) * ow ..][0..ow];
                for (row, src) |*r, v| r.* += wt * v;
            }
            for (0..ow) |x| out[c * hw + i * ow + x] = norm(row[x], s, c);
        };
    }
    var h = std.crypto.hash.sha2.Sha256.init(.{});
    h.update(VERSION);
    var gbuf: [64]u8 = undefined;
    h.update(std.fmt.bufPrint(&gbuf, "[{d}, {d}]", .{ g[1], g[0] }) catch unreachable);
    h.update(std.mem.sliceAsBytes(out));
    var digest: [32]u8 = undefined;
    h.final(&digest);
    return .{ .patches = out, .rows = g[1], .cols = g[0], .digest = digest, .size = .{ image.w, image.h } };
}

/// ``(t / 255.0 - mean) / std`` in float32, to bf16.
fn norm(t: f32, s: *const Settings, c: usize) u16 {
    return bf16((t / 255.0 - s.mean[c]) / s.std[c]);
}

test "the processor's grid: 1600 x 1200 is 100 x 76 patches, a wide strip 376 x 32, small images scale up to the minimum" {
    const s: Settings = .{};
    try std.testing.expectEqual([2]u32{ 100, 76 }, plan(1600, 1200, &s));
    try std.testing.expectEqual([2]u32{ 376, 32 }, plan(6000, 500, &s));
    const small = plan(100, 50, &s);
    try std.testing.expect(small[0] * small[1] >= 1024 and small[0] * small[1] <= 13312 and small[0] % 2 == 0 and small[1] % 2 == 0);
}

test "python round: halves to even" {
    try std.testing.expectEqual(@as(i64, 2), pyRound(2.5));
    try std.testing.expectEqual(@as(i64, 4), pyRound(3.5));
    try std.testing.expectEqual(@as(i64, 3), pyRound(2.6));
}
