//! TF_DSV41_PREFILL_PIPE: a rank's pipeline stage copies.

const std = @import("std");
const cuda = @import("cuda");
const Config = @import("config.zig").Config;
const Pack = @import("pack.zig").Pack;
const plan = @import("plan.zig");
const named = @import("named.zig");
const load = @import("cuda_weights.zig");
const pipe = @import("pipe.zig");
const names = @import("pipe_names.zig");

const log = std.log.scoped(.dsv41);

/// Rank `rank`'s stage copies into `w`; returns their world-1 plan (the experts' layouts the pointer tables read).
pub fn loadStage(gpa: std.mem.Allocator, io: std.Io, d: *const cuda.Driver, pack: *const Pack, cfg: *const Config, s: pipe.Settings, rank: u32, w: *load.Weights) !plan.Plan {
    var lb: [64]u32 = undefined;
    const layers = s.stage(rank).weightLayers(&lb);
    var pl = try plan.build(gpa, cfg, pack, .{ .rank = 0, .world = 1, .blocks = layers });
    errdefer pl.deinit();
    const before = w.bytes;
    const t0 = std.Io.Clock.awake.now(io).toNanoseconds();
    for (pl.layers, 0..) |*l, i| {
        var b: named.Builder = .{ .gpa = gpa, .io = io, .pack = pack };
        defer b.deinit();
        _ = i;
        try b.layer(l, cfg.o_groups, 0, 1);
        try names.keepSharded(&b);
        try w.upload(d, &b);
    }
    const ms = @as(f64, @floatFromInt(std.Io.Clock.awake.now(io).toNanoseconds() - t0)) / 1e6;
    log.info("pipeline: rank {d}'s stage copies (layers {d}-{d} at world 1, sharded tensors only): {d:.1} GiB in {d:.0} ms", .{ rank, layers[0], layers[layers.len - 1], @as(f64, @floatFromInt(w.bytes - before)) / (1 << 30), ms });
    return pl;
}

/// x3gm's pointer table of a stage layer's routed experts ("s.P.L<i>.gm.t{g,u,d}", int64 [E]): each expert's trellis.
pub fn gmTable(a: std.mem.Allocator, w: *const load.Weights, pl: *const plan.Plan, L: u32, proj: usize) ![]u8 {
    const lp = for (pl.layers) |*l| {
        if (l.index == L) break l;
    } else return error.LayerNotPlanned;
    const lay = &lp.experts[proj].layout;
    const wn = ([_][]const u8{ "w1", "w2", "w3" })[proj];
    const data = w.get(try std.fmt.allocPrint(a, names.prefix ++ "L{d}.moe.{s}.data", .{ L, wn })) orelse uniform: {
        const t = w.get(try std.fmt.allocPrint(a, names.prefix ++ "L{d}.moe.{s}.trellis", .{ L, wn })) orelse return error.MissingWeight;
        if (t.len != lay.trellisBytes()) return error.ExpertStackSize;
        break :uniform t;
    };
    const E = lay.words.len;
    const out = try a.alloc(u8, 8 * E);
    for (0..E) |e| std.mem.writeInt(u64, out[8 * e ..][0..8], data.ptr + lay.byteOffset(e), .little);
    return out;
}
