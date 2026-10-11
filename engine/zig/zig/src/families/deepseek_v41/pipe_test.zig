//! Host tests of the pipelined prompt reading's rules (pipe.zig, docs/ZIG-PREFILL-PIPE.md): the knobs and their
//! refusals, the stages (Python's bounds_for, the layers a stage runs and loads, the selection it enters and leaves
//! with), the round schedule and its drain, the rows a segment writes (Python's _rows_of), the exchanges' layouts and
//! the stage weights' names (only the TP-sharded tensors and their forms are the stage's own).
const std = @import("std");
const testing = std.testing;
const calls = @import("calls.zig");
const block = @import("block.zig");
const pipe = @import("pipe.zig");
const Config = @import("config.zig").Config;

const Fake = struct {
    var pairs: []const [2][]const u8 = &.{};
    fn get(name: []const u8) ?[]const u8 {
        for (pairs) |p| if (std.mem.eql(u8, p[0], name)) return p[1];
        return null;
    }
};

fn settingsOf(pairs: []const [2][]const u8, world: u32) !?pipe.Settings {
    Fake.pairs = pairs;
    const cfg: Config = .{};
    return pipe.settings(&Fake.get, &cfg, world);
}

test "TF_DSV41_PREFILL_PIPE: off unless set, Python's stages at TP=4, refusals named" {
    try testing.expect(try settingsOf(&.{}, 4) == null);
    try testing.expect(try settingsOf(&.{.{ "TF_DSV41_PREFILL_PIPE", "0" }}, 4) == null);
    const s = (try settingsOf(&.{.{ "TF_DSV41_PREFILL_PIPE", "1024" }}, 4)).?;
    try testing.expectEqual(@as(u32, 1024), s.min_rows);
    try testing.expectEqual(@as(u32, 2048), s.rows);
    try testing.expectEqual(@as(u32, 4), s.world);
    try testing.expectEqualSlices(u32, &.{ 0, 5, 10, 15, 21 }, s.boundsSlice());
    try testing.expect(!s.trace);
    // 4,096-row stages, explicit bounds, the trace
    const t = (try settingsOf(&.{ .{ "TF_DSV41_PREFILL_PIPE", "2048" }, .{ "TF_DSV41_PIPE_ROWS", "4096" }, .{ "TF_DSV41_PREFILL_PIPE_BOUNDS", "0,4,9,14,21" }, .{ "TF_DSV41_PIPE_TRACE", "1" } }, 4)).?;
    try testing.expectEqual(@as(u32, 4096), t.rows);
    try testing.expectEqualSlices(u32, &.{ 0, 4, 9, 14, 21 }, t.boundsSlice());
    try testing.expect(t.trace);
    // a pipeline needs TP > 1; rows a segment of the line at least pipe.min_seg; 2,048 or 4,096 rows a stage
    try testing.expectError(error.BadKnob, settingsOf(&.{.{ "TF_DSV41_PREFILL_PIPE", "1024" }}, 1));
    for ([_][]const u8{ "-1", "abc", "16", "32" }) |v| try testing.expectError(error.BadKnob, settingsOf(&.{.{ "TF_DSV41_PREFILL_PIPE", v }}, 4));
    for ([_][]const u8{ "1000", "1024", "8192", "x" }) |v| try testing.expectError(error.BadKnob, settingsOf(&.{ .{ "TF_DSV41_PREFILL_PIPE", "1024" }, .{ "TF_DSV41_PIPE_ROWS", v } }, 4));
    // bounds: world + 1 increasing values from 0 to the decoder's first layer + 1
    for ([_][]const u8{ "0,5,10,21", "0,5,5,15,21", "0,5,10,15,20", "1,5,10,15,21", "0,5,10,15,22", "0,5,a,15,21", "0,5,10,15,21,22" }) |v|
        try testing.expectError(error.BadBounds, settingsOf(&.{ .{ "TF_DSV41_PREFILL_PIPE", "1024" }, .{ "TF_DSV41_PREFILL_PIPE_BOUNDS", v } }, 4));
}

test "stage bounds: Python's bounds_for (round half to even), the last stage through the decoder's first layer" {
    const cfg: Config = .{};
    var b: [pipe.max_world + 1]u32 = undefined;
    try testing.expectEqualSlices(u32, &.{ 0, 10, 21 }, try pipe.boundsFor(&cfg, 2, null, &b));
    try testing.expectEqualSlices(u32, &.{ 0, 7, 13, 21 }, try pipe.boundsFor(&cfg, 3, null, &b));
    try testing.expectEqualSlices(u32, &.{ 0, 5, 10, 15, 21 }, try pipe.boundsFor(&cfg, 4, null, &b));
    // 20 x s / 8: 2.5 -> 2, 7.5 -> 8, 12.5 -> 12, 17.5 -> 18 (Python's round)
    try testing.expectEqualSlices(u32, &.{ 0, 2, 5, 8, 10, 12, 15, 18, 21 }, try pipe.boundsFor(&cfg, 8, null, &b));
}

test "a stage's layers, its loaded layers, its owner and the selections it enters / leaves with" {
    const cfg: Config = .{};
    const s = (try settingsOf(&.{.{ "TF_DSV41_PREFILL_PIPE", "1024" }}, 4)).?;
    var buf: [64]u32 = undefined;
    const want = [_][]const u32{ &.{ 0, 1, 2, 3, 4 }, &.{ 5, 6, 7, 8, 9 }, &.{ 10, 11, 12, 13, 14 }, &.{ 15, 16, 17, 18, 19, 20 } };
    const loads = [_][]const u32{ &.{ 0, 1, 2, 3, 4 }, &.{ 5, 6, 7, 8, 9 }, &.{ 10, 11, 12, 13, 14 }, &.{ 15, 16, 17, 18, 19 } };
    const sel_in = [_]?u32{ null, 2, 8, 14 };
    const sel_out = [_]?u32{ 2, 8, 14, null };
    for (0..4) |r| {
        const st = s.stage(@intCast(r));
        try testing.expectEqual(r == 3, st.last);
        try testing.expectEqualSlices(u32, want[r], st.layers(&buf));
        try testing.expectEqualSlices(u32, loads[r], st.weightLayers(&buf));
        try testing.expectEqual(sel_in[r], st.selIn(&cfg));
        try testing.expectEqual(sel_out[r], st.selOut(&cfg, s.boundsSlice()));
    }
    for (0..21) |L| try testing.expectEqual(@as(u32, @intCast(L / 5 - @as(usize, @intFromBool(L == 20)))), s.owner(@intCast(L)));
    // bounds 0,4,9,14,21: rank 3 starts at the kv source 14 (its own selection), rank 1 at layer 4 (2's)
    const t = (try settingsOf(&.{ .{ "TF_DSV41_PREFILL_PIPE", "1024" }, .{ "TF_DSV41_PREFILL_PIPE_BOUNDS", "0,4,9,14,21" } }, 4)).?;
    try testing.expectEqual(@as(?u32, 2), t.stage(1).selIn(&cfg));
    try testing.expectEqual(@as(?u32, 8), t.stage(2).selIn(&cfg));
    try testing.expectEqual(@as(?u32, null), t.stage(3).selIn(&cfg));
    try testing.expectEqual(@as(?u32, null), t.stage(2).selOut(&cfg, t.boundsSlice()));
    // a stage of SWA layers only (0, 1) leaves with no selection; the next starts at the source 2 itself
    const u = (try settingsOf(&.{ .{ "TF_DSV41_PREFILL_PIPE", "1024" }, .{ "TF_DSV41_PREFILL_PIPE_BOUNDS", "0,2,9,14,21" } }, 4)).?;
    try testing.expectEqual(@as(?u32, null), u.stage(0).selOut(&cfg, u.boundsSlice()));
    try testing.expectEqual(@as(?u32, null), u.stage(1).selIn(&cfg));
    // the kv sources a stage writes (its layers' compressors): 2 / 8 / 14 / 20 at Python's bounds
    try testing.expectEqualSlices(u32, &.{2}, s.stage(0).sources(&cfg, &buf));
    try testing.expectEqualSlices(u32, &.{20}, s.stage(3).sources(&cfg, &buf));
}

test "the compressed rows a segment can write (Python's _rows_of: the one before it too)" {
    try testing.expectEqual([2]u64{ 0, 1024 }, pipe.rowsOf(2, 0, 2048));
    try testing.expectEqual([2]u64{ 1023, 2048 }, pipe.rowsOf(2, 2048, 2048));
    try testing.expectEqual([2]u64{ 2047, 4096 }, pipe.rowsOf(1, 2048, 2048));
    try testing.expectEqual([2]u64{ 0, 16 }, pipe.rowsOf(2, 0, 33));
    try testing.expectEqual([2]u64{ 2047, 2065 }, pipe.rowsOf(2, 4097, 33));
    try testing.expectEqual([2]u64{ 0, 33 }, pipe.rowsOf(1, 0, 33));
}

test "the round schedule: round t, stage s runs segment t - s; the drain runs the line empty" {
    var sch: pipe.Schedule = .{ .world = 4 };
    try testing.expect(!sch.busy());
    const segs = [_]pipe.Seg{ .{ .start = 0, .n = 2048 }, .{ .start = 2048, .n = 2048 }, .{ .start = 4096, .n = 2048 }, .{ .start = 6144, .n = 2048 }, .{ .start = 8192, .n = 100 } };
    for (segs, 0..) |sg, t| {
        sch.enter(sg);
        try testing.expectEqual(@as(u64, t), sch.round());
        for (0..4) |s| {
            const got = sch.segOf(@intCast(s));
            if (s > t) try testing.expect(got == null) else try testing.expectEqual(segs[t - s], got.?);
        }
        try testing.expect(sch.busy());
    }
    // three drain rounds: the last segment leaves the last stage in the third
    var drains: usize = 0;
    while (sch.busy()) {
        sch.enter(null);
        drains += 1;
        try testing.expect(sch.segOf(0) == null);
        try testing.expectEqual(segs[4], sch.segOf(@intCast(drains)).?);
    }
    try testing.expectEqual(@as(usize, 3), drains);
    // a fresh line after a drain (or an abandon)
    sch.clear();
    try testing.expect(!sch.busy());
    sch.enter(segs[0]);
    try testing.expectEqual(@as(u64, 0), sch.round());
}

test "the round's rows exchange: a stride a rank, each owned source's comp rows then index keys, 16-byte aligned" {
    const cfg: Config = .{};
    const s = (try settingsOf(&.{.{ "TF_DSV41_PREFILL_PIPE", "1024" }}, 4)).?;
    const lay = pipe.RowsLayout.of(&cfg, s);
    // rank 3's source 20 (ratio 1): 2,048 / 1 + 2 rows of 584 + 132 bytes; ranks 0-2: 1,026 rows (ratio 2)
    try testing.expectEqual(@as(u64, 2050 * 584), lay.comp_bytes[3]);
    try testing.expectEqual(@as(u64, 1026 * 132), lay.ik_bytes[0]);
    try testing.expectEqual(@as(u64, std.mem.alignForward(u64, 2050 * 584, 16) + std.mem.alignForward(u64, 2050 * 132, 16)), lay.stride);
    for (0..4) |r| try testing.expect(lay.used[r] <= lay.stride and lay.used[r] % 16 == 0);
}

test "the drain's hand-off: each stage's SWA rings and ratio-2 carries, the last stage's CED stash, one stride a rank" {
    const cfg: Config = .{};
    const s = (try settingsOf(&.{.{ "TF_DSV41_PREFILL_PIPE", "1024" }}, 4)).?;
    const fin = pipe.FinalsLayout.of(&cfg, s);
    const ring: u64 = 2 * @as(u64, cfg.window) * (576 + 8);
    const carry: u64 = 2 * @as(u64, cfg.head_dim) * 4;
    const stash: u64 = 128 * (4 * @as(u64, cfg.hidden) * 2 + @as(u64, cfg.hidden) * 2 + (4 + 4 + 16) * 4);
    try testing.expectEqual(5 * ring + carry, fin.used[0]); // layers 0-4, the source 2's carry
    try testing.expectEqual(5 * ring + carry, fin.used[1]);
    try testing.expectEqual(5 * ring + carry, fin.used[2]);
    try testing.expectEqual(5 * ring + stash, fin.used[3]); // layers 15-19 (20 is the decoder's), no ratio-2 source
    try testing.expectEqual(fin.used[3], fin.stride);
}

test "stage weights: only the TP-sharded tensors and their run-time forms take the P. prefix" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const a = arena.allocator();
    const W = struct {
        fn w(x: []const u8) calls.Role {
            return .{ .weight = x };
        }
        fn b(x: []const u8) calls.Role {
            return .{ .buf = x };
        }
    };
    const cases = [_]struct { calls.Role, []const u8 }{
        .{ W.w("L5.attn.wq_b.trellis"), "P.L5.attn.wq_b.trellis" },
        .{ W.w("L5.attn.wq_b.svh"), "P.L5.attn.wq_b.svh" },
        .{ W.w("L5.attn.wo_a.3.suh"), "P.L5.attn.wo_a.3.suh" },
        .{ W.w("L5.attn.wo_b.trellis"), "P.L5.attn.wo_b.trellis" },
        .{ W.w("L5.attn.sink"), "P.L5.attn.sink" },
        .{ W.w("L5.moe.w1.suh"), "P.L5.moe.w1.suh" },
        .{ W.w("L5.moe.w2.svh"), "P.L5.moe.w2.svh" },
        .{ W.w("L5.moe.shared.0.w3.trellis"), "P.L5.moe.shared.0.w3.trellis" },
        .{ W.w("L5.attn.wq_a.suh"), "L5.attn.wq_a.suh" },
        .{ W.w("L5.attn.ix_wq_b.svh"), "L5.attn.ix_wq_b.svh" },
        .{ W.w("L5.attn.kv_norm"), "L5.attn.kv_norm" },
        .{ W.w("L5.moe.gate"), "L5.moe.gate" },
        .{ W.w("L5.moe.bias"), "L5.moe.bias" },
        .{ W.w("L14.engram.wkv.suh"), "L14.engram.wkv.suh" },
        .{ W.w("norm"), "norm" },
        .{ W.b("s.L5.attn.wq_b.lanes"), "s.P.L5.attn.wq_b.lanes" },
        .{ W.b("s.L5.attn.wo_b.T"), "s.P.L5.attn.wo_b.T" },
        .{ W.b("s.L5.moe.shared.0.w1.lanes"), "s.P.L5.moe.shared.0.w1.lanes" },
        .{ W.b("s.L5.attn.wq_a.lanes"), "s.L5.attn.wq_a.lanes" },
        .{ W.b("s.L5.gm.tg"), "s.P.L5.gm.tg" },
        .{ W.b("s.L5.gm.td"), "s.P.L5.gm.td" },
        .{ W.b("s.L5.gm.k2g"), "s.L5.gm.k2g" },
        .{ W.b("s.L5.ex.tp_g"), "s.P.L5.ex.tp_g" },
        .{ W.b("s.L5.carry"), "s.L5.carry" },
        .{ W.b("s.L5.swa.v"), "s.L5.swa.v" },
        .{ W.b("s.L5.hc_attn.fn16"), "s.L5.hc_attn.fn16" },
        .{ W.b("s.kv.comp.L2"), "s.kv.comp.L2" },
        .{ W.b("w.x0"), "w.x0" },
        .{ W.b("L.pp~recv.moe"), "L.pp~recv.moe" },
    };
    for (cases) |c| {
        const got = try pipe.renameRole(a, c[0]);
        const name = switch (got) {
            .weight => |x| x,
            .buf => |x| x,
            .empty => "",
        };
        try testing.expectEqualStrings(c[1], name);
        try testing.expectEqual(std.meta.activeTag(c[0]), std.meta.activeTag(got));
    }
    try testing.expect(pipe.sharded("attn.wq_b") and pipe.sharded("moe.w3") and pipe.sharded("moe.shared.0.w2") and pipe.sharded("attn.wo_a.7"));
    try testing.expect(!pipe.sharded("attn.wq_a") and !pipe.sharded("moe.gate") and !pipe.sharded("engram.wkv") and !pipe.sharded("hc_attn.fn"));
}

test "a stage's options: world 1 rank 0, the line's rows, no taps / TBO / exchange pieces / images" {
    var o: block.Options = .{ .world = 4, .rank = 2, .taps = true, .pf_tbo = true, .pf_overlap = 2, .pf_overlap_site = true, .image_keep = true, .image_rows = 5, .prefill_rows = 2048, .gm_v2 = .one, .r1 = true, .limit = 420096 };
    const x = pipe.stageOptions(o, 2048);
    try testing.expectEqual(@as(u32, 1), x.world);
    try testing.expectEqual(@as(u32, 0), x.rank);
    try testing.expect(!x.taps and !x.pf_tbo and x.pf_overlap == 0 and !x.pf_overlap_site and !x.image_keep and x.image_rows == 0 and !x.pf4k);
    try testing.expectEqual(@as(i64, 2048), x.prefill_rows);
    try testing.expect(x.r1 and x.gm_v2 == .one and x.limit == 420096);
    const cfg: Config = .{};
    try testing.expectEqual(@as(i64, 2304), x.expertWidth(&cfg));
    o.prefill_rows = 2048;
    const y = pipe.stageOptions(o, 4096);
    try testing.expect(y.pf4k and y.prefill_rows == 4096);
}
