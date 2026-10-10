//! Host tests of TP=4 prefill planning (the second GPU window's boot failure, 2026-10-10): every rank's 2,048-row
//! segment (TF_DSV41_PF_4K's fallback segments, and the forward's own 2K programs) is on the prefill path at the boot's
//! options (`f.opts`: pf4k off), at the uneven expert widths (640 on ranks 0 / 1, 512 on ranks 2 / 3).
//!
//! The boot refused it on every rank ("a 2048-row segment is not on the prefill path at any position"): x3gm's block
//! must fit the decode-sized expert scratch (block_prefill.moeRows: expert_rows x (topk + 1) x max(8 I, D) floats), which
//! a 2,048-row block's X / Y (P x D, P = rows x topk) does at TP=2 (I 1,152: 8 I > D) but not at TP=4 (8 I <= D).
const std = @import("std");
const testing = std.testing;
const calls = @import("calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const buffers = @import("buffers.zig");
const check = @import("m1_check.zig");
const Config = @import("config.zig").Config;

fn widths(a: std.mem.Allocator) !block.Widths {
    var w: block.Widths = .{};
    try check.parseDense(a, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(a, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    return w;
}

/// The boot's prefill options at TP=4 (config/tp4-zig.env.example's switches that reach the emitter), pf4k off as in
/// `f.opts`: the 4K segments' options are forward_prefill.k4Options, not these.
fn bootOptions(rank: u32) block.Options {
    const limit: i64 = 420096;
    return .{ .world = 4, .rank = rank, .gm_v2 = .one, .r1 = true, .branches = true, .mhc_defer = true, .expert_topp = 0.85, .limit = limit, .rope_rows = limit + 2048, .prefill_rows = 2048, .index_budget = 64 << 20 };
}

fn count(cs: []const calls.Call, name: []const u8) usize {
    var n: usize = 0;
    for (cs) |c| n += @intFromBool(std.mem.eql(u8, c.name, name));
    return n;
}

test "TP=4: a 2,048-row segment is on the prefill path on every rank (640- and 512-wide experts), x3gm one block" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const layers = [_]u32{ 1, 2, 3 }; // Engram, a ratio-2 kv source (compressor + indexer), a plain layer
    var enc: [21]u32 = undefined; // CED's encoder pass: layers 0 .. 20
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    const D: i64 = cfg.hidden;
    for (0..4) |r| {
        const o = bootOptions(@intCast(r));
        const I: i64 = o.expertWidth(&cfg);
        try testing.expectEqual(@as(i64, if (r < 2) 640 else 512), I);
        for ([_]i64{ 0, 300000 }) |start| {
            const cs = try bp.emitPrefill(aa, &cfg, &w, o, &layers, 2048, start, true);
            const moes = count(cs, "tf_dsv41_x3gm_v1.rot");
            try testing.expectEqual(@as(usize, layers.len), moes);
            try testing.expectEqual(moes, count(cs, "tf_dsv41_x3gm_v1.gateup2"));
            // the block is the segment: P = 2,048 x 6 pairs, X / Y in s.ex.z sized by these views (the buffer plan)
            var plan: buffers.Plan = .{ .a = aa };
            try plan.add(cs);
            for (cs) |c| if (std.mem.eql(u8, c.name, "tf_dsv41_x3gm_v1.rot")) try testing.expectEqual(@as(i64, 2048 * 6), c.args[9].arg.i);
            try testing.expect(plan.sizes.get("s.ex.z").? >= 2 * 2048 * 6 * @as(u64, @intCast(D)) * 2);
            try testing.expect(plan.sizes.get("s.ex.xg").? >= 2048 * 6 * @as(u64, @intCast(I)) * 2);
        }
        // CED replay's encoder segment and a two-segment piece at 2,048 rows, the decoder's tails
        _ = try bp.emitReplay(aa, &cfg, &w, o, &enc, 2048, 0, .encoder);
        const sp = [_]bp.Span{ .{ .slot = 0, .start = 0, .n = 2048 - 17 }, .{ .slot = 1, .start = 0, .n = 17 } };
        _ = try bp.emitMulti(aa, &cfg, &w, o, &enc, &sp, .encoder);
        // the smaller segments the forward plans (17 and 16 rows) were never short of scratch
        _ = try bp.emitPrefill(aa, &cfg, &w, o, &layers, 17, 0, true);
        _ = try bp.emitPrefill(aa, &cfg, &w, o, &layers, 16, 0, true);
    }
}

/// The same Triton specialization, strictly: a tensor's role, dtype and 16-byte alignment (what aot_needs.reduce keys a
/// pointer by), every other argument (ints, constexprs, floats) equal.
fn sameArg(x: calls.Arg, y: calls.Arg) bool {
    if (std.meta.activeTag(x) != std.meta.activeTag(y)) return false;
    return switch (x) {
        .t => x.t.dt == y.t.dt and (@mod(x.t.offset, 16) == 0) == (@mod(y.t.offset, 16) == 0) and
            std.meta.activeTag(x.t.role) == std.meta.activeTag(y.t.role) and std.mem.eql(u8, roleName(x.t.role), roleName(y.t.role)),
        else => std.meta.eql(x, y),
    };
}

fn roleName(r: calls.Role) []const u8 {
    return switch (r) {
        .buf => |b| b,
        .weight => |w| w,
        .empty => "",
    };
}

test "TP=4: the boot's 2K segments (pf4k off) launch the Triton variants aot-needs listed (pf4k on): no new AOT variant" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    var enc: [21]u32 = undefined;
    for (&enc, 0..) |*l, i| l.* = @intCast(i);
    for ([_]u32{ 0, 2 }) |r| {
        // the boot: kv_state sets prefill_rows to 4,096 under TF_DSV41_PF_4K, f.opts.pf4k stays off; aot-needs: pf4k on
        var boot = bootOptions(r);
        boot.prefill_rows = 4096;
        var aot = boot;
        aot.pf4k = true;
        for ([_]u32{ 2048, 17 }) |n| for ([_]i64{ 0, 300000 }) |start| {
            const a = try bp.emitReplay(aa, &cfg, &w, boot, &enc, n, start, .encoder);
            const b = try bp.emitReplay(aa, &cfg, &w, aot, &enc, n, start, .encoder);
            try testing.expectEqual(a.len, b.len);
            var tritons: usize = 0;
            for (a, b) |x, y| {
                try testing.expectEqualStrings(x.name, y.name);
                if (!x.triton) continue;
                tritons += 1;
                try testing.expectEqual(x.args.len, y.args.len);
                for (x.args, y.args) |p, q| if (!sameArg(p.arg, q.arg)) {
                    std.debug.print("rank {d} rows {d}: {s} arg {s} differs: {any} vs {any}\n", .{ r, n, x.name, p.name, p.arg, q.arg });
                    return error.TestUnexpectedResult;
                };
            }
            try testing.expect(tritons > 0);
        };
    }
}

test "TP=2 keeps Python's limit: past what the scratch holds without 4K is refused (x3gm's rows would halve)" {
    var arena = std.heap.ArenaAllocator.init(testing.allocator);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    const w = try widths(aa);
    const layers = [_]u32{ 1, 2, 3 };
    var o: block.Options = .{ .gm_v2 = .one, .limit = 1 << 16, .rope_rows = (1 << 16) + 2048, .prefill_rows = 4096 };
    try testing.expectError(error.Unsupported, bp.emitPrefill(aa, &cfg, &w, o, &layers, 4096, 0, true));
    // and its 2,048-row segment fits the scratch as it always did
    o.prefill_rows = 2048;
    _ = try bp.emitPrefill(aa, &cfg, &w, o, &layers, 2048, 0, true);
}
