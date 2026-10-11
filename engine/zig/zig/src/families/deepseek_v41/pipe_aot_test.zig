//! Host test of the AOT fill's coverage of the pipeline's stage programs (TF_DSV41_PREFILL_PIPE, aot_needs.enumerateStage):
//! at TP=4's served config (TF_DSV41_CONTEXT=420096: 1,641 pages), on every rank, every Triton launch of every stage
//! program the boot plans (pipe.planned) and of the stage programs prompts run (any segment past pipe.min_seg rows at
//! any 16-row-grid start, as the prompt pieces and snapshots fall) reduces, through the served Triton signatures
//! (fixtures/triton-sigs.json: `triton_fill.py sigs` of the Python tree the fill compiles), to a variant aot-needs
//! lists. A boot-requested variant the host pass did not list cost a window (2026-10-10); this pins the stage set.
const std = @import("std");
const testing = std.testing;
const calls = @import("calls.zig");
const block = @import("block.zig");
const bp = @import("block_prefill.zig");
const check = @import("m1_check.zig");
const aot = @import("aot_needs.zig");
const pipe = @import("pipe.zig");
const Config = @import("config.zig").Config;

const limit: i64 = 420096;

/// aot-needs' options for config/tp4-zig.env (Served.fromEnv: TF_TP_WORLD=4, TF_TP_RANK=r, the PF_4K workspace set,
/// x3gm v2, INDEX_BOUND, the 64 MiB index budget, MHC_PFDEC=17, R1, drafts), the pool as it plans it (KV not split).
fn served(rank: u32) block.Options {
    const pages = @divExact(limit, block.Pool.page);
    return .{ .world = 4, .rank = rank, .r1 = true, .expert_topp = 0.85, .limit = limit, .rope_rows = limit + 2048, .prefill_rows = 4096, .pf4k = true, .gm_v2 = .one, .index_bound = true, .index_budget = 64 << 20, .mhc_pf_rows = 17, .taps = true, .pool = .{ .comp_pages = pages + 1, .ik_pages = pages + 1, .pts = pages } };
}

fn settings() !pipe.Settings {
    const F = struct {
        fn get(name: []const u8) ?[]const u8 {
            return if (std.mem.eql(u8, name, "TF_DSV41_PREFILL_PIPE")) "1024" else null;
        }
    };
    const cfg: Config = .{};
    return (try pipe.settings(&F.get, &cfg, 4)).?;
}

test "TP=4 (CONTEXT=420096): the boot's and the prompts' stage programs are in the AOT fill's list, every rank" {
    const gpa = testing.allocator;
    const parsed = try std.json.parseFromSlice(struct { kernels: aot.Sigs }, gpa, @embedFile("fixtures/triton-sigs.json"), .{ .ignore_unknown_fields = true, .allocate = .alloc_always });
    defer parsed.deinit();
    var arena = std.heap.ArenaAllocator.init(gpa);
    defer arena.deinit();
    const aa = arena.allocator();
    const cfg: Config = .{};
    var w: block.Widths = .{};
    try check.parseDense(aa, &w, @embedFile("fixtures/q28v2-dense-k2.txt"));
    try check.parseExperts(aa, &w, @embedFile("fixtures/q28v2-expert-k2.txt"));
    try pipe.fillGroups(aa, &w, &cfg);
    const s = try settings();
    var discard: std.Io.Writer.Discarding = .init(&.{});
    var buf: [64]u32 = undefined;
    for (0..4) |r| {
        const o1 = pipe.stageOptions(served(@intCast(r)), s.rows);
        var needs: aot.Needs = .{ .gpa = gpa, .sigs = &parsed.value.kernels };
        defer needs.deinit();
        try aot.enumerateStage(gpa, &needs, &cfg, &w, o1, s, @intCast(r), &discard.writer);
        try testing.expect(needs.keys.count() > 0);
        const st = s.stage(@intCast(r));
        const io: bp.StageIo = .{ .entered = r > 0, .last = st.last, .sel_in = st.selIn(&cfg), .sel_out = st.selOut(&cfg, s.boundsSlice()) };
        // the boot's plan, then prompt segments: whole ones, a piece's or a resumed prompt's ragged ones, the shortest
        var cases: std.ArrayList([2]i64) = .empty;
        for (try pipe.planned(aa, &cfg, &w, o1, s, @intCast(r))) |p| try cases.append(aa, p);
        for ([_]i64{ 2048, 2047, 1500, 1000, 100, 33 }) |n| for ([_]i64{ 0, 16, 2048, 4112, 37984, 154176, 300000, 417984 }) |start| {
            if (start + n <= limit) try cases.append(aa, .{ n, start });
        };
        for (cases.items) |p| {
            var pa = std.heap.ArenaAllocator.init(gpa);
            defer pa.deinit();
            const cs = try bp.emitStage(pa.allocator(), &cfg, &w, o1, st.layers(&buf), p[0], p[1], io);
            for (cs) |*c| if (c.triton) {
                const key = try needs.reduce(pa.allocator(), c);
                if (needs.keys.get(key) == null) {
                    std.debug.print("rank {d}: a {d}-row stage at {d} launches a variant aot-needs did not list: {s}\n", .{ r, p[0], p[1], key });
                    return error.TestUnexpectedResult;
                }
            };
        }
    }
}
