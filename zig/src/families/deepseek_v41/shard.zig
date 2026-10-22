//! A width split over the TP ranks on whole 128-value blocks (EXL3's Hadamard blocks).
//!
//! Two shardings of one vocabulary share these rules and are not interchangeable: the embedding's rows
//! split evenly (`vocab / world`, so the plan requires `vocab % world == 0`), while the head's columns
//! split on whole blocks (`of`, uneven at world 4). `vocabShard` is the head's sharding; the embedding
//! sites (batch_graph.zig, forward_glue.zig, dspark_gpu_device.zig, dspark_slots_device.zig) deliberately
//! still read `vocab / world`. Relaxing the plan's `vocab % world` check without changing those sites
//! would silently mis-index the embedding gather.
const std = @import("std");

pub const block: u32 = 128;

/// A rank's run of a split width: [lo, lo + len), plus the widest and narrowest rank's len.
pub const Shard = struct {
    lo: u32,
    len: u32,
    max: u32,
    min: u32,

    pub fn even(s: Shard) bool {
        return s.max == s.min;
    }

    /// The global id of this rank's element `i`.
    pub fn id(s: Shard, i: usize) usize {
        return @as(usize, s.lo) + i;
    }
};

/// `rank`'s blocks of `n_blocks` whole blocks: each rank `n_blocks / world`, the first `n_blocks % world` one more.
pub fn blocks(n_blocks: u32, world: u32, rank: u32) !struct { lo: u32, len: u32 } {
    if (world == 0 or rank >= world) return error.BadWorld;
    const base = n_blocks / world;
    const extra = n_blocks % world;
    if (base == 0) return error.SplitNotOnBlocks;
    return .{ .lo = rank * base + @min(rank, extra), .len = base + @intFromBool(rank < extra) };
}

/// `rank`'s part of `n` values (a multiple of 128) split over `world` on whole blocks.
pub fn of(n: u32, world: u32, rank: u32) !Shard {
    if (n % block != 0) return error.SplitNotOnBlocks;
    const b = try blocks(n / block, world, rank);
    const base = n / block / world;
    const wide = base + @intFromBool(n / block % world != 0);
    return .{ .lo = b.lo * block, .len = b.len * block, .max = wide * block, .min = base * block };
}

/// `of`, else (a width that is not whole blocks, as in test vocabularies) the equal parts when `world` divides it.
/// The one fallback rule: `checked` applies the same rule on serving paths.
pub fn ofOrEven(n: u32, world: u32, rank: u32) !Shard {
    if (of(n, world, rank)) |s| return s else |_| {}
    if (world == 0 or rank >= world or n % world != 0) return error.SplitNotEven;
    const per = n / world;
    return .{ .lo = rank * per, .len = per, .max = per, .min = per };
}

/// The widest rank's part of `n` over `world` (a buffer bound that needs no rank): `of`'s max, else the equal parts.
pub fn widest(n: u32, world: u32) u32 {
    if (n % block == 0 and world != 0 and n / block >= world) return (n / block + world - 1) / world * block;
    return n / world;
}

/// `ofOrEven` for a width the plan and the admission gate already checked: the equal-parts fallback is
/// reached only by test vocabularies, never by a served width (plan.build refuses those). A width neither
/// rule can split here is a bug, not a fallback.
pub fn checked(n: u32, world: u32, rank: u32) Shard {
    return ofOrEven(n, world, rank) catch unreachable;
}

test "checked is ofOrEven's rule on every width" {
    try testing.expectEqual(try ofOrEven(129280, 4, 2), checked(129280, 4, 2));
    try testing.expectEqual(try ofOrEven(1500, 3, 1), checked(1500, 3, 1));
    try testing.expectEqual(try ofOrEven(2304, 2, 0), checked(2304, 2, 0));
}

const testing = std.testing;

test "even splits are the equal parts" {
    for (0..2) |r| {
        const s = try of(2304, 2, @intCast(r));
        try testing.expectEqual(@as(u32, 1152), s.len);
        try testing.expectEqual(@as(u32, @intCast(r)) * 1152, s.lo);
        try testing.expect(s.even());
    }
    const v = try of(129280, 2, 1);
    try testing.expectEqual(Shard{ .lo = 64640, .len = 64640, .max = 64640, .min = 64640 }, v);
    try testing.expectEqual(Shard{ .lo = 3 * 8192, .len = 8192, .max = 8192, .min = 8192 }, try of(32768, 4, 3));
}

test "four ranks: experts 640/640/512/512, vocabulary 32,384/32,384/32,256/32,256" {
    const ex = [_][2]u32{ .{ 0, 640 }, .{ 640, 640 }, .{ 1280, 512 }, .{ 1792, 512 } };
    const vo = [_][2]u32{ .{ 0, 32384 }, .{ 32384, 32384 }, .{ 64768, 32256 }, .{ 97024, 32256 } };
    for (0..4) |r| {
        const e = try of(2304, 4, @intCast(r));
        try testing.expectEqual(ex[r], [2]u32{ e.lo, e.len });
        try testing.expectEqual(@as(u32, 640), e.max);
        try testing.expectEqual(@as(u32, 512), e.min);
        const v = try of(129280, 4, @intCast(r));
        try testing.expectEqual(vo[r], [2]u32{ v.lo, v.len });
        try testing.expectEqual(@as(u32, 32384), v.max);
        try testing.expectEqual(@as(u32, 32256), v.min);
        try testing.expect(!v.even());
    }
    var at: u32 = 0;
    for (0..4) |r| {
        const v = try of(129280, 4, @intCast(r));
        try testing.expectEqual(at, v.lo);
        at += v.len;
    }
    try testing.expectEqual(@as(u32, 129280), at);
}

test "ofOrEven takes whole blocks where they split, else the equal parts" {
    try testing.expectEqual(try of(129280, 4, 2), try ofOrEven(129280, 4, 2));
    try testing.expectEqual(Shard{ .lo = 1000, .len = 500, .max = 500, .min = 500 }, try ofOrEven(1500, 3, 2));
    try testing.expectError(error.SplitNotEven, ofOrEven(1501, 3, 0));
}

test "widest is the widest rank's part, else the equal parts" {
    try testing.expectEqual(@as(u32, 640), widest(2304, 4));
    try testing.expectEqual(@as(u32, 32384), widest(129280, 4));
    try testing.expectEqual(@as(u32, 1152), widest(2304, 2));
    try testing.expectEqual(@as(u32, 8), widest(8, 1));
}

test "a split off the blocks, with fewer blocks than ranks, or past the world is refused" {
    try testing.expectError(error.SplitNotOnBlocks, of(200, 2, 0));
    try testing.expectError(error.SplitNotOnBlocks, of(128, 2, 0));
    try testing.expectError(error.BadWorld, of(256, 2, 2));
    try testing.expectError(error.BadWorld, of(256, 0, 0));
}
