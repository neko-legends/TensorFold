//! Tensor parallelism for Nemotron 3.5 Super across the Sparks (TF_TP_WORLD ranks, one GPU each; fabric/tp from the
//! DeepSeek TP-4 tree). Every rank holds a slice of each projection, as Megatron splits them:
//!   Mamba-2:   heads (and their groups) split; in_proj rows z/x/B/C/dt of the rank's heads, out_proj columns -> sum
//!   attention: q heads split, each rank keeps the one KV head its q heads read; o columns -> sum
//!   latent MoE: router replicated; fc1 rows -> gather; every routed expert's width split (padded to 64), down columns
//!              -> latent sum; fc2 columns; shared expert up rows / down columns -> one sum with fc2's
//!   lm_head and the draft head: vocabulary rows -> gather
//! The norms, embedding and router are whole on every rank. The sums are bf16 in rank order (fabric/tp's contract), so
//! every rank holds the same bits after each, and draws the same tokens.
const std = @import("std");
const cuda = @import("cuda");
const tp = @import("tp");
const tp_kernel = @import("tp_kernel");

/// This rank's place and its communicator; world 1 (no TF_TP_WORLD): every call here is a no-op.
pub const Shard = struct {
    rank: usize = 0,
    world: usize = 1,
    comm: ?tp.Collective = null,
    session: ?*tp.Session = null,
    /// TF_NEMOTRON_TP_NOCOMM=1 (timing only, wrong numbers): the sums and gathers are skipped, to price them.
    nocomm: bool = false,

    pub fn on(s: *const Shard) bool {
        return s.world > 1;
    }

    /// This rank's share of `n` (which every caller has checked divides).
    pub fn part(s: *const Shard, n: usize) usize {
        return n / s.world;
    }

    /// Where this rank's share of `n` starts.
    pub fn first(s: *const Shard, n: usize) usize {
        return s.rank * (n / s.world);
    }

    /// bf16 rows summed across the ranks in place (each rank's partial projection output).
    pub fn sum(s: *const Shard, buf: u64, count: usize, stream: cuda.Stream) !void {
        const c = s.comm orelse return;
        if (s.nocomm) return;
        try c.allReduce(buf, buf, count, .bf16, .sum, stream.handle);
    }

    /// `rows` rows of `width` elements (`elem` bytes) from every rank side by side: dst [rows][world * width], rank r's
    /// at r * width. More than one row gathers into `scratch` ([world][rows][width]) and lays the ranks out from there.
    pub fn gatherCols(s: *const Shard, o: anytype, src: u64, dst: u64, rows: usize, width: usize, elem: usize, scratch: u64) !void {
        const c = s.comm orelse return o.copy(dst, src, rows * width * elem);
        const w = s.world;
        const row = width * elem;
        if (s.nocomm) return o.copyRows(dst, w * row, src, row, row, rows);
        if (rows == 1) return c.allGather(src, dst, row, .u8, o.s.handle);
        try c.allGather(src, scratch, rows * row, .u8, o.s.handle);
        for (0..w) |r| try o.copyRows(dst + r * row, w * row, scratch + r * rows * row, row, row, rows);
    }
};

var session: tp.Session = undefined;
var shard: Shard = .{};
var opened = false;

/// TF_TP_WORLD > 1: this process's rank, its session brought up once (every engine of the process shares it).
pub fn open(d: *const cuda.Driver) !*Shard {
    if (opened) return &shard;
    const world = std.fmt.parseInt(usize, std.mem.span(std.c.getenv("TF_TP_WORLD") orelse "1"), 10) catch return error.BadTpWorld;
    if (world <= 1) {
        opened = true;
        return &shard;
    }
    const cfg = try tp.Config.fromEnv();
    try session.open(d, cfg, tp_kernel.mailbox);
    shard = .{ .rank = cfg.rank, .world = cfg.world, .comm = session.comm(), .session = &session, .nocomm = std.c.getenv("TF_NEMOTRON_TP_NOCOMM") != null };
    opened = true;
    std.log.info("nemotron tp: rank {d} of {d} ({t})", .{ cfg.rank, cfg.world, session.comm().kind() });
    return &shard;
}

/// The shard this process opened (world 1 before `open` or without TF_TP_WORLD).
pub fn current() *Shard {
    return &shard;
}

/// Every rank stops together (each calls this as its engine goes): rank 0 marks the stop as expected so no rank's
/// fail-fast takes the hang-ups for a failure, then all meet once and close the session.
pub fn close() void {
    if (!opened or shard.world <= 1) return;
    if (shard.rank == 0) session.announceStop();
    if (shard.comm) |c| c.barrier() catch {};
    session.close();
    shard = .{};
    opened = false;
}
