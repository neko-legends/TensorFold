//! A replayed operation's error classes for a follower (forward.zig `follow`).
//!
//! Rank 0 serves requests: a request-level failure ends that request (the client sees the error, the engine
//! serves on). A follower replaying the same operation must survive the same classes — the 2026-10-10 window:
//! an evicted session's resume failed per-request on rank 0 (NotInRam) while the followers' replay made it
//! process-fatal, and the fate channel took the whole fleet down. Genuine process, device and transport faults
//! stay process-fatal: they are what the tp fail-fast design exists for.

const std = @import("std");

/// True when a replayed operation's error must end the follower: the replica cannot meaningfully continue
/// (its allocator, its device or the operations stream itself failed, or its pool state provably diverged
/// from rank 0's). Everything else — session-store state (an evicted, trimmed or damaged entry) and the NVMe
/// tier's I/O, a pool a request priced out — is a request-level error rank 0 already ends only the request
/// for, so the follower logs it and serves on. One carve-out: a spill's park that failed on this rank
/// (ParkFailed) IS the tier's I/O, yet it is not classed request-level — a park cannot be mirrored or rolled
/// back locally (where the write landed the entry's pages were freed and the reservation taken), so a failure
/// there cannot be recovered consistently; it joins the pool-divergence fatal class, and the fate channel's
/// restart is the recovery. (Its prevention: rank 0 refuses the operations whose parks its tier cannot
/// hold — kv/sess.zig's tier gate, TierFull, for an admission's spills and a save's trims alike — so a
/// full tier on every rank degrades to refusals instead of a ParkFailed crash loop.)
pub fn processFatal(err: anyerror) bool {
    return switch (err) {
        // the process itself cannot mirror rank 0's work
        error.OutOfMemory => true,
        // the operations stream: a peer or a framing failure is not a request's
        error.PeerClosed, error.TooLarge, error.BadPlan => true,
        // the device and its libraries: a replica whose GPU faulted pairs no more collectives
        error.DriverUnavailable, error.MissingSymbol, error.Unsupported, error.CudaFailed, error.OutOfDeviceMemory, error.NotReady, error.NotFound, error.Invalid, error.CublasFailed, error.NoAlgorithm, error.NcclFailed => true,
        // pool integrity: state that can only mean the ranks' pools diverged (fail-fast's own)
        error.ResidueMismatch, error.DoubleFree, error.PastCapacity, error.SlotNotEmpty, error.BadPool => true,
        // a spill's park that failed on this rank: the pools diverged by construction the moment another
        // rank's write landed (sessions/store.zig's settle maps it, world > 1 only — the NVMe read path's
        // own errors stay request-level, folded by restore()'s agreement)
        error.ParkFailed => true,
        // request-level: session-store state, the NVMe tier's I/O, a request's pool price — rank 0 ends the
        // request for these and serves on, so the follower does too
        else => false,
    };
}

test "replayed request-level errors are survivable; process, device and transport faults fatal" {
    // the fleet-fatal class of the w4b window: an evicted session's resume
    try std.testing.expect(!processFatal(error.NotInRam));
    // the rest of the session store's and the NVMe tier's request-level state
    try std.testing.expect(!processFatal(error.NotOnDisk));
    try std.testing.expect(!processFatal(error.BadBlob));
    try std.testing.expect(!processFatal(error.BadChain));
    try std.testing.expect(!processFatal(error.FileNotFound));
    try std.testing.expect(!processFatal(error.BadHeader));
    try std.testing.expect(!processFatal(error.Truncated));
    try std.testing.expect(!processFatal(error.ChecksumMismatch));
    try std.testing.expect(!processFatal(error.NoDisk));
    try std.testing.expect(!processFatal(error.NotParkable));
    try std.testing.expect(!processFatal(error.Busy));
    try std.testing.expect(!processFatal(error.BadWindow));
    try std.testing.expect(!processFatal(error.InputOutput));
    try std.testing.expect(!processFatal(error.NoSpaceLeft));
    // a request priced the pool out (rank 0 ends the request; every rank's pool agreed on it)
    try std.testing.expect(!processFatal(error.PoolExhausted));
    try std.testing.expect(!processFatal(error.SlotTooShort));
    // rank 0's tier gate refused the admission pre-broadcast (nothing was applied, on any rank)
    try std.testing.expect(!processFatal(error.TierFull));
    // a spill's park that failed on this rank only: the ranks' pools diverged by construction (the entry's
    // pages freed where the write landed, held here) — the fate channel's clean restart, as a replica's OOM
    try std.testing.expect(processFatal(error.ParkFailed));
    // the process itself cannot mirror rank 0's work
    try std.testing.expect(processFatal(error.OutOfMemory));
    // the operations stream (a peer or a framing failure is not a request's)
    try std.testing.expect(processFatal(error.PeerClosed));
    try std.testing.expect(processFatal(error.TooLarge));
    try std.testing.expect(processFatal(error.BadPlan));
    // the device and its libraries: a replica whose GPU faulted pairs no more collectives
    try std.testing.expect(processFatal(error.DriverUnavailable));
    try std.testing.expect(processFatal(error.MissingSymbol));
    try std.testing.expect(processFatal(error.Unsupported));
    try std.testing.expect(processFatal(error.CudaFailed));
    try std.testing.expect(processFatal(error.OutOfDeviceMemory));
    try std.testing.expect(processFatal(error.NotReady));
    try std.testing.expect(processFatal(error.NotFound));
    try std.testing.expect(processFatal(error.Invalid));
    try std.testing.expect(processFatal(error.CublasFailed));
    try std.testing.expect(processFatal(error.NoAlgorithm));
    try std.testing.expect(processFatal(error.NcclFailed));
    // pool integrity: state that can only mean the ranks' pools diverged (fail-fast's own)
    try std.testing.expect(processFatal(error.ResidueMismatch));
    try std.testing.expect(processFatal(error.DoubleFree));
    try std.testing.expect(processFatal(error.PastCapacity));
    try std.testing.expect(processFatal(error.SlotNotEmpty));
    try std.testing.expect(processFatal(error.BadPool));
}
