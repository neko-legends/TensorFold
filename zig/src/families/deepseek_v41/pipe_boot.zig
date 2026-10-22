//! TF_DSV41_PREFILL_PIPE's boot side: the line's buffer layout, the boot's refusals and what it prices.
const std = @import("std");
const cuda = @import("cuda");
const buffers = @import("buffers.zig");
const pipe = @import("pipe.zig");
const eh = @import("engram_host.zig");
const pf = @import("forward_prefill.zig");
const pipe_run = @import("pipe_run.zig");
const Forward = @import("forward.zig").Forward;
const Config = @import("config.zig").Config;

const log = std.log.scoped(.dsv41);

pub const Bufs = struct {
    x0: ?cuda.DeviceBuffer = null,
    emb_send: ?cuda.DeviceBuffer = null,
    emb_recv: ?cuda.DeviceBuffer = null,
    dev_ids: ?cuda.DeviceBuffer = null,
    eng_rows: [eh.max_layers]?cuda.DeviceBuffer = @splat(null),
    eng_send: ?cuda.DeviceBuffer = null,
    eng_gathered: ?cuda.DeviceBuffer = null,
    send_f: ?cuda.DeviceBuffer = null,
    recv_f: ?cuda.DeviceBuffer = null,
    rows_send: ?cuda.DeviceBuffer = null,
    rows_recv: ?cuda.DeviceBuffer = null,
    fin_send: ?cuda.DeviceBuffer = null,
    fin_recv: ?cuda.DeviceBuffer = null,

    pub fn all(b: *Bufs) [12 + eh.max_layers]*?cuda.DeviceBuffer {
        var out: [12 + eh.max_layers]*?cuda.DeviceBuffer = undefined;
        const fixed = [_]*?cuda.DeviceBuffer{ &b.x0, &b.emb_send, &b.emb_recv, &b.dev_ids, &b.eng_send, &b.eng_gathered, &b.send_f, &b.recv_f, &b.rows_send, &b.rows_recv, &b.fin_send, &b.fin_recv };
        @memcpy(out[0..fixed.len], &fixed);
        for (&b.eng_rows, 0..) |*x, i| out[fixed.len + i] = x;
        return out;
    }
};

/// The device buffers' bytes (`bind` allocates them at boot; the boot check prices them before the Engram host is.
pub fn bufferBytes(cfg: *const Config, s: pipe.Settings, rank: u32) u64 {
    const W: u64 = s.world;
    const R: u64 = s.rows;
    const D: u64 = cfg.hidden;
    const hand = pipe.HandoffLayout.of(cfg, s.rows, true, true).total;
    var total: u64 = R * 4 * D * 2 + R * D * 2 + W * R * D * 2 + R * 8 + 2 * hand;
    total += pipe.RowsLayout.of(cfg, s).stride * (W + 1) + pipe.FinalsLayout.of(cfg, s).stride * (W + 1);
    if (cfg.engram_layers.len > 0) {
        const heads: u64 = cfg.hashCols();
        const cols: u64 = (heads + W - 1) / W * cfg.engram_head_dim;
        const K: u64 = heads * cfg.engram_head_dim;
        total += R * cols * 2 + W * R * cols * 2;
        const st = s.stage(rank);
        for (cfg.engram_layers.items()) |L| if (L >= st.lo and L < st.hi) {
            total += R * K * 2;
        };
    }
    return total;
}

/// The active slot (the pool's current one; the line runs on its views alone).
pub fn activeSlot(f: *const Forward) u32 {
    return if (f.kv) |kx| kx.slot.id else 0;
}

/// The refusals of a boot with TF_DSV41_PREFILL_PIPE (every rank reads the same environment, so every rank refuses).
pub fn check(f: *const Forward, s: pipe.Settings) !void {
    if (pf.state(f).mode != .replay) {
        log.warn("TF_DSV41_PREFILL_PIPE: pipelines CED's encoder pass; needs TF_DSV41_PREFILL=replay", .{});
        return error.BadKnob;
    }
    const pool = f.opts.pool orelse {
        log.warn("TF_DSV41_PREFILL_PIPE: needs the paged pool (TF_DSV41_POOL_TOKENS)", .{});
        return error.BadKnob;
    };
    if (pool.split) {
        log.warn("TF_DSV41_PREFILL_PIPE: a stage reads its sources' whole history; needs TF_DSV41_KV_SPLIT=0", .{});
        return error.BadKnob;
    }
    if (f.opts.gm_v2 != .one) {
        // 0 / split: gm_plan's width bits read the forward's K2 table by its bare name (pipe.planned refuses it too).
        log.warn("TF_DSV41_PREFILL_PIPE: a stage's MoE needs TF_DSV41_GM_V2=1, not {t}", .{f.opts.gm_v2});
        return error.BadKnob;
    }
    if (f.cfg.vocab % s.world != 0) return error.BadKnob;
}

/// The stage programs' roles into `p` (forward_prefill.plan): pipe.planned's programs.
pub fn planInto(f: *Forward, ln: *const pipe_run.Line, p: *buffers.Plan) !void {
    var arena = std.heap.ArenaAllocator.init(f.gpa);
    defer arena.deinit();
    const a = arena.allocator();
    for (try pipe.planned(a, f.cfg, f.widths, ln.o1, ln.s, ln.rank)) |x| {
        var pa = std.heap.ArenaAllocator.init(f.gpa);
        defer pa.deinit();
        try p.add(try pipe.program(pa.allocator(), f.cfg, f.widths, ln.o1, &ln.s, ln.rank, x[0], x[1]));
    }
}

/// The bytes the line adds at boot beyond its stage copies (priced by kv_state.bootCheck as resident).
pub fn residentBytes(f: *Forward, ln: *const pipe_run.Line) !u64 {
    return try pipe.stageRoleBytes(f.gpa, f.cfg, f.widths, ln.o1, ln.s, ln.rank) + bufferBytes(f.cfg, ln.s, ln.rank);
}
