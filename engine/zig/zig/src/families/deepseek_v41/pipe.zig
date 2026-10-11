//! TF_DSV41_PREFILL_PIPE (patches/0003 pipe.py, docs/ZIG-PREFILL-PIPE.md): CED's encoder pass as an assembly line over
//! the TP ranks. Rank s also holds encoder layers [B[s], B[s + 1]) at full width (only their TP-sharded tensors, under
//! the `P.` prefix: `rename`), and a long prompt's segments flow rank 0 -> W - 1: in round t, rank s runs segment
//! t - s through its layers as a world-1 program (block_prefill.emitStage), no collective inside it. The last stage ends
//! at the decoder's first layer's site and compressor, as the encoder pass does.
//!
//! This file holds the rules every rank applies alike (host only, tested in pipe_test.zig): the knobs, the stages and
//! their selections in and out, the round schedule, the rows a segment writes, the exchanges' layouts and the stage
//! weights' names. pipe_run.zig runs the rounds; block_prefill.emitStage emits a stage.
//!
//! - `TF_DSV41_PREFILL_PIPE=<rows>` (0 / unset: off): a replay-mode segment of at least that many rows opens the line
//!   (at least `min_seg`); while it is open every segment of its slot past `min_seg` - 1 rows goes through it.
//! - `TF_DSV41_PIPE_ROWS` = 2048 (default, Python's pchunk) | 4096 (the stages' segments under x3gm one block).
//! - `TF_DSV41_PREFILL_PIPE_BOUNDS` = explicit stages ("0,5,10,15,21"): world + 1 increasing values from 0 to the
//!   decoder's first layer + 1 (Python's).
//! - `TF_DSV41_PIPE_TRACE=1`: each line's rounds timed (pre-steps, stage, exchange), logged when it drains.

const std = @import("std");
const calls = @import("calls.zig");
const block = @import("block.zig");
const ced = @import("ced.zig");
const Config = @import("config.zig").Config;

pub const Error = error{ BadKnob, BadBounds };
pub const Get = *const fn ([]const u8) ?[]const u8;

const log = std.log.scoped(.dsv41);

/// the most ranks a line spans
pub const max_world: usize = 16;
/// A segment of the line has more rows than the mHC sites' split programs (block_prefill split_rows, 32): its stage
/// programs then never take the out-of-place boundary streams, so every stage starts at streams buffer 0 and
/// coefficient set 0, exactly where the one-process encoder pass is at that layer (pipe_emit_test.zig). Shorter
/// segments (a prompt's ragged tail) drain the line and run tensor parallel.
pub const min_seg: u32 = 33;
/// the weight prefix of a stage's full-width copies
pub const prefix = "P.";
/// the stage programs' namespace (block_prefill.ownCalls: window, layer and scratch roles; the slot's state shared)
pub const tag = "pp~";

pub const Settings = struct {
    /// a segment of at least this many rows opens the line
    min_rows: u32,
    /// the line's segment rows (2,048 or 4,096)
    rows: u32,
    world: u32,
    bounds: [max_world + 1]u32,
    trace: bool = false,

    pub fn boundsSlice(s: *const Settings) []const u32 {
        return s.bounds[0 .. s.world + 1];
    }

    pub fn stage(s: *const Settings, r: u32) Stage {
        std.debug.assert(r < s.world);
        return .{ .rank = r, .lo = s.bounds[r], .hi = s.bounds[r + 1], .d = s.bounds[s.world] - 1, .last = r + 1 == s.world };
    }

    /// The stage that runs layer L (L <= the decoder's first).
    pub fn owner(s: *const Settings, L: u32) u32 {
        var r: u32 = 0;
        while (r + 1 < s.world and L >= s.bounds[r + 1]) r += 1;
        return r;
    }
};

/// Rank `rank`'s stage: layers [lo, hi), the last through the decoder's first layer `d` (its site and compressor).
pub const Stage = struct {
    rank: u32,
    lo: u32,
    hi: u32,
    d: u32,
    last: bool,

    /// The layers its program runs.
    pub fn layers(st: Stage, buf: []u32) []const u32 {
        for (st.lo..st.hi, 0..) |L, i| buf[i] = @intCast(L);
        return buf[0 .. st.hi - st.lo];
    }

    /// The layers whose sharded tensors it loads at full width (the decoder's first runs replicated tensors alone).
    pub fn weightLayers(st: Stage, buf: []u32) []const u32 {
        const hi = @min(st.hi, st.d);
        for (st.lo..hi, 0..) |L, i| buf[i] = @intCast(L);
        return buf[0 .. hi - st.lo];
    }

    /// The kv sources its layers' compressors write (the round's rows exchange).
    pub fn sources(st: Stage, cfg: *const Config, buf: []u32) []const u32 {
        var n: usize = 0;
        for (st.lo..st.hi) |L| if (cfg.isKvSource(@intCast(L))) {
            buf[n] = @intCast(L);
            n += 1;
        };
        return buf[0..n];
    }

    /// The index source whose selection its first compressed layers reuse from an earlier stage ("w.sel<src>"), or null
    /// (its first compressed layer selects itself, or it has none).
    pub fn selIn(st: Stage, cfg: *const Config) ?u32 {
        for (st.lo..st.hi) |i| {
            const L: u32 = @intCast(i);
            if (cfg.compressRatio(L) == 0) continue;
            const src = cfg.indexSource(L) orelse return null;
            return if (src < st.lo) src else null;
        }
        return null;
    }

    /// The selection its output carries: the next stage's `selIn`.
    pub fn selOut(st: Stage, cfg: *const Config, bounds: []const u32) ?u32 {
        if (st.last) return null;
        const next: Stage = .{ .rank = st.rank + 1, .lo = bounds[st.rank + 1], .hi = bounds[st.rank + 2], .d = st.d, .last = st.rank + 2 == bounds.len - 1 };
        return next.selIn(cfg);
    }
};

fn envInt(get: Get, name: []const u8, default: u32) !u32 {
    const raw = std.mem.trim(u8, get(name) orelse return default, " ");
    return std.fmt.parseInt(u32, raw, 10) catch {
        log.warn("{s}={s}: expected a whole number", .{ name, raw });
        return error.BadKnob;
    };
}

/// The line's settings, or null when TF_DSV41_PREFILL_PIPE is unset or 0 (every rank reads the same environment).
pub fn settings(get: Get, cfg: *const Config, world: u32) !?Settings {
    const min_rows = try envInt(get, "TF_DSV41_PREFILL_PIPE", 0);
    if (min_rows == 0) return null;
    if (world < 2 or world > max_world) {
        log.warn("TF_DSV41_PREFILL_PIPE={d}: a pipeline runs over 2-{d} TP ranks (world {d})", .{ min_rows, max_world, world });
        return error.BadKnob;
    }
    if (min_rows < min_seg) {
        log.warn("TF_DSV41_PREFILL_PIPE={d}: 0 (off) or the rows that open a pipeline, at least {d}", .{ min_rows, min_seg });
        return error.BadKnob;
    }
    const rows = try envInt(get, "TF_DSV41_PIPE_ROWS", 2048);
    if (rows != 2048 and rows != 4096) {
        log.warn("TF_DSV41_PIPE_ROWS={d}: 2048 or 4096", .{rows});
        return error.BadKnob;
    }
    var s: Settings = .{ .min_rows = min_rows, .rows = rows, .world = world, .bounds = @splat(0) };
    _ = try boundsFor(cfg, world, get("TF_DSV41_PREFILL_PIPE_BOUNDS"), &s.bounds);
    if (get("TF_DSV41_PIPE_TRACE")) |v| s.trace = std.mem.eql(u8, std.mem.trim(u8, v, " "), "1");
    return s;
}

/// round(num / den) with ties to even (Python's round).
fn roundHalfEven(num: u64, den: u64) u64 {
    const q = num / den;
    const r = num % den;
    if (2 * r > den) return q + 1;
    if (2 * r == den) return q + (q & 1);
    return q;
}

/// Stage boundaries (pipe.bounds_for): the encoder layers [0, D) in `world` near-equal parts, the last stage through
/// layer D's site and compressor; `raw`: TF_DSV41_PREFILL_PIPE_BOUNDS.
pub fn boundsFor(cfg: *const Config, world: u32, raw: ?[]const u8, out: []u32) ![]const u32 {
    const d = ced.decoderStart(cfg) catch |e| {
        log.warn("TF_DSV41_PREFILL_PIPE: no decoder layer in this model ({t})", .{e});
        return error.BadBounds;
    };
    const b = out[0 .. world + 1];
    if (raw) |text| {
        var it = std.mem.tokenizeScalar(u8, text, ',');
        var n: usize = 0;
        while (it.next()) |v| : (n += 1) {
            if (n > world) return badBounds(text, world, d);
            b[n] = std.fmt.parseInt(u32, std.mem.trim(u8, v, " "), 10) catch return badBounds(text, world, d);
        }
        if (n != world + 1) return badBounds(text, world, d);
    } else {
        for (0..world) |s| b[s] = @intCast(roundHalfEven(@as(u64, d) * s, world));
        b[world] = d + 1;
    }
    if (b[0] != 0 or b[world] != d + 1) return badBounds(raw orelse "(default)", world, d);
    for (0..world) |i| if (b[i] >= b[i + 1]) return badBounds(raw orelse "(default)", world, d);
    return b;
}

fn badBounds(text: []const u8, world: u32, d: u32) error{BadBounds} {
    log.warn("TF_DSV41_PREFILL_PIPE_BOUNDS={s}: {d} increasing stage starts from 0, then {d} (the decoder's first layer + 1)", .{ text, world, d + 1 });
    return error.BadBounds;
}

/// The compressed rows [a, b) of a ratio-`ratio` kv source a segment [start, start + n) can write, the one before it
/// included (pipe._rows_of: a ratio-2 row straddling the segment's first position is rewritten with the carry).
pub fn rowsOf(ratio: u32, start: u64, n: u32) [2]u64 {
    const r: u64 = @max(1, ratio);
    const a = start / r;
    return .{ a -| 1, (start + n) / r };
}

/// One segment of the line: positions [start, start + n) of its slot.
pub const Seg = struct { start: u64, n: u32 };

/// The rounds since the line opened (pipe.Pipeline.entered): round t's new segment or null (a drain round); in round t
/// stage s runs the segment entered in round t - s.
pub const Schedule = struct {
    world: u32,
    ring: [max_world]?Seg = @splat(null),
    /// rounds entered
    t: u64 = 0,

    /// Round t's new segment (null: a drain round): the next round.
    pub fn enter(s: *Schedule, seg: ?Seg) void {
        s.ring[@intCast(s.t % max_world)] = seg;
        s.t += 1;
    }

    /// The current round (the last entered).
    pub fn round(s: *const Schedule) u64 {
        return s.t - 1;
    }

    /// The segment stage `stage_` runs in the current round.
    pub fn segOf(s: *const Schedule, stage_: u32) ?Seg {
        if (s.t == 0 or stage_ > s.t - 1) return null;
        return s.ring[@intCast((s.t - 1 - stage_) % max_world)];
    }

    /// A segment is still in a stage: one of the last world - 1 rounds entered one.
    pub fn busy(s: *const Schedule) bool {
        var k: u64 = 0;
        while (k + 1 < s.world and k < s.t) : (k += 1) if (s.ring[@intCast((s.t - 1 - k) % max_world)] != null) return true;
        return false;
    }

    pub fn clear(s: *Schedule) void {
        s.* = .{ .world = s.world };
    }
};

fn a16(x: u64) u64 {
    return std.mem.alignForward(u64, x, 16);
}

/// The round's rows exchange (one all-gather, a stride a rank): rank q's part holds, for each kv source its stage
/// writes (in layer order), the comp rows `rowsOf` names (584-byte pool records) then their index keys (132 bytes),
/// each part 16-byte aligned, sized for a whole segment of the line (rows / ratio + 2 rows).
pub const RowsLayout = struct {
    comp_bytes: [max_world]u64 = @splat(0),
    ik_bytes: [max_world]u64 = @splat(0),
    used: [max_world]u64 = @splat(0),
    stride: u64 = 0,

    /// Rows a segment of the line can write into a ratio-`ratio` source.
    pub fn cap(s: Settings, ratio: u32) u64 {
        return s.rows / @max(1, ratio) + 2;
    }

    pub fn of(cfg: *const Config, s: Settings) RowsLayout {
        var lay: RowsLayout = .{};
        var buf: [64]u32 = undefined;
        for (0..s.world) |q| {
            for (s.stage(@intCast(q)).sources(cfg, &buf)) |L| {
                const c = cap(s, cfg.compressRatio(L));
                lay.comp_bytes[q] += c * block.Pool.row_bytes;
                lay.ik_bytes[q] += c * block.Pool.ik_bytes;
                lay.used[q] += a16(c * block.Pool.row_bytes) + a16(c * block.Pool.ik_bytes);
            }
            lay.stride = @max(lay.stride, lay.used[q]);
        }
        return lay;
    }

    /// Source L's comp rows and index keys within its owner's part: their byte offsets.
    pub fn at(cfg: *const Config, s: Settings, L: u32) struct { comp: u64, ik: u64 } {
        var buf: [64]u32 = undefined;
        var off: u64 = 0;
        for (s.stage(s.owner(L)).sources(cfg, &buf)) |x| {
            const c = cap(s, cfg.compressRatio(x));
            if (x == L) return .{ .comp = off, .ik = off + a16(c * block.Pool.row_bytes) };
            off += a16(c * block.Pool.row_bytes) + a16(c * block.Pool.ik_bytes);
        }
        unreachable;
    }
};

/// The drain's hand-off (one all-gather, a stride a rank): rank q's part holds each of its encoder layers' SWA ring (the
/// slot's 2 x window rows: values then scales), each of its ratio-2 kv sources' carry (fp32 [kv | gate] of position
/// pos - 1), and the last stage's CED stash ring (ced.roles), each part 16-byte aligned.
pub const FinalsLayout = struct {
    used: [max_world]u64 = @splat(0),
    stride: u64 = 0,

    pub fn ringBytes(cfg: *const Config) [2]u64 {
        const r: u64 = 2 * @as(u64, cfg.window);
        return .{ r * 576, r * 8 };
    }

    pub fn carryBytes(cfg: *const Config) u64 {
        return 2 * @as(u64, cfg.head_dim) * 4;
    }

    /// ced.roles' bytes: the streams, the normed input, pre / post / comb, `ced.ring` rows each.
    pub fn stashBytes(cfg: *const Config) [5]u64 {
        const R: u64 = ced.ring;
        const D: u64 = cfg.hidden;
        return .{ R * 4 * D * 2, R * D * 2, R * 4 * 4, R * 4 * 4, R * 16 * 4 };
    }

    pub fn of(cfg: *const Config, s: Settings) FinalsLayout {
        var lay: FinalsLayout = .{};
        const ring = ringBytes(cfg);
        for (0..s.world) |q| {
            const st = s.stage(@intCast(q));
            for (st.lo..@min(st.hi, st.d)) |L| {
                lay.used[q] += a16(ring[0]) + a16(ring[1]);
                if (cfg.isKvSource(@intCast(L)) and cfg.compressRatio(@intCast(L)) == 2) lay.used[q] += a16(carryBytes(cfg));
            }
            if (st.last) for (stashBytes(cfg)) |b| {
                lay.used[q] += a16(b);
            };
            lay.stride = @max(lay.stride, lay.used[q]);
        }
        return lay;
    }
};

// -- the stage weights' names --------------------------------------------------------------------------------------

/// A layer tensor's field (names.zig's TP splits): col (wq_b, the routed and shared experts' gate / up), row (wo_b, their
/// down), own groups (wo_a.<g>) or a per-head native (attn.sink). Everything else is replicated: the stage reads the
/// TP copy.
pub fn sharded(field: []const u8) bool {
    const whole = [_][]const u8{ "attn.wq_b", "attn.wo_b", "attn.sink", "moe.w1", "moe.w2", "moe.w3" };
    for (whole) |w| if (std.mem.eql(u8, field, w) or (std.mem.startsWith(u8, field, w) and field.len > w.len and field[w.len] == '.')) return true;
    return std.mem.startsWith(u8, field, "attn.wo_a.") or std.mem.startsWith(u8, field, "moe.shared.");
}

/// "L<i>.<field>" -> the field (null: not a layer tensor's name).
fn fieldOf(name: []const u8) ?[]const u8 {
    if (name.len < 3 or name[0] != 'L') return null;
    const dot = std.mem.indexOfScalar(u8, name, '.') orelse return null;
    if (dot == 1) return null;
    for (name[1..dot]) |c| if (!std.ascii.isDigit(c)) return null;
    return name[dot + 1 ..];
}

/// A stage program's role: a sharded tensor's weight and run-time forms (`s.<w>.lanes` / `.T`), the expert tables
/// (`s.L<i>.ex.*`) and x3gm's pointer tables (`s.L<i>.gm.t{g,u,d}`) take the `P.` prefix (the stage's full-width
/// copies); everything else stays (the replicated tensors and their forms, the slot's state, scratch).
pub fn renameRole(a: std.mem.Allocator, r: calls.Role) !calls.Role {
    switch (r) {
        .weight => |w| if (fieldOf(w)) |f| if (sharded(f)) return .{ .weight = try std.fmt.allocPrint(a, prefix ++ "{s}", .{w}) },
        .buf => |b| if (std.mem.startsWith(u8, b, "s.")) {
            const rest = b[2..];
            if (fieldOf(rest)) |f| {
                const mine = std.mem.startsWith(u8, f, "ex.") or std.mem.eql(u8, f, "gm.tg") or std.mem.eql(u8, f, "gm.tu") or std.mem.eql(u8, f, "gm.td") or
                    (std.mem.endsWith(u8, f, ".lanes") and sharded(f[0 .. f.len - ".lanes".len])) or
                    (std.mem.endsWith(u8, f, ".T") and sharded(f[0 .. f.len - ".T".len]));
                if (mine) return .{ .buf = try std.fmt.allocPrint(a, "s." ++ prefix ++ "{s}", .{rest}) };
            }
        },
        .empty => {},
    }
    return r;
}

fn renameArg(a: std.mem.Allocator, x: calls.Arg) !calls.Arg {
    return switch (x) {
        .t, .opaque_table => |t| blk: {
            var u = t;
            u.role = try renameRole(a, t.role);
            break :blk if (x == .t) .{ .t = u } else .{ .opaque_table = u };
        },
        .list => |l| blk: {
            const out = try a.alloc(calls.Arg, l.len);
            for (l, out) |y, *z| z.* = try renameArg(a, y);
            break :blk .{ .list = out };
        },
        else => x,
    };
}

/// `cs` with every role through `renameRole`.
pub fn rename(a: std.mem.Allocator, cs: []const calls.Call) ![]calls.Call {
    const out = try a.alloc(calls.Call, cs.len);
    for (cs, out) |c, *x| {
        x.* = c;
        const args = try a.alloc(calls.Named, c.args.len);
        for (c.args, args) |y, *z| z.* = .{ .name = y.name, .arg = try renameArg(a, y.arg) };
        x.args = args;
    }
    return out;
}

/// The stage programs' dense widths: the emitter asks a group's K2 by its bare name (`L<i>.attn.wo_a.<g>` for every one
/// of the model's groups at world 1, where a TP rank holds only its own), and the renaming happens after it. Every
/// stage copy's width under its bare name, where the TP rank has none (pipe_load.zig loaded "P.<name>").
pub fn widthsFromCopies(a: std.mem.Allocator, w: *block.Widths) !void {
    var add: std.ArrayList(struct { []const u8, u32 }) = .empty;
    defer add.deinit(a);
    var it = w.dense.iterator();
    while (it.next()) |kv| if (std.mem.startsWith(u8, kv.key_ptr.*, prefix) and w.dense.get(kv.key_ptr.*[prefix.len..]) == null) {
        try add.append(a, .{ kv.key_ptr.*[prefix.len..], kv.value_ptr.* });
    };
    for (add.items) |x| try w.dense.put(a, try a.dupe(u8, x[0]), x[1]);
}

/// Every `wo_a` group's width from group 0's where a listing has only a TP rank's groups (one checkpoint tensor a layer,
/// one K2: the host listings aot-needs and the tests read).
pub fn fillGroups(a: std.mem.Allocator, w: *block.Widths, cfg: *const Config) !void {
    var nb: [64]u8 = undefined;
    for (0..cfg.allLayers()) |L| {
        const k2 = w.dense.get(try std.fmt.bufPrint(&nb, "L{d}.attn.wo_a.0", .{L})) orelse continue;
        for (1..cfg.o_groups) |g| {
            const name = try std.fmt.bufPrint(&nb, "L{d}.attn.wo_a.{d}", .{ L, g });
            if (w.dense.get(name) == null) try w.dense.put(a, try a.dupe(u8, name), k2);
        }
    }
}

/// The (rows, start) of the stage programs the boot plans. Not planned yet.
pub fn planned(a: std.mem.Allocator, cfg: *const Config, w: *const block.Widths, o1: block.Options, s: Settings, rank: u32) ![]const [2]i64 {
    _ = cfg;
    _ = w;
    _ = o1;
    _ = s;
    _ = rank;
    return a.alloc([2]i64, 0);
}

/// A stage program's options: the forward's at world 1, rank 0 (every head, every wo_a group, every expert column), the
/// line's segment rows (4,096: x3gm one block, as TF_DSV41_PF_4K), and none of the TP-only or per-request paths (taps,
/// TBO pairs, exchange pieces, image rows: an image segment never enters the line).
pub fn stageOptions(o: block.Options, rows: u32) block.Options {
    var x = o;
    x.world = 1;
    x.rank = 0;
    x.taps = false;
    x.pf_tbo = false;
    x.pf_overlap = 0;
    x.pf_overlap_site = false;
    x.image_keep = false;
    x.image_rows = 0;
    x.prefill_rows = rows;
    x.pf4k = rows > 2048;
    return x;
}
