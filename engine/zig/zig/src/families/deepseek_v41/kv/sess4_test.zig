//! Sessions over 4 live slots against prod: sess4_golden.json is prod's own session store, KV pool and NVMe tier
//! (Python 8474f31) driven on the CPU as prod's batcher drives them (tools/zig/dsv41_sess4/golden.py). This test runs
//! the same steps on the Zig store, pool and tier with the rules the GPU side uses (sess.zig, sessions_gpu.zig's
//! policy), and every step must decide as prod did: the request's pages and spills, the tier it resumes from and how
//! far, the snapshot's RAM measure and duplicate, the RAM tier's entries, the NVMe tier's files, the pool's free and
//! unreserved pages. Slots interleave: a chat resumes while others hold pages, reservations and shared RAM pages.
//! Every resume is checked row for row against what the slot that saved it wrote (a fresh prefill's rows: a function
//! of the position's id), and the restored bounded state against the saved one.

const std = @import("std");
const testing = std.testing;
const sessions = @import("sessions");
const sess = @import("sess.zig");

const Pool = sessions.Pool;
const Slot = sessions.Slot;
const Family = sessions.Family;
const Store = sessions.Store;
const Bounded = sessions.store.Bounded;
const Blob = sessions.disk.Blob;

const golden_json = @embedFile("sess4_golden.json");

const Saved = struct { at: u64, nbytes: u64, ds_rows: u64, dup: bool };
const Step = struct {
    op: []const u8,
    label: []const u8 = "",
    slot: u32 = 0,
    chat: u32 = 0,
    branch: u32 = 0,
    n: u64 = 0,
    max_new: u64 = 0,
    reply: u64 = 0,
    tier: []const u8 = "",
    cached: u64 = 0,
    need: u32 = 0,
    want: u32 = 0,
    spills: []const []const u8 = &.{},
    saved: ?Saved = null,
    wait: ?bool = null,
    ram: []const []const u8,
    disk: []const []const u8,
    available: u32,
    free: u32,
};
const MeasureIn = struct { layers: u64, decoder_start: u64, window: u64, carries: u64, head_dim: u64, hidden: u64, hc: u64, ds_blocks: u64, keep: u64, row_bytes: u64, lookback: u64 };
const Golden = struct {
    tag: u32,
    page: u32,
    pages: u32,
    slots: u32,
    capacity: u64,
    slack: u64,
    ram_bytes: u64,
    disk_min: u64,
    measure: MeasureIn,
    steps: []const Step,
};

/// The pool's families as the test stores them (any sizes: the decisions are in pages; the rows are checked)
const fams = [_]Family{
    .{ .name = "comp.2", .ratio = 2, .row_bytes = 584 },
    .{ .name = "index_k.2", .ratio = 2, .row_bytes = 132 },
    .{ .name = "comp.20", .ratio = 1, .row_bytes = 584 },
};

/// A snapshot's bounded state in host memory: what the slot's state at `pos` carries for the next resume (the drafter's
/// rows, the stash's first position), its RAM measure the one prod's arrays have.
const HostBounded = struct {
    gpa: std.mem.Allocator,
    pos: u64,
    ds_rows: u64,
    stash_from: u64,
    measure: u64,

    const vt: Bounded.VTable = .{ .nbytes = nbytes, .blobs = blobs, .pieces = pieces, .release = release };
    fn pieces(_: *anyopaque) u32 {
        return 1;
    }

    fn make(gpa: std.mem.Allocator, b: HostBounded) !Bounded {
        const h = try gpa.create(HostBounded);
        h.* = b;
        h.gpa = gpa;
        return .{ .ptr = h, .vtable = &vt };
    }
    fn nbytes(p: *anyopaque) u64 {
        const h: *HostBounded = @ptrCast(@alignCast(p));
        return h.measure;
    }
    fn blobs(p: *anyopaque, gpa: std.mem.Allocator) anyerror![]Blob {
        const h: *HostBounded = @ptrCast(@alignCast(p));
        var b: [24]u8 = undefined;
        std.mem.writeInt(u64, b[0..8], h.pos, .little);
        std.mem.writeInt(u64, b[8..16], h.ds_rows, .little);
        std.mem.writeInt(u64, b[16..24], h.stash_from, .little);
        const out = try gpa.alloc(Blob, 1);
        out[0] = .{ .name = try gpa.dupe(u8, "slot"), .bytes = try gpa.dupe(u8, &b) };
        return out;
    }
    fn release(p: *anyopaque) void {
        const h: *HostBounded = @ptrCast(@alignCast(p));
        h.gpa.destroy(h);
    }
};

/// A chat's ids (golden.py chat_ids): its own stream; a branch rewrites its last 100 tokens.
fn chatIds(out: []i32, chat: u32, branch: u32) void {
    const n = out.len;
    for (out, 0..) |*t, i| t.* = @intCast(chat * 100_000 + i);
    if (branch > 0) {
        const cut = n - 100;
        for (out[cut..], 0..) |*t, i| t.* = @intCast(chat * 100_000 + 50_000 + branch * 1000 + i);
    }
}

/// A row's bytes: a function of the id at its first position (what a fresh prefill of the same prefix writes there).
fn rowByte(id: i32, pos: u64, fi: usize, i: usize) u8 {
    return @truncate(@as(u64, @bitCast(@as(i64, id))) *% 31 +% pos *% 3 +% i *% 7 +% fi);
}

const World = struct {
    gpa: std.mem.Allocator,
    g: Golden,
    pool: Pool,
    host: sessions.HostPool,
    disk: *sessions.Disk,
    store: Store,
    slots: [4]*Slot = undefined,
    /// per slot: drafter.valid and the stash's first position (sessions_gpu.zig keeps the same)
    valid: [4]u64 = @splat(0),
    stash_from: [4]u64 = @splat(0),
    names: std.AutoHashMapUnmanaged(u32, []const u8) = .empty,
    arena: std.heap.ArenaAllocator,

    fn measure(w: *const World) sess.Measure {
        const m = w.g.measure;
        return .{ .rings = m.decoder_start, .window = m.window, .carries = m.carries, .head_dim = m.head_dim, .hidden = m.hidden, .hc = m.hc, .ds_blocks = m.ds_blocks, .replay = true };
    }

    /// Writes the rows of positions [from, to) of the slot's pages (each family's rows that start there).
    fn write(w: *World, s: *const Slot, ids: []const i32, from: u64, to: u64) void {
        const page = w.pool.page;
        for (fams, 0..) |f, fi| {
            var p = from - from % f.ratio + (if (from % f.ratio != 0) f.ratio else 0);
            while (p < to) : (p += f.ratio) {
                const pg = s.mapped()[@intCast(p / page)];
                const v = w.host.pageOf(@intCast(fi), w.pool.familyPage(f, pg));
                const at: usize = @intCast((p % page) / f.ratio * f.row_bytes);
                for (v[at..][0..f.row_bytes], 0..) |*b, i| b.* = rowByte(ids[@intCast(p)], p, fi, i);
            }
        }
    }

    /// Every row below `pos` of the slot's pages holds what a fresh prefill of `ids` writes there.
    fn check(w: *World, s: *const Slot, ids: []const i32, pos: u64) !void {
        const page = w.pool.page;
        for (fams, 0..) |f, fi| {
            var p: u64 = 0;
            while (p + f.ratio <= pos) : (p += f.ratio) {
                const pg = s.mapped()[@intCast(p / page)];
                const v = w.host.pageOf(@intCast(fi), w.pool.familyPage(f, pg));
                const at: usize = @intCast((p % page) / f.ratio * f.row_bytes);
                for (v[at..][0..f.row_bytes], 0..) |b, i| if (b != rowByte(ids[@intCast(p)], p, fi, i)) {
                    std.debug.print("row at position {d} (family {s}) is not the prompt's\n", .{ p, f.name });
                    return error.WrongRow;
                };
            }
        }
    }

    fn nameOf(w: *World, id: u32) []const u8 {
        return w.names.get(id) orelse "?";
    }

    fn sortNames(list: [][]const u8) void {
        std.mem.sort([]const u8, list, {}, struct {
            fn less(_: void, a: []const u8, b: []const u8) bool {
                return std.mem.order(u8, a, b) == .lt;
            }
        }.less);
    }

    /// The RAM tier's entries, the NVMe tier's files, the pool's pages == prod's after this step.
    fn expectState(w: *World, st: *const Step) !void {
        const a = w.arena.allocator();
        var ram: std.ArrayList([]const u8) = .empty;
        var disk: std.ArrayList([]const u8) = .empty;
        for (w.store.index.entries.items, 0..) |e, id| if (e.live) {
            if (e.ram) try ram.append(a, w.nameOf(@intCast(id)));
            if (e.extra.file) try disk.append(a, w.nameOf(@intCast(id)));
        };
        sortNames(ram.items);
        sortNames(disk.items);
        const j = struct {
            fn of(x: std.mem.Allocator, l: []const []const u8) []const u8 {
                return std.mem.join(x, " ", l) catch "?";
            }
        }.of;
        errdefer std.debug.print("step '{s}' {s} slot {d}: RAM {s} (prod {s}), NVMe {s} (prod {s}), available {d} (prod {d}), free {d} (prod {d})\n", .{ st.op, st.label, st.slot, j(a, ram.items), j(a, st.ram), j(a, disk.items), j(a, st.disk), w.pool.available(), st.available, w.pool.freePages(), st.free });
        try testing.expectEqual(st.ram.len, ram.items.len);
        for (st.ram, ram.items) |x, y| try testing.expectEqualStrings(x, y);
        try testing.expectEqual(st.disk.len, disk.items.len);
        for (st.disk, disk.items) |x, y| try testing.expectEqualStrings(x, y);
        try testing.expectEqual(st.available, w.pool.available());
        try testing.expectEqual(st.free, w.pool.freePages());
    }

    fn admit(w: *World, st: *const Step) !void {
        const a = w.arena.allocator();
        const ids = try a.alloc(i32, @intCast(st.n));
        chatIds(ids, st.chat, st.branch);
        const hit = try w.store.find(w.g.tag, ids);
        var spills: std.ArrayList(u32) = .empty;
        defer spills.deinit(w.gpa);
        const keep: []const u32 = if (hit) |h| &.{h.id} else &.{};
        const plan = try sess.admit(&w.store, hit, st.n, st.max_new, w.g.capacity, keep, 0, &spills);
        if (std.mem.eql(u8, st.op, "probe")) {
            try testing.expectEqual(st.wait.?, plan.wait);
            try testing.expectEqual(st.need, plan.need);
            try testing.expectEqual(st.want, plan.want);
            return;
        }
        errdefer std.debug.print("admit '{s}': tier {s} cached {d} need {d} want {d} spills {d}\n", .{ st.label, st.tier, st.cached, plan.need, plan.want, spills.items.len });
        try testing.expect(!plan.wait);
        try testing.expectEqual(st.need, plan.need);
        try testing.expectEqual(st.want, plan.want);
        try testing.expectEqual(st.spills.len, spills.items.len);
        var names: std.ArrayList([]const u8) = .empty;
        for (spills.items) |id| try names.append(a, w.nameOf(id));
        sortNames(names.items);
        for (st.spills, names.items) |x, y| try testing.expectEqualStrings(x, y);
        // the leader's spills in its order, then the slot's reservation (rounds.py `_admit`: reserve, reset, restore)
        for (spills.items) |id| _ = try w.store.evict(id);
        try w.store.settleAll();
        const s = w.slots[st.slot];
        try testing.expectEqual(@as(u32, 0), s.len);
        s.reserve(plan.need);
        w.valid[st.slot] = 0;
        w.stash_from[st.slot] = 0;
        var cached: u64 = 0;
        var tier: []const u8 = "none";
        if (hit) |h| {
            if (h.ram) {
                const b = try w.store.restoreRam(h.id, s);
                const hb: *HostBounded = @ptrCast(@alignCast(b.ptr));
                try testing.expectEqual(h.pos, hb.pos);
                w.valid[st.slot] = hb.pos - hb.ds_rows;
                w.stash_from[st.slot] = hb.stash_from;
                tier = "ram";
            } else {
                const j = try w.store.beginRestore(h.id, s);
                var got = (try w.store.settle(j)).?;
                defer got.deinit(w.gpa);
                const b = got.blobs[0].bytes;
                try testing.expectEqual(h.pos, std.mem.readInt(u64, b[0..8], .little));
                w.valid[st.slot] = h.pos - std.mem.readInt(u64, b[8..16], .little);
                w.stash_from[st.slot] = std.mem.readInt(u64, b[16..24], .little);
                tier = "disk";
            }
            cached = h.pos;
            try w.check(s, ids, cached);
        }
        try testing.expectEqualStrings(st.tier, tier);
        try testing.expectEqual(st.cached, cached);
        // the prompt to its replay point, the snapshot there (rounds.py `_save(slot, "prompt")`), the rest of it, the reply
        var at = cached;
        if (sess.savePoint(st.n, cached)) |save_at| {
            try s.ensure(save_at);
            w.write(s, ids, at, save_at);
            at = save_at;
            const ds = sess.dsRows(save_at, w.valid[st.slot], w.g.measure.window);
            const stash = sess.stashRows(save_at, w.stash_from[st.slot]);
            const m = w.measure().bytes(save_at, stash, ds);
            const want = st.saved orelse return error.ProdSavedNothing;
            try testing.expectEqual(want.at, save_at);
            try testing.expectEqual(want.ds_rows, ds);
            try testing.expectEqual(want.nbytes, m);
            const b = try HostBounded.make(w.gpa, .{ .gpa = w.gpa, .pos = save_at, .ds_rows = ds, .stash_from = save_at - stash, .measure = m });
            const id = try w.store.save(s, b, ids[0..@intCast(save_at)], w.g.tag, 1);
            try testing.expectEqual(want.dup, id == null);
            if (id) |x| try w.names.put(w.gpa, x, try std.fmt.allocPrint(a, "c{d}b{d}@{d}", .{ st.chat, st.branch, save_at }));
            try w.store.settleAll();
        } else try testing.expect(st.saved == null);
        try s.ensure(st.n - 1);
        w.write(s, ids, at, st.n - 1);
        try s.ensure(st.n + st.reply);
    }
};

test "4 slots, interleaved resumes, spills, RAM trims and NVMe parks decide as prod's store, pool and tier" {
    const gpa = testing.allocator;
    var parsed = try std.json.parseFromSlice(Golden, gpa, golden_json, .{ .ignore_unknown_fields = true });
    defer parsed.deinit();
    const g = parsed.value;
    try testing.expectEqual(@as(u64, sess.slack), g.slack);
    try testing.expectEqual(sess.row_bytes, g.measure.row_bytes);
    try testing.expectEqual(sess.keep, g.measure.keep);
    try testing.expectEqual(sess.lookback, g.measure.lookback);
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(testing.io, "tier");
    const root = try tmp.dir.realPathFileAlloc(testing.io, "tier", gpa);
    defer gpa.free(root);

    var w: World = .{ .gpa = gpa, .g = g, .pool = undefined, .host = undefined, .disk = undefined, .store = undefined, .arena = .init(gpa) };
    defer w.arena.deinit();
    defer w.names.deinit(gpa);
    w.pool = try Pool.init(gpa, .{ .families = &fams, .page = g.page }, @as(u64, g.pages) * g.page, 1, 0);
    defer w.pool.deinit();
    w.host = try sessions.HostPool.init(gpa, &w.pool);
    defer w.host.deinit();
    w.disk = try sessions.Disk.open(gpa, testing.io, .{ .root = root, .compat = @splat(9), .min_tokens = g.disk_min, .chunk = 1 << 20, .staging = 4, .lanes = 2, .direct = false });
    defer w.disk.close();
    // prod's store: its RAM budget, no residency budget, no covered-entry or delta rules (sessions_gpu.optionsFromEnv)
    w.store = try Store.init(gpa, testing.io, &w.pool, w.host.store(), w.disk, .{ .ram_bytes = g.ram_bytes, .skip_covered = false, .delta = false });
    defer w.store.deinit();
    for (&w.slots) |*s| s.* = try w.pool.newSlot(g.capacity);
    try testing.expectEqual(@as(usize, g.slots), w.slots.len);
    var spills: u32 = 0;
    var hits: [2]u32 = .{ 0, 0 };
    for (g.steps) |*st| {
        if (std.mem.eql(u8, st.op, "admit") or std.mem.eql(u8, st.op, "probe")) {
            try w.admit(st);
            spills += @intCast(st.spills.len);
            if (std.mem.eql(u8, st.tier, "ram")) hits[0] += 1;
            if (std.mem.eql(u8, st.tier, "disk")) hits[1] += 1;
        } else if (std.mem.eql(u8, st.op, "finish")) {
            _ = try w.slots[st.slot].releaseAll();
        } else return error.UnknownStep;
        try w.expectState(st);
    }
    // the golden exercises what it claims: spills, both tiers' resumes, parks and drops
    try testing.expect(spills >= 2 and hits[0] >= 2 and hits[1] >= 3);
    try testing.expect(w.store.stats.parks >= 3 and w.store.stats.drops >= 2);
}

test "a full NVMe tier refuses the admission pre-broadcast (the ParkFailed crash loop's gate)" {
    // the 2026-10-10 review's F2: with a spill park's failure process-fatal (ParkFailed, tp/replay.zig),
    // accepts() pricing tokens only (sessions/disk.zig) meant a tier full on every rank turned one
    // admission into a fleet crash loop — the parks met the full tier at write time, every follower died,
    // and the restart's empty start freed nothing (boots never scan the tier), so the next admission's
    // spills met the same wall. Rank 0's planner now prices the spills' files against the tier's real
    // free space (Store.tierAdmits, over Disk.freeBytes' statfs) before anything is broadcast and refuses
    // the request instead: every rank's store untouched, the failure request-level.
    const gpa = testing.allocator;
    const page = 64;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(testing.io, "tier");
    const root = try tmp.dir.realPathFileAlloc(testing.io, "tier", gpa);
    defer gpa.free(root);
    var pool = try Pool.init(gpa, .{ .families = &fams, .page = page }, 128 * page, 1, 0);
    defer pool.deinit();
    var host = try sessions.HostPool.init(gpa, &pool);
    defer host.deinit();
    const disk = try sessions.Disk.open(gpa, testing.io, .{ .root = root, .compat = @splat(9), .min_tokens = 1, .chunk = 64 << 10, .staging = 4, .lanes = 2, .direct = false });
    defer disk.close();
    var store = try Store.init(gpa, testing.io, &pool, host.store(), disk, .{});
    defer store.deinit();
    // a chat idle in RAM, 80 of the pool's 128 pages
    const s = try pool.newSlot(128 * page);
    const ids = try gpa.alloc(i32, 80 * page);
    defer gpa.free(ids);
    chatIds(ids, 1, 0);
    try s.ensure(ids.len);
    const id = (try store.save(s, try HostBounded.make(gpa, .{ .gpa = undefined, .pos = 5120, .ds_rows = 0, .stash_from = 0, .measure = 24 }), ids, 1, 0)).?;
    _ = try s.releaseAll();
    var spills: std.ArrayList(u32) = .empty;
    defer spills.deinit(gpa);
    const held: []const u32 = &.{};
    // the statfs probe itself answers on a real tier dir
    try testing.expect(try disk.freeBytes() > 0);
    // ample space: the admission plans and spills the idle chat, as before
    disk.opts.free_bytes = 1 << 40;
    const plan = try sess.admit(&store, null, 60 * page, 0, 128 * page, held, 0, &spills);
    try testing.expect(!plan.wait);
    try testing.expectEqualSlices(u32, &.{id}, spills.items);
    const need = store.parkBytes(id);
    try testing.expect(need > 0);
    // the boundary: exactly the spills' bytes fits; one byte less refuses the request
    disk.opts.free_bytes = need;
    _ = try sess.admit(&store, null, 60 * page, 0, 128 * page, held, 0, &spills);
    disk.opts.free_bytes = need - 1;
    const free_pages = pool.freePages();
    try testing.expectError(error.TierFull, sess.admit(&store, null, 60 * page, 0, 128 * page, held, 0, &spills));
    // the refusal decided nothing: the entry still holds its pages in RAM, the tier untouched
    try testing.expect(store.entry(id).ram);
    try testing.expectEqual(free_pages, pool.freePages());
    try testing.expectEqual(@as(u64, 0), disk.used);
    // the crash loop's condition itself — a full tier on every rank — is the same request-level refusal
    disk.opts.free_bytes = 0;
    try testing.expectError(error.TierFull, sess.admit(&store, null, 60 * page, 0, 128 * page, held, 0, &spills));
    // what still serves on a full tier: an admission the pool holds without spilling never parks
    const small = try sess.admit(&store, null, 10 * page, 0, 128 * page, held, 0, &spills);
    try testing.expect(!small.wait);
    try testing.expectEqual(@as(usize, 0), spills.items.len);
    // an empty spill list never refuses, probed or not
    try store.tierAdmits(&.{}, 0);
}

test "a park's price covers the file it writes (the free == payload boundary the payload sum let through)" {
    // the 2026-10-10 re-review's F2: parkBytes summed the payload only, but the file adds the padded
    // header (its per-chunk digest slots) and every segment's 4 KiB chunk strides and tails
    // (sessions/format.zig's Layout.build), so the gate's pinned boundary (free == price) could still
    // ENOSPC at write time — the fatal ParkFailed loop for exactly that slice. The price is the file
    // now: free == payload refuses, free == price fits the write.
    const gpa = testing.allocator;
    const page = 64;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(testing.io, "tier");
    const root = try tmp.dir.realPathFileAlloc(testing.io, "tier", gpa);
    defer gpa.free(root);
    var pool = try Pool.init(gpa, .{ .families = &fams, .page = page }, 128 * page, 1, 0);
    defer pool.deinit();
    var host = try sessions.HostPool.init(gpa, &pool);
    defer host.deinit();
    const disk = try sessions.Disk.open(gpa, testing.io, .{ .root = root, .compat = @splat(9), .min_tokens = 1, .chunk = 64 << 10, .staging = 4, .lanes = 2, .direct = false });
    defer disk.close();
    var store = try Store.init(gpa, testing.io, &pool, host.store(), disk, .{});
    defer store.deinit();
    const s = try pool.newSlot(128 * page);
    const ids = try gpa.alloc(i32, 80 * page);
    defer gpa.free(ids);
    chatIds(ids, 1, 0);
    try s.ensure(ids.len);
    const id = (try store.save(s, try HostBounded.make(gpa, .{ .gpa = undefined, .pos = 5120, .ds_rows = 0, .stash_from = 0, .measure = 24 }), ids, 1, 0)).?;
    _ = try s.releaseAll();
    // the payload sum the price used to be: the ids, this rank's pages of every family, the bounded state
    var payload: u64 = 4 * @as(u64, ids.len) + 24;
    for (fams) |f| payload += 80 * f.pageBytes(page);
    const price = store.parkBytes(id);
    try testing.expect(price > payload);
    // the boundary that escaped: free space the payload sum called enough, the file does not fit
    var spills: std.ArrayList(u32) = .empty;
    defer spills.deinit(gpa);
    const held: []const u32 = &.{};
    disk.opts.free_bytes = payload;
    try testing.expectError(error.TierFull, sess.admit(&store, null, 60 * page, 0, 128 * page, held, 0, &spills));
    // the boundary that holds: the price itself fits — and the write below proves it does
    disk.opts.free_bytes = price;
    _ = try sess.admit(&store, null, 60 * page, 0, 128 * page, held, 0, &spills);
    try testing.expectEqualSlices(u32, &.{id}, spills.items);
    // the file a real park writes: bigger than the payload, no bigger than the price, and close to it
    disk.opts.free_bytes = null;
    _ = try store.evict(id);
    try testing.expect(store.entry(id).extra.file);
    const real = disk.used;
    try testing.expect(real > payload);
    try testing.expect(price >= real);
    try testing.expect(price - real < 2 * 4096);
}

test "a save's RAM-budget trims meet the tier gate pre-broadcast (the save-path crash loop's gate)" {
    // the 2026-10-10 re-review's F1: the tier gate priced admission spills only. A save's own trims
    // (Store.save's trimRam/enforce) park ungated, and the save op replays on every follower, so a
    // tier full on every rank crash-looped through the save path with the admission gate green (the
    // probe's own TF_DSV41_SESSION_RAM_MIB knob drives exactly that path). Rank 0 now prices the
    // save's parks through the same predicate (kv/sess.zig's saveAdmits over Store.planSaveParks +
    // tierAdmits) before the op is broadcast and ends the request instead; a follower never gates
    // (its probe is its own tier's — a follower-local full tier keeps its clean ParkFailed restart).
    const gpa = testing.allocator;
    const page = 64;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(testing.io, "tier");
    const root = try tmp.dir.realPathFileAlloc(testing.io, "tier", gpa);
    defer gpa.free(root);
    var pool = try Pool.init(gpa, .{ .families = &fams, .page = page }, 128 * page, 1, 0);
    defer pool.deinit();
    var host = try sessions.HostPool.init(gpa, &pool);
    defer host.deinit();
    const disk = try sessions.Disk.open(gpa, testing.io, .{ .root = root, .compat = @splat(9), .min_tokens = 1, .chunk = 64 << 10, .staging = 4, .lanes = 2, .direct = false });
    defer disk.close();
    // the RAM budget two entries of 100 bytes cross: the second save trims the first
    var store = try Store.init(gpa, testing.io, &pool, host.store(), disk, .{ .ram_bytes = 150, .skip_covered = false, .delta = false });
    defer store.deinit();
    const idsA = try gpa.alloc(i32, 40 * page);
    defer gpa.free(idsA);
    chatIds(idsA, 1, 0);
    const idsB = try gpa.alloc(i32, 40 * page);
    defer gpa.free(idsB);
    chatIds(idsB, 2, 0);
    const sa = try pool.newSlot(128 * page);
    try sa.ensure(idsA.len);
    const idA = (try store.save(sa, try HostBounded.make(gpa, .{ .gpa = undefined, .pos = 2560, .ds_rows = 0, .stash_from = 0, .measure = 100 }), idsA, 1, 0)).?;
    _ = try sa.releaseAll();
    var priced: std.ArrayList(u32) = .empty;
    defer priced.deinit(gpa);
    // a save that trims nothing parks nothing and never refuses, full tier or not
    disk.opts.free_bytes = 0;
    try sess.saveAdmits(&store, 0, 0, 1, &priced);
    try testing.expectEqual(@as(usize, 0), priced.items.len);
    // ample space: the gate passes and names the entry the save's trim would park
    disk.opts.free_bytes = 1 << 40;
    try sess.saveAdmits(&store, 40 * page, 100, 1, &priced);
    try testing.expectEqualSlices(u32, &.{idA}, priced.items);
    // a full tier on every rank: the request is refused, not parked into a fleet crash loop
    disk.opts.free_bytes = 0;
    try testing.expectError(error.TierFull, sess.saveAdmits(&store, 40 * page, 100, 1, &priced));
    // the refusal decided nothing: the entry still holds its pages in RAM, the tier untouched
    try testing.expect(store.entry(idA).ram);
    try testing.expectEqual(@as(u64, 0), disk.used);
    // the boundary: exactly the park's price fits; one byte less refuses
    const need = store.parkBytes(idA);
    disk.opts.free_bytes = need;
    try sess.saveAdmits(&store, 40 * page, 100, 1, &priced);
    disk.opts.free_bytes = need - 1;
    try testing.expectError(error.TierFull, sess.saveAdmits(&store, 40 * page, 100, 1, &priced));
    // the plan is what the save parks: ample space, the second save, and the priced entry is the parked one
    disk.opts.free_bytes = 1 << 40;
    const sb = try pool.newSlot(128 * page);
    try sb.ensure(idsB.len);
    const idB = (try store.save(sb, try HostBounded.make(gpa, .{ .gpa = undefined, .pos = 2560, .ds_rows = 0, .stash_from = 0, .measure = 100 }), idsB, 1, 0)).?;
    _ = try sb.releaseAll();
    try testing.expect(store.entry(idA).extra.file);
    try testing.expect(!store.entry(idA).ram);
    try testing.expectEqual(@as(u64, 1), store.stats.parks);
    try testing.expect(store.entry(idB).ram);
}

test "a save's residency-budget trims meet the tier gate too" {
    // the same gate for Store.save's enforce (idle chats past the residency budget parked): the plan
    // names the coldest idle chat's leaf, a full tier refuses, and the plan is what enforce parks.
    const gpa = testing.allocator;
    const page = 64;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(testing.io, "tier");
    const root = try tmp.dir.realPathFileAlloc(testing.io, "tier", gpa);
    defer gpa.free(root);
    var pool = try Pool.init(gpa, .{ .families = &fams, .page = page }, 128 * page, 1, 0);
    defer pool.deinit();
    var host = try sessions.HostPool.init(gpa, &pool);
    defer host.deinit();
    const disk = try sessions.Disk.open(gpa, testing.io, .{ .root = root, .compat = @splat(9), .min_tokens = 1, .chunk = 64 << 10, .staging = 4, .lanes = 2, .direct = false });
    defer disk.close();
    var store = try Store.init(gpa, testing.io, &pool, host.store(), disk, .{ .chats = 1, .skip_covered = false, .delta = false });
    defer store.deinit();
    var ids: [2][]i32 = undefined;
    for (&ids, 0..) |*buf, c| {
        buf.* = try gpa.alloc(i32, 40 * page);
        chatIds(buf.*, @intCast(c + 1), 0);
    }
    defer for (ids) |buf| gpa.free(buf);
    var leaves: [2]u32 = undefined;
    for (&leaves, 0..) |*leaf, c| {
        const sl = try pool.newSlot(128 * page);
        try sl.ensure(ids[c].len);
        leaf.* = (try store.save(sl, try HostBounded.make(gpa, .{ .gpa = undefined, .pos = 2560, .ds_rows = 0, .stash_from = 0, .measure = 24 }), ids[c], 1, 0)).?;
        _ = try sl.releaseAll();
    }
    // two idle chats over a budget of one: the coldest leaf is the save's park
    var priced: std.ArrayList(u32) = .empty;
    defer priced.deinit(gpa);
    disk.opts.free_bytes = 1 << 40;
    try sess.saveAdmits(&store, 40 * page, 24, 1, &priced);
    try testing.expectEqualSlices(u32, &.{leaves[0]}, priced.items);
    disk.opts.free_bytes = 0;
    try testing.expectError(error.TierFull, sess.saveAdmits(&store, 40 * page, 24, 1, &priced));
    try testing.expect(store.entry(leaves[0]).ram);
    try testing.expectEqual(@as(u64, 0), disk.used);
    // the plan is what the trim parks
    disk.opts.free_bytes = 1 << 40;
    try testing.expectEqual(@as(u32, 1), try store.enforce());
    try testing.expect(store.entry(leaves[0]).extra.file);
    try testing.expect(!store.entry(leaves[0]).ram);
    try testing.expectEqual(@as(u64, 1), store.stats.parks);
    try testing.expect(store.entry(leaves[1]).ram);
}

test "the save gate prices the incoming entry itself when the trims' walk would reach it" {
    // trimRam's walk ends at the newest entry — the one being saved: when every older entry's park is
    // still in flight (its job holds it) and the budget stays crossed, the real walk evicts the
    // incoming entry too, so its file rides along in the gate's price (planSaveParks' `incoming`).
    const gpa = testing.allocator;
    const page = 64;
    var tmp = testing.tmpDir(.{});
    defer tmp.cleanup();
    try tmp.dir.createDirPath(testing.io, "tier");
    const root = try tmp.dir.realPathFileAlloc(testing.io, "tier", gpa);
    defer gpa.free(root);
    var pool = try Pool.init(gpa, .{ .families = &fams, .page = page }, 128 * page, 1, 0);
    defer pool.deinit();
    var host = try sessions.HostPool.init(gpa, &pool);
    defer host.deinit();
    const disk = try sessions.Disk.open(gpa, testing.io, .{ .root = root, .compat = @splat(9), .min_tokens = 1, .chunk = 64 << 10, .staging = 4, .lanes = 2, .direct = false });
    defer disk.close();
    // async parks: a job in flight holds its entry in RAM, so the next save's walk passes over it
    var store = try Store.init(gpa, testing.io, &pool, host.store(), disk, .{ .ram_bytes = 100, .skip_covered = false, .delta = false, .async_io = true });
    defer store.deinit();
    const idsA = try gpa.alloc(i32, 40 * page);
    defer gpa.free(idsA);
    chatIds(idsA, 1, 0);
    const idsB = try gpa.alloc(i32, 40 * page);
    defer gpa.free(idsB);
    chatIds(idsB, 2, 0);
    const sa = try pool.newSlot(128 * page);
    try sa.ensure(idsA.len);
    const idA = (try store.save(sa, try HostBounded.make(gpa, .{ .gpa = undefined, .pos = 2560, .ds_rows = 0, .stash_from = 0, .measure = 60 }), idsA, 1, 0)).?;
    _ = try sa.releaseAll();
    // a park of A still in flight: the next save's trim walk skips it, the budget stays crossed
    _ = (try store.beginPark(idA)).?;
    var priced: std.ArrayList(u32) = .empty;
    defer priced.deinit(gpa);
    var incoming: u64 = 0;
    try store.planSaveParks(40 * page, 60, 1, &priced, &incoming);
    try testing.expectEqual(@as(usize, 0), priced.items.len); // A's park is in flight — priced when it was admitted
    try testing.expect(incoming > 0); // the incoming entry's own file is the walk's last eviction
    disk.opts.free_bytes = incoming;
    try sess.saveAdmits(&store, 40 * page, 60, 1, &priced);
    disk.opts.free_bytes = incoming - 1;
    try testing.expectError(error.TierFull, sess.saveAdmits(&store, 40 * page, 60, 1, &priced));
    // the plan is what the save parks: the walk reaches the incoming entry itself, nothing else
    disk.opts.free_bytes = 1 << 40;
    const sb = try pool.newSlot(128 * page);
    try sb.ensure(idsB.len);
    const idB = (try store.save(sb, try HostBounded.make(gpa, .{ .gpa = undefined, .pos = 2560, .ds_rows = 0, .stash_from = 0, .measure = 60 }), idsB, 1, 0)).?;
    _ = try sb.releaseAll();
    try store.settleAll();
    try testing.expectEqual(@as(u64, 2), store.stats.parks);
    try testing.expect(store.entry(idA).extra.file and !store.entry(idA).ram);
    try testing.expect(store.entry(idB).extra.file and !store.entry(idB).ram);
    // both files written: the price of the second covered the incoming entry's own
    const sizeB = disk.used - disk.get(store.entry(idA).key).?.size;
    try testing.expect(sizeB > 0 and sizeB <= incoming);
}
