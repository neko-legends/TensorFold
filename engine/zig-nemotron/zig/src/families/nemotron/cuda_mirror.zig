//! Tensor parallelism's serving: rank 0 runs the server and the round loop; every call its lane backend gets that does
//! GPU work goes first to the other ranks over the plan link (fabric/tp planlink, little-endian int64 frames), and each
//! of them makes the same call on its own backend with a shadow of the stream. The calls then launch the same kernels
//! and collectives in the same order on every rank, and their draws agree (the sums are bit-identical on every rank).
//!
//! Host-only calls (read, first, first_row, held) are not sent. A prompt pass is not cancelled midway under TP (the
//! other ranks cannot see the check).
//!
//! The prompt cache: every rank keeps its own copy of each state rank 0's cache keeps, under a key rank 0 gives it. A
//! prefill call carries the key of the state it starts from and the first key of the marks it keeps at; every rank
//! saves at each mark (rank 0 through its cache, the others into their table), and the keys rank 0's cache did not
//! keep, or dropped since, go to the others as a drop call ahead of the next call.
//!
//! Its disk tier (TF_NEMOTRON_DISK, cuda_reuse.zig's files): each rank writes and reads its own share under the same
//! disk key. A state rank 0 writes goes to the others as a book call ahead of the next call (before its drop); one it
//! reads back is read on every rank at once, and each answers whether it could (a rank whose file is gone or damaged
//! makes it a miss on every rank, never a prefill that runs apart).
const std = @import("std");
const lanes = @import("lanes");
const tp = @import("tp");

const be = lanes.backend;
const Stream = lanes.Stream;
const PlanLink = tp.planlink.PlanLink;

const Op = enum(i64) { prefill = 1, verify, keep, draft, first_masked, release, image, drop, book, load, sync, stop };

/// A book call's entries: (spill, mirror key, disk key) writes a kept copy to disk; (forget, disk key, 0) removes it.
const Book = enum(i64) { spill = 1, forget };

const vstore = @import("vision_store.zig");
const reuse = @import("cuda_reuse.zig");

/// The family's image path on this rank (cuda_native.zig's VisionHook).
pub const Images = struct { ctx: *anyopaque, embed: *const fn (ctx: *anyopaque, img: *const vstore.HeldImage) anyerror!void };

/// Rank 0's side: the family's backend behind one that sends each call first.
pub const Mirror = struct {
    gpa: std.mem.Allocator,
    inner: be.Backend,
    link: *PlanLink,
    mutex: std.atomic.Mutex = .unlocked,
    ids: std.AutoHashMapUnmanaged(*const Stream, i64) = .empty,
    next_id: i64 = 1,
    buf: std.ArrayList(i64) = .empty,
    vt: be.Backend.VTable = undefined,
    images: ?Images = null, // rank 0's own image path, which `embedImage` runs after sending the image on
    // the prompt cache's keys (under `book`: the cache keeps and drops from inside a prefill call and from its own)
    book: std.atomic.Mutex = .unlocked,
    keys: std.AutoHashMapUnmanaged(*anyopaque, i64) = .empty, // rank 0's copies, by the key every rank knows them by
    next_key: i64 = 1,
    marks: []const u32 = &.{}, // the prefill call running: its marks, their first key, which of them were kept
    first_key: i64 = 0,
    saved: std.ArrayList(bool) = .empty,
    drops: std.ArrayList(i64) = .empty, // keys the other ranks drop with the next call
    drop_frame: std.ArrayList(i64) = .empty,
    disk_book: std.ArrayList(i64) = .empty, // the disk tier's writes and removals for the other ranks, in order, before the drops

    /// Calls arrive from the round loop (one thread at a time in practice); a spin keeps any other in line.
    fn lock(m: *Mirror) void {
        while (!m.mutex.tryLock()) std.Thread.yield() catch {};
    }

    pub fn init(gpa: std.mem.Allocator, inner: be.Backend, link: *PlanLink) Mirror {
        return .{ .gpa = gpa, .inner = inner, .link = link };
    }

    /// Tells the other ranks to stop following (they leave their loop and close with rank 0).
    pub fn deinit(m: *Mirror) void {
        m.lock();
        // a stopping server's last disk writes (Store.spillAll) reach the others, and each has written them before the
        // stop: a rank 0 that exits first ends the others through the fate channel within a second
        m.flushDrops() catch {};
        m.buf.clearRetainingCapacity();
        m.buf.append(m.gpa, @intFromEnum(Op.sync)) catch {};
        if (m.link.send(m.buf.items)) |_| {
            _ = m.link.answers() catch false;
        } else |_| {}
        m.buf.clearRetainingCapacity();
        m.buf.append(m.gpa, @intFromEnum(Op.stop)) catch {};
        m.link.send(m.buf.items) catch {};
        m.mutex.unlock();
        m.ids.deinit(m.gpa);
        m.buf.deinit(m.gpa);
        m.keys.deinit(m.gpa);
        m.saved.deinit(m.gpa);
        m.drops.deinit(m.gpa);
        m.drop_frame.deinit(m.gpa);
        m.disk_book.deinit(m.gpa);
    }

    fn lockBook(m: *Mirror) void {
        while (!m.book.tryLock()) std.Thread.yield() catch {};
    }

    /// cuda_reuse.save on rank 0: the copy at mark `at` of the running prefill call takes that mark's key.
    pub fn kept(m: *Mirror, at: u32, copy: *anyopaque) void {
        m.lockBook();
        defer m.book.unlock();
        const k = std.mem.indexOfScalar(u32, m.marks, at) orelse return; // no mark: no rank but this one has it
        m.keys.put(m.gpa, copy, m.first_key + @as(i64, @intCast(k))) catch return;
        m.saved.items[k] = true;
    }

    /// cuda_reuse.drop on rank 0: the others drop theirs with the next call.
    pub fn dropped(m: *Mirror, copy: *anyopaque) void {
        m.lockBook();
        defer m.book.unlock();
        const kv = m.keys.fetchRemove(copy) orelse return;
        m.drops.append(m.gpa, kv.value) catch std.log.err("tp mirror: kept state {d} stays on the other ranks", .{kv.value});
    }

    /// Whether the other ranks hold `copy` (kept at a mirrored call's mark, or read back from disk on every rank).
    pub fn mirrored(m: *Mirror, copy: *anyopaque) bool {
        m.lockBook();
        defer m.book.unlock();
        return m.keys.contains(copy);
    }

    /// cuda_reuse.write on rank 0: the others write their copy of `copy` as disk state `key` with the next call.
    pub fn spilled(m: *Mirror, copy: *anyopaque, key: u64) void {
        m.lockBook();
        defer m.book.unlock();
        const k = m.keys.get(copy) orelse return;
        m.disk_book.appendSlice(m.gpa, &.{ @intFromEnum(Book.spill), k, @bitCast(key) }) catch std.log.err("tp mirror: disk state {x} stays on rank 0 alone", .{key});
    }

    /// cuda_reuse.forget on rank 0: the others remove disk state `key` with the next call.
    pub fn forgot(m: *Mirror, key: u64) void {
        m.lockBook();
        defer m.book.unlock();
        m.disk_book.appendSlice(m.gpa, &.{ @intFromEnum(Book.forget), @bitCast(key), 0 }) catch {};
    }

    /// cuda_reuse.read on rank 0 (`copy` read from disk state `key`): every other rank reads its own under a new key,
    /// and all answer; error.PeerDiskRead when any could not (the ones that did drop theirs with the next call).
    pub fn loaded(m: *Mirror, copy: *anyopaque, key: u64, at: u32) !void {
        m.lock();
        defer m.mutex.unlock();
        const k = blk: {
            m.lockBook();
            defer m.book.unlock();
            const k = m.next_key;
            m.next_key += 1;
            break :blk k;
        };
        try m.begin(.load);
        try m.put(@as(i64, @bitCast(key)));
        try m.put(k);
        try m.put(at);
        try m.send(); // the book and drops go first: a spill or removal of this key lands before the read
        const all = try m.link.answers();
        m.lockBook();
        defer m.book.unlock();
        if (!all) {
            m.drops.append(m.gpa, k) catch {};
            return error.PeerDiskRead;
        }
        try m.keys.put(m.gpa, copy, k);
    }

    /// The wrapping backend; its table lives in the mirror (some entries follow the family's).
    pub fn backend(m: *Mirror) be.Backend {
        const v = m.inner.vtable;
        m.vt = .{
            .prefill = prefillFn,
            .first = firstFn,
            .queue = queueFn,
            .read = readFn,
            .verify = verifyFn,
            .keep = keepFn,
            .draft = draftFn,
            .release = releaseFn,
            .first_row = if (v.first_row != null) firstRowFn else null,
            .first_masked = if (v.first_masked != null) firstMaskedFn else null,
            .held = if (v.held != null) heldFn else null,
            .prompt_rows = v.prompt_rows,
        };
        return .{ .ptr = m, .vtable = &m.vt };
    }

    fn of(ptr: *anyopaque) *Mirror {
        return @ptrCast(@alignCast(ptr));
    }

    fn put(m: *Mirror, v: anytype) !void {
        try m.buf.append(m.gpa, switch (@typeInfo(@TypeOf(v))) {
            .bool => @intFromBool(v),
            .float, .comptime_float => @bitCast(@as(f64, v)),
            .@"enum" => @intFromEnum(v),
            else => @intCast(v),
        });
    }

    fn putSlice(m: *Mirror, xs: anytype) !void {
        try m.put(xs.len);
        for (xs) |x| try m.put(x);
    }

    fn begin(m: *Mirror, op: Op) !void {
        m.buf.clearRetainingCapacity();
        try m.put(op);
    }

    fn send(m: *Mirror) !void {
        try m.flushDrops();
        try m.link.send(m.buf.items);
    }

    /// The disk tier's book since the last call, then the keys dropped since, each as one call (under the mirror's lock).
    fn flushDrops(m: *Mirror) !void {
        m.lockBook();
        defer m.book.unlock();
        if (m.disk_book.items.len > 0) {
            m.drop_frame.clearRetainingCapacity();
            try m.drop_frame.append(m.gpa, @intFromEnum(Op.book));
            try m.drop_frame.append(m.gpa, @intCast(m.disk_book.items.len / 3));
            try m.drop_frame.appendSlice(m.gpa, m.disk_book.items);
            try m.link.send(m.drop_frame.items);
            m.disk_book.clearRetainingCapacity();
        }
        if (m.drops.items.len == 0) return;
        m.drop_frame.clearRetainingCapacity();
        try m.drop_frame.append(m.gpa, @intFromEnum(Op.drop));
        try m.drop_frame.append(m.gpa, @intCast(m.drops.items.len));
        try m.drop_frame.appendSlice(m.gpa, m.drops.items);
        try m.link.send(m.drop_frame.items);
        m.drops.clearRetainingCapacity();
    }

    /// The key of the state a prefill starts from, -1 for none: a copy no other rank has, or one its restore would
    /// refuse (cuda_lanes.restoreKept's checks), is not restored on any rank.
    fn restoreKey(m: *Mirror, s: *Stream) i64 {
        const saved = s.reuse.saved orelse return -1;
        const copy: *const reuse.Copy = @ptrCast(@alignCast(saved));
        m.lockBook();
        const key = m.keys.get(saved);
        m.book.unlock();
        if (key == null or !copy.whole or copy.pos != s.reuse.at or s.reuse.at >= s.prompt().len) {
            s.reuse.saved = null;
            s.reuse.at = 0;
            return -1;
        }
        return key.?;
    }

    fn idOf(m: *Mirror, s: *const Stream) !i64 {
        return m.ids.get(s) orelse error.UnknownStream;
    }

    // -- the vtable: send, then the same call here, under one lock so the ranks see one order ------------------------

    fn prefillFn(ptr: *anyopaque, s: *Stream) anyerror!void {
        const m = of(ptr);
        m.lock();
        defer m.mutex.unlock();
        const gop = try m.ids.getOrPut(m.gpa, s);
        if (!gop.found_existing) {
            gop.value_ptr.* = m.next_id;
            m.next_id += 1;
        }
        try m.begin(.prefill);
        try m.put(gop.value_ptr.*);
        try m.put(s.max_new);
        try putSampling(m, s.sampling);
        try m.put(if (s.logprobs) |k| @as(i64, k) else -1);
        try m.put(if (s.prompt_logprobs) |k| @as(i64, k) else -1);
        try m.put(s.untaught);
        try m.putSlice(s.prompt());
        // the prompt cache: the state it starts from, the marks it keeps at and their first key
        const restore = m.restoreKey(s);
        try m.put(restore);
        try m.put(if (restore >= 0) s.reuse.at else 0);
        {
            m.lockBook();
            defer m.book.unlock();
            m.marks = s.reuse.marks;
            m.first_key = m.next_key;
            m.next_key += @intCast(s.reuse.marks.len);
            m.saved.clearRetainingCapacity();
            try m.saved.appendNTimes(m.gpa, false, s.reuse.marks.len);
        }
        try m.put(m.first_key);
        try m.putSlice(s.reuse.marks);
        try m.send();
        // the prompt pass runs to its end on every rank (no rank but this one sees a cancel)
        const cancel = s.cancel_check;
        const yields = s.yield_check;
        s.cancel_check = null;
        s.yield_check = null;
        defer {
            s.cancel_check = cancel;
            s.yield_check = yields;
            m.lockBook();
            // the marks this rank's cache did not keep: the others drop theirs
            for (m.saved.items, 0..) |kept_, k| if (!kept_) m.drops.append(m.gpa, m.first_key + @as(i64, @intCast(k))) catch {};
            m.marks = &.{};
            m.saved.clearRetainingCapacity();
            m.book.unlock();
            m.flushDrops() catch |e| std.log.err("tp mirror: drops not sent ({t})", .{e});
        }
        try m.inner.prefill(s);
    }

    fn putSampling(m: *Mirror, s: ?lanes.Sampling) !void {
        const x = s orelse return m.put(@as(i64, 0));
        try m.put(@as(i64, 1));
        try m.put(@as(i64, @bitCast(x.seed)));
        try m.put(x.temperature);
        try m.put(x.top_k);
        try m.put(x.top_p);
        try m.put(x.min_p);
    }

    fn firstFn(ptr: *anyopaque, s: *Stream, position: u64) anyerror!u64 {
        return of(ptr).inner.first(s, position);
    }

    fn queueFn(ptr: *anyopaque, s: *Stream, feed: be.Feed, position: u64) anyerror!u64 {
        return of(ptr).inner.queue(s, feed, position);
    }

    fn readFn(ptr: *anyopaque, handle: u64) anyerror!u32 {
        return of(ptr).inner.read(handle);
    }

    fn firstRowFn(ptr: *anyopaque, s: *Stream) anyerror!lanes.LogprobRow {
        return of(ptr).inner.firstRow(s);
    }

    fn heldFn(ptr: *anyopaque, s: *Stream, out: []u32) anyerror!void {
        return of(ptr).inner.vtable.held.?(of(ptr).inner.ptr, s, out);
    }

    fn verifyFn(ptr: *anyopaque, windows: []const be.Window, out: []be.Verified) anyerror!void {
        const m = of(ptr);
        m.lock();
        defer m.mutex.unlock();
        try m.begin(.verify);
        try m.put(windows.len);
        for (windows, out) |w, o| {
            if (w.parents != null) return error.TreesNotBuilt;
            try m.put(try m.idOf(w.stream));
            try m.put(w.pending);
            try m.put(w.held);
            try m.put(w.early);
            try m.putSlice(w.tokens);
            try m.putSlice(w.positions);
            try m.putSlice(w.masks);
            try m.put(o.sampled.len);
            try m.put(o.drafts.len);
            try m.put(o.rows.len);
        }
        try m.send();
        try m.inner.verify(windows, out);
    }

    fn keepFn(ptr: *anyopaque, windows: []const be.Window, paths: []const []const u32) anyerror!void {
        const m = of(ptr);
        m.lock();
        defer m.mutex.unlock();
        try m.begin(.keep);
        try m.put(windows.len);
        for (windows, paths) |w, p| {
            try m.put(try m.idOf(w.stream));
            try m.putSlice(p);
        }
        try m.send();
        try m.inner.keep(windows, paths);
    }

    fn draftFn(ptr: *anyopaque, requests: []const be.DraftRequest) anyerror!void {
        const m = of(ptr);
        m.lock();
        defer m.mutex.unlock();
        try m.begin(.draft);
        try m.put(requests.len);
        for (requests) |r| {
            if (r.lanes != null) return error.TreesNotBuilt;
            try m.put(try m.idOf(r.stream));
            try m.putSlice(r.follow);
            if (r.first) |f| switch (f) {
                .handle => |h| {
                    try m.put(@as(i64, 1));
                    try m.put(h);
                },
                .value => |v| {
                    try m.put(@as(i64, 2));
                    try m.put(v);
                },
            } else {
                try m.put(@as(i64, 0));
                try m.put(@as(i64, 0));
            }
            if (r.rows) |rows| try m.putSlice(rows) else try m.put(@as(i64, -1));
            try m.put(r.start);
            try m.put(r.position);
            try m.put(r.depth);
            try m.put(r.early);
            try m.put(r.ranks);
        }
        try m.send();
        try m.inner.draft(requests);
    }

    fn firstMaskedFn(ptr: *anyopaque, s: *Stream, position: u64, mask: []const u32) anyerror!u64 {
        const m = of(ptr);
        m.lock();
        defer m.mutex.unlock();
        try m.begin(.first_masked);
        try m.put(try m.idOf(s));
        try m.put(position);
        try m.putSlice(mask);
        try m.send();
        return m.inner.vtable.first_masked.?(m.inner.ptr, s, position, mask);
    }

    /// An image's rows on every rank: its pixels and ids go to the others (each runs the tower on its own GPU: the
    /// rows land in every rank's image table before any prompt that reads them), then this rank embeds it.
    pub fn embedImage(m: *Mirror, img: *const vstore.HeldImage) !void {
        const own = m.images orelse return error.NoVision;
        {
            m.lock();
            defer m.mutex.unlock();
            try m.begin(.image);
            try m.put(img.rows);
            try m.put(img.cols);
            for (0..4) |i| try m.put(@as(i64, @bitCast(std.mem.readInt(u64, img.digest[i * 8 ..][0..8], .little))));
            try m.putSlice(img.vids[0..img.n_vids]);
            const n = 3 * 256 * @as(usize, img.rows) * img.cols;
            try m.put(n);
            const words = (n + 3) / 4;
            try m.buf.ensureUnusedCapacity(m.gpa, words);
            var i: usize = 0;
            while (i < n) : (i += 4) {
                var w: u64 = 0;
                for (0..4) |j| if (i + j < n) {
                    w |= @as(u64, img.pixels[i + j]) << @intCast(16 * j);
                };
                m.buf.appendAssumeCapacity(@bitCast(w));
            }
            try m.send();
        }
        try own.embed(own.ctx, img);
    }

    fn releaseFn(ptr: *anyopaque, s: *Stream) void {
        const m = of(ptr);
        m.lock();
        defer m.mutex.unlock();
        const kv = m.ids.fetchRemove(s) orelse return m.inner.release(s);
        m.begin(.release) catch {};
        m.put(kv.value) catch {};
        m.send() catch |e| std.log.err("tp mirror: release not sent ({t})", .{e});
        m.inner.release(s);
    }
};

/// A frame's words read in order.
const Reader = struct {
    w: []const i64,
    at: usize = 0,

    fn int(r: *Reader, comptime T: type) !T {
        if (r.at >= r.w.len) return error.ShortFrame;
        const v = r.w[r.at];
        r.at += 1;
        return switch (@typeInfo(T)) {
            .bool => v != 0,
            .float => @floatCast(@as(f64, @bitCast(v))),
            else => @intCast(v),
        };
    }

    /// A length-prefixed run of words as T values in `out` (grown as needed).
    fn slice(r: *Reader, comptime T: type, gpa: std.mem.Allocator, out: *std.ArrayList(T)) ![]T {
        const n = try r.int(usize);
        if (r.at + n > r.w.len) return error.ShortFrame;
        out.clearRetainingCapacity();
        try out.ensureTotalCapacity(gpa, n);
        for (r.w[r.at..][0..n]) |v| out.appendAssumeCapacity(@intCast(v));
        r.at += n;
        return out.items;
    }
};

/// A follower's kept states (the prompt cache's, rank 0's keys) and where it copies them from.
const Kept = struct {
    gpa: std.mem.Allocator,
    target: *reuse.Target,
    copies: std.AutoHashMapUnmanaged(i64, *anyopaque) = .empty,
    // the prefill call running: its marks and their first key
    marks: []const u32 = &.{},
    first: i64 = 0,
    disk: ?[:0]u8 = null, // the disk tier's folder on this rank (TF_NEMOTRON_DISK/<identity>), null when off

    fn deinit(k: *Kept) void {
        var it = k.copies.valueIterator();
        while (it.next()) |c| reuse.drop(k.target, c.*);
        k.copies.deinit(k.gpa);
        if (k.disk) |d| k.gpa.free(d);
    }

    /// The prefill's pass stands at a mark (or arms one inside its next chunk): this rank's copy, under its key.
    fn at(ptr: *anyopaque, _: *Stream, row: u32) void {
        const k: *Kept = @ptrCast(@alignCast(ptr));
        const i = std.mem.indexOfScalar(u32, k.marks, row) orelse return;
        const copy = reuse.save(k.target, null, row) catch |e| {
            std.log.warn("tp follower: no copy at row {d} ({t}); a later prefill from it stops this rank", .{ row, e });
            return;
        };
        k.copies.put(k.gpa, k.first + @as(i64, @intCast(i)), copy) catch reuse.drop(k.target, copy);
    }
};

/// A rank above 0: makes rank 0's calls on `inner` until rank 0 stops (or its link closes). `target`: this rank's
/// engine, whose states the prompt cache keeps (null: rank 0 sends no keys).
pub fn follow(gpa: std.mem.Allocator, inner: be.Backend, link: *PlanLink, images: ?Images, target: ?*reuse.Target) !void {
    var kept: ?Kept = if (target) |t| .{ .gpa = gpa, .target = t } else null;
    defer if (kept) |*k| k.deinit();
    if (kept) |*k| k.disk = reuse.diskDir(gpa, k.target) catch |e| blk: {
        std.log.warn("tp follower: no disk tier on this rank ({t}); a state rank 0 reads from disk is a miss everywhere", .{e});
        break :blk null;
    };
    var streams: std.AutoHashMapUnmanaged(i64, *Stream) = .empty;
    defer {
        var it = streams.valueIterator();
        while (it.next()) |s| {
            inner.release(s.*);
            s.*.deinit(gpa);
            gpa.destroy(s.*);
        }
        streams.deinit(gpa);
    }
    var frame: std.ArrayList(i64) = .empty;
    defer frame.deinit(gpa);
    var scratch: Scratch = .{};
    defer scratch.deinit(gpa);
    var calls: u64 = 0;
    while (true) {
        const words = link.recvGrow(gpa, &frame) catch |e| switch (e) {
            error.PeerClosed => {
                std.log.info("tp follower: rank 0 closed the plan link after {d} calls", .{calls});
                return;
            },
            else => return e,
        };
        var r: Reader = .{ .w = words };
        const raw = try r.int(i64);
        if (raw < 1 or raw > @intFromEnum(Op.stop)) return error.BadMirrorFrame;
        const op: Op = @enumFromInt(raw);
        if (op == .stop) {
            std.log.info("tp follower: rank 0 stopped after {d} calls", .{calls});
            return;
        }
        calls += 1;
        if (op == .load) { // every rank answers, whatever happens here: rank 0 waits for it
            try link.answer(@intFromBool(load(&r, if (kept) |*k| k else null)));
            continue;
        }
        if (op == .sync) { // every call before it is done here
            try link.answer(1);
            continue;
        }
        apply(gpa, inner, images, op, &r, &streams, &scratch, if (kept) |*k| k else null) catch |e| {
            // a state rank 0 starts from that this rank lacks: its prefill would run apart from rank 0's
            if (e == error.MissingKept) return e;
            // rank 0 met the same failure at the same point (the calls are deterministic); keep following
            std.log.warn("tp follower: {t} call {d} failed: {t}", .{ op, calls, e });
        };
    }
}

/// A load call: disk state `key` read into this rank's copy under rank 0's new key; whether it could.
fn load(r: *Reader, kept: ?*Kept) bool {
    const key: u64 = @bitCast(r.int(i64) catch return false);
    const mk = r.int(i64) catch return false;
    const at = r.int(u32) catch return false;
    const k = kept orelse return false;
    const dir = k.disk orelse return false;
    const copy = reuse.readFile(k.target, dir, key, at) catch |e| {
        std.log.warn("tp follower: disk state {x} ({d} tokens) not read ({t}): a miss on every rank", .{ key, at, e });
        return false;
    };
    k.copies.put(k.gpa, mk, copy) catch {
        reuse.drop(k.target, copy);
        return false;
    };
    return true;
}

/// The follower's reusable buffers for decoded calls.
const Scratch = struct {
    a: [8]std.ArrayList(u32) = @splat(.empty),
    pos: [8]std.ArrayList(u64) = @splat(.empty),
    masks: [8]std.ArrayList(u32) = @splat(.empty),
    sampled: [8]std.ArrayList(u32) = @splat(.empty),
    drafts: [8]std.ArrayList(u32) = @splat(.empty),
    rows: [8]std.ArrayList(lanes.LogprobRow) = @splat(.empty),
    rows2: [8]std.ArrayList(u32) = @splat(.empty),
    prompt: std.ArrayList(u32) = .empty,
    marks: std.ArrayList(u32) = .empty,
    drops: std.ArrayList(i64) = .empty,

    fn deinit(s: *Scratch, gpa: std.mem.Allocator) void {
        inline for (.{ &s.a, &s.pos, &s.masks, &s.sampled, &s.drafts, &s.rows, &s.rows2 }) |set| for (set) |*l| l.deinit(gpa);
        s.prompt.deinit(gpa);
        s.marks.deinit(gpa);
        s.drops.deinit(gpa);
    }
};

const max_windows = 8;

fn apply(gpa: std.mem.Allocator, inner: be.Backend, images: ?Images, op: Op, r: *Reader, streams: *std.AutoHashMapUnmanaged(i64, *Stream), x: *Scratch, kept: ?*Kept) !void {
    switch (op) {
        .image => {
            const own = images orelse return error.NoVision;
            var img: vstore.HeldImage = undefined;
            img.rows = try r.int(u32);
            img.cols = try r.int(u32);
            for (0..4) |i| std.mem.writeInt(u64, img.digest[i * 8 ..][0..8], @bitCast(try r.int(i64)), .little);
            const vids = try r.slice(u32, gpa, &x.prompt);
            img.vids = vids.ptr;
            img.n_vids = vids.len;
            const n = try r.int(usize);
            if (n != 3 * 256 * @as(usize, img.rows) * img.cols or r.at + (n + 3) / 4 > r.w.len) return error.BadMirrorFrame;
            const px = try gpa.alloc(u16, n);
            defer gpa.free(px);
            for (0..n) |i| px[i] = @truncate(@as(u64, @bitCast(r.w[r.at + i / 4])) >> @intCast(16 * (i % 4)));
            r.at += (n + 3) / 4;
            img.pixels = px.ptr;
            try own.embed(own.ctx, &img);
        },
        .prefill => {
            const id = try r.int(i64);
            const max_new = try r.int(u32);
            const sampling: ?lanes.Sampling = if (try r.int(i64) == 0) null else .{
                .seed = @bitCast(try r.int(i64)),
                .temperature = try r.int(f64),
                .top_k = try r.int(u32),
                .top_p = try r.int(f64),
                .min_p = try r.int(f64),
            };
            const logprobs = try r.int(i64);
            const prompt_logprobs = try r.int(i64);
            const untaught = try r.int(bool);
            const prompt = try r.slice(u32, gpa, &x.prompt);
            const restore = try r.int(i64);
            const restore_at = try r.int(u32);
            const first_key = try r.int(i64);
            const marks = try r.slice(u32, gpa, &x.marks);
            var saved: ?*anyopaque = null;
            if (restore >= 0) {
                const k = kept orelse return error.MissingKept;
                saved = k.copies.get(restore) orelse return error.MissingKept;
                const copy: *const reuse.Copy = @ptrCast(@alignCast(saved.?));
                if (!copy.whole or copy.pos != restore_at) return error.MissingKept;
            }
            const spec: lanes.stream.Spec = .{
                .id = "tp-follower",
                .prompt = prompt,
                .max_new = max_new,
                .sampling = sampling,
                .logprobs = if (logprobs >= 0) @intCast(logprobs) else null,
                .prompt_logprobs = if (prompt_logprobs >= 0) @intCast(prompt_logprobs) else null,
                .untaught = untaught,
            };
            const gop = try streams.getOrPut(gpa, id);
            if (gop.found_existing) {
                // a stream prefilled again keeps its address: the backend finds its lane by it
                gop.value_ptr.*.deinit(gpa);
                gop.value_ptr.*.* = try Stream.init(gpa, spec);
            } else {
                const s = try gpa.create(Stream);
                s.* = Stream.init(gpa, spec) catch |e| {
                    gpa.destroy(s);
                    _ = streams.remove(id);
                    return e;
                };
                gop.value_ptr.* = s;
            }
            const s = gop.value_ptr.*;
            if (kept) |k| {
                k.marks = marks;
                k.first = first_key;
                s.reuse = .{ .saved = saved, .at = if (saved != null) restore_at else 0, .marks = marks, .hook = .{ .ptr = k, .at = Kept.at } };
            }
            defer s.reuse = .{};
            try inner.prefill(s);
        },
        .drop => {
            const keys = try r.slice(i64, gpa, &x.drops);
            const k = kept orelse return;
            for (keys) |key| if (k.copies.fetchRemove(key)) |kv| reuse.drop(k.target, kv.value);
        },
        .book => {
            const n = try r.int(usize);
            for (0..n) |_| {
                const kind = try r.int(i64);
                const a = try r.int(i64);
                const b = try r.int(i64);
                const k = kept orelse continue;
                const dir = k.disk orelse continue;
                switch (kind) {
                    @intFromEnum(Book.spill) => {
                        const copy = k.copies.get(a) orelse {
                            std.log.warn("tp follower: no copy {d} to write as disk state {x}", .{ a, @as(u64, @bitCast(b)) });
                            continue;
                        };
                        _ = reuse.writeFile(k.target, copy, dir, @bitCast(b)) catch |e| std.log.warn("tp follower: disk state {x} not written ({t}); reading it back will be a miss", .{ @as(u64, @bitCast(b)), e });
                    },
                    @intFromEnum(Book.forget) => reuse.unlinkFile(dir, @bitCast(a)),
                    else => return error.BadMirrorFrame,
                }
            }
        },
        .load, .sync => unreachable, // follow answers them
        .verify => {
            const n = try r.int(usize);
            if (n > max_windows) return error.BadMirrorFrame;
            var windows: [max_windows]be.Window = undefined;
            var out: [max_windows]be.Verified = undefined;
            for (0..n) |k| {
                const s = streams.get(try r.int(i64)) orelse return error.UnknownStream;
                const pending = try r.int(u32);
                const held = try r.int(u32);
                const early = try r.int(bool);
                const tokens = try r.slice(u32, gpa, &x.a[k]);
                const positions = try r.slice(u64, gpa, &x.pos[k]);
                const masks = try r.slice(u32, gpa, &x.masks[k]);
                windows[k] = .{ .stream = s, .pending = pending, .held = held, .tokens = tokens, .parents = null, .positions = positions, .early = early, .masks = masks };
                try x.sampled[k].resize(gpa, try r.int(usize));
                try x.drafts[k].resize(gpa, try r.int(usize));
                try x.rows[k].resize(gpa, try r.int(usize));
                out[k] = .{ .sampled = x.sampled[k].items, .drafts = x.drafts[k].items, .rows = x.rows[k].items };
            }
            try inner.verify(windows[0..n], out[0..n]);
        },
        .keep => {
            const n = try r.int(usize);
            if (n > max_windows) return error.BadMirrorFrame;
            var windows: [max_windows]be.Window = undefined;
            var paths: [max_windows][]const u32 = undefined;
            for (0..n) |k| {
                const s = streams.get(try r.int(i64)) orelse return error.UnknownStream;
                windows[k] = .{ .stream = s, .pending = 0, .held = 0, .tokens = &.{}, .parents = null, .positions = &.{} };
                paths[k] = try r.slice(u32, gpa, &x.a[k]);
            }
            try inner.keep(windows[0..n], paths[0..n]);
        },
        .draft => {
            const n = try r.int(usize);
            if (n > max_windows) return error.BadMirrorFrame;
            var requests: [max_windows]be.DraftRequest = undefined;
            for (0..n) |k| {
                const s = streams.get(try r.int(i64)) orelse return error.UnknownStream;
                const follow_ = try r.slice(u32, gpa, &x.a[k]);
                const tag = try r.int(i64);
                const val = try r.int(i64);
                const first: ?be.Feed = switch (tag) {
                    0 => null,
                    1 => .{ .handle = @intCast(val) },
                    2 => .{ .value = @intCast(val) },
                    else => return error.BadMirrorFrame,
                };
                const rows_len = try r.int(i64);
                var rows: ?[]const u32 = null;
                if (rows_len >= 0) {
                    r.at -= 1; // the length is the slice's own prefix
                    rows = try r.slice(u32, gpa, &x.rows2[k]);
                }
                requests[k] = .{
                    .stream = s,
                    .follow = follow_,
                    .first = first,
                    .rows = rows,
                    .start = try r.int(u64),
                    .position = try r.int(u64),
                    .depth = try r.int(u32),
                    .early = try r.int(bool),
                    .ranks = try r.int(bool),
                };
            }
            try inner.draft(requests[0..n]);
        },
        .first_masked => {
            const s = streams.get(try r.int(i64)) orelse return error.UnknownStream;
            const position = try r.int(u64);
            const mask = try r.slice(u32, gpa, &x.masks[0]);
            _ = try inner.vtable.first_masked.?(inner.ptr, s, position, mask);
        },
        .release => {
            const kv = streams.fetchRemove(try r.int(i64)) orelse return error.UnknownStream;
            inner.release(kv.value);
            kv.value.deinit(gpa);
            gpa.destroy(kv.value);
        },
        .stop => unreachable,
    }
}
