//! Tensor parallelism's serving: rank 0 runs the server and the round loop; every call its lane backend gets that does
//! GPU work goes first to the other ranks over the plan link (fabric/tp planlink, little-endian int64 frames), and each
//! of them makes the same call on its own backend with a shadow of the stream. The calls then launch the same kernels
//! and collectives in the same order on every rank, and their draws agree (the sums are bit-identical on every rank).
//!
//! Host-only calls (read, first, first_row, held) are not sent. A prompt pass is not cancelled midway under TP (the
//! other ranks cannot see the check), and the prompt cache is off (a restored state would exist on rank 0 alone).
const std = @import("std");
const lanes = @import("lanes");
const tp = @import("tp");

const be = lanes.backend;
const Stream = lanes.Stream;
const PlanLink = tp.planlink.PlanLink;

const Op = enum(i64) { prefill = 1, verify, keep, draft, first_masked, release, image, stop };

const vstore = @import("vision_store.zig");

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
        m.buf.clearRetainingCapacity();
        m.buf.append(m.gpa, @intFromEnum(Op.stop)) catch {};
        m.link.send(m.buf.items) catch {};
        m.mutex.unlock();
        m.ids.deinit(m.gpa);
        m.buf.deinit(m.gpa);
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
        try m.link.send(m.buf.items);
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
        try m.send();
        // the prompt pass runs to its end on every rank (no rank but this one sees a cancel), from row 0
        const cancel = s.cancel_check;
        const yields = s.yield_check;
        const reuse = s.reuse;
        s.cancel_check = null;
        s.yield_check = null;
        s.reuse = .{};
        defer {
            s.cancel_check = cancel;
            s.yield_check = yields;
            s.reuse = reuse;
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

/// A rank above 0: makes rank 0's calls on `inner` until rank 0 stops (or its link closes).
pub fn follow(gpa: std.mem.Allocator, inner: be.Backend, link: *PlanLink, images: ?Images) !void {
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
        apply(gpa, inner, images, op, &r, &streams, &scratch) catch |e| {
            // rank 0 met the same failure at the same point (the calls are deterministic); keep following
            std.log.warn("tp follower: {t} call {d} failed: {t}", .{ op, calls, e });
        };
    }
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

    fn deinit(s: *Scratch, gpa: std.mem.Allocator) void {
        inline for (.{ &s.a, &s.pos, &s.masks, &s.sampled, &s.drafts, &s.rows, &s.rows2 }) |set| for (set) |*l| l.deinit(gpa);
        s.prompt.deinit(gpa);
    }
};

const max_windows = 8;

fn apply(gpa: std.mem.Allocator, inner: be.Backend, images: ?Images, op: Op, r: *Reader, streams: *std.AutoHashMapUnmanaged(i64, *Stream), x: *Scratch) !void {
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
            try inner.prefill(gop.value_ptr.*);
        },
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
