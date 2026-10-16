//! Nemotron CUDA prompt states for the shared cache: a prompt's key and value rows, the Mamba states, the head's rows.

const std = @import("std");
const cuda = @import("cuda");
const config = @import("config.zig");
const Config = config.Config;
const state = @import("cuda_state.zig");
const Engine = @import("cuda_engine.zig").Engine;
const Head = @import("cuda_mtp.zig").Head;
const marks = @import("cuda_marks.zig");

/// One kept state: its Mamba states in one device buffer, its key and value rows in another, where the engine stood.
pub const Copy = struct {
    mamba: cuda.DeviceBuffer,
    kv: cuda.DeviceBuffer,
    head: bool,
    pos: usize,
    parity: usize,
    rows: usize,
    whole: bool = true, // false while an armed mark's chunk has not written all its parts
};

/// The engine and head a cache save copies. The native host keeps one of these for the loaded model.
pub const Target = struct {
    e: *Engine,
    head: ?*Head,
    mirror: ?*Mirror = null, // tensor parallelism's rank 0: every rank keeps the same states (cuda_mirror.zig)
};

const Mirror = @import("cuda_mirror.zig").Mirror;

/// `len` bytes at `live` in the bound sequence's planes; `mamba`: the SSM or conv state, which an armed chunk writes.
const Part = struct { live: u64, len: usize, mamba: bool = false };

/// Where the bound sequence keeps what a state copies: KV planes, SSM and conv planes, the head's KV planes.
const Planes = struct { k: u64, v: u64, ssm: u64, conv: u64, head: ?[2]u64 = null };

const max_parts = 2 * config.max_layers + 4;

/// cuMemAlloc gives a buffer this size whole 2 MiB pages: a state allocates, and the budget counts, whole pages.
const page: usize = 2 << 20;

fn allocBytes(n: usize) usize {
    return std.mem.alignForward(usize, n, page);
}

/// Bytes of a state at `at` tokens: the Mamba states, then a key and a value row a token a layer (the head's too).
pub fn stateBytes(c: Config, at: usize, head: bool) usize {
    const m = state.mambaPlanes(c);
    return m[0] + m[1] + kvBytes(c, at, head);
}

fn kvBytes(c: Config, at: usize, head: bool) usize {
    return (c.count(.attention) + @intFromBool(head)) * 2 * at * state.kvRowBytes(c);
}

/// A state's Mamba buffer, the same for every state: a spare one takes any.
fn mambaBytes(c: Config) usize {
    const m = state.mambaPlanes(c);
    return allocBytes(m[0] + m[1]);
}

/// A state's parts in copy order: every layer's first `at` key rows, its value rows, the Mamba states, the head's.
fn parts(c: Config, max_len: usize, p: Planes, at: usize, out: *[max_parts]Part) []const Part {
    const row = state.kvRowBytes(c);
    const m = state.mambaPlanes(c);
    var n: usize = 0;
    for ([2]u64{ p.k, p.v }) |plane| for (0..c.count(.attention)) |l| {
        out[n] = .{ .live = plane + l * max_len * row, .len = at * row };
        n += 1;
    };
    out[n] = .{ .live = p.ssm, .len = m[0], .mamba = true };
    out[n + 1] = .{ .live = p.conv, .len = m[1], .mamba = true };
    n += 2;
    if (p.head) |h| for (h) |plane| {
        out[n] = .{ .live = plane, .len = at * row };
        n += 1;
    };
    return out[0..n];
}

fn planes(t: *const Target) Planes {
    const b = &t.e.b;
    return .{ .k = b.k_cache, .v = b.v_cache, .ssm = b.ssm, .conv = b.conv_base, .head = if (t.head) |h| .{ h.k_cache, h.v_cache } else null };
}

/// Copies between the bound sequence and `copy` on the engine's stream (`into`: live to copy); the bytes moved.
fn move(t: *const Target, at: usize, copy: *const Copy, into: bool, mamba: bool) !usize {
    const o = t.e.ops();
    var store: [max_parts]Part = undefined;
    var off = [2]usize{ 0, 0 }; // into the key and value buffer, into the Mamba buffer
    for (parts(t.e.c, t.e.max_len, planes(t), at, &store)) |p| {
        const buf = if (p.mamba) copy.mamba.ptr + off[1] else copy.kv.ptr + off[0];
        if (mamba or !p.mamba) if (into) try o.copy(buf, p.live, p.len) else try o.copy(p.live, buf, p.len);
        off[@intFromBool(p.mamba)] += p.len;
    }
    return off[0] + off[1];
}

/// Bytes a save at `at` takes, as the budget counts them: what `save` allocates, spare or new.
pub fn bytes(ptr: *anyopaque, at: u32) u64 {
    const t: *Target = @ptrCast(@alignCast(ptr));
    return mambaBytes(t.e.c) + allocBytes(kvBytes(t.e.c, at, t.head != null));
}

/// Sets spare Mamba buffers aside in a cache of `budget` bytes, a quarter of it at most; the bytes they take.
pub fn ready(t: *Target, budget: u64) !u64 {
    const e = t.e;
    const len = mambaBytes(e.c);
    e.spares.want = @min(marks.spares_max, budget / 4 / len);
    e.spares.fill(e.ctx.d, len);
    return e.spares.want * len;
}

/// Copy the live state after `at` prompt tokens. The pass is standing there.
pub fn save(ptr: *anyopaque, owner: ?*anyopaque, at: u32) anyerror!*anyopaque {
    _ = owner;
    const t: *Target = @ptrCast(@alignCast(ptr));
    const e = t.e;
    // a prompt mark: no window's rows wait to replay, so raw, xc and dt hold nothing a later pass reads
    if (e.prev_keep != 0) return error.NotAtMark;
    if (e.armed.open) |span| if (at > span[0] and at < span[1]) {
        const armed = try arm(t, at);
        if (t.mirror) |m| m.kept(at, armed);
        return armed;
    };
    if (e.pos != at) return error.NotAtMark;
    if (t.head) |h| if (h.pos != at) return error.NotAtMark;
    const copy = try alloc(t, at, true);
    errdefer drop(ptr, copy);
    if (try move(t, at, copy, true, true) != stateBytes(e.c, at, t.head != null)) return error.SnapshotSize;
    if (t.mirror) |m| m.kept(at, copy);
    return copy;
}

/// A copy with a spare Mamba buffer when one waits, and its key and value buffer now (`kv`) or at `finish`.
fn alloc(t: *const Target, at: usize, kv: bool) !*Copy {
    const e = t.e;
    const copy = try e.gpa.create(Copy);
    errdefer e.gpa.destroy(copy);
    var m = e.spares.take() orelse try cuda.DeviceBuffer.alloc(e.ctx.d, mambaBytes(e.c));
    errdefer if (!e.spares.give(m)) m.free();
    const rows = try cuda.DeviceBuffer.alloc(e.ctx.d, if (kv) allocBytes(kvBytes(e.c, at, t.head != null)) else 0);
    copy.* = .{ .mamba = m, .kv = rows, .head = t.head != null, .pos = at, .parity = e.parity, .rows = e.rows };
    return copy;
}

/// A row inside the chunk about to run: the chunk writes the copy's Mamba parts there, `finish` its key and value rows.
fn arm(t: *const Target, at: usize) !*anyopaque {
    const e = t.e;
    if (e.armed.n == marks.max) return error.TooManyMarks;
    const copy = try alloc(t, at, false); // nothing allocated before the chunk when a spare waits
    copy.whole = false;
    e.armed.list[e.armed.n] = .{ .at = at, .ssm = copy.mamba.ptr, .conv = copy.mamba.ptr + state.mambaPlanes(e.c)[0], .copy = copy };
    e.armed.n += 1;
    return copy;
}

/// Once a chunk is queued, while the GPU runs it: the armed rows' key and value rows (the head's too), then new spares.
pub fn finish(t: Target) !void {
    const e = t.e;
    for (e.armed.marks()) |m| {
        const copy: *Copy = @ptrCast(@alignCast(m.copy));
        copy.kv = cuda.DeviceBuffer.alloc(e.ctx.d, allocBytes(kvBytes(e.c, m.at, copy.head))) catch continue; // stays not whole
        _ = try move(&t, m.at, copy, true, false);
        copy.whole = true;
    }
    e.spares.fill(e.ctx.d, mambaBytes(e.c));
}

/// What a kept state holds on the device, both buffers.
pub fn held(saved: *anyopaque) usize {
    const copy: *Copy = @ptrCast(@alignCast(saved));
    return copy.mamba.len + copy.kv.len;
}

/// A kept state's bytes in `out` (its key and value rows, then its Mamba states), after the engine's stream.
pub fn download(t: *const Target, saved: *anyopaque, out: []u8) !void {
    const copy: *Copy = @ptrCast(@alignCast(saved));
    const kv = kvBytes(t.e.c, copy.pos, copy.head);
    try t.e.ops().download(out[0..kv], copy.kv.ptr);
    try t.e.ops().download(out[kv..], copy.mamba.ptr);
    try t.e.stream.synchronize();
}

/// Make the bound sequence `saved`'s; rows past its position stay, since every kernel writes a key before reading it.
pub fn restore(ptr: *anyopaque, owner: ?*anyopaque, saved: *anyopaque) anyerror!void {
    _ = owner;
    const t: *Target = @ptrCast(@alignCast(ptr));
    const copy: *Copy = @ptrCast(@alignCast(saved));
    const e = t.e;
    if (copy.head != (t.head != null)) return error.NoHead; // checked before any copy: nothing half restored
    if (!copy.whole) return error.Incomplete; // its chunk failed before `finish`
    _ = try move(t, copy.pos, copy, false, true);
    e.pos = copy.pos;
    e.parity = copy.parity;
    e.prev_keep = 0;
    e.rows = copy.rows;
    if (t.head) |h| h.pos = copy.pos;
}

// -- the disk tier (TF_NEMOTRON_DISK): a kept state as a file, so an evicted state comes back and a restart keeps them --

const disk_magic: u64 = 0x314b53444f4d454e; // "NEMODSK1"
/// A state file's layout; a change here gives new files a new identity (their own folder), so old ones are never read.
const disk_version: u64 = 1;
const disk_head = 9; // words: magic, version, pos, parity, rows, head, key and value bytes, Mamba bytes, payload hash

/// diskIdentity for the cache's family pointer.
pub fn identityOf(ptr: *anyopaque) u64 {
    return diskIdentity(@ptrCast(@alignCast(ptr)));
}

/// Where this rank's state files go under `root`: a folder per checkpoint, layout, rank and world, the same name every
/// start. The checkpoint's hash keeps another checkpoint of the same shapes (a fine-tune, a revision) from reading
/// these states; adding it moved every folder, so the states written before it are never read.
pub fn diskIdentity(t: *const Target) u64 {
    const c = t.e.c;
    const m = state.mambaPlanes(c);
    const words = [_]u64{ disk_version, state.kvRowBytes(c), m[0], m[1], c.count(.attention), c.layers, c.tp_rank, c.tp_world, @intFromBool(t.head != null), t.e.w.checkpoint_hash };
    return std.hash.Wyhash.hash(0x4e454d4f, std.mem.sliceAsBytes(&words));
}

/// TF_NEMOTRON_DISK's folder for this rank's states (made when missing); null when the variable is unset or empty.
pub fn diskDir(gpa: std.mem.Allocator, t: *const Target) !?[:0]u8 {
    const root = std.mem.trim(u8, std.mem.span(std.c.getenv("TF_NEMOTRON_DISK") orelse return null), " ");
    if (root.len == 0) return null;
    const dir = try std.fmt.allocPrintSentinel(gpa, "{s}/{x:0>16}", .{ root, diskIdentity(t) }, 0);
    errdefer gpa.free(dir);
    for (1..dir.len) |i| if (dir[i] == '/') { // each parent, then the folder (one that exists stays)
        dir[i] = 0;
        _ = std.c.mkdir(dir[0..i :0], 0o700);
        dir[i] = '/';
    };
    _ = std.c.mkdir(dir, 0o700);
    const fd = std.c.open(dir, .{ .ACCMODE = .RDONLY, .DIRECTORY = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.DiskDir;
    _ = std.c.close(fd);
    return dir;
}

extern "c" fn flock(fd: c_int, operation: c_int) c_int;

/// This process's hold on disk folder `dir` (`dir`/lock, held until it exits); error.DiskBusy while another server
/// of the same identity holds it, which then keeps no disk tier rather than share the index and files with it.
pub fn lockDir(dir: []const u8) !void {
    var pb: [1100]u8 = undefined;
    const path = try std.fmt.bufPrintSentinel(&pb, "{s}/lock", .{dir}, 0);
    const fd = std.c.open(path, .{ .ACCMODE = .RDONLY, .CREAT = true, .NOFOLLOW = true, .CLOEXEC = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.DiskLock;
    if (flock(fd, std.c.LOCK.EX | std.c.LOCK.NB) != 0) {
        _ = std.c.close(fd);
        return error.DiskBusy;
    }
}

/// Rank 0's disk folder (made when missing) under this process's hold (lockDir); nothing to hold with TF_NEMOTRON_DISK off.
pub fn lockOf(ptr: *anyopaque) anyerror!void {
    const t: *const Target = @ptrCast(@alignCast(ptr));
    const gpa = t.e.gpa;
    const dir = try diskDir(gpa, t) orelse return;
    defer gpa.free(dir);
    try lockDir(dir);
}

/// State files in `dir` no index names (`keep`), and torn `.part` ones, removed: a follower's half of rank 0's
/// removeUnindexed. Only exact `<key>.bin` and `<key>.bin.part` names go.
pub fn reapDir(dir: [:0]const u8, keep: *const std.AutoHashMapUnmanaged(u64, void)) usize {
    const d = std.c.opendir(dir) orelse return 0;
    defer _ = std.c.closedir(d);
    var n: usize = 0;
    while (std.c.readdir(d)) |ent| {
        if (ent.type != std.c.DT.REG) continue;
        const name = std.mem.sliceTo(&ent.name, 0);
        const key = fileKey(name) orelse continue;
        if (!key.part and keep.contains(key.key)) continue;
        var pb: [1100]u8 = undefined;
        const path = std.fmt.bufPrintSentinel(&pb, "{s}/{s}", .{ dir, name }, 0) catch continue;
        if (std.c.unlink(path) == 0) n += 1;
    }
    return n;
}

/// The key a state file's name holds (diskPath's names), and whether it is a torn `.part`.
fn fileKey(name: []const u8) ?struct { key: u64, part: bool } {
    if (name.len < 20 or !std.mem.eql(u8, name[16..20], ".bin")) return null;
    const part = std.mem.eql(u8, name[20..], ".part");
    if (name.len != 20 and !part) return null;
    for (name[0..16]) |ch| if (!std.ascii.isDigit(ch) and !(ch >= 'a' and ch <= 'f')) return null;
    return .{ .key = std.fmt.parseInt(u64, name[0..16], 16) catch return null, .part = part };
}

/// Rank 0's disk index opened (`keys`): the other ranks remove the state files it does not name, with the next call.
pub fn reapPeers(ptr: *anyopaque, keys: []const u64) void {
    const t: *Target = @ptrCast(@alignCast(ptr));
    if (t.mirror) |m| m.reap(keys);
}

fn diskPath(buf: []u8, dir: []const u8, key: u64, part: bool) ![:0]u8 {
    return std.fmt.bufPrintSentinel(buf, "{s}/{x:0>16}.bin{s}", .{ dir, key, if (part) ".part" else "" }, 0);
}

/// The bytes a state of `at` tokens takes on disk.
pub fn diskBytes(t: *const Target, at: u32) u64 {
    const m = state.mambaPlanes(t.e.c);
    return disk_head * 8 + kvBytes(t.e.c, at, t.head != null) + m[0] + m[1];
}

/// Kept state `saved` as `dir`/<key>.bin: a header, its key and value rows, its Mamba states (written whole, then
/// renamed, so a reader never sees half a file); the bytes written.
pub fn writeFile(t: *const Target, saved: *anyopaque, dir: []const u8, key: u64) !u64 {
    const copy: *Copy = @ptrCast(@alignCast(saved));
    if (!copy.whole) return error.Incomplete;
    if (copy.head != (t.head != null)) return error.NoHead;
    try t.e.ctx.makeCurrent(); // a lookup's thread may not have made a GPU call yet
    const kv = kvBytes(t.e.c, copy.pos, copy.head);
    const m = state.mambaPlanes(t.e.c);
    const buf = try t.e.gpa.alloc(u8, disk_head * 8 + kv + m[0] + m[1]);
    defer t.e.gpa.free(buf);
    const payload = buf[disk_head * 8 ..];
    try download(t, saved, payload);
    const words = [disk_head]u64{ disk_magic, disk_version, copy.pos, copy.parity, copy.rows, @intFromBool(copy.head), kv, m[0] + m[1], std.hash.Wyhash.hash(key, payload) };
    @memcpy(buf[0 .. disk_head * 8], std.mem.sliceAsBytes(&words));
    var pb: [1100]u8 = undefined;
    var fb: [1100]u8 = undefined;
    const part = try diskPath(&pb, dir, key, true);
    const fd = std.c.open(part, .{ .ACCMODE = .WRONLY, .CREAT = true, .TRUNC = true, .NOFOLLOW = true }, @as(std.c.mode_t, 0o600));
    if (fd < 0) return error.DiskWrite;
    {
        defer _ = std.c.close(fd);
        var off: usize = 0;
        while (off < buf.len) {
            const n = std.c.write(fd, buf[off..].ptr, buf.len - off);
            if (n <= 0) {
                _ = std.c.unlink(part);
                return error.DiskWrite;
            }
            off += @intCast(n);
        }
    }
    if (std.c.rename(part, try diskPath(&fb, dir, key, false)) != 0) {
        _ = std.c.unlink(part);
        return error.DiskWrite;
    }
    return buf.len;
}

/// State `key` of `at` tokens read back from `dir` into a new copy; any header, size or hash mismatch refuses it.
pub fn readFile(t: *const Target, dir: []const u8, key: u64, at: u32) !*Copy {
    try t.e.ctx.makeCurrent(); // a lookup's thread may not have made a GPU call yet
    var pb: [1100]u8 = undefined;
    const fd = std.c.open(try diskPath(&pb, dir, key, false), .{ .ACCMODE = .RDONLY, .NOFOLLOW = true }, @as(std.c.mode_t, 0));
    if (fd < 0) return error.DiskMissing;
    defer _ = std.c.close(fd);
    const kv = kvBytes(t.e.c, at, t.head != null);
    const m = state.mambaPlanes(t.e.c);
    const total = disk_head * 8 + kv + m[0] + m[1];
    if (std.c.lseek(fd, 0, std.c.SEEK.END) != @as(std.c.off_t, @intCast(total)) or std.c.lseek(fd, 0, std.c.SEEK.SET) != 0) return error.DiskSize;
    const buf = try t.e.gpa.alloc(u8, total);
    defer t.e.gpa.free(buf);
    var off: usize = 0;
    while (off < total) {
        const n = std.c.read(fd, buf[off..].ptr, total - off);
        if (n <= 0) return error.DiskRead;
        off += @intCast(n);
    }
    var words: [disk_head]u64 = undefined;
    @memcpy(std.mem.sliceAsBytes(&words), buf[0 .. disk_head * 8]);
    const payload = buf[disk_head * 8 ..];
    if (words[0] != disk_magic or words[1] != disk_version or words[2] != at or words[5] != @intFromBool(t.head != null) or words[6] != kv or words[7] != m[0] + m[1]) return error.DiskHeader;
    if (words[8] != std.hash.Wyhash.hash(key, payload)) return error.DiskHash;
    const copy = try alloc(t, at, true);
    errdefer drop(@constCast(t), copy);
    try t.e.ops().upload(copy.kv.ptr, payload[0..kv]);
    try t.e.ops().upload(copy.mamba.ptr, payload[kv..]);
    try t.e.stream.synchronize();
    copy.parity = @intCast(words[3]);
    copy.rows = @intCast(words[4]);
    return copy;
}

/// State `key`'s file removed (none there: nothing to do).
pub fn unlinkFile(dir: []const u8, key: u64) void {
    var pb: [1100]u8 = undefined;
    _ = std.c.unlink(diskPath(&pb, dir, key, false) catch return);
}

/// The cache's disk tier on rank 0 (prompt_cache.Snapshots.write): this rank's file, then the others' (cuda_mirror.zig).
pub fn write(ptr: *anyopaque, saved: *anyopaque, dir: [:0]const u8, key: u64) anyerror!void {
    const t: *Target = @ptrCast(@alignCast(ptr));
    if (t.mirror) |m| if (!m.mirrored(saved)) return error.NotMirrored; // a copy rank 0 alone holds
    _ = try writeFile(t, saved, dir, key);
    if (t.mirror) |m| m.spilled(saved, key);
}

/// Disk state `key` read back into a new copy, on every rank under tensor parallelism (all of them, or a miss).
pub fn read(ptr: *anyopaque, dir: [:0]const u8, key: u64, at: u32) anyerror!*anyopaque {
    const t: *Target = @ptrCast(@alignCast(ptr));
    const copy = try readFile(t, dir, key, at);
    if (t.mirror) |m| m.loaded(copy, key, at) catch |e| {
        drop(t, copy);
        return e;
    };
    return copy;
}

/// Disk state `key` removed here and on the other ranks.
pub fn forget(ptr: *anyopaque, dir: [:0]const u8, key: u64) void {
    const t: *Target = @ptrCast(@alignCast(ptr));
    unlinkFile(dir, key);
    if (t.mirror) |m| m.forgot(key);
}

pub fn forgetChecked(ptr: *anyopaque, dir: [:0]const u8, key: u64) anyerror!void {
    forget(ptr, dir, key);
}

/// Free a kept state's device copy; its Mamba buffer waits as a spare while fewer than the cache set aside do.
pub fn drop(ptr: *anyopaque, saved: *anyopaque) void {
    const t: *Target = @ptrCast(@alignCast(ptr));
    const copy: *Copy = @ptrCast(@alignCast(saved));
    if (t.mirror) |m| m.dropped(saved); // the other ranks drop theirs
    t.e.armed.forget(saved); // a copy evicted before its chunk ran: the chunk must not write it
    if (!t.e.spares.give(copy.mamba)) copy.mamba.free();
    copy.kv.free();
    t.e.gpa.destroy(copy);
}

test "a follower's reap names only exact state files" {
    try std.testing.expectEqual(@as(u64, 0x0123456789abcdef), fileKey("0123456789abcdef.bin").?.key);
    try std.testing.expect(!fileKey("0123456789abcdef.bin").?.part);
    try std.testing.expect(fileKey("0123456789abcdef.bin.part").?.part);
    for ([_][]const u8{ "lock", "index", "0123456789abcdef.bin.old", "0123456789ABCDEF.bin", "0123456789abcde.bin", "0123456789abcdefx.bin" }) |n|
        try std.testing.expect(fileKey(n) == null);
}

fn testConfig() Config {
    var c: Config = undefined;
    c.mamba_heads = 64;
    c.mamba_head_dim = 64;
    c.groups = 8;
    c.state = 128;
    c.head_dim = 128;
    c.kv_heads = 2;
    c.layers = 5;
    c.kinds[0..5].* = .{ .mamba, .attention, .moe, .mamba, .attention };
    return c;
}

test "a kept state holds its prompt's key and value rows beside the whole Mamba states" {
    const c = testConfig();
    const fixed = 2 * (64 * 64 * 128 * 4 + 3 * 6144 * 2);
    try std.testing.expectEqual(@as(usize, fixed), stateBytes(c, 0, true));
    try std.testing.expectEqual(@as(usize, 3 * 2 * 2100 * 512), stateBytes(c, 2100, true) - fixed); // two layers, the head
    try std.testing.expectEqual(@as(usize, 2 * 2 * 2100 * 512), stateBytes(c, 2100, false) - fixed);
}

test "a state's Mamba parts fill its Mamba buffer, SSM first, where an armed row's chunk writes them" {
    const c = testConfig();
    const m = state.mambaPlanes(c);
    const p: Planes = .{ .k = 1 << 40, .v = 2 << 40, .ssm = 3 << 40, .conv = 4 << 40, .head = .{ 5 << 40, 6 << 40 } };
    var store: [max_parts]Part = undefined;
    var off = [2]usize{ 0, 0 };
    for (parts(c, 70144, p, 1500, &store)) |x| {
        if (x.live == p.ssm) try std.testing.expectEqual(@as(usize, 0), off[1]);
        if (x.live == p.conv) try std.testing.expectEqual(m[0], off[1]);
        off[@intFromBool(x.mamba)] += x.len;
    }
    try std.testing.expectEqual(kvBytes(c, 1500, true), off[0]);
    try std.testing.expect(off[1] <= mambaBytes(c));
}

test "a state is charged the whole pages it allocates" {
    try std.testing.expectEqual(@as(usize, 48 << 20), allocBytes(48 << 20));
    try std.testing.expectEqual(@as(usize, 50 << 20), allocBytes((48 << 20) + 1));
    try std.testing.expectEqual(@as(usize, 48 << 20), allocBytes(49082368)); // Nemotron 3.5 Lightning's Mamba buffer
    try std.testing.expectEqual(@as(usize, 14 << 20), allocBytes(7 * 2 * 2048 * 512)); // and its rows at 2,048 tokens
}

test "a state's parts start at each layer's row 0 and add up to its bytes" {
    const c = testConfig();
    const max_len: usize = 70144;
    const row: usize = 512;
    const m = state.mambaPlanes(c);
    const p: Planes = .{ .k = 1 << 40, .v = 2 << 40, .ssm = 3 << 40, .conv = 4 << 40, .head = .{ 5 << 40, 6 << 40 } };
    var store: [max_parts]Part = undefined;
    const want = [_]Part{
        .{ .live = 1 << 40, .len = 2100 * row },          .{ .live = (1 << 40) + max_len * row, .len = 2100 * row },
        .{ .live = 2 << 40, .len = 2100 * row },          .{ .live = (2 << 40) + max_len * row, .len = 2100 * row },
        .{ .live = 3 << 40, .len = m[0], .mamba = true }, .{ .live = 4 << 40, .len = m[1], .mamba = true },
        .{ .live = 5 << 40, .len = 2100 * row },          .{ .live = 6 << 40, .len = 2100 * row },
    };
    const got = parts(c, max_len, p, 2100, &store);
    try std.testing.expectEqualSlices(Part, &want, got);
    var total: usize = 0;
    for (got) |x| total += x.len;
    try std.testing.expectEqual(stateBytes(c, 2100, true), total);
    total = 0;
    var p2 = p;
    p2.head = null;
    for (parts(c, max_len, p2, 2100, &store)) |x| total += x.len;
    try std.testing.expectEqual(stateBytes(c, 2100, false), total);
}
