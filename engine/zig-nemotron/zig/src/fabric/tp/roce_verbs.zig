//! The RoCE wire over verbs (the Python engine's roce.cpp): one RC queue pair a HCA and peer, a send slot striped over the HCAs to every peer, each stripe followed by its seq flag.
const std = @import("std");
const verbs = @import("verbs");
const abi = verbs.abi;
const Verbs = verbs.Verbs;

pub const max_hcas = 2;
/// Largest world the RoCE wire takes (roce.cpp's MAX_PEERS).
pub const max_peers = 16;
/// Send and receive slots by seq parity.
pub const slots = 2;
/// Work requests a queue pair holds; at most a quarter outstanding (two requests a stripe), as in roce.cpp.
const send_depth = 256;
pub const flag_stride = 128;
const link_ethernet = 2;
const record_magic: u32 = 0x5446_5244 + 2; // roce.cpp's "TFRD", bumped: this record's layout is the Zig engine's (N ranks)

pub const Error = error{ NoHca, VerbsFailed, PostFailed, Completion, BadRecord };

/// One rank's region, the same layout on every rank (roce.cpp's, with ctrl first):
///   ctrl [4 KiB]; flags [src][slot][max_hcas] at `flag_stride` (each source rank's seq after that HCA's stripe);
///   send [slot]; recv [src][slot] (filled by rank src's writes). A rank's own src entries are unused.
/// mailbox.cu's tp_roce computes the same addresses from `flag_off`, `recv_off` and the world.
pub const Layout = struct {
    world: u32,
    /// Bytes a slot (a multiple of 4 KiB).
    slot: usize,

    pub const ctrl: usize = 0;
    pub const flags: usize = 4096;

    pub fn of(world: u32, slot: usize) Layout {
        return .{ .world = world, .slot = slot };
    }

    pub fn flagBytes(l: Layout) usize {
        return @as(usize, l.world) * slots * max_hcas * flag_stride;
    }

    pub fn send(l: Layout) usize {
        return flags + std.mem.alignForward(usize, l.flagBytes(), 4096);
    }

    pub fn recv(l: Layout) usize {
        return l.send() + slots * l.slot;
    }

    pub fn bytes(l: Layout) usize {
        return l.recv() + @as(usize, l.world) * slots * l.slot;
    }

    pub fn sendAt(l: Layout, slot: usize) usize {
        return l.send() + slot * l.slot;
    }

    /// Where rank `src`'s slot `slot` lands in the receiver's region.
    pub fn recvAt(l: Layout, src: u32, slot: usize) usize {
        return l.recv() + (@as(usize, src) * slots + slot) * l.slot;
    }

    /// Where rank `src`'s seq for slot `slot` on its HCA `hca` lands in the receiver's region.
    pub fn flagAt(_: Layout, src: u32, slot: usize, hca: usize) usize {
        return flags + ((@as(usize, src) * slots + slot) * max_hcas + hca) * flag_stride;
    }
};

/// The `i`-th peer `me` posts to (i in 0..world-1): the next rank first, so the senders do not all hit the same
/// receiver at once (patches/0003's roce.cpp post order, incast).
pub fn peerAt(me: u32, world: u32, i: u32) u32 {
    return (me + 1 + i) % world;
}

/// One HCA as configured: its device name and RoCE v2 GID index (null: detected).
pub const HcaSpec = struct { name: [64]u8 = @splat(0), name_len: u8 = 0, gid: ?u8 = null };

/// What a rank tells every other rank to connect (every rank builds it the same way; all-gathered through the bootstrap).
pub const Record = extern struct {
    magic: u32 = record_magic,
    n_hca: u32 = 0,
    region: u64 = 0,
    rkey: [max_hcas]u32 = @splat(0),
    lid: [max_hcas]u32 = @splat(0),
    mtu: [max_hcas]u32 = @splat(0),
    /// qpn[h][p]: HCA h's queue pair to rank p (0 at this rank's own index), as roce.cpp's Blob.qp_num.
    qpn: [max_hcas][max_peers]u32 = @splat(@splat(0)),
    gid: [max_hcas][16]u8 = @splat(@splat(0)),
};

const Hca = struct {
    ctx: *abi.Context,
    pd: *abi.Pd,
    mr: *abi.Mr,
    cq: *abi.Cq,
    /// qp[p]: the queue pair to rank p (null at this rank and past the world).
    qp: [max_peers]?*abi.Qp = @splat(null),
    gid_index: u8,
    gid: [16]u8,
    mtu: u32,
    lid: u16,
    /// Signalled requests not yet reaped, a peer.
    outstanding: [max_peers]u32 = @splat(0),
};

/// Parses `name[:gid],name[:gid]` (TF_TP_ROCE_HCA).
pub fn parseSpecs(text: []const u8, out: *[max_hcas]HcaSpec) !usize {
    var n: usize = 0;
    var it = std.mem.tokenizeScalar(u8, text, ',');
    while (it.next()) |item| {
        if (n == max_hcas) return error.TooManyHcas;
        var parts = std.mem.splitScalar(u8, std.mem.trim(u8, item, " "), ':');
        const name = parts.next().?;
        if (name.len == 0 or name.len >= 64) return error.BadHcaName;
        out[n] = .{ .name_len = @intCast(name.len) };
        @memcpy(out[n].name[0..name.len], name);
        if (parts.next()) |g| out[n].gid = try std.fmt.parseInt(u8, g, 10);
        n += 1;
    }
    return n;
}

/// RoCE v2 by sysfs (the kernel's own word for each GID), as roce.py detects it; null when sysfs cannot say.
fn sysfsRoceV2(name: []const u8, index: u32) ?bool {
    var path: [256]u8 = undefined;
    const p = std.fmt.bufPrintSentinel(&path, "/sys/class/infiniband/{s}/ports/1/gid_attrs/types/{d}", .{ name, index }, 0) catch return null;
    const fd = std.c.open(p, .{ .ACCMODE = .RDONLY });
    if (fd < 0) return null;
    defer _ = std.c.close(fd);
    var buf: [32]u8 = undefined;
    const n = std.c.read(fd, &buf, buf.len);
    if (n <= 0) return false;
    return std.mem.startsWith(u8, buf[0..@intCast(n)], "RoCE v2");
}

/// An IPv4-mapped GID (::ffff:a.b.c.d): the address the HCA's netdev carries.
pub fn ipv4Mapped(g: [16]u8) bool {
    return std.mem.allEqual(u8, g[0..10], 0) and g[10] == 0xff and g[11] == 0xff;
}

/// The lowest RoCE v2 GID index that is IPv4-mapped (GID indices move on reboot: never hard-coded).
fn detectGid(v: *const Verbs, ctx: *abi.Context, name: []const u8, table: c_int) ?u8 {
    var i: u32 = 0;
    while (i < @min(@as(u32, @intCast(@max(table, 0))), 255)) : (i += 1) {
        const g = v.gid(ctx, 1, i) catch continue;
        if (!ipv4Mapped(g.raw)) continue;
        const v2 = sysfsRoceV2(name, i) orelse (if (v.gidType(ctx, 1, i)) |t| t == 2 else true);
        if (v2) return @intCast(i);
    }
    return null;
}

/// Every device whose port 1 is an active Ethernet port with an IPv4 RoCE v2 GID, by name order, at most `limit`.
pub fn autoSpecs(v: *const Verbs, out: *[max_hcas]HcaSpec, limit: usize) usize {
    var count: c_int = 0;
    const list = v.api.ibv_get_device_list(&count) orelse return 0;
    defer v.api.ibv_free_device_list(list);
    var n: usize = 0;
    var i: usize = 0;
    while (list[i]) |dev| : (i += 1) {
        if (n == limit) break;
        const name = std.mem.span(v.api.ibv_get_device_name(dev) orelse continue);
        const ctx = v.api.ibv_open_device(dev) orelse continue;
        defer _ = v.api.ibv_close_device(ctx);
        const port = v.port(ctx, 1) catch continue;
        if (port.state != abi.port_active or port.link_layer != link_ethernet) continue;
        const g = detectGid(v, ctx, name, port.gid_tbl_len) orelse continue;
        out[n] = .{ .name_len = @intCast(@min(name.len, 63)), .gid = g };
        @memcpy(out[n].name[0..out[n].name_len], name[0..out[n].name_len]);
        n += 1;
    }
    return n;
}

pub const VerbsWire = struct {
    v: Verbs,
    hcas: [max_hcas]Hca = undefined,
    n_hca: u32 = 0,
    traffic_class: u8,
    me: u32,
    /// Every rank's region address (this rank's entry unused).
    peer_region: [max_peers]u64 = @splat(0),
    /// peer_rkey[p][i]: the rkey of rank p's HCA that this rank's HCA i writes to.
    peer_rkey: [max_peers][max_hcas]u32 = @splat(@splat(0)),
    /// Every rank's region has this layout.
    layout: Layout,
    region: []u8,
    posted: u64 = 0,
    completed: u64 = 0,

    /// Opens the HCAs, registers `region` on each and creates a queue pair a peer (INIT); `record` then goes to every
    /// rank.
    pub fn open(specs: []const HcaSpec, region: []u8, layout: Layout, me: u32, traffic_class: u8) !VerbsWire {
        if (specs.len == 0 or specs.len > max_hcas) return error.NoHca;
        if (layout.world < 2 or layout.world > max_peers or me >= layout.world or region.len < layout.bytes()) return error.NoHca;
        var w: VerbsWire = .{ .v = try Verbs.open(null), .traffic_class = traffic_class, .me = me, .layout = layout, .region = region };
        errdefer w.close();
        for (specs) |s| {
            try w.openHca(s);
        }
        return w;
    }

    fn openHca(w: *VerbsWire, s: HcaSpec) !void {
        const api = w.v.api;
        const name = s.name[0..s.name_len];
        const ctx = try w.v.openDevice(name);
        errdefer _ = api.ibv_close_device(ctx);
        const port = try w.v.port(ctx, 1);
        if (port.state != abi.port_active) {
            std.log.warn("tp roce: {s} port 1 is not active", .{name});
            return error.NoHca;
        }
        const gi = s.gid orelse detectGid(&w.v, ctx, name, port.gid_tbl_len) orelse {
            std.log.warn("tp roce: {s} has no IPv4 RoCE v2 GID", .{name});
            return error.NoHca;
        };
        const gid = try w.v.gid(ctx, 1, gi);
        const pd = api.ibv_alloc_pd(ctx) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_dealloc_pd(pd);
        const mr = api.ibv_reg_mr(pd, w.region.ptr, w.region.len, abi.access_local_write | abi.access_remote_write) orelse {
            std.log.warn("tp roce: ibv_reg_mr on {s} failed (memlock limit? ulimit -l unlimited)", .{name});
            return error.VerbsFailed;
        };
        errdefer _ = api.ibv_dereg_mr(mr);
        // one completion queue a HCA for every peer's queue pair (each holds at most a quarter of its depth signalled)
        const cq = api.ibv_create_cq(ctx, @intCast(send_depth * (w.layout.world - 1)), null, null, 0) orelse return error.VerbsFailed;
        errdefer _ = api.ibv_destroy_cq(cq);
        var h: Hca = .{ .ctx = ctx, .pd = pd, .mr = mr, .cq = cq, .gid_index = gi, .gid = gid.raw, .mtu = @intCast(port.active_mtu), .lid = port.lid };
        errdefer w.destroyQps(&h);
        for (0..w.layout.world) |p| {
            if (p == w.me) continue;
            var init: abi.QpInitAttr = .{ .send_cq = cq, .recv_cq = cq, .cap = .{ .max_send_wr = send_depth, .max_recv_wr = 1, .max_send_sge = 1, .max_recv_sge = 1, .max_inline_data = 16 }, .qp_type = abi.qpt_rc };
            const qp = api.ibv_create_qp(pd, &init) orelse return error.VerbsFailed;
            h.qp[p] = qp;
            var a: abi.QpAttr = .{ .qp_state = abi.qps_init, .port_num = 1, .qp_access_flags = abi.access_remote_write };
            if (api.ibv_modify_qp(qp, &a, abi.mask.state | abi.mask.pkey_index | abi.mask.port | abi.mask.access_flags) != 0) return error.VerbsFailed;
        }
        w.hcas[w.n_hca] = h;
        w.n_hca += 1;
        std.log.info("[tensorfold] tp roce: {s} GID {d} ({d}.{d}.{d}.{d}), MTU enum {d}, {d} queue pairs", .{ name, gi, gid.raw[12], gid.raw[13], gid.raw[14], gid.raw[15], port.active_mtu, w.layout.world - 1 });
    }

    fn destroyQps(w: *VerbsWire, h: *Hca) void {
        for (&h.qp) |*q| if (q.*) |qp| {
            _ = w.v.api.ibv_destroy_qp(qp);
            q.* = null;
        };
    }

    pub fn close(w: *VerbsWire) void {
        const api = w.v.api;
        for (w.hcas[0..w.n_hca]) |*h| {
            w.destroyQps(h);
            _ = api.ibv_destroy_cq(h.cq);
            _ = api.ibv_dereg_mr(h.mr);
            _ = api.ibv_dealloc_pd(h.pd);
            _ = api.ibv_close_device(h.ctx);
        }
        w.n_hca = 0;
        w.v.close();
    }

    pub fn record(w: *const VerbsWire) Record {
        var r: Record = .{ .n_hca = w.n_hca, .region = @intFromPtr(w.region.ptr) };
        for (w.hcas[0..w.n_hca], 0..) |h, i| {
            r.rkey[i] = h.mr.rkey;
            r.lid[i] = h.lid;
            r.mtu[i] = h.mtu;
            for (h.qp, 0..) |q, p| r.qpn[i][p] = if (q) |qp| qp.qp_num else 0;
            r.gid[i] = h.gid;
        }
        return r;
    }

    /// The peer HCA for each of ours: the same IPv4 /24 (the CX7 functions are paired by subnet, names may differ;
    /// roce.py's `pair` keys on the netdev's own prefix, the Sparks' link subnets are /24), else the same position.
    /// One-to-one, or `error.BadRecord`. A function of the two records alone, so both ends of a pair compute it.
    pub fn pairing(mine: Record, peer: Record) ![max_hcas]u32 {
        if (peer.magic != record_magic or mine.magic != record_magic) return error.BadRecord;
        if (peer.n_hca != mine.n_hca or mine.n_hca == 0 or mine.n_hca > max_hcas) return error.BadRecord;
        var out: [max_hcas]u32 = undefined;
        for (0..mine.n_hca) |i| {
            out[i] = @intCast(i);
            var hits: u32 = 0;
            for (0..peer.n_hca) |j| if (std.mem.eql(u8, mine.gid[i][12..15], peer.gid[j][12..15])) {
                out[i] = @intCast(j);
                hits += 1;
            };
            if (hits > 1) out[i] = @intCast(i); // one subnet for both: keep the order
            // No subnet in common: the positional fallback would *connect* a pair no
            // subnet agrees on (review F1). Refuse instead, so the ranks take the NCCL
            // fallback rather than a topology Python would never have built.
            if (hits == 0) return error.BadRecord;
        }
        // two of ours on one of the peer's: the stripes would cross (each HCA i must reach a distinct peer HCA)
        for (0..mine.n_hca) |i| for (0..i) |k| if (out[i] == out[k]) return error.BadRecord;
        return out;
    }

    /// `pairing` checked from the peer's side as well: our HCA i to its j exactly when its j comes back to our i (the
    /// queue pair at each end must name the other's).
    pub fn pairBoth(mine: Record, peer: Record) ![max_hcas]u32 {
        const there = try pairing(mine, peer);
        const back = try pairing(peer, mine);
        for (0..mine.n_hca) |i| if (back[there[i]] != i) return error.BadRecord;
        return there;
    }

    /// INIT to RTR to RTS against every other rank's record (roce.cpp's connect / connect_qp, field for field): our
    /// HCA i's queue pair to rank p meets rank p's paired HCA's queue pair to us. `all`: every rank's record, rank order.
    pub fn connect(w: *VerbsWire, all: []const Record) !void {
        if (all.len != w.layout.world) return error.BadRecord;
        const mine = w.record();
        const api = w.v.api;
        const m = abi.mask;
        for (all, 0..) |peer, pi| {
            const p: u32 = @intCast(pi);
            if (p == w.me) continue;
            const pairs = try pairBoth(mine, peer);
            w.peer_region[p] = peer.region;
            for (w.hcas[0..w.n_hca], 0..) |*h, i| {
                const j = pairs[i];
                const qp = h.qp[p] orelse return error.BadRecord;
                if (peer.qpn[j][w.me] == 0) return error.BadRecord;
                w.peer_rkey[p][i] = peer.rkey[j];
                var rtr: abi.QpAttr = .{ .qp_state = abi.qps_rtr, .path_mtu = @intCast(@min(peer.mtu[j], h.mtu)), .dest_qp_num = peer.qpn[j][w.me], .rq_psn = 0, .max_dest_rd_atomic = 1, .min_rnr_timer = 12 };
                rtr.ah_attr = .{ .grh = .{ .dgid = .{ .raw = peer.gid[j] }, .sgid_index = h.gid_index, .hop_limit = 64, .traffic_class = w.traffic_class }, .dlid = @intCast(peer.lid[j]), .is_global = 1, .port_num = 1 };
                if (api.ibv_modify_qp(qp, &rtr, m.state | m.av | m.path_mtu | m.dest_qpn | m.rq_psn | m.max_dest_rd_atomic | m.min_rnr_timer) != 0) return error.VerbsFailed;
                var rts: abi.QpAttr = .{ .qp_state = abi.qps_rts, .timeout = 14, .retry_cnt = 7, .rnr_retry = 7, .sq_psn = 0, .max_rd_atomic = 1 };
                if (api.ibv_modify_qp(qp, &rts, m.state | m.timeout | m.retry_cnt | m.rnr_retry | m.sq_psn | m.max_qp_rd_atomic) != 0) return error.VerbsFailed;
            }
            std.log.info("[tensorfold] tp roce: rank {d} -> rank {d}: HCA pairs {any}", .{ w.me, p, pairs[0..w.n_hca] });
        }
    }

    /// Slot `slot`'s first `padded` bytes (a multiple of 16) to every peer's receive slot for this rank, the next rank
    /// first, striped over the HCAs, each stripe followed by `seq` at that HCA's flag on the same queue pair (so the
    /// flag cannot land first).
    pub fn post(w: *VerbsWire, slot: u32, seq: u32, padded: u32) !void {
        const packs = padded / 16;
        var seq_copy = seq;
        const l = w.layout;
        for (0..l.world - 1) |k| {
            const p = peerAt(w.me, l.world, @intCast(k));
            var offset: u64 = 0;
            for (w.hcas[0..w.n_hca], 0..) |*h, i| {
                while (h.outstanding[p] >= send_depth / 4) try w.reapHca(h);
                var stripe = packs / w.n_hca;
                if (i < packs % w.n_hca) stripe += 1;
                const bytes: u32 = stripe * 16;
                const id: u64 = (@as(u64, p) << 32) | seq;
                var flag_sge: abi.Sge = .{ .addr = @intFromPtr(&seq_copy), .length = 4, .lkey = 0 };
                var flag_wr: abi.SendWr = .{ .wr_id = id, .sg_list = @ptrCast(&flag_sge), .num_sge = 1, .opcode = abi.wr_rdma_write, .send_flags = abi.send_signaled | abi.send_inline, .remote_addr = w.peer_region[p] + l.flagAt(w.me, slot, i), .rkey = w.peer_rkey[p][i] };
                var data_sge: abi.Sge = .{ .addr = @intFromPtr(w.region.ptr) + l.sendAt(slot) + offset, .length = bytes, .lkey = h.mr.lkey };
                var data_wr: abi.SendWr = .{ .wr_id = id, .next = &flag_wr, .sg_list = @ptrCast(&data_sge), .num_sge = 1, .opcode = abi.wr_rdma_write, .send_flags = 0, .remote_addr = w.peer_region[p] + l.recvAt(w.me, slot) + offset, .rkey = w.peer_rkey[p][i] };
                var bad: ?*abi.SendWr = null;
                const first: *abi.SendWr = if (bytes > 0) &data_wr else &flag_wr;
                if (h.ctx.ops.post_send(h.qp[p].?, first, &bad) != 0) return error.PostFailed;
                h.outstanding[p] += 1;
                offset += bytes;
            }
        }
        w.posted += 1;
    }

    fn reapHca(w: *VerbsWire, h: *Hca) !void {
        var wc: [32]abi.Wc = undefined;
        const n = h.ctx.ops.poll_cq(h.cq, 32, &wc);
        if (n < 0) return error.Completion;
        for (wc[0..@intCast(n)]) |c| {
            const p: usize = @intCast(c.wr_id >> 32);
            if (c.status != abi.wc_success) {
                std.log.err("tp roce: RDMA write of seq {d} to rank {d} failed: {s} (vendor 0x{x})", .{ c.wr_id & 0xffff_ffff, p, w.v.statusText(c.status), c.vendor_err });
                return error.Completion;
            }
            if (p >= max_peers or h.outstanding[p] == 0) return error.Completion;
            h.outstanding[p] -= 1;
        }
        w.completed += @intCast(n);
    }

    /// Reap every HCA's completions (after each op, and in the proxy's idle loop).
    pub fn reap(w: *VerbsWire) !void {
        for (w.hcas[0..w.n_hca]) |*h| try w.reapHca(h);
    }
};

test "HCA lists parse and subnets pair the functions" {
    var specs: [max_hcas]HcaSpec = undefined;
    const n = try parseSpecs("rocep1s0f1:3, roceP2p1s0f1", &specs);
    try std.testing.expectEqual(@as(usize, 2), n);
    try std.testing.expectEqualStrings("rocep1s0f1", specs[0].name[0..specs[0].name_len]);
    try std.testing.expectEqual(@as(?u8, 3), specs[0].gid);
    try std.testing.expectEqual(@as(?u8, null), specs[1].gid);
    var a: Record = .{ .n_hca = 2 };
    var b: Record = .{ .n_hca = 2 };
    const sub100 = [16]u8{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, 100, 10 };
    var sub101 = sub100;
    sub101[14] = 101;
    a.gid = .{ sub100, sub101 };
    b.gid = .{ sub101, sub100 }; // the peer lists its functions the other way round
    b.gid[0][15] = 11;
    b.gid[1][15] = 11;
    const p = try VerbsWire.pairing(a, b);
    try std.testing.expectEqual(@as(u32, 1), p[0]);
    try std.testing.expectEqual(@as(u32, 0), p[1]);
    try std.testing.expect(ipv4Mapped(sub100));
    b.n_hca = 1;
    try std.testing.expectError(error.BadRecord, VerbsWire.pairing(a, b));
}

/// An IPv4-mapped GID on 192.168.`net`.`host`.
fn testGid(net: u8, host: u8) [16]u8 {
    return .{ 0, 0, 0, 0, 0, 0, 0, 0, 0, 0, 0xff, 0xff, 192, 168, net, host };
}

test "four ranks: HCA h of every rank pairs by subnet, whatever order each node lists its functions in" {
    // the four Sparks: each CX7 function on its own switched subnet (100, 101); ranks 1 and 3 list them the other way
    var recs: [4]Record = undefined;
    for (&recs, 0..) |*r, i| {
        const host: u8 = @intCast(10 + i);
        r.* = .{ .n_hca = 2 };
        r.gid = if (i % 2 == 1) .{ testGid(101, host), testGid(100, host) } else .{ testGid(100, host), testGid(101, host) };
        for (0..2) |h| for (0..4) |p| {
            r.qpn[h][p] = if (p == i) 0 else @intCast(1000 * (i + 1) + 10 * h + p);
        };
    }
    for (0..4) |i| for (0..4) |p| {
        if (i == p) continue;
        const pairs = try VerbsWire.pairBoth(recs[i], recs[p]);
        for (0..2) |h| {
            const j = pairs[h];
            // the same subnet at both ends, and the queue pair at the far end is rank p's HCA j's towards rank i
            try std.testing.expectEqualSlices(u8, recs[i].gid[h][12..15], recs[p].gid[j][12..15]);
            try std.testing.expectEqual(@as(u32, @intCast(1000 * (p + 1) + 10 * j + i)), recs[p].qpn[j][i]);
        }
        // the far end pairs back the same way (both ends compute it alone)
        const back = try VerbsWire.pairBoth(recs[p], recs[i]);
        for (0..2) |h| try std.testing.expectEqual(@as(u32, @intCast(h)), back[pairs[h]]);
    };
    // the records all-gather through the bootstrap: sixteen of them fit one blob table
    try std.testing.expect(@sizeOf(Record) * max_peers <= 64 * 1024);
}

test "a pairing that is not one-to-one is refused at both ends (the stripes would cross)" {
    var a: Record = .{ .n_hca = 2 };
    var b: Record = .{ .n_hca = 2 };
    // a's two functions on one subnet, b's on two: both of a's would write b's HCA on that subnet
    a.gid = .{ testGid(100, 10), testGid(100, 20) };
    b.gid = .{ testGid(101, 11), testGid(100, 11) };
    try std.testing.expectError(error.BadRecord, VerbsWire.pairBoth(a, b));
    try std.testing.expectError(error.BadRecord, VerbsWire.pairBoth(b, a));
    // one subnet for every function on both: the listed order, as before
    b.gid = .{ testGid(100, 11), testGid(100, 21) };
    const p = try VerbsWire.pairBoth(a, b);
    try std.testing.expectEqual(@as(u32, 0), p[0]);
    try std.testing.expectEqual(@as(u32, 1), p[1]);
    // a record of another build
    b.magic +%= 1;
    try std.testing.expectError(error.BadRecord, VerbsWire.pairBoth(a, b));
}
