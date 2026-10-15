//! A tensor-parallel rank's share of the Super VL ModelOpt checkpoint (cuda_tp.zig's split): rows and input columns cut
//! out of each projection before it is packed, the routed experts' width padded to whole 64-input blocks with zeros.
const std = @import("std");
const cuda = @import("cuda");
const core = @import("core");
const W = @import("cuda_weights.zig");
const mo = @import("cuda_modelopt.zig");
const kern = @import("cuda_kernels.zig");
const Config = @import("config.zig").Config;
const draft_ids = @import("draft_ids.zig");

const Loader = W.Loader;
const Tensor = core.checkpoint.Tensor;
const Nvfp4Experts = cuda.experts.Nvfp4Experts;

/// Rows [r0, r0 + rows) and inputs [k0, k0 + cols) of `{pre}.weight`; rows or cols 0: all of them from there.
pub const Cut = struct { pre: []const u8, r0: usize = 0, rows: usize = 0, k0: usize = 0, cols: usize = 0 };

const max_cuts = 8;

/// Rows [r0, r0 + rows) of a 2-d tensor: a view of the same bytes.
pub fn rowView(t: Tensor, r0: usize, rows: usize) Tensor {
    var out = t;
    const per = t.bytes.len / t.dim(0);
    out.shape[0] = rows;
    out.bytes = t.bytes[r0 * per ..][0 .. rows * per];
    return out;
}

/// Bytes a row of `k` inputs takes in `kind`.
fn rowBytes(kind: mo.Kind, k: usize) usize {
    return switch (kind) {
        .bf16 => 2 * k,
        .fp8 => k,
        .nvfp4 => k / 2,
    };
}

/// Row-stacked cuts of one kind as a lane weight (mo.dense's packing); FP8 parts with scales of their own stack as bf16.
pub fn dense(L: *Loader, ck: *core.Checkpoint, label: []const u8, cuts: []const Cut) !kern.QLinear {
    const m = try mo.kernels(L);
    if (cuts.len == 0 or cuts.len > max_cuts) return error.UnexpectedTensor;
    var parts: [max_cuts]mo.Part = undefined;
    var k0s: [max_cuts]usize = undefined;
    var fulls: [max_cuts]usize = undefined;
    var n: usize = 0;
    for (cuts, 0..) |ct, i| {
        var p = try mo.part(ck, ct.pre);
        const rows = if (ct.rows == 0) p.n - ct.r0 else ct.rows;
        const cols = if (ct.cols == 0) p.k - ct.k0 else ct.cols;
        if (ct.r0 + rows > p.n or ct.k0 + cols > p.k or ct.k0 % 64 != 0 or cols % 64 != 0) return error.UnexpectedTensor;
        p.w = rowView(p.w, ct.r0, rows);
        if (p.blocks) |b| p.blocks = rowView(b, ct.r0, rows);
        fulls[i] = p.k;
        k0s[i] = ct.k0;
        p.n = rows;
        p.k = cols;
        parts[i] = p;
        n += rows;
    }
    const ps = parts[0..cuts.len];
    const k = ps[0].k;
    var kind = ps[0].kind;
    var scale = ps[0].scale;
    var mixed = false;
    for (ps) |p| {
        if (p.k != k or p.kind != ps[0].kind) return error.MixedQuantization;
        if (p.scale != scale) {
            if (p.kind != .fp8) return error.MixedQuantization;
            mixed = true;
        }
    }
    if (mixed) {
        kind = .bf16;
        scale = 1.0;
    }
    // device scratch: the packed input (codes, then NVFP4's block scales), then a stage for a cut's whole rows
    const codes = n * rowBytes(kind, k);
    const blocks = if (kind == .nvfp4) n * k / 16 else 0;
    var stage_bytes: usize = 0;
    for (ps) |p| stage_bytes = @max(stage_bytes, p.w.bytes.len + if (p.blocks) |b| b.bytes.len else 0);
    const base = try L.tmp(codes + blocks + stage_bytes);
    const stage = base + codes + blocks;
    var at = [2]usize{ 0, codes };
    for (ps, k0s[0..ps.len], fulls[0..ps.len]) |p, k0, full| {
        if (mixed) {
            // e4m3 x scale, rounded to bf16 on the host, this cut's inputs alone
            const out = try L.staging(p.n * k * 2);
            const in = try L.src.view(p.w.bytes);
            for (0..p.n) |r| for (0..k) |j| {
                const bits: u32 = @bitCast(mo.e4m3(in[r * full + k0 + j]) * p.scale);
                const rounded = bits +% 0x7FFF +% ((bits >> 16) & 1);
                std.mem.writeInt(u16, out[2 * (r * k + j) ..][0..2], @intCast(rounded >> 16), .little);
            };
            try L.ops.upload(base + at[0], out);
            at[0] += out.len;
            continue;
        }
        const ts = [2]?Tensor{ p.w, p.blocks };
        const per = [2]usize{ rowBytes(kind, full), full / 16 };
        const want = [2]usize{ rowBytes(kind, k), k / 16 };
        const skip = [2]usize{ rowBytes(kind, k0), k0 / 16 };
        var st = stage;
        for (ts, per, want, skip, &at) |maybe, row, width, off, *o| {
            const t = maybe orelse continue;
            if (k0 == 0 and k == full) {
                try L.src.upload(base + o.*, t.bytes);
            } else {
                try L.src.upload(st, t.bytes);
                try L.src.flush(); // the stage's bytes are queued before the copy that reads them
                try L.ops.copyRows(base + o.*, width, st + off, row, width, p.n);
                st += t.bytes.len;
            }
            o.* += p.n * width;
        }
    }
    var buf: [128]u8 = undefined;
    const s = L.ops.s;
    try L.src.flush();
    switch (kind) {
        .bf16 => {
            const out = try L.alloc(try mo.name(&buf, "{s}.weight", .{label}), cuda.modelopt.padded(n) * k * 2);
            try m.pack.bf16Lane(s, base, out, n, k);
            return .{ .bf16 = cuda.modelopt.bf16Weight(out, n, k) };
        },
        .fp8 => {
            const out = try L.alloc(try mo.name(&buf, "{s}.weight", .{label}), cuda.modelopt.padded(n) * k);
            try m.pack.fp8Lane(s, base, out, n, k);
            return .{ .fp8 = cuda.modelopt.fp8Weight(out, scale, n, k) };
        },
        .nvfp4 => {
            const words = try L.alloc(try mo.name(&buf, "{s}.weight", .{label}), cuda.modelopt.padded(n) * k / 2);
            const tiles = try L.alloc(try mo.name(&buf, "{s}.scales", .{label}), cuda.modelopt.padded(n) * k / 16);
            try m.pack.fp4Lane(s, base, base + codes, words, tiles, n, k);
            return .{ .nvfp4 = cuda.nvfp4.weight(words, tiles, scale, n, k) };
        },
    }
}

/// A Mamba-2 mixer's share: the rank's heads (and their groups) in in_proj's z, x, B, C and dt rows, the conv channels,
/// A, D, dt_bias and the gated norm of those heads, and out_proj's columns for them.
pub fn mamba(L: *Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, c: Config) !W.Mamba {
    const r = c.tp_rank;
    const w = c.tp_world;
    const inner = c.inner(); // this rank's
    const gs = c.groups * c.state;
    const inner_full = inner * w;
    const gs_full = gs * w;
    var a: [128]u8 = undefined;
    var n: [5][192]u8 = undefined;
    const in_name = try mo.name(&n[0], "{s}in_proj", .{pre});
    var m: W.Mamba = undefined;
    m.out_rest = 0;
    m.adapter = null;
    m.in_proj = try dense(L, ck, try mo.name(&a, "{s}.in_proj", .{label}), &.{
        .{ .pre = in_name, .r0 = r * inner, .rows = inner }, // z
        .{ .pre = in_name, .r0 = inner_full + r * inner, .rows = inner }, // x
        .{ .pre = in_name, .r0 = 2 * inner_full + r * gs, .rows = gs }, // B
        .{ .pre = in_name, .r0 = 2 * inner_full + gs_full + r * gs, .rows = gs }, // C
        .{ .pre = in_name, .r0 = 2 * inner_full + 2 * gs_full + r * c.mamba_heads, .rows = c.mamba_heads }, // dt
    });
    m.out_proj = try dense(L, ck, try mo.name(&a, "{s}.out_proj", .{label}), &.{.{ .pre = try mo.name(&n[1], "{s}out_proj", .{pre}), .k0 = r * inner, .cols = inner }});
    // conv channels: x, B, C of the rank's heads and groups, [4][ch] fp32 as the loader lays them out
    const conv = try ck.get(try mo.name(&n[1], "{s}conv1d.weight", .{pre}));
    if (conv.rank != 3 or conv.dim(0) != inner_full + 2 * gs_full or conv.dim(1) * conv.dim(2) != 4 or conv.dtype != .bf16) return error.UnexpectedTensor;
    const conv_b = try ck.get(try mo.name(&n[2], "{s}conv1d.bias", .{pre}));
    const ch = inner + 2 * gs;
    const firsts = [3][2]usize{ .{ r * inner, inner }, .{ inner_full + r * gs, gs }, .{ inner_full + gs_full + r * gs, gs } };
    {
        const taps = std.mem.bytesAsSlice(u32, try L.staging(ch * 4 * 4));
        const in = try L.src.view(conv.bytes);
        var o: usize = 0;
        for (firsts) |f| for (f[0]..f[0] + f[1]) |cc| {
            for (0..4) |t| taps[t * ch + o] = @as(u32, std.mem.readInt(u16, in[(cc * 4 + t) * 2 ..][0..2], .little)) << 16;
            o += 1;
        };
        m.conv_w = try L.alloc(try mo.name(&a, "{s}.conv_w", .{label}), L.host.items.len);
        try L.ops.upload(m.conv_w, L.host.items);
    }
    m.conv_b = try widenRuns(L, try mo.name(&a, "{s}.conv_b", .{label}), conv_b, &firsts);
    const alog = try widenRuns(L, "", try ck.get(try mo.name(&n[3], "{s}A_log", .{pre})), &.{.{ r * c.mamba_heads, c.mamba_heads }});
    m.a = try L.alloc(try mo.name(&a, "{s}.a", .{label}), c.mamba_heads * 4);
    try L.ops.torch().mambaA(alog, m.a, c.mamba_heads);
    m.d = try widenRuns(L, try mo.name(&a, "{s}.d", .{label}), try ck.get(try mo.name(&n[3], "{s}D", .{pre})), &.{.{ r * c.mamba_heads, c.mamba_heads }});
    m.dt_bias = try widenRuns(L, try mo.name(&a, "{s}.dt_bias", .{label}), try ck.get(try mo.name(&n[3], "{s}dt_bias", .{pre})), &.{.{ r * c.mamba_heads, c.mamba_heads }});
    const gnorm = try ck.expect(try mo.name(&n[4], "{s}norm.weight", .{pre}), .bf16, &.{inner_full});
    var view = gnorm;
    view.shape[0] = inner;
    view.bytes = gnorm.bytes[r * inner * 2 ..][0 .. inner * 2];
    m.gnorm = try L.raw(try mo.name(&a, "{s}.gnorm", .{label}), view);
    return m;
}

/// Runs [first, first + count) of a 1-d bf16 tensor widened to fp32 on the GPU (label "": loader scratch, not kept).
fn widenRuns(L: *Loader, label: []const u8, t: Tensor, runs: []const [2]usize) !u64 {
    if (t.dtype != .bf16) return error.UnexpectedTensor;
    var total: usize = 0;
    for (runs) |x| total += x[1];
    const out = std.mem.bytesAsSlice(u32, try L.staging(total * 4));
    const in = try L.src.view(t.bytes);
    var o: usize = 0;
    for (runs) |x| for (x[0]..x[0] + x[1]) |i| {
        out[o] = @as(u32, std.mem.readInt(u16, in[2 * i ..][0..2], .little)) << 16;
        o += 1;
    };
    const ptr = if (label.len == 0) try L.tmp(total * 4) else try L.alloc(label, total * 4);
    try L.ops.upload(ptr, L.host.items);
    return ptr;
}

/// Attention's share: the rank's q heads, the KV head they read (kept whole on each rank that reads it), o's columns.
pub fn attention(L: *Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, c: Config) !W.Attention {
    const qd = c.heads * c.head_dim;
    const kvd = c.kv_heads * c.head_dim;
    const kv0 = c.kvFirst() * c.head_dim;
    var a: [128]u8 = undefined;
    var q: [192]u8 = undefined;
    var k: [192]u8 = undefined;
    var v: [192]u8 = undefined;
    const qkv = try dense(L, ck, try mo.name(&a, "{s}.qkv", .{label}), &.{
        .{ .pre = try mo.name(&q, "{s}q_proj", .{pre}), .r0 = c.tp_rank * qd, .rows = qd },
        .{ .pre = try mo.name(&k, "{s}k_proj", .{pre}), .r0 = kv0, .rows = kvd },
        .{ .pre = try mo.name(&v, "{s}v_proj", .{pre}), .r0 = kv0, .rows = kvd },
    });
    const o = try dense(L, ck, try mo.name(&a, "{s}.o", .{label}), &.{.{ .pre = try mo.name(&q, "{s}o_proj", .{pre}), .k0 = c.tp_rank * qd, .cols = qd }});
    return .{ .qkv = qkv, .o = o };
}

/// A latent MoE layer's share: the router and its bias whole, fc1's rows and fc2's columns of the rank's latent slice,
/// every routed expert's width slice, the shared expert's up rows and down columns.
pub fn moe(L: *Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, c: Config) !W.MoE {
    var a: [128]u8 = undefined;
    var b: [192]u8 = undefined;
    const r = c.tp_rank;
    const lat = c.latent / c.tp_world;
    const sw = c.shared_width;
    const router = try mo.router(L, try mo.name(&a, "{s}.router", .{label}), try ck.get(try mo.name(&b, "{s}gate.weight", .{pre})), c);
    const bias = try ck.expect(try mo.name(&b, "{s}gate.e_score_correction_bias", .{pre}), .f32, &.{c.experts});
    var out: W.MoE = .{ .router = router, .bias = try L.raw(try mo.name(&a, "{s}.bias", .{label}), bias), .experts = try routedExperts(L, ck, label, pre, c) };
    var n: [192]u8 = undefined;
    out.latent = .{
        .fc1 = try dense(L, ck, try mo.name(&a, "{s}.fc1", .{label}), &.{.{ .pre = try mo.name(&n, "{s}fc1_latent_proj", .{pre}), .r0 = r * lat, .rows = lat }}),
        .fc2 = try dense(L, ck, try mo.name(&a, "{s}.fc2", .{label}), &.{.{ .pre = try mo.name(&n, "{s}fc2_latent_proj", .{pre}), .k0 = r * lat, .cols = lat }}),
        .shared_up = try dense(L, ck, try mo.name(&a, "{s}.shared_up", .{label}), &.{.{ .pre = try mo.name(&n, "{s}shared_experts.up_proj", .{pre}), .r0 = r * sw, .rows = sw }}),
        .shared_down = try dense(L, ck, try mo.name(&a, "{s}.shared_down", .{label}), &.{.{ .pre = try mo.name(&n, "{s}shared_experts.down_proj", .{pre}), .k0 = r * sw, .cols = sw }}),
    };
    return out;
}

/// Every routed expert's share of its width (up's rows, down's inputs), padded with zero rows and inputs to the rank's
/// width (c.expert_width, whole 64-input blocks): relu^2 of a zero row is zero, so the padding adds nothing.
fn routedExperts(L: *Loader, ck: *core.Checkpoint, label: []const u8, pre: []const u8, c: Config) !kern.Experts {
    const m = try mo.kernels(L);
    const E = c.experts;
    const Wd = c.expert_width;
    const full = c.expert_full;
    const share = full / c.tp_world;
    const r0 = c.tp_rank * share;
    const D = c.latent;
    var layer: kern.Experts = .{ .format = .nvfp4, .up = 0, .down = 0, .width = Wd, .dims = D, .experts = E, .relu2 = true };
    const gs = try L.gpa.alloc(f32, 2 * E);
    defer L.gpa.free(gs);
    var a: [128]u8 = undefined;
    var b: [192]u8 = undefined;
    const st = L.ops.s;
    {
        // up [E][Wd][D]: rows r0.. of each expert, the padded rows zero
        const codes = Wd * D / 2;
        const blocks = Wd * D / 16;
        const base = try L.tmp(E * (codes + blocks));
        try L.ops.fill32(base, 0, E * (codes + blocks) / 4);
        for (0..E) |i| {
            const p = try mo.part(ck, try mo.name(&b, "{s}experts.{d}.up_proj", .{ pre, i }));
            if (p.kind != .nvfp4 or p.n != full or p.k != D) return error.UnexpectedTensor;
            try L.src.upload(base + i * codes, rowView(p.w, r0, share).bytes);
            try L.src.upload(base + E * codes + i * blocks, rowView(p.blocks.?, r0, share).bytes);
            gs[i] = p.scale;
        }
        const out = try L.alloc(try mo.name(&a, "{s}.experts.up", .{label}), Nvfp4Experts.bytes(E, Wd, D, 1));
        try L.src.flush();
        try m.pack.fp4Experts(st, base, base + E * codes, out, E, Wd, D, 0, D, 1, 0);
        layer.up = out;
    }
    {
        // down [E][D][Wd]: inputs r0.. of each expert's rows, the padded inputs zero; staged a chunk of experts at a time
        const codes = D * Wd / 2;
        const blocks = D * Wd / 16;
        const full_codes = D * full / 2;
        const full_blocks = D * full / 16;
        const chunk: usize = 64;
        const base = try L.tmp(E * (codes + blocks) + chunk * (full_codes + full_blocks));
        const stage = base + E * (codes + blocks);
        try L.ops.fill32(base, 0, E * (codes + blocks) / 4);
        var e0: usize = 0;
        while (e0 < E) : (e0 += chunk) {
            const e1 = @min(E, e0 + chunk);
            for (e0..e1) |i| {
                const p = try mo.part(ck, try mo.name(&b, "{s}experts.{d}.down_proj", .{ pre, i }));
                if (p.kind != .nvfp4 or p.n != D or p.k != full) return error.UnexpectedTensor;
                try L.src.upload(stage + (i - e0) * full_codes, p.w.bytes);
                try L.src.upload(stage + chunk * full_codes + (i - e0) * full_blocks, p.blocks.?.bytes);
                gs[E + i] = p.scale;
            }
            try L.src.flush();
            try L.ops.copyRows(base + e0 * codes, Wd / 2, stage + r0 / 2, full / 2, share / 2, (e1 - e0) * D);
            try L.ops.copyRows(base + E * codes + e0 * blocks, Wd / 16, stage + chunk * full_codes + r0 / 16, full / 16, share / 16, (e1 - e0) * D);
        }
        const out = try L.alloc(try mo.name(&a, "{s}.experts.down", .{label}), Nvfp4Experts.bytes(E, D, Wd, 1));
        try m.pack.fp4Experts(st, base, base + E * codes, out, E, D, Wd, 0, Wd, 1, 0);
        layer.down = out;
    }
    layer.up_scale = try mo.scales(L, try mo.name(&a, "{s}.experts.up_scale", .{label}), gs[0..E]);
    layer.down_scale = try mo.scales(L, try mo.name(&a, "{s}.experts.down_scale", .{label}), gs[E..]);
    return layer;
}

/// lm_head's vocabulary rows of the rank: its logits are gathered after the head.
pub fn head(L: *Loader, ck: *core.Checkpoint, c: Config) !kern.QLinear {
    const part = c.vocab / c.tp_world;
    return dense(L, ck, "head", &.{.{ .pre = "lm_head", .r0 = c.tp_rank * part, .rows = part }});
}

/// The MTP head's share: eh_proj's output rows (gathered), attention and the latent MoE as the backbone's.
pub fn mtp(L: *Loader, ck: *core.Checkpoint, c: Config) !void {
    const p0 = "mtp.layers.0.";
    const part = c.hidden / c.tp_world;
    L.w.mtp = .{
        .enorm = try L.raw("mtp.enorm", try ck.get(p0 ++ "enorm.weight")),
        .hnorm = try L.raw("mtp.hnorm", try ck.get(p0 ++ "hnorm.weight")),
        .eh_proj = try dense(L, ck, "mtp.eh_proj", &.{.{ .pre = p0 ++ "eh_proj", .r0 = c.tp_rank * part, .rows = part }}),
        .attn_norm = try L.raw("mtp.attn_norm", try ck.get(p0 ++ "norm.weight")),
        .attn = try attention(L, ck, "mtp.attn", p0 ++ "mixer.", c),
        .moe_norm = try L.raw("mtp.moe_norm", try ck.get("mtp.layers.1.norm.weight")),
        .moe = try moe(L, ck, "mtp.moe", "mtp.layers.1.mixer.", c),
        .final_norm = try L.raw("mtp.final_norm", try ck.get("mtp.layers.1.final_layernorm.weight")),
    };
    try draftHead(L, ck, c);
}

/// The draft head's rows of the rank: its share of the draft ids (in their order), from the FP8 lm_head; the ids table
/// stays whole, since the draws read the gathered logits.
fn draftHead(L: *Loader, ck: *core.Checkpoint, c: Config) !void {
    const m = try mo.kernels(L);
    const ids = try draft_ids.load(L.gpa, c.vocab);
    defer L.gpa.free(ids);
    if (ids.len % c.tp_world != 0) return error.TpUneven;
    const p = try mo.part(ck, "lm_head");
    if (p.kind != .fp8 or p.n != c.vocab) return error.UnexpectedTensor;
    const n = ids.len / c.tp_world;
    const mine = ids[c.tp_rank * n ..][0..n];
    const host = try L.staging(n * p.k);
    const whole = try L.gpa.alloc(u8, p.w.bytes.len);
    defer L.gpa.free(whole);
    try L.src.read(whole, p.w.bytes);
    for (mine, 0..) |id, r| @memcpy(host[r * p.k ..][0..p.k], whole[id * p.k ..][0..p.k]);
    const base = try L.tmp(host.len);
    try L.ops.upload(base, host);
    const out = try L.alloc("draft_head.weight", cuda.modelopt.padded(n) * p.k);
    try m.pack.fp8Lane(L.ops.s, base, out, n, p.k);
    L.w.draft_head = .{ .fp8 = cuda.modelopt.fp8Weight(out, p.scale, n, p.k) };
    try W.draftIds(L, ids);
}
