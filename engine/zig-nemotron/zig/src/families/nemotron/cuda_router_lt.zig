//! A prompt chunk's router logits through cuBLASLt (bf16 in, fp32 accumulate and out): x [rows, D] . W[E, D]^T into
//! the router's partial buffer as one slice. The products of bf16 values are exact in fp32, but the sum's order differs
//! from tf_nemo_router's K-slice chains, so near-tied picks can go the other way: greedy text changes. Opt-in
//! (TF_NEMOTRON_ROUTER_LT=1); the default is tf_nemo_router_tiles, the chains' own bits. ~0.6 ms a 4,096-row chunk a
//! layer.
const std = @import("std");
const cuda = @import("cuda");
const lt_mod = cuda.cublaslt;

/// Rows are rounded up to this, one plan a rounded count (the rows past the chunk's are read and their logits unused).
const step = 128;
const max_plans = 64; // 8,192 rows

var lib: ?lt_mod.Library = null;
var failed = false;
var plans: [max_plans]?lt_mod.Linear = @splat(null);
var dims: [2]u64 = .{ 0, 0 };

pub fn enabled() bool {
    if (failed) return false;
    const v = std.c.getenv("TF_NEMOTRON_ROUTER_LT") orelse return false;
    return std.mem.eql(u8, std.mem.span(v), "1");
}

/// The rows a call reads and writes: `rows` rounded up to the plan's count.
pub fn paddedRows(rows: usize) usize {
    return (rows + step - 1) / step * step;
}

/// PART[0][row][e] = x[row] . w[e] for `paddedRows(rows)` rows, on `s`; false when cuBLASLt is not there (once: the
/// tiled router serves from then on).
pub fn logits(s: cuda.Stream, x: u64, w: u64, part: u64, rows: usize, d: usize, e: usize) !bool {
    const m = paddedRows(rows);
    const slot = m / step - 1;
    if (slot >= max_plans) return false;
    if (lib == null) {
        lib = lt_mod.Library.open() catch |err| {
            std.log.warn("router: cuBLASLt not available ({t}); prompt chunks use the tiled router", .{err});
            failed = true;
            return false;
        };
        dims = .{ d, e };
    }
    if (dims[0] != d or dims[1] != e) return false; // one router shape a process (every layer's is the same)
    if (plans[slot] == null) plans[slot] = try lt_mod.Linear.init(&lib.?, m, e, d, .f32, 0);
    try plans[slot].?.run(x, w, part, 0, 0, s.handle);
    return true;
}
