// Nemotron's router: fp32 logits in K slices (one fma chain a slice), then each row's top-k and the shared slots.

#include <cuda_bf16.h>
#include <math.h>
#include <stdint.h>

#include "glue_math.cuh"

using namespace tf_glue;

// Block (16-row tile, 16-expert block, K slice s): PART[s][row][e] = x[row] . w[e] over the slice, in index order.
extern "C" __global__ void __launch_bounds__(256) tf_nemo_router(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                   float* __restrict__ PART, int R, int D, int E, int SK) {
    extern __shared__ uint32_t tile[];  // 16 input rows, then 16 expert rows, each per / 2 words plus one of padding
    const int per = D / SK, words = per / 2, stride = words + 1;
    const int row0 = blockIdx.x * 16, e0 = blockIdx.y * 16, k0 = blockIdx.z * per;
    uint32_t* xs = tile;
    uint32_t* ws = tile + 16 * stride;
    for (int c = threadIdx.x; c < 16 * words; c += 256) {
        const int r = c / words, w = c % words;
        const uint32_t* xr = reinterpret_cast<const uint32_t*>(X + static_cast<int64_t>(row0 + r) * D + k0);
        const uint32_t* wr = reinterpret_cast<const uint32_t*>(W + static_cast<int64_t>(e0 + r) * D + k0);
        xs[r * stride + w] = row0 + r < R ? xr[w] : 0u;
        ws[r * stride + w] = e0 + r < E ? wr[w] : 0u;
    }
    __syncthreads();
    const int rr = threadIdx.x / 16, ee = threadIdx.x % 16;
    const uint32_t* xp = xs + rr * stride;
    const uint32_t* wp = ws + ee * stride;
    float acc = 0.0f;
    for (int w = 0; w < words; ++w) {
        const uint32_t a = xp[w], b = wp[w];
        acc = __fmaf_rn(__uint_as_float(a << 16), __uint_as_float(b << 16), acc);
        acc = __fmaf_rn(__uint_as_float(a & 0xFFFF0000u), __uint_as_float(b & 0xFFFF0000u), acc);
    }
    if (row0 + rr < R && e0 + ee < E) PART[(static_cast<int64_t>(blockIdx.z) * R + row0 + rr) * E + e0 + ee] = acc;
}

// Prompt rows: tf_nemo_router's sums, bit for bit (each output one fma chain over its slice in index order, from 0),
// a block 128 rows x 64 experts of slice blockIdx.z and a thread an 8 x 4 tile of them, staged 32 elements at a time
// as fp32 in [k][row] order: three 16-byte shared loads feed 32 fmas (3.5x the 16 x 16 tile's speed at 4,096 rows).
extern "C" __global__ void __launch_bounds__(256) tf_nemo_router_tiles(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                         float* __restrict__ PART, int R, int D, int E, int SK) {
    constexpr int RT = 8, ET = 4, TR = 16 * RT, TE = 16 * ET, KC = 32;
    __shared__ __align__(16) float xs[KC][TR], ws[KC][TE];
    const int per = D / SK, row0 = blockIdx.x * TR, e0 = blockIdx.y * TE, k0 = blockIdx.z * per;
    const int tr = threadIdx.x / 16, te = threadIdx.x % 16;
    float acc[RT][ET];
#pragma unroll
    for (int i = 0; i < RT; ++i)
#pragma unroll
        for (int j = 0; j < ET; ++j) acc[i][j] = 0.0f;
    for (int kc = 0; kc < per; kc += KC) {
        const int n = min(KC, per - kc);
        __syncthreads();
        // 8 elements (16 bytes) a load; a warp's lanes on consecutive rows, so its shared stores never collide
        for (int c = threadIdx.x; c < (TR + TE) * (KC / 8); c += 256) {
            const bool isx = c < TR * (KC / 8);
            const int cc = isx ? c : c - TR * (KC / 8), T = isx ? TR : TE;
            const int r = cc % T, k8 = (cc / T) * 8;
            uint4 v = make_uint4(0u, 0u, 0u, 0u);
            if (k8 < n) {
                if (isx && row0 + r < R) v = *reinterpret_cast<const uint4*>(X + static_cast<int64_t>(row0 + r) * D + k0 + kc + k8);
                if (!isx && e0 + r < E) v = *reinterpret_cast<const uint4*>(W + static_cast<int64_t>(e0 + r) * D + k0 + kc + k8);
            }
            const uint32_t u[4] = {v.x, v.y, v.z, v.w};
#pragma unroll
            for (int q = 0; q < 4; ++q) {
                const float lo = __uint_as_float(u[q] << 16), hi = __uint_as_float(u[q] & 0xFFFF0000u);
                if (isx) xs[k8 + 2 * q][r] = lo, xs[k8 + 2 * q + 1][r] = hi;
                else ws[k8 + 2 * q][r] = lo, ws[k8 + 2 * q + 1][r] = hi;
            }
        }
        __syncthreads();
#pragma unroll 4
        for (int k = 0; k < n; ++k) {
            float av[RT], bv[ET];
#pragma unroll
            for (int i = 0; i < RT; i += 4) {
                const float4 a = *reinterpret_cast<const float4*>(&xs[k][RT * tr + i]);
                av[i] = a.x, av[i + 1] = a.y, av[i + 2] = a.z, av[i + 3] = a.w;
            }
            const float4 b = *reinterpret_cast<const float4*>(&ws[k][ET * te]);
            bv[0] = b.x, bv[1] = b.y, bv[2] = b.z, bv[3] = b.w;
#pragma unroll
            for (int i = 0; i < RT; ++i)
#pragma unroll
                for (int j = 0; j < ET; ++j) acc[i][j] = __fmaf_rn(av[i], bv[j], acc[i][j]);
        }
    }
#pragma unroll
    for (int i = 0; i < RT; ++i) {
        const int row = row0 + RT * tr + i;
        if (row >= R) continue;
#pragma unroll
        for (int j = 0; j < ET; ++j) {
            const int e = e0 + ET * te + j;
            if (e < E) PART[(static_cast<int64_t>(blockIdx.z) * R + row) * E + e] = acc[i][j];
        }
    }
}

// Window rows (R <= 16, D a multiple of 256): a warp an expert reads its weight row once in 16-byte loads against every
// row, PART[row][e] (one slice). Decode reads the 4 MB router at full bandwidth instead of staging 16-row tiles.
extern "C" __global__ void __launch_bounds__(256) tf_nemo_router_rows(const __nv_bfloat16* __restrict__ X, const __nv_bfloat16* __restrict__ W,
                                                                        float* __restrict__ PART, int R, int D, int E) {
    const int warp = threadIdx.x / 32, lane = threadIdx.x % 32;
    const int e = blockIdx.x * 8 + warp;
    if (e >= E) return;
    float acc[16];
#pragma unroll
    for (int r = 0; r < 16; ++r) acc[r] = 0.0f;
    const uint4* wr = reinterpret_cast<const uint4*>(W + static_cast<int64_t>(e) * D);
    const int n8 = D / 8;
    for (int c = lane; c < n8; c += 32) {
        const uint4 wv = __ldg(wr + c);
        const uint32_t ww[4] = {wv.x, wv.y, wv.z, wv.w};
#pragma unroll
        for (int r = 0; r < 16; ++r) {
            if (r >= R) break;
            const uint4 xv = __ldg(reinterpret_cast<const uint4*>(X + static_cast<int64_t>(r) * D) + c);
            const uint32_t xx[4] = {xv.x, xv.y, xv.z, xv.w};
            float a = acc[r];
#pragma unroll
            for (int j = 0; j < 4; ++j) {
                a = __fmaf_rn(__uint_as_float(xx[j] << 16), __uint_as_float(ww[j] << 16), a);
                a = __fmaf_rn(__uint_as_float(xx[j] & 0xFFFF0000u), __uint_as_float(ww[j] & 0xFFFF0000u), a);
            }
            acc[r] = a;
        }
    }
#pragma unroll
    for (int r = 0; r < 16; ++r) {
        if (r >= R) break;
        float v = acc[r];
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) v += __shfl_xor_sync(0xffffffffu, v, off);
        if (lane == 0) PART[static_cast<int64_t>(r) * E + e] = v;
    }
}

namespace {

constexpr int MAXC = 16;  // experts a lane holds: up to 512 experts (Super VL)
constexpr int MAXK = 32;  // routed picks kept per row: top-k up to 32 (Super VL picks 22)

// The larger of two (value, index) picks: a higher value, or the lower index on a tie (values are never NaN).
__device__ __forceinline__ bool beats(float v, int i, float bv, int bi) { return v > bv || (v == bv && i < bi); }

}  // namespace

// Sum logits' slices in order; rank sigmoid scores plus bias before adding the shared slots.
extern "C" __global__ void __launch_bounds__(32) tf_nemo_topk(const float* __restrict__ PART, const float* __restrict__ BIAS,
                                                               int* __restrict__ IDX, float* __restrict__ WT, int R, float scaling, int E,
                                                               int SK, int TOPK, int NS, int norm) {
    const int64_t r = blockIdx.x;
    const int lane = threadIdx.x;
    float score[MAXC], sel[MAXC];
#pragma unroll
    for (int c = 0; c < MAXC; ++c) {
        const int e = lane + 32 * c;
        score[c] = 0.0f;
        sel[c] = -INFINITY;
        if (e >= E) continue;
        float logit = 0.0f;
        for (int s = 0; s < SK; ++s) logit = __fadd_rn(logit, PART[(s * R + r) * E + e]);
        score[c] = tf_sigmoid(logit);
        const float v = __fadd_rn(score[c], BIAS[e]);
        sel[c] = v != v ? -INFINITY : v;  // a NaN logit is never picked ahead of a number
    }
    float total = 0.0f;
    float probs[MAXK];
    int ids[MAXK];
    for (int k = 0; k < TOPK; ++k) {
        float bv = -INFINITY;
        int bi = 0x7fffffff;
#pragma unroll
        for (int c = 0; c < MAXC; ++c)
            if (lane + 32 * c < E && beats(sel[c], lane + 32 * c, bv, bi)) bv = sel[c], bi = lane + 32 * c;
#pragma unroll
        for (int off = 16; off > 0; off >>= 1) {
            const float ov = __shfl_xor_sync(0xffffffffu, bv, off);
            const int oi = __shfl_xor_sync(0xffffffffu, bi, off);
            if (beats(ov, oi, bv, bi)) bv = ov, bi = oi;
        }
        float p = 0.0f;
#pragma unroll
        for (int c = 0; c < MAXC; ++c)
            if (bi == lane + 32 * c) p = score[c], sel[c] = -INFINITY;
        p = __shfl_sync(0xffffffffu, p, bi % 32);
        if (k < MAXK) probs[k] = p, ids[k] = bi;
        total = __fadd_rn(total, p);
    }
    if (lane != 0) return;
    for (int k = 0; k < NS; ++k) {
        float w = 1.0f;
        int id = E + (k - TOPK);
        if (k < TOPK) {
            id = ids[k];
            w = norm ? __fmul_rn(__fdiv_rn(probs[k], __fadd_rn(total, 1e-20f)), scaling) : __fmul_rn(probs[k], scaling);
        }
        IDX[r * NS + k] = id;
        WT[r * NS + k] = w;
    }
}

// tf_nemo_topk's picks and weights with a thread an expert (E <= blockDim.x <= 512): each expert's rank among the
// others is its count of experts that beat it, so the top-k land in tf_nemo_topk's order (value, then lower index)
// and the weights sum in that order: the same bits, without TOPK rounds of a one-warp argmax.
extern "C" __global__ void __launch_bounds__(512) tf_nemo_topk_wide(const float* __restrict__ PART, const float* __restrict__ BIAS,
                                                                     int* __restrict__ IDX, float* __restrict__ WT, int R, float scaling,
                                                                     int E, int SK, int TOPK, int NS, int norm) {
    __shared__ float sel_s[512];
    __shared__ float probs[MAXK];
    __shared__ int ids[MAXK];
    const int64_t r = blockIdx.x;
    const int e = threadIdx.x;
    float score = 0.0f, sel = -INFINITY;
    if (e < E) {
        float logit = 0.0f;
        for (int s = 0; s < SK; ++s) logit = __fadd_rn(logit, PART[(s * R + r) * E + e]);
        score = tf_sigmoid(logit);
        const float v = __fadd_rn(score, BIAS[e]);
        sel = v != v ? -INFINITY : v;
    }
    sel_s[e] = sel;
    __syncthreads();
    if (e < E) {
        int rank = 0;
        for (int j = 0; j < E; ++j) rank += beats(sel_s[j], j, sel, e);
        if (rank < TOPK && rank < MAXK) ids[rank] = e, probs[rank] = score;
    }
    __syncthreads();
    if (e != 0) return;
    float total = 0.0f;
    for (int k = 0; k < TOPK; ++k) total = __fadd_rn(total, probs[k]);
    for (int k = 0; k < NS; ++k) {
        float w = 1.0f;
        int id = E + (k - TOPK);
        if (k < TOPK) {
            id = ids[k];
            w = norm ? __fmul_rn(__fdiv_rn(probs[k], __fadd_rn(total, 1e-20f)), scaling) : __fmul_rn(probs[k], scaling);
        }
        IDX[r * NS + k] = id;
        WT[r * NS + k] = w;
    }
}
