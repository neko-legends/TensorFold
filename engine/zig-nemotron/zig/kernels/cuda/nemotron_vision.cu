// Nemotron 3.5 Super VL's vision path (M3): the C-RADIO tower (32 pre-norm blocks, LayerNorm + full attention, hdim 80,
// GELU MLP, CPE patch generator) and the projector (final LayerNorm, pixel shuffle 2 x 2, RMSNorm, ReLU^2 MLP).
// Matrix products run in cuBLASLt (bias epilogue); these are the glue ops between them, bf16 in and out with fp32 math.
//   layernorm   F.layer_norm over D (affine, eps), fp32 mean / variance, one block a row
//   patchify    pixels bf16 [3, H, W] -> patches [rows * cols, 3 * 16 * 16] (c, py, px order)
//   embed       patch rows + bilinear-resampled CPE position table (fp32 interpolate, rounded to bf16, then the bf16
//               add) after the 10 cls / register rows: x [10 + rows * cols, D]
//   attn        FA2-style forward (m16n8k16 bf16 MMA, fp32 online softmax, P in bf16) straight from the qkv rows
//   gelu        F.gelu (erf), in place;  add  x += y;  relu2  relu(x)^2 in place
//   shuffle_rms the 2 x 2 pixel shuffle ((dr * 2 + dc) * C + c channel order) fused with the projector's RMSNorm
#include <cuda_bf16.h>
#include <cstdint>

namespace nemotron_vision {

__device__ __forceinline__ float bf(const __nv_bfloat16 v) { return __bfloat162float(v); }
__device__ __forceinline__ __nv_bfloat16 rn(const float v) { return __float2bfloat16_rn(v); }

__device__ float block_sum(float v, float* red) {
    for (int off = 16; off > 0; off >>= 1) v += __shfl_down_sync(0xffffffffu, v, off);
    const int lane = threadIdx.x & 31, warp = threadIdx.x >> 5;
    __syncthreads();
    if (lane == 0) red[warp] = v;
    __syncthreads();
    const int nw = (blockDim.x + 31) >> 5;
    float t = lane < nw ? red[lane] : 0.f;
    for (int off = 16; off > 0; off >>= 1) t += __shfl_down_sync(0xffffffffu, t, off);
    return __shfl_sync(0xffffffffu, t, 0);
}

extern "C" __global__ void tf_vis_layernorm(const __nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ w,
                                            const __nv_bfloat16* __restrict__ b, __nv_bfloat16* __restrict__ out, int D, float eps) {
    __shared__ float red[32];
    const __nv_bfloat16* xr = x + (size_t)blockIdx.x * D;
    float s = 0.f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) s += bf(xr[j]);
    const float mean = block_sum(s, red) / (float)D;
    float v = 0.f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        const float d = bf(xr[j]) - mean;
        v += d * d;
    }
    const float r = rsqrtf(block_sum(v, red) / (float)D + eps);
    for (int j = threadIdx.x; j < D; j += blockDim.x)
        out[(size_t)blockIdx.x * D + j] = rn((bf(xr[j]) - mean) * r * bf(w[j]) + bf(b[j]));
}

extern "C" __global__ void tf_vis_patchify(const __nv_bfloat16* __restrict__ px, __nv_bfloat16* __restrict__ out, int rows, int cols) {
    const int p = blockIdx.x;
    const int pr = p / cols, pc = p % cols;
    const int H = rows * 16, W = cols * 16;
    for (int k = threadIdx.x; k < 768; k += blockDim.x) {
        const int c = k / 256, py = (k / 16) % 16, pxx = k % 16;
        out[(size_t)p * 768 + k] = px[((size_t)c * H + pr * 16 + py) * W + pc * 16 + pxx];
    }
}

// x [10 + rows * cols, D]: prefix rows, then bf16(e + bf16(bilinear(pos 128 x 128 -> md x md, cropped)))
extern "C" __global__ void tf_vis_embed(const __nv_bfloat16* __restrict__ e, const __nv_bfloat16* __restrict__ pos,
                                        const __nv_bfloat16* __restrict__ prefix, __nv_bfloat16* __restrict__ x,
                                        int rows, int cols, int D, int nprefix, int table) {
    const int i = blockIdx.x;
    if (i < nprefix) {
        for (int c = threadIdx.x; c < D; c += blockDim.x) x[(size_t)i * D + c] = prefix[(size_t)i * D + c];
        return;
    }
    const int p = i - nprefix;
    const int pr = p / cols, pc = p % cols;
    const int md = rows > cols ? rows : cols;
    const float scale = (float)table / (float)md;
    float sy = ((float)pr + 0.5f) * scale - 0.5f;
    float sx = ((float)pc + 0.5f) * scale - 0.5f;
    if (sy < 0.f) sy = 0.f;
    if (sx < 0.f) sx = 0.f;
    int y0 = (int)sy, x0 = (int)sx;
    const int y1 = y0 + 1 < table ? y0 + 1 : table - 1;
    const int x1 = x0 + 1 < table ? x0 + 1 : table - 1;
    if (y0 > table - 1) y0 = table - 1;
    if (x0 > table - 1) x0 = table - 1;
    const float ly = sy - (float)y0, lx = sx - (float)x0;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        const float v00 = bf(pos[((size_t)y0 * table + x0) * D + c]), v01 = bf(pos[((size_t)y0 * table + x1) * D + c]);
        const float v10 = bf(pos[((size_t)y1 * table + x0) * D + c]), v11 = bf(pos[((size_t)y1 * table + x1) * D + c]);
        const float top = v00 * (1.f - lx) + v01 * lx, bot = v10 * (1.f - lx) + v11 * lx;
        const float pv = bf(rn(top * (1.f - ly) + bot * ly));
        x[(size_t)i * D + c] = rn(bf(e[(size_t)p * D + c]) + pv);
    }
}

// ---- attention (hdim 80, 128 x 128 tiles, 4 warps, m16n8k16 bf16 MMA) ---------------------------------------------
__device__ __forceinline__ void mma16816(float* d, const uint32_t* a, const uint32_t* b) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}
__device__ __forceinline__ uint32_t pack2(__nv_bfloat16 lo, __nv_bfloat16 hi) {
    return (uint32_t)__bfloat16_as_ushort(lo) | ((uint32_t)__bfloat16_as_ushort(hi) << 16);
}

constexpr int BM = 128, BN = 128, HD = 80;

// qkv [n, 3 * heads * 80] rows (q | k | v); one CTA: 128 query rows of one head; warp w owns rows w*16 + {0, 64}
extern "C" __global__ void __launch_bounds__(128) tf_vis_attn(const __nv_bfloat16* __restrict__ QKV, __nv_bfloat16* __restrict__ O,
                                                              int n, int heads, float scale_log2) {
    __shared__ __nv_bfloat16 sk[BN][HD + 8];
    __shared__ __nv_bfloat16 sv[BN][HD + 8];
    const int head = blockIdx.y, m0 = blockIdx.x * BM;
    const int tid = threadIdx.x, warp = tid / 32, lane = tid % 32;
    const int g = lane / 4, q4 = lane % 4;
    const size_t rs = (size_t)3 * heads * HD;
    const __nv_bfloat16* Qh = QKV + (size_t)head * HD;
    const __nv_bfloat16* Kh = QKV + (size_t)heads * HD + (size_t)head * HD;
    const __nv_bfloat16* Vh = QKV + (size_t)2 * heads * HD + (size_t)head * HD;
    uint32_t qa[2][HD / 16][4];
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int kk = 0; kk < HD / 16; kk++)
#pragma unroll
            for (int e = 0; e < 4; e++) {
                const int r = m0 + warp * 16 + mi * 64 + g + (e & 1) * 8;
                const int c = kk * 16 + q4 * 2 + (e >> 1) * 8;
                const __nv_bfloat16 z = __float2bfloat16_rn(0.f);
                qa[mi][kk][e] = r < n ? pack2(Qh[(size_t)r * rs + c], Qh[(size_t)r * rs + c + 1]) : pack2(z, z);
            }
    float acc_o[2][HD / 8][4];
    float row_max[2][2], row_sum[2][2];
#pragma unroll
    for (int mi = 0; mi < 2; mi++) {
#pragma unroll
        for (int d = 0; d < HD / 8; d++) acc_o[mi][d][0] = acc_o[mi][d][1] = acc_o[mi][d][2] = acc_o[mi][d][3] = 0.f;
        row_max[mi][0] = row_max[mi][1] = -INFINITY;
        row_sum[mi][0] = row_sum[mi][1] = 0.f;
    }
    const int nblocks = (n + BN - 1) / BN;
    for (int nb = 0; nb < nblocks; nb++) {
        __syncthreads();
        for (int i = tid; i < BN * HD; i += 128) {
            const int r = i / HD, c = i % HD, key = nb * BN + r;
            sk[r][c] = key < n ? Kh[(size_t)key * rs + c] : __float2bfloat16_rn(0.f);
            sv[r][c] = key < n ? Vh[(size_t)key * rs + c] : __float2bfloat16_rn(0.f);
        }
        __syncthreads();
#pragma unroll
        for (int mi = 0; mi < 2; mi++) {
            float s[16][4];
#pragma unroll
            for (int ni = 0; ni < 16; ni++) s[ni][0] = s[ni][1] = s[ni][2] = s[ni][3] = 0.f;
#pragma unroll
            for (int kk = 0; kk < HD / 16; kk++) {
                const uint32_t* a = qa[mi][kk];
#pragma unroll
                for (int ni = 0; ni < 16; ni++) {
                    uint32_t b[2];
                    b[0] = pack2(sk[ni * 8 + g][kk * 16 + q4 * 2], sk[ni * 8 + g][kk * 16 + q4 * 2 + 1]);
                    b[1] = pack2(sk[ni * 8 + g][kk * 16 + q4 * 2 + 8], sk[ni * 8 + g][kk * 16 + q4 * 2 + 9]);
                    mma16816(s[ni], a, b);
                }
            }
#pragma unroll
            for (int ni = 0; ni < 16; ni++)
#pragma unroll
                for (int e = 0; e < 4; e++)
                    if (nb * BN + ni * 8 + q4 * 2 + (e & 1) >= n) s[ni][e] = -INFINITY;
#pragma unroll
            for (int hr = 0; hr < 2; hr++) {
                float mx = row_max[mi][hr];
#pragma unroll
                for (int ni = 0; ni < 16; ni++)
#pragma unroll
                    for (int e = 0; e < 2; e++) mx = fmaxf(mx, s[ni][hr * 2 + e]);
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 2));
                mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, 1));
                const float prev = row_max[mi][hr];
                const float sc = prev == -INFINITY ? 0.f : exp2f((prev - mx) * scale_log2);
                row_sum[mi][hr] *= sc;
#pragma unroll
                for (int d = 0; d < HD / 8; d++) {
                    acc_o[mi][d][hr * 2] *= sc;
                    acc_o[mi][d][hr * 2 + 1] *= sc;
                }
                row_max[mi][hr] = mx;
                const float ms = mx * scale_log2;
                float sum = row_sum[mi][hr];
#pragma unroll
                for (int ni = 0; ni < 16; ni++)
#pragma unroll
                    for (int e = 0; e < 2; e++) {
                        const float p = exp2f(s[ni][hr * 2 + e] * scale_log2 - ms);
                        s[ni][hr * 2 + e] = p;
                        sum += p;
                    }
                row_sum[mi][hr] = sum;
            }
#pragma unroll
            for (int kk = 0; kk < BN / 16; kk++) {
                uint32_t a[4];
                a[0] = pack2(rn(s[2 * kk][0]), rn(s[2 * kk][1]));
                a[1] = pack2(rn(s[2 * kk][2]), rn(s[2 * kk][3]));
                a[2] = pack2(rn(s[2 * kk + 1][0]), rn(s[2 * kk + 1][1]));
                a[3] = pack2(rn(s[2 * kk + 1][2]), rn(s[2 * kk + 1][3]));
#pragma unroll
                for (int d = 0; d < HD / 8; d++) {
                    uint32_t b[2];
                    b[0] = pack2(sv[kk * 16 + q4 * 2][d * 8 + g], sv[kk * 16 + q4 * 2 + 1][d * 8 + g]);
                    b[1] = pack2(sv[kk * 16 + q4 * 2 + 8][d * 8 + g], sv[kk * 16 + q4 * 2 + 9][d * 8 + g]);
                    mma16816(acc_o[mi][d], a, b);
                }
            }
        }
    }
#pragma unroll
    for (int mi = 0; mi < 2; mi++)
#pragma unroll
        for (int hr = 0; hr < 2; hr++) {
            float sum = row_sum[mi][hr];
            sum += __shfl_xor_sync(0xffffffffu, sum, 2);
            sum += __shfl_xor_sync(0xffffffffu, sum, 1);
            const float inv = (sum == 0.f || sum != sum) ? 1.f : 1.f / sum;
            const int row = m0 + warp * 16 + mi * 64 + g + hr * 8;
            if (row >= n) continue;
#pragma unroll
            for (int d = 0; d < HD / 8; d++) {
                const int col = d * 8 + q4 * 2;
                __nv_bfloat16* o = O + (size_t)row * heads * HD + head * HD + col;
                o[0] = rn(acc_o[mi][d][hr * 2] * inv);
                o[1] = rn(acc_o[mi][d][hr * 2 + 1] * inv);
            }
        }
}

extern "C" __global__ void tf_vis_gelu(__nv_bfloat16* __restrict__ x, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = bf(x[i]);
    x[i] = rn(v * 0.5f * (1.0f + erff(v * 0.70710678118654752440f)));
}

extern "C" __global__ void tf_vis_relu2(__nv_bfloat16* __restrict__ x, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = fmaxf(bf(x[i]), 0.f);
    x[i] = rn(bf(rn(v)) * bf(rn(v)));
}

extern "C" __global__ void tf_vis_add(__nv_bfloat16* __restrict__ x, const __nv_bfloat16* __restrict__ y, long long n) {
    const long long i = (long long)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    x[i] = rn(bf(x[i]) + bf(y[i]));
}

// feats [rows * cols, C] (row-major patches) -> out [(rows / 2) * (cols / 2), 4 C]: token (r2, c2) = the four patches
// (2 r2 + dr, 2 c2 + dc) at channel (dr * 2 + dc) * C + c, then RMSNorm over 4 C: bf16(w * (x * rsqrt(mean(x^2) + eps)))
extern "C" __global__ void tf_vis_shuffle_rms(const __nv_bfloat16* __restrict__ feats, const __nv_bfloat16* __restrict__ w,
                                              __nv_bfloat16* __restrict__ out, int rows, int cols, int C, float eps) {
    __shared__ float red[32];
    const int t = blockIdx.x;
    const int c2n = cols / 2;
    const int r2 = t / c2n, c2 = t % c2n;
    const int D = 4 * C;
    float s = 0.f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        const int q = j / C, c = j % C;
        const float v = bf(feats[((size_t)(2 * r2 + (q >> 1)) * cols + 2 * c2 + (q & 1)) * C + c]);
        s += v * v;
    }
    const float r = rsqrtf(block_sum(s, red) / (float)D + eps);
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        const int q = j / C, c = j % C;
        const float v = bf(feats[((size_t)(2 * r2 + (q >> 1)) * cols + 2 * c2 + (q & 1)) * C + c]);
        out[(size_t)t * D + j] = rn(bf(w[j]) * (v * r));
    }
}

} // namespace nemotron_vision
