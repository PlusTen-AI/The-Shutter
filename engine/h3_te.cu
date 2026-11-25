// shutter engine — MiniMax H3 text encoder: Qwen3-VL-32B cut at 50 LM layers (nvfp4 + AWQ LLM, bf16 vision tower), C API
// Numerics follow ComfyUI (comfy/text_encoders/minimax.py, qwen3vl.py, llama.py, qwen35.py; comfy/ops.py): the encoder
// runs with fp32 activations and full_precision_matrix_mult, i.e. y = (x * pre_quant_scale) . W^T with the nvfp4 weight
// dequantized to fp32, W = e2m1 * e4m3_block_scale * tensor_scale. W / tensor_scale = e2m1 * e4m3 has <= 6 significant
// bits, so it is exact in fp16 (range 2^-10 .. 2688). We expand each linear to that fp16 W' just before its GEMM, split
// the fp32 input rows into two fp16 planes (x = sc (h0 + h1), sc a power of two per row, ~22 significant bits), run one
// fp16 tensor-core GEMM over the stacked planes with fp32 accumulation (products are exact), and combine in fp32.
// Token embeddings: int8 rows * row scale rounded to bf16 once. Norms, RoPE, attention and the vision tower (bf16
// weights cast to fp32; patch embed in TF32 like cuDNN's Conv3d) are fp32.
// Output = every row of the last layer's hidden state (no final norm), fp32.
#include "common.cuh"
#include "st.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace {
constexpr int D = 5120, NL = 50, NH = 64, HD = 128, NKV = 8, FF = 25600;
constexpr int QN = NH * HD, KVN = NKV * HD, QKVN = QN + 2 * KVN;
constexpr int VD = 1152, VNH = 16, VHD = 72, VFF = 4304, VNB = 27, VMD = 4 * VD, VPK = 1536, VGRID = 48;
constexpr int NDS = 3;
constexpr int DS_AT[NDS] = {8, 16, 24};

// ---------------------------------------------------------------------------------------------- kernels
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ __forceinline__ float warp_max(float v) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    return v;
}
template <bool MAX>
__device__ float block_reduce(float v, float* red) {
    v = MAX ? warp_max(v) : warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (l == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        v = l < nw ? red[l] : 0.f;
        v = MAX ? warp_max(v) : warp_sum(v);
        if (l == 0) red[0] = v;
    }
    __syncthreads();
    v = red[0];
    __syncthreads();
    return v;
}

// nvfp4 [N, K/2] (hi nibble = even element) + swizzled e4m3 block scales (cuBLAS 128x4 tiles) -> fp16 W' [N, K]
__device__ __forceinline__ float e4m3(uint32_t b) {
    const uint32_t e = (b >> 3) & 15, m = b & 7;
    const float v = e ? __uint_as_float(((e + 120) << 23) | (m << 20)) : (float)m * 0x1p-9f;
    return (b & 0x80) ? -v : v;
}
// one thread per 16-element block: 8 packed bytes, one scale; e2m1 * 2 = nibble of 0xC8643210 (exact products)
__global__ void k_expand(const uint8_t* __restrict__ q, const uint8_t* __restrict__ bs, int N, int K, __half* __restrict__ w) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int per_row = K / 16;
    if (i >= (size_t)N * per_row) return;
    const int n = (int)(i / per_row), c = (int)(i % per_row), ncb = K / 64;
    const size_t so = (size_t)((n >> 7) * ncb + (c >> 2)) * 512 + (n & 31) * 16 + ((n >> 5) & 3) * 4 + (c & 3);
    const float s = 0.5f * e4m3(bs[so]);
    const uint2 p = *(const uint2*)(q + i * 8);
    __align__(16) __half o[16];
    for (int b = 0; b < 8; b++) {
        const uint32_t by = ((b < 4 ? p.x : p.y) >> (8 * (b & 3))) & 0xff;
        for (int z = 0; z < 2; z++) {
            const uint32_t cd = z ? by & 15 : by >> 4;
            const float v = (float)((0xC8643210u >> (4 * (cd & 7))) & 15) * s;
            o[2 * b + z] = __float2half_rn((cd & 8) ? -v : v);
        }
    }
    *(uint4*)(w + i * 16) = *(const uint4*)o;
    *(uint4*)(w + i * 16 + 8) = *(const uint4*)(o + 8);
}

// fp32 rows -> pre-op -> two fp16 planes (plane stride `plane`), x = sc * (h0 + h1), sc = 2^k with |x / sc| < 2^15
enum { S_RMS = 0, S_PQS = 1, S_SWIGLU = 2 };
template <int MODE>
__device__ __forceinline__ float pre(const float* x, int j, int K, float r, const float* w) {
    if (MODE == S_RMS) return x[j] * r * w[j];
    if (MODE == S_PQS) return x[j] * w[j];
    const float g = x[j], u = x[K + j];
    return g / (1.f + expf(-g)) * u * w[j];
}
template <int MODE>
__global__ void __launch_bounds__(256) k_split(const float* __restrict__ in, int ld, int K, const float* __restrict__ w,
                                               __half* __restrict__ h, size_t plane, float* __restrict__ sc) {
    __shared__ float red[32];
    const float* x = in + (size_t)blockIdx.x * ld;
    float r = 1.f;
    if (MODE == S_RMS) {
        float s = 0.f;
        for (int j = threadIdx.x; j < K; j += blockDim.x) s += x[j] * x[j];
        r = rsqrtf(block_reduce<false>(s, red) / K + 1e-6f);
    }
    float mx = 0.f;
    for (int j = threadIdx.x; j < K; j += blockDim.x) mx = fmaxf(mx, fabsf(pre<MODE>(x, j, K, r, w)));
    mx = block_reduce<true>(mx, red);
    const int e = mx > 0x1p-90f ? ilogbf(mx) - 14 : -104;  // mx / 2^e in [2^14, 2^15) (or below)
    const float inv = ldexpf(1.f, -e);                        // power of two: scaling is exact
    __half* h0 = h + (size_t)blockIdx.x * K;
    for (int j = threadIdx.x; j < K; j += blockDim.x) {
        const float v = pre<MODE>(x, j, K, r, w) * inv;
        const __half a = __float2half_rn(v);
        h0[j] = a;
        h0[plane + j] = __float2half_rn(v - __half2float(a));
    }
    if (threadIdx.x == 0) sc[blockIdx.x] = ldexpf(1.f, e);
}

// per-column tensor scale: t0 below n1, t1 below n2, t2 above
struct Seg { int n1, n2; float t0, t1, t2; };
// out[m, n] (=, or +=) (acc[m, n] + acc[Mp + m, n]) * sc[m] * ts(n)
__global__ void k_comb(const float* __restrict__ acc, int M, int Mp, int N, const float* __restrict__ sc, Seg sg,
                       float* __restrict__ out, int add) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * N) return;
    const int m = (int)(i / N), n = (int)(i % N);
    const float ts = n < sg.n1 ? sg.t0 : n < sg.n2 ? sg.t1 : sg.t2;
    const float v = (acc[i] + acc[(size_t)Mp * N + i]) * sc[m] * ts;
    out[i] = add ? out[i] + v : v;
}

// int8 embedding rows * row scale -> bf16 -> fp32 (ids < 0: vision rows, left alone)
__global__ void k_embed(const int8_t* __restrict__ tab, const float* __restrict__ sc, const int* __restrict__ ids,
                        float* __restrict__ x) {
    const int id = ids[blockIdx.x];
    if (id < 0) return;
    const int8_t* r = tab + (size_t)id * D;
    for (int j = threadIdx.x; j < D; j += blockDim.x)
        x[(size_t)blockIdx.x * D + j] = __bfloat162float(__float2bfloat16((float)r[j] * sc[id]));
}

// one warp per (row, head 0..71): q heads then k heads in qkv [M, QKVN]; per-head RMSNorm, split-half RoPE, in place
__global__ void k_qkrope(float* __restrict__ qkv, int M, const float2* __restrict__ cs, const float* __restrict__ wq,
                         const float* __restrict__ wk) {
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int hh = gw % (NH + NKV), m = gw / (NH + NKV);
    if (m >= M) return;
    float* p = qkv + (size_t)m * QKVN + hh * HD;
    const float* w = hh < NH ? wq : wk;
    float a[4];
    float ss = 0.f;
    for (int k = 0; k < 4; k++) { a[k] = p[lane + 32 * k]; ss += a[k] * a[k]; }
    const float r = rsqrtf(warp_sum(ss) / HD + 1e-6f);
    for (int k = 0; k < 4; k++) a[k] = a[k] * r * w[lane + 32 * k];
    for (int k = 0; k < 2; k++) {
        const int i = lane + 32 * k;
        const float2 c = cs[(size_t)m * 64 + i];
        p[i] = a[k] * c.x - a[k + 2] * c.y;
        p[i + 64] = a[k + 2] * c.x + a[k] * c.y;
    }
}

// GQA: K/V heads repeated to 64 heads, [M, QN] each
// kv rows [M, ld]: k at columns 0..KVN, v at KVN..2*KVN -> K, V [M, QN] (GQA heads repeated)
__global__ void k_expand_kv(const float* __restrict__ kv, int ld, int M, float* __restrict__ K, float* __restrict__ V) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * QN) return;
    const int m = (int)(i / QN), c = (int)(i % QN);
    const int src = (c / HD / (NH / NKV)) * HD + c % HD;
    K[i] = kv[(size_t)m * ld + src];
    V[i] = kv[(size_t)m * ld + KVN + src];
}

// Fused fp32 attention (flash style, plain FMA, no tensor cores): the score matrix never leaves the SM. One block = FBR
// query rows of one head (8 warps x 16 rows), key tiles of FBC. Per tile: S = Q.K^T * scale (thread: 4 rows x 4 keys;
// even / odd groups of 4 d in two sums), mask (causal: query i sees keys <= off + i), online softmax inside the row's
// warp (running max m), the tile's P.V in its own sums (thread: 4 rows x HD/8 columns), O = O * exp(m_old - m) + tile,
// row sum l likewise (Kahan, per lane), O / l at the end. More accurate than the materialized SGEMM + softmax path.
// Shared memory: Q^T [HD][FBR], the K tile [FBC][HD+4] (its space takes P^T [FBC][FBR+4] once S is done), the V tile
// [FBC][HD]; the next K / V tile is fetched into registers while this one computes.
constexpr int FBR = 128, FBC = 32;
template <int HD> struct Fa {
    static constexpr int KP = HD + 4, PP = FBR + 4;   // pitches: float4 reads of 8 K rows / 4 P^T rows are conflict-free
    static constexpr int QS = HD * FBR, KS = FBC * (KP > PP ? KP : PP), VS = FBC * HD, SMEM = (QS + KS + VS) * 4;
    static constexpr int C4 = HD / 4, NT = (FBC * C4 + 255) / 256;          // float4 per row; per thread per K or V tile
    static constexpr int NV = HD / 32, NR = HD % 32 / 8, NO = 4 * NV + NR;  // P.V columns per lane: float4 groups, singles
};
__device__ __forceinline__ float f4(const float4& v, int e) { return e == 0 ? v.x : e == 1 ? v.y : e == 2 ? v.z : v.w; }
template <int HD>
__global__ void __launch_bounds__(256, 1) k_flash(const float* __restrict__ Q, int ldq, const float* __restrict__ K, int ldk,
                                                  const float* __restrict__ V, int ldv, float* __restrict__ O, int ldo,
                                                  int Lq, int Lk, int off, int causal, float scale) {
    static_assert(HD % 8 == 0 && HD % 32 % 8 == 0, "head dim");
    using F = Fa<HD>;
    extern __shared__ float4 fsm[];
    float* Qs = (float*)fsm;
    float* Ks = Qs + F::QS;   // K tile, then P^T
    float* Vs = Ks + F::KS;
    const int tid = threadIdx.x, lx = tid & 7, rb = 16 * (tid >> 5) + 4 * ((tid >> 3) & 3);   // rows rb..rb+3
    const int q0 = (causal ? gridDim.x - 1 - blockIdx.x : blockIdx.x) * FBR;                  // causal: long rows first
    const size_t hc = (size_t)blockIdx.y * HD;
    Q += hc, K += hc, V += hc, O += hc;
    for (int c = tid; c < FBR * F::C4; c += 256) {   // Q^T (rows past Lq: zero)
        const int r = c % FBR, d = c / FBR * 4;
        const float4 v = q0 + r < Lq ? *(const float4*)(Q + (size_t)(q0 + r) * ldq + d) : make_float4(0.f, 0.f, 0.f, 0.f);
        for (int e = 0; e < 4; e++) Qs[(d + e) * FBR + r] = f4(v, e);
    }
    const int nk = causal ? min(Lk, off + min(q0 + FBR, Lq)) : Lk;   // keys this block needs
    int lim[4];
    float m[4], l[4], lc[4], o[4][F::NO];
    for (int i = 0; i < 4; i++) {
        lim[i] = causal ? min(Lk, off + q0 + rb + i + 1) : Lk;
        m[i] = -INFINITY;
        l[i] = lc[i] = 0.f;
        for (int c = 0; c < F::NO; c++) o[i][c] = 0.f;
    }
    float4 kr[F::NT], vr[F::NT];
    auto fetch = [&](int k0) {
        for (int e = 0; e < F::NT; e++) {
            const int c = tid + 256 * e, r = c / F::C4, d = c % F::C4 * 4;
            kr[e] = vr[e] = make_float4(0.f, 0.f, 0.f, 0.f);
            if (c < FBC * F::C4 && k0 + r < Lk) {
                kr[e] = *(const float4*)(K + (size_t)(k0 + r) * ldk + d);
                vr[e] = *(const float4*)(V + (size_t)(k0 + r) * ldv + d);
            }
        }
    };
    fetch(0);
    for (int k0 = 0; k0 < nk; k0 += FBC) {
        __syncthreads();   // the last tile's P^T / V reads (and the Q^T stores) are done
        for (int e = 0; e < F::NT; e++) {
            const int c = tid + 256 * e, r = c / F::C4, d = c % F::C4 * 4;
            if (c < FBC * F::C4) {
                *(float4*)(Ks + r * F::KP + d) = kr[e];
                *(float4*)(Vs + r * HD + d) = vr[e];
            }
        }
        __syncthreads();
        if (k0 + FBC < nk) fetch(k0 + FBC);
        float s[2][4][4] = {};   // [d / 4 even, odd][row][key]
#pragma unroll
        for (int d = 0; d < HD; d += 4) {
            float4 kk[4];
            for (int j = 0; j < 4; j++) kk[j] = *(const float4*)(Ks + (lx + 8 * j) * F::KP + d);
            for (int e = 0; e < 4; e++) {
                const float4 q = *(const float4*)(Qs + (d + e) * FBR + rb);
                for (int i = 0; i < 4; i++)
                    for (int j = 0; j < 4; j++) s[d / 4 & 1][i][j] = fmaf(f4(q, i), f4(kk[j], e), s[d / 4 & 1][i][j]);
            }
        }
        __syncthreads();   // K tile done: its space takes P^T
        float a[4];
        for (int i = 0; i < 4; i++) {
            float mx = m[i];
            for (int j = 0; j < 4; j++) {
                s[0][i][j] = k0 + lx + 8 * j < lim[i] ? (s[0][i][j] + s[1][i][j]) * scale : -INFINITY;
                mx = fmaxf(mx, s[0][i][j]);
            }
            for (int x = 1; x < 8; x <<= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, x));
            a[i] = expf(m[i] - mx);   // first tile: 0 (key 0 is always visible, so mx is finite)
            m[i] = mx;
            for (int j = 0; j < 4; j++) s[0][i][j] = expf(s[0][i][j] - mx);
            const float la = l[i] * a[i], y = (s[0][i][0] + s[0][i][1]) + (s[0][i][2] + s[0][i][3]) - lc[i] * a[i];
            l[i] = la + y;
            lc[i] = (l[i] - la) - y;
        }
        for (int j = 0; j < 4; j++)
            *(float4*)(Ks + (lx + 8 * j) * F::PP + rb) = make_float4(s[0][0][j], s[0][1][j], s[0][2][j], s[0][3][j]);
        __syncwarp();      // a warp reads only its own P^T rows
        float t[4][F::NO] = {};
#pragma unroll
        for (int j = 0; j < FBC; j++) {
            const float4 p = *(const float4*)(Ks + j * F::PP + rb);
            for (int g = 0; g < F::NV; g++) {
                const float4 v = *(const float4*)(Vs + j * HD + 32 * g + 4 * lx);
                for (int i = 0; i < 4; i++)
                    for (int e = 0; e < 4; e++) t[i][4 * g + e] = fmaf(f4(p, i), f4(v, e), t[i][4 * g + e]);
            }
            for (int g = 0; g < F::NR; g++) {
                const float v = Vs[j * HD + 32 * F::NV + 8 * g + lx];
                for (int i = 0; i < 4; i++) t[i][4 * F::NV + g] = fmaf(f4(p, i), v, t[i][4 * F::NV + g]);
            }
        }
        for (int i = 0; i < 4; i++)
            for (int c = 0; c < F::NO; c++) o[i][c] = fmaf(o[i][c], a[i], t[i][c]);
    }
    for (int i = 0; i < 4; i++) {
        float s = l[i] - lc[i];
        for (int x = 1; x < 8; x <<= 1) s += __shfl_xor_sync(0xffffffffu, s, x);
        const int r = q0 + rb + i;
        if (r >= Lq) continue;
        const float inv = 1.f / s;
        float* y = O + (size_t)r * ldo;
        for (int g = 0; g < F::NV; g++)
            *(float4*)(y + 32 * g + 4 * lx) = make_float4(o[i][4 * g] * inv, o[i][4 * g + 1] * inv, o[i][4 * g + 2] * inv,
                                                          o[i][4 * g + 3] * inv);
        for (int g = 0; g < F::NR; g++) y[32 * F::NV + 8 * g + lx] = o[i][4 * F::NV + g] * inv;
    }
}

__global__ void __launch_bounds__(256) k_ln(const float* __restrict__ x, int K, const float* __restrict__ w,
                                            const float* __restrict__ b, float* __restrict__ y) {
    __shared__ float red[32];
    const float* xr = x + (size_t)blockIdx.x * K;
    float s = 0.f;
    for (int j = threadIdx.x; j < K; j += blockDim.x) s += xr[j];
    const float mean = block_reduce<false>(s, red) / K;
    float s2 = 0.f;
    for (int j = threadIdx.x; j < K; j += blockDim.x) { const float d = xr[j] - mean; s2 += d * d; }
    const float rstd = rsqrtf(block_reduce<false>(s2, red) / K + 1e-6f);
    for (int j = threadIdx.x; j < K; j += blockDim.x) y[(size_t)blockIdx.x * K + j] = (xr[j] - mean) * rstd * w[j] + b[j];
}

// v = y + bias; dst ? dst += v : y = act(v)   act: 0 none, 1 gelu tanh, 2 gelu erf
__global__ void k_bias(float* __restrict__ y, const float* __restrict__ b, size_t total, int N, int act, float* __restrict__ dst) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= total) return;
    float v = y[i] + b[i % N];
    if (dst) { dst[i] += v; return; }
    if (act == 1) v = 0.5f * v * (1.f + tanhf(0.7978845608028654f * (v + 0.044715f * v * v * v)));
    else if (act == 2) v = 0.5f * v * (1.f + erff(v * 0.7071067811865476f));
    y[i] = v;
}

__global__ void k_add(float* __restrict__ x, const float* __restrict__ y, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] += y[i];
}

__global__ void k_bf16_to_f32(const bf16* __restrict__ in, float* __restrict__ out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

// fp32 -> tf32 (round to nearest, ties away), kept in fp32: products of two such values are exact in fp32
__device__ __forceinline__ float tf32(float v) { return __uint_as_float((__float_as_uint(v) + 0x1000u) & ~0x1fffu); }
__global__ void k_tf32(float* __restrict__ x, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tf32(x[i]);
}

// frames [T, H, W, 3] in [0, 1] (T = 1: the image fills both temporal slots, T = 2: a video frame pair) ->
// normalized patches [Np, 3*2*16*16] in merge order (block row, block col, 2x2), element (c, t, py, px), tf32
__global__ void k_patch(const float* __restrict__ img, int T, int H, int W, float* __restrict__ P) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int gw = W / 16, Np = (H / 16) * gw;
    if (i >= (size_t)Np * VPK) return;
    const int p = (int)(i / VPK), e = (int)(i % VPK);
    const int mw = p & 1, mh = (p >> 1) & 1, blk = p >> 2, bw = blk % (gw / 2), bh = blk / (gw / 2);
    const int row = bh * 2 + mh, col = bw * 2 + mw;
    const int c = e / 512, t = T > 1 ? (e / 256) & 1 : 0, py = (e / 16) % 16, px = e % 16;
    P[i] = tf32((img[(((size_t)t * H + row * 16 + py) * W + col * 16 + px) * 3 + c] - 0.5f) / 0.5f);
}

// bilinear position embedding, bf16 math as ComfyUI: sum_k bf16(table[idx_k] * w_k), summed in bf16; x += it
__global__ void k_posemb(const bf16* __restrict__ tab, const int* __restrict__ idx, const bf16* __restrict__ wt, int Np,
                         float* __restrict__ x) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)Np * VD) return;
    const int t = (int)(i / VD), c = (int)(i % VD);
    bf16 s;
    for (int k = 0; k < 4; k++) {
        const bf16 p = __float2bfloat16(__bfloat162float(tab[(size_t)idx[k * Np + t] * VD + c]) * __bfloat162float(wt[k * Np + t]));
        s = k ? __float2bfloat16(__bfloat162float(s) + __bfloat162float(p)) : p;
    }
    x[i] += __bfloat162float(s);
}

// vision RoPE on q and k of qkv [Np, 3*1152]: one warp per (row, 32 heads), head dim 72, pairs (i, i+36)
__global__ void k_vrope(float* __restrict__ qkv, int M, const float2* __restrict__ cs) {
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int hh = gw % (2 * VNH), m = gw / (2 * VNH);
    if (m >= M) return;
    float* p = qkv + (size_t)m * 3 * VD + hh * VHD;  // q heads 0..15, k heads 16..31 (k starts at 1152 = 16*72)
    for (int i = lane; i < VHD / 2; i += 32) {
        const float2 c = cs[(size_t)m * (VHD / 2) + i];
        const float lo = p[i], hi = p[i + VHD / 2];
        p[i] = lo * c.x - hi * c.y;
        p[i + VHD / 2] = hi * c.x + lo * c.y;
    }
}

inline int nb(size_t n, int t) { return (int)((n + t - 1) / t); }

// ---------------------------------------------------------------------------------------------- host side
struct Buf {
    void* p = nullptr;
    size_t n = 0;
    template <class T> T* get(size_t count) {
        const size_t bytes = std::max<size_t>(count * sizeof(T), 16);
        if (bytes > n) {
            cudaFree(p);
            CK(cudaMalloc(&p, bytes));
            n = bytes;
        }
        return (T*)p;
    }
};

struct Q4 {  // one nvfp4 linear: packed weight [N, K/2], swizzled block scales, tensor scale
    uint8_t *w, *bs;
    int N, K;
    float ts;
};
struct Layer {
    Q4 q, k, v, o, g, u, d;
    float *pqs_o, *pqs_d, *ln1, *ln2, *qn, *kn;
};
struct VBlock {
    float *n1w, *n1b, *n2w, *n2b, *qkvw, *qkvb, *pw, *pb, *f1w, *f1b, *f2w, *f2b;
};
struct VMerger {
    float *nw, *nb, *f1w, *f1b, *f2w, *f2b;
};

struct Te {
    cudaStream_t st;
    cublasLtHandle_t lh;
    void* ws = nullptr;
    size_t wsz = 64u << 20;
    StFile f;
    std::vector<void*> weights;
    int8_t* emb;
    float* embs;
    Layer l[NL];
    float *pew, *peb;
    bf16* pos;
    VBlock vb[VNB];
    VMerger vm, ds[NDS];
    Buf tmp, qkv, K, V, att, gu, hp, hs, acc, wexp, rows, cs;     // LLM
    Buf vp, vx, vh, vqkv, vatt, vf, vcs, vidx, vw, vout, vds;        // vision
    // prefix cache: the first P rows (reference pictures) of the last keyed prompt — per layer their roped k / v rows
    // [NL][P][2*KVN] and their output rows [P][D]. Rows only see earlier rows (causal), so they don't depend on the rest.
    unsigned long long pkey = 0;
    int pP = 0;
    Buf pkv, pxo;
    float *tap_embed = nullptr, *tap_layers = nullptr, *tap_vblocks = nullptr, *tap_vmerged = nullptr, *tap_vds = nullptr;
};

void* dmalloc(Te* t, size_t n) {
    void* p;
    CK(cudaMalloc(&p, n));
    t->weights.push_back(p);
    return p;
}

const StTensor& tensor(Te* t, const std::string& name, const char* dtype, size_t elems) {
    const StTensor& s = t->f.get(name);
    const size_t es = s.dtype == "I8" || s.dtype == "U8" || s.dtype == "F8_E4M3" ? 1 : s.dtype == "F32" ? 4 : 2;
    if (s.dtype != dtype) DIE("%s: dtype %s, expected %s", name.c_str(), s.dtype.c_str(), dtype);
    if (s.bytes() != elems * es) DIE("%s: %llu bytes, expected %zu elems", name.c_str(), (unsigned long long)s.bytes(), elems);
    return s;
}

void* up(Te* t, const std::string& name, const char* dtype, size_t elems) {
    const StTensor& s = tensor(t, name, dtype, elems);
    void* p = dmalloc(t, s.bytes());
    CK(cudaMemcpy(p, t->f.ptr(s), s.bytes(), cudaMemcpyHostToDevice));
    return p;
}

// bf16 tensor -> fp32 on the device (exact). Copy and convert on our stream: a pageable cudaMemcpy may return
// before its DMA lands, and our stream does not wait for the legacy stream.
float* upf(Te* t, const std::string& name, size_t elems) {
    const StTensor& s = tensor(t, name, "BF16", elems);
    bf16* b = t->tmp.get<bf16>(elems);
    CK(cudaMemcpyAsync(b, t->f.ptr(s), s.bytes(), cudaMemcpyHostToDevice, t->st));
    float* p = (float*)dmalloc(t, elems * 4);
    k_bf16_to_f32<<<nb(elems, 256), 256, 0, t->st>>>(b, p, elems);
    CK(cudaStreamSynchronize(t->st));
    return p;
}

Q4 upq(Te* t, const std::string& name, int N, int K) {
    const StTensor& c = t->f.get(name + ".comfy_quant");
    const std::string js((const char*)t->f.ptr(c), c.bytes());
    if (js.find("\"nvfp4\"") == std::string::npos) DIE("%s: unexpected quant config %s", name.c_str(), js.c_str());
    if (N % 128 || K % 64) DIE("%s: %d x %d not tile aligned", name.c_str(), N, K);
    Q4 q;
    q.N = N;
    q.K = K;
    q.w = (uint8_t*)up(t, name + ".weight", "U8", (size_t)N * K / 2);
    q.bs = (uint8_t*)up(t, name + ".weight_scale", "F8_E4M3", (size_t)N * K / 16);
    const StTensor& s = tensor(t, name + ".weight_scale_2", "F32", 1);
    q.ts = *(const float*)t->f.ptr(s);
    return q;
}

// row-major cuBLASLt GEMM, heuristic's first algo (deterministic): C[M,N] = alpha * A[M,K] . (bT ? B[N,K]^T : B[K,N])
// + beta * C, strided batches. ab: input type (fp32: no TF32, fp16: tensor cores), fp32 accumulate and output.
void mm(Te* t, cudaDataType_t ab, int M, int N, int K, float alpha, const void* A, int lda, long long sa, const void* B,
        int ldb, long long sb, bool bT, float beta, void* C, int ldc, long long sc, int batch) {
    const cudaDataType_t ct = CUDA_R_32F;
    cublasLtMatmulDesc_t op;
    CKL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = bT ? CUBLAS_OP_T : CUBLAS_OP_N, tb = CUBLAS_OP_N;
    CKL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CKL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    cublasLtMatrixLayout_t la, lb, lc;
    CKL(cublasLtMatrixLayoutCreate(&la, ab, bT ? K : N, bT ? N : K, ldb));
    CKL(cublasLtMatrixLayoutCreate(&lb, ab, K, M, lda));
    CKL(cublasLtMatrixLayoutCreate(&lc, ct, N, M, ldc));
    if (batch > 1) {
        const std::pair<cublasLtMatrixLayout_t, long long> ls[3] = {{la, sb}, {lb, sa}, {lc, sc}};
        for (auto& l : ls) {
            CKL(cublasLtMatrixLayoutSetAttribute(l.first, CUBLASLT_MATRIX_LAYOUT_BATCH_COUNT, &batch, sizeof(batch)));
            CKL(cublasLtMatrixLayoutSetAttribute(l.first, CUBLASLT_MATRIX_LAYOUT_STRIDED_BATCH_OFFSET, &l.second, sizeof(l.second)));
        }
    }
    cublasLtMatmulPreference_t pref;
    CKL(cublasLtMatmulPreferenceCreate(&pref));
    CKL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &t->wsz, sizeof(t->wsz)));
    cublasLtMatmulHeuristicResult_t res;
    int n = 0;
    CKL(cublasLtMatmulAlgoGetHeuristic(t->lh, op, la, lb, lc, lc, pref, 1, &res, &n));
    if (n == 0) DIE("no cublasLt algo for type=%d M=%d N=%d K=%d batch=%d", (int)ab, M, N, K, batch);
    CKL(cublasLtMatmul(t->lh, op, &alpha, B, la, A, lb, &beta, C, lc, C, lc, &res.algo, t->ws, t->wsz, t->st));
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
}

void sgemm(Te* t, int M, int N, int K, float alpha, const float* A, int lda, long long sa, const float* B, int ldb,
           long long sb, bool bT, float beta, float* C, int ldc, long long sc, int batch = 1) {
    mm(t, CUDA_R_32F, M, N, K, alpha, A, lda, sa, B, ldb, sb, bT, beta, C, ldc, sc, batch);
}

// y = x . W^T + b (fp32), optional activation, or dst += x . W^T + b
void linear(Te* t, int M, int N, int K, const float* x, const float* W, const float* b, float* y, int act = 0,
            float* dst = nullptr) {
    sgemm(t, M, N, K, 1.f, x, K, 0, W, K, 0, true, 0.f, y, N, 0);
    k_bias<<<nb((size_t)M * N, 256), 256, 0, t->st>>>(y, b, (size_t)M * N, N, act, dst);
    CK(cudaGetLastError());
}

template <int HD>
void flash(Te* t, const float* Q, int ldq, const float* K, int ldk, const float* V, int ldv, float* O, int ldo, int Lq,
           int Lk, int off, int heads, bool causal) {
    static const bool init = [] {
        CK(cudaFuncSetAttribute(k_flash<HD>, cudaFuncAttributeMaxDynamicSharedMemorySize, Fa<HD>::SMEM));
        return true;
    }();
    (void)init;
    k_flash<HD><<<dim3((Lq + FBR - 1) / FBR, heads), 256, Fa<HD>::SMEM, t->st>>>(Q, ldq, K, ldk, V, ldv, O, ldo, Lq, Lk, off,
                                                                                causal, 1.f / std::sqrt((float)HD));
    CK(cudaGetLastError());
}

// fp32 attention (SDPA): heads x hd, rows of Q/K/V/O at the given strides, head h at column h*hd
// Lq query rows against Lk keys; causal: query i is row off + i of the sequence
void attention32(Te* t, const float* Q, int ldq, const float* K, int ldk, const float* V, int ldv, float* O, int ldo,
                 int Lq, int Lk, int off, int heads, int hd, bool causal) {
    if ((ldq | ldk | ldv | ldo) & 3 || ((uintptr_t)Q | (uintptr_t)K | (uintptr_t)V | (uintptr_t)O) & 15)
        DIE("attention32: rows not 16-byte aligned");
    if (hd == VHD) flash<VHD>(t, Q, ldq, K, ldk, V, ldv, O, ldo, Lq, Lk, off, heads, causal);
    else if (hd == HD) flash<HD>(t, Q, ldq, K, ldk, V, ldv, O, ldo, Lq, Lk, off, heads, causal);
    else DIE("attention32: head dim %d", hd);
}

// rows of `in` -> fp16 planes for the next lin4() (M rounded up to 64 rows per plane, so GEMM plans repeat)
void split(Te* t, int mode, const float* in, int ld, int M, int K, const float* w) {
    const int Mp = (M + 63) & ~63;
    const size_t plane = (size_t)Mp * K;
    __half* h = t->hp.get<__half>((size_t)Mp * FF * 2);
    float* sc = t->hs.get<float>(M);
    if (mode == S_RMS) k_split<S_RMS><<<M, 256, 0, t->st>>>(in, ld, K, w, h, plane, sc);
    else if (mode == S_PQS) k_split<S_PQS><<<M, 256, 0, t->st>>>(in, ld, K, w, h, plane, sc);
    else k_split<S_SWIGLU><<<M, 256, 0, t->st>>>(in, ld, K, w, h, plane, sc);
    CK(cudaGetLastError());
}

// out (=, or +=) split rows . [W_0; W_1; ...]^T for up to three nvfp4 linears with the same K (outputs concatenated)
void lin4(Te* t, int M, std::initializer_list<const Q4*> ws, float* out, bool add) {
    const int Mp = (M + 63) & ~63, K = (*ws.begin())->K;
    int N = 0;
    float ts[3] = {0.f, 0.f, 0.f};
    int ends[3] = {0, 0, 0};
    __half* W = t->wexp.get<__half>((size_t)2 * FF * D);
    int j = 0;
    for (const Q4* q : ws) {
        k_expand<<<nb((size_t)q->N * q->K / 16, 256), 256, 0, t->st>>>(q->w, q->bs, q->N, q->K, W + (size_t)N * K);
        N += q->N;
        ts[j] = q->ts;
        ends[j++] = N;
    }
    CK(cudaGetLastError());
    float* acc = t->acc.get<float>((size_t)Mp * 2 * 2 * FF);
    mm(t, CUDA_R_16F, 2 * Mp, N, K, 1.f, t->hp.p, K, 0, W, K, 0, true, 0.f, acc, N, 0, 1);
    const Seg sg{ends[0], ends[1] ? ends[1] : N, ts[0], ts[1], ts[2]};
    k_comb<<<nb((size_t)M * N, 256), 256, 0, t->st>>>(acc, M, Mp, N, (const float*)t->hs.p, sg, out, add ? 1 : 0);
    CK(cudaGetLastError());
}

// torch.linspace(0, n_grid - 1, steps) on CUDA, float
float linspace_at(int steps, int i) {
    if (steps == 1) return 0.f;
    const float end = (float)(VGRID - 1);
    const float step = end / (float)(steps - 1);
    return i < steps / 2 ? step * (float)i : end - step * (float)(steps - i - 1);
}

// Vision tower for one image or frame pair [T, H, W, 3] (device fp32, H and W multiples of 32): merged [Np/4, D] and
// the three DeepStack features (dsout + k*ds_stride). tap_blk: this call's rows in the block tap (nullptr = off).
void vision(Te* t, const float* img, int T, int H, int W, float* merged, float* dsout, size_t ds_stride, float* tap_blk,
            size_t tap_stride) {
    cudaStream_t st = t->st;
    const int gh = H / 16, gw = W / 16, Np = gh * gw, Nm = Np / 4;
    // host tables in merge order: bilinear pos-embed indices / weights and rotary cos/sin
    std::vector<float> hv(gh), wv(gw);
    for (int i = 0; i < gh; i++) hv[i] = linspace_at(gh, i);
    for (int i = 0; i < gw; i++) wv[i] = linspace_at(gw, i);
    std::vector<int> idx(4 * (size_t)Np);
    std::vector<bf16> wt(4 * (size_t)Np);
    std::vector<float2> cs((size_t)Np * (VHD / 2));
    float inv[VHD / 4];
    for (int i = 0; i < VHD / 4; i++) inv[i] = 1.f / (float)std::pow(10000.0, (double)((float)(2 * i) / (float)(VHD / 2)));
    for (int p = 0; p < Np; p++) {
        const int mw = p & 1, mh = (p >> 1) & 1, blk = p >> 2, bw = blk % (gw / 2), bh = blk / (gw / 2);
        const int y = bh * 2 + mh, x = bw * 2 + mw;
        const int fy = (int)hv[y], fx = (int)wv[x];
        const int cy = std::min(fy + 1, VGRID - 1), cx = std::min(fx + 1, VGRID - 1);
        const float dy = hv[y] - (float)fy, dx = wv[x] - (float)fx;
        const int id[4] = {fy * VGRID + fx, fy * VGRID + cx, cy * VGRID + fx, cy * VGRID + cx};
        const float ww[4] = {(1.f - dy) * (1.f - dx), (1.f - dy) * dx, dy * (1.f - dx), dy * dx};
        for (int k = 0; k < 4; k++) {
            idx[(size_t)k * Np + p] = id[k];
            wt[(size_t)k * Np + p] = f2bf(ww[k]);
        }
        for (int d = 0; d < VHD / 2; d++) {
            const float a = d < VHD / 4 ? (float)y * inv[d] : (float)x * inv[d - VHD / 4];
            cs[(size_t)p * (VHD / 2) + d] = make_float2((float)std::cos((double)a), (float)std::sin((double)a));
        }
    }
    int* didx = t->vidx.get<int>(idx.size());
    bf16* dwt = t->vw.get<bf16>(wt.size());
    float2* dcs = t->vcs.get<float2>(cs.size());
    CK(cudaMemcpyAsync(didx, idx.data(), idx.size() * 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(dwt, wt.data(), wt.size() * 2, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(dcs, cs.data(), cs.size() * 8, cudaMemcpyHostToDevice, st));

    float* P = t->vp.get<float>((size_t)Np * VPK);
    float* x = t->vx.get<float>((size_t)Np * VD);
    float* h = t->vh.get<float>((size_t)Np * VD);
    float* qkv = t->vqkv.get<float>((size_t)Np * 3 * VD);
    float* att = t->vatt.get<float>((size_t)Np * VD);
    float* f = t->vf.get<float>((size_t)Np * VFF);
    k_patch<<<nb((size_t)Np * VPK, 256), 256, 0, st>>>(img, T, H, W, P);
    linear(t, Np, VD, VPK, P, t->pew, t->peb, x);
    k_posemb<<<nb((size_t)Np * VD, 256), 256, 0, st>>>(t->pos, didx, dwt, Np, x);
    CK(cudaGetLastError());
    int dsi = 0;
    for (int i = 0; i < VNB; i++) {
        const VBlock& b = t->vb[i];
        k_ln<<<Np, 256, 0, st>>>(x, VD, b.n1w, b.n1b, h);
        linear(t, Np, 3 * VD, VD, h, b.qkvw, b.qkvb, qkv);
        k_vrope<<<nb((size_t)Np * 2 * VNH * 32, 256), 256, 0, st>>>(qkv, Np, dcs);
        attention32(t, qkv, 3 * VD, qkv + VD, 3 * VD, qkv + 2 * VD, 3 * VD, att, VD, Np, Np, 0, VNH, VHD, false);
        linear(t, Np, VD, VD, att, b.pw, b.pb, h, 0, x);
        k_ln<<<Np, 256, 0, st>>>(x, VD, b.n2w, b.n2b, h);
        linear(t, Np, VFF, VD, h, b.f1w, b.f1b, f, 1);
        linear(t, Np, VD, VFF, f, b.f2w, b.f2b, h, 0, x);
        if (tap_blk) CK(cudaMemcpyAsync(tap_blk + (size_t)i * tap_stride, x, (size_t)Np * VD * 4, cudaMemcpyDeviceToDevice, st));
        if (dsi < NDS && i == DS_AT[dsi]) {  // DeepStack merger: LayerNorm after the 2x2 merge
            const VMerger& m = t->ds[dsi];
            k_ln<<<Nm, 256, 0, st>>>(x, VMD, m.nw, m.nb, h);
            linear(t, Nm, VMD, VMD, h, m.f1w, m.f1b, f, 2);
            linear(t, Nm, D, VMD, f, m.f2w, m.f2b, dsout + dsi * ds_stride);
            dsi++;
        }
    }
    k_ln<<<Np, 256, 0, st>>>(x, VD, t->vm.nw, t->vm.nb, h);
    linear(t, Nm, VMD, VMD, h, t->vm.f1w, t->vm.f1b, f, 2);
    linear(t, Nm, D, VMD, f, t->vm.f2w, t->vm.f2b, merged);
}
}  // namespace

extern "C" {

void* h3_te_new(const char* path) {
    Te* t = new Te();
    if (!t->f.open(path)) DIE("cannot open %s", path);
    CK(cudaStreamCreate(&t->st));   // blocking on purpose (common.cuh)
    CKL(cublasLtCreate(&t->lh));
    CK(cudaMalloc(&t->ws, t->wsz));
    const std::string M = "model.";
    {
        const StTensor& c = t->f.get(M + "embed_tokens.comfy_quant");
        const std::string js((const char*)t->f.ptr(c), c.bytes());
        if (js.find("int8_tensorwise") == std::string::npos || js.find("convrot") != std::string::npos)
            DIE("embed_tokens: unexpected quant config %s", js.c_str());
    }
    const size_t V = t->f.get(M + "embed_tokens.weight").shape[0];
    t->emb = (int8_t*)up(t, M + "embed_tokens.weight", "I8", V * D);
    t->embs = (float*)up(t, M + "embed_tokens.weight_scale", "F32", V);
    if (t->f.has(M + "layers." + std::to_string(NL) + ".input_layernorm.weight") || t->f.has(M + "norm.weight"))
        DIE("expected exactly %d layers and no final norm", NL);
    for (int i = 0; i < NL; i++) {
        const std::string p = M + "layers." + std::to_string(i) + ".";
        Layer& l = t->l[i];
        l.q = upq(t, p + "self_attn.q_proj", QN, D);
        l.k = upq(t, p + "self_attn.k_proj", KVN, D);
        l.v = upq(t, p + "self_attn.v_proj", KVN, D);
        l.o = upq(t, p + "self_attn.o_proj", D, QN);
        l.g = upq(t, p + "mlp.gate_proj", FF, D);
        l.u = upq(t, p + "mlp.up_proj", FF, D);
        l.d = upq(t, p + "mlp.down_proj", D, FF);
        for (const char* n : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "mlp.gate_proj", "mlp.up_proj"})
            if (t->f.has(p + n + ".pre_quant_scale")) DIE("%s%s: unexpected pre_quant_scale", p.c_str(), n);
        l.pqs_o = upf(t, p + "self_attn.o_proj.pre_quant_scale", QN);
        l.pqs_d = upf(t, p + "mlp.down_proj.pre_quant_scale", FF);
        l.ln1 = upf(t, p + "input_layernorm.weight", D);
        l.ln2 = upf(t, p + "post_attention_layernorm.weight", D);
        l.qn = upf(t, p + "self_attn.q_norm.weight", HD);
        l.kn = upf(t, p + "self_attn.k_norm.weight", HD);
    }
    const std::string v = "visual.";
    t->pew = upf(t, v + "patch_embed.proj.weight", (size_t)VD * VPK);
    k_tf32<<<nb((size_t)VD * VPK, 256), 256, 0, t->st>>>(t->pew, (size_t)VD * VPK);
    t->peb = upf(t, v + "patch_embed.proj.bias", VD);
    t->pos = (bf16*)up(t, v + "pos_embed.weight", "BF16", (size_t)VGRID * VGRID * VD);
    for (int i = 0; i < VNB; i++) {
        const std::string p = v + "blocks." + std::to_string(i) + ".";
        VBlock& b = t->vb[i];
        b.n1w = upf(t, p + "norm1.weight", VD);
        b.n1b = upf(t, p + "norm1.bias", VD);
        b.n2w = upf(t, p + "norm2.weight", VD);
        b.n2b = upf(t, p + "norm2.bias", VD);
        b.qkvw = upf(t, p + "attn.qkv.weight", (size_t)3 * VD * VD);
        b.qkvb = upf(t, p + "attn.qkv.bias", 3 * VD);
        b.pw = upf(t, p + "attn.proj.weight", (size_t)VD * VD);
        b.pb = upf(t, p + "attn.proj.bias", VD);
        b.f1w = upf(t, p + "mlp.linear_fc1.weight", (size_t)VFF * VD);
        b.f1b = upf(t, p + "mlp.linear_fc1.bias", VFF);
        b.f2w = upf(t, p + "mlp.linear_fc2.weight", (size_t)VD * VFF);
        b.f2b = upf(t, p + "mlp.linear_fc2.bias", VD);
    }
    for (int k = 0; k <= NDS; k++) {
        const std::string p = k < NDS ? v + "deepstack_merger_list." + std::to_string(k) + "." : v + "merger.";
        VMerger& m = k < NDS ? t->ds[k] : t->vm;
        const int nd = k < NDS ? VMD : VD;
        m.nw = upf(t, p + "norm.weight", nd);
        m.nb = upf(t, p + "norm.bias", nd);
        m.f1w = upf(t, p + "linear_fc1.weight", (size_t)VMD * VMD);
        m.f1b = upf(t, p + "linear_fc1.bias", VMD);
        m.f2w = upf(t, p + "linear_fc2.weight", (size_t)D * VMD);
        m.f2b = upf(t, p + "linear_fc2.bias", D);
    }
    CK(cudaDeviceSynchronize());  // all weight copies landed
    t->f.close();
    cudaFree(t->tmp.p);
    t->tmp = Buf();
    return t;
}

// Encode one prompt. ids: host int32 [n] from H3Tokenizer, -1 = one vision block (expanded here to (H/32)*(W/32) rows).
// vis[k]: device fp32 [T, H, W, 3] in [0, 1] (T = 1 image, T = 2 video frame pair; H and W multiples of 32, already
// at their final size), vis_thw = {T0, H0, W0, T1, ...}. Writes out (device fp32 [rows, 5120]): all rows, vision rows
// included. Returns rows, or -1 if rows > out_cap.
// key / P (h3_te_encode_prefix): the first P expanded rows (every vision block among them) are the same as in the last
// call with this key — reuse them and compute only rows P..; key 0 = no cache.
int h3_te_encode_prefix(void* h, const int* ids, int n, int nvis, const void* const* vis, const int* vis_thw, void* out,
                        int out_cap, unsigned long long key, int P) {
    Te* t = (Te*)h;
    cudaStream_t st = t->st;
    std::vector<int> nm(nvis), voff(nvis + 1, 0), poff(nvis + 1, 0);
    for (int k = 0; k < nvis; k++) {
        const int T = vis_thw[3 * k], H = vis_thw[3 * k + 1], W = vis_thw[3 * k + 2];
        if ((T != 1 && T != 2) || H % 32 || W % 32 || H * W < 3136 || (long long)H * W > 12845056)
            DIE("vision input %d: %dx%dx%d not supported", k, T, H, W);
        nm[k] = (H / 32) * (W / 32);
        voff[k + 1] = voff[k] + nm[k];
        poff[k + 1] = poff[k] + 4 * nm[k];
    }
    // expanded sequence: row ids (-1 = vision), block spans, positions (MRoPE as qwen2vl_mrope_position_ids)
    std::vector<int> rid, start(nvis);
    for (int i = 0, k = 0; i < n; i++) {
        if (ids[i] < 0) {
            if (k >= nvis) DIE("more vision slots than vision inputs");
            start[k] = (int)rid.size();
            rid.insert(rid.end(), nm[k], -1);
            k++;
        } else rid.push_back(ids[i]);
    }
    const int L = (int)rid.size();
    if (L > out_cap) return -1;
    if (key && (P <= 0 || P >= L)) key = 0;
    for (int r = P; key && r < L; r++)
        if (rid[r] < 0) DIE("prefix cache: vision row %d after the prefix (P=%d)", r, P);
    const bool hit = key && key == t->pkey && P == t->pP;
    // rows are computed in phases [r0, end). A keyed call that misses computes the prefix 0..P first and then P.. exactly
    // as a hit does, so the output never depends on the cache state (same GEMM shapes either way).
    struct Ph { int r0, end; };
    const std::vector<Ph> ph = !key ? std::vector<Ph>{{0, L}} : hit ? std::vector<Ph>{{P, L}} : std::vector<Ph>{{0, P}, {P, L}};
    int mx = 0;
    for (const Ph& p : ph) mx = std::max(mx, p.end - p.r0);
    std::vector<float> pos(3 * (size_t)L);
    for (int r = 0; r < L; r++) pos[r] = pos[L + r] = pos[2 * L + r] = (float)r;
    float offset = 0.f;
    for (int k = 0; k < nvis; k++) {
        const int s = start[k], e = s + nm[k], gh = vis_thw[3 * k + 1] / 16, gw = vis_thw[3 * k + 2] / 16;
        const int len_max = std::max(1, std::max(gh, gw)) / 2, mw = gw / 2;
        for (int r = e; r < L; r++) pos[r] = pos[L + r] = pos[2 * L + r] = (float)(s + len_max + (r - e)) + offset;
        for (int r = s; r < e; r++) {
            pos[r] = (float)s + offset;
            pos[L + r] = (float)(s + (r - s) / mw) + offset;
            pos[2 * L + r] = (float)(s + (r - s) % mw) + offset;
        }
        offset += (float)(len_max - nm[k]);
    }
    // RoPE table [L][64] (cos, sin): Qwen3-VL interleaved MRoPE, H/W on every 3rd frequency below 60
    std::vector<float2> cs((size_t)L * 64);
    for (int j = 0; j < 64; j++) {
        const float inv = 1.f / (float)std::pow(5000000.0, (double)((float)(2 * j) / (float)HD));
        const int ax = (j % 3 == 1 && j < 60) ? 1 : (j % 3 == 2 && j < 60) ? 2 : 0;
        for (int r = 0; r < L; r++) {
            const float a = inv * pos[(size_t)ax * L + r];
            cs[(size_t)r * 64 + j] = make_float2((float)std::cos((double)a), (float)std::sin((double)a));
        }
    }

    // SHUTTER_PROF=1: vision tower / LLM time on stderr
    static const bool prof = getenv("SHUTTER_PROF") != nullptr;
    cudaEvent_t ev[3];
    if (prof) for (auto& e : ev) { CK(cudaEventCreate(&e)); }
    if (prof) CK(cudaEventRecord(ev[0], st));
    // vision: every block through the tower
    float* vout = t->vout.get<float>((size_t)voff[nvis] * D);
    float* vds = t->vds.get<float>((size_t)NDS * voff[nvis] * D);
    for (int k = 0; k < nvis && !hit; k++)
        vision(t, (const float*)vis[k], vis_thw[3 * k], vis_thw[3 * k + 1], vis_thw[3 * k + 2], vout + (size_t)voff[k] * D,
               vds + (size_t)voff[k] * D, (size_t)voff[nvis] * D,
               t->tap_vblocks ? t->tap_vblocks + (size_t)poff[k] * VD : nullptr, (size_t)poff[nvis] * VD);
    if (t->tap_vmerged && nvis) CK(cudaMemcpyAsync(t->tap_vmerged, vout, (size_t)voff[nvis] * D * 4, cudaMemcpyDeviceToDevice, st));
    if (t->tap_vds && nvis) CK(cudaMemcpyAsync(t->tap_vds, vds, (size_t)NDS * voff[nvis] * D * 4, cudaMemcpyDeviceToDevice, st));

    if (prof) CK(cudaEventRecord(ev[1], st));
    // LLM
    int* drows = t->rows.get<int>(L);
    float2* dcs = t->cs.get<float2>(cs.size());
    float* x = (float*)out;
    float* qkv = t->qkv.get<float>((size_t)mx * QKVN);
    float* K = t->K.get<float>((size_t)L * QN);
    float* V = t->V.get<float>((size_t)L * QN);
    float* att = t->att.get<float>((size_t)mx * QN);
    float* gu = t->gu.get<float>((size_t)mx * 2 * FF);
    float* pkv = key ? t->pkv.get<float>((size_t)NL * P * 2 * KVN) : nullptr;
    float* pxo = key ? t->pxo.get<float>((size_t)P * D) : nullptr;
    const int e0 = ph[0].r0;   // first row computed this call
    CK(cudaMemcpyAsync(drows, rid.data(), (size_t)L * 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(dcs, cs.data(), cs.size() * 8, cudaMemcpyHostToDevice, st));
    k_embed<<<L - e0, 256, 0, st>>>(t->emb, t->embs, drows + e0, x + (size_t)e0 * D);
    CK(cudaGetLastError());
    for (int k = 0; k < nvis && !hit; k++)
        CK(cudaMemcpyAsync(x + (size_t)start[k] * D, vout + (size_t)voff[k] * D, (size_t)nm[k] * D * 4, cudaMemcpyDeviceToDevice, st));
    if (t->tap_embed) CK(cudaMemcpyAsync(t->tap_embed, x, (size_t)L * D * 4, cudaMemcpyDeviceToDevice, st));
    for (const Ph& p : ph) {
        const int r0 = p.r0, Lc = p.end - p.r0;
        const bool reuse = key && r0 == P;   // prefix K / V of each layer from the cache (a hit, or a miss's second phase)
        float* xc = x + (size_t)r0 * D;
        for (int i = 0; i < NL; i++) {
            const Layer& l = t->l[i];
            split(t, S_RMS, xc, D, Lc, D, l.ln1);
            lin4(t, Lc, {&l.q, &l.k, &l.v}, qkv, false);
            k_qkrope<<<nb((size_t)Lc * (NH + NKV) * 32, 256), 256, 0, st>>>(qkv, Lc, dcs + (size_t)r0 * 64, l.qn, l.kn);
            float* lkv = pkv ? pkv + (size_t)i * P * 2 * KVN : nullptr;
            if (reuse) k_expand_kv<<<nb((size_t)P * QN, 256), 256, 0, st>>>(lkv, 2 * KVN, P, K, V);
            else if (lkv)   // the prefix phase (Lc == P): keep its k / v rows
                CK(cudaMemcpy2DAsync(lkv, 2 * KVN * 4, qkv + QN, QKVN * 4, 2 * KVN * 4, P, cudaMemcpyDeviceToDevice, st));
            k_expand_kv<<<nb((size_t)Lc * QN, 256), 256, 0, st>>>(qkv + QN, QKVN, Lc, K + (size_t)r0 * QN, V + (size_t)r0 * QN);
            attention32(t, qkv, QKVN, K, QN, V, QN, att, QN, Lc, p.end, r0, NH, HD, true);
            split(t, S_PQS, att, QN, Lc, QN, l.pqs_o);
            lin4(t, Lc, {&l.o}, xc, true);
            split(t, S_RMS, xc, D, Lc, D, l.ln2);
            lin4(t, Lc, {&l.g, &l.u}, gu, false);
            split(t, S_SWIGLU, gu, 2 * FF, Lc, FF, l.pqs_d);
            lin4(t, Lc, {&l.d}, xc, true);
            if (t->tap_layers && !key) CK(cudaMemcpyAsync(t->tap_layers + (size_t)i * L * D, x, (size_t)L * D * 4, cudaMemcpyDeviceToDevice, st));
            if (i < NDS && !reuse)  // DeepStack: vision features of block DS_AT[i] added at the vision rows after layer i
                for (int k = 0; k < nvis; k++)
                    k_add<<<nb((size_t)nm[k] * D, 256), 256, 0, st>>>(x + (size_t)start[k] * D,
                        vds + (size_t)i * voff[nvis] * D + (size_t)voff[k] * D, (size_t)nm[k] * D);
        }
    }
    if (hit) CK(cudaMemcpyAsync(x, pxo, (size_t)P * D * 4, cudaMemcpyDeviceToDevice, st));
    else if (key) {
        CK(cudaMemcpyAsync(pxo, x, (size_t)P * D * 4, cudaMemcpyDeviceToDevice, st));
        t->pkey = key;
        t->pP = P;
    }
    CK(cudaGetLastError());
    if (prof) CK(cudaEventRecord(ev[2], st));
    CK(cudaStreamSynchronize(st));
    if (prof) {
        float a, b;
        CK(cudaEventElapsedTime(&a, ev[0], ev[1]));
        CK(cudaEventElapsedTime(&b, ev[1], ev[2]));
        fprintf(stderr, "[prof] te rows %d (computed %d): vision %.1f ms, llm %.1f ms\n", L, L - e0, a, b);
        for (auto& e : ev) cudaEventDestroy(e);
    }
    return L;
}

int h3_te_encode(void* h, const int* ids, int n, int nvis, const void* const* vis, const int* vis_thw, void* out,
                 int out_cap) {
    return h3_te_encode_prefix(h, ids, n, nvis, vis, vis_thw, out, out_cap, 0, 0);
}

// debug taps (device fp32, caller-owned, nullptr = off): embed [L, 5120] (vision rows placed), layers [50, L, 5120]
// (each layer's output, before DeepStack), vision blocks [27, sum Np, 1152] (calls side by side in each block),
// merged [sum Np/4, 5120], deepstack [3, sum Np/4, 5120]
void h3_te_taps(void* h, void* embed, void* layers, void* vblocks, void* vmerged, void* vdeep) {
    Te* t = (Te*)h;
    t->tap_embed = (float*)embed;
    t->tap_layers = (float*)layers;
    t->tap_vblocks = (float*)vblocks;
    t->tap_vmerged = (float*)vmerged;
    t->tap_vds = (float*)vdeep;
}

void h3_te_free(void* h) {
    Te* t = (Te*)h;
    cudaStreamSynchronize(t->st);
    for (void* p : t->weights) cudaFree(p);
    for (Buf* b : {&t->tmp, &t->qkv, &t->K, &t->V, &t->att, &t->gu, &t->hp, &t->hs, &t->acc, &t->wexp, &t->rows,
                   &t->cs, &t->vp, &t->vx, &t->vh, &t->vqkv, &t->vatt, &t->vf, &t->vcs, &t->vidx, &t->vw, &t->vout,
                   &t->vds, &t->pkv, &t->pxo})   // pkv / pxo: prefix cache, GBs at ref_size max
        cudaFree(b->p);
    cudaFree(t->ws);
    cublasLtDestroy(t->lh);
    cudaStreamDestroy(t->st);
    delete t;
}
}  // extern "C"
