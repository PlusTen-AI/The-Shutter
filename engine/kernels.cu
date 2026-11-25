// shutter engine — Qwen-Image 2.1 kernels (GB10 / sm_121)
// Numerics follow ComfyUI + comfy_kitchen CUDA paths: fp32 math, bf16 rounding where torch rounds.
#include "common.cuh"

#include <cmath>

namespace {

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
        v = l < nw ? red[l] : (MAX ? 0.f : 0.f);
        v = MAX ? warp_max(v) : warp_sum(v);
        if (l == 0) red[0] = v;
    }
    __syncthreads();
    v = red[0];
    __syncthreads();
    return v;
}
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16(v)); }

// ---------------------------------------------------------------------------------------------
// pre-op -> ConvRot (regular Hadamard, group 256, normalized) -> row-wise int8
template <int MODE>
__global__ void __launch_bounds__(256) k_rotquant(const bf16* __restrict__ in, int ld, int K, const bf16* __restrict__ ms,
                                                  int8_t* __restrict__ q, float* __restrict__ qs, float eps,
                                                  const int32_t* __restrict__ acc = nullptr, const float* __restrict__ axs = nullptr,
                                                  const float* __restrict__ aws = nullptr, const bf16* __restrict__ nw = nullptr,
                                                  const int* __restrict__ mrow = nullptr, int ldm = 0) {
    extern __shared__ float buf[];
    __shared__ float red[32];
    const int row = blockIdx.x;
    const bf16* x = in + (size_t)row * ld;
    if (MODE == RQ_ADALN) {
        float s = 0.f;
        for (int j = threadIdx.x; j < K; j += blockDim.x) { float v = __bfloat162float(x[j]); buf[j] = v; s += v; }
        const float mean = block_reduce<false>(s, red) / K;
        float s2 = 0.f;
        for (int j = threadIdx.x; j < K; j += blockDim.x) { float d = buf[j] - mean; s2 += d * d; }
        const float rstd = rsqrtf(block_reduce<false>(s2, red) / K + eps);
        for (int j = threadIdx.x; j < K; j += blockDim.x)
            buf[j] = rbf((buf[j] - mean) * rstd * (1.f + __bfloat162float(ms[j])));  // adaln output is bf16
    } else if (MODE == RQ_RMSMOD) {  // torch: rms_norm (fused, one rounding), then mul_(1 + scale), add_(shift) in bf16
        float s = 0.f;
        for (int j = threadIdx.x; j < K; j += blockDim.x) { float v = __bfloat162float(x[j]); buf[j] = v; s += v * v; }
        const float r = rsqrtf(block_reduce<false>(s, red) / K + eps);
        const bf16* mr = ms + (size_t)mrow[row] * ldm;
        for (int j = threadIdx.x; j < K; j += blockDim.x) {
            const float n = rbf(buf[j] * r * __bfloat162float(nw[j]));
            const float opl = rbf(1.f + __bfloat162float(mr[K + j]));
            buf[j] = rbf(rbf(n * opl) + __bfloat162float(mr[j]));
        }
    } else if (MODE == RQ_SWIGLU) {
        for (int j = threadIdx.x; j < K; j += blockDim.x) {
            float g = __bfloat162float(x[j]), u = __bfloat162float(x[K + j]);
            buf[j] = g / (1.f + __expf(-g)) * u;
        }
    } else if (MODE == RQ_SWIGLU_ACC) {  // same values as dequant (bf16 rounding) then swiglu
        const int32_t* a = acc + (size_t)row * 2 * K;
        const float s = axs[row];
        for (int j = threadIdx.x; j < K; j += blockDim.x) {
            const float g = rbf(((float)a[j] * s) * aws[j]), u = rbf(((float)a[K + j] * s) * aws[K + j]);
            buf[j] = g / (1.f + __expf(-g)) * u;
        }
    } else {
        for (int j = threadIdx.x; j < K; j += blockDim.x) buf[j] = __bfloat162float(x[j]);
    }
    __syncthreads();
    // H = h4 (x) h4 (x) h4 (x) h4, h4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]] (symmetric): one radix-4 pass per digit
    const int nq = K >> 2;
    for (int stride = 1; stride < 256; stride <<= 2) {
        for (int qi = threadIdx.x; qi < nq; qi += blockDim.x) {
            const int r = qi & 63;
            const int b = (qi >> 6) * 256 + (r / stride) * (stride * 4) + (r % stride);
            const float a0 = buf[b], a1 = buf[b + stride], a2 = buf[b + 2 * stride], a3 = buf[b + 3 * stride];
            const float s = a0 + a1 + a2 + a3;
            buf[b] = s - 2.f * a3;
            buf[b + stride] = s - 2.f * a2;
            buf[b + 2 * stride] = s - 2.f * a1;
            buf[b + 3 * stride] = s - 2.f * a0;
        }
        __syncthreads();
    }
    float mx = 0.f;
    for (int j = threadIdx.x; j < K; j += blockDim.x) { float v = buf[j] * 0.0625f; buf[j] = v; mx = fmaxf(mx, fabsf(v)); }
    mx = block_reduce<true>(mx, red);
    // comfy_kitchen CUDA quantizer, matched bit for bit: s = amax/127 (fp32); q = rne(bf16(bf16(x) / bf16(s)))
    const float sc = fmaxf(mx / 127.f, 1e-30f);
    const float scb = rbf(sc);
    int8_t* qo = q + (size_t)row * K;
    for (int j = threadIdx.x; j < K; j += blockDim.x) {
        float v = rintf(rbf(rbf(buf[j]) / scb));
        qo[j] = (int8_t)fminf(fmaxf(v, -128.f), 127.f);
    }
    if (threadIdx.x == 0) qs[row] = sc;
}

// ---------------------------------------------------------------------------------------------
__global__ void k_dequant(const int32_t* __restrict__ acc, int M, int N, const float* __restrict__ xs,
                          const float* __restrict__ ws, bf16* __restrict__ out) {
    const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= (size_t)M * N) return;
    const int m = (int)(i / N), n = (int)(i % N);
    const int4 a = *(const int4*)(acc + i);
    const float s = xs[m];
    const float4 w = *(const float4*)(ws + n);
    __nv_bfloat162 o0 = __floats2bfloat162_rn(((float)a.x * s) * w.x, ((float)a.y * s) * w.y);
    __nv_bfloat162 o1 = __floats2bfloat162_rn(((float)a.z * s) * w.z, ((float)a.w * s) * w.w);
    *(__nv_bfloat162*)(out + i) = o0;
    *(__nv_bfloat162*)(out + i + 2) = o1;
}

__global__ void k_dequant_resid(const int32_t* __restrict__ acc, int M, int N, const float* __restrict__ xs,
                                const float* __restrict__ ws, const bf16* __restrict__ gate, bf16* __restrict__ x,
                                int ldg = 0, const int* __restrict__ mrow = nullptr) {
    const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= (size_t)M * N) return;
    const int m = (int)(i / N), n = (int)(i % N);
    if (mrow) gate += (size_t)mrow[m] * ldg;
    const int4 a = *(const int4*)(acc + i);
    const float s = xs[m];
    const int av[4] = {a.x, a.y, a.z, a.w};
    for (int j = 0; j < 4; j++) {
        const float y = rbf(((float)av[j] * s) * ws[n + j]);
        x[i + j] = __float2bfloat16(__bfloat162float(x[i + j]) + y * __bfloat162float(gate[n + j]));
    }
}

// ---------------------------------------------------------------------------------------------
// fused qkv dequant + RMSNorm + RoPE: reads the int32 accumulators of the qkv GEMM (same bf16 rounding as dequant)
__global__ void k_rms_rope_acc(const int32_t* __restrict__ acc, const float* __restrict__ xs, const float* __restrict__ ws,
                               int M, const float* __restrict__ pe, const bf16* __restrict__ wq, const bf16* __restrict__ wk,
                               float eps, bf16* __restrict__ q, bf16* __restrict__ k, bf16* __restrict__ v) {
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int which = gw % 3, hh = (gw / 3) & 31, m = gw / 96;
    if (m >= M) return;
    const int c0 = which * 4096 + hh * 128;
    const int32_t* src = acc + (size_t)m * 12288 + c0;
    const float s = xs[m];
    if (which == 2) {
        const int4 a = *(const int4*)(src + lane * 4);
        const float4 w = *(const float4*)(ws + c0 + lane * 4);
        bf16* dst = v + (size_t)m * 4096 + hh * 128 + lane * 4;
        *(__nv_bfloat162*)dst = __floats2bfloat162_rn(((float)a.x * s) * w.x, ((float)a.y * s) * w.y);
        *(__nv_bfloat162*)(dst + 2) = __floats2bfloat162_rn(((float)a.z * s) * w.z, ((float)a.w * s) * w.w);
        return;
    }
    const int2 ia = *(const int2*)(src + 2 * lane), ib = *(const int2*)(src + 64 + 2 * lane);
    float2 a = make_float2(rbf(((float)ia.x * s) * ws[c0 + 2 * lane]), rbf(((float)ia.y * s) * ws[c0 + 2 * lane + 1]));
    float2 b = make_float2(rbf(((float)ib.x * s) * ws[c0 + 64 + 2 * lane]), rbf(((float)ib.y * s) * ws[c0 + 65 + 2 * lane]));
    const float ss = warp_sum(a.x * a.x + a.y * a.y + b.x * b.x + b.y * b.y);
    const float r = rsqrtf(ss / 128.f + eps);
    const bf16* w = which ? wk : wq;
    a.x = rbf(a.x * r * __bfloat162float(w[2 * lane]));        // ck rms_rope: normalized value rounded to bf16, then rope
    a.y = rbf(a.y * r * __bfloat162float(w[2 * lane + 1]));
    b.x = rbf(b.x * r * __bfloat162float(w[64 + 2 * lane]));
    b.y = rbf(b.y * r * __bfloat162float(w[64 + 2 * lane + 1]));
    const float4 pa = *(const float4*)(pe + ((size_t)m * 64 + lane) * 4);
    const float4 pb = *(const float4*)(pe + ((size_t)m * 64 + 32 + lane) * 4);
    bf16* dst = (which ? k : q) + (size_t)m * 4096 + hh * 128;
    *(__nv_bfloat162*)(dst + 2 * lane) = __floats2bfloat162_rn(pa.x * a.x + pa.y * a.y, pa.z * a.x + pa.w * a.y);
    *(__nv_bfloat162*)(dst + 64 + 2 * lane) = __floats2bfloat162_rn(pb.x * b.x + pb.y * b.y, pb.z * b.x + pb.w * b.y);
}

// one warp per (row, head, which): which 0 = q, 1 = k (RMSNorm + RoPE), 2 = v (copy)
__global__ void k_rms_rope(const bf16* __restrict__ qkv, int M, const float* __restrict__ pe, const bf16* __restrict__ wq,
                           const bf16* __restrict__ wk, float eps, bf16* __restrict__ q, bf16* __restrict__ k,
                           bf16* __restrict__ v) {
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int which = gw % 3, hh = (gw / 3) & 31, m = gw / 96;
    if (m >= M) return;
    const bf16* src = qkv + (size_t)m * 12288 + which * 4096 + hh * 128;
    if (which == 2) {
        *(uint2*)(v + (size_t)m * 4096 + hh * 128 + lane * 4) = *(const uint2*)(src + lane * 4);
        return;
    }
    float2 a = __bfloat1622float2(*(const __nv_bfloat162*)(src + 2 * lane));
    float2 b = __bfloat1622float2(*(const __nv_bfloat162*)(src + 64 + 2 * lane));
    const float ss = warp_sum(a.x * a.x + a.y * a.y + b.x * b.x + b.y * b.y);
    const float r = rsqrtf(ss / 128.f + eps);
    const bf16* w = which ? wk : wq;
    a.x = rbf(a.x * r * __bfloat162float(w[2 * lane]));        // ck rms_rope: normalized value rounded to bf16, then rope
    a.y = rbf(a.y * r * __bfloat162float(w[2 * lane + 1]));
    b.x = rbf(b.x * r * __bfloat162float(w[64 + 2 * lane]));
    b.y = rbf(b.y * r * __bfloat162float(w[64 + 2 * lane + 1]));
    const float4 pa = *(const float4*)(pe + ((size_t)m * 64 + lane) * 4);
    const float4 pb = *(const float4*)(pe + ((size_t)m * 64 + 32 + lane) * 4);
    bf16* dst = (which ? k : q) + (size_t)m * 4096 + hh * 128;
    *(__nv_bfloat162*)(dst + 2 * lane) = __floats2bfloat162_rn(pa.x * a.x + pa.y * a.y, pa.z * a.x + pa.w * a.y);
    *(__nv_bfloat162*)(dst + 64 + 2 * lane) = __floats2bfloat162_rn(pb.x * b.x + pb.y * b.y, pb.z * b.x + pb.w * b.y);
}

// ---------------------------------------------------------------------------------------------
// flash attention (FA2 style), head dim 128, 128 queries x 64 keys per tile, 8 warps, mma.sync bf16
constexpr int HD = 128;
__device__ __forceinline__ int swz(int row, int col) { return row * HD + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7); }
__device__ __forceinline__ void cp16(void* s, const void* g, int bytes) {
    unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(g), "r"(bytes));
}
__device__ __forceinline__ void cp_commit_wait() {
    asm volatile("cp.async.commit_group;\n" ::);
    asm volatile("cp.async.wait_group 0;\n" ::);
}
__device__ __forceinline__ void ldm4(uint32_t* r, const void* s) {
    unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa));
}
__device__ __forceinline__ void ldm4t(uint32_t* r, const void* s) {
    unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa));
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ uint32_t pack2(float lo, float hi) {
    __nv_bfloat162 v = __floats2bfloat162_rn(lo, hi);
    return *(uint32_t*)&v;
}

// W warps (16 query rows each) share each K/V tile; the next K/V tile is loaded (cp.async) while this one computes.
// smem: two K/V stages (64 KB); Q is staged in stage 1 and moved to registers before the loop.
// (Replaced a 4-warp single-stage kernel with the same per-row math: bit-identical output, H3 S=85k 62.8 -> 75.2 TFLOPS.)
constexpr int W = 8;
__global__ void __launch_bounds__(32 * W) k_flash(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                                                   const bf16* __restrict__ V, bf16* __restrict__ O, int Mq, int Nk,
                                                   const int* __restrict__ qlim, float sl2, int ld_in, int ldo) {
    constexpr int NT = 32 * W, BQ = 16 * W, TILE = 64 * HD;
    extern __shared__ __align__(128) bf16 sm[];
    __shared__ int s_kmax;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int q0 = blockIdx.x * BQ, h = blockIdx.y;
    const size_t ld = ld_in;
    bf16* sQ = sm + 2 * TILE;   // stage 1 (K and V halves: room for 128 query rows)
    for (int c = tid; c < BQ * 16; c += NT) {
        const int r = c >> 4, ch = c & 15, gr = q0 + r;
        cp16(&sQ[swz(r, ch * 8)], Q + (size_t)(gr < Mq ? gr : 0) * ld + h * HD + ch * 8, gr < Mq ? 16 : 0);
    }
    if (tid == 0) s_kmax = 0;
    __syncthreads();
    if (qlim && tid < BQ && q0 + tid < Mq) atomicMax(&s_kmax, qlim[q0 + tid]);
    auto load_kv = [&](int kb, int stage) {
        bf16* sK = sm + stage * 2 * TILE;
        bf16* sV = sK + TILE;
        for (int c = tid; c < 64 * 16; c += NT) {
            const int r = c >> 4, ch = c & 15, gr = kb + r;
            const bool ok = gr < Nk;
            const size_t off = (size_t)(ok ? gr : 0) * ld + h * HD + ch * 8;
            cp16(&sK[swz(r, ch * 8)], K + off, ok ? 16 : 0);
            cp16(&sV[swz(r, ch * 8)], V + off, ok ? 16 : 0);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    load_kv(0, 0);   // same commit group as Q
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    const int kmax = qlim ? min(s_kmax, Nk) : Nk;
    const int g = lane >> 2, t = lane & 3;
    const int r0 = q0 + warp * 16 + g, r1 = r0 + 8;
    const int lim0 = qlim ? (r0 < Mq ? qlim[r0] : 0) : Nk;
    const int lim1 = qlim ? (r1 < Mq ? qlim[r1] : 0) : Nk;

    uint32_t qf[8][4];
#pragma unroll
    for (int kk = 0; kk < 8; kk++) ldm4(qf[kk], &sQ[swz(warp * 16 + (lane & 15), kk * 16 + (lane >> 4) * 8)]);
    __syncthreads();   // stage 1 is free for K/V
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    for (int kb = 0, stage = 0; kb < kmax; kb += 64, stage ^= 1) {
        if (kb + 64 < kmax) load_kv(kb + 64, stage ^ 1);
        const bf16* sK = sm + stage * 2 * TILE;
        const bf16* sV = sK + TILE;

        float s[8][4];
#pragma unroll
        for (int i = 0; i < 8; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 8; kk++) {
#pragma unroll
            for (int p = 0; p < 4; p++) {
                uint32_t b[4];
                ldm4(b, &sK[swz(p * 16 + (lane & 7) + (lane >> 4) * 8, kk * 16 + ((lane >> 3) & 1) * 8)]);
                mma(s[2 * p], qf[kk], b[0], b[1]);
                mma(s[2 * p + 1], qf[kk], b[2], b[3]);
            }
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            const int key = kb + nt * 8 + t * 2;
            s[nt][0] = key < lim0 ? s[nt][0] * sl2 : -INFINITY;
            s[nt][1] = key + 1 < lim0 ? s[nt][1] * sl2 : -INFINITY;
            s[nt][2] = key < lim1 ? s[nt][2] * sl2 : -INFINITY;
            s[nt][3] = key + 1 < lim1 ? s[nt][3] * sl2 : -INFINITY;
            mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float b0 = n0 == -INFINITY ? 0.f : n0, b1 = n1 == -INFINITY ? 0.f : n1;
        const float c0 = exp2f(m0 - b0), c1 = exp2f(m1 - b1);
        float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            s[nt][0] = exp2f(s[nt][0] - b0);
            s[nt][1] = exp2f(s[nt][1] - b0);
            s[nt][2] = exp2f(s[nt][2] - b1);
            s[nt][3] = exp2f(s[nt][3] - b1);
            rs0 += s[nt][0] + s[nt][1];
            rs1 += s[nt][2] + s[nt][3];
        }
        m0 = n0;
        m1 = n1;
        l0 = l0 * c0 + rs0;
        l1 = l1 * c1 + rs1;
#pragma unroll
        for (int i = 0; i < 16; i++) { o[i][0] *= c0; o[i][1] *= c0; o[i][2] *= c1; o[i][3] *= c1; }
#pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            const uint32_t a[4] = {pack2(s[2 * kc][0], s[2 * kc][1]), pack2(s[2 * kc][2], s[2 * kc][3]),
                                   pack2(s[2 * kc + 1][0], s[2 * kc + 1][1]), pack2(s[2 * kc + 1][2], s[2 * kc + 1][3])};
#pragma unroll
            for (int dp = 0; dp < 8; dp++) {
                uint32_t b[4];
                ldm4t(b, &sV[swz(kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, dp * 16 + (lane >> 4) * 8)]);
                mma(o[2 * dp], a, b[0], b[1]);
                mma(o[2 * dp + 1], a, b[2], b[3]);
            }
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();   // next stage landed; everyone is done with this one before it is refilled
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
    for (int dt = 0; dt < 16; dt++) {
        const int col = h * HD + dt * 8 + t * 2;
        if (r0 < Mq) *(uint32_t*)(O + (size_t)r0 * ldo + col) = pack2(o[dt][0] * i0, o[dt][1] * i0);
        if (r1 < Mq) *(uint32_t*)(O + (size_t)r1 * ldo + col) = pack2(o[dt][2] * i1, o[dt][3] * i1);
    }
}

// ---------------------------------------------------------------------------------------------
__global__ void k_elementwise(int op, const bf16* __restrict__ in, bf16* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = __bfloat162float(in[i]);
    float y;
    if (op == EW_SILU) y = x / (1.f + expf(-x));
    else if (op == EW_TANH) y = tanhf(x);
    else y = 0.5f * x * (1.f + tanhf(0.7978845608028654f * (x + 0.044715f * x * x * x)));
    out[i] = __float2bfloat16(y);
}

__global__ void __launch_bounds__(256) k_rmsnorm_zc(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                    bf16* __restrict__ out, int K, float eps) {
    __shared__ float red[32];
    const bf16* xr = x + (size_t)blockIdx.x * K;
    float s = 0.f;
    for (int j = threadIdx.x; j < K; j += blockDim.x) { float v = __bfloat162float(xr[j]); s += v * v; }
    const float r = rsqrtf(block_reduce<false>(s, red) / K + eps);
    for (int j = threadIdx.x; j < K; j += blockDim.x)
        out[(size_t)blockIdx.x * K + j] = __float2bfloat16(__bfloat162float(xr[j]) * r * (__bfloat162float(w[j]) + 1.f));
}

__global__ void __launch_bounds__(256) k_adaln(const bf16* __restrict__ x, const bf16* __restrict__ sc,
                                               bf16* __restrict__ out, int K, float eps) {
    __shared__ float red[32];
    const bf16* xr = x + (size_t)blockIdx.x * K;
    float s = 0.f;
    for (int j = threadIdx.x; j < K; j += blockDim.x) s += __bfloat162float(xr[j]);
    const float mean = block_reduce<false>(s, red) / K;
    float s2 = 0.f;
    for (int j = threadIdx.x; j < K; j += blockDim.x) { float d = __bfloat162float(xr[j]) - mean; s2 += d * d; }
    const float rstd = rsqrtf(block_reduce<false>(s2, red) / K + eps);
    for (int j = threadIdx.x; j < K; j += blockDim.x)
        out[(size_t)blockIdx.x * K + j] =
            __float2bfloat16((__bfloat162float(xr[j]) - mean) * rstd * (1.f + __bfloat162float(sc[j])));
}

__global__ void k_chw_to_tokens(const bf16* __restrict__ x, bf16* __restrict__ t, int C, int HW) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C * HW) return;
    const int c = i % C, p = i / C;
    t[i] = x[(size_t)c * HW + p];
}
__global__ void k_tokens_to_chw(const bf16* __restrict__ t, bf16* __restrict__ x, int C, int HW) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= C * HW) return;
    const int p = i % HW, c = i / HW;
    x[i] = t[(size_t)p * C + c];
}
__global__ void k_f32_to_bf16(const float* __restrict__ in, bf16* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(in[i]);
}
__global__ void k_euler(float* __restrict__ x, const bf16* __restrict__ out, int n, float s, float sn) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xv = x[i];
    const float den = xv - __bfloat162float(out[i]) * s;
    x[i] = xv + (xv - den) / s * (sn - s);
}

__global__ void k_euler_cfg(float* __restrict__ x, const bf16* __restrict__ oc, const bf16* __restrict__ ou, int n, float s,
                            float sn, float cfg) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float xv = x[i];
    const float dc = xv - __bfloat162float(oc[i]) * s;
    const float du = xv - __bfloat162float(ou[i]) * s;
    const float den = du + (dc - du) * cfg;
    x[i] = xv + (xv - den) / s * (sn - s);
}

inline int nb(size_t n, int t) { return (int)((n + t - 1) / t); }
}  // namespace

// ------------------------------------------------------------------------------------------------- launchers
constexpr int RQ_KMAX = 16384;
static void rq_init() {
    static bool init = false;
    if (init) return;
    CK(cudaFuncSetAttribute(k_rotquant<RQ_PLAIN>, cudaFuncAttributeMaxDynamicSharedMemorySize, RQ_KMAX * 4));
    CK(cudaFuncSetAttribute(k_rotquant<RQ_ADALN>, cudaFuncAttributeMaxDynamicSharedMemorySize, RQ_KMAX * 4));
    CK(cudaFuncSetAttribute(k_rotquant<RQ_SWIGLU>, cudaFuncAttributeMaxDynamicSharedMemorySize, RQ_KMAX * 4));
    CK(cudaFuncSetAttribute(k_rotquant<RQ_SWIGLU_ACC>, cudaFuncAttributeMaxDynamicSharedMemorySize, RQ_KMAX * 4));
    CK(cudaFuncSetAttribute(k_rotquant<RQ_RMSMOD>, cudaFuncAttributeMaxDynamicSharedMemorySize, RQ_KMAX * 4));
    init = true;
}

void rotquant(cudaStream_t st, int mode, const bf16* in, int ld_in, int M, int K, const bf16* ms, int8_t* q, float* qs) {
    if (K % 256 || K > RQ_KMAX) DIE("rotquant: K=%d", K);
    const size_t smem = (size_t)K * sizeof(float);
    rq_init();
    const float eps = 1e-6f;
    if (mode == RQ_ADALN) k_rotquant<RQ_ADALN><<<M, 256, smem, st>>>(in, ld_in, K, ms, q, qs, eps);
    else if (mode == RQ_SWIGLU) k_rotquant<RQ_SWIGLU><<<M, 256, smem, st>>>(in, ld_in, K, ms, q, qs, eps);
    else k_rotquant<RQ_PLAIN><<<M, 256, smem, st>>>(in, ld_in, K, ms, q, qs, eps);
    CK(cudaGetLastError());
}

void swiglu_rotquant_acc(cudaStream_t st, const int32_t* acc, const float* xs, const float* ws, int M, int K, int8_t* q,
                         float* qs) {
    if (K % 256 || K > RQ_KMAX) DIE("swiglu_rotquant_acc: K=%d", K);
    rq_init();
    k_rotquant<RQ_SWIGLU_ACC><<<M, 256, (size_t)K * 4, st>>>(nullptr, 0, K, nullptr, q, qs, 1e-6f, acc, xs, ws);
    CK(cudaGetLastError());
}

void rotquant_rmsmod(cudaStream_t st, const bf16* x, int M, int K, const bf16* nw, float eps, const bf16* mod, int ldm,
                     const int* mrow, int8_t* q, float* qs) {
    if (K % 256 || K > RQ_KMAX) DIE("rotquant_rmsmod: K=%d", K);
    rq_init();
    k_rotquant<RQ_RMSMOD><<<M, 256, (size_t)K * 4, st>>>(x, K, K, mod, q, qs, eps, nullptr, nullptr, nullptr, nw, mrow, ldm);
    CK(cudaGetLastError());
}

void rms_rope_acc(cudaStream_t st, const int32_t* acc, const float* xs, const float* ws, int M, const float* pe,
                  const bf16* wq, const bf16* wk, float eps, bf16* q, bf16* k, bf16* v) {
    k_rms_rope_acc<<<nb((size_t)M * 96 * 32, 256), 256, 0, st>>>(acc, xs, ws, M, pe, wq, wk, eps, q, k, v);
    CK(cudaGetLastError());
}

void dequant(cudaStream_t st, const int32_t* acc, int M, int N, const float* xs, const float* ws, bf16* out) {
    k_dequant<<<nb((size_t)M * N / 4, 256), 256, 0, st>>>(acc, M, N, xs, ws, out);
    CK(cudaGetLastError());
}
void dequant_resid(cudaStream_t st, const int32_t* acc, int M, int N, const float* xs, const float* ws, const bf16* gate,
                   bf16* x) {
    k_dequant_resid<<<nb((size_t)M * N / 4, 256), 256, 0, st>>>(acc, M, N, xs, ws, gate, x);
    CK(cudaGetLastError());
}
void rms_rope(cudaStream_t st, const bf16* qkv, int M, const float* pe, const bf16* wq, const bf16* wk, float eps,
              bf16* q, bf16* k, bf16* v) {
    k_rms_rope<<<nb((size_t)M * 96 * 32, 256), 256, 0, st>>>(qkv, M, pe, wq, wk, eps, q, k, v);
    CK(cudaGetLastError());
}
void attention_ex(cudaStream_t st, const bf16* q, const bf16* k, const bf16* v, bf16* o, int Mq, int Nk, const int* qlim,
                  int heads, int ld, int ldo, float scale) {
    static bool init = false;
    const int smem = 4 * 64 * HD * sizeof(bf16);
    if (!init) {
        CK(cudaFuncSetAttribute(k_flash, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        init = true;
    }
    k_flash<<<dim3((Mq + 16 * W - 1) / (16 * W), heads), 32 * W, smem, st>>>(q, k, v, o, Mq, Nk, qlim,
                                                                           1.4426950408889634f * (scale > 0.f ? scale : 1.f / sqrtf(128.f)), ld, ldo);
    CK(cudaGetLastError());
}
void attention(cudaStream_t st, const bf16* q, const bf16* k, const bf16* v, bf16* o, int Mq, int Nk, const int* qlim) {
    attention_ex(st, q, k, v, o, Mq, Nk, qlim, 32, 4096, 4096);
}
void elementwise(cudaStream_t st, int op, const bf16* in, bf16* out, int n) {
    k_elementwise<<<nb(n, 256), 256, 0, st>>>(op, in, out, n);
    CK(cudaGetLastError());
}
void rmsnorm_zc(cudaStream_t st, const bf16* x, const bf16* w, bf16* out, int M, int K, float eps) {
    k_rmsnorm_zc<<<M, 256, 0, st>>>(x, w, out, K, eps);
    CK(cudaGetLastError());
}
void adaln(cudaStream_t st, const bf16* x, const bf16* scale, bf16* out, int M, int K, float eps) {
    k_adaln<<<M, 256, 0, st>>>(x, scale, out, K, eps);
    CK(cudaGetLastError());
}
void chw_to_tokens(cudaStream_t st, const bf16* x, bf16* t, int C, int HW) {
    k_chw_to_tokens<<<nb((size_t)C * HW, 256), 256, 0, st>>>(x, t, C, HW);
    CK(cudaGetLastError());
}
void tokens_to_chw(cudaStream_t st, const bf16* t, bf16* x, int C, int HW) {
    k_tokens_to_chw<<<nb((size_t)C * HW, 256), 256, 0, st>>>(t, x, C, HW);
    CK(cudaGetLastError());
}
void f32_to_bf16(cudaStream_t st, const float* in, bf16* out, int n) {
    k_f32_to_bf16<<<nb(n, 256), 256, 0, st>>>(in, out, n);
    CK(cudaGetLastError());
}
void euler(cudaStream_t st, float* x, const bf16* out, int n, float s, float sn) {
    k_euler<<<nb(n, 256), 256, 0, st>>>(x, out, n, s, sn);
    CK(cudaGetLastError());
}
void euler_cfg(cudaStream_t st, float* x, const bf16* oc, const bf16* ou, int n, float s, float sn, float cfg) {
    k_euler_cfg<<<nb(n, 256), 256, 0, st>>>(x, oc, ou, n, s, sn, cfg);
    CK(cudaGetLastError());
}
