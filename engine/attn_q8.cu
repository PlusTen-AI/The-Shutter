// shutter engine — quantized flash attention (lossy switch), the SageAttention2 recipe ComfyUI runs on GB10
// (sageattn_qk_int8_pv_fp8_cuda, qk_quant_gran per_warp, smooth_k off):
//   Q int8 per 16 query rows x head (softmax scale * log2 e folded into the scale), K int8 per 64 keys x head,
//   S = int8 mma (s32), softmax in fp32, P = e4m3, V = e4m3 with a per-(head, channel) scale, P.V = e4m3 mma (f32).
// V is stored transposed per head ([h][128][Npad] bytes) with the keys of each 32-key group permuted so that the P
// fragment comes straight out of the score accumulators (see vperm).
#include "common.cuh"

#include <cmath>

namespace {

constexpr int HD = 128, W = 8, BQ = 16 * W, BK = 64;
constexpr float E4M3_MAX = 448.f;
// P is kept at 448 x its value (as SageAttention2) so it fills the e4m3 range; the row sum l carries the same factor.
// Unscaled, the small probabilities of 30k+ keys fall below e4m3's 2^-9 and vanish (1-step DiT error 14.5% vs Sage 6.3%).
constexpr float P_LOG2 = 8.807354922057604f;   // log2(448)

__device__ __forceinline__ void cp16(void* s, const void* g, int bytes) {
    unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(g), "r"(bytes));
}
__device__ __forceinline__ void ldm4(uint32_t* r, const void* s) {
    unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa));
}
__device__ __forceinline__ void mma_s8(int* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.s32.s8.s8.s32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+r"(c[0]), "+r"(c[1]), "+r"(c[2]), "+r"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ void mma_e4m3(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k32.row.col.f32.e4m3.e4m3.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
__device__ __forceinline__ float ex2(float x) {   // MUFU exp2 (as SageAttention); -inf -> 0
    float y;
    asm("ex2.approx.ftz.f32 %0, %1;\n" : "=f"(y) : "f"(x));
    return y;
}
// 4 floats -> 4 e4m3 bytes, x0 in the lowest byte
__device__ __forceinline__ uint32_t e4m3x4(float x0, float x1, float x2, float x3) {
    unsigned short lo, hi;
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(lo) : "f"(x1), "f"(x0));
    asm("cvt.rn.satfinite.e4m3x2.f32 %0, %1, %2;\n" : "=h"(hi) : "f"(x3), "f"(x2));
    return (uint32_t)lo | ((uint32_t)hi << 16);
}
// 128-byte rows (Q, K): 16-byte chunk ^ (row & 7); 64-byte rows (V^T): chunk ^ ((row >> 1) & 3)
__device__ __forceinline__ int sw128(int row, int byte) { return row * 128 + ((((byte >> 4) ^ (row & 7))) << 4) + (byte & 15); }
__device__ __forceinline__ int sw64(int row, int byte) { return row * 64 + ((((byte >> 4) ^ ((row >> 1) & 3))) << 4) + (byte & 15); }
// position p of a 32-key group in V^T holds key vperm(p): the order the P fragment (two n8 score tiles per register) has
__device__ __forceinline__ int vperm(int p) {
    const int half = p >> 4, r = p & 15, t = r >> 2, i = r & 3;
    return half * 16 + (i < 2 ? 2 * t + i : 8 + 2 * t + i - 2);
}

template <int G>
__device__ float block_max(float v, float* red) {
    for (int o = 16; o; o >>= 1) v = fmaxf(v, __shfl_xor_sync(0xffffffffu, v, o));
    if ((threadIdx.x & 31) == 0) red[threadIdx.x >> 5] = v;
    __syncthreads();
    v = 0.f;
    for (int i = 0; i < (int)(blockDim.x >> 5); i++) v = fmaxf(v, red[i]);
    return v;
}

// rows [g*G, g*G + G) of head h -> int8 (row-major like the input, ld bytes), scale = amax / 127 (* mult) per group
template <int G>
__global__ void __launch_bounds__(256) k_quant(const bf16* __restrict__ x, int M, int ld, int8_t* __restrict__ q,
                                               float* __restrict__ s, int ngroups, float mult) {
    __shared__ float red[8];
    const int g = blockIdx.x, h = blockIdx.y;
    float amax = 0.f;
    for (int e = threadIdx.x * 8; e < G * HD; e += 256 * 8) {
        const int r = g * G + e / HD;
        if (r < M) {
            const uint4 u = *(const uint4*)(x + (size_t)r * ld + h * HD + e % HD);
            const bf16* v = (const bf16*)&u;
            for (int j = 0; j < 8; j++) amax = fmaxf(amax, fabsf(__bfloat162float(v[j])));
        }
    }
    amax = block_max<G>(amax, red);
    const float sc = amax > 0.f ? amax / 127.f : 1.f;
    for (int e = threadIdx.x * 8; e < G * HD; e += 256 * 8) {
        const int r = g * G + e / HD;
        if (r < M) {
            const uint4 u = *(const uint4*)(x + (size_t)r * ld + h * HD + e % HD);
            const bf16* v = (const bf16*)&u;
            uint2 o;
            int8_t* ob = (int8_t*)&o;
            for (int j = 0; j < 8; j++) ob[j] = (int8_t)max(-127, min(127, __float2int_rn(__bfloat162float(v[j]) / sc)));
            *(uint2*)(q + (size_t)r * ld + h * HD + e % HD) = o;
        }
    }
    if (threadIdx.x == 0) s[h * ngroups + g] = sc * mult;
}

// per (head, channel) |V| max over all keys (float bits as int: every value >= 0)
__global__ void k_vamax(const bf16* __restrict__ v, int N, int ld, int rows_per, int* __restrict__ amax) {
    const int h = blockIdx.y, c = threadIdx.x;
    float m = 0.f;
    const int r1 = min(N, (blockIdx.x + 1) * rows_per);
    for (int r = blockIdx.x * rows_per; r < r1; r++) m = fmaxf(m, fabsf(__bfloat162float(v[(size_t)r * ld + h * HD + c])));
    atomicMax(amax + h * HD + c, __float_as_int(m));
}

// V [N, ld] bf16 -> V^T e4m3 [h][128][Npad], scaled by 1 / (amax / 448), keys of each 32-group in vperm order, zero pad
__global__ void __launch_bounds__(256) k_vt(const bf16* __restrict__ v, int N, int ld, const int* __restrict__ amax,
                                            uint8_t* __restrict__ vt, int Npad, float* __restrict__ vs) {
    __shared__ float tile[BK][HD + 1];
    const int k0 = blockIdx.x * BK, h = blockIdx.y;
    for (int e = threadIdx.x; e < BK * HD; e += 256) {
        const int r = e / HD, c = e % HD;
        tile[r][c] = k0 + r < N ? __bfloat162float(v[(size_t)(k0 + r) * ld + h * HD + c]) : 0.f;
    }
    __syncthreads();
    for (int e = threadIdx.x; e < HD * BK / 4; e += 256) {
        const int c = e / (BK / 4), p = (e % (BK / 4)) * 4;   // 4 consecutive positions of channel c
        const float a = __int_as_float(amax[h * HD + c]);
        const float inv = a > 0.f ? E4M3_MAX / a : 1.f;
        const int base = p & ~31;
        *(uint32_t*)(vt + ((size_t)h * HD + c) * Npad + k0 + p) =
            e4m3x4(tile[base + vperm((p & 31) + 0)][c] * inv, tile[base + vperm((p & 31) + 1)][c] * inv,
                   tile[base + vperm((p & 31) + 2)][c] * inv, tile[base + vperm((p & 31) + 3)][c] * inv);
    }
    if (blockIdx.x == 0 && threadIdx.x < HD) {
        const float a = __int_as_float(amax[h * HD + threadIdx.x]);
        vs[h * HD + threadIdx.x] = a > 0.f ? a / E4M3_MAX : 1.f;
    }
}

// W warps x 16 query rows share each 64-key tile; K / V^T tiles double-buffered with cp.async.
__global__ void __launch_bounds__(32 * W, 2) k_flash_q8(const int8_t* __restrict__ Q, const float* __restrict__ QS, int nqg,
                                                      const int8_t* __restrict__ K, const float* __restrict__ KS, int nkg,
                                                      const uint8_t* __restrict__ VT, const float* __restrict__ VS, int Npad,
                                                      bf16* __restrict__ O, int Mq, int Nk, int ld, int ldo) {
    constexpr int NT = 32 * W, KT = BK * HD, STAGE = 2 * KT;   // bytes: K tile + V^T tile
    extern __shared__ __align__(128) uint8_t smb[];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int q0 = blockIdx.x * BQ, h = blockIdx.y;
    uint8_t* sQ = smb + 2 * STAGE;   // the Q tile (128 rows x 128 B) stays in shared memory (registers go to 2 CTAs / SM)
    for (int c = tid; c < BQ * 8; c += NT) {
        const int r = c >> 3, ch = c & 7, gr = q0 + r;
        cp16(&sQ[sw128(r, ch * 16)], Q + (size_t)(gr < Mq ? gr : 0) * ld + h * HD + ch * 16, gr < Mq ? 16 : 0);
    }
    auto load = [&](int kb, int stage) {
        uint8_t* sK = smb + stage * STAGE;
        uint8_t* sV = sK + KT;
        for (int c = tid; c < BK * 8; c += NT) {
            const int r = c >> 3, ch = c & 7, gr = kb + r;
            cp16(&sK[sw128(r, ch * 16)], K + (size_t)(gr < Nk ? gr : 0) * ld + h * HD + ch * 16, gr < Nk ? 16 : 0);
        }
        for (int c = tid; c < HD * 4; c += NT) {
            const int r = c >> 2, ch = c & 3;
            cp16(&sV[sw64(r, ch * 16)], VT + ((size_t)h * HD + r) * Npad + kb + ch * 16, 16);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    load(0, 0);
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    const int g = lane >> 2, t = lane & 3;
    const int r0 = q0 + warp * 16 + g, r1 = r0 + 8;
    const int qg = (q0 + warp * 16) / 16;
    const float qs = qg < nqg ? QS[h * nqg + qg] : 0.f;
    float o[16][4];
#pragma unroll
    for (int i = 0; i < 16; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;

    for (int kb = 0, stage = 0; kb < Nk; kb += BK, stage ^= 1) {
        if (kb + BK < Nk) load(kb + BK, stage ^ 1);
        const uint8_t* sK = smb + stage * STAGE;
        const uint8_t* sV = sK + KT;
        int si[8][4];
#pragma unroll
        for (int i = 0; i < 8; i++) si[i][0] = si[i][1] = si[i][2] = si[i][3] = 0;
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
            uint32_t qf[4];
            ldm4(qf, &sQ[sw128(warp * 16 + (lane & 15), kk * 32 + (lane >> 4) * 16)]);
#pragma unroll
            for (int p = 0; p < 4; p++) {
                uint32_t b[4];
                ldm4(b, &sK[sw128(p * 16 + (lane & 7) + (lane >> 4) * 8, kk * 32 + ((lane >> 3) & 1) * 16)]);
                mma_s8(si[2 * p], qf, b[0], b[1]);
                mma_s8(si[2 * p + 1], qf, b[2], b[3]);
            }
        }
        const float f = qs * KS[h * nkg + kb / BK];
        float s[8][4];
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {   // raw scores; f > 0 is applied inside the exponent below
            s[nt][0] = (float)si[nt][0];
            s[nt][1] = (float)si[nt][1];
            s[nt][2] = (float)si[nt][2];
            s[nt][3] = (float)si[nt][3];
            if (kb + BK > Nk) {   // last tile only: keys past Nk
                const int key = kb + nt * 8 + t * 2;
                if (key >= Nk) s[nt][0] = s[nt][2] = -INFINITY;
                if (key + 1 >= Nk) s[nt][1] = s[nt][3] = -INFINITY;
            }
            mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
        }
        mx0 *= f;
        mx1 *= f;
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float b0 = n0 == -INFINITY ? 0.f : n0, b1 = n1 == -INFINITY ? 0.f : n1;
        const float c0 = ex2(m0 - b0), c1 = ex2(m1 - b1);
        float rs0 = 0.f, rs1 = 0.f;
        const float e0 = P_LOG2 - b0, e1 = P_LOG2 - b1;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            s[nt][0] = ex2(fmaf(s[nt][0], f, e0));
            s[nt][1] = ex2(fmaf(s[nt][1], f, e0));
            s[nt][2] = ex2(fmaf(s[nt][2], f, e1));
            s[nt][3] = ex2(fmaf(s[nt][3], f, e1));
            rs0 += s[nt][0] + s[nt][1];
            rs1 += s[nt][2] + s[nt][3];
        }
        l0 = l0 * c0 + rs0;
        l1 = l1 * c1 + rs1;
        m0 = n0;
        m1 = n1;
        if (__any_sync(0xffffffffu, c0 != 1.f || c1 != 1.f)) {   // most tiles leave every row max unchanged
#pragma unroll
            for (int i = 0; i < 16; i++) { o[i][0] *= c0; o[i][1] *= c0; o[i][2] *= c1; o[i][3] *= c1; }
        }
#pragma unroll
        for (int kc = 0; kc < 2; kc++) {   // 32 keys = score tiles 4kc .. 4kc+3
            const int a = 4 * kc;
            const uint32_t pa[4] = {e4m3x4(s[a][0], s[a][1], s[a + 1][0], s[a + 1][1]),
                                    e4m3x4(s[a][2], s[a][3], s[a + 1][2], s[a + 1][3]),
                                    e4m3x4(s[a + 2][0], s[a + 2][1], s[a + 3][0], s[a + 3][1]),
                                    e4m3x4(s[a + 2][2], s[a + 2][3], s[a + 3][2], s[a + 3][3])};
#pragma unroll
            for (int dp = 0; dp < 8; dp++) {
                uint32_t b[4];
                ldm4(b, &sV[sw64(dp * 16 + (lane & 7) + (lane >> 4) * 8, kc * 32 + ((lane >> 3) & 1) * 16)]);
                mma_e4m3(o[2 * dp], pa, b[0], b[1]);
                mma_e4m3(o[2 * dp + 1], pa, b[2], b[3]);
            }
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = l0 > 0.f ? 1.f / l0 : 0.f, i1 = l1 > 0.f ? 1.f / l1 : 0.f;
#pragma unroll
    for (int dt = 0; dt < 16; dt++) {
        const int d = dt * 8 + t * 2;
        const float v0 = VS[h * HD + d], v1 = VS[h * HD + d + 1];
        __nv_bfloat162 a = __floats2bfloat162_rn(o[dt][0] * i0 * v0, o[dt][1] * i0 * v1);
        __nv_bfloat162 b = __floats2bfloat162_rn(o[dt][2] * i1 * v0, o[dt][3] * i1 * v1);
        if (r0 < Mq) *(__nv_bfloat162*)(O + (size_t)r0 * ldo + h * HD + d) = a;
        if (r1 < Mq) *(__nv_bfloat162*)(O + (size_t)r1 * ldo + h * HD + d) = b;
    }
}

struct Ws {
    void* p = nullptr;
    size_t n = 0;
} g_ws;

}  // namespace

// Free the workspace (about 1.2 GB at H3 refine size) when the model that used it is unloaded. Caller has synced.
void attention_q8_release() {
    if (g_ws.p) CK(cudaFree(g_ws.p));
    g_ws = Ws{};
}

// Quantized attention (lossy). q, k, v [rows, ld] bf16 (heads x 128), o [Mq, ldo]. Workspace grows on demand.
void attention_q8(cudaStream_t st, const bf16* q, const bf16* k, const bf16* v, bf16* o, int Mq, int Nk, int heads, int ld,
                  int ldo) {
    const int nqg = (Mq + 15) / 16, nkg = (Nk + BK - 1) / BK, Npad = nkg * BK;
    auto al = [](size_t n) { return (n + 255) & ~size_t(255); };
    const size_t bq = al((size_t)Mq * ld), bk = al((size_t)Nk * ld), bv = al((size_t)heads * HD * Npad),
                 sq = al((size_t)heads * nqg * 4), sk = al((size_t)heads * nkg * 4), sv = al((size_t)heads * HD * 4),
                 am = al((size_t)heads * HD * 4);
    const size_t need = bq + bk + bv + sq + sk + sv + am;
    if (need > g_ws.n) {
        if (g_ws.p) CK(cudaFree(g_ws.p));
        CK(cudaMalloc(&g_ws.p, need));
        g_ws.n = need;
    }
    uint8_t* w = (uint8_t*)g_ws.p;
    int8_t* q8 = (int8_t*)w; w += bq;
    int8_t* k8 = (int8_t*)w; w += bk;
    uint8_t* vt = w; w += bv;
    float* qs = (float*)w; w += sq;
    float* ks = (float*)w; w += sk;
    float* vs = (float*)w; w += sv;
    int* vam = (int*)w;
    k_quant<16><<<dim3(nqg, heads), 256, 0, st>>>(q, Mq, ld, q8, qs, nqg, 1.4426950408889634f / sqrtf((float)HD));
    k_quant<BK><<<dim3(nkg, heads), 256, 0, st>>>(k, Nk, ld, k8, ks, nkg, 1.f);
    CK(cudaMemsetAsync(vam, 0, (size_t)heads * HD * 4, st));
    const int rows_per = 512;
    k_vamax<<<dim3((Nk + rows_per - 1) / rows_per, heads), HD, 0, st>>>(v, Nk, ld, rows_per, vam);
    k_vt<<<dim3(nkg, heads), 256, 0, st>>>(v, Nk, ld, vam, vt, Npad, vs);
    static bool init = false;
    const int smem = 2 * 2 * BK * HD + BQ * HD;   // two K / V^T stages + the Q tile
    if (!init) {
        CK(cudaFuncSetAttribute(k_flash_q8, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        init = true;
    }
    k_flash_q8<<<dim3((Mq + BQ - 1) / BQ, heads), 32 * W, smem, st>>>(q8, qs, nqg, k8, ks, nkg, vt, vs, Npad, o, Mq, Nk, ld,
                                                                      ldo);
    CK(cudaGetLastError());
}

// test entry: one attention on device pointers (default stream), q8 = 0 the bf16 flash kernel, 1 the quantized one
extern "C" void shutter_attention2(const void* q, const void* k, const void* v, void* o, int Mq, int Nk, int heads, int ld,
                                  int q8) {
    if (q8) attention_q8(0, (const bf16*)q, (const bf16*)k, (const bf16*)v, (bf16*)o, Mq, Nk, heads, ld, ld);
    else attention_ex(0, (const bf16*)q, (const bf16*)k, (const bf16*)v, (bf16*)o, Mq, Nk, nullptr, heads, ld, ld);
    CK(cudaStreamSynchronize(0));
}
