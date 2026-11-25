// shutter engine — MiniMax H3 video VAE (ComfyUI comfy/ldm/minimax/vae.py MiniMaxH3VideoVAE, stock ViT3D decoder), C API.
// Same tiling as ComfyUI: 256 px spatial tiles (64 px min overlap, bf16 linear blends), 17-frame encode clips,
// 7-latent decode chunks (5 + 2 overlap, 3 pre-padding frames dropped, 5-frame temporal blend).
// Numerics follow what ComfyUI ran with --bf16-vae (probed against comfy_kitchen 0.2.41 CUDA ops on GB10):
//  - encoder: bf16 activations NDHWC, per-frame GroupNorm and SiLU each rounded, reflect spatial / zero causal
//    padding, conv = fp32-accumulated GEMM rounded to bf16, then + bias and + residual each rounded.
//  - decoder int8 ConvRot linears: input act (rms_norm / swiglu) in fp32, Hadamard-256 rotation in fp32,
//    q = rne(bf16(bf16(v) / bf16(s))) with s = amax(v)/127, out = bf16((acc*xs)*ws + b),
//    residual = bf16(x + scale * out). q/k: per-head RMS rounded, split-half rope (48 of 64) rounded.
//  - attention: exact softmax (ComfyUI used comfy_kitchen int8 attention here; ~1.4e-2 rel error, not reproduced).
// Self-contained: own cublasLt wrapper and kernels (only common.cuh macros and st.h are shared).
#include "common.cuh"
#include "st.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <array>
#include <cmath>
#include <map>
#include <string>
#include <vector>

namespace {
inline size_t nblk(size_t n, int t) { return (n + t - 1) / t; }
inline int rup(int a, int b) { return (a + b - 1) / b * b; }
__device__ __forceinline__ float rd(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
inline float hround(float v) { return __bfloat162float(__float2bfloat16(v)); }  // host bf16 rounding

void* dmalloc(size_t n) {
    void* p;
    CK(cudaMalloc(&p, std::max<size_t>(n, 16)));
    return p;
}

// H3VAE_PROF=1: per-section GPU time (events between marks), printed to stderr after each call
struct Prof {
    bool on = getenv("H3VAE_PROF") != nullptr;
    std::vector<std::pair<const char*, cudaEvent_t>> ev;
    void mark(cudaStream_t st, const char* name) {
        if (!on) return;
        cudaEvent_t e;
        CK(cudaEventCreate(&e));
        CK(cudaEventRecord(e, st));
        ev.push_back({name, e});
    }
    void report(const char* what) {
        if (!on || ev.size() < 2) return;
        CK(cudaEventSynchronize(ev.back().second));
        std::map<std::string, float> s;
        float tot = 0.f;
        for (size_t i = 1; i < ev.size(); i++) {
            float ms;
            CK(cudaEventElapsedTime(&ms, ev[i - 1].second, ev[i].second));
            s[ev[i].first] += ms, tot += ms;
        }
        fprintf(stderr, "[h3_vae prof %s] total %.0f ms:", what, tot);
        for (auto& kv : s) fprintf(stderr, " %s %.0f", kv.first.c_str(), kv.second);
        fprintf(stderr, "\n");
        for (auto& x : ev) cudaEventDestroy(x.second);
        ev.clear();
    }
};
Prof prof;

// ---------------- cuBLASLt (row-major C[M,N] = A[M,K] . W[N,K]^T, leading dims in elements) ----------------
enum { G_BF16 = 0, G_F32 = 1, G_I8 = 2 };
struct Gemm {
    cublasLtHandle_t h;
    cudaStream_t st;
    void* ws;
    size_t wsz = 64u << 20;
    struct Plan {
        cublasLtMatmulDesc_t op;
        cublasLtMatrixLayout_t lw, la, lc;
        cublasLtMatmulAlgo_t algo;
    };
    std::map<std::array<int, 7>, Plan> plans;
};

void gemm_run(Gemm* g, Gemm::Plan& p, int type, const void* A, const void* W, void* C, const cublasLtMatmulAlgo_t* algo) {
    int32_t ai = 1, bi = 0;
    float af = 1.f, bf = 0.f;
    const void* al = type == G_I8 ? (const void*)&ai : (const void*)&af;
    const void* be = type == G_I8 ? (const void*)&bi : (const void*)&bf;
    CKL(cublasLtMatmul(g->h, p.op, al, W, p.lw, A, p.la, be, C, p.lc, C, p.lc, algo, g->ws, g->wsz, g->st));
}

void gemm(Gemm* g, int type, int M, int N, int K, const void* A, int lda, const void* W, int ldw, void* C) {
    const std::array<int, 7> key{type, M, N, K, lda, ldw, 0};
    auto it = g->plans.find(key);
    if (it == g->plans.end()) {
        Gemm::Plan p;
        const bool i8 = type == G_I8;
        CKL(cublasLtMatmulDescCreate(&p.op, i8 ? CUBLAS_COMPUTE_32I : CUBLAS_COMPUTE_32F, i8 ? CUDA_R_32I : CUDA_R_32F));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        CKL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        CKL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        const cudaDataType_t ab = i8 ? CUDA_R_8I : CUDA_R_16BF;
        const cudaDataType_t ct = i8 ? CUDA_R_32I : type == G_F32 ? CUDA_R_32F : CUDA_R_16BF;
        CKL(cublasLtMatrixLayoutCreate(&p.lw, ab, K, N, ldw));
        CKL(cublasLtMatrixLayoutCreate(&p.la, ab, K, M, lda));
        CKL(cublasLtMatrixLayoutCreate(&p.lc, ct, N, M, N));
        cublasLtMatmulPreference_t pref;
        CKL(cublasLtMatmulPreferenceCreate(&pref));
        CKL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g->wsz, sizeof(g->wsz)));
        cublasLtMatmulHeuristicResult_t res[8];
        int n = 0;
        CKL(cublasLtMatmulAlgoGetHeuristic(g->h, p.op, p.lw, p.la, p.lc, p.lc, pref, 8, res, &n));
        cublasLtMatmulPreferenceDestroy(pref);
        if (n == 0) DIE("h3_vae: no cublasLt algo type %d M=%d N=%d K=%d", type, M, N, K);
        int best = 0;
        if (i8 && n > 1 && (double)M * N * K > 1e9) {  // int8 is exact: time the candidates once per shape
            cudaEvent_t e0, e1;
            CK(cudaEventCreate(&e0));
            CK(cudaEventCreate(&e1));
            float bt = 1e30f;
            for (int i = 0; i < n; i++) {
                gemm_run(g, p, type, A, W, C, &res[i].algo);
                CK(cudaEventRecord(e0, g->st));
                for (int r = 0; r < 3; r++) gemm_run(g, p, type, A, W, C, &res[i].algo);
                CK(cudaEventRecord(e1, g->st));
                CK(cudaEventSynchronize(e1));
                float ms;
                CK(cudaEventElapsedTime(&ms, e0, e1));
                if (ms < bt) bt = ms, best = i;
            }
            cudaEventDestroy(e0);
            cudaEventDestroy(e1);
        }
        p.algo = res[best].algo;  // bf16: heuristic's first choice (deterministic)
        it = g->plans.emplace(key, p).first;
    }
    gemm_run(g, it->second, type, A, W, C, &it->second.algo);
}

// ---------------- small kernels ----------------
__device__ float block_sum(float v, float* sh) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = l < nw ? sh[l] : 0.f;
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}

// The pre-op values are rounded to fp32 one by one (__fmul_rn) before the rotation, like the torch ops: left to itself
// nvcc contracts those products into the rotation's sums (fma), and where it does so changes with unrelated edits.
// int8 ConvRot quantizer, rows of K (2048 or 8192), 128 threads: K/2048 warps per row (4 rows per block for 2048).
// Lane l of a row-warp owns 64 contiguous elements of Hadamard group l/4 (digit d3 = l%4); MODE 0: plain bf16 row;
// 1: rms_norm(x) * w (fp32); 2: swiglu of the previous GEMM's [gate | up], already dequantized + bias in bf16 (gemm_i8_epi DQ_F32B)
__device__ __forceinline__ void h4(float& a0, float& a1, float& a2, float& a3) {
    const float s = a0 + a1 + a2 + a3, b0 = a0, b1 = a1;
    a0 = s - 2.f * a3, a1 = s - 2.f * a2, a2 = s - 2.f * b1, a3 = s - 2.f * b0;
}
template <int MODE>
__global__ void __launch_bounds__(128) k_rq(const void* __restrict__ in, int M, int K, const float* __restrict__ p0,
                                           int8_t* __restrict__ q, float* __restrict__ qs) {
    __shared__ float red[4];
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31, wpr = K / 2048, rpb = 4 / wpr;
    const size_t row = (size_t)blockIdx.x * rpb + warp / wpr;
    const int slot = (warp / wpr) * wpr, off = (warp % wpr) * 2048 + (lane >> 2) * 256 + (lane & 3) * 64;
    const bool ok = row < (size_t)M;
    float v[64];
    if (MODE == 2) {
        const bf16* a = (const bf16*)in + (ok ? row : 0) * 2 * K;
#pragma unroll
        for (int j = 0; j < 64; j += 8) {
            const uint4 g8 = *(const uint4*)(a + off + j), u8 = *(const uint4*)(a + K + off + j);
            const __nv_bfloat162 *g2 = (const __nv_bfloat162*)&g8, *u2 = (const __nv_bfloat162*)&u8;
#pragma unroll
            for (int e = 0; e < 8; e++) {
                const float g = e & 1 ? __high2float(g2[e / 2]) : __low2float(g2[e / 2]);
                const float u = e & 1 ? __high2float(u2[e / 2]) : __low2float(u2[e / 2]);
                v[j + e] = __fmul_rn(g / (1.f + expf(-g)), u);   // rounded: else nvcc fuses it into the rotation sums
            }
        }
    } else {
        const bf16* x = (const bf16*)in + (ok ? row : 0) * K + off;
        float ss = 0.f;
#pragma unroll
        for (int j = 0; j < 64; j += 8) {
            const uint4 r = *(const uint4*)(x + j);
            const __nv_bfloat162* h = (const __nv_bfloat162*)&r;
#pragma unroll
            for (int e = 0; e < 4; e++) {
                v[j + 2 * e] = __low2float(h[e]), v[j + 2 * e + 1] = __high2float(h[e]);
                ss += v[j + 2 * e] * v[j + 2 * e] + v[j + 2 * e + 1] * v[j + 2 * e + 1];
            }
        }
        if (MODE == 1) {
            for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
            if (wpr > 1) {
                if (lane == 0) red[warp] = ss;
                __syncthreads();
                ss = red[slot] + red[slot + 1] + red[slot + 2] + red[slot + 3];
            }
            const float r = rsqrtf(ss / K + 1e-5f);
#pragma unroll
            for (int j = 0; j < 64; j++) v[j] = __fmul_rn(v[j] * r, p0[off + j]);   // rounded, as above
        }
    }
#pragma unroll
    for (int st = 1; st < 64; st *= 4)
#pragma unroll
        for (int j = 0; j < 64; j++)
            if ((j / st) % 4 == 0) h4(v[j], v[j + st], v[j + 2 * st], v[j + 3 * st]);
    float mx = 0.f;
#pragma unroll
    for (int j = 0; j < 64; j++) {  // digit d3 across the 4 lanes of the group: lane k gets s - 2 a_(3-k)
        float s = v[j] + __shfl_xor_sync(0xffffffffu, v[j], 1);
        s += __shfl_xor_sync(0xffffffffu, s, 2);
        v[j] = (s - 2.f * __shfl_xor_sync(0xffffffffu, v[j], 3)) * 0.0625f;
        mx = fmaxf(mx, fabsf(v[j]));
    }
    for (int o = 16; o; o >>= 1) mx = fmaxf(mx, __shfl_xor_sync(0xffffffffu, mx, o));
    if (wpr > 1) {
        __syncthreads();
        if (lane == 0) red[warp] = mx;
        __syncthreads();
        mx = fmaxf(fmaxf(red[slot], red[slot + 1]), fmaxf(red[slot + 2], red[slot + 3]));
    }
    if (!ok) return;
    const float sc = fmaxf(mx / 127.f, 1e-30f), sb = rd(sc);
    int8_t* qo = q + row * K + off;
#pragma unroll
    for (int j = 0; j < 64; j += 16) {
        uint32_t w[4];
#pragma unroll
        for (int e = 0; e < 4; e++) {
            uint32_t b = 0;
#pragma unroll
            for (int k = 0; k < 4; k++) {
                const float t = fminf(fmaxf(rintf(rd(rd(v[j + 4 * e + k]) / sb)), -128.f), 127.f);
                b |= ((uint32_t)(int)t & 0xffu) << (8 * k);
            }
            w[e] = b;
        }
        *(uint4*)(qo + j) = make_uint4(w[0], w[1], w[2], w[3]);
    }
    if (lane == 0 && warp % wpr == 0) qs[row] = sc;
}

// qkv accumulators [M, 6144] (per head: q 64 | k 64 | v 64) -> q, k (RMS rounded, split-half rope on 48 dims), v
// y [M, 6144] = bf16((acc * xs) * ws + b) from gemm_i8_epi DQ_F32B
__global__ void k_qkv(const bf16* __restrict__ y, const float2* __restrict__ rope, int L, int M, bf16* __restrict__ Q,
                      bf16* __restrict__ Kk, bf16* __restrict__ V) {
    const size_t gw = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const int which = (int)(gw % 3), h = (int)((gw / 3) & 31);
    const size_t m = gw / 96;
    if (m >= (size_t)M) return;
    const int c0 = h * 192 + which * 64;
    const bf16* a = y + m * 6144 + c0;
    float y0 = ld(a + lane), y1 = ld(a + lane + 32);
    bf16* dst = (which == 0 ? Q : which == 1 ? Kk : V) + m * 2048 + h * 64;
    if (which < 2) {
        float ss = y0 * y0 + y1 * y1;
        for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
        const float r = rsqrtf(ss / 64.f + 1e-5f);
        y0 = rd(y0 * r), y1 = rd(y1 * r);
        const int p0 = lane < 24 ? lane + 24 : lane - 24;  // partner of element lane (only used if lane < 48)
        const int p1 = lane + 32 - 24;                       // partner of element lane+32 (lane+32 < 48)
        const float a0 = __shfl_sync(0xffffffffu, y0, p0 & 31), b0 = __shfl_sync(0xffffffffu, y1, p0 & 31);
        const float a1 = __shfl_sync(0xffffffffu, y0, p1 & 31), b1 = __shfl_sync(0xffffffffu, y1, p1 & 31);
        const float n0 = p0 < 32 ? a0 : b0, n1 = p1 < 32 ? a1 : b1;
        const float2* rp = rope + (size_t)(m % L) * 24;
        {
            const float2 cs = rp[lane % 24];  // element lane < 32 < 48 always rotates
            y0 = lane < 24 ? rd(y0 * cs.x - n0 * cs.y) : rd(y0 * cs.x + n0 * cs.y);
        }
        if (lane < 16) {  // element lane + 32 in [32, 48): second half of pair lane + 8
            const float2 cs = rp[lane + 8];
            y1 = rd(y1 * cs.x + n1 * cs.y);
        }
    }
    dst[lane] = __float2bfloat16(y0);
    dst[lane + 32] = __float2bfloat16(y1);
}


// ---------------- flash attention, head dim 64, 32 heads, rows of 2048, batch over blockIdx.z ----------------
__device__ __forceinline__ int swz(int row, int col) { return row * 64 + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7); }
__device__ __forceinline__ void cp16(void* s, const void* g, int bytes) {
    unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(g), "r"(bytes));
}
__device__ __forceinline__ void cp_wait() {
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

__global__ void __launch_bounds__(128) k_flash64(const bf16* __restrict__ Q, const bf16* __restrict__ K,
                                                 const bf16* __restrict__ V, bf16* __restrict__ O, int L, float sl2) {
    __shared__ __align__(128) bf16 sQ[64 * 64], sK[64 * 64], sV[64 * 64];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int q0 = blockIdx.x * 64, hc = blockIdx.y * 64;
    const size_t base = (size_t)blockIdx.z * L * 2048;
    Q += base, K += base, V += base, O += base;
    for (int c = tid; c < 64 * 8; c += 128) {
        const int r = c >> 3, ch = c & 7, gr = q0 + r;
        cp16(&sQ[swz(r, ch * 8)], Q + (size_t)(gr < L ? gr : 0) * 2048 + hc + ch * 8, gr < L ? 16 : 0);
    }
    cp_wait();
    __syncthreads();
    const int g = lane >> 2, t = lane & 3;
    const int r0 = q0 + warp * 16 + g, r1 = r0 + 8;
    uint32_t qf[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; kk++) ldm4(qf[kk], &sQ[swz(warp * 16 + (lane & 15), kk * 16 + (lane >> 4) * 8)]);
    float o[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    for (int kb = 0; kb < L; kb += 64) {
        __syncthreads();
        for (int c = tid; c < 64 * 8; c += 128) {
            const int r = c >> 3, ch = c & 7, gr = kb + r;
            const bool ok = gr < L;
            const size_t off = (size_t)(ok ? gr : 0) * 2048 + hc + ch * 8;
            cp16(&sK[swz(r, ch * 8)], K + off, ok ? 16 : 0);
            cp16(&sV[swz(r, ch * 8)], V + off, ok ? 16 : 0);
        }
        cp_wait();
        __syncthreads();
        float s[8][4];
#pragma unroll
        for (int i = 0; i < 8; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
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
            s[nt][0] = key < L ? s[nt][0] * sl2 : -INFINITY;
            s[nt][1] = key + 1 < L ? s[nt][1] * sl2 : -INFINITY;
            s[nt][2] = key < L ? s[nt][2] * sl2 : -INFINITY;
            s[nt][3] = key + 1 < L ? s[nt][3] * sl2 : -INFINITY;
            mx0 = fmaxf(mx0, fmaxf(s[nt][0], s[nt][1]));
            mx1 = fmaxf(mx1, fmaxf(s[nt][2], s[nt][3]));
        }
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 1));
        mx0 = fmaxf(mx0, __shfl_xor_sync(0xffffffffu, mx0, 2));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 1));
        mx1 = fmaxf(mx1, __shfl_xor_sync(0xffffffffu, mx1, 2));
        const float n0 = fmaxf(m0, mx0), n1 = fmaxf(m1, mx1);
        const float c0 = exp2f(m0 - n0), c1 = exp2f(m1 - n1);
        float rs0 = 0.f, rs1 = 0.f;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
            s[nt][0] = exp2f(s[nt][0] - n0);
            s[nt][1] = exp2f(s[nt][1] - n0);
            s[nt][2] = exp2f(s[nt][2] - n1);
            s[nt][3] = exp2f(s[nt][3] - n1);
            rs0 += s[nt][0] + s[nt][1];
            rs1 += s[nt][2] + s[nt][3];
        }
        m0 = n0, m1 = n1;
        l0 = l0 * c0 + rs0;
        l1 = l1 * c1 + rs1;
#pragma unroll
        for (int i = 0; i < 8; i++) o[i][0] *= c0, o[i][1] *= c0, o[i][2] *= c1, o[i][3] *= c1;
#pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            const uint32_t a[4] = {pack2(s[2 * kc][0], s[2 * kc][1]), pack2(s[2 * kc][2], s[2 * kc][3]),
                                   pack2(s[2 * kc + 1][0], s[2 * kc + 1][1]), pack2(s[2 * kc + 1][2], s[2 * kc + 1][3])};
#pragma unroll
            for (int dp = 0; dp < 4; dp++) {
                uint32_t b[4];
                ldm4t(b, &sV[swz(kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, dp * 16 + (lane >> 4) * 8)]);
                mma(o[2 * dp], a, b[0], b[1]);
                mma(o[2 * dp + 1], a, b[2], b[3]);
            }
        }
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float i0 = 1.f / l0, i1 = 1.f / l1;
#pragma unroll
    for (int dt = 0; dt < 8; dt++) {
        const int col = hc + dt * 8 + t * 2;
        if (r0 < L) *(uint32_t*)(O + (size_t)r0 * 2048 + col) = pack2(o[dt][0] * i0, o[dt][1] * i0);
        if (r1 < L) *(uint32_t*)(O + (size_t)r1 * 2048 + col) = pack2(o[dt][2] * i1, o[dt][3] * i1);
    }
}

// ---------------- decoder in/out ----------------
// latent chunk [24][T][lh][lw] f32 (normalized), times frames t0..t0+n (clamped to Tz-1 = repeat of the last latent)
// -> zq [n*lh*lw][24] bf16 = post_quant_conv(bf16(bf16(z) * std + mean))
__global__ void k_dec_prep(const float* __restrict__ lat, int Tz, int t0, int n, int HW, const float* __restrict__ sd,
                           const float* __restrict__ mn, const float* __restrict__ w, const float* __restrict__ b,
                           bf16* __restrict__ zq) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * HW * 24) return;
    const int o = (int)(i % 24);
    const size_t p = i / 24;
    const int t = min(t0 + (int)(p / HW), Tz - 1);
    const size_t pos = (size_t)t * HW + p % HW;
    float acc = 0.f;
    for (int c = 0; c < 24; c++) {
        const float z = rd(rd(rd(lat[(size_t)c * Tz * HW + pos]) * sd[c]) + mn[c]);
        acc += w[o * 24 + c] * z;
    }
    zq[i] = __float2bfloat16(rd(acc) + b[o]);
}

// tokens of G tiles: x [G][L][2048]: rows < L0 = x_embedder(zq token), then 4 register tokens and a zero row
__global__ void k_embed(const bf16* __restrict__ zq, int lh, int lw, int T, int th, int tw, const int2* __restrict__ org,
                        const float* __restrict__ w, const float* __restrict__ b, const float* __restrict__ reg, int L,
                        int G, bf16* __restrict__ x) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)G * L * 2048) return;
    const int o = (int)(i % 2048);
    const size_t rr = i / 2048;
    const int r = (int)(rr % L), gi = (int)(rr / L), L0 = T * th * tw;
    float v = 0.f;
    if (r < L0) {
        const int xx = r % tw, yy = (r / tw) % th, t = r / (tw * th);
        const bf16* z = zq + (((size_t)t * lh + org[gi].x + yy) * lw + org[gi].y + xx) * 24;
        float acc = 0.f;
        for (int c = 0; c < 24; c++) acc += w[o * 24 + c] * ld(z + c);
        v = acc + b[o];
    } else if (r < L0 + 4) {
        v = reg[(r - L0) * 2048 + o];
    }
    x[i] = __float2bfloat16(v);
}

// LayerNorm (affine, eps 1e-5) of the first L0 rows of each tile -> out [G*L0][2048]
__global__ void __launch_bounds__(256) k_ln(const bf16* __restrict__ x, int L, int L0, const float* __restrict__ w,
                                            const float* __restrict__ b, bf16* __restrict__ out) {
    __shared__ float red[32];
    const size_t ro = blockIdx.x;
    const bf16* xr = x + ((ro / L0) * L + ro % L0) * 2048;
    float s = 0.f;
    for (int j = threadIdx.x; j < 2048; j += blockDim.x) s += ld(xr + j);
    const float mean = block_sum(s, red) / 2048.f;
    float s2 = 0.f;
    for (int j = threadIdx.x; j < 2048; j += blockDim.x) {
        const float d = ld(xr + j) - mean;
        s2 += d * d;
    }
    const float rstd = rsqrtf(block_sum(s2, red) / 2048.f + 1e-5f);
    for (int j = threadIdx.x; j < 2048; j += blockDim.x)
        out[ro * 2048 + j] = __float2bfloat16((ld(xr + j) - mean) * rstd * w[j] + b[j]);
}

// proj_out accumulators [G*L0][3072] (+ bias) -> tile pixels bf16 [3][4T][16 th][16 tw], tile g at tiles + g * tsz
__global__ void k_unpatch(const float* __restrict__ acc, const float* __restrict__ b, int T, int th, int tw, int G,
                          bf16* __restrict__ tiles, size_t tsz) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int L0 = T * th * tw;
    if (i >= (size_t)G * L0 * 3072) return;
    const int n = (int)(i % 3072);
    const size_t row = i / 3072;
    const int r = (int)(row % L0), gi = (int)(row / L0);
    const int xx = r % tw, yy = (r / tw) % th, t = r / (tw * th);
    const int c = n >> 10, pt = (n >> 8) & 3, py = (n >> 4) & 15, px = n & 15;
    const size_t W = (size_t)tw * 16, H = (size_t)th * 16;
    tiles[gi * tsz + (((size_t)c * 4 * T + t * 4 + pt) * H + yy * 16 + py) * W + xx * 16 + px] =
        __float2bfloat16(acc[i] + b[n]);
}

// ---------------- tiling helpers (planes of [rows][cols] bf16) ----------------
// B[p][r][c] = bf16(bf16(A[p][r][c] * wa) + bf16(B[p][r][c] * wb)), wb = bf16(k / e), wa = bf16(1 - wb),
// k = r (vertical) or c; ComfyUI's blend() in bf16
__global__ void k_blend2d(int P, int R, int C, const bf16* __restrict__ A, size_t aps, int ald, bf16* __restrict__ B,
                          size_t bps, int bld, int vertical, int e) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)P * R * C) return;
    const int c = (int)(i % C), r = (int)((i / C) % R);
    const size_t p = i / ((size_t)R * C);
    const float wb = rd((float)(vertical ? r : c) / (float)e), wa = rd(1.f - wb);
    bf16* bp = B + p * bps + (size_t)r * bld + c;
    *bp = __float2bfloat16(rd(ld(A + p * aps + (size_t)r * ald + c) * wa) + rd(ld(bp) * wb));
}
__global__ void k_copy2d(int P, int R, int C, const bf16* __restrict__ S, size_t sps, int sld, bf16* __restrict__ D,
                         size_t dps, int dld) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)P * R * C) return;
    const int c = (int)(i % C), r = (int)((i / C) % R);
    const size_t p = i / ((size_t)R * C);
    D[p * dps + (size_t)r * dld + c] = S[p * sps + (size_t)r * sld + c];
}

// frames f0..f0+nf of canvas [3][Fc][H][W] (first e frames blended with ov [3][e][H][W]) -> out [nf][H][W][3] f32
__global__ void k_final(const bf16* __restrict__ cv, int Fc, int f0, int nf, const bf16* __restrict__ ov, int e, int HW,
                        const float* __restrict__ sd, const float* __restrict__ mn, float* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)nf * HW * 3) return;
    const int c = (int)(i % 3);
    const size_t p = (i / 3) % HW;
    const int f = (int)(i / 3 / HW);
    float v = ld(cv + ((size_t)c * Fc + f0 + f) * HW + p);
    if (ov && f < e) {
        const float wb = rd((float)f / (float)e), wa = rd(1.f - wb);
        v = rd(rd(ld(ov + ((size_t)c * e + f) * HW + p) * wa) + rd(v * wb));
    }
    out[i] = fminf(fmaxf(v * sd[c] + mn[c], 0.f), 1.f);
}

// ---------------- encoder kernels (NDHWC bf16) ----------------
// pixels [F][H][W][3] f32 0..1, frames fr0.. (clamped to F-1), tile at (y0, x0) th x tw -> bf16 normalized [T][th][tw][3]
__global__ void k_enc_in(const float* __restrict__ px, int F, int H, int W, int fr0, int T, int y0, int x0, int th,
                         int tw, const float* __restrict__ mn, const float* __restrict__ sd, bf16* __restrict__ x) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * th * tw * 3) return;
    const int c = (int)(i % 3);
    const size_t p = i / 3;
    const int xx = (int)(p % tw), yy = (int)((p / tw) % th), t = (int)(p / ((size_t)tw * th));
    const int f = min(fr0 + t, F - 1);
    float v = rd(px[(((size_t)f * H + y0 + yy) * W + x0 + xx) * 3 + c] * 2.f - 1.f);
    v = rd(rd(rd(v + 1.f) * 0.5f) - mn[c]);
    x[i] = __float2bfloat16(v / sd[c]);
}

// per-frame GroupNorm(32) statistics, deterministic and coalesced: per-(frame, pixel chunk, channel) partial sums
// (of x, then of (x - mean)^2), reduced per (frame, group). Thread c of a block sums channel c over 256 pixels.
__global__ void k_gn_part(const bf16* __restrict__ x, int HW, int C, int nch, const float2* __restrict__ st,
                          float* __restrict__ part) {
    const int t = blockIdx.x / nch, ch = blockIdx.x % nch, c = threadIdx.x;
    const int p0 = ch * 256, p1 = min(p0 + 256, HW);
    const bf16* xs = x + (size_t)t * HW * C + c;
    const float mean = st ? st[t * 32 + c / (C / 32)].x : 0.f;
    float s = 0.f;
    for (int p = p0; p < p1; p++) {
        const float d = ld(xs + (size_t)p * C) - mean;
        s += st ? d * d : d;
    }
    part[(size_t)blockIdx.x * C + c] = s;
}
// one warp per (frame, group): pass 0 writes (mean, 0), pass 1 writes (mean, rstd)
__global__ void k_gn_fin(const float* __restrict__ part, int HW, int C, int nch, float eps, int pass, float2* __restrict__ st) {
    const int t = blockIdx.x / 32, g = blockIdx.x % 32, cg = C / 32, lane = threadIdx.x;
    float s = 0.f;
    for (int i = lane; i < nch * cg; i += 32) s += part[((size_t)t * nch + i / cg) * C + g * cg + i % cg];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane) return;
    const float n = (float)HW * cg;
    st[blockIdx.x] = pass == 0 ? make_float2(s / n, 0.f) : make_float2(st[blockIdx.x].x, rsqrtf(s / n + eps));
}

// padded [T+front][H+pt+pb][W+pl+pr][C] = reflect-pad(optional silu(groupnorm(x))), zero front frames;
// E channels per thread (8 when C % 8 == 0)
template <int E>
__global__ void k_pad(const bf16* __restrict__ x, int T, int H, int W, int C, const float2* __restrict__ st,
                      const float* __restrict__ gw, const float* __restrict__ gb, int pl, int pr, int pt, int pb,
                      int front, bf16* __restrict__ P) {
    const int Hp = H + pt + pb, Wp = W + pl + pr, CE = C / E;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)(T + front) * Hp * Wp * CE) return;
    const int c0 = (int)(i % CE) * E;
    const size_t p = i / CE;
    bf16* dst = P + p * C + c0;
    const int xp = (int)(p % Wp), yp = (int)((p / Wp) % Hp), tp = (int)(p / ((size_t)Wp * Hp));
    if (tp < front) {
        for (int e = 0; e < E; e++) dst[e] = __float2bfloat16(0.f);
        return;
    }
    const int t = tp - front;
    int y = yp - pt, xx = xp - pl;
    y = y < 0 ? -y : y >= H ? 2 * (H - 1) - y : y;
    xx = xx < 0 ? -xx : xx >= W ? 2 * (W - 1) - xx : xx;
    const bf16* src = x + (((size_t)t * H + y) * W + xx) * C + c0;
    bf16 o[E];
    for (int e = 0; e < E; e++) {
        float v = ld(src + e);
        if (st) {
            const int c = c0 + e;
            const float2 s = st[t * 32 + c / (C / 32)];
            v = rd((v - s.x) * s.y * gw[c] + gb[c]);
            v = rd(v / (1.f + expf(-v)));
        }
        o[e] = __float2bfloat16(v);
    }
    for (int e = 0; e < E; e++) dst[e] = o[e];
}

// implicit-GEMM 3x3x3 (or 1x3x3 for single frames) conv over padded P [Tp][Hp][Wp][C] (C % 32 == 0):
// out[m][n] = bf16(bf16(sum_k A[m][k] W[n][k]) + b[n]) (+ resid, rounded), K order (kt, ky, kx, c), fp32 accumulate.
// 128 x 128 x 32 tiles, 8 warps (2 x 4, warp tile 64 x 32), cp.async double buffer, ldmatrix + mma.sync bf16.
__device__ __forceinline__ int sw32(int row, int ch) { return row * 32 + ((ch ^ ((row >> 1) & 3)) << 3); }
__global__ void __launch_bounds__(256) k_iconv(const bf16* __restrict__ P, int Hp, int Wp, int C, int nt, int st,
                                               int ss, int Ho, int Wo, int M, const bf16* __restrict__ Wt, int kp,
                                               const float* __restrict__ bias, const bf16* __restrict__ resid, int N,
                                               bf16* __restrict__ out) {
    __shared__ __align__(128) bf16 sA[2][128 * 32], sB[2][128 * 32];
    __shared__ long long base[128];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp >> 2, wn = warp & 3;
    const int m0 = blockIdx.x * 128, n0 = blockIdx.y * 128;
    if (tid < 128) {
        const int m = min(m0 + tid, M - 1);
        const int xo = m % Wo, yo = (m / Wo) % Ho, to = m / (Wo * Ho);
        base[tid] = (((long long)to * st * Hp + (long long)yo * ss) * Wp + (long long)xo * ss) * C;
    }
    __syncthreads();
    const int cpt = C / 32, nk = nt * 9 * cpt;
    auto load = [&](int kc, int buf) {
        const int tap = kc / cpt, c0 = (kc % cpt) * 32;
        const int kt = tap / 9, ky = (tap / 3) % 3, kx = tap % 3;
        const long long toff = (((long long)kt * Hp + ky) * Wp + kx) * C + c0;
        for (int i = tid; i < 512; i += 256) {
            const int r = i >> 2, ch = i & 3;
            const bool ok = m0 + r < M;
            cp16(&sA[buf][sw32(r, ch)], P + (ok ? base[r] + toff : 0) + ch * 8, ok ? 16 : 0);
            cp16(&sB[buf][sw32(r, ch)], Wt + (size_t)(n0 + r) * kp + (size_t)kc * 32 + ch * 8, 16);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    float acc[4][4][4];
#pragma unroll
    for (int i = 0; i < 4; i++)
#pragma unroll
        for (int j = 0; j < 4; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
    load(0, 0);
    for (int kc = 0; kc < nk; kc++) {
        const int buf = kc & 1;
        if (kc + 1 < nk) {
            load(kc + 1, buf ^ 1);
            asm volatile("cp.async.wait_group 1;\n" ::);
        } else {
            asm volatile("cp.async.wait_group 0;\n" ::);
        }
        __syncthreads();
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            uint32_t af[4][4], bfr[2][4];
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
                ldm4(af[mt], &sA[buf][sw32(wm * 64 + mt * 16 + (lane & 15), ks * 2 + (lane >> 4))]);
#pragma unroll
            for (int p = 0; p < 2; p++)
                ldm4(bfr[p], &sB[buf][sw32(wn * 32 + p * 16 + (lane & 7) + (lane >> 4) * 8, ks * 2 + ((lane >> 3) & 1))]);
#pragma unroll
            for (int mt = 0; mt < 4; mt++)
#pragma unroll
                for (int p = 0; p < 2; p++) {
                    mma(acc[mt][2 * p], af[mt], bfr[p][0], bfr[p][1]);
                    mma(acc[mt][2 * p + 1], af[mt], bfr[p][2], bfr[p][3]);
                }
        }
        __syncthreads();
    }
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mt = 0; mt < 4; mt++)
#pragma unroll
        for (int nt8 = 0; nt8 < 4; nt8++) {
            const int n = n0 + wn * 32 + nt8 * 8 + t * 2;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int m = m0 + wm * 64 + mt * 16 + g + h * 8;
                if (m >= M) continue;
                float v0 = rd(rd(acc[mt][nt8][2 * h]) + bias[n]), v1 = rd(rd(acc[mt][nt8][2 * h + 1]) + bias[n + 1]);
                const size_t o = (size_t)m * N + n;
                if (resid) {
                    const __nv_bfloat162 r = *(const __nv_bfloat162*)(resid + o);
                    v0 += __low2float(r), v1 += __high2float(r);
                }
                *(__nv_bfloat162*)(out + o) = __floats2bfloat162_rn(v0, v1);
            }
        }
}

// im2col for k=3 conv over padded P [Tp][Hp][Wp][C]; output voxel m (to, yo, xo), tap (kt < nt, ky, kx), K order
// (kt, ky, kx, c); E = 8: C % 8 == 0; E = 1: any C, columns >= nt*9*C zero (row width kp)
template <int E>
__global__ void k_im2col(const bf16* __restrict__ P, int Hp, int Wp, int C, int nt, int st, int ss, int Ho, int Wo,
                         size_t m0, int rows, int kp, bf16* __restrict__ col) {
    const int CE = C / E, G = E == 8 ? nt * 9 * CE : kp;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * G) return;
    const int g = (int)(i % G);
    const size_t m = m0 + i / G;
    bf16* dst = col + (i / G) * kp + (size_t)g * E;
    if (E == 1 && g >= nt * 9 * C) { *dst = __float2bfloat16(0.f); return; }
    const int tap = g / CE, c = (g % CE) * E, kt = tap / 9, ky = (tap / 3) % 3, kx = tap % 3;
    const int xo = (int)(m % Wo), yo = (int)((m / Wo) % Ho), to = (int)(m / ((size_t)Wo * Ho));
    const bf16* src = P + (((size_t)(to * st + kt) * Hp + yo * ss + ky) * Wp + xo * ss + kx) * C + c;
    if (E == 8) *(uint4*)dst = *(const uint4*)src;
    else *dst = *src;
}

// y = bf16(bf16(y + b) + r)
__global__ void k_bias_res(bf16* __restrict__ y, size_t M, int N, const float* __restrict__ b, const bf16* __restrict__ r) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    float v = rd(ld(y + i) + b[i % N]);
    if (r) v += ld(r + i);
    y[i] = __float2bfloat16(v);
}

// moments NDHWC [T][h][w][48] -> planar [48][T][h][w]
__global__ void k_to_planar(const bf16* __restrict__ m, int T, int HW, bf16* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)48 * T * HW) return;
    const int c = (int)(i / ((size_t)T * HW));
    out[i] = m[(i % ((size_t)T * HW)) * 48 + c];
}

// clip canvas planar [48][5][Hl][Wl] -> out [24][Tout][Hl][Wl] f32 at token offset t0 (tokens >= Tout dropped)
__global__ void k_enc_out(const bf16* __restrict__ cv, int Tc, int HW, int t0, int Tout, const float* __restrict__ lm,
                          const float* __restrict__ ls, float* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)24 * Tc * HW) return;
    const int c = (int)(i / ((size_t)Tc * HW)), t = (int)((i / HW) % Tc);
    if (t0 + t >= Tout) return;
    out[((size_t)c * Tout + t0 + t) * HW + i % HW] = (ld(cv + i) - lm[c]) / ls[c];
}

// ---------------- model ----------------
struct Lin {  // int8 ConvRot linear
    int8_t* w;
    float *ws, *b;
    int n, k;
};
struct Blk {
    Lin qkv, out, w1, w2;
    float *n1, *n2, *s1, *s2;
};
struct Conv {  // [np][kp] bf16 = [cout][kt][ky][kx][cin] (+ last-temporal-tap copy w1 for single frames)
    bf16 *w = nullptr, *w1 = nullptr;
    float* b = nullptr;
    int cin = 0, cout = 0, k = 1, kp = 0, kp1 = 0;
};
struct Res {
    float *g1w, *g1b, *g2w, *g2b;
    Conv c1, c2, sc;
};
struct Lvl {
    Res r[2];
    Conv down;
    int ts = 1, ss = 1;
    bool has_down = false;
};

struct Vae {
    cudaStream_t st;
    Gemm g;
    StFile f;
    // decoder
    std::vector<Blk> blk;
    float *pq_w, *pq_b, *emb_w, *emb_b, *reg, *no_w, *no_b, *po_b;
    bf16* po_w;
    float *lat_mean, *lat_std, *px_mean, *px_std;  // bf16-rounded
    // encoder
    Conv cin, cout, qc;
    Lvl lv[6];
    float *no_gw, *no_gb;
    // decoder work
    int G = 0, Lw = 0;
    bf16 *x = nullptr, *Qb, *Kb, *Vb, *Ob, *lnb;
    int8_t* q8;
    float *qs, *qs2, *pacc;
    int32_t* acc;
    float2* rope = nullptr;
    int rope_key[3] = {0, 0, 0};
    // encoder work
    size_t ecap = 0, pcap = 0, colcap = 96u << 20;
    bf16 *ea = nullptr, *eb = nullptr, *ec = nullptr, *ed = nullptr, *ep = nullptr, *col = nullptr;
    float2* gst = nullptr;
    float* gpart = nullptr;
    size_t gpcap = 0;
};

template <typename T>
T* up(const std::vector<T>& h) {
    T* d = (T*)dmalloc(h.size() * sizeof(T));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

std::vector<float> hf(Vae* v, const std::string& name, size_t n, bool round = true) {  // F32/F16 -> float (bf16-rounded)
    const StTensor& t = v->f.get(name);
    std::vector<float> o(n);
    const uint8_t* p = v->f.ptr(t);
    if (t.dtype == "F32" && t.bytes() == n * 4) {
        for (size_t i = 0; i < n; i++) o[i] = ((const float*)p)[i];
    } else if (t.dtype == "F16" && t.bytes() == n * 2) {
        for (size_t i = 0; i < n; i++) o[i] = __half2float(((const __half*)p)[i]);
    } else {
        DIE("%s: %s %llu bytes, expected %zu values", name.c_str(), t.dtype.c_str(), (unsigned long long)t.bytes(), n);
    }
    if (round)
        for (float& x : o) x = hround(x);
    return o;
}
float* upf(Vae* v, const std::string& name, size_t n) { return up(hf(v, name, n)); }

Lin load_lin(Vae* v, const std::string& p, int n, int k) {
    Lin l{nullptr, nullptr, nullptr, n, k};
    const StTensor& t = v->f.get(p + ".weight");
    if (t.dtype != "I8" || t.bytes() != (size_t)n * k) DIE("%s.weight: expected I8 [%d, %d]", p.c_str(), n, k);
    l.w = (int8_t*)dmalloc((size_t)n * k);
    CK(cudaMemcpy(l.w, v->f.ptr(t), (size_t)n * k, cudaMemcpyHostToDevice));
    l.ws = up(hf(v, p + ".weight_scale", n, false));
    l.b = upf(v, p + ".bias", n);
    return l;
}

Conv load_conv(Vae* v, const std::string& p, int cin, int cout, int k) {
    Conv c;
    c.cin = cin, c.cout = cout, c.k = k;
    const int taps = k * k * k;
    c.kp = rup(taps * cin, 8);
    std::vector<float> w = hf(v, p + ".weight", (size_t)cout * cin * taps);
    std::vector<bf16> r((size_t)cout * c.kp, __float2bfloat16(0.f));
    for (int o = 0; o < cout; o++)
        for (int i = 0; i < cin; i++)
            for (int t = 0; t < taps; t++) r[(size_t)o * c.kp + t * cin + i] = __float2bfloat16(w[((size_t)o * cin + i) * taps + t]);
    c.w = up(r);
    if (k == 3) {  // single-frame input: ComfyUI keeps only the last temporal tap (causal_zero)
        c.kp1 = rup(9 * cin, 8);
        std::vector<bf16> r1((size_t)cout * c.kp1, __float2bfloat16(0.f));
        for (int o = 0; o < cout; o++)
            for (int j = 0; j < 9 * cin; j++) r1[(size_t)o * c.kp1 + j] = r[(size_t)o * c.kp + 18 * cin + j];
        c.w1 = up(r1);
    }
    c.b = upf(v, p + ".bias", cout);
    return c;
}

Res load_res(Vae* v, const std::string& p, int cin, int cout) {
    Res r;
    r.g1w = upf(v, p + ".norm1.weight", cin), r.g1b = upf(v, p + ".norm1.bias", cin);
    r.g2w = upf(v, p + ".norm2.weight", cout), r.g2b = upf(v, p + ".norm2.bias", cout);
    r.c1 = load_conv(v, p + ".conv1", cin, cout, 3);
    r.c2 = load_conv(v, p + ".conv2", cout, cout, 3);
    if (cin != cout) r.sc = load_conv(v, p + ".nin_shortcut", cin, cout, 1);
    return r;
}

// ---------------- encoder layers ----------------
struct Act {  // NDHWC activation
    bf16* p;
    int T, H, W, C;
};

void ensure_enc(Vae* v, size_t e, size_t pe) {
    if (e > v->ecap) {
        for (bf16* q : {v->ea, v->eb, v->ec, v->ed}) cudaFree(q);
        v->ecap = e;
        v->ea = (bf16*)dmalloc(e * 2), v->eb = (bf16*)dmalloc(e * 2), v->ec = (bf16*)dmalloc(e * 2), v->ed = (bf16*)dmalloc(e * 2);
    }
    if (pe > v->pcap) {
        cudaFree(v->ep);
        v->pcap = pe;
        v->ep = (bf16*)dmalloc(pe * 2);
    }
}

// CausalConv3d: [optional GN+SiLU] -> reflect pad (l, r, t, b), causal zero frames -> conv (+ bias, + residual)
Act conv3(Vae* v, const Conv& c, const Act& x, const float* gw, const float* gb, int pl, int pr, int pt, int pb,
          int ts, int ss, bf16* out, const bf16* resid) {
    const cudaStream_t st = v->st;
    Act y{out, 0, 0, 0, c.cout};
    if (c.k == 1) {
        y.T = x.T, y.H = x.H, y.W = x.W;
        prof.mark(st, "other");
        gemm(&v->g, G_BF16, (int)((size_t)x.T * x.H * x.W), c.cout, c.cin, x.p, c.cin, c.w, c.kp, out);
        prof.mark(st, "conv1x1");
    } else {
        const bool single = x.T == 1;  // ComfyUI: no front padding, last temporal tap only
        const int front = single ? 0 : 2, nt = single ? 1 : 3;
        const int Tp = x.T + front, Hp = x.H + pt + pb, Wp = x.W + pl + pr;
        const size_t pn = (size_t)Tp * Hp * Wp * x.C;
        if (pn > v->pcap) DIE("h3_vae: pad buffer too small");
        const float2* gs = nullptr;
        if (gw) {
            const int HW = x.H * x.W, nch = (HW + 255) / 256;
            if ((size_t)x.T * nch * x.C > v->gpcap) {
                cudaFree(v->gpart);
                v->gpcap = (size_t)x.T * nch * x.C;
                v->gpart = (float*)dmalloc(v->gpcap * 4);
            }
            for (int pass = 0; pass < 2; pass++) {
                k_gn_part<<<x.T * nch, x.C, 0, st>>>(x.p, HW, x.C, nch, pass ? v->gst : nullptr, v->gpart);
                k_gn_fin<<<x.T * 32, 32, 0, st>>>(v->gpart, HW, x.C, nch, 1e-6f, pass, v->gst);
            }
            gs = v->gst;
        }
        prof.mark(st, "other");
        if (x.C % 8 == 0)
            k_pad<8><<<nblk(pn / 8, 256), 256, 0, st>>>(x.p, x.T, x.H, x.W, x.C, gs, gw, gb, pl, pr, pt, pb, front, v->ep);
        else
            k_pad<1><<<nblk(pn, 256), 256, 0, st>>>(x.p, x.T, x.H, x.W, x.C, gs, gw, gb, pl, pr, pt, pb, front, v->ep);
        prof.mark(st, "gn_pad");
        y.T = (Tp - nt) / ts + 1, y.H = (Hp - 3) / ss + 1, y.W = (Wp - 3) / ss + 1;
        const int kp = single ? c.kp1 : c.kp;
        const bf16* w = single ? c.w1 : c.w;
        const size_t M = (size_t)y.T * y.H * y.W;
        if (x.C % 32 == 0 && c.cout % 128 == 0) {
            k_iconv<<<dim3((unsigned)nblk(M, 128), c.cout / 128), 256, 0, st>>>(v->ep, Hp, Wp, x.C, nt, ts, ss, y.H, y.W,
                                                                                (int)M, w, kp, c.b, resid, c.cout, out);
            prof.mark(st, "iconv");
            return y;
        }
        const int rows = (int)std::max<size_t>(1, std::min<size_t>(M, v->colcap / kp));
        for (size_t m0 = 0; m0 < M; m0 += rows) {
            const int n = (int)std::min<size_t>(rows, M - m0);
            if (x.C % 8 == 0) {
                const size_t t = (size_t)n * nt * 9 * (x.C / 8);
                k_im2col<8><<<nblk(t, 256), 256, 0, st>>>(v->ep, Hp, Wp, x.C, nt, ts, ss, y.H, y.W, m0, n, kp, v->col);
            } else {
                const size_t t = (size_t)n * kp;
                k_im2col<1><<<nblk(t, 256), 256, 0, st>>>(v->ep, Hp, Wp, x.C, nt, ts, ss, y.H, y.W, m0, n, kp, v->col);
            }
            prof.mark(st, "im2col");
            gemm(&v->g, G_BF16, n, c.cout, kp, v->col, kp, w, kp, out + m0 * c.cout);
            prof.mark(st, "conv_gemm");
        }
    }
    const size_t M = (size_t)y.T * y.H * y.W;
    k_bias_res<<<nblk(M * c.cout, 256), 256, 0, st>>>(out, M, c.cout, c.b, resid);
    prof.mark(st, "bias");
    return y;
}

// one spatial tile of one clip: normalized pixels in v->ea -> moments NDHWC [T'][h][w][48] in returned buffer
Act enc_tile(Vae* v, Act x) {
    bf16* bufs[4] = {v->ea, v->eb, v->ec, v->ed};
    auto other = [&](std::initializer_list<const bf16*> used) {
        for (bf16* b : bufs) {
            bool u = false;
            for (const bf16* q : used) u |= q == b;
            if (!u) return b;
        }
        DIE("h3_vae: no free buffer");
        return (bf16*)nullptr;
    };
    x = conv3(v, v->cin, x, nullptr, nullptr, 1, 1, 1, 1, 1, 1, other({x.p}), nullptr);
    for (int l = 0; l < 6; l++) {
        const Lvl& L = v->lv[l];
        for (int j = 0; j < 2; j++) {
            const Res& r = L.r[j];
            Act h = conv3(v, r.c1, x, r.g1w, r.g1b, 1, 1, 1, 1, 1, 1, other({x.p}), nullptr);
            Act s = x;
            if (r.sc.w) s = conv3(v, r.sc, x, nullptr, nullptr, 0, 0, 0, 0, 1, 1, other({x.p, h.p}), nullptr);
            x = conv3(v, r.c2, h, r.g2w, r.g2b, 1, 1, 1, 1, 1, 1, other({h.p, s.p}), s.p);
        }
        if (L.has_down) {
            const int p = L.ss == 2 ? 1 : 0;
            x = conv3(v, L.down, x, nullptr, nullptr, 0, p, 0, p, L.ts, L.ss, other({x.p}), nullptr);
        }
    }
    x = conv3(v, v->cout, x, v->no_gw, v->no_gb, 1, 1, 1, 1, 1, 1, other({x.p}), nullptr);
    return conv3(v, v->qc, x, nullptr, nullptr, 0, 0, 0, 0, 1, 1, other({x.p}), nullptr);
}

// ComfyUI split_tiles: (starts, lens, overlaps) for a length in pixels, tile 256, min overlap 64, unit 16
void split_tiles(int n, std::vector<int>& s, std::vector<int>& len, std::vector<int>& ov) {
    s.clear(), len.clear(), ov.clear();
    if (256 >= n) { s = {0}, len = {n}; return; }
    int N = (n + 255) / 256, rem;
    for (;;) {
        ov.assign(N - 1, 64);
        rem = 256 * N - 64 * (N - 1) - n;
        if (rem < 0) N++;
        else break;
    }
    for (int i = 0; i < rem / 16; i++) ov[i % (N - 1)] += 16;
    s.push_back(0);
    for (int i = 0; i < N - 1; i++) s.push_back(s.back() + 256 - ov[i]);
    len.assign(N, 256);
}

void blend2d(cudaStream_t st, int P, int R, int C, const bf16* A, size_t aps, int ald, bf16* B, size_t bps, int bld,
             bool vertical, int e) {
    if (R <= 0 || C <= 0 || e <= 0) return;
    k_blend2d<<<nblk((size_t)P * R * C, 256), 256, 0, st>>>(P, R, C, A, aps, ald, B, bps, bld, vertical, e);
}
void copy2d(cudaStream_t st, int P, int R, int C, const bf16* S, size_t sps, int sld, bf16* D, size_t dps, int dld) {
    if (R <= 0 || C <= 0) return;
    k_copy2d<<<nblk((size_t)P * R * C, 256), 256, 0, st>>>(P, R, C, S, sps, sld, D, dps, dld);
}

// ---------------- decoder ----------------
void ensure_dec(Vae* v, int G, int L) {
    if (G <= v->G && L <= v->Lw) return;
    for (void* p : {(void*)v->x, (void*)v->Qb, (void*)v->Kb, (void*)v->Vb, (void*)v->Ob, (void*)v->lnb, (void*)v->q8,
                    (void*)v->qs, (void*)v->qs2, (void*)v->acc})   // pacc aliases acc: freeing it too was a double free
        CK(cudaFree(p));
    v->G = std::max(G, v->G), v->Lw = std::max(L, v->Lw);
    const size_t M = (size_t)v->G * v->Lw;
    v->x = (bf16*)dmalloc(M * 2048 * 2), v->Qb = (bf16*)dmalloc(M * 2048 * 2), v->Kb = (bf16*)dmalloc(M * 2048 * 2);
    v->Vb = (bf16*)dmalloc(M * 2048 * 2), v->Ob = (bf16*)dmalloc(M * 2048 * 2), v->lnb = (bf16*)dmalloc(M * 2048 * 2);
    v->q8 = (int8_t*)dmalloc(M * 8192), v->qs = (float*)dmalloc(M * 4), v->qs2 = (float*)dmalloc(M * 4);
    v->acc = (int32_t*)dmalloc(M * 16384 * 4);
    v->pacc = (float*)v->acc;  // proj_out accumulators reuse the int32 buffer (M x 3072 floats)
}

// rope table for one tile shape (T, th, tw): [L][24] (cos, sin), ids/table rounded as ComfyUI (bf16 coords and table)
void make_rope(Vae* v, int T, int th, int tw) {
    if (v->rope_key[0] == T && v->rope_key[1] == th && v->rope_key[2] == tw) return;
    const int L0 = T * th * tw, L = L0 + 5;
    float inv[8];
    for (int k = 0; k < 8; k++) inv[k] = hround(1.f / powf(100.f, 0.125f * k));
    auto coord = [](int k, int n) { return hround(hround(2.f * hround((k + 0.5f) / (float)n)) - 1.f); };
    std::vector<float2> tab((size_t)L * 24, make_float2(1.f, 0.f));
    const float sc = (float)6.283185307179586;
    for (int r = 0; r < L0; r++) {
        const int ids[3] = {r / (th * tw), (r / tw) % th, r % tw}, dims[3] = {T, th, tw};
        for (int a = 0; a < 3; a++)
            for (int k = 0; k < 8; k++) {
                const float ang = sc * coord(ids[a], dims[a]) * inv[k];
                tab[(size_t)r * 24 + a * 8 + k] = make_float2(hround(cosf(ang)), hround(sinf(ang)));
            }
    }
    cudaFree(v->rope);
    v->rope = up(tab);
    v->rope_key[0] = T, v->rope_key[1] = th, v->rope_key[2] = tw;
}

void rq(Vae* v, int mode, const void* in, int M, int K, const float* p0, float* qs) {
    const unsigned nb = (unsigned)nblk(M, 4 / (K / 2048));
    if (mode == 0) k_rq<0><<<nb, 128, 0, v->st>>>(in, M, K, p0, v->q8, qs);
    else if (mode == 1) k_rq<1><<<nb, 128, 0, v->st>>>(in, M, K, p0, v->q8, qs);
    else k_rq<2><<<nb, 128, 0, v->st>>>(in, M, K, p0, v->q8, qs);
}

// G tiles (same shape) of one latent chunk -> tile pixels [G][3][4T][16th][16tw] bf16
void dec_tiles(Vae* v, const bf16* zq, int lh, int lw, int T, int th, int tw, const std::vector<int2>& org,
               bf16* tiles, size_t tsz) {
    const cudaStream_t st = v->st;
    const int G = (int)org.size(), L0 = T * th * tw, L = L0 + 5, M = G * L;
    ensure_dec(v, G, L);
    make_rope(v, T, th, tw);
    int2* dorg = up(org);
    prof.mark(st, "other");
    k_embed<<<nblk((size_t)M * 2048, 256), 256, 0, st>>>(zq, lh, lw, T, th, tw, dorg, v->emb_w, v->emb_b, v->reg, L, G, v->x);
    const float sl2 = 0.125f * 1.4426950408889634f;
    for (const Blk& b : v->blk) {
        prof.mark(st, "other");
        rq(v, 1, v->x, M, 2048, b.n1, v->qs);
        prof.mark(st, "rq");
        bf16* y = (bf16*)v->acc;   // dequantized GEMM outputs (bf16, half the int32 accumulators' bytes)
        gemm_i8_epi(st, DQ_F32B, M, 6144, 2048, v->q8, b.qkv.w, v->qs, b.qkv.ws, b.qkv.b, nullptr, y);
        prof.mark(st, "gemm_qkv");
        k_qkv<<<nblk((size_t)M * 96 * 32, 256), 256, 0, st>>>(y, v->rope, L, M, v->Qb, v->Kb, v->Vb);
        prof.mark(st, "qkv_post");
        k_flash64<<<dim3((L + 63) / 64, 32, G), 128, 0, st>>>(v->Qb, v->Kb, v->Vb, v->Ob, L, sl2);
        prof.mark(st, "attn");
        rq(v, 0, v->Ob, M, 2048, nullptr, v->qs);
        prof.mark(st, "rq");
        gemm_i8_epi(st, DQ_RES, M, 2048, 2048, v->q8, b.out.w, v->qs, b.out.ws, b.out.b, b.s1, v->x);   // x += s1 * out
        prof.mark(st, "gemm_out");
        rq(v, 1, v->x, M, 2048, b.n2, v->qs);
        prof.mark(st, "rq");
        gemm_i8_epi(st, DQ_F32B, M, 16384, 2048, v->q8, b.w1.w, v->qs, b.w1.ws, b.w1.b, nullptr, y);
        prof.mark(st, "gemm_w1");
        rq(v, 2, y, M, 8192, nullptr, v->qs2);
        prof.mark(st, "rq_swiglu");
        gemm_i8_epi(st, DQ_RES, M, 2048, 8192, v->q8, b.w2.w, v->qs2, b.w2.ws, b.w2.b, b.s2, v->x);   // x += s2 * w2
        prof.mark(st, "gemm_w2");
    }
    k_ln<<<G * L0, 256, 0, st>>>(v->x, L, L0, v->no_w, v->no_b, v->lnb);
    gemm(&v->g, G_F32, G * L0, 3072, 2048, v->lnb, 2048, v->po_w, 2048, v->pacc);
    k_unpatch<<<nblk((size_t)G * L0 * 3072, 256), 256, 0, st>>>(v->pacc, v->po_b, T, th, tw, G, tiles, tsz);
    prof.mark(st, "final");
    CK(cudaStreamSynchronize(st));
    cudaFree(dorg);
}

// ComfyUI _decode_temporal_chunks / _decode_temporal_frame_plan (clip 17, ratio 4, token drop 3)
const int TCH = 5, TOV = 2, FPRE = 3, FOV = 5;
void tplan(int T, int& pad, int& chunks) {
    const int pseudo = T + 3;
    pad = (TCH - pseudo % TCH) % TCH;
    chunks = (pseudo + pad) / TCH - 1;
    if (chunks < 1) pad += TCH, chunks += 1;
}
int frames_for(int T) {
    if (T == 1) return 1;
    int pad, chunks;
    tplan(T, pad, chunks);
    const int zl = T + pad;
    int total = 0, fin = 0;
    for (int i = 0; i < chunks; i++) {
        const int s = i * TCH, e = s + TCH + TOV;
        const int ctl = std::max(0, std::min(e, zl) - std::min(s, zl)), cfl = ctl * 4;
        for (int j = 0; j < 2; j++) {
            const int f0 = j * 20, f1 = std::min(f0 + 20, cfl), cf = std::max(0, f1 - f0 - FPRE);
            if (j == 0) total += cf;
            else fin = cf;
        }
    }
    total += fin;
    int padf = 0;
    for (int k = 0; k < pad; k++) padf += ((T + k) % TCH == 0) ? 1 : 4;  // intra tail 17 % 4 = 1
    return total - padf;
}
}  // namespace

// ================= audio VAE decoder (comfy/ldm/minimax/audio_vae.py: BigVGAN, 32 kHz, bf16 as --bf16-vae) =================
// Activations [T][C] bf16; every torch op rounds to bf16 (snake step by step, conv then + bias, residual adds).
namespace {
// latent [32][2][T] f32 (normalized), stereo channel s -> z [T][32] bf16 = bf16(bf16(bf16(z) * std) + mean)
__global__ void k_aud_in(const float* __restrict__ lat, int T, int s, const float* __restrict__ sd,
                         const float* __restrict__ mn, bf16* __restrict__ z) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= T * 32) return;
    const int c = i % 32, t = i / 32;
    z[i] = __float2bfloat16(rd(rd(lat[((size_t)c * 2 + s) * T + t]) * sd[c]) + mn[c]);
}

// conv1d im2col: col [T][k*C] = x[t + j*d - p][c] (zero outside), C % 8 == 0
__global__ void k_im2col1(const bf16* __restrict__ x, int T, int C, int k, int d, int p, bf16* __restrict__ col) {
    const int C8 = C / 8;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * k * C8) return;
    const int c = (int)(i % C8) * 8, j = (int)((i / C8) % k);
    const int t = (int)(i / ((size_t)C8 * k)), ti = t + j * d - p;
    *(uint4*)(col + i * 8) = (ti >= 0 && ti < T) ? *(const uint4*)(x + (size_t)ti * C + c) : make_uint4(0, 0, 0, 0);
}

// conv_transpose1d overlap-add: Y [Tin][cout*k] f32 -> out [Tout][cout] = bf16(bf16(sum) + b)
__global__ void k_col2im(const float* __restrict__ Y, int Tin, int co, int k, int u, int p, int Tout,
                         const float* __restrict__ b, bf16* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)Tout * co) return;
    const int o = (int)(i % co), t = (int)(i / co);
    float s = 0.f;
    for (int j = (t + p) % u; j < k; j += u) {
        const int ti = (t + p - j) / u;
        if (ti >= 0 && ti < Tin) s += Y[((size_t)ti * co + o) * k + j];
    }
    out[i] = __float2bfloat16(rd(s) + b[o]);
}

// Activation1d first half: replicate pad 5, depthwise conv_transpose (stride 2, 12 taps) rounded, x2, crop 15,
// then SnakeBeta step by step in bf16: x + bf16(1/beta) * sin(alpha x)^2. x [T][C] -> u [2T][C]
__global__ void k_upsnake(const bf16* __restrict__ x, int T, int C, const float* __restrict__ f,
                          const float* __restrict__ a, const float* __restrict__ ib, bf16* __restrict__ u) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)2 * T * C) return;
    const int c = (int)(i % C), m = (int)(i / C), n = m + 15;
    float s = 0.f;
    for (int j = n & 1; j < 12; j += 2) {
        const int xi = (n - j) / 2;
        if (xi >= 0 && xi < T + 10) s += ld(x + (size_t)min(max(xi - 5, 0), T - 1) * C + c) * f[j];
    }
    const float v = rd(s) * 2.f;
    float t = rd(sinf(rd(a[c] * v)));
    t = rd(t * t);
    t = rd(t * ib[c]);
    u[i] = __float2bfloat16(t + v);
}

// Activation1d second half: replicate pad (5, 6), depthwise conv1d stride 2, 12 taps. s [2T][C] -> out [T][C]
__global__ void k_down(const bf16* __restrict__ s, int T, int C, const float* __restrict__ f, bf16* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * C) return;
    const int c = (int)(i % C), t = (int)(i / C);
    float acc = 0.f;
    for (int j = 0; j < 12; j++) acc += ld(s + (size_t)min(max(2 * t + j - 5, 0), 2 * T - 1) * C + c) * f[j];
    out[i] = __float2bfloat16(acc);
}

// mode 0: out = bf16(a + b); mode 1: out = bf16(a / 3)
__global__ void k_ew(int mode, const bf16* __restrict__ a, const bf16* __restrict__ b, bf16* __restrict__ out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(mode == 0 ? ld(a + i) + ld(b + i) : ld(a + i) / 3.f);
}

// conv_post (8 -> 1, 7 taps, pad 3, no bias), rounded, clamped -> out f32
__global__ void k_post(const bf16* __restrict__ x, int L, const float* __restrict__ w, float* __restrict__ out) {
    const int t = blockIdx.x * blockDim.x + threadIdx.x;
    if (t >= L) return;
    float s = 0.f;
    for (int j = 0; j < 7; j++) {
        const int ti = t + j - 3;
        if (ti < 0 || ti >= L) continue;
        for (int c = 0; c < 8; c++) s += ld(x + (size_t)ti * 8 + c) * w[c * 7 + j];
    }
    out[t] = fminf(fmaxf(rd(s), -1.f), 1.f);
}

__global__ void k_sumsq(const float* __restrict__ x, size_t n, double* __restrict__ acc) {
    double s = 0.0, s2 = 0.0;
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        s += x[i];
        s2 += (double)x[i] * x[i];
    }
    atomicAdd(acc, s);
    atomicAdd(acc + 1, s2);
}
__global__ void k_scale(float* __restrict__ x, size_t n, float d) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] /= d;
}

struct AConv {  // conv1d: w [cout][k*cin] bf16; convT: w [cout*k][cin] bf16
    bf16* w = nullptr;
    float* b = nullptr;
    int cin = 0, cout = 0, k = 1, d = 1, p = 0, u = 1;
};
struct AAct {
    float *a, *ib, *fu, *fd;
};
struct AmpBlk {
    AConv c1[3], c2[3];
    AAct act[6];
};
struct Aud {
    cudaStream_t st;
    Gemm g;
    StFile f;
    float *mean, *std, *post;
    AConv in, pre, ups[7];
    AmpBlk rb[21];
    AAct post_act;
    size_t cap = 0, ccap = 0;
    bf16 *b[6] = {}, *col = nullptr;
    float* Y = nullptr;
    double* red = nullptr;
};

std::vector<float> ahf(Aud* a, const std::string& name, size_t n) {
    const StTensor& t = a->f.get(name);
    if (t.dtype != "F32" || t.bytes() != n * 4) DIE("%s: %s, expected F32 x %zu", name.c_str(), t.dtype.c_str(), n);
    const float* p = (const float*)a->f.ptr(t);
    std::vector<float> o(p, p + n);
    for (float& x : o) x = hround(x);
    return o;
}

AConv aconv(Aud* a, const std::string& p, int cin, int cout, int k, int d, bool transposed, int u = 1) {
    AConv c;
    c.cin = cin, c.cout = cout, c.k = k, c.d = d, c.u = u;
    c.p = transposed ? (k - u) / 2 : (k * d - d) / 2;
    std::vector<float> w = ahf(a, p + ".weight", (size_t)cin * cout * k);
    std::vector<bf16> r(w.size());
    for (int o = 0; o < cout; o++)
        for (int i = 0; i < cin; i++)
            for (int j = 0; j < k; j++) {
                if (transposed) r[((size_t)o * k + j) * cin + i] = __float2bfloat16(w[((size_t)i * cout + o) * k + j]);
                else r[(size_t)o * k * cin + j * cin + i] = __float2bfloat16(w[((size_t)o * cin + i) * k + j]);
            }
    c.w = up(r);
    c.b = up(ahf(a, p + ".bias", cout));
    return c;
}

AAct aact(Aud* a, const std::string& p, int C) {
    std::vector<float> al = ahf(a, p + ".act.alpha", C), be = ahf(a, p + ".act.beta", C);
    for (int c = 0; c < C; c++) {
        al[c] = hround(expf(al[c]));
        be[c] = hround(1.f / hround(hround(expf(be[c])) + 1e-9f));
    }
    return AAct{up(al), up(be), up(ahf(a, p + ".upsample.filter", 12)), up(ahf(a, p + ".downsample.lowpass.filter", 12))};
}

// conv1d (zero pad, bias): x [T][cin] -> out [T][cout]; resid: out = bf16(out + resid)
void aconv1(Aud* a, const AConv& c, const bf16* x, int T, bf16* out, const bf16* resid) {
    const int K = c.k * c.cin;
    const bf16* A = x;
    if (c.k > 1) {
        k_im2col1<<<nblk((size_t)T * K / 8, 256), 256, 0, a->st>>>(x, T, c.cin, c.k, c.d, c.p, a->col);
        A = a->col;
    }
    gemm(&a->g, G_BF16, T, c.cout, K, A, K, c.w, K, out);
    k_bias_res<<<nblk((size_t)T * c.cout, 256), 256, 0, a->st>>>(out, (size_t)T, c.cout, c.b, resid);
}

void aact1(Aud* a, const AAct& f, const bf16* x, int T, int C, bf16* u, bf16* out) {
    k_upsnake<<<nblk((size_t)2 * T * C, 256), 256, 0, a->st>>>(x, T, C, f.fu, f.a, f.ib, u);
    k_down<<<nblk((size_t)T * C, 256), 256, 0, a->st>>>(u, T, C, f.fd, out);
}

// AMPBlock1 on x [T][C] -> out (b[2..5] scratch)
void amp(Aud* a, const AmpBlk& r, const bf16* x, int T, int C, bf16* out) {
    bf16 *t1 = a->b[3], *t2 = a->b[4], *u = a->b[5];
    CK(cudaMemcpyAsync(out, x, (size_t)T * C * 2, cudaMemcpyDeviceToDevice, a->st));
    for (int l = 0; l < 3; l++) {
        aact1(a, r.act[2 * l], out, T, C, u, t1);
        aconv1(a, r.c1[l], t1, T, t2, nullptr);
        aact1(a, r.act[2 * l + 1], t2, T, C, u, t1);
        aconv1(a, r.c2[l], t1, T, t2, out);  // t2 = bf16(conv + b) + out
        CK(cudaMemcpyAsync(out, t2, (size_t)T * C * 2, cudaMemcpyDeviceToDevice, a->st));
    }
}
}  // namespace

extern "C" {

void* h3_vae_new(const char* path) {
    Vae* v = new Vae();
    if (!v->f.open(path)) DIE("cannot open %s", path);
    CK(cudaStreamCreate(&v->st));   // blocking on purpose (common.cuh)
    v->g.st = v->st;
    CKL(cublasLtCreate(&v->g.h));
    v->g.ws = dmalloc(v->g.wsz);
    // decoder
    for (int i = 0; i < 36; i++) {
        const std::string p = "decoder.transformer_blocks." + std::to_string(i) + ".";
        Blk b;
        b.qkv = load_lin(v, p + "attn.to_qkv", 6144, 2048);
        b.out = load_lin(v, p + "attn.to_out", 2048, 2048);
        b.w1 = load_lin(v, p + "ff.w1", 16384, 2048);
        b.w2 = load_lin(v, p + "ff.w2", 2048, 8192);
        b.n1 = upf(v, p + "norm1.weight", 2048), b.n2 = upf(v, p + "norm2.weight", 2048);
        b.s1 = upf(v, p + "scale1", 2048), b.s2 = upf(v, p + "scale2", 2048);
        v->blk.push_back(b);
    }
    v->pq_w = upf(v, "post_quant_conv.weight", 24 * 24), v->pq_b = upf(v, "post_quant_conv.bias", 24);
    v->emb_w = upf(v, "decoder.x_embedder.weight", 2048 * 24), v->emb_b = upf(v, "decoder.x_embedder.bias", 2048);
    v->reg = upf(v, "decoder.register_tokens", 4 * 2048);
    v->no_w = upf(v, "decoder.norm_out.weight", 2048), v->no_b = upf(v, "decoder.norm_out.bias", 2048);
    {
        std::vector<float> w = hf(v, "decoder.proj_out.weight", (size_t)3072 * 2048);
        std::vector<bf16> h(w.size());
        for (size_t i = 0; i < w.size(); i++) h[i] = __float2bfloat16(w[i]);
        v->po_w = up(h);
    }
    v->po_b = upf(v, "decoder.proj_out.bias", 3072);
    v->lat_mean = upf(v, "latents_mean", 24), v->lat_std = upf(v, "latents_std", 24);
    v->px_mean = up(std::vector<float>{hround(0.485f), hround(0.456f), hround(0.406f)});
    v->px_std = up(std::vector<float>{hround(0.229f), hround(0.224f), hround(0.225f)});
    // encoder: ch 128, mult (1, 2, 2, 4, 4, 8), space down (2, 2, 2, 2, 1, 1), time down (1, 2, 2, 1, 1, 1)
    const int mid[6] = {128, 256, 256, 512, 512, 1024}, sd[6] = {2, 2, 2, 2, 1, 1}, td[6] = {1, 2, 2, 1, 1, 1};
    v->cin = load_conv(v, "encoder.conv_in", 3, 128, 3);
    for (int l = 0; l < 6; l++) {
        const std::string p = "encoder.down." + std::to_string(l) + ".";
        const int bin = l ? mid[l - 1] : 128;
        v->lv[l].r[0] = load_res(v, p + "block.0", bin, mid[l]);
        v->lv[l].r[1] = load_res(v, p + "block.1", mid[l], mid[l]);
        if (sd[l] * td[l] > 1) {
            v->lv[l].has_down = true, v->lv[l].ts = td[l], v->lv[l].ss = sd[l];
            v->lv[l].down = load_conv(v, p + "downsample.conv", mid[l], mid[l], 3);
        }
    }
    v->no_gw = upf(v, "encoder.norm_out.weight", 1024), v->no_gb = upf(v, "encoder.norm_out.bias", 1024);
    v->cout = load_conv(v, "encoder.conv_out", 1024, 48, 3);
    v->qc = load_conv(v, "quant_conv", 48, 48, 1);
    v->f.close();
    v->col = (bf16*)dmalloc(v->colcap * 2);
    v->gst = (float2*)dmalloc(32 * 32 * sizeof(float2));
    CK(cudaDeviceSynchronize());  // uploads complete before the non-blocking stream uses them
    return v;
}

// number of frames h3_vae_decode writes for T latent frames (17k+5 frames <-> 5k+2 latents)
int h3_vae_frames(int T) { return frames_for(T); }

// latent [24, T, lh, lw] f32 device, normalized (what VAEDecode receives)
// -> out [F, 16 lh, 16 lw, 3] f32 device in 0..1, F = h3_vae_frames(T). Returns F.
int h3_vae_decode(void* h, const float* lat, int T, int lh, int lw, float* out) {
    Vae* v = (Vae*)h;
    const cudaStream_t st = v->st;
    const int H = lh * 16, W = lw * 16, HW = H * W, F = frames_for(T);
    std::vector<int> ys, yl, yo, xs, xl, xo;
    split_tiles(H, ys, yl, yo);
    split_tiles(W, xs, xl, xo);
    const int ny = (int)ys.size(), nx = (int)xs.size(), th = yl[0] / 16, tw = xl[0] / 16;
    const bool still = T == 1;
    int pad = 0, chunks = 1;
    if (!still) tplan(T, pad, chunks);
    const int ct = still ? TCH : TCH + TOV, Fc = ct * 4, P = 3 * Fc;
    const size_t tsz = (size_t)P * th * 16 * tw * 16;
    bf16* zq = (bf16*)dmalloc((size_t)ct * lh * lw * 24 * 2);
    bf16* tiles = (bf16*)dmalloc(tsz * ny * nx * 2);
    bf16* cv = (bf16*)dmalloc((size_t)P * HW * 2);
    bf16* strip = (bf16*)dmalloc((size_t)P * 256 * W * 2);
    bf16* nstrip = (bf16*)dmalloc((size_t)P * 256 * W * 2);
    bf16* ov = (bf16*)dmalloc((size_t)3 * FOV * HW * 2);
    const int gmax = 8, ntile = ny * nx, ng = (ntile + gmax - 1) / gmax, gsz = (ntile + ng - 1) / ng;
    int wpos = 0;
    bool have_ov = false;
    for (int ci = 0; ci < chunks; ci++) {
        const int t0 = still ? 0 : ci * TCH;  // repeated still / padded tail = copies of the last latent
        k_dec_prep<<<nblk((size_t)ct * lh * lw * 24, 256), 256, 0, st>>>(lat, still ? 1 : T, still ? 0 : t0, ct, lh * lw,
                                                                          v->lat_std, v->lat_mean, v->pq_w, v->pq_b, zq);
        std::vector<int2> org;
        for (int i = 0; i < ny; i++)
            for (int j = 0; j < nx; j++) org.push_back(make_int2(ys[i] / 16, xs[j] / 16));
        for (int g0 = 0; g0 < ntile; g0 += gsz) {
            std::vector<int2> o(org.begin() + g0, org.begin() + std::min(ntile, g0 + gsz));
            dec_tiles(v, zq, lh, lw, ct, th, tw, o, tiles + g0 * tsz, tsz);
        }
        // spatial compose (ComfyUI tiled_decode): vertical blend with the previous row's strip, horizontal blend
        // with the left neighbour's tail, crop overlaps, keep the bottom overlap as the next strip
        const int TH = th * 16, TW = tw * 16;
        int oy = 0;
        for (int i = 0; i < ny; i++) {
            int ox = 0, chh = TH;
            for (int j = 0; j < nx; j++) {
                bf16* tb = tiles + (size_t)(i * nx + j) * tsz;
                const size_t ps = (size_t)TH * TW;
                if (i > 0) blend2d(st, P, std::min(yo[i - 1], TH), TW, strip + xs[j], (size_t)yo[i - 1] * W, W, tb, ps, TW, true, std::min(yo[i - 1], TH));
                if (j > 0) {
                    const int e = std::min(xo[j - 1], TW);
                    blend2d(st, P, TH, e, tiles + (size_t)(i * nx + j - 1) * tsz + (TW - xo[j - 1]), ps, TW, tb, ps, TW, false, e);
                }
                const int cw = j < nx - 1 ? TW - xo[j] : TW;
                chh = i < ny - 1 ? TH - yo[i] : TH;
                if (i < ny - 1) copy2d(st, P, yo[i], cw, tb + (size_t)(TH - yo[i]) * TW, ps, TW, nstrip + ox, (size_t)yo[i] * W, W);
                copy2d(st, P, chh, cw, tb, ps, TW, cv + (size_t)oy * W + ox, (size_t)HW, W);
                ox += cw;
            }
            std::swap(strip, nstrip);
            oy += chh;
        }
        if (still) {  // frame 3 of the repeated clip
            k_final<<<nblk((size_t)HW * 3, 256), 256, 0, st>>>(cv, Fc, FPRE, 1, nullptr, 0, HW, v->px_std, v->px_mean, out);
            wpos = 1;
            break;
        }
        // temporal: part 0 = frames [3, 20) (blended with the previous overlap), part 1 = [23, 28) kept as overlap
        const int n0 = std::max(0, std::min(20, Fc) - FPRE), n1 = std::max(0, Fc - 20 - FPRE);
        const int w0 = std::min(n0, F - wpos);
        if (w0 > 0)
            k_final<<<nblk((size_t)w0 * HW * 3, 256), 256, 0, st>>>(cv, Fc, FPRE, w0, have_ov ? ov : nullptr, FOV, HW, v->px_std,
                                                                     v->px_mean, out + (size_t)wpos * HW * 3);
        wpos += std::max(0, w0);
        if (n1 > 0) {
            CK(cudaStreamSynchronize(st));
            for (int c = 0; c < 3; c++)
                CK(cudaMemcpyAsync(ov + (size_t)c * FOV * HW, cv + ((size_t)c * Fc + 20 + FPRE) * HW, (size_t)n1 * HW * 2,
                                   cudaMemcpyDeviceToDevice, st));
            have_ov = true;
        }
        if (ci == chunks - 1 && have_ov) {
            const int w1 = std::min(n1, F - wpos);
            if (w1 > 0)
                k_final<<<nblk((size_t)w1 * HW * 3, 256), 256, 0, st>>>(ov, FOV, 0, w1, nullptr, 0, HW, v->px_std, v->px_mean,
                                                                         out + (size_t)wpos * HW * 3);
            wpos += std::max(0, w1);
        }
        CK(cudaStreamSynchronize(st));
    }
    CK(cudaStreamSynchronize(st));
    for (void* p : {(void*)zq, (void*)tiles, (void*)cv, (void*)strip, (void*)nstrip, (void*)ov}) cudaFree(p);
    prof.report("decode");
    CK(cudaGetLastError());   // launches here are unchecked: surface their errors in this call
    return wpos;
}

// latent frames h3_vae_encode returns for F frames
int h3_vae_latent_frames(int F) { return F == 1 ? 1 : (F + 16) / 17 * 5 - 3; }

// pixels [F, H, W, 3] f32 device in 0..1 (H, W multiples of 16) -> out [24, T', H/16, W/16] f32 device: the
// normalized latent mean (what VAEEncode gives), T' = h3_vae_latent_frames(F). Returns T'.
int h3_vae_encode(void* h, const float* px, int F, int H, int W, float* out) {
    Vae* v = (Vae*)h;
    const cudaStream_t st = v->st;
    if (H % 16 || W % 16) DIE("h3_vae_encode: %dx%d is not a multiple of 16", W, H);
    std::vector<int> ys, yl, yo, xs, xl, xo;
    split_tiles(H, ys, yl, yo);
    split_tiles(W, xs, xl, xo);
    const int ny = (int)ys.size(), nx = (int)xs.size(), th = yl[0], tw = xl[0];
    const int Tc = F == 1 ? 1 : 17, Tl = F == 1 ? 1 : 5, Tout = h3_vae_latent_frames(F);
    const int clips = F == 1 ? 1 : (F + 16) / 17;
    const int Hl = H / 16, Wl = W / 16, hl = th / 16, wl = tw / 16;
    ensure_enc(v, (size_t)Tc * th * tw * 128, (size_t)(Tc + 2) * (th + 2) * (tw + 2) * 128);
    const size_t msz = (size_t)48 * Tl * hl * wl;
    bf16* mom = (bf16*)dmalloc(msz * ny * nx * 2);  // raw tile moments, planar [48][Tl][hl][wl]
    bf16* tmp = (bf16*)dmalloc(msz * 2);
    bf16* cv = (bf16*)dmalloc((size_t)48 * Tl * Hl * Wl * 2);
    const int P = 48 * Tl;
    for (int ci = 0; ci < clips; ci++) {
        for (int i = 0; i < ny; i++)
            for (int j = 0; j < nx; j++) {
                const size_t n = (size_t)Tc * th * tw * 3;
                k_enc_in<<<nblk(n, 256), 256, 0, st>>>(px, F, H, W, ci * 17, Tc, ys[i], xs[j], th, tw, v->px_mean, v->px_std, v->ea);
                Act m = enc_tile(v, Act{v->ea, Tc, th, tw, 3});
                if (m.T != Tl || m.H != hl || m.W != wl || m.C != 48) DIE("h3_vae_encode: unexpected moments shape");
                k_to_planar<<<nblk(msz, 256), 256, 0, st>>>(m.p, Tl, hl * wl, mom + (size_t)(i * nx + j) * msz);
            }
        // ComfyUI tiled_encode: blend each raw tile with its raw upper / left neighbours, crop, concatenate
        int oy = 0;
        for (int i = 0; i < ny; i++) {
            int ox = 0, chh = hl;
            for (int j = 0; j < nx; j++) {
                const size_t ps = (size_t)hl * wl;
                CK(cudaMemcpyAsync(tmp, mom + (size_t)(i * nx + j) * msz, msz * 2, cudaMemcpyDeviceToDevice, st));
                if (i > 0) {
                    const int e = std::min(yo[i - 1] / 16, hl);
                    blend2d(st, P, e, wl, mom + (size_t)((i - 1) * nx + j) * msz + (size_t)(hl - e) * wl, ps, wl, tmp, ps, wl, true, e);
                }
                if (j > 0) {
                    const int e = std::min(xo[j - 1] / 16, wl);
                    blend2d(st, P, hl, e, mom + (size_t)(i * nx + j - 1) * msz + (wl - e), ps, wl, tmp, ps, wl, false, e);
                }
                chh = i < ny - 1 ? hl - yo[i] / 16 : hl;
                const int cw = j < nx - 1 ? wl - xo[j] / 16 : wl;
                copy2d(st, P, chh, cw, tmp, ps, wl, cv + (size_t)oy * Wl + ox, (size_t)Hl * Wl, Wl);
                ox += cw;
            }
            oy += chh;
        }
        k_enc_out<<<nblk((size_t)24 * Tl * Hl * Wl, 256), 256, 0, st>>>(cv, Tl, Hl * Wl, ci * 5, Tout, v->lat_mean, v->lat_std, out);
        CK(cudaStreamSynchronize(st));
    }
    prof.report("encode");
    for (void* p : {(void*)mom, (void*)tmp, (void*)cv}) cudaFree(p);
    CK(cudaGetLastError());   // launches here are unchecked: surface their errors in this call
    return Tout;
}

void h3_vae_free(void* h) {
    Vae* v = (Vae*)h;
    cudaStreamSynchronize(v->st);
    for (Blk& b : v->blk) {
        for (Lin* l : {&b.qkv, &b.out, &b.w1, &b.w2}) cudaFree(l->w), cudaFree(l->ws), cudaFree(l->b);
        for (float* p : {b.n1, b.n2, b.s1, b.s2}) cudaFree(p);
    }
    auto fc = [](Conv& c) { cudaFree(c.w), cudaFree(c.w1), cudaFree(c.b); };
    for (Lvl& l : v->lv) {
        for (Res& r : l.r) {
            for (float* p : {r.g1w, r.g1b, r.g2w, r.g2b}) cudaFree(p);
            fc(r.c1), fc(r.c2), fc(r.sc);
        }
        fc(l.down);
    }
    fc(v->cin), fc(v->cout), fc(v->qc);
    void* ps[] = {v->pq_w, v->pq_b, v->emb_w, v->emb_b, v->reg, v->no_w, v->no_b, v->po_b, v->po_w, v->lat_mean,
                  v->lat_std, v->px_mean, v->px_std, v->no_gw, v->no_gb, v->x, v->Qb, v->Kb, v->Vb, v->Ob, v->lnb,
                  v->q8, v->qs, v->qs2, v->acc, v->rope, v->ea, v->eb, v->ec, v->ed, v->ep, v->col, v->gst, v->gpart, v->g.ws};
    for (void* p : ps) cudaFree(p);
    for (auto& kv : v->g.plans) {
        cublasLtMatmulDescDestroy(kv.second.op);
        cublasLtMatrixLayoutDestroy(kv.second.lw);
        cublasLtMatrixLayoutDestroy(kv.second.la);
        cublasLtMatrixLayoutDestroy(kv.second.lc);
    }
    cublasLtDestroy(v->g.h);
    cudaStreamDestroy(v->st);
    delete v;
}
// ---------------- audio ----------------
void* h3_audio_new(const char* path) {
    Aud* a = new Aud();
    if (!a->f.open(path)) DIE("cannot open %s", path);
    CK(cudaStreamCreate(&a->st));   // blocking on purpose (common.cuh)
    a->g.st = a->st;
    CKL(cublasLtCreate(&a->g.h));
    a->g.ws = dmalloc(a->g.wsz);
    a->mean = up(ahf(a, "latents_mean", 32)), a->std = up(ahf(a, "latents_std", 32));
    a->in = aconv(a, "dec_in_proj", 32, 2048, 1, 1, false);
    a->pre = aconv(a, "decoder.conv_pre", 2048, 1024, 7, 1, false);
    const int ur[7] = {5, 5, 2, 2, 2, 2, 2}, uk[7] = {9, 9, 4, 4, 4, 4, 4}, rk[3] = {3, 7, 11}, dil[3] = {1, 3, 5};
    for (int i = 0; i < 7; i++) {
        const int ci = 1024 >> i, co = ci / 2;
        a->ups[i] = aconv(a, "decoder.ups." + std::to_string(i) + ".0", ci, co, uk[i], 1, true, ur[i]);
        for (int j = 0; j < 3; j++) {
            AmpBlk& r = a->rb[i * 3 + j];
            const std::string p = "decoder.resblocks." + std::to_string(i * 3 + j) + ".";
            for (int l = 0; l < 3; l++) {
                r.c1[l] = aconv(a, p + "convs1." + std::to_string(l), co, co, rk[j], dil[l], false);
                r.c2[l] = aconv(a, p + "convs2." + std::to_string(l), co, co, rk[j], 1, false);
            }
            for (int l = 0; l < 6; l++) r.act[l] = aact(a, p + "activations." + std::to_string(l), co);
        }
    }
    a->post_act = aact(a, "decoder.activation_post", 8);
    a->post = up(ahf(a, "decoder.conv_post.weight", 56));
    a->f.close();
    a->red = (double*)dmalloc(16);
    CK(cudaDeviceSynchronize());
    return a;
}

// latent [32, 2, T] f32 device, normalized (the audio part of the H3 latent) -> out [2, 800 T] f32 device at 32 kHz,
// scaled like ComfyUI VAEDecodeAudio (divided by 5 * std when that exceeds 1). Returns the sample count per channel.
int h3_audio_decode(void* h, const float* lat, int T, float* out) {
    Aud* a = (Aud*)h;
    const cudaStream_t st = a->st;
    const int L = T * 800;
    // largest activation: 2 * T_stage * C_stage (upsampled) or the 2048-wide input; im2col up to T_stage * 11 * C
    size_t need = (size_t)T * 2048, cneed = (size_t)T * 7 * 2048, yneed = 0;
    for (int i = 0, t = T; i < 7; i++) {
        const int co = 512 >> i;
        yneed = std::max(yneed, (size_t)t * co * a->ups[i].k);
        t *= a->ups[i].u;
        need = std::max(need, (size_t)2 * t * co);
        cneed = std::max(cneed, (size_t)t * 11 * co);
    }
    if (need > a->cap || cneed > a->ccap) {
        for (bf16*& p : a->b) cudaFree(p), p = (bf16*)dmalloc(need * 2);
        cudaFree(a->col), cudaFree(a->Y);
        a->col = (bf16*)dmalloc(cneed * 2), a->Y = (float*)dmalloc(std::max(yneed, (size_t)1) * 4);
        a->cap = need, a->ccap = cneed;
    }
    bf16 *x = a->b[0], *xs = a->b[1], *y = a->b[2];
    for (int s = 0; s < 2; s++) {
        k_aud_in<<<nblk((size_t)T * 32, 256), 256, 0, st>>>(lat, T, s, a->std, a->mean, y);
        aconv1(a, a->in, y, T, x, nullptr);
        aconv1(a, a->pre, x, T, y, nullptr);
        std::swap(x, y);
        int t = T, C = 1024;
        for (int i = 0; i < 7; i++) {
            const AConv& u = a->ups[i];
            const int to = t * u.u;
            gemm(&a->g, G_F32, t, u.cout * u.k, u.cin, x, u.cin, u.w, u.cin, a->Y);
            k_col2im<<<nblk((size_t)to * u.cout, 256), 256, 0, st>>>(a->Y, t, u.cout, u.k, u.u, u.p, to, u.b, y);
            t = to, C = u.cout;
            amp(a, a->rb[i * 3], y, t, C, xs);
            amp(a, a->rb[i * 3 + 1], y, t, C, x);
            k_ew<<<nblk((size_t)t * C, 256), 256, 0, st>>>(0, xs, x, xs, (size_t)t * C);
            amp(a, a->rb[i * 3 + 2], y, t, C, x);
            k_ew<<<nblk((size_t)t * C, 256), 256, 0, st>>>(0, xs, x, xs, (size_t)t * C);
            k_ew<<<nblk((size_t)t * C, 256), 256, 0, st>>>(1, xs, nullptr, x, (size_t)t * C);
        }
        aact1(a, a->post_act, x, t, C, a->b[5], y);
        k_post<<<nblk(L, 256), 256, 0, st>>>(y, L, a->post, out + (size_t)s * L);
    }
    // VAEDecodeAudio: std over everything (unbiased) * 5, divide if >= 1
    CK(cudaMemsetAsync(a->red, 0, 16, st));
    k_sumsq<<<256, 256, 0, st>>>(out, (size_t)2 * L, a->red);
    double r[2];
    CK(cudaMemcpyAsync(r, a->red, 16, cudaMemcpyDeviceToHost, st));
    CK(cudaStreamSynchronize(st));
    const double n = 2.0 * L, var = (r[1] - r[0] * r[0] / n) / (n - 1);
    const float sd = (float)(std::sqrt(std::max(var, 0.0)) * 5.0);
    if (sd >= 1.f) {
        k_scale<<<nblk((size_t)2 * L, 256), 256, 0, st>>>(out, (size_t)2 * L, sd);
        CK(cudaStreamSynchronize(st));
    }
    CK(cudaGetLastError());   // launches here are unchecked: surface their errors in this call
    return L;
}

void h3_audio_free(void* h) {
    Aud* a = (Aud*)h;
    cudaStreamSynchronize(a->st);
    auto fc = [](AConv& c) { cudaFree(c.w), cudaFree(c.b); };
    auto fa = [](AAct& f) { cudaFree(f.a), cudaFree(f.ib), cudaFree(f.fu), cudaFree(f.fd); };
    fc(a->in), fc(a->pre);
    for (AConv& c : a->ups) fc(c);
    for (AmpBlk& r : a->rb) {
        for (int l = 0; l < 3; l++) fc(r.c1[l]), fc(r.c2[l]);
        for (AAct& f : r.act) fa(f);
    }
    fa(a->post_act);
    for (bf16* p : a->b) cudaFree(p);
    void* ps[] = {a->mean, a->std, a->post, a->col, a->Y, a->red, a->g.ws};
    for (void* p : ps) cudaFree(p);
    for (auto& kv : a->g.plans) {
        cublasLtMatmulDescDestroy(kv.second.op);
        cublasLtMatrixLayoutDestroy(kv.second.lw);
        cublasLtMatrixLayoutDestroy(kv.second.la);
        cublasLtMatrixLayoutDestroy(kv.second.lc);
    }
    cublasLtDestroy(a->g.h);
    cudaStreamDestroy(a->st);
    delete a;
}
}  // extern "C"

// ---------------- decoder-block kernels shared with h3_dwd.cu (the pi-H3 DWD refiner layers are these blocks) ----------------
// Launchers only: the kernels above are unchanged, so the VAE's numbers stay the same.
void h3vit_rq(cudaStream_t st, int mode, const void* in, int M, int K, const float* p0, int8_t* q, float* qs) {
    const unsigned nb = (unsigned)nblk(M, 4 / (K / 2048));
    if (mode == 0) k_rq<0><<<nb, 128, 0, st>>>(in, M, K, p0, q, qs);
    else if (mode == 1) k_rq<1><<<nb, 128, 0, st>>>(in, M, K, p0, q, qs);
    else k_rq<2><<<nb, 128, 0, st>>>(in, M, K, p0, q, qs);
}
void h3vit_qkv(cudaStream_t st, const bf16* y, const float2* rope, int L, int M, bf16* Q, bf16* K, bf16* V) {
    k_qkv<<<nblk((size_t)M * 96 * 32, 256), 256, 0, st>>>(y, rope, L, M, Q, K, V);
}
void h3vit_ln(cudaStream_t st, const bf16* x, int G, int L, int L0, const float* w, const float* b, bf16* out) {
    k_ln<<<G * L0, 256, 0, st>>>(x, L, L0, w, b, out);
}
