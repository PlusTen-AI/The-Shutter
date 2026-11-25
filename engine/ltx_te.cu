// shutter engine — LTX-2.5 text encoder: Gemma4-12B (int8 ConvRot weights) + the LTX dual projection, C API
// Numerics follow ComfyUI e9afbccc2 (comfy/text_encoders/gemma4.py, lt.py, sd1_clip.py; comfy/ops.py; comfy_kitchen):
// the encoder runs with fp32 activations and full-precision matmuls — each int8 ConvRot weight is dequantized to fp32,
// W = unrotate(q * s), then a plain fp32 GEMM. We compute the same product as (x H) . q^T * s: x is rotated in fp32 and
// split into three int8 digits (x H / sc = d0 + d1/254 + d2/254^2, ~fp32 precision), one int8 GEMM over the stacked
// digits, combined in fp32 (as qi21_te.cu). Token embeddings: bf16 rows * sqrt(3840) in fp32. Norms, RoPE (split half,
// fp32 tables), attention (scale 1, fp32 SGEMM + softmax) are fp32.
// Gemma4-12B: 48 layers, d 3840, MLP 15360 (gelu tanh); 5 sliding layers (16 x 256 heads, 8 KV heads, window 1024,
// theta 1e4) : 1 global (16 x 512 heads, one KV head with V = rms_norm(raw K), 64 of the 256 pairs rotated, theta 1e6).
// q / k get RMSNorm with weight then RoPE, v RMSNorm without weight. Per layer:
//   x = x + post_attention_norm(o_proj(attn(input_norm(x))))
//   x = (x + post_feedforward_norm(down(gelu_tanh(gate) * up)(pre_feedforward_norm(x)))) * layer_scalar
// ComfyUI pads the prompt on the left to 1024 tokens with an additive mask: padded keys never reach real rows, so only
// the n real rows are computed, at positions max(0, 1024 - n) + i. Sliding layers see keys i - 1023 .. i (binds only
// past 1024 tokens).
// Projection (bf16, as ComfyUI casts the hidden states to bf16): for each of the 49 hidden states (48 layer inputs +
// final norm) per-token RMS over 3840 (eps 1e-6, op-by-op bf16 rounding), flattened d-major / layer-minor to 188160,
// scaled by sqrt(out / 3840), Linear to video 4096 and audio 2048 (bf16 GEMM, fp32 accumulate, + bias, bf16).
#include "common.cuh"
#include "st.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace {
constexpr int D = 3840, NL = 48, NH = 16, FF = 15360, NHS = NL + 1, VOCAB = 262144;
constexpr int HS = 256, KVS = 8, HG = 512, ROTG = 64, WIN = 1024, MIN_LEN = 1024;
constexpr int QKVS = NH * HS + 2 * KVS * HS, QKVG = NH * HG + HG;   // q | k | v (sliding), q | k (global)
constexpr int PK = D * NHS, PV = 4096, PA = 2048, PO = PV + PA;
constexpr float EMB_SCALE = 61.96773353931867f;                    // sqrt(3840)
constexpr size_t SCAP = 256u << 20;                                // attention score buffer, floats (1 GiB)
inline bool is_global(int i) { return i % 6 == 5; }

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
__device__ __forceinline__ float bfr(float v) { return __bfloat162float(__float2bfloat16(v)); }   // round to bf16

// regular Hadamard 256 per group (unnormalized), in shared memory; caller scales by 1/16 (same as qi21_te.cu)
__device__ void hadamard(float* buf, int K) {
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
}

// fp32 rows: pre-op -> ConvRot -> three int8 digit planes (row scale sc = absmax / 127, plane stride `plane`)
enum { R_PLAIN = 0, R_RMS = 1, R_GEGLU = 2 };
template <int MODE>
__global__ void __launch_bounds__(256) k_rq(const float* __restrict__ in, int ld, int K, const float* __restrict__ w,
                                            int8_t* __restrict__ q, size_t plane, float* __restrict__ qs) {
    extern __shared__ float buf[];
    __shared__ float red[32];
    const float* x = in + (size_t)blockIdx.x * ld;
    if (MODE == R_RMS) {
        float s = 0.f;
        for (int j = threadIdx.x; j < K; j += blockDim.x) { const float v = x[j]; buf[j] = v; s += v * v; }
        const float r = rsqrtf(block_reduce<false>(s, red) / K + 1e-6f);
        for (int j = threadIdx.x; j < K; j += blockDim.x) buf[j] = buf[j] * r * w[j];
    } else if (MODE == R_GEGLU) {   // gelu_tanh(gate) * up, torch's formula
        for (int j = threadIdx.x; j < K; j += blockDim.x) {
            const float g = x[j], u = x[K + j];
            buf[j] = 0.5f * g * (1.f + tanhf(0.7978845608028654f * (g + 0.044715f * (g * g * g)))) * u;
        }
    } else {
        for (int j = threadIdx.x; j < K; j += blockDim.x) buf[j] = x[j];
    }
    __syncthreads();
    hadamard(buf, K);
    float mx = 0.f;
    for (int j = threadIdx.x; j < K; j += blockDim.x) { const float v = buf[j] * 0.0625f; buf[j] = v; mx = fmaxf(mx, fabsf(v)); }
    mx = block_reduce<true>(mx, red);
    const float sc = fmaxf(mx / 127.f, 1e-30f);
    int8_t* qo = q + (size_t)blockIdx.x * K;
    for (int j = threadIdx.x; j < K; j += blockDim.x) {
        float v = buf[j] / sc;
        for (int d = 0; d < 3; d++) {
            const float a = rintf(v);  // |a| <= 127: v in [-127, 127], then remainders * 254 in [-127, 127]
            qo[d * plane + j] = (int8_t)a;
            v = (v - a) * 254.f;
        }
    }
    if (threadIdx.x == 0) qs[blockIdx.x] = sc;
}

// digits of row m are acc rows m, Mp + m, 2 Mp + m: out = (a0 + (a1 + a2 / 254) / 254) * (xs[m] * ws[n])
__global__ void k_deq(const int32_t* __restrict__ acc, int M, int Mp, int N, const float* __restrict__ xs,
                      const float* __restrict__ ws, float* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * N) return;
    const int m = (int)(i / N), n = (int)(i % N);
    const size_t pl = (size_t)Mp * N;
    const float a = (float)acc[i] + ((float)acc[pl + i] + (float)acc[2 * pl + i] * (1.f / 254.f)) * (1.f / 254.f);
    out[i] = a * (xs[m] * ws[n]);
}

// bf16 embedding rows -> fp32 * sqrt(3840)
__global__ void k_embed(const bf16* __restrict__ tab, const int* __restrict__ ids, float* __restrict__ x) {
    const bf16* r = tab + (size_t)ids[blockIdx.x] * D;
    for (int j = threadIdx.x; j < D; j += blockDim.x) x[(size_t)blockIdx.x * D + j] = __bfloat162float(r[j]) * EMB_SCALE;
}

// RoPE table row i (position p0 + i): (cos, sin) of inv[j] * pos in fp32, as Gemma4Transformer._freqs_from_inv
__global__ void k_rope_tab(const float* __restrict__ inv, int nf, int p0, float2* __restrict__ cs) {
    const int j = threadIdx.x;
    if (j >= nf) return;
    const float a = inv[j] * (float)(p0 + (int)blockIdx.x);
    cs[(size_t)blockIdx.x * nf + j] = make_float2(cosf(a), sinf(a));
}

// one head of HD in a warp: RMSNorm (weight w, or none) then split-half RoPE (pairs (i, i + HD/2), table cs for
// i < nrot, identity above; cs == nullptr: no RoPE) -> dst
template <int HD>
__device__ __forceinline__ void head(const float* __restrict__ src, const float* __restrict__ w,
                                     const float2* __restrict__ cs, int nrot, float* __restrict__ dst, int lane) {
    constexpr int E = HD / 32;
    float a[E], ss = 0.f;
#pragma unroll
    for (int k = 0; k < E; k++) { a[k] = src[lane + 32 * k]; ss += a[k] * a[k]; }
    const float r = rsqrtf(warp_sum(ss) / HD + 1e-6f);
#pragma unroll
    for (int k = 0; k < E; k++) a[k] = w ? a[k] * r * w[lane + 32 * k] : a[k] * r;
    if (!cs) {
#pragma unroll
        for (int k = 0; k < E; k++) dst[lane + 32 * k] = a[k];
        return;
    }
#pragma unroll
    for (int k = 0; k < E / 2; k++) {
        const int i = lane + 32 * k;
        const float2 c = i < nrot ? cs[i] : make_float2(1.f, 0.f);
        dst[i] = a[k] * c.x - a[k + E / 2] * c.y;
        dst[i + HD / 2] = a[k + E / 2] * c.x + a[k] * c.y;
    }
}

// one warp per (row m, query head h): q head h, and k / v of its KV head g = h / (NH / NKV), written per query head
// (GQA heads repeated) into Q, K, V [M, NH*HD]. qkv rows [M, ld]: q heads, k heads, v heads (KEQV: no v, v = raw k).
template <int HD, int NKV, bool KEQV>
__global__ void k_qkv(const float* __restrict__ qkv, int ld, int M, const float2* __restrict__ cs, int nrot,
                      const float* __restrict__ wq, const float* __restrict__ wk, float* __restrict__ Q,
                      float* __restrict__ K, float* __restrict__ V) {
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int h = gw % NH, m = gw / NH;
    if (m >= M) return;
    const int g = h / (NH / NKV);
    const float* row = qkv + (size_t)m * ld;
    const size_t o = (size_t)m * NH * HD + (size_t)h * HD;
    const float2* c = cs + (size_t)m * nrot;
    head<HD>(row + h * HD, wq, c, nrot, Q + o, lane);
    head<HD>(row + NH * HD + g * HD, wk, c, nrot, K + o, lane);
    head<HD>(row + NH * HD + (KEQV ? 0 : NKV * HD) + g * HD, nullptr, nullptr, 0, V + o, lane);
}

// softmax over the visible keys of each score row (row b: query i = b % L sees keys lo..i, lo = max(0, i - win + 1),
// win 0 = no window); the rest -> 0
__global__ void __launch_bounds__(256) k_softmax(float* __restrict__ S, int L, int win) {
    __shared__ float red[32];
    float* s = S + (size_t)blockIdx.x * L;
    const int i = (int)(blockIdx.x % L), lo = win ? max(0, i - win + 1) : 0, hi = i + 1;
    float mx = -INFINITY;
    for (int j = lo + threadIdx.x; j < hi; j += blockDim.x) mx = fmaxf(mx, s[j]);
    mx = block_reduce<true>(mx, red);
    float sum = 0.f;
    for (int j = lo + threadIdx.x; j < hi; j += blockDim.x) { const float e = expf(s[j] - mx); s[j] = e; sum += e; }
    sum = block_reduce<false>(sum, red);
    const float inv = 1.f / sum;
    for (int j = threadIdx.x; j < L; j += blockDim.x) s[j] = j >= lo && j < hi ? s[j] * inv : 0.f;
}

// x = (x + rmsnorm(h) * w) * ls, rows of D (post-attention / post-feedforward norm, residual, layer_scalar)
__global__ void __launch_bounds__(256) k_post(float* __restrict__ x, const float* __restrict__ h,
                                              const float* __restrict__ w, float ls) {
    __shared__ float red[32];
    const size_t o = (size_t)blockIdx.x * D;
    float s = 0.f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) { const float v = h[o + j]; s += v * v; }
    const float r = rsqrtf(block_reduce<false>(s, red) / D + 1e-6f);
    for (int j = threadIdx.x; j < D; j += blockDim.x) x[o + j] = (x[o + j] + h[o + j] * r * w[j]) * ls;
}

// y = rmsnorm(x) * w, rows of D (final norm)
__global__ void __launch_bounds__(256) k_norm(const float* __restrict__ x, const float* __restrict__ w,
                                              float* __restrict__ y) {
    __shared__ float red[32];
    const size_t o = (size_t)blockIdx.x * D;
    float s = 0.f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) { const float v = x[o + j]; s += v * v; }
    const float r = rsqrtf(block_reduce<false>(s, red) / D + 1e-6f);
    for (int j = threadIdx.x; j < D; j += blockDim.x) y[o + j] = x[o + j] * r * w[j];
}

// projection input for hidden state l of each row (DualLinearProjection, bf16 op by op): b = bf16(x),
// r = bf16(rsqrt(bf16(bf16(mean(bf16(b^2))) + 1e-6))), y = bf16(b * r); xv / xa[m, d*49 + l] = bf16(y * sv / sa)
__global__ void __launch_bounds__(256) k_pin(const float* __restrict__ x, int l, float sv, float sa,
                                             bf16* __restrict__ xv, bf16* __restrict__ xa) {
    __shared__ float red[32];
    const float* r = x + (size_t)blockIdx.x * D;
    float s = 0.f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) { const float b = bfr(r[j]); s += bfr(b * b); }
    const float ms = bfr(block_reduce<false>(s, red) * (1.f / D));
    const float rs = bfr(rsqrtf(bfr(ms + 1e-6f)));
    const size_t o = (size_t)blockIdx.x * PK + l;
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        const float y = bfr(bfr(r[j]) * rs);
        xv[o + (size_t)j * NHS] = __float2bfloat16(y * sv);
        xa[o + (size_t)j * NHS] = __float2bfloat16(y * sa);
    }
}

// out[m] = bf16(video acc + bias) | bf16(audio acc + bias)
__global__ void k_pout(const float* __restrict__ av, const float* __restrict__ aa, const bf16* __restrict__ bv,
                       const bf16* __restrict__ ba, int M, bf16* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * PO) return;
    const int m = (int)(i / PO), c = (int)(i % PO);
    const float v = c < PV ? av[(size_t)m * PV + c] + __bfloat162float(bv[c])
                           : aa[(size_t)m * PA + c - PV] + __bfloat162float(ba[c - PV]);
    out[i] = __float2bfloat16(v);
}

__global__ void k_bf16_to_f32(const bf16* __restrict__ in, float* __restrict__ out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
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

struct Layer {
    bool g;                                       // global attention layer
    int8_t *wqkv, *wo, *wgu, *wd;                 // rotated int8 weights (q|k|v, o, gate|up, down)
    float *sqkv, *so, *sgu, *sd;                  // their row scales
    float *n_in, *n_pa, *n_pf, *n_po, *qn, *kn;   // input / post-attention / pre-ff / post-ff norms, q / k norms
    float ls;                                     // layer_scalar
};

struct Te {
    cudaStream_t st;
    cublasLtHandle_t lh;     // fp32 batched attention GEMMs
    void* ws = nullptr;
    size_t wsz = 64u << 20;
    Lt* lt = nullptr;        // int8 / bf16 GEMMs (gemm.cpp)
    StFile f;
    std::vector<void*> weights;
    bf16* emb;
    Layer l[NL];
    float *fn, *inv_s, *inv_g;
    bf16 *pwv, *pwa, *pbv, *pba;
    float sv, sa;
    Buf tmp, ids, x, h, qkv, Q, K, V, att, gu, aq, as, acc, S, css, csg, xv, xa, pv, pa;
    float* tap = nullptr;
};

void* dmalloc(Te* t, size_t n) {
    void* p;
    CK(cudaMalloc(&p, n));
    t->weights.push_back(p);
    return p;
}

const StTensor& tensor(Te* t, const std::string& name, const char* dtype, size_t elems) {
    const StTensor& s = t->f.get(name);
    const size_t es = s.dtype == "I8" || s.dtype == "U8" ? 1 : s.dtype == "F32" ? 4 : 2;
    if (s.dtype != dtype) DIE("%s: dtype %s, expected %s", name.c_str(), s.dtype.c_str(), dtype);
    if (s.bytes() != elems * es) DIE("%s: %llu bytes, expected %zu elems", name.c_str(), (unsigned long long)s.bytes(), elems);
    return s;
}

void up_into(Te* t, const std::string& name, const char* dtype, size_t elems, void* dst) {
    const StTensor& s = tensor(t, name, dtype, elems);
    CK(cudaMemcpy(dst, t->f.ptr(s), s.bytes(), cudaMemcpyHostToDevice));
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

float* upv(Te* t, const std::vector<float>& v) {
    float* p = (float*)dmalloc(t, v.size() * 4);
    CK(cudaMemcpy(p, v.data(), v.size() * 4, cudaMemcpyHostToDevice));
    return p;
}

void check_convrot(Te* t, const std::string& layer) {
    const StTensor& s = t->f.get(layer + ".comfy_quant");
    const std::string js((const char*)t->f.ptr(s), s.bytes());
    if (js.find("int8_tensorwise") == std::string::npos || js.find("\"convrot\": true") == std::string::npos ||
        js.find("\"convrot_groupsize\": 256") == std::string::npos)
        DIE("%s: unexpected quant config %s", layer.c_str(), js.c_str());
}

// int8 linears stacked along N into one weight [sum N, K] + row scales
void upq(Te* t, const std::string& p, std::initializer_list<std::pair<const char*, int>> parts, int K, int8_t** w,
         float** s) {
    int N = 0;
    for (const auto& q : parts) N += q.second;
    *w = (int8_t*)dmalloc(t, (size_t)N * K);
    *s = (float*)dmalloc(t, (size_t)N * 4);
    int off = 0;
    for (const auto& q : parts) {
        check_convrot(t, p + q.first);
        up_into(t, p + q.first + ".weight", "I8", (size_t)q.second * K, *w + (size_t)off * K);
        up_into(t, p + q.first + ".weight_scale", "F32", q.second, *s + off);
        off += q.second;
    }
}

// row-major fp32 cuBLASLt GEMM (no TF32), heuristic's first algo, strided batches:
// C[M,N] = A[M,K] . (bT ? B[N,K]^T : B[K,N])
void sgemm(Te* t, int M, int N, int K, const float* A, int lda, long long sa, const float* B, int ldb, long long sb,
           bool bT, float* C, int ldc, long long sc, int batch) {
    const float one = 1.f, zero = 0.f;
    cublasLtMatmulDesc_t op;
    CKL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = bT ? CUBLAS_OP_T : CUBLAS_OP_N, tb = CUBLAS_OP_N;
    CKL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CKL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    cublasLtMatrixLayout_t la, lb, lc;
    CKL(cublasLtMatrixLayoutCreate(&la, CUDA_R_32F, bT ? K : N, bT ? N : K, ldb));
    CKL(cublasLtMatrixLayoutCreate(&lb, CUDA_R_32F, K, M, lda));
    CKL(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, N, M, ldc));
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
    if (n == 0) DIE("no cublasLt algo for M=%d N=%d K=%d batch=%d", M, N, K, batch);
    CKL(cublasLtMatmul(t->lh, op, &one, B, la, A, lb, &zero, C, lc, C, lc, &res.algo, t->ws, t->wsz, t->st));
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
}

// fp32 attention over n rows, NH heads of hd (Q / K / V / O rows of NH*hd, KV already repeated), scale 1, causal,
// sliding window win (0 = none)
void attention(Te* t, const float* Q, const float* K, const float* V, float* O, int n, int hd, int win) {
    const int ld = NH * hd;
    const size_t per = (size_t)n * n;
    const int hc = (int)std::max<size_t>(1, std::min<size_t>(NH, SCAP / per));
    float* S = t->S.get<float>(per * hc);
    for (int h0 = 0; h0 < NH; h0 += hc) {
        const int c = std::min(hc, NH - h0);
        sgemm(t, n, n, hd, Q + h0 * hd, ld, hd, K + h0 * hd, ld, hd, true, S, n, per, c);
        k_softmax<<<c * n, 256, 0, t->st>>>(S, n, win);
        sgemm(t, n, hd, n, S, n, per, V + h0 * hd, ld, hd, false, O + h0 * hd, ld, hd, c);
    }
    CK(cudaGetLastError());
}

// rows of `in` -> rotated int8 digit planes for the next lin8() (M rounded up to 64 rows per plane, so GEMM plans repeat)
void rq(Te* t, int mode, const float* in, int ld, int M, int K, const float* w) {
    static const bool init = [] {
        for (auto f : {k_rq<R_PLAIN>, k_rq<R_RMS>, k_rq<R_GEGLU>})
            CK(cudaFuncSetAttribute(f, cudaFuncAttributeMaxDynamicSharedMemorySize, FF * 4));
        return true;
    }();
    (void)init;
    const int Mp = (M + 63) & ~63;
    const size_t plane = (size_t)Mp * K;
    int8_t* q = t->aq.get<int8_t>((size_t)Mp * FF * 3);
    float* qs = t->as.get<float>(M);
    const size_t smem = (size_t)K * 4;
    if (mode == R_RMS) k_rq<R_RMS><<<M, 256, smem, t->st>>>(in, ld, K, w, q, plane, qs);
    else if (mode == R_GEGLU) k_rq<R_GEGLU><<<M, 256, smem, t->st>>>(in, ld, K, w, q, plane, qs);
    else k_rq<R_PLAIN><<<M, 256, smem, t->st>>>(in, ld, K, w, q, plane, qs);
    CK(cudaGetLastError());
}

// out = x . W^T (fp32) on the rows just split by rq(): one int8 GEMM over the three digit planes
void lin8(Te* t, int M, int N, int K, const int8_t* w, const float* ws, float* out) {
    const int Mp = (M + 63) & ~63;
    int32_t* acc = t->acc.get<int32_t>((size_t)Mp * 2 * FF * 3);
    gemm_i8(t->lt, 3 * Mp, N, K, (const int8_t*)t->aq.p, w, acc);
    k_deq<<<nb((size_t)M * N, 256), 256, 0, t->st>>>(acc, M, Mp, N, (const float*)t->as.p, ws, out);
    CK(cudaGetLastError());
}

// hidden state l ([n, D] fp32): debug tap, projection input
void hidden(Te* t, const float* x, int n, int l, bf16* xv, bf16* xa) {
    if (t->tap) CK(cudaMemcpyAsync(t->tap + (size_t)l * n * D, x, (size_t)n * D * 4, cudaMemcpyDeviceToDevice, t->st));
    k_pin<<<n, 256, 0, t->st>>>(x, l, t->sv, t->sa, xv, xa);
    CK(cudaGetLastError());
}

void project(Te* t, int n, const bf16* xv, const bf16* xa, bf16* out) {
    float* av = t->pv.get<float>((size_t)n * PV);
    float* aa = t->pa.get<float>((size_t)n * PA);
    gemm_bf16_f32(t->lt, n, PV, PK, xv, t->pwv, av);
    gemm_bf16_f32(t->lt, n, PA, PK, xa, t->pwa, aa);
    k_pout<<<nb((size_t)n * PO, 256), 256, 0, t->st>>>(av, aa, t->pbv, t->pba, n, out);
    CK(cudaGetLastError());
}
}  // namespace

extern "C" {

void* ltx_te_new(const char* path) {
    Te* t = new Te();
    if (!t->f.open(path)) DIE("cannot open %s", path);
    CK(cudaStreamCreate(&t->st));   // blocking on purpose (common.cuh)
    CKL(cublasLtCreate(&t->lh));
    CK(cudaMalloc(&t->ws, t->wsz));
    t->lt = lt_new(t->st);
    const std::string M = "model.";
    if (t->f.has(M + "layers." + std::to_string(NL) + ".input_layernorm.weight")) DIE("expected exactly %d layers", NL);
    t->emb = (bf16*)up(t, M + "embed_tokens.weight", "BF16", (size_t)VOCAB * D);
    for (int i = 0; i < NL; i++) {
        const std::string p = M + "layers." + std::to_string(i) + ".";
        Layer& l = t->l[i];
        l.g = is_global(i);
        const int hd = l.g ? HG : HS;
        if (l.g != !t->f.has(p + "self_attn.v_proj.weight")) DIE("layer %d: v_proj does not match the layer type", i);
        if (l.g) upq(t, p, {{"self_attn.q_proj", NH * HG}, {"self_attn.k_proj", HG}}, D, &l.wqkv, &l.sqkv);
        else upq(t, p, {{"self_attn.q_proj", NH * HS}, {"self_attn.k_proj", KVS * HS}, {"self_attn.v_proj", KVS * HS}}, D,
                 &l.wqkv, &l.sqkv);
        upq(t, p, {{"self_attn.o_proj", D}}, NH * hd, &l.wo, &l.so);
        upq(t, p, {{"mlp.gate_proj", FF}, {"mlp.up_proj", FF}}, D, &l.wgu, &l.sgu);
        upq(t, p, {{"mlp.down_proj", D}}, FF, &l.wd, &l.sd);
        l.n_in = upf(t, p + "input_layernorm.weight", D);
        l.n_pa = upf(t, p + "post_attention_layernorm.weight", D);
        l.n_pf = upf(t, p + "pre_feedforward_layernorm.weight", D);
        l.n_po = upf(t, p + "post_feedforward_layernorm.weight", D);
        l.qn = upf(t, p + "self_attn.q_norm.weight", hd);
        l.kn = upf(t, p + "self_attn.k_norm.weight", hd);
        l.ls = bf2f(*(const bf16*)t->f.ptr(tensor(t, p + "layer_scalar", "BF16", 1)));
    }
    t->fn = upf(t, M + "norm.weight", D);
    const std::string P = "text_embedding_projection.";
    t->pwv = (bf16*)up(t, P + "video_aggregate_embed.weight", "BF16", (size_t)PV * PK);
    t->pwa = (bf16*)up(t, P + "audio_aggregate_embed.weight", "BF16", (size_t)PA * PK);
    t->pbv = (bf16*)up(t, P + "video_aggregate_embed.bias", "BF16", PV);
    t->pba = (bf16*)up(t, P + "audio_aggregate_embed.bias", "BF16", PA);
    t->sv = (float)std::sqrt((double)PV / D);
    t->sa = (float)std::sqrt((double)PA / D);
    // inverse frequencies as torch computes them on the CPU in fp32: 1 / theta ** (2j / head_dim)
    std::vector<float> is(HS / 2), ig(ROTG);
    for (int j = 0; j < HS / 2; j++) is[j] = 1.f / std::pow(10000.f, (float)(2 * j) / (float)HS);
    for (int j = 0; j < ROTG; j++) ig[j] = 1.f / std::pow(1000000.f, (float)(2 * j) / (float)HG);
    t->inv_s = upv(t, is);
    t->inv_g = upv(t, ig);
    CK(cudaDeviceSynchronize());  // all weight copies landed
    t->f.close();
    cudaFree(t->tmp.p);
    t->tmp = Buf();
    return t;
}

// Encode one prompt. ids: host int32 [n], BOS (2) + the prompt tokens, no padding (LtxTokenizer). Writes out (device
// bf16 [n, 6144] = video 4096 | audio 2048, ComfyUI's conditioning). Returns n, or -1 if n > out_cap.
int ltx_te_encode(void* te, const int* ids, int n, void* out, int out_cap) {
    Te* t = (Te*)te;
    if (n <= 0 || n > out_cap) return -1;
    for (int i = 0; i < n; i++)
        if (ids[i] < 0 || ids[i] >= VOCAB) DIE("token id %d out of range at %d", ids[i], i);
    cudaStream_t st = t->st;
    static const bool prof = getenv("SHUTTER_PROF") != nullptr;
    cudaEvent_t ev[3];
    if (prof) for (auto& e : ev) { CK(cudaEventCreate(&e)); }
    if (prof) CK(cudaEventRecord(ev[0], st));
    const int p0 = std::max(0, MIN_LEN - n);
    float2* css = t->css.get<float2>((size_t)n * (HS / 2));
    float2* csg = t->csg.get<float2>((size_t)n * ROTG);
    k_rope_tab<<<n, HS / 2, 0, st>>>(t->inv_s, HS / 2, p0, css);
    k_rope_tab<<<n, ROTG, 0, st>>>(t->inv_g, ROTG, p0, csg);
    int* dids = t->ids.get<int>(n);
    float* x = t->x.get<float>((size_t)n * D);
    float* hb = t->h.get<float>((size_t)n * D);
    float* qkv = t->qkv.get<float>((size_t)n * QKVG);
    float* Q = t->Q.get<float>((size_t)n * NH * HG);
    float* K = t->K.get<float>((size_t)n * NH * HG);
    float* V = t->V.get<float>((size_t)n * NH * HG);
    float* att = t->att.get<float>((size_t)n * NH * HG);
    float* gu = t->gu.get<float>((size_t)n * 2 * FF);
    bf16* xv = t->xv.get<bf16>((size_t)n * PK);
    bf16* xa = t->xa.get<bf16>((size_t)n * PK);
    CK(cudaMemcpyAsync(dids, ids, (size_t)n * 4, cudaMemcpyHostToDevice, st));
    k_embed<<<n, 256, 0, st>>>(t->emb, dids, x);
    CK(cudaGetLastError());
    for (int i = 0; i < NL; i++) {
        const Layer& l = t->l[i];
        const int hd = l.g ? HG : HS, qn = l.g ? QKVG : QKVS, an = NH * hd;
        hidden(t, x, n, i, xv, xa);
        rq(t, R_RMS, x, D, n, D, l.n_in);
        lin8(t, n, qn, D, l.wqkv, l.sqkv, qkv);
        const int g = nb((size_t)n * NH * 32, 256);
        if (l.g) k_qkv<HG, 1, true><<<g, 256, 0, st>>>(qkv, qn, n, csg, ROTG, l.qn, l.kn, Q, K, V);
        else k_qkv<HS, KVS, false><<<g, 256, 0, st>>>(qkv, qn, n, css, HS / 2, l.qn, l.kn, Q, K, V);
        CK(cudaGetLastError());
        attention(t, Q, K, V, att, n, hd, l.g ? 0 : WIN);
        rq(t, R_PLAIN, att, an, n, an, nullptr);
        lin8(t, n, D, an, l.wo, l.so, hb);
        k_post<<<n, 256, 0, st>>>(x, hb, l.n_pa, 1.f);
        rq(t, R_RMS, x, D, n, D, l.n_pf);
        lin8(t, n, 2 * FF, D, l.wgu, l.sgu, gu);
        rq(t, R_GEGLU, gu, 2 * FF, n, FF, nullptr);
        lin8(t, n, D, FF, l.wd, l.sd, hb);
        k_post<<<n, 256, 0, st>>>(x, hb, l.n_po, l.ls);
        CK(cudaGetLastError());
    }
    k_norm<<<n, 256, 0, st>>>(x, t->fn, hb);
    hidden(t, hb, n, NL, xv, xa);
    if (prof) CK(cudaEventRecord(ev[1], st));
    project(t, n, xv, xa, (bf16*)out);
    if (prof) CK(cudaEventRecord(ev[2], st));
    CK(cudaStreamSynchronize(st));
    if (prof) {
        float a, b;
        CK(cudaEventElapsedTime(&a, ev[0], ev[1]));
        CK(cudaEventElapsedTime(&b, ev[1], ev[2]));
        fprintf(stderr, "[prof] ltx te %d tokens: gemma %.1f ms, projection %.1f ms\n", n, a, b);
        for (auto& e : ev) cudaEventDestroy(e);
    }
    return n;
}

// Projection only (golden check): hidden = device fp32 [49, n, 3840] -> out device bf16 [n, 6144]
void ltx_te_project(void* te, const void* hidden_states, int n, void* out) {
    Te* t = (Te*)te;
    bf16* xv = t->xv.get<bf16>((size_t)n * PK);
    bf16* xa = t->xa.get<bf16>((size_t)n * PK);
    for (int l = 0; l < NHS; l++)
        k_pin<<<n, 256, 0, t->st>>>((const float*)hidden_states + (size_t)l * n * D, l, t->sv, t->sa, xv, xa);
    CK(cudaGetLastError());
    project(t, n, xv, xa, (bf16*)out);
    CK(cudaStreamSynchronize(t->st));
}

// debug tap (device fp32 [49, n, 3840], caller-owned, nullptr = off): the 48 layer inputs and the final norm
void ltx_te_taps(void* te, void* hidden_states) { ((Te*)te)->tap = (float*)hidden_states; }

void ltx_te_free(void* te) {
    Te* t = (Te*)te;
    cudaStreamSynchronize(t->st);
    for (void* p : t->weights) cudaFree(p);
    for (Buf* b : {&t->tmp, &t->ids, &t->x, &t->h, &t->qkv, &t->Q, &t->K, &t->V, &t->att, &t->gu, &t->aq, &t->as, &t->acc,
                   &t->S, &t->css, &t->csg, &t->xv, &t->xa, &t->pv, &t->pa})
        cudaFree(b->p);
    lt_free(t->lt);
    cudaFree(t->ws);
    cublasLtDestroy(t->lh);
    cudaStreamDestroy(t->st);
    delete t;
}
}  // extern "C"
