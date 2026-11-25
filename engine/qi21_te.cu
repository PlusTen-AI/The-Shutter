// shutter engine — Qwen-Image 2.1 text encoder: Qwen3-VL 8B (int8 ConvRot W8A8 LLM + bf16 vision tower), C API
// Numerics follow ComfyUI (comfy/text_encoders/qwen3vl.py, llama.py, qwen35.py): the encoder runs with fp32
// activations and full-precision matmuls (sd1_clip: full_precision_mm), i.e. y = x . W^T with the int8 ConvRot
// weight dequantized to fp32, W = unrotate(q * s). We compute the same product as (x H) . q^T * s: x is rotated in
// fp32 and split into three int8 digits (x H / sc = d0 + d1/254 + d2/254^2, ~fp32 precision), one int8 GEMM over the
// stacked digits, combined in fp32. Token embeddings are int8 rows un-rotated and rounded to bf16 once; norms, RoPE,
// attention and the vision tower (bf16 weights cast to fp32) are fp32.
// Output = last layer hidden state without the final norm, system turn and vision rows dropped.
#include "common.cuh"
#include "st.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <vector>

namespace {
constexpr int D = 4096, NL = 36, NH = 32, HD = 128, NKV = 8, FF = 12288, QKVN = D + 2 * NKV * HD;
constexpr int VD = 1152, VNH = 16, VHD = 72, VFF = 4304, VNB = 27, VMD = 4 * VD, VPK = 1536, VGRID = 48;
constexpr int NDS = 3;
constexpr int DS_AT[NDS] = {8, 16, 24};
constexpr int TOK_IM_START = 151644;
constexpr size_t SCAP = 256u << 20;  // attention score buffer, floats (1 GiB)

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

// regular Hadamard 256 per group (unnormalized), in shared memory; caller scales by 1/16
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
enum { R_PLAIN = 0, R_RMS = 1, R_SWIGLU = 2 };
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
    } else if (MODE == R_SWIGLU) {
        for (int j = threadIdx.x; j < K; j += blockDim.x) {
            const float g = x[j], u = x[K + j];
            buf[j] = g / (1.f + expf(-g)) * u;
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

// digits of row m are acc rows m, Mp + m, 2 Mp + m: out = (a0 + (a1 + a2 / 254) / 254) * (xs[m] * ws[n]), or out += that
__global__ void k_deq(const int32_t* __restrict__ acc, int M, int Mp, int N, const float* __restrict__ xs,
                      const float* __restrict__ ws, float* __restrict__ out, int add) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * N) return;
    const int m = (int)(i / N), n = (int)(i % N);
    const size_t pl = (size_t)Mp * N;
    const float a = (float)acc[i] + ((float)acc[pl + i] + (float)acc[2 * pl + i] * (1.f / 254.f)) * (1.f / 254.f);
    const float v = a * (xs[m] * ws[n]);
    out[i] = add ? out[i] + v : v;
}

// rotated int8 embedding rows -> un-rotate -> bf16 -> fp32 (ids < 0: vision rows, left alone)
__global__ void __launch_bounds__(256) k_embed(const int8_t* __restrict__ tab, const float* __restrict__ sc,
                                               const int* __restrict__ ids, float* __restrict__ x) {
    __shared__ float buf[D];
    const int id = ids[blockIdx.x];
    if (id < 0) return;
    const int8_t* r = tab + (size_t)id * D;
    const float s = sc[id];
    for (int j = threadIdx.x; j < D; j += blockDim.x) buf[j] = (float)r[j] * s;
    __syncthreads();
    hadamard(buf, D);
    for (int j = threadIdx.x; j < D; j += blockDim.x)
        x[(size_t)blockIdx.x * D + j] = __bfloat162float(__float2bfloat16(buf[j] * 0.0625f));
}

// one warp per (row, head 0..39): q heads then k heads in qkv [M, 6144]; per-head RMSNorm, split-half RoPE, in place
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

// GQA: K/V heads repeated to 32 heads, [M, 4096] each
__global__ void k_expand_kv(const float* __restrict__ qkv, int M, float* __restrict__ K, float* __restrict__ V) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * D) return;
    const int m = (int)(i / D), c = (int)(i % D);
    const int src = (c / HD / (NH / NKV)) * HD + c % HD;
    K[i] = qkv[(size_t)m * QKVN + D + src];
    V[i] = qkv[(size_t)m * QKVN + D + NKV * HD + src];
}

// row softmax over L columns (scores already scaled); causal: row i sees keys 0..i
__global__ void __launch_bounds__(256) k_softmax(float* __restrict__ S, int L, int causal) {
    __shared__ float red[32];
    float* s = S + (size_t)blockIdx.x * L;
    const int n = causal ? (int)(blockIdx.x % L) + 1 : L;
    float mx = -INFINITY;
    for (int j = threadIdx.x; j < n; j += blockDim.x) mx = fmaxf(mx, s[j]);
    mx = block_reduce<true>(mx, red);
    float sum = 0.f;
    for (int j = threadIdx.x; j < n; j += blockDim.x) { const float e = expf(s[j] - mx); s[j] = e; sum += e; }
    sum = block_reduce<false>(sum, red);
    const float inv = 1.f / sum;
    for (int j = threadIdx.x; j < L; j += blockDim.x) s[j] = j < n ? s[j] * inv : 0.f;
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

__global__ void k_neq(const float* __restrict__ a, const float* __restrict__ b, size_t n, int* __restrict__ flag) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n && __float_as_uint(a[i]) != __float_as_uint(b[i])) *flag = 1;
}

__global__ void k_bf16_to_f32(const bf16* __restrict__ in, float* __restrict__ out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

__global__ void k_rows_out(const float* __restrict__ x, const int* __restrict__ rows, int n, bf16* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)n * D) return;
    out[i] = __float2bfloat16(x[(size_t)rows[i / D] * D + i % D]);
}

// fp32 -> tf32 (round to nearest, ties away), kept in fp32: products of two such values are exact in fp32
__device__ __forceinline__ float tf32(float v) { return __uint_as_float((__float_as_uint(v) + 0x1000u) & ~0x1fffu); }
__global__ void k_tf32(float* __restrict__ x, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = tf32(x[i]);
}

// image [H, W, 3] in [0, 1] -> normalized patches [Np, 3*2*16*16] in merge order (block row, block col, 2x2), tf32
// (ComfyUI's patch embed is a cuDNN Conv3d, which runs in TF32)
__global__ void k_patch(const float* __restrict__ img, int H, int W, float* __restrict__ P) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int gw = W / 16, Np = (H / 16) * gw;
    if (i >= (size_t)Np * VPK) return;
    const int p = (int)(i / VPK), e = (int)(i % VPK);
    const int mw = p & 1, mh = (p >> 1) & 1, blk = p >> 2, bw = blk % (gw / 2), bh = blk / (gw / 2);
    const int row = bh * 2 + mh, col = bw * 2 + mw;
    const int c = e / 512, py = (e / 16) % 16, px = e % 16;  // (c, t, py, px); both t copies are the same frame
    P[i] = tf32((img[((size_t)(row * 16 + py) * W + col * 16 + px) * 3 + c] - 0.5f) / 0.5f);
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

struct Layer {
    int8_t *wqkv, *wo, *wgu, *wd;
    float *sqkv, *so, *sgu, *sd, *ln1, *ln2, *qn, *kn;
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
    Buf tmp, x, qkv, K, V, att, gu, aq, as, acc, rows, cs, S;           // LLM
    Buf vp, vx, vh, vqkv, vatt, vf, vcs, vidx, vw, vout, vds;            // vision
    Buf vimg, vflag;           // images of the last vision run (vout/vds are reused when the next call has the same)
    std::vector<int> vhw;
    float *tap_embed = nullptr, *tap_layers = nullptr, *tap_vblocks = nullptr, *tap_vmerged = nullptr;
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

void* up(Te* t, const std::string& name, const char* dtype, size_t elems) {
    const StTensor& s = tensor(t, name, dtype, elems);
    void* p = dmalloc(t, s.bytes());
    CK(cudaMemcpy(p, t->f.ptr(s), s.bytes(), cudaMemcpyHostToDevice));
    return p;
}

void up_into(Te* t, const std::string& name, const char* dtype, size_t elems, void* dst) {
    const StTensor& s = tensor(t, name, dtype, elems);
    CK(cudaMemcpy(dst, t->f.ptr(s), s.bytes(), cudaMemcpyHostToDevice));
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

void check_convrot(Te* t, const std::string& layer) {
    const StTensor& s = t->f.get(layer + ".comfy_quant");
    std::string js((const char*)t->f.ptr(s), s.bytes());
    if (js.find("int8_tensorwise") == std::string::npos || js.find("\"convrot\": true") == std::string::npos ||
        js.find("256") == std::string::npos)
        DIE("%s: unexpected quant config %s", layer.c_str(), js.c_str());
}

// row-major cuBLASLt GEMM, heuristic's first algo (deterministic): C[M,N] = alpha * A[M,K] . (bT ? B[N,K]^T : B[K,N])
// + beta * C, strided batches. i8: int8 inputs, int32 out (exact); else fp32 (no TF32).
void mm(Te* t, bool i8, int M, int N, int K, const void* alpha, const void* A, int lda, long long sa, const void* B,
        int ldb, long long sb, bool bT, const void* beta, void* C, int ldc, long long sc, int batch) {
    const cudaDataType_t ab = i8 ? CUDA_R_8I : CUDA_R_32F, ct = i8 ? CUDA_R_32I : CUDA_R_32F;
    cublasLtMatmulDesc_t op;
    CKL(cublasLtMatmulDescCreate(&op, i8 ? CUBLAS_COMPUTE_32I : CUBLAS_COMPUTE_32F, ct));
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
    if (n == 0) DIE("no cublasLt algo for i8=%d M=%d N=%d K=%d batch=%d", (int)i8, M, N, K, batch);
    CKL(cublasLtMatmul(t->lh, op, alpha, B, la, A, lb, beta, C, lc, C, lc, &res.algo, t->ws, t->wsz, t->st));
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(la);
    cublasLtMatrixLayoutDestroy(lb);
    cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
}

void sgemm(Te* t, int M, int N, int K, float alpha, const float* A, int lda, long long sa, const float* B, int ldb,
           long long sb, bool bT, float beta, float* C, int ldc, long long sc, int batch = 1) {
    mm(t, false, M, N, K, &alpha, A, lda, sa, B, ldb, sb, bT, &beta, C, ldc, sc, batch);
}

// y = x . W^T + b (fp32), optional activation, or dst += x . W^T + b
void linear(Te* t, int M, int N, int K, const float* x, const float* W, const float* b, float* y, int act = 0,
            float* dst = nullptr) {
    sgemm(t, M, N, K, 1.f, x, K, 0, W, K, 0, true, 0.f, y, N, 0);
    k_bias<<<nb((size_t)M * N, 256), 256, 0, t->st>>>(y, b, (size_t)M * N, N, act, dst);
    CK(cudaGetLastError());
}

// fp32 attention (SDPA): heads x hd, rows of Q/K/V/O at the given strides, head h at column h*hd
void attention32(Te* t, const float* Q, int ldq, const float* K, int ldk, const float* V, int ldv, float* O, int ldo,
                 int L, int heads, int hd, bool causal) {
    const size_t per = (size_t)L * L;
    const int hc = (int)std::max<size_t>(1, std::min<size_t>(heads, SCAP / per));
    float* S = t->S.get<float>(per * hc);
    for (int h0 = 0; h0 < heads; h0 += hc) {
        const int n = std::min(hc, heads - h0);
        sgemm(t, L, L, hd, 1.f / std::sqrt((float)hd), Q + h0 * hd, ldq, hd, K + h0 * hd, ldk, hd, true, 0.f, S, L, per, n);
        k_softmax<<<n * L, 256, 0, t->st>>>(S, L, causal);
        sgemm(t, L, hd, L, 1.f, S, L, per, V + h0 * hd, ldv, hd, false, 0.f, O + h0 * hd, ldo, hd, n);
    }
}

void rq(Te* t, int mode, const float* in, int ld, int M, int K, const float* w) {
    static bool init = false;
    if (!init) {
        CK(cudaFuncSetAttribute(k_rq<R_PLAIN>, cudaFuncAttributeMaxDynamicSharedMemorySize, FF * 4));
        CK(cudaFuncSetAttribute(k_rq<R_RMS>, cudaFuncAttributeMaxDynamicSharedMemorySize, FF * 4));
        CK(cudaFuncSetAttribute(k_rq<R_SWIGLU>, cudaFuncAttributeMaxDynamicSharedMemorySize, FF * 4));
        init = true;
    }
    const size_t plane = (size_t)((M + 63) & ~63) * K;
    int8_t* q = t->aq.get<int8_t>((size_t)((M + 63) & ~63) * FF * 3);
    float* qs = t->as.get<float>(M);
    const size_t smem = (size_t)K * 4;
    if (mode == R_RMS) k_rq<R_RMS><<<M, 256, smem, t->st>>>(in, ld, K, w, q, plane, qs);
    else if (mode == R_SWIGLU) k_rq<R_SWIGLU><<<M, 256, smem, t->st>>>(in, ld, K, w, q, plane, qs);
    else k_rq<R_PLAIN><<<M, 256, smem, t->st>>>(in, ld, K, w, q, plane, qs);
    CK(cudaGetLastError());
}

// linear on the rows just split by rq(): out = x . W^T or out += x . W^T (fp32).
// Each digit plane has M rounded up to 64 rows (spare rows ignored), so cuBLASLt plans are shared across lengths.
void lin8(Te* t, int M, int N, int K, const int8_t* w, const float* ws, float* out, bool add) {
    const int Mp = (M + 63) & ~63;
    int32_t* acc = t->acc.get<int32_t>((size_t)Mp * 2 * FF * 3);
    const int32_t one = 1, zero = 0;
    mm(t, true, 3 * Mp, N, K, &one, t->aq.p, K, 0, w, K, 0, true, &zero, acc, N, 0, 1);
    k_deq<<<nb((size_t)M * N, 256), 256, 0, t->st>>>(acc, M, Mp, N, (const float*)t->as.p, ws, out, add ? 1 : 0);
    CK(cudaGetLastError());
}

// torch.linspace(0, n_grid - 1, steps) on CUDA, float
float linspace_at(int steps, int i) {
    if (steps == 1) return 0.f;
    const float end = (float)(VGRID - 1);
    const float step = end / (float)(steps - 1);
    return i < steps / 2 ? step * (float)i : end - step * (float)(steps - i - 1);
}

// Vision tower for one image [H, W, 3] (device fp32, H and W multiples of 32): merged [Np/4, 4096] and the three
// DeepStack features (dsout + k*ds_stride) for the LLM.
void vision(Te* t, const float* img, int H, int W, float* merged, float* dsout, size_t ds_stride) {
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
    k_patch<<<nb((size_t)Np * VPK, 256), 256, 0, st>>>(img, H, W, P);
    linear(t, Np, VD, VPK, P, t->pew, t->peb, x);
    k_posemb<<<nb((size_t)Np * VD, 256), 256, 0, st>>>(t->pos, didx, dwt, Np, x);
    CK(cudaGetLastError());
    int dsi = 0;
    for (int i = 0; i < VNB; i++) {
        const VBlock& b = t->vb[i];
        k_ln<<<Np, 256, 0, st>>>(x, VD, b.n1w, b.n1b, h);
        linear(t, Np, 3 * VD, VD, h, b.qkvw, b.qkvb, qkv);
        k_vrope<<<nb((size_t)Np * 2 * VNH * 32, 256), 256, 0, st>>>(qkv, Np, dcs);
        attention32(t, qkv, 3 * VD, qkv + VD, 3 * VD, qkv + 2 * VD, 3 * VD, att, VD, Np, VNH, VHD, false);
        linear(t, Np, VD, VD, att, b.pw, b.pb, h, 0, x);
        k_ln<<<Np, 256, 0, st>>>(x, VD, b.n2w, b.n2b, h);
        linear(t, Np, VFF, VD, h, b.f1w, b.f1b, f, 1);
        linear(t, Np, VD, VFF, f, b.f2w, b.f2b, h, 0, x);
        if (t->tap_vblocks) CK(cudaMemcpyAsync(t->tap_vblocks + (size_t)i * Np * VD, x, (size_t)Np * VD * 4, cudaMemcpyDeviceToDevice, st));
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
    if (t->tap_vmerged) CK(cudaMemcpyAsync(t->tap_vmerged, merged, (size_t)Nm * D * 4, cudaMemcpyDeviceToDevice, st));
}
}  // namespace

extern "C" {

void* qi21_te_new(const char* path) {
    Te* t = new Te();
    if (!t->f.open(path)) DIE("cannot open %s", path);
    CK(cudaStreamCreate(&t->st));   // blocking on purpose (common.cuh)
    CKL(cublasLtCreate(&t->lh));
    CK(cudaMalloc(&t->ws, t->wsz));
    const std::string M = "model.";
    check_convrot(t, M + "embed_tokens");
    const size_t V = t->f.get(M + "embed_tokens.weight").shape[0];
    t->emb = (int8_t*)up(t, M + "embed_tokens.weight", "I8", V * D);
    t->embs = (float*)up(t, M + "embed_tokens.weight_scale", "F32", V);
    for (int i = 0; i < NL; i++) {
        const std::string p = M + "layers." + std::to_string(i) + ".";
        Layer& l = t->l[i];
        for (const char* n : {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj", "self_attn.o_proj", "mlp.gate_proj",
                              "mlp.up_proj", "mlp.down_proj"})
            check_convrot(t, p + n);
        l.wqkv = (int8_t*)dmalloc(t, (size_t)QKVN * D);
        l.sqkv = (float*)dmalloc(t, (size_t)QKVN * 4);
        const char* qkv[3] = {"self_attn.q_proj", "self_attn.k_proj", "self_attn.v_proj"};
        const int rows[3] = {D, NKV * HD, NKV * HD}, off[3] = {0, D, D + NKV * HD};
        for (int j = 0; j < 3; j++) {
            up_into(t, p + qkv[j] + ".weight", "I8", (size_t)rows[j] * D, l.wqkv + (size_t)off[j] * D);
            up_into(t, p + qkv[j] + ".weight_scale", "F32", rows[j], l.sqkv + off[j]);
        }
        l.wgu = (int8_t*)dmalloc(t, (size_t)2 * FF * D);
        l.sgu = (float*)dmalloc(t, (size_t)2 * FF * 4);
        up_into(t, p + "mlp.gate_proj.weight", "I8", (size_t)FF * D, l.wgu);
        up_into(t, p + "mlp.up_proj.weight", "I8", (size_t)FF * D, l.wgu + (size_t)FF * D);
        up_into(t, p + "mlp.gate_proj.weight_scale", "F32", FF, l.sgu);
        up_into(t, p + "mlp.up_proj.weight_scale", "F32", FF, l.sgu + FF);
        l.wo = (int8_t*)up(t, p + "self_attn.o_proj.weight", "I8", (size_t)D * D);
        l.so = (float*)up(t, p + "self_attn.o_proj.weight_scale", "F32", D);
        l.wd = (int8_t*)up(t, p + "mlp.down_proj.weight", "I8", (size_t)D * FF);
        l.sd = (float*)up(t, p + "mlp.down_proj.weight_scale", "F32", D);
        l.ln1 = upf(t, p + "input_layernorm.weight", D);
        l.ln2 = upf(t, p + "post_attention_layernorm.weight", D);
        l.qn = upf(t, p + "self_attn.q_norm.weight", HD);
        l.kn = upf(t, p + "self_attn.k_norm.weight", HD);
    }
    const std::string v = M + "visual.";
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

// Encode one prompt. ids: host int32 [n] from the tokenizer, -1 = one image (expanded here to (H/32)*(W/32) vision
// tokens). images: device fp32 [H, W, 3] in [0, 1] (RGB, already at its final size, H and W multiples of 32),
// img_hw = {H0, W0, H1, W1, ...}. Writes out (device bf16 [rows, 4096]): the tokens from the second <|im_start|> on,
// vision rows dropped; slots[k] (host) = row where image k goes. Returns rows, or -1 if rows > out_cap.
int qi21_te_encode(void* h, const int* ids, int n, int nimg, const void* const* images, const int* img_hw, void* out,
                   int out_cap, int* slots) {
    Te* t = (Te*)h;
    cudaStream_t st = t->st;
    // vision tokens per image
    std::vector<int> nm(nimg), voff(nimg + 1, 0);
    for (int k = 0; k < nimg; k++) {
        const int H = img_hw[2 * k], W = img_hw[2 * k + 1];
        if (H % 32 || W % 32 || H * W < 3136 || (long long)H * W > 12845056) DIE("image %d: %dx%d not supported", k, H, W);
        nm[k] = (H / 32) * (W / 32);
        voff[k + 1] = voff[k] + nm[k];
    }
    // expanded sequence: row ids (-1 = vision), image spans, positions (MRoPE as qwen2vl_mrope_position_ids)
    std::vector<int> rid, start(nimg);
    for (int i = 0, k = 0; i < n; i++) {
        if (ids[i] < 0) {
            if (k >= nimg) DIE("more image slots than images");
            start[k] = (int)rid.size();
            rid.insert(rid.end(), nm[k], -1);
            k++;
        } else rid.push_back(ids[i]);
    }
    const int L = (int)rid.size();
    std::vector<float> pos(3 * (size_t)L);
    for (int r = 0; r < L; r++) pos[r] = pos[L + r] = pos[2 * L + r] = (float)r;
    float offset = 0.f;
    for (int k = 0; k < nimg; k++) {
        const int s = start[k], e = s + nm[k], gh = img_hw[2 * k] / 16, gw = img_hw[2 * k + 1] / 16;
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
    // kept rows: from the second <|im_start|>, without vision rows
    int first = 0;
    for (int r = 0, c = 0; r < L; r++)
        if (rid[r] == TOK_IM_START && ++c == 2) { first = r; break; }
    std::vector<int> keep;
    for (int r = first, k = 0; r < L; r++) {
        if (k < nimg && r == start[k]) {
            slots[k] = (int)keep.size();
            r += nm[k] - 1;
            k++;
            continue;
        }
        keep.push_back(r);
    }
    const int R = (int)keep.size();
    if (R > out_cap) return -1;

    // vision (skipped when the images are the same as in the previous call, e.g. positive then negative prompt)
    std::vector<int> hw(img_hw, img_hw + 2 * nimg);
    std::vector<size_t> ioff(nimg + 1, 0);
    for (int k = 0; k < nimg; k++) ioff[k + 1] = ioff[k] + (size_t)hw[2 * k] * hw[2 * k + 1] * 3;
    bool same = nimg > 0 && hw == t->vhw;
    if (same) {
        int* flag = t->vflag.get<int>(1);
        CK(cudaMemsetAsync(flag, 0, 4, st));
        for (int k = 0; k < nimg; k++)
            k_neq<<<nb(ioff[k + 1] - ioff[k], 256), 256, 0, st>>>((const float*)images[k], (const float*)t->vimg.p + ioff[k],
                                                               ioff[k + 1] - ioff[k], flag);
        int f = 1;
        CK(cudaMemcpyAsync(&f, flag, 4, cudaMemcpyDeviceToHost, st));
        CK(cudaStreamSynchronize(st));
        same = f == 0;
    }
    float* vout = t->vout.get<float>((size_t)voff[nimg] * D);
    float* vds = t->vds.get<float>((size_t)NDS * voff[nimg] * D);
    if (!same && nimg > 0) {
        t->vhw.clear();
        float* keep_img = t->vimg.get<float>(ioff[nimg]);
        for (int k = 0; k < nimg; k++) {
            vision(t, (const float*)images[k], hw[2 * k], hw[2 * k + 1], vout + (size_t)voff[k] * D,
                   vds + (size_t)voff[k] * D, (size_t)voff[nimg] * D);
            CK(cudaMemcpyAsync(keep_img + ioff[k], images[k], (ioff[k + 1] - ioff[k]) * 4, cudaMemcpyDeviceToDevice, st));
        }
        t->vhw = hw;
    }

    // LLM
    int* drows = t->rows.get<int>(std::max(L, R));
    float2* dcs = t->cs.get<float2>(cs.size());
    float* x = t->x.get<float>((size_t)L * D);
    float* qkv = t->qkv.get<float>((size_t)L * QKVN);
    float* K = t->K.get<float>((size_t)L * D);
    float* V = t->V.get<float>((size_t)L * D);
    float* att = t->att.get<float>((size_t)L * D);
    float* gu = t->gu.get<float>((size_t)L * 2 * FF);
    CK(cudaMemcpyAsync(drows, rid.data(), (size_t)L * 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(dcs, cs.data(), cs.size() * 8, cudaMemcpyHostToDevice, st));
    k_embed<<<L, 256, 0, st>>>(t->emb, t->embs, drows, x);
    CK(cudaGetLastError());
    for (int k = 0; k < nimg; k++)
        CK(cudaMemcpyAsync(x + (size_t)start[k] * D, vout + (size_t)voff[k] * D, (size_t)nm[k] * D * 4, cudaMemcpyDeviceToDevice, st));
    if (t->tap_embed) CK(cudaMemcpyAsync(t->tap_embed, x, (size_t)L * D * 4, cudaMemcpyDeviceToDevice, st));
    for (int i = 0; i < NL; i++) {
        const Layer& l = t->l[i];
        rq(t, R_RMS, x, D, L, D, l.ln1);
        lin8(t, L, QKVN, D, l.wqkv, l.sqkv, qkv, false);
        k_qkrope<<<nb((size_t)L * (NH + NKV) * 32, 256), 256, 0, st>>>(qkv, L, dcs, l.qn, l.kn);
        k_expand_kv<<<nb((size_t)L * D, 256), 256, 0, st>>>(qkv, L, K, V);
        attention32(t, qkv, QKVN, K, D, V, D, att, D, L, NH, HD, true);
        rq(t, R_PLAIN, att, D, L, D, nullptr);
        lin8(t, L, D, D, l.wo, l.so, x, true);
        rq(t, R_RMS, x, D, L, D, l.ln2);
        lin8(t, L, 2 * FF, D, l.wgu, l.sgu, gu, false);
        rq(t, R_SWIGLU, gu, 2 * FF, L, FF, nullptr);
        lin8(t, L, D, FF, l.wd, l.sd, x, true);
        if (t->tap_layers) CK(cudaMemcpyAsync(t->tap_layers + (size_t)i * L * D, x, (size_t)L * D * 4, cudaMemcpyDeviceToDevice, st));
        if (i < NDS)  // DeepStack: vision features of block DS_AT[i] added at the image rows after layer i
            for (int k = 0; k < nimg; k++)
                k_add<<<nb((size_t)nm[k] * D, 256), 256, 0, st>>>(x + (size_t)start[k] * D,
                    vds + (size_t)i * voff[nimg] * D + (size_t)voff[k] * D, (size_t)nm[k] * D);
    }
    CK(cudaMemcpyAsync(drows, keep.data(), (size_t)R * 4, cudaMemcpyHostToDevice, st));
    k_rows_out<<<nb((size_t)R * D, 256), 256, 0, st>>>(x, drows, R, (bf16*)out);
    CK(cudaGetLastError());
    CK(cudaStreamSynchronize(st));
    return R;
}

// debug taps (device fp32, caller-owned, nullptr = off): embed [L, 4096] (after vision rows are placed),
// layers [36, L, 4096] (each layer's output, before DeepStack), vision blocks [27, Np, 1152], merged [Np/4, 4096]
// (vision taps cover the last image run through the tower)
void qi21_te_taps(void* h, void* embed, void* layers, void* vblocks, void* vmerged) {
    Te* t = (Te*)h;
    t->tap_embed = (float*)embed;
    t->tap_layers = (float*)layers;
    t->tap_vblocks = (float*)vblocks;
    t->tap_vmerged = (float*)vmerged;
}

void qi21_te_free(void* h) {
    Te* t = (Te*)h;
    cudaStreamSynchronize(t->st);
    for (void* p : t->weights) cudaFree(p);
    for (Buf* b : {&t->tmp, &t->x, &t->qkv, &t->K, &t->V, &t->att, &t->gu, &t->aq, &t->as, &t->acc, &t->rows, &t->cs, &t->S,
                   &t->vp, &t->vx, &t->vh, &t->vqkv, &t->vatt, &t->vf, &t->vcs, &t->vidx, &t->vw, &t->vout, &t->vds,
                   &t->vimg, &t->vflag})
        cudaFree(b->p);
    cudaFree(t->ws);
    cublasLtDestroy(t->lh);
    cudaStreamDestroy(t->st);
    delete t;
}
}  // extern "C"
