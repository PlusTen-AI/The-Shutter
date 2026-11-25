// shutter engine — LTX-2.5 22B audio+video DiT and its two text connectors (int8 ConvRot W8A8), C API
// Follows ComfyUI comfy/ldm/lightricks/{av_model,model,embeddings_connector}.py (e9afbccc2) as it runs on GB10 with
// --bf16-unet. Op numerics pinned against the golden dump (golden/test_ltx_dit.py):
//  - int8 linears: comfy_kitchen ConvRot quantizer (Hadamard 256 + row int8, k_rq below = kernels.cu k_rotquant) and the
//    CUTLASS dequant y = bf16(fma(acc * xs, ws, b)) (gemm_i8_epi); the FF input GELU-tanh is fused into the quantizer in
//    fp32 (not rounded to bf16), as kitchen's input_act path does (ff.net.2 outputs still differ in 6 % / 1.8 % of the
//    elements by about one bf16 step: no fp32 GELU formula changes that, it is inside kitchen's fused kernel)
//  - ck.rms_adaln: bf16(fma(x * r, 1 + scale, shift)); plain rms_norm: bf16(x * r); q/k RMSNorm over all heads
//    (eps 1e-5): bf16((x * r) * w)
//  - ck rope split half: out_j = bf16(fma(x1, f[j][0], bf16(x2 * f[j][1]))) per pair (i, i + hd/2)
//  - residuals: addcmul_ = bf16(fma(y, g, x)) (DQ_RES), text cross-attention x + bf16(y * g) (DQ_GATE), the audio
//    modulations and the text-context modulation are eager bf16 ops
//  - bf16 linears (adaLN, patchify, gate logits, output heads) run on cuBLAS: same math, not the same accumulation order
//  - attention: bf16 flash attention (kernels.cu); head dim 64 (audio side) runs zero-padded to 128 with scale 1/8
// Timestep embedding: the caller gives the sinusoidal inputs of the distinct timestep values (U rows); the engine runs
// the eight adaLN-single modules on those rows and indexes them per video frame / audio token.
#include "common.cuh"
#include "st.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int DV = 4096, DA = 2048, NH = 32, FFV = 16384, FFA = 8192, NB = 48, NCB = 8;
// CTX: the text context length for prompts up to 1024 tokens; longer prompts get ceil(n / NREG) * NREG rows (ComfyUI
// Embeddings1DConnector: registers tiled to max(1024, n)), up to CTXMAX.
constexpr int CTX = 1024, CTXMAX = 4096, NREG = 128, TIN = 256, PIN = 128, HP = 128, CH = 4096, MAXK = 16384;
constexpr float EPS = 1e-6f, QK_EPS = 1e-5f;
const std::string PFX = "model.diffusion_model.";

struct Lin {   // int8 ConvRot linear (parts concatenated along N)
    int8_t* w = nullptr;
    float *s = nullptr, *b = nullptr;   // weight scale [N], bias fp32 [N] (bf16 values) or null
    bf16* bh = nullptr;                 // bias bf16 (DQ_GATE)
    int N = 0, K = 0;
};
struct LinH {  // bf16 linear
    bf16* w = nullptr;
    float* b = nullptr;
    int N = 0, K = 0;
};
struct Att {   // q holds [q | k | v] for self-attention (kv unused)
    Lin q, kv, o;
    bf16 *nq = nullptr, *nk = nullptr;
    LinH g;
};
struct Blk {
    Att a1, a2, aa1, aa2, a2v, v2a;
    Lin f1, f2, af1, af2;
};
struct Conn {
    Att a[NCB];
    Lin f1[NCB], f2[NCB];
    bf16* reg = nullptr;
    int C = 0, hd = 0, ff = 0;
};
struct Ada {
    LinH l1, l2, lin;
    int D = 0, coef = 0;
};
enum { A_V = 0, A_VP, A_A, A_AP, A_CAV, A_GAV, A_CAA, A_GAA, NADA };   // adaLN modules (Ada list order)

// taps (block 0 / connector debug): device destinations set by the caller
enum {
    T_V_N1 = 0, T_V_Q1, T_V_QR1, T_V_KR1, T_V_ATT1, T_V_GL1, T_V_O1, T_V_X1, T_V_N2, T_V_CTX2, T_V_ATT2, T_V_O2, T_V_X2,
    T_A_N1, T_A_QR1, T_A_ATT1, T_A_O1, T_A_X1, T_A_N2, T_A_ATT2, T_A_X2,
    T_V_NA2V, T_A_CA2V, T_A2V_Q, T_A2V_K, T_A2V_ATT, T_V_X3, T_A_NV2A, T_V2A_ATT, T_A_X3,
    T_V_NFF, T_V_FF1, T_A_NFF, T_A_FF1, NTAP
};

struct Dit {
    cudaStream_t st;
    Lt* lt;
    StFile f;
    Blk b[NB];
    Conn cv, ca;
    Ada ada[NADA];
    LinH pv, pa, ov, oa;   // patchify / output heads
    bf16* kfe = nullptr;   // keyframe embedding [DV]
    // per-block tables (bf16 values): [NB][...]
    bf16 *sst = nullptr, *asst = nullptr, *psst = nullptr, *apsst = nullptr, *cav4 = nullptr, *cavg = nullptr,
         *caa4 = nullptr, *caag = nullptr, *fsst = nullptr, *afsst = nullptr;
    float* ones = nullptr;
    int attn_q8 = 0, used_q8 = 0;   // lossy switch: SageAttention2-style attention for the video self-attention
    int qnoise = 0;                 // debug: +-1 flips per 1000 int8 activations (noise floor)
    uint32_t qcall = 0;
    // job (prepare)
    int nv = 0, ppf = 0, na = 0, F = 0, nctx = CTX;                           // nctx: text context rows (CTX..CTXMAX)
    bf16 *vctx = nullptr, *actx = nullptr;                                     // [nctx, DV], [nctx, DA]
    bf16 *vcs = nullptr, *vsn = nullptr, *acs = nullptr, *asn = nullptr, *xvcs = nullptr, *xvsn = nullptr, *xacs = nullptr,
         *xasn = nullptr;                                                      // rope cos / sin tables
    // forward conditioning
    int U = 0, ucap = 0, u_ga = 0, u_gv = 0, u_pv = 0, u_pa = 0;
    bf16 *sinus = nullptr, *t1 = nullptr, *t2 = nullptr, *aout[NADA] = {}, *vemb = nullptr, *aemb = nullptr;
    bf16 *vmod = nullptr, *vcam = nullptr, *amod = nullptr, *acam = nullptr, *vkv = nullptr, *akv = nullptr, *vfin = nullptr,
         *afin = nullptr;
    float *vg = nullptr, *ag = nullptr, *vga = nullptr, *agv = nullptr;
    int *vrow = nullptr, *arow = nullptr;
    std::vector<int> vr, ar;   // (r0, r1, u) triples of rows with one timestep
    // work
    int cap = 0, acap = 0;
    bf16 *vx = nullptr, *y = nullptr, *att = nullptr, *hn = nullptr, *gl = nullptr, *kv = nullptr, *cm = nullptr, *cx = nullptr;
    int8_t* aq = nullptr;
    float *as = nullptr, *f32 = nullptr;
    bf16 *ax = nullptr, *yA = nullptr, *PA = nullptr, *attA = nullptr, *hnA = nullptr, *glA = nullptr;
    // debug
    void* tap[NTAP] = {};
    bf16 *tap_bv = nullptr, *tap_ba = nullptr;
    int tapping = 0;
};

void* dmalloc(size_t n) {
    void* p;
    CK(cudaMalloc(&p, std::max<size_t>(n, 16)));
    return p;
}

uint8_t* g_stage = nullptr;
constexpr size_t STAGE = 64u << 20;
void h2d(void* dst, const uint8_t* src, size_t n) {
    for (size_t o = 0; o < n; o += STAGE) {
        const size_t c = std::min(STAGE, n - o);
        memcpy(g_stage, src + o, c);
        CK(cudaMemcpy((uint8_t*)dst + o, g_stage, c, cudaMemcpyHostToDevice));
    }
}

const StTensor& tensor(Dit* d, const std::string& name, const char* dtype, size_t elems) {
    const StTensor& t = d->f.get(PFX + name);
    if (t.dtype != dtype) DIE("%s: dtype %s, expected %s", name.c_str(), t.dtype.c_str(), dtype);
    const size_t es = t.dtype == "I8" ? 1 : t.dtype == "F32" ? 4 : 2;
    if (t.bytes() != elems * es) DIE("%s: %llu bytes, expected %zu elems", name.c_str(), (unsigned long long)t.bytes(), elems);
    return t;
}

// BF16 / F32 tensor -> bf16 on the device (F32 tables are cast to bf16 by ComfyUI before use: get_ada_values .to(dtype))
bf16* upload_bf16(Dit* d, const std::string& name, size_t elems, bf16* dst = nullptr) {
    const StTensor& t = d->f.get(PFX + name);
    const uint8_t* p = d->f.ptr(t);
    std::vector<bf16> h(elems);
    if (t.dtype == "BF16" && t.bytes() == elems * 2) memcpy(h.data(), p, elems * 2);
    else if (t.dtype == "F32" && t.bytes() == elems * 4)
        for (size_t i = 0; i < elems; i++) {
            float v;
            memcpy(&v, p + 4 * i, 4);
            h[i] = f2bf(v);
        }
    else DIE("%s: dtype %s / %llu bytes, expected %zu elems", name.c_str(), t.dtype.c_str(), (unsigned long long)t.bytes(), elems);
    bf16* o = dst ? dst : (bf16*)dmalloc(elems * 2);
    CK(cudaMemcpy(o, h.data(), elems * 2, cudaMemcpyHostToDevice));
    return o;
}

// bf16 bias -> fp32 (exact) on the device
float* upload_bias_f32(Dit* d, const std::string& name, size_t n, float* dst, bf16* dsth) {
    const StTensor& t = tensor(d, name, "BF16", n);
    std::vector<float> h(n);
    const bf16* p = (const bf16*)d->f.ptr(t);
    for (size_t i = 0; i < n; i++) h[i] = bf2f(p[i]);
    CK(cudaMemcpy(dst, h.data(), n * 4, cudaMemcpyHostToDevice));
    if (dsth) CK(cudaMemcpy(dsth, p, n * 2, cudaMemcpyHostToDevice));
    return dst;
}

void check_convrot(Dit* d, const std::string& layer) {
    const StTensor& t = d->f.get(PFX + layer + ".comfy_quant");
    std::string js((const char*)d->f.ptr(t), t.bytes());
    if (js.find("int8_tensorwise") == std::string::npos || js.find("\"convrot\": true") == std::string::npos ||
        js.find("256") == std::string::npos)
        DIE("%s: unexpected quant config %s", layer.c_str(), js.c_str());
}

// int8 linear from one or more layers (concatenated output features), K input features
Lin load_lin(Dit* d, const std::vector<std::string>& layers, int K, bool bias) {
    Lin L;
    L.K = K;
    for (const auto& l : layers) L.N += (int)d->f.get(PFX + l + ".weight").shape[0];
    L.w = (int8_t*)dmalloc((size_t)L.N * K);
    L.s = (float*)dmalloc((size_t)L.N * 4);
    if (bias) {
        L.b = (float*)dmalloc((size_t)L.N * 4);
        L.bh = (bf16*)dmalloc((size_t)L.N * 2);
    }
    int n0 = 0;
    for (const auto& l : layers) {
        check_convrot(d, l);
        const int n = (int)d->f.get(PFX + l + ".weight").shape[0];
        const StTensor& w = tensor(d, l + ".weight", "I8", (size_t)n * K);
        h2d(L.w + (size_t)n0 * K, d->f.ptr(w), w.bytes());
        const StTensor& s = tensor(d, l + ".weight_scale", "F32", n);
        CK(cudaMemcpy(L.s + n0, d->f.ptr(s), (size_t)n * 4, cudaMemcpyHostToDevice));
        if (bias) upload_bias_f32(d, l + ".bias", n, L.b + n0, L.bh + n0);
        n0 += n;
    }
    return L;
}

LinH load_linh(Dit* d, const std::string& l, int N, int K) {
    LinH L;
    L.N = N, L.K = K;
    L.w = upload_bf16(d, l + ".weight", (size_t)N * K);
    L.b = (float*)dmalloc((size_t)N * 4);
    upload_bias_f32(d, l + ".bias", N, L.b, nullptr);
    return L;
}

Att load_att(Dit* d, const std::string& p, bool self, int qdim, int cdim, int inner) {
    Att a;
    if (self) a.q = load_lin(d, {p + ".to_q", p + ".to_k", p + ".to_v"}, qdim, true);
    else {
        a.q = load_lin(d, {p + ".to_q"}, qdim, true);
        a.kv = load_lin(d, {p + ".to_k", p + ".to_v"}, cdim, true);
    }
    a.o = load_lin(d, {p + ".to_out.0"}, inner, true);
    a.nq = upload_bf16(d, p + ".q_norm.weight", inner);
    a.nk = upload_bf16(d, p + ".k_norm.weight", inner);
    a.g = load_linh(d, p + ".to_gate_logits", NH, qdim);
    return a;
}

inline int nb(size_t n, int t) { return (int)((n + t - 1) / t); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16(v)); }
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

// torch GeluCUDAKernelImpl tanh in fp32, steps as gemm_dq.cu gelu_t
__device__ __forceinline__ float gelu_t(float y) {
    constexpr float kBeta = (float)(M_SQRT2 * M_2_SQRTPI * 0.5), kKappa = 0.044715f;
    const float in = __fmul_rn(__fmaf_rn(__fmul_rn(y, __fmul_rn(y, y)), kKappa, y), kBeta);
    return __fmul_rn(__fmul_rn(y, 0.5f), __fadd_rn(tanhf(in), 1.f));
}

// ------------------------------------------------------------------------------------------------- kernels
// pre-op -> (optional bf16 copy of the pre-op output) -> ConvRot (Hadamard 256, normalized) -> row-wise int8.
// The rotation / quantizer part is kernels.cu k_rotquant (comfy_kitchen CUDA quantizer, bit for bit).
//   RQ_PLAIN   x
//   RQ_RMS     bf16(x * r)                                     rms_norm (connector)
//   RQ_ADALN   bf16(fma(x * r, 1 + scale, shift))              ck.rms_adaln
//   RQ_RMSMOD  bf16(bf16(bf16(x * r) * bf16(1 + scale)) + shift)   eager audio modulation
//   RQ_GATE    bf16(o * 2 bf16(sigmoid(logit[head])))          per-head gate before to_out (o rows: head stride hs)
//   RQ_GELU    gelu_tanh(x) in fp32                            kitchen input_act before ff.net.2
// scale / shift of row m: mod[mrow[m] * ldm + sco + j] / [.. + sho + j]
enum { RQ_PLAIN = 0, RQ_RMS, RQ_ADALN, RQ_RMSMOD, RQ_GATE, RQ_GELU };
template <int MODE>
__global__ void __launch_bounds__(256) k_rq(const bf16* __restrict__ x, int ldx, int K, const bf16* __restrict__ mod, int ldm,
                                            const int* __restrict__ mrow, int sco, int sho, const bf16* __restrict__ gl, int hd,
                                            int hs, bf16* __restrict__ copy, int8_t* __restrict__ q, float* __restrict__ qs) {
    extern __shared__ float buf[];
    __shared__ float red[32];
    const int row = blockIdx.x;
    const bf16* xr = x + (size_t)row * ldx;
    if (MODE == RQ_RMS || MODE == RQ_ADALN || MODE == RQ_RMSMOD) {
        float s = 0.f;
        for (int j = threadIdx.x; j < K; j += blockDim.x) {
            const float v = __bfloat162float(xr[j]);
            buf[j] = v;
            s += v * v;
        }
        const float r = rsqrtf(block_reduce<false>(s, red) / K + EPS);
        const bf16* mr = MODE == RQ_RMS ? nullptr : mod + (size_t)mrow[row] * ldm;
        for (int j = threadIdx.x; j < K; j += blockDim.x) {
            const float xn = __fmul_rn(buf[j], r);
            float v;
            if (MODE == RQ_RMS) v = rbf(xn);
            else if (MODE == RQ_ADALN)
                v = rbf(__fmaf_rn(xn, __fadd_rn(1.f, __bfloat162float(mr[sco + j])), __bfloat162float(mr[sho + j])));
            else
                v = rbf(__fadd_rn(rbf(__fmul_rn(rbf(xn), rbf(__fadd_rn(1.f, __bfloat162float(mr[sco + j]))))),
                                  __bfloat162float(mr[sho + j])));
            buf[j] = v;
            if (copy) copy[(size_t)row * K + j] = __float2bfloat16(v);
        }
    } else if (MODE == RQ_GATE) {
        for (int j = threadIdx.x; j < K; j += blockDim.x) {
            const int h = j / hd, e = j - h * hd;
            const float lg = __bfloat162float(gl[(size_t)row * NH + h]);
            const float g = 2.f * rbf(1.f / (1.f + expf(-lg)));
            buf[j] = rbf(__fmul_rn(__bfloat162float(xr[h * hs + e]), g));
        }
    } else if (MODE == RQ_GELU) {
        for (int j = threadIdx.x; j < K; j += blockDim.x) buf[j] = gelu_t(__bfloat162float(xr[j]));
    } else {
        for (int j = threadIdx.x; j < K; j += blockDim.x) buf[j] = __bfloat162float(xr[j]);
    }
    __syncthreads();
    // H = h4 (x) h4 (x) h4 (x) h4, h4 = [[1,1,1,-1],[1,1,-1,1],[1,-1,1,1],[-1,1,1,1]]: one radix-4 pass per digit
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
    for (int j = threadIdx.x; j < K; j += blockDim.x) {
        const float v = buf[j] * 0.0625f;
        buf[j] = v;
        mx = fmaxf(mx, fabsf(v));
    }
    mx = block_reduce<true>(mx, red);
    // comfy_kitchen CUDA quantizer: s = amax/127 (fp32); q = rne(bf16(bf16(x) / bf16(s)))
    const float sc = fmaxf(mx / 127.f, 1e-30f);
    const float scb = rbf(sc);
    int8_t* qo = q + (size_t)row * K;
    for (int j = threadIdx.x; j < K; j += blockDim.x) {
        const float v = rintf(rbf(rbf(buf[j]) / scb));
        qo[j] = (int8_t)fminf(fmaxf(v, -128.f), 127.f);
    }
    if (threadIdx.x == 0) qs[row] = sc;
}

// q / k: RMSNorm over all heads (C = 32 hd, weight w, eps 1e-5) -> bf16 -> split-half RoPE per head (cos / sin rows of
// C/2, pair p = head * hd/2 + i rotates dims i and i + hd/2; null = no rope) -> out rows (stride ldo, head stride hs,
// dims hd..hs zeroed). In place is fine (the row is staged in shared memory first).
__global__ void __launch_bounds__(256) k_qk_norm(const bf16* __restrict__ y, int ldy, int C, int hd, const bf16* __restrict__ w,
                                                 const bf16* __restrict__ cs, const bf16* __restrict__ sn, bf16* out, int ldo,
                                                 int hs) {
    extern __shared__ float buf[];
    __shared__ float red[32];
    const int row = blockIdx.x;
    const bf16* yr = y + (size_t)row * ldy;
    float s = 0.f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        const float v = __bfloat162float(yr[j]);
        buf[j] = v;
        s += v * v;
    }
    const float r = rsqrtf(block_reduce<false>(s, red) / C + QK_EPS);
    for (int j = threadIdx.x; j < C; j += blockDim.x) buf[j] = rbf(__fmul_rn(__fmul_rn(buf[j], r), __bfloat162float(w[j])));
    __syncthreads();
    const int half = hd >> 1;
    bf16* o = out + (size_t)row * ldo;
    for (int p = threadIdx.x; p < C / 2; p += blockDim.x) {
        const int h = p / half, i = p - h * half;
        float n1 = buf[h * hd + i], n2 = buf[h * hd + i + half];
        if (cs) {
            const float c = __bfloat162float(cs[(size_t)row * (C / 2) + p]), sv = __bfloat162float(sn[(size_t)row * (C / 2) + p]);
            const float o1 = rbf(__fmaf_rn(n1, c, -rbf(__fmul_rn(n2, sv))));
            const float o2 = rbf(__fmaf_rn(n1, sv, rbf(__fmul_rn(n2, c))));
            n1 = o1, n2 = o2;
        }
        o[h * hs + i] = __float2bfloat16(n1);
        o[h * hs + i + half] = __float2bfloat16(n2);
    }
    if (hs > hd)
        for (int j = threadIdx.x; j < NH * (hs - hd); j += blockDim.x) {
            const int h = j / (hs - hd);
            o[h * hs + hd + (j - h * (hs - hd))] = __float2bfloat16(0.f);
        }
}

// v rows [M, 32 hd] (ld) -> head-padded rows (stride ldo, head stride hs)
__global__ void k_pad(const bf16* __restrict__ v, int ld, int M, int hd, bf16* __restrict__ out, int ldo, int hs) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * NH * hs) return;
    const int m = (int)(i / (NH * hs)), c = (int)(i % (NH * hs)), h = c / hs, e = c - h * hs;
    out[(size_t)m * ldo + c] = e < hd ? v[(size_t)m * ld + h * hd + e] : __float2bfloat16(0.f);
}

__global__ void k_bias_round(const float* __restrict__ in, const float* __restrict__ b, int M, int N, bf16* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)M * N) out[i] = __float2bfloat16(__fadd_rn(in[i], b[i % N]));
}

__global__ void k_silu(const bf16* __restrict__ in, bf16* __restrict__ out, int n) {   // torch silu on bf16
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float x = __bfloat162float(in[i]);
    out[i] = __float2bfloat16(x / (1.f + expf(-x)));
}

// mod tables: out[b][u][e] = bf16(tab[b][e] + src[uu][e % se]), uu = sel < 0 ? u : sel (all bf16 values)
__global__ void k_mod(const bf16* __restrict__ tab, int nblk, int E, const bf16* __restrict__ src, int se, int U, int sel,
                      bf16* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)nblk * U * E) return;
    const int e = (int)(i % E), u = (int)((i / E) % U), b = (int)(i / ((size_t)E * U));
    const int uu = sel < 0 ? u : sel;
    out[i] = __float2bfloat16(__fadd_rn(__bfloat162float(tab[(size_t)b * E + e]), __bfloat162float(src[(size_t)uu * se + e % se])));
}

// fp32 copies of gate rows: out[b][u][g][j] = mod[b][u][row_g * D + j], rows r0, r1
__global__ void k_gates(const bf16* __restrict__ mod, int nblk, int U, int E, int D, int r0, int r1, float* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)nblk * U * 2 * D) return;
    const int j = (int)(i % D), g = (int)((i / D) % 2);
    const size_t bu = i / (2 * (size_t)D);
    out[i] = __bfloat162float(mod[bu * E + (size_t)(g ? r1 : r0) * D + j]);
}
__global__ void k_to_f32(const bf16* __restrict__ in, float* __restrict__ out, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __bfloat162float(in[i]);
}

// text context modulation (eager bf16): out = bf16(bf16(ctx * bf16(1 + scale)) + shift), kvmod = [shift | scale]
__global__ void k_ctx_mod(const bf16* __restrict__ ctx, int L, int C, const bf16* __restrict__ kvmod, bf16* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)L * C) return;
    const int j = (int)(i % C);
    const float opl = rbf(__fadd_rn(1.f, __bfloat162float(kvmod[C + j])));
    out[i] = __float2bfloat16(__fadd_rn(rbf(__fmul_rn(__bfloat162float(ctx[i]), opl)), __bfloat162float(kvmod[j])));
}

__global__ void k_add_row(bf16* __restrict__ x, const bf16* __restrict__ e, int M, int C) {   // x = bf16(x + e)
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)M * C) x[i] = __float2bfloat16(__fadd_rn(__bfloat162float(x[i]), __bfloat162float(e[i % C])));
}

// connector input: rows < n from x, rows >= n = learnable register (j % 128)
__global__ void k_regs(const bf16* __restrict__ x, int n, const bf16* __restrict__ reg, int L, int C, bf16* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)L * C) return;
    const int r = (int)(i / C), c = (int)(i % C);
    out[i] = r < n ? x[(size_t)r * C + c] : reg[(size_t)(r % NREG) * C + c];
}

__global__ void __launch_bounds__(256) k_rms_out(const bf16* __restrict__ x, int C, bf16* __restrict__ out) {   // rms_norm
    __shared__ float red[32];
    const bf16* xr = x + (size_t)blockIdx.x * C;
    float s = 0.f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        const float v = __bfloat162float(xr[j]);
        s += v * v;
    }
    const float r = rsqrtf(block_reduce<false>(s, red) / C + EPS);
    for (int j = threadIdx.x; j < C; j += blockDim.x)
        out[(size_t)blockIdx.x * C + j] = __float2bfloat16(__fmul_rn(__bfloat162float(xr[j]), r));
}

// output head pre-op: LayerNorm (no affine, eps 1e-6) -> bf16, then eager bf16(bf16(n * bf16(1 + scale)) + shift),
// fin row mrow[m] = [shift | scale]
__global__ void __launch_bounds__(256) k_final(const bf16* __restrict__ x, int C, const bf16* __restrict__ fin,
                                               const int* __restrict__ mrow, bf16* __restrict__ out) {
    extern __shared__ float buf[];
    __shared__ float red[32];
    const int row = blockIdx.x;
    const bf16* xr = x + (size_t)row * C;
    float s = 0.f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        const float v = __bfloat162float(xr[j]);
        buf[j] = v;
        s += v;
    }
    const float mean = block_reduce<false>(s, red) / C;
    float s2 = 0.f;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        const float dd = buf[j] - mean;
        s2 += dd * dd;
    }
    const float rstd = rsqrtf(block_reduce<false>(s2, red) / C + EPS);
    const bf16* fr = fin + (size_t)mrow[row] * 2 * C;
    for (int j = threadIdx.x; j < C; j += blockDim.x) {
        const float n = rbf(__fmul_rn(__fsub_rn(buf[j], mean), rstd));
        const float opl = rbf(__fadd_rn(1.f, __bfloat162float(fr[C + j])));
        out[(size_t)row * C + j] = __float2bfloat16(__fadd_rn(rbf(__fmul_rn(n, opl)), __bfloat162float(fr[j])));
    }
}

// debug: flip int8 activations by +-1 at `rate` per 1000 (hash of index and call), to measure the noise floor
__global__ void k_qnoise(int8_t* q, size_t n, int rate, uint32_t call) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    uint32_t h = (uint32_t)i * 0x9E3779B1u ^ call * 0x85EBCA77u;
    h ^= h >> 15; h *= 0x2C1B3C6Du; h ^= h >> 12; h *= 0x297A2D39u; h ^= h >> 15;
    if ((int)(h % 1000u) < rate) q[i] = (int8_t)max(-128, min(127, (int)q[i] + ((h >> 16) & 1 ? 1 : -1)));
}

// ------------------------------------------------------------------------------------------------- profiling
const char* PH_NAME[] = {"temb/embed/final", "video self", "video text", "audio self+text", "a2v+v2a", "video ff", "audio ff"};
constexpr int NPH = 7;
bool g_prof = getenv("SHUTTER_PROF") && *getenv("SHUTTER_PROF");
std::vector<cudaEvent_t> g_ev;
std::vector<int> g_ph;
double g_ms[NPH] = {};
int g_prof_n = 0;
void mark(Dit* d, int ph) {
    if (!g_prof) return;
    if (g_ph.size() == g_ev.size()) {
        cudaEvent_t e;
        CK(cudaEventCreate(&e));
        g_ev.push_back(e);
    }
    CK(cudaEventRecord(g_ev[g_ph.size()], d->st));
    g_ph.push_back(ph);
}
void prof_end(Dit* d) {
    if (!g_prof) return;
    mark(d, 0);
    CK(cudaStreamSynchronize(d->st));
    for (size_t i = 0; i + 1 < g_ph.size(); i++) {
        float ms;
        CK(cudaEventElapsedTime(&ms, g_ev[i], g_ev[i + 1]));
        g_ms[g_ph[i]] += ms;
    }
    g_ph.clear();
    static int last_nv = -1;   // averages per job size
    if (d->nv != last_nv) {
        for (double& m : g_ms) m = 0;
        g_prof_n = 0, last_nv = d->nv;
        return;   // first call of a size picks the GEMM tiles: not counted
    }
    g_prof_n++;
    double tot = 0;
    for (int k = 0; k < NPH; k++) tot += g_ms[k];
    fprintf(stderr, "[prof] per forward (nv=%d na=%d, avg of %d): total %.1f ms\n", d->nv, d->na, g_prof_n, tot / g_prof_n);
    for (int k = 0; k < NPH; k++) fprintf(stderr, "[prof]   %-18s %9.2f ms  %4.1f%%\n", PH_NAME[k], g_ms[k] / g_prof_n, 100 * g_ms[k] / tot);
}

// ------------------------------------------------------------------------------------------------- pieces
void rq_init() {
    static bool init = false;
    if (init) return;
    CK(cudaFuncSetAttribute(k_rq<RQ_PLAIN>, cudaFuncAttributeMaxDynamicSharedMemorySize, MAXK * 4));
    CK(cudaFuncSetAttribute(k_rq<RQ_RMS>, cudaFuncAttributeMaxDynamicSharedMemorySize, MAXK * 4));
    CK(cudaFuncSetAttribute(k_rq<RQ_ADALN>, cudaFuncAttributeMaxDynamicSharedMemorySize, MAXK * 4));
    CK(cudaFuncSetAttribute(k_rq<RQ_RMSMOD>, cudaFuncAttributeMaxDynamicSharedMemorySize, MAXK * 4));
    CK(cudaFuncSetAttribute(k_rq<RQ_GATE>, cudaFuncAttributeMaxDynamicSharedMemorySize, MAXK * 4));
    CK(cudaFuncSetAttribute(k_rq<RQ_GELU>, cudaFuncAttributeMaxDynamicSharedMemorySize, MAXK * 4));
    init = true;
}

struct RqIn {
    const bf16* x;
    int ldx;
    const bf16* mod = nullptr;
    int ldm = 0;
    const int* mrow = nullptr;
    int sco = 0, sho = 0;
    const bf16* gl = nullptr;
    int hd = 0, hs = 0;
    bf16* copy = nullptr;
};
void rq(Dit* d, int mode, int M, int K, const RqIn& a) {
    if (K % 256 || K > MAXK) DIE("rq: K=%d", K);
    if (M <= 0) return;
    rq_init();
    const size_t sm = (size_t)K * 4;
#define RQ_L(MD) k_rq<MD><<<M, 256, sm, d->st>>>(a.x, a.ldx, K, a.mod, a.ldm, a.mrow, a.sco, a.sho, a.gl, a.hd, a.hs, a.copy, d->aq, d->as)
    switch (mode) {
    case RQ_PLAIN: RQ_L(RQ_PLAIN); break;
    case RQ_RMS: RQ_L(RQ_RMS); break;
    case RQ_ADALN: RQ_L(RQ_ADALN); break;
    case RQ_RMSMOD: RQ_L(RQ_RMSMOD); break;
    case RQ_GATE: RQ_L(RQ_GATE); break;
    default: RQ_L(RQ_GELU); break;
    }
#undef RQ_L
    if (d->qnoise) k_qnoise<<<nb((size_t)M * K, 256), 256, 0, d->st>>>(d->aq, (size_t)M * K, d->qnoise, ++d->qcall);
}

// int8 linear on the quantized rows [r0, r0 + m) of d->aq
void lin(Dit* d, int epi, const Lin& L, int r0, int m, bf16* out, const void* aux = nullptr) {
    if (m <= 0) return;
    const void* b = epi == DQ_GATE ? (const void*)L.bh : (const void*)L.b;
    gemm_i8_epi(d->st, epi, m, L.N, L.K, d->aq + (size_t)r0 * L.K, L.w, d->as + r0, L.s, b, aux, out + (size_t)r0 * L.N);
}

// bf16 linear: out [M, N] = bf16(x W^T + b), row chunks bound the fp32 scratch
void linh(Dit* d, const bf16* x, int M, const LinH& L, bf16* out) {
    const int ch = std::max(1, CH * DV / L.N);
    for (int r0 = 0; r0 < M; r0 += ch) {
        const int m = std::min(ch, M - r0);
        gemm_bf16_f32(d->lt, m, L.N, L.K, x + (size_t)r0 * L.K, L.w, d->f32);
        k_bias_round<<<nb((size_t)m * L.N, 256), 256, 0, d->st>>>(d->f32, L.b, m, L.N, out + (size_t)r0 * L.N);
    }
}

void qk_norm(Dit* d, const bf16* y, int ldy, int M, int C, int hd, const bf16* w, const bf16* cs, const bf16* sn, bf16* out,
             int ldo, int hs) {
    if (M <= 0) return;
    k_qk_norm<<<M, 256, (size_t)C * 4, d->st>>>(y, ldy, C, hd, w, cs, sn, out, ldo, hs);
}
void pad(Dit* d, const bf16* v, int ld, int M, int hd, bf16* out, int ldo) {
    if (M <= 0) return;
    k_pad<<<nb((size_t)M * NH * HP, 256), 256, 0, d->st>>>(v, ld, M, hd, out, ldo, HP);
}
void attn(Dit* d, const bf16* q, const bf16* k, const bf16* v, bf16* o, int Mq, int Nk, int ld, float scale, bool q8 = false) {
    if (Mq <= 0 || Nk <= 0) return;
    if (q8) attention_q8(d->st, q, k, v, o, Mq, Nk, NH, ld, 4096), d->used_q8 = 1;
    else attention_ex(d->st, q, k, v, o, Mq, Nk, nullptr, NH, ld, 4096, scale);
}

void tap(Dit* d, int id, const bf16* src, int rows, int cols, int ld) {
    if (!d->tapping || !d->tap[id] || rows <= 0) return;
    CK(cudaMemcpy2DAsync(d->tap[id], (size_t)cols * 2, src, (size_t)ld * 2, (size_t)cols * 2, rows, cudaMemcpyDeviceToDevice, d->st));
}

// residual GEMMs over row ranges with one timestep: aux row = gate of that range
void lin_res(Dit* d, int epi, const Lin& L, const std::vector<int>& rg, bf16* out, const float* g32, const bf16* g16, size_t ustride) {
    for (size_t k = 0; k + 2 < rg.size(); k += 3) {
        const int r0 = rg[k], r1 = rg[k + 1], u = rg[k + 2];
        if (epi == DQ_RES) lin(d, DQ_RES, L, r0, r1 - r0, out, g32 + (size_t)u * ustride);
        else lin(d, DQ_GATE, L, r0, r1 - r0, out, g16 + (size_t)u * ustride);
    }
}

void ensure(Dit* d, int nv, int na) {
    const int R = std::max(nv, d->nctx);
    if (R > d->cap) {
        void* ps[] = {d->vx, d->y, d->att, d->hn, d->gl, d->aq, d->as, d->vrow};
        for (void* p : ps) cudaFree(p);
        d->cap = R;
        d->vx = (bf16*)dmalloc((size_t)R * DV * 2);
        d->y = (bf16*)dmalloc((size_t)R * FFV * 2);
        d->att = (bf16*)dmalloc((size_t)R * 4096 * 2);
        d->hn = (bf16*)dmalloc((size_t)R * DV * 2);
        d->gl = (bf16*)dmalloc((size_t)R * NH * 2);
        d->aq = (int8_t*)dmalloc((size_t)R * MAXK);
        d->as = (float*)dmalloc((size_t)R * 4);
        d->vrow = (int*)dmalloc((size_t)R * 4);
    }
    const int RA = std::max(na, d->nctx);
    if (RA > d->acap) {
        void* ps[] = {d->ax, d->yA, d->PA, d->attA, d->hnA, d->glA, d->arow};
        for (void* p : ps) cudaFree(p);
        d->acap = RA;
        d->ax = (bf16*)dmalloc((size_t)RA * DA * 2);
        d->yA = (bf16*)dmalloc((size_t)RA * FFA * 2);
        d->PA = (bf16*)dmalloc((size_t)RA * 3 * 4096 * 2);
        d->attA = (bf16*)dmalloc((size_t)RA * 4096 * 2);
        d->hnA = (bf16*)dmalloc((size_t)RA * DA * 2);
        d->glA = (bf16*)dmalloc((size_t)RA * NH * 2);
        d->arow = (int*)dmalloc((size_t)RA * 4);
    }
}

void ensure_u(Dit* d, int U) {
    if (U <= d->ucap) return;
    void* ps[] = {d->sinus, d->t1, d->t2, d->vemb, d->aemb, d->vmod, d->vcam, d->amod, d->acam, d->vfin, d->afin, d->vg, d->ag};
    for (void* p : ps) cudaFree(p);
    for (int k = 0; k < NADA; k++) cudaFree(d->aout[k]);
    d->ucap = U;
    d->sinus = (bf16*)dmalloc((size_t)U * TIN * 2);
    d->t1 = (bf16*)dmalloc((size_t)U * DV * 2);
    d->t2 = (bf16*)dmalloc((size_t)U * DV * 2);
    d->vemb = (bf16*)dmalloc((size_t)U * DV * 2);
    d->aemb = (bf16*)dmalloc((size_t)U * DA * 2);
    for (int k = 0; k < NADA; k++) d->aout[k] = (bf16*)dmalloc((size_t)U * d->ada[k].coef * d->ada[k].D * 2);
    d->vmod = (bf16*)dmalloc((size_t)NB * U * 9 * DV * 2);
    d->vcam = (bf16*)dmalloc((size_t)NB * U * 4 * DV * 2);
    d->amod = (bf16*)dmalloc((size_t)NB * U * 9 * DA * 2);
    d->acam = (bf16*)dmalloc((size_t)NB * U * 4 * DA * 2);
    d->vfin = (bf16*)dmalloc((size_t)U * 2 * DV * 2);
    d->afin = (bf16*)dmalloc((size_t)U * 2 * DA * 2);
    d->vg = (float*)dmalloc((size_t)NB * U * 2 * DV * 4);
    d->ag = (float*)dmalloc((size_t)NB * U * 2 * DA * 4);
}

// adaLN-single on the U sinusoid rows: emb = l2(silu(l1(sinus))), out = lin(silu(emb))
void run_ada(Dit* d, int k, bf16* emb_keep) {
    const Ada& A = d->ada[k];
    const int U = d->U, D = A.D;
    linh(d, d->sinus, U, A.l1, d->t1);
    k_silu<<<nb((size_t)U * D, 256), 256, 0, d->st>>>(d->t1, d->t1, U * D);
    bf16* emb = emb_keep ? emb_keep : d->t2;
    linh(d, d->t1, U, A.l2, emb);
    k_silu<<<nb((size_t)U * D, 256), 256, 0, d->st>>>(emb, d->t1, U * D);
    linh(d, d->t1, U, A.lin, d->aout[k]);
}

void modt(Dit* d, const bf16* tab, int nblk, int E, const bf16* src, int se, int U, int sel, bf16* out) {
    k_mod<<<nb((size_t)nblk * U * E, 256), 256, 0, d->st>>>(tab, nblk, E, src, se, U, sel, out);
}

void conditioning(Dit* d) {
    const int U = d->U;
    run_ada(d, A_V, d->vemb);
    run_ada(d, A_A, d->aemb);
    for (int k : {A_VP, A_AP, A_CAV, A_GAV, A_CAA, A_GAA}) run_ada(d, k, nullptr);
    modt(d, d->sst, NB, 9 * DV, d->aout[A_V], 9 * DV, U, -1, d->vmod);
    modt(d, d->asst, NB, 9 * DA, d->aout[A_A], 9 * DA, U, -1, d->amod);
    modt(d, d->cav4, NB, 4 * DV, d->aout[A_CAV], 4 * DV, U, -1, d->vcam);
    modt(d, d->caa4, NB, 4 * DA, d->aout[A_CAA], 4 * DA, U, -1, d->acam);
    modt(d, d->psst, NB, 2 * DV, d->aout[A_VP], 2 * DV, 1, d->u_pv, d->vkv);
    modt(d, d->apsst, NB, 2 * DA, d->aout[A_AP], 2 * DA, 1, d->u_pa, d->akv);
    // a2v (video side) / v2a (audio side) gates, one timestep each: bf16 rows via d->hn, kept in fp32
    modt(d, d->cavg, NB, DV, d->aout[A_GAV], DV, 1, d->u_ga, d->hn);
    k_to_f32<<<nb((size_t)NB * DV, 256), 256, 0, d->st>>>(d->hn, d->vga, (size_t)NB * DV);
    modt(d, d->caag, NB, DA, d->aout[A_GAA], DA, 1, d->u_gv, d->hn);
    k_to_f32<<<nb((size_t)NB * DA, 256), 256, 0, d->st>>>(d->hn, d->agv, (size_t)NB * DA);
    k_gates<<<nb((size_t)NB * U * 2 * DV, 256), 256, 0, d->st>>>(d->vmod, NB, U, 9 * DV, DV, 2, 5, d->vg);
    k_gates<<<nb((size_t)NB * U * 2 * DA, 256), 256, 0, d->st>>>(d->amod, NB, U, 9 * DA, DA, 2, 5, d->ag);
    modt(d, d->fsst, 1, 2 * DV, d->vemb, DV, U, -1, d->vfin);
    modt(d, d->afsst, 1, 2 * DA, d->aemb, DA, U, -1, d->afin);
}

// one transformer block on d->vx [nv, DV] / d->ax [na, DA]
void block(Dit* d, int i) {
    const Blk& b = d->b[i];
    const int N = d->nv, A = d->na, U = d->U;
    const bf16* vm = d->vmod + (size_t)i * U * 9 * DV;   // [U][9][DV]: shift, scale, gate (msa), shift, scale, gate (mlp), q shift, scale, gate
    const bf16* am = d->amod + (size_t)i * U * 9 * DA;
    const bf16* vca = d->vcam + (size_t)i * U * 4 * DV;  // [U][4][DV]: scale a2v, shift a2v, scale v2a, shift v2a
    const bf16* aca = d->acam + (size_t)i * U * 4 * DA;
    const float* vg = d->vg + (size_t)i * U * 2 * DV;    // [U][2][DV]: msa gate, mlp gate (fp32)
    const float* ag = d->ag + (size_t)i * U * 2 * DA;
    const bool tp = d->tapping;
    bf16 *vx = d->vx, *ax = d->ax, *y = d->y, *att = d->att, *hn = d->hn;

    // ---- video self-attention
    mark(d, 1);
    {
        RqIn a{vx, DV};
        a.mod = vm, a.ldm = 9 * DV, a.mrow = d->vrow, a.sco = DV, a.sho = 0, a.copy = hn;
        rq(d, RQ_ADALN, N, DV, a);
        lin(d, DQ_F32B, b.a1.q, 0, N, y);
        if (tp) tap(d, T_V_N1, hn, N, DV, DV), tap(d, T_V_Q1, y, N, DV, 3 * DV);
        qk_norm(d, y, 3 * DV, N, DV, 128, b.a1.nq, d->vcs, d->vsn, y, 3 * DV, 128);
        qk_norm(d, y + DV, 3 * DV, N, DV, 128, b.a1.nk, d->vcs, d->vsn, y + DV, 3 * DV, 128);
        if (tp) tap(d, T_V_QR1, y, N, DV, 3 * DV), tap(d, T_V_KR1, y + DV, N, DV, 3 * DV);
        attn(d, y, y + DV, y + 2 * DV, att, N, N, 3 * DV, 0.f, d->attn_q8);
        linh(d, hn, N, b.a1.g, d->gl);
        if (tp) tap(d, T_V_ATT1, att, N, DV, DV), tap(d, T_V_GL1, d->gl, N, NH, NH);
        RqIn g{att, 4096};
        g.gl = d->gl, g.hd = 128, g.hs = 128;
        rq(d, RQ_GATE, N, DV, g);
        if (tp && d->tap[T_V_O1]) lin(d, DQ_F32B, b.a1.o, 0, N, (bf16*)d->tap[T_V_O1]);
        lin_res(d, DQ_RES, b.a1.o, d->vr, vx, vg, nullptr, 2 * DV);
        if (tp) tap(d, T_V_X1, vx, N, DV, DV);
    }
    // ---- video text cross-attention (adaLN on q and on the context)
    mark(d, 2);
    {
        RqIn a{vx, DV};
        a.mod = vm, a.ldm = 9 * DV, a.mrow = d->vrow, a.sco = 7 * DV, a.sho = 6 * DV, a.copy = hn;
        rq(d, RQ_ADALN, N, DV, a);
        lin(d, DQ_F32B, b.a2.q, 0, N, att);
        qk_norm(d, att, DV, N, DV, 128, b.a2.nq, nullptr, nullptr, y, 2 * DV, 128);
        if (tp) tap(d, T_V_N2, hn, N, DV, DV);
        const int L = d->nctx;
        k_ctx_mod<<<nb((size_t)L * DV, 256), 256, 0, d->st>>>(d->vctx, L, DV, d->vkv + (size_t)i * 2 * DV, d->cm);
        if (tp) tap(d, T_V_CTX2, d->cm, L, DV, DV);
        RqIn c{d->cm, DV};
        rq(d, RQ_PLAIN, L, DV, c);
        lin(d, DQ_F32B, b.a2.kv, 0, L, d->kv);
        qk_norm(d, d->kv, 2 * DV, L, DV, 128, b.a2.nk, nullptr, nullptr, d->kv, 2 * DV, 128);
        attn(d, y, d->kv, d->kv + DV, att, N, L, 2 * DV, 0.f);
        linh(d, hn, N, b.a2.g, d->gl);
        if (tp) tap(d, T_V_ATT2, att, N, DV, DV);
        RqIn g{att, 4096};
        g.gl = d->gl, g.hd = 128, g.hs = 128;
        rq(d, RQ_GATE, N, DV, g);
        if (tp && d->tap[T_V_O2]) lin(d, DQ_F32B, b.a2.o, 0, N, (bf16*)d->tap[T_V_O2]);
        lin_res(d, DQ_GATE, b.a2.o, d->vr, vx, nullptr, vm + 8 * DV, 9 * DV);
        if (tp) tap(d, T_V_X2, vx, N, DV, DV);
    }
    // ---- audio self-attention + audio text cross-attention
    mark(d, 3);
    if (A > 0) {
        RqIn a{ax, DA};
        a.mod = am, a.ldm = 9 * DA, a.mrow = d->arow, a.sco = DA, a.sho = 0, a.copy = d->hnA;
        rq(d, RQ_RMSMOD, A, DA, a);
        lin(d, DQ_F32B, b.aa1.q, 0, A, d->yA);
        qk_norm(d, d->yA, 3 * DA, A, DA, 64, b.aa1.nq, d->acs, d->asn, d->PA, 3 * 4096, HP);
        qk_norm(d, d->yA + DA, 3 * DA, A, DA, 64, b.aa1.nk, d->acs, d->asn, d->PA + 4096, 3 * 4096, HP);
        pad(d, d->yA + 2 * DA, 3 * DA, A, 64, d->PA + 2 * 4096, 3 * 4096);
        if (tp) tap(d, T_A_N1, d->hnA, A, DA, DA), tap(d, T_A_QR1, d->PA, A, 4096, 3 * 4096);
        attn(d, d->PA, d->PA + 4096, d->PA + 2 * 4096, d->attA, A, A, 3 * 4096, 0.125f);
        linh(d, d->hnA, A, b.aa1.g, d->glA);
        if (tp) tap(d, T_A_ATT1, d->attA, A, 4096, 4096);
        RqIn g{d->attA, 4096};
        g.gl = d->glA, g.hd = 64, g.hs = HP;
        rq(d, RQ_GATE, A, DA, g);
        if (tp && d->tap[T_A_O1]) lin(d, DQ_F32B, b.aa1.o, 0, A, (bf16*)d->tap[T_A_O1]);
        lin_res(d, DQ_RES, b.aa1.o, d->ar, ax, ag, nullptr, 2 * DA);
        if (tp) tap(d, T_A_X1, ax, A, DA, DA);

        RqIn a2{ax, DA};
        a2.mod = am, a2.ldm = 9 * DA, a2.mrow = d->arow, a2.sco = 7 * DA, a2.sho = 6 * DA, a2.copy = d->hnA;
        rq(d, RQ_ADALN, A, DA, a2);
        lin(d, DQ_F32B, b.aa2.q, 0, A, d->yA);
        qk_norm(d, d->yA, DA, A, DA, 64, b.aa2.nq, nullptr, nullptr, d->PA, 2 * 4096, HP);
        if (tp) tap(d, T_A_N2, d->hnA, A, DA, DA);
        const int L = d->nctx;
        k_ctx_mod<<<nb((size_t)L * DA, 256), 256, 0, d->st>>>(d->actx, L, DA, d->akv + (size_t)i * 2 * DA, d->cm);
        RqIn c{d->cm, DA};
        rq(d, RQ_PLAIN, L, DA, c);
        lin(d, DQ_F32B, b.aa2.kv, 0, L, d->cm);
        qk_norm(d, d->cm, 2 * DA, L, DA, 64, b.aa2.nk, nullptr, nullptr, d->kv, 2 * 4096, HP);
        pad(d, d->cm + DA, 2 * DA, L, 64, d->kv + 4096, 2 * 4096);
        attn(d, d->PA, d->kv, d->kv + 4096, d->attA, A, L, 2 * 4096, 0.125f);
        linh(d, d->hnA, A, b.aa2.g, d->glA);
        if (tp) tap(d, T_A_ATT2, d->attA, A, 4096, 4096);
        RqIn g2{d->attA, 4096};
        g2.gl = d->glA, g2.hd = 64, g2.hs = HP;
        rq(d, RQ_GATE, A, DA, g2);
        lin_res(d, DQ_GATE, b.aa2.o, d->ar, ax, nullptr, am + 8 * DA, 9 * DA);
        if (tp) tap(d, T_A_X2, ax, A, DA, DA);
    }
    // ---- audio -> video and video -> audio cross-attention (temporal RoPE only)
    mark(d, 4);
    if (A > 0) {
        // a2v: q from video (rms_adaln), k / v from audio (rms_norm then eager modulation)
        RqIn a{vx, DV};
        a.mod = vca, a.ldm = 4 * DV, a.mrow = d->vrow, a.sco = 0, a.sho = DV, a.copy = hn;
        rq(d, RQ_ADALN, N, DV, a);
        lin(d, DQ_F32B, b.a2v.q, 0, N, att);
        qk_norm(d, att, DA, N, DA, 64, b.a2v.nq, d->xvcs, d->xvsn, y, 2 * 4096, HP);
        if (tp) tap(d, T_V_NA2V, hn, N, DV, DV), tap(d, T_A2V_Q, y, N, 4096, 2 * 4096);
        RqIn c{ax, DA};
        c.mod = aca, c.ldm = 4 * DA, c.mrow = d->arow, c.sco = 0, c.sho = DA, c.copy = tp ? d->hnA : nullptr;
        rq(d, RQ_RMSMOD, A, DA, c);
        if (tp) tap(d, T_A_CA2V, d->hnA, A, DA, DA);
        lin(d, DQ_F32B, b.a2v.kv, 0, A, d->yA);
        qk_norm(d, d->yA, 2 * DA, A, DA, 64, b.a2v.nk, d->xacs, d->xasn, d->PA, 2 * 4096, HP);
        pad(d, d->yA + DA, 2 * DA, A, 64, d->PA + 4096, 2 * 4096);
        if (tp) tap(d, T_A2V_K, d->PA, A, 4096, 2 * 4096);
        attn(d, y, d->PA, d->PA + 4096, att, N, A, 2 * 4096, 0.125f);
        linh(d, hn, N, b.a2v.g, d->gl);
        if (tp) tap(d, T_A2V_ATT, att, N, 4096, 4096);
        RqIn g{att, 4096};
        g.gl = d->gl, g.hd = 64, g.hs = HP;
        rq(d, RQ_GATE, N, DA, g);
        lin(d, DQ_RES, b.a2v.o, 0, N, vx, d->vga + (size_t)i * DV);
        if (tp) tap(d, T_V_X3, vx, N, DV, DV);

        // v2a: q from audio, k / v from video (updated by a2v)
        RqIn qa{ax, DA};
        qa.mod = aca, qa.ldm = 4 * DA, qa.mrow = d->arow, qa.sco = 2 * DA, qa.sho = 3 * DA, qa.copy = d->hnA;
        rq(d, RQ_RMSMOD, A, DA, qa);
        lin(d, DQ_F32B, b.v2a.q, 0, A, d->yA);
        qk_norm(d, d->yA, DA, A, DA, 64, b.v2a.nq, d->xacs, d->xasn, d->PA, 2 * 4096, HP);
        if (tp) tap(d, T_A_NV2A, d->hnA, A, DA, DA);
        RqIn cv{vx, DV};
        cv.mod = vca, cv.ldm = 4 * DV, cv.mrow = d->vrow, cv.sco = 2 * DV, cv.sho = 3 * DV;
        rq(d, RQ_ADALN, N, DV, cv);
        lin(d, DQ_F32B, b.v2a.kv, 0, N, att);
        qk_norm(d, att, 2 * DA, N, DA, 64, b.v2a.nk, d->xvcs, d->xvsn, y, 2 * 4096, HP);
        pad(d, att + DA, 2 * DA, N, 64, y + 4096, 2 * 4096);
        attn(d, d->PA, y, y + 4096, d->attA, A, N, 2 * 4096, 0.125f);
        linh(d, d->hnA, A, b.v2a.g, d->glA);
        if (tp) tap(d, T_V2A_ATT, d->attA, A, 4096, 4096);
        RqIn g2{d->attA, 4096};
        g2.gl = d->glA, g2.hd = 64, g2.hs = HP;
        rq(d, RQ_GATE, A, DA, g2);
        lin(d, DQ_RES, b.v2a.o, 0, A, ax, d->agv + (size_t)i * DA);
        if (tp) tap(d, T_A_X3, ax, A, DA, DA);
    }
    // ---- video feed-forward
    mark(d, 5);
    {
        RqIn a{vx, DV};
        a.mod = vm, a.ldm = 9 * DV, a.mrow = d->vrow, a.sco = 4 * DV, a.sho = 3 * DV, a.copy = tp ? hn : nullptr;
        rq(d, RQ_ADALN, N, DV, a);
        if (tp) tap(d, T_V_NFF, hn, N, DV, DV);
        lin(d, DQ_MUL, b.f1, 0, N, y);
        if (tp) tap(d, T_V_FF1, y, N, FFV, FFV);
        RqIn g{y, FFV};
        rq(d, RQ_GELU, N, FFV, g);
        lin_res(d, DQ_RES, b.f2, d->vr, vx, vg + DV, nullptr, 2 * DV);
    }
    // ---- audio feed-forward
    mark(d, 6);
    if (A > 0) {
        RqIn a{ax, DA};
        a.mod = am, a.ldm = 9 * DA, a.mrow = d->arow, a.sco = 4 * DA, a.sho = 3 * DA, a.copy = tp ? d->hnA : nullptr;
        rq(d, RQ_RMSMOD, A, DA, a);
        if (tp) tap(d, T_A_NFF, d->hnA, A, DA, DA);
        lin(d, DQ_F32B, b.af1, 0, A, d->yA);
        if (tp) tap(d, T_A_FF1, d->yA, A, FFA, FFA);
        RqIn g{d->yA, FFA};
        rq(d, RQ_GELU, A, FFA, g);
        lin_res(d, DQ_RES, b.af2, d->ar, ax, ag + DA, nullptr, 2 * DA);
    }
}

// row ranges of consecutive rows with one timestep index
std::vector<int> ranges(const std::vector<int>& u) {
    std::vector<int> r;
    for (size_t a = 0; a < u.size();) {
        size_t b = a;
        while (b < u.size() && u[b] == u[a]) b++;
        r.push_back((int)a), r.push_back((int)b), r.push_back(u[a]);
        a = b;
    }
    return r;
}

void connector(Dit* d, const Conn& c, const bf16* x, int n, const bf16* cs, const bf16* sn, bf16* out, bf16* tapb) {
    const int L = std::max(CTX, (n + NREG - 1) / NREG * NREG), C = c.C, hd = c.hd;
    if (L > d->nctx) d->nctx = L;   // scratch rows for the connector (ensure)
    ensure(d, 0, 0);
    if (!d->cx) d->cx = (bf16*)dmalloc((size_t)CTXMAX * DV * 2);
    bf16* X = d->cx;
    k_regs<<<nb((size_t)L * C, 256), 256, 0, d->st>>>(x, n, c.reg, L, C, X);
    for (int j = 0; j < NCB; j++) {
        const Att& a = c.a[j];
        RqIn r{X, C};
        r.copy = d->hn;
        rq(d, RQ_RMS, L, C, r);
        lin(d, DQ_F32B, a.q, 0, L, d->y);
        const bf16* q = d->y;
        int ld = 3 * C;
        if (hd == 128) {
            qk_norm(d, d->y, 3 * C, L, C, hd, a.nq, cs, sn, d->y, 3 * C, hd);
            qk_norm(d, d->y + C, 3 * C, L, C, hd, a.nk, cs, sn, d->y + C, 3 * C, hd);
        } else {
            qk_norm(d, d->y, 3 * C, L, C, hd, a.nq, cs, sn, d->PA, 3 * 4096, HP);
            qk_norm(d, d->y + C, 3 * C, L, C, hd, a.nk, cs, sn, d->PA + 4096, 3 * 4096, HP);
            pad(d, d->y + 2 * C, 3 * C, L, hd, d->PA + 2 * 4096, 3 * 4096);
            q = d->PA, ld = 3 * 4096;
        }
        attn(d, q, q + (hd == 128 ? C : 4096), q + 2 * (hd == 128 ? C : 4096), d->att, L, L, ld, hd == 128 ? 0.f : 0.125f);
        linh(d, d->hn, L, a.g, d->gl);
        RqIn g{d->att, 4096};
        g.gl = d->gl, g.hd = hd, g.hs = HP;
        rq(d, RQ_GATE, L, C, g);
        lin(d, DQ_RES, a.o, 0, L, X, d->ones);
        RqIn r2{X, C};
        rq(d, RQ_RMS, L, C, r2);
        lin(d, DQ_F32B, c.f1[j], 0, L, d->y);
        RqIn ge{d->y, c.ff};
        rq(d, RQ_GELU, L, c.ff, ge);
        lin(d, DQ_RES, c.f2[j], 0, L, X, d->ones);
        if (tapb) CK(cudaMemcpyAsync(tapb + (size_t)j * L * C, X, (size_t)L * C * 2, cudaMemcpyDeviceToDevice, d->st));
    }
    k_rms_out<<<L, 256, 0, d->st>>>(X, C, out);
}

Conn load_conn(Dit* d, const std::string& p, int C, int hd, int ff) {
    Conn c;
    c.C = C, c.hd = hd, c.ff = ff;
    for (int j = 0; j < NCB; j++) {
        const std::string q = p + ".transformer_1d_blocks." + std::to_string(j) + ".";
        c.a[j] = load_att(d, q + "attn1", true, C, C, C);
        c.f1[j] = load_lin(d, {q + "ff.net.0.proj"}, C, true);
        c.f2[j] = load_lin(d, {q + "ff.net.2"}, ff, true);
    }
    c.reg = upload_bf16(d, p + ".learnable_registers", (size_t)NREG * C);
    return c;
}

Ada load_ada(Dit* d, const std::string& p, int D, int coef) {
    Ada a;
    a.D = D, a.coef = coef;
    a.l1 = load_linh(d, p + ".emb.timestep_embedder.linear_1", D, TIN);
    a.l2 = load_linh(d, p + ".emb.timestep_embedder.linear_2", D, D);
    a.lin = load_linh(d, p + ".linear", coef * D, D);
    return a;
}

void free_lin(Lin& L) {
    cudaFree(L.w), cudaFree(L.s), cudaFree(L.b), cudaFree(L.bh);
}
void free_linh(LinH& L) { cudaFree(L.w), cudaFree(L.b); }
void free_att(Att& a) {
    free_lin(a.q), free_lin(a.kv), free_lin(a.o), free_linh(a.g);
    cudaFree(a.nq), cudaFree(a.nk);
}
}  // namespace

extern "C" {

void* ltx_dit_new(const char* path) {
    Dit* d = new Dit();
    if (!d->f.open(path)) DIE("cannot open %s", path);
    CK(cudaStreamCreate(&d->st));   // blocking on purpose (common.cuh)
    d->lt = lt_new(d->st);
    CK(cudaMallocHost(&g_stage, STAGE));
    d->pv = load_linh(d, "patchify_proj", DV, PIN);
    d->pa = load_linh(d, "audio_patchify_proj", DA, PIN);
    d->ov = load_linh(d, "proj_out", PIN, DV);
    d->oa = load_linh(d, "audio_proj_out", PIN, DA);
    d->kfe = upload_bf16(d, "keyframes_abs_pos_embedding", DV);
    d->fsst = upload_bf16(d, "scale_shift_table", 2 * DV);
    d->afsst = upload_bf16(d, "audio_scale_shift_table", 2 * DA);
    d->ada[A_V] = load_ada(d, "adaln_single", DV, 9);
    d->ada[A_VP] = load_ada(d, "prompt_adaln_single", DV, 2);
    d->ada[A_A] = load_ada(d, "audio_adaln_single", DA, 9);
    d->ada[A_AP] = load_ada(d, "audio_prompt_adaln_single", DA, 2);
    d->ada[A_CAV] = load_ada(d, "av_ca_video_scale_shift_adaln_single", DV, 4);
    d->ada[A_GAV] = load_ada(d, "av_ca_a2v_gate_adaln_single", DV, 1);
    d->ada[A_CAA] = load_ada(d, "av_ca_audio_scale_shift_adaln_single", DA, 4);
    d->ada[A_GAA] = load_ada(d, "av_ca_v2a_gate_adaln_single", DA, 1);
    d->cv = load_conn(d, "video_embeddings_connector", DV, 128, FFV);
    d->ca = load_conn(d, "audio_embeddings_connector", DA, 64, FFA);
    d->sst = (bf16*)dmalloc((size_t)NB * 9 * DV * 2);
    d->asst = (bf16*)dmalloc((size_t)NB * 9 * DA * 2);
    d->psst = (bf16*)dmalloc((size_t)NB * 2 * DV * 2);
    d->apsst = (bf16*)dmalloc((size_t)NB * 2 * DA * 2);
    d->cav4 = (bf16*)dmalloc((size_t)NB * 4 * DV * 2);
    d->cavg = (bf16*)dmalloc((size_t)NB * DV * 2);
    d->caa4 = (bf16*)dmalloc((size_t)NB * 4 * DA * 2);
    d->caag = (bf16*)dmalloc((size_t)NB * DA * 2);
    for (int i = 0; i < NB; i++) {
        const std::string p = "transformer_blocks." + std::to_string(i) + ".";
        Blk& b = d->b[i];
        b.a1 = load_att(d, p + "attn1", true, DV, DV, DV);
        b.a2 = load_att(d, p + "attn2", false, DV, DV, DV);
        b.aa1 = load_att(d, p + "audio_attn1", true, DA, DA, DA);
        b.aa2 = load_att(d, p + "audio_attn2", false, DA, DA, DA);
        b.a2v = load_att(d, p + "audio_to_video_attn", false, DV, DA, DA);
        b.v2a = load_att(d, p + "video_to_audio_attn", false, DA, DV, DA);
        b.f1 = load_lin(d, {p + "ff.net.0.proj"}, DV, false);
        b.f2 = load_lin(d, {p + "ff.net.2"}, FFV, false);
        b.af1 = load_lin(d, {p + "audio_ff.net.0.proj"}, DA, true);
        b.af2 = load_lin(d, {p + "audio_ff.net.2"}, FFA, true);
        upload_bf16(d, p + "scale_shift_table", 9 * DV, d->sst + (size_t)i * 9 * DV);
        upload_bf16(d, p + "audio_scale_shift_table", 9 * DA, d->asst + (size_t)i * 9 * DA);
        upload_bf16(d, p + "prompt_scale_shift_table", 2 * DV, d->psst + (size_t)i * 2 * DV);
        upload_bf16(d, p + "audio_prompt_scale_shift_table", 2 * DA, d->apsst + (size_t)i * 2 * DA);
        bf16* t = upload_bf16(d, p + "scale_shift_table_a2v_ca_video", 5 * DV);
        CK(cudaMemcpy(d->cav4 + (size_t)i * 4 * DV, t, (size_t)4 * DV * 2, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(d->cavg + (size_t)i * DV, t + 4 * DV, (size_t)DV * 2, cudaMemcpyDeviceToDevice));
        cudaFree(t);
        t = upload_bf16(d, p + "scale_shift_table_a2v_ca_audio", 5 * DA);
        CK(cudaMemcpy(d->caa4 + (size_t)i * 4 * DA, t, (size_t)4 * DA * 2, cudaMemcpyDeviceToDevice));
        CK(cudaMemcpy(d->caag + (size_t)i * DA, t + 4 * DA, (size_t)DA * 2, cudaMemcpyDeviceToDevice));
        cudaFree(t);
    }
    std::vector<float> one(DV, 1.f);
    d->ones = (float*)dmalloc(DV * 4);
    CK(cudaMemcpy(d->ones, one.data(), DV * 4, cudaMemcpyHostToDevice));
    d->vga = (float*)dmalloc((size_t)NB * DV * 4);
    d->agv = (float*)dmalloc((size_t)NB * DA * 4);
    d->vkv = (bf16*)dmalloc((size_t)NB * 2 * DV * 2);
    d->akv = (bf16*)dmalloc((size_t)NB * 2 * DA * 2);
    d->vctx = (bf16*)dmalloc((size_t)CTXMAX * DV * 2);
    d->actx = (bf16*)dmalloc((size_t)CTXMAX * DA * 2);
    d->kv = (bf16*)dmalloc((size_t)CTXMAX * 8192 * 2);
    d->cm = (bf16*)dmalloc((size_t)CTXMAX * DV * 2);
    d->f32 = (float*)dmalloc((size_t)CH * DV * 4);   // bf16-linear scratch: CH rows x 4096, or U rows x 36864
    d->f.close();
    CK(cudaDeviceSynchronize());
    CK(cudaFreeHost(g_stage));
    g_stage = nullptr;
    return d;
}

// text connector: x [n, C] bf16 (device, n <= CTXMAX; C = 4096 video / 2048 audio) -> out [L, C] bf16 (device),
// L = max(1024, ceil(n / 128) * 128). cs / sn: RoPE cos / sin [L, C/2] bf16 (positions 0..L-1, max_pos 4096).
// tap_blocks (optional): [8][L][C].
int ltx_dit_connector(void* h, int audio, const void* x, int n, const void* cs, const void* sn, void* out, void* tap_blocks) {
    Dit* d = (Dit*)h;
    if (n > CTXMAX) DIE("ltx_dit_connector: %d text tokens > %d", n, CTXMAX);
    connector(d, audio ? d->ca : d->cv, (const bf16*)x, n, (const bf16*)cs, (const bf16*)sn, (bf16*)out, (bf16*)tap_blocks);
    CK(cudaStreamSynchronize(d->st));
    CK(cudaGetLastError());
    return 0;
}

// Prepare a job: text contexts (connector outputs, device bf16 [nctx, 4096] / [nctx, 2048]), nv video tokens (the
// generated frames of ppf tokens each, then any IC-LoRA guide tokens), na audio tokens, RoPE cos / sin tables (device
// bf16): video self [nv, 2048], audio self [na, 1024], cross-modal video side [nv, 1024], audio side [na, 1024].
// The keyframe marker goes on the first ppf tokens (generated frame 0; guide tokens never get it, as ComfyUI).
int ltx_dit_prepare(void* h, const void* vctx, const void* actx, int nctx, int nv, int ppf, int na, const void* vcs, const void* vsn,
                    const void* acs, const void* asn, const void* xvcs, const void* xvsn, const void* xacs, const void* xasn) {
    Dit* d = (Dit*)h;
    if (ppf <= 0 || nv < ppf) DIE("ltx_dit_prepare: nv %d < ppf %d", nv, ppf);
    if (nctx < CTX || nctx > CTXMAX || nctx % NREG) DIE("ltx_dit_prepare: text context %d rows", nctx);
    d->nv = nv, d->ppf = ppf, d->na = na, d->F = nv / ppf, d->nctx = nctx;
    ensure(d, nv, na);
    CK(cudaMemcpyAsync(d->vctx, vctx, (size_t)nctx * DV * 2, cudaMemcpyDeviceToDevice, d->st));
    CK(cudaMemcpyAsync(d->actx, actx, (size_t)nctx * DA * 2, cudaMemcpyDeviceToDevice, d->st));
    bf16** tabs[] = {&d->vcs, &d->vsn, &d->acs, &d->asn, &d->xvcs, &d->xvsn, &d->xacs, &d->xasn};
    const void* src[] = {vcs, vsn, acs, asn, xvcs, xvsn, xacs, xasn};
    const size_t n[] = {(size_t)nv * 2048, (size_t)nv * 2048, (size_t)na * 1024, (size_t)na * 1024, (size_t)nv * 1024,
                        (size_t)nv * 1024, (size_t)na * 1024, (size_t)na * 1024};
    for (int k = 0; k < 8; k++) {
        cudaFree(*tabs[k]);
        *tabs[k] = (bf16*)dmalloc(n[k] * 2);
        CK(cudaMemcpyAsync(*tabs[k], src[k], n[k] * 2, cudaMemcpyDeviceToDevice, d->st));
    }
    CK(cudaStreamSynchronize(d->st));
    CK(cudaGetLastError());
    return 0;
}

// One DiT call. vx [nv, 128] / ax [na, 128] bf16 (device): patchified latents (tokens (f, h, w), channels c for video;
// tokens t, channels (c, freq) for audio). sinus [U, 256] bf16 (device): timestep_embedding(t * 1000) of the U distinct
// timestep values. vfu [F] (host): timestep index of each video frame; atu [na] (host): of each audio token;
// sel[4] (host): index of the a2v gate (max audio), v2a gate (max video), video prompt (max video), audio prompt (max
// audio) timesteps. Writes the velocity rows vout [nv, 128], aout [na, 128] bf16 (device).
int ltx_dit_forward_tok(void* h, const void* vx, const void* ax, const void* sinus, int U, const int* vtu, const int* atu,
                        const int* sel, void* vout, void* aout);

int ltx_dit_forward(void* h, const void* vx, const void* ax, const void* sinus, int U, const int* vfu, const int* atu,
                    const int* sel, void* vout, void* aout) {
    Dit* d = (Dit*)h;
    std::vector<int> vu(d->nv);
    for (int m = 0; m < d->nv; m++) vu[m] = vfu[m / d->ppf];
    return ltx_dit_forward_tok(h, vx, ax, sinus, U, vu.data(), atu, sel, vout, aout);
}

// As ltx_dit_forward with vtu [nv] (host): the timestep index of every video token (generated frames and IC-LoRA guide
// tokens; ComfyUI runs the adaLN per token when a dilated guide makes the denoise mask vary inside a frame — same rows).
int ltx_dit_forward_tok(void* h, const void* vx, const void* ax, const void* sinus, int U, const int* vtu, const int* atu,
                        const int* sel, void* vout, void* aout) {
    Dit* d = (Dit*)h;
    const int N = d->nv, A = d->na;
    ensure(d, N, A);
    ensure_u(d, U);
    d->U = U;
    d->u_ga = sel[0], d->u_gv = sel[1], d->u_pv = sel[2], d->u_pa = sel[3];
    std::vector<int> vu(vtu, vtu + N), au(atu, atu + A);
    for (int u : vu) if (u < 0 || u >= U) DIE("ltx_dit_forward: video timestep index %d / %d", u, U);
    for (int u : au) if (u < 0 || u >= U) DIE("ltx_dit_forward: audio timestep index %d / %d", u, U);
    for (int k = 0; k < 4; k++) if (sel[k] < 0 || sel[k] >= U) DIE("ltx_dit_forward: sel[%d] = %d / %d", k, sel[k], U);
    d->vr = ranges(vu), d->ar = ranges(au);
    mark(d, 0);
    CK(cudaMemcpyAsync(d->vrow, vu.data(), (size_t)N * 4, cudaMemcpyHostToDevice, d->st));
    if (A) CK(cudaMemcpyAsync(d->arow, au.data(), (size_t)A * 4, cudaMemcpyHostToDevice, d->st));
    CK(cudaMemcpyAsync(d->sinus, sinus, (size_t)U * TIN * 2, cudaMemcpyDeviceToDevice, d->st));
    conditioning(d);
    linh(d, (const bf16*)vx, N, d->pv, d->vx);
    k_add_row<<<nb((size_t)d->ppf * DV, 256), 256, 0, d->st>>>(d->vx, d->kfe, d->ppf, DV);   // keyframe marker: frame 0
    if (A) linh(d, (const bf16*)ax, A, d->pa, d->ax);
    for (int i = 0; i < NB; i++) {
        d->tapping = i == 0 && d->tap_bv;   // sub-op taps only with block taps on
        block(d, i);
        d->tapping = 0;
        if (d->tap_bv) CK(cudaMemcpyAsync(d->tap_bv + (size_t)i * N * DV, d->vx, (size_t)N * DV * 2, cudaMemcpyDeviceToDevice, d->st));
        if (d->tap_ba && A) CK(cudaMemcpyAsync(d->tap_ba + (size_t)i * A * DA, d->ax, (size_t)A * DA * 2, cudaMemcpyDeviceToDevice, d->st));
    }
    mark(d, 0);
    k_final<<<N, 256, DV * 4, d->st>>>(d->vx, DV, d->vfin, d->vrow, d->hn);
    linh(d, d->hn, N, d->ov, (bf16*)vout);
    if (A) {
        k_final<<<A, 256, DA * 4, d->st>>>(d->ax, DA, d->afin, d->arow, d->hnA);
        linh(d, d->hnA, A, d->oa, (bf16*)aout);
    }
    if (g_prof && !d->tap_bv) prof_end(d);
    else g_ph.clear();
    CK(cudaStreamSynchronize(d->st));
    CK(cudaGetLastError());   // launches here are unchecked: surface their errors in this call
    return 0;
}

// debug: block i alone on vin [nv, 4096] / ain [na, 2048] with the conditioning of the last forward()
int ltx_dit_block(void* h, int i, const void* vin, const void* ain, void* vout, void* aout) {
    Dit* d = (Dit*)h;
    CK(cudaMemcpyAsync(d->vx, vin, (size_t)d->nv * DV * 2, cudaMemcpyDeviceToDevice, d->st));
    if (d->na) CK(cudaMemcpyAsync(d->ax, ain, (size_t)d->na * DA * 2, cudaMemcpyDeviceToDevice, d->st));
    block(d, i);
    g_ph.clear();
    CK(cudaMemcpyAsync(vout, d->vx, (size_t)d->nv * DV * 2, cudaMemcpyDeviceToDevice, d->st));
    if (d->na) CK(cudaMemcpyAsync(aout, d->ax, (size_t)d->na * DA * 2, cudaMemcpyDeviceToDevice, d->st));
    CK(cudaStreamSynchronize(d->st));
    CK(cudaGetLastError());
    return 0;
}

// debug taps: id < NTAP = block-0 sub-op destination (see enum in ltx_dit.cu); blocks: [48][nv][4096] / [48][na][2048]
void ltx_dit_tap(void* h, int id, void* dst) { ((Dit*)h)->tap[id] = dst; }
void ltx_dit_tap_blocks(void* h, void* bv, void* ba) {
    Dit* d = (Dit*)h;
    d->tap_bv = (bf16*)bv;
    d->tap_ba = (bf16*)ba;
}
// debug: copy the adaLN rows of the last forward to dst (device): k = module (A_V, A_VP, A_A, A_AP, A_CAV, A_GAV, A_CAA,
// A_GAA) -> [U][coef * D] bf16, k = 8 / 9 -> video / audio embedded timestep [U][D]
void ltx_dit_temb(void* h, int k, void* dst) {
    Dit* d = (Dit*)h;
    const void* src = k < NADA ? (void*)d->aout[k] : k == NADA ? (void*)d->vemb : (void*)d->aemb;
    const size_t n = (size_t)d->U * (k < NADA ? d->ada[k].coef * d->ada[k].D : k == NADA ? DV : DA);
    CK(cudaMemcpy(dst, src, n * 2, cudaMemcpyDeviceToDevice));
}
void ltx_dit_attn_q8(void* h, int on) { ((Dit*)h)->attn_q8 = on; }
void ltx_dit_qnoise(void* h, int rate) { ((Dit*)h)->qnoise = rate; }
// debug: the engine attention on given q / k / v rows (32 heads x 128, row stride ld) -> o [Mq, 4096]
void ltx_dit_attention(void* h, const void* q, const void* k, const void* v, void* o, int Mq, int Nk, int ld, float scale) {
    Dit* d = (Dit*)h;
    attn(d, (const bf16*)q, (const bf16*)k, (const bf16*)v, (bf16*)o, Mq, Nk, ld, scale);
    CK(cudaStreamSynchronize(d->st));
}

void ltx_dit_free(void* h) {
    Dit* d = (Dit*)h;
    cudaStreamSynchronize(d->st);
    for (int i = 0; i < NB; i++) {
        Blk& b = d->b[i];
        for (Att* a : {&b.a1, &b.a2, &b.aa1, &b.aa2, &b.a2v, &b.v2a}) free_att(*a);
        for (Lin* l : {&b.f1, &b.f2, &b.af1, &b.af2}) free_lin(*l);
    }
    for (Conn* c : {&d->cv, &d->ca}) {
        for (int j = 0; j < NCB; j++) free_att(c->a[j]), free_lin(c->f1[j]), free_lin(c->f2[j]);
        cudaFree(c->reg);
    }
    for (int k = 0; k < NADA; k++) free_linh(d->ada[k].l1), free_linh(d->ada[k].l2), free_linh(d->ada[k].lin), cudaFree(d->aout[k]);
    for (LinH* l : {&d->pv, &d->pa, &d->ov, &d->oa}) free_linh(*l);
    void* ps[] = {d->kfe, d->sst, d->asst, d->psst, d->apsst, d->cav4, d->cavg, d->caa4, d->caag, d->fsst, d->afsst, d->ones,
                  d->vctx, d->actx, d->vcs, d->vsn, d->acs, d->asn, d->xvcs, d->xvsn, d->xacs, d->xasn, d->sinus, d->t1, d->t2,
                  d->vemb, d->aemb, d->vmod, d->vcam, d->amod, d->acam, d->vkv, d->akv, d->vfin, d->afin, d->vg, d->ag,
                  d->vga, d->agv, d->vrow, d->arow, d->vx, d->y, d->att, d->hn, d->gl, d->kv, d->cm, d->cx, d->aq, d->as,
                  d->f32, d->ax, d->yA, d->PA, d->attA, d->hnA, d->glA};
    for (void* p : ps) cudaFree(p);
    lt_free(d->lt);
    if (d->used_q8) attention_q8_release();   // its global workspace (one model loaded at a time)
    cudaStreamDestroy(d->st);
    delete d;
}
}  // extern "C"
