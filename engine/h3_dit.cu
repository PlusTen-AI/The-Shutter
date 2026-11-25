// shutter engine — MiniMax H3 Ref2VA DiT (int8 ConvRot W8A8), C API
// Follows ComfyUI comfy/ldm/minimax/model.py (e9afbccc2) as ComfyUI runs it with --bf16-unet:
//  - the fp32 "island" weights (patch projections, output heads, adaln curve projections) are bf16-rounded, fp32 math
//  - packed sequence [text | cond / ref rows | target audio | target video], dense self-attention (no mask)
//  - text rows follow the video timestep and every row attends to every row, so nothing is step-independent:
//    prepare() embeds the condition rows once, each step runs the full sequence.
#include "common.cuh"
#include "st.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <unordered_map>
#include <vector>

namespace {
constexpr int D = 5376, NH = 56, HDIM = 128, IN = NH * HDIM, QKV = 3 * IN, FF = 14336, NB = 50, NR = 2, TXT = 5120;
constexpr int TD = 8, NMOD = 6 * D, NADA = 3 * NMOD, NFIN = 2 * D, VP = 96, AP = 32, ROT = 48;
constexpr int CH = 8192;  // rows per linear chunk: bounds the work buffers (CH x 28672)
constexpr int RAW_B0 = 46;  // pi-H3 adapter: first block whose adaln bias it sets (docs/pih3.md §1)
constexpr float EPS = 1e-5f;
enum { K_TEXT = 0, K_VIS = 1, K_AUD = 2, K_TAUD = 3, K_TVID = 4 };

struct Block {
    int8_t *wqkv, *wo, *w1, *w2;
    float *sqkv, *so, *s1, *s2, *aw, *ab;
    bf16 *n1, *n2, *nq, *nk;
};
struct Ref {
    bf16 *wqkv, *wo, *w1, *w2, *n1, *n2, *nq, *nk;
};

// A loaded weight, by its file name, so a load-time merge (engine/py/merge.py) can overwrite it: raw = the file's bytes
// as uploaded (int8, f32 scales, bf16), else an fp32 copy that may be bf16-rounded (upload_f32).
struct Slot {
    void* p;
    size_t elems;
    std::string dtype;  // raw dtype ("I8", "F32", "BF16") or "f32" for an upload_f32 tensor
    bool round;
};

struct Dit {
    cudaStream_t st;
    Lt* lt;
    StFile f;
    std::unordered_map<std::string, Slot> slots;
    int ntab = 0;
    float *tab, *inv, *cpb, *vpw, *vpb, *apw, *apb, *faw, *fab, *vow, *vob, *aow, *aob, *aw_all, *ab_all;
    bf16 *cpw, *rfn, *fnw;
    Ref r[NR];
    Block b[NB];
    // job
    int S = 0, P = 0, na = 0, nv = 0, cap = 0, ucap = 0;
    bf16* x0 = nullptr;    // [P, D] embedded condition rows
    float* pe = nullptr;   // [S, 48, 2] rope (cos, sin), bf16 values
    int* mrow = nullptr;   // [S] mod row per token (t index * 3 + modality tag), per step
    float *tv = nullptr, *temb = nullptr, *fmod = nullptr;
    bf16* mod = nullptr;   // [NB][U*3][NMOD]
    // work
    bf16 *x = nullptr, *q = nullptr, *k = nullptr, *v = nullptr, *att = nullptr;
    int8_t* aq = nullptr;
    float *as = nullptr, *fo = nullptr;
    bf16* y = nullptr;     // [CH, 2 FF] linear outputs (bf16, gemm_i8_epi)
    // debug taps (device pointers owned by the caller)
    bf16 *tap_blocks = nullptr, *tap_qkv = nullptr, *tap_att = nullptr, *tap_fc1 = nullptr;
    int U3 = 0, qnoise = 0;  // last forward's mod rows; qnoise: debug +-1 flips per 1000 int8 activations
    int attn_q8 = 0;         // lossy switch: SageAttention2-style int8 QK / e4m3 PV (attn_q8.cu)
    // attention hook (option vsa, h3_vsa.py): for the blocks in hook_on the qkv projection [S, QKV] goes to hook_qkv
    // (before q/k norm and rope), hook(i) computes the attention into hook_att [S, IN], o-proj reads it from there
    int (*hook)(int) = nullptr;
    bf16 *hook_qkv = nullptr, *hook_att = nullptr;
    unsigned char hook_on[NB] = {};
    int raw_first = 0;       // gate only: the pi-H3 adapter's float set tensors are unrounded until the first forward ends
    uint32_t qcall = 0;
};

void* dmalloc(size_t n) {
    void* p;
    CK(cudaMalloc(&p, n));
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
    const StTensor& t = d->f.get(name);
    if (t.dtype != dtype) DIE("%s: dtype %s, expected %s", name.c_str(), t.dtype.c_str(), dtype);
    const size_t es = t.dtype == "I8" ? 1 : t.dtype == "F32" ? 4 : 2;
    if (t.bytes() != elems * es) DIE("%s: %llu bytes, expected %zu elems", name.c_str(), (unsigned long long)t.bytes(), elems);
    return t;
}

void* upload(Dit* d, const std::string& name, const char* dtype, size_t elems) {
    const StTensor& t = tensor(d, name, dtype, elems);
    void* p = dmalloc(t.bytes());
    h2d(p, d->f.ptr(t), t.bytes());
    d->slots[name] = {p, elems, dtype, false};
    return p;
}

// F32 / F16 / BF16 tensor -> fp32 on the device. round: through bf16 first (ComfyUI --bf16-unet casts these weights).
float* upload_f32(Dit* d, const std::string& name, size_t elems, bool round, float* dst = nullptr) {
    const StTensor& t = d->f.get(name);
    std::vector<float> h(elems);
    const uint8_t* p = d->f.ptr(t);
    if (t.bytes() != elems * (t.dtype == "F32" ? 4 : 2)) DIE("%s: %llu bytes, expected %zu elems", name.c_str(), (unsigned long long)t.bytes(), elems);
    for (size_t i = 0; i < elems; i++) {
        float v;
        if (t.dtype == "F32") memcpy(&v, p + 4 * i, 4);
        else if (t.dtype == "F16") v = __half2float(((const __half*)p)[i]);
        else if (t.dtype == "BF16") v = bf2f(((const bf16*)p)[i]);
        else DIE("%s: dtype %s", name.c_str(), t.dtype.c_str());
        h[i] = round ? bf2f(f2bf(v)) : v;
    }
    float* o = dst ? dst : (float*)dmalloc(elems * 4);
    CK(cudaMemcpy(o, h.data(), elems * 4, cudaMemcpyHostToDevice));
    d->slots[name] = {o, elems, "f32", round};
    return o;
}

void check_convrot(Dit* d, const std::string& layer) {
    const StTensor& t = d->f.get(layer + ".comfy_quant");
    std::string js((const char*)d->f.ptr(t), t.bytes());
    if (js.find("int8_tensorwise") == std::string::npos || js.find("\"convrot\": true") == std::string::npos ||
        js.find("256") == std::string::npos)
        DIE("%s: unexpected quant config %s", layer.c_str(), js.c_str());
}

inline int nb(size_t n, int t) { return (int)((n + t - 1) / t); }
__device__ __forceinline__ float rbf(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float warp_sum(float v) {
    for (int o = 16; o; o >>= 1) v += __shfl_xor_sync(0xffffffffu, v, o);
    return v;
}
__device__ float block_sum(float v, float* red) {
    v = warp_sum(v);
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
    if (l == 0) red[w] = v;
    __syncthreads();
    if (w == 0) {
        v = l < nw ? red[l] : 0.f;
        v = warp_sum(v);
        if (l == 0) red[0] = v;
    }
    __syncthreads();
    v = red[0];
    __syncthreads();
    return v;
}

// ------------------------------------------------------------------------------------------------- kernels
// adaln curve: t -> lerp of the table rows (torch.lerp formula)
__global__ void k_temb(const float* __restrict__ tab, int ntab, const float* __restrict__ tv, int U, float* __restrict__ out) {
    const int i = threadIdx.x;
    if (i >= U * TD) return;
    const int u = i / TD, c = i % TD;
    const float pos = fminf(fmaxf(tv[u], 0.f), 1.f) * (float)(ntab - 1);
    const int i0 = min((int)floorf(pos), ntab - 2);
    const float w = pos - (float)i0, a = tab[i0 * TD + c], b = tab[(i0 + 1) * TD + c];
    out[i] = fabsf(w) < 0.5f ? a + w * (b - a) : b - (b - a) * (1.f - w);
}

// out[blk][u][n] = temb[u] . w[blk][n] + b[blk][n] (K = 8, fp32); bf16 or fp32 output
__global__ void k_ada(const float* __restrict__ w, const float* __restrict__ b, int nblk, int N, const float* __restrict__ temb,
                      int U, bf16* __restrict__ ob, float* __restrict__ of) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)nblk * N) return;
    const int blk = (int)(i / N), n = (int)(i % N);
    const float4 w0 = *(const float4*)(w + i * TD), w1 = *(const float4*)(w + i * TD + 4);
    for (int u = 0; u < U; u++) {
        const float* t = temb + u * TD;
        float a = w0.x * t[0];
        a = fmaf(w0.y, t[1], a); a = fmaf(w0.z, t[2], a); a = fmaf(w0.w, t[3], a);
        a = fmaf(w1.x, t[4], a); a = fmaf(w1.y, t[5], a); a = fmaf(w1.z, t[6], a); a = fmaf(w1.w, t[7], a);
        a += b[i];
        const size_t o = ((size_t)blk * U + u) * N + n;
        if (ob) ob[o] = __float2bfloat16(a);
        else of[o] = a;
    }
}

// rope table: angle = pos[axis] * inv_freq[i] (fp32), (cos, sin) rounded to bf16 like rope_rotation_table(.., bf16)
__global__ void k_rope_table(const float* __restrict__ pos, const float* __restrict__ inv, int S, float* __restrict__ pe) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= S * ROT) return;
    const int r = i / ROT, j = i % ROT;
    const float ang = pos[r * 3 + j / 16] * inv[j % 16];
    pe[2 * i] = rbf(cosf(ang));
    pe[2 * i + 1] = rbf(sinf(ang));
}

// qkv y = bf16((acc * xs) * ws) (gemm_i8_epi DQ_MUL) -> per-head RMSNorm (one rounding, like ck.rms_rope_split_half_) +
// split-half RoPE on the first 96 dims (pair i <-> i + 48). One warp per (row, head, q|k|v); lane holds dims
// 4*lane..4*lane+3, partner = lane +- 12. fp32 steps pinned (__fmul_rn / __fmaf_rn) to what nvcc + ptxas emitted before.
__global__ void k_rms_rope(const bf16* __restrict__ y, int M, const float* __restrict__ pe, const bf16* __restrict__ wq,
                           const bf16* __restrict__ wk, bf16* __restrict__ q, bf16* __restrict__ k, bf16* __restrict__ v) {
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int which = gw % 3, hh = (gw / 3) % NH, m = gw / (3 * NH);
    if (m >= M) return;
    const int c0 = which * IN + hh * HDIM + lane * 4;
    const uint2 a = *(const uint2*)(y + (size_t)m * QKV + c0);
    const size_t o = (size_t)m * IN + hh * HDIM + lane * 4;
    if (which == 2) {
        *(uint2*)(v + o) = a;
        return;
    }
    const float2 a0 = __bfloat1622float2(*(const __nv_bfloat162*)&a.x), a1 = __bfloat1622float2(*(const __nv_bfloat162*)&a.y);
    float e[4] = {a0.x, a0.y, a1.x, a1.y};
    const float ss = __fmaf_rn(e[3], e[3], __fmaf_rn(e[2], e[2], __fmaf_rn(e[0], e[0], __fmul_rn(e[1], e[1]))));
    const float r = rsqrtf(__fmaf_rn(warp_sum(ss), 1.f / HDIM, EPS));
    const bf16* nw = which ? wk : wq;
    for (int j = 0; j < 4; j++) e[j] = rbf(__fmul_rn(__fmul_rn(e[j], r), __bfloat162float(nw[lane * 4 + j])));
    const int src = lane < 12 ? lane + 12 : lane < 24 ? lane - 12 : lane;
    float p[4];
    for (int j = 0; j < 4; j++) p[j] = __shfl_sync(0xffffffffu, e[j], src);
    if (lane < 24) {
        const int pi = (lane < 12 ? lane : lane - 12) * 4;  // pair index of dim 0 of this lane
        const float4 t0 = *(const float4*)(pe + ((size_t)m * ROT + pi) * 2), t1 = *(const float4*)(pe + ((size_t)m * ROT + pi + 2) * 2);
        const float cs[4] = {t0.x, t0.z, t1.x, t1.z}, sn[4] = {t0.y, t0.w, t1.y, t1.w};
        for (int j = 0; j < 4; j++) {   // e cs -+ p sn: ptxas fused the e cs product (SASS)
            const float ps = __fmul_rn(p[j], sn[j]);
            e[j] = __fmaf_rn(e[j], cs[j], lane < 12 ? -ps : ps);
        }
    }
    bf16* dst = (which ? k : q) + o;
    *(__nv_bfloat162*)dst = __floats2bfloat162_rn(e[0], e[1]);
    *(__nv_bfloat162*)(dst + 2) = __floats2bfloat162_rn(e[2], e[3]);
}

// gated residual with a gate row per output row: x = bf16(fma(y, gate[mrow[m] * ldg + n], x)), y from gemm_i8_epi DQ_MUL
// (the old dequant_resid_rows on int32 accumulators did the same fma)
__global__ void k_resid_rows(const bf16* __restrict__ y, int M, int N, const bf16* __restrict__ gate, int ldg,
                             const int* __restrict__ mrow, bf16* __restrict__ x) {
    const size_t i = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) * 4;
    if (i >= (size_t)M * N) return;
    const int m = (int)(i / N), n = (int)(i % N);
    const uint2 yv = *(const uint2*)(y + i), gv = *(const uint2*)(gate + (size_t)mrow[m] * ldg + n);
    uint2 xv = *(const uint2*)(x + i);
    __nv_bfloat162* xh = (__nv_bfloat162*)&xv;
    const __nv_bfloat162 *yh = (const __nv_bfloat162*)&yv, *gh = (const __nv_bfloat162*)&gv;
    for (int j = 0; j < 2; j++) {
        const float2 a = __bfloat1622float2(yh[j]), g = __bfloat1622float2(gh[j]), b = __bfloat1622float2(xh[j]);
        xh[j] = __floats2bfloat162_rn(__fmaf_rn(a.x, g.x, b.x), __fmaf_rn(a.y, g.y, b.y));
    }
    *(uint2*)(x + i) = xv;
}

// token refiner: per-head RMSNorm of q, k inside the qkv rows (in place, one rounding)
__global__ void k_head_norm(bf16* __restrict__ qkv, int M, const bf16* __restrict__ wq, const bf16* __restrict__ wk) {
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int which = gw & 1, hh = (gw >> 1) % NH, m = gw / (2 * NH);
    if (m >= M) return;
    bf16* p = qkv + (size_t)m * QKV + which * IN + hh * HDIM + lane * 4;
    float e[4];
    for (int j = 0; j < 4; j++) e[j] = __bfloat162float(p[j]);
    const float r = rsqrtf(warp_sum(e[0] * e[0] + e[1] * e[1] + e[2] * e[2] + e[3] * e[3]) / HDIM + EPS);
    const bf16* nw = which ? wk : wq;
    for (int j = 0; j < 4; j++) p[j] = __float2bfloat16(e[j] * r * __bfloat162float(nw[lane * 4 + j]));
}

// rms_norm with weight (fused: one rounding); rows of D
__global__ void __launch_bounds__(256) k_rmsnorm(const bf16* __restrict__ x, const bf16* __restrict__ w, bf16* __restrict__ out) {
    __shared__ float red[32];
    const bf16* xr = x + (size_t)blockIdx.x * D;
    float s = 0.f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) { const float v = __bfloat162float(xr[j]); s += v * v; }
    const float r = rsqrtf(block_sum(s, red) / D + EPS);
    for (int j = threadIdx.x; j < D; j += blockDim.x)
        out[(size_t)blockIdx.x * D + j] = __float2bfloat16(__bfloat162float(xr[j]) * r * __bfloat162float(w[j]));
}

// final layer pre-head: bf16 rms_norm, then fp32 (n * (1 + scale) + shift) with the stream's t row
__global__ void __launch_bounds__(256) k_final_mod(const bf16* __restrict__ x, const bf16* __restrict__ w,
                                                   const float* __restrict__ fm, float* __restrict__ out) {
    __shared__ float red[32];
    const bf16* xr = x + (size_t)blockIdx.x * D;
    float s = 0.f;
    for (int j = threadIdx.x; j < D; j += blockDim.x) { const float v = __bfloat162float(xr[j]); s += v * v; }
    const float r = rsqrtf(block_sum(s, red) / D + EPS);
    for (int j = threadIdx.x; j < D; j += blockDim.x) {
        const float n = rbf(__bfloat162float(xr[j]) * r * __bfloat162float(w[j]));
        out[(size_t)blockIdx.x * D + j] = __fadd_rn(__fmul_rn(n, __fadd_rn(1.f, fm[D + j])), fm[j]);
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

__global__ void k_round_bf16(float* __restrict__ p, size_t n) {  // in place: fp32 -> bf16 value
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) p[i] = rbf(p[i]);
}

__global__ void k_add(bf16* __restrict__ x, const bf16* __restrict__ y, size_t n) {  // x = bf16(y + x)
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) x[i] = __float2bfloat16(__bfloat162float(y[i]) + __bfloat162float(x[i]));
}

__global__ void k_swiglu(const bf16* __restrict__ in, bf16* __restrict__ out, int M) {  // eager: silu(gate) * up, bf16 ops
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * FF) return;
    const size_t m = i / FF, j = i % FF;
    const float g = __bfloat162float(in[m * 2 * FF + j]), u = __bfloat162float(in[m * 2 * FF + FF + j]);
    out[i] = __float2bfloat16(rbf(g / (1.f + expf(-g))) * u);
}

__global__ void k_bias_bf16(const float* __restrict__ in, const float* __restrict__ b, int M, int N, bf16* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)M * N) out[i] = __float2bfloat16(in[i] + b[i % N]);
}
__global__ void k_bias_f32(float* __restrict__ x, const float* __restrict__ b, int M, int N) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)M * N) x[i] += b[i % N];
}

// ------------------------------------------------------------------------------------------------- profiling
// SHUTTER_PROF=1: events at phase boundaries; the time to the next mark is charged to the phase
const char* PH_NAME[] = {"adaln/embed/final", "norm1+mod+rq", "qkv gemm", "qkv deq+norm+rope", "attention", "o rq", "o gemm",
                         "o deq+res", "norm2+mod+rq", "fc1 gemm", "fc1 deq+swiglu+rq", "fc2 gemm", "fc2 deq+res"};
constexpr int NPH = 13;
bool g_prof = getenv("SHUTTER_PROF") != nullptr;
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
void prof_end(Dit* d, int S) {
    if (!g_prof) return;
    mark(d, 0);
    CK(cudaStreamSynchronize(d->st));
    for (size_t i = 0; i + 1 < g_ph.size(); i++) {
        float ms;
        CK(cudaEventElapsedTime(&ms, g_ev[i], g_ev[i + 1]));
        g_ms[g_ph[i]] += ms;
    }
    g_ph.clear();
    g_prof_n++;
    double tot = 0;
    for (int k = 0; k < NPH; k++) tot += g_ms[k];
    fprintf(stderr, "[prof] per forward (S=%d, avg of %d): total %.1f ms\n", S, g_prof_n, tot / g_prof_n);
    for (int k = 0; k < NPH; k++) fprintf(stderr, "[prof]   %-20s %8.2f ms  %4.1f%%\n", PH_NAME[k], g_ms[k] / g_prof_n, 100 * g_ms[k] / tot);
}

// ------------------------------------------------------------------------------------------------- pieces
void ensure(Dit* d, int S, int U) {
    if (S > d->cap) {
        void* ps[] = {d->x, d->q, d->k, d->v, d->att, d->aq, d->as, d->y, d->fo, d->mrow, d->pe};
        for (void* p : ps) cudaFree(p);
        d->cap = S;
        const int c = std::min(S, CH);
        d->x = (bf16*)dmalloc((size_t)S * D * 2);
        d->q = (bf16*)dmalloc((size_t)S * IN * 2);
        d->k = (bf16*)dmalloc((size_t)S * IN * 2);
        d->v = (bf16*)dmalloc((size_t)S * IN * 2);
        d->att = (bf16*)dmalloc((size_t)S * IN * 2);
        d->aq = (int8_t*)dmalloc((size_t)c * FF);
        d->as = (float*)dmalloc((size_t)c * 4);
        d->y = (bf16*)dmalloc((size_t)c * 2 * FF * 2);
        d->fo = (float*)dmalloc((size_t)c * D * 4);
        d->mrow = (int*)dmalloc((size_t)S * 4);
        d->pe = (float*)dmalloc((size_t)S * ROT * 2 * 4);
    }
    if (U > d->ucap) {
        void* ps[] = {d->tv, d->temb, d->mod, d->fmod};
        for (void* p : ps) cudaFree(p);
        d->ucap = U;
        d->tv = (float*)dmalloc(U * 4);
        d->temb = (float*)dmalloc(U * TD * 4);
        d->mod = (bf16*)dmalloc((size_t)NB * U * NADA * 2);
        d->fmod = (float*)dmalloc((size_t)U * NFIN * 4);
    }
}

// fp32 rows [n, K] -> fp32 linear (bf16-rounded weights) + bias -> bf16 rows [n, D]
void embed_rows(Dit* d, const float* in, int n, int K, const float* w, const float* b, bf16* out) {
    for (int r0 = 0; r0 < n; r0 += CH) {
        const int m = std::min(CH, n - r0);
        gemm_f32(d->lt, m, D, K, in + (size_t)r0 * K, w, d->fo);
        k_bias_bf16<<<nb((size_t)m * D, 256), 256, 0, d->st>>>(d->fo, b, m, D, out + (size_t)r0 * D);
    }
}

// target rows [x0, x0 + n) -> final norm/modulation (t row ti) -> fp32 head [n, N] + bias
void head(Dit* d, int x0, int n, int ti, const float* w, const float* b, int N, float* out) {
    for (int r0 = 0; r0 < n; r0 += CH) {
        const int m = std::min(CH, n - r0);
        k_final_mod<<<m, 256, 0, d->st>>>(d->x + (size_t)(x0 + r0) * D, d->fnw, d->fmod + (size_t)ti * NFIN, d->fo);
        gemm_f32(d->lt, m, N, D, d->fo, w, out + (size_t)r0 * N);
        k_bias_f32<<<nb((size_t)m * N, 256), 256, 0, d->st>>>(out + (size_t)r0 * N, b, m, N);
    }
}

void qn(Dit* d, int m, int K) {
    if (d->qnoise) k_qnoise<<<nb((size_t)m * K, 256), 256, 0, d->st>>>(d->aq, (size_t)m * K, d->qnoise, ++d->qcall);
}

void block(Dit* d, int i, int M, int U3) {
    const Block& b = d->b[i];
    cudaStream_t st = d->st;
    const bf16* mod = d->mod + (size_t)i * U3 * NMOD;  // rows: shift, scale, gate (msa), shift, scale, gate (mlp)
    const bool tap = i == 0 && (d->tap_qkv || d->tap_att || d->tap_fc1);
    const bool hk = d->hook && d->hook_on[i];
    for (int r0 = 0; r0 < M; r0 += CH) {
        const int m = std::min(CH, M - r0);
        mark(d, 1);
        rotquant_rmsmod(st, d->x + (size_t)r0 * D, m, D, b.n1, EPS, mod, NMOD, d->mrow + r0, d->aq, d->as);
        qn(d, m, D);
        mark(d, 2);
        gemm_i8_epi(st, DQ_MUL, m, QKV, D, d->aq, b.wqkv, d->as, b.sqkv, nullptr, nullptr,
                    hk ? d->hook_qkv + (size_t)r0 * QKV : d->y);
        mark(d, 3);
        if (hk) continue;
        if (tap && d->tap_qkv) CK(cudaMemcpyAsync(d->tap_qkv + (size_t)r0 * QKV, d->y, (size_t)m * QKV * 2, cudaMemcpyDeviceToDevice, st));
        k_rms_rope<<<nb((size_t)m * 3 * NH * 32, 256), 256, 0, st>>>(d->y, m, d->pe + (size_t)r0 * ROT * 2, b.nq, b.nk,
                                                                   d->q + (size_t)r0 * IN, d->k + (size_t)r0 * IN,
                                                                   d->v + (size_t)r0 * IN);
    }
    mark(d, 4);
    const bf16* att = d->att;
    if (hk) {   // the hook runs on torch's stream: hand over finished qkv rows, get back finished attention rows
        CK(cudaStreamSynchronize(st));
        d->hook(i);
        att = d->hook_att;
    } else if (d->attn_q8) attention_q8(st, d->q, d->k, d->v, d->att, M, M, NH, IN, IN);
    else attention_ex(st, d->q, d->k, d->v, d->att, M, M, nullptr, NH, IN, IN);
    if (tap && d->tap_att && !hk) CK(cudaMemcpyAsync(d->tap_att, d->att, (size_t)M * IN * 2, cudaMemcpyDeviceToDevice, st));
    for (int r0 = 0; r0 < M; r0 += CH) {
        const int m = std::min(CH, M - r0);
        bf16* x = d->x + (size_t)r0 * D;
        mark(d, 5);
        rotquant(st, RQ_PLAIN, att + (size_t)r0 * IN, IN, m, IN, nullptr, d->aq, d->as);
        qn(d, m, IN);
        mark(d, 6);
        gemm_i8_epi(st, DQ_MUL, m, D, IN, d->aq, b.wo, d->as, b.so, nullptr, nullptr, d->y);
        mark(d, 7);
        k_resid_rows<<<nb((size_t)m * D / 4, 256), 256, 0, st>>>(d->y, m, D, mod + 2 * D, NMOD, d->mrow + r0, x);
        mark(d, 8);
        rotquant_rmsmod(st, x, m, D, b.n2, EPS, mod + 3 * D, NMOD, d->mrow + r0, d->aq, d->as);
        qn(d, m, D);
        mark(d, 9);
        gemm_i8_epi(st, DQ_MUL, m, 2 * FF, D, d->aq, b.w1, d->as, b.s1, nullptr, nullptr, d->y);
        mark(d, 10);
        if (tap && d->tap_fc1) CK(cudaMemcpyAsync(d->tap_fc1 + (size_t)r0 * 2 * FF, d->y, (size_t)m * 2 * FF * 2, cudaMemcpyDeviceToDevice, st));
        rotquant(st, RQ_SWIGLU, d->y, 2 * FF, m, FF, nullptr, d->aq, d->as);
        qn(d, m, FF);
        mark(d, 11);
        gemm_i8_epi(st, DQ_MUL, m, D, FF, d->aq, b.w2, d->as, b.s2, nullptr, nullptr, d->y);
        mark(d, 12);
        k_resid_rows<<<nb((size_t)m * D / 4, 256), 256, 0, st>>>(d->y, m, D, mod + 5 * D, NMOD, d->mrow + r0, x);
    }
}
}  // namespace

extern "C" {

void* h3_dit_new(const char* path) {
    Dit* d = new Dit();
    if (!d->f.open(path)) DIE("cannot open %s", path);
    d->f.pfx = "model.diffusion_model.";   // Singularity keeps ComfyUI's full key prefix
    CK(cudaStreamCreate(&d->st));   // blocking on purpose (common.cuh)
    d->lt = lt_new(d->st);
    CK(cudaMallocHost(&g_stage, STAGE));
    const StTensor& tt = d->f.get("adaln_t_table");
    d->ntab = (int)(tt.bytes() / (TD * 4));
    d->tab = upload_f32(d, "adaln_t_table", (size_t)d->ntab * TD, false);  // buffers keep fp32
    d->inv = upload_f32(d, "rope.inv_freq", 16, false);
    d->cpw = (bf16*)upload(d, "condition_proj.weight", "BF16", (size_t)D * TXT);
    d->cpb = upload_f32(d, "condition_proj.bias", D, false);
    for (int r = 0; r < NR; r++) {
        const std::string p = "token_refiner.blocks." + std::to_string(r) + ".";
        Ref& R = d->r[r];
        R.wqkv = (bf16*)upload(d, p + "attn.qkv_proj.weight", "BF16", (size_t)QKV * D);
        R.wo = (bf16*)upload(d, p + "attn.out_proj.weight", "BF16", (size_t)D * IN);
        R.w1 = (bf16*)upload(d, p + "mlp.fc1.weight", "BF16", (size_t)2 * FF * D);
        R.w2 = (bf16*)upload(d, p + "mlp.fc2.weight", "BF16", (size_t)D * FF);
        R.n1 = (bf16*)upload(d, p + "norm1.weight", "BF16", D);
        R.n2 = (bf16*)upload(d, p + "norm2.weight", "BF16", D);
        R.nq = (bf16*)upload(d, p + "attn.q_norm.weight", "BF16", HDIM);
        R.nk = (bf16*)upload(d, p + "attn.k_norm.weight", "BF16", HDIM);
    }
    d->rfn = (bf16*)upload(d, "token_refiner.final_norm.weight", "BF16", D);
    // Gate only (SHUTTER_H3_FIRST_UNROUNDED=1, golden/test_pih3_pipe.py): ComfyUI's first forward after loading a pi-H3
    // model computes with the adapter's float "set" tensors as stored and casts them to bf16 for every later forward
    // (docs/pih3.md §2.3): all adaln weights, the adaln biases of blocks 46-49 and the final layer's adaln / heads. The
    // switch loads those unrounded and rounds them in place after the first forward; by default they are bf16 from the
    // start (= ComfyUI from its second forward on, and on every later job while the model stays loaded).
    const bool raw = getenv("SHUTTER_H3_FIRST_UNROUNDED") != nullptr;
    d->raw_first = raw;
    d->vpw = upload_f32(d, "video_patch_proj.weight", (size_t)D * VP, true);
    d->vpb = upload_f32(d, "video_patch_proj.bias", D, true);
    d->apw = upload_f32(d, "audio_patch_proj.weight", (size_t)D * AP, true);
    d->apb = upload_f32(d, "audio_patch_proj.bias", D, true);
    d->vow = upload_f32(d, "final_layer.video_out.weight", (size_t)VP * D, !raw);
    d->vob = upload_f32(d, "final_layer.video_out.bias", VP, !raw);
    d->aow = upload_f32(d, "final_layer.audio_out.weight", (size_t)AP * D, !raw);
    d->aob = upload_f32(d, "final_layer.audio_out.bias", AP, !raw);
    d->faw = upload_f32(d, "final_layer.adaln_proj.linear.weight", (size_t)NFIN * TD, !raw);
    d->fab = upload_f32(d, "final_layer.adaln_proj.linear.bias", NFIN, !raw);
    d->fnw = (bf16*)upload(d, "final_layer.norm.weight", "BF16", D);
    d->aw_all = (float*)dmalloc((size_t)NB * NADA * TD * 4);
    d->ab_all = (float*)dmalloc((size_t)NB * NADA * 4);
    for (int i = 0; i < NB; i++) {
        const std::string p = "blocks." + std::to_string(i) + ".";
        Block& b = d->b[i];
        for (const char* l : {"attn.qkv_proj", "attn.out_proj", "mlp.fc1", "mlp.fc2"}) check_convrot(d, p + l);
        b.wqkv = (int8_t*)upload(d, p + "attn.qkv_proj.weight", "I8", (size_t)QKV * D);
        b.sqkv = (float*)upload(d, p + "attn.qkv_proj.weight_scale", "F32", QKV);
        b.wo = (int8_t*)upload(d, p + "attn.out_proj.weight", "I8", (size_t)D * IN);
        b.so = (float*)upload(d, p + "attn.out_proj.weight_scale", "F32", D);
        b.w1 = (int8_t*)upload(d, p + "mlp.fc1.weight", "I8", (size_t)2 * FF * D);
        b.s1 = (float*)upload(d, p + "mlp.fc1.weight_scale", "F32", 2 * FF);
        b.w2 = (int8_t*)upload(d, p + "mlp.fc2.weight", "I8", (size_t)D * FF);
        b.s2 = (float*)upload(d, p + "mlp.fc2.weight_scale", "F32", D);
        b.n1 = (bf16*)upload(d, p + "norm1.weight", "BF16", D);
        b.n2 = (bf16*)upload(d, p + "norm2.weight", "BF16", D);
        b.nq = (bf16*)upload(d, p + "attn.q_norm.weight", "BF16", HDIM);
        b.nk = (bf16*)upload(d, p + "attn.k_norm.weight", "BF16", HDIM);
        b.aw = upload_f32(d, p + "adaln_proj.linear.weight", (size_t)NADA * TD, !raw, d->aw_all + (size_t)i * NADA * TD);
        b.ab = upload_f32(d, p + "adaln_proj.linear.bias", NADA, !(raw && i >= RAW_B0), d->ab_all + (size_t)i * NADA);
    }
    d->f.close();
    CK(cudaDeviceSynchronize());
    CK(cudaFreeHost(g_stage));
    g_stage = nullptr;
    return d;
}

// Overwrite one loaded weight with a merged value (engine/py/merge.py): src = device data, fp32 for an upload_f32
// tensor (bf16-rounded here when the load rounded it) or for a bf16 one (rounded to nearest), else the raw dtype.
int h3_dit_set(void* h, const char* name, const void* src, size_t bytes) {
    Dit* d = (Dit*)h;
    auto it = d->slots.find(name);
    if (it == d->slots.end()) DIE("h3_dit_set: no weight %s", name);
    const Slot& s = it->second;
    const bool f32_in = bytes == s.elems * 4;
    if (s.dtype == "f32" || (s.dtype == "BF16" && f32_in)) {
        if (!f32_in) DIE("h3_dit_set %s: %zu bytes, expected fp32 x %zu", name, bytes, s.elems);
        std::vector<float> v(s.elems);
        CK(cudaMemcpy(v.data(), src, bytes, cudaMemcpyDeviceToHost));
        if (s.dtype == "BF16") {
            std::vector<bf16> b(s.elems);
            for (size_t i = 0; i < s.elems; i++) b[i] = f2bf(v[i]);
            CK(cudaMemcpy(s.p, b.data(), s.elems * 2, cudaMemcpyHostToDevice));
        } else {
            if (s.round)
                for (auto& x : v) x = bf2f(f2bf(x));
            CK(cudaMemcpy(s.p, v.data(), bytes, cudaMemcpyHostToDevice));
        }
        return 0;
    }
    const size_t es = s.dtype == "I8" ? 1 : s.dtype == "F32" ? 4 : 2;
    if (bytes != s.elems * es) DIE("h3_dit_set %s: %zu bytes, expected %zu x %s", name, bytes, s.elems, s.dtype.c_str());
    CK(cudaMemcpy(s.p, src, bytes, cudaMemcpyDeviceToDevice));
    return 0;
}

// condition_proj + token refiner (bf16), once per job: ctx [L, 5120] bf16 -> out [L, 5376] bf16 (device).
// cp_tap (optional): condition_proj output.
int h3_dit_refine(void* h, const void* ctx, int L, void* out, void* cp_tap) {
    Dit* d = (Dit*)h;
    cudaStream_t st = d->st;
    bf16 *xa = (bf16*)dmalloc((size_t)L * D * 2), *ln = (bf16*)dmalloc((size_t)L * D * 2);
    bf16 *qkv = (bf16*)dmalloc((size_t)L * QKV * 2), *att = (bf16*)dmalloc((size_t)L * IN * 2);
    bf16 *g = (bf16*)dmalloc((size_t)L * 2 * FF * 2), *sw = (bf16*)dmalloc((size_t)L * FF * 2);
    float* tmp = (float*)dmalloc((size_t)L * D * 4);
    const size_t n = (size_t)L * D;
    gemm_bf16_f32(d->lt, L, D, TXT, (const bf16*)ctx, d->cpw, tmp);
    k_bias_bf16<<<nb(n, 256), 256, 0, st>>>(tmp, d->cpb, L, D, xa);
    if (cp_tap) CK(cudaMemcpyAsync(cp_tap, xa, n * 2, cudaMemcpyDeviceToDevice, st));
    for (int r = 0; r < NR; r++) {
        const Ref& R = d->r[r];
        k_rmsnorm<<<L, 256, 0, st>>>(xa, R.n1, ln);
        gemm_bf16(d->lt, L, QKV, D, ln, R.wqkv, qkv);
        k_head_norm<<<nb((size_t)L * 2 * NH * 32, 256), 256, 0, st>>>(qkv, L, R.nq, R.nk);
        attention_ex(st, qkv, qkv + IN, qkv + 2 * IN, att, L, L, nullptr, NH, QKV, IN);
        gemm_bf16(d->lt, L, D, IN, att, R.wo, ln);
        k_add<<<nb(n, 256), 256, 0, st>>>(xa, ln, n);
        k_rmsnorm<<<L, 256, 0, st>>>(xa, R.n2, ln);
        gemm_bf16(d->lt, L, 2 * FF, D, ln, R.w1, g);
        k_swiglu<<<nb((size_t)L * FF, 256), 256, 0, st>>>(g, sw, L);
        gemm_bf16(d->lt, L, D, FF, sw, R.w2, ln);
        k_add<<<nb(n, 256), 256, 0, st>>>(xa, ln, n);
    }
    k_rmsnorm<<<L, 256, 0, st>>>(xa, d->rfn, (bf16*)out);
    CK(cudaStreamSynchronize(st));
    void* ps[] = {xa, ln, qkv, att, g, sw, tmp};
    for (void* p : ps) cudaFree(p);
    CK(cudaGetLastError());   // launches here are unchecked: surface their errors in this call
    return 0;
}

// Prepare a job. kind[S] (host): 0 text (rows of ctx in order), 1 visual condition row (rows of vis [*, 96] fp32,
// patchified + noise-augmented), 2 audio condition row (rows of aud [*, 32]), 3 target audio, 4 target video; targets
// must be the last rows, audio then video. pos [S, 3] fp32 (host): rope positions (t, h, w).
int h3_dit_prepare(void* h, const void* ctx, int L, const void* vis, int nvis, const void* aud, int naud, const int* kind,
                   const float* pos, int S) {
    Dit* d = (Dit*)h;
    cudaStream_t st = d->st;
    int P = 0;
    while (P < S && kind[P] <= K_AUD) P++;
    int na = 0, nv = 0;
    for (int i = P; i < S; i++) {
        if (kind[i] == K_TAUD && nv == 0) na++;
        else if (kind[i] == K_TVID) nv++;
        else DIE("h3_dit_prepare: row %d kind %d after the condition rows", i, kind[i]);
    }
    d->S = S;
    d->P = P;
    d->na = na;
    d->nv = nv;
    ensure(d, S, 1);
    cudaFree(d->x0);
    d->x0 = (bf16*)dmalloc((size_t)std::max(P, 1) * D * 2);
    int it = 0, iv = 0, ia = 0;
    for (int a = 0; a < P;) {
        int b = a;
        while (b < P && kind[b] == kind[a]) b++;
        const int n = b - a;
        bf16* dst = d->x0 + (size_t)a * D;
        if (kind[a] == K_TEXT) {
            if (it + n > L) DIE("h3_dit_prepare: text rows %d > L %d", it + n, L);
            CK(cudaMemcpyAsync(dst, (const bf16*)ctx + (size_t)it * D, (size_t)n * D * 2, cudaMemcpyDeviceToDevice, st));
            it += n;
        } else if (kind[a] == K_VIS) {
            if (iv + n > nvis) DIE("h3_dit_prepare: visual rows %d > %d", iv + n, nvis);
            embed_rows(d, (const float*)vis + (size_t)iv * VP, n, VP, d->vpw, d->vpb, dst);
            iv += n;
        } else {
            if (ia + n > naud) DIE("h3_dit_prepare: audio rows %d > %d", ia + n, naud);
            embed_rows(d, (const float*)aud + (size_t)ia * AP, n, AP, d->apw, d->apb, dst);
            ia += n;
        }
        a = b;
    }
    float* dpos = (float*)dmalloc((size_t)S * 3 * 4);
    CK(cudaMemcpyAsync(dpos, pos, (size_t)S * 3 * 4, cudaMemcpyHostToDevice, st));
    k_rope_table<<<nb((size_t)S * ROT, 256), 256, 0, st>>>(dpos, d->inv, S, d->pe);
    CK(cudaStreamSynchronize(st));
    cudaFree(dpos);
    CK(cudaGetLastError());   // launches here are unchecked: surface their errors in this call
    return 0;
}

// One DiT call (ComfyUI _forward): target rows vrows [nv, 96] / arows [na, 32] fp32 (patchified / packed latents, device),
// U distinct timesteps tvals (host, sorted as ComfyUI's unique_t), mrow [S] (host) = t index * 3 + modality tag per row,
// iv / ia = t index of the video / audio stream. Writes the raw head outputs vout [nv, 96], aout [na, 32] fp32 (device).
int h3_dit_forward(void* h, const void* vrows, const void* arows, int U, const float* tvals, const int* mrow, int iv, int ia,
                   void* vout, void* aout) {
    Dit* d = (Dit*)h;
    cudaStream_t st = d->st;
    const int S = d->S, P = d->P;
    ensure(d, S, U);
    mark(d, 0);
    CK(cudaMemcpyAsync(d->tv, tvals, U * 4, cudaMemcpyHostToDevice, st));
    CK(cudaMemcpyAsync(d->mrow, mrow, (size_t)S * 4, cudaMemcpyHostToDevice, st));
    k_temb<<<1, 64, 0, st>>>(d->tab, d->ntab, d->tv, U, d->temb);
    k_ada<<<nb((size_t)NB * NADA, 256), 256, 0, st>>>(d->aw_all, d->ab_all, NB, NADA, d->temb, U, d->mod, nullptr);
    k_ada<<<nb(NFIN, 256), 256, 0, st>>>(d->faw, d->fab, 1, NFIN, d->temb, U, nullptr, d->fmod);
    if (P) CK(cudaMemcpyAsync(d->x, d->x0, (size_t)P * D * 2, cudaMemcpyDeviceToDevice, st));
    if (d->na) embed_rows(d, (const float*)arows, d->na, AP, d->apw, d->apb, d->x + (size_t)P * D);
    embed_rows(d, (const float*)vrows, d->nv, VP, d->vpw, d->vpb, d->x + (size_t)(P + d->na) * D);
    d->U3 = 3 * U;
    for (int i = 0; i < NB; i++) {
        block(d, i, S, 3 * U);
        if (d->tap_blocks) CK(cudaMemcpyAsync(d->tap_blocks + (size_t)i * S * D, d->x, (size_t)S * D * 2, cudaMemcpyDeviceToDevice, st));
    }
    mark(d, 0);
    if (d->na) head(d, P, d->na, ia, d->aow, d->aob, AP, (float*)aout);
    head(d, P + d->na, d->nv, iv, d->vow, d->vob, VP, (float*)vout);
    if (d->raw_first) {   // gate only: ComfyUI's cast to bf16 after its first forward (see h3_dit_new)
        const struct { float* p; size_t n; } set[] = {
            {d->aw_all, (size_t)NB * NADA * TD}, {d->ab_all + (size_t)RAW_B0 * NADA, (size_t)(NB - RAW_B0) * NADA},
            {d->faw, (size_t)NFIN * TD}, {d->fab, NFIN}, {d->vow, (size_t)VP * D}, {d->vob, VP}, {d->aow, (size_t)AP * D},
            {d->aob, AP}};
        for (const auto& s : set) k_round_bf16<<<nb(s.n, 256), 256, 0, st>>>(s.p, s.n);
        d->raw_first = 0;
    }
    if (g_prof && !d->tap_blocks) prof_end(d, S);
    else g_ph.clear();
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());   // launches here are unchecked: surface their errors in this call
    return 0;
}

// debug: block i alone on xin [S, D] -> xout, with the conditioning of the last forward(); qnoise as above
int h3_dit_block(void* h, int i, const void* xin, void* xout) {
    Dit* d = (Dit*)h;
    CK(cudaMemcpyAsync(d->x, xin, (size_t)d->S * D * 2, cudaMemcpyDeviceToDevice, d->st));
    block(d, i, d->S, d->U3);
    g_ph.clear();
    CK(cudaMemcpyAsync(xout, d->x, (size_t)d->S * D * 2, cudaMemcpyDeviceToDevice, d->st));
    CK(cudaStreamSynchronize(d->st));
    CK(cudaGetLastError());   // launches here are unchecked: surface their errors in this call
    return 0;
}
// After a forward: the last block's target-video rows (bf16 [nv, D], before final_layer) into vout and that call's time
// embedding rows (fp32 [U, 8], U = the forward's distinct timesteps) into temb = the DWD head's inputs (docs/pih3.md
// §3.5). Valid until the next forward. Returns nv.
int h3_dit_video_hidden(void* h, void* vout, void* temb) {
    Dit* d = (Dit*)h;
    CK(cudaMemcpyAsync(vout, d->x + (size_t)(d->P + d->na) * D, (size_t)d->nv * D * 2, cudaMemcpyDeviceToDevice, d->st));
    if (temb) CK(cudaMemcpyAsync(temb, d->temb, (size_t)(d->U3 / 3) * TD * 4, cudaMemcpyDeviceToDevice, d->st));
    CK(cudaStreamSynchronize(d->st));
    return d->nv;
}
// attention hook (option vsa): cb(i) computes block i's attention from qkv [S, QKV] (projection rows before q/k norm
// and rope) into att [S, IN] (device buffers owned by the caller, valid while the hook is set) for the blocks with
// on[i] != 0, and returns after its work has finished. cb nullptr = off (the normal path, unchanged).
void h3_dit_attn_hook(void* h, int (*cb)(int), void* qkv, void* att, const unsigned char* on) {
    Dit* d = (Dit*)h;
    d->hook = cb;
    d->hook_qkv = (bf16*)qkv;
    d->hook_att = (bf16*)att;
    for (int i = 0; i < NB; i++) d->hook_on[i] = cb && on ? on[i] : 0;
}
// the current job's rope table (fp32 [S, 48, 2] = (cos, sin) per pair, bf16 values) and block i's q / k norm weights
// (bf16 [128] each) into caller buffers — what the vsa hook's kernel needs besides qkv
void h3_dit_rope_copy(void* h, void* dst) {
    Dit* d = (Dit*)h;
    CK(cudaMemcpy(dst, d->pe, (size_t)d->S * ROT * 2 * 4, cudaMemcpyDeviceToDevice));
}
void h3_dit_qk_norm_copy(void* h, int i, void* q, void* k) {
    Dit* d = (Dit*)h;
    CK(cudaMemcpy(q, d->b[i].nq, HDIM * 2, cudaMemcpyDeviceToDevice));
    CK(cudaMemcpy(k, d->b[i].nk, HDIM * 2, cudaMemcpyDeviceToDevice));
}
void h3_dit_qnoise(void* h, int rate) { ((Dit*)h)->qnoise = rate; }
void h3_dit_attn_q8(void* h, int on) { ((Dit*)h)->attn_q8 = on; }

void h3_dit_taps(void* h, void* blocks, void* qkv0, void* att0, void* fc10) {
    Dit* d = (Dit*)h;
    d->tap_blocks = (bf16*)blocks;
    d->tap_qkv = (bf16*)qkv0;
    d->tap_att = (bf16*)att0;
    d->tap_fc1 = (bf16*)fc10;
}

void h3_dit_free(void* h) {
    Dit* d = (Dit*)h;
    cudaStreamSynchronize(d->st);
    void* ps[] = {d->tab, d->inv, d->cpb, d->vpw, d->vpb, d->apw, d->apb, d->faw, d->fab, d->vow, d->vob, d->aow, d->aob,
                  d->aw_all, d->ab_all, d->cpw, d->rfn, d->fnw, d->x0, d->pe, d->mrow, d->tv, d->temb, d->fmod, d->mod,
                  d->x, d->q, d->k, d->v, d->att, d->aq, d->as, d->fo, d->y};
    for (void* p : ps) cudaFree(p);
    for (int r = 0; r < NR; r++) {
        Ref& R = d->r[r];
        void* rs[] = {R.wqkv, R.wo, R.w1, R.w2, R.n1, R.n2, R.nq, R.nk};
        for (void* p : rs) cudaFree(p);
    }
    for (int i = 0; i < NB; i++) {
        Block& b = d->b[i];
        void* bs[] = {b.wqkv, b.wo, b.w1, b.w2, b.sqkv, b.so, b.s1, b.s2, b.n1, b.n2, b.nq, b.nk};
        for (void* p : bs) cudaFree(p);
    }
    lt_free(d->lt);
    attention_q8_release();   // the only user of the q8 workspace
    cudaStreamDestroy(d->st);
    delete d;
}
}  // extern "C"
