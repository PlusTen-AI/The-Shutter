// shutter engine — SeedVR2 7B DiT (NaDiT, int8 ConvRot W8A8), C API
// Follows ComfyUI comfy/ldm/seedvr/model.py (e9afbccc2) as ComfyUI runs it with --bf16-unet, cfg 1 (positive only):
//  - 36 MMDiT blocks, vid and txt weights per block; txt = txt_in(positive_conditioning [58, 5120]) is constant
//  - block: rms_adaln (no weight; scale arg bf16(bf16(sA + sB) - 1)) -> int8 linear -> per-head RMSNorm (+ RoPE on vid)
//    -> Swin window attention, every window = [its vid tokens | all 58 txt tokens], txt output = mean over windows
//    -> int8 proj_out (+bias) -> * gate (bf16) -> + residual (bf16); MLP the same with GELU-tanh, chunked 8192 rows
//  - windows (4,3,3) on the 720p grid, shifted by half a window on odd blocks; RoPE: first 60 dims of q/k, adjacent pairs,
//    window-local positions linspace(-1, 1), freqs linspace(1, 128, 10) * pi per axis (t, h, w)
//  - F16 weights are rounded to bf16 at load (--bf16-unet); emb_in / txt_in / vid_in / vid_out run in bf16
// Window attention: q/k/v rows are scattered into a packed buffer, one contiguous sequence per window
// ([vid rows of the window | 58 txt rows]), and the shared flash kernel (attention_ex) runs once per window.
#include "common.cuh"
#include "st.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int D = 3072, NH = 24, HD = 128, QKV = 3 * D, FF = 12288, NB = 36, TXT_IN = 5120, NT = 58;
constexpr int PIN = 132, PINP = 136, POUT = 64, TS = 256, NEMB = 6 * D, NPAIR = 30;
constexpr int CH = 8192;   // rows per linear chunk (bounds the work buffers, CH x 12288)
constexpr float EPS = 1e-5f;
// sinusoid freqs exp(arange(128) * -(log(10000) / 128)) exactly as torch computes them on the CPU (Sleef expf; libm
// differs by 1 ulp in 14 entries)
const float TFREQ[128] = {
    0x1.p+0f, 0x1.dc73f0p-1f, 0x1.bb5facp-1f, 0x1.9c9758p-1f, 0x1.7ff222p-1f, 0x1.654a0cp-1f, 0x1.4c7bc0p-1f, 0x1.356656p-1f,
    0x1.1feb34p-1f, 0x1.0beddcp-1f, 0x1.f2a79cp-2f, 0x1.d008c0p-2f, 0x1.afd134p-2f, 0x1.91d648p-2f, 0x1.75f034p-2f, 0x1.5bf9fep-2f,
    0x1.43d136p-2f, 0x1.2d55d4p-2f, 0x1.186a06p-2f, 0x1.04f210p-2f, 0x1.e5a844p-3f, 0x1.c3f06cp-3f, 0x1.a48fdap-3f, 0x1.875cf8p-3f,
    0x1.6c310ep-3f, 0x1.52e812p-3f, 0x1.3b6080p-3f, 0x1.257b20p-3f, 0x1.111aeep-3f, 0x1.fc49bep-4f, 0x1.d8ffaap-4f, 0x1.b828c8p-4f,
    0x1.999998p-4f, 0x1.7d2990p-4f, 0x1.62b2f0p-4f, 0x1.4a12aep-4f, 0x1.33281ap-4f, 0x1.1dd4d8p-4f, 0x1.09fc98p-4f, 0x1.ef0a22p-5f,
    0x1.ccab82p-5f, 0x1.acafc6p-5f, 0x1.8eec80p-5f, 0x1.733a30p-5f, 0x1.59742cp-5f, 0x1.417836p-5f, 0x1.2b2690p-5f, 0x1.166196p-5f,
    0x1.030dc4p-5f, 0x1.e222eep-6f, 0x1.c0a9a2p-6f, 0x1.a18350p-6f, 0x1.84869cp-6f, 0x1.698d24p-6f, 0x1.507312p-6f, 0x1.39172cp-6f,
    0x1.235a74p-6f, 0x1.0f2012p-6f, 0x1.f89a60p-7f, 0x1.d591ccp-7f, 0x1.b4f7e2p-7f, 0x1.96a16ap-7f, 0x1.7a661cp-7f, 0x1.6020a0p-7f,
    0x1.47ae14p-7f, 0x1.30ee0cp-7f, 0x1.1bc25cp-7f, 0x1.080eeep-7f, 0x1.eb735ap-8f, 0x1.c9548ap-8f, 0x1.a9942cp-8f, 0x1.8c0814p-8f,
    0x1.708934p-8f, 0x1.56f304p-8f, 0x1.3f2398p-8f, 0x1.28fb5ep-8f, 0x1.145ceap-8f, 0x1.012cf8p-8f, 0x1.dea41ap-9f, 0x1.bd68f4p-9f,
    0x1.9e7c66p-9f, 0x1.81b584p-9f, 0x1.66ee1ap-9f, 0x1.4e02a6p-9f, 0x1.36d21cp-9f, 0x1.213db0p-9f, 0x1.0d28dap-9f, 0x1.f4f1e0p-10f,
    0x1.d22a50p-10f, 0x1.b1cce8p-10f, 0x1.93aeb2p-10f, 0x1.77a7d4p-10f, 0x1.5d931cp-10f, 0x1.454deep-10f, 0x1.2eb816p-10f, 0x1.19b3b2p-10f,
    0x1.0624dcp-10f, 0x1.e7e344p-11f, 0x1.c603c6p-11f, 0x1.a67e48p-11f, 0x1.892914p-11f, 0x1.6ddd3cp-11f, 0x1.547688p-11f, 0x1.3cd344p-11f,
    0x1.26d428p-11f, 0x1.125c02p-11f, 0x1.fe9f58p-12f, 0x1.db2bc8p-12f, 0x1.ba2e44p-12f, 0x1.9b7b26p-12f, 0x1.7ee9acp-12f, 0x1.6453f6p-12f,
    0x1.4b96c0p-12f, 0x1.349140p-12f, 0x1.1f24eap-12f, 0x1.0b3548p-12f, 0x1.f1501ap-13f, 0x1.cec918p-13f, 0x1.aea7c4p-13f, 0x1.90c180p-13f,
    0x1.74eea6p-13f, 0x1.5b0a54p-13f, 0x1.42f232p-13f, 0x1.2c864cp-13f, 0x1.17a8dap-13f, 0x1.043e50p-13f, 0x1.e459bap-14f, 0x1.c2b91cp-14f,
};

struct Lin {   // int8 ConvRot linear: weight [N, K] int8, per-row scale [N] fp32, bias [N] bf16 (or null)
    int8_t* w = nullptr;
    float* s = nullptr;
    bf16* b = nullptr;
};
struct Branch {   // one modality of a block
    Lin qkv, out, fc1, fc2;
    bf16 *nq, *nk;
};
struct WinSet {   // window plan for one grid and one window method (normal / shifted)
    std::vector<int> win;    // [nwin * 6]: t0 t1 h0 h1 w0 w1
    std::vector<int> ps;     // packed start row per window
    std::vector<int> nv;     // vid tokens per window
    int M = 0, P = 0;        // vid tokens, packed rows (vid + 58 per window)
    int *prow = nullptr;     // [M] raster token -> packed row
    int *wrow = nullptr;     // [M] window-order token -> packed row (taps)
    int *trow = nullptr;     // [nwin] first txt row of each window
    float* pe = nullptr;     // [M][30][2] (cos, sin) per raster token
};

struct Dit {
    cudaStream_t st;
    Lt* lt;
    StFile f;
    size_t bytes = 0, peak = 0;
    Branch v[NB], t[NB];
    bf16 *ada = nullptr;     // [NB][2][6][D] learned (vid, txt) x (attn shift, scale, gate, mlp shift, scale, gate)
    bf16 *e1w, *e1b, *e2w, *e2b, *e3w, *e3b, *viw, *vib, *vow, *vob;
    bf16* txt0 = nullptr;    // [58, D] txt_in(positive_conditioning)
    float* tfreq = nullptr;  // [128] sinusoid frequencies
    // per forward
    bf16 *emb = nullptr, *mod = nullptr;   // [NEMB], [NB][2][6][D] = (shift, scale - 1, gate) per layer
    bf16 *e256 = nullptr, *eh = nullptr;
    float* ef = nullptr;
    // per grid
    int T = 0, H = 0, W = 0, M = 0;
    WinSet ws[2];
    // work
    int cap = 0, pcap = 0;
    bf16 *x = nullptr, *Q = nullptr, *K = nullptr, *V = nullptr, *O = nullptr, *h = nullptr, *hb = nullptr, *pin = nullptr;
    bf16 *xt = nullptr, *ht = nullptr, *att_t = nullptr;
    int8_t* aq = nullptr;
    float *as = nullptr, *fo = nullptr;
    bf16* y = nullptr;       // [CH, QKV] dequantized qkv
    // debug taps (device pointers owned by the caller)
    bf16 *tap_vid = nullptr, *tap_txt = nullptr, *tap_emb = nullptr, *tap_qk = nullptr, *tap_att = nullptr;
};

Dit* g_d = nullptr;   // for allocation accounting in dmalloc
void* dmalloc(size_t n) {
    void* p;
    CK(cudaMalloc(&p, n));
    if (g_d) {
        g_d->bytes += n;
        g_d->peak = std::max(g_d->peak, g_d->bytes);
    }
    return p;
}
void dfree(void* p, size_t n) {
    if (!p) return;
    cudaFree(p);
    if (g_d) g_d->bytes -= n;
}

const StTensor& tensor(Dit* d, const std::string& name, size_t elems) {
    const StTensor& t = d->f.get(name);
    const size_t es = t.dtype == "I8" ? 1 : t.dtype == "F32" ? 4 : 2;
    if (t.bytes() != elems * es) DIE("%s: %llu bytes, expected %zu elems", name.c_str(), (unsigned long long)t.bytes(), elems);
    return t;
}

// raw upload (I8 / F32)
void* upload_raw(Dit* d, const std::string& name, const char* dtype, size_t elems) {
    const StTensor& t = tensor(d, name, elems);
    if (t.dtype != dtype) DIE("%s: dtype %s, expected %s", name.c_str(), t.dtype.c_str(), dtype);
    void* p = dmalloc(t.bytes());
    CK(cudaMemcpy(p, d->f.ptr(t), t.bytes(), cudaMemcpyHostToDevice));
    return p;
}

// F16 / BF16 / F32 -> bf16 (round to nearest even, as --bf16-unet loads them); rows of k padded to kp with zeros
bf16* upload_bf16(Dit* d, const std::string& name, size_t rows, size_t k = 1, size_t kp = 0) {
    if (!kp) kp = k;
    const StTensor& t = tensor(d, name, rows * k);
    const uint8_t* p = d->f.ptr(t);
    std::vector<bf16> h(rows * kp, f2bf(0.f));
    for (size_t r = 0; r < rows; r++)
        for (size_t c = 0; c < k; c++) {
            const size_t i = r * k + c;
            float v;
            if (t.dtype == "F16") v = __half2float(((const __half*)p)[i]);
            else if (t.dtype == "BF16") v = bf2f(((const bf16*)p)[i]);
            else if (t.dtype == "F32") memcpy(&v, p + 4 * i, 4);
            else DIE("%s: dtype %s", name.c_str(), t.dtype.c_str());
            h[r * kp + c] = f2bf(v);
        }
    bf16* o = (bf16*)dmalloc(h.size() * 2);
    CK(cudaMemcpy(o, h.data(), h.size() * 2, cudaMemcpyHostToDevice));
    return o;
}

Lin load_lin(Dit* d, const std::string& p, int N, int K, bool bias) {
    const StTensor& q = d->f.get(p + ".comfy_quant");
    std::string js((const char*)d->f.ptr(q), q.bytes());
    if (js.find("int8_tensorwise") == std::string::npos || js.find("\"convrot\": true") == std::string::npos ||
        js.find("256") == std::string::npos)
        DIE("%s: unexpected quant config %s", p.c_str(), js.c_str());
    Lin l;
    l.w = (int8_t*)upload_raw(d, p + ".weight", "I8", (size_t)N * K);
    l.s = (float*)upload_raw(d, p + ".weight_scale", "F32", N);
    if (bias) l.b = upload_bf16(d, p + ".bias", N);
    return l;
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
// sinusoidal timestep embedding (flip_sin_to_cos=False, shift 0): [sin(t f) | cos(t f)] -> bf16
__global__ void k_tsin(const float* __restrict__ f, float t, bf16* __restrict__ out) {
    const int i = threadIdx.x;
    const float a = t * f[i & 127];
    out[i] = __float2bfloat16(i < 128 ? sinf(a) : cosf(a));
}

// out = bf16(in + bias); act 1: silu after the rounding (nn.SiLU on the bf16 linear output)
__global__ void k_bias(const float* __restrict__ in, const bf16* __restrict__ b, int M, int N, int act, bf16* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * N) return;
    float y = rbf(in[i] + __bfloat162float(b[i % N]));
    if (act) y = y / (1.f + expf(-y));
    out[i] = __float2bfloat16(y);
}

// AdaSingle: emb.reshape(D, 2, 3)[:, layer, (shift, scale, gate)] + learned -> bf16 adds as torch does them;
// the "in" scale is handed to rms_adaln as bf16(scale - 1)
__global__ void k_mod(const bf16* __restrict__ emb, const bf16* __restrict__ ada, bf16* __restrict__ mod) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;   // over NB * 2 * 6 * D
    if (i >= NB * 2 * 6 * D) return;
    const int c = i % D, k = (i / D) % 6;
    float v = rbf(__bfloat162float(emb[c * 6 + k]) + __bfloat162float(ada[i]));
    if (k % 3 == 1) v = rbf(v - 1.f);
    mod[i] = __float2bfloat16(v);
}

// patchify (1, 2, 2): token (t, i, j), feature (ph * 2 + pw) * 33 + c over [noise 16 | cond 17]; 4 zero pad columns
__global__ void k_patch_in(const bf16* __restrict__ nz, const bf16* __restrict__ cd, int T, int H, int W, bf16* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int h = H / 2, w = W / 2;
    const size_t M = (size_t)T * h * w;
    if (i >= M * PINP) return;
    const int f = (int)(i % PINP);
    const size_t m = i / PINP;
    if (f >= PIN) { out[i] = __float2bfloat16(0.f); return; }
    const int c = f % 33, p = f / 33, ph = p >> 1, pw = p & 1;
    const int jj = (int)(m % w), ii = (int)((m / w) % h), tt = (int)(m / ((size_t)w * h));
    const size_t px = ((size_t)tt * H + 2 * ii + ph) * W + 2 * jj + pw;
    const size_t plane = (size_t)T * H * W;
    out[i] = c < 16 ? nz[c * plane + px] : cd[(c - 16) * plane + px];
}

// unpatchify: y [M, 64] fp32 (+ bias, bf16 rounding) -> v [16, T, H, W] bf16
__global__ void k_patch_out(const float* __restrict__ y, const bf16* __restrict__ b, int r0, int m, int T, int H, int W,
                            bf16* __restrict__ out) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= m * POUT) return;
    const int o = i % POUT, row = r0 + i / POUT, h = H / 2, w = W / 2;
    const int c = o % 16, p = o / 16, ph = p >> 1, pw = p & 1;
    const int jj = row % w, ii = (row / w) % h, tt = row / (w * h);
    out[(size_t)c * T * H * W + ((size_t)tt * H + 2 * ii + ph) * W + 2 * jj + pw] = __float2bfloat16(y[i] + __bfloat162float(b[o]));
}

// rms_adaln (comfy_kitchen, fused): bf16(rmsnorm(x) * (1 + scale) + shift), fp32 math, rows of D
__global__ void __launch_bounds__(256) k_rms_adaln(const bf16* __restrict__ x, const bf16* __restrict__ shift,
                                                   const bf16* __restrict__ scale, bf16* __restrict__ out) {
    __shared__ float red[32];
    const bf16* xr = x + (size_t)blockIdx.x * D;
    float v[D / 256], s = 0.f;
#pragma unroll
    for (int k = 0; k < D / 256; k++) { v[k] = __bfloat162float(xr[threadIdx.x + k * 256]); s += v[k] * v[k]; }
    const float r = rsqrtf(block_sum(s, red) / D + EPS);
#pragma unroll
    for (int k = 0; k < D / 256; k++) {
        const int j = threadIdx.x + k * 256;
        out[(size_t)blockIdx.x * D + j] = __float2bfloat16(v[k] * r * (1.f + __bfloat162float(scale[j])) + __bfloat162float(shift[j]));
    }
}

// qkv y = bf16((acc * xs) * ws) (gemm_i8_epi DQ_MUL) -> per-head RMSNorm (weight, one rounding) -> RoPE on the first 60
// dims (adjacent pairs, vid only) -> rows of the packed q/k/v. One warp per (row, head, q|k|v), 4 dims per lane.
// vid: dst row prow[m], rope table pe; txt: one copy per window at trow[w] + m.
// fp32 steps pinned (__fmul_rn / __fmaf_rn) to what nvcc emitted before, so the bits don't move with unrelated edits.
__global__ void k_qkv(const bf16* __restrict__ y, int M, const int* __restrict__ prow, const float* __restrict__ pe,
                      const int* __restrict__ trow, int ntw, const bf16* __restrict__ wq, const bf16* __restrict__ wk,
                      bf16* __restrict__ Q, bf16* __restrict__ K, bf16* __restrict__ V) {
    const int gw = (blockIdx.x * blockDim.x + threadIdx.x) >> 5, lane = threadIdx.x & 31;
    const int which = gw % 3, hh = (gw / 3) % NH, m = gw / (3 * NH);
    if (m >= M) return;
    const int c0 = which * D + hh * HD + lane * 4;
    const uint2 a = *(const uint2*)(y + (size_t)m * QKV + c0);
    const float2 a0 = __bfloat1622float2(*(const __nv_bfloat162*)&a.x), a1 = __bfloat1622float2(*(const __nv_bfloat162*)&a.y);
    float e[4] = {a0.x, a0.y, a1.x, a1.y};
    if (which < 2) {
        const float ss = __fmaf_rn(e[3], e[3], __fmaf_rn(e[2], e[2], __fmaf_rn(e[0], e[0], __fmul_rn(e[1], e[1]))));
        const float r = rsqrtf(__fmaf_rn(warp_sum(ss), 1.f / HD, EPS));
        const bf16* nw = which ? wk : wq;
        for (int j = 0; j < 4; j++) e[j] = rbf(__fmul_rn(__fmul_rn(e[j], r), __bfloat162float(nw[lane * 4 + j])));
        if (pe && lane < NPAIR / 2) {
            const float4 t = *(const float4*)(pe + ((size_t)m * NPAIR + 2 * lane) * 2);   // (cos, sin) of pairs 2l, 2l+1
            const float x0 = e[0], x1 = e[1], x2 = e[2], x3 = e[3];
            e[0] = __fmaf_rn(t.x, x0, -__fmul_rn(t.y, x1));   // ptxas fused the a*b - c*d this way (SASS)
            e[1] = __fmaf_rn(t.y, x0, __fmul_rn(t.x, x1));
            e[2] = __fmaf_rn(t.z, x2, -__fmul_rn(t.w, x3));
            e[3] = __fmaf_rn(t.w, x2, __fmul_rn(t.z, x3));
        }
    }
    bf16* dst = which == 0 ? Q : which == 1 ? K : V;
    const __nv_bfloat162 o0 = __floats2bfloat162_rn(e[0], e[1]), o1 = __floats2bfloat162_rn(e[2], e[3]);
    const int col = hh * HD + lane * 4;
    if (prow) {
        bf16* p = dst + (size_t)prow[m] * D + col;
        *(__nv_bfloat162*)p = o0;
        *(__nv_bfloat162*)(p + 2) = o1;
    } else {
        for (int wi = 0; wi < ntw; wi++) {
            bf16* p = dst + (size_t)(trow[wi] + m) * D + col;
            *(__nv_bfloat162*)p = o0;
            *(__nv_bfloat162*)(p + 2) = o1;
        }
    }
}

// rows: out[i] = in[idx[i]] (D bf16 per row)
__global__ void k_rows(const bf16* __restrict__ in, const int* __restrict__ idx, bf16* __restrict__ out) {
    const uint4* s = (const uint4*)(in + (size_t)idx[blockIdx.x] * D);
    uint4* o = (uint4*)(out + (size_t)blockIdx.x * D);
    for (int j = threadIdx.x; j < D / 8; j += blockDim.x) o[j] = s[j];
}

// txt attention output = mean over the window copies (fp32 sum * (1 / n), like torch's mean)
__global__ void k_txt_mean(const bf16* __restrict__ O, const int* __restrict__ trow, int ntw, bf16* __restrict__ out) {
    const int r = blockIdx.x;
    const float inv = 1.f / (float)ntw;
    for (int c = threadIdx.x; c < D; c += blockDim.x) {
        float s = 0.f;
        for (int wi = 0; wi < ntw; wi++) s += __bfloat162float(O[(size_t)(trow[wi] + r) * D + c]);
        out[(size_t)r * D + c] = __float2bfloat16(s * inv);
    }
}

// RoPE table for one window set: angle = pos[axis] * freq[k] (fp32), (cos, sin)
__global__ void k_rope_table(const float* __restrict__ pos, const float* __restrict__ freq, int M, float* __restrict__ pe) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * NPAIR) return;
    const int m = i / NPAIR, k = i % NPAIR;
    const float a = pos[m * 3 + k / 10] * freq[k % 10];
    pe[2 * i] = cosf(a);
    pe[2 * i + 1] = sinf(a);
}

// ------------------------------------------------------------------------------------------------- host pieces
// ComfyUI make_720Pwindows_bysize / make_shifted_720Pwindows_bysize, window list order (w outer, t inner)
std::vector<int> windows(int t, int h, int w, bool shifted) {
    const double scale = std::sqrt(3600.0 / ((double)h * w));
    const int rh = (int)std::nearbyint(h * scale), rw = (int)std::nearbyint(w * scale);   // Python round(): ties to even
    const int wh = (rh + 2) / 3, ww = (rw + 2) / 3, wt = (std::min(t, 30) + 3) / 4;
    std::vector<int> out;
    auto add = [&](int t0, int t1, int h0, int h1, int w0, int w1) { out.insert(out.end(), {t0, t1, h0, h1, w0, w1}); };
    if (!shifted) {
        const int nt = (t + wt - 1) / wt, nh = (h + wh - 1) / wh, nw = (w + ww - 1) / ww;
        for (int iw = 0; iw < nw; iw++)
            for (int ih = 0; ih < nh; ih++)
                for (int it = 0; it < nt; it++)
                    add(it * wt, std::min((it + 1) * wt, t), ih * wh, std::min((ih + 1) * wh, h), iw * ww, std::min((iw + 1) * ww, w));
        return out;
    }
    const double st = wt < t ? 0.5 : 0, sh = wh < h ? 0.5 : 0, sw = ww < w ? 0.5 : 0;
    const int nt = st > 0 ? (int)std::ceil((t - st) / wt) + 1 : 1, nh = sh > 0 ? (int)std::ceil((h - sh) / wh) + 1 : 1,
              nw = sw > 0 ? (int)std::ceil((w - sw) / ww) + 1 : 1;
    auto lo = [](int i, double s, int n) { return std::max((int)((i - s) * n), 0); };
    auto hi = [](int i, double s, int n, int lim) { return std::min((int)((i - s + 1) * n), lim); };
    for (int iw = 0; iw < nw; iw++) {
        if (hi(iw, sw, ww, w) <= lo(iw, sw, ww)) continue;
        for (int ih = 0; ih < nh; ih++) {
            if (hi(ih, sh, wh, h) <= lo(ih, sh, wh)) continue;
            for (int it = 0; it < nt; it++) {
                if (hi(it, st, wt, t) <= lo(it, st, wt)) continue;
                add(lo(it, st, wt), hi(it, st, wt, t), lo(ih, sh, wh), hi(ih, sh, wh, h), lo(iw, sw, ww), hi(iw, sw, ww, w));
            }
        }
    }
    return out;
}

// torch.linspace(-1, 1, n) as the CUDA kernel computes it (float step, fma from both ends)
float lin(int i, int n) {
    if (n == 1) return -1.f;
    const float step = 2.f / (float)(n - 1);
    return i < n / 2 ? std::fma(step, (float)i, -1.f) : std::fma(-step, (float)(n - 1 - i), 1.f);
}

void free_ws(WinSet& s) {
    dfree(s.prow, (size_t)s.M * 4);
    dfree(s.wrow, (size_t)s.M * 4);
    dfree(s.trow, s.ps.size() * 4);
    dfree(s.pe, (size_t)s.M * NPAIR * 2 * 4);
    s = WinSet();
}

void build_ws(Dit* d, WinSet& s, bool shifted, const float* dfreq) {
    const int t = d->T, h = d->H / 2, w = d->W / 2, M = d->M;
    s.win = windows(t, h, w, shifted);
    const int nwin = (int)s.win.size() / 6;
    std::vector<int> prow(M), wrow, trow(nwin);
    std::vector<float> pos((size_t)M * 3);
    s.ps.resize(nwin);
    s.nv.resize(nwin);
    int p = 0;
    for (int k = 0; k < nwin; k++) {
        const int* b = &s.win[6 * k];
        const int ft = b[1] - b[0], fh = b[3] - b[2], fw = b[5] - b[4];
        s.ps[k] = p;
        s.nv[k] = ft * fh * fw;
        for (int a = 0; a < ft; a++)
            for (int c = 0; c < fh; c++)
                for (int e = 0; e < fw; e++) {
                    const int m = ((b[0] + a) * h + b[2] + c) * w + b[4] + e;
                    prow[m] = p;
                    wrow.push_back(p);
                    pos[3 * (size_t)m] = lin(a, ft);
                    pos[3 * (size_t)m + 1] = lin(c, fh);
                    pos[3 * (size_t)m + 2] = lin(e, fw);
                    p++;
                }
        trow[k] = p;
        p += NT;
    }
    if ((int)wrow.size() != M) DIE("windows cover %zu of %d tokens", wrow.size(), M);
    s.M = M;
    s.P = p;
    s.prow = (int*)dmalloc((size_t)M * 4);
    s.wrow = (int*)dmalloc((size_t)M * 4);
    s.trow = (int*)dmalloc((size_t)nwin * 4);
    s.pe = (float*)dmalloc((size_t)M * NPAIR * 2 * 4);
    float* dpos = (float*)dmalloc(pos.size() * 4);
    CK(cudaMemcpy(s.prow, prow.data(), (size_t)M * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s.wrow, wrow.data(), (size_t)M * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(s.trow, trow.data(), (size_t)nwin * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(dpos, pos.data(), pos.size() * 4, cudaMemcpyHostToDevice));
    k_rope_table<<<nb((size_t)M * NPAIR, 256), 256, 0, d->st>>>(dpos, dfreq, M, s.pe);
    CK(cudaStreamSynchronize(d->st));
    dfree(dpos, pos.size() * 4);
}

#define FREE(p, n) do { dfree(p, n); p = nullptr; } while (0)
void set_grid(Dit* d, int T, int H, int W) {
    if (T == d->T && H == d->H && W == d->W) return;
    if (H % 2 || W % 2 || T < 1) DIE("sv2_dit: latent grid %dx%dx%d (H, W must be even)", T, H, W);
    for (WinSet& s : d->ws) free_ws(s);
    d->T = T, d->H = H, d->W = W, d->M = T * (H / 2) * (W / 2);
    // RoPE freqs: torch.linspace(1, 128, 10) (CPU, float step) * pi
    float fr[10];
    const float step = 127.f / 9.f;
    for (int k = 0; k < 10; k++) {
        const volatile float p = step * (float)(k < 5 ? k : 9 - k);
        fr[k] = (k < 5 ? 1.f + p : 128.f - p) * (float)M_PI;
    }
    float* dfreq = (float*)dmalloc(40);
    CK(cudaMemcpy(dfreq, fr, 40, cudaMemcpyHostToDevice));
    build_ws(d, d->ws[0], false, dfreq);
    build_ws(d, d->ws[1], true, dfreq);
    dfree(dfreq, 40);
    const int M = d->M, P = std::max(d->ws[0].P, d->ws[1].P);
    if (M > d->cap) {
        const size_t c = d->cap;
        FREE(d->x, c * D * 2);
        FREE(d->pin, c * PINP * 2);
        d->cap = M;
        d->x = (bf16*)dmalloc((size_t)M * D * 2);
        d->pin = (bf16*)dmalloc((size_t)M * PINP * 2);
    }
    if (P > d->pcap) {
        const size_t c = d->pcap;
        FREE(d->Q, c * D * 2);
        FREE(d->K, c * D * 2);
        FREE(d->V, c * D * 2);
        FREE(d->O, c * D * 2);
        d->pcap = P;
        d->Q = (bf16*)dmalloc((size_t)P * D * 2);
        d->K = (bf16*)dmalloc((size_t)P * D * 2);
        d->V = (bf16*)dmalloc((size_t)P * D * 2);
        d->O = (bf16*)dmalloc((size_t)P * D * 2);
    }
}

// SHUTTER_PROF=1: events at phase boundaries, totals per forward on stderr
const char* PH_NAME[] = {"embed/in/out", "norm+qkv", "attention", "unwindow", "proj_out", "mlp"};
constexpr int NPH = 6;
const bool g_prof = getenv("SHUTTER_PROF") != nullptr;
std::vector<cudaEvent_t> g_ev;
std::vector<int> g_ph;
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
    double ms[NPH] = {}, tot = 0;
    for (size_t i = 0; i + 1 < g_ph.size(); i++) {
        float t;
        CK(cudaEventElapsedTime(&t, g_ev[i], g_ev[i + 1]));
        ms[g_ph[i]] += t;
        tot += t;
    }
    g_ph.clear();
    fprintf(stderr, "[prof] forward M=%d: %.1f ms\n", d->M, tot);
    for (int k = 0; k < NPH; k++) fprintf(stderr, "[prof]   %-14s %9.1f ms %5.1f%%\n", PH_NAME[k], ms[k], 100 * ms[k] / tot);
}

// int8 linear over rows: rotquant (ConvRot + row int8) -> GEMM with the epilogue fused (gemm_dq.cu), y = bf16(acc * xs *
// ws + bias); DQ_GATE: out = bf16(out + bf16(y * gate)) (AdaSingle out + residual); DQ_GELU: out = bf16(gelu_tanh(y))
void lin8(Dit* d, int epi, const Lin& l, const bf16* in, int m, int N, int Kd, const bf16* gate, bf16* out) {
    rotquant(d->st, RQ_PLAIN, in, Kd, m, Kd, nullptr, d->aq, d->as);
    gemm_i8_epi(d->st, epi, m, N, Kd, d->aq, l.w, d->as, l.s, l.b, gate, out);
}

// attn -> residual, then mlp -> residual, for one modality's rows [x, x + M) (vid in CH chunks, txt in one)
void mlp(Dit* d, const Branch& b, const bf16* mod, bf16* x, int M) {
    for (int r0 = 0; r0 < M; r0 += CH) {
        const int m = std::min(CH, M - r0);
        bf16* xr = x + (size_t)r0 * D;
        k_rms_adaln<<<m, 256, 0, d->st>>>(xr, mod + 3 * D, mod + 4 * D, d->h);
        lin8(d, DQ_GELU, b.fc1, d->h, m, FF, D, nullptr, d->hb);
        lin8(d, DQ_GATE, b.fc2, d->hb, m, D, FF, mod + 5 * D, xr);
    }
}

void block(Dit* d, int i) {
    cudaStream_t st = d->st;
    const WinSet& s = d->ws[i & 1];
    const int M = d->M, nwin = (int)s.ps.size();
    const bf16* mv = d->mod + (size_t)(i * 2) * 6 * D;
    const bf16* mt = mv + 6 * D;
    const Branch &bv = d->v[i], &bt = d->t[i];
    // q, k, v into the packed window sequences
    mark(d, 1);
    for (int r0 = 0; r0 < M; r0 += CH) {
        const int m = std::min(CH, M - r0);
        k_rms_adaln<<<m, 256, 0, st>>>(d->x + (size_t)r0 * D, mv, mv + D, d->h);
        rotquant(st, RQ_PLAIN, d->h, D, m, D, nullptr, d->aq, d->as);
        gemm_i8_epi(st, DQ_MUL, m, QKV, D, d->aq, bv.qkv.w, d->as, bv.qkv.s, nullptr, nullptr, d->y);
        k_qkv<<<nb((size_t)m * 3 * NH * 32, 256), 256, 0, st>>>(d->y, m, s.prow + r0, s.pe + (size_t)r0 * NPAIR * 2, nullptr, 0,
                                                              bv.nq, bv.nk, d->Q, d->K, d->V);
    }
    k_rms_adaln<<<NT, 256, 0, st>>>(d->xt, mt, mt + D, d->ht);
    rotquant(st, RQ_PLAIN, d->ht, D, NT, D, nullptr, d->aq, d->as);
    gemm_i8_epi(st, DQ_MUL, NT, QKV, D, d->aq, bt.qkv.w, d->as, bt.qkv.s, nullptr, nullptr, d->y);
    k_qkv<<<nb((size_t)NT * 3 * NH * 32, 256), 256, 0, st>>>(d->y, NT, nullptr, nullptr, s.trow, nwin, bt.nq, bt.nk, d->Q, d->K,
                                                           d->V);
    if (i < 2 && d->tap_qk) {
        k_rows<<<M, 128, 0, st>>>(d->Q, s.wrow, d->tap_qk + (size_t)(2 * i) * M * D);
        k_rows<<<M, 128, 0, st>>>(d->K, s.wrow, d->tap_qk + (size_t)(2 * i + 1) * M * D);
    }
    // window attention, one launch per window
    mark(d, 2);
    for (int w = 0; w < nwin; w++) {
        const size_t o = (size_t)s.ps[w] * D;
        const int n = s.nv[w] + NT;
        attention_ex(st, d->Q + o, d->K + o, d->V + o, d->O + o, n, n, nullptr, NH, D, D);
    }
    // back to raster rows (vid, into Q) and the window mean (txt)
    mark(d, 3);
    bf16* att = d->Q;
    k_rows<<<M, 128, 0, st>>>(d->O, s.prow, att);
    k_txt_mean<<<NT, 256, 0, st>>>(d->O, s.trow, nwin, d->att_t);
    if (i == 0 && d->tap_att) {
        CK(cudaMemcpyAsync(d->tap_att, att, (size_t)M * D * 2, cudaMemcpyDeviceToDevice, st));
        CK(cudaMemcpyAsync(d->tap_att + (size_t)M * D, d->att_t, (size_t)NT * D * 2, cudaMemcpyDeviceToDevice, st));
    }
    mark(d, 4);
    for (int r0 = 0; r0 < M; r0 += CH) {
        const int m = std::min(CH, M - r0);
        lin8(d, DQ_GATE, bv.out, att + (size_t)r0 * D, m, D, D, mv + 2 * D, d->x + (size_t)r0 * D);
    }
    lin8(d, DQ_GATE, bt.out, d->att_t, NT, D, D, mt + 2 * D, d->xt);
    mark(d, 5);
    mlp(d, bv, mv, d->x, M);
    mlp(d, bt, mt, d->xt, NT);
}

// bf16 linear rows [m, K] -> fp32 (+ bias, bf16, optional silu) in chunks
void lin16(Dit* d, const bf16* in, int m, int N, int K, const bf16* w, const bf16* b, int act, bf16* out) {
    for (int r0 = 0; r0 < m; r0 += CH) {
        const int c = std::min(CH, m - r0);
        gemm_bf16_f32(d->lt, c, N, K, in + (size_t)r0 * K, w, d->fo);
        k_bias<<<nb((size_t)c * N, 256), 256, 0, d->st>>>(d->fo, b, c, N, act, out + (size_t)r0 * N);
    }
}
}  // namespace

extern "C" {

// Load the SeedVR2 7B int8 ConvRot checkpoint (seedvr2_7b_int8_convrot.safetensors).
void* sv2_dit_new(const char* path) {
    Dit* d = new Dit();
    g_d = d;
    if (!d->f.open(path)) DIE("cannot open %s", path);
    CK(cudaStreamCreate(&d->st));   // blocking on purpose (common.cuh)
    d->lt = lt_new(d->st);
    d->e1w = upload_bf16(d, "emb_in.proj_in.weight", D, TS);
    d->e1b = upload_bf16(d, "emb_in.proj_in.bias", D);
    d->e2w = upload_bf16(d, "emb_in.proj_hid.weight", D, D);
    d->e2b = upload_bf16(d, "emb_in.proj_hid.bias", D);
    d->e3w = upload_bf16(d, "emb_in.proj_out.weight", NEMB, D);
    d->e3b = upload_bf16(d, "emb_in.proj_out.bias", NEMB);
    d->viw = upload_bf16(d, "vid_in.proj.weight", D, PIN, PINP);
    d->vib = upload_bf16(d, "vid_in.proj.bias", D);
    d->vow = upload_bf16(d, "vid_out.proj.weight", POUT, D);
    d->vob = upload_bf16(d, "vid_out.proj.bias", POUT);
    std::vector<bf16> ada((size_t)NB * 2 * 6 * D);
    const char* AK[6] = {"attn_shift", "attn_scale", "attn_gate", "mlp_shift", "mlp_scale", "mlp_gate"};
    for (int i = 0; i < NB; i++) {
        const std::string p = "blocks." + std::to_string(i) + ".";
        for (int br = 0; br < 2; br++) {
            const std::string m = br ? "txt" : "vid";
            Branch& b = br ? d->t[i] : d->v[i];
            b.qkv = load_lin(d, p + "attn.proj_qkv." + m, QKV, D, false);
            b.out = load_lin(d, p + "attn.proj_out." + m, D, D, true);
            b.fc1 = load_lin(d, p + "mlp." + m + ".proj_in", FF, D, true);
            b.fc2 = load_lin(d, p + "mlp." + m + ".proj_out", D, FF, true);
            b.nq = upload_bf16(d, p + "attn.norm_q." + m + ".weight", HD);
            b.nk = upload_bf16(d, p + "attn.norm_k." + m + ".weight", HD);
            for (int k = 0; k < 6; k++) {
                const StTensor& t = tensor(d, p + "ada." + m + "." + AK[k], D);
                if (t.dtype != "F16") DIE("%s: dtype %s", (p + "ada." + m + "." + AK[k]).c_str(), t.dtype.c_str());
                const __half* src = (const __half*)d->f.ptr(t);
                for (int c = 0; c < D; c++) ada[(((size_t)i * 2 + br) * 6 + k) * D + c] = f2bf(__half2float(src[c]));
            }
        }
    }
    d->ada = (bf16*)dmalloc(ada.size() * 2);
    CK(cudaMemcpy(d->ada, ada.data(), ada.size() * 2, cudaMemcpyHostToDevice));
    // work buffers (sized for CH rows)
    d->h = (bf16*)dmalloc((size_t)CH * D * 2);
    d->hb = (bf16*)dmalloc((size_t)CH * FF * 2);
    d->aq = (int8_t*)dmalloc((size_t)CH * FF);
    d->as = (float*)dmalloc((size_t)CH * 4);
    d->y = (bf16*)dmalloc((size_t)CH * QKV * 2);
    d->fo = (float*)dmalloc((size_t)CH * D * 4);
    d->xt = (bf16*)dmalloc((size_t)NT * D * 2);
    d->ht = (bf16*)dmalloc((size_t)NT * D * 2);
    d->att_t = (bf16*)dmalloc((size_t)NT * D * 2);
    d->emb = (bf16*)dmalloc((size_t)NEMB * 2);
    d->mod = (bf16*)dmalloc((size_t)NB * 2 * 6 * D * 2);
    d->e256 = (bf16*)dmalloc(TS * 2);
    d->eh = (bf16*)dmalloc((size_t)2 * D * 2);
    // txt = txt_in(positive_conditioning): constant
    bf16* pc = upload_bf16(d, "positive_conditioning", (size_t)NT * TXT_IN);
    bf16* tw = upload_bf16(d, "txt_in.weight", D, TXT_IN);
    bf16* tb = upload_bf16(d, "txt_in.bias", D);
    d->txt0 = (bf16*)dmalloc((size_t)NT * D * 2);
    lin16(d, pc, NT, D, TXT_IN, tw, tb, 0, d->txt0);
    d->tfreq = (float*)dmalloc(128 * 4);
    CK(cudaMemcpy(d->tfreq, TFREQ, 128 * 4, cudaMemcpyHostToDevice));
    CK(cudaStreamSynchronize(d->st));
    dfree(pc, (size_t)NT * TXT_IN * 2);
    dfree(tw, (size_t)D * TXT_IN * 2);
    dfree(tb, (size_t)D * 2);
    d->f.close();
    CK(cudaGetLastError());
    return d;
}

// One DiT call (ComfyUI NaDiT.forward, batch 1, positive text only = cfg 1).
//   noise [16, T, H, W], cond [17, T, H, W]: bf16 device, channels first, the latent grid (H, W even) — what ComfyUI
//   feeds the model: xc.to(bf16) and the SeedVR2Conditioning condition (latent 16 + mask 1).to(bf16)
//   timestep: sigma * 1000
//   v [16, T, H, W] bf16 device: the raw model output (the flow velocity; denoised = x - v * sigma)
int sv2_dit_forward(void* hd, const void* noise, const void* cond, int T, int H, int W, float timestep, void* v) {
    Dit* d = (Dit*)hd;
    g_d = d;
    cudaStream_t st = d->st;
    set_grid(d, T, H, W);
    const int M = d->M;
    mark(d, 0);
    // time embedding -> modulation for every block
    k_tsin<<<1, TS, 0, st>>>(d->tfreq, timestep, d->e256);
    lin16(d, d->e256, 1, D, TS, d->e1w, d->e1b, 1, d->eh);
    lin16(d, d->eh, 1, D, D, d->e2w, d->e2b, 1, d->eh + D);
    lin16(d, d->eh + D, 1, NEMB, D, d->e3w, d->e3b, 0, d->emb);
    k_mod<<<nb((size_t)NB * 2 * 6 * D, 256), 256, 0, st>>>(d->emb, d->ada, d->mod);
    // inputs
    k_patch_in<<<nb((size_t)M * PINP, 256), 256, 0, st>>>((const bf16*)noise, (const bf16*)cond, T, H, W, d->pin);
    lin16(d, d->pin, M, D, PINP, d->viw, d->vib, 0, d->x);
    CK(cudaMemcpyAsync(d->xt, d->txt0, (size_t)NT * D * 2, cudaMemcpyDeviceToDevice, st));
    if (d->tap_emb) CK(cudaMemcpyAsync(d->tap_emb, d->emb, (size_t)NEMB * 2, cudaMemcpyDeviceToDevice, st));
    for (int i = 0; i <= NB; i++) {
        if (i) block(d, i - 1);
        if (d->tap_vid) CK(cudaMemcpyAsync(d->tap_vid + (size_t)i * M * D, d->x, (size_t)M * D * 2, cudaMemcpyDeviceToDevice, st));
        if (d->tap_txt) CK(cudaMemcpyAsync(d->tap_txt + (size_t)i * NT * D, d->xt, (size_t)NT * D * 2, cudaMemcpyDeviceToDevice, st));
    }
    // vid_out + unpatchify
    mark(d, 0);
    for (int r0 = 0; r0 < M; r0 += CH) {
        const int m = std::min(CH, M - r0);
        gemm_bf16_f32(d->lt, m, POUT, D, d->x + (size_t)r0 * D, d->vow, d->fo);
        k_patch_out<<<nb((size_t)m * POUT, 256), 256, 0, st>>>(d->fo, d->vob, r0, m, T, H, W, (bf16*)v);
    }
    prof_end(d);
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());   // launches here are unchecked: surface their errors in this call
    return 0;
}

// debug taps (device buffers, nullptr = off), M = T * (H/2) * (W/2) of the next forward:
//   vid [37][M][3072]: 0 = vid_in output, i + 1 = after block i (raster token order)
//   txt [37][58][3072]: 0 = txt_in output, i + 1 = after block i
//   emb [18432]: emb_in output
//   qk [4][M][3072]: q, k after norm + RoPE of block 0, then block 1 (window order, vid tokens)
//   att [M + 58][3072]: block 0 attention output before proj_out (vid raster, then the txt window mean)
void sv2_dit_taps(void* hd, void* vid, void* txt, void* emb, void* qk, void* att) {
    Dit* d = (Dit*)hd;
    d->tap_vid = (bf16*)vid;
    d->tap_txt = (bf16*)txt;
    d->tap_emb = (bf16*)emb;
    d->tap_qk = (bf16*)qk;
    d->tap_att = (bf16*)att;
}

// debug: block i alone on vid [M][3072] / txt [58][3072] (bf16 device) -> vout / tout, with the grid and timestep of
// the last forward (teacher forcing against a golden dump)
int sv2_dit_block(void* hd, int i, const void* vin, const void* tin, void* vout, void* tout) {
    Dit* d = (Dit*)hd;
    cudaStream_t st = d->st;
    CK(cudaMemcpyAsync(d->x, vin, (size_t)d->M * D * 2, cudaMemcpyDeviceToDevice, st));
    CK(cudaMemcpyAsync(d->xt, tin, (size_t)NT * D * 2, cudaMemcpyDeviceToDevice, st));
    block(d, i);
    g_ph.clear();
    CK(cudaMemcpyAsync(vout, d->x, (size_t)d->M * D * 2, cudaMemcpyDeviceToDevice, st));
    CK(cudaMemcpyAsync(tout, d->xt, (size_t)NT * D * 2, cudaMemcpyDeviceToDevice, st));
    CK(cudaStreamSynchronize(st));
    CK(cudaGetLastError());
    return 0;
}

// window list of the latent grid [T, H, W] (shifted 0 / 1) into out[6 * n] (t0 t1 h0 h1 w0 w1, token grid); returns n
int sv2_dit_windows(int T, int H, int W, int shifted, int* out, int cap) {
    const std::vector<int> w = windows(T, H / 2, W / 2, shifted != 0);
    const int n = (int)w.size() / 6;
    if (out) memcpy(out, w.data(), std::min(n, cap) * 6 * sizeof(int));
    return n;
}

// device bytes held by the engine: current (peak = 0) or the peak since load (peak = 1)
size_t sv2_dit_mem(void* hd, int peak) {
    Dit* d = (Dit*)hd;
    return peak ? d->peak : d->bytes;
}

void sv2_dit_free(void* hd) {
    Dit* d = (Dit*)hd;
    cudaStreamSynchronize(d->st);
    for (WinSet& s : d->ws) free_ws(s);
    for (int i = 0; i < NB; i++)
        for (Branch* b : {&d->v[i], &d->t[i]}) {
            for (Lin* l : {&b->qkv, &b->out, &b->fc1, &b->fc2}) {
                cudaFree(l->w);
                cudaFree(l->s);
                cudaFree(l->b);
            }
            cudaFree(b->nq);
            cudaFree(b->nk);
        }
    void* ps[] = {d->ada, d->e1w, d->e1b, d->e2w, d->e2b, d->e3w, d->e3b, d->viw, d->vib, d->vow, d->vob, d->txt0, d->tfreq,
                  d->emb, d->mod, d->e256, d->eh, d->x, d->Q, d->K, d->V, d->O, d->h, d->hb, d->pin, d->xt, d->ht, d->att_t,
                  d->aq, d->as, d->fo, d->y};
    for (void* p : ps) cudaFree(p);
    lt_free(d->lt);
    cudaStreamDestroy(d->st);
    if (g_d == d) g_d = nullptr;
    delete d;
}
}  // extern "C"
