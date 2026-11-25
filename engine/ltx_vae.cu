// shutter engine — LTX-2.5 video VAE and latent spatial upsampler x2, C API.
// Model code: ComfyUI e9afbccc2 comfy/ldm/lightricks/vae/na_diffusion_decoder.py CausalDiffusionVAE (the causal conv3d
// Encoder of causal_video_autoencoder.py + NADiffusionDecoder) and comfy/ldm/lightricks/latent_upsampler.py
// (LatentUpsampler as the LTXVLatentUpsampler node runs it). Numerics follow ComfyUI with --bf16-vae: bf16 tensors,
// every torch op rounded to bf16, the decoder's attention through comfy_kitchen's fused na3d / rms_rope_.
//  - Encoder: pixels -> bf16(2p - 1) -> patch 4 (channel c*16 + r*4 + q) -> causal 3x3x3 convs (first frame repeated
//    twice in front, zero spatial border); conv = fp32 implicit GEMM rounded, + bias rounded, + residual rounded.
//    ResnetBlock3D: PixelNorm (bf16(x / bf16(sqrt(bf16(mean bf16(x^2)) + 1e-8)))) -> SiLU -> conv, twice, + x.
//    SpaceToDepthDownsample: conv to C/s, space/time-to-depth, + bf16(mean over groups of s2d(x)); a time stride
//    first duplicates frame 0. conv_out (1024 -> 129) keeps the 128 means; latent = bf16(bf16(m - mean) / std).
//  - Decoder (no convs): z = bf16(bf16(lat * std) + mean), the last latent frame repeated twice behind (trailing
//    pad), conv_in Linear 128 -> 2048 (mm rounded, + bias rounded: the input is a permuted view), 4 stages of NA blocks
//    (2048/1024/512/512 ch, depth 4/6/4/2, kernels 3x7x7 / 3x5x5, head 64) each followed by a Linear pixel-shuffle
//    upsample (temporal x2 drops the duplicated leading frame), the trailing pad cropped -> context [F][H/4][W/4][256].
//    Stage 5: x = conv_in_x_t(patchify4(x_t)) (mm + bias, two roundings), 8 diffusion NA blocks (256 ch, kernel 11^3)
//    with x += context_proj(context) first and AdaLN modulate x * (1 + scale) + shift (scale/shift = shared AdaLN of
//    t = 1 + the block table: constant per checkpoint, computed at load), norm_out, conv_out -> x0 = the pixels.
//    NA block: x += proj(na3d(rms_rope(qkv(rmsnorm(x))))); x += w_down(silu(w_gate(n)) * w_up(n)), n = rmsnorm(x).
//    Linears on contiguous inputs: cuBLASLt with the bias in the epilogue (one rounding, as torch addmm).
//    RMSNorm = torch's vectorized rms_norm kernel order (bit-exact). rms_rope: fp32 RMS over the head, normed value
//    rounded, interleaved-pair rotation (pairs 0-7 frame, 8-19 row, 20-31 column; dims 16/24/24, base 1e4), q weight
//    = bf16(q_norm * 1/8) (the attention scale folded in), so na3d runs with scale 1.
//    na3d (NATTEN semantics): per axis a window of exactly min(k, L) positions centred on the query, shifted inward at
//    the borders; flash-style here (bf16 mma, fp32 online softmax, P rounded to bf16), 128-query tiles against the
//    union of their windows with a per-element mask.
//    Memory: each block streams over time chunks; q/k/v live in a ring of chunk + kt - 1 frames (the key frames a
//    chunk needs are produced from the block input before any of them is updated), so only x and the context are
//    full-size. The decode is untiled: one pass, the model's own output (no seams, one noise tensor).
//  - Latent upsampler: z = bf16(bf16(lat * std) + mean) -> Conv3d 3x3x3 (zero padding in t, h, w) -> GroupNorm(32,
//    eps 1e-5, statistics over the whole [T][H][W] volume) -> SiLU -> 4 ResBlocks (conv, GN, SiLU, conv, GN,
//    SiLU(x + residual)) -> per-frame Conv2d 3x3 1024 -> 4096 + PixelShuffle 2 -> 4 ResBlocks -> Conv3d -> 128,
//    normalized back (bf16(bf16(y - mean) / std)).
// Env: LTXVAE_PROF=1 prints GPU time per section and peak device memory per call; LTXVAE_TAPS=<dir> writes decoder
// activations (golden/test_ltx_vae.py compares them with the dump); LTXVAE_ROWS=<n> sets the time-chunk size (rows per
// chunk, default 262144; debug).
#include "common.cuh"
#include "st.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <tuple>
#include <vector>

namespace {
inline size_t nblk(size_t n, int t) { return (n + t - 1) / t; }
inline int rup(int a, int b) { return (a + b - 1) / b * b; }
__device__ __forceinline__ float rd(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float ldf(const bf16* p) { return __bfloat162float(*p); }
__device__ __forceinline__ float silu(float v) { return v / (1.f + expf(-v)); }

// ---------------- device memory with a running total (LTXVAE_PROF=1 prints the peak per call)
std::map<void*, size_t> g_sz;
size_t g_cur = 0, g_peak = 0;
void* dmalloc(size_t n) {
    void* p;
    CK(cudaMalloc(&p, std::max<size_t>(n, 16)));
    g_sz[p] = n, g_cur += n, g_peak = std::max(g_peak, g_cur);
    return p;
}
void dfree(void* p) {
    if (!p) return;
    auto it = g_sz.find(p);
    if (it != g_sz.end()) g_cur -= it->second, g_sz.erase(it);
    CK(cudaFree(p));
}
struct Prof {
    bool on = getenv("LTXVAE_PROF") != nullptr;
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
        fprintf(stderr, "[ltx_vae %s] %.0f ms:", what, tot);
        for (auto& kv : s) fprintf(stderr, " %s %.0f", kv.first.c_str(), kv.second);
        fprintf(stderr, "; peak device memory %.2f GiB (weights included)\n", g_peak / 1073741824.0);
        for (auto& x : ev) cudaEventDestroy(x.second);
        ev.clear();
    }
};
Prof prof;

// ---------------- GEMM: cuBLASLt bf16, out[M, N] (row stride ldc) = A[M, K] (row stride lda) . W[N, K]^T.
// With a bias the epilogue adds it before the one rounding (torch addmm / F.linear on a contiguous input),
// without it plain mm. Heuristic's first algorithm per shape (deterministic).
struct Gemm {
    cublasLtHandle_t h = nullptr;
    cudaStream_t st = nullptr;
    void* ws = nullptr;
    size_t wsz = 32u << 20;
    struct Plan {
        cublasLtMatmulDesc_t op;
        cublasLtMatrixLayout_t lw, la, lc;
        cublasLtMatmulAlgo_t algo;
    };
    std::map<std::tuple<int, int, int, int, int, int>, Plan> plans;
};
void gemm_init(Gemm& g, cudaStream_t st) {
    g.st = st;
    CKL(cublasLtCreate(&g.h));
    g.ws = dmalloc(g.wsz);
}
void gemm_free(Gemm& g) {
    for (auto& kv : g.plans) {
        cublasLtMatmulDescDestroy(kv.second.op);
        cublasLtMatrixLayoutDestroy(kv.second.lw);
        cublasLtMatrixLayoutDestroy(kv.second.la);
        cublasLtMatrixLayoutDestroy(kv.second.lc);
    }
    g.plans.clear();
    dfree(g.ws);
    cublasLtDestroy(g.h);
}
void lin(Gemm& g, int M, int N, int K, const bf16* A, int lda, const bf16* W, const bf16* bias, bf16* C, int ldc) {
    if (M <= 0) return;
    const auto key = std::make_tuple(M, N, K, lda, ldc, bias ? 1 : 0);
    auto it = g.plans.find(key);
    if (it == g.plans.end()) {
        Gemm::Plan p;
        CKL(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        CKL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        CKL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        if (bias) {
            cublasLtEpilogue_t ep = CUBLASLT_EPILOGUE_BIAS;
            cudaDataType_t bt = CUDA_R_16BF;
            CKL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_EPILOGUE, &ep, sizeof(ep)));
            CKL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE, &bt, sizeof(bt)));
            CKL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
        }
        CKL(cublasLtMatrixLayoutCreate(&p.lw, CUDA_R_16BF, K, N, K));
        CKL(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, K, M, lda));
        CKL(cublasLtMatrixLayoutCreate(&p.lc, CUDA_R_16BF, N, M, ldc));
        cublasLtMatmulPreference_t pref;
        CKL(cublasLtMatmulPreferenceCreate(&pref));
        CKL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g.wsz, sizeof(g.wsz)));
        cublasLtMatmulHeuristicResult_t res[1];
        int n = 0;
        CKL(cublasLtMatmulAlgoGetHeuristic(g.h, p.op, p.lw, p.la, p.lc, p.lc, pref, 1, res, &n));
        cublasLtMatmulPreferenceDestroy(pref);
        if (n == 0) DIE("ltx_vae: no cublasLt algo for M=%d N=%d K=%d", M, N, K);
        p.algo = res[0].algo;
        it = g.plans.emplace(key, p).first;
    }
    Gemm::Plan& p = it->second;
    if (bias) CKL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_BIAS_POINTER, &bias, sizeof(bias)));
    const float one = 1.f, zero = 0.f;
    CKL(cublasLtMatmul(g.h, p.op, &one, W, p.lw, A, p.la, &zero, C, p.lc, C, p.lc, &p.algo, g.ws, g.wsz, g.st));
}

// ================================================================ decoder kernels
// RMSNorm (+ AdaLN modulate) over rows of N (N % 128 == 0, <= 2048), one warp per row. The sum of squares follows
// torch's vectorized rms_norm kernel (256 threads x 4-element vectors, fma per element, shfl_down tree per warp, then
// warps pairwise 4, 2, 1), so y = bf16(w * (rstd * x)) is F.rms_norm bit for bit. With ope/sh (modulate,
// x * (1 + scale) + shift with ope = bf16(1 + scale)): out = bf16(bf16(y * ope) + sh).
__global__ void __launch_bounds__(256) k_rmsmod(const bf16* __restrict__ x, int M, int N, const bf16* __restrict__ w,
                                                const bf16* __restrict__ ope, const bf16* __restrict__ sh, float eps,
                                                bf16* __restrict__ out) {
    const int warp = threadIdx.x >> 5, lane = threadIdx.x & 31;
    const size_t row = (size_t)blockIdx.x * 8 + warp;
    if (row >= (size_t)M) return;
    const bf16* xr = x + row * N;
    const int nv = N / 4;
    float s[8];
#pragma unroll
    for (int j = 0; j < 8; j++) {
        s[j] = 0.f;
        for (int i = lane + 32 * j; i < nv; i += 256) {
            const uint2 r = *(const uint2*)(xr + 4 * i);
            const __nv_bfloat162* h = (const __nv_bfloat162*)&r;
            const float v0 = __low2float(h[0]), v1 = __high2float(h[0]), v2 = __low2float(h[1]), v3 = __high2float(h[1]);
            s[j] = __fmaf_rn(v0, v0, s[j]);
            s[j] = __fmaf_rn(v1, v1, s[j]);
            s[j] = __fmaf_rn(v2, v2, s[j]);
            s[j] = __fmaf_rn(v3, v3, s[j]);
        }
    }
#pragma unroll
    for (int j = 0; j < 8; j++)
#pragma unroll
        for (int o = 16; o; o >>= 1) s[j] = __fadd_rn(s[j], __shfl_down_sync(0xffffffffu, s[j], o));
#pragma unroll
    for (int o = 4; o; o >>= 1)
#pragma unroll
        for (int j = 0; j < o; j++) s[j] = __fadd_rn(s[j], s[j + o]);
    const float tot = __shfl_sync(0xffffffffu, s[0], 0);
    const float rstd = rsqrtf(__fadd_rn(__fdiv_rn(tot, (float)N), eps));
    bf16* orow = out + row * N;
    for (int i = lane; i < nv; i += 32) {
        const uint2 r = *(const uint2*)(xr + 4 * i);
        const __nv_bfloat162* h = (const __nv_bfloat162*)&r;
        float v[4] = {__low2float(h[0]), __high2float(h[0]), __low2float(h[1]), __high2float(h[1])};
#pragma unroll
        for (int e = 0; e < 4; e++) {
            const int c = 4 * i + e;
            float y = rd(__fmul_rn(ldf(w + c), __fmul_rn(rstd, v[e])));
            if (ope) y = rd(__fadd_rn(rd(__fmul_rn(y, ldf(ope + c))), ldf(sh + c)));
            v[e] = y;
        }
        uint2 o;
        __nv_bfloat162* ho = (__nv_bfloat162*)&o;
        ho[0] = __floats2bfloat162_rn(v[0], v[1]);
        ho[1] = __floats2bfloat162_rn(v[2], v[3]);
        *(uint2*)(orow + 4 * i) = o;
    }
}

// q/k RMSNorm over each 64-dim head + absolute 3D RoPE, in place on ring rows (comfy_kitchen rms_rope_: fp32 RMS, the
// normed value rounded to bf16, then the rotation c x0 - s x1, s x0 + c x1 in fp32, rounded). One warp per (token,
// q|k, head), a pair per lane: pairs 0-7 use the frame table, 8-19 the row table, 20-31 the column table.
__global__ void k_rms_rope(bf16* __restrict__ ring, int ld, int C, int heads, int f0, int nf, int R, int H, int W,
                           const bf16* __restrict__ wq, const bf16* __restrict__ wk, const float2* __restrict__ tt,
                           const float2* __restrict__ th, const float2* __restrict__ tw) {
    const size_t gw = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    const size_t ntok = (size_t)nf * H * W;
    if (gw >= ntok * 2 * heads) return;
    const int hd = (int)(gw % heads), which = (int)((gw / heads) & 1);
    const size_t tk = gw / (2 * heads);
    const int w = (int)(tk % W), h = (int)((tk / W) % H), f = f0 + (int)(tk / ((size_t)W * H));
    bf16* p = ring + (((size_t)(f % R) * H + h) * W + w) * ld + which * C + hd * 64 + 2 * lane;
    const __nv_bfloat162 xv = *(__nv_bfloat162*)p;
    const float x0 = __low2float(xv), x1 = __high2float(xv);
    float ss = x0 * x0 + x1 * x1;
#pragma unroll
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float r = rsqrtf(ss / 64.f + 1e-6f);
    const bf16* wt = which ? wk : wq;
    const float y0 = rd(__fmul_rn(__fmul_rn(x0, r), ldf(wt + 2 * lane)));
    const float y1 = rd(__fmul_rn(__fmul_rn(x1, r), ldf(wt + 2 * lane + 1)));
    const float2 cs = lane < 8 ? tt[f * 8 + lane] : lane < 20 ? th[h * 12 + lane - 8] : tw[w * 12 + lane - 20];
    *(__nv_bfloat162*)p = __floats2bfloat162_rn(__fadd_rn(__fmul_rn(cs.x, y0), __fmul_rn(-cs.y, y1)),
                                                __fadd_rn(__fmul_rn(cs.y, y0), __fmul_rn(cs.x, y1)));
}

// RoPE table: tab[p * n + i] = (cos, sin)(p * inv[i]) in fp32, p < L
__global__ void k_rope_tab(int L, int n, const float* __restrict__ inv, float2* __restrict__ tab) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= L * n) return;
    const float a = __fmul_rn((float)(i / n), inv[i % n]);
    tab[i] = make_float2(cosf(a), sinf(a));
}

// ---- 3D neighborhood attention, head dim 64, 128-query tiles (TQ x HQ x WQ), 8 warps, 64-key chunks of the tile's key
// region (the union of its queries' windows), mask per (query, key), flash-style online softmax (exp2, P in bf16).
// q/k/v rows of stride ld (head hd at column hd * 64) in a ring of R frames (frame f in slot f % R); queries are frames
// [t0, t1), output rows ((t - t0) H + h) W + w of stride ldo.
struct NaArgs {
    const bf16 *q, *k, *v;
    bf16* o;
    int ld, ldo, T, H, W, R, kt, kh, kw, t0, t1, TQ, HQ, WQ, ntw, nth, heads;
    float sl2;
};
__device__ __forceinline__ int swz64(int row, int col) { return row * 64 + ((((col >> 3) ^ (row & 7))) << 3) + (col & 7); }
__device__ __forceinline__ void cp16(void* s, const void* g, int bytes) {
    unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(g), "r"(bytes));
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
__device__ __forceinline__ int wstart(int i, int L, int k) { return min(max(i - k / 2, 0), L - k); }

__global__ void __launch_bounds__(256) k_na3d(NaArgs a) {
    constexpr int TILE = 64 * 64;
    extern __shared__ __align__(128) bf16 sm[];   // 2 stages x (K | V) 64 x 64; Q (128 x 64) staged in stage 1
    __shared__ int skey[2][64];                   // key coordinates t << 20 | h << 10 | w per stage
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31;
    const int hd = blockIdx.x % a.heads, tile = blockIdx.x / a.heads;
    const int tw = tile % a.ntw, th = (tile / a.ntw) % a.nth, ti = tile / (a.ntw * a.nth);
    const int qt0 = a.t0 + ti * a.TQ, qh0 = th * a.HQ, qw0 = tw * a.WQ;
    const int qt1 = min(qt0 + a.TQ, a.t1), qh1 = min(qh0 + a.HQ, a.H), qw1 = min(qw0 + a.WQ, a.W);
    const int rt0 = wstart(qt0, a.T, a.kt), rh0 = wstart(qh0, a.H, a.kh), rw0 = wstart(qw0, a.W, a.kw);
    const int rH = wstart(qh1 - 1, a.H, a.kh) + a.kh - rh0, rW = wstart(qw1 - 1, a.W, a.kw) + a.kw - rw0;
    const int nreg = (wstart(qt1 - 1, a.T, a.kt) + a.kt - rt0) * rH * rW;
    const int HQW = a.HQ * a.WQ;
    auto tok = [&](int t, int h, int w) -> size_t { return (((size_t)(t % a.R)) * a.H + h) * a.W + w; };
    bf16* sQ = sm + 2 * TILE;
    for (int c = tid; c < 128 * 8; c += 256) {
        const int r = c >> 3, ch = c & 7;
        const int t = qt0 + r / HQW, h = qh0 + (r / a.WQ) % a.HQ, w = qw0 + r % a.WQ;
        const bool ok = t < qt1 && h < qh1 && w < qw1;
        cp16(&sQ[swz64(r, ch * 8)], a.q + (ok ? tok(t, h, w) * a.ld : 0) + hd * 64 + ch * 8, ok ? 16 : 0);
    }
    auto load_kv = [&](int kb, int stage) {
        bf16* sK = sm + stage * 2 * TILE;
        bf16* sV = sK + TILE;
        for (int c = tid; c < 64 * 8; c += 256) {
            const int r = c >> 3, ch = c & 7, j = kb + r;
            const bool ok = j < nreg;
            int t = 0, h = 0, w = 0;
            if (ok) w = rw0 + j % rW, h = rh0 + (j / rW) % rH, t = rt0 + j / (rW * rH);
            const size_t off = (ok ? tok(t, h, w) * a.ld : 0) + hd * 64 + ch * 8;
            cp16(&sK[swz64(r, ch * 8)], a.k + off, ok ? 16 : 0);
            cp16(&sV[swz64(r, ch * 8)], a.v + off, ok ? 16 : 0);
            if (ch == 0) skey[stage][r] = ok ? (t << 20 | h << 10 | w) : (1023 << 20);
        }
        asm volatile("cp.async.commit_group;\n" ::);
    };
    load_kv(0, 0);   // same commit group as Q
    asm volatile("cp.async.wait_group 0;\n" ::);
    __syncthreads();
    const int g = lane >> 2, tq = lane & 3;
    int st[2], sh[2], sw[2], rt[2], rh[2], rw[2];
    bool rv[2];
#pragma unroll
    for (int i = 0; i < 2; i++) {
        const int r = warp * 16 + g + i * 8;
        rt[i] = qt0 + r / HQW, rh[i] = qh0 + (r / a.WQ) % a.HQ, rw[i] = qw0 + r % a.WQ;
        rv[i] = rt[i] < qt1 && rh[i] < qh1 && rw[i] < qw1;
        st[i] = rv[i] ? wstart(rt[i], a.T, a.kt) : 1 << 20;   // invalid rows see no key
        sh[i] = wstart(rh[i], a.H, a.kh), sw[i] = wstart(rw[i], a.W, a.kw);
    }
    uint32_t qf[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; kk++) ldm4(qf[kk], &sQ[swz64(warp * 16 + (lane & 15), kk * 16 + (lane >> 4) * 8)]);
    __syncthreads();   // stage 1 is free for K/V
    float o[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    const unsigned KT = a.kt, KH = a.kh, KW = a.kw;
    for (int kb = 0, stage = 0; kb < nreg; kb += 64, stage ^= 1) {
        if (kb + 64 < nreg) load_kv(kb + 64, stage ^ 1);
        const bf16* sK = sm + stage * 2 * TILE;
        const bf16* sV = sK + TILE;
        float s[8][4];
#pragma unroll
        for (int i = 0; i < 8; i++) s[i][0] = s[i][1] = s[i][2] = s[i][3] = 0.f;
#pragma unroll
        for (int kk = 0; kk < 4; kk++) {
#pragma unroll
            for (int p = 0; p < 4; p++) {
                uint32_t b[4];
                ldm4(b, &sK[swz64(p * 16 + (lane & 7) + (lane >> 4) * 8, kk * 16 + ((lane >> 3) & 1) * 8)]);
                mma(s[2 * p], qf[kk], b[0], b[1]);
                mma(s[2 * p + 1], qf[kk], b[2], b[3]);
            }
        }
        float mx0 = -INFINITY, mx1 = -INFINITY;
#pragma unroll
        for (int nt = 0; nt < 8; nt++) {
#pragma unroll
            for (int e = 0; e < 2; e++) {
                const int key = skey[stage][nt * 8 + tq * 2 + e];
                const int kt_ = key >> 20, kh_ = (key >> 10) & 1023, kw_ = key & 1023;
#pragma unroll
                for (int i = 0; i < 2; i++) {
                    const bool vis = (unsigned)(kt_ - st[i]) < KT && (unsigned)(kh_ - sh[i]) < KH && (unsigned)(kw_ - sw[i]) < KW;
                    float& sv = s[nt][2 * i + e];
                    sv = vis ? sv * a.sl2 : -INFINITY;
                }
            }
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
        m0 = n0, m1 = n1;
        l0 = l0 * c0 + rs0;
        l1 = l1 * c1 + rs1;
#pragma unroll
        for (int i = 0; i < 8; i++) o[i][0] *= c0, o[i][1] *= c0, o[i][2] *= c1, o[i][3] *= c1;
#pragma unroll
        for (int kc = 0; kc < 4; kc++) {
            const uint32_t pa[4] = {pack2(s[2 * kc][0], s[2 * kc][1]), pack2(s[2 * kc][2], s[2 * kc][3]),
                                    pack2(s[2 * kc + 1][0], s[2 * kc + 1][1]), pack2(s[2 * kc + 1][2], s[2 * kc + 1][3])};
#pragma unroll
            for (int dp = 0; dp < 4; dp++) {
                uint32_t b[4];
                ldm4t(b, &sV[swz64(kc * 16 + (lane & 7) + ((lane >> 3) & 1) * 8, dp * 16 + (lane >> 4) * 8)]);
                mma(o[2 * dp], pa, b[0], b[1]);
                mma(o[2 * dp + 1], pa, b[2], b[3]);
            }
        }
        asm volatile("cp.async.wait_group 0;\n" ::);
        __syncthreads();   // next stage landed; everyone is done with this one before it is refilled
    }
    l0 += __shfl_xor_sync(0xffffffffu, l0, 1);
    l0 += __shfl_xor_sync(0xffffffffu, l0, 2);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 1);
    l1 += __shfl_xor_sync(0xffffffffu, l1, 2);
    const float il[2] = {l0 > 0.f ? 1.f / l0 : 0.f, l1 > 0.f ? 1.f / l1 : 0.f};
#pragma unroll
    for (int i = 0; i < 2; i++) {
        if (!rv[i]) continue;
        bf16* orow = a.o + (((size_t)(rt[i] - a.t0) * a.H + rh[i]) * a.W + rw[i]) * a.ldo + hd * 64 + tq * 2;
#pragma unroll
        for (int dt = 0; dt < 8; dt++) *(uint32_t*)(orow + dt * 8) = pack2(o[dt][2 * i] * il[i], o[dt][2 * i + 1] * il[i]);
    }
}

// x = bf16(x + y), n % 8 == 0
__global__ void k_add(bf16* __restrict__ x, const bf16* __restrict__ y, size_t n8) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n8) return;
    uint4 a = ((uint4*)x)[i];
    const uint4 b = ((const uint4*)y)[i];
    __nv_bfloat162* ha = (__nv_bfloat162*)&a;
    const __nv_bfloat162* hb = (const __nv_bfloat162*)&b;
#pragma unroll
    for (int e = 0; e < 4; e++)
        ha[e] = __floats2bfloat162_rn(__low2float(ha[e]) + __low2float(hb[e]), __high2float(ha[e]) + __high2float(hb[e]));
    ((uint4*)x)[i] = a;
}
// y = bf16(y + b[col]) over rows of N
__global__ void k_bias(bf16* __restrict__ y, size_t n, int N, const bf16* __restrict__ b) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) y[i] = __float2bfloat16(ldf(y + i) + ldf(b + i % N));
}
// SwiGLU in place: gu rows [gate hid | up hid] -> gate slot = bf16(bf16(silu(gate)) * up)
__global__ void k_swiglu(bf16* __restrict__ gu, size_t M, int hid) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * hid) return;
    bf16* r = gu + (i / hid) * 2 * hid + i % hid;
    r[0] = __float2bfloat16(rd(silu(ldf(r))) * ldf(r + hid));
}
// latent [128][T][HW] f32 (normalized) -> bf16 -> bf16(bf16(z * std) + mean), channels-last [Tp][HW][128]; frames
// t >= T repeat frame T - 1 (decoder trailing pad)
__global__ void k_lat_in(const float* __restrict__ lat, int T, int Tp, int HW, const bf16* __restrict__ sd,
                         const bf16* __restrict__ mu, bf16* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)Tp * HW * 128) return;
    const int c = (int)(i % 128);
    const size_t p = (i / 128) % HW;
    const int t = min((int)(i / ((size_t)128 * HW)), T - 1);
    const float z = rd(lat[((size_t)c * T + t) * HW + p]);
    out[i] = __float2bfloat16(rd(z * ldf(sd + c)) + ldf(mu + c));
}
// LinearPixelShuffleUpsample scatter: y = proj rows of input frames fa.. ([H][W] each), P * Co channels with channel
// ((c p1 + a) p2 + b) p3 + e -> out [To][H p2][W p3][Co], frame t p1 + a - drop (frames outside [0, To) skipped).
// One block per input token (row staged in shared memory).
__global__ void k_shuffle(const bf16* __restrict__ y, int fa, int H, int W, int Co, int p1, int p2, int p3, int drop, int To,
                          bf16* __restrict__ out) {
    extern __shared__ __align__(16) bf16 srow[];
    const int P = p1 * p2 * p3, n = P * Co;
    const size_t tk = blockIdx.x;
    const bf16* yr = y + tk * n;
    for (int i = threadIdx.x; i < n / 8; i += blockDim.x) ((uint4*)srow)[i] = ((const uint4*)yr)[i];
    __syncthreads();
    const int w = (int)(tk % W), h = (int)((tk / W) % H), t = fa + (int)(tk / ((size_t)W * H));
    const int Ho = H * p2, Wo = W * p3;
    for (int i = threadIdx.x; i < n; i += blockDim.x) {
        const int c = i % Co, q = i / Co, e = q % p3, b = (q / p3) % p2, aa = q / (p3 * p2);
        const int to = t * p1 + aa - drop;
        if (to < 0 || to >= To) continue;
        out[(((size_t)to * Ho + h * p2 + b) * Wo + w * p3 + e) * Co + c] = srow[((c * p1 + aa) * p2 + b) * p3 + e];
    }
}
// stage-5 patchify: x_t [3][F][H][W] bf16, frames f0.. -> A [nf][H/4][W/4][48], channel c*16 + r*4 + q = x_t[c][f][4y+q][4x+r]
__global__ void k_patch(const bf16* __restrict__ xt, int F, int H, int W, int f0, int nf, bf16* __restrict__ A) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int h4 = H / 4, w4 = W / 4;
    if (i >= (size_t)nf * h4 * w4 * 48) return;
    const int ch = (int)(i % 48);
    const size_t tk = i / 48;
    const int x = (int)(tk % w4), y = (int)((tk / w4) % h4), f = f0 + (int)(tk / ((size_t)w4 * h4));
    const int c = ch / 16, r = (ch / 4) % 4, q = ch % 4;
    A[i] = xt[(((size_t)c * F + f) * H + 4 * y + q) * W + 4 * x + r];
}
// conv_out rows [nf][H/4][W/4][48] -> raw [3][F][H][W] bf16 and/or frames [F][H][W][3] f32 = clamp((raw + 1) / 2, 0, 1)
__global__ void k_unpatch(const bf16* __restrict__ y, int F, int H, int W, int f0, int nf, bf16* __restrict__ raw,
                          float* __restrict__ frames) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)nf * H * W) return;
    const int X = (int)(i % W), Y = (int)((i / W) % H), f = f0 + (int)(i / ((size_t)W * H));
    const bf16* r = y + ((((size_t)(f - f0) * (H / 4) + Y / 4) * (W / 4)) + X / 4) * 48 + (X % 4) * 4 + Y % 4;
#pragma unroll
    for (int c = 0; c < 3; c++) {
        const bf16 v = r[c * 16];
        const size_t o = (((size_t)f * H + Y) * W + X);
        if (raw) raw[(size_t)c * F * H * W + o] = v;
        if (frames) frames[o * 3 + c] = fminf(fmaxf(__fmul_rn(__fadd_rn(__bfloat162float(v), 1.f), 0.5f), 0.f), 1.f);
    }
}
// timestep embedding of t (torch get_timestep_embedding, 256, flip_sin_to_cos, shift 0): [cos(t f_i) | sin(t f_i)],
// f_i = exp(-ln(1e4) i / 128), fp32, rounded to bf16
__global__ void k_temb_freq(float t, bf16* __restrict__ emb) {
    const int i = threadIdx.x;
    if (i >= 128) return;
    const float e = expf(__fmul_rn(__fmul_rn(-9.210340371976184f, (float)i), 1.f / 128.f));
    const float a = __fmul_rn(t, e);
    emb[i] = __float2bfloat16(cosf(a));
    emb[128 + i] = __float2bfloat16(sinf(a));
}
__global__ void k_silu(const bf16* __restrict__ in, bf16* __restrict__ out, int n) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __float2bfloat16(silu(ldf(in + i)));
}
// per diffusion block: ope = bf16(1 + bf16(mod[j] + table[j])), sh = bf16(mod[j + 1] + table[j + 1]) for j = 0, 3
__global__ void k_modprep(const bf16* __restrict__ mod, const bf16* __restrict__ tab, int C, bf16* __restrict__ ope1,
                          bf16* __restrict__ sh1, bf16* __restrict__ ope2, bf16* __restrict__ sh2) {
    const int c = blockIdx.x * blockDim.x + threadIdx.x;
    if (c >= C) return;
    auto at = [&](int j) { return rd(ldf(mod + j * C + c) + ldf(tab + j * C + c)); };
    ope1[c] = __float2bfloat16(1.f + at(0));
    sh1[c] = __float2bfloat16(at(1));
    ope2[c] = __float2bfloat16(1.f + at(3));
    sh2[c] = __float2bfloat16(at(4));
}

// ================================================================ encoder / upsampler kernels (channels-last bf16)
// pixels [F][H][W][3] f32 in 0..1 -> bf16(2p - 1) patchified by 4: x [F][H/4][W/4][48], channel c*16 + r*4 + q
__global__ void k_px_patch(const float* __restrict__ px, int F, int H, int W, bf16* __restrict__ x) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    const int h4 = H / 4, w4 = W / 4;
    if (i >= (size_t)F * h4 * w4 * 48) return;
    const int ch = (int)(i % 48);
    const size_t tk = i / 48;
    const int xx = (int)(tk % w4), y = (int)((tk / w4) % h4), f = (int)(tk / ((size_t)w4 * h4));
    const int c = ch / 16, r = (ch / 4) % 4, q = ch % 4;
    const float v = px[(((size_t)f * H + 4 * y + q) * W + 4 * xx + r) * 3 + c];
    x[i] = __float2bfloat16(__fsub_rn(__fmul_rn(v, 2.f), 1.f));
}
// PixelNorm denominator per pixel (torch bf16 ops): bf16(sqrt(bf16(bf16(mean_c bf16(x^2)) + 1e-8))), one warp a pixel
__global__ void k_pn(const bf16* __restrict__ x, size_t npix, int C, float* __restrict__ pn) {
    const size_t p = ((size_t)blockIdx.x * blockDim.x + threadIdx.x) >> 5;
    const int lane = threadIdx.x & 31;
    if (p >= npix) return;
    const bf16* r = x + p * C;
    float s = 0.f;
    for (int c = lane; c < C; c += 32) {
        const float v = ldf(r + c);
        s += rd(v * v);
    }
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane == 0) pn[p] = rd(sqrtf(rd(rd(s / (float)C) + 1e-8f)));
}
// GroupNorm(32) statistics over a whole [n][C] volume, deterministic: per-(256-pixel chunk, channel) partial sums
// (pass 0: x, pass 1: (x - mean)^2), reduced per group by one warp. A thread sums 8 channels of a chunk; 4 chunks a block.
template <int PASS>
__global__ void k_gn_part(const bf16* __restrict__ x, size_t n, int C, int nch, const float2* __restrict__ st,
                          float* __restrict__ part) {
    const int CG = C / 8, q = threadIdx.x / CG, c0 = (threadIdx.x % CG) * 8, k = blockIdx.x * 4 + q;
    if (k >= nch) return;
    const size_t p0 = (size_t)k * 256, p1 = p0 + 256 < n ? p0 + 256 : n;
    float mean[8], s[8];
#pragma unroll
    for (int e = 0; e < 8; e++) mean[e] = PASS ? st[(c0 + e) / (C / 32)].x : 0.f, s[e] = 0.f;
    for (size_t p = p0; p < p1; p++) {
        const uint4 r = *(const uint4*)(x + p * C + c0);
        const __nv_bfloat162* h = (const __nv_bfloat162*)&r;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const float d = __fsub_rn(e & 1 ? __high2float(h[e / 2]) : __low2float(h[e / 2]), mean[e]);
            s[e] = __fadd_rn(s[e], PASS ? __fmul_rn(d, d) : d);
        }
    }
#pragma unroll
    for (int e = 0; e < 8; e++) part[(size_t)k * C + c0 + e] = s[e];
}
__global__ void k_gn_fin(const float* __restrict__ part, size_t n, int C, int nch, int pass, float eps, float2* __restrict__ st) {
    const int g = blockIdx.x, cg = C / 32, lane = threadIdx.x;
    float s = 0.f;
    for (int i = lane; i < nch * cg; i += 32) s += part[(size_t)(i / cg) * C + g * cg + i % cg];
#pragma unroll
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane) return;
    const float cnt = (float)((double)n * cg);
    st[g] = pass == 0 ? make_float2(s / cnt, 0.f) : make_float2(st[g].x, rsqrtf(s / cnt + eps));
}
// conv input P [Tp][H + 2][W + 2][Cp] from x [T][H][W][C]: zero spatial border and channels >= C. Time: zt == 0 reads
// x[clamp(tp - front, 0, T - 1)] (causal: frame 0 repeated front times; front 0 = a per-frame 2D conv), zt == 1 reads
// x[tp - 1] or zeros (Conv3d zero padding 1). Transform: mode 1 PixelNorm + SiLU, bf16(silu(bf16(x / pn[pixel])));
// mode 2 GroupNorm + SiLU, bf16(silu(bf16(x a + b))), a = rstd w, b = beta - a mean. E channels per thread.
template <int E>
__global__ void k_pad(const bf16* __restrict__ x, int T, int H, int W, int C, int Cp, int Tp, int front, int zt, int mode,
                      const float* __restrict__ pn, const float2* __restrict__ gs, const float* __restrict__ gw,
                      const float* __restrict__ gb, bf16* __restrict__ P) {
    const int Hp = H + 2, Wp = W + 2, CE = Cp / E;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)Tp * Hp * Wp * CE) return;
    const int c0 = (int)(i % CE) * E;
    const size_t p = i / CE;
    const int xp = (int)(p % Wp), yp = (int)((p / Wp) % Hp), tp = (int)(p / ((size_t)Wp * Hp));
    const int y = yp - 1, xx = xp - 1, t = zt ? tp - 1 : min(max(tp - front, 0), T - 1);
    float v[E];
#pragma unroll
    for (int e = 0; e < E; e++) v[e] = 0.f;
    if (y >= 0 && y < H && xx >= 0 && xx < W && t >= 0 && t < T && c0 < C) {
        const size_t pix = ((size_t)t * H + y) * W + xx;
        const bf16* src = x + pix * C + c0;
        if (E == 8) {
            const uint4 r = *(const uint4*)src;
            const __nv_bfloat162* h = (const __nv_bfloat162*)&r;
#pragma unroll
            for (int e = 0; e < 4; e++) v[2 * e] = __low2float(h[e]), v[2 * e + 1] = __high2float(h[e]);
        } else {
            v[0] = ldf(src);
        }
        if (mode == 1) {
            const float d = pn[pix];
#pragma unroll
            for (int e = 0; e < E; e++) v[e] = rd(silu(rd(__fdiv_rn(v[e], d))));
        } else if (mode == 2) {
#pragma unroll
            for (int e = 0; e < E; e++) {
                const int c = c0 + e;
                const float2 s = gs[c / (C / 32)];
                const float a = s.y * gw[c], b = gb[c] - a * s.x;
                v[e] = rd(silu(rd(a * v[e] + b)));
            }
        }
    }
    bf16* dst = P + p * Cp + c0;
    if (E == 8) {
        uint4 r;
        __nv_bfloat162* h = (__nv_bfloat162*)&r;
#pragma unroll
        for (int e = 0; e < 4; e++) h[e] = __floats2bfloat162_rn(v[2 * e], v[2 * e + 1]);
        *(uint4*)dst = r;
    } else {
        dst[0] = __float2bfloat16(v[0]);
    }
}

// implicit-GEMM conv (stride 1) over the padded P [Tp][Hp][Wp][C] (C % 32 == 0), kernel (nt, 3, 3):
// out[m][n] = bf16(bf16(bf16(sum_k A[m][k] W[n][k]) + b[n]) + resid), n < N, K order (kt, ky, kx, c), fp32 accumulate.
// BM x BN x 32 tiles, warps of WM x WN, ST-stage cp.async pipeline, ldmatrix + mma.sync bf16 (seedvr2_vae.cu k_iconv
// with a contiguous padded input and a 1-D grid, n blocks fastest so they share A in L2).
__device__ __forceinline__ int sw32(int row, int ch) { return row * 32 + ((ch ^ ((row >> 1) & 3)) << 3); }
template <int BM, int BN, int WM, int WN, int ST>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32) k_iconv(const bf16* __restrict__ P, int Hp, int Wp, int C, int nt,
                                                                       int Ho, int Wo, int M, const bf16* __restrict__ Wt, int kp,
                                                                       const float* __restrict__ bias, const bf16* __restrict__ resid,
                                                                       int N, int nbn, bf16* __restrict__ out) {
    constexpr int MT = WM / 16, NP = WN / 16, WGN = BN / WN, NT = (BM / WM) * WGN * 32;
    extern __shared__ __align__(128) unsigned char smem[];
    bf16* sA = (bf16*)smem;
    bf16* sB = sA + ST * BM * 32;
    __shared__ const bf16* rowp[BM];
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp / WGN, wn = warp % WGN;
    const int m0 = (int)(blockIdx.x / nbn) * BM, n0 = (int)(blockIdx.x % nbn) * BN;
    const size_t fs = (size_t)Hp * Wp * C;
    for (int r = tid; r < BM; r += NT) {
        const int m = min(m0 + r, M - 1);
        const int xo = m % Wo, yo = (m / Wo) % Ho, to = m / (Wo * Ho);
        rowp[r] = P + (size_t)to * fs + ((size_t)yo * Wp + xo) * C;
    }
    __syncthreads();
    const int cpt = C / 32, nk = nt * 9 * cpt;
    int lk = 0, lc = 0, ltap = 0;
    auto load = [&](int slot) {
        const int kt = ltap / 9, ky = (ltap / 3) % 3, kx = ltap % 3;
        const size_t toff = kt * fs + (size_t)(ky * Wp + kx) * C + lc * 32;
        bf16 *a = sA + slot * BM * 32, *b = sB + slot * BN * 32;
#pragma unroll
        for (int i = tid; i < BM * 4; i += NT) {
            const int r = i >> 2, ch = i & 3;
            const bool ok = m0 + r < M;
            cp16(a + sw32(r, ch), rowp[r] + (ok ? toff : 0) + ch * 8, ok ? 16 : 0);
        }
#pragma unroll
        for (int i = tid; i < BN * 4; i += NT) {
            const int r = i >> 2, ch = i & 3;
            cp16(b + sw32(r, ch), Wt + (size_t)(n0 + r) * kp + (size_t)lk * 32 + ch * 8, 16);
        }
        lk++;
        if (++lc == cpt) lc = 0, ltap++;
    };
    float acc[MT][2 * NP][4];
#pragma unroll
    for (int i = 0; i < MT; i++)
#pragma unroll
        for (int j = 0; j < 2 * NP; j++) acc[i][j][0] = acc[i][j][1] = acc[i][j][2] = acc[i][j][3] = 0.f;
#pragma unroll
    for (int s = 0; s < ST - 1; s++) {
        if (s < nk) load(s);
        asm volatile("cp.async.commit_group;\n" ::);
    }
    for (int kc = 0; kc < nk; kc++) {
        asm volatile("cp.async.wait_group %0;\n" ::"n"(ST - 2));
        __syncthreads();
        if (kc + ST - 1 < nk) load((kc + ST - 1) % ST);
        asm volatile("cp.async.commit_group;\n" ::);
        const bf16 *a = sA + (kc % ST) * BM * 32, *b = sB + (kc % ST) * BN * 32;
#pragma unroll
        for (int ks = 0; ks < 2; ks++) {
            uint32_t af[MT][4], bfr[NP][4];
#pragma unroll
            for (int mt = 0; mt < MT; mt++) ldm4(af[mt], a + sw32(wm * WM + mt * 16 + (lane & 15), ks * 2 + (lane >> 4)));
#pragma unroll
            for (int p = 0; p < NP; p++)
                ldm4(bfr[p], b + sw32(wn * WN + p * 16 + (lane & 7) + (lane >> 4) * 8, ks * 2 + ((lane >> 3) & 1)));
#pragma unroll
            for (int mt = 0; mt < MT; mt++)
#pragma unroll
                for (int p = 0; p < NP; p++) {
                    mma(acc[mt][2 * p], af[mt], bfr[p][0], bfr[p][1]);
                    mma(acc[mt][2 * p + 1], af[mt], bfr[p][2], bfr[p][3]);
                }
        }
    }
    const int g = lane >> 2, t = lane & 3;
#pragma unroll
    for (int mt = 0; mt < MT; mt++)
#pragma unroll
        for (int j = 0; j < 2 * NP; j++) {
            const int n = n0 + wn * WN + j * 8 + t * 2;
            if (n >= N) continue;
#pragma unroll
            for (int h = 0; h < 2; h++) {
                const int m = m0 + wm * WM + mt * 16 + g + h * 8;
                if (m >= M) continue;
                const size_t o = (size_t)m * N + n;
                float v0 = rd(rd(acc[mt][j][2 * h]) + bias[n]);
                if (resid) v0 += ldf(resid + o);
                if (n + 1 < N) {
                    float v1 = rd(rd(acc[mt][j][2 * h + 1]) + bias[n + 1]);
                    if (resid) v1 += ldf(resid + o + 1);
                    if (N % 2 == 0) {
                        *(__nv_bfloat162*)(out + o) = __floats2bfloat162_rn(v0, v1);
                        continue;
                    }
                    out[o + 1] = __float2bfloat16(v1);
                }
                out[o] = __float2bfloat16(v0);
            }
        }
}

// SpaceToDepthDownsample tail: out [To][H/ss][W/ss][Co] = bf16(s2d(y)[o] + bf16(sum_g s2d(xd)[o g + i] / g)),
// s2d channel ((c st + i) ss + j) ss + k, y = conv output over the frames of xd (xd = x with frame 0 duplicated in front
// when st == 2: xd[j] = x[max(j - st + 1, 0)]).
__global__ void k_s2d_add(const bf16* __restrict__ y, const bf16* __restrict__ x, int H, int W, int Cy, int Cx, int st,
                          int ss, int g, int To, bf16* __restrict__ out) {
    const int Ho = H / ss, Wo = W / ss, S = st * ss * ss, Co = Cy * S;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)To * Ho * Wo * Co) return;
    const int o = (int)(i % Co);
    const size_t p = i / Co;
    const int wo = (int)(p % Wo), ho = (int)((p / Wo) % Ho), d = (int)(p / ((size_t)Wo * Ho));
    auto at = [&](const bf16* src, int C, int ch, int shift) {   // s2d element ch of a source with C channels
        const int c = ch / S, r = ch % S, ii = r / (ss * ss), jj = (r / ss) % ss, kk = r % ss;
        const int f = max(d * st + ii - shift, 0);
        return ldf(src + (((size_t)f * H + ho * ss + jj) * W + wo * ss + kk) * C + c);
    };
    float s = 0.f;
    for (int e = 0; e < g; e++) s += at(x, Cx, o * g + e, st - 1);
    out[i] = __float2bfloat16(at(y, Cy, o, 0) + rd(s / (float)g));
}
// [T][HW][128] bf16 -> f32 [128][T][HW] = bf16(bf16(z - mean) / std)
__global__ void k_lat_out(const bf16* __restrict__ z, int T, int HW, const bf16* __restrict__ sd, const bf16* __restrict__ mu,
                          float* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * HW * 128) return;
    const int c = (int)(i % 128);
    const size_t p = i / 128;
    const int t = (int)(p / HW);
    out[((size_t)c * T + t) * HW + p % HW] = rd(rd(ldf(z + i) - ldf(mu + c)) / ldf(sd + c));
}
// GroupNorm + SiLU (+ residual first): x = bf16(silu(bf16(bf16(y a + b) + x))) or, without resid, bf16(silu(bf16(y a + b)))
__global__ void k_gn_silu(const bf16* __restrict__ y, const float2* __restrict__ gs, const float* __restrict__ gw,
                          const float* __restrict__ gb, int C, size_t n, const bf16* __restrict__ resid, bf16* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const int c = (int)(i % C);
    const float2 s = gs[c / (C / 32)];
    const float a = s.y * gw[c], b = gb[c] - a * s.x;
    float v = rd(a * ldf(y + i) + b);
    if (resid) v = rd(v + ldf(resid + i));
    out[i] = __float2bfloat16(silu(v));
}
// PixelShuffle 2D: y [T][h][w][4C] (channel c*4 + p1*2 + p2) -> x [T][2h][2w][C]
__global__ void k_ps2d(const bf16* __restrict__ y, int T, int h, int w, int C, bf16* __restrict__ x) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * 4 * h * w * C) return;
    const int c = (int)(i % C);
    const size_t p = i / C;
    const int X = (int)(p % (2 * w)), Y = (int)((p / (2 * w)) % (2 * h)), t = (int)(p / ((size_t)4 * w * h));
    x[i] = y[(((size_t)t * h + Y / 2) * w + X / 2) * 4 * C + c * 4 + (Y % 2) * 2 + X % 2];
}

// ================================================================ model
struct Lin {
    bf16 *w = nullptr, *b = nullptr;
    int n = 0, k = 0;
};
struct NABlk {
    bf16 *n1 = nullptr, *n2 = nullptr, *qn = nullptr, *kn = nullptr;
    Lin qkv, proj, gu, down, ctx;   // gu = [w_gate; w_up]
    bf16 *ope1 = nullptr, *sh1 = nullptr, *ope2 = nullptr, *sh2 = nullptr;
};
struct Conv {
    bf16* w = nullptr;  // [np][nt][3][3][cp] zero-padded
    float* b = nullptr;
    int cin = 0, cp = 0, cout = 0, np = 0, nt = 3, kp = 0;
};
struct Res {
    Conv c1, c2;
    float *g1w = nullptr, *g1b = nullptr, *g2w = nullptr, *g2b = nullptr;   // upsampler GroupNorms
};
constexpr int DC[4] = {2048, 1024, 512, 512}, DD[4] = {4, 6, 4, 2}, DK[4][3] = {{3, 7, 7}, {3, 7, 7}, {3, 5, 5}, {3, 5, 5}};
constexpr int UP[4][4] = {{1, 2, 2, 2}, {2, 1, 1, 2}, {2, 2, 2, 1}, {2, 2, 2, 2}};   // stride (t, h, w), channel reduction

struct Vae {
    cudaStream_t st;
    Gemm g;
    StFile f;
    bf16 *sd = nullptr, *mu = nullptr;     // per-channel statistics
    // encoder
    Conv ecin, edown[4], ecout;
    std::vector<Res> eres[5];
    // decoder
    Lin cin, up[4], xin, cout;
    std::vector<NABlk> det[4], diff;
    bf16* nout = nullptr;
    float* inv = nullptr;                  // RoPE inverse frequencies: 8 (frames) then 12 (rows / columns)
    bf16 *temb = nullptr, *mod = nullptr;  // t_emb [384], shared AdaLN [7 x 256] at t = 1 (taps)
    const char* tapdir = nullptr;
};
struct Ups {
    cudaStream_t st;
    StFile f;
    bf16 *sd = nullptr, *mu = nullptr;
    Conv cin, upc, cout;
    float *n0w = nullptr, *n0b = nullptr;
    std::vector<Res> pre, post;
};

template <typename T>
T* up(const std::vector<T>& h) {
    T* d = (T*)dmalloc(h.size() * sizeof(T));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
const uint16_t* raw16(StFile& f, const std::string& name, size_t n) {
    const StTensor& t = f.get(name);
    if (t.dtype != "BF16" || t.bytes() != n * 2)
        DIE("%s: %s %llu bytes, expected BF16 x %zu", name.c_str(), t.dtype.c_str(), (unsigned long long)t.bytes(), n);
    return (const uint16_t*)f.ptr(t);
}
bf16* upb(StFile& f, const std::string& name, size_t n) {
    const uint16_t* p = raw16(f, name, n);
    return up(std::vector<bf16>((const bf16*)p, (const bf16*)p + n));
}
float hb(uint16_t v) {
    uint32_t u = (uint32_t)v << 16;
    float x;
    memcpy(&x, &u, 4);
    return x;
}
float* upf(StFile& f, const std::string& name, size_t n) {
    const uint16_t* p = raw16(f, name, n);
    std::vector<float> o(n);
    for (size_t i = 0; i < n; i++) o[i] = hb(p[i]);
    return up(o);
}
Lin load_lin(StFile& f, const std::string& p, int n, int k, bool bias) {
    Lin l;
    l.n = n, l.k = k, l.w = upb(f, p + ".weight", (size_t)n * k);
    if (bias) l.b = upb(f, p + ".bias", n);
    return l;
}
// conv weight [cout][cin][nt][3][3] -> [np][nt][3][3][cp]; keep = output channels kept (encoder conv_out: the means)
Conv load_conv(StFile& f, const std::string& p, int cin, int cout, int nt, int keep = 0) {
    Conv c;
    c.cin = cin, c.cout = keep ? keep : cout, c.nt = nt, c.cp = rup(cin, 32), c.np = rup(c.cout, c.cout % 128 ? 16 : 128);
    const int taps = nt * 9;
    c.kp = taps * c.cp;
    const uint16_t* w = raw16(f, p + ".weight", (size_t)cout * cin * taps);
    const uint16_t* b = raw16(f, p + ".bias", cout);
    std::vector<uint16_t> r((size_t)c.np * c.kp, 0);
    for (int o = 0; o < c.cout; o++)
        for (int i = 0; i < cin; i++)
            for (int t = 0; t < taps; t++) r[(size_t)o * c.kp + (size_t)t * c.cp + i] = w[((size_t)o * cin + i) * taps + t];
    c.w = (bf16*)up(r);
    std::vector<float> bf(c.cout);
    for (int o = 0; o < c.cout; o++) bf[o] = hb(b[o]);
    c.b = up(bf);
    return c;
}
void free_conv(Conv& c) { dfree(c.w), dfree(c.b), c.w = nullptr, c.b = nullptr; }
void free_lin(Lin& l) { dfree(l.w), dfree(l.b), l.w = l.b = nullptr; }

// NABlock / DiffusionNABlock weights
NABlk load_blk(Vae* v, const std::string& p, int C, bool diff) {
    StFile& f = v->f;
    NABlk b;
    const int hid = (C * 4 + 15) / 16 * 16;
    b.n1 = upb(f, p + ".norm1.weight", C), b.n2 = upb(f, p + ".norm2.weight", C);
    b.qkv = load_lin(f, p + ".attn.qkv", 3 * C, C, true);
    b.proj = load_lin(f, p + ".attn.proj", C, C, true);
    b.down = load_lin(f, p + ".mlp.w_down", C, hid, false);
    const uint16_t* wg = raw16(f, p + ".mlp.w_gate.weight", (size_t)hid * C);
    const uint16_t* wu = raw16(f, p + ".mlp.w_up.weight", (size_t)hid * C);
    std::vector<uint16_t> gu((size_t)2 * hid * C);
    std::copy(wg, wg + (size_t)hid * C, gu.begin());
    std::copy(wu, wu + (size_t)hid * C, gu.begin() + (size_t)hid * C);
    b.gu.n = 2 * hid, b.gu.k = C, b.gu.w = (bf16*)up(gu);
    const uint16_t* qn = raw16(f, p + ".attn.q_norm.weight", 64);
    std::vector<bf16> qs(64);
    for (int i = 0; i < 64; i++) qs[i] = __float2bfloat16(hb(qn[i]) * 0.125f);   // head_dim^-0.5 folded in (exact)
    b.qn = up(qs), b.kn = upb(f, p + ".attn.k_norm.weight", 64);
    if (diff) b.ctx = load_lin(f, p + ".context_proj", C, C, true);
    return b;
}
void free_blk(NABlk& b) {
    for (bf16* p : {b.n1, b.n2, b.qn, b.kn, b.ope1, b.sh1, b.ope2, b.sh2}) dfree(p);
    free_lin(b.qkv), free_lin(b.proj), free_lin(b.gu), free_lin(b.down), free_lin(b.ctx);
}

void tap(Vae* v, const std::string& name, const bf16* p, size_t rows, int cols, int ld) {
    if (!v->tapdir) return;
    CK(cudaStreamSynchronize(v->st));
    std::vector<bf16> h(rows * cols);
    CK(cudaMemcpy2D(h.data(), cols * 2, p, (size_t)ld * 2, cols * 2, rows, cudaMemcpyDeviceToHost));
    FILE* fp = fopen((std::string(v->tapdir) + "/" + name + ".bin").c_str(), "wb");
    if (!fp) DIE("cannot write tap %s", name.c_str());
    const int64_t hd[2] = {(int64_t)rows, cols};
    fwrite(hd, 8, 2, fp);
    fwrite(h.data(), 2, h.size(), fp);
    fclose(fp);
}

// ---------------- NA stage executor
struct SP {   // one NA stage on a [T][H][W][C] volume
    int T, H, W, C, heads, kt, kh, kw, TQ, HQ, WQ, chunk, R, hid;
};
SP plan(int T, int H, int W, int C, const int* k, int TQ, int HQ, int WQ) {
    SP s;
    s.T = T, s.H = H, s.W = W, s.C = C, s.heads = C / 64, s.hid = (C * 4 + 15) / 16 * 16;
    s.kt = std::min(k[0], T), s.kh = std::min(k[1], H), s.kw = std::min(k[2], W);
    s.TQ = TQ, s.HQ = HQ, s.WQ = WQ;
    const int hw = H * W;
    static const long rows = getenv("LTXVAE_ROWS") ? atol(getenv("LTXVAE_ROWS")) : 262144;   // time-chunk size (debug override)
    s.chunk = std::max<int>(TQ, (int)(rows / hw) / TQ * TQ);
    if (s.chunk >= T) s.chunk = T;
    s.R = std::min(T, s.chunk + s.kt - 1);
    return s;
}
inline int wstart_h(int i, int L, int k) { return std::min(std::max(i - k / 2, 0), L - k); }
constexpr size_t MLP_ROWS = 65536;
struct Work {
    bf16 *ring = nullptr, *A = nullptr, *na = nullptr, *y = nullptr, *hid = nullptr;
    size_t cring = 0, cA = 0, cna = 0, cy = 0, chid = 0;   // element capacities
    float2* tab = nullptr;                                  // RoPE tables (frames | rows | columns)
};
void work_need(Work& w, const SP& s) {
    const size_t hw = (size_t)s.H * s.W, rows = (size_t)s.chunk * hw;
    w.cring = std::max(w.cring, (size_t)s.R * hw * 3 * s.C);
    w.cA = std::max(w.cA, rows * s.C);
    w.cna = std::max(w.cna, rows * s.C);
    w.cy = std::max(w.cy, rows * s.C);
    w.chid = std::max(w.chid, std::min(rows, MLP_ROWS) * 2 * s.hid);
}
void work_alloc(Work& w) {
    w.ring = (bf16*)dmalloc(w.cring * 2), w.A = (bf16*)dmalloc(w.cA * 2), w.na = (bf16*)dmalloc(w.cna * 2);
    w.y = (bf16*)dmalloc(w.cy * 2), w.hid = (bf16*)dmalloc(w.chid * 2);
    w.tab = (float2*)dmalloc((size_t)(1024 * 8 + 2048 * 12 * 2) * sizeof(float2));
}
void work_free(Work& w) {
    for (void* p : {(void*)w.ring, (void*)w.A, (void*)w.na, (void*)w.y, (void*)w.hid, (void*)w.tab}) dfree(p);
    w = Work();
}

void add(cudaStream_t st, bf16* x, const bf16* y, size_t n) { k_add<<<nblk(n / 8, 256), 256, 0, st>>>(x, y, n / 8); }
void rmsmod(cudaStream_t st, const bf16* x, size_t M, int N, const bf16* w, const bf16* ope, const bf16* sh, bf16* out) {
    k_rmsmod<<<nblk(M, 8), 256, 0, st>>>(x, (int)M, N, w, ope, sh, 1e-6f, out);
}
void na_launch(cudaStream_t st, const bf16* q, const bf16* k, const bf16* v, int ld, bf16* o, int ldo, int T, int H, int W,
               int R, int kt, int kh, int kw, int t0, int t1, int TQ, int HQ, int WQ, int heads) {
    NaArgs a{q, k, v, o, ld, ldo, T, H, W, R, kt, kh, kw, t0, t1, TQ, HQ, WQ, (W + WQ - 1) / WQ, (H + HQ - 1) / HQ, heads,
             1.4426950408889634f};
    if (T > 1022 || H > 1023 || W > 1023 || TQ * HQ * WQ != 128) DIE("ltx_vae na3d: %dx%dx%d tile %dx%dx%d", T, H, W, TQ, HQ, WQ);
    const size_t grid = (size_t)((t1 - t0 + TQ - 1) / TQ) * a.nth * a.ntw * heads;
    k_na3d<<<(unsigned)grid, 256, 4 * 64 * 64 * sizeof(bf16), st>>>(a);
}

// one NA block in place on x [T][H][W][C]; ctx (stage 5): x += context_proj(ctx) first. tap0: write stage-0 taps.
void na_block(Vae* v, Work& wk, const SP& s, NABlk& b, bf16* x, const bf16* ctx, bool tap0) {
    const cudaStream_t st = v->st;
    const size_t hw = (size_t)s.H * s.W;
    const int C = s.C, C3 = 3 * C;
    const float2 *tt = wk.tab, *th = wk.tab + 1024 * 8, *tw = th + 2048 * 12;
    int produced = 0;
    for (int t0 = 0; t0 < s.T; t0 += s.chunk) {
        const int t1 = std::min(t0 + s.chunk, s.T);
        const int need = wstart_h(t1 - 1, s.T, s.kt) + s.kt;
        while (produced < need) {   // q/k/v of frames [produced, need) from the block input, in runs that do not wrap
            const int n = std::min(std::min(need - produced, s.chunk), s.R - produced % s.R);
            const size_t M = (size_t)n * hw;
            bf16* xr = x + (size_t)produced * hw * C;
            prof.mark(st, "other");
            if (ctx) {
                lin(v->g, (int)M, C, C, ctx + (size_t)produced * hw * C, C, b.ctx.w, b.ctx.b, wk.y, C);
                add(st, xr, wk.y, M * C);
            }
            rmsmod(st, xr, M, C, b.n1, b.ope1, b.sh1, wk.A);
            bf16* rr = wk.ring + (size_t)(produced % s.R) * hw * C3;
            lin(v->g, (int)M, C3, C, wk.A, C, b.qkv.w, b.qkv.b, rr, C3);
            prof.mark(st, "linear");
            k_rms_rope<<<nblk(M * 2 * s.heads * 32, 256), 256, 0, st>>>(wk.ring, C3, C, s.heads, produced, n, s.R, s.H, s.W,
                                                                         b.qn, b.kn, tt, th, tw);
            prof.mark(st, "rms_rope");
            produced += n;
        }
        if (tap0 && t0 == 0) tap(v, "dec.rms_rope0.q_out", wk.ring, (size_t)std::min(s.R, s.T) * hw, C, C3);
        na_launch(st, wk.ring, wk.ring + C, wk.ring + 2 * C, C3, wk.na, C, s.T, s.H, s.W, s.R, s.kt, s.kh, s.kw, t0, t1,
                  s.TQ, s.HQ, s.WQ, s.heads);
        prof.mark(st, C == 256 ? "na3d_11" : "na3d_small");
        if (tap0 && t0 == 0) tap(v, "dec.na3d0.out", wk.na, (size_t)(t1 - t0) * hw, C, C);
        const size_t Mq = (size_t)(t1 - t0) * hw;
        bf16* xq = x + (size_t)t0 * hw * C;
        lin(v->g, (int)Mq, C, C, wk.na, C, b.proj.w, b.proj.b, wk.y, C);
        add(st, xq, wk.y, Mq * C);
        for (size_t r0 = 0; r0 < Mq; r0 += MLP_ROWS) {
            const size_t m = std::min(MLP_ROWS, Mq - r0);
            bf16* xm = xq + r0 * C;
            rmsmod(st, xm, m, C, b.n2, b.ope2, b.sh2, wk.A);
            lin(v->g, (int)m, 2 * s.hid, C, wk.A, C, b.gu.w, nullptr, wk.hid, 2 * s.hid);
            k_swiglu<<<nblk(m * s.hid, 256), 256, 0, st>>>(wk.hid, m, s.hid);
            lin(v->g, (int)m, C, s.hid, wk.hid, 2 * s.hid, b.down.w, nullptr, wk.y, C);
            add(st, xm, wk.y, m * C);
        }
        prof.mark(st, "linear");
    }
}
// RoPE tables for a [T][H][W] stage volume (global positions)
void rope_tabs(Vae* v, Work& wk, int T, int H, int W) {
    const cudaStream_t st = v->st;
    if (T > 1024 || H > 2048 || W > 2048) DIE("ltx_vae: rope table %dx%dx%d", T, H, W);
    k_rope_tab<<<nblk((size_t)T * 8, 256), 256, 0, st>>>(T, 8, v->inv, wk.tab);
    k_rope_tab<<<nblk((size_t)H * 12, 256), 256, 0, st>>>(H, 12, v->inv + 8, wk.tab + 1024 * 8);
    k_rope_tab<<<nblk((size_t)W * 12, 256), 256, 0, st>>>(W, 12, v->inv + 8, wk.tab + 1024 * 8 + 2048 * 12);
}
// LinearPixelShuffleUpsample of x [T][H][W][C] -> out [To][H p2][W p3][Co] (frames past To not computed)
void upsample(Vae* v, Work& wk, Lin& l, const bf16* x, int T, int H, int W, int C, int p1, int p2, int p3, int Co, int To,
              bf16* out) {
    const cudaStream_t st = v->st;
    const size_t hw = (size_t)H * W;
    const int P = p1 * p2 * p3, drop = p1 == 2 ? 1 : 0, N = P * Co;
    const int tmax = std::min(T, (To - 1 + drop) / p1 + 1);
    const int n = std::max<int>(1, (int)std::min<size_t>(wk.cy / (hw * N), 1u << 20));
    if (hw * N > wk.cy) DIE("ltx_vae: upsample buffer");
    for (int fa = 0; fa < tmax; fa += n) {
        const int nf = std::min(n, tmax - fa);
        lin(v->g, (int)(nf * hw), N, C, x + (size_t)fa * hw * C, C, l.w, l.b, wk.y, N);
        k_shuffle<<<(unsigned)(nf * hw), 256, N * sizeof(bf16), st>>>(wk.y, fa, H, W, Co, p1, p2, p3, drop, To, out);
    }
}

// stage 5: x = conv_in_x_t(patchify(x_t)), 8 diffusion blocks over the context, norm_out, conv_out -> pixels
void stage5(Vae* v, Work& wk, const SP& p5, const bf16* ctx, bf16* x5, const bf16* noise, float* frames, bf16* raw) {
    const cudaStream_t st = v->st;
    const int F = p5.T, H = 4 * p5.H, W = 4 * p5.W;
    const size_t hw5 = (size_t)p5.H * p5.W;
    for (int f0 = 0; f0 < F; f0 += p5.chunk) {
        const int nf = std::min(p5.chunk, F - f0);
        const size_t M = (size_t)nf * hw5;
        k_patch<<<nblk(M * 48, 256), 256, 0, st>>>(noise, F, H, W, f0, nf, wk.A);
        lin(v->g, (int)M, 256, 48, wk.A, 48, v->xin.w, nullptr, x5 + (size_t)f0 * hw5 * 256, 256);
        k_bias<<<nblk(M * 256, 256), 256, 0, st>>>(x5 + (size_t)f0 * hw5 * 256, M * 256, 256, v->xin.b);
    }
    rope_tabs(v, wk, p5.T, p5.H, p5.W);
    prof.mark(st, "stage5_in");
    for (NABlk& b : v->diff) na_block(v, wk, p5, b, x5, ctx, false);
    for (int f0 = 0; f0 < F; f0 += p5.chunk) {
        const int nf = std::min(p5.chunk, F - f0);
        const size_t M = (size_t)nf * hw5;
        rmsmod(st, x5 + (size_t)f0 * hw5 * 256, M, 256, v->nout, nullptr, nullptr, wk.A);
        lin(v->g, (int)M, 48, 256, wk.A, 256, v->cout.w, v->cout.b, wk.y, 48);
        k_unpatch<<<nblk(M * 16, 256), 256, 0, st>>>(wk.y, F, H, W, f0, nf, raw, frames);
    }
}

// ---------------- encoder / upsampler conv helpers
struct CW {   // per-call conv work: padded input, pixel-norm scales, group-norm partials and statistics
    bf16* P = nullptr;
    float *pn = nullptr, *gpart = nullptr;
    float2* gst = nullptr;
    size_t cP = 0, cpn = 0, cgp = 0;
};
void pad(cudaStream_t st, CW& w, const bf16* x, int T, int H, int W, int C, int Cp, int Tp, int front, int zt, int mode,
         const float2* gs = nullptr, const float* gw = nullptr, const float* gb = nullptr) {
    const size_t m = (size_t)Tp * (H + 2) * (W + 2) * Cp;
    if (m > w.cP) DIE("ltx_vae: pad buffer %zu > %zu", m, w.cP);
    if (C % 8 == 0)
        k_pad<8><<<nblk(m / 8, 256), 256, 0, st>>>(x, T, H, W, C, Cp, Tp, front, zt, mode, w.pn, gs, gw, gb, w.P);
    else
        k_pad<1><<<nblk(m, 256), 256, 0, st>>>(x, T, H, W, C, Cp, Tp, front, zt, mode, w.pn, gs, gw, gb, w.P);
}
template <int BM, int BN, int WM, int WN, int ST>
void iconv_t(cudaStream_t s, const bf16* P, int Hp, int Wp, int Tp, const Conv& c, const bf16* resid, bf16* out) {
    constexpr int smem = ST * (BM + BN) * 32 * 2;
    static bool init = false;
    if (!init) {
        CK(cudaFuncSetAttribute(k_iconv<BM, BN, WM, WN, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        init = true;
    }
    const int Ho = Hp - 2, Wo = Wp - 2, To = Tp - c.nt + 1, M = To * Ho * Wo, nbn = c.np / BN;
    k_iconv<BM, BN, WM, WN, ST><<<(unsigned)(nblk(M, BM) * nbn), (BM / WM) * (BN / WN) * 32, smem, s>>>(
        P, Hp, Wp, c.cp, c.nt, Ho, Wo, M, c.w, c.kp, c.b, resid, c.cout, nbn, out);
}
// conv over the padded input w.P [Tp][H + 2][W + 2][cp] -> out [Tp - nt + 1][H][W][cout]
void conv(cudaStream_t st, CW& w, const Conv& c, int Tp, int H, int W, const bf16* resid, bf16* out) {
    if (c.cout % 128 == 0) iconv_t<128, 128, 64, 32, 2>(st, w.P, H + 2, W + 2, Tp, c, resid, out);
    else iconv_t<256, 16, 32, 16, 2>(st, w.P, H + 2, W + 2, Tp, c, resid, out);
}
void pixnorm(cudaStream_t st, CW& w, const bf16* x, size_t npix, int C) {
    if (npix > w.cpn) DIE("ltx_vae: pixel-norm buffer");
    k_pn<<<nblk(npix * 32, 256), 256, 0, st>>>(x, npix, C, w.pn);
}
void gn_stats(cudaStream_t st, CW& w, const bf16* x, size_t n, int C, float eps) {
    const int nch = (int)nblk(n, 256);
    if ((size_t)nch * C > w.cgp) DIE("ltx_vae: group-norm buffer");
    k_gn_part<0><<<(nch + 3) / 4, C / 2, 0, st>>>(x, n, C, nch, nullptr, w.gpart);
    k_gn_fin<<<32, 32, 0, st>>>(w.gpart, n, C, nch, 0, eps, w.gst);
    k_gn_part<1><<<(nch + 3) / 4, C / 2, 0, st>>>(x, n, C, nch, w.gst, w.gpart);
    k_gn_fin<<<32, 32, 0, st>>>(w.gpart, n, C, nch, 1, eps, w.gst);
}
void cw_free(CW& w) {
    for (void* p : {(void*)w.P, (void*)w.pn, (void*)w.gpart, (void*)w.gst}) dfree(p);
    w = CW();
}
}  // namespace

extern "C" {

void* ltx_vae_new(const char* path) {
    Vae* v = new Vae();
    if (!v->f.open(path)) DIE("cannot open %s", path);
    StFile& f = v->f;
    CK(cudaStreamCreate(&v->st));   // blocking on purpose (common.cuh)
    gemm_init(v->g, v->st);
    v->tapdir = getenv("LTXVAE_TAPS");
    v->sd = upb(f, "per_channel_statistics.std-of-means", 128), v->mu = upb(f, "per_channel_statistics.mean-of-means", 128);
    // encoder: conv_in 48 -> 128, res_x 4/6/4/2/2 at 128/256/512/1024/1024, downsamplers between
    v->ecin = load_conv(f, "encoder.conv_in.conv", 48, 128, 3);
    const int ech[5] = {128, 256, 512, 1024, 1024}, enl[5] = {4, 6, 4, 2, 2}, eblk[5] = {0, 2, 4, 6, 8};
    for (int l = 0; l < 5; l++)
        for (int j = 0; j < enl[l]; j++) {
            const std::string p = "encoder.down_blocks." + std::to_string(eblk[l]) + ".res_blocks." + std::to_string(j);
            Res r;
            r.c1 = load_conv(f, p + ".conv1.conv", ech[l], ech[l], 3), r.c2 = load_conv(f, p + ".conv2.conv", ech[l], ech[l], 3);
            v->eres[l].push_back(r);
        }
    const int dcin[4] = {128, 256, 512, 1024}, dcout[4] = {64, 256, 128, 128};
    for (int l = 0; l < 4; l++)
        v->edown[l] = load_conv(f, "encoder.down_blocks." + std::to_string(2 * l + 1) + ".conv.conv", dcin[l], dcout[l], 3);
    v->ecout = load_conv(f, "encoder.conv_out.conv", 1024, 129, 3, 128);
    // decoder
    v->cin = load_lin(f, "decoder.conv_in", 2048, 128, true);
    for (int s = 0; s < 4; s++) {
        for (int j = 0; j < DD[s]; j++)
            v->det[s].push_back(load_blk(v, "decoder.det_stages." + std::to_string(s) + "." + std::to_string(j), DC[s], false));
        const int P = UP[s][0] * UP[s][1] * UP[s][2];
        v->up[s] = load_lin(f, "decoder.upsamples." + std::to_string(s) + ".proj", P * DC[s] / UP[s][3], DC[s], true);
    }
    v->xin = load_lin(f, "decoder.conv_in_x_t", 256, 48, true);
    for (int j = 0; j < 8; j++) v->diff.push_back(load_blk(v, "decoder.diff_blocks." + std::to_string(j), 256, true));
    v->nout = upb(f, "decoder.norm_out.weight", 256);
    v->cout = load_lin(f, "decoder.conv_out", 48, 256, true);
    std::vector<float> inv(20);
    for (int i = 0; i < 8; i++) inv[i] = (float)(1.0 / std::pow(10000.0, (double)(2 * i) / 16.0));
    for (int i = 0; i < 12; i++) inv[8 + i] = (float)(1.0 / std::pow(10000.0, (double)(2 * i) / 24.0));
    v->inv = up(inv);
    // the decoder's single step runs at t = 1: t_emb and the shared AdaLN modulation are constants of the checkpoint
    {
        const cudaStream_t st = v->st;
        Lin m0 = load_lin(f, "decoder.t_embedder.mlp.0", 384, 256, true), m2 = load_lin(f, "decoder.t_embedder.mlp.2", 384, 384, true);
        Lin ad = load_lin(f, "decoder.shared_adaln.proj", 7 * 256, 384, true);
        bf16* emb = (bf16*)dmalloc(256 * 2);
        bf16* hbuf = (bf16*)dmalloc(384 * 2);
        v->temb = (bf16*)dmalloc(384 * 2), v->mod = (bf16*)dmalloc(7 * 256 * 2);
        k_temb_freq<<<1, 128, 0, st>>>(1000.f, emb);   // timestep_scale_multiplier * t, t = 1 (bf16 1000 exact)
        lin(v->g, 1, 384, 256, emb, 256, m0.w, m0.b, hbuf, 384);
        k_silu<<<2, 256, 0, st>>>(hbuf, hbuf, 384);
        lin(v->g, 1, 384, 384, hbuf, 384, m2.w, m2.b, v->temb, 384);
        k_silu<<<2, 256, 0, st>>>(v->temb, hbuf, 384);
        lin(v->g, 1, 7 * 256, 384, hbuf, 384, ad.w, ad.b, v->mod, 7 * 256);
        for (int j = 0; j < 8; j++) {
            NABlk& b = v->diff[j];
            bf16* tab = upb(f, "decoder.diff_blocks." + std::to_string(j) + ".scale_shift_table", 7 * 256);
            for (bf16** p : {&b.ope1, &b.sh1, &b.ope2, &b.sh2}) *p = (bf16*)dmalloc(256 * 2);
            k_modprep<<<1, 256, 0, st>>>(v->mod, tab, 256, b.ope1, b.sh1, b.ope2, b.sh2);
            CK(cudaStreamSynchronize(st));
            dfree(tab);
        }
        CK(cudaStreamSynchronize(st));
        tap(v, "dec.t_emb", v->temb, 1, 384, 384);
        tap(v, "dec.modulation", v->mod, 1, 7 * 256, 7 * 256);
        free_lin(m0), free_lin(m2), free_lin(ad), dfree(emb), dfree(hbuf);
    }
    v->f.close();
    CK(cudaDeviceSynchronize());
    return v;
}

int ltx_vae_frames(int T) { return 8 * T - 7; }
int ltx_vae_latent_frames(int F) { return (F - 1) / 8 + 1; }

// px [F][H][W][3] f32 device in 0..1 (frames past the last 8k + 1 are dropped, as CausalDiffusionVAE.encode),
// H and W multiples of 32 -> out [128][T][H/32][W/32] f32 device, the normalized latent (what vae.encode returns).
int ltx_vae_encode(void* h, const float* px, int F, int H, int W, float* out) {
    Vae* v = (Vae*)h;
    const cudaStream_t st = v->st;
    if (F < 1 || H % 32 || W % 32) DIE("ltx_vae_encode: %d frames %dx%d (H, W multiples of 32)", F, W, H);
    const int Fu = 1 + (F - 1) / 8 * 8, T = ltx_vae_latent_frames(Fu);
    g_peak = g_cur;
    prof.mark(st, "start");
    // level sizes: (frames, h, w, channels) of the activations and padded conv inputs
    struct Lv { int t, h, w; };
    const Lv L0{Fu, H / 4, W / 4}, L1{Fu, H / 8, W / 8}, L2{(Fu + 1) / 2, H / 8, W / 8}, L3{(L2.t + 1) / 2, H / 16, W / 16},
        L4{(L3.t + 1) / 2, H / 32, W / 32};
    auto vol = [](const Lv& l, int c) { return (size_t)l.t * l.h * l.w * c; };
    const size_t cact = std::max({vol(L0, 128), vol(L1, 256), vol(L2, 512), vol(L3, 1024), vol(L4, 1024),
                                  (size_t)(L1.t + 1) * L1.h * L1.w * 256, (size_t)(L2.t + 1) * L2.h * L2.w * 128});
    CW w;
    auto pv = [](const Lv& l, int front, int c) { return (size_t)(l.t + front) * (l.h + 2) * (l.w + 2) * c; };
    w.cP = std::max({pv(L0, 2, 128), pv(L1, 3, 256), pv(L2, 3, 512), pv(L3, 3, 1024), pv(L4, 2, 1024)});
    w.cpn = (size_t)L0.t * L0.h * L0.w;
    w.P = (bf16*)dmalloc(w.cP * 2), w.pn = (float*)dmalloc(w.cpn * 4);
    bf16* buf[3];
    for (bf16*& b : buf) b = (bf16*)dmalloc(cact * 2);
    k_px_patch<<<nblk(vol(L0, 48), 256), 256, 0, st>>>(px, Fu, H, W, buf[0]);
    pad(st, w, buf[0], L0.t, L0.h, L0.w, 48, v->ecin.cp, L0.t + 2, 2, 0, 0);
    conv(st, w, v->ecin, L0.t + 2, L0.h, L0.w, nullptr, buf[1]);
    bf16 *x = buf[1], *t1 = buf[0], *t2 = buf[2];
    auto res_level = [&](int l, const Lv& lv, int C) {
        for (Res& r : v->eres[l]) {
            const size_t npix = (size_t)lv.t * lv.h * lv.w;
            pixnorm(st, w, x, npix, C);
            pad(st, w, x, lv.t, lv.h, lv.w, C, C, lv.t + 2, 2, 0, 1);
            conv(st, w, r.c1, lv.t + 2, lv.h, lv.w, nullptr, t1);
            pixnorm(st, w, t1, npix, C);
            pad(st, w, t1, lv.t, lv.h, lv.w, C, C, lv.t + 2, 2, 0, 1);
            conv(st, w, r.c2, lv.t + 2, lv.h, lv.w, x, t2);
            std::swap(x, t2);
        }
    };
    // SpaceToDepthDownsample: conv over x (frame 0 duplicated first when st == 2), then s2d + grouped mean of s2d(x)
    auto down = [&](int l, const Lv& lv, int Cx, int stt, int ss, int Co) {
        const Conv& c = v->edown[l];
        const int front = stt == 2 ? 3 : 2, Ty = lv.t + stt - 1;
        pad(st, w, x, lv.t, lv.h, lv.w, Cx, Cx, Ty + 2, front, 0, 0);
        conv(st, w, c, Ty + 2, lv.h, lv.w, nullptr, t1);
        const int To = Ty / stt, g = Cx * stt * ss * ss / Co;
        const size_t n = (size_t)To * (lv.h / ss) * (lv.w / ss) * Co;
        k_s2d_add<<<nblk(n, 256), 256, 0, st>>>(t1, x, lv.h, lv.w, c.cout, Cx, stt, ss, g, To, t2);
        std::swap(x, t2);
    };
    res_level(0, L0, 128);
    down(0, L0, 128, 1, 2, 256);
    res_level(1, L1, 256);
    down(1, L1, 256, 2, 1, 512);
    res_level(2, L2, 512);
    down(2, L2, 512, 2, 2, 1024);
    res_level(3, L3, 1024);
    down(3, L3, 1024, 2, 2, 1024);
    res_level(4, L4, 1024);
    pixnorm(st, w, x, (size_t)L4.t * L4.h * L4.w, 1024);
    pad(st, w, x, L4.t, L4.h, L4.w, 1024, 1024, L4.t + 2, 2, 0, 1);
    conv(st, w, v->ecout, L4.t + 2, L4.h, L4.w, nullptr, t1);
    k_lat_out<<<nblk(vol(L4, 128), 256), 256, 0, st>>>(t1, L4.t, L4.h * L4.w, v->sd, v->mu, out);
    prof.mark(st, "encode");
    CK(cudaStreamSynchronize(st));
    prof.report("encode");
    for (bf16* b : buf) dfree(b);
    cw_free(w);
    CK(cudaGetLastError());
    return T;
}

// lat [128][T][lh][lw] f32 device (normalized, what the sampler returns), noise [3][F][32 lh][32 lw] bf16 device (the
// decoder's x_t, F = ltx_vae_frames(T)) -> frames [F][32 lh][32 lw][3] f32 in 0..1 and/or raw [3][F][H][W] bf16 x0
// (either may be null). Untiled. Returns F.
int ltx_vae_decode(void* h, const float* lat, int T, int lh, int lw, const void* noise, float* frames, void* raw) {
    Vae* v = (Vae*)h;
    const cudaStream_t st = v->st;
    if (T < 1) DIE("ltx_vae_decode: %d latents", T);
    const int T0 = T + 2, F = ltx_vae_frames(T), H5 = 8 * lh, W5 = 8 * lw;
    g_peak = g_cur;
    prof.mark(st, "start");
    // stage volumes: 0 [T0][lh][lw], 1 [T0][2lh][2lw], 2 [2T0-1][2lh][2lw], 3 [4T0-3][4lh][4lw], 5 [F][8lh][8lw]
    const int ST_[4][3] = {{T0, lh, lw}, {T0, 2 * lh, 2 * lw}, {2 * T0 - 1, 2 * lh, 2 * lw}, {4 * T0 - 3, 4 * lh, 4 * lw}};
    const int TL[5][3] = {{2, 8, 8}, {2, 8, 8}, {2, 8, 8}, {2, 8, 8}, {4, 4, 8}};   // query tiles
    SP sp[5];
    Work wk;
    size_t cx = 0;
    for (int s = 0; s < 4; s++) {
        sp[s] = plan(ST_[s][0], ST_[s][1], ST_[s][2], DC[s], DK[s], TL[s][0], TL[s][1], TL[s][2]);
        work_need(wk, sp[s]);
        cx = std::max(cx, (size_t)ST_[s][0] * ST_[s][1] * ST_[s][2] * DC[s]);
    }
    const int K5[3] = {11, 11, 11};
    sp[4] = plan(F, H5, W5, 256, K5, TL[4][0], TL[4][1], TL[4][2]);
    work_need(wk, sp[4]);
    for (int s = 0; s < 4; s++) {   // upsample GEMM output for one input frame
        const size_t hw = (size_t)ST_[s][1] * ST_[s][2];
        wk.cy = std::max(wk.cy, hw * UP[s][0] * UP[s][1] * UP[s][2] * DC[s] / UP[s][3]);
    }
    wk.cA = std::max(wk.cA, (size_t)sp[4].chunk * H5 * W5 * 48);
    work_alloc(wk);
    const size_t n5 = (size_t)F * H5 * W5 * 256;
    bf16* xa = (bf16*)dmalloc(cx * 2);
    bf16* xb = (bf16*)dmalloc(cx * 2);
    bf16* ctx = (bf16*)dmalloc(n5 * 2);
    // stage 0 input
    const size_t hw0 = (size_t)lh * lw;
    k_lat_in<<<nblk((size_t)T0 * hw0 * 128, 256), 256, 0, st>>>(lat, T, T0, (int)hw0, v->sd, v->mu, xb);
    lin(v->g, (int)(T0 * hw0), 2048, 128, xb, 128, v->cin.w, nullptr, xa, 2048);
    k_bias<<<nblk((size_t)T0 * hw0 * 2048, 256), 256, 0, st>>>(xa, (size_t)T0 * hw0 * 2048, 2048, v->cin.b);
    tap(v, "dec.conv_in_out", xa, (size_t)T0 * hw0, 2048, 2048);
    prof.mark(st, "stage0_in");
    bf16 *x = xa, *nx = xb;
    for (int s = 0; s < 4; s++) {
        const SP& p = sp[s];
        rope_tabs(v, wk, p.T, p.H, p.W);
        for (size_t j = 0; j < v->det[s].size(); j++) {
            na_block(v, wk, p, v->det[s][j], x, nullptr, s == 0 && j == 0);
            if (s == 0 && j == 0) tap(v, "dec.stage0.block0_out", x, (size_t)p.T * p.H * p.W, p.C, p.C);
        }
        const int Co = DC[s] / UP[s][3];
        const int To = s < 3 ? (UP[s][0] == 2 ? 2 * p.T - 1 : p.T) : F;   // the last upsample writes the cropped context
        bf16* dst = s < 3 ? nx : ctx;
        prof.mark(st, "other");
        upsample(v, wk, v->up[s], x, p.T, p.H, p.W, p.C, UP[s][0], UP[s][1], UP[s][2], Co, To, dst);
        prof.mark(st, "upsample");
        if (s < 3) {
            std::swap(x, nx);
            tap(v, "dec.up" + std::to_string(s) + "_out", x, (size_t)To * p.H * UP[s][1] * p.W * UP[s][2], Co, Co);
        }
    }
    dfree(xb);
    xb = nullptr;
    tap(v, "dec.context", ctx, (size_t)F * H5 * W5, 256, 256);
    // stage 5 (x5 reuses the stage-0..3 buffer when large enough)
    bf16* x5 = xa;
    if (cx < n5) dfree(xa), xa = nullptr, x5 = (bf16*)dmalloc(n5 * 2);
    stage5(v, wk, sp[4], ctx, x5, (const bf16*)noise, frames, (bf16*)raw);
    prof.mark(st, "out");
    CK(cudaStreamSynchronize(st));
    prof.report("decode");
    dfree(x5), dfree(ctx);
    if (xa && xa != x5) dfree(xa);
    work_free(wk);
    CK(cudaGetLastError());
    return F;
}

// test entry: stage 5 alone on a given context [F][h5][w5][256] bf16 device (h5 = 8 lh), noise as ltx_vae_decode
void ltx_vae_stage5(void* h, const void* ctx, int F, int h5, int w5, const void* noise, float* frames, void* raw) {
    Vae* v = (Vae*)h;
    const int K5[3] = {11, 11, 11};
    const SP p5 = plan(F, h5, w5, 256, K5, 4, 4, 8);
    Work wk;
    work_need(wk, p5);
    wk.cA = std::max(wk.cA, (size_t)p5.chunk * h5 * w5 * 48);
    work_alloc(wk);
    bf16* x5 = (bf16*)dmalloc((size_t)F * h5 * w5 * 256 * 2);
    stage5(v, wk, p5, (const bf16*)ctx, x5, (const bf16*)noise, frames, (bf16*)raw);
    CK(cudaStreamSynchronize(v->st));
    dfree(x5);
    work_free(wk);
    CK(cudaGetLastError());
}

// test entry: na3d on contiguous q, k, v, out [T][H][W][heads][64] bf16 device, kernel (kt, kh, kw), scale 1
void ltx_na3d(const void* q, const void* k, const void* v, void* o, int T, int H, int W, int heads, int kt, int kh, int kw,
              int TQ, int HQ, int WQ) {
    const int C = heads * 64;
    na_launch(0, (const bf16*)q, (const bf16*)k, (const bf16*)v, C, (bf16*)o, C, T, H, W, T, std::min(kt, T), std::min(kh, H),
              std::min(kw, W), 0, T, TQ, HQ, WQ, heads);
    CK(cudaDeviceSynchronize());
    CK(cudaGetLastError());
}

void ltx_vae_free(void* h) {
    Vae* v = (Vae*)h;
    cudaStreamSynchronize(v->st);
    free_conv(v->ecin), free_conv(v->ecout);
    for (Conv& c : v->edown) free_conv(c);
    for (auto& l : v->eres)
        for (Res& r : l) free_conv(r.c1), free_conv(r.c2);
    free_lin(v->cin), free_lin(v->xin), free_lin(v->cout);
    for (Lin& l : v->up) free_lin(l);
    for (auto& s : v->det)
        for (NABlk& b : s) free_blk(b);
    for (NABlk& b : v->diff) free_blk(b);
    for (void* p : {(void*)v->sd, (void*)v->mu, (void*)v->nout, (void*)v->inv, (void*)v->temb, (void*)v->mod}) dfree(p);
    gemm_free(v->g);
    cudaStreamDestroy(v->st);
    delete v;
}

// ---- latent upsampler x2 (weights: the upscaler file; per-channel statistics: the video VAE file)
void* ltx_up_new(const char* path, const char* vae_path) {
    Ups* u = new Ups();
    CK(cudaStreamCreate(&u->st));
    StFile vf;
    if (!vf.open(vae_path)) DIE("cannot open %s", vae_path);
    u->sd = upb(vf, "per_channel_statistics.std-of-means", 128), u->mu = upb(vf, "per_channel_statistics.mean-of-means", 128);
    vf.close();
    if (!u->f.open(path)) DIE("cannot open %s", path);
    StFile& f = u->f;
    u->cin = load_conv(f, "initial_conv", 128, 1024, 3);
    u->n0w = upf(f, "initial_norm.weight", 1024), u->n0b = upf(f, "initial_norm.bias", 1024);
    for (int k = 0; k < 2; k++)
        for (int j = 0; j < 4; j++) {
            const std::string p = std::string(k ? "post_upsample_res_blocks." : "res_blocks.") + std::to_string(j);
            Res r;
            r.c1 = load_conv(f, p + ".conv1", 1024, 1024, 3), r.c2 = load_conv(f, p + ".conv2", 1024, 1024, 3);
            r.g1w = upf(f, p + ".norm1.weight", 1024), r.g1b = upf(f, p + ".norm1.bias", 1024);
            r.g2w = upf(f, p + ".norm2.weight", 1024), r.g2b = upf(f, p + ".norm2.bias", 1024);
            (k ? u->post : u->pre).push_back(r);
        }
    u->upc = load_conv(f, "upsampler.0", 1024, 4096, 1);
    u->cout = load_conv(f, "final_conv", 1024, 128, 3);
    u->f.close();
    CK(cudaDeviceSynchronize());
    return u;
}

// lat [128][T][lh][lw] f32 device (normalized) -> out [128][T][2 lh][2 lw] f32 device (normalized), as the
// LTXVLatentUpsampler node (bf16 model, un_normalize / normalize with the VAE statistics)
void ltx_up_run(void* h, const float* lat, int T, int lh, int lw, float* out) {
    Ups* u = (Ups*)h;
    const cudaStream_t st = u->st;
    g_peak = g_cur;
    prof.mark(st, "start");
    const int H2 = 2 * lh, W2 = 2 * lw;
    const size_t n1 = (size_t)T * lh * lw, n2 = (size_t)T * H2 * W2;
    CW w;
    w.cP = std::max((size_t)(T + 2) * (H2 + 2) * (W2 + 2) * 1024, (size_t)T * (lh + 2) * (lw + 2) * 1024);
    w.cgp = nblk(n2, 256) * 1024;
    w.P = (bf16*)dmalloc(w.cP * 2), w.gpart = (float*)dmalloc(w.cgp * 4), w.gst = (float2*)dmalloc(32 * sizeof(float2));
    bf16* buf[3];
    for (bf16*& b : buf) b = (bf16*)dmalloc(std::max(n2 * 1024, n1 * 4096) * 2);
    k_lat_in<<<nblk(n1 * 128, 256), 256, 0, st>>>(lat, T, T, lh * lw, u->sd, u->mu, buf[0]);
    pad(st, w, buf[0], T, lh, lw, 128, 128, T + 2, 0, 1, 0);
    conv(st, w, u->cin, T + 2, lh, lw, nullptr, buf[1]);
    gn_stats(st, w, buf[1], n1, 1024, 1e-5f);
    k_gn_silu<<<nblk(n1 * 1024, 256), 256, 0, st>>>(buf[1], w.gst, u->n0w, u->n0b, 1024, n1 * 1024, nullptr, buf[0]);
    bf16 *x = buf[0], *t1 = buf[1], *t2 = buf[2];
    auto res = [&](Res& r, int hh, int ww) {
        const size_t n = (size_t)T * hh * ww;
        pad(st, w, x, T, hh, ww, 1024, 1024, T + 2, 0, 1, 0);
        conv(st, w, r.c1, T + 2, hh, ww, nullptr, t1);
        gn_stats(st, w, t1, n, 1024, 1e-5f);
        pad(st, w, t1, T, hh, ww, 1024, 1024, T + 2, 0, 1, 2, w.gst, r.g1w, r.g1b);
        conv(st, w, r.c2, T + 2, hh, ww, nullptr, t2);
        gn_stats(st, w, t2, n, 1024, 1e-5f);
        k_gn_silu<<<nblk(n * 1024, 256), 256, 0, st>>>(t2, w.gst, r.g2w, r.g2b, 1024, n * 1024, x, x);
    };
    for (Res& r : u->pre) res(r, lh, lw);
    pad(st, w, x, T, lh, lw, 1024, 1024, T, 0, 0, 0);   // per-frame Conv2d
    conv(st, w, u->upc, T, lh, lw, nullptr, t1);
    k_ps2d<<<nblk(n2 * 1024, 256), 256, 0, st>>>(t1, T, lh, lw, 1024, x);
    for (Res& r : u->post) res(r, H2, W2);
    pad(st, w, x, T, H2, W2, 1024, 1024, T + 2, 0, 1, 0);
    conv(st, w, u->cout, T + 2, H2, W2, nullptr, t1);
    k_lat_out<<<nblk(n2 * 128, 256), 256, 0, st>>>(t1, T, H2 * W2, u->sd, u->mu, out);
    prof.mark(st, "upsampler");
    CK(cudaStreamSynchronize(st));
    prof.report("upsample");
    for (bf16* b : buf) dfree(b);
    cw_free(w);
    CK(cudaGetLastError());
}

void ltx_up_free(void* h) {
    Ups* u = (Ups*)h;
    cudaStreamSynchronize(u->st);
    free_conv(u->cin), free_conv(u->upc), free_conv(u->cout);
    for (auto* l : {&u->pre, &u->post})
        for (Res& r : *l) {
            free_conv(r.c1), free_conv(r.c2);
            for (float* p : {r.g1w, r.g1b, r.g2w, r.g2b}) dfree(p);
        }
    for (void* p : {(void*)u->sd, (void*)u->mu, (void*)u->n0w, (void*)u->n0b}) dfree(p);
    cudaStreamDestroy(u->st);
    delete u;
}
}  // extern "C"
