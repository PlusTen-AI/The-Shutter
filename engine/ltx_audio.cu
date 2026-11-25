// shutter engine — LTX-2.5 audio decode (ComfyUI e9afbccc2 LTXVAudioVAEDecode -> comfy/ldm/lightricks/vae/audio_vae.py
// AudioVAE.decode), C API:
//   normalized latent [8, T, 16] -> per-channel denormalize in patch space (feature c*16 + f) -> CausalAudioAutoencoder
//   decoder (2-D convs over (time, mel), causal in time, PixelNorm, no attention) -> mel [2, 4T-3, 64] -> BigVGAN-v2
//   vocoder (AMP1, SnakeBeta, kaiser anti-alias, 16 kHz) -> BWE: causal STFT log-mel -> second vocoder (residual) +
//   hann x3 resampler (skip), clamp -> [2, 480 (4T-3)] at 48 kHz.
// Numerics as ComfyUI ran it with --bf16-vae (plain torch, bf16 weights and activations, nothing through comfy_kitchen):
// every torch op rounds to bf16 — convs accumulate in fp32 and round, then + bias rounds; PixelNorm, SiLU, SnakeBeta
// and the STFT magnitude step by step; residual adds rounded; x / 3 as x * (1/3) (torch's CPU-scalar division).
// Layouts: decoder activations NHWC [t][f][C]; vocoder activations [t][C] (stereo mel = channel c*64 + mel bin);
// waveforms planar [2][samples].
// Self-contained: own cuBLASLt wrapper and kernels (only common.cuh macros and st.h are shared).
#include "common.cuh"
#include "st.h"

#include <algorithm>
#include <array>
#include <cmath>
#include <cstring>
#include <map>
#include <string>
#include <vector>

namespace {
inline size_t nblk(size_t n, int t) { return (n + t - 1) / t; }
__device__ __forceinline__ float rd(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
inline float hround(float v) { return __bfloat162float(__float2bfloat16(v)); }  // host bf16 rounding

constexpr int COL_CAP = 16 << 20;  // im2col chunk (elements): convs run in row chunks of at most this many columns

void* dmalloc(size_t n) {
    void* p;
    CK(cudaMalloc(&p, std::max<size_t>(n, 16)));
    return p;
}
template <class T>
T* up(const std::vector<T>& h) {
    T* d = (T*)dmalloc(h.size() * sizeof(T));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}
std::vector<bf16> tobf(const std::vector<float>& v) {
    std::vector<bf16> o(v.size());
    for (size_t i = 0; i < v.size(); i++) o[i] = __float2bfloat16(v[i]);
    return o;
}

// ---------------- cuBLASLt: row-major C[M,N] (ldc) = A[M,K] (lda) . W[N,K]^T (ldw), bf16 in, fp32 accumulate ----------------
enum { G_BF16 = 0, G_F32 = 1 };  // C type
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
    std::map<std::array<int, 7>, Plan> plans;
};

void gemm(Gemm* g, int type, int M, int N, int K, const bf16* A, int lda, const bf16* W, int ldw, void* C, int ldc) {
    const std::array<int, 7> key{type, M, N, K, lda, ldw, ldc};
    auto it = g->plans.find(key);
    if (it == g->plans.end()) {
        Gemm::Plan p;
        CKL(cublasLtMatmulDescCreate(&p.op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
        cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
        CKL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
        CKL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
        CKL(cublasLtMatrixLayoutCreate(&p.lw, CUDA_R_16BF, K, N, ldw));
        CKL(cublasLtMatrixLayoutCreate(&p.la, CUDA_R_16BF, K, M, lda));
        CKL(cublasLtMatrixLayoutCreate(&p.lc, type == G_F32 ? CUDA_R_32F : CUDA_R_16BF, N, M, ldc));
        cublasLtMatmulPreference_t pref;
        CKL(cublasLtMatmulPreferenceCreate(&pref));
        CKL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &g->wsz, sizeof(g->wsz)));
        cublasLtMatmulHeuristicResult_t res;
        int n = 0;
        CKL(cublasLtMatmulAlgoGetHeuristic(g->h, p.op, p.lw, p.la, p.lc, p.lc, pref, 1, &res, &n));
        cublasLtMatmulPreferenceDestroy(pref);
        if (n == 0) DIE("ltx_audio: no cublasLt algo type %d M=%d N=%d K=%d", type, M, N, K);
        p.algo = res.algo;
        it = g->plans.emplace(key, p).first;
    }
    const float al = 1.f, be = 0.f;
    Gemm::Plan& p = it->second;
    CKL(cublasLtMatmul(g->h, p.op, &al, W, p.lw, A, p.la, &be, C, p.lc, C, p.lc, &p.algo, g->ws, g->wsz, g->st));
}

// ================= kernels: audio VAE decoder (NHWC [t][f][C]) =================
// latent [8][T][16] f32 -> x [T][16][8] = bf16(bf16(bf16(z) * std) + mean), statistics of feature c*16 + f; tap [8][T][16]
__global__ void k_denorm(const float* __restrict__ lat, int T, const float* __restrict__ sd, const float* __restrict__ mn,
                         bf16* __restrict__ x, bf16* __restrict__ tap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 8 * T * 16) return;
    const int c = i / (T * 16), t = (i / 16) % T, f = i % 16, s = c * 16 + f;
    const bf16 v = __float2bfloat16(rd(rd(lat[i]) * sd[s]) + mn[s]);
    x[((size_t)t * 16 + f) * 8 + c] = v;
    if (tap) tap[i] = v;
}

// PixelNorm over channels then SiLU, op by op: m = bf16(sum(bf16(x^2)) * (1/C)), d = bf16(sqrt(bf16(m + 1e-6))),
// y = bf16(x / d), out = bf16(y / (1 + exp(-y))). One warp per pixel, x [P][C]
__global__ void k_pnsilu(const bf16* __restrict__ x, int P, int C, bf16* __restrict__ out) {
    const int w = (int)((blockIdx.x * (size_t)blockDim.x + threadIdx.x) >> 5), l = threadIdx.x & 31;
    if (w >= P) return;
    const bf16* r = x + (size_t)w * C;
    float s = 0.f;
    for (int c = l; c < C; c += 32) {
        const float v = ld(r + c);
        s += rd(v * v);
    }
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    const float d = rd(sqrtf(rd(rd(s * (1.f / C)) + 1e-6f)));
    for (int c = l; c < C; c += 32) {
        const float v = rd(ld(r + c) / d);
        out[(size_t)w * C + c] = __float2bfloat16(v / (1.f + expf(-v)));
    }
}

// 3x3 causal conv im2col (time padded 2 before / 0 after, mel 1 each side, zeros) on the input nearest-upsampled by
// `up` (1 or 2), output rows shifted by r0 (Upsample drops the first time row): col [n][9 C] for output pixels
// p0 .. p0+n-1 of the [Ho][Wo] grid. C % 8 == 0
__global__ void k_im2col2(const bf16* __restrict__ x, int H, int W, int C, int up, int r0, int Wo, int p0, int n,
                          bf16* __restrict__ col) {
    const int C8 = C / 8;
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)n * 9 * C8) return;
    const int c = (int)(i % C8) * 8, tap = (int)((i / C8) % 9), p = p0 + (int)(i / ((size_t)9 * C8));
    const int yy = p / Wo + r0 + tap / 3 - 2, xx = p % Wo + tap % 3 - 1;
    uint4 v = make_uint4(0, 0, 0, 0);
    if (yy >= 0 && yy < up * H && xx >= 0 && xx < up * W)
        v = *(const uint4*)(x + ((size_t)(yy / up) * W + xx / up) * C + c);
    *(uint4*)(col + i * 8) = v;
}

// y [M][N] (GEMM out, bf16) -> bf16(y + b), then + r rounded when r is given
__global__ void k_bias(bf16* __restrict__ y, size_t M, int N, const float* __restrict__ b, const bf16* __restrict__ r) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    float v = rd(ld(y + i) + b[i % N]);
    if (r) v += ld(r + i);
    y[i] = __float2bfloat16(v);
}

// conv_out GEMM out y [M*64][16] (channels 0, 1 real) -> vocoder input x [M][128] (channel c*64 + mel bin) = bf16(y + b);
// tap = ComfyUI's mel [2][M][64]
__global__ void k_melout(const bf16* __restrict__ y, int M, const float* __restrict__ b, bf16* __restrict__ x,
                         bf16* __restrict__ tap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * 128) return;
    const int c = i & 1, p = i >> 1, t = p / 64, f = p % 64;
    const bf16 v = __float2bfloat16(ld(y + (size_t)p * 16 + c) + b[c]);
    x[(size_t)t * 128 + c * 64 + f] = v;
    if (tap) tap[((size_t)c * M + t) * 64 + f] = v;
}

// mel [2][M][64] -> vocoder input x [M][128]
__global__ void k_melin(const bf16* __restrict__ mel, int M, bf16* __restrict__ x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * M * 64) return;
    const int c = i / (M * 64), t = (i / 64) % M, f = i % 64;
    x[(size_t)t * 128 + c * 64 + f] = mel[i];
}

// ================= kernels: vocoder ([t][C]) =================
// conv1d im2col: col [n][k C] = x[t + j d - p][c] for t = t0 .. t0+n-1 (zero outside), C % 8 == 0
__global__ void k_im2col1(const bf16* __restrict__ x, int T, int C, int k, int d, int p, int t0, int n,
                          bf16* __restrict__ col) {
    const int C8 = C / 8;
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)n * k * C8) return;
    const int c = (int)(i % C8) * 8, j = (int)((i / C8) % k);
    const int t = t0 + (int)(i / ((size_t)C8 * k)), ti = t + j * d - p;
    *(uint4*)(col + i * 8) = (ti >= 0 && ti < T) ? *(const uint4*)(x + (size_t)ti * C + c) : make_uint4(0, 0, 0, 0);
}

// conv_transpose1d overlap-add: Y [Tin][cout*k] f32 -> out [Tout][cout] = bf16(bf16(sum) + b)
__global__ void k_col2im(const float* __restrict__ Y, int Tin, int co, int k, int u, int p, int Tout,
                         const float* __restrict__ b, bf16* __restrict__ out) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)Tout * co) return;
    const int o = (int)(i % co), t = (int)(i / co);
    float s = 0.f;
    for (int j = (t + p) % u; j < k; j += u) {
        const int ti = (t + p - j) / u;
        if (ti >= 0 && ti < Tin) s += Y[((size_t)ti * co + o) * k + j];
    }
    out[i] = __float2bfloat16(rd(s) + b[o]);
}

// Activation1d first half: UpSample1d (replicate pad 5, depthwise conv_transpose stride 2 with the 12-tap kaiser
// filter, rounded, x2, crop 15), then SnakeBeta step by step: x + bf16(1/beta) * bf16(sin(bf16(alpha x))^2).
// x [T][C] -> u [2T][C]
__global__ void k_upsnake(const bf16* __restrict__ x, int T, int C, const float* __restrict__ f,
                          const float* __restrict__ a, const float* __restrict__ ib, bf16* __restrict__ u) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
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

// Activation1d second half: DownSample1d = replicate pad (5, 6), depthwise conv1d stride 2, 12 taps. s [2T][C] -> [T][C]
__global__ void k_down(const bf16* __restrict__ s, int T, int C, const float* __restrict__ f, bf16* __restrict__ out) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)T * C) return;
    const int c = (int)(i % C), t = (int)(i / C);
    float acc = 0.f;
    for (int j = 0; j < 12; j++) acc += ld(s + (size_t)min(max(2 * t + j - 5, 0), 2 * T - 1) * C + c) * f[j];
    out[i] = __float2bfloat16(acc);
}

// mode 0: out = bf16(a + b); mode 1: out = bf16(a * (1/3))
__global__ void k_ew(int mode, const bf16* __restrict__ a, const bf16* __restrict__ b, bf16* __restrict__ out, size_t n) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= n) return;
    out[i] = __float2bfloat16(mode == 0 ? ld(a + i) + ld(b + i) : ld(a + i) * (1.f / 3.f));
}

// conv_post (C -> 2, 7 taps, pad 3, no bias), rounded, clamped to [-1, 1] when clamp: x [L][C] -> out planar [2][ldo]
__global__ void k_post(const bf16* __restrict__ x, int L, int C, const float* __restrict__ w, int clamp,
                       bf16* __restrict__ out, int ldo) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * L) return;
    const int o = i / L, t = i % L;
    float s = 0.f;
    for (int j = 0; j < 7; j++) {
        const int ti = t + j - 3;
        if (ti < 0 || ti >= L) continue;
        const bf16* xr = x + (size_t)ti * C;
        const float* wr = w + (size_t)o * C * 7 + j;
        for (int c = 0; c < C; c++) s += ld(xr + c) * wr[c * 7];
    }
    float v = rd(s);
    if (clamp) v = fminf(fmaxf(v, -1.f), 1.f);
    out[(size_t)o * ldo + t] = __float2bfloat16(v);
}

// ================= kernels: BWE =================
// causal STFT frames (zero left pad 432 = n_fft - hop, hop 80, 512 taps): col [2F][512], row c*F + i = wav[c][80 i + k - 432]
__global__ void k_frames(const bf16* __restrict__ w, int Lp, int F, bf16* __restrict__ col) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)2 * F * 64) return;
    const int k = (int)(i % 64) * 8, r = (int)(i / 64), c = r / F, s = (r % F) * 80 + k - 432;
    *(uint4*)(col + i * 8) = s >= 0 ? *(const uint4*)(w + (size_t)c * Lp + s) : make_uint4(0, 0, 0, 0);
}

// spec [R][520] (real 0..256, imag 257..513) -> mag [R][264] = bf16(sqrt(bf16(bf16(re^2) + bf16(im^2)))), zero padded
__global__ void k_mag(const bf16* __restrict__ spec, int R, bf16* __restrict__ mag) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)R * 264) return;
    const int f = (int)(i % 264);
    const bf16* r = spec + (i / 264) * 520;
    float v = 0.f;
    if (f < 257) {
        const float re = ld(r + f), im = ld(r + 257 + f);
        v = sqrtf(rd(rd(re * re) + rd(im * im)));
    }
    mag[i] = __float2bfloat16(v);
}

// log-mel: m [2F][64] -> generator input x [F][128] (channel c*64 + j) = bf16(log(max(m, bf16(1e-5)))); tap [2][64][F]
__global__ void k_logmel(const bf16* __restrict__ m, int F, bf16* __restrict__ x, bf16* __restrict__ tap) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= 2 * F * 64) return;
    const int j = i % 64, r = i / 64, c = r / F, fi = r % F;
    const float lo = __bfloat162float(__float2bfloat16(1e-5f));
    const bf16 v = __float2bfloat16(logf(fmaxf(ld(m + i), lo)));
    x[(size_t)fi * 128 + c * 64 + j] = v;
    if (tap) tap[((size_t)c * 64 + j) * F + fi] = v;
}

// resampler: UpSample1d window "hann" ratio 3 (replicate pad 7, depthwise conv_transpose stride 3 with 43 taps,
// rounded, x3 rounded, crop 42): w [2][Lp] -> s [2][3 Lp]
__global__ void k_hann(const bf16* __restrict__ w, int Lp, const float* __restrict__ f, bf16* __restrict__ s) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)6 * Lp) return;
    const int c = (int)(i / (3 * (size_t)Lp)), n = (int)(i % (3 * (size_t)Lp)) + 42;
    const bf16* r = w + (size_t)c * Lp;
    float acc = 0.f;
    for (int j = n % 3; j < 43; j += 3) acc += ld(r + min(max((n - j) / 3 - 7, 0), Lp - 1)) * f[j];
    s[i] = __float2bfloat16(rd(acc) * 3.f);
}

// out [2][n] f32 = clamp(bf16(res + skip), -1, 1); res / skip rows of ldr
__global__ void k_final(const bf16* __restrict__ r, const bf16* __restrict__ s, int ldr, int n, float* __restrict__ out) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)2 * n) return;
    const size_t j = (i / n) * ldr + i % n;
    out[i] = fminf(fmaxf(rd(ld(r + j) + ld(s + j)), -1.f), 1.f);
}

// ================= model =================
struct Conv2 {  // 2-D conv, k 3 (causal) or 1: w [np][k*k*cin] bf16 ((kt, kf, cin) order), np = cout rounded up for N
    bf16* w = nullptr;
    float* b = nullptr;
    int cin = 0, cout = 0, np = 0, k = 3;
};
struct Res2 {  // ResnetBlock (PixelNorm, no temb); sc = nin_shortcut when cin != cout
    Conv2 c1, c2, sc;
};
struct AConv {  // conv1d: w [cout][k*cin]; convT: w [cout*k][cin]
    bf16* w = nullptr;
    float* b = nullptr;
    int cin = 0, cout = 0, k = 1, d = 1, p = 0, u = 1;
};
struct AAct {  // Activation1d(SnakeBeta): exp(alpha), bf16(1 / (exp(beta) + 1e-9)), up / down kaiser filters
    float *a = nullptr, *ib = nullptr, *fu = nullptr, *fd = nullptr;
};
struct AmpBlk {
    AConv c1[3], c2[3];
    AAct a1[3], a2[3];
};
struct Voc {  // BigVGAN-v2 generator (resblock AMP1, snakebeta), stereo mel in (128 channels)
    int c0 = 0, clast = 0;
    AConv pre;
    std::vector<AConv> ups;
    std::vector<AmpBlk> rb;
    AAct post_act;
    float* post = nullptr;  // conv_post [2][clast][7], no bias
};
struct Buf {
    void* p = nullptr;
    size_t cap = 0;
};

struct Aud {
    cudaStream_t st;
    Gemm g;
    StFile f;
    float *mean = nullptr, *std = nullptr;
    Conv2 cin, ups[3], cout;  // ups[1], ups[2] = up.{1,2}.upsample.conv
    Res2 mid[2], blk[3][3];   // blk[level][block]
    Voc voc, bwe;
    bf16 *basis = nullptr, *melb = nullptr;  // STFT forward basis [520][512] (514 real), mel basis [64][264] (257 real)
    float* hann = nullptr;                   // resampler filter, 43 taps
    Buf col, dec[4], vin, vb[6], Y, wav, fcol, spec, mag, melv, res, skip;
    bf16* tap[6] = {};  // denorm, mel, voc16k, bwe_mel, bwe_residual, bwe_skip (ComfyUI layouts)
};

// grow a scratch buffer (waits for the stream before freeing the old one)
void* grow(Aud* a, Buf& b, size_t bytes) {
    if (bytes > b.cap) {
        CK(cudaStreamSynchronize(a->st));
        cudaFree(b.p);
        b.p = dmalloc(bytes);
        b.cap = bytes;
    }
    return b.p;
}

std::vector<float> wget(Aud* a, const std::string& name, size_t n) {  // BF16 / F32 tensor -> float
    const StTensor& t = a->f.get(name);
    std::vector<float> o(n);
    if (t.dtype == "BF16" && t.bytes() == n * 2) {
        const uint16_t* p = (const uint16_t*)a->f.ptr(t);
        for (size_t i = 0; i < n; i++) {
            const uint32_t u = (uint32_t)p[i] << 16;
            memcpy(&o[i], &u, 4);
        }
    } else if (t.dtype == "F32" && t.bytes() == n * 4) {
        memcpy(o.data(), a->f.ptr(t), n * 4);
        for (float& x : o) x = hround(x);
    } else {
        DIE("%s: %s %llu bytes, expected %zu values", name.c_str(), t.dtype.c_str(), (unsigned long long)t.bytes(), n);
    }
    return o;
}

Conv2 load_conv2(Aud* a, const std::string& p, int cin, int cout, int k) {
    Conv2 c;
    c.cin = cin, c.cout = cout, c.k = k, c.np = (cout + 15) / 16 * 16;
    const std::vector<float> w = wget(a, p + ".conv.weight", (size_t)cout * cin * k * k);
    std::vector<float> r((size_t)c.np * k * k * cin, 0.f);
    for (int o = 0; o < cout; o++)
        for (int i = 0; i < cin; i++)
            for (int t = 0; t < k * k; t++) r[((size_t)o * k * k + t) * cin + i] = w[((size_t)o * cin + i) * k * k + t];
    c.w = up(tobf(r));
    c.b = up(wget(a, p + ".conv.bias", cout));
    return c;
}

Res2 load_res2(Aud* a, const std::string& p, int cin, int cout) {
    Res2 r;
    r.c1 = load_conv2(a, p + ".conv1", cin, cout, 3);
    r.c2 = load_conv2(a, p + ".conv2", cout, cout, 3);
    if (cin != cout) r.sc = load_conv2(a, p + ".nin_shortcut", cin, cout, 1);
    return r;
}

AConv load_aconv(Aud* a, const std::string& p, int cin, int cout, int k, int d, bool transposed, int u = 1) {
    AConv c;
    c.cin = cin, c.cout = cout, c.k = k, c.d = d, c.u = u;
    c.p = transposed ? (k - u) / 2 : (k * d - d) / 2;
    const std::vector<float> w = wget(a, p + ".weight", (size_t)cin * cout * k);
    std::vector<float> r(w.size());
    for (int o = 0; o < cout; o++)
        for (int i = 0; i < cin; i++)
            for (int j = 0; j < k; j++) {
                if (transposed) r[((size_t)o * k + j) * cin + i] = w[((size_t)i * cout + o) * k + j];
                else r[(size_t)o * k * cin + j * cin + i] = w[((size_t)o * cin + i) * k + j];
            }
    c.w = up(tobf(r));
    c.b = up(wget(a, p + ".bias", cout));
    return c;
}

AAct load_act(Aud* a, const std::string& p, int C) {
    std::vector<float> al = wget(a, p + ".act.alpha", C), be = wget(a, p + ".act.beta", C);
    for (int c = 0; c < C; c++) {
        al[c] = hround(expf(al[c]));
        be[c] = hround(1.f / hround(hround(expf(be[c])) + 1e-9f));
    }
    return AAct{up(al), up(be), up(wget(a, p + ".upsample.filter", 12)), up(wget(a, p + ".downsample.lowpass.filter", 12))};
}

Voc load_voc(Aud* a, const std::string& p, int c0, const std::vector<int>& ur, const std::vector<int>& uk) {
    Voc v;
    v.c0 = c0;
    v.pre = load_aconv(a, p + "conv_pre", 128, c0, 7, 1, false);
    const int rk[3] = {3, 7, 11}, dil[3] = {1, 3, 5};
    int ci = c0;
    for (size_t i = 0; i < ur.size(); i++) {
        const int co = ci / 2;
        v.ups.push_back(load_aconv(a, p + "ups." + std::to_string(i), ci, co, uk[i], 1, true, ur[i]));
        for (int j = 0; j < 3; j++) {
            AmpBlk r;
            const std::string q = p + "resblocks." + std::to_string(i * 3 + j) + ".";
            for (int l = 0; l < 3; l++) {
                const std::string s = std::to_string(l);
                r.c1[l] = load_aconv(a, q + "convs1." + s, co, co, rk[j], dil[l], false);
                r.c2[l] = load_aconv(a, q + "convs2." + s, co, co, rk[j], 1, false);
                r.a1[l] = load_act(a, q + "acts1." + s, co);
                r.a2[l] = load_act(a, q + "acts2." + s, co);
            }
            v.rb.push_back(r);
        }
        ci = co;
    }
    v.clast = ci;
    v.post_act = load_act(a, p + "act_post", ci);
    v.post = up(wget(a, p + "conv_post.weight", (size_t)2 * ci * 7));
    return v;
}

// UpSample1d(ratio 3, window "hann") filter, computed in fp32 in torch's op order, then cast to bf16 with the module
std::vector<float> hann_filter() {
    const float roll = 0.99f, pi = 3.14159265358979323846f;
    const int ratio = 3, lpw = 6, width = (int)std::ceil(lpw / 0.99), K = 2 * width * ratio + 1;  // width 7, 43 taps
    std::vector<float> f(K);
    for (int i = 0; i < K; i++) {
        const float t = ((float)i / (float)ratio - (float)width) * roll;
        const float tc = std::min(std::max(t, -(float)lpw), (float)lpw);
        float w = std::cos(tc * pi / (float)lpw / 2.f);
        w = w * w;
        const float pt = pi * t;
        const float sinc = t == 0.f ? 1.f : std::sin(pt) / pt;
        f[i] = hround(sinc * w * roll / (float)ratio);
    }
    return f;
}

// ---------------- runners ----------------
bf16* col_buf(Aud* a) { return (bf16*)grow(a, a->col, (size_t)COL_CAP * 2); }

// 2-D conv: x [H*W][cin] -> out [Ho*Wo][np] (Ho = up H - r0, Wo = up W); + bias (+ resid) when np == cout
void conv2(Aud* a, const Conv2& c, const bf16* x, int H, int W, int up, int r0, bf16* out, const bf16* resid) {
    const int Ho = up * H - r0, Wo = up * W, P = Ho * Wo, N = c.np;
    if (c.k == 1) {
        gemm(&a->g, G_BF16, P, N, c.cin, x, c.cin, c.w, c.cin, out, N);
    } else {
        const int K = 9 * c.cin, rows = std::max(1, COL_CAP / K);
        bf16* col = col_buf(a);
        for (int p0 = 0; p0 < P; p0 += rows) {
            const int n = std::min(rows, P - p0);
            k_im2col2<<<nblk((size_t)n * K / 8, 256), 256, 0, a->st>>>(x, H, W, c.cin, up, r0, Wo, p0, n, col);
            gemm(&a->g, G_BF16, n, N, K, col, K, c.w, K, out + (size_t)p0 * N, N);
        }
    }
    if (N == c.cout) k_bias<<<nblk((size_t)P * N, 256), 256, 0, a->st>>>(out, (size_t)P, N, c.b, resid);
}

void pnsilu(Aud* a, const bf16* x, int P, int C, bf16* out) {
    k_pnsilu<<<nblk((size_t)P * 32, 256), 256, 0, a->st>>>(x, P, C, out);
}

// ResnetBlock: x [P][cin] -> out [P][cout] = shortcut(x) + conv2(silu(pn(conv1(silu(pn(x)))))); t1, t2 scratch
void res2(Aud* a, const Res2& r, const bf16* x, int H, int W, bf16* out, bf16* t1, bf16* t2) {
    const int P = H * W;
    pnsilu(a, x, P, r.c1.cin, t1);
    conv2(a, r.c1, t1, H, W, 1, 0, t2, nullptr);
    pnsilu(a, t2, P, r.c1.cout, t1);
    const bf16* s = x;
    if (r.sc.w) {
        conv2(a, r.sc, x, H, W, 1, 0, t2, nullptr);
        s = t2;
    }
    conv2(a, r.c2, t1, H, W, 1, 0, out, s);
}

// denormalize + decoder: latent [8][T][16] f32 -> vocoder input vin [4T-3][128]; returns the mel frame count
int dec_run(Aud* a, const float* lat, int T, bf16* vin) {
    const size_t need = std::max({(size_t)8192 * T, (size_t)16384 * (2 * T - 1), (size_t)16384 * (4 * T - 3)});
    bf16* d[4];
    for (int i = 0; i < 4; i++) d[i] = (bf16*)grow(a, a->dec[i], need * 2);
    const cudaStream_t st = a->st;
    k_denorm<<<nblk((size_t)T * 128, 256), 256, 0, st>>>(lat, T, a->std, a->mean, d[3], a->tap[0]);
    int H = T, W = 16;
    bf16 *x = d[0], *o = d[1];
    conv2(a, a->cin, d[3], H, W, 1, 0, x, nullptr);
    for (int i = 0; i < 2; i++) res2(a, a->mid[i], x, H, W, o, d[2], d[3]), std::swap(x, o);
    for (int lvl = 2; lvl >= 0; lvl--) {
        for (int b = 0; b < 3; b++) res2(a, a->blk[lvl][b], x, H, W, o, d[2], d[3]), std::swap(x, o);
        if (lvl) {
            conv2(a, a->ups[lvl], x, H, W, 2, 1, o, nullptr);
            std::swap(x, o);
            H = 2 * H - 1, W *= 2;
        }
    }
    pnsilu(a, x, H * W, 128, o);
    conv2(a, a->cout, o, H, W, 1, 0, d[2], nullptr);
    k_melout<<<nblk((size_t)H * 128, 256), 256, 0, st>>>(d[2], H, a->cout.b, vin, a->tap[1]);
    return H;
}

// conv1d (zero pad, bias): x [T][cin] -> out [T][cout]; resid: out = bf16(out + resid)
void aconv1(Aud* a, const AConv& c, const bf16* x, int T, bf16* out, const bf16* resid) {
    const int K = c.k * c.cin;
    if (c.k == 1) {
        gemm(&a->g, G_BF16, T, c.cout, K, x, K, c.w, K, out, c.cout);
    } else {
        const int rows = std::max(1, COL_CAP / K);
        bf16* col = col_buf(a);
        for (int t0 = 0; t0 < T; t0 += rows) {
            const int n = std::min(rows, T - t0);
            k_im2col1<<<nblk((size_t)n * K / 8, 256), 256, 0, a->st>>>(x, T, c.cin, c.k, c.d, c.p, t0, n, col);
            gemm(&a->g, G_BF16, n, c.cout, K, col, K, c.w, K, out + (size_t)t0 * c.cout, c.cout);
        }
    }
    k_bias<<<nblk((size_t)T * c.cout, 256), 256, 0, a->st>>>(out, (size_t)T, c.cout, c.b, resid);
}

void aact1(Aud* a, const AAct& f, const bf16* x, int T, int C, bf16* u, bf16* out) {
    k_upsnake<<<nblk((size_t)2 * T * C, 256), 256, 0, a->st>>>(x, T, C, f.fu, f.a, f.ib, u);
    k_down<<<nblk((size_t)T * C, 256), 256, 0, a->st>>>(u, T, C, f.fd, out);
}

// AMPBlock1: x [T][C] -> out (b[3..5] scratch)
void amp(Aud* a, const AmpBlk& r, const bf16* x, int T, int C, bf16* out, bf16* const* b) {
    bf16 *t1 = b[3], *t2 = b[4], *u = b[5];
    CK(cudaMemcpyAsync(out, x, (size_t)T * C * 2, cudaMemcpyDeviceToDevice, a->st));
    for (int l = 0; l < 3; l++) {
        aact1(a, r.a1[l], out, T, C, u, t1);
        aconv1(a, r.c1[l], t1, T, t2, nullptr);
        aact1(a, r.a2[l], t2, T, C, u, t1);
        aconv1(a, r.c2[l], t1, T, t2, out);  // t2 = bf16(bf16(conv + b) + out)
        CK(cudaMemcpyAsync(out, t2, (size_t)T * C * 2, cudaMemcpyDeviceToDevice, a->st));
    }
}

// generator: x [T][128] -> planar out [2][ldo] (T * prod(rates) samples), clamped to [-1, 1] when clamp
int voc_run(Aud* a, const Voc& v, const bf16* x, int T, bf16* out, int ldo, bool clamp) {
    size_t bneed = (size_t)T * v.c0, yneed = 1;
    for (size_t i = 0, t = T; i < v.ups.size(); i++) {
        yneed = std::max(yneed, t * v.ups[i].cout * v.ups[i].k);
        t *= v.ups[i].u;
        bneed = std::max(bneed, 2 * t * v.ups[i].cout);
    }
    bf16* b[6];
    for (int i = 0; i < 6; i++) b[i] = (bf16*)grow(a, a->vb[i], bneed * 2);
    float* Y = (float*)grow(a, a->Y, yneed * 4);
    const cudaStream_t st = a->st;
    bf16 *xa = b[0], *xs = b[1], *y = b[2];
    aconv1(a, v.pre, x, T, xa, nullptr);
    int t = T, C = v.c0;
    for (size_t i = 0; i < v.ups.size(); i++) {
        const AConv& u = v.ups[i];
        const int to = t * u.u;
        gemm(&a->g, G_F32, t, u.cout * u.k, u.cin, xa, u.cin, u.w, u.cin, Y, u.cout * u.k);
        k_col2im<<<nblk((size_t)to * u.cout, 256), 256, 0, st>>>(Y, t, u.cout, u.k, u.u, u.p, to, u.b, y);
        t = to, C = u.cout;
        const size_t n = (size_t)t * C;
        amp(a, v.rb[i * 3], y, t, C, xs, b);
        amp(a, v.rb[i * 3 + 1], y, t, C, xa, b);
        k_ew<<<nblk(n, 256), 256, 0, st>>>(0, xs, xa, xs, n);
        amp(a, v.rb[i * 3 + 2], y, t, C, xa, b);
        k_ew<<<nblk(n, 256), 256, 0, st>>>(0, xs, xa, xs, n);
        k_ew<<<nblk(n, 256), 256, 0, st>>>(1, xs, nullptr, xa, n);
    }
    aact1(a, v.post_act, xa, t, C, b[5], y);
    k_post<<<nblk((size_t)2 * t, 256), 256, 0, st>>>(y, t, C, v.post, clamp ? 1 : 0, out, ldo);
    return t;
}

// BWE: wav [2][Lp] (Lp = L rounded up to the hop 80, zero tail) -> out [2][3L] f32 at 48 kHz
void bwe_run(Aud* a, const bf16* wav, int L, int Lp, float* out) {
    const cudaStream_t st = a->st;
    const int F = Lp / 80, R = 2 * F, L3 = 3 * Lp;
    bf16* col = (bf16*)grow(a, a->fcol, (size_t)R * 512 * 2);
    bf16* spec = (bf16*)grow(a, a->spec, (size_t)R * 520 * 2);
    bf16* mag = (bf16*)grow(a, a->mag, (size_t)R * 264 * 2);
    bf16* melv = (bf16*)grow(a, a->melv, (size_t)R * 64 * 2);
    bf16* vin = (bf16*)grow(a, a->vin, (size_t)F * 128 * 2);
    bf16* res = (bf16*)grow(a, a->res, (size_t)2 * L3 * 2);
    bf16* skip = (bf16*)grow(a, a->skip, (size_t)2 * L3 * 2);
    k_frames<<<nblk((size_t)R * 64, 256), 256, 0, st>>>(wav, Lp, F, col);
    gemm(&a->g, G_BF16, R, 520, 512, col, 512, a->basis, 512, spec, 520);
    k_mag<<<nblk((size_t)R * 264, 256), 256, 0, st>>>(spec, R, mag);
    gemm(&a->g, G_BF16, R, 64, 264, mag, 264, a->melb, 264, melv, 64);
    k_logmel<<<nblk((size_t)R * 64, 256), 256, 0, st>>>(melv, F, vin, a->tap[3]);
    voc_run(a, a->bwe, vin, F, res, L3, false);
    k_hann<<<nblk((size_t)2 * L3, 256), 256, 0, st>>>(wav, Lp, a->hann, skip);
    if (a->tap[4]) CK(cudaMemcpyAsync(a->tap[4], res, (size_t)2 * L3 * 2, cudaMemcpyDeviceToDevice, st));
    if (a->tap[5]) CK(cudaMemcpyAsync(a->tap[5], skip, (size_t)2 * L3 * 2, cudaMemcpyDeviceToDevice, st));
    k_final<<<nblk((size_t)6 * L, 256), 256, 0, st>>>(res, skip, L3, 3 * L, out);
}

int finish(Aud* a, int r) {
    CK(cudaStreamSynchronize(a->st));
    CK(cudaGetLastError());  // launches here are unchecked: surface their errors in this call
    return r;
}
}  // namespace

extern "C" {

void* ltx_audio_new(const char* path) {
    Aud* a = new Aud();
    if (!a->f.open(path)) DIE("cannot open %s", path);
    CK(cudaStreamCreate(&a->st));  // blocking on purpose (common.cuh)
    a->g.st = a->st;
    CKL(cublasLtCreate(&a->g.h));
    a->g.ws = dmalloc(a->g.wsz);
    const std::string D = "audio_vae.decoder.";
    a->mean = up(wget(a, "audio_vae.per_channel_statistics.mean-of-means", 128));
    a->std = up(wget(a, "audio_vae.per_channel_statistics.std-of-means", 128));
    a->cin = load_conv2(a, D + "conv_in", 8, 512, 3);
    a->mid[0] = load_res2(a, D + "mid.block_1", 512, 512);
    a->mid[1] = load_res2(a, D + "mid.block_2", 512, 512);
    const int ch[3] = {128, 256, 512};
    int bi = 512;
    for (int lvl = 2; lvl >= 0; lvl--) {
        for (int b = 0; b < 3; b++) {
            a->blk[lvl][b] = load_res2(a, D + "up." + std::to_string(lvl) + ".block." + std::to_string(b), bi, ch[lvl]);
            bi = ch[lvl];
        }
        if (lvl) a->ups[lvl] = load_conv2(a, D + "up." + std::to_string(lvl) + ".upsample.conv", bi, bi, 3);
    }
    a->cout = load_conv2(a, D + "conv_out", 128, 2, 3);
    a->voc = load_voc(a, "vocoder.vocoder.", 1536, {5, 2, 2, 2, 2, 2}, {11, 4, 4, 4, 4, 4});
    a->bwe = load_voc(a, "vocoder.bwe_generator.", 512, {6, 5, 2, 2, 2}, {12, 11, 4, 4, 4});
    std::vector<float> fb = wget(a, "vocoder.mel_stft.stft_fn.forward_basis", (size_t)514 * 512);
    fb.resize((size_t)520 * 512, 0.f);
    a->basis = up(tobf(fb));
    const std::vector<float> mb = wget(a, "vocoder.mel_stft.mel_basis", (size_t)64 * 257);
    std::vector<float> mp((size_t)64 * 264, 0.f);
    for (int m = 0; m < 64; m++) std::copy(mb.begin() + m * 257, mb.begin() + (m + 1) * 257, mp.begin() + m * 264);
    a->melb = up(tobf(mp));
    a->hann = up(hann_filter());
    a->f.close();
    CK(cudaDeviceSynchronize());
    return a;
}

// Optional copies of intermediates in ComfyUI layouts (bf16 device, nullptr = off), filled by later calls:
// 0 denorm [8][T][16], 1 mel [2][4T-3][64], 2 voc16k [2][L], 3 bwe_mel [2][64][L/80], 4 bwe_residual [2][3L],
// 5 bwe_skip [2][3L] (L = 16 kHz samples rounded up to 80; ltx_audio_decode: L = 160 (4T-3))
void ltx_audio_taps(void* h, void* const* taps) {
    Aud* a = (Aud*)h;
    for (int i = 0; i < 6; i++) a->tap[i] = taps ? (bf16*)taps[i] : nullptr;
}

// latent [8, T, 16] f32 device, normalized (ComfyUI audio latent [1, 8, T, 16]) -> out [2, 480 (4T-3)] f32 device
// at 48 kHz. Returns the sample count per channel.
int ltx_audio_decode(void* h, const float* lat, int T, float* out) {
    Aud* a = (Aud*)h;
    if (T < 1) DIE("ltx_audio_decode: T = %d", T);
    const int M = 4 * T - 3, L = 160 * M;
    bf16* vin = (bf16*)grow(a, a->vin, (size_t)M * 128 * 2);
    dec_run(a, lat, T, vin);
    bf16* wav = (bf16*)grow(a, a->wav, (size_t)2 * L * 2);
    voc_run(a, a->voc, vin, M, wav, L, true);
    if (a->tap[2]) CK(cudaMemcpyAsync(a->tap[2], wav, (size_t)2 * L * 2, cudaMemcpyDeviceToDevice, a->st));
    bwe_run(a, wav, L, L, out);
    return finish(a, 3 * L);
}

// vocoder only: mel [2, M, 64] bf16 device (ComfyUI mel layout) -> wav [2, 160 M] bf16 device at 16 kHz
int ltx_audio_vocode(void* h, const void* mel, int M, void* wav) {
    Aud* a = (Aud*)h;
    bf16* vin = (bf16*)grow(a, a->vin, (size_t)M * 128 * 2);
    k_melin<<<nblk((size_t)M * 128, 256), 256, 0, a->st>>>((const bf16*)mel, M, vin);
    return finish(a, voc_run(a, a->voc, vin, M, (bf16*)wav, 160 * M, true));
}

// BWE only: wav [2, L] bf16 device at 16 kHz -> out [2, 3L] f32 device at 48 kHz
int ltx_audio_bwe(void* h, const void* wav, int L, float* out) {
    Aud* a = (Aud*)h;
    const int Lp = (L + 79) / 80 * 80;
    bf16* w = (bf16*)grow(a, a->wav, (size_t)2 * Lp * 2);
    CK(cudaMemsetAsync(w, 0, (size_t)2 * Lp * 2, a->st));
    CK(cudaMemcpy2DAsync(w, (size_t)Lp * 2, wav, (size_t)L * 2, (size_t)L * 2, 2, cudaMemcpyDeviceToDevice, a->st));
    bwe_run(a, w, L, Lp, out);
    return finish(a, 3 * L);
}

void ltx_audio_free(void* h) {
    Aud* a = (Aud*)h;
    cudaStreamSynchronize(a->st);
    auto f2 = [](Conv2& c) { cudaFree(c.w), cudaFree(c.b); };
    auto fr = [&](Res2& r) { f2(r.c1), f2(r.c2), f2(r.sc); };
    auto fc = [](AConv& c) { cudaFree(c.w), cudaFree(c.b); };
    auto fa = [](AAct& f) { cudaFree(f.a), cudaFree(f.ib), cudaFree(f.fu), cudaFree(f.fd); };
    auto fv = [&](Voc& v) {
        fc(v.pre);
        for (AConv& c : v.ups) fc(c);
        for (AmpBlk& r : v.rb)
            for (int l = 0; l < 3; l++) fc(r.c1[l]), fc(r.c2[l]), fa(r.a1[l]), fa(r.a2[l]);
        fa(v.post_act);
        cudaFree(v.post);
    };
    f2(a->cin), f2(a->ups[1]), f2(a->ups[2]), f2(a->cout);
    fr(a->mid[0]), fr(a->mid[1]);
    for (auto& l : a->blk)
        for (Res2& r : l) fr(r);
    fv(a->voc), fv(a->bwe);
    Buf* bufs[] = {&a->col, &a->dec[0], &a->dec[1], &a->dec[2], &a->dec[3], &a->vin, &a->vb[0], &a->vb[1], &a->vb[2],
                   &a->vb[3], &a->vb[4], &a->vb[5], &a->Y, &a->wav, &a->fcol, &a->spec, &a->mag, &a->melv, &a->res, &a->skip};
    for (Buf* b : bufs) cudaFree(b->p);
    void* ps[] = {a->mean, a->std, a->basis, a->melb, a->hann, a->g.ws};
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
