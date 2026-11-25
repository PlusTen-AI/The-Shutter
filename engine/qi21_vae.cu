// shutter engine — Qwen-Image 2.1 VAE (Wan 2.2 layout, temporal kernel 1, no patchify, RGBA, 16x), C API.
// Single image only: follows ComfyUI comfy/ldm/wan/vae2_2.py WanVAE.encode/decode with a 4D input (no temporal cache,
// so time_conv is never used and every conv is 2D). Activations are NHWC bf16; convs are im2col + gemm_bf16 in row
// strips; bf16 rounding follows torch (conv out, + bias, + shortcut each rounded; RMS norm rounds per op).
#include "common.cuh"
#include "st.h"

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

namespace {
inline int rup(int a, int b) { return (a + b - 1) / b * b; }
inline int nblk(size_t n, int t) { return (int)((n + t - 1) / t); }
__device__ __forceinline__ float rd(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }

struct Conv {  // weight [np][kp] = [cout][ky][kx][cin] zero-padded, bias [np]
    bf16 *w = nullptr, *b = nullptr;
    int cin = 0, cout = 0, k = 1, kp = 0, np = 0;
};
struct Res {
    bf16 *g1, *g2;
    Conv c1, c2, sc;
};
struct Attn {
    bf16* g;
    Conv qkv, proj;
};
struct Stage {
    std::vector<Res> r;
    Conv rs;
    int cin, cout, ft;
    bool resample;
};

struct Vae {
    cudaStream_t st;
    Lt* lt;
    cublasLtHandle_t lh;
    void* lws;
    size_t lwsz = 32u << 20;
    StFile f;
    Conv conv1, conv2, dconv1, dhead, econv1, ehead;
    bf16 *dhead_g, *ehead_g;
    Res dmid[2], emid[2];
    Attn dattn, eattn;
    std::vector<Stage> up, down;
    size_t cap = 0, colcap = 128u << 20, scap = 0;
    bf16 *x = nullptr, *t1 = nullptr, *t2 = nullptr, *keep = nullptr, *col = nullptr, *P = nullptr;
    float* S = nullptr;
};

void* dmalloc(size_t n) {
    void* p;
    CK(cudaMalloc(&p, n));
    return p;
}

const uint16_t* host(Vae* v, const std::string& name, size_t elems) {
    const StTensor& t = v->f.get(name);
    if (t.dtype != "BF16" || t.bytes() != elems * 2)
        DIE("%s: %s %llu bytes, expected BF16 x %zu", name.c_str(), t.dtype.c_str(), (unsigned long long)t.bytes(), elems);
    return (const uint16_t*)v->f.ptr(t);
}

bf16* up_vec(Vae* v, const std::string& name, int n, int npad) {
    std::vector<uint16_t> h(npad, 0);
    const uint16_t* s = host(v, name, n);
    std::copy(s, s + n, h.begin());
    bf16* d = (bf16*)dmalloc((size_t)npad * 2);
    CK(cudaMemcpy(d, h.data(), (size_t)npad * 2, cudaMemcpyHostToDevice));
    return d;
}

Conv load_conv(Vae* v, const std::string& p, int cin, int cout, int k) {
    Conv c;
    c.cin = cin, c.cout = cout, c.k = k, c.kp = rup(k * k * cin, 8), c.np = rup(cout, 8);
    const uint16_t* w = host(v, p + ".weight", (size_t)cout * cin * k * k);
    std::vector<uint16_t> r((size_t)c.np * c.kp, 0);
    for (int o = 0; o < cout; o++)
        for (int i = 0; i < cin; i++)
            for (int t = 0; t < k * k; t++) r[(size_t)o * c.kp + t * cin + i] = w[((size_t)o * cin + i) * k * k + t];
    c.w = (bf16*)dmalloc(r.size() * 2);
    CK(cudaMemcpy(c.w, r.data(), r.size() * 2, cudaMemcpyHostToDevice));
    c.b = up_vec(v, p + ".bias", cout, c.np);
    return c;
}

Res load_res(Vae* v, const std::string& p, int cin, int cout) {
    Res r;
    r.g1 = up_vec(v, p + ".residual.0.gamma", cin, cin);
    r.c1 = load_conv(v, p + ".residual.2", cin, cout, 3);
    r.g2 = up_vec(v, p + ".residual.3.gamma", cout, cout);
    r.c2 = load_conv(v, p + ".residual.6", cout, cout, 3);
    if (cin != cout) r.sc = load_conv(v, p + ".shortcut", cin, cout, 1);
    return r;
}

Attn load_attn(Vae* v, const std::string& p, int c) {
    return Attn{up_vec(v, p + ".norm.gamma", c, c), load_conv(v, p + ".to_qkv", c, 3 * c, 1), load_conv(v, p + ".proj", c, c, 1)};
}

// ---------------- kernels ----------------
// im2col for 3x3 convs. mode 0: stride 1 pad 1; 1: ZeroPad2d(0,1,0,1) + stride 2; 2: nearest 2x upsample + pad 1.
// E = 8: C % 8 == 0 (vector copies, kp == 9C); E = 1: any C, columns >= 9C zero.
template <int E>
__global__ void k_im2col(const bf16* __restrict__ x, int H, int W, int C, int mode, int Wo, int r0, int rows, int kp,
                         bf16* __restrict__ col) {
    const int CE = C / E, G = E == 8 ? 9 * CE : kp;
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * Wo * G) return;
    const int g = (int)(i % G);
    const size_t p = i / G;
    const int ox = (int)(p % Wo), oy = r0 + (int)(p / Wo);
    bf16* dst = col + p * kp + (size_t)g * E;
    if (E == 1 && g >= 9 * C) { *dst = __float2bfloat16(0.f); return; }
    const int tap = g / CE, c = (g % CE) * E, ky = tap / 3, kx = tap % 3;
    int iy, ix;
    bool ok;
    if (mode == 0) {
        iy = oy + ky - 1, ix = ox + kx - 1;
        ok = iy >= 0 && iy < H && ix >= 0 && ix < W;
    } else if (mode == 1) {
        iy = 2 * oy + ky, ix = 2 * ox + kx;
        ok = iy < H && ix < W;
    } else {
        const int uy = oy + ky - 1, ux = ox + kx - 1;
        ok = uy >= 0 && uy < 2 * H && ux >= 0 && ux < 2 * W;
        iy = uy >> 1, ix = ux >> 1;
    }
    const bf16* src = x + ((size_t)iy * W + ix) * C + c;
    if (E == 8) *(uint4*)dst = ok ? *(const uint4*)src : make_uint4(0, 0, 0, 0);
    else *dst = ok ? *src : __float2bfloat16(0.f);
}

// Wan RMS_norm (F.normalize * sqrt(C) * gamma), each torch op rounded to bf16, optional SiLU; one warp per row
__global__ void k_norm(const bf16* x, bf16* y, const bf16* __restrict__ g, int M, int C, float scale, int silu) {
    const int row = blockIdx.x * 8 + threadIdx.x / 32, lane = threadIdx.x & 31;
    if (row >= M) return;
    const bf16* xr = x + (size_t)row * C;
    bf16* yr = y + (size_t)row * C;
    float ss = 0.f;
    for (int c = lane; c < C; c += 32) {
        const float a = ld(xr + c);
        ss += a * a;
    }
    for (int o = 16; o; o >>= 1) ss += __shfl_xor_sync(0xffffffffu, ss, o);
    const float n = fmaxf(rd(sqrtf(ss)), 1e-12f);
    for (int c = lane; c < C; c += 32) {
        float a = rd(ld(xr + c) / n);
        a = rd(a * scale);
        a = rd(a * ld(g + c));
        if (silu) a = rd(a / (1.f + expf(-a)));
        yr[c] = __float2bfloat16(a);
    }
}

// out[p][o] = bf16(bf16(y[p][o] + b[o]) + s). s: SC_NONE 0, SC_RES r[p][o], SC_DUP DupUp3D(r) (first chunk),
// SC_AVG AvgDown3D(r) (zero frame padded in front when ft = 2), rounded to bf16 first. r has Cin channels for DUP/AVG.
enum { SC_NONE = 0, SC_RES = 1, SC_DUP = 2, SC_AVG = 3 };
__global__ void k_bias_sc(int mode, const bf16* y, int ldy, const bf16* __restrict__ b, const bf16* r, bf16* out,
                          int Ho, int Wo, int N, int Cin, int ft) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)Ho * Wo * N) return;
    const int o = (int)(i % N);
    const size_t p = i / N;
    float v = ld(y + p * ldy + o);
    if (b) v = rd(v + ld(b + o));
    if (mode == SC_RES) {
        v += ld(r + i);
    } else if (mode == SC_DUP) {
        const int oy = (int)(p / Wo), ox = (int)(p % Wo), factor = ft * 4, rep = N * factor / Cin;
        const int ci = (o * factor + (ft - 1) * 4 + (oy & 1) * 2 + (ox & 1)) / rep;
        v += ld(r + ((size_t)(oy >> 1) * (Wo >> 1) + (ox >> 1)) * Cin + ci);
    } else if (mode == SC_AVG) {
        const int oy = (int)(p / Wo), ox = (int)(p % Wo), factor = ft * 4, gs = Cin * factor / N, Wi = 2 * Wo;
        float s = 0.f;
        for (int gi = 0; gi < gs; gi++) {
            const int n = o * gs + gi, c = n / factor, k = n % factor;
            if (ft == 2 && k < 4) continue;  // padded zero frame
            s += ld(r + ((size_t)(2 * oy + ((k >> 1) & 1)) * Wi + 2 * ox + (k & 1)) * Cin + c);
        }
        v += rd(s / (float)gs);
    }
    out[i] = __float2bfloat16(v);
}

__global__ void k_bias_rows(bf16* y, const bf16* __restrict__ b, int R, int L) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < (size_t)R * L) y[i] = __float2bfloat16(ld(y + i) + ld(b + i / L));
}

__device__ float block_reduce(float v, bool mx) {
    __shared__ float sh[32];
    for (int o = 16; o; o >>= 1) {
        const float u = __shfl_xor_sync(0xffffffffu, v, o);
        v = mx ? fmaxf(v, u) : v + u;
    }
    const int w = threadIdx.x / 32, l = threadIdx.x & 31, nw = blockDim.x / 32;
    __syncthreads();
    if (l == 0) sh[w] = v;
    __syncthreads();
    v = l < nw ? sh[l] : (mx ? -INFINITY : 0.f);
    for (int o = 16; o; o >>= 1) {
        const float u = __shfl_xor_sync(0xffffffffu, v, o);
        v = mx ? fmaxf(v, u) : v + u;
    }
    return v;
}

// P = bf16(softmax(S * scale)) per row, fp32 math
__global__ void k_softmax(const float* __restrict__ S, bf16* __restrict__ P, int L, float scale) {
    const float* s = S + (size_t)blockIdx.x * L;
    bf16* p = P + (size_t)blockIdx.x * L;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < L; j += blockDim.x) m = fmaxf(m, s[j] * scale);
    m = block_reduce(m, true);
    float l = 0.f;
    for (int j = threadIdx.x; j < L; j += blockDim.x) l += expf(s[j] * scale - m);
    l = block_reduce(l, false);
    const float il = 1.f / l;
    for (int j = threadIdx.x; j < L; j += blockDim.x) p[j] = __float2bfloat16(expf(s[j] * scale - m) * il);
}

// latent [C, HW] (f32 or bf16) -> bf16 [HW, C]
__global__ void k_lat_in(const void* lat, int f32, int C, int HW, bf16* x) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)C * HW) return;
    const size_t s = (i % C) * HW + i / C;
    x[i] = f32 ? __float2bfloat16(((const float*)lat)[s]) : ((const bf16*)lat)[s];
}

// bf16 [HW, C] -> float [C, HW]
__global__ void k_lat_out(const bf16* y, int C, int HW, float* out) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i < (size_t)C * HW) out[i] = ld(y + (i % HW) * C + i / HW);
}

// image float [Hs, Ws, Cs] 0..1, crop at (y0, x0) to H x W, first 4 channels (alpha padded with 1), x*2-1 -> bf16 [HW, 4]
__global__ void k_img_in(const float* img, int Ws, int Cs, int y0, int x0, int H, int W, bf16* x) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= (size_t)H * W * 4) return;
    const int c = (int)(i & 3);
    const size_t p = i >> 2;
    const float v = c < Cs ? img[((size_t)(y0 + p / W) * Ws + x0 + p % W) * Cs + c] : 1.f;
    x[i] = __float2bfloat16(v * 2.f - 1.f);
}

// decoder head output (+ bias) -> float [HW, 4], (v + 1) / 2 clamped to 0..1
__global__ void k_img_out(const bf16* y, int ldy, const bf16* __restrict__ b, size_t M, float* out) {
    const size_t i = blockIdx.x * (size_t)blockDim.x + threadIdx.x;
    if (i >= M * 4) return;
    const int c = (int)(i & 3);
    const float v = rd(ld(y + (i >> 2) * ldy + c) + ld(b + c));
    out[i] = fminf(fmaxf((v + 1.f) * 0.5f, 0.f), 1.f);
}

// ---------------- layers ----------------
void ensure(Vae* v, size_t elems) {
    if (elems <= v->cap) return;
    cudaFree(v->x), cudaFree(v->t1), cudaFree(v->t2), cudaFree(v->keep);
    v->cap = elems;
    v->x = (bf16*)dmalloc(elems * 2), v->t1 = (bf16*)dmalloc(elems * 2);
    v->t2 = (bf16*)dmalloc(elems * 2), v->keep = (bf16*)dmalloc(elems * 2);
}

void bias_sc(Vae* v, int mode, const bf16* y, int ldy, const bf16* b, const bf16* r, bf16* out, int Ho, int Wo, int N,
             int Cin = 0, int ft = 1) {
    const size_t n = (size_t)Ho * Wo * N;
    k_bias_sc<<<nblk(n, 256), 256, 0, v->st>>>(mode, y, ldy, b, r, out, Ho, Wo, N, Cin, ft);
}

void norm(Vae* v, const bf16* x, bf16* y, const bf16* g, int M, int C, bool silu) {
    k_norm<<<nblk(M, 8), 256, 0, v->st>>>(x, y, g, M, C, (float)std::sqrt((double)C), silu);
}

// y [Ho*Wo, np] = conv(x [H*W, cin]) without bias; mode as k_im2col (ignored for 1x1)
void conv(Vae* v, const Conv& c, const bf16* x, int H, int W, int mode, bf16* y) {
    if (c.k == 1) { gemm_bf16(v->lt, H * W, c.np, c.kp, x, c.w, y); return; }
    const int Ho = mode == 1 ? H / 2 : mode == 2 ? 2 * H : H, Wo = mode == 1 ? W / 2 : mode == 2 ? 2 * W : W;
    const int rows = std::max(1, std::min(Ho, (int)(v->colcap / ((size_t)Wo * c.kp))));
    for (int r0 = 0; r0 < Ho; r0 += rows) {
        const int n = std::min(rows, Ho - r0);
        if (c.cin % 8 == 0) {
            const size_t t = (size_t)n * Wo * 9 * (c.cin / 8);
            k_im2col<8><<<nblk(t, 256), 256, 0, v->st>>>(x, H, W, c.cin, mode, Wo, r0, n, c.kp, v->col);
        } else {
            const size_t t = (size_t)n * Wo * c.kp;
            k_im2col<1><<<nblk(t, 256), 256, 0, v->st>>>(x, H, W, c.cin, mode, Wo, r0, n, c.kp, v->col);
        }
        gemm_bf16(v->lt, n * Wo, c.np, c.kp, v->col, c.w, y + (size_t)r0 * Wo * c.np);
    }
}

// ResidualBlock on v->x [H*W, cin] -> v->x [H*W, cout]
void res(Vae* v, const Res& r, int H, int W) {
    const int M = H * W, co = r.c1.cout;
    norm(v, v->x, v->t1, r.g1, M, r.c1.cin, true);
    conv(v, r.c1, v->t1, H, W, 0, v->t2);
    bias_sc(v, SC_NONE, v->t2, co, r.c1.b, nullptr, v->t2, H, W, co);
    norm(v, v->t2, v->t2, r.g2, M, co, true);
    conv(v, r.c2, v->t2, H, W, 0, v->t1);
    if (r.sc.w) {
        conv(v, r.sc, v->x, H, W, 0, v->t2);
        bias_sc(v, SC_NONE, v->t2, co, r.sc.b, nullptr, v->t2, H, W, co);
        bias_sc(v, SC_RES, v->t1, co, r.c2.b, v->t2, v->x, H, W, co);
    } else {
        bias_sc(v, SC_RES, v->t1, co, r.c2.b, v->x, v->x, H, W, co);
    }
}

// row-major C[M,N] (fp32) = A[M,K] . W[N,K]^T, bf16 inputs, fp32 accumulate
void gemm_f32(Vae* v, int M, int N, int K, const bf16* A, const bf16* W, float* C) {
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t lw, la, lc;
    CKL(cublasLtMatmulDescCreate(&op, CUBLAS_COMPUTE_32F, CUDA_R_32F));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    CKL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CKL(cublasLtMatmulDescSetAttribute(op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    CKL(cublasLtMatrixLayoutCreate(&lw, CUDA_R_16BF, K, N, K));
    CKL(cublasLtMatrixLayoutCreate(&la, CUDA_R_16BF, K, M, K));
    CKL(cublasLtMatrixLayoutCreate(&lc, CUDA_R_32F, N, M, N));
    cublasLtMatmulPreference_t pref;
    CKL(cublasLtMatmulPreferenceCreate(&pref));
    CKL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &v->lwsz, sizeof(v->lwsz)));
    cublasLtMatmulHeuristicResult_t res;
    int n = 0;
    CKL(cublasLtMatmulAlgoGetHeuristic(v->lh, op, lw, la, lc, lc, pref, 1, &res, &n));
    if (n == 0) DIE("no cublasLt f32 algo for M=%d N=%d K=%d", M, N, K);
    const float one = 1.f, zero = 0.f;
    CKL(cublasLtMatmul(v->lh, op, &one, W, lw, A, la, &zero, C, lc, C, lc, &res.algo, v->lws, v->lwsz, v->st));
    cublasLtMatmulPreferenceDestroy(pref);
    cublasLtMatrixLayoutDestroy(lw), cublasLtMatrixLayoutDestroy(la), cublasLtMatrixLayoutDestroy(lc);
    cublasLtMatmulDescDestroy(op);
}

// single-head AttentionBlock on v->x [L, C] (sdpa, scale 1/sqrt(C)), in place
void attn(Vae* v, const Attn& a, int H, int W) {
    const int L = H * W, C = a.proj.cin;
    if ((size_t)L * L > v->scap) {
        cudaFree(v->S), cudaFree(v->P);
        v->scap = (size_t)L * L;
        v->S = (float*)dmalloc(v->scap * 4), v->P = (bf16*)dmalloc(v->scap * 2);
    }
    bf16 *q = v->t2, *k = q + (size_t)L * C, *vt = k + (size_t)L * C;
    const bf16 *w = a.qkv.w, *b = a.qkv.b;
    norm(v, v->x, v->t1, a.g, L, C, false);
    gemm_bf16(v->lt, L, C, C, v->t1, w, q);
    bias_sc(v, SC_NONE, q, C, b, nullptr, q, L, 1, C);
    gemm_bf16(v->lt, L, C, C, v->t1, w + (size_t)C * C, k);
    bias_sc(v, SC_NONE, k, C, b + C, nullptr, k, L, 1, C);
    gemm_bf16(v->lt, C, L, C, w + (size_t)2 * C * C, v->t1, vt);  // v^T [C, L]
    k_bias_rows<<<nblk((size_t)C * L, 256), 256, 0, v->st>>>(vt, b + 2 * C, C, L);
    gemm_f32(v, L, L, C, q, k, v->S);
    k_softmax<<<L, 512, 0, v->st>>>(v->S, v->P, L, (float)(1.0 / std::sqrt((double)C)));
    gemm_bf16(v->lt, L, C, L, v->P, vt, v->t1);
    gemm_bf16(v->lt, L, C, C, v->t1, a.proj.w, q);
    bias_sc(v, SC_RES, q, C, a.proj.b, v->x, v->x, L, 1, C);
}
}  // namespace

extern "C" {

void* qi21_vae_new(const char* path) {
    Vae* v = new Vae();
    if (!v->f.open(path)) DIE("cannot open %s", path);
    CK(cudaStreamCreate(&v->st));   // blocking on purpose (common.cuh)
    v->lt = lt_new(v->st);
    CKL(cublasLtCreate(&v->lh));
    v->lws = dmalloc(v->lwsz);
    v->col = (bf16*)dmalloc(v->colcap * 2);
    // decoder: dims 144 * [8, 8, 8, 4, 2, 1], 3 res blocks per stage, upsample on the first four
    v->conv2 = load_conv(v, "conv2", 64, 64, 1);
    v->dconv1 = load_conv(v, "decoder.conv1", 64, 1152, 3);
    v->dmid[0] = load_res(v, "decoder.middle.0", 1152, 1152);
    v->dattn = load_attn(v, "decoder.middle.1", 1152);
    v->dmid[1] = load_res(v, "decoder.middle.2", 1152, 1152);
    const int dd[6] = {1152, 1152, 1152, 576, 288, 144};
    for (int i = 0; i < 5; i++) {
        Stage s{{}, Conv(), dd[i], dd[i + 1], i < 3 ? 2 : 1, i < 4};
        const std::string p = "decoder.upsamples." + std::to_string(i) + ".upsamples.";
        for (int j = 0; j < 3; j++) s.r.push_back(load_res(v, p + std::to_string(j), j ? s.cout : s.cin, s.cout));
        if (s.resample) s.rs = load_conv(v, p + "3.resample.1", s.cout, s.cout, 3);
        v->up.push_back(s);
    }
    v->dhead_g = up_vec(v, "decoder.head.0.gamma", 144, 144);
    v->dhead = load_conv(v, "decoder.head.2", 144, 4, 3);
    // encoder: dims 96 * [1, 1, 2, 4, 8, 8], 2 res blocks per stage, downsample on the first four
    v->econv1 = load_conv(v, "encoder.conv1", 4, 96, 3);
    const int ed[6] = {96, 96, 192, 384, 768, 768};
    for (int i = 0; i < 5; i++) {
        Stage s{{}, Conv(), ed[i], ed[i + 1], (i >= 1 && i <= 3) ? 2 : 1, i < 4};
        const std::string p = "encoder.downsamples." + std::to_string(i) + ".downsamples.";
        for (int j = 0; j < 2; j++) s.r.push_back(load_res(v, p + std::to_string(j), j ? s.cout : s.cin, s.cout));
        if (s.resample) s.rs = load_conv(v, p + "2.resample.1", s.cout, s.cout, 3);
        v->down.push_back(s);
    }
    v->emid[0] = load_res(v, "encoder.middle.0", 768, 768);
    v->eattn = load_attn(v, "encoder.middle.1", 768);
    v->emid[1] = load_res(v, "encoder.middle.2", 768, 768);
    v->ehead_g = up_vec(v, "encoder.head.0.gamma", 768, 768);
    v->ehead = load_conv(v, "encoder.head.2", 768, 128, 3);
    v->conv1 = load_conv(v, "conv1", 128, 128, 1);
    v->f.close();
    CK(cudaDeviceSynchronize());  // uploads complete before the non-blocking stream uses them
    return v;
}

// latent [64, lh, lw] device (f32 != 0: float32, else bf16), raw VAE space (what VAEDecode receives)
// -> out [16 lh, 16 lw, 4] float device, RGBA in 0..1 (VAEDecode's IMAGE)
int qi21_vae_decode(void* h, const void* latent, int f32, int lh, int lw, float* out) {
    Vae* v = (Vae*)h;
    int H = lh, W = lw;
    const size_t L = (size_t)H * W;
    ensure(v, std::max<size_t>(288 * 256 * L, 3 * 1152 * L));
    k_lat_in<<<nblk(64 * L, 256), 256, 0, v->st>>>(latent, f32, 64, (int)L, v->t1);
    conv(v, v->conv2, v->t1, H, W, 0, v->x);
    bias_sc(v, SC_NONE, v->x, 64, v->conv2.b, nullptr, v->x, H, W, 64);
    conv(v, v->dconv1, v->x, H, W, 0, v->t1);
    bias_sc(v, SC_NONE, v->t1, 1152, v->dconv1.b, nullptr, v->t1, H, W, 1152);
    std::swap(v->x, v->t1);
    res(v, v->dmid[0], H, W);
    attn(v, v->dattn, H, W);
    res(v, v->dmid[1], H, W);
    for (const Stage& s : v->up) {
        if (s.resample) CK(cudaMemcpyAsync(v->keep, v->x, (size_t)H * W * s.cin * 2, cudaMemcpyDeviceToDevice, v->st));
        for (const Res& r : s.r) res(v, r, H, W);
        if (!s.resample) continue;
        conv(v, s.rs, v->x, H, W, 2, v->t1);
        H *= 2, W *= 2;
        bias_sc(v, SC_DUP, v->t1, s.cout, s.rs.b, v->keep, v->x, H, W, s.cout, s.cin, s.ft);
    }
    norm(v, v->x, v->t1, v->dhead_g, H * W, 144, true);
    conv(v, v->dhead, v->t1, H, W, 0, v->t2);
    k_img_out<<<nblk((size_t)H * W * 4, 256), 256, 0, v->st>>>(v->t2, v->dhead.np, v->dhead.b, (size_t)H * W, out);
    CK(cudaStreamSynchronize(v->st));
    return 0;
}

// image [H, W, C] float device in 0..1 (C = 3: alpha 1, C = 4: RGBA), center-cropped to multiples of 16 as ComfyUI
// -> out [64, H/16, W/16] float device: the latent mean (what vae.encode returns, raw VAE space)
int qi21_vae_encode(void* h, const float* img, int Hs, int Ws, int C, float* out) {
    Vae* v = (Vae*)h;
    int H = Hs / 16 * 16, W = Ws / 16 * 16;
    ensure(v, (size_t)96 * H * W);
    k_img_in<<<nblk((size_t)H * W * 4, 256), 256, 0, v->st>>>(img, Ws, C, (Hs % 16) / 2, (Ws % 16) / 2, H, W, v->x);
    conv(v, v->econv1, v->x, H, W, 0, v->t1);
    bias_sc(v, SC_NONE, v->t1, 96, v->econv1.b, nullptr, v->t1, H, W, 96);
    std::swap(v->x, v->t1);
    for (const Stage& s : v->down) {
        CK(cudaMemcpyAsync(v->keep, v->x, (size_t)H * W * s.cin * 2, cudaMemcpyDeviceToDevice, v->st));
        for (const Res& r : s.r) res(v, r, H, W);
        if (s.resample) {
            conv(v, s.rs, v->x, H, W, 1, v->t1);
            H /= 2, W /= 2;
            bias_sc(v, SC_AVG, v->t1, s.cout, s.rs.b, v->keep, v->x, H, W, s.cout, s.cin, s.ft);
        } else {
            bias_sc(v, SC_RES, v->x, s.cout, nullptr, v->keep, v->x, H, W, s.cout);
        }
    }
    res(v, v->emid[0], H, W);
    attn(v, v->eattn, H, W);
    res(v, v->emid[1], H, W);
    norm(v, v->x, v->t1, v->ehead_g, H * W, 768, true);
    conv(v, v->ehead, v->t1, H, W, 0, v->t2);
    bias_sc(v, SC_NONE, v->t2, 128, v->ehead.b, nullptr, v->t2, H, W, 128);
    gemm_bf16(v->lt, H * W, 64, 128, v->t2, v->conv1.w, v->t1);  // mu = first 64 channels of conv1
    bias_sc(v, SC_NONE, v->t1, 64, v->conv1.b, nullptr, v->t1, H, W, 64);
    k_lat_out<<<nblk((size_t)64 * H * W, 256), 256, 0, v->st>>>(v->t1, 64, H * W, out);
    CK(cudaStreamSynchronize(v->st));
    return 0;
}

void qi21_vae_free(void* h) {
    Vae* v = (Vae*)h;
    cudaStreamSynchronize(v->st);
    auto fc = [](Conv& c) { cudaFree(c.w), cudaFree(c.b); };
    auto fr = [&](Res& r) { cudaFree(r.g1), cudaFree(r.g2), fc(r.c1), fc(r.c2), fc(r.sc); };
    for (std::vector<Stage>* ss : {&v->up, &v->down})
        for (Stage& s : *ss) {
            for (Res& r : s.r) fr(r);
            fc(s.rs);
        }
    for (int i = 0; i < 2; i++) fr(v->dmid[i]), fr(v->emid[i]);
    for (Attn* a : {&v->dattn, &v->eattn}) cudaFree(a->g), fc(a->qkv), fc(a->proj);
    for (Conv* c : {&v->conv1, &v->conv2, &v->dconv1, &v->dhead, &v->econv1, &v->ehead}) fc(*c);
    void* ps[] = {v->dhead_g, v->ehead_g, v->x, v->t1, v->t2, v->keep, v->col, v->P, v->S, v->lws};
    for (void* p : ps) cudaFree(p);
    cublasLtDestroy(v->lh);
    lt_free(v->lt);
    cudaStreamDestroy(v->st);
    delete v;
}
}  // extern "C"
