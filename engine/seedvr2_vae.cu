// shutter engine — SeedVR2 video VAE (ComfyUI e9afbccc2 comfy/ldm/seedvr/vae.py VideoAutoencoderKLWrapper), C API.
// Reproduces VAEEncodeTiled / VAEDecodeTiled with the blueprint widgets [512, 128, 64, 8]. For this VAE sd.py hands
// tiling to the model (handles_tiling -> encode_tiled / decode_tiled -> tiled_vae):
//  - spatial tiles only: encode 512 px tiles, overlap 128 px; decode 64 latent tiles, overlap 16 (128 px). Each tile
//    runs over all frames, then tiles are blended in fp32 with cosine ramps (linspace(0, 1, ov), 0.5 - 0.5 cos(pi t))
//    and divided by the summed weights, then cast to bf16. The nodes' temporal_size / overlap are discarded: the VAE
//    slices time itself with causal caches, which is exact, so this engine slices time its own way (SLICE latents).
//  - encode: x = bf16(2p - 1), latent = mean (first 16 of 32 channels), out = fp32(bf16(blend)) * 0.9152.
//    decode: z = bf16(bf16(lat) * (1 / 0.9152)), pixels = clamp((fp32(bf16(blend)) + 1) / 2, 0, 1).
// Numerics follow ComfyUI with --bf16-vae (the eager path; its kitchen kernels are fp16 only):
//  - bf16 activations NDHWC; per-frame GroupNorm(32, 1e-6) rounded (y = a x + b, a = rstd w, b = bias - a mean),
//    SiLU rounded; conv = fp32-accumulated implicit GEMM rounded, + bias rounded, + residual rounded (cuDNN conv,
//    then separate adds). Causal 3x3x3 convs: zero spatial border, the conv input's first frame repeated twice in
//    front (later slices: the cached last frames of the padded input, kept in a ping-pong buffer the pad kernel
//    writes directly). Downsamplers: (0,1,0,1) zero pad, stride 2
//    (temporal stride 2 with a 2-frame head / 1-frame cache). Upsamplers: 1x1x1 conv (rounded, + bias rounded),
//    pixel shuffle (sr, sr, tr, c), first slice drops the first frame's second phase, then a causal 3x3x3 conv.
//  - mid-block attention per frame, single head 512: q/k/v/out = bf16 matmul then + bias rounded (F.linear on a
//    transposed input takes matmul + add_), softmax fp32 with bf16 probabilities (SDPA mem-efficient), + residual.
// Weights F16 -> bf16 (RNE) as ComfyUI loads them into the bf16 model. GEMMs: gemm.cpp (cuBLASLt).
// Env: SV2VAE_PROF=1 prints GPU time per section and peak device memory per call; SV2VAE_TAPS=<dir> writes tile 0 /
// pass 0 activations (golden/test_seedvr2_vae.py compares them with the dump's taps).
#include "common.cuh"
#include "st.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <initializer_list>
#include <map>
#include <string>
#include <vector>

namespace {
constexpr int SLICE = 1;                     // latents per pass after a tile's first pass (which has one more)
constexpr float SCALE = 0.9152f;             // BYTEDANCE_VAE_SCALING_FACTOR
constexpr int ETILE = 512, EOV = 128;        // encode tiles (pixels)
constexpr int DTILE = 64, DOV = 16;          // decode tiles (latents)

inline size_t nblk(size_t n, int t) { return (n + t - 1) / t; }
inline int rup(int a, int b) { return (a + b - 1) / b * b; }
__device__ __forceinline__ float rd(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
inline float hround(float v) { return __bfloat162float(__float2bfloat16(v)); }

// device memory with a running total (SV2VAE_PROF=1 prints the peak per call)
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
// SV2VAE_PROF=1: GPU time per section (events between marks) and peak device memory, printed after each call
struct Prof {
    bool on = getenv("SV2VAE_PROF") != nullptr;
    std::vector<std::pair<const char*, cudaEvent_t>> ev;
    std::map<std::string, double> fl;  // FLOPs per section (TFLOPS in the report)
    void flops(const char* name, double f) {
        if (on) fl[name] += f;
    }
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
        fprintf(stderr, "[seedvr2_vae %s] %.0f ms:", what, tot);
        for (auto& kv : s) {
            fprintf(stderr, " %s %.0f", kv.first.c_str(), kv.second);
            if (fl.count(kv.first)) fprintf(stderr, " (%.0f TFLOPS)", fl[kv.first] / kv.second / 1e9);
        }
        fl.clear();
        fprintf(stderr, "; peak device memory %.2f GiB (weights included)\n", g_peak / 1073741824.0);
        for (auto& x : ev) cudaEventDestroy(x.second);
        ev.clear();
    }
};
Prof prof;

template <typename T>
T* up(const std::vector<T>& h) {
    T* d = (T*)dmalloc(h.size() * sizeof(T));
    CK(cudaMemcpy(d, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return d;
}

// ---------------- kernels ----------------
__device__ float block_red(float v, float* sh, bool mx) {
    for (int o = 16; o; o >>= 1) {
        const float u = __shfl_xor_sync(0xffffffffu, v, o);
        v = mx ? fmaxf(v, u) : v + u;
    }
    const int w = threadIdx.x >> 5, l = threadIdx.x & 31, nw = blockDim.x >> 5;
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

// pixels [F][H][W][3] f32, frames f0.., tile (y0, x0) th x tw -> bf16(2p - 1) [T][th][tw][3]
__global__ void k_px_in(const float* __restrict__ px, int H, int W, int f0, int T, int y0, int x0, int th, int tw,
                        bf16* __restrict__ x) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * th * tw * 3) return;
    const int c = (int)(i % 3);
    const size_t p = i / 3;
    const int xx = (int)(p % tw), yy = (int)((p / tw) % th), t = (int)(p / ((size_t)tw * th));
    const float v = px[(((size_t)(f0 + t) * H + y0 + yy) * W + x0 + xx) * 3 + c];
    x[i] = __float2bfloat16(__fsub_rn(__fmul_rn(v, 2.f), 1.f));
}

// latent [16][Tz][Hl][Wl] f32 (scaled), frames t0.., tile (y0, x0) -> bf16(bf16(z) * inv) [T][th][tw][16]
__global__ void k_lat_in(const float* __restrict__ lat, int Tz, int Hl, int Wl, int t0, int T, int y0, int x0, int th,
                         int tw, float inv, bf16* __restrict__ x) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * th * tw * 16) return;
    const int c = (int)(i % 16);
    const size_t p = i / 16;
    const int xx = (int)(p % tw), yy = (int)((p / tw) % th), t = (int)(p / ((size_t)tw * th));
    x[i] = __float2bfloat16(__fmul_rn(rd(lat[(((size_t)c * Tz + t0 + t) * Hl + y0 + yy) * Wl + x0 + xx]), inv));
}

// per-frame GroupNorm(32) statistics, deterministic: per-(frame, 256-pixel chunk, channel) partial sums in pixel order
// (pass 0: x, pass 1: (x - mean)^2, no fma), reduced per (frame, group). A thread sums 8 channels of one chunk
// (uint4 loads); a block takes 4 chunks.
template <int PASS>
__global__ void k_gn_part(const bf16* __restrict__ x, int HW, int C, int nch, int nchunks, const float2* __restrict__ st,
                          float* __restrict__ part) {
    const int CG = C / 8, q = threadIdx.x / CG, c0 = (threadIdx.x % CG) * 8, k = blockIdx.x * 4 + q;
    if (k >= nchunks) return;
    const int t = k / nch, p0 = (k % nch) * 256, p1 = min(p0 + 256, HW);
    const bf16* xs = x + (size_t)t * HW * C + c0;
    float mean[8], s[8];
#pragma unroll
    for (int e = 0; e < 8; e++) mean[e] = PASS ? st[t * 32 + (c0 + e) / (C / 32)].x : 0.f, s[e] = 0.f;
#pragma unroll 4
    for (int p = p0; p < p1; p++) {
        const uint4 r = *(const uint4*)(xs + (size_t)p * C);
        const __nv_bfloat162* h = (const __nv_bfloat162*)&r;
#pragma unroll
        for (int e = 0; e < 8; e++) {
            const float d = __fsub_rn(e & 1 ? __high2float(h[e / 2]) : __low2float(h[e / 2]), mean[e]);
            s[e] = __fadd_rn(s[e], PASS ? __fmul_rn(d, d) : d);
        }
    }
    float* o = part + (size_t)k * C + c0;
#pragma unroll
    for (int e = 0; e < 8; e++) o[e] = s[e];
}
// one warp per (frame, group): pass 0 writes (mean, 0), pass 1 (mean, rstd)
__global__ void k_gn_fin(const float* __restrict__ part, int HW, int C, int nch, int pass, float2* __restrict__ st) {
    const int t = blockIdx.x / 32, cg = C / 32, g = blockIdx.x % 32, lane = threadIdx.x;
    float s = 0.f;
    for (int i = lane; i < nch * cg; i += 32) s += part[((size_t)t * nch + i / cg) * C + g * cg + i % cg];
    for (int o = 16; o; o >>= 1) s += __shfl_xor_sync(0xffffffffu, s, o);
    if (lane) return;
    const float n = (float)HW * cg;
    st[blockIdx.x] = pass == 0 ? make_float2(s / n, 0.f) : make_float2(st[blockIdx.x].x, rsqrtf(s / n + 1e-6f));
}

// padded frames [T][Hp][Wp][Cp] from x [T][H][W][C] at offset (pt, pl): zero border and channels >= C; with st,
// per-frame GroupNorm (rounded) and SiLU (rounded) if silu. Frames t < nP go to P, the rest to P2 (the next causal
// cache). E channels per thread (8 when C % 8 == 0).
template <int E>
__global__ void k_pad(const bf16* __restrict__ x, int T, int H, int W, int C, int Cp, const float2* __restrict__ st,
                      const float* __restrict__ gw, const float* __restrict__ gb, int silu, int pt, int pl, int Hp, int Wp,
                      bf16* __restrict__ P, int nP, bf16* __restrict__ P2) {
    const int CE = Cp / E;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)T * Hp * Wp * CE) return;
    const int c0 = (int)(i % CE) * E;
    const size_t p = i / CE;
    const int xp = (int)(p % Wp), yp = (int)((p / Wp) % Hp), t = (int)(p / ((size_t)Wp * Hp));
    const int y = yp - pt, xx = xp - pl;
    float v[E];
    for (int e = 0; e < E; e++) v[e] = 0.f;
    if (y >= 0 && y < H && xx >= 0 && xx < W && c0 < C) {
        const bf16* src = x + (((size_t)t * H + y) * W + xx) * C + c0;
        if (E == 8) {
            const uint4 r = *(const uint4*)src;
            const __nv_bfloat162* h = (const __nv_bfloat162*)&r;
            for (int e = 0; e < 4; e++) v[2 * e] = __low2float(h[e]), v[2 * e + 1] = __high2float(h[e]);
        } else {
            v[0] = ld(src);
        }
        if (st) {
            for (int e = 0; e < E; e++) {
                const int c = c0 + e;
                const float2 s = st[t * 32 + c / (C / 32)];
                const float a = s.y * gw[c], b = gb[c] - a * s.x;
                v[e] = rd(a * v[e] + b);
                if (silu) v[e] = rd(v[e] / (1.f + expf(-v[e])));
            }
        }
    }
    bf16* dst = (t < nP ? P + p * Cp : P2 + (p - (size_t)nP * Hp * Wp) * Cp) + c0;
    if (E == 8) {
        uint4 r;
        __nv_bfloat162* h = (__nv_bfloat162*)&r;
        for (int e = 0; e < 4; e++) h[e] = __floats2bfloat162_rn(v[2 * e], v[2 * e + 1]);
        *(uint4*)dst = r;
    } else {
        dst[0] = __float2bfloat16(v[0]);
    }
}

// implicit-GEMM conv over padded P [Tp][Hp][Wp][C] (C % 32 == 0), kernel (nt, 3, 3), strides (st, ss, ss):
// out[m][n] = bf16(bf16(bf16(sum_k A[m][k] W[n][k]) + b[n]) + resid), n < N, K order (kt, ky, kx, c), fp32 accumulate.
// BM x BN x 32 tiles, warps of WM x WN, ST-stage cp.async pipeline, ldmatrix + mma.sync bf16 (from h3_vae.cu k_iconv).
// Every output accumulates its k16 steps in the same order whatever the tile shape, so the result does not depend on it.
// Input frames come from a table (halo frames read in place from the causal cache, no copies).
__device__ __forceinline__ int sw32(int row, int ch) { return row * 32 + ((ch ^ ((row >> 1) & 3)) << 3); }
__device__ __forceinline__ void cp16(void* s, const void* g, int bytes) {
    unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("cp.async.cg.shared.global [%0], [%1], 16, %2;\n" ::"r"(sa), "l"(g), "r"(bytes));
}
__device__ __forceinline__ void ldm4(uint32_t* r, const void* s) {
    unsigned sa = (unsigned)__cvta_generic_to_shared(s);
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3]) : "r"(sa));
}
__device__ __forceinline__ void mma(float* c, const uint32_t* a, uint32_t b0, uint32_t b1) {
    asm volatile("mma.sync.aligned.m16n8k16.row.col.f32.bf16.bf16.f32 {%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
                 : "+f"(c[0]), "+f"(c[1]), "+f"(c[2]), "+f"(c[3])
                 : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}
struct Frames {  // padded input frames [Hp][Wp][C] of one pass: halo (cache) frames, then the new ones
    const bf16* p[4 * SLICE + 3];
};
template <int BM, int BN, int WM, int WN, int ST>
__global__ void __launch_bounds__((BM / WM) * (BN / WN) * 32) k_iconv(Frames fr, int Hp, int Wp, int C, int nt, int st, int ss, int Ho, int Wo,
                                               int M, const bf16* __restrict__ Wt, int kp, const float* __restrict__ bias,
                                               const bf16* __restrict__ resid, int N, bf16* __restrict__ out) {
    constexpr int MT = WM / 16, NP = WN / 16, WGN = BN / WN, NT = (BM / WM) * WGN * 32;
    extern __shared__ __align__(128) unsigned char smem[];
    bf16* sA = (bf16*)smem;           // [ST][BM * 32]
    bf16* sB = sA + ST * BM * 32;     // [ST][BN * 32]
    __shared__ const bf16* rowp[3][BM];  // per row and temporal tap: its window's corner
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, wm = warp / WGN, wn = warp % WGN;
    const int m0 = blockIdx.y * BM, n0 = blockIdx.x * BN;  // n fastest: the n blocks of one m block share A in L2
    for (int r = tid; r < BM; r += NT) {
        const int m = min(m0 + r, M - 1);
        const int xo = m % Wo, yo = (m / Wo) % Ho, to = m / (Wo * Ho);
        const size_t sp = ((size_t)yo * ss * Wp + (size_t)xo * ss) * C;
        for (int kt = 0; kt < nt; kt++) rowp[kt][r] = fr.p[to * st + kt] + sp;
    }
    __syncthreads();
    const int cpt = C / 32, nk = nt * 9 * cpt;
    int lk = 0, lc = 0, ltap = 0;  // next k chunk to load: index, channel chunk, tap
    auto load = [&](int slot) {
        const int kt = ltap / 9, ky = (ltap / 3) % 3, kx = ltap % 3;
        const int toff = (ky * Wp + kx) * C + lc * 32;
        bf16 *a = sA + slot * BM * 32, *b = sB + slot * BN * 32;
#pragma unroll
        for (int i = tid; i < BM * 4; i += NT) {
            const int r = i >> 2, ch = i & 3;
            const bool ok = m0 + r < M;
            cp16(a + sw32(r, ch), rowp[kt][r] + (ok ? toff : 0) + ch * 8, ok ? 16 : 0);
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
        __syncthreads();  // chunk kc landed for all threads; slot (kc - 1) % ST is free
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
                if (resid) v0 += ld(resid + o);
                if (n + 1 < N) {
                    float v1 = rd(rd(acc[mt][j][2 * h + 1]) + bias[n + 1]);
                    if (resid) v1 += ld(resid + o + 1);
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

struct IconvArgs {
    Frames fr;
    int Hp, Wp, C, nt, st, ss, Ho, Wo, M;
    const bf16* w;
    int kp;
    const float* b;
    const bf16* resid;
    int N;
    bf16* out;
};
template <int BM, int BN, int WM, int WN, int ST>
void iconv(cudaStream_t s, const IconvArgs& a, int np) {
    constexpr int smem = ST * (BM + BN) * 32 * 2;
    static bool init = false;
    if (!init) {
        CK(cudaFuncSetAttribute(k_iconv<BM, BN, WM, WN, ST>, cudaFuncAttributeMaxDynamicSharedMemorySize, smem));
        init = true;
    }
    k_iconv<BM, BN, WM, WN, ST><<<dim3(np / BN, (unsigned)nblk(a.M, BM)), (BM / WM) * (BN / WN) * 32, smem, s>>>(
        a.fr, a.Hp, a.Wp, a.C, a.nt, a.st, a.ss, a.Ho, a.Wo, a.M, a.w, a.kp, a.b, a.resid, a.N, a.out);
}

// y = bf16(bf16(y + b) + r) over [M][N]
__global__ void k_bias_res(bf16* __restrict__ y, size_t M, int N, const float* __restrict__ b, const bf16* __restrict__ r) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= M * N) return;
    float v = rd(ld(y + i) + b[i % N]);
    if (r) v += ld(r + i);
    y[i] = __float2bfloat16(v);
}
// rows of [R][L]: y = bf16(y + b[row])
__global__ void k_bias_rows(bf16* __restrict__ y, int R, int L, const float* __restrict__ b) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)R * L) y[i] = __float2bfloat16(ld(y + i) + b[i / L]);
}

// upsampler pixel shuffle straight into the conv's padded frames: y [T][H][W][4 tr C] (1x1 conv output, channel
// ((i 2 + j) tr + t) C + c) -> [To][2H + 2][2W + 2][C] = bf16(y + b) with a zero border; frame to = f tr + t, the first
// pass drops frame 1 (the first frame's phase 1). Frames < nP go to P, the rest to P2 (as k_pad). 8 channels a thread.
__global__ void k_shuffle_pad(const bf16* __restrict__ y, int H, int W, int C, int tr, int drop, int To,
                              const float* __restrict__ b, bf16* __restrict__ P, int nP, bf16* __restrict__ P2) {
    const int Hp = 2 * H + 2, Wp = 2 * W + 2, C8 = C / 8;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)To * Hp * Wp * C8) return;
    const int c0 = (int)(i % C8) * 8;
    const size_t p = i / C8;
    const int xp = (int)(p % Wp), yp = (int)((p / Wp) % Hp), to = (int)(p / ((size_t)Wp * Hp));
    const int Y = yp - 1, X = xp - 1;
    uint4 r = make_uint4(0, 0, 0, 0);
    if (Y >= 0 && Y < 2 * H && X >= 0 && X < 2 * W) {
        const int fi = drop && to > 0 ? to + 1 : to, f = fi / tr, t = fi % tr;
        const int k = (((Y & 1) * 2 + (X & 1)) * tr + t) * C + c0;
        const uint4 s = *(const uint4*)(y + (((size_t)f * H + (Y >> 1)) * W + (X >> 1)) * 4 * tr * C + k);
        const __nv_bfloat162* hs = (const __nv_bfloat162*)&s;
        __nv_bfloat162* hr = (__nv_bfloat162*)&r;
        for (int e = 0; e < 4; e++)
            hr[e] = __floats2bfloat162_rn(__low2float(hs[e]) + b[k + 2 * e], __high2float(hs[e]) + b[k + 2 * e + 1]);
    }
    *(uint4*)((to < nP ? P + p * C : P2 + (p - (size_t)nP * Hp * Wp) * C) + c0) = r;
}

// attention row softmax: S [L][L] f32 row r -> P = bf16(exp((s - max) * scale)) (unnormalized), l[r] = sum (fp32)
__global__ void k_softmax(const float* __restrict__ S, bf16* __restrict__ P, float* __restrict__ l, int L, float scale) {
    __shared__ float sh[32];
    const float* s = S + (size_t)blockIdx.x * L;
    bf16* p = P + (size_t)blockIdx.x * L;
    float m = -INFINITY;
    for (int j = threadIdx.x; j < L; j += blockDim.x) m = fmaxf(m, s[j]);
    m = block_red(m, sh, true);
    float sum = 0.f;
    for (int j = threadIdx.x; j < L; j += blockDim.x) {
        const float e = expf((s[j] - m) * scale);
        sum += e;
        p[j] = __float2bfloat16(e);
    }
    sum = block_red(sum, sh, false);
    if (threadIdx.x == 0) l[blockIdx.x] = sum;
}
// o = bf16(O * (1 / l[row])) over [L][C]
__global__ void k_attn_norm(const float* __restrict__ O, const float* __restrict__ l, int L, int C, bf16* __restrict__ o) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < (size_t)L * C) o[i] = __float2bfloat16(O[i] * (1.f / l[i / C]));
}

// tiled_vae blend (fp32, as torch on the CPU): cnt += wy x wx; out += v * (wy x wx); out = bf16(out / max(cnt, 1e-6))
__global__ void k_cnt(const float* __restrict__ wy, const float* __restrict__ wx, int th, int tw, int y0, int x0, int Wc,
                      float* __restrict__ cnt) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= th * tw) return;
    float* c = cnt + (size_t)(y0 + i / tw) * Wc + x0 + i % tw;
    *c = __fadd_rn(*c, __fmul_rn(wy[i / tw], wx[i % tw]));
}
// encode: tile latent [nT][th][tw][16] -> out [16][T][Hl][Wl] at (t0, y0, x0)
__global__ void k_enc_acc(const bf16* __restrict__ z, int nT, int th, int tw, const float* __restrict__ wy,
                          const float* __restrict__ wx, int T, int Hl, int Wl, int t0, int y0, int x0, float* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)16 * nT * th * tw) return;
    const int x = (int)(i % tw), y = (int)((i / tw) % th), t = (int)((i / ((size_t)tw * th)) % nT);
    const int c = (int)(i / ((size_t)tw * th * nT));
    const float v = __fmul_rn(ld(z + (((size_t)t * th + y) * tw + x) * 16 + c), __fmul_rn(wy[y], wx[x]));
    float* o = out + (((size_t)c * T + t0 + t) * Hl + y0 + y) * Wl + x0 + x;
    *o = __fadd_rn(*o, v);
}
__global__ void k_enc_fin(float* __restrict__ out, const float* __restrict__ cnt, int HW, size_t n, float scale) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) out[i] = __fmul_rn(rd(__fdiv_rn(out[i], fmaxf(cnt[i % HW], 1e-6f))), scale);
}
// decode: tile pixels [nf][th][tw][3] -> out [F][H][W][3] at (f0, y0, x0)
__global__ void k_dec_acc(const bf16* __restrict__ y, int nf, int th, int tw, const float* __restrict__ wy,
                          const float* __restrict__ wx, int H, int W, int f0, int y0, int x0, float* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)nf * th * tw * 3) return;
    const size_t p = i / 3;
    const int x = (int)(p % tw), yy = (int)((p / tw) % th), f = (int)(p / ((size_t)tw * th));
    const float v = __fmul_rn(ld(y + i), __fmul_rn(wy[yy], wx[x]));
    float* o = out + (((size_t)(f0 + f) * H + y0 + yy) * W + x0 + x) * 3 + i % 3;
    *o = __fadd_rn(*o, v);
}
__global__ void k_dec_fin(float* __restrict__ out, const float* __restrict__ cnt, int HW, size_t n) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= n) return;
    const float v = rd(__fdiv_rn(out[i], fmaxf(cnt[(i / 3) % HW], 1e-6f)));
    out[i] = fminf(fmaxf(__fmul_rn(__fadd_rn(v, 1.f), 0.5f), 0.f), 1.f);
}

// ---------------- model ----------------
struct Act {  // NDHWC bf16
    bf16* p;
    int T, H, W, C;
};
struct Norm {
    float *w = nullptr, *b = nullptr;
};
struct Conv {
    bf16* w = nullptr;  // k3: [np][kt][3][3][cp] zero-padded; 1x1x1: [cout][cin]
    float* b = nullptr;
    int cin = 0, cp = 0, cout = 0, np = 0, kt = 1, ks = 1, st = 1, ss = 1, kp = 0;
    int pt = 1, pb = 1, pl = 1, pr = 1;  // spatial zero border (downsamplers: 0, 1, 0, 1)
    bf16* cache[2] = {};                  // causal tail (last kt - st padded input frames), ping-pong per pass
    size_t ccap = 0;
    int cur = 0;
};
struct Res {
    Norm n1, n2;
    Conv c1, c2, sc;
};
struct Attn {
    Norm n;
    bf16 *wq, *wk, *wv, *wo;
    float *bq, *bk, *bv, *bo;
};
struct Mid {
    Res r0, r1;
    Attn a;
};
struct Up {
    Conv e, conv;  // 1x1x1 expansion (cout 4 tr C), causal conv
    int tr = 1;
};

struct Vae {
    cudaStream_t st;
    Lt* lt;
    StFile f;
    Conv ecin, edown[3], ecout, dcin, dcout;
    Res er[4][2], dr[4][3];
    Mid em, dm;
    Up du[3];
    Norm eno, dno;
    // work (per call)
    size_t cap = 0;
    bf16 *buf[4] = {}, *P = nullptr;
    float2* gst = nullptr;
    float* gpart = nullptr;
    size_t gpcap = 0;
    bf16 *xn = nullptr, *q = nullptr, *k = nullptr, *vt = nullptr, *ao = nullptr, *Pm = nullptr;
    float *S = nullptr, *O = nullptr, *l = nullptr;
    float *wy = nullptr, *wx = nullptr;
    // SV2VAE_TAPS=<dir>: write tile 0 / pass 0 activations there (raw bf16 NDHWC, header int32 T H W C)
    const char* tapdir = nullptr;
    bool tapping = false;
};

std::vector<float> hf(Vae* v, const std::string& name, size_t n) {  // F16 -> float, bf16-rounded
    const StTensor& t = v->f.get(name);
    if (t.dtype != "F16" || t.bytes() != n * 2) DIE("%s: %s %llu bytes, expected F16 x %zu", name.c_str(), t.dtype.c_str(),
                                                    (unsigned long long)t.bytes(), n);
    const __half* p = (const __half*)v->f.ptr(t);
    std::vector<float> o(n);
    for (size_t i = 0; i < n; i++) o[i] = hround(__half2float(p[i]));
    return o;
}
float* upf(Vae* v, const std::string& name, size_t n) { return up(hf(v, name, n)); }
bf16* upb(const std::vector<float>& w) {
    std::vector<bf16> h(w.size());
    for (size_t i = 0; i < w.size(); i++) h[i] = __float2bfloat16(w[i]);
    return up(h);
}
Norm load_norm(Vae* v, const std::string& p, int c) { return Norm{upf(v, p + ".weight", c), upf(v, p + ".bias", c)}; }

// conv weight [cout][cin][kt][ks][ks]; keep = output channels kept (encoder conv_out: the mean half)
Conv load_conv(Vae* v, const std::string& p, int cin, int cout, int kt, int ks, int st = 1, int ss = 1, int keep = 0) {
    Conv c;
    c.cin = cin, c.cout = keep ? keep : cout, c.kt = kt, c.ks = ks, c.st = st, c.ss = ss;
    const int taps = kt * ks * ks;
    std::vector<float> w = hf(v, p + ".weight", (size_t)cout * cin * taps), b = hf(v, p + ".bias", cout);
    if (ks == 1) {
        c.cp = cin, c.np = c.cout, c.kp = cin;
    } else {
        c.cp = rup(cin, 32), c.np = rup(c.cout, c.cout % 128 ? 16 : 128), c.kp = taps * c.cp;
    }
    std::vector<float> r((size_t)c.np * c.kp, 0.f);
    for (int o = 0; o < c.cout; o++)
        for (int i = 0; i < cin; i++)
            for (int t = 0; t < taps; t++) r[(size_t)o * c.kp + (size_t)t * c.cp + i] = w[((size_t)o * cin + i) * taps + t];
    c.w = upb(r);
    b.resize(c.cout);
    c.b = up(b);
    return c;
}
Res load_res(Vae* v, const std::string& p, int cin, int cout) {
    Res r;
    r.n1 = load_norm(v, p + ".norm1", cin), r.n2 = load_norm(v, p + ".norm2", cout);
    r.c1 = load_conv(v, p + ".conv1", cin, cout, 3, 3);
    r.c2 = load_conv(v, p + ".conv2", cout, cout, 3, 3);
    if (cin != cout) r.sc = load_conv(v, p + ".conv_shortcut", cin, cout, 1, 1);
    return r;
}
Mid load_mid(Vae* v, const std::string& p) {
    Mid m;
    m.r0 = load_res(v, p + ".resnets.0", 512, 512), m.r1 = load_res(v, p + ".resnets.1", 512, 512);
    const std::string a = p + ".attentions.0.";
    m.a.n = load_norm(v, a + "group_norm", 512);
    m.a.wq = upb(hf(v, a + "to_q.weight", 512 * 512)), m.a.bq = upf(v, a + "to_q.bias", 512);
    m.a.wk = upb(hf(v, a + "to_k.weight", 512 * 512)), m.a.bk = upf(v, a + "to_k.bias", 512);
    m.a.wv = upb(hf(v, a + "to_v.weight", 512 * 512)), m.a.bv = upf(v, a + "to_v.bias", 512);
    m.a.wo = upb(hf(v, a + "to_out.0.weight", 512 * 512)), m.a.bo = upf(v, a + "to_out.0.bias", 512);
    return m;
}

void tap(Vae* v, const std::string& name, const Act& a) {
    if (!v->tapping) return;
    CK(cudaStreamSynchronize(v->st));
    std::vector<bf16> h((size_t)a.T * a.H * a.W * a.C);
    CK(cudaMemcpy(h.data(), a.p, h.size() * 2, cudaMemcpyDeviceToHost));
    FILE* fp = fopen((std::string(v->tapdir) + "/" + name + ".bin").c_str(), "wb");
    if (!fp) DIE("cannot write tap %s", name.c_str());
    const int hd[4] = {a.T, a.H, a.W, a.C};
    fwrite(hd, 4, 4, fp);
    fwrite(h.data(), 2, h.size(), fp);
    fclose(fp);
}

bf16* free_buf(Vae* v, std::initializer_list<const void*> used) {
    for (bf16* b : v->buf) {
        bool u = false;
        for (const void* q : used) u |= q == b;
        if (!u) return b;
    }
    DIE("seedvr2_vae: no free buffer");
    return nullptr;
}

const float2* gn_stats(Vae* v, const Act& x) {
    const int HW = x.H * x.W, nch = (HW + 255) / 256;
    const size_t need = (size_t)x.T * nch * x.C;
    if (x.T > 4 * SLICE + 1) DIE("seedvr2_vae: %d frames in one pass", x.T);
    if (need > v->gpcap) {
        CK(cudaStreamSynchronize(v->st));
        dfree(v->gpart);
        v->gpcap = need, v->gpart = (float*)dmalloc(need * 4);
    }
    for (int pass = 0; pass < 2; pass++) {
        const int nk = x.T * nch;
        if (pass) k_gn_part<1><<<(nk + 3) / 4, x.C / 2, 0, v->st>>>(x.p, HW, x.C, nch, nk, v->gst, v->gpart);
        else k_gn_part<0><<<(nk + 3) / 4, x.C / 2, 0, v->st>>>(x.p, HW, x.C, nch, nk, nullptr, v->gpart);
        k_gn_fin<<<x.T * 32, 32, 0, v->st>>>(v->gpart, HW, x.C, nch, pass, v->gst);
    }
    return v->gst;
}

void pad(Vae* v, const Act& x, int Cp, const Norm* n, int silu, int pt, int pl, int Hp, int Wp, bf16* P, int nP,
         bf16* P2) {
    prof.mark(v->st, "other");
    const float2* gs = n ? gn_stats(v, x) : nullptr;
    prof.mark(v->st, "gn_stats");
    const float *gw = n ? n->w : nullptr, *gb = n ? n->b : nullptr;
    const size_t m = (size_t)x.T * Hp * Wp * Cp;
    if (x.C % 8 == 0)
        k_pad<8><<<nblk(m / 8, 256), 256, 0, v->st>>>(x.p, x.T, x.H, x.W, x.C, Cp, gs, gw, gb, silu, pt, pl, Hp, Wp, P, nP, P2);
    else
        k_pad<1><<<nblk(m, 256), 256, 0, v->st>>>(x.p, x.T, x.H, x.W, x.C, Cp, gs, gw, gb, silu, pt, pl, Hp, Wp, P, nP, P2);
    prof.mark(v->st, "pad");
}

// InflatedCausalConv3d (optional GroupNorm + SiLU first). first: a tile's first pass (head = first frame twice).
struct Shuf {  // an upsampler's 1x1 conv output, shuffled into the conv's padded input instead of x
    const bf16* y;
    int H, W, tr, drop;
    const float* b;
};
Act conv(Vae* v, Conv& c, const Act& x, const Norm* n, bool first, bf16* out, const bf16* resid,
         const Shuf* sh = nullptr) {
    const cudaStream_t st = v->st;
    if (x.C != c.cin) DIE("seedvr2_vae: conv expects %d channels, got %d", c.cin, x.C);
    if (c.ks == 1) {  // 1x1x1: matmul rounded, + bias rounded (+ resid rounded)
        const size_t M = (size_t)x.T * x.H * x.W;
        prof.mark(st, "other");
        gemm_bf16(v->lt, (int)M, c.cout, c.cin, x.p, c.w, out);
        k_bias_res<<<nblk(M * c.cout, 256), 256, 0, st>>>(out, M, c.cout, c.b, resid);
        prof.mark(st, "conv1x1");
        return Act{out, x.T, x.H, x.W, c.cout};
    }
    // padded input sequence = [halo, new frames]: halo = the first frame twice (a tile's first pass) or the cached tail.
    // New frames go to P, except the last cs (the next tail), which go straight to the other cache buffer.
    const int halo = c.kt == 1 ? 0 : first ? 2 : c.kt - c.st, cs = c.kt - c.st, T = x.T;
    const int Hp = x.H + c.pt + c.pb, Wp = x.W + c.pl + c.pr, Tp = halo + T, nP = std::max(0, T - cs);
    const size_t fs = (size_t)Hp * Wp * c.cp;
    if (nP * fs > v->cap || Tp > 4 * SLICE + 3) DIE("seedvr2_vae: pad buffer too small");
    if (cs * fs > c.ccap) {
        CK(cudaStreamSynchronize(st));
        dfree(c.cache[0]), dfree(c.cache[1]);
        c.ccap = cs * fs, c.cache[0] = (bf16*)dmalloc(c.ccap * 2), c.cache[1] = (bf16*)dmalloc(c.ccap * 2);
    }
    bf16* nxt = c.cache[c.cur ^ 1];
    auto newf = [&](int t) -> const bf16* { return t < T - cs ? v->P + t * fs : nxt + (t - (T - cs)) * fs; };
    if (sh) {
        prof.mark(st, "other");
        k_shuffle_pad<<<nblk((size_t)T * Hp * Wp * x.C / 8, 256), 256, 0, st>>>(sh->y, sh->H, sh->W, x.C, sh->tr, sh->drop, T,
                                                                               sh->b, v->P, T - cs, nxt);
        prof.mark(st, "upsample_1x1");
    } else {
        pad(v, x, c.cp, n, 1, c.pt, c.pl, Hp, Wp, v->P, T - cs, nxt);
    }
    Frames fr{};
    for (int f = 0; f < halo; f++) fr.p[f] = first ? newf(0) : c.cache[c.cur] + f * fs;
    for (int t = 0; t < T; t++) fr.p[halo + t] = newf(t);
    for (int j = 0; j < cs - T; j++)  // a pass shorter than the tail: the tail's head comes from the halo
        CK(cudaMemcpyAsync(nxt + j * fs, fr.p[halo + T - cs + j], fs * 2, cudaMemcpyDeviceToDevice, st));
    if (cs) c.cur ^= 1;
    Act y{out, (Tp - c.kt) / c.st + 1, (Hp - 3) / c.ss + 1, (Wp - 3) / c.ss + 1, c.cout};
    const int M = y.T * y.H * y.W;
    prof.mark(st, "other");
    // tile sweep on GB10 (1088x1920 decode): 128 x 128, 2 stages, 2 blocks / SM beat 3-4 stages and 64 x 64 warp tiles
    const IconvArgs ia{fr, Hp, Wp, c.cp, c.kt, c.st, c.ss, y.H, y.W, M, c.w, c.kp, c.b, resid, c.cout, out};
    if (c.cout % 128 == 0) iconv<128, 128, 64, 32, 2>(st, ia, c.np);
    else iconv<256, 16, 32, 16, 2>(st, ia, c.np);
    prof.mark(st, c.cout == 128 ? "conv128" : c.cout == 256 ? "conv256" : c.cout == 512 ? "conv512" : "conv_small");
    prof.flops(c.cout == 128 ? "conv128" : c.cout == 256 ? "conv256" : c.cout == 512 ? "conv512" : "conv_small",
               2.0 * M * c.np * c.kp);
    return y;
}

Act res(Vae* v, Res& r, Act x, bool first) {
    const Act h = conv(v, r.c1, x, &r.n1, first, free_buf(v, {x.p}), nullptr);
    const bf16* rs = x.p;
    if (r.sc.w) rs = conv(v, r.sc, x, nullptr, first, free_buf(v, {x.p, h.p}), nullptr).p;
    return conv(v, r.c2, h, &r.n2, first, free_buf(v, {x.p, h.p, rs}), rs);
}

// Attention (mid block), per frame: GroupNorm -> q, k, v -> softmax(q k^T / sqrt(512)) v -> out proj + x
Act attn(Vae* v, Attn& a, Act x) {
    const cudaStream_t st = v->st;
    const int L = x.H * x.W, C = 512;
    const size_t M = (size_t)x.T * L;
    pad(v, x, C, &a.n, 0, 0, 0, x.H, x.W, v->xn, x.T, nullptr);
    prof.mark(st, "other");
    gemm_bf16(v->lt, (int)M, C, C, v->xn, a.wq, v->q);
    k_bias_res<<<nblk(M * C, 256), 256, 0, st>>>(v->q, M, C, a.bq, nullptr);
    gemm_bf16(v->lt, (int)M, C, C, v->xn, a.wk, v->k);
    k_bias_res<<<nblk(M * C, 256), 256, 0, st>>>(v->k, M, C, a.bk, nullptr);
    const float scale = 1.f / sqrtf((float)C);
    for (int t = 0; t < x.T; t++) {
        const size_t o = (size_t)t * L * C;
        gemm_bf16(v->lt, C, L, C, a.wv, v->xn + o, v->vt);  // v^T [C][L]
        k_bias_rows<<<nblk((size_t)C * L, 256), 256, 0, st>>>(v->vt, C, L, a.bv);
        gemm_bf16_f32(v->lt, L, L, C, v->q + o, v->k + o, v->S);
        k_softmax<<<L, 256, 0, st>>>(v->S, v->Pm, v->l, L, scale);
        gemm_bf16_f32(v->lt, L, C, L, v->Pm, v->vt, v->O);
        k_attn_norm<<<nblk((size_t)L * C, 256), 256, 0, st>>>(v->O, v->l, L, C, v->ao + o);
    }
    bf16* out = free_buf(v, {x.p});
    gemm_bf16(v->lt, (int)M, C, C, v->ao, a.wo, out);
    k_bias_res<<<nblk(M * C, 256), 256, 0, st>>>(out, M, C, a.bo, x.p);
    prof.mark(st, "attn");
    return Act{out, x.T, x.H, x.W, C};
}

Act mid(Vae* v, Mid& m, Act x, bool first, const char* tp) {
    x = res(v, m.r0, x, first);
    tap(v, std::string(tp) + "mid.r0", x);
    x = attn(v, m.a, x);
    tap(v, std::string(tp) + "mid.attn", x);
    return res(v, m.r1, x, first);
}

Act upsample(Vae* v, Up& u, Act x, bool first) {
    const int C = x.C, drop = u.tr == 2 && first;
    bf16* y = free_buf(v, {x.p});
    prof.mark(v->st, "other");
    gemm_bf16(v->lt, x.T * x.H * x.W, u.e.cout, C, x.p, u.e.w, y);
    prof.mark(v->st, "upsample_1x1");
    const Shuf sh{y, x.H, x.W, u.tr, drop, u.e.b};
    return conv(v, u.conv, Act{nullptr, x.T * u.tr - drop, 2 * x.H, 2 * x.W, C}, nullptr, first, free_buf(v, {y}), nullptr, &sh);
}

// one pass of the encoder over pixels x [T][h][w][3] -> latent mean [T'][h/8][w/8][16]
Act enc_pass(Vae* v, Act x, bool first) {
    tap(v, "enc.in", x);
    x = conv(v, v->ecin, x, nullptr, first, free_buf(v, {x.p}), nullptr);
    tap(v, "enc.conv_in", x);
    for (int l = 0; l < 4; l++) {
        for (int j = 0; j < 2; j++) {
            x = res(v, v->er[l][j], x, first);
            tap(v, "enc.down" + std::to_string(l) + ".res" + std::to_string(j), x);
        }
        if (l < 3) {
            x = conv(v, v->edown[l], x, nullptr, first, free_buf(v, {x.p}), nullptr);
            tap(v, "enc.down" + std::to_string(l) + ".down", x);
        }
    }
    x = mid(v, v->em, x, first, "enc.");
    tap(v, "enc.mid", x);
    x = conv(v, v->ecout, x, &v->eno, first, free_buf(v, {x.p}), nullptr);
    tap(v, "enc.conv_out", x);
    return x;
}

// one pass of the decoder over latents z [T][h][w][16] (unscaled bf16) -> pixels [T'][8h][8w][3]
Act dec_pass(Vae* v, Act x, bool first) {
    tap(v, "dec.in", x);
    x = conv(v, v->dcin, x, nullptr, first, free_buf(v, {x.p}), nullptr);
    tap(v, "dec.conv_in", x);
    x = mid(v, v->dm, x, first, "dec.");
    tap(v, "dec.mid", x);
    for (int l = 0; l < 4; l++) {
        for (int j = 0; j < 3; j++) {
            x = res(v, v->dr[l][j], x, first);
            tap(v, "dec.up" + std::to_string(l) + ".res" + std::to_string(j), x);
        }
        if (l < 3) {
            x = upsample(v, v->du[l], x, first);
            tap(v, "dec.up" + std::to_string(l) + ".up", x);
        }
    }
    x = conv(v, v->dcout, x, &v->dno, first, free_buf(v, {x.p}), nullptr);
    tap(v, "dec.conv_out", x);
    return x;
}

// tiled_vae tile ranges over (h, w) in its units: stride ti - ov, tiles of <= ov past the start are skipped
struct Tile {
    int y0, y1, x0, x1;
};
std::vector<Tile> tiles(int h, int w, int ti, int ov) {
    ov = std::min(ov, ti - 1);
    std::vector<Tile> r;
    for (int y = 0; y < h; y += ti - ov) {
        const int ye = std::min(y + ti, h);
        if (y > 0 && ye - y <= ov) continue;
        for (int x = 0; x < w; x += ti - ov) {
            const int xe = std::min(x + ti, w);
            if (x > 0 && xe - x <= ov) continue;
            r.push_back({y, ye, x, xe});
        }
    }
    return r;
}

// tiled_vae weights along one axis of n outputs: ramp r = 0.5 - 0.5 cos(pi linspace(0, 1, k)) over the first k
// (when not at the image start) and 1 - r over the last k (when not at the end), k = min(ov, n / 2), fp32 like torch
std::vector<float> axis_w(int n, int ov, bool lo, bool hi) {
    std::vector<float> w(n, 1.f);
    const int k = std::min(ov, n / 2);
    if (k <= 0) return w;
    const float step = k > 1 ? 1.f / (float)(k - 1) : 0.f;
    for (int i = 0; i < k; i++) {
        const float t = k == 1 ? 0.f : i < k / 2 ? step * (float)i : 1.f - step * (float)(k - 1 - i);
        const float c = (float)std::cos((double)(t * (float)M_PI));
        const float r = 0.5f - 0.5f * c;
        if (lo) w[i] = r;
        if (hi) w[n - k + i] = 1.f - r;
    }
    return w;
}

// work buffers for tiles of th x tw output pixels: activations up to (4 SLICE + 3) frames x (th+2)(tw+2) x cmax
void work_alloc(Vae* v, int th, int tw, int cmax, int lh, int lw) {
    v->cap = (size_t)(4 * SLICE + 3) * (th + 2) * (tw + 2) * cmax;
    for (bf16*& b : v->buf) b = (bf16*)dmalloc(v->cap * 2);
    v->P = (bf16*)dmalloc(v->cap * 2);
    const size_t L = (size_t)lh * lw, M = (size_t)(SLICE + 1) * L * 512;
    v->xn = (bf16*)dmalloc(M * 2), v->q = (bf16*)dmalloc(M * 2), v->k = (bf16*)dmalloc(M * 2), v->ao = (bf16*)dmalloc(M * 2);
    v->vt = (bf16*)dmalloc(L * 512 * 2), v->O = (float*)dmalloc(L * 512 * 4), v->l = (float*)dmalloc(L * 4);
    v->S = (float*)dmalloc(L * L * 4), v->Pm = (bf16*)dmalloc(L * L * 2);
    v->wy = (float*)dmalloc((size_t)std::max(th, tw) * 4), v->wx = (float*)dmalloc((size_t)std::max(th, tw) * 4);
    v->gst = (float2*)dmalloc((size_t)(4 * SLICE + 1) * 32 * sizeof(float2));
    v->tapdir = getenv("SV2VAE_TAPS");
    g_peak = g_cur;
    prof.mark(v->st, "start");
}
void work_free(Vae* v, const char* what) {
    prof.mark(v->st, "other");
    CK(cudaStreamSynchronize(v->st));
    prof.report(what);
    std::vector<Conv*> cs = {&v->ecin, &v->ecout, &v->dcin, &v->dcout};
    for (Conv& c : v->edown) cs.push_back(&c);
    for (auto& l : v->er)
        for (Res& r : l) cs.push_back(&r.c1), cs.push_back(&r.c2);
    for (auto& l : v->dr)
        for (Res& r : l) cs.push_back(&r.c1), cs.push_back(&r.c2);
    for (Mid* m : {&v->em, &v->dm}) cs.push_back(&m->r0.c1), cs.push_back(&m->r0.c2), cs.push_back(&m->r1.c1), cs.push_back(&m->r1.c2);
    for (Up& u : v->du) cs.push_back(&u.conv);
    for (Conv* c : cs) dfree(c->cache[0]), dfree(c->cache[1]), c->cache[0] = c->cache[1] = nullptr, c->ccap = 0, c->cur = 0;
    for (bf16*& b : v->buf) dfree(b), b = nullptr;
    void* ps[] = {v->P, v->gst, v->gpart, v->xn, v->q, v->k, v->ao, v->vt, v->O, v->l, v->S, v->Pm, v->wy, v->wx};
    for (void* p : ps) dfree(p);
    v->P = nullptr, v->gst = nullptr, v->gpart = nullptr, v->gpcap = 0, v->cap = 0;
    v->tapping = false;
}

void tile_w(Vae* v, const Tile& t, int h, int w, int sc, int ov, int& th, int& tw) {
    th = (t.y1 - t.y0) * sc, tw = (t.x1 - t.x0) * sc;
    const std::vector<float> wy = axis_w(th, ov, t.y0 > 0, t.y1 < h), wx = axis_w(tw, ov, t.x0 > 0, t.x1 < w);
    CK(cudaStreamSynchronize(v->st));
    CK(cudaMemcpy(v->wy, wy.data(), th * 4, cudaMemcpyHostToDevice));
    CK(cudaMemcpy(v->wx, wx.data(), tw * 4, cudaMemcpyHostToDevice));
}
}  // namespace

extern "C" {

void* sv2_vae_new(const char* path) {
    Vae* v = new Vae();
    if (!v->f.open(path)) DIE("cannot open %s", path);
    CK(cudaStreamCreate(&v->st));   // blocking on purpose (common.cuh)
    v->lt = lt_new(v->st);
    // encoder: channels (128, 256, 512, 512), 2 resnets per block, downsample spatial / temporal+spatial x2
    const int ch[4] = {128, 256, 512, 512};
    v->ecin = load_conv(v, "encoder.conv_in", 3, 128, 3, 3);
    for (int l = 0; l < 4; l++) {
        const std::string p = "encoder.down_blocks." + std::to_string(l);
        const int cin = l ? ch[l - 1] : 128;
        v->er[l][0] = load_res(v, p + ".resnets.0", cin, ch[l]);
        v->er[l][1] = load_res(v, p + ".resnets.1", ch[l], ch[l]);
        if (l < 3) {
            Conv& d = v->edown[l];
            d = load_conv(v, p + ".downsamplers.0.conv", ch[l], ch[l], l ? 3 : 1, 3, l ? 2 : 1, 2);
            d.pt = 0, d.pb = 1, d.pl = 0, d.pr = 1;
        }
    }
    v->em = load_mid(v, "encoder.mid_block");
    v->eno = load_norm(v, "encoder.conv_norm_out", 512);
    v->ecout = load_conv(v, "encoder.conv_out", 512, 32, 3, 3, 1, 1, 16);
    // decoder: conv_in, mid, up blocks (512, 512, 256, 128) x 3 resnets, upsample temporal+spatial x2, spatial
    const int dch[4] = {512, 512, 256, 128};
    v->dcin = load_conv(v, "decoder.conv_in", 16, 512, 3, 3);
    v->dm = load_mid(v, "decoder.mid_block");
    for (int l = 0; l < 4; l++) {
        const std::string p = "decoder.up_blocks." + std::to_string(l);
        const int cin = l ? dch[l - 1] : 512;
        for (int j = 0; j < 3; j++) v->dr[l][j] = load_res(v, p + ".resnets." + std::to_string(j), j ? dch[l] : cin, dch[l]);
        if (l < 3) {
            Up& u = v->du[l];
            u.tr = l < 2 ? 2 : 1;
            u.e = load_conv(v, p + ".upsamplers.0.upscale_conv", dch[l], dch[l] * 4 * u.tr, 1, 1);
            u.conv = load_conv(v, p + ".upsamplers.0.conv", dch[l], dch[l], 3, 3);
        }
    }
    v->dno = load_norm(v, "decoder.conv_norm_out", 128);
    v->dcout = load_conv(v, "decoder.conv_out", 128, 3, 3, 3);
    v->f.close();
    CK(cudaDeviceSynchronize());  // uploads complete before the non-blocking stream uses them
    return v;
}

// frames sv2_vae_decode writes for T latents, latents sv2_vae_encode writes for F = 4n + 1 frames
int sv2_vae_frames(int T) { return 4 * (T - 1) + 1; }
int sv2_vae_latent_frames(int F) { return (F + 3) / 4; }

// px [F][H][W][3] f32 device in 0..1, F = 4n + 1, H and W multiples of 16 -> out [16][T][H/8][W/8] f32 device,
// T = sv2_vae_latent_frames(F): exactly what VAEEncodeTiled [512, 128, 64, 8] returns (scaled by 0.9152).
int sv2_vae_encode(void* h, const float* px, int F, int H, int W, float* out) {
    Vae* v = (Vae*)h;
    const cudaStream_t st = v->st;
    if (F < 1 || F % 4 != 1 || H % 16 || W % 16) DIE("sv2_vae_encode: %d frames %dx%d (need 4n+1, multiples of 16)", F, W, H);
    const int T = sv2_vae_latent_frames(F), Hl = H / 8, Wl = W / 8;
    const std::vector<Tile> ts = tiles(H, W, ETILE, EOV);
    work_alloc(v, std::min(ETILE, H), std::min(ETILE, W), 128, std::min(ETILE, H) / 8, std::min(ETILE, W) / 8);
    float* cnt = (float*)dmalloc((size_t)Hl * Wl * 4);
    CK(cudaMemsetAsync(cnt, 0, (size_t)Hl * Wl * 4, st));
    CK(cudaMemsetAsync(out, 0, (size_t)16 * T * Hl * Wl * 4, st));
    for (size_t ti = 0; ti < ts.size(); ti++) {
        const Tile& t = ts[ti];
        int lh, lw;
        tile_w(v, Tile{t.y0 / 8, t.y1 / 8, t.x0 / 8, t.x1 / 8}, Hl, Wl, 1, EOV / 8, lh, lw);
        k_cnt<<<(unsigned)nblk((size_t)lh * lw, 256), 256, 0, st>>>(v->wy, v->wx, lh, lw, t.y0 / 8, t.x0 / 8, Wl, cnt);
        const int th = t.y1 - t.y0, tw = t.x1 - t.x0;
        for (int f0 = 0, t0 = 0, s = 0; f0 < F; s++) {
            const int nf = s == 0 ? std::min(F, 1 + 4 * SLICE) : std::min(F - f0, 4 * SLICE);
            v->tapping = v->tapdir && ti == 0 && s == 0;
            k_px_in<<<nblk((size_t)nf * th * tw * 3, 256), 256, 0, st>>>(px, H, W, f0, nf, t.y0, t.x0, th, tw, v->buf[0]);
            const Act z = enc_pass(v, Act{v->buf[0], nf, th, tw, 3}, s == 0);
            k_enc_acc<<<nblk((size_t)16 * z.T * lh * lw, 256), 256, 0, st>>>(z.p, z.T, lh, lw, v->wy, v->wx, T, Hl, Wl, t0,
                                                                            t.y0 / 8, t.x0 / 8, out);
            f0 += nf, t0 += z.T;
        }
    }
    k_enc_fin<<<nblk((size_t)16 * T * Hl * Wl, 256), 256, 0, st>>>(out, cnt, Hl * Wl, (size_t)16 * T * Hl * Wl, SCALE);
    dfree(cnt);
    work_free(v, "encode");
    CK(cudaGetLastError());
    return T;
}

// lat [16][T][lh][lw] f32 device (scaled, what the sampler returns) -> out [F][8 lh][8 lw][3] f32 device in 0..1,
// F = sv2_vae_frames(T): exactly what VAEDecodeTiled [512, 128, 64, 8] returns. Returns F.
int sv2_vae_decode(void* h, const float* lat, int T, int lh, int lw, float* out) {
    Vae* v = (Vae*)h;
    const cudaStream_t st = v->st;
    if (T < 1) DIE("sv2_vae_decode: %d latents", T);
    const int F = sv2_vae_frames(T), H = 8 * lh, W = 8 * lw;
    const std::vector<Tile> ts = tiles(lh, lw, DTILE, DOV);
    work_alloc(v, 8 * std::min(DTILE, lh), 8 * std::min(DTILE, lw), 256, std::min(DTILE, lh), std::min(DTILE, lw));
    float* cnt = (float*)dmalloc((size_t)H * W * 4);
    CK(cudaMemsetAsync(cnt, 0, (size_t)H * W * 4, st));
    CK(cudaMemsetAsync(out, 0, (size_t)F * H * W * 3 * 4, st));
    const float inv = 1.f / SCALE;
    for (size_t ti = 0; ti < ts.size(); ti++) {
        const Tile& t = ts[ti];
        int ph, pw;
        tile_w(v, t, lh, lw, 8, 8 * DOV, ph, pw);
        k_cnt<<<(unsigned)nblk((size_t)ph * pw, 256), 256, 0, st>>>(v->wy, v->wx, ph, pw, 8 * t.y0, 8 * t.x0, W, cnt);
        const int th = t.y1 - t.y0, tw = t.x1 - t.x0;
        for (int t0 = 0, f0 = 0, s = 0; t0 < T; s++) {
            const int nt = s == 0 ? std::min(T, 1 + SLICE) : std::min(T - t0, SLICE);
            v->tapping = v->tapdir && ti == 0 && s == 0;
            k_lat_in<<<nblk((size_t)nt * th * tw * 16, 256), 256, 0, st>>>(lat, T, lh, lw, t0, nt, t.y0, t.x0, th, tw, inv,
                                                                           v->buf[0]);
            const Act y = dec_pass(v, Act{v->buf[0], nt, th, tw, 16}, s == 0);
            k_dec_acc<<<nblk((size_t)y.T * ph * pw * 3, 256), 256, 0, st>>>(y.p, y.T, ph, pw, v->wy, v->wx, H, W, f0,
                                                                            8 * t.y0, 8 * t.x0, out);
            t0 += nt, f0 += y.T;
        }
    }
    k_dec_fin<<<nblk((size_t)F * H * W * 3, 256), 256, 0, st>>>(out, cnt, H * W, (size_t)F * H * W * 3);
    dfree(cnt);
    work_free(v, "decode");
    CK(cudaGetLastError());
    return F;
}

void sv2_vae_free(void* h) {
    Vae* v = (Vae*)h;
    cudaStreamSynchronize(v->st);
    auto fc = [](Conv& c) { dfree(c.w), dfree(c.b), dfree(c.cache[0]), dfree(c.cache[1]); };
    auto fn = [](Norm& n) { dfree(n.w), dfree(n.b); };
    auto fr = [&](Res& r) { fn(r.n1), fn(r.n2), fc(r.c1), fc(r.c2), fc(r.sc); };
    auto fm = [&](Mid& m) {
        fr(m.r0), fr(m.r1), fn(m.a.n);
        for (void* p : {(void*)m.a.wq, (void*)m.a.wk, (void*)m.a.wv, (void*)m.a.wo, (void*)m.a.bq, (void*)m.a.bk,
                        (void*)m.a.bv, (void*)m.a.bo})
            dfree(p);
    };
    fc(v->ecin), fc(v->ecout), fc(v->dcin), fc(v->dcout);
    for (Conv& c : v->edown) fc(c);
    for (auto& l : v->er)
        for (Res& r : l) fr(r);
    for (auto& l : v->dr)
        for (Res& r : l) fr(r);
    fm(v->em), fm(v->dm), fn(v->eno), fn(v->dno);
    for (Up& u : v->du) fc(u.e), fc(u.conv);
    lt_free(v->lt);
    cudaStreamDestroy(v->st);
    delete v;
}
}  // extern "C"
