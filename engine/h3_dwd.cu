// shutter engine — pi-H3 DWD head, C API: the last DiT step's final-block hidden states of the target video rows ->
// pixels, in place of the video VAE decode. Ported from comfyui-piflow 1.4.1 (modules/architectures/minimax_h3_dwd.py,
// sol_h3.py) on ComfyUI e9afbccc2 (comfy/ldm/minimax/vae.py NonCausalViT3DDecoder: geometry and layer loop).
//  1. dwd_norm_out: (scale, shift) = bf16(Linear(silu(t_emb row))) in fp32 (scale first);
//     x = bf16(bf16(bf16(LayerNorm(h)) * bf16(1 + scale)) + shift), LayerNorm without affine, eps 1e-5.
//  2. feature_expand 5376 -> 2 x 2 x 2048 (bf16 weights, fp32 accumulate + bias, one rounding): a DiT token (32 px)
//     becomes 2 x 2 Sol tokens (16 px) on the (T, lh, lw) latent grid. Rows are block-major (blocks of 5 latents):
//     a block's tokens in (t, y, x) order, then its 5 global rows (4 register tokens + a zero row).
//  3. 8 refiner layers = the H3 VAE decoder TransformerBlock (h3_vae.cu kernels: rms_norm + ConvRot int8, int8 GEMMs,
//     per-head RMS + split-half rope 48/64, gated SiLU FFN, scale1/2 residuals) with local attention:
//     16 x 16 spatial x 5 latent windows, odd layers shifted by 8 / 2 and clipped at the edges (no wrap); keys = the
//     window's tokens + the global rows of the 1-2 blocks it spans; global rows attend to their whole block. Rope ids are
//     window-local in [-1, 1] (fp32 tables, then bf16), global rows 0. Exact softmax (ComfyUI production runs the
//     windows with comfy_kitchen int8 attention: PSNR 59.8 dB from this exact path on the golden clip).
//     ComfyUI streams each layer over blocks with a ring of 2 q/k/v slots; the values are the same as doing the whole
//     clip per layer, which is what this does (row chunks for the linears, one launch for all windows).
//  4. per block: norm_out (LayerNorm affine) -> first_frame_proj_out on its first latent (1 frame) and proj_out on the
//     rest (4 frames each), bf16 weights, outputs rounded to bf16.
//  5. colour (fp32, op by op): clamp(p * std + mean, -1, 1); fp16((p + 1) * 0.5) -> [F, H, W, 3],
//     F = sum over blocks of 1 + 4 (n - 1) = 4 T - 3 ceil(T / 5).
// Weights are the dwd_* tensors as ComfyUI runs them: F16 params cast to bf16 (biases, norms, scales, register tokens,
// dense weights), int8 weights + fp32 row scales as stored, dwd_norm_out.linear in fp32.
#include "common.cuh"
#include "st.h"

#include <cuda_fp16.h>

#include <algorithm>
#include <cmath>
#include <string>
#include <utility>
#include <vector>

// h3_vae.cu (decoder-block kernels, same numbers as the VAE)
void h3vit_rq(cudaStream_t st, int mode, const void* in, int M, int K, const float* p0, int8_t* q, float* qs);
void h3vit_qkv(cudaStream_t st, const bf16* y, const float2* rope, int L, int M, bf16* Q, bf16* K, bf16* V);
void h3vit_ln(cudaStream_t st, const bf16* x, int G, int L, int L0, const float* w, const float* b, bf16* out);

namespace {
constexpr int DIT = 5376, D = 2048, NMOD = 2 * DIT, TB = 5, NG = 5, NL = 8;
constexpr int CH = 16384;   // Sol rows per linear chunk (FFN 16384 wide: 512 MB of bf16 per chunk)
constexpr int FCH = 8192;   // DiT rows per expand chunk
enum { TAP_MOD = 0, TAP_EXP = 1, TAP_L0 = 2, TAP_NORM = 10, TAP_PRE = 11, NTAP = 12 };
const float COLOR_MEAN[3] = {-0.03f, -0.088f, -0.188f}, COLOR_STD[3] = {0.458f, 0.448f, 0.45f};

inline size_t nblk(size_t n, int t) { return (n + t - 1) / t; }
__device__ __forceinline__ float rd(float v) { return __bfloat162float(__float2bfloat16(v)); }
__device__ __forceinline__ float ld(const bf16* p) { return __bfloat162float(*p); }
inline float hround(float v) { return __bfloat162float(__float2bfloat16(v)); }

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

// ---------------- front ----------------
// t [8] fp32 (the video stream's t_emb row) -> mod [2][5376] bf16 = (bf16(1 + scale), shift), (scale, shift) =
// bf16(silu(t) . w[n] + b[n]) in fp32
__global__ void k_tmod(const float* __restrict__ w, const float* __restrict__ b, const float* __restrict__ t,
                       bf16* __restrict__ mod) {
    const int n = blockIdx.x * blockDim.x + threadIdx.x;
    if (n >= NMOD) return;
    float a = 0.f;
    for (int k = 0; k < 8; k++) a = fmaf(t[k] / (1.f + expf(-t[k])), w[n * 8 + k], a);
    const float v = rd(a + b[n]);
    mod[n] = __float2bfloat16(n < DIT ? 1.f + v : v);
}

// DiT rows of 5376: out = bf16(bf16(bf16(layer_norm(h)) * mod[j]) + mod[5376 + j])
__global__ void __launch_bounds__(256) k_modln(const bf16* __restrict__ h, const bf16* __restrict__ mod,
                                               bf16* __restrict__ out) {
    __shared__ float red[32];
    const bf16* xr = h + (size_t)blockIdx.x * DIT;
    float s = 0.f;
    for (int j = threadIdx.x; j < DIT; j += blockDim.x) s += ld(xr + j);
    const float mean = block_sum(s, red) / DIT;
    float s2 = 0.f;
    for (int j = threadIdx.x; j < DIT; j += blockDim.x) {
        const float d = ld(xr + j) - mean;
        s2 += d * d;
    }
    const float rstd = rsqrtf(block_sum(s2, red) / DIT + 1e-5f);
    bf16* o = out + (size_t)blockIdx.x * DIT;
    for (int j = threadIdx.x; j < DIT; j += blockDim.x) {
        const float v = rd(rd(__fmul_rn(ld(xr + j) - mean, rstd)) * ld(mod + j));
        o[j] = __float2bfloat16(v + ld(mod + DIT + j));
    }
}

// Sol row of latent (t, y, x) on a W-wide, HW-sized grid: blocks of 5 latents, each followed by its 5 global rows
__host__ __device__ __forceinline__ int srow(int t, int y, int x, int W, int HW) { return t * HW + t / TB * NG + y * W + x; }

// feature_expand accumulators acc [m][8192] (DiT rows r0..r0+m of the (t, h, w) grid) + bias -> Sol rows of x:
// channel (eh, ew, d) of DiT token (t, yy, xx) = Sol token (t, 2 yy + eh, 2 xx + ew)
__global__ void k_expand(const float* __restrict__ acc, const float* __restrict__ b, int r0, int m, int h, int w,
                         bf16* __restrict__ x) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)m * 4 * D) return;
    const int c = (int)(i % (4 * D)), dr = r0 + (int)(i / (4 * D));
    const int t = dr / (h * w), yy = (dr / w) % h, xx = dr % w;
    const int row = srow(t, 2 * yy + (c >> 12), 2 * xx + ((c >> 11) & 1), 2 * w, 4 * h * w);
    x[(size_t)row * D + (c & (D - 1))] = __float2bfloat16(acc[i] + b[c]);
}

// the 5 global rows of every block: register tokens (bf16) and a zero row
__global__ void k_glob(const float* __restrict__ reg, int T, int HW, int nb, bf16* __restrict__ x) {
    const int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= nb * NG * D) return;
    const int d = i % D, j = (i / D) % NG, b = i / (D * NG);
    const size_t row = (size_t)std::min((b + 1) * TB, T) * HW + b * NG + j;
    x[row * D + d] = __float2bfloat16(j < 4 ? reg[j * D + d] : 0.f);
}

// ---------------- windowed flash attention (head dim 64, 32 heads, rows of 2048) ----------------
// k_flash64 of h3_vae.cu with row lists: CTA (tile, head), tile = (qoff, qn, koff, kn): queries qidx[qoff, qoff + qn)
// (qn <= 64), keys / values kidx[koff, koff + kn); the output goes to the query rows. O may alias Q: each (query row,
// head) belongs to one CTA, which reads Q before it writes O.
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

__global__ void __launch_bounds__(128) k_wattn(const bf16* Q, const bf16* __restrict__ K, const bf16* __restrict__ V,
                                               bf16* O, const int4* __restrict__ tiles, const int* __restrict__ qidx,
                                               const int* __restrict__ kidx, float sl2) {
    __shared__ __align__(128) bf16 sQ[64 * 64], sK[64 * 64], sV[64 * 64];
    const int4 tl = tiles[blockIdx.x];
    const int* qi = qidx + tl.x;
    const int* ki = kidx + tl.z;
    const int qn = tl.y, kn = tl.w;
    const int tid = threadIdx.x, warp = tid >> 5, lane = tid & 31, hc = blockIdx.y * 64;
    for (int c = tid; c < 64 * 8; c += 128) {
        const int r = c >> 3, ch = c & 7;
        const bool ok = r < qn;
        cp16(&sQ[swz(r, ch * 8)], Q + (size_t)(ok ? qi[r] : 0) * D + hc + ch * 8, ok ? 16 : 0);
    }
    cp_wait();
    __syncthreads();
    const int g = lane >> 2, t = lane & 3;
    const int r0 = warp * 16 + g, r1 = r0 + 8;
    uint32_t qf[4][4];
#pragma unroll
    for (int kk = 0; kk < 4; kk++) ldm4(qf[kk], &sQ[swz(warp * 16 + (lane & 15), kk * 16 + (lane >> 4) * 8)]);
    float o[8][4];
#pragma unroll
    for (int i = 0; i < 8; i++) o[i][0] = o[i][1] = o[i][2] = o[i][3] = 0.f;
    float m0 = -INFINITY, m1 = -INFINITY, l0 = 0.f, l1 = 0.f;
    for (int kb = 0; kb < kn; kb += 64) {
        __syncthreads();
        for (int c = tid; c < 64 * 8; c += 128) {
            const int r = c >> 3, ch = c & 7;
            const bool ok = kb + r < kn;
            const size_t off = (size_t)(ok ? ki[kb + r] : 0) * D + hc + ch * 8;
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
            s[nt][0] = key < kn ? s[nt][0] * sl2 : -INFINITY;
            s[nt][1] = key + 1 < kn ? s[nt][1] * sl2 : -INFINITY;
            s[nt][2] = key < kn ? s[nt][2] * sl2 : -INFINITY;
            s[nt][3] = key + 1 < kn ? s[nt][3] * sl2 : -INFINITY;
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
        if (r0 < qn) *(uint32_t*)(O + (size_t)qi[r0] * D + col) = pack2(o[dt][0] * i0, o[dt][1] * i0);
        if (r1 < qn) *(uint32_t*)(O + (size_t)qi[r1] * D + col) = pack2(o[dt][2] * i1, o[dt][3] * i1);
    }
}

// ---------------- back ----------------
// projection accumulators [rows][3 pt 256] of latents (tl, y, x) + bias -> bf16 frames pc [f][3][16 H][16 W],
// frame = f0 + tl * pt + p (first_frame_proj_out: pt 1, proj_out: pt 4)
__global__ void k_unpatch(const float* __restrict__ acc, const float* __restrict__ b, int rows, int pt, int H, int W,
                          int f0, bf16* __restrict__ pc) {
    const int N = 3 * pt * 256;
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)rows * N) return;
    const int n = (int)(i % N), r = (int)(i / N);
    const int xx = r % W, yy = (r / W) % H, tl = r / (W * H);
    const int c = n / (pt * 256), p = (n >> 8) % pt, py = (n >> 4) & 15, px = n & 15;
    const size_t Hp = (size_t)H * 16, Wp = (size_t)W * 16;
    pc[(((size_t)(f0 + tl * pt + p) * 3 + c) * Hp + yy * 16 + py) * Wp + xx * 16 + px] = __float2bfloat16(acc[i] + b[n]);
}

// pc [nf][3][Hp][Wp] bf16 -> out [nf][Hp][Wp][3] fp16 = fp16((clamp(p * std + mean, -1, 1) + 1) * 0.5), fp32 op by op
__global__ void k_color(const bf16* __restrict__ pc, int nf, int HW, float3 sd, float3 mn, __half* __restrict__ out) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)nf * HW * 3) return;
    const int c = (int)(i % 3);
    const size_t p = (i / 3) % HW, f = i / 3 / HW;
    float v = ld(pc + (f * 3 + c) * HW + p);
    v = __fadd_rn(__fmul_rn(v, c == 0 ? sd.x : c == 1 ? sd.y : sd.z), c == 0 ? mn.x : c == 1 ? mn.y : mn.z);
    v = fminf(fmaxf(v, -1.f), 1.f);
    out[i] = __float2half_rn(__fmul_rn(__fadd_rn(v, 1.f), 0.5f));
}

// ---------------- model ----------------
struct Lin {  // int8 ConvRot linear
    int8_t* w;
    float *ws, *b;
};
struct Blk {
    Lin qkv, out, w1, w2;
    float *n1, *n2, *s1, *s2;
};
struct Win {  // attention jobs of one layer parity
    int4* tiles = nullptr;
    int *qi = nullptr, *ki = nullptr;
    int ntiles = 0;
    size_t nq = 0, nk = 0;
};
struct Dwd {
    cudaStream_t st;
    Lt* lt;
    StFile f;
    size_t bytes = 0;
    float *tw, *tb, *feb, *reg, *no_w, *no_b, *pb, *fb;
    bf16 *few, *pw, *fw;
    std::vector<Blk> blk;
    // per shape
    int key[3] = {0, 0, 0}, R = 0;
    bf16 *x = nullptr, *Q = nullptr, *K = nullptr, *V = nullptr, *mod = nullptr;
    float2* rope[2] = {nullptr, nullptr};
    Win win[2];
    // work: ws holds the FFN chunk (y) / the expand chunk / one block's back end
    size_t wcap = 0;
    char* ws = nullptr;
    int8_t* q8 = nullptr;
    float *qs = nullptr, *qs2 = nullptr;
    void* tap_out[NTAP] = {};
    void* tap_in[NTAP] = {};
};

void* dalloc(Dwd* d, size_t n) {
    void* p;
    CK(cudaMalloc(&p, std::max<size_t>(n, 16)));
    d->bytes += std::max<size_t>(n, 16);
    return p;
}
void dfree(Dwd* d, void* p, size_t n) {
    if (!p) return;
    CK(cudaFree(p));
    d->bytes -= std::max<size_t>(n, 16);
}
template <typename T>
T* up(Dwd* d, const std::vector<T>& h) {
    T* p = (T*)dalloc(d, h.size() * sizeof(T));
    CK(cudaMemcpy(p, h.data(), h.size() * sizeof(T), cudaMemcpyHostToDevice));
    return p;
}
std::vector<float> hf(Dwd* d, const std::string& name, size_t n, bool round) {  // F16 / F32 -> float (bf16-rounded)
    const StTensor& t = d->f.get(name);
    std::vector<float> o(n);
    const uint8_t* p = d->f.ptr(t);
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
float* upf(Dwd* d, const std::string& name, size_t n) { return up(d, hf(d, name, n, true)); }
bf16* upw(Dwd* d, const std::string& name, size_t n) {
    std::vector<float> w = hf(d, name, n, false);
    std::vector<bf16> h(n);
    for (size_t i = 0; i < n; i++) h[i] = __float2bfloat16(w[i]);
    return up(d, h);
}
Lin load_lin(Dwd* d, const std::string& p, int n, int k) {
    const StTensor& t = d->f.get(p + ".weight");
    if (t.dtype != "I8" || t.bytes() != (size_t)n * k) DIE("%s.weight: expected I8 [%d, %d]", p.c_str(), n, k);
    Lin l;
    l.w = (int8_t*)dalloc(d, (size_t)n * k);
    CK(cudaMemcpy(l.w, d->f.ptr(t), (size_t)n * k, cudaMemcpyHostToDevice));
    l.ws = up(d, hf(d, p + ".weight_scale", n, false));
    l.b = upf(d, p + ".bias", n);
    return l;
}

std::vector<std::pair<int, int>> spans(int n, int shift, int size) {  // vae.py _build_geometry spans()
    std::vector<std::pair<int, int>> r;
    for (int i = 0; i < (n + shift + size - 1) / size; i++) {
        const int lo = std::max(i * size - shift, 0), hi = std::min((i + 1) * size - shift, n);
        if (hi > lo) r.push_back({lo, hi});
    }
    return r;
}

void free_shape(Dwd* d) {
    const size_t R = d->R, rb = R * D * 2;
    for (bf16* p : {d->x, d->Q, d->K, d->V}) dfree(d, p, rb);
    for (int s = 0; s < 2; s++) {
        Win& w = d->win[s];
        dfree(d, d->rope[s], R * 24 * sizeof(float2));
        dfree(d, w.tiles, w.ntiles * sizeof(int4)), dfree(d, w.qi, w.nq * 4), dfree(d, w.ki, w.nk * 4);
        w = Win();
        d->rope[s] = nullptr;
    }
    d->x = d->Q = d->K = d->V = nullptr;
    d->R = 0;
}

// geometry of one latent shape: rope tables and attention jobs of both layer parities, row buffers, work size
void prepare(Dwd* d, int T, int H, int W) {
    if (d->key[0] == T && d->key[1] == H && d->key[2] == W) return;
    free_shape(d);
    const int HW = H * W, nb = (T + TB - 1) / TB, R = T * HW + nb * NG;
    auto start = [&](int b) { return std::min(b * TB, T) * HW + b * NG; };
    float inv[8];   // RotaryEmbeddingND(48, 100).inv_freq: an fp32 buffer (the DWD module is not cast)
    for (int k = 0; k < 8; k++) inv[k] = 1.f / powf(100.f, 0.125f * k);
    const float sc = (float)6.283185307179586;
    for (int s = 0; s < 2; s++) {
        const int ss = s ? 8 : 0, st = s ? 2 : 0;
        // window-local rope ids ((i + shift) % size + 0.5) / size * 2 - 1 in fp32, bf16 like ids.to(x.dtype)
        auto id = [](int i, int o, int m) { return hround(((float)((i + o) % m) + 0.5f) / (float)m * 2.f - 1.f); };
        std::vector<float2> tab((size_t)R * 24, make_float2(1.f, 0.f));
        for (int t = 0; t < T; t++)
            for (int y = 0; y < H; y++)
                for (int x = 0; x < W; x++) {
                    const float ids[3] = {id(t, st, TB), id(y, ss, 16), id(x, ss, 16)};
                    float2* e = &tab[(size_t)srow(t, y, x, W, HW) * 24];
                    for (int a = 0; a < 3; a++)
                        for (int k = 0; k < 8; k++) {
                            const float ang = sc * ids[a] * inv[k];
                            e[a * 8 + k] = make_float2(hround(cosf(ang)), hround(sinf(ang)));
                        }
                }
        d->rope[s] = up(d, tab);
        // jobs: global rows first (longest key lists), then the windows
        std::vector<int> qi, ki;
        std::vector<int4> tiles;
        auto add = [&](const std::vector<int>& q, const std::vector<int>& k) {
            const int qo = (int)qi.size(), ko = (int)ki.size(), n = (int)q.size();
            qi.insert(qi.end(), q.begin(), q.end());
            ki.insert(ki.end(), k.begin(), k.end());
            for (int q0 = 0; q0 < n; q0 += 64) tiles.push_back(make_int4(qo + q0, std::min(64, n - q0), ko, (int)k.size()));
        };
        std::vector<int> q, k;
        for (int b = 0; b < nb; b++) {
            q.clear(), k.clear();
            for (int j = 0; j < NG; j++) q.push_back(start(b + 1) - NG + j);
            for (int r = start(b); r < start(b + 1); r++) k.push_back(r);
            add(q, k);
        }
        for (auto [t0, t1] : spans(T, st, TB)) {
            const int b0 = t0 / TB, b1 = (t1 - 1) / TB;
            for (auto [y0, y1] : spans(H, ss, 16))
                for (auto [x0, x1] : spans(W, ss, 16)) {
                    q.clear();
                    for (int t = t0; t < t1; t++)
                        for (int y = y0; y < y1; y++)
                            for (int x = x0; x < x1; x++) q.push_back(srow(t, y, x, W, HW));
                    k = q;
                    for (int b = b0; b <= b1; b++)
                        for (int j = 0; j < NG; j++) k.push_back(start(b + 1) - NG + j);
                    add(q, k);
                }
        }
        if ((int)qi.size() != R) DIE("h3_dwd: %zu query rows, expected %d", qi.size(), R);
        Win& w = d->win[s];
        w.tiles = up(d, tiles), w.qi = up(d, qi), w.ki = up(d, ki);
        w.ntiles = (int)tiles.size(), w.nq = qi.size(), w.nk = ki.size();
    }
    d->R = R;
    const size_t rb = (size_t)R * D * 2;
    d->x = (bf16*)dalloc(d, rb), d->Q = (bf16*)dalloc(d, rb), d->K = (bf16*)dalloc(d, rb), d->V = (bf16*)dalloc(d, rb);
    // work: FFN chunk [CH][16384] bf16 | expand chunk (ln [FCH][5376] bf16 + acc [FCH][8192] f32) |
    // one block's back end (ln [5 HW][2048] bf16 + acc [4 HW][3072] f32 + frames [17][3][256 HW] bf16)
    const size_t need = std::max({(size_t)CH * 16384 * 2, (size_t)FCH * (DIT * 2 + 4 * D * 4),
                                  (size_t)HW * (TB * D * 2 + 4 * 3072 * 4 + 17 * 3 * 256 * 2)});
    if (need > d->wcap) {
        dfree(d, d->ws, d->wcap);
        d->ws = (char*)dalloc(d, need), d->wcap = need;
    }
    d->key[0] = T, d->key[1] = H, d->key[2] = W;
}

void tap_out(Dwd* d, int i, const void* src, size_t off, size_t n) {
    if (d->tap_out[i]) CK(cudaMemcpyAsync((char*)d->tap_out[i] + off, src, n, cudaMemcpyDeviceToDevice, d->st));
}
void tap_in(Dwd* d, int i, void* dst, size_t off, size_t n) {
    if (d->tap_in[i]) CK(cudaMemcpyAsync(dst, (const char*)d->tap_in[i] + off, n, cudaMemcpyDeviceToDevice, d->st));
}

// H3DWD_PROF=1: GPU time per phase on stderr
struct Prof {
    bool on = getenv("H3DWD_PROF") != nullptr;
    std::vector<std::pair<const char*, cudaEvent_t>> ev;
    void mark(cudaStream_t st, const char* name) {
        if (!on) return;
        cudaEvent_t e;
        CK(cudaEventCreate(&e));
        CK(cudaEventRecord(e, st));
        ev.push_back({name, e});
    }
    void report() {
        if (!on || ev.size() < 2) return;
        CK(cudaEventSynchronize(ev.back().second));
        fprintf(stderr, "[h3_dwd prof]");
        float a = 0.f, l = 0.f;
        for (size_t i = 1; i < ev.size(); i++) {
            float ms;
            CK(cudaEventElapsedTime(&ms, ev[i - 1].second, ev[i].second));
            if (ev[i].first[0] == 'a') a += ms;
            else if (ev[i].first[0] == 'l') l += ms;
            else fprintf(stderr, " %s %.0f", ev[i].first, ms);
        }
        fprintf(stderr, " attn %.0f linear %.0f ms\n", a, l);
        for (auto& x : ev) cudaEventDestroy(x.second);
        ev.clear();
    }
};
Prof prof;
}  // namespace

extern "C" {

void* h3_dwd_new(const char* path) {
    Dwd* d = new Dwd();
    if (!d->f.open(path)) DIE("cannot open %s", path);
    CK(cudaStreamCreate(&d->st));   // blocking on purpose (common.cuh)
    d->lt = lt_new(d->st);
    const std::string r = "dwd_pixel_refiner.";
    d->tw = up(d, hf(d, "dwd_norm_out.linear.weight", (size_t)NMOD * 8, false));   // fp32 compute
    d->tb = up(d, hf(d, "dwd_norm_out.linear.bias", NMOD, false));
    d->few = upw(d, r + "feature_expand.weight", (size_t)4 * D * DIT), d->feb = upf(d, r + "feature_expand.bias", 4 * D);
    for (int i = 0; i < NL; i++) {
        const std::string p = r + "blocks." + std::to_string(i) + ".";
        Blk b;
        b.qkv = load_lin(d, p + "attn.to_qkv", 3 * D, D);
        b.out = load_lin(d, p + "attn.to_out", D, D);
        b.w1 = load_lin(d, p + "ff.w1", 8 * D, D);
        b.w2 = load_lin(d, p + "ff.w2", D, 4 * D);
        b.n1 = upf(d, p + "norm1.weight", D), b.n2 = upf(d, p + "norm2.weight", D);
        b.s1 = upf(d, p + "scale1", D), b.s2 = upf(d, p + "scale2", D);
        d->blk.push_back(b);
    }
    d->reg = upf(d, r + "register_tokens", 4 * D);
    d->no_w = upf(d, r + "norm_out.weight", D), d->no_b = upf(d, r + "norm_out.bias", D);
    d->pw = upw(d, r + "proj_out.weight", (size_t)3072 * D), d->pb = upf(d, r + "proj_out.bias", 3072);
    d->fw = upw(d, r + "first_frame_proj_out.weight", (size_t)768 * D), d->fb = upf(d, r + "first_frame_proj_out.bias", 768);
    d->f.close();
    d->mod = (bf16*)dalloc(d, NMOD * 2);
    d->q8 = (int8_t*)dalloc(d, (size_t)CH * 8192);
    d->qs = (float*)dalloc(d, CH * 4), d->qs2 = (float*)dalloc(d, CH * 4);
    CK(cudaDeviceSynchronize());
    return d;
}

// frames h3_dwd_decode writes for T latent frames: 1 + 4 (n - 1) per block of n <= 5 latents
int h3_dwd_frames(int T) { return 4 * T - 3 * ((T + TB - 1) / TB); }

// device bytes the head holds (weights + the last shape's buffers)
long long h3_dwd_bytes(void* h) { return (long long)((Dwd*)h)->bytes; }

// gate: NTAP device buffers each (null = off) — out: stage outputs are copied there; in: the stage's output is replaced
// by it before the next stage reads it (out still gets the engine's own). Stages: 0 dwd_norm_out [N, 5376], 1 expanded rows [R, 2048], 2..9 refiner layer
// outputs [R, 2048], 10 norm_out [T HW, 2048], 11 pre-colour projections [F, 3, H, W] (bf16, golden dump layout).
void h3_dwd_taps(void* h, void* const* out, void* const* in) {
    Dwd* d = (Dwd*)h;
    for (int i = 0; i < NTAP; i++) d->tap_out[i] = out ? out[i] : nullptr, d->tap_in[i] = in ? in[i] : nullptr;
}

// hid [T (lh/2) (lw/2), 5376] bf16 device: the last DiT block's output rows of the target video (DiT token order t, y, x),
// temb [8] fp32 device: the t_emb row of the video stream at that step, (T, lh, lw) = video latent grid
// -> out [F, 16 lh, 16 lw, 3] fp16 device in [0, 1], F = h3_dwd_frames(T). Returns F.
int h3_dwd_decode(void* h, const void* hid, const float* temb, int T, int lh, int lw, void* out) {
    Dwd* d = (Dwd*)h;
    const cudaStream_t st = d->st;
    if (T < 1 || lh % 2 || lw % 2) DIE("h3_dwd_decode: bad latent grid %d x %d x %d", T, lh, lw);
    prepare(d, T, lh, lw);
    const int HW = lh * lw, nb = (T + TB - 1) / TB, gh = lh / 2, gw = lw / 2, N = T * gh * gw, R = d->R;
    prof.mark(st, "start");
    // 1-2: modulated norm, feature_expand, global rows
    k_tmod<<<nblk(NMOD, 256), 256, 0, st>>>(d->tw, d->tb, temb, d->mod);
    bf16* ln = (bf16*)d->ws;
    float* facc = (float*)(d->ws + (size_t)FCH * DIT * 2);
    for (int r0 = 0; r0 < N; r0 += FCH) {
        const int m = std::min(FCH, N - r0);
        k_modln<<<m, 256, 0, st>>>((const bf16*)hid + (size_t)r0 * DIT, d->mod, ln);
        tap_out(d, TAP_MOD, ln, (size_t)r0 * DIT * 2, (size_t)m * DIT * 2);
        tap_in(d, TAP_MOD, ln, (size_t)r0 * DIT * 2, (size_t)m * DIT * 2);
        gemm_bf16_f32(d->lt, m, 4 * D, DIT, ln, d->few, facc);
        k_expand<<<nblk((size_t)m * 4 * D, 256), 256, 0, st>>>(facc, d->feb, r0, m, gh, gw, d->x);
    }
    k_glob<<<nblk(nb * NG * D, 256), 256, 0, st>>>(d->reg, T, HW, nb, d->x);
    tap_out(d, TAP_EXP, d->x, 0, (size_t)R * D * 2);
    prof.mark(st, "expand");
    // 3: refiner layers
    const float sl2 = 0.125f * 1.4426950408889634f;
    bf16* y = (bf16*)d->ws;
    for (int li = 0; li < NL; li++) {
        const Blk& b = d->blk[li];
        const int s = li & 1;
        tap_in(d, TAP_EXP + li, d->x, 0, (size_t)R * D * 2);
        for (int c0 = 0; c0 < R; c0 += CH) {
            const int m = std::min(CH, R - c0);
            const size_t o = (size_t)c0 * D;
            h3vit_rq(st, 1, d->x + o, m, D, b.n1, d->q8, d->qs);
            gemm_i8_epi(st, DQ_F32B, m, 3 * D, D, d->q8, b.qkv.w, d->qs, b.qkv.ws, b.qkv.b, nullptr, y);
            h3vit_qkv(st, y, d->rope[s] + (size_t)c0 * 24, m, m, d->Q + o, d->K + o, d->V + o);
        }
        prof.mark(st, "lin");
        k_wattn<<<dim3(d->win[s].ntiles, 32), 128, 0, st>>>(d->Q, d->K, d->V, d->Q, d->win[s].tiles, d->win[s].qi,
                                                            d->win[s].ki, sl2);
        prof.mark(st, "attn");
        for (int c0 = 0; c0 < R; c0 += CH) {
            const int m = std::min(CH, R - c0);
            const size_t o = (size_t)c0 * D;
            h3vit_rq(st, 0, d->Q + o, m, D, nullptr, d->q8, d->qs);
            gemm_i8_epi(st, DQ_RES, m, D, D, d->q8, b.out.w, d->qs, b.out.ws, b.out.b, b.s1, d->x + o);   // x += s1 * out
            h3vit_rq(st, 1, d->x + o, m, D, b.n2, d->q8, d->qs);
            gemm_i8_epi(st, DQ_F32B, m, 8 * D, D, d->q8, b.w1.w, d->qs, b.w1.ws, b.w1.b, nullptr, y);
            h3vit_rq(st, 2, y, m, 4 * D, nullptr, d->q8, d->qs2);
            gemm_i8_epi(st, DQ_RES, m, D, 4 * D, d->q8, b.w2.w, d->qs2, b.w2.ws, b.w2.b, b.s2, d->x + o);  // x += s2 * w2
        }
        prof.mark(st, "lin");
        tap_out(d, TAP_L0 + li, d->x, 0, (size_t)R * D * 2);
    }
    tap_in(d, TAP_NORM - 1, d->x, 0, (size_t)R * D * 2);
    // 4-5: per block norm_out, projections, colour
    const int Hp = lh * 16, Wp = lw * 16;
    const size_t fpx = (size_t)3 * Hp * Wp;
    bf16* lb = (bf16*)d->ws;
    float* pacc = (float*)(d->ws + (size_t)TB * HW * D * 2);
    bf16* pc = (bf16*)((char*)pacc + (size_t)4 * HW * 3072 * 4);
    const float3 sd = make_float3(COLOR_STD[0], COLOR_STD[1], COLOR_STD[2]);
    const float3 mn = make_float3(COLOR_MEAN[0], COLOR_MEAN[1], COLOR_MEAN[2]);
    int f0 = 0;
    for (int b = 0; b < nb; b++) {
        const int n = std::min(TB, T - b * TB), ntok = n * HW, nf = 1 + 4 * (n - 1);
        const size_t tok0 = (size_t)b * TB * HW;
        h3vit_ln(st, d->x + (size_t)(tok0 + b * NG) * D, 1, ntok, ntok, d->no_w, d->no_b, lb);
        tap_out(d, TAP_NORM, lb, tok0 * D * 2, (size_t)ntok * D * 2);
        tap_in(d, TAP_NORM, lb, tok0 * D * 2, (size_t)ntok * D * 2);
        gemm_bf16_f32(d->lt, HW, 768, D, lb, d->fw, pacc);
        k_unpatch<<<nblk((size_t)HW * 768, 256), 256, 0, st>>>(pacc, d->fb, HW, 1, lh, lw, 0, pc);
        if (n > 1) {
            gemm_bf16_f32(d->lt, (n - 1) * HW, 3072, D, lb + (size_t)HW * D, d->pw, pacc);
            k_unpatch<<<nblk((size_t)(n - 1) * HW * 3072, 256), 256, 0, st>>>(pacc, d->pb, (n - 1) * HW, 4, lh, lw, 1, pc);
        }
        tap_out(d, TAP_PRE, pc, (size_t)f0 * fpx * 2, (size_t)nf * fpx * 2);
        tap_in(d, TAP_PRE, pc, (size_t)f0 * fpx * 2, (size_t)nf * fpx * 2);
        k_color<<<nblk((size_t)nf * fpx, 256), 256, 0, st>>>(pc, nf, Hp * Wp, sd, mn, (__half*)out + (size_t)f0 * fpx);
        f0 += nf;
    }
    prof.mark(st, "back");
    CK(cudaStreamSynchronize(st));
    prof.report();
    CK(cudaGetLastError());   // launches here are unchecked: surface their errors in this call
    return f0;
}

void h3_dwd_free(void* h) {
    Dwd* d = (Dwd*)h;
    cudaStreamSynchronize(d->st);
    free_shape(d);
    for (Blk& b : d->blk) {
        for (Lin* l : {&b.qkv, &b.out, &b.w1, &b.w2}) cudaFree(l->w), cudaFree(l->ws), cudaFree(l->b);
        for (float* p : {b.n1, b.n2, b.s1, b.s2}) cudaFree(p);
    }
    void* ps[] = {d->tw, d->tb, d->feb, d->reg, d->no_w, d->no_b, d->pb, d->fb, d->few, d->pw, d->fw, d->mod, d->ws,
                  d->q8, d->qs, d->qs2};
    for (void* p : ps) cudaFree(p);
    lt_free(d->lt);
    cudaStreamDestroy(d->st);
    delete d;
}
}  // extern "C"
