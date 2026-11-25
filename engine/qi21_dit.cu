// shutter engine — Qwen-Image 2.1 DiT (int8 ConvRot W8A8), C API
// Structure follows ComfyUI comfy/ldm/qwen_image21/model.py: text/reference prefix runs once per job
// (t = 0 modulation, causal), its K/V are kept per block; each step runs the target tokens only.
#include "common.cuh"
#include "st.h"

#include <algorithm>
#include <cmath>
#include <cstring>
#include <string>
#include <vector>

namespace {
constexpr int D = 4096, NB = 32, QKV = 3 * D, FF = 12288;

struct Block {
    int8_t *wqkv, *wo, *wgu, *wd;
    float *sqkv, *so, *sgu, *sd;
    bf16 *nq, *nk;
};

struct Dit {
    cudaStream_t st;
    Lt* lt;
    StFile f;
    bf16 *t1, *t2, *txt_norm, *txt1, *txt2, *img_in, *modw, *nout, *pout;
    Block b[NB];
    // job state
    int P = 0, H = 0, W = 0, HW = 0, cap = 0, kvcap = 0;
    bf16* K[NB] = {};
    bf16* V[NB] = {};
    float *pe_p = nullptr, *pe_t = nullptr;
    int* qlim_p = nullptr;
    // CFG: the negative prompt's prefix waits here, swapped in for the uncond forward (same target size)
    struct Ctx {
        int P = 0, kvcap = 0;
        bf16* K[NB] = {};
        bf16* V[NB] = {};
        float *pe_p = nullptr, *pe_t = nullptr;
        int* qlim_p = nullptr;
    } alt;
    bf16* cfgb = nullptr;  // [3][64*HW]: x in bf16, cond out, uncond out
    int cfgn = 0;
    // work buffers (cap rows)
    bf16 *x = nullptr, *qkv = nullptr, *q = nullptr, *att = nullptr, *gu = nullptr, *tok = nullptr, *ln = nullptr;
    int8_t* aq = nullptr;
    float* as = nullptr;
    int32_t* acc = nullptr;
    // conditioning vectors
    bf16 *temb_in, *temb_h, *temb, *temb_s, *modv, *fs;  // modv [2, 16384]: row 0 target t, row 1 prefix t=0
    // debug taps (device pointers owned by caller)
    bf16 *tap_blocks = nullptr, *tap_prefix = nullptr, *tap_qkv = nullptr, *tap_att = nullptr, *tap_gu = nullptr;
};

void* dmalloc(size_t n) {
    void* p;
    CK(cudaMalloc(&p, n));
    return p;
}

// host (mmap) -> device through a pinned staging buffer: much faster than cudaMemcpy from pageable mmap memory
uint8_t* g_stage = nullptr;
constexpr size_t STAGE = 64u << 20;
void h2d(void* dst, const uint8_t* src, size_t n) {
    for (size_t o = 0; o < n; o += STAGE) {
        const size_t c = std::min(STAGE, n - o);
        memcpy(g_stage, src + o, c);
        CK(cudaMemcpy((uint8_t*)dst + o, g_stage, c, cudaMemcpyHostToDevice));
    }
}

void* upload(Dit* d, const std::string& name, const char* dtype, size_t elems) {
    const StTensor& t = d->f.get(name);
    if (t.dtype != dtype) DIE("%s: dtype %s, expected %s", name.c_str(), t.dtype.c_str(), dtype);
    if (elems && t.bytes() != elems * (t.dtype == "I8" ? 1 : t.dtype == "F32" ? 4 : 2))
        DIE("%s: %llu bytes, expected %zu elems", name.c_str(), (unsigned long long)t.bytes(), elems);
    void* p = dmalloc(t.bytes());
    h2d(p, d->f.ptr(t), t.bytes());
    return p;
}

void upload_into(Dit* d, const std::string& name, void* dst, size_t bytes) {
    const StTensor& t = d->f.get(name);
    if (t.bytes() != bytes) DIE("%s: %llu bytes, expected %zu", name.c_str(), (unsigned long long)t.bytes(), bytes);
    h2d(dst, d->f.ptr(t), bytes);
}

void check_convrot(Dit* d, const std::string& layer) {
    const StTensor& t = d->f.get(layer + ".comfy_quant");
    std::string js((const char*)d->f.ptr(t), t.bytes());
    if (js.find("int8_tensorwise") == std::string::npos || js.find("\"convrot\": true") == std::string::npos ||
        js.find("256") == std::string::npos)
        DIE("%s: unexpected quant config %s", layer.c_str(), js.c_str());
}

// int8 linear: act rows (already quantized in d->aq/d->as) x W -> acc
inline void lin8(Dit* d, int M, int N, int K, const int8_t* w) { gemm_i8(d->lt, M, N, K, d->aq, w, d->acc); }

// sinusoidal embedding as torch does it (fp32), then bf16
void temb_input(float t, bf16* out) {
    const int half = 128;
    const float c = (float)(-std::log(10000.0));
    for (int j = 0; j < half; j++) {
        const float fr = std::exp(c * (float)j / (float)half);
        const float a = (t * 1000.f) * fr;
        out[j] = f2bf(std::cos(a));
        out[half + j] = f2bf(std::sin(a));
    }
}

// modulation for target t and prefix t = 0, plus the final-layer scale for t
void conditioning(Dit* d, float sigma) {
    // pipeline rounding: t = bf16(bf16(sigma * 1000) / 1000)
    const float t = bf2f(f2bf(bf2f(f2bf(sigma * 1000.f)) / 1000.f));
    bf16 h[512];
    temb_input(t, h);
    temb_input(0.f, h + 256);
    CK(cudaMemcpyAsync(d->temb_in, h, sizeof(h), cudaMemcpyHostToDevice, d->st));
    gemm_bf16(d->lt, 2, D, 256, d->temb_in, d->t1, d->temb_h);
    elementwise(d->st, EW_SILU, d->temb_h, d->temb_h, 2 * D);
    gemm_bf16(d->lt, 2, D, D, d->temb_h, d->t2, d->temb);
    elementwise(d->st, EW_SILU, d->temb, d->temb_s, 2 * D);
    gemm_bf16(d->lt, 2, 4 * D, D, d->temb_s, d->modw, d->modv);
    for (int r = 0; r < 2; r++) {  // gates are tanh'd
        elementwise(d->st, EW_TANH, d->modv + r * 4 * D + D, d->modv + r * 4 * D + D, D);
        elementwise(d->st, EW_TANH, d->modv + r * 4 * D + 3 * D, d->modv + r * 4 * D + 3 * D, D);
    }
    gemm_bf16(d->lt, 1, D, D, d->temb_s, d->nout, d->fs);  // LastLayer: linear(silu(temb[target]))
}

void ensure(Dit* d, int rows, int kvrows) {
    if (rows > d->cap) {
        cudaFree(d->x); cudaFree(d->qkv); cudaFree(d->q); cudaFree(d->att); cudaFree(d->gu); cudaFree(d->tok);
        cudaFree(d->ln); cudaFree(d->aq); cudaFree(d->as); cudaFree(d->acc);
        d->cap = rows;
        d->x = (bf16*)dmalloc((size_t)rows * D * 2);
        d->qkv = (bf16*)dmalloc((size_t)rows * QKV * 2);
        d->q = (bf16*)dmalloc((size_t)rows * D * 2);
        d->att = (bf16*)dmalloc((size_t)rows * D * 2);
        d->gu = (bf16*)dmalloc((size_t)rows * 2 * FF * 2);
        d->tok = (bf16*)dmalloc((size_t)rows * 64 * 2);
        d->ln = (bf16*)dmalloc((size_t)rows * D * 2);
        d->aq = (int8_t*)dmalloc((size_t)rows * FF);
        d->as = (float*)dmalloc((size_t)rows * 4);
        d->acc = (int32_t*)dmalloc((size_t)rows * 2 * FF * 4);
    }
    if (kvrows > d->kvcap) {
        for (int i = 0; i < NB; i++) { cudaFree(d->K[i]); cudaFree(d->V[i]); }
        d->kvcap = kvrows;
        for (int i = 0; i < NB; i++) {
            d->K[i] = (bf16*)dmalloc((size_t)kvrows * D * 2);
            d->V[i] = (bf16*)dmalloc((size_t)kvrows * D * 2);
        }
    }
}

// optional phase timing (env SHUTTER_PROF=1): events between the phases of block(), summed per forward
const char* PH_NAME[] = {"adaln+rq", "qkv gemm", "qkv deq+rope", "-", "attention", "o rq", "o gemm", "o deq+res",
                         "mlp adaln+rq", "gate_up gemm", "gu deq+swiglu", "-", "down gemm", "down deq+res"};
constexpr int NPH = 14;
bool g_prof = getenv("SHUTTER_PROF") != nullptr;
cudaEvent_t g_ev[NB][NPH + 1];
double g_ms[NPH] = {};
int g_prof_n = 0;
#define PH(k) do { if (g_prof && tap0 == false && i >= 0) CK(cudaEventRecord(g_ev[i][k], st)); } while (0)

// one transformer block on M rows of d->x; rows' K/V go to cache row kv0; attention over Nk keys
// mod: [scale1, gate1, scale2, gate2] vectors (each D) for these rows
void block(Dit* d, int i, int M, const bf16* mod, const float* pe, int kv0, int Nk, const int* qlim, bool tap0) {
    const Block& b = d->b[i];
    cudaStream_t st = d->st;
    tap0 = tap0 && (d->tap_qkv || d->tap_att || d->tap_gu);
    PH(0);
    rotquant(st, RQ_ADALN, d->x, D, M, D, mod, d->aq, d->as);
    PH(1);
    lin8(d, M, QKV, D, b.wqkv);
    PH(2);
    if (tap0) {  // debug: materialize qkv for the tap, unfused path
        dequant(st, d->acc, M, QKV, d->as, b.sqkv, d->qkv);
        if (d->tap_qkv) CK(cudaMemcpyAsync(d->tap_qkv, d->qkv, (size_t)M * QKV * 2, cudaMemcpyDeviceToDevice, st));
        rms_rope(st, d->qkv, M, pe, b.nq, b.nk, 1e-6f, d->q, d->K[i] + (size_t)kv0 * D, d->V[i] + (size_t)kv0 * D);
    } else {
        rms_rope_acc(st, d->acc, d->as, b.sqkv, M, pe, b.nq, b.nk, 1e-6f, d->q, d->K[i] + (size_t)kv0 * D,
                     d->V[i] + (size_t)kv0 * D);
    }
    PH(3);
    PH(4);
    attention(st, d->q, d->K[i], d->V[i], d->att, M, Nk, qlim);
    PH(5);
    if (tap0 && d->tap_att) CK(cudaMemcpyAsync(d->tap_att, d->att, (size_t)M * D * 2, cudaMemcpyDeviceToDevice, st));
    rotquant(st, RQ_PLAIN, d->att, D, M, D, nullptr, d->aq, d->as);
    PH(6);
    lin8(d, M, D, D, b.wo);
    PH(7);
    dequant_resid(st, d->acc, M, D, d->as, b.so, mod + D, d->x);
    PH(8);
    rotquant(st, RQ_ADALN, d->x, D, M, D, mod + 2 * D, d->aq, d->as);
    PH(9);
    lin8(d, M, 2 * FF, D, b.wgu);
    PH(10);
    if (tap0) {  // debug: materialize gate_up for the tap, unfused path
        dequant(st, d->acc, M, 2 * FF, d->as, b.sgu, d->gu);
        if (d->tap_gu) CK(cudaMemcpyAsync(d->tap_gu, d->gu, (size_t)M * 2 * FF * 2, cudaMemcpyDeviceToDevice, st));
        rotquant(st, RQ_SWIGLU, d->gu, 2 * FF, M, FF, nullptr, d->aq, d->as);
    } else {
        swiglu_rotquant_acc(st, d->acc, d->as, b.sgu, M, FF, d->aq, d->as);
    }
    PH(11);
    PH(12);
    lin8(d, M, D, FF, b.wd);
    PH(13);
    dequant_resid(st, d->acc, M, D, d->as, b.sd, mod + 3 * D, d->x);
    PH(14);
}

// swap the job's prefix context (P, K/V cache, rope, key limits) with the second one
void swap_ctx(Dit* d) {
    Dit::Ctx& a = d->alt;
    std::swap(d->P, a.P);
    std::swap(d->kvcap, a.kvcap);
    for (int i = 0; i < NB; i++) { std::swap(d->K[i], a.K[i]); std::swap(d->V[i], a.V[i]); }
    std::swap(d->pe_p, a.pe_p);
    std::swap(d->pe_t, a.pe_t);
    std::swap(d->qlim_p, a.qlim_p);
}

// rope table for positions ids [n][3] -> pe [n][64][4] (cos, -sin, sin, cos), computed in double as torch does
void rope_table(const std::vector<float>& ids, int n, std::vector<float>& pe) {
    const int axes[3] = {16, 56, 56};
    pe.assign((size_t)n * 64 * 4, 0.f);
    for (int r = 0; r < n; r++) {
        int pair = 0;
        for (int a = 0; a < 3; a++) {
            const int dim = axes[a];
            for (int i = 0; i < dim / 2; i++, pair++) {
                const double scale = (double)(2 * i) / dim;
                const double omega = 1.0 / std::pow(10000.0, scale);
                const double ang = (double)ids[(size_t)r * 3 + a] * omega;
                float* p = &pe[((size_t)r * 64 + pair) * 4];
                p[0] = (float)std::cos(ang);
                p[1] = (float)-std::sin(ang);
                p[2] = (float)std::sin(ang);
                p[3] = (float)std::cos(ang);
            }
        }
    }
}
}  // namespace

extern "C" {

void* qi21_dit_new(const char* path) {
    Dit* d = new Dit();
    if (!d->f.open(path)) DIE("cannot open %s", path);
    CK(cudaStreamCreate(&d->st));   // blocking on purpose (common.cuh)
    d->lt = lt_new(d->st);
    CK(cudaMallocHost(&g_stage, STAGE));
    if (g_prof)
        for (int i = 0; i < NB; i++)
            for (int k = 0; k <= NPH; k++) CK(cudaEventCreate(&g_ev[i][k]));
    d->t1 = (bf16*)upload(d, "time_text_embed.timestep_embedder.linear_1.weight", "BF16", (size_t)D * 256);
    d->t2 = (bf16*)upload(d, "time_text_embed.timestep_embedder.linear_2.weight", "BF16", (size_t)D * D);
    d->txt_norm = (bf16*)upload(d, "txt_in.text_norm.weight", "BF16", D);
    d->txt1 = (bf16*)upload(d, "txt_in.in_layer.weight", "BF16", (size_t)D * D);
    d->txt2 = (bf16*)upload(d, "txt_in.out_layer.weight", "BF16", (size_t)D * D);
    d->img_in = (bf16*)upload(d, "img_in.weight", "BF16", (size_t)D * 64);
    d->modw = (bf16*)upload(d, "modulation.1.weight", "BF16", (size_t)4 * D * D);
    d->nout = (bf16*)upload(d, "norm_out.linear.weight", "BF16", (size_t)D * D);
    d->pout = (bf16*)upload(d, "proj_out.weight", "BF16", (size_t)64 * D);
    for (int i = 0; i < NB; i++) {
        const std::string p = "transformer_blocks." + std::to_string(i) + ".";
        Block& b = d->b[i];
        for (const char* l : {"attn.to_q", "attn.to_k", "attn.to_v", "attn.to_out.0", "img_mlp.gate_up", "img_mlp.out"})
            check_convrot(d, p + l);
        b.wqkv = (int8_t*)dmalloc((size_t)QKV * D);
        b.sqkv = (float*)dmalloc((size_t)QKV * 4);
        const char* qkv[3] = {"attn.to_q", "attn.to_k", "attn.to_v"};
        for (int j = 0; j < 3; j++) {
            upload_into(d, p + qkv[j] + ".weight", b.wqkv + (size_t)j * D * D, (size_t)D * D);
            upload_into(d, p + qkv[j] + ".weight_scale", b.sqkv + j * D, (size_t)D * 4);
        }
        b.wo = (int8_t*)upload(d, p + "attn.to_out.0.weight", "I8", (size_t)D * D);
        b.so = (float*)upload(d, p + "attn.to_out.0.weight_scale", "F32", D);
        b.wgu = (int8_t*)upload(d, p + "img_mlp.gate_up.weight", "I8", (size_t)2 * FF * D);
        b.sgu = (float*)upload(d, p + "img_mlp.gate_up.weight_scale", "F32", 2 * FF);
        b.wd = (int8_t*)upload(d, p + "img_mlp.out.weight", "I8", (size_t)D * FF);
        b.sd = (float*)upload(d, p + "img_mlp.out.weight_scale", "F32", D);
        b.nq = (bf16*)upload(d, p + "attn.norm_q.weight", "BF16", 128);
        b.nk = (bf16*)upload(d, p + "attn.norm_k.weight", "BF16", 128);
    }
    d->f.close();
    CK(cudaDeviceSynchronize());  // weights fully on the device before any non-blocking-stream kernel reads them
    CK(cudaFreeHost(g_stage));
    g_stage = nullptr;
    d->temb_in = (bf16*)dmalloc(2 * 256 * 2);
    d->temb_h = (bf16*)dmalloc(2 * D * 2);
    d->temb = (bf16*)dmalloc(2 * D * 2);
    d->temb_s = (bf16*)dmalloc(2 * D * 2);
    d->modv = (bf16*)dmalloc(2 * 4 * D * 2);
    d->fs = (bf16*)dmalloc(D * 2);
    return d;
}

// Prepare a job: text context [L, 4096] (device bf16), reference latents [64, h, w] (device bf16) spliced into the
// text at image_slots, target latent size H x W (latent pixels). Runs the prefix through all blocks once.
int qi21_dit_prefix(void* h, const void* ctx, int L, int nref, const void* const* refs, const int* ref_hw,
                    const int* slots, int H, int W) {
    Dit* d = (Dit*)h;
    cudaStream_t st = d->st;
    int P = L;
    for (int r = 0; r < nref; r++) P += ref_hw[2 * r] * ref_hw[2 * r + 1];
    d->P = P;
    d->H = H;
    d->W = W;
    d->HW = H * W;
    int maxref = 0;
    for (int r = 0; r < nref; r++) maxref = std::max(maxref, ref_hw[2 * r] * ref_hw[2 * r + 1]);
    ensure(d, std::max(std::max(P, d->HW), std::max(L, maxref)), P + d->HW);

    // positions, segments and per-query key limits, exactly as build_sequence
    std::vector<float> ids;
    std::vector<int> qlim;
    std::vector<int> bounds = {0};
    for (int r = 0; r < nref; r++) bounds.push_back(slots ? slots[r] : L);
    bounds.push_back(L);
    int pos = 0, len = 0;
    std::vector<std::pair<int, int>> txt_rows;  // (dst row, src row)
    std::vector<std::pair<int, int>> ref_rows;  // (dst row start, ref index)
    for (int s = 0; s <= nref; s++) {
        const int a = bounds[s], b = bounds[s + 1];
        for (int j = a; j < b; j++) {
            ids.insert(ids.end(), {(float)pos, (float)pos, (float)pos});
            qlim.push_back(len + 1);
            txt_rows.push_back({len, j});
            pos++;
            len++;
        }
        const int hh = s < nref ? ref_hw[2 * s] : H, ww = s < nref ? ref_hw[2 * s + 1] : W;
        for (int y = 0; y < hh; y++)
            for (int x = 0; x < ww; x++) {
                const float fy = (float)y - (float)(hh - hh / 2) + 0.5f * (float)(hh % 2 - H % 2);
                const float fx = (float)x - (float)(ww - ww / 2) + 0.5f * (float)(ww % 2 - W % 2);
                ids.insert(ids.end(), {(float)pos, fy, fx});
                if (s < nref) qlim.push_back(len + hh * ww);
            }
        if (s < nref) ref_rows.push_back({len, s});
        pos += std::max(hh, ww);
        len += hh * ww;
    }
    std::vector<float> pe;
    rope_table(ids, P + d->HW, pe);
    cudaFree(d->pe_p);
    cudaFree(d->qlim_p);
    d->pe_p = (float*)dmalloc(pe.size() * 4);
    CK(cudaMemcpyAsync(d->pe_p, pe.data(), pe.size() * 4, cudaMemcpyHostToDevice, st));
    d->pe_t = d->pe_p + (size_t)P * 64 * 4;
    d->qlim_p = (int*)dmalloc(std::max<size_t>(qlim.size(), 1) * 4);
    if (!qlim.empty()) CK(cudaMemcpyAsync(d->qlim_p, qlim.data(), qlim.size() * 4, cudaMemcpyHostToDevice, st));

    conditioning(d, 0.f);
    if (P == 0) { CK(cudaStreamSynchronize(st)); return 0; }
    // text: txt_in(context) into att (scratch), then scatter rows into x
    rmsnorm_zc(st, (const bf16*)ctx, d->txt_norm, d->ln, L, D, 1e-6f);
    gemm_bf16(d->lt, L, D, D, d->ln, d->txt1, d->att);
    elementwise(st, EW_GELU_TANH, d->att, d->att, L * D);
    gemm_bf16(d->lt, L, D, D, d->att, d->txt2, d->q);
    for (auto& tr : txt_rows)
        CK(cudaMemcpyAsync(d->x + (size_t)tr.first * D, d->q + (size_t)tr.second * D, D * 2, cudaMemcpyDeviceToDevice, st));
    for (auto& rr : ref_rows) {
        const int n = ref_hw[2 * rr.second] * ref_hw[2 * rr.second + 1];
        chw_to_tokens(st, (const bf16*)refs[rr.second], d->tok, 64, n);
        gemm_bf16(d->lt, n, D, 64, d->tok, d->img_in, d->x + (size_t)rr.first * D);
    }
    const bf16* modp = d->modv + 4 * D;  // prefix row (t = 0)
    for (int i = 0; i < NB; i++) {
        block(d, i, P, modp, d->pe_p, 0, P, d->qlim_p, false);
        if (d->tap_prefix) CK(cudaMemcpyAsync(d->tap_prefix + (size_t)i * P * D, d->x, (size_t)P * D * 2, cudaMemcpyDeviceToDevice, st));
    }
    CK(cudaStreamSynchronize(st));
    return 0;
}

// Exact DiT call: x [64, H, W] bf16 (device), timestep sigma -> out [64, H, W] bf16 (device)
int qi21_dit_forward(void* h, const void* x, float sigma, void* out) {
    Dit* d = (Dit*)h;
    cudaStream_t st = d->st;
    const int M = d->HW;
    conditioning(d, sigma);
    chw_to_tokens(st, (const bf16*)x, d->tok, 64, M);
    gemm_bf16(d->lt, M, D, 64, d->tok, d->img_in, d->x);
    for (int i = 0; i < NB; i++) {
        block(d, i, M, d->modv, d->pe_t, d->P, d->P + M, nullptr, i == 0);
        if (d->tap_blocks) CK(cudaMemcpyAsync(d->tap_blocks + (size_t)i * M * D, d->x, (size_t)M * D * 2, cudaMemcpyDeviceToDevice, st));
    }
    adaln(st, d->x, d->fs, d->ln, M, D, 1e-6f);
    gemm_bf16(d->lt, M, 64, D, d->ln, d->pout, d->tok);
    tokens_to_chw(st, d->tok, (bf16*)out, 64, M);
    CK(cudaStreamSynchronize(st));
    if (g_prof && !d->tap_qkv) {
        for (int i = 0; i < NB; i++)
            for (int k = 0; k < NPH; k++) { float ms; CK(cudaEventElapsedTime(&ms, g_ev[i][k], g_ev[i][k + 1])); g_ms[k] += ms; }
        if (++g_prof_n % 5 == 0) {
            double tot = 0;
            for (int k = 0; k < NPH; k++) tot += g_ms[k];
            fprintf(stderr, "[prof] per forward (M=%d, avg of %d): total %.1f ms\n", M, g_prof_n, tot / g_prof_n);
            for (int k = 0; k < NPH; k++) fprintf(stderr, "[prof]   %-14s %7.2f ms  %4.1f%%\n", PH_NAME[k], g_ms[k] / g_prof_n, 100 * g_ms[k] / tot);
        }
    }
    return 0;
}

// Sampler step on the fp32 latent x [64, H, W] (device), in place: flow euler sigma -> sigma_next
int qi21_dit_step(void* h, void* x, float sigma, float sigma_next) {
    Dit* d = (Dit*)h;
    const int n = 64 * d->HW;
    bf16* xin = d->ln;  // scratch: [64*HW] bf16 fits in ln ([cap, 4096])
    bf16* o = d->ln + n;
    f32_to_bf16(d->st, (const float*)x, xin, n);
    qi21_dit_forward(h, xin, sigma, o);
    euler(d->st, (float*)x, o, n, sigma, sigma_next);
    CK(cudaStreamSynchronize(d->st));
    return 0;
}

// CFG (negative prompt): swap the two prefix contexts (prefix → swap → prefix leaves the negative waiting in the second)
void qi21_dit_swap(void* h) { swap_ctx((Dit*)h); }

// release the second context (a job without a negative prompt)
void qi21_dit_drop_alt(void* h) {
    Dit* d = (Dit*)h;
    CK(cudaStreamSynchronize(d->st));
    Dit::Ctx& a = d->alt;
    for (int i = 0; i < NB; i++) { cudaFree(a.K[i]); cudaFree(a.V[i]); }
    cudaFree(a.pe_p);
    cudaFree(a.qlim_p);
    a = Dit::Ctx();
}

// sampler step with CFG: cond forward on the current context, uncond on the second, euler on the mixed denoised values
int qi21_dit_step_cfg(void* h, void* x, float sigma, float sigma_next, float cfg) {
    Dit* d = (Dit*)h;
    const int n = 64 * d->HW;
    if (n > d->cfgn) {
        cudaFree(d->cfgb);
        d->cfgb = (bf16*)dmalloc((size_t)3 * n * 2);
        d->cfgn = n;
    }
    bf16 *xin = d->cfgb, *oc = xin + n, *ou = oc + n;
    f32_to_bf16(d->st, (const float*)x, xin, n);
    qi21_dit_forward(h, xin, sigma, oc);
    swap_ctx(d);
    qi21_dit_forward(h, xin, sigma, ou);
    swap_ctx(d);
    euler_cfg(d->st, (float*)x, oc, ou, n, sigma, sigma_next, cfg);
    CK(cudaStreamSynchronize(d->st));
    return 0;
}

void qi21_dit_taps(void* h, void* blocks, void* prefix, void* qkv0, void* att0, void* gu0) {
    Dit* d = (Dit*)h;
    d->tap_blocks = (bf16*)blocks;
    d->tap_prefix = (bf16*)prefix;
    d->tap_qkv = (bf16*)qkv0;
    d->tap_att = (bf16*)att0;
    d->tap_gu = (bf16*)gu0;
}

void qi21_dit_free(void* h) {
    Dit* d = (Dit*)h;
    qi21_dit_drop_alt(h);
    cudaFree(d->cfgb);
    void* ps[] = {d->t1, d->t2, d->txt_norm, d->txt1, d->txt2, d->img_in, d->modw, d->nout, d->pout, d->pe_p, d->qlim_p,
                  d->x, d->qkv, d->q, d->att, d->gu, d->tok, d->ln, d->aq, d->as, d->acc, d->temb_in, d->temb_h,
                  d->temb, d->temb_s, d->modv, d->fs};
    for (void* p : ps) cudaFree(p);
    for (int i = 0; i < NB; i++) {
        Block& b = d->b[i];
        void* bs[] = {b.wqkv, b.wo, b.wgu, b.wd, b.sqkv, b.so, b.sgu, b.sd, b.nq, b.nk, d->K[i], d->V[i]};
        for (void* p : bs) cudaFree(p);
    }
    lt_free(d->lt);
    cudaStreamDestroy(d->st);
    delete d;
}
}  // extern "C"

// bench/debug: the flash attention kernel alone (q [Mq,4096], k/v [Nk,4096], o [Mq,4096] bf16 device)
extern "C" int shutter_attention(const void* q, const void* k, const void* v, void* o, int Mq, int Nk) {
    attention(0, (const bf16*)q, (const bf16*)k, (const bf16*)v, (bf16*)o, Mq, Nk, nullptr);
    CK(cudaDeviceSynchronize());
    return 0;
}
