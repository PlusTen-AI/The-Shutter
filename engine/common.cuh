// shutter engine — shared helpers
#pragma once
#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublasLt.h>
#include <cstdint>
#include <cstdio>
#include <cstdlib>

typedef __nv_bfloat16 bf16;

// Engine streams are plain (blocking) streams, not cudaStreamNonBlocking: a pageable cudaMemcpy H2D on the legacy
// stream returns before its DMA lands, and a kernel on a non-blocking stream right after can read the old bytes
// (measured on GB10: 32 stale reads in 20000 launches, 0 with a blocking stream). Blocking streams also order the
// engine against torch's (legacy) stream on both sides.

#define CK(x) do { cudaError_t e_ = (x); if (e_ != cudaSuccess) { \
    fprintf(stderr, "[engine] CUDA %s at %s:%d\n", cudaGetErrorString(e_), __FILE__, __LINE__); abort(); } } while (0)
#define CKL(x) do { cublasStatus_t s_ = (x); if (s_ != CUBLAS_STATUS_SUCCESS) { \
    fprintf(stderr, "[engine] cublasLt status %d at %s:%d\n", (int)s_, __FILE__, __LINE__); abort(); } } while (0)
#define DIE(...) do { fprintf(stderr, "[engine] " __VA_ARGS__); fputc('\n', stderr); abort(); } while (0)

static inline float bf2f(bf16 v) { return __bfloat162float(v); }
static inline bf16 f2bf(float v) { return __float2bfloat16(v); }  // round to nearest even

// ---- GEMM (cuBLASLt), row-major: C[M,N] = A[M,K] . W[N,K]^T ----
struct Lt;
Lt* lt_new(cudaStream_t st);
void lt_free(Lt*);
void gemm_bf16(Lt*, int M, int N, int K, const bf16* A, const bf16* W, bf16* C);
void gemm_i8(Lt*, int M, int N, int K, const int8_t* A, const int8_t* W, int32_t* C);
void gemm_bf16_f32(Lt*, int M, int N, int K, const bf16* A, const bf16* W, float* C);  // fp32 out
void gemm_f32(Lt*, int M, int N, int K, const float* A, const float* W, float* C);     // fp32 (no TF32)
// int8 GEMM with the epilogue fused (gemm_dq.cu, CUTLASS): A [M, K], W [N, K] int8 rows, row scale xs, column scale ws,
// a = acc * xs (fp32), out [M, N] bf16 rows:
//   DQ_F32B out = bf16(fma(a, ws, b)), b fp32 (nullptr = 0)        DQ_RES  x = bf16(fma(sc, y, x)), y = DQ_F32B, aux = sc fp32
//   DQ_MUL  out = bf16(a * ws)                                       DQ_GATE x = bf16(x + bf16(y * gate)), y = bf16(fma(a, ws, b)),
//   DQ_GELU out = bf16(gelu_tanh(y)), y as DQ_GATE                           b bf16, aux = gate bf16
// Same bits as gemm_i8 + an epilogue kernel doing those fp32 operations, without the int32 round trip.
enum { DQ_F32B = 0, DQ_RES = 1, DQ_MUL = 2, DQ_GATE = 3, DQ_GELU = 4 };
void gemm_i8_epi(cudaStream_t, int epi, int M, int N, int K, const int8_t* A, const int8_t* W, const float* xs,
                 const float* ws, const void* b, const void* aux, bf16* out);

// ---- kernels (kernels.cu) ----
// rows of K (<= 16384, multiple of 256): optional pre-op, then ConvRot (regular Hadamard 256) + row-wise int8 quant
enum { RQ_PLAIN = 0, RQ_ADALN = 1, RQ_SWIGLU = 2, RQ_SWIGLU_ACC = 3, RQ_RMSMOD = 4 };
void rotquant(cudaStream_t, int mode, const bf16* in, int ld_in, int M, int K, const bf16* mod_scale,
              int8_t* q, float* qs);
// H3 pre-op: h = bf16(bf16(rmsnorm(x) * w) * bf16(1 + scale) + shift) per row (torch op-by-op rounding), then ConvRot + int8.
// Row m uses mod row mrow[m]: shift = mod[mrow*ldm + j], scale = mod[mrow*ldm + K + j].
void rotquant_rmsmod(cudaStream_t, const bf16* x, int M, int K, const bf16* nw, float eps, const bf16* mod, int ldm,
                     const int* mrow, int8_t* q, float* qs);
// fused: [gate|up] int32 accumulators (+ scales) -> dequant (bf16 rounding) -> swiglu -> ConvRot -> int8 (K = half width)
void swiglu_rotquant_acc(cudaStream_t, const int32_t* acc, const float* xs, const float* ws, int M, int K, int8_t* q, float* qs);
// fused: qkv int32 accumulators -> dequant (bf16 rounding) -> per-head RMSNorm + RoPE (q, k) / copy (v)
void rms_rope_acc(cudaStream_t, const int32_t* acc, const float* xs, const float* ws, int M, const float* pe,
                  const bf16* wq, const bf16* wk, float eps, bf16* q, bf16* k, bf16* v);
// y = bf16(acc * xs[m] * ws[n]); mode 0 writes y, mode 1 does x = bf16(x + y * gate[n])
void dequant(cudaStream_t, const int32_t* acc, int M, int N, const float* xs, const float* ws, bf16* out);
void dequant_resid(cudaStream_t, const int32_t* acc, int M, int N, const float* xs, const float* ws,
                   const bf16* gate, bf16* x);
// per-head RMSNorm (weights wq/wk) + RoPE on q,k from qkv [M, 3*4096]; writes q [M,4096], k/v rows to the cache
void rms_rope(cudaStream_t, const bf16* qkv, int M, const float* pe, const bf16* wq, const bf16* wk, float eps,
              bf16* q, bf16* k, bf16* v);
// block-causal flash attention, heads 32 x 128; qlim[i] = keys visible to query i (nullptr = all Nk)
void attention(cudaStream_t, const bf16* q, const bf16* k, const bf16* v, bf16* o, int Mq, int Nk, const int* qlim);
// same, any head count (head dim 128): q/k/v rows of stride ld, output rows of stride ldo
void attention_ex(cudaStream_t, const bf16* q, const bf16* k, const bf16* v, bf16* o, int Mq, int Nk, const int* qlim,
                  int heads, int ld, int ldo, float scale = 0.f);   // scale 0 = 1/sqrt(128)
// quantized attention (lossy switch, attn_q8.cu): int8 Q/K, e4m3 P/V like SageAttention2 on sm_121; no mask
void attention_q8(cudaStream_t, const bf16* q, const bf16* k, const bf16* v, bf16* o, int Mq, int Nk, int heads, int ld,
                  int ldo);
void attention_q8_release();   // its global workspace, at model unload
// small elementwise (bf16 in/out, fp32 math, one rounding)
enum { EW_SILU = 0, EW_TANH = 1, EW_GELU_TANH = 2 };
void elementwise(cudaStream_t, int op, const bf16* in, bf16* out, int n);
// out = bf16(rmsnorm(x) * (w + 1)) (fp32 math), rows of K
void rmsnorm_zc(cudaStream_t, const bf16* x, const bf16* w, bf16* out, int M, int K, float eps);
// out = bf16(layernorm(x) * (1 + scale)), rows of K (final layer)
void adaln(cudaStream_t, const bf16* x, const bf16* scale, bf16* out, int M, int K, float eps);
// [C,H,W] <-> [H*W, C]
void chw_to_tokens(cudaStream_t, const bf16* x, bf16* t, int C, int HW);
void tokens_to_chw(cudaStream_t, const bf16* t, bf16* x, int C, int HW);
void f32_to_bf16(cudaStream_t, const float* in, bf16* out, int n);
// flow euler in fp32: den = x - out*s; x += (x - den)/s * (s_next - s)
void euler(cudaStream_t, float* x, const bf16* out, int n, float s, float s_next);
// the same step with CFG on the denoised values (ComfyUI cfg_function): den = den_u + (den_c - den_u) * cfg
void euler_cfg(cudaStream_t, float* x, const bf16* out_c, const bf16* out_u, int n, float s, float s_next, float cfg);
