// int8 GEMM with the epilogue fused (CUTLASS 2.x, sm80 mma.sync s8 path, runs on sm_121). a = (float)acc * xs[m], then
// per epilogue (DqEpi, common.cuh):
//   DQ_F32B  out = bf16(fma(a, ws, b))                    b fp32 (null -> 0)                       H3 VAE
//   DQ_RES   x = bf16(fma(sc, y, x)), y = DQ_F32B          aux = sc fp32, in place                  H3 VAE
//   DQ_MUL   out = bf16(a * ws)                                                                     SeedVR2 qkv
//   DQ_GATE  x = bf16(x + bf16(y * gate)), y = bf16(fma(a, ws, b))   b bf16, aux = gate bf16, in place   SeedVR2
//   DQ_GELU  out = bf16(gelu_tanh(y)), y as DQ_GATE                                                 SeedVR2 fc1
// Same numbers as cuBLASLt int32 accumulators + the separate epilogue kernels (integer accumulation is exact; the fp32
// steps are pinned with __fmul_rn / __fmaf_rn to the operations nvcc emitted for those kernels), but writes bf16
// instead of an int32 round trip. Tile shape is picked per GEMM shape by timing (all tiles give the same bits).
#include "common.cuh"

#include <map>
#include <tuple>
#include <vector>

// include order as in CUTLASS example 47 (visitors.hpp leans on the gemm / conversion headers before it)
#include "cutlass/cutlass.h"
#include "cutlass/numeric_conversion.h"
#include "cutlass/gemm/device/gemm_universal.h"
#include "cutlass/epilogue/threadblock/fusion/visitors.hpp"
#include "cutlass/gemm/kernel/default_gemm_universal_with_visitor.h"
#include "cutlass/gemm/device/gemm_universal_adapter.h"

namespace {
using cute::Stride, cute::_0, cute::_1;   // not "using namespace cute": its anonymous namespace clashes with ours (nvcc stub)
namespace tb = cutlass::epilogue::threadblock;
using cutlass::bfloat16_t;
template <int N> using AF = cutlass::Array<float, N>;
constexpr auto RN = cutlass::FloatRoundStyle::round_to_nearest;

// fp32 steps in fixed roundings (no contraction across them)
template <class T> struct Mul;
template <int N> struct Mul<AF<N>> {
    CUTLASS_DEVICE AF<N> operator()(AF<N> const& a, AF<N> const& b) const {
        AF<N> r;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; i++) r[i] = __fmul_rn(a[i], b[i]);
        return r;
    }
};
template <class T> struct Add;
template <int N> struct Add<AF<N>> {
    CUTLASS_DEVICE AF<N> operator()(AF<N> const& a, AF<N> const& b) const {
        AF<N> r;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; i++) r[i] = __fadd_rn(a[i], b[i]);
        return r;
    }
};
template <class T> struct Fma;   // one rounding: what nvcc contracts the reference kernels' a * b + c into
template <int N> struct Fma<AF<N>> {
    CUTLASS_DEVICE AF<N> operator()(AF<N> const& a, AF<N> const& b, AF<N> const& c) const {
        AF<N> r;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; i++) r[i] = __fmaf_rn(a[i], b[i], c[i]);
        return r;
    }
};
// torch GeluCUDAKernelImpl tanh: 0.5 y (1 + tanh(beta (y + kappa y^3))), steps as nvcc emitted seedvr2 k_epi<1>
__device__ __forceinline__ float gelu_t(float y) {
    constexpr float kBeta = (float)(M_SQRT2 * M_2_SQRTPI * 0.5), kKappa = 0.044715f;
    const float in = __fmul_rn(__fmaf_rn(__fmul_rn(y, __fmul_rn(y, y)), kKappa, y), kBeta);
    return __fmul_rn(__fmul_rn(y, 0.5f), __fadd_rn(tanhf(in), 1.f));
}
template <class T> struct Gelu;
template <int N> struct Gelu<AF<N>> {
    CUTLASS_DEVICE AF<N> operator()(AF<N> const& a) const {
        AF<N> r;
        CUTLASS_PRAGMA_UNROLL
        for (int i = 0; i < N; i++) r[i] = gelu_t(a[i]);
        return r;
    }
};

using SR = Stride<_0, _1, int32_t>;     // one value per column
using SC = Stride<_1, _0, int32_t>;     // one value per row
using SA = Stride<int64_t, _1, int64_t>;

template <class TB, class WS, int STAGES, int EPI>
struct Cfg {
    using TM = tb::OutputTileThreadLayout<TB, WS, bfloat16_t, 8, 1>;
    using Acc = tb::VisitorAccFetch;
    using Col = tb::VisitorColBroadcast<TM, float, SC>;
    using RowF = tb::VisitorRowBroadcast<TM, float, SR>;
    using RowH = tb::VisitorRowBroadcast<TM, bfloat16_t, SR>;
    using X = tb::VisitorAuxLoad<TM, bfloat16_t, SA>;
    template <template <class> class F, class... C>
    using Op = tb::Sm80EVT<tb::VisitorCompute<F, bfloat16_t, float, RN>, C...>;   // fp32 math, bf16 out
    using A = tb::Sm80EVT<tb::VisitorCompute<Mul, float, float, RN>, Acc, Col>;   // acc * xs
    using Yf = Op<Fma, A, RowF, RowF>;
    using Yh = Op<Fma, A, RowF, RowH>;
    using G = Op<Mul, Yh, RowH>;
    using Top = std::conditional_t<EPI == DQ_F32B, Yf,
                std::conditional_t<EPI == DQ_RES, Op<Fma, RowF, Yf, X>,
                std::conditional_t<EPI == DQ_MUL, Op<Mul, A, RowF>,
                std::conditional_t<EPI == DQ_GATE, Op<Add, X, G>, Op<Gelu, Yh>>>>>;
    using Out = tb::VisitorAuxStore<TM, bfloat16_t, RN, SA>;
    using EVT = tb::Sm80EVT<Out, Top>;
    using Kernel = typename cutlass::gemm::kernel::DefaultGemmWithVisitor<
        int8_t, cutlass::layout::RowMajor, cutlass::ComplexTransform::kNone, 16,
        int8_t, cutlass::layout::ColumnMajor, cutlass::ComplexTransform::kNone, 16,
        bfloat16_t, cutlass::layout::RowMajor, 8, int32_t, float, cutlass::arch::OpClassTensorOp, cutlass::arch::Sm80, TB,
        WS, cutlass::gemm::GemmShape<16, 8, 32>, EVT, cutlass::gemm::threadblock::GemmIdentityThreadblockSwizzle<8>,
        STAGES, cutlass::arch::OpMultiplyAddSaturate, 1>::GemmKernel;
    using Gemm = cutlass::gemm::device::GemmUniversalAdapter<Kernel>;

    // A [M, K] int8 rows, W [N, K] int8 rows -> out [M, N] bf16 rows (in place for DQ_RES / DQ_GATE)
    static cudaError_t run(cudaStream_t st, int M, int N, int K, const int8_t* A_, const int8_t* W, const float* xs,
                           const float* ws, const void* b, const void* aux, bf16* out) {
        const SR sr{_0{}, _1{}, N};
        const SA so{(int64_t)N, _1{}, (int64_t)M * N};
        const typename A::Arguments a{{}, {xs, 0.f, SC{_1{}, _0{}, M}}, {}};
        const typename RowF::Arguments w{ws, 0.f, sr};
        const typename X::Arguments x{(bfloat16_t*)out, bfloat16_t(0), so};
        const auto rf = [&](const void* p) { return typename RowF::Arguments{(const float*)p, 0.f, sr}; };
        const auto rh = [&](const void* p) { return typename RowH::Arguments{(const bfloat16_t*)p, bfloat16_t(0), sr}; };
        const auto top = [&] {
            if constexpr (EPI == DQ_F32B) return typename Top::Arguments{a, w, rf(b), {}};
            else if constexpr (EPI == DQ_RES) return typename Top::Arguments{rf(aux), {a, w, rf(b), {}}, x, {}};
            else if constexpr (EPI == DQ_MUL) return typename Top::Arguments{a, w, {}};
            else if constexpr (EPI == DQ_GATE) return typename Top::Arguments{x, {{a, w, rh(b), {}}, rh(aux), {}}, {}};
            else return typename Top::Arguments{{a, w, rh(b), {}}, {}};
        }();
        const typename EVT::Arguments ea{top, {(bfloat16_t*)out, so}};
        typename Gemm::Arguments args(cutlass::gemm::GemmUniversalMode::kGemm, {M, N, K}, 1, ea, A_, W, nullptr, nullptr,
                                      (int64_t)M * K, (int64_t)N * K, 0, 0, K, K, 0, 0);
        Gemm g;
        if (g.can_implement(args) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
        if (g.initialize(args, nullptr, st) != cutlass::Status::kSuccess) return cudaErrorInvalidValue;
        if (g.run(st) != cutlass::Status::kSuccess) return cudaErrorLaunchFailure;
        return cudaGetLastError();
    }
};

template <int M, int N, int K> using GS = cutlass::gemm::GemmShape<M, N, K>;
typedef cudaError_t (*RunFn)(cudaStream_t, int, int, int, const int8_t*, const int8_t*, const float*, const float*,
                             const void*, const void*, bf16*);
constexpr int NTILE = 4;
template <int EPI>
const RunFn* tiles() {
    static const RunFn t[NTILE] = {
        &Cfg<GS<128, 256, 64>, GS<64, 64, 64>, 3, EPI>::run,
        &Cfg<GS<256, 128, 64>, GS<64, 64, 64>, 3, EPI>::run,
        &Cfg<GS<128, 128, 64>, GS<64, 64, 64>, 4, EPI>::run,
        &Cfg<GS<128, 128, 64>, GS<64, 64, 64>, 5, EPI>::run,
    };
    return t;
}

// fastest tile per (M, N, K, epilogue), timed once on the real call (every tile writes the same bits)
std::map<std::tuple<int, int, int, int>, int> g_pick;
int g_force = -1;   // test: force one tile

cudaError_t dq_run(cudaStream_t st, int epi, int M, int N, int K, const int8_t* A, const int8_t* W, const float* xs,
                   const float* ws, const void* b, const void* aux, bf16* out) {
    static const RunFn* const all[] = {tiles<DQ_F32B>(), tiles<DQ_RES>(), tiles<DQ_MUL>(), tiles<DQ_GATE>(), tiles<DQ_GELU>()};
    const RunFn* t = all[epi];
    if (g_force >= 0) return t[g_force % NTILE](st, M, N, K, A, W, xs, ws, b, aux, out);
    const auto key = std::make_tuple(M, N, K, epi);
    auto it = g_pick.find(key);
    if (it != g_pick.end()) return t[it->second](st, M, N, K, A, W, xs, ws, b, aux, out);
    // in-place epilogues read-modify-write x: time on a scratch copy so the real output is written exactly once
    const bool inplace = epi == DQ_RES || epi == DQ_GATE;
    bf16* tmp = nullptr;
    if (inplace) {
        CK(cudaMallocAsync(&tmp, (size_t)M * N * 2, st));
        CK(cudaMemcpyAsync(tmp, out, (size_t)M * N * 2, cudaMemcpyDeviceToDevice, st));
    }
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    int best = -1;
    float bt = 1e30f;
    for (int i = 0; i < NTILE; i++) {
        bf16* o = inplace ? tmp : out;
        if (t[i](st, M, N, K, A, W, xs, ws, b, aux, o) != cudaSuccess) continue;   // warm / can't
        CK(cudaEventRecord(e0, st));
        CK(t[i](st, M, N, K, A, W, xs, ws, b, aux, o));
        CK(cudaEventRecord(e1, st));
        CK(cudaEventSynchronize(e1));
        float ms;
        CK(cudaEventElapsedTime(&ms, e0, e1));
        if (ms < bt) bt = ms, best = i;
    }
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    if (tmp) CK(cudaFreeAsync(tmp, st));
    if (best < 0) DIE("gemm_i8_epi: no tile fits %dx%dx%d", M, N, K);
    g_pick[key] = best;
    return t[best](st, M, N, K, A, W, xs, ws, b, aux, out);
}
}  // namespace

void gemm_i8_epi(cudaStream_t st, int epi, int M, int N, int K, const int8_t* A, const int8_t* W, const float* xs,
                 const float* ws, const void* b, const void* aux, bf16* out) {
    CK(dq_run(st, epi, M, N, K, A, W, xs, ws, b, aux, out));
}

// ---- test entry: the fused GEMM against cuBLASLt int32 + a plain epilogue kernel, bits and time ----
namespace {
__global__ void k_ref(int epi, const int32_t* acc, int M, int N, const float* xs, const float* ws, const void* b,
                      const void* aux, bf16* x) {
    const size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= (size_t)M * N) return;
    const int n = (int)(i % N);
    const float a = __fmul_rn((float)acc[i], xs[i / N]);
    const float* bf = (const float*)b;
    const bf16* bh = (const bf16*)b;
    const auto r = [](float v) { return __bfloat162float(__float2bfloat16(v)); };
    float o;
    if (epi == DQ_F32B) o = __fmaf_rn(a, ws[n], bf[n]);
    else if (epi == DQ_RES) o = __fmaf_rn(((const float*)aux)[n], r(__fmaf_rn(a, ws[n], bf[n])), __bfloat162float(x[i]));
    else if (epi == DQ_MUL) o = __fmul_rn(a, ws[n]);
    else {
        const float y = r(__fmaf_rn(a, ws[n], __bfloat162float(bh[n])));
        o = epi == DQ_GATE ? __fadd_rn(__bfloat162float(x[i]), r(__fmul_rn(y, __bfloat162float(((const bf16*)aux)[n]))))
                           : gelu_t(y);
    }
    x[i] = __float2bfloat16(o);
}
__global__ void k_fill(uint32_t* p, size_t n, uint32_t seed, int kind) {   // 0 int8 bytes, 1 scales, 2 bf16 pairs
    for (size_t i = (size_t)blockIdx.x * blockDim.x + threadIdx.x; i < n; i += (size_t)gridDim.x * blockDim.x) {
        uint32_t h = (uint32_t)i * 2654435761u ^ seed;
        h ^= h >> 15, h *= 2246822519u, h ^= h >> 13, h *= 3266489917u, h ^= h >> 16;
        if (kind == 1) ((float*)p)[i] = 1e-4f + (h >> 8) * (1.f / (1 << 24)) * 1e-3f;
        else if (kind == 2) {
            const float a = ((int)(h & 0xffff) - 32768) / 4096.f, c = ((int)(h >> 16) - 32768) / 4096.f;
            ((__nv_bfloat162*)p)[i] = __floats2bfloat162_rn(a, c);
        } else p[i] = h;
    }
}
}  // namespace

// returns the number of differing output elements; times in ms (reference = cuBLASLt + kernel, fused) via t[2]
extern "C" long shutter_gemm_dq_test(int M, int N, int K, int epi, int tile, int iters, float* t) {
    int8_t *A, *W;
    float *xs, *ws, *b, *aux;
    int32_t* acc;
    bf16 *y0, *y1;
    CK(cudaMalloc(&A, (size_t)M * K));
    CK(cudaMalloc(&W, (size_t)N * K));
    CK(cudaMalloc(&xs, M * 4));
    CK(cudaMalloc(&ws, N * 4));
    CK(cudaMalloc(&b, N * 4));
    CK(cudaMalloc(&aux, N * 4));
    CK(cudaMalloc(&acc, (size_t)M * N * 4));
    CK(cudaMalloc(&y0, (size_t)M * N * 2));
    CK(cudaMalloc(&y1, (size_t)M * N * 2));
    const bool half = epi == DQ_GATE || epi == DQ_GELU;   // bf16 bias / gate
    k_fill<<<1024, 256>>>((uint32_t*)A, (size_t)M * K / 4, 1, 0);
    k_fill<<<1024, 256>>>((uint32_t*)W, (size_t)N * K / 4, 2, 0);
    k_fill<<<64, 256>>>((uint32_t*)xs, M, 3, 1);
    k_fill<<<64, 256>>>((uint32_t*)ws, N, 4, 1);
    k_fill<<<64, 256>>>((uint32_t*)b, half ? N / 2 : N, 5, half ? 2 : 1);
    k_fill<<<64, 256>>>((uint32_t*)aux, half ? N / 2 : N, 6, half ? 2 : 1);
    k_fill<<<1024, 256>>>((uint32_t*)y0, (size_t)M * N / 2, 7, 2);
    CK(cudaMemcpy(y1, y0, (size_t)M * N * 2, cudaMemcpyDeviceToDevice));
    CK(cudaDeviceSynchronize());
    Lt* lt = lt_new(0);
    cudaEvent_t e0, e1;
    CK(cudaEventCreate(&e0));
    CK(cudaEventCreate(&e1));
    const unsigned nb = (unsigned)(((size_t)M * N + 255) / 256);
    auto ref = [&]() {
        gemm_i8(lt, M, N, K, A, W, acc);
        k_ref<<<nb, 256>>>(epi, acc, M, N, xs, ws, b, aux, y0);
    };
    g_force = tile;
    auto fused = [&]() { CK(dq_run(0, epi, M, N, K, A, W, xs, ws, b, aux, y1)); };
    ref(), fused();   // once each for the bit compare (in place: both applied once to the same start)
    CK(cudaDeviceSynchronize());
    std::vector<uint16_t> h0((size_t)M * N), h1((size_t)M * N);
    CK(cudaMemcpy(h0.data(), y0, h0.size() * 2, cudaMemcpyDeviceToHost));
    CK(cudaMemcpy(h1.data(), y1, h1.size() * 2, cudaMemcpyDeviceToHost));
    long diff = 0;
    for (size_t i = 0; i < h0.size(); i++) diff += h0[i] != h1[i];
    auto time = [&](auto f, float* ms) {
        f();
        CK(cudaEventRecord(e0));
        for (int i = 0; i < iters; i++) f();
        CK(cudaEventRecord(e1));
        CK(cudaEventSynchronize(e1));
        CK(cudaEventElapsedTime(ms, e0, e1));
        *ms /= iters;
    };
    time(ref, &t[0]);
    time(fused, &t[1]);
    g_force = -1;
    cudaEventDestroy(e0);
    cudaEventDestroy(e1);
    lt_free(lt);
    for (void* p : {(void*)A, (void*)W, (void*)xs, (void*)ws, (void*)b, (void*)aux, (void*)acc, (void*)y0, (void*)y1}) cudaFree(p);
    return diff;
}
