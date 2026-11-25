// cuBLASLt GEMMs, row-major C[M,N] = A[M,K] . W[N,K]^T (TN in column-major terms).
// Each shape is planned once (int8: fastest of the heuristic's candidates; bf16: first candidate, deterministic).
#include "common.cuh"

#include <map>
#include <tuple>

namespace {
struct Plan {
    cublasLtMatmulDesc_t op;
    cublasLtMatrixLayout_t lw, la, lc;
    cublasLtMatmulAlgo_t algo;
};
}  // namespace

struct Lt {
    cublasLtHandle_t h;
    cudaStream_t st;
    void* ws = nullptr;
    size_t wsz = 64u << 20;
    std::map<std::tuple<int, int, int, int>, Plan> plans;
};

Lt* lt_new(cudaStream_t st) {
    Lt* lt = new Lt();
    lt->st = st;
    CKL(cublasLtCreate(&lt->h));
    CK(cudaMalloc(&lt->ws, lt->wsz));
    return lt;
}

void lt_free(Lt* lt) {
    for (auto& kv : lt->plans) {
        cublasLtMatmulDescDestroy(kv.second.op);
        cublasLtMatrixLayoutDestroy(kv.second.lw);
        cublasLtMatrixLayoutDestroy(kv.second.la);
        cublasLtMatrixLayoutDestroy(kv.second.lc);
    }
    cudaFree(lt->ws);
    cublasLtDestroy(lt->h);
    delete lt;
}

// modes: 0 bf16 -> bf16, 1 int8 -> int32, 2 bf16 -> fp32, 3 fp32 -> fp32 (fp32 compute, no TF32)
static void run(Lt* lt, Plan& p, int mode, const void* A, const void* W, void* C, const cublasLtMatmulAlgo_t* algo) {
    const bool i8 = mode == 1;
    int32_t ai = 1, bi = 0;
    float af = 1.f, bf = 0.f;
    const void* alpha = i8 ? (const void*)&ai : (const void*)&af;
    const void* beta = i8 ? (const void*)&bi : (const void*)&bf;
    CKL(cublasLtMatmul(lt->h, p.op, alpha, W, p.lw, A, p.la, beta, C, p.lc, C, p.lc, algo, lt->ws, lt->wsz, lt->st));
}

static Plan& plan(Lt* lt, int M, int N, int K, int mode, const void* A, const void* W, void* C) {
    const bool i8 = mode == 1;
    auto key = std::make_tuple(M, N, K, mode);
    auto it = lt->plans.find(key);
    if (it != lt->plans.end()) return it->second;
    Plan p;
    cublasComputeType_t ct = i8 ? CUBLAS_COMPUTE_32I : CUBLAS_COMPUTE_32F;
    cudaDataType_t st = i8 ? CUDA_R_32I : CUDA_R_32F;
    cudaDataType_t ab = i8 ? CUDA_R_8I : mode == 3 ? CUDA_R_32F : CUDA_R_16BF;
    cudaDataType_t ct_out = i8 ? CUDA_R_32I : mode >= 2 ? CUDA_R_32F : CUDA_R_16BF;
    CKL(cublasLtMatmulDescCreate(&p.op, ct, st));
    cublasOperation_t ta = CUBLAS_OP_T, tb = CUBLAS_OP_N;
    CKL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSA, &ta, sizeof(ta)));
    CKL(cublasLtMatmulDescSetAttribute(p.op, CUBLASLT_MATMUL_DESC_TRANSB, &tb, sizeof(tb)));
    CKL(cublasLtMatrixLayoutCreate(&p.lw, ab, K, N, K));  // W [N,K] row-major = col-major K x N
    CKL(cublasLtMatrixLayoutCreate(&p.la, ab, K, M, K));  // A [M,K] row-major = col-major K x M
    CKL(cublasLtMatrixLayoutCreate(&p.lc, ct_out, N, M, N));
    cublasLtMatmulPreference_t pref;
    CKL(cublasLtMatmulPreferenceCreate(&pref));
    CKL(cublasLtMatmulPreferenceSetAttribute(pref, CUBLASLT_MATMUL_PREF_MAX_WORKSPACE_BYTES, &lt->wsz, sizeof(lt->wsz)));
    cublasLtMatmulHeuristicResult_t res[8];
    int n = 0;
    CKL(cublasLtMatmulAlgoGetHeuristic(lt->h, p.op, p.lw, p.la, p.lc, p.lc, pref, 8, res, &n));
    cublasLtMatmulPreferenceDestroy(pref);
    if (n == 0) DIE("no cublasLt algo for M=%d N=%d K=%d mode=%d", M, N, K, mode);
    int best = 0;
    // int8 accumulates exactly, so any algo gives the same result: time candidates for big int8 shapes.
    // bf16 keeps the heuristic's first choice so the same seed always gives the same image.
    if (n > 1 && i8 && (double)M * N * K > 1e9) {
        cudaEvent_t e0, e1;
        CK(cudaEventCreate(&e0));
        CK(cudaEventCreate(&e1));
        float bt = 1e30f;
        for (int i = 0; i < n; i++) {
            run(lt, p, mode, A, W, C, &res[i].algo);  // warm
            CK(cudaEventRecord(e0, lt->st));
            for (int r = 0; r < 3; r++) run(lt, p, mode, A, W, C, &res[i].algo);
            CK(cudaEventRecord(e1, lt->st));
            CK(cudaEventSynchronize(e1));
            float ms;
            CK(cudaEventElapsedTime(&ms, e0, e1));
            if (ms < bt) { bt = ms; best = i; }
        }
        cudaEventDestroy(e0);
        cudaEventDestroy(e1);
    }
    p.algo = res[best].algo;
    return lt->plans.emplace(key, p).first->second;
}

void gemm_bf16(Lt* lt, int M, int N, int K, const bf16* A, const bf16* W, bf16* C) {
    Plan& p = plan(lt, M, N, K, 0, A, W, C);
    run(lt, p, 0, A, W, C, &p.algo);
}

void gemm_i8(Lt* lt, int M, int N, int K, const int8_t* A, const int8_t* W, int32_t* C) {
    Plan& p = plan(lt, M, N, K, 1, A, W, C);
    run(lt, p, 1, A, W, C, &p.algo);
}

void gemm_bf16_f32(Lt* lt, int M, int N, int K, const bf16* A, const bf16* W, float* C) {
    Plan& p = plan(lt, M, N, K, 2, A, W, C);
    run(lt, p, 2, A, W, C, &p.algo);
}

void gemm_f32(Lt* lt, int M, int N, int K, const float* A, const float* W, float* C) {
    Plan& p = plan(lt, M, N, K, 3, A, W, C);
    run(lt, p, 3, A, W, C, &p.algo);
}
