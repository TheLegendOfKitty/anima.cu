#include "linear.h"
#include "cuda_utils.cuh"
#include "kernels/kernels.h"
#include <cublasLt.h>

// ========================= Lifecycle =========================

void Linear::GELUCache::destroy() {
    if (Adesc) { cublasLtMatrixLayoutDestroy(Adesc); Adesc = nullptr; }
    if (Bdesc) { cublasLtMatrixLayoutDestroy(Bdesc); Bdesc = nullptr; }
    if (Cdesc) { cublasLtMatrixLayoutDestroy(Cdesc); Cdesc = nullptr; }
    if (matmulDesc) { cublasLtMatmulDescDestroy(matmulDesc); matmulDesc = nullptr; }
    M = 0;
}

Linear::~Linear() {
    gelu_cache_.destroy();
}

Linear::Linear(Linear&& o) noexcept
    : weight_(std::move(o.weight_)), bias_(std::move(o.bias_)),
      in_(o.in_), out_(o.out_), gelu_cache_(o.gelu_cache_) {
    o.in_ = 0; o.out_ = 0;
    // Transfer cache ownership — zero out source so it doesn't double-destroy
    o.gelu_cache_ = {};
}

Linear& Linear::operator=(Linear&& o) noexcept {
    if (this != &o) {
        gelu_cache_.destroy();
        weight_ = std::move(o.weight_);
        bias_ = std::move(o.bias_);
        in_ = o.in_; out_ = o.out_;
        gelu_cache_ = o.gelu_cache_;
        o.in_ = 0; o.out_ = 0;
        o.gelu_cache_ = {};
    }
    return *this;
}

// ========================= Load =========================

void Linear::load(Tensor weight, Tensor bias) {
    assert(weight.ndim() == 2);
    out_ = (int)weight.dim(0);
    in_  = (int)weight.dim(1);
    weight_ = std::move(weight);
    if (!bias.empty()) {
        assert(bias.ndim() == 1 && bias.dim(0) == out_);
        bias_ = std::move(bias);
    }
    gelu_cache_.destroy();  // invalidate cache if reloaded
}

// ========================= Forward =========================

void Linear::forward(cublasHandle_t handle,
                     const __nv_bfloat16* x, __nv_bfloat16* out,
                     int M, cudaStream_t stream) const {
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    float alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out_, M, in_,
        &alpha,
        weight_.data_ptr(), CUDA_R_16BF, in_,
        x,               CUDA_R_16BF, in_,
        &beta,
        out,             CUDA_R_16BF, out_,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    if (!bias_.empty()) {
        bias_add_bf16(out, bias_.bf16_ptr(), out, M, out_, stream);
    }
}

void Linear::forward_gemm_only(cublasHandle_t handle,
                                const __nv_bfloat16* x, __nv_bfloat16* out,
                                int M, cudaStream_t stream) const {
    CUBLAS_CHECK(cublasSetStream(handle, stream));

    float alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T, CUBLAS_OP_N,
        out_, M, in_,
        &alpha,
        weight_.data_ptr(), CUDA_R_16BF, in_,
        x,               CUDA_R_16BF, in_,
        &beta,
        out,             CUDA_R_16BF, out_,
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));
}

// ========================= Forward with fused GELU (cached cublasLt) =========================

static cublasLtHandle_t get_cublaslt_handle() {
    static cublasLtHandle_t handle = nullptr;
    if (!handle) {
        cublasLtCreate(&handle);
    }
    return handle;
}

void Linear::forward_gelu(cublasHandle_t handle,
                           const __nv_bfloat16* x, __nv_bfloat16* out,
                           int M, cudaStream_t stream) const {
    cublasLtHandle_t ltHandle = get_cublaslt_handle();

    // Rebuild cache if M changed (matmulDesc + Adesc are M-independent but we
    // rebuild everything for simplicity — only happens once in steady state)
    if (gelu_cache_.M != M) {
        gelu_cache_.destroy();

        cublasLtMatmulDescCreate(&gelu_cache_.matmulDesc, CUBLAS_COMPUTE_32F, CUDA_R_32F);

        cublasOperation_t transA = CUBLAS_OP_T;
        cublasOperation_t transB = CUBLAS_OP_N;
        cublasLtMatmulDescSetAttribute(gelu_cache_.matmulDesc, CUBLASLT_MATMUL_DESC_TRANSA,
                                       &transA, sizeof(transA));
        cublasLtMatmulDescSetAttribute(gelu_cache_.matmulDesc, CUBLASLT_MATMUL_DESC_TRANSB,
                                       &transB, sizeof(transB));

        cublasLtEpilogue_t epilogue;
        if (!bias_.empty()) {
            epilogue = CUBLASLT_EPILOGUE_GELU_BIAS;
            const void* biasPtr = bias_.data_ptr();
            cublasLtMatmulDescSetAttribute(gelu_cache_.matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_POINTER,
                                           &biasPtr, sizeof(biasPtr));
            cudaDataType_t biasDt = CUDA_R_16BF;
            cublasLtMatmulDescSetAttribute(gelu_cache_.matmulDesc, CUBLASLT_MATMUL_DESC_BIAS_DATA_TYPE,
                                           &biasDt, sizeof(biasDt));
        } else {
            epilogue = CUBLASLT_EPILOGUE_GELU;
        }
        cublasLtMatmulDescSetAttribute(gelu_cache_.matmulDesc, CUBLASLT_MATMUL_DESC_EPILOGUE,
                                       &epilogue, sizeof(epilogue));

        cublasLtMatrixLayoutCreate(&gelu_cache_.Adesc, CUDA_R_16BF, in_, out_, in_);
        cublasLtMatrixLayoutCreate(&gelu_cache_.Bdesc, CUDA_R_16BF, in_, M, in_);
        cublasLtMatrixLayoutCreate(&gelu_cache_.Cdesc, CUDA_R_16BF, out_, M, out_);

        gelu_cache_.M = M;
    }

    float alpha = 1.0f, beta = 0.0f;

    cublasStatus_t status = cublasLtMatmul(
        ltHandle, gelu_cache_.matmulDesc,
        &alpha,
        weight_.data_ptr(), gelu_cache_.Adesc,
        x, gelu_cache_.Bdesc,
        &beta,
        out, gelu_cache_.Cdesc,
        out, gelu_cache_.Cdesc,
        nullptr, nullptr, 0,
        stream);

    if (status != CUBLAS_STATUS_SUCCESS) {
        fprintf(stderr, "cublasLtMatmul (GELU) failed: %d\n", (int)status);
        exit(1);
    }
}
