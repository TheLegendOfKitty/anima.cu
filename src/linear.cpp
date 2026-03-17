#include "linear.h"
#include "cuda_utils.cuh"
#include "kernels/kernels.h"

void Linear::load(Tensor weight, Tensor bias) {
    assert(weight.ndim() == 2);
    out_ = (int)weight.dim(0);
    in_  = (int)weight.dim(1);
    weight_ = std::move(weight);
    if (!bias.empty()) {
        assert(bias.ndim() == 1 && bias.dim(0) == out_);
        bias_ = std::move(bias);
    }
}

void Linear::forward(cublasHandle_t handle,
                     const __nv_bfloat16* x, __nv_bfloat16* out,
                     int M, cudaStream_t stream) const {
    // y = x @ W^T + b
    // cuBLAS does: C = alpha * op(A) * op(B) + beta * C
    // We want: out[M, out] = x[M, in] @ W^T[in, out]
    //
    // cuBLAS is column-major. If we treat row-major matrices as their transposes:
    //   Row-major x[M, in]  = column-major x^T[in, M]
    //   Row-major W[out, in] = column-major W^T[in, out]
    //   Row-major out[M, out] = column-major out^T[out, M]
    //
    // We need: out^T = W^T^T @ x^T  (but we want W^T, not W)
    // Actually: out = x @ W^T  =>  out^T = W @ x^T
    // So in column-major: C[out, M] = A[out, in] @ B[in, M]
    //   A = W (stored as column-major, which is W[out, in] row-major = W^T[in, out] col-major)
    //   Wait, W is stored row-major as [out, in].
    //   In column-major, this is W^T[in, out].
    //   We need A[out, in] in column-major = op(W^T) = CUBLAS_OP_T applied to W^T gives W.
    //
    // Let's be precise:
    //   cuBLAS sees memory as column-major.
    //   W is stored in memory as row-major [out, in] = col-major layout with dims [in, out], i.e. W^T.
    //   x is stored as row-major [M, in] = col-major [in, M], i.e. x^T.
    //   We want out row-major [M, out] = x @ W^T.
    //   In col-major: out^T[out, M] = W[out, in] @ x^T^T[in, M]
    //   But x^T is what cuBLAS sees as [in, M], and x^T^T = x, so op(B) = CUBLAS_OP_N gives x^T.
    //   W in cuBLAS is seen as [in, out] (W^T). op(A) = CUBLAS_OP_T gives W[out, in].
    //
    // So: cublasGemmEx(handle, CUBLAS_OP_T, CUBLAS_OP_N, out, M, in, ...)
    //     A = W^T[in, out] with OP_T -> W[out, in]
    //     B = x^T[in, M]  with OP_N -> x^T[in, M]
    //     C = out^T[out, M]

    CUBLAS_CHECK(cublasSetStream(handle, stream));

    float alpha = 1.0f, beta = 0.0f;

    CUBLAS_CHECK(cublasGemmEx(
        handle,
        CUBLAS_OP_T,    // op(A): transpose W^T -> W
        CUBLAS_OP_N,    // op(B): no-op on x^T
        out_,           // m: rows of C = out_features
        M,              // n: cols of C = batch
        in_,            // k: inner dim = in_features
        &alpha,
        weight_.data_ptr(), CUDA_R_16BF, in_,   // A = W^T, lda = in (leading dim of col-major storage)
        x,               CUDA_R_16BF, in_,       // B = x^T, ldb = in
        &beta,
        out,             CUDA_R_16BF, out_,       // C = out^T, ldc = out
        CUBLAS_COMPUTE_32F,
        CUBLAS_GEMM_DEFAULT_TENSOR_OP
    ));

    // Add bias: out[M, out_features] += bias[out_features]
    if (!bias_.empty()) {
        bias_add_bf16(out, bias_.bf16_ptr(), out, M, out_, stream);
    }
}
