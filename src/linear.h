#pragma once

#include "tensor.h"
#include <cublas_v2.h>
#include <cublasLt.h>

// Dense linear layer: y = x @ W^T + bias
// Weights: BF16 [out_features, in_features]
// Bias:    BF16 [out_features] (optional)
// Input:   BF16 [*, in_features]
// Output:  BF16 [*, out_features]
class Linear {
public:
    Linear() = default;

    // Load from safetensors
    void load(Tensor weight, Tensor bias);

    // Forward: out must be pre-allocated [M, out_features]
    // x: [M, in_features] where M = product of all leading dims
    void forward(cublasHandle_t handle,
                 const __nv_bfloat16* x, __nv_bfloat16* out,
                 int M,  // rows of x (batch * seq_len)
                 cudaStream_t stream = 0) const;

    // Forward with fused GELU epilogue via cublasLt:
    // out = GELU(x @ W^T + bias)
    // Eliminates the separate GELU kernel and extra memory round-trip.
    void forward_gelu(cublasHandle_t handle,
                      const __nv_bfloat16* x, __nv_bfloat16* out,
                      int M,
                      cudaStream_t stream = 0) const;

    int in_features()  const { return in_; }
    int out_features() const { return out_; }
    bool has_bias()    const { return !bias_.empty(); }

    const Tensor& weight() const { return weight_; }
    const Tensor& bias()   const { return bias_; }

private:
    Tensor weight_;  // [out, in]
    Tensor bias_;    // [out] or empty
    int in_  = 0;
    int out_ = 0;
};
