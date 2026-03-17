#include "cuda_utils.cuh"
#include <math_constants.h>

// GELU(tanh) approximation:
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// BF16 in/out, FP32 compute.

static constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2/pi)
static constexpr float GELU_COEFF = 0.044715f;

__global__ void gelu_tanh_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float v = bf16_to_float(x[idx]);
    float inner = SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v);
    float result = 0.5f * v * (1.0f + tanhf(inner));
    out[idx] = float_to_bf16(result);
}

// Fused: out = gelu_tanh(x + bias)
__global__ void gelu_tanh_bias_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int64_t N,
    int D  // bias dimension (last dim)
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float v = bf16_to_float(x[idx]) + bf16_to_float(bias[idx % D]);
    float inner = SQRT_2_OVER_PI * (v + GELU_COEFF * v * v * v);
    float result = 0.5f * v * (1.0f + tanhf(inner));
    out[idx] = float_to_bf16(result);
}

void gelu_tanh_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t N, cudaStream_t stream) {
    int threads = 256;
    int blocks = ceil_div((int)N, threads);
    gelu_tanh_bf16_kernel<<<blocks, threads, 0, stream>>>(x, out, N);
}

void gelu_tanh_bias_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* bias,
    __nv_bfloat16* out, int64_t N, int D, cudaStream_t stream
) {
    int threads = 256;
    int blocks = ceil_div((int)N, threads);
    gelu_tanh_bias_bf16_kernel<<<blocks, threads, 0, stream>>>(x, bias, out, N, D);
}
