#include "cuda_utils.cuh"

// SiLU: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// BF16 in/out, FP32 compute.

__global__ void silu_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float v = bf16_to_float(x[idx]);
    float result = v / (1.0f + expf(-v));
    out[idx] = float_to_bf16(result);
}

// Fused: out = silu(x + bias)
__global__ void silu_bias_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int64_t N,
    int D
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float v = bf16_to_float(x[idx]) + bf16_to_float(bias[idx % D]);
    float result = v / (1.0f + expf(-v));
    out[idx] = float_to_bf16(result);
}

void silu_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t N, cudaStream_t stream) {
    int threads = 256;
    int blocks = ceil_div((int)N, threads);
    silu_bf16_kernel<<<blocks, threads, 0, stream>>>(x, out, N);
}

void silu_bias_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* bias,
    __nv_bfloat16* out, int64_t N, int D, cudaStream_t stream
) {
    int threads = 256;
    int blocks = ceil_div((int)N, threads);
    silu_bias_bf16_kernel<<<blocks, threads, 0, stream>>>(x, bias, out, N, D);
}
