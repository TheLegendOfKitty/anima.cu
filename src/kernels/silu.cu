#include "cuda_utils.cuh"

// SiLU: silu(x) = x * sigmoid(x) = x / (1 + exp(-x))
// BF16 in/out, FP32 compute.
// Vectorized with BF16x2 loads/stores.

__global__ void silu_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int64_t N2  // N / 2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;

    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);

    __nv_bfloat162 val = x2[idx];
    float v0 = __low2float(val), v1 = __high2float(val);
    float r0 = v0 / (1.0f + expf(-v0));
    float r1 = v1 / (1.0f + expf(-v1));
    out2[idx] = __floats2bfloat162_rn(r0, r1);
}

// Fused: out = silu(x + bias)
__global__ void silu_bias_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int64_t N2,
    int D2  // D / 2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;

    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const __nv_bfloat162* bias2 = reinterpret_cast<const __nv_bfloat162*>(bias);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);

    __nv_bfloat162 xv = x2[idx];
    __nv_bfloat162 bv = bias2[idx % D2];
    float v0 = __low2float(xv) + __low2float(bv);
    float v1 = __high2float(xv) + __high2float(bv);
    float r0 = v0 / (1.0f + expf(-v0));
    float r1 = v1 / (1.0f + expf(-v1));
    out2[idx] = __floats2bfloat162_rn(r0, r1);
}

void silu_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t N, cudaStream_t stream) {
    int64_t N2 = N / 2;
    int threads = 256;
    int blocks = ceil_div((int)N2, threads);
    silu_bf16_kernel<<<blocks, threads, 0, stream>>>(x, out, N2);
}

void silu_bias_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* bias,
    __nv_bfloat16* out, int64_t N, int D, cudaStream_t stream
) {
    int64_t N2 = N / 2;
    int threads = 256;
    int blocks = ceil_div((int)N2, threads);
    silu_bias_bf16_kernel<<<blocks, threads, 0, stream>>>(x, bias, out, N2, D / 2);
}
