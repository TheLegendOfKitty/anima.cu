#include "cuda_utils.cuh"
#include <math_constants.h>

// GELU(tanh) approximation:
// gelu(x) = 0.5 * x * (1 + tanh(sqrt(2/pi) * (x + 0.044715 * x^3)))
// BF16 in/out, FP32 compute.
// Vectorized with BF16x2 loads/stores for 2x memory throughput.

static constexpr float SQRT_2_OVER_PI = 0.7978845608028654f;  // sqrt(2/pi)
static constexpr float GELU_COEFF = 0.044715f;

__global__ void gelu_tanh_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int64_t N2  // N / 2 (number of BF16x2 pairs)
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;

    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);

    __nv_bfloat162 val = x2[idx];
    float v0 = __low2float(val), v1 = __high2float(val);

    float v0_sq = v0 * v0, v1_sq = v1 * v1;
    float inner0 = SQRT_2_OVER_PI * __fmaf_rn(GELU_COEFF * v0_sq, v0, v0);
    float inner1 = SQRT_2_OVER_PI * __fmaf_rn(GELU_COEFF * v1_sq, v1, v1);
    float half_v0 = 0.5f * v0, half_v1 = 0.5f * v1;
    float r0 = __fmaf_rn(half_v0, tanhf(inner0), half_v0);
    float r1 = __fmaf_rn(half_v1, tanhf(inner1), half_v1);

    out2[idx] = __floats2bfloat162_rn(r0, r1);
}

// Fused: out = gelu_tanh(x + bias)
__global__ void gelu_tanh_bias_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int64_t N2,  // N / 2
    int D2       // D / 2 (bias dimension / 2)
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

    float v0_sq = v0 * v0, v1_sq = v1 * v1;
    float inner0 = SQRT_2_OVER_PI * __fmaf_rn(GELU_COEFF * v0_sq, v0, v0);
    float inner1 = SQRT_2_OVER_PI * __fmaf_rn(GELU_COEFF * v1_sq, v1, v1);
    float half_v0 = 0.5f * v0, half_v1 = 0.5f * v1;
    float r0 = __fmaf_rn(half_v0, tanhf(inner0), half_v0);
    float r1 = __fmaf_rn(half_v1, tanhf(inner1), half_v1);

    out2[idx] = __floats2bfloat162_rn(r0, r1);
}

void gelu_tanh_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t N, cudaStream_t stream) {
    int64_t N2 = N / 2;
    int threads = 256;
    int blocks = ceil_div((int)N2, threads);
    gelu_tanh_bf16_kernel<<<blocks, threads, 0, stream>>>(x, out, N2);
}

void gelu_tanh_bias_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* bias,
    __nv_bfloat16* out, int64_t N, int D, cudaStream_t stream
) {
    int64_t N2 = N / 2;
    int threads = 256;
    int blocks = ceil_div((int)N2, threads);
    gelu_tanh_bias_bf16_kernel<<<blocks, threads, 0, stream>>>(x, bias, out, N2, D / 2);
}
