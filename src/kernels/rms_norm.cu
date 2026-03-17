#include "cuda_utils.cuh"

// RMS normalization: x = x * rsqrt(mean(x^2) + eps)
// Optionally scales by weight.
// Input/output: BF16 [N, D], weight: BF16 [D] or nullptr
// Internal accumulation in FP32.

// One block per row. Supports D up to ~8192 with 256 threads.
__global__ void rms_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,  // nullable
    __nv_bfloat16* __restrict__ out,
    int D,
    float eps
) {
    int row = blockIdx.x;
    const __nv_bfloat16* xr = x + (int64_t)row * D;
    __nv_bfloat16* yr = out + (int64_t)row * D;

    // Compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(xr[i]);
        sum_sq += v * v;
    }

    // Block-wide reduction
    sum_sq = warp_reduce_sum(sum_sq);

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;

    if (lane == 0) shared[warp] = sum_sq;
    __syncthreads();

    if (warp == 0) {
        sum_sq = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        sum_sq = warp_reduce_sum(sum_sq);
    }
    __syncthreads();

    // Broadcast
    if (threadIdx.x == 0) shared[0] = sum_sq;
    __syncthreads();
    sum_sq = shared[0];

    float rms_inv = rsqrtf(sum_sq / D + eps);

    // Normalize and optionally scale
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(xr[i]) * rms_inv;
        if (weight) v *= bf16_to_float(weight[i]);
        yr[i] = float_to_bf16(v);
    }
}

void rms_norm_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    __nv_bfloat16* out,
    int N, int D,
    float eps,
    cudaStream_t stream
) {
    int threads = (D <= 1024) ? ((D + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    rms_norm_bf16_kernel<<<N, threads, 0, stream>>>(x, weight, out, D, eps);
}
