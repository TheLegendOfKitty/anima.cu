#include "cuda_utils.cuh"

// Fused AdaLN: out = rms_norm(x, eps) * (1 + scale) + shift
// x:     BF16 [N, D]
// scale: BF16 [N, D] (broadcast from [N, 1, D])
// shift: BF16 [N, D]
// out:   BF16 [N, D]
// Internal FP32 accumulation.

__global__ void adaln_rms_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_bfloat16* __restrict__ out,
    int D,
    float eps
) {
    int row = blockIdx.x;
    const __nv_bfloat16* xr = x + (int64_t)row * D;
    const __nv_bfloat16* sr = scale + (int64_t)row * D;
    const __nv_bfloat16* hr = shift + (int64_t)row * D;
    __nv_bfloat16* yr = out + (int64_t)row * D;

    // RMS norm: compute sum of squares
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(xr[i]);
        sum_sq += v * v;
    }

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
    if (threadIdx.x == 0) shared[0] = sum_sq;
    __syncthreads();
    float rms_inv = rsqrtf(shared[0] / D + eps);

    // Apply: norm(x) * (1 + scale) + shift
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(xr[i]) * rms_inv;
        float s = bf16_to_float(sr[i]);
        float h = bf16_to_float(hr[i]);
        yr[i] = float_to_bf16(v * (1.0f + s) + h);
    }
}

void adaln_rms_norm_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* scale,
    const __nv_bfloat16* shift,
    __nv_bfloat16* out,
    int N, int D,
    float eps,
    cudaStream_t stream
) {
    int threads = (D <= 1024) ? ((D + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    adaln_rms_norm_bf16_kernel<<<N, threads, 0, stream>>>(x, scale, shift, out, D, eps);
}
