#include "cuda_utils.cuh"

// LayerNorm with elementwise_affine=False:
// out = (x - mean(x)) / sqrt(var(x) + eps)
// Input/output: BF16 [N, D]. Internal FP32.

__global__ void layer_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int D,
    float eps
) {
    int row = blockIdx.x;
    const __nv_bfloat16* xr = x + (int64_t)row * D;
    __nv_bfloat16* yr = out + (int64_t)row * D;

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += bf16_to_float(xr[i]);
    }

    sum = warp_reduce_sum(sum);
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    if (lane == 0) shared[warp] = sum;
    __syncthreads();
    if (warp == 0) {
        sum = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        sum = warp_reduce_sum(sum);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = sum;
    __syncthreads();
    float mean = shared[0] / D;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(xr[i]) - mean;
        var_sum += v * v;
    }

    var_sum = warp_reduce_sum(var_sum);
    if (lane == 0) shared[warp] = var_sum;
    __syncthreads();
    if (warp == 0) {
        var_sum = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        var_sum = warp_reduce_sum(var_sum);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = var_sum;
    __syncthreads();
    float inv_std = rsqrtf(shared[0] / D + eps);

    // Normalize
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = (bf16_to_float(xr[i]) - mean) * inv_std;
        yr[i] = float_to_bf16(v);
    }
}

void layer_norm_bf16(
    const __nv_bfloat16* x,
    __nv_bfloat16* out,
    int N, int D,
    float eps,
    cudaStream_t stream
) {
    int threads = (D <= 1024) ? ((D + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    layer_norm_bf16_kernel<<<N, threads, 0, stream>>>(x, out, D, eps);
}
