#include "cuda_utils.cuh"

// GroupNorm for 5D tensors [B, C, D, H, W] (3D convolutions).
// Groups divide channels: each group normalizes C/G channels across all spatial dims.
// With learnable weight and bias.
// BF16 in/out, FP32 accumulation.

// One block per (batch, group).
__global__ void group_norm_5d_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,  // [C]
    const __nv_bfloat16* __restrict__ bias,    // [C]
    __nv_bfloat16* __restrict__ out,
    int C, int spatial,  // D*H*W
    int groups,
    float eps
) {
    int b = blockIdx.x;
    int g = blockIdx.y;
    int cpg = C / groups;  // channels per group
    int c_start = g * cpg;

    int64_t group_size = (int64_t)cpg * spatial;
    int64_t base = (int64_t)b * C * spatial + (int64_t)c_start * spatial;

    // Compute mean
    float sum = 0.0f;
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_local = i / spatial;
        int s = i % spatial;
        int64_t idx = base + (int64_t)c_local * spatial + s;
        sum += bf16_to_float(x[idx]);
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
    float mean = shared[0] / group_size;

    // Compute variance
    float var_sum = 0.0f;
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_local = i / spatial;
        int s = i % spatial;
        int64_t idx = base + (int64_t)c_local * spatial + s;
        float v = bf16_to_float(x[idx]) - mean;
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
    float inv_std = rsqrtf(shared[0] / group_size + eps);

    // Normalize + affine
    for (int64_t i = threadIdx.x; i < group_size; i += blockDim.x) {
        int c_local = i / spatial;
        int s = i % spatial;
        int c_abs = c_start + c_local;
        int64_t idx = base + (int64_t)c_local * spatial + s;
        float v = (bf16_to_float(x[idx]) - mean) * inv_std;
        if (weight) v = v * bf16_to_float(weight[c_abs]) + bf16_to_float(bias[c_abs]);
        out[idx] = float_to_bf16(v);
    }
}

void group_norm_5d_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* weight,
    const __nv_bfloat16* bias,
    __nv_bfloat16* out,
    int B, int C, int spatial,
    int groups, float eps,
    cudaStream_t stream
) {
    dim3 grid(B, groups);
    int threads = 256;
    group_norm_5d_bf16_kernel<<<grid, threads, 0, stream>>>(
        x, weight, bias, out, C, spatial, groups, eps);
}
