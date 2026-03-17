#include "cuda_utils.cuh"

// Row-wise softmax for attention scores.
// Input:  BF16 [N, D]  (each row is softmax'd independently)
// Output: BF16 [N, D]
// Uses online max tracking for numerical stability, FP32 accumulation.

__global__ void softmax_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int D
) {
    int row = blockIdx.x;
    const __nv_bfloat16* xr = x + (int64_t)row * D;
    __nv_bfloat16* yr = out + (int64_t)row * D;

    // Pass 1: find max
    float max_val = -1e30f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        max_val = fmaxf(max_val, bf16_to_float(xr[i]));
    }

    max_val = warp_reduce_max(max_val);
    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    if (lane == 0) shared[warp] = max_val;
    __syncthreads();
    if (warp == 0) {
        max_val = (lane < (blockDim.x + 31) / 32) ? shared[lane] : -1e30f;
        max_val = warp_reduce_max(max_val);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = max_val;
    __syncthreads();
    max_val = shared[0];

    // Pass 2: compute exp sum
    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum_exp += expf(bf16_to_float(xr[i]) - max_val);
    }

    sum_exp = warp_reduce_sum(sum_exp);
    if (lane == 0) shared[warp] = sum_exp;
    __syncthreads();
    if (warp == 0) {
        sum_exp = (lane < (blockDim.x + 31) / 32) ? shared[lane] : 0.0f;
        sum_exp = warp_reduce_sum(sum_exp);
    }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = sum_exp;
    __syncthreads();
    float inv_sum = 1.0f / shared[0];

    // Pass 3: normalize
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        yr[i] = float_to_bf16(expf(bf16_to_float(xr[i]) - max_val) * inv_sum);
    }
}

// Causal mask: set scores[h, i, j] = -inf for j > i (upper triangular)
// BF16 version for Qwen3
__global__ void causal_mask_bf16_kernel(
    __nv_bfloat16* __restrict__ scores, int T
) {
    int h = blockIdx.x;
    __nv_bfloat16* s = scores + (int64_t)h * T * T;
    for (int idx = threadIdx.x; idx < T * T; idx += blockDim.x) {
        int i = idx / T, j = idx % T;
        if (j > i) s[idx] = __float2bfloat16(-1e9f);
    }
}

void causal_mask_bf16(
    __nv_bfloat16* scores, int num_heads, int T, cudaStream_t stream
) {
    int threads = min(T * T, 1024);
    causal_mask_bf16_kernel<<<num_heads, threads, 0, stream>>>(scores, T);
}

// F32 softmax (for attention scores precision)
__global__ void softmax_f32_kernel(
    const float* __restrict__ x, float* __restrict__ out, int D
) {
    int row = blockIdx.x;
    const float* xr = x + (int64_t)row * D;
    float* yr = out + (int64_t)row * D;

    float max_val = -1e30f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        max_val = fmaxf(max_val, xr[i]);
    max_val = warp_reduce_max(max_val);
    __shared__ float shared[32];
    int lane = threadIdx.x % 32, warp = threadIdx.x / 32;
    if (lane == 0) shared[warp] = max_val;
    __syncthreads();
    if (warp == 0) { max_val = (lane < (blockDim.x+31)/32) ? shared[lane] : -1e30f; max_val = warp_reduce_max(max_val); }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = max_val;
    __syncthreads();
    max_val = shared[0];

    float sum_exp = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        sum_exp += expf(xr[i] - max_val);
    sum_exp = warp_reduce_sum(sum_exp);
    if (lane == 0) shared[warp] = sum_exp;
    __syncthreads();
    if (warp == 0) { sum_exp = (lane < (blockDim.x+31)/32) ? shared[lane] : 0.0f; sum_exp = warp_reduce_sum(sum_exp); }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = sum_exp;
    __syncthreads();
    float inv_sum = 1.0f / shared[0];

    for (int i = threadIdx.x; i < D; i += blockDim.x)
        yr[i] = expf(xr[i] - max_val) * inv_sum;
}

void softmax_f32(const float* x, float* out, int N, int D, cudaStream_t stream) {
    int threads = (D <= 1024) ? ((D + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    softmax_f32_kernel<<<N, threads, 0, stream>>>(x, out, D);
}

void softmax_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    int N, int D, cudaStream_t stream
) {
    int threads = (D <= 1024) ? ((D + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    softmax_bf16_kernel<<<N, threads, 0, stream>>>(x, out, D);
}
