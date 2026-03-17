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

// Fused F32 softmax -> BF16 output.
// Reads F32 input, computes softmax in F32 (numerical stability), writes BF16 directly.
// Uses online 2-pass softmax with vectorized float4 loads for memory throughput.
// One block per row, configurable threads.
__global__ void softmax_f32_to_bf16_kernel(
    const float* __restrict__ x, __nv_bfloat16* __restrict__ out, int D
) {
    int row = blockIdx.x;
    const float* xr = x + (int64_t)row * D;
    __nv_bfloat16* yr = out + (int64_t)row * D;

    // Pass 1: online fused max + sum_exp
    // Each thread tracks (max, sum_exp) and merges on the fly.
    float local_max = -1e30f;
    float local_sum = 0.0f;

    // Vectorized float4 loads for the aligned portion
    int D4 = D / 4;
    int rem = D - D4 * 4;
    const float4* xr4 = reinterpret_cast<const float4*>(xr);

    for (int i = threadIdx.x; i < D4; i += blockDim.x) {
        float4 v = xr4[i];
        float vals[4] = {v.x, v.y, v.z, v.w};
        #pragma unroll
        for (int k = 0; k < 4; k++) {
            float xi = vals[k];
            if (xi > local_max) {
                local_sum = local_sum * expf(local_max - xi) + 1.0f;
                local_max = xi;
            } else {
                local_sum += expf(xi - local_max);
            }
        }
    }
    // Handle remainder elements
    for (int i = D4 * 4 + threadIdx.x; i < D; i += blockDim.x) {
        float xi = xr[i];
        if (xi > local_max) {
            local_sum = local_sum * expf(local_max - xi) + 1.0f;
            local_max = xi;
        } else {
            local_sum += expf(xi - local_max);
        }
    }

    // Cross-warp reduction of (max, sum) using online merge
    // First: warp-level reduction
    for (int offset = 16; offset > 0; offset >>= 1) {
        float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
        float other_sum = __shfl_down_sync(0xffffffff, local_sum, offset);
        if (other_max > local_max) {
            local_sum = local_sum * expf(local_max - other_max) + other_sum;
            local_max = other_max;
        } else {
            local_sum = local_sum + other_sum * expf(other_max - local_max);
        }
    }

    // Cross-warp via shared memory
    __shared__ float s_max[32];
    __shared__ float s_sum[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int num_warps = (blockDim.x + 31) / 32;

    if (lane == 0) {
        s_max[warp] = local_max;
        s_sum[warp] = local_sum;
    }
    __syncthreads();

    if (warp == 0) {
        local_max = (lane < num_warps) ? s_max[lane] : -1e30f;
        local_sum = (lane < num_warps) ? s_sum[lane] : 0.0f;

        for (int offset = 16; offset > 0; offset >>= 1) {
            float other_max = __shfl_down_sync(0xffffffff, local_max, offset);
            float other_sum = __shfl_down_sync(0xffffffff, local_sum, offset);
            if (other_max > local_max) {
                local_sum = local_sum * expf(local_max - other_max) + other_sum;
                local_max = other_max;
            } else {
                local_sum = local_sum + other_sum * expf(other_max - local_max);
            }
        }
    }
    __syncthreads();

    if (threadIdx.x == 0) {
        s_max[0] = local_max;
        s_sum[0] = local_sum;
    }
    __syncthreads();

    float max_val = s_max[0];
    float inv_sum = 1.0f / s_sum[0];

    // Pass 2: normalize and write BF16 output (vectorized reads)
    for (int i = threadIdx.x; i < D4; i += blockDim.x) {
        float4 v = xr4[i];
        int base = i * 4;
        yr[base + 0] = __float2bfloat16(expf(v.x - max_val) * inv_sum);
        yr[base + 1] = __float2bfloat16(expf(v.y - max_val) * inv_sum);
        yr[base + 2] = __float2bfloat16(expf(v.z - max_val) * inv_sum);
        yr[base + 3] = __float2bfloat16(expf(v.w - max_val) * inv_sum);
    }
    // Handle remainder
    for (int i = D4 * 4 + threadIdx.x; i < D; i += blockDim.x) {
        yr[i] = __float2bfloat16(expf(xr[i] - max_val) * inv_sum);
    }
}

void softmax_f32_to_bf16(const float* x, __nv_bfloat16* out, int N, int D, cudaStream_t stream) {
    int threads = (D <= 1024) ? ((D + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    softmax_f32_to_bf16_kernel<<<N, threads, 0, stream>>>(x, out, D);
}
