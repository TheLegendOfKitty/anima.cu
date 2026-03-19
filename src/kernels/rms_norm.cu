#include "cuda_utils.cuh"

// RMS normalization: x = x * rsqrt(mean(x^2) + eps)
// Optionally scales by weight.
// Input/output: BF16 [N, D], weight: BF16 [D] or nullptr
// Internal accumulation in FP32.
// Vectorized with BF16x2 loads/stores for 2x memory throughput.

__global__ void rms_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,  // nullable
    __nv_bfloat16* __restrict__ out,
    int D,
    float eps
) {
    int row = blockIdx.x;
    int D2 = D / 2;
    const __nv_bfloat162* xr2 = reinterpret_cast<const __nv_bfloat162*>(x + (int64_t)row * D);
    __nv_bfloat162* yr2 = reinterpret_cast<__nv_bfloat162*>(out + (int64_t)row * D);

    // Pass 1: Compute sum of squares (vectorized BF16x2 loads)
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D2; i += blockDim.x) {
        __nv_bfloat162 val = xr2[i];
        float v0 = __low2float(val), v1 = __high2float(val);
        sum_sq = __fmaf_rn(v0, v0, sum_sq);
        sum_sq = __fmaf_rn(v1, v1, sum_sq);
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
    if (threadIdx.x == 0) shared[0] = sum_sq;
    __syncthreads();

    float rms_inv = rsqrtf(shared[0] / D + eps);

    // Pass 2: Normalize and optionally scale (vectorized)
    if (weight) {
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(weight);
        for (int i = threadIdx.x; i < D2; i += blockDim.x) {
            __nv_bfloat162 val = xr2[i];
            __nv_bfloat162 wv = w2[i];
            float v0 = __low2float(val) * rms_inv * __low2float(wv);
            float v1 = __high2float(val) * rms_inv * __high2float(wv);
            yr2[i] = __floats2bfloat162_rn(v0, v1);
        }
    } else {
        for (int i = threadIdx.x; i < D2; i += blockDim.x) {
            __nv_bfloat162 val = xr2[i];
            float v0 = __low2float(val) * rms_inv;
            float v1 = __high2float(val) * rms_inv;
            yr2[i] = __floats2bfloat162_rn(v0, v1);
        }
    }
}

// Multi-row variant: packs multiple rows per CTA for small D.
// Each warp handles one row independently (warp-level reduction only, no shared memory).
// For D=128 (head_dim): 32 rows/block × 32 threads/warp = 1024 threads.
__global__ void rms_norm_multirow_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,  // nullable
    __nv_bfloat16* __restrict__ out,
    int N, int D,
    float eps
) {
    int warp_id = threadIdx.x / 32;
    int lane = threadIdx.x % 32;
    int rows_per_block = blockDim.x / 32;
    int row = blockIdx.x * rows_per_block + warp_id;
    if (row >= N) return;

    int D2 = D / 2;
    const __nv_bfloat162* xr2 = reinterpret_cast<const __nv_bfloat162*>(x + (int64_t)row * D);
    __nv_bfloat162* yr2 = reinterpret_cast<__nv_bfloat162*>(out + (int64_t)row * D);

    // Pass 1: sum of squares (each lane strides over D2)
    float sum_sq = 0.0f;
    for (int i = lane; i < D2; i += 32) {
        __nv_bfloat162 val = xr2[i];
        float v0 = __low2float(val), v1 = __high2float(val);
        sum_sq = __fmaf_rn(v0, v0, sum_sq);
        sum_sq = __fmaf_rn(v1, v1, sum_sq);
    }
    sum_sq = warp_reduce_sum(sum_sq);
    float rms_inv = rsqrtf(sum_sq / D + eps);

    // Pass 2: normalize + scale
    if (weight) {
        const __nv_bfloat162* w2 = reinterpret_cast<const __nv_bfloat162*>(weight);
        for (int i = lane; i < D2; i += 32) {
            __nv_bfloat162 val = xr2[i];
            __nv_bfloat162 wv = w2[i];
            float v0 = __low2float(val) * rms_inv * __low2float(wv);
            float v1 = __high2float(val) * rms_inv * __high2float(wv);
            yr2[i] = __floats2bfloat162_rn(v0, v1);
        }
    } else {
        for (int i = lane; i < D2; i += 32) {
            __nv_bfloat162 val = xr2[i];
            float v0 = __low2float(val) * rms_inv;
            float v1 = __high2float(val) * rms_inv;
            yr2[i] = __floats2bfloat162_rn(v0, v1);
        }
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
    int D2 = D / 2;

    int threads = (D2 <= 1024) ? ((D2 + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    rms_norm_bf16_kernel<<<N, threads, 0, stream>>>(x, weight, out, D, eps);
}
