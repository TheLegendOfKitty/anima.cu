#include "cuda_utils.cuh"

// Fused AdaLN: out = rms_norm(x, eps) * (1 + scale) + shift
// x:     BF16 [N, D]
// scale: BF16 [N, D] (broadcast from [N, 1, D])
// shift: BF16 [N, D]
// out:   BF16 [N, D]
// Internal FP32 accumulation.

// Vectorized with BF16x2 loads/stores.
__global__ void adaln_rms_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_bfloat16* __restrict__ out,
    int D,
    float eps
) {
    int row = blockIdx.x;
    int D2 = D / 2;
    const __nv_bfloat162* xr2 = reinterpret_cast<const __nv_bfloat162*>(x + (int64_t)row * D);
    const __nv_bfloat162* sr2 = reinterpret_cast<const __nv_bfloat162*>(scale + (int64_t)row * D);
    const __nv_bfloat162* hr2 = reinterpret_cast<const __nv_bfloat162*>(shift + (int64_t)row * D);
    __nv_bfloat162* yr2 = reinterpret_cast<__nv_bfloat162*>(out + (int64_t)row * D);

    // RMS norm: compute sum of squares (vectorized)
    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D2; i += blockDim.x) {
        __nv_bfloat162 val = xr2[i];
        float v0 = __low2float(val), v1 = __high2float(val);
        sum_sq = __fmaf_rn(v0, v0, sum_sq);
        sum_sq = __fmaf_rn(v1, v1, sum_sq);
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

    // Apply: norm(x) * (1 + scale) + shift (vectorized)
    for (int i = threadIdx.x; i < D2; i += blockDim.x) {
        __nv_bfloat162 xv = xr2[i], sv = sr2[i], hv = hr2[i];
        float x0 = __low2float(xv) * rms_inv, x1 = __high2float(xv) * rms_inv;
        float r0 = __fmaf_rn(x0, 1.0f + __low2float(sv), __low2float(hv));
        float r1 = __fmaf_rn(x1, 1.0f + __high2float(sv), __high2float(hv));
        yr2[i] = __floats2bfloat162_rn(r0, r1);
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
    int D2 = D / 2;
    int threads = (D2 <= 1024) ? ((D2 + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    adaln_rms_norm_bf16_kernel<<<N, threads, 0, stream>>>(x, scale, shift, out, D, eps);
}
