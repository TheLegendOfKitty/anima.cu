#include "cuda_utils.cuh"

// Patchify: transpose [B, C, F*H*W] -> [B, F*H*W, C]
// With patch_size=1, this is just a 2D transpose of the last two dims per batch.
// BF16 in/out.

__global__ void patchify_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,    // [B, C, T]
    __nv_bfloat16* __restrict__ out,         // [B, T, C]
    int C, int T
) {
    int b = blockIdx.z;
    int t = blockIdx.x * blockDim.x + threadIdx.x;
    int c = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= T || c >= C) return;

    int64_t src_idx = (int64_t)b * C * T + (int64_t)c * T + t;
    int64_t dst_idx = (int64_t)b * T * C + (int64_t)t * C + c;
    out[dst_idx] = x[src_idx];
}

void patchify_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    int B, int C, int T, cudaStream_t stream
) {
    dim3 threads(16, 16);
    dim3 blocks(ceil_div(T, 16), ceil_div(C, 16), B);
    patchify_bf16_kernel<<<blocks, threads, 0, stream>>>(x, out, C, T);
}

// Unpatchify: transpose [B, T, C] -> [B, C, T]
// Same kernel, just swapped semantics.
__global__ void unpatchify_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,    // [B, T, C]
    __nv_bfloat16* __restrict__ out,         // [B, C, T]
    int T, int C
) {
    int b = blockIdx.z;
    int c = blockIdx.x * blockDim.x + threadIdx.x;
    int t = blockIdx.y * blockDim.y + threadIdx.y;
    if (t >= T || c >= C) return;

    int64_t src_idx = (int64_t)b * T * C + (int64_t)t * C + c;
    int64_t dst_idx = (int64_t)b * C * T + (int64_t)c * T + t;
    out[dst_idx] = x[src_idx];
}

void unpatchify_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    int B, int T, int C, cudaStream_t stream
) {
    dim3 threads(16, 16);
    dim3 blocks(ceil_div(C, 16), ceil_div(T, 16), B);
    unpatchify_bf16_kernel<<<blocks, threads, 0, stream>>>(x, out, T, C);
}
