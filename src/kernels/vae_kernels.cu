/*
 * VAE-specific CUDA kernels:
 *   - nearest_upsample_2x: 2x nearest-neighbor spatial upsampling
 *   - channel_scale: per-channel multiplicative scale for [B, C, H, W]
 */

#include "kernels.h"
#include "../cuda_utils.cuh"

// ========================= Pixel Norm (RMS norm over channel dim) =========================

// For each spatial position, normalize over C channels:
// out[b,c,s] = x[b,c,s] / sqrt(mean_c(x[b,:,s]^2) + eps)
// Equivalent to F.normalize(x, dim=channel) * sqrt(C)
__global__ void pixel_norm_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int C, int spatial, float eps) {
    // Each thread handles one spatial position
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= spatial) return;

    // Compute sum of squares over C channels
    float sum_sq = 0.0f;
    for (int c = 0; c < C; c++) {
        float v = __bfloat162float(x[(int64_t)c * spatial + s]);
        sum_sq += v * v;
    }

    // RMS = sqrt(mean(x^2)) = sqrt(sum(x^2) / C)
    float rms = sqrtf(sum_sq / (float)C + eps);
    float inv_rms = 1.0f / rms;

    // Normalize
    for (int c = 0; c < C; c++) {
        int64_t idx = (int64_t)c * spatial + s;
        float v = __bfloat162float(x[idx]);
        out[idx] = __float2bfloat16(v * inv_rms);
    }
}

void pixel_norm_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    int B, int C, int spatial, float eps, cudaStream_t stream) {
    int threads = 256;
    for (int b = 0; b < B; b++) {
        int blocks = (spatial + threads - 1) / threads;
        pixel_norm_kernel<<<blocks, threads, 0, stream>>>(
            x + (int64_t)b * C * spatial,
            out + (int64_t)b * C * spatial,
            C, spatial, eps);
    }
}

// ========================= Fused RMS Norm Channel (norm + gamma in one pass) =========================

// For each spatial position, normalize over C channels AND multiply by gamma
// in a single pass to avoid BF16 roundtrip between norm and scale.
// out[c,s] = (x[c,s] / sqrt(mean_c(x^2) + eps)) * gamma[c]
__global__ void rms_norm_channel_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ gamma,
    __nv_bfloat16* __restrict__ out,
    int C, int spatial, float eps) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= spatial) return;

    float sum_sq = 0.0f;
    for (int c = 0; c < C; c++) {
        float v = __bfloat162float(x[(int64_t)c * spatial + s]);
        sum_sq += v * v;
    }
    float rms = rsqrtf(sum_sq / (float)C + eps);

    for (int c = 0; c < C; c++) {
        int64_t idx = (int64_t)c * spatial + s;
        float v = __bfloat162float(x[idx]) * rms;
        float g = __bfloat162float(gamma[c]);
        out[idx] = __float2bfloat16(v * g);
    }
}

void rms_norm_channel_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gamma,
    __nv_bfloat16* out, int B, int C, int spatial,
    float eps, cudaStream_t stream) {
    int threads = 256;
    for (int b = 0; b < B; b++) {
        int blocks = (spatial + threads - 1) / threads;
        rms_norm_channel_kernel<<<blocks, threads, 0, stream>>>(
            x + (int64_t)b * C * spatial,
            gamma,
            out + (int64_t)b * C * spatial,
            C, spatial, eps);
    }
}

// ========================= Nearest Upsample 2x =========================

// Input: [B, C, H, W]  ->  Output: [B, C, 2H, 2W]
__global__ void nearest_upsample_2x_kernel(
    const __nv_bfloat16* __restrict__ in,
    __nv_bfloat16* __restrict__ out,
    int C, int H, int W, int H2, int W2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * H2 * W2;
    if (idx >= total) return;

    int w2 = idx % W2;
    int h2 = (idx / W2) % H2;
    int c  = idx / (H2 * W2);

    int h = h2 / 2;
    int w = w2 / 2;

    out[idx] = in[(int64_t)c * H * W + h * W + w];
}

void nearest_upsample_2x_bf16(
    const __nv_bfloat16* in, __nv_bfloat16* out,
    int B, int C, int H, int W, cudaStream_t stream) {
    int H2 = H * 2, W2 = W * 2;
    int total_per_batch = C * H2 * W2;
    int threads = 256;

    for (int b = 0; b < B; b++) {
        int blocks = (total_per_batch + threads - 1) / threads;
        nearest_upsample_2x_kernel<<<blocks, threads, 0, stream>>>(
            in + (int64_t)b * C * H * W,
            out + (int64_t)b * C * H2 * W2,
            C, H, W, H2, W2);
    }
}

// ========================= Channel Scale =========================

// out[b, c, s] = x[b, c, s] * scale[c]
// x: [B, C, spatial], scale: [C]
__global__ void channel_scale_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    __nv_bfloat16* __restrict__ out,
    int C, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * spatial;
    if (idx >= total) return;

    int c = idx / spatial;
    float xv = __bfloat162float(x[idx]);
    float sv = __bfloat162float(scale[c]);
    out[idx] = __float2bfloat16(xv * sv);
}

void channel_scale_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* scale,
    __nv_bfloat16* out, int B, int C, int spatial,
    cudaStream_t stream) {
    int total_per_batch = C * spatial;
    int threads = 256;

    for (int b = 0; b < B; b++) {
        int blocks = (total_per_batch + threads - 1) / threads;
        channel_scale_kernel<<<blocks, threads, 0, stream>>>(
            x + (int64_t)b * total_per_batch,
            scale,
            out + (int64_t)b * total_per_batch,
            C, spatial);
    }
}

// ========================= Channel Scale + Shift =========================

// out[b, c, s] = x[b, c, s] * scale[c] + shift[c]
__global__ void channel_scale_shift_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_bfloat16* __restrict__ out,
    int C, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = C * spatial;
    if (idx >= total) return;

    int c = idx / spatial;
    float xv = __bfloat162float(x[idx]);
    float sv = __bfloat162float(scale[c]);
    float hv = __bfloat162float(shift[c]);
    out[idx] = __float2bfloat16(xv * sv + hv);
}

void channel_scale_shift_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* out, int B, int C, int spatial,
    cudaStream_t stream) {
    int total_per_batch = C * spatial;
    int threads = 256;

    for (int b = 0; b < B; b++) {
        int blocks = (total_per_batch + threads - 1) / threads;
        channel_scale_shift_kernel<<<blocks, threads, 0, stream>>>(
            x + (int64_t)b * total_per_batch,
            scale, shift,
            out + (int64_t)b * total_per_batch,
            C, spatial);
    }
}
