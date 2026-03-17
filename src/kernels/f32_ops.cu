// Native F32 kernels for VAE decode (avoids BF16 round-trips)
#include "../cuda_utils.cuh"
#include <cmath>

__global__ void silu_f32_kernel(const float* x, float* out, int64_t N) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    float v = x[i];
    out[i] = v / (1.0f + expf(-v));
}
void silu_f32(const float* x, float* out, int64_t N, cudaStream_t s) {
    int t = 256; silu_f32_kernel<<<(int)((N+t-1)/t), t, 0, s>>>(x, out, N);
}

__global__ void add_f32_kernel(const float* a, const float* b, float* out, int64_t N) {
    int64_t i = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (i >= N) return;
    out[i] = a[i] + b[i];
}
void add_f32(const float* a, const float* b, float* out, int64_t N, cudaStream_t s) {
    int t = 256; add_f32_kernel<<<(int)((N+t-1)/t), t, 0, s>>>(a, b, out, N);
}

// Pixel norm: RMS normalize over C channels per spatial position
__global__ void pixel_norm_f32_kernel(float* x, int C, int spatial, float eps) {
    int s = blockIdx.x * blockDim.x + threadIdx.x;
    if (s >= spatial) return;
    float sum_sq = 0.0f;
    for (int c = 0; c < C; c++) { float v = x[(int64_t)c * spatial + s]; sum_sq += v * v; }
    float inv = 1.0f / sqrtf(sum_sq / (float)C + eps);
    for (int c = 0; c < C; c++) { int64_t idx = (int64_t)c * spatial + s; x[idx] *= inv; }
}
void pixel_norm_f32(float* x, int B, int C, int spatial, float eps, cudaStream_t s) {
    int t = 256;
    for (int b = 0; b < B; b++)
        pixel_norm_f32_kernel<<<(spatial+t-1)/t, t, 0, s>>>(x + (int64_t)b*C*spatial, C, spatial, eps);
}

// Channel scale: out[c,s] = x[c,s] * scale[c]
__global__ void channel_scale_f32_kernel(const float* x, const float* scale, float* out, int C, int spatial) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * spatial) return;
    out[idx] = x[idx] * scale[idx / spatial];
}
void channel_scale_f32(const float* x, const float* scale, float* out, int B, int C, int spatial, cudaStream_t s) {
    int t = 256, total = C * spatial;
    for (int b = 0; b < B; b++)
        channel_scale_f32_kernel<<<(total+t-1)/t, t, 0, s>>>(x+(int64_t)b*total, scale, out+(int64_t)b*total, C, spatial);
}

// Nearest upsample 2x
__global__ void nearest_upsample_2x_f32_kernel(const float* in, float* out, int C, int H, int W, int H2, int W2) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= C * H2 * W2) return;
    int w2 = idx % W2, h2 = (idx / W2) % H2, c = idx / (H2 * W2);
    out[idx] = in[(int64_t)c * H * W + (h2/2) * W + (w2/2)];
}
void nearest_upsample_2x_f32(const float* in, float* out, int B, int C, int H, int W, cudaStream_t s) {
    int H2=H*2, W2=W*2, total=C*H2*W2, t=256;
    for (int b = 0; b < B; b++)
        nearest_upsample_2x_f32_kernel<<<(total+t-1)/t, t, 0, s>>>(in+(int64_t)b*C*H*W, out+(int64_t)b*total, C, H, W, H2, W2);
}
