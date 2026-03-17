#include "cuda_utils.cuh"

// ---- Fused residual + gate: out = x + y * gate ----
// x, y, gate, out: BF16, all [N] elements total.
// gate is broadcast along last dim (same gate per row).
// Variant 1: gate is per-element (same shape as x, y)
__global__ void residual_gate_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ out,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float xv = bf16_to_float(x[idx]);
    float yv = bf16_to_float(y[idx]);
    float gv = bf16_to_float(gate[idx]);
    out[idx] = float_to_bf16(xv + yv * gv);
}

void residual_gate_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* y, const __nv_bfloat16* gate,
    __nv_bfloat16* out, int64_t N, cudaStream_t stream
) {
    int threads = 256;
    int blocks = ceil_div((int)N, threads);
    residual_gate_bf16_kernel<<<blocks, threads, 0, stream>>>(x, y, gate, out, N);
}

// ---- Simple add: out = a + b ----
__global__ void add_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = float_to_bf16(bf16_to_float(a[idx]) + bf16_to_float(b[idx]));
}

void add_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    __nv_bfloat16* out, int64_t N, cudaStream_t stream
) {
    int threads = 256;
    int blocks = ceil_div((int)N, threads);
    add_bf16_kernel<<<blocks, threads, 0, stream>>>(a, b, out, N);
}

// ---- Scale: out = x * scale (scalar) ----
__global__ void scale_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    float scale,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = float_to_bf16(bf16_to_float(x[idx]) * scale);
}

void scale_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    float scale, int64_t N, cudaStream_t stream
) {
    int threads = 256;
    int blocks = ceil_div((int)N, threads);
    scale_bf16_kernel<<<blocks, threads, 0, stream>>>(x, out, scale, N);
}

// ---- Multiply elementwise: out = a * b ----
__global__ void mul_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = float_to_bf16(bf16_to_float(a[idx]) * bf16_to_float(b[idx]));
}

void mul_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    __nv_bfloat16* out, int64_t N, cudaStream_t stream
) {
    int threads = 256;
    int blocks = ceil_div((int)N, threads);
    mul_bf16_kernel<<<blocks, threads, 0, stream>>>(a, b, out, N);
}

// ---- Fused scale-shift: out = x * (1 + scale) + shift ----
// x, scale, shift: BF16 [N]
__global__ void scale_shift_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_bfloat16* __restrict__ out,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float v = bf16_to_float(x[idx]);
    float s = bf16_to_float(scale[idx]);
    float h = bf16_to_float(shift[idx]);
    out[idx] = float_to_bf16(v * (1.0f + s) + h);
}

void scale_shift_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* out, int64_t N, cudaStream_t stream
) {
    int threads = 256;
    int blocks = ceil_div((int)N, threads);
    scale_shift_bf16_kernel<<<blocks, threads, 0, stream>>>(x, scale, shift, out, N);
}

// ---- Broadcast bias add: out[M, D] = x[M, D] + bias[D] ----
__global__ void bias_add_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int64_t total,
    int D
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    out[idx] = float_to_bf16(bf16_to_float(x[idx]) + bf16_to_float(bias[idx % D]));
}

void bias_add_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* bias,
    __nv_bfloat16* out, int M, int D, cudaStream_t stream
) {
    int64_t total = (int64_t)M * D;
    int threads = 256;
    int blocks = ceil_div((int)total, threads);
    bias_add_bf16_kernel<<<blocks, threads, 0, stream>>>(x, bias, out, total, D);
}

// ---- Broadcast scale-shift: out[M,D] = x[M,D] * (1 + scale[D]) + shift[D] ----
__global__ void scale_shift_bcast_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_bfloat16* __restrict__ out,
    int64_t total,
    int D
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = idx % D;
    float v = bf16_to_float(x[idx]);
    float s = bf16_to_float(scale[d]);
    float h = bf16_to_float(shift[d]);
    out[idx] = float_to_bf16(v * (1.0f + s) + h);
}

void scale_shift_bcast_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* out, int M, int D, cudaStream_t stream
) {
    int64_t total = (int64_t)M * D;
    int threads = 256;
    int blocks = ceil_div((int)total, threads);
    scale_shift_bcast_bf16_kernel<<<blocks, threads, 0, stream>>>(x, scale, shift, out, total, D);
}

// ---- Broadcast residual+gate: out[M,D] = x[M,D] + y[M,D] * gate[D] ----
__global__ void residual_gate_bcast_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ out,
    int64_t total,
    int D
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total) return;
    int d = idx % D;
    float xv = bf16_to_float(x[idx]);
    float yv = bf16_to_float(y[idx]);
    float gv = bf16_to_float(gate[d]);
    out[idx] = float_to_bf16(xv + yv * gv);
}

void residual_gate_bcast_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* y, const __nv_bfloat16* gate,
    __nv_bfloat16* out, int M, int D, cudaStream_t stream
) {
    int64_t total = (int64_t)M * D;
    int threads = 256;
    int blocks = ceil_div((int)total, threads);
    residual_gate_bcast_bf16_kernel<<<blocks, threads, 0, stream>>>(x, y, gate, out, total, D);
}

// ---- to_denoised: out = sample - velocity * sigma (FP32 compute) ----
__global__ void to_denoised_bf16_kernel(
    const __nv_bfloat16* __restrict__ sample,
    const __nv_bfloat16* __restrict__ velocity,
    __nv_bfloat16* __restrict__ out,
    float sigma,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float s = bf16_to_float(sample[idx]);
    float v = bf16_to_float(velocity[idx]);
    out[idx] = float_to_bf16(s - v * sigma);
}

void to_denoised_bf16(
    const __nv_bfloat16* sample, const __nv_bfloat16* velocity,
    __nv_bfloat16* out, float sigma, int64_t N, cudaStream_t stream
) {
    int threads = 256;
    int blocks = ceil_div((int)N, threads);
    to_denoised_bf16_kernel<<<blocks, threads, 0, stream>>>(sample, velocity, out, sigma, N);
}

// ---- F32 to BF16 conversion ----
__global__ void f32_to_bf16_kernel(
    const float* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = float_to_bf16(x[idx]);
}

void f32_to_bf16(const float* x, __nv_bfloat16* out, int64_t N, cudaStream_t stream) {
    int threads = 256;
    int blocks = ceil_div((int)N, threads);
    f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(x, out, N);
}

__global__ void bf16_to_f32_kernel(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ out,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    out[idx] = bf16_to_float(x[idx]);
}

void bf16_to_f32(const __nv_bfloat16* x, float* out, int64_t N, cudaStream_t stream) {
    int threads = 256;
    int blocks = ceil_div((int)N, threads);
    bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(x, out, N);
}

// ---- Euler step: out = (sample + velocity * dt) in FP32, result BF16 ----
// velocity = (sample - denoised) / sigma
// sample = sample + velocity * dt = sample + (sample - denoised) / sigma * (sigma_next - sigma)
__global__ void euler_step_bf16_kernel(
    const __nv_bfloat16* __restrict__ sample,
    const __nv_bfloat16* __restrict__ denoised,
    __nv_bfloat16* __restrict__ out,
    float sigma,
    float dt,  // sigma_next - sigma
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;
    float s = bf16_to_float(sample[idx]);
    float d = bf16_to_float(denoised[idx]);
    float velocity = (s - d) / sigma;
    out[idx] = float_to_bf16(s + velocity * dt);
}

void euler_step_bf16(
    const __nv_bfloat16* sample, const __nv_bfloat16* denoised,
    __nv_bfloat16* out,
    float sigma, float sigma_next,
    int64_t N, cudaStream_t stream
) {
    float dt = sigma_next - sigma;
    int threads = 256;
    int blocks = ceil_div((int)N, threads);
    euler_step_bf16_kernel<<<blocks, threads, 0, stream>>>(sample, denoised, out, sigma, dt, N);
}
