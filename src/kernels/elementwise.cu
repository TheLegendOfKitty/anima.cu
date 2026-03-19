#include "cuda_utils.cuh"

// All elementwise kernels vectorized with BF16x2 loads/stores for 2x memory throughput.

// ---- Fused residual + gate: out = x + y * gate ----
__global__ void residual_gate_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ out,
    int64_t N2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(y);
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(gate);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);

    __nv_bfloat162 xv = x2[idx], yv = y2[idx], gv = g2[idx];
    float r0 = __fmaf_rn(__low2float(yv), __low2float(gv), __low2float(xv));
    float r1 = __fmaf_rn(__high2float(yv), __high2float(gv), __high2float(xv));
    out2[idx] = __floats2bfloat162_rn(r0, r1);
}

void residual_gate_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* y, const __nv_bfloat16* gate,
    __nv_bfloat16* out, int64_t N, cudaStream_t stream
) {
    int64_t N2 = N / 2;
    int threads = 256;
    int blocks = ceil_div((int)N2, threads);
    residual_gate_bf16_kernel<<<blocks, threads, 0, stream>>>(x, y, gate, out, N2);
}

// ---- Simple add: out = a + b ----
__global__ void add_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int64_t N2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(a);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(b);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);

    __nv_bfloat162 av = a2[idx], bv = b2[idx];
    float r0 = __low2float(av) + __low2float(bv);
    float r1 = __high2float(av) + __high2float(bv);
    out2[idx] = __floats2bfloat162_rn(r0, r1);
}

void add_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    __nv_bfloat16* out, int64_t N, cudaStream_t stream
) {
    int64_t N2 = N / 2;
    int threads = 256;
    int blocks = ceil_div((int)N2, threads);
    add_bf16_kernel<<<blocks, threads, 0, stream>>>(a, b, out, N2);
}

// ---- Three-way add: out = a + b + c ----
__global__ void add3_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    const __nv_bfloat16* __restrict__ c,
    __nv_bfloat16* __restrict__ out,
    int64_t N2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(a);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(b);
    const __nv_bfloat162* c2 = reinterpret_cast<const __nv_bfloat162*>(c);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);

    __nv_bfloat162 av = a2[idx], bv = b2[idx], cv = c2[idx];
    float r0 = __low2float(av) + __low2float(bv) + __low2float(cv);
    float r1 = __high2float(av) + __high2float(bv) + __high2float(cv);
    out2[idx] = __floats2bfloat162_rn(r0, r1);
}

void add3_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b, const __nv_bfloat16* c,
    __nv_bfloat16* out, int64_t N, cudaStream_t stream
) {
    int64_t N2 = N / 2;
    int threads = 256;
    int blocks = ceil_div((int)N2, threads);
    add3_bf16_kernel<<<blocks, threads, 0, stream>>>(a, b, c, out, N2);
}

// ---- Scale: out = x * scale (scalar) ----
__global__ void scale_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    float scale,
    int64_t N2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);

    __nv_bfloat162 val = x2[idx];
    float r0 = __low2float(val) * scale;
    float r1 = __high2float(val) * scale;
    out2[idx] = __floats2bfloat162_rn(r0, r1);
}

void scale_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    float scale, int64_t N, cudaStream_t stream
) {
    int64_t N2 = N / 2;
    int threads = 256;
    int blocks = ceil_div((int)N2, threads);
    scale_bf16_kernel<<<blocks, threads, 0, stream>>>(x, out, scale, N2);
}

// ---- Multiply elementwise: out = a * b ----
__global__ void mul_bf16_kernel(
    const __nv_bfloat16* __restrict__ a,
    const __nv_bfloat16* __restrict__ b,
    __nv_bfloat16* __restrict__ out,
    int64_t N2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    const __nv_bfloat162* a2 = reinterpret_cast<const __nv_bfloat162*>(a);
    const __nv_bfloat162* b2 = reinterpret_cast<const __nv_bfloat162*>(b);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);

    __nv_bfloat162 av = a2[idx], bv = b2[idx];
    float r0 = __low2float(av) * __low2float(bv);
    float r1 = __high2float(av) * __high2float(bv);
    out2[idx] = __floats2bfloat162_rn(r0, r1);
}

void mul_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    __nv_bfloat16* out, int64_t N, cudaStream_t stream
) {
    int64_t N2 = N / 2;
    int threads = 256;
    int blocks = ceil_div((int)N2, threads);
    mul_bf16_kernel<<<blocks, threads, 0, stream>>>(a, b, out, N2);
}

// ---- Fused scale-shift: out = x * (1 + scale) + shift ----
__global__ void scale_shift_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_bfloat16* __restrict__ out,
    int64_t N2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(scale);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(shift);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);

    __nv_bfloat162 xv = x2[idx], sv = s2[idx], hv = h2[idx];
    float r0 = __fmaf_rn(__low2float(xv), 1.0f + __low2float(sv), __low2float(hv));
    float r1 = __fmaf_rn(__high2float(xv), 1.0f + __high2float(sv), __high2float(hv));
    out2[idx] = __floats2bfloat162_rn(r0, r1);
}

void scale_shift_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* out, int64_t N, cudaStream_t stream
) {
    int64_t N2 = N / 2;
    int threads = 256;
    int blocks = ceil_div((int)N2, threads);
    scale_shift_bf16_kernel<<<blocks, threads, 0, stream>>>(x, scale, shift, out, N2);
}

// ---- Broadcast bias add: out[M, D] = x[M, D] + bias[D] ----
__global__ void bias_add_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ bias,
    __nv_bfloat16* __restrict__ out,
    int64_t total2,
    int D2  // D / 2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total2) return;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const __nv_bfloat162* bias2 = reinterpret_cast<const __nv_bfloat162*>(bias);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);

    __nv_bfloat162 xv = x2[idx];
    __nv_bfloat162 bv = bias2[idx % D2];
    float r0 = __low2float(xv) + __low2float(bv);
    float r1 = __high2float(xv) + __high2float(bv);
    out2[idx] = __floats2bfloat162_rn(r0, r1);
}

void bias_add_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* bias,
    __nv_bfloat16* out, int M, int D, cudaStream_t stream
) {
    int64_t total2 = (int64_t)M * (D / 2);
    int threads = 256;
    int blocks = ceil_div((int)total2, threads);
    bias_add_bf16_kernel<<<blocks, threads, 0, stream>>>(x, bias, out, total2, D / 2);
}

// ---- Broadcast scale-shift: out[M,D] = x[M,D] * (1 + scale[D]) + shift[D] ----
__global__ void scale_shift_bcast_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_bfloat16* __restrict__ out,
    int64_t total2,
    int D2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total2) return;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(scale);
    const __nv_bfloat162* h2 = reinterpret_cast<const __nv_bfloat162*>(shift);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);

    int d2 = idx % D2;
    __nv_bfloat162 xv = x2[idx], sv = s2[d2], hv = h2[d2];
    float r0 = __fmaf_rn(__low2float(xv), 1.0f + __low2float(sv), __low2float(hv));
    float r1 = __fmaf_rn(__high2float(xv), 1.0f + __high2float(sv), __high2float(hv));
    out2[idx] = __floats2bfloat162_rn(r0, r1);
}

void scale_shift_bcast_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* out, int M, int D, cudaStream_t stream
) {
    int64_t total2 = (int64_t)M * (D / 2);
    int threads = 256;
    int blocks = ceil_div((int)total2, threads);
    scale_shift_bcast_bf16_kernel<<<blocks, threads, 0, stream>>>(x, scale, shift, out, total2, D / 2);
}

// ---- Broadcast residual+gate: out[M,D] = x[M,D] + y[M,D] * gate[D] ----
__global__ void residual_gate_bcast_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ y,
    const __nv_bfloat16* __restrict__ gate,
    __nv_bfloat16* __restrict__ out,
    int64_t total2,
    int D2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total2) return;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    const __nv_bfloat162* y2 = reinterpret_cast<const __nv_bfloat162*>(y);
    const __nv_bfloat162* g2 = reinterpret_cast<const __nv_bfloat162*>(gate);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);

    int d2 = idx % D2;
    __nv_bfloat162 xv = x2[idx], yv = y2[idx], gv = g2[d2];
    float r0 = __fmaf_rn(__low2float(yv), __low2float(gv), __low2float(xv));
    float r1 = __fmaf_rn(__high2float(yv), __high2float(gv), __high2float(xv));
    out2[idx] = __floats2bfloat162_rn(r0, r1);
}

void residual_gate_bcast_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* y, const __nv_bfloat16* gate,
    __nv_bfloat16* out, int M, int D, cudaStream_t stream
) {
    int64_t total2 = (int64_t)M * (D / 2);
    int threads = 256;
    int blocks = ceil_div((int)total2, threads);
    residual_gate_bcast_bf16_kernel<<<blocks, threads, 0, stream>>>(x, y, gate, out, total2, D / 2);
}

// ---- to_denoised: out = sample - velocity * sigma (FP32 compute) ----
__global__ void to_denoised_bf16_kernel(
    const __nv_bfloat16* __restrict__ sample,
    const __nv_bfloat16* __restrict__ velocity,
    __nv_bfloat16* __restrict__ out,
    float sigma,
    int64_t N2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(sample);
    const __nv_bfloat162* v2 = reinterpret_cast<const __nv_bfloat162*>(velocity);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);

    __nv_bfloat162 sv = s2[idx], vv = v2[idx];
    float r0 = __low2float(sv) - __low2float(vv) * sigma;
    float r1 = __high2float(sv) - __high2float(vv) * sigma;
    out2[idx] = __floats2bfloat162_rn(r0, r1);
}

void to_denoised_bf16(
    const __nv_bfloat16* sample, const __nv_bfloat16* velocity,
    __nv_bfloat16* out, float sigma, int64_t N, cudaStream_t stream
) {
    int64_t N2 = N / 2;
    int threads = 256;
    int blocks = ceil_div((int)N2, threads);
    to_denoised_bf16_kernel<<<blocks, threads, 0, stream>>>(sample, velocity, out, sigma, N2);
}

// ---- F32 to BF16 conversion ----
__global__ void f32_to_bf16_kernel(
    const float* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int64_t N2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    const float2* x2 = reinterpret_cast<const float2*>(x);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);

    float2 val = x2[idx];
    out2[idx] = __floats2bfloat162_rn(val.x, val.y);
}

void f32_to_bf16(const float* x, __nv_bfloat16* out, int64_t N, cudaStream_t stream) {
    int64_t N2 = N / 2;
    int threads = 256;
    int blocks = ceil_div((int)N2, threads);
    f32_to_bf16_kernel<<<blocks, threads, 0, stream>>>(x, out, N2);
}

__global__ void bf16_to_f32_kernel(
    const __nv_bfloat16* __restrict__ x,
    float* __restrict__ out,
    int64_t N2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    const __nv_bfloat162* x2 = reinterpret_cast<const __nv_bfloat162*>(x);
    float2* out2 = reinterpret_cast<float2*>(out);

    __nv_bfloat162 val = x2[idx];
    out2[idx] = make_float2(__low2float(val), __high2float(val));
}

void bf16_to_f32(const __nv_bfloat16* x, float* out, int64_t N, cudaStream_t stream) {
    int64_t N2 = N / 2;
    int threads = 256;
    int blocks = ceil_div((int)N2, threads);
    bf16_to_f32_kernel<<<blocks, threads, 0, stream>>>(x, out, N2);
}

// ---- Euler step ----
__global__ void euler_step_bf16_kernel(
    const __nv_bfloat16* __restrict__ sample,
    const __nv_bfloat16* __restrict__ denoised,
    __nv_bfloat16* __restrict__ out,
    float sigma,
    float dt,
    int64_t N2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N2) return;
    const __nv_bfloat162* s2 = reinterpret_cast<const __nv_bfloat162*>(sample);
    const __nv_bfloat162* d2 = reinterpret_cast<const __nv_bfloat162*>(denoised);
    __nv_bfloat162* out2 = reinterpret_cast<__nv_bfloat162*>(out);

    __nv_bfloat162 sv = s2[idx], dv = d2[idx];
    float s0 = __low2float(sv), s1 = __high2float(sv);
    float d0 = __low2float(dv), d1 = __high2float(dv);
    float inv_sigma = 1.0f / sigma;
    float r0 = __fmaf_rn((s0 - d0) * inv_sigma, dt, s0);
    float r1 = __fmaf_rn((s1 - d1) * inv_sigma, dt, s1);
    out2[idx] = __floats2bfloat162_rn(r0, r1);
}

void euler_step_bf16(
    const __nv_bfloat16* sample, const __nv_bfloat16* denoised,
    __nv_bfloat16* out,
    float sigma, float sigma_next,
    int64_t N, cudaStream_t stream
) {
    float dt = sigma_next - sigma;
    int64_t N2 = N / 2;
    int threads = 256;
    int blocks = ceil_div((int)N2, threads);
    euler_step_bf16_kernel<<<blocks, threads, 0, stream>>>(sample, denoised, out, sigma, dt, N2);
}
