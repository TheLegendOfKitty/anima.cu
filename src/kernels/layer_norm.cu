#include "cuda_utils.cuh"

// LayerNorm with elementwise_affine=False:
// out = (x - mean(x)) / sqrt(var(x) + eps)
// Input/output: BF16 [N, D]. Internal FP32.

__global__ void layer_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    __nv_bfloat16* __restrict__ out,
    int D,
    float eps
) {
    int row = blockIdx.x;
    const __nv_bfloat16* xr = x + (int64_t)row * D;
    __nv_bfloat16* yr = out + (int64_t)row * D;

    // Compute mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        sum += bf16_to_float(xr[i]);
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
    float mean = shared[0] / D;

    // Compute variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(xr[i]) - mean;
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
    float inv_std = rsqrtf(shared[0] / D + eps);

    // Normalize
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = (bf16_to_float(xr[i]) - mean) * inv_std;
        yr[i] = float_to_bf16(v);
    }
}

void layer_norm_bf16(
    const __nv_bfloat16* x,
    __nv_bfloat16* out,
    int N, int D,
    float eps,
    cudaStream_t stream
) {
    int threads = (D <= 1024) ? ((D + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    layer_norm_bf16_kernel<<<N, threads, 0, stream>>>(x, out, D, eps);
}

// ========================= Fused adaLN: LayerNorm + scale_shift_bcast =========================
// out[n,d] = LayerNorm(x[n,:])[d] * (1 + scale[d]) + shift[d]
// scale, shift are [D] broadcast across N rows.
// Fuses two separate kernels (layer_norm + scale_shift_bcast) into one,
// eliminating one full read+write of the [N, D] intermediate.

__global__ void adaln_layernorm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ scale,
    const __nv_bfloat16* __restrict__ shift,
    __nv_bfloat16* __restrict__ out,
    int D, float eps
) {
    int row = blockIdx.x;
    const __nv_bfloat16* xr = x + (int64_t)row * D;
    __nv_bfloat16* yr = out + (int64_t)row * D;

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int num_warps = (blockDim.x + 31) / 32;

    // Pass 1: mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x)
        sum += bf16_to_float(xr[i]);
    sum = warp_reduce_sum(sum);
    if (lane == 0) shared[warp] = sum;
    __syncthreads();
    if (warp == 0) { sum = (lane < num_warps) ? shared[lane] : 0.0f; sum = warp_reduce_sum(sum); }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = sum;
    __syncthreads();
    float mean = shared[0] / D;

    // Pass 2: variance
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(xr[i]) - mean;
        var_sum += v * v;
    }
    var_sum = warp_reduce_sum(var_sum);
    if (lane == 0) shared[warp] = var_sum;
    __syncthreads();
    if (warp == 0) { var_sum = (lane < num_warps) ? shared[lane] : 0.0f; var_sum = warp_reduce_sum(var_sum); }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = var_sum;
    __syncthreads();
    float inv_std = rsqrtf(shared[0] / D + eps);

    // Pass 3: normalize + scale + shift (fused)
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = (bf16_to_float(xr[i]) - mean) * inv_std;
        float s = bf16_to_float(scale[i]);
        float h = bf16_to_float(shift[i]);
        yr[i] = float_to_bf16(v * (1.0f + s) + h);
    }
}

void adaln_layernorm_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* out, int N, int D, float eps, cudaStream_t stream
) {
    int threads = (D <= 1024) ? ((D + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    adaln_layernorm_bf16_kernel<<<N, threads, 0, stream>>>(x, scale, shift, out, D, eps);
}

// ========================= Fused residual_gate + adaLN =========================
// Computes:
//   hidden[n,d] = x[n,d] + y[n,d] * gate[d]           (residual + gated output)
//   normed[n,d] = LayerNorm(hidden[n,:])[d] * (1 + scale[d]) + shift[d]  (adaLN)
//
// Fuses residual_gate_bcast + layer_norm + scale_shift_bcast into one kernel.
// hidden is both read (as x) and written (residual result), normed is the adaLN output.

__global__ void residual_gate_adaln_bf16_kernel(
    __nv_bfloat16* __restrict__ hidden,         // [N, D] in/out (residual target)
    const __nv_bfloat16* __restrict__ sub_out,  // [N, D] sub-layer output
    const __nv_bfloat16* __restrict__ gate,     // [D] gate vector
    const __nv_bfloat16* __restrict__ scale,    // [D] adaLN scale
    const __nv_bfloat16* __restrict__ shift,    // [D] adaLN shift
    __nv_bfloat16* __restrict__ normed,         // [N, D] adaLN output
    int D, float eps
) {
    int row = blockIdx.x;
    __nv_bfloat16* hr = hidden + (int64_t)row * D;
    const __nv_bfloat16* yr = sub_out + (int64_t)row * D;
    __nv_bfloat16* nr = normed + (int64_t)row * D;

    __shared__ float shared[32];
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int num_warps = (blockDim.x + 31) / 32;

    // Pass 1: compute residual and accumulate mean
    float sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float xv = bf16_to_float(hr[i]);
        float yv = bf16_to_float(yr[i]);
        float gv = bf16_to_float(gate[i]);
        float h = xv + yv * gv;
        hr[i] = float_to_bf16(h);
        sum += h;
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) shared[warp] = sum;
    __syncthreads();
    if (warp == 0) { sum = (lane < num_warps) ? shared[lane] : 0.0f; sum = warp_reduce_sum(sum); }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = sum;
    __syncthreads();
    float mean = shared[0] / D;

    // Pass 2: variance of hidden (re-read from BF16 — small precision loss but avoids register pressure)
    float var_sum = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(hr[i]) - mean;
        var_sum += v * v;
    }
    var_sum = warp_reduce_sum(var_sum);
    if (lane == 0) shared[warp] = var_sum;
    __syncthreads();
    if (warp == 0) { var_sum = (lane < num_warps) ? shared[lane] : 0.0f; var_sum = warp_reduce_sum(var_sum); }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = var_sum;
    __syncthreads();
    float inv_std = rsqrtf(shared[0] / D + eps);

    // Pass 3: normalize + adaLN scale/shift
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = (bf16_to_float(hr[i]) - mean) * inv_std;
        float s = bf16_to_float(scale[i]);
        float h = bf16_to_float(shift[i]);
        nr[i] = float_to_bf16(v * (1.0f + s) + h);
    }
}

void residual_gate_adaln_bf16(
    __nv_bfloat16* hidden,
    const __nv_bfloat16* sub_out, const __nv_bfloat16* gate,
    const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* normed,
    int N, int D, float eps, cudaStream_t stream
) {
    int threads = (D <= 1024) ? ((D + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    residual_gate_adaln_bf16_kernel<<<N, threads, 0, stream>>>(
        hidden, sub_out, gate, scale, shift, normed, D, eps);
}
