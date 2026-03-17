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

// ========================= Shared reduction helper =========================

// Reduce a float value across the entire CTA using warp shuffles + shared memory.
// Returns the reduced value broadcast to all threads.
__device__ __forceinline__ float cta_reduce_sum(float val, float* shared) {
    int lane = threadIdx.x % 32;
    int warp = threadIdx.x / 32;
    int num_warps = (blockDim.x + 31) / 32;
    val = warp_reduce_sum(val);
    if (lane == 0) shared[warp] = val;
    __syncthreads();
    if (warp == 0) { val = (lane < num_warps) ? shared[lane] : 0.0f; val = warp_reduce_sum(val); }
    __syncthreads();
    if (threadIdx.x == 0) shared[0] = val;
    __syncthreads();
    return shared[0];
}

// ========================= Fused adaLN: LayerNorm + scale_shift_bcast =========================
// out[n,d] = LayerNorm(x[n,:])[d] * (1 + scale[d]) + shift[d]
// scale, shift are [D] broadcast across N rows.
//
// Optimizations vs naive:
//   - 2-pass instead of 3: single pass computes sum + sum_sq simultaneously
//     (variance = sum_sq/D - mean², valid for F32 accumulation on BF16 data)
//   - Vectorized BF16x2 loads/stores via __nv_bfloat162 (2x memory throughput)

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

    __shared__ float shared[33];  // 32 for warp reduce + 1 for broadcast
    int D2 = D / 2;

    // Pass 1: single-pass mean + sum_of_squares (vectorized BF16x2 loads)
    float sum = 0.0f, sum_sq = 0.0f;
    const __nv_bfloat162* xr2 = reinterpret_cast<const __nv_bfloat162*>(xr);
    for (int i = threadIdx.x; i < D2; i += blockDim.x) {
        __nv_bfloat162 val = xr2[i];
        float v0 = __low2float(val), v1 = __high2float(val);
        sum += v0 + v1;
        sum_sq += v0 * v0 + v1 * v1;
    }
    float mean = cta_reduce_sum(sum, shared) / D;
    float var = cta_reduce_sum(sum_sq, shared) / D - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Pass 2: normalize + scale + shift (vectorized BF16x2 loads/stores)
    const __nv_bfloat162* scale2 = reinterpret_cast<const __nv_bfloat162*>(scale);
    const __nv_bfloat162* shift2 = reinterpret_cast<const __nv_bfloat162*>(shift);
    __nv_bfloat162* yr2 = reinterpret_cast<__nv_bfloat162*>(yr);
    for (int i = threadIdx.x; i < D2; i += blockDim.x) {
        __nv_bfloat162 xv = xr2[i];
        __nv_bfloat162 sv = scale2[i];
        __nv_bfloat162 hv = shift2[i];
        float x0 = __low2float(xv), x1 = __high2float(xv);
        float s0 = __low2float(sv), s1 = __high2float(sv);
        float h0 = __low2float(hv), h1 = __high2float(hv);
        float o0 = (x0 - mean) * inv_std * (1.0f + s0) + h0;
        float o1 = (x1 - mean) * inv_std * (1.0f + s1) + h1;
        yr2[i] = __floats2bfloat162_rn(o0, o1);
    }
}

void adaln_layernorm_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* out, int N, int D, float eps, cudaStream_t stream
) {
    int threads = (D / 2 <= 1024) ? ((D / 2 + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    adaln_layernorm_bf16_kernel<<<N, threads, 0, stream>>>(x, scale, shift, out, D, eps);
}

// ========================= Fused residual_gate + adaLN =========================
// Computes:
//   hidden[n,d] = x[n,d] + y[n,d] * gate[d]           (residual + gated output)
//   normed[n,d] = LayerNorm(hidden[n,:])[d] * (1 + scale[d]) + shift[d]  (adaLN)
//
// 2-pass with vectorized BF16x2:
//   Pass 1: residual + accumulate sum + sum_sq (vectorized loads, write hidden)
//   Pass 2: normalize + scale + shift (vectorized loads/stores)

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
    int D2 = D / 2;
    __nv_bfloat162* hr2 = reinterpret_cast<__nv_bfloat162*>(hidden + (int64_t)row * D);
    const __nv_bfloat162* yr2 = reinterpret_cast<const __nv_bfloat162*>(sub_out + (int64_t)row * D);
    const __nv_bfloat162* gate2 = reinterpret_cast<const __nv_bfloat162*>(gate);
    __nv_bfloat162* nr2 = reinterpret_cast<__nv_bfloat162*>(normed + (int64_t)row * D);

    __shared__ float shared[33];

    // Pass 1: residual + single-pass mean + sum_sq (vectorized)
    float sum = 0.0f, sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D2; i += blockDim.x) {
        __nv_bfloat162 xv = hr2[i];
        __nv_bfloat162 yv = yr2[i];
        __nv_bfloat162 gv = gate2[i];
        float x0 = __low2float(xv), x1 = __high2float(xv);
        float y0 = __low2float(yv), y1 = __high2float(yv);
        float g0 = __low2float(gv), g1 = __high2float(gv);
        float h0 = x0 + y0 * g0, h1 = x1 + y1 * g1;
        hr2[i] = __floats2bfloat162_rn(h0, h1);
        sum += h0 + h1;
        sum_sq += h0 * h0 + h1 * h1;
    }
    float mean = cta_reduce_sum(sum, shared) / D;
    float var = cta_reduce_sum(sum_sq, shared) / D - mean * mean;
    float inv_std = rsqrtf(var + eps);

    // Pass 2: normalize + adaLN (vectorized)
    const __nv_bfloat162* scale2 = reinterpret_cast<const __nv_bfloat162*>(scale);
    const __nv_bfloat162* shift2 = reinterpret_cast<const __nv_bfloat162*>(shift);
    for (int i = threadIdx.x; i < D2; i += blockDim.x) {
        __nv_bfloat162 hv = hr2[i];
        __nv_bfloat162 sv = scale2[i];
        __nv_bfloat162 shv = shift2[i];
        float h0 = __low2float(hv), h1 = __high2float(hv);
        float s0 = __low2float(sv), s1 = __high2float(sv);
        float sh0 = __low2float(shv), sh1 = __high2float(shv);
        float o0 = (h0 - mean) * inv_std * (1.0f + s0) + sh0;
        float o1 = (h1 - mean) * inv_std * (1.0f + s1) + sh1;
        nr2[i] = __floats2bfloat162_rn(o0, o1);
    }
}

void residual_gate_adaln_bf16(
    __nv_bfloat16* hidden,
    const __nv_bfloat16* sub_out, const __nv_bfloat16* gate,
    const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* normed,
    int N, int D, float eps, cudaStream_t stream
) {
    int threads = (D / 2 <= 1024) ? ((D / 2 + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    residual_gate_adaln_bf16_kernel<<<N, threads, 0, stream>>>(
        hidden, sub_out, gate, scale, shift, normed, D, eps);
}
