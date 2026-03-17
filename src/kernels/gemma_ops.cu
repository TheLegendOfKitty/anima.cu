#include "cuda_utils.cuh"

// ---- Gated GELU MLP: out = down_proj(gelu(gate) * up) ----
// Fused gate*up part: out[i] = gelu_tanh(gate[i]) * up[i]
__global__ void gated_gelu_bf16_kernel(
    const __nv_bfloat16* __restrict__ gate,
    const __nv_bfloat16* __restrict__ up,
    __nv_bfloat16* __restrict__ out,
    int64_t N
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float g = bf16_to_float(gate[idx]);
    float u = bf16_to_float(up[idx]);

    // GELU(tanh) on gate
    float inner = 0.7978845608f * (g + 0.044715f * g * g * g);
    float gelu_g = 0.5f * g * (1.0f + tanhf(inner));

    out[idx] = float_to_bf16(gelu_g * u);
}

void gated_gelu_bf16(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* out, int64_t N, cudaStream_t stream
) {
    int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);
    gated_gelu_bf16_kernel<<<blocks, threads, 0, stream>>>(gate, up, out, N);
}

// ---- Embedding lookup: out[t, d] = table[ids[t], d] ----
__global__ void embedding_lookup_bf16_kernel(
    const __nv_bfloat16* __restrict__ table,  // [V, D]
    const int* __restrict__ ids,               // [T]
    __nv_bfloat16* __restrict__ out,           // [T, D]
    int D
) {
    int t = blockIdx.x;
    int id = ids[t];
    for (int d = threadIdx.x; d < D; d += blockDim.x) {
        out[(int64_t)t * D + d] = table[(int64_t)id * D + d];
    }
}

void embedding_lookup_bf16(
    const __nv_bfloat16* table, const int* ids,
    __nv_bfloat16* out, int T, int D, cudaStream_t stream
) {
    int threads = (D <= 256) ? D : 256;
    embedding_lookup_bf16_kernel<<<T, threads, 0, stream>>>(table, ids, out, D);
}

// ---- RoPE for Gemma (standard, not interleaved) ----
// Gemma uses "standard" RoPE: split x into first half and second half
// x1 = x[:half], x2 = x[half:]
// out[:half]  = x1 * cos - x2 * sin
// out[half:] = x2 * cos + x1 * sin
// Applied per-head: x is [B, H, T, D], cos/sin are [T, D/2]
// We flatten: x is [total_pairs] pairs.
__global__ void rope_standard_bf16_kernel(
    __nv_bfloat16* __restrict__ x,  // [N, D] in-place (N = B*H*T)
    const float* __restrict__ cos_cache,  // [T, D/2]
    const float* __restrict__ sin_cache,  // [T, D/2]
    int T, int D, int total_rows  // total_rows = B*H*T
) {
    int row = blockIdx.x;
    if (row >= total_rows) return;

    int t = row % T;
    int half = D / 2;

    for (int d = threadIdx.x; d < half; d += blockDim.x) {
        float c = cos_cache[t * half + d];
        float s = sin_cache[t * half + d];

        int64_t i1 = (int64_t)row * D + d;
        int64_t i2 = (int64_t)row * D + half + d;

        float x1 = bf16_to_float(x[i1]);
        float x2 = bf16_to_float(x[i2]);

        x[i1] = float_to_bf16(x1 * c - x2 * s);
        x[i2] = float_to_bf16(x2 * c + x1 * s);
    }
}

void rope_standard_bf16(
    __nv_bfloat16* x,
    const float* cos_cache, const float* sin_cache,
    int total_rows, int T, int D, cudaStream_t stream
) {
    int threads = (D/2 <= 256) ? (D/2) : 256;
    rope_standard_bf16_kernel<<<total_rows, threads, 0, stream>>>(
        x, cos_cache, sin_cache, T, D, total_rows);
}

// ---- Expand GQA keys/values: repeat KV heads to match Q heads ----
// K/V: [B, KV_H, T, D] -> [B, Q_H, T, D] by repeating each KV head (Q_H/KV_H) times
__global__ void expand_kv_bf16_kernel(
    const __nv_bfloat16* __restrict__ kv,  // [B, KV_H, T, D]
    __nv_bfloat16* __restrict__ out,        // [B, Q_H, T, D]
    int KV_H, int Q_H, int T, int D
) {
    int b = blockIdx.z;
    int qh = blockIdx.y;
    int td = blockIdx.x * blockDim.x + threadIdx.x;
    if (td >= T * D) return;

    int ratio = Q_H / KV_H;
    int kvh = qh / ratio;

    int64_t src = ((int64_t)b * KV_H + kvh) * T * D + td;
    int64_t dst = ((int64_t)b * Q_H + qh) * T * D + td;
    out[dst] = kv[src];
}

void expand_kv_bf16(
    const __nv_bfloat16* kv, __nv_bfloat16* out,
    int B, int KV_H, int Q_H, int T, int D, cudaStream_t stream
) {
    int threads = 256;
    int blocks_x = (T * D + threads - 1) / threads;
    dim3 grid(blocks_x, Q_H, B);
    expand_kv_bf16_kernel<<<grid, threads, 0, stream>>>(kv, out, KV_H, Q_H, T, D);
}

// ---- RMS norm with weight (Gemma uses learnable weight + adds 1.0) ----
// Gemma norm: rms_norm(x) * (1 + weight)
// This is different from standard RMS norm which does rms_norm(x) * weight
__global__ void gemma_rms_norm_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ weight,
    __nv_bfloat16* __restrict__ out,
    int D, float eps
) {
    int row = blockIdx.x;
    const __nv_bfloat16* xr = x + (int64_t)row * D;
    __nv_bfloat16* yr = out + (int64_t)row * D;

    float sum_sq = 0.0f;
    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(xr[i]);
        sum_sq += v * v;
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

    for (int i = threadIdx.x; i < D; i += blockDim.x) {
        float v = bf16_to_float(xr[i]) * rms_inv;
        float w = bf16_to_float(weight[i]);
        yr[i] = float_to_bf16(v * (1.0f + w));  // Gemma: (1 + weight)
    }
}

void gemma_rms_norm_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    __nv_bfloat16* out, int N, int D, float eps, cudaStream_t stream
) {
    int threads = (D <= 1024) ? ((D + 31) / 32 * 32) : 1024;
    if (threads < 32) threads = 32;
    gemma_rms_norm_bf16_kernel<<<N, threads, 0, stream>>>(x, weight, out, D, eps);
}
