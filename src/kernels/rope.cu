#include "cuda_utils.cuh"

// ========================= Cosmos RoPE (halved pattern) =========================
//
// For each head h, spatial position s, element i in [0, HD/2):
//   out[h,s,i]       = x[h,s,i]       * cos[s,i] - x[h,s,i+HD/2] * sin[s,i]
//   out[h,s,i+HD/2]  = x[h,s,i+HD/2]  * cos[s,i] + x[h,s,i]     * sin[s,i]
//
// cos/sin tables are [S, HD] with first/second half identical (cos[i] = cos[i+HD/2]).
// x is [H, S, HD]. cos/sin broadcast across all heads.

__global__ void rope_cosmos_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_freq,
    const __nv_bfloat16* __restrict__ sin_freq,
    __nv_bfloat16* __restrict__ out,
    int S, int HD, int64_t total_pairs  // total_pairs = H * S * HD/2
) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= total_pairs) return;

    int half_hd = HD / 2;
    int s_half = S * half_hd;

    // Decompose: idx -> (head, spatial_pos, half_element)
    int h   = (int)(idx / s_half);
    int rem = (int)(idx % s_half);
    int s   = rem / half_hd;
    int i   = rem % half_hd;

    // Indices into x [H, S, HD]
    int64_t base   = (int64_t)h * S * HD + (int64_t)s * HD;
    int64_t idx_re = base + i;
    int64_t idx_im = base + i + half_hd;

    // cos/sin from [S, HD], use first half (since duplicated)
    int64_t cs_idx = (int64_t)s * HD + i;

    float x_re = bf16_to_float(x[idx_re]);
    float x_im = bf16_to_float(x[idx_im]);
    float c    = bf16_to_float(cos_freq[cs_idx]);
    float sv   = bf16_to_float(sin_freq[cs_idx]);

    // Complex rotation: (x_re + j*x_im) * (c + j*s)
    out[idx_re] = float_to_bf16(x_re * c - x_im * sv);
    out[idx_im] = float_to_bf16(x_im * c + x_re * sv);
}

void rope_cosmos_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* cos_freq,
    const __nv_bfloat16* sin_freq,
    __nv_bfloat16* out,
    int H, int S, int HD,
    cudaStream_t stream
) {
    int64_t total_pairs = (int64_t)H * S * (HD / 2);
    int threads = 256;
    int blocks = (int)((total_pairs + threads - 1) / threads);
    rope_cosmos_bf16_kernel<<<blocks, threads, 0, stream>>>(
        x, cos_freq, sin_freq, out, S, HD, total_pairs);
}

// ========================= Interleaved RoPE (legacy) =========================
//
// For each pair (x[2i], x[2i+1]):
//   out[2i]   = x[2i]   * cos[2i]   - x[2i+1] * sin[2i]
//   out[2i+1] = x[2i+1] * cos[2i+1] + x[2i]   * sin[2i+1]
//
// Note: In the Python code, cos and sin are already repeat_interleaved(2).
// So cos[2i] == cos[2i+1] and sin[2i] == sin[2i+1].
//
// Input:  x [total_elems] BF16  (flattened [B, T, H*D] or similar)
// cos_freq, sin_freq: BF16 [total_elems] (pre-broadcast to match x shape)
// Output: out [total_elems] BF16

__global__ void rope_interleaved_bf16_kernel(
    const __nv_bfloat16* __restrict__ x,
    const __nv_bfloat16* __restrict__ cos_freq,
    const __nv_bfloat16* __restrict__ sin_freq,
    __nv_bfloat16* __restrict__ out,
    int64_t total_pairs  // total_elems / 2
) {
    int64_t pair_idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (pair_idx >= total_pairs) return;

    int64_t base = pair_idx * 2;

    float x0 = bf16_to_float(x[base]);
    float x1 = bf16_to_float(x[base + 1]);
    float c  = bf16_to_float(cos_freq[base]);     // cos[2i] == cos[2i+1]
    float s  = bf16_to_float(sin_freq[base]);     // sin[2i] == sin[2i+1]

    // Interleaved rotation: rotate_half pairs adjacent elements
    // rotate_half: (-x1, x0) interleaved
    out[base]     = float_to_bf16(x0 * c - x1 * s);
    out[base + 1] = float_to_bf16(x1 * c + x0 * s);
}

void rope_interleaved_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* cos_freq,
    const __nv_bfloat16* sin_freq,
    __nv_bfloat16* out,
    int64_t total_elems,
    cudaStream_t stream
) {
    int64_t pairs = total_elems / 2;
    int threads = 256;
    int blocks = ceil_div((int)pairs, threads);
    rope_interleaved_bf16_kernel<<<blocks, threads, 0, stream>>>(x, cos_freq, sin_freq, out, pairs);
}
