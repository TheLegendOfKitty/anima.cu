#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cublas_v2.h>
#include <cstdint>

// ---- RMS Norm ----
void rms_norm_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, __nv_bfloat16* out,
    int N, int D, float eps = 1e-6f, cudaStream_t stream = 0);

// ---- Layer Norm (no affine) ----
void layer_norm_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    int N, int D, float eps = 1e-6f, cudaStream_t stream = 0);

// ---- Fused adaLN: LayerNorm(x) * (1+scale) + shift ----
void adaln_layernorm_bf16(
    const __nv_bfloat16* x,
    const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* out, int N, int D, float eps = 1e-6f, cudaStream_t stream = 0);

// ---- Fused residual_gate + adaLN: hidden += sub_out * gate; normed = adaLN(hidden) ----
void residual_gate_adaln_bf16(
    __nv_bfloat16* hidden,
    const __nv_bfloat16* sub_out, const __nv_bfloat16* gate,
    const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* normed,
    int N, int D, float eps = 1e-6f, cudaStream_t stream = 0);

// ---- GELU(tanh) ----
void gelu_tanh_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t N, cudaStream_t stream = 0);
void gelu_tanh_bias_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* bias,
    __nv_bfloat16* out, int64_t N, int D, cudaStream_t stream = 0);

// ---- SiLU ----
void silu_bf16(const __nv_bfloat16* x, __nv_bfloat16* out, int64_t N, cudaStream_t stream = 0);
void silu_bias_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* bias,
    __nv_bfloat16* out, int64_t N, int D, cudaStream_t stream = 0);

// ---- RoPE (Cosmos halved pattern, broadcast across heads) ----
// x: [H, S, HD], cos/sin: [S, HD]. Pairs (x[i], x[i+HD/2]).
void rope_cosmos_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* cos_freq, const __nv_bfloat16* sin_freq,
    __nv_bfloat16* out, int H, int S, int HD, cudaStream_t stream = 0);

// ---- RoPE (Cosmos halved, strided [B*S, H*HD] layout — no transpose needed) ----
void rope_cosmos_strided_bf16(
    __nv_bfloat16* x, const __nv_bfloat16* cos_freq, const __nv_bfloat16* sin_freq,
    int B, int S, int H, int HD, cudaStream_t stream = 0);

// ---- RoPE (interleaved, legacy) ----
void rope_interleaved_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* cos_freq, const __nv_bfloat16* sin_freq,
    __nv_bfloat16* out, int64_t total_elems, cudaStream_t stream = 0);

// ---- Fused AdaLN: out = rms_norm(x) * (1+scale) + shift ----
void adaln_rms_norm_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* out, int N, int D, float eps = 1e-6f, cudaStream_t stream = 0);

// ---- Elementwise ----
void residual_gate_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* y, const __nv_bfloat16* gate,
    __nv_bfloat16* out, int64_t N, cudaStream_t stream = 0);
void add_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    __nv_bfloat16* out, int64_t N, cudaStream_t stream = 0);
void scale_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    float scale, int64_t N, cudaStream_t stream = 0);
void mul_bf16(
    const __nv_bfloat16* a, const __nv_bfloat16* b,
    __nv_bfloat16* out, int64_t N, cudaStream_t stream = 0);
void bias_add_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* bias,
    __nv_bfloat16* out, int M, int D, cudaStream_t stream = 0);
void scale_shift_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* out, int64_t N, cudaStream_t stream = 0);
void scale_shift_bcast_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* out, int M, int D, cudaStream_t stream = 0);
void residual_gate_bcast_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* y, const __nv_bfloat16* gate,
    __nv_bfloat16* out, int M, int D, cudaStream_t stream = 0);
void to_denoised_bf16(
    const __nv_bfloat16* sample, const __nv_bfloat16* velocity,
    __nv_bfloat16* out, float sigma, int64_t N, cudaStream_t stream = 0);
void f32_to_bf16(const float* x, __nv_bfloat16* out, int64_t N, cudaStream_t stream = 0);
void bf16_to_f32(const __nv_bfloat16* x, float* out, int64_t N, cudaStream_t stream = 0);
void euler_step_bf16(
    const __nv_bfloat16* sample, const __nv_bfloat16* denoised,
    __nv_bfloat16* out, float sigma, float sigma_next,
    int64_t N, cudaStream_t stream = 0);

// ---- Causal mask: set upper triangular to -inf ----
void causal_mask_bf16(
    __nv_bfloat16* scores, int num_heads, int T,
    cudaStream_t stream = 0);

// ---- Softmax (row-wise) ----
void softmax_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    int N, int D, cudaStream_t stream = 0);
void softmax_f32(
    const float* x, float* out,
    int N, int D, cudaStream_t stream = 0);
void softmax_f32_to_bf16(
    const float* x, __nv_bfloat16* out,
    int N, int D, cudaStream_t stream = 0);

// ---- Patchify / Unpatchify ----
void patchify_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    int B, int C, int T, cudaStream_t stream = 0);
void unpatchify_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    int B, int T, int C, cudaStream_t stream = 0);

// ---- GroupNorm 5D ----
void group_norm_5d_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    __nv_bfloat16* out, int B, int C, int spatial, int groups, float eps = 1e-5f,
    cudaStream_t stream = 0);

// ---- PixelShuffle 2D ----
void pixel_shuffle_2d_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    int B, int C, int H, int W, int r, cudaStream_t stream = 0);

// ---- Gemma ops ----
void gated_gelu_bf16(
    const __nv_bfloat16* gate, const __nv_bfloat16* up,
    __nv_bfloat16* out, int64_t N, cudaStream_t stream = 0);
void embedding_lookup_bf16(
    const __nv_bfloat16* table, const int* ids,
    __nv_bfloat16* out, int T, int D, cudaStream_t stream = 0);
void rope_standard_bf16(
    __nv_bfloat16* x, const float* cos_cache, const float* sin_cache,
    int total_rows, int T, int D, cudaStream_t stream = 0);
void expand_kv_bf16(
    const __nv_bfloat16* kv, __nv_bfloat16* out,
    int B, int KV_H, int Q_H, int T, int D, cudaStream_t stream = 0);
void gemma_rms_norm_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* weight,
    __nv_bfloat16* out, int N, int D, float eps = 1e-6f, cudaStream_t stream = 0);

// ---- PixelNorm (over channel dim per spatial position) ----
void pixel_norm_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    int B, int C, int spatial, float eps = 1e-6f, cudaStream_t stream = 0);

// ---- Depth-to-Space 3D ----
void depth_to_space_3d_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    int B, int C, int D, int H, int W,
    int sd, int sh, int sw, cudaStream_t stream = 0);

// ---- VAE Unpatchify ----
void vae_unpatchify_bf16(
    const __nv_bfloat16* x, __nv_bfloat16* out,
    int B, int C, int F, int H, int W, int p, cudaStream_t stream = 0);

// ---- Nearest Upsample 2x ----
void nearest_upsample_2x_bf16(
    const __nv_bfloat16* in, __nv_bfloat16* out,
    int B, int C, int H, int W, cudaStream_t stream = 0);

// ---- Fused RMS Norm Channel + Gamma (single BF16 write) ----
void rms_norm_channel_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* gamma,
    __nv_bfloat16* out, int B, int C, int spatial,
    float eps = 1e-6f, cudaStream_t stream = 0);

// ---- Channel Scale: out[b,c,s] = x[b,c,s] * scale[c] ----
void channel_scale_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* scale,
    __nv_bfloat16* out, int B, int C, int spatial,
    cudaStream_t stream = 0);

// ---- Channel Scale+Shift: out[b,c,s] = x[b,c,s] * scale[c] + shift[c] ----
void channel_scale_shift_bf16(
    const __nv_bfloat16* x, const __nv_bfloat16* scale, const __nv_bfloat16* shift,
    __nv_bfloat16* out, int B, int C, int spatial,
    cudaStream_t stream = 0);

// ---- Head Transpose: [T, H*HD] -> [H, T, HD] ----
void head_transpose_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int H, int T, int HD, cudaStream_t stream = 0);

// ---- Head Untranspose: [H, T, HD] -> [T, H*HD] ----
void head_untranspose_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int H, int T, int HD, cudaStream_t stream = 0);

// ---- Batched Head Transpose: [B*S, H*HD] -> [B, H, S, HD] ----
void head_transpose_batched_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int B, int H, int S, int HD, cudaStream_t stream = 0);

// ---- Batched Head Untranspose: [B, H, S, HD] -> [B*S, H*HD] ----
void head_untranspose_batched_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int B, int H, int S, int HD, cudaStream_t stream = 0);

// ---- Patchify 3D: [B, C, T, H, W] -> [B, S, patch_dim] ----
void patchify_3d_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int B, int C, int T, int H, int W,
    int pT, int pH, int pW, cudaStream_t stream = 0);

// ---- Unpatchify 3D: [B, S, patch_dim] -> [B, C, T, H, W] ----
void unpatchify_3d_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int B, int C, int T, int H, int W,
    int pT, int pH, int pW, cudaStream_t stream = 0);

// ---- Causal Conv3D (im2col + GEMM, for VAE) ----
void causal_conv3d_forward(
    cublasHandle_t cublas,
    const __nv_bfloat16* input, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    __nv_bfloat16* output,
    int C_in, int T, int H, int W,
    int C_out, int kT, int kH, int kW,
    int padH, int padW,
    cudaStream_t stream = 0);

// ---- Native F32 ops for VAE ----
void silu_f32(const float* x, float* out, int64_t N, cudaStream_t stream = 0);
void add_f32(const float* a, const float* b, float* out, int64_t N, cudaStream_t stream = 0);
void pixel_norm_f32(float* x, int B, int C, int spatial, float eps = 1e-6f, cudaStream_t stream = 0);
void channel_scale_f32(const float* x, const float* scale, float* out, int B, int C, int spatial, cudaStream_t stream = 0);
void nearest_upsample_2x_f32(const float* in, float* out, int B, int C, int H, int W, cudaStream_t stream = 0);

// ---- Fused CFG + Euler Step ----
void cfg_euler_step_bf16(
    const __nv_bfloat16* sample,
    const __nv_bfloat16* noise_pos,
    const __nv_bfloat16* noise_neg,
    __nv_bfloat16* out,
    float cfg_scale, float sigma, float sigma_next,
    int64_t N, cudaStream_t stream = 0);

// ---- Fused CFG + Euler Ancestral RF Step ----
void cfg_euler_a_rf_step_bf16(
    const __nv_bfloat16* sample,
    const __nv_bfloat16* noise_pos,
    const __nv_bfloat16* noise_neg,
    const __nv_bfloat16* rand_noise,
    __nv_bfloat16* out,
    float cfg_scale, float sigma, float sigma_next,
    float eta, float s_noise,
    int64_t N, cudaStream_t stream = 0);
