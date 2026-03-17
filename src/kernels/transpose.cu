/*
 * Transpose and reshape kernels:
 *   - head_transpose: [T, H*HD] -> [H, T, HD] for multi-head attention
 *   - head_untranspose: [H, T, HD] -> [T, H*HD] for output
 *   - patchify_3d: [B, C, T, H, W] -> [B, S, C*pT*pH*pW]
 *   - unpatchify_3d: [B, S, C*pT*pH*pW] -> [B, C, T, H, W]
 *   - cfg_euler_step: fused CFG + Euler step on GPU
 */

#include "kernels.h"
#include "../cuda_utils.cuh"

// ========================= Head Transpose =========================

// [T, H*HD] -> [H, T, HD]
// src[t, h*HD + d] -> dst[h, t, d]
__global__ void head_transpose_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int H, int T, int HD) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * T * HD;
    if (idx >= total) return;

    int d = idx % HD;
    int t = (idx / HD) % T;
    int h = idx / (T * HD);

    int64_t src_idx = (int64_t)t * H * HD + (int64_t)h * HD + d;
    dst[idx] = src[src_idx];
}

void head_transpose_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int H, int T, int HD, cudaStream_t stream) {
    int total = H * T * HD;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    head_transpose_kernel<<<blocks, threads, 0, stream>>>(src, dst, H, T, HD);
}

// [H, T, HD] -> [T, H*HD]
// src[h, t, d] -> dst[t, h*HD + d]
__global__ void head_untranspose_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int H, int T, int HD) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = H * T * HD;
    if (idx >= total) return;

    int d = idx % HD;
    int t = (idx / HD) % T;
    int h = idx / (T * HD);

    int64_t dst_idx = (int64_t)t * H * HD + (int64_t)h * HD + d;
    dst[dst_idx] = src[idx];
}

void head_untranspose_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int H, int T, int HD, cudaStream_t stream) {
    int total = H * T * HD;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;
    head_untranspose_kernel<<<blocks, threads, 0, stream>>>(src, dst, H, T, HD);
}

// ========================= Batched Head Transpose =========================

// [B*S, H*HD] -> [B, H, S, HD]
// src[b*S + s, h*HD + d] -> dst[b, h, s, d] = dst[(b*H + h)*S*HD + s*HD + d]
__global__ void head_transpose_batched_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int B, int H, int S, int HD) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * H * S * HD;
    if (idx >= total) return;

    int d  = idx % HD;
    int s  = (idx / HD) % S;
    int h  = (idx / (HD * S)) % H;
    int b  = idx / (HD * S * H);

    int64_t src_idx = ((int64_t)b * S + s) * H * HD + (int64_t)h * HD + d;
    dst[idx] = src[src_idx];
}

void head_transpose_batched_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int B, int H, int S, int HD, cudaStream_t stream) {
    int64_t total = (int64_t)B * H * S * HD;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    head_transpose_batched_kernel<<<blocks, threads, 0, stream>>>(src, dst, B, H, S, HD);
}

// [B, H, S, HD] -> [B*S, H*HD]
// src[b, h, s, d] -> dst[b*S + s, h*HD + d]
__global__ void head_untranspose_batched_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int B, int H, int S, int HD) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)B * H * S * HD;
    if (idx >= total) return;

    int d  = idx % HD;
    int s  = (idx / HD) % S;
    int h  = (idx / (HD * S)) % H;
    int b  = idx / (HD * S * H);

    int64_t dst_idx = ((int64_t)b * S + s) * H * HD + (int64_t)h * HD + d;
    dst[dst_idx] = src[idx];
}

void head_untranspose_batched_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int B, int H, int S, int HD, cudaStream_t stream) {
    int64_t total = (int64_t)B * H * S * HD;
    int threads = 256;
    int blocks = (int)((total + threads - 1) / threads);
    head_untranspose_batched_kernel<<<blocks, threads, 0, stream>>>(src, dst, B, H, S, HD);
}

// ========================= Patchify 3D =========================

// [B, C, T, H, W] -> [B, S, C*pT*pH*pW]
// S = (T/pT) * (H/pH) * (W/pW)
// Patch dim order: c, pt, ph, pw (c varies slowest)
__global__ void patchify_3d_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int C, int T, int H, int W,
    int pT, int pH, int pW,
    int pe_t, int pe_h, int pe_w,
    int S, int patch_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * patch_dim;  // per batch
    if (idx >= total) return;

    int tok = idx / patch_dim;
    int d   = idx % patch_dim;

    // Decode patch element: d -> (c, pt, ph, pw)
    int ppw = d % pW; d /= pW;
    int pph = d % pH; d /= pH;
    int pt  = d % pT; d /= pT;
    int c   = d;

    // Decode token index: tok -> (ft, fh, fw)
    int fw = tok % pe_w;
    int fh = (tok / pe_w) % pe_h;
    int ft = tok / (pe_w * pe_h);

    int t_idx = ft * pT + pt;
    int h_idx = fh * pH + pph;
    int w_idx = fw * pW + ppw;

    int64_t src_idx = (((int64_t)c * T + t_idx) * H + h_idx) * W + w_idx;
    dst[(int64_t)tok * patch_dim + (idx % patch_dim)] = src[src_idx];
}

void patchify_3d_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int B, int C, int T, int H, int W,
    int pT, int pH, int pW,
    cudaStream_t stream) {
    int pe_t = T / pT, pe_h = H / pH, pe_w = W / pW;
    int S = pe_t * pe_h * pe_w;
    int patch_dim = C * pT * pH * pW;
    int total = S * patch_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    for (int b = 0; b < B; b++) {
        patchify_3d_kernel<<<blocks, threads, 0, stream>>>(
            src + (int64_t)b * C * T * H * W,
            dst + (int64_t)b * S * patch_dim,
            C, T, H, W, pT, pH, pW, pe_t, pe_h, pe_w, S, patch_dim);
    }
}

// ========================= Unpatchify 3D =========================

// [B, S, patch_dim] -> [B, C, T, H, W]
// IMPORTANT: The Cosmos output projection produces elements in [pH, pW, pT, C] order
// (C varies fastest), NOT [C, pT, pH, pW] order like the patchify.
// See the NOTE in diffusers transformer_cosmos.py about permutation order.
__global__ void unpatchify_3d_kernel(
    const __nv_bfloat16* __restrict__ src,
    __nv_bfloat16* __restrict__ dst,
    int C, int T, int H, int W,
    int pT, int pH, int pW,
    int pe_t, int pe_h, int pe_w,
    int S, int patch_dim) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    int total = S * patch_dim;
    if (idx >= total) return;

    int tok = idx / patch_dim;
    int d   = idx % patch_dim;

    // Decode d as [pH, pW, pT, C] with C varying fastest
    // (matches Python's unflatten(2, (p_h, p_w, p_t, -1)))
    int d_tmp = d;
    int c   = d_tmp % C;   d_tmp /= C;
    int pt  = d_tmp % pT;  d_tmp /= pT;
    int ppw = d_tmp % pW;  d_tmp /= pW;
    int pph = d_tmp;

    int fw = tok % pe_w;
    int fh = (tok / pe_w) % pe_h;
    int ft = tok / (pe_w * pe_h);

    int t_idx = ft * pT + pt;
    int h_idx = fh * pH + pph;
    int w_idx = fw * pW + ppw;

    int64_t dst_idx = (((int64_t)c * T + t_idx) * H + h_idx) * W + w_idx;
    dst[dst_idx] = src[(int64_t)tok * patch_dim + d];
}

void unpatchify_3d_bf16(
    const __nv_bfloat16* src, __nv_bfloat16* dst,
    int B, int C, int T, int H, int W,
    int pT, int pH, int pW,
    cudaStream_t stream) {
    int pe_t = T / pT, pe_h = H / pH, pe_w = W / pW;
    int S = pe_t * pe_h * pe_w;
    int patch_dim = C * pT * pH * pW;
    int total = S * patch_dim;
    int threads = 256;
    int blocks = (total + threads - 1) / threads;

    for (int b = 0; b < B; b++) {
        unpatchify_3d_kernel<<<blocks, threads, 0, stream>>>(
            src + (int64_t)b * S * patch_dim,
            dst + (int64_t)b * C * T * H * W,
            C, T, H, W, pT, pH, pW, pe_t, pe_h, pe_w, S, patch_dim);
    }
}

// ========================= Fused CFG + Euler Step =========================

// noise = neg + cfg * (pos - neg)
// if last step: out = sample - sigma * noise
// else: out = sample + (sigma_next - sigma) * noise
__global__ void cfg_euler_step_kernel(
    const __nv_bfloat16* __restrict__ sample,
    const __nv_bfloat16* __restrict__ noise_pos,
    const __nv_bfloat16* __restrict__ noise_neg,
    __nv_bfloat16* __restrict__ out,
    float cfg_scale, float sigma, float sigma_next,
    int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float s = bf16_to_float(sample[idx]);
    float np = bf16_to_float(noise_pos[idx]);
    float nn = bf16_to_float(noise_neg[idx]);

    float noise = nn + cfg_scale * (np - nn);

    float result;
    if (sigma_next == 0.0f) {
        result = s - sigma * noise;
    } else {
        float denoised = s - sigma * noise;
        float d = (s - denoised) / sigma;  // = noise
        result = s + d * (sigma_next - sigma);
    }

    out[idx] = float_to_bf16(result);
}

void cfg_euler_step_bf16(
    const __nv_bfloat16* sample,
    const __nv_bfloat16* noise_pos,
    const __nv_bfloat16* noise_neg,
    __nv_bfloat16* out,
    float cfg_scale, float sigma, float sigma_next,
    int64_t N, cudaStream_t stream) {
    int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);
    cfg_euler_step_kernel<<<blocks, threads, 0, stream>>>(
        sample, noise_pos, noise_neg, out,
        cfg_scale, sigma, sigma_next, N);
}

// ========================= Fused CFG + Euler Ancestral RF Step =========================
//
// Python reference (sampling.py):
//   denoised = latents - sigma * noise_pred
//   downstep_ratio = 1 + (sigma_next/sigma - 1) * eta
//   sigma_down = sigma_next * downstep_ratio
//   alpha_ip1 = 1 - sigma_next
//   alpha_down = 1 - sigma_down
//   renoise_sq = sigma_next^2 - sigma_down^2 * alpha_ip1^2 / alpha_down^2
//   renoise_coeff = sqrt(max(0, renoise_sq))
//   sigma_down_ratio = sigma_down / sigma
//   latents = sigma_down_ratio * latents + (1 - sigma_down_ratio) * denoised
//   if eta > 0: latents = (alpha_ip1/alpha_down) * latents + noise * s_noise * renoise_coeff

__global__ void cfg_euler_a_rf_step_kernel(
    const __nv_bfloat16* __restrict__ sample,
    const __nv_bfloat16* __restrict__ noise_pos,
    const __nv_bfloat16* __restrict__ noise_neg,
    const __nv_bfloat16* __restrict__ rand_noise,
    __nv_bfloat16* __restrict__ out,
    float cfg_scale, float sigma, float sigma_next,
    float eta, float s_noise,
    float sigma_down_ratio, float alpha_ratio, float renoise_coeff,
    int64_t N) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    if (idx >= N) return;

    float s = bf16_to_float(sample[idx]);
    float np = bf16_to_float(noise_pos[idx]);
    float nn = bf16_to_float(noise_neg[idx]);

    float noise_pred = nn + cfg_scale * (np - nn);
    float denoised = s - sigma * noise_pred;

    // Ancestral step
    float result = sigma_down_ratio * s + (1.0f - sigma_down_ratio) * denoised;

    // Re-noise
    if (eta > 0.0f && renoise_coeff > 0.0f) {
        float rn = bf16_to_float(rand_noise[idx]);
        result = alpha_ratio * result + rn * s_noise * renoise_coeff;
    }

    out[idx] = float_to_bf16(result);
}

void cfg_euler_a_rf_step_bf16(
    const __nv_bfloat16* sample,
    const __nv_bfloat16* noise_pos,
    const __nv_bfloat16* noise_neg,
    const __nv_bfloat16* rand_noise,
    __nv_bfloat16* out,
    float cfg_scale, float sigma, float sigma_next,
    float eta, float s_noise,
    int64_t N, cudaStream_t stream) {
    // Precompute coefficients on CPU
    float downstep_ratio = 1.0f + (sigma_next / sigma - 1.0f) * eta;
    float sigma_down = sigma_next * downstep_ratio;
    float alpha_ip1 = 1.0f - sigma_next;
    float alpha_down = 1.0f - sigma_down;
    float renoise_sq = sigma_next * sigma_next
                     - sigma_down * sigma_down * alpha_ip1 * alpha_ip1 / (alpha_down * alpha_down);
    float renoise_coeff = (renoise_sq > 0.0f) ? sqrtf(renoise_sq) : 0.0f;
    float sigma_down_ratio = sigma_down / sigma;
    float alpha_ratio = alpha_ip1 / alpha_down;

    int threads = 256;
    int blocks = (int)((N + threads - 1) / threads);
    cfg_euler_a_rf_step_kernel<<<blocks, threads, 0, stream>>>(
        sample, noise_pos, noise_neg, rand_noise, out,
        cfg_scale, sigma, sigma_next, eta, s_noise,
        sigma_down_ratio, alpha_ratio, renoise_coeff, N);
}
