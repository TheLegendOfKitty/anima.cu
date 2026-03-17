#pragma once

#include "anima.h"
#include "tensor.h"
#include "linear.h"
#include <cublas_v2.h>
#include <vector>

// CosmosTransformer3DModel for Anima diffusion.
//
// 28 transformer blocks with adaLN-Zero modulation.
// Self-attention + cross-attention + GELU FFN.
// Cosmos 3D RoPE for spatial/temporal positional encoding.
//
// Input:  latents [B, 16, 1, H, W] + timestep scalar
// Output: noise prediction [B, 16, 1, H, W]

struct CosmosAdaLNZero {
    // SiLU(embedded_timestep) -> linear_1(2048, 256) -> linear_2(256, 6144) -> chunk(3) -> (shift, scale, gate)
    Linear linear_1;  // [256, 2048]
    Linear linear_2;  // [6144, 256]
};

struct CosmosAdaLN {
    // Output norm (shift + scale only, no gate) -> linear_2 outputs 4096 = 2*2048
    Linear linear_1;  // [256, 2048]
    Linear linear_2;  // [4096, 256]
};

struct CosmosTransformerBlock {
    CosmosAdaLNZero norm1;  // self-attention adaLN
    CosmosAdaLNZero norm2;  // cross-attention adaLN
    CosmosAdaLNZero norm3;  // FFN adaLN

    // Self-attention (2048-dim, 16 heads, head_dim=128)
    Linear sa_q_proj;  // [2048, 2048]
    Linear sa_k_proj;  // [2048, 2048]
    Linear sa_v_proj;  // [2048, 2048]
    Linear sa_o_proj;  // [2048, 2048]
    Tensor sa_q_norm;  // [128]
    Tensor sa_k_norm;  // [128]

    // Cross-attention (Q from 2048, K/V from 1024)
    Linear ca_q_proj;  // [2048, 2048]
    Linear ca_k_proj;  // [2048, 1024]
    Linear ca_v_proj;  // [2048, 1024]
    Linear ca_o_proj;  // [2048, 2048]
    Tensor ca_q_norm;  // [128]
    Tensor ca_k_norm;  // [128]

    // Feed-forward
    Linear ff_proj1;  // [8192, 2048] (GELU gate)
    Linear ff_proj2;  // [2048, 8192]
};

class CosmosTransformer {
public:
    CosmosTransformer() = default;

    // Load from safetensors file (keys have "net." prefix in Anima format).
    void load(const class SafeTensorsFile& file);

    // Forward pass: single denoising step.
    //   latents:       [B, 16, 1, latent_h, latent_w] BF16
    //   timestep:      scalar (sigma * ANIMA_SAMPLING_MULTIPLIER)
    //   encoder_cond:  [B, S_text, 1024] BF16 (text conditioning)
    // Returns: noise prediction [B, 16, 1, latent_h, latent_w] BF16
    Tensor forward(const Tensor& latents, float timestep,
                   const __nv_bfloat16* encoder_cond, int S_text,
                   int batch_size, int latent_h, int latent_w);

private:
    cublasHandle_t cublas_ = nullptr;

    // Patch embedding
    Linear patch_proj_;  // [2048, 68] (17 channels * 1*2*2 patch)

    // Timestep embedding
    Tensor time_norm_weight_;   // [2048] RMSNorm
    Linear time_linear_1_;      // [2048, 2048]
    Linear time_linear_2_;      // [6144, 2048]

    // Transformer blocks
    std::vector<CosmosTransformerBlock> blocks_;

    // Output
    CosmosAdaLN output_norm_;
    Linear output_proj_;  // [64, 2048]

    // Helpers
    void compute_timestep_embedding(float timestep, __nv_bfloat16* embedded_ts,
                                     __nv_bfloat16* temb, cudaStream_t stream);
    void compute_3d_rope(int num_frames, int height, int width,
                          Tensor& cos_out, Tensor& sin_out);
    void forward_block(int block_idx,
                        __nv_bfloat16* hidden, int S,
                        const __nv_bfloat16* encoder_cond, int S_text,
                        const __nv_bfloat16* embedded_ts,
                        const __nv_bfloat16* temb,
                        const __nv_bfloat16* rope_cos,
                        const __nv_bfloat16* rope_sin,
                        int batch_size,
                        cudaStream_t stream);
};
