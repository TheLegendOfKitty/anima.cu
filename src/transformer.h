#pragma once

#include "anima.h"
#include "tensor.h"
#include "linear.h"
#include "cudnn_sdpa.h"
#include <cublas_v2.h>
#include <cudnn.h>
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
    //   output_buf:    caller-provided buffer [B, 16, 1, H, W] BF16 to write result into
    void forward(const Tensor& latents, float timestep,
                 const __nv_bfloat16* encoder_cond, int S_text,
                 int batch_size, int latent_h, int latent_w,
                 __nv_bfloat16* output_buf);

private:
    cublasHandle_t cublas_ = nullptr;
    cudnnHandle_t cudnn_ = nullptr;
    CudnnSDPA sdpa_;

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

    // ---- RoPE cache (reused across denoising steps for same resolution) ----
    Tensor rope_cos_cache_, rope_sin_cache_;
    int rope_cache_h_ = 0, rope_cache_w_ = 0;

    // ---- Pre-allocated scratch buffers (reused across all 28 blocks and steps) ----
    struct Scratch {
        // adaLN outputs (reused across 3 sub-layers per block)
        Tensor mod;          // [6144] BF16
        Tensor normed;       // [S, D] BF16
        Tensor gate;         // [D] BF16
        Tensor sub_out;      // [S, D] BF16 — for sa_out, ca_out, ff_out

        // FFN
        Tensor ff_buf;       // [S, MLP_DIM] BF16

        // Attention internals (sized for self-attn = max, reused for cross-attn)
        Tensor q_buf;        // [S, D] BF16
        Tensor k_buf;        // [max_kv, D] BF16
        Tensor v_buf;        // [max_kv, D] BF16
        Tensor q_h;          // [H, S, HD] BF16
        Tensor k_h;          // [H, max_kv, HD] BF16
        Tensor v_h;          // [H, max_kv, HD] BF16
        Tensor attn_out;     // [H, S, HD] BF16
        Tensor attn_flat;    // [S, D] BF16

        // Cached per-forward-call (SiLU(embedded_ts) computed once, reused 84+ times)
        Tensor silu_ts;      // [D] BF16
        Tensor adaln_mid;    // [ADALN_DIM] BF16

        // Timestep embedding buffers (reused across denoising steps)
        Tensor embedded_ts;  // [D] BF16
        Tensor temb;         // [6144] BF16
        Tensor ts_sin_buf;   // [D] BF16 — sinusoidal upload in compute_timestep_embedding
        Tensor ts_mlp_buf;   // [D] BF16 — MLP buffer in compute_timestep_embedding

        // Forward-pass temporaries (reused across calls, eliminates per-call cudaMalloc)
        Tensor padded;       // [B, 17, 1, H, W] BF16 — also backs proj_out (non-overlapping lifetimes)
        Tensor patches;      // [B*S, patch_dim] BF16
        Tensor hidden;       // [B*S, D] BF16
        Tensor proj_out;     // [B*S, out_patch_dim] BF16 — non-owning view into padded's memory

        int S = 0;
        int S_text = 0;
    } scratch_;

    // Allocate/resize scratch buffers for given spatial and text token counts.
    void ensure_scratch(int S, int S_text);

    void compute_timestep_embedding(float timestep, __nv_bfloat16* embedded_ts,
                                     __nv_bfloat16* temb, cudaStream_t stream);

    // Compute and cache 3D RoPE (no-op if resolution unchanged).
    void compute_3d_rope(int num_frames, int height, int width);

    // Run a single transformer block using scratch buffers.
    void forward_block(int block_idx,
                        __nv_bfloat16* hidden, int S,
                        const __nv_bfloat16* encoder_cond, int S_text,
                        const __nv_bfloat16* temb,
                        int batch_size,
                        cudaStream_t stream);

    // Attention sub-layer using pre-allocated scratch buffers.
    void run_attention(const Linear& q_proj, const Linear& k_proj,
                       const Linear& v_proj, const Linear& o_proj,
                       const Tensor& q_norm_w, const Tensor& k_norm_w,
                       const __nv_bfloat16* q_input, int T_q,
                       const __nv_bfloat16* kv_input, int T_kv,
                       __nv_bfloat16* output,
                       bool apply_rope,
                       int batch_size,
                       cudaStream_t stream);
};
