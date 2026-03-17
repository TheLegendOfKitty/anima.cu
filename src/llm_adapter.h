#pragma once

#include "anima.h"
#include "tensor.h"
#include "linear.h"
#include <cublas_v2.h>
#include <vector>

// Anima LLM Adapter: converts Qwen3 hidden states + T5 token IDs into
// conditioning tensors for the diffusion transformer.
//
// Architecture:
//   embed: Embedding(32128, 1024) for T5 token IDs
//   6x AdapterBlock:
//     self-attention(1024, 16 heads, head_dim=64, RoPE theta=10000)
//     cross-attention(1024, 16 heads, context=Qwen3 hidden states)
//     MLP: Linear(1024, 4096, bias) -> GELU -> Linear(4096, 1024, bias)
//   out_proj: Linear(1024, 1024, bias)
//   norm: RMSNorm(1024)
//
// Output: [B, 512, 1024] (padded to 512 tokens)

struct AdapterAttention {
    Linear q_proj;    // [1024, 1024]
    Linear k_proj;    // [1024, 1024]
    Linear v_proj;    // [1024, 1024]
    Linear o_proj;    // [1024, 1024]
    Tensor q_norm_weight;  // [64] (head_dim)
    Tensor k_norm_weight;  // [64]
};

struct AdapterBlock {
    Tensor norm_self_attn_weight;   // [1024]
    AdapterAttention self_attn;
    Tensor norm_cross_attn_weight;  // [1024]
    AdapterAttention cross_attn;
    Tensor norm_mlp_weight;         // [1024]
    Linear mlp_fc1;   // [4096, 1024] with bias
    Linear mlp_fc2;   // [1024, 4096] with bias
};

class LLMAdapter {
public:
    LLMAdapter() = default;

    // Load weights from the transformer safetensors file.
    // Keys have "net.llm_adapter." prefix.
    void load(const class SafeTensorsFile& file);

    // Forward pass:
    //   source_hidden: [T_src, 1024] — Qwen3 output
    //   target_ids:    [T_tgt] — T5 token IDs (int32 on GPU)
    //   t5_weights:    [T_tgt, 1] — per-token weights (BF16 on GPU), or nullptr for all-ones
    // Returns: conditioning tensor [512, 1024] on GPU, padded to 512 tokens
    Tensor forward(const __nv_bfloat16* source_hidden, int T_src,
                   const int* target_ids, int T_tgt,
                   const __nv_bfloat16* t5_weights = nullptr);

private:
    cublasHandle_t cublas_ = nullptr;

    Tensor embed_weight_;  // [32128, 1024]
    std::vector<AdapterBlock> blocks_;
    Linear out_proj_;
    Tensor norm_weight_;   // [1024]

    // ---- Pre-allocated scratch buffers (reused across 6 blocks) ----
    struct Scratch {
        Tensor q_buf, k_buf, v_buf;         // attention projections
        Tensor q_heads, k_heads, v_heads;   // head-transposed
        Tensor scores;                       // [H, T_q, max_kv]
        Tensor attn_out;                     // [H, T_q, HD]
        Tensor attn_flat;                    // [T_q, D] untransposed output
        Tensor norm_buf;                     // [max(T_src, T_tgt), D]
        Tensor attn_buf;                     // [max(T_src, T_tgt), D]
        Tensor mlp_buf;                      // [T_tgt, 4096]
        Tensor mlp_out;                      // [T_tgt, D]
        int T_src = 0, T_tgt = 0;
    } scratch_;

    void ensure_scratch(int T_src, int T_tgt);

    // ---- RoPE cache (reused if max_len unchanged) ----
    Tensor rope_cos_cache_, rope_sin_cache_;
    int rope_cache_len_ = 0;

    // Run attention sublayer
    void run_attention(const AdapterAttention& attn,
                       const __nv_bfloat16* query_in, int T_q,
                       const __nv_bfloat16* kv_in, int T_kv,
                       __nv_bfloat16* out,
                       const float* cos_q, const float* sin_q,
                       const float* cos_k, const float* sin_k,
                       cudaStream_t stream);
};
