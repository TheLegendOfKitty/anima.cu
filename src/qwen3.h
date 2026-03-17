#pragma once

#include "anima.h"
#include "tensor.h"
#include "linear.h"
#include <cublas_v2.h>
#include <string>
#include <vector>

// Qwen3-0.6B text encoder for Anima.
// GQA: 16 Q heads, 8 KV heads, head_dim=128
// 28 layers, hidden=1024, intermediate=3072, SwiGLU MLP
// Output: last_hidden_state [T, 1024]

struct Qwen3Layer {
    // Self-attention
    Linear q_proj;   // [2048, 1024]  (16 heads * 128 head_dim)
    Linear k_proj;   // [1024, 1024]  (8 kv_heads * 128 head_dim)
    Linear v_proj;   // [1024, 1024]
    Linear o_proj;   // [1024, 2048]

    // QK normalization (per head_dim)
    Tensor q_norm_weight;  // [128]
    Tensor k_norm_weight;  // [128]

    // MLP (SwiGLU)
    Linear gate_proj;  // [3072, 1024]
    Linear up_proj;    // [3072, 1024]
    Linear down_proj;  // [1024, 3072]

    // Norms
    Tensor input_layernorm_weight;       // [1024]
    Tensor post_attention_layernorm_weight;  // [1024]
};

class Qwen3Encoder {
public:
    Qwen3Encoder() = default;

    // Load weights from a single safetensors file.
    // Keys have "model." prefix which is stripped.
    void load(const class SafeTensorsFile& file);

    // Forward pass: returns hidden states [T, 1024] on GPU.
    // input_ids: [T] token IDs on GPU (as int32)
    // attention_mask: [T] attention mask on GPU (as int32, 1=real, 0=pad)
    Tensor forward(const int* input_ids, const int* attention_mask, int T);

private:
    cublasHandle_t cublas_ = nullptr;

    Tensor embed_tokens_;    // [151936, 1024] BF16
    Tensor final_norm_weight_;  // [1024]
    std::vector<Qwen3Layer> layers_;

    // ---- Pre-allocated scratch buffers (reused across all 28 layers) ----
    struct Scratch {
        Tensor norm_buf;     // [T, D]
        Tensor q_buf;        // [T, QD]
        Tensor k_buf;        // [T, KVD]
        Tensor v_buf;        // [T, KVD]
        Tensor q_heads;      // [QH, T, HD]
        Tensor k_heads;      // [KVH, T, HD]
        Tensor v_heads;      // [KVH, T, HD]
        Tensor k_expanded;   // [QH, T, HD]
        Tensor v_expanded;   // [QH, T, HD]
        Tensor scores;       // [QH, T, T] BF16
        Tensor attn_out;     // [QH, T, HD]
        Tensor attn_flat;    // [T, QD]
        Tensor o_out;        // [T, D]
        Tensor gate_buf;     // [T, INTER]
        Tensor up_buf;       // [T, INTER]
        Tensor mlp_out;      // [T, D]
        int T = 0;
    } scratch_;

    void ensure_scratch(int T);

    // ---- RoPE cache (reused if T unchanged) ----
    Tensor rope_cos_cache_, rope_sin_cache_;
    int rope_cache_T_ = 0;

    void forward_layer(int layer_idx, __nv_bfloat16* x, int T,
                       const float* cos_cache, const float* sin_cache,
                       cudaStream_t stream);
};
