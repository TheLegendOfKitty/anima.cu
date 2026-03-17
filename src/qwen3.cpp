#include "qwen3.h"
#include "safetensors.h"
#include "kernels/kernels.h"
#include "cuda_utils.cuh"

#include <cmath>
#include <cstdio>
#include <algorithm>

// ========================= RoPE Cache =========================

static void build_rope_cache(float* cos_out, float* sin_out,
                              int max_len, int head_dim, float theta) {
    int half = head_dim / 2;
    for (int t = 0; t < max_len; t++) {
        for (int d = 0; d < half; d++) {
            float freq = 1.0f / std::pow(theta, (float)(2 * d) / head_dim);
            float angle = (float)t * freq;
            cos_out[t * half + d] = std::cos(angle);
            sin_out[t * half + d] = std::sin(angle);
        }
    }
}

// ========================= Load =========================

void Qwen3Encoder::load(const SafeTensorsFile& file) {
    fprintf(stderr, "[qwen3] loading encoder weights\n");

    CUBLAS_CHECK(cublasCreate(&cublas_));

    // Helper to load with "model." prefix
    auto load_t = [&](const std::string& name) -> Tensor {
        return file.load("model." + name);
    };

    auto has_t = [&](const std::string& name) -> bool {
        return file.has("model." + name);
    };

    // Embedding
    embed_tokens_ = load_t("embed_tokens.weight");
    fprintf(stderr, "[qwen3]   embed_tokens: [%lld, %lld]\n",
            (long long)embed_tokens_.dim(0), (long long)embed_tokens_.dim(1));

    // Final norm
    final_norm_weight_ = load_t("norm.weight");

    // Layers
    layers_.resize(QWEN3_LAYERS);
    for (int i = 0; i < QWEN3_LAYERS; i++) {
        auto& L = layers_[i];
        std::string lp = "layers." + std::to_string(i);

        auto load_lin = [&](Linear& lin, const std::string& suffix) {
            Tensor w = load_t(lp + "." + suffix + ".weight");
            Tensor b = has_t(lp + "." + suffix + ".bias")
                       ? load_t(lp + "." + suffix + ".bias") : Tensor();
            lin.load(std::move(w), std::move(b));
        };

        load_lin(L.q_proj, "self_attn.q_proj");
        load_lin(L.k_proj, "self_attn.k_proj");
        load_lin(L.v_proj, "self_attn.v_proj");
        load_lin(L.o_proj, "self_attn.o_proj");

        L.q_norm_weight = load_t(lp + ".self_attn.q_norm.weight");
        L.k_norm_weight = load_t(lp + ".self_attn.k_norm.weight");

        load_lin(L.gate_proj, "mlp.gate_proj");
        load_lin(L.up_proj, "mlp.up_proj");
        load_lin(L.down_proj, "mlp.down_proj");

        L.input_layernorm_weight = load_t(lp + ".input_layernorm.weight");
        L.post_attention_layernorm_weight = load_t(lp + ".post_attention_layernorm.weight");

        if ((i + 1) % 10 == 0 || i == QWEN3_LAYERS - 1) {
            fprintf(stderr, "[qwen3]   loaded layer %d/%d\n", i + 1, QWEN3_LAYERS);
        }
    }

    fprintf(stderr, "[qwen3] encoder loaded: %d layers\n", QWEN3_LAYERS);
}

// ========================= Forward Layer =========================

void Qwen3Encoder::forward_layer(int li, __nv_bfloat16* x, int T,
                                  const float* cos_cache, const float* sin_cache,
                                  cudaStream_t stream) {
    auto& L = layers_[li];
    constexpr int D = QWEN3_HIDDEN;        // 1024
    constexpr int QH = QWEN3_HEADS;        // 16
    constexpr int KVH = QWEN3_KV_HEADS;    // 8
    constexpr int HD = QWEN3_HEAD_DIM;     // 128
    constexpr int QD = QH * HD;            // 2048
    constexpr int KVD = KVH * HD;          // 1024
    constexpr int INTER = QWEN3_INTERMEDIATE; // 3072

    // Ensure scratch buffers
    if (norm_buf_.empty() || norm_buf_.numel() < (int64_t)T * D) {
        norm_buf_ = Tensor({(int64_t)T, (int64_t)D}, DType::BF16);
        q_buf_ = Tensor({(int64_t)T, (int64_t)QD}, DType::BF16);
        k_buf_ = Tensor({(int64_t)T, (int64_t)KVD}, DType::BF16);
        v_buf_ = Tensor({(int64_t)T, (int64_t)KVD}, DType::BF16);
    }

    // ---- Self-attention ----

    // 1. Input layernorm: RMSNorm(x)
    rms_norm_bf16(x, L.input_layernorm_weight.bf16_ptr(),
                  norm_buf_.bf16_ptr(), T, D, QWEN3_RMS_EPS, stream);

    // 2. QKV projections
    L.q_proj.forward(cublas_, norm_buf_.bf16_ptr(), q_buf_.bf16_ptr(), T, stream);
    L.k_proj.forward(cublas_, norm_buf_.bf16_ptr(), k_buf_.bf16_ptr(), T, stream);
    L.v_proj.forward(cublas_, norm_buf_.bf16_ptr(), v_buf_.bf16_ptr(), T, stream);

    // 3. Reshape to [num_heads, T, head_dim] for attention
    Tensor q_heads({(int64_t)QH, (int64_t)T, (int64_t)HD}, DType::BF16);
    Tensor k_heads({(int64_t)KVH, (int64_t)T, (int64_t)HD}, DType::BF16);
    Tensor v_heads({(int64_t)KVH, (int64_t)T, (int64_t)HD}, DType::BF16);

    // Transpose [T, H*HD] -> [H, T, HD] via GPU kernel
    head_transpose_bf16(q_buf_.bf16_ptr(), q_heads.bf16_ptr(), QH, T, HD, stream);
    head_transpose_bf16(k_buf_.bf16_ptr(), k_heads.bf16_ptr(), KVH, T, HD, stream);
    head_transpose_bf16(v_buf_.bf16_ptr(), v_heads.bf16_ptr(), KVH, T, HD, stream);

    // 4. Per-head QK normalization: RMSNorm each head's Q and K
    // Q: [QH*T, HD] — normalize each row of HD elements
    rms_norm_bf16(q_heads.bf16_ptr(), L.q_norm_weight.bf16_ptr(),
                  q_heads.bf16_ptr(), QH * T, HD, QWEN3_RMS_EPS, stream);
    rms_norm_bf16(k_heads.bf16_ptr(), L.k_norm_weight.bf16_ptr(),
                  k_heads.bf16_ptr(), KVH * T, HD, QWEN3_RMS_EPS, stream);

    // 5. Apply RoPE to Q and K
    rope_standard_bf16(q_heads.bf16_ptr(), cos_cache, sin_cache,
                       QH * T, T, HD, stream);
    rope_standard_bf16(k_heads.bf16_ptr(), cos_cache, sin_cache,
                       KVH * T, T, HD, stream);

    // 6. Expand KV heads for GQA: [KVH, T, HD] -> [QH, T, HD]
    Tensor k_expanded({(int64_t)QH, (int64_t)T, (int64_t)HD}, DType::BF16);
    Tensor v_expanded({(int64_t)QH, (int64_t)T, (int64_t)HD}, DType::BF16);
    expand_kv_bf16(k_heads.bf16_ptr(), k_expanded.bf16_ptr(), 1, KVH, QH, T, HD, stream);
    expand_kv_bf16(v_heads.bf16_ptr(), v_expanded.bf16_ptr(), 1, KVH, QH, T, HD, stream);

    // 7. Attention: scores = Q @ K^T / sqrt(HD)
    float scale = 1.0f / std::sqrt((float)HD);

    CUBLAS_CHECK(cublasSetStream(cublas_, stream));

    Tensor scores({(int64_t)QH, (int64_t)T, (int64_t)T}, DType::BF16);
    {
        float alpha = scale, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
            T, T, HD, &alpha,
            k_expanded.bf16_ptr(), CUDA_R_16BF, HD, (int64_t)T * HD,
            q_heads.bf16_ptr(), CUDA_R_16BF, HD, (int64_t)T * HD,
            &beta,
            scores.bf16_ptr(), CUDA_R_16BF, T, (int64_t)T * T,
            QH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Apply causal mask (Qwen3 is a causal LM)
    causal_mask_bf16(scores.bf16_ptr(), QH, T, stream);

    // Softmax each row
    softmax_bf16(scores.bf16_ptr(), scores.bf16_ptr(), QH * T, T, stream);

    // attn_out = scores @ V: [QH, T, HD]
    Tensor attn_out({(int64_t)QH, (int64_t)T, (int64_t)HD}, DType::BF16);
    {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
            HD, T, T, &alpha,
            v_expanded.bf16_ptr(), CUDA_R_16BF, HD, (int64_t)T * HD,
            scores.bf16_ptr(), CUDA_R_16BF, T, (int64_t)T * T,
            &beta,
            attn_out.bf16_ptr(), CUDA_R_16BF, HD, (int64_t)T * HD,
            QH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // 8. Transpose back: [QH, T, HD] -> [T, QH*HD] via GPU kernel
    Tensor attn_flat({(int64_t)T, (int64_t)QD}, DType::BF16);
    head_untranspose_bf16(attn_out.bf16_ptr(), attn_flat.bf16_ptr(), QH, T, HD, stream);

    // 9. Output projection
    Tensor o_out({(int64_t)T, (int64_t)D}, DType::BF16);
    L.o_proj.forward(cublas_, attn_flat.bf16_ptr(), o_out.bf16_ptr(), T, stream);

    // Residual add
    add_bf16(x, o_out.bf16_ptr(), x, (int64_t)T * D, stream);

    // ---- MLP (SwiGLU) ----

    // Post-attention layernorm
    rms_norm_bf16(x, L.post_attention_layernorm_weight.bf16_ptr(),
                  norm_buf_.bf16_ptr(), T, D, QWEN3_RMS_EPS, stream);

    // Gate + Up projections
    Tensor gate_buf({(int64_t)T, (int64_t)INTER}, DType::BF16);
    Tensor up_buf({(int64_t)T, (int64_t)INTER}, DType::BF16);
    L.gate_proj.forward(cublas_, norm_buf_.bf16_ptr(), gate_buf.bf16_ptr(), T, stream);
    L.up_proj.forward(cublas_, norm_buf_.bf16_ptr(), up_buf.bf16_ptr(), T, stream);

    // SwiGLU: silu(gate) * up  (reuse gated_gelu_bf16 but need SiLU variant)
    // Actually we need silu, not gelu. Let me use silu then mul.
    silu_bf16(gate_buf.bf16_ptr(), gate_buf.bf16_ptr(), (int64_t)T * INTER, stream);
    mul_bf16(gate_buf.bf16_ptr(), up_buf.bf16_ptr(), gate_buf.bf16_ptr(),
             (int64_t)T * INTER, stream);

    // Down projection
    Tensor mlp_out({(int64_t)T, (int64_t)D}, DType::BF16);
    L.down_proj.forward(cublas_, gate_buf.bf16_ptr(), mlp_out.bf16_ptr(), T, stream);

    // Residual add
    add_bf16(x, mlp_out.bf16_ptr(), x, (int64_t)T * D, stream);
}

// ========================= Forward =========================

Tensor Qwen3Encoder::forward(const int* input_ids, const int* attention_mask, int T) {
    fprintf(stderr, "[qwen3] forward: T=%d\n", T);

    // 1. Embedding lookup
    Tensor x({(int64_t)T, (int64_t)QWEN3_HIDDEN}, DType::BF16);
    embedding_lookup_bf16(embed_tokens_.bf16_ptr(), input_ids,
                          x.bf16_ptr(), T, QWEN3_HIDDEN);

    // 2. Build RoPE cache [T, HD/2] as float on GPU
    int half_hd = QWEN3_HEAD_DIM / 2;
    std::vector<float> cos_host(T * half_hd), sin_host(T * half_hd);
    build_rope_cache(cos_host.data(), sin_host.data(), T, QWEN3_HEAD_DIM, QWEN3_ROPE_THETA);

    Tensor cos_gpu({(int64_t)T, (int64_t)half_hd}, DType::F32);
    Tensor sin_gpu({(int64_t)T, (int64_t)half_hd}, DType::F32);
    cos_gpu.copy_from_host(cos_host.data(), cos_host.size() * sizeof(float));
    sin_gpu.copy_from_host(sin_host.data(), sin_host.size() * sizeof(float));

    // 3. Run all 28 layers
    for (int i = 0; i < QWEN3_LAYERS; i++) {
        forward_layer(i, x.bf16_ptr(), T,
                      cos_gpu.f32_ptr(), sin_gpu.f32_ptr(), 0);

        if ((i + 1) % 10 == 0) {
            fprintf(stderr, "[qwen3]   layer %d/%d done\n", i + 1, QWEN3_LAYERS);
        }
    }

    // 4. Final RMSNorm
    rms_norm_bf16(x.bf16_ptr(), final_norm_weight_.bf16_ptr(),
                  x.bf16_ptr(), T, QWEN3_HIDDEN, QWEN3_RMS_EPS);

    fprintf(stderr, "[qwen3] forward complete: [%d, %d]\n", T, QWEN3_HIDDEN);
    return x;
}
