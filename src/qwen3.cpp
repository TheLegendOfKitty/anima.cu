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

// ========================= ensure_scratch =========================

void Qwen3Encoder::ensure_scratch(int T) {
    if (T <= scratch_.T) return;  // high-water-mark: only grow

    constexpr int D = QWEN3_HIDDEN;
    constexpr int QH = QWEN3_HEADS;
    constexpr int KVH = QWEN3_KV_HEADS;
    constexpr int HD = QWEN3_HEAD_DIM;
    constexpr int QD = QH * HD;
    constexpr int KVD = KVH * HD;
    constexpr int INTER = QWEN3_INTERMEDIATE;

    fprintf(stderr, "[qwen3] allocating scratch buffers: T=%d\n", T);

    scratch_.norm_buf = Tensor({(int64_t)T, (int64_t)D}, DType::BF16);
    scratch_.q_buf = Tensor({(int64_t)T, (int64_t)QD}, DType::BF16);
    scratch_.k_buf = Tensor({(int64_t)T, (int64_t)KVD}, DType::BF16);
    scratch_.v_buf = Tensor({(int64_t)T, (int64_t)KVD}, DType::BF16);
    scratch_.q_heads = Tensor({(int64_t)QH, (int64_t)T, (int64_t)HD}, DType::BF16);
    scratch_.k_heads = Tensor({(int64_t)KVH, (int64_t)T, (int64_t)HD}, DType::BF16);
    scratch_.v_heads = Tensor({(int64_t)KVH, (int64_t)T, (int64_t)HD}, DType::BF16);
    scratch_.k_expanded = Tensor({(int64_t)QH, (int64_t)T, (int64_t)HD}, DType::BF16);
    scratch_.v_expanded = Tensor({(int64_t)QH, (int64_t)T, (int64_t)HD}, DType::BF16);
    scratch_.scores = Tensor({(int64_t)QH, (int64_t)T, (int64_t)T}, DType::BF16);
    scratch_.attn_out = Tensor({(int64_t)QH, (int64_t)T, (int64_t)HD}, DType::BF16);
    scratch_.attn_flat = Tensor({(int64_t)T, (int64_t)QD}, DType::BF16);
    scratch_.o_out = Tensor({(int64_t)T, (int64_t)D}, DType::BF16);
    scratch_.gate_buf = Tensor({(int64_t)T, (int64_t)INTER}, DType::BF16);
    scratch_.up_buf = Tensor({(int64_t)T, (int64_t)INTER}, DType::BF16);
    scratch_.mlp_out = Tensor({(int64_t)T, (int64_t)D}, DType::BF16);

    scratch_.T = T;

    size_t total = 0;
    auto add = [&](const Tensor& t) { total += t.size_bytes(); };
    add(scratch_.norm_buf); add(scratch_.q_buf); add(scratch_.k_buf); add(scratch_.v_buf);
    add(scratch_.q_heads); add(scratch_.k_heads); add(scratch_.v_heads);
    add(scratch_.k_expanded); add(scratch_.v_expanded);
    add(scratch_.scores); add(scratch_.attn_out); add(scratch_.attn_flat);
    add(scratch_.o_out); add(scratch_.gate_buf); add(scratch_.up_buf); add(scratch_.mlp_out);
    fprintf(stderr, "[qwen3] scratch memory: %.1f MB\n", total / (1024.0 * 1024.0));
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

    // Use pre-allocated scratch buffers
    auto* norm_buf = scratch_.norm_buf.bf16_ptr();
    auto* q_buf = scratch_.q_buf.bf16_ptr();
    auto* k_buf = scratch_.k_buf.bf16_ptr();
    auto* v_buf = scratch_.v_buf.bf16_ptr();
    auto* q_heads = scratch_.q_heads.bf16_ptr();
    auto* k_heads = scratch_.k_heads.bf16_ptr();
    auto* v_heads = scratch_.v_heads.bf16_ptr();

    // ---- Self-attention ----

    // 1. Input layernorm: RMSNorm(x)
    rms_norm_bf16(x, L.input_layernorm_weight.bf16_ptr(),
                  norm_buf, T, D, QWEN3_RMS_EPS, stream);

    // 2. QKV projections
    L.q_proj.forward(cublas_, norm_buf, q_buf, T, stream);
    L.k_proj.forward(cublas_, norm_buf, k_buf, T, stream);
    L.v_proj.forward(cublas_, norm_buf, v_buf, T, stream);

    // 3. Reshape to [num_heads, T, head_dim] for attention
    head_transpose_bf16(q_buf, q_heads, QH, T, HD, stream);
    head_transpose_bf16(k_buf, k_heads, KVH, T, HD, stream);
    head_transpose_bf16(v_buf, v_heads, KVH, T, HD, stream);

    // 4. Per-head QK normalization: RMSNorm each head's Q and K
    rms_norm_bf16(q_heads, L.q_norm_weight.bf16_ptr(),
                  q_heads, QH * T, HD, QWEN3_RMS_EPS, stream);
    rms_norm_bf16(k_heads, L.k_norm_weight.bf16_ptr(),
                  k_heads, KVH * T, HD, QWEN3_RMS_EPS, stream);

    // 5. Apply RoPE to Q and K
    rope_standard_bf16(q_heads, cos_cache, sin_cache,
                       QH * T, T, HD, stream);
    rope_standard_bf16(k_heads, cos_cache, sin_cache,
                       KVH * T, T, HD, stream);

    // 6. Expand KV heads for GQA: [KVH, T, HD] -> [QH, T, HD]
    auto* k_expanded = scratch_.k_expanded.bf16_ptr();
    auto* v_expanded = scratch_.v_expanded.bf16_ptr();
    expand_kv_bf16(k_heads, k_expanded, 1, KVH, QH, T, HD, stream);
    expand_kv_bf16(v_heads, v_expanded, 1, KVH, QH, T, HD, stream);

    // 7. Attention: scores = Q @ K^T / sqrt(HD)
    float scale = 1.0f / std::sqrt((float)HD);

    CUBLAS_CHECK(cublasSetStream(cublas_, stream));

    auto* scores = scratch_.scores.bf16_ptr();
    {
        float alpha = scale, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
            T, T, HD, &alpha,
            k_expanded, CUDA_R_16BF, HD, (int64_t)T * HD,
            q_heads, CUDA_R_16BF, HD, (int64_t)T * HD,
            &beta,
            scores, CUDA_R_16BF, T, (int64_t)T * T,
            QH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Apply causal mask (Qwen3 is a causal LM)
    causal_mask_bf16(scores, QH, T, stream);

    // Softmax each row
    softmax_bf16(scores, scores, QH * T, T, stream);

    // attn_out = scores @ V: [QH, T, HD]
    auto* attn_out = scratch_.attn_out.bf16_ptr();
    {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
            HD, T, T, &alpha,
            v_expanded, CUDA_R_16BF, HD, (int64_t)T * HD,
            scores, CUDA_R_16BF, T, (int64_t)T * T,
            &beta,
            attn_out, CUDA_R_16BF, HD, (int64_t)T * HD,
            QH, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // 8. Transpose back: [QH, T, HD] -> [T, QH*HD] via GPU kernel
    auto* attn_flat = scratch_.attn_flat.bf16_ptr();
    head_untranspose_bf16(attn_out, attn_flat, QH, T, HD, stream);

    // 9. Output projection
    auto* o_out = scratch_.o_out.bf16_ptr();
    L.o_proj.forward(cublas_, attn_flat, o_out, T, stream);

    // Residual add
    add_bf16(x, o_out, x, (int64_t)T * D, stream);

    // ---- MLP (SwiGLU) ----

    // Post-attention layernorm
    rms_norm_bf16(x, L.post_attention_layernorm_weight.bf16_ptr(),
                  norm_buf, T, D, QWEN3_RMS_EPS, stream);

    // Gate + Up projections
    auto* gate_buf = scratch_.gate_buf.bf16_ptr();
    auto* up_buf = scratch_.up_buf.bf16_ptr();
    L.gate_proj.forward(cublas_, norm_buf, gate_buf, T, stream);
    L.up_proj.forward(cublas_, norm_buf, up_buf, T, stream);

    // SwiGLU: silu(gate) * up
    silu_bf16(gate_buf, gate_buf, (int64_t)T * INTER, stream);
    mul_bf16(gate_buf, up_buf, gate_buf,
             (int64_t)T * INTER, stream);

    // Down projection
    auto* mlp_out = scratch_.mlp_out.bf16_ptr();
    L.down_proj.forward(cublas_, gate_buf, mlp_out, T, stream);

    // Residual add
    add_bf16(x, mlp_out, x, (int64_t)T * D, stream);
}

// ========================= Forward =========================

Tensor Qwen3Encoder::forward(const int* input_ids, const int* attention_mask, int T) {
    fprintf(stderr, "[qwen3] forward: T=%d\n", T);

    // Ensure scratch buffers (no-op if T unchanged)
    ensure_scratch(T);

    // 1. Embedding lookup
    Tensor x({(int64_t)T, (int64_t)QWEN3_HIDDEN}, DType::BF16);
    embedding_lookup_bf16(embed_tokens_.bf16_ptr(), input_ids,
                          x.bf16_ptr(), T, QWEN3_HIDDEN);

    // 2. Build RoPE cache (high-water-mark: only rebuild if T exceeds cache)
    if (T > rope_cache_T_) {
        int half_hd = QWEN3_HEAD_DIM / 2;
        std::vector<float> cos_host(T * half_hd), sin_host(T * half_hd);
        build_rope_cache(cos_host.data(), sin_host.data(), T, QWEN3_HEAD_DIM, QWEN3_ROPE_THETA);

        rope_cos_cache_ = Tensor({(int64_t)T, (int64_t)half_hd}, DType::F32);
        rope_sin_cache_ = Tensor({(int64_t)T, (int64_t)half_hd}, DType::F32);
        rope_cos_cache_.copy_from_host(cos_host.data(), cos_host.size() * sizeof(float));
        rope_sin_cache_.copy_from_host(sin_host.data(), sin_host.size() * sizeof(float));
        rope_cache_T_ = T;
        fprintf(stderr, "[qwen3] computed RoPE cache for T=%d\n", T);
    }

    // 3. Run all 28 layers
    for (int i = 0; i < QWEN3_LAYERS; i++) {
        forward_layer(i, x.bf16_ptr(), T,
                      rope_cos_cache_.f32_ptr(), rope_sin_cache_.f32_ptr(), 0);

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
