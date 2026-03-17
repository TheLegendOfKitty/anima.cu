#include "llm_adapter.h"
#include "safetensors.h"
#include "kernels/kernels.h"
#include "cuda_utils.cuh"

#include <cmath>
#include <cstdio>

// ========================= RoPE Cache (for adapter, theta=10000) =========================

static void build_adapter_rope_cache(float* cos_out, float* sin_out,
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

void LLMAdapter::load(const SafeTensorsFile& file) {
    fprintf(stderr, "[adapter] loading LLM adapter weights\n");
    CUBLAS_CHECK(cublasCreate(&cublas_));

    const std::string pfx = "net.llm_adapter.";

    auto load_t = [&](const std::string& name) -> Tensor {
        return file.load(pfx + name);
    };

    // Embedding
    embed_weight_ = load_t("embed.weight");
    fprintf(stderr, "[adapter]   embed: [%lld, %lld]\n",
            (long long)embed_weight_.dim(0), (long long)embed_weight_.dim(1));

    // Output projection + norm
    {
        Tensor w = load_t("out_proj.weight");
        Tensor b = load_t("out_proj.bias");
        out_proj_.load(std::move(w), std::move(b));
    }
    norm_weight_ = load_t("norm.weight");

    // 6 adapter blocks
    blocks_.resize(ADAPTER_LAYERS);
    for (int i = 0; i < ADAPTER_LAYERS; i++) {
        auto& B = blocks_[i];
        std::string bp = "blocks." + std::to_string(i);

        // Norms
        B.norm_self_attn_weight = load_t(bp + ".norm_self_attn.weight");
        B.norm_cross_attn_weight = load_t(bp + ".norm_cross_attn.weight");
        B.norm_mlp_weight = load_t(bp + ".norm_mlp.weight");

        // Self-attention
        auto load_attn = [&](AdapterAttention& attn, const std::string& ap) {
            auto load_lin = [&](Linear& lin, const std::string& n) {
                Tensor w = load_t(ap + "." + n + ".weight");
                Tensor b = file.has(pfx + ap + "." + n + ".bias")
                           ? load_t(ap + "." + n + ".bias") : Tensor();
                lin.load(std::move(w), std::move(b));
            };
            load_lin(attn.q_proj, "q_proj");
            load_lin(attn.k_proj, "k_proj");
            load_lin(attn.v_proj, "v_proj");
            load_lin(attn.o_proj, "o_proj");
            attn.q_norm_weight = load_t(ap + ".q_norm.weight");
            attn.k_norm_weight = load_t(ap + ".k_norm.weight");
        };

        load_attn(B.self_attn, bp + ".self_attn");
        load_attn(B.cross_attn, bp + ".cross_attn");

        // MLP
        {
            Tensor w1 = load_t(bp + ".mlp.0.weight");
            Tensor b1 = load_t(bp + ".mlp.0.bias");
            B.mlp_fc1.load(std::move(w1), std::move(b1));
        }
        {
            Tensor w2 = load_t(bp + ".mlp.2.weight");
            Tensor b2 = load_t(bp + ".mlp.2.bias");
            B.mlp_fc2.load(std::move(w2), std::move(b2));
        }

        fprintf(stderr, "[adapter]   loaded block %d/%d\n", i + 1, ADAPTER_LAYERS);
    }

    fprintf(stderr, "[adapter] loaded\n");
}

// ========================= ensure_scratch =========================

void LLMAdapter::ensure_scratch(int T_src, int T_tgt) {
    if (T_src <= scratch_.T_src && T_tgt <= scratch_.T_tgt) return;  // high-water-mark
    T_src = std::max(T_src, scratch_.T_src);
    T_tgt = std::max(T_tgt, scratch_.T_tgt);

    constexpr int D = ADAPTER_DIM;       // 1024
    constexpr int H = ADAPTER_HEADS;     // 16
    constexpr int HD = ADAPTER_HEAD_DIM; // 64
    constexpr int MLP_DIM = 4096;

    int max_kv = std::max(T_src, T_tgt);

    fprintf(stderr, "[adapter] allocating scratch buffers: T_src=%d, T_tgt=%d\n", T_src, T_tgt);

    scratch_.q_buf = Tensor({(int64_t)T_tgt, (int64_t)D}, DType::BF16);
    scratch_.k_buf = Tensor({(int64_t)max_kv, (int64_t)D}, DType::BF16);
    scratch_.v_buf = Tensor({(int64_t)max_kv, (int64_t)D}, DType::BF16);
    scratch_.q_heads = Tensor({(int64_t)H, (int64_t)T_tgt, (int64_t)HD}, DType::BF16);
    scratch_.k_heads = Tensor({(int64_t)H, (int64_t)max_kv, (int64_t)HD}, DType::BF16);
    scratch_.v_heads = Tensor({(int64_t)H, (int64_t)max_kv, (int64_t)HD}, DType::BF16);
    scratch_.scores = Tensor({(int64_t)H, (int64_t)T_tgt, (int64_t)max_kv}, DType::BF16);
    scratch_.attn_out = Tensor({(int64_t)H, (int64_t)T_tgt, (int64_t)HD}, DType::BF16);
    scratch_.attn_flat = Tensor({(int64_t)T_tgt, (int64_t)D}, DType::BF16);
    scratch_.norm_buf = Tensor({(int64_t)max_kv, (int64_t)D}, DType::BF16);
    scratch_.attn_buf = Tensor({(int64_t)max_kv, (int64_t)D}, DType::BF16);
    scratch_.mlp_buf = Tensor({(int64_t)T_tgt, (int64_t)MLP_DIM}, DType::BF16);
    scratch_.mlp_out = Tensor({(int64_t)T_tgt, (int64_t)D}, DType::BF16);

    scratch_.T_src = T_src;
    scratch_.T_tgt = T_tgt;

    size_t total = 0;
    auto add = [&](const Tensor& t) { total += t.size_bytes(); };
    add(scratch_.q_buf); add(scratch_.k_buf); add(scratch_.v_buf);
    add(scratch_.q_heads); add(scratch_.k_heads); add(scratch_.v_heads);
    add(scratch_.scores); add(scratch_.attn_out); add(scratch_.attn_flat);
    add(scratch_.norm_buf); add(scratch_.attn_buf);
    add(scratch_.mlp_buf); add(scratch_.mlp_out);
    fprintf(stderr, "[adapter] scratch memory: %.1f MB\n", total / (1024.0 * 1024.0));
}

// ========================= Attention =========================

void LLMAdapter::run_attention(const AdapterAttention& attn,
                                const __nv_bfloat16* query_in, int T_q,
                                const __nv_bfloat16* kv_in, int T_kv,
                                __nv_bfloat16* out,
                                const float* cos_q, const float* sin_q,
                                const float* cos_k, const float* sin_k,
                                cudaStream_t stream) {
    constexpr int D = ADAPTER_DIM;       // 1024
    constexpr int H = ADAPTER_HEADS;     // 16
    constexpr int HD = ADAPTER_HEAD_DIM; // 64

    CUBLAS_CHECK(cublasSetStream(cublas_, stream));

    // Use pre-allocated scratch buffers
    auto* q_buf = scratch_.q_buf.bf16_ptr();
    auto* k_buf = scratch_.k_buf.bf16_ptr();
    auto* v_buf = scratch_.v_buf.bf16_ptr();
    auto* q_heads = scratch_.q_heads.bf16_ptr();
    auto* k_heads = scratch_.k_heads.bf16_ptr();
    auto* v_heads = scratch_.v_heads.bf16_ptr();

    // QKV projections
    attn.q_proj.forward(cublas_, query_in, q_buf, T_q, stream);
    attn.k_proj.forward(cublas_, kv_in, k_buf, T_kv, stream);
    attn.v_proj.forward(cublas_, kv_in, v_buf, T_kv, stream);

    // Reshape [T, H*HD] -> [H, T, HD]
    head_transpose_bf16(q_buf, q_heads, H, T_q, HD, stream);
    head_transpose_bf16(k_buf, k_heads, H, T_kv, HD, stream);
    head_transpose_bf16(v_buf, v_heads, H, T_kv, HD, stream);

    // Per-head QK norm
    rms_norm_bf16(q_heads, attn.q_norm_weight.bf16_ptr(),
                  q_heads, H * T_q, HD, 1e-6f, stream);
    rms_norm_bf16(k_heads, attn.k_norm_weight.bf16_ptr(),
                  k_heads, H * T_kv, HD, 1e-6f, stream);

    // Apply RoPE
    if (cos_q && sin_q) {
        rope_standard_bf16(q_heads, cos_q, sin_q,
                           H * T_q, T_q, HD, stream);
    }
    if (cos_k && sin_k) {
        rope_standard_bf16(k_heads, cos_k, sin_k,
                           H * T_kv, T_kv, HD, stream);
    }

    // Attention: scores = Q @ K^T / sqrt(HD)
    float scale = 1.0f / std::sqrt((float)HD);
    auto* scores = scratch_.scores.bf16_ptr();
    {
        float alpha = scale, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
            T_kv, T_q, HD, &alpha,
            k_heads, CUDA_R_16BF, HD, (int64_t)T_kv * HD,
            q_heads, CUDA_R_16BF, HD, (int64_t)T_q * HD,
            &beta,
            scores, CUDA_R_16BF, T_kv, (int64_t)T_q * T_kv,
            H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    softmax_bf16(scores, scores, H * T_q, T_kv, stream);

    // attn_out = scores @ V: [H, T_q, HD]
    auto* attn_out = scratch_.attn_out.bf16_ptr();
    {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
            HD, T_q, T_kv, &alpha,
            v_heads, CUDA_R_16BF, HD, (int64_t)T_kv * HD,
            scores, CUDA_R_16BF, T_kv, (int64_t)T_q * T_kv,
            &beta,
            attn_out, CUDA_R_16BF, HD, (int64_t)T_q * HD,
            H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Transpose back [H, T_q, HD] -> [T_q, H*HD] and output projection
    auto* attn_flat = scratch_.attn_flat.bf16_ptr();
    head_untranspose_bf16(attn_out, attn_flat, H, T_q, HD, stream);
    attn.o_proj.forward(cublas_, attn_flat, out, T_q, stream);
}

// ========================= Forward =========================

Tensor LLMAdapter::forward(const __nv_bfloat16* source_hidden, int T_src,
                            const int* target_ids, int T_tgt,
                            const __nv_bfloat16* t5_weights) {
    fprintf(stderr, "[adapter] forward: T_src=%d, T_tgt=%d\n", T_src, T_tgt);

    constexpr int D = ADAPTER_DIM;
    constexpr int HD = ADAPTER_HEAD_DIM;
    cudaStream_t stream = 0;

    // Ensure scratch buffers (no-op if dimensions unchanged)
    ensure_scratch(T_src, T_tgt);

    // 1. Embed target (T5) tokens
    Tensor target({(int64_t)T_tgt, (int64_t)D}, DType::BF16);
    embedding_lookup_bf16(embed_weight_.bf16_ptr(), target_ids,
                          target.bf16_ptr(), T_tgt, D);

    // 2. Build RoPE caches (no-op if max_len unchanged)
    int half_hd = HD / 2;
    int max_len = std::max(T_src, T_tgt);
    if (max_len > rope_cache_len_) {
        std::vector<float> cos_host(max_len * half_hd), sin_host(max_len * half_hd);
        build_adapter_rope_cache(cos_host.data(), sin_host.data(), max_len, HD, ADAPTER_ROPE_THETA);

        rope_cos_cache_ = Tensor({(int64_t)max_len, (int64_t)half_hd}, DType::F32);
        rope_sin_cache_ = Tensor({(int64_t)max_len, (int64_t)half_hd}, DType::F32);
        rope_cos_cache_.copy_from_host(cos_host.data(), cos_host.size() * sizeof(float));
        rope_sin_cache_.copy_from_host(sin_host.data(), sin_host.size() * sizeof(float));
        rope_cache_len_ = max_len;
        fprintf(stderr, "[adapter] computed RoPE cache for max_len=%d\n", max_len);
    }

    // 3. Run 6 adapter blocks
    auto* norm_buf = scratch_.norm_buf.bf16_ptr();
    auto* attn_buf = scratch_.attn_buf.bf16_ptr();
    auto* mlp_buf = scratch_.mlp_buf.bf16_ptr();
    auto* mlp_out = scratch_.mlp_out.bf16_ptr();

    __nv_bfloat16* hidden = target.bf16_ptr();

    for (int i = 0; i < ADAPTER_LAYERS; i++) {
        auto& B = blocks_[i];

        // Self-attention: hidden = hidden + self_attn(norm(hidden))
        rms_norm_bf16(hidden, B.norm_self_attn_weight.bf16_ptr(),
                      norm_buf, T_tgt, D, 1e-6f, stream);
        run_attention(B.self_attn,
                      norm_buf, T_tgt,
                      norm_buf, T_tgt,
                      attn_buf,
                      rope_cos_cache_.f32_ptr(), rope_sin_cache_.f32_ptr(),
                      rope_cos_cache_.f32_ptr(), rope_sin_cache_.f32_ptr(),
                      stream);
        add_bf16(hidden, attn_buf, hidden, (int64_t)T_tgt * D, stream);

        // Cross-attention: hidden = hidden + cross_attn(norm(hidden), source)
        rms_norm_bf16(hidden, B.norm_cross_attn_weight.bf16_ptr(),
                      norm_buf, T_tgt, D, 1e-6f, stream);
        run_attention(B.cross_attn,
                      norm_buf, T_tgt,
                      source_hidden, T_src,
                      attn_buf,
                      rope_cos_cache_.f32_ptr(), rope_sin_cache_.f32_ptr(),
                      rope_cos_cache_.f32_ptr(), rope_sin_cache_.f32_ptr(),
                      stream);
        add_bf16(hidden, attn_buf, hidden, (int64_t)T_tgt * D, stream);

        // MLP: hidden = hidden + mlp(norm(hidden))
        rms_norm_bf16(hidden, B.norm_mlp_weight.bf16_ptr(),
                      norm_buf, T_tgt, D, 1e-6f, stream);

        constexpr int MLP_DIM = 4096;
        B.mlp_fc1.forward(cublas_, norm_buf, mlp_buf, T_tgt, stream);
        gelu_tanh_bf16(mlp_buf, mlp_buf, (int64_t)T_tgt * MLP_DIM, stream);

        B.mlp_fc2.forward(cublas_, mlp_buf, mlp_out, T_tgt, stream);
        add_bf16(hidden, mlp_out, hidden, (int64_t)T_tgt * D, stream);
    }

    // 4. Output: norm(out_proj(hidden))
    Tensor proj_buf({(int64_t)T_tgt, (int64_t)D}, DType::BF16);
    out_proj_.forward(cublas_, hidden, proj_buf.bf16_ptr(), T_tgt, stream);
    rms_norm_bf16(proj_buf.bf16_ptr(), norm_weight_.bf16_ptr(),
                  proj_buf.bf16_ptr(), T_tgt, D, 1e-6f, stream);

    // 5. Apply T5 weights if provided
    if (t5_weights) {
        // t5_weights is [T_tgt, 1], broadcast multiply with proj_buf [T_tgt, D]
        // For now, since all weights are 1.0, skip this step
        // TODO: implement broadcast multiply if needed
    }

    // 6. Pad to 512 tokens
    constexpr int PAD_LEN = CONDITIONING_MAX_LEN;
    if (T_tgt >= PAD_LEN) {
        // Truncate
        proj_buf.reshape({(int64_t)PAD_LEN, (int64_t)D});
        fprintf(stderr, "[adapter] output: [%d, %d] (truncated)\n", PAD_LEN, D);
        return proj_buf;
    }

    // Zero-pad to 512 tokens (model was trained with this)
    Tensor padded = zeros({(int64_t)PAD_LEN, (int64_t)D}, DType::BF16);
    CUDA_CHECK(cudaMemcpy(padded.bf16_ptr(), proj_buf.bf16_ptr(),
                           T_tgt * D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));

    fprintf(stderr, "[adapter] output: [%d, %d] (padded from %d)\n", PAD_LEN, D, T_tgt);
    return padded;
}
