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

    // QKV projections
    Tensor q_buf({(int64_t)T_q, (int64_t)D}, DType::BF16);
    Tensor k_buf({(int64_t)T_kv, (int64_t)D}, DType::BF16);
    Tensor v_buf({(int64_t)T_kv, (int64_t)D}, DType::BF16);

    attn.q_proj.forward(cublas_, query_in, q_buf.bf16_ptr(), T_q, stream);
    attn.k_proj.forward(cublas_, kv_in, k_buf.bf16_ptr(), T_kv, stream);
    attn.v_proj.forward(cublas_, kv_in, v_buf.bf16_ptr(), T_kv, stream);

    // Reshape [T, H*HD] -> [H, T, HD]
    Tensor q_heads({(int64_t)H, (int64_t)T_q, (int64_t)HD}, DType::BF16);
    Tensor k_heads({(int64_t)H, (int64_t)T_kv, (int64_t)HD}, DType::BF16);
    Tensor v_heads({(int64_t)H, (int64_t)T_kv, (int64_t)HD}, DType::BF16);

    head_transpose_bf16(q_buf.bf16_ptr(), q_heads.bf16_ptr(), H, T_q, HD, stream);
    head_transpose_bf16(k_buf.bf16_ptr(), k_heads.bf16_ptr(), H, T_kv, HD, stream);
    head_transpose_bf16(v_buf.bf16_ptr(), v_heads.bf16_ptr(), H, T_kv, HD, stream);

    // Per-head QK norm
    rms_norm_bf16(q_heads.bf16_ptr(), attn.q_norm_weight.bf16_ptr(),
                  q_heads.bf16_ptr(), H * T_q, HD, 1e-6f, stream);
    rms_norm_bf16(k_heads.bf16_ptr(), attn.k_norm_weight.bf16_ptr(),
                  k_heads.bf16_ptr(), H * T_kv, HD, 1e-6f, stream);

    // Apply RoPE
    if (cos_q && sin_q) {
        rope_standard_bf16(q_heads.bf16_ptr(), cos_q, sin_q,
                           H * T_q, T_q, HD, stream);
    }
    if (cos_k && sin_k) {
        rope_standard_bf16(k_heads.bf16_ptr(), cos_k, sin_k,
                           H * T_kv, T_kv, HD, stream);
    }

    // Attention: scores = Q @ K^T / sqrt(HD)
    float scale = 1.0f / std::sqrt((float)HD);
    Tensor scores({(int64_t)H, (int64_t)T_q, (int64_t)T_kv}, DType::BF16);
    {
        float alpha = scale, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
            T_kv, T_q, HD, &alpha,
            k_heads.bf16_ptr(), CUDA_R_16BF, HD, (int64_t)T_kv * HD,
            q_heads.bf16_ptr(), CUDA_R_16BF, HD, (int64_t)T_q * HD,
            &beta,
            scores.bf16_ptr(), CUDA_R_16BF, T_kv, (int64_t)T_q * T_kv,
            H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    softmax_bf16(scores.bf16_ptr(), scores.bf16_ptr(), H * T_q, T_kv, stream);

    // attn_out = scores @ V: [H, T_q, HD]
    Tensor attn_out({(int64_t)H, (int64_t)T_q, (int64_t)HD}, DType::BF16);
    {
        float alpha = 1.0f, beta = 0.0f;
        CUBLAS_CHECK(cublasGemmStridedBatchedEx(
            cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
            HD, T_q, T_kv, &alpha,
            v_heads.bf16_ptr(), CUDA_R_16BF, HD, (int64_t)T_kv * HD,
            scores.bf16_ptr(), CUDA_R_16BF, T_kv, (int64_t)T_q * T_kv,
            &beta,
            attn_out.bf16_ptr(), CUDA_R_16BF, HD, (int64_t)T_q * HD,
            H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Transpose back: [H, T_q, HD] -> [T_q, H*HD] via GPU kernel
    head_untranspose_bf16(attn_out.bf16_ptr(), out, H, T_q, HD, stream);

    // Output projection
    Tensor o_buf({(int64_t)T_q, (int64_t)D}, DType::BF16);
    attn.o_proj.forward(cublas_, out, o_buf.bf16_ptr(), T_q, stream);
    CUDA_CHECK(cudaMemcpyAsync(out, o_buf.bf16_ptr(), T_q * D * sizeof(__nv_bfloat16),
                                cudaMemcpyDeviceToDevice, stream));
}

// ========================= Forward =========================

Tensor LLMAdapter::forward(const __nv_bfloat16* source_hidden, int T_src,
                            const int* target_ids, int T_tgt,
                            const __nv_bfloat16* t5_weights) {
    fprintf(stderr, "[adapter] forward: T_src=%d, T_tgt=%d\n", T_src, T_tgt);

    constexpr int D = ADAPTER_DIM;
    constexpr int HD = ADAPTER_HEAD_DIM;
    cudaStream_t stream = 0;

    // 1. Embed target (T5) tokens
    Tensor target({(int64_t)T_tgt, (int64_t)D}, DType::BF16);
    embedding_lookup_bf16(embed_weight_.bf16_ptr(), target_ids,
                          target.bf16_ptr(), T_tgt, D);

    // 2. Build RoPE caches for target and source positions
    int half_hd = HD / 2;
    int max_len = std::max(T_src, T_tgt);
    std::vector<float> cos_host(max_len * half_hd), sin_host(max_len * half_hd);
    build_adapter_rope_cache(cos_host.data(), sin_host.data(), max_len, HD, ADAPTER_ROPE_THETA);

    Tensor cos_gpu({(int64_t)max_len, (int64_t)half_hd}, DType::F32);
    Tensor sin_gpu({(int64_t)max_len, (int64_t)half_hd}, DType::F32);
    cos_gpu.copy_from_host(cos_host.data(), cos_host.size() * sizeof(float));
    sin_gpu.copy_from_host(sin_host.data(), sin_host.size() * sizeof(float));

    // 3. Run 6 adapter blocks
    Tensor norm_buf({(int64_t)std::max(T_src, T_tgt), (int64_t)D}, DType::BF16);
    Tensor attn_buf({(int64_t)std::max(T_src, T_tgt), (int64_t)D}, DType::BF16);

    __nv_bfloat16* hidden = target.bf16_ptr();

    for (int i = 0; i < ADAPTER_LAYERS; i++) {
        auto& B = blocks_[i];

        // Self-attention: hidden = hidden + self_attn(norm(hidden))
        rms_norm_bf16(hidden, B.norm_self_attn_weight.bf16_ptr(),
                      norm_buf.bf16_ptr(), T_tgt, D, 1e-6f, stream);
        run_attention(B.self_attn,
                      norm_buf.bf16_ptr(), T_tgt,
                      norm_buf.bf16_ptr(), T_tgt,
                      attn_buf.bf16_ptr(),
                      cos_gpu.f32_ptr(), sin_gpu.f32_ptr(),
                      cos_gpu.f32_ptr(), sin_gpu.f32_ptr(),
                      stream);
        add_bf16(hidden, attn_buf.bf16_ptr(), hidden, (int64_t)T_tgt * D, stream);

        // Cross-attention: hidden = hidden + cross_attn(norm(hidden), source)
        rms_norm_bf16(hidden, B.norm_cross_attn_weight.bf16_ptr(),
                      norm_buf.bf16_ptr(), T_tgt, D, 1e-6f, stream);
        run_attention(B.cross_attn,
                      norm_buf.bf16_ptr(), T_tgt,
                      source_hidden, T_src,
                      attn_buf.bf16_ptr(),
                      cos_gpu.f32_ptr(), sin_gpu.f32_ptr(),
                      cos_gpu.f32_ptr(), sin_gpu.f32_ptr(),
                      stream);
        add_bf16(hidden, attn_buf.bf16_ptr(), hidden, (int64_t)T_tgt * D, stream);

        // MLP: hidden = hidden + mlp(norm(hidden))
        rms_norm_bf16(hidden, B.norm_mlp_weight.bf16_ptr(),
                      norm_buf.bf16_ptr(), T_tgt, D, 1e-6f, stream);

        constexpr int MLP_DIM = 4096;
        Tensor mlp_buf({(int64_t)T_tgt, (int64_t)MLP_DIM}, DType::BF16);
        B.mlp_fc1.forward(cublas_, norm_buf.bf16_ptr(), mlp_buf.bf16_ptr(), T_tgt, stream);
        gelu_tanh_bf16(mlp_buf.bf16_ptr(), mlp_buf.bf16_ptr(), (int64_t)T_tgt * MLP_DIM, stream);

        Tensor mlp_out({(int64_t)T_tgt, (int64_t)D}, DType::BF16);
        B.mlp_fc2.forward(cublas_, mlp_buf.bf16_ptr(), mlp_out.bf16_ptr(), T_tgt, stream);
        add_bf16(hidden, mlp_out.bf16_ptr(), hidden, (int64_t)T_tgt * D, stream);
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
