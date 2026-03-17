#include "transformer.h"
#include "safetensors.h"
#include "kernels/kernels.h"
#include "cuda_utils.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>

// ========================= Weight Loading =========================

void CosmosTransformer::load(const SafeTensorsFile& file) {
    fprintf(stderr, "[transformer] loading Cosmos transformer weights\n");
    CUBLAS_CHECK(cublasCreate(&cublas_));

    // All keys in the Anima checkpoint have "net." prefix
    auto load_t = [&](const std::string& name) -> Tensor {
        return file.load("net." + name);
    };

    auto load_lin = [&](Linear& lin, const std::string& name) {
        Tensor w = load_t(name + ".weight");
        Tensor b = file.has("net." + name + ".bias")
                   ? load_t(name + ".bias") : Tensor();
        lin.load(std::move(w), std::move(b));
    };

    // Patch embedding: x_embedder.proj.1.weight [2048, 68]
    load_lin(patch_proj_, "x_embedder.proj.1");
    fprintf(stderr, "[transformer]   patch_embed: [%d, %d]\n",
            patch_proj_.out_features(), patch_proj_.in_features());

    // Timestep embedding
    time_norm_weight_ = load_t("t_embedding_norm.weight");
    load_lin(time_linear_1_, "t_embedder.1.linear_1");
    load_lin(time_linear_2_, "t_embedder.1.linear_2");
    fprintf(stderr, "[transformer]   timestep: linear_1=[%d,%d] linear_2=[%d,%d]\n",
            time_linear_1_.out_features(), time_linear_1_.in_features(),
            time_linear_2_.out_features(), time_linear_2_.in_features());

    // 28 transformer blocks
    blocks_.resize(COSMOS_LAYERS);
    for (int i = 0; i < COSMOS_LAYERS; i++) {
        auto& B = blocks_[i];
        std::string bp = "blocks." + std::to_string(i);

        // adaLN-Zero modules
        auto load_adaln = [&](CosmosAdaLNZero& aln, const std::string& name) {
            load_lin(aln.linear_1, name + ".1");
            load_lin(aln.linear_2, name + ".2");
        };
        load_adaln(B.norm1, bp + ".adaln_modulation_self_attn");
        load_adaln(B.norm2, bp + ".adaln_modulation_cross_attn");
        load_adaln(B.norm3, bp + ".adaln_modulation_mlp");

        // Self-attention
        load_lin(B.sa_q_proj, bp + ".self_attn.q_proj");
        load_lin(B.sa_k_proj, bp + ".self_attn.k_proj");
        load_lin(B.sa_v_proj, bp + ".self_attn.v_proj");
        load_lin(B.sa_o_proj, bp + ".self_attn.output_proj");
        B.sa_q_norm = load_t(bp + ".self_attn.q_norm.weight");
        B.sa_k_norm = load_t(bp + ".self_attn.k_norm.weight");

        // Cross-attention
        load_lin(B.ca_q_proj, bp + ".cross_attn.q_proj");
        load_lin(B.ca_k_proj, bp + ".cross_attn.k_proj");
        load_lin(B.ca_v_proj, bp + ".cross_attn.v_proj");
        load_lin(B.ca_o_proj, bp + ".cross_attn.output_proj");
        B.ca_q_norm = load_t(bp + ".cross_attn.q_norm.weight");
        B.ca_k_norm = load_t(bp + ".cross_attn.k_norm.weight");

        // Feed-forward
        load_lin(B.ff_proj1, bp + ".mlp.layer1");
        load_lin(B.ff_proj2, bp + ".mlp.layer2");

        if ((i + 1) % 10 == 0 || i == COSMOS_LAYERS - 1) {
            fprintf(stderr, "[transformer]   loaded block %d/%d\n", i + 1, COSMOS_LAYERS);
        }
    }

    // Output norm + projection
    load_lin(output_norm_.linear_1, "final_layer.adaln_modulation.1");
    load_lin(output_norm_.linear_2, "final_layer.adaln_modulation.2");
    load_lin(output_proj_, "final_layer.linear");
    fprintf(stderr, "[transformer]   output_proj: [%d, %d]\n",
            output_proj_.out_features(), output_proj_.in_features());

    fprintf(stderr, "[transformer] loaded: %d blocks\n", COSMOS_LAYERS);
}

// ========================= Timestep Embedding =========================

void CosmosTransformer::compute_timestep_embedding(
    float timestep, __nv_bfloat16* embedded_ts, __nv_bfloat16* temb,
    cudaStream_t stream) {

    constexpr int D = COSMOS_HIDDEN;  // 2048
    constexpr int TEMB_DIM = 6144;    // 3 * 2048

    // 1. Sinusoidal timestep projection (Timesteps class, flip_sin_to_cos=True)
    // half_dim = D // 2 = 1024
    // freqs = exp(-log(10000) * arange(0, half_dim) / half_dim) * timestep
    // emb = [cos(freqs), sin(freqs)]
    int half_dim = D / 2;
    std::vector<float> sinusoidal(D);
    for (int i = 0; i < half_dim; i++) {
        float freq = std::exp(-std::log(10000.0f) * (float)i / (float)half_dim);
        float angle = timestep * freq;
        sinusoidal[i] = std::cos(angle);           // flip_sin_to_cos: cos first
        sinusoidal[half_dim + i] = std::sin(angle);
    }

    // Upload sinusoidal to GPU as BF16
    Tensor sin_gpu({(int64_t)D}, DType::BF16);
    {
        std::vector<__nv_bfloat16> sin_bf16(D);
        for (int i = 0; i < D; i++) sin_bf16[i] = __float2bfloat16(sinusoidal[i]);
        sin_gpu.copy_from_host(sin_bf16.data(), D * sizeof(__nv_bfloat16));
    }

    // 2. RMSNorm(sinusoidal) -> embedded_timestep [D]
    rms_norm_bf16(sin_gpu.bf16_ptr(), time_norm_weight_.bf16_ptr(),
                  embedded_ts, 1, D, 1e-6f, stream);

    // 3. TimestepEmbedding MLP: Linear(D, D) -> SiLU -> Linear(D, TEMB_DIM)
    // Input is raw sinusoidal (not normed)
    Tensor mlp_buf({(int64_t)D}, DType::BF16);
    time_linear_1_.forward(cublas_, sin_gpu.bf16_ptr(), mlp_buf.bf16_ptr(), 1, stream);
    silu_bf16(mlp_buf.bf16_ptr(), mlp_buf.bf16_ptr(), D, stream);
    time_linear_2_.forward(cublas_, mlp_buf.bf16_ptr(), temb, 1, stream);

}

// ========================= 3D RoPE =========================

void CosmosTransformer::compute_3d_rope(int num_frames, int height, int width,
                                          Tensor& cos_out, Tensor& sin_out) {
    constexpr int HD = COSMOS_HEAD_DIM;  // 128

    // Dimension splits
    int dim_h = (HD / 6) * 2;   // 42
    int dim_w = (HD / 6) * 2;   // 42
    int dim_t = HD - dim_h - dim_w;  // 44

    // Patch sizes
    int pe_t = num_frames / COSMOS_PATCH_T;  // frames / 1
    int pe_h = height / COSMOS_PATCH_H;      // height / 2
    int pe_w = width / COSMOS_PATCH_W;        // width / 2
    int S = pe_t * pe_h * pe_w;  // total spatial tokens

    // NTK scaling factors
    float rope_scale_t = 1.0f, rope_scale_h = 4.0f, rope_scale_w = 4.0f;
    float h_ntk = std::pow(rope_scale_h, (float)dim_h / (float)(dim_h - 2));
    float w_ntk = std::pow(rope_scale_w, (float)dim_w / (float)(dim_w - 2));
    float t_ntk = std::pow(rope_scale_t, (float)dim_t / (float)(dim_t - 2));

    float h_theta = 10000.0f * h_ntk;
    float w_theta = 10000.0f * w_ntk;
    float t_theta = 10000.0f * t_ntk;

    // Compute frequencies for each dimension
    auto make_freqs = [](int dim, float theta) -> std::vector<float> {
        int half = dim / 2;
        std::vector<float> freqs(half);
        for (int i = 0; i < half; i++) {
            freqs[i] = 1.0f / std::pow(theta, (float)(2 * i) / (float)dim);
        }
        return freqs;
    };

    auto h_freqs = make_freqs(dim_h, h_theta);
    auto w_freqs = make_freqs(dim_w, w_theta);
    auto t_freqs = make_freqs(dim_t, t_theta);

    // Build [S, HD] cos and sin arrays
    // For each spatial position (t, h, w), concatenate [emb_t, emb_h, emb_w] * 2
    // where emb_x = outer(position, freqs_x) — shape [half_dim_x]
    std::vector<float> cos_host(S * HD), sin_host(S * HD);

    for (int ft = 0; ft < pe_t; ft++) {
        for (int fh = 0; fh < pe_h; fh++) {
            for (int fw = 0; fw < pe_w; fw++) {
                int idx = (ft * pe_h + fh) * pe_w + fw;
                float* cos_row = &cos_host[idx * HD];
                float* sin_row = &sin_host[idx * HD];

                int offset = 0;

                // emb_t (half of dim_t values)
                for (int d = 0; d < dim_t / 2; d++) {
                    float angle = (float)ft * t_freqs[d];
                    cos_row[offset + d] = std::cos(angle);
                    sin_row[offset + d] = std::sin(angle);
                }
                offset += dim_t / 2;

                // emb_h
                for (int d = 0; d < dim_h / 2; d++) {
                    float angle = (float)fh * h_freqs[d];
                    cos_row[offset + d] = std::cos(angle);
                    sin_row[offset + d] = std::sin(angle);
                }
                offset += dim_h / 2;

                // emb_w
                for (int d = 0; d < dim_w / 2; d++) {
                    float angle = (float)fw * w_freqs[d];
                    cos_row[offset + d] = std::cos(angle);
                    sin_row[offset + d] = std::sin(angle);
                }
                offset += dim_w / 2;

                // Repeat: [emb_t, emb_h, emb_w] again (the "* 2" in cat)
                int first_half = offset;
                for (int d = 0; d < first_half; d++) {
                    cos_row[offset + d] = cos_row[d];
                    sin_row[offset + d] = sin_row[d];
                }
            }
        }
    }

    // Upload to GPU as BF16
    cos_out = Tensor({(int64_t)S, (int64_t)HD}, DType::BF16);
    sin_out = Tensor({(int64_t)S, (int64_t)HD}, DType::BF16);

    std::vector<__nv_bfloat16> cos_bf16(S * HD), sin_bf16(S * HD);
    for (int i = 0; i < S * HD; i++) {
        cos_bf16[i] = __float2bfloat16(cos_host[i]);
        sin_bf16[i] = __float2bfloat16(sin_host[i]);
    }
    cos_out.copy_from_host(cos_bf16.data(), S * HD * sizeof(__nv_bfloat16));
    sin_out.copy_from_host(sin_bf16.data(), S * HD * sizeof(__nv_bfloat16));
}

// ========================= Transformer Block =========================

void CosmosTransformer::forward_block(
    int block_idx,
    __nv_bfloat16* hidden, int S,
    const __nv_bfloat16* encoder_cond, int S_text,
    const __nv_bfloat16* embedded_ts,
    const __nv_bfloat16* temb,
    const __nv_bfloat16* rope_cos,
    const __nv_bfloat16* rope_sin,
    int batch_size,
    cudaStream_t stream) {

    auto& B = blocks_[block_idx];
    constexpr int D = COSMOS_HIDDEN;       // 2048
    constexpr int H = COSMOS_HEADS;        // 16
    constexpr int HD = COSMOS_HEAD_DIM;    // 128
    constexpr int ADALN_DIM = COSMOS_ADALN_DIM; // 256
    constexpr int TEXT_DIM = COSMOS_TEXT_DIM;    // 1024
    constexpr int MLP_DIM = COSMOS_MLP_DIM;     // 8192

    CUBLAS_CHECK(cublasSetStream(cublas_, stream));

    // Helper: compute adaLN-Zero modulation
    // Returns (shift, scale, gate) each [D] in a single [3*D] buffer
    auto compute_adaln_zero = [&](const CosmosAdaLNZero& aln, __nv_bfloat16* out_6144) {
        // SiLU(embedded_ts) -> linear_1 -> linear_2 + temb -> out [6144]
        Tensor silu_buf({(int64_t)D}, DType::BF16);
        CUDA_CHECK(cudaMemcpyAsync(silu_buf.bf16_ptr(), embedded_ts,
                                    D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
        silu_bf16(silu_buf.bf16_ptr(), silu_buf.bf16_ptr(), D, stream);

        Tensor adaln_mid({(int64_t)ADALN_DIM}, DType::BF16);
        aln.linear_1.forward(cublas_, silu_buf.bf16_ptr(), adaln_mid.bf16_ptr(), 1, stream);
        aln.linear_2.forward(cublas_, adaln_mid.bf16_ptr(), out_6144, 1, stream);

        // Add temb (6144-dim)
        add_bf16(out_6144, temb, out_6144, 3 * D, stream);
    };

    // Helper: apply adaLN modulation
    // normed = LayerNorm(hidden) * (1 + scale) + shift
    auto apply_adaln = [&](const __nv_bfloat16* x, const __nv_bfloat16* mod_6144,
                            __nv_bfloat16* normed, __nv_bfloat16* gate_out) {
        // mod_6144 = [shift(D), scale(D), gate(D)]
        const __nv_bfloat16* shift = mod_6144;
        const __nv_bfloat16* scale = mod_6144 + D;
        const __nv_bfloat16* gate = mod_6144 + 2 * D;

        // LayerNorm(x)
        layer_norm_bf16(x, normed, S, D, 1e-6f, stream);

        // normed = normed * (1 + scale) + shift  (broadcast scale/shift over S tokens)
        scale_shift_bcast_bf16(normed, scale, shift, normed, S, D, stream);

        // Copy gate
        CUDA_CHECK(cudaMemcpyAsync(gate_out, gate, D * sizeof(__nv_bfloat16),
                                    cudaMemcpyDeviceToDevice, stream));
    };

    // Helper: run attention (self or cross)
    auto run_attention = [&](const Linear& q_proj, const Linear& k_proj,
                              const Linear& v_proj, const Linear& o_proj,
                              const Tensor& q_norm_w, const Tensor& k_norm_w,
                              const __nv_bfloat16* q_input, int T_q,
                              const __nv_bfloat16* kv_input, int T_kv,
                              __nv_bfloat16* output,
                              bool apply_rope) {
        int QD = H * HD;  // 2048

        Tensor q_buf({(int64_t)T_q, (int64_t)QD}, DType::BF16);
        Tensor k_buf({(int64_t)T_kv, (int64_t)QD}, DType::BF16);
        Tensor v_buf({(int64_t)T_kv, (int64_t)QD}, DType::BF16);

        q_proj.forward(cublas_, q_input, q_buf.bf16_ptr(), T_q, stream);
        k_proj.forward(cublas_, kv_input, k_buf.bf16_ptr(), T_kv, stream);
        v_proj.forward(cublas_, kv_input, v_buf.bf16_ptr(), T_kv, stream);

        // Reshape [T, H*HD] -> [H, T, HD] via GPU kernel
        Tensor q_h({(int64_t)H, (int64_t)T_q, (int64_t)HD}, DType::BF16);
        Tensor k_h({(int64_t)H, (int64_t)T_kv, (int64_t)HD}, DType::BF16);
        Tensor v_h({(int64_t)H, (int64_t)T_kv, (int64_t)HD}, DType::BF16);

        head_transpose_bf16(q_buf.bf16_ptr(), q_h.bf16_ptr(), H, T_q, HD, stream);
        head_transpose_bf16(k_buf.bf16_ptr(), k_h.bf16_ptr(), H, T_kv, HD, stream);
        head_transpose_bf16(v_buf.bf16_ptr(), v_h.bf16_ptr(), H, T_kv, HD, stream);

        // Per-head QK norm
        rms_norm_bf16(q_h.bf16_ptr(), q_norm_w.bf16_ptr(),
                      q_h.bf16_ptr(), H * T_q, HD, 1e-6f, stream);
        rms_norm_bf16(k_h.bf16_ptr(), k_norm_w.bf16_ptr(),
                      k_h.bf16_ptr(), H * T_kv, HD, 1e-6f, stream);

        // Apply 3D RoPE (self-attention only)
        if (apply_rope && rope_cos && rope_sin) {
            // Cosmos 3D RoPE with use_real=True, use_real_unbind_dim=-2
            // cos/sin are [S, HD] BF16, broadcast across H heads.
            // Halved pattern: pairs (x[i], x[i+HD/2]) rotated as complex numbers.
            rope_cosmos_bf16(q_h.bf16_ptr(), rope_cos, rope_sin,
                             q_h.bf16_ptr(), H, T_q, HD, stream);
            rope_cosmos_bf16(k_h.bf16_ptr(), rope_cos, rope_sin,
                             k_h.bf16_ptr(), H, T_kv, HD, stream);
        }

        // Attention: scores = Q @ K^T / sqrt(HD) in F32 for precision
        float scale = 1.0f / std::sqrt((float)HD);
        Tensor scores({(int64_t)H, (int64_t)T_q, (int64_t)T_kv}, DType::F32);

        {
            float alpha = scale, beta = 0.0f;
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                cublas_, CUBLAS_OP_T, CUBLAS_OP_N,
                T_kv, T_q, HD, &alpha,
                k_h.bf16_ptr(), CUDA_R_16BF, HD, (int64_t)T_kv * HD,
                q_h.bf16_ptr(), CUDA_R_16BF, HD, (int64_t)T_q * HD,
                &beta,
                scores.f32_ptr(), CUDA_R_32F, T_kv, (int64_t)T_q * T_kv,
                H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }

        // Softmax in F32, then convert to BF16 for V multiplication
        softmax_f32(scores.f32_ptr(), scores.f32_ptr(), H * T_q, T_kv, stream);
        Tensor scores_bf16({(int64_t)H, (int64_t)T_q, (int64_t)T_kv}, DType::BF16);
        f32_to_bf16(scores.f32_ptr(), scores_bf16.bf16_ptr(), (int64_t)H * T_q * T_kv, stream);

        // attn_out = scores @ V
        Tensor attn_out({(int64_t)H, (int64_t)T_q, (int64_t)HD}, DType::BF16);
        {
            float alpha = 1.0f, beta = 0.0f;
            CUBLAS_CHECK(cublasGemmStridedBatchedEx(
                cublas_, CUBLAS_OP_N, CUBLAS_OP_N,
                HD, T_q, T_kv, &alpha,
                v_h.bf16_ptr(), CUDA_R_16BF, HD, (int64_t)T_kv * HD,
                scores_bf16.bf16_ptr(), CUDA_R_16BF, T_kv, (int64_t)T_q * T_kv,
                &beta,
                attn_out.bf16_ptr(), CUDA_R_16BF, HD, (int64_t)T_q * HD,
                H, CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
        }

        // Transpose back [H, T_q, HD] -> [T_q, H*HD] via GPU kernel
        Tensor attn_flat({(int64_t)T_q, (int64_t)QD}, DType::BF16);
        head_untranspose_bf16(attn_out.bf16_ptr(), attn_flat.bf16_ptr(), H, T_q, HD, stream);

        o_proj.forward(cublas_, attn_flat.bf16_ptr(), output, T_q, stream);
    };

    // --- Block execution ---

    // 1. Self-attention with adaLN-Zero
    Tensor mod1({3 * (int64_t)D}, DType::BF16);
    compute_adaln_zero(B.norm1, mod1.bf16_ptr());

    Tensor normed1({(int64_t)S, (int64_t)D}, DType::BF16);
    Tensor gate1({(int64_t)D}, DType::BF16);
    apply_adaln(hidden, mod1.bf16_ptr(), normed1.bf16_ptr(), gate1.bf16_ptr());

    Tensor sa_out({(int64_t)S, (int64_t)D}, DType::BF16);
    run_attention(B.sa_q_proj, B.sa_k_proj, B.sa_v_proj, B.sa_o_proj,
                  B.sa_q_norm, B.sa_k_norm,
                  normed1.bf16_ptr(), S, normed1.bf16_ptr(), S,
                  sa_out.bf16_ptr(), true);

    // Residual: hidden = hidden + gate * sa_out
    residual_gate_bcast_bf16(hidden, sa_out.bf16_ptr(), gate1.bf16_ptr(),
                              hidden, S, D, stream);

    // 2. Cross-attention with adaLN-Zero
    Tensor mod2({3 * (int64_t)D}, DType::BF16);
    compute_adaln_zero(B.norm2, mod2.bf16_ptr());

    Tensor normed2({(int64_t)S, (int64_t)D}, DType::BF16);
    Tensor gate2({(int64_t)D}, DType::BF16);
    apply_adaln(hidden, mod2.bf16_ptr(), normed2.bf16_ptr(), gate2.bf16_ptr());

    Tensor ca_out({(int64_t)S, (int64_t)D}, DType::BF16);
    run_attention(B.ca_q_proj, B.ca_k_proj, B.ca_v_proj, B.ca_o_proj,
                  B.ca_q_norm, B.ca_k_norm,
                  normed2.bf16_ptr(), S, encoder_cond, S_text,
                  ca_out.bf16_ptr(), false);

    residual_gate_bcast_bf16(hidden, ca_out.bf16_ptr(), gate2.bf16_ptr(),
                              hidden, S, D, stream);

    // 3. Feed-forward with adaLN-Zero
    Tensor mod3({3 * (int64_t)D}, DType::BF16);
    compute_adaln_zero(B.norm3, mod3.bf16_ptr());

    Tensor normed3({(int64_t)S, (int64_t)D}, DType::BF16);
    Tensor gate3({(int64_t)D}, DType::BF16);
    apply_adaln(hidden, mod3.bf16_ptr(), normed3.bf16_ptr(), gate3.bf16_ptr());

    Tensor ff_buf({(int64_t)S, (int64_t)MLP_DIM}, DType::BF16);
    B.ff_proj1.forward(cublas_, normed3.bf16_ptr(), ff_buf.bf16_ptr(), S, stream);
    gelu_tanh_bf16(ff_buf.bf16_ptr(), ff_buf.bf16_ptr(), (int64_t)S * MLP_DIM, stream);

    Tensor ff_out({(int64_t)S, (int64_t)D}, DType::BF16);
    B.ff_proj2.forward(cublas_, ff_buf.bf16_ptr(), ff_out.bf16_ptr(), S, stream);

    residual_gate_bcast_bf16(hidden, ff_out.bf16_ptr(), gate3.bf16_ptr(),
                              hidden, S, D, stream);
}

// ========================= Forward =========================

Tensor CosmosTransformer::forward(const Tensor& latents, float timestep,
                                   const __nv_bfloat16* encoder_cond, int S_text,
                                   int batch_size, int latent_h, int latent_w) {
    constexpr int D = COSMOS_HIDDEN;
    constexpr int C_IN = COSMOS_IN_CHANNELS;  // 16
    cudaStream_t stream = 0;
    CUBLAS_CHECK(cublasSetStream(cublas_, stream));

    // Spatial dimensions after patching
    int pe_h = latent_h / COSMOS_PATCH_H;  // latent_h / 2
    int pe_w = latent_w / COSMOS_PATCH_W;  // latent_w / 2
    int pe_t = 1;  // single frame / COSMOS_PATCH_T=1
    int S = pe_t * pe_h * pe_w;  // total spatial tokens

    fprintf(stderr, "[transformer] forward: latent=[%d,%d], patches=[%d,%d], S=%d, timestep=%.4f\n",
            latent_h, latent_w, pe_h, pe_w, S, timestep);

    // 1. Concatenate padding mask (all zeros) as 17th channel
    // Input: [B, 16, 1, H, W] -> [B, 17, 1, H, W]
    // For now, create combined tensor with padding mask = 0
    int C_total = C_IN + 1;  // 17
    Tensor padded({(int64_t)batch_size, (int64_t)C_total, 1, (int64_t)latent_h, (int64_t)latent_w}, DType::BF16);
    CUDA_CHECK(cudaMemset(padded.data_ptr(), 0, padded.size_bytes()));
    // Copy latent channels
    CUDA_CHECK(cudaMemcpy(padded.bf16_ptr(), latents.data_ptr(),
                           batch_size * C_IN * latent_h * latent_w * sizeof(__nv_bfloat16),
                           cudaMemcpyDeviceToDevice));

    // 2. Patchify: [B, 17, 1, H, W] -> [B, S, 68] on GPU
    int patch_dim = C_total * COSMOS_PATCH_T * COSMOS_PATCH_H * COSMOS_PATCH_W;  // 68
    Tensor patches({(int64_t)(batch_size * S), (int64_t)patch_dim}, DType::BF16);
    patchify_3d_bf16(padded.bf16_ptr(), patches.bf16_ptr(),
                     batch_size, C_total, 1, latent_h, latent_w,
                     COSMOS_PATCH_T, COSMOS_PATCH_H, COSMOS_PATCH_W, stream);

    // 3. Linear projection: [B, S, 68] -> [B, S, 2048]
    Tensor hidden({(int64_t)(batch_size * S), (int64_t)D}, DType::BF16);
    patch_proj_.forward(cublas_, patches.bf16_ptr(), hidden.bf16_ptr(), batch_size * S, stream);

    // 4. Compute timestep embedding
    Tensor embedded_ts({(int64_t)D}, DType::BF16);
    Tensor temb({6144}, DType::BF16);
    compute_timestep_embedding(timestep, embedded_ts.bf16_ptr(), temb.bf16_ptr(), stream);

    // 5. Compute 3D RoPE
    Tensor rope_cos, rope_sin;
    compute_3d_rope(1, latent_h, latent_w, rope_cos, rope_sin);

    // 6. Run 28 transformer blocks
    for (int i = 0; i < COSMOS_LAYERS; i++) {
        forward_block(i, hidden.bf16_ptr(), S,
                      encoder_cond, S_text,
                      embedded_ts.bf16_ptr(), temb.bf16_ptr(),
                      rope_cos.bf16_ptr(), rope_sin.bf16_ptr(),
                      batch_size, stream);

        if ((i + 1) % 10 == 0) {
            fprintf(stderr, "[transformer]   block %d/%d done\n", i + 1, COSMOS_LAYERS);
        }
    }

    // 7. Output norm (CosmosAdaLayerNorm — shift + scale only, no gate)
    {
        Tensor silu_buf({(int64_t)D}, DType::BF16);
        CUDA_CHECK(cudaMemcpy(silu_buf.bf16_ptr(), embedded_ts.bf16_ptr(),
                               D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice));
        silu_bf16(silu_buf.bf16_ptr(), silu_buf.bf16_ptr(), D, stream);

        Tensor adaln_mid({(int64_t)COSMOS_ADALN_DIM}, DType::BF16);
        output_norm_.linear_1.forward(cublas_, silu_buf.bf16_ptr(), adaln_mid.bf16_ptr(), 1, stream);

        Tensor mod_4096({4096}, DType::BF16);
        output_norm_.linear_2.forward(cublas_, adaln_mid.bf16_ptr(), mod_4096.bf16_ptr(), 1, stream);

        // Add first 4096 elements of temb (temb is 6144-dim, output norm uses 4096)
        add_bf16(mod_4096.bf16_ptr(), temb.bf16_ptr(), mod_4096.bf16_ptr(), 4096, stream);

        // Split into shift[D], scale[D]
        const __nv_bfloat16* shift = mod_4096.bf16_ptr();
        const __nv_bfloat16* scale = mod_4096.bf16_ptr() + D;

        layer_norm_bf16(hidden.bf16_ptr(), hidden.bf16_ptr(), S, D, 1e-6f, stream);
        scale_shift_bcast_bf16(hidden.bf16_ptr(), scale, shift, hidden.bf16_ptr(), S, D, stream);
    }

    // 8. Output projection: [B*S, 2048] -> [B*S, patch_dim]
    int out_patch_dim = COSMOS_PATCH_T * COSMOS_PATCH_H * COSMOS_PATCH_W * COSMOS_OUT_CHANNELS;
    Tensor proj_out({(int64_t)(batch_size * S), (int64_t)out_patch_dim}, DType::BF16);
    output_proj_.forward(cublas_, hidden.bf16_ptr(), proj_out.bf16_ptr(), batch_size * S, stream);

    // 9. Unpatchify: [B, S, patch_dim] -> [B, C, T, H, W]
    Tensor output({(int64_t)batch_size, (int64_t)COSMOS_OUT_CHANNELS, 1, (int64_t)latent_h, (int64_t)latent_w}, DType::BF16);
    unpatchify_3d_bf16(proj_out.bf16_ptr(), output.bf16_ptr(),
                       batch_size, COSMOS_OUT_CHANNELS, 1, latent_h, latent_w,
                       COSMOS_PATCH_T, COSMOS_PATCH_H, COSMOS_PATCH_W, stream);

    fprintf(stderr, "[transformer] forward complete\n");
    return output;
}
