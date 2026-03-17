#include "transformer.h"
#include "safetensors.h"
#include "kernels/kernels.h"
#include "cuda_utils.cuh"

#include <cmath>
#include <cstdio>
#include <cstring>
#include <vector>
#include <algorithm>

// ========================= Weight Loading =========================

void CosmosTransformer::load(const SafeTensorsFile& file) {
    fprintf(stderr, "[transformer] loading Cosmos transformer weights\n");
    CUBLAS_CHECK(cublasCreate(&cublas_));

    // Initialize cuDNN for flash attention
    cudnnCreate(&cudnn_);
    sdpa_.init(cudnn_);

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

// ========================= ensure_scratch =========================

void CosmosTransformer::ensure_scratch(int S, int S_text) {
    if (S <= scratch_.S && S_text <= scratch_.S_text) return;

    constexpr int D = COSMOS_HIDDEN;
    constexpr int H = COSMOS_HEADS;
    constexpr int HD = COSMOS_HEAD_DIM;
    constexpr int MLP_DIM = COSMOS_MLP_DIM;
    constexpr int ADALN_DIM = COSMOS_ADALN_DIM;
    constexpr int C_IN = COSMOS_IN_CHANNELS;
    constexpr int C_total = C_IN + 1;  // 17

    int max_kv = std::max(S, S_text);

    int patch_dim = C_total * COSMOS_PATCH_T * COSMOS_PATCH_H * COSMOS_PATCH_W;  // 68
    int out_patch_dim = COSMOS_PATCH_T * COSMOS_PATCH_H * COSMOS_PATCH_W * COSMOS_OUT_CHANNELS;  // 64

    fprintf(stderr, "[transformer] allocating scratch buffers: S=%d, S_text=%d, max_kv=%d\n",
            S, S_text, max_kv);

    // adaLN reusable buffers
    scratch_.mod = Tensor({3 * (int64_t)D}, DType::BF16);
    scratch_.normed = Tensor({(int64_t)S, (int64_t)D}, DType::BF16);
    scratch_.gate = Tensor({(int64_t)D}, DType::BF16);
    scratch_.sub_out = Tensor({(int64_t)S, (int64_t)D}, DType::BF16);
    scratch_.ff_buf = Tensor({(int64_t)S, (int64_t)MLP_DIM}, DType::BF16);

    // Attention buffers — strided SDPA reads [B*S, H*HD] directly, no transpose needed
    scratch_.q_buf = Tensor({(int64_t)S, (int64_t)D}, DType::BF16);
    scratch_.k_buf = Tensor({(int64_t)max_kv, (int64_t)D}, DType::BF16);
    scratch_.v_buf = Tensor({(int64_t)max_kv, (int64_t)D}, DType::BF16);
    scratch_.attn_flat = Tensor({(int64_t)S, (int64_t)D}, DType::BF16);

    // Cached per-forward-call
    scratch_.silu_ts = Tensor({(int64_t)D}, DType::BF16);
    scratch_.adaln_mid = Tensor({(int64_t)ADALN_DIM}, DType::BF16);

    // Timestep embedding buffers (reused across denoising steps)
    scratch_.embedded_ts = Tensor({(int64_t)D}, DType::BF16);
    scratch_.temb = Tensor({6144}, DType::BF16);
    scratch_.ts_sin_buf = Tensor({(int64_t)D}, DType::BF16);
    scratch_.ts_mlp_buf = Tensor({(int64_t)D}, DType::BF16);

    // Forward-pass temporaries
    scratch_.padded = Tensor({(int64_t)S * (int64_t)patch_dim}, DType::BF16);
    scratch_.patches = Tensor({(int64_t)S, (int64_t)patch_dim}, DType::BF16);
    scratch_.hidden = Tensor({(int64_t)S, (int64_t)D}, DType::BF16);
    scratch_.proj_out = Tensor(scratch_.padded.data_ptr(),
                               {(int64_t)S, (int64_t)out_patch_dim},
                               DType::BF16, /*owned=*/false);

    scratch_.S = S;
    scratch_.S_text = S_text;

    // Log scratch memory usage
    size_t total = 0;
    auto add = [&](const Tensor& t) { if (t.data_ptr()) total += t.size_bytes(); };
    add(scratch_.mod); add(scratch_.normed); add(scratch_.gate);
    add(scratch_.sub_out); add(scratch_.ff_buf);
    add(scratch_.q_buf); add(scratch_.k_buf); add(scratch_.v_buf);
    add(scratch_.attn_flat);
    add(scratch_.silu_ts); add(scratch_.adaln_mid);
    add(scratch_.embedded_ts); add(scratch_.temb);
    add(scratch_.ts_sin_buf); add(scratch_.ts_mlp_buf);
    add(scratch_.padded); add(scratch_.patches); add(scratch_.hidden);
    // proj_out is non-owning (shares padded memory), don't double-count
    fprintf(stderr, "[transformer] scratch memory: %.1f MB\n", total / (1024.0 * 1024.0));
}

// ========================= Timestep Embedding =========================

void CosmosTransformer::compute_timestep_embedding(
    float timestep, __nv_bfloat16* embedded_ts, __nv_bfloat16* temb,
    cudaStream_t stream) {

    constexpr int D = COSMOS_HIDDEN;  // 2048

    // 1. Sinusoidal timestep projection (Timesteps class, flip_sin_to_cos=True)
    int half_dim = D / 2;
    std::vector<float> sinusoidal(D);
    for (int i = 0; i < half_dim; i++) {
        float freq = std::exp(-std::log(10000.0f) * (float)i / (float)half_dim);
        float angle = timestep * freq;
        sinusoidal[i] = std::cos(angle);           // flip_sin_to_cos: cos first
        sinusoidal[half_dim + i] = std::sin(angle);
    }

    // Upload sinusoidal to GPU as BF16 (reuse scratch buffer)
    auto* sin_buf = scratch_.ts_sin_buf.bf16_ptr();
    {
        std::vector<__nv_bfloat16> sin_bf16(D);
        for (int i = 0; i < D; i++) sin_bf16[i] = __float2bfloat16(sinusoidal[i]);
        scratch_.ts_sin_buf.copy_from_host(sin_bf16.data(), D * sizeof(__nv_bfloat16));
    }

    // 2. RMSNorm(sinusoidal) -> embedded_timestep [D]
    rms_norm_bf16(sin_buf, time_norm_weight_.bf16_ptr(),
                  embedded_ts, 1, D, 1e-6f, stream);

    // 3. TimestepEmbedding MLP: Linear(D, D) -> SiLU -> Linear(D, TEMB_DIM)
    // Input is raw sinusoidal (not normed)
    auto* mlp_buf = scratch_.ts_mlp_buf.bf16_ptr();
    time_linear_1_.forward(cublas_, sin_buf, mlp_buf, 1, stream);
    silu_bf16(mlp_buf, mlp_buf, D, stream);
    time_linear_2_.forward(cublas_, mlp_buf, temb, 1, stream);
}

// ========================= 3D RoPE (cached) =========================

void CosmosTransformer::compute_3d_rope(int num_frames, int height, int width) {
    constexpr int HD = COSMOS_HEAD_DIM;  // 128

    int pe_h = height / COSMOS_PATCH_H;
    int pe_w = width / COSMOS_PATCH_W;

    // Return cached if resolution hasn't changed
    if (pe_h == rope_cache_h_ && pe_w == rope_cache_w_ && !rope_cos_cache_.empty()) return;

    fprintf(stderr, "[transformer] computing 3D RoPE: pe=[%d, %d, %d]\n",
            num_frames, pe_h, pe_w);

    // Dimension splits
    int dim_h = (HD / 6) * 2;   // 42
    int dim_w = (HD / 6) * 2;   // 42
    int dim_t = HD - dim_h - dim_w;  // 44

    // Patch sizes
    int pe_t = num_frames / COSMOS_PATCH_T;  // frames / 1
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
    rope_cos_cache_ = Tensor({(int64_t)S, (int64_t)HD}, DType::BF16);
    rope_sin_cache_ = Tensor({(int64_t)S, (int64_t)HD}, DType::BF16);

    std::vector<__nv_bfloat16> cos_bf16(S * HD), sin_bf16(S * HD);
    for (int i = 0; i < S * HD; i++) {
        cos_bf16[i] = __float2bfloat16(cos_host[i]);
        sin_bf16[i] = __float2bfloat16(sin_host[i]);
    }
    rope_cos_cache_.copy_from_host(cos_bf16.data(), S * HD * sizeof(__nv_bfloat16));
    rope_sin_cache_.copy_from_host(sin_bf16.data(), S * HD * sizeof(__nv_bfloat16));

    rope_cache_h_ = pe_h;
    rope_cache_w_ = pe_w;
}

// ========================= Attention (scratch-based) =========================

void CosmosTransformer::run_attention(
    const Linear& q_proj, const Linear& k_proj,
    const Linear& v_proj, const Linear& o_proj,
    const Tensor& q_norm_w, const Tensor& k_norm_w,
    const __nv_bfloat16* q_input, int T_q,
    const __nv_bfloat16* kv_input, int T_kv,
    __nv_bfloat16* output,
    bool apply_rope,
    int batch_size,
    cudaStream_t stream) {

    constexpr int H = COSMOS_HEADS;
    constexpr int HD = COSMOS_HEAD_DIM;
    constexpr int D = H * HD;
    int B = batch_size;

    // Use pre-allocated scratch buffers — stay in [B*T, H*HD] layout throughout.
    // No physical transpose: cuDNN SDPA reads via strided descriptors.
    auto* q_buf = scratch_.q_buf.bf16_ptr();
    auto* k_buf = scratch_.k_buf.bf16_ptr();
    auto* v_buf = scratch_.v_buf.bf16_ptr();

    // QKV projections: [B*T, D] @ [D, D] = [B*T, H*HD]
    q_proj.forward(cublas_, q_input, q_buf, B * T_q, stream);
    k_proj.forward(cublas_, kv_input, k_buf, B * T_kv, stream);
    v_proj.forward(cublas_, kv_input, v_buf, B * T_kv, stream);

    // Per-head QK norm — [B*T, H*HD] viewed as [B*T*H, HD] rows (HD contiguous per head)
    rms_norm_bf16(q_buf, q_norm_w.bf16_ptr(), q_buf, B * T_q * H, HD, 1e-6f, stream);
    rms_norm_bf16(k_buf, k_norm_w.bf16_ptr(), k_buf, B * T_kv * H, HD, 1e-6f, stream);

    // Apply 3D RoPE on [B*S, H*HD] layout (self-attention only)
    if (apply_rope && !rope_cos_cache_.empty()) {
        rope_cosmos_strided_bf16(q_buf, rope_cos_cache_.bf16_ptr(), rope_sin_cache_.bf16_ptr(),
                                 B, T_q, H, HD, stream);
        rope_cosmos_strided_bf16(k_buf, rope_cos_cache_.bf16_ptr(), rope_sin_cache_.bf16_ptr(),
                                 B, T_kv, H, HD, stream);
    }

    // Flash attention via cuDNN SDPA with strided layout — no transpose needed.
    // Physical layout [B*T, H*HD] described as logical [B, H, T, HD] with strides:
    //   stride_B = T * H * HD,  stride_H = HD,  stride_T = H * HD,  stride_D = 1
    float scale = 1.0f / std::sqrt((float)HD);
    int64_t q_stride[4] = {(int64_t)T_q * D, (int64_t)HD, (int64_t)D, 1};
    int64_t kv_stride[4] = {(int64_t)T_kv * D, (int64_t)HD, (int64_t)D, 1};

    // Output goes to attn_flat [B*T_q, H*HD] — same strided layout, skip untranspose
    auto* attn_flat = scratch_.attn_flat.bf16_ptr();
    sdpa_.forward(q_buf, k_buf, v_buf, attn_flat,
                  B, H, T_q, T_kv, HD,
                  q_stride, kv_stride, q_stride,  // O uses same strides as Q
                  scale, stream);

    // O projection: [B*T_q, D] -> [B*T_q, D] — reads directly from strided SDPA output
    o_proj.forward(cublas_, attn_flat, output, B * T_q, stream);
}

// ========================= Transformer Block =========================

void CosmosTransformer::forward_block(
    int block_idx,
    __nv_bfloat16* hidden, int S,
    const __nv_bfloat16* encoder_cond, int S_text,
    const __nv_bfloat16* temb,
    int batch_size,
    cudaStream_t stream) {

    auto& B = blocks_[block_idx];
    constexpr int D = COSMOS_HIDDEN;
    constexpr int MLP_DIM = COSMOS_MLP_DIM;
    int BS = batch_size * S;  // total rows for batch-agnostic ops

    CUBLAS_CHECK(cublasSetStream(cublas_, stream));

    // Scratch pointers (pre-allocated, reused across blocks)
    auto* mod = scratch_.mod.bf16_ptr();
    auto* normed = scratch_.normed.bf16_ptr();
    auto* gate = scratch_.gate.bf16_ptr();
    auto* sub_out = scratch_.sub_out.bf16_ptr();
    auto* silu_ts = scratch_.silu_ts.bf16_ptr();
    auto* adaln_mid = scratch_.adaln_mid.bf16_ptr();

    // Helper: compute adaLN-Zero modulation (uses pre-computed silu_ts)
    auto compute_adaln_zero = [&](const CosmosAdaLNZero& aln) {
        aln.linear_1.forward(cublas_, silu_ts, adaln_mid, 1, stream);
        aln.linear_2.forward(cublas_, adaln_mid, mod, 1, stream);
        add_bf16(mod, temb, mod, 3 * D, stream);
    };

    // Helper: extract adaLN shift/scale/gate pointers from mod buffer
    auto get_adaln_ptrs = [&]() {
        return std::make_tuple(mod, mod + D, mod + 2 * D);  // shift, scale, gate
    };

    // --- Block execution ---

    // 1. Self-attention with adaLN-Zero
    compute_adaln_zero(B.norm1);
    {
        auto [shift, scale, g] = get_adaln_ptrs();
        // Fused: LayerNorm(hidden) * (1+scale) + shift → normed
        adaln_layernorm_bf16(hidden, scale, shift, normed, BS, D, 1e-6f, stream);
        CUDA_CHECK(cudaMemcpyAsync(gate, g, D * sizeof(__nv_bfloat16),
                                    cudaMemcpyDeviceToDevice, stream));
    }

    run_attention(B.sa_q_proj, B.sa_k_proj, B.sa_v_proj, B.sa_o_proj,
                  B.sa_q_norm, B.sa_k_norm,
                  normed, S, normed, S,
                  sub_out, true, batch_size, stream);

    // 2. Cross-attention: fuse residual_gate + adaLN of sub-layer 2
    compute_adaln_zero(B.norm2);
    {
        auto [shift, scale, g] = get_adaln_ptrs();
        // Fused: hidden += sub_out * gate; normed = adaLN(hidden)
        residual_gate_adaln_bf16(hidden, sub_out, gate, scale, shift,
                                  normed, BS, D, 1e-6f, stream);
        CUDA_CHECK(cudaMemcpyAsync(gate, g, D * sizeof(__nv_bfloat16),
                                    cudaMemcpyDeviceToDevice, stream));
    }

    run_attention(B.ca_q_proj, B.ca_k_proj, B.ca_v_proj, B.ca_o_proj,
                  B.ca_q_norm, B.ca_k_norm,
                  normed, S, encoder_cond, S_text,
                  sub_out, false, batch_size, stream);

    // 3. FFN: fuse residual_gate + adaLN of sub-layer 3
    compute_adaln_zero(B.norm3);
    {
        auto [shift, scale, g] = get_adaln_ptrs();
        residual_gate_adaln_bf16(hidden, sub_out, gate, scale, shift,
                                  normed, BS, D, 1e-6f, stream);
        CUDA_CHECK(cudaMemcpyAsync(gate, g, D * sizeof(__nv_bfloat16),
                                    cudaMemcpyDeviceToDevice, stream));
    }

    auto* ff_buf = scratch_.ff_buf.bf16_ptr();
    B.ff_proj1.forward(cublas_, normed, ff_buf, BS, stream);
    gelu_tanh_bf16(ff_buf, ff_buf, (int64_t)BS * MLP_DIM, stream);
    B.ff_proj2.forward(cublas_, ff_buf, sub_out, BS, stream);

    // Final residual of this block (no subsequent adaLN to fuse with)
    residual_gate_bcast_bf16(hidden, sub_out, gate, hidden, BS, D, stream);
}

// ========================= Forward =========================

void CosmosTransformer::forward(const Tensor& latents, float timestep,
                                const __nv_bfloat16* encoder_cond, int S_text,
                                int batch_size, int latent_h, int latent_w,
                                __nv_bfloat16* output_buf) {
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

    // Ensure scratch buffers are allocated (no-op if already sufficient size)
    ensure_scratch(S, S_text);

    // Cache 3D RoPE (no-op if resolution unchanged)
    compute_3d_rope(1, latent_h, latent_w);

    // Use pre-allocated scratch buffers instead of per-call allocations
    auto* padded_ptr = scratch_.padded.bf16_ptr();
    auto* patches_ptr = scratch_.patches.bf16_ptr();
    auto* hidden_ptr = scratch_.hidden.bf16_ptr();

    // 1. Concatenate padding mask (all zeros) as 17th channel
    int C_total = C_IN + 1;  // 17
    size_t padded_bytes = (size_t)batch_size * C_total * 1 * latent_h * latent_w * sizeof(__nv_bfloat16);
    size_t latent_bytes_1 = (size_t)C_IN * latent_h * latent_w * sizeof(__nv_bfloat16);
    size_t padded_bytes_1 = (size_t)C_total * latent_h * latent_w * sizeof(__nv_bfloat16);
    CUDA_CHECK(cudaMemset(padded_ptr, 0, padded_bytes));
    // Copy latent into each batch element's first 16 channels (17th channel stays zero)
    for (int b = 0; b < batch_size; b++) {
        CUDA_CHECK(cudaMemcpy(padded_ptr + b * (padded_bytes_1 / sizeof(__nv_bfloat16)),
                               latents.data_ptr(),
                               latent_bytes_1, cudaMemcpyDeviceToDevice));
    }

    // 2. Patchify: [B, 17, 1, H, W] -> [B, S, 68] on GPU
    patchify_3d_bf16(padded_ptr, patches_ptr,
                     batch_size, C_total, 1, latent_h, latent_w,
                     COSMOS_PATCH_T, COSMOS_PATCH_H, COSMOS_PATCH_W, stream);

    // 3. Linear projection: [B, S, 68] -> [B, S, 2048]
    patch_proj_.forward(cublas_, patches_ptr, hidden_ptr, batch_size * S, stream);

    // 4. Compute timestep embedding (uses scratch buffers internally)
    auto* embedded_ts = scratch_.embedded_ts.bf16_ptr();
    auto* temb = scratch_.temb.bf16_ptr();
    compute_timestep_embedding(timestep, embedded_ts, temb, stream);

    // 5. Pre-compute SiLU(embedded_ts) once (reused 84+ times across 28 blocks)
    CUDA_CHECK(cudaMemcpyAsync(scratch_.silu_ts.bf16_ptr(), embedded_ts,
                                D * sizeof(__nv_bfloat16), cudaMemcpyDeviceToDevice, stream));
    silu_bf16(scratch_.silu_ts.bf16_ptr(), scratch_.silu_ts.bf16_ptr(), D, stream);

    // 6. Run 28 transformer blocks
    for (int i = 0; i < COSMOS_LAYERS; i++) {
        forward_block(i, hidden_ptr, S,
                      encoder_cond, S_text,
                      temb, batch_size, stream);

        if ((i + 1) % 10 == 0) {
            fprintf(stderr, "[transformer]   block %d/%d done\n", i + 1, COSMOS_LAYERS);
        }
    }

    // 7. Output norm (reuses scratch buffers — adaln_mid and mod are free after block loop)
    {
        auto* adaln_mid = scratch_.adaln_mid.bf16_ptr();
        output_norm_.linear_1.forward(cublas_, scratch_.silu_ts.bf16_ptr(), adaln_mid, 1, stream);

        // Reuse first 4096 elements of scratch_.mod (which is 6144 — blocks are done)
        auto* mod_4096 = scratch_.mod.bf16_ptr();
        output_norm_.linear_2.forward(cublas_, adaln_mid, mod_4096, 1, stream);

        // Add first 4096 elements of temb (temb is 6144-dim, output norm uses 4096)
        add_bf16(mod_4096, temb, mod_4096, 4096, stream);

        // Split into shift[D], scale[D]
        const __nv_bfloat16* shift = mod_4096;
        const __nv_bfloat16* scale = mod_4096 + D;

        layer_norm_bf16(hidden_ptr, hidden_ptr, batch_size * S, D, 1e-6f, stream);
        scale_shift_bcast_bf16(hidden_ptr, scale, shift, hidden_ptr, batch_size * S, D, stream);
    }

    // 8. Output projection: [B*S, 2048] -> [B*S, out_patch_dim]
    // proj_out shares memory with padded (non-overlapping lifetimes: padded is done after step 2)
    auto* proj_out_ptr = scratch_.proj_out.bf16_ptr();
    output_proj_.forward(cublas_, hidden_ptr, proj_out_ptr, batch_size * S, stream);

    // 9. Unpatchify: [B, S, patch_dim] -> [B, C, T, H, W] into caller-provided buffer
    unpatchify_3d_bf16(proj_out_ptr, output_buf,
                       batch_size, COSMOS_OUT_CHANNELS, 1, latent_h, latent_w,
                       COSMOS_PATCH_T, COSMOS_PATCH_H, COSMOS_PATCH_W, stream);

    fprintf(stderr, "[transformer] forward complete\n");
}
