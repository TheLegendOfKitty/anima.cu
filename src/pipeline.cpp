#include "pipeline.h"
#include "safetensors.h"
#include "kernels/kernels.h"
#include "cuda_utils.cuh"

#include <cstdio>
#include <cmath>
#include <vector>
#include <curand.h>

#define STB_IMAGE_WRITE_IMPLEMENTATION
#include <stb_image_write.h>

// ========================= Load =========================

void AnimaPipeline::load(const std::string& model_dir, const AnimaOptions& opts) {
    char path[512];

    // Qwen3 text encoder
    snprintf(path, sizeof(path), "%s/text_encoders/qwen_3_06b_base.safetensors", model_dir.c_str());
    SafeTensorsFile text_encoder_file(path);
    qwen3_.load(text_encoder_file);

    // Transformer + adapter
    snprintf(path, sizeof(path), "%s/diffusion_models/anima-preview2.safetensors", model_dir.c_str());
    SafeTensorsFile transformer_file(path);
    adapter_.load(transformer_file);
    transformer_.load(transformer_file);

    // VAE decoder
    snprintf(path, sizeof(path), "%s/vae/qwen_image_vae.safetensors", model_dir.c_str());
    SafeTensorsFile vae_file(path);
    vae_.load(vae_file);

    // Tokenizers (optional — falls back to byte-level if not found)
    if (!opts.qwen3_tokenizer_path.empty()) {
        has_qwen3_tok_ = qwen3_tok_.load(opts.qwen3_tokenizer_path);
    }
    if (!opts.t5_tokenizer_path.empty()) {
        has_t5_tok_ = t5_tok_.load(opts.t5_tokenizer_path);
    }

    fprintf(stderr, "[pipeline] all components loaded (qwen3_tok=%s, t5_tok=%s)\n",
            has_qwen3_tok_ ? "yes" : "fallback", has_t5_tok_ ? "yes" : "fallback");
}

// ========================= Tokenization =========================

std::pair<std::vector<int>, std::vector<int>> AnimaPipeline::tokenize_qwen3(const std::string& text) {
    std::vector<int> ids;

    if (has_qwen3_tok_) {
        // Real BPE tokenizer (add_special_tokens=False)
        ids = qwen3_tok_.tokenize(text);
    } else {
        // Byte-level fallback
        for (unsigned char c : text)
            ids.push_back((int)c + 256);
    }
    if (ids.empty()) ids.push_back(QWEN3_DEFAULT_PAD);

    std::vector<int> mask(ids.size(), 1);
    return {ids, mask};
}

std::vector<int> AnimaPipeline::tokenize_t5(const std::string& text) {
    if (has_t5_tok_) {
        return t5_tok_.tokenize(text);
    }

    // Byte-level fallback
    std::vector<int> ids;
    for (unsigned char c : text)
        ids.push_back((int)c + 100);
    ids.push_back(1);  // EOS
    return ids;
}

// ========================= Prompt Encoding =========================

std::pair<Tensor, int> AnimaPipeline::encode_prompt(const std::string& prompt) {
    // 1. Tokenize for Qwen3
    auto [qwen_ids, qwen_mask] = tokenize_qwen3(prompt);
    int T_qwen = (int)qwen_ids.size();

    fprintf(stderr, "[encode] qwen3 tokens: %d\n", T_qwen);

    // Upload to GPU
    Tensor qwen_ids_gpu({(int64_t)T_qwen}, DType::F32);  // int32 via F32
    qwen_ids_gpu.copy_from_host(qwen_ids.data(), T_qwen * sizeof(int));
    Tensor qwen_mask_gpu({(int64_t)T_qwen}, DType::F32);
    qwen_mask_gpu.copy_from_host(qwen_mask.data(), T_qwen * sizeof(int));

    // 2. Run Qwen3 forward
    Tensor qwen_hidden = qwen3_.forward(
        reinterpret_cast<const int*>(qwen_ids_gpu.data_ptr()),
        reinterpret_cast<const int*>(qwen_mask_gpu.data_ptr()),
        T_qwen);

    // 3. Tokenize for T5
    auto t5_ids = tokenize_t5(prompt);
    int T_t5 = (int)t5_ids.size();

    fprintf(stderr, "[encode] t5 tokens: %d\n", T_t5);

    Tensor t5_ids_gpu({(int64_t)T_t5}, DType::F32);
    t5_ids_gpu.copy_from_host(t5_ids.data(), T_t5 * sizeof(int));

    // 4. Run LLM adapter (produces [512, 1024] padded output)
    Tensor cond = adapter_.forward(
        qwen_hidden.bf16_ptr(), T_qwen,
        reinterpret_cast<const int*>(t5_ids_gpu.data_ptr()), T_t5);

    // Return with actual token count (not padded length)
    return {std::move(cond), T_t5};
}

// ========================= Sampler =========================

Tensor AnimaPipeline::sample_euler(Tensor latents, const Tensor& pos_cond, const Tensor& neg_cond,
                                    const std::vector<float>& sigmas, float guidance_scale,
                                    int latent_h, int latent_w, int S_text) {
    int64_t numel = latents.numel();
    bool use_ancestral = (opts_.sampler == "euler_a_rf");

    // Pre-allocate output buffers for forward() — reused across all denoising steps.
    // Each forward() writes into its own buffer so pos/neg results coexist.
    Tensor noise_pos({1, (int64_t)COSMOS_OUT_CHANNELS, 1, (int64_t)latent_h, (int64_t)latent_w}, DType::BF16);
    Tensor noise_neg({1, (int64_t)COSMOS_OUT_CHANNELS, 1, (int64_t)latent_h, (int64_t)latent_w}, DType::BF16);

    // Pre-allocate noise buffers for ancestral sampler
    Tensor rand_noise;
    Tensor noise_f32;
    curandGenerator_t gen = nullptr;
    if (use_ancestral) {
        rand_noise = Tensor({numel}, DType::BF16);
        noise_f32 = Tensor({numel}, DType::F32);
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, opts_.seed + 1000);
    }

    for (int i = 0; i < (int)sigmas.size() - 1; i++) {
        float sigma = sigmas[i];
        float sigma_next = sigmas[i + 1];

        fprintf(stderr, "[sampler] step %d/%d: sigma=%.4f -> %.4f\n",
                i + 1, (int)sigmas.size() - 1, sigma, sigma_next);

        transformer_.forward(latents, sigma,
                             pos_cond.bf16_ptr(), S_text,
                             1, latent_h, latent_w,
                             noise_pos.bf16_ptr());
        transformer_.forward(latents, sigma,
                             neg_cond.bf16_ptr(), S_text,
                             1, latent_h, latent_w,
                             noise_neg.bf16_ptr());

        if (use_ancestral && sigma_next > 0.0f) {
            // Generate fresh noise for this step (reuse pre-allocated noise_f32)
            curandGenerateNormal(gen, noise_f32.f32_ptr(), numel, 0.0f, 1.0f);
            f32_to_bf16(noise_f32.f32_ptr(), rand_noise.bf16_ptr(), numel);

            cfg_euler_a_rf_step_bf16(
                latents.bf16_ptr(), noise_pos.bf16_ptr(), noise_neg.bf16_ptr(),
                rand_noise.bf16_ptr(), latents.bf16_ptr(),
                guidance_scale, sigma, sigma_next,
                opts_.eta, opts_.s_noise, numel, 0);
        } else {
            // Plain Euler (or last step of ancestral)
            if (use_ancestral && sigma_next == 0.0f) {
                // Last step: just compute denoised = sample - sigma * noise
                cfg_euler_step_bf16(latents.bf16_ptr(), noise_pos.bf16_ptr(), noise_neg.bf16_ptr(),
                                    latents.bf16_ptr(), guidance_scale, sigma, 0.0f, numel, 0);
            } else {
                cfg_euler_step_bf16(latents.bf16_ptr(), noise_pos.bf16_ptr(), noise_neg.bf16_ptr(),
                                    latents.bf16_ptr(), guidance_scale, sigma, sigma_next, numel, 0);
            }
        }
    }

    if (gen) curandDestroyGenerator(gen);
    return latents;
}

// ========================= Generate =========================

void AnimaPipeline::generate(const AnimaOptions& opts) {
    opts_ = opts;  // cache for sampler access
    fprintf(stderr, "\n[pipeline] generating image: %dx%d, steps=%d, cfg=%.1f, seed=%lu, sampler=%s\n",
            opts.width, opts.height, opts.num_steps, opts.guidance_scale, opts.seed, opts.sampler.c_str());

    // Compute latent dimensions
    int step = VAE_SCALE_FACTOR * COSMOS_PATCH_H;  // 8 * 2 = 16
    int height = (opts.height / step) * step;
    int width = (opts.width / step) * step;
    int latent_h = height / VAE_SCALE_FACTOR;
    int latent_w = width / VAE_SCALE_FACTOR;

    fprintf(stderr, "[pipeline] image=%dx%d, latent=%dx%d\n", width, height, latent_h, latent_w);

    // 1. Encode prompts
    fprintf(stderr, "\n--- Encoding positive prompt ---\n");
    auto [pos_cond, pos_T] = encode_prompt(opts.prompt);

    fprintf(stderr, "\n--- Encoding negative prompt ---\n");
    auto [neg_cond, neg_T] = encode_prompt(opts.negative_prompt);

    // Use padded 512 conditioning (model trained with this)
    int S_text = CONDITIONING_MAX_LEN;
    fprintf(stderr, "[pipeline] conditioning tokens: pos=%d, neg=%d, S_text=%d\n", pos_T, neg_T, S_text);

    // 2. Generate initial noise
    int64_t latent_numel = 1 * COSMOS_IN_CHANNELS * 1 * latent_h * latent_w;
    Tensor latents({1, (int64_t)COSMOS_IN_CHANNELS, 1, (int64_t)latent_h, (int64_t)latent_w}, DType::BF16);

    {
        Tensor noise_f32({latent_numel}, DType::F32);
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, opts.seed);
        curandGenerateNormal(gen, noise_f32.f32_ptr(), latent_numel, 0.0f, 1.0f);
        curandDestroyGenerator(gen);
        f32_to_bf16(noise_f32.f32_ptr(), latents.bf16_ptr(), latent_numel);
    }

    // 3. Build sigma schedule
    int num_train_timesteps = 1000;
    float shift = 3.0f;

    std::vector<float> sigmas;
    if (opts.sigma_schedule == "beta") {
        sigmas = build_beta_sigmas(opts.num_steps, num_train_timesteps, shift,
                                    opts.beta_alpha, opts.beta_beta);
    } else if (opts.sigma_schedule == "simple") {
        sigmas = build_simple_sigmas(opts.num_steps, num_train_timesteps, shift);
    } else if (opts.sigma_schedule == "normal") {
        sigmas = build_normal_sigmas(opts.num_steps, num_train_timesteps, shift);
    } else {
        sigmas = build_uniform_sigmas(opts.num_steps, num_train_timesteps, shift);
    }

    fprintf(stderr, "[pipeline] sigma schedule (%s): %zu steps, range [%.4f, %.4f]\n",
            opts.sigma_schedule.c_str(), sigmas.size() - 1, sigmas.front(), sigmas[sigmas.size()-2]);

    // Scale initial noise by first sigma
    scale_bf16(latents.bf16_ptr(), latents.bf16_ptr(), sigmas[0], latent_numel);



    // 4. Denoising loop
    fprintf(stderr, "\n--- Denoising ---\n");
    latents = sample_euler(std::move(latents), pos_cond, neg_cond,
                            sigmas, opts.guidance_scale, latent_h, latent_w, S_text);

    // Save denoised latents for Python VAE decode fallback
    {
        std::string lp = opts.output_path + ".latents";
        std::vector<__nv_bfloat16> lh(latents.numel());
        latents.copy_to_host(lh.data(), latents.size_bytes());
        FILE* fp = fopen(lp.c_str(), "wb");
        if (fp) { fwrite(lh.data(), 2, lh.size(), fp); fclose(fp); }
        fprintf(stderr, "[pipeline] saved latents to %s\n", lp.c_str());
    }

    // 5. VAE decode
    fprintf(stderr, "\n--- VAE decode ---\n");

    // Reshape latents from [1, 16, 1, H, W] to [1, 16, H, W] for the decoder
    latents.reshape({1, (int64_t)COSMOS_IN_CHANNELS, (int64_t)latent_h, (int64_t)latent_w});


    Tensor decoded = vae_.decode(latents);

    // Convert decoded [1, 3, H, W] BF16 (range roughly [-1, 1]) to RGB uint8
    {
        int out_h = (int)decoded.dim(2);
        int out_w = (int)decoded.dim(3);
        int64_t pixels = (int64_t)out_h * out_w;

        std::vector<__nv_bfloat16> dec_host(decoded.numel());
        decoded.copy_to_host(dec_host.data(), decoded.size_bytes());

        std::vector<uint8_t> rgb(pixels * 3);
        for (int64_t i = 0; i < pixels; i++) {
            for (int c = 0; c < 3; c++) {
                float v = __bfloat162float(dec_host[c * pixels + i]);
                v = v / 2.0f + 0.5f;  // [-1,1] -> [0,1]
                v = std::max(0.0f, std::min(1.0f, v));
                rgb[i * 3 + c] = (uint8_t)(v * 255.0f);
            }
        }

        stbi_write_png(opts.output_path.c_str(), out_w, out_h, 3,
                        rgb.data(), out_w * 3);
        fprintf(stderr, "[pipeline] saved %s (%dx%d)\n",
                opts.output_path.c_str(), out_w, out_h);
    }

    fprintf(stderr, "\n[pipeline] generation complete\n");
}

// Debug: generate with external conditioning
void AnimaPipeline::generate_with_cond(const AnimaOptions& opts,
    const std::string& pos_path, const std::string& neg_path) {
    opts_ = opts;
    int step = VAE_SCALE_FACTOR * COSMOS_PATCH_H;
    int height = (opts.height / step) * step;
    int width = (opts.width / step) * step;
    int latent_h = height / VAE_SCALE_FACTOR;
    int latent_w = width / VAE_SCALE_FACTOR;
    
    // Load conditioning from files (BF16 as uint16)
    Tensor pos_cond({CONDITIONING_MAX_LEN, 1024}, DType::BF16);
    Tensor neg_cond({CONDITIONING_MAX_LEN, 1024}, DType::BF16);
    {
        FILE* f = fopen(pos_path.c_str(), "rb");
        std::vector<uint16_t> buf(512*1024);
        fread(buf.data(), 2, buf.size(), f); fclose(f);
        pos_cond.copy_from_host(buf.data(), buf.size()*2);
        f = fopen(neg_path.c_str(), "rb");
        fread(buf.data(), 2, buf.size(), f); fclose(f);
        neg_cond.copy_from_host(buf.data(), buf.size()*2);
    }
    fprintf(stderr, "[pipeline] loaded external conditioning\n");
    
    int S_text = CONDITIONING_MAX_LEN;
    int64_t latent_numel = COSMOS_IN_CHANNELS * latent_h * latent_w;
    Tensor latents({1, (int64_t)COSMOS_IN_CHANNELS, 1, (int64_t)latent_h, (int64_t)latent_w}, DType::BF16);
    {
        Tensor noise_f32({latent_numel}, DType::F32);
        curandGenerator_t gen;
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, opts.seed);
        curandGenerateNormal(gen, noise_f32.f32_ptr(), latent_numel, 0.0f, 1.0f);
        curandDestroyGenerator(gen);
        f32_to_bf16(noise_f32.f32_ptr(), latents.bf16_ptr(), latent_numel);
    }
    
    auto sigmas = build_beta_sigmas(opts.num_steps, 1000, 3.0f, opts.beta_alpha, opts.beta_beta);
    scale_bf16(latents.bf16_ptr(), latents.bf16_ptr(), sigmas[0], latent_numel);
    
    latents = sample_euler(std::move(latents), pos_cond, neg_cond,
                            sigmas, opts.guidance_scale, latent_h, latent_w, S_text);
    
    latents.reshape({1, (int64_t)COSMOS_IN_CHANNELS, (int64_t)latent_h, (int64_t)latent_w});
    Tensor decoded = vae_.decode(latents);
    
    int out_h = (int)decoded.dim(2), out_w = (int)decoded.dim(3);
    int64_t pixels = (int64_t)out_h * out_w;
    std::vector<__nv_bfloat16> dec_host(decoded.numel());
    decoded.copy_to_host(dec_host.data(), decoded.size_bytes());
    std::vector<uint8_t> rgb(pixels * 3);
    for (int64_t i = 0; i < pixels; i++)
        for (int c = 0; c < 3; c++) {
            float v = __bfloat162float(dec_host[c * pixels + i]);
            v = std::max(0.0f, std::min(1.0f, v / 2.0f + 0.5f));
            rgb[i * 3 + c] = (uint8_t)(v * 255.0f);
        }
    stbi_write_png(opts.output_path.c_str(), out_w, out_h, 3, rgb.data(), out_w * 3);
    fprintf(stderr, "[pipeline] saved %s\n", opts.output_path.c_str());
}

void AnimaPipeline::vae_decode_from_file(const AnimaOptions& opts, const std::string& latent_path) {
    opts_ = opts;
    int latent_h = opts.height / VAE_SCALE_FACTOR;
    int latent_w = opts.width / VAE_SCALE_FACTOR;
    int64_t numel = COSMOS_IN_CHANNELS * latent_h * latent_w;
    
    Tensor latents({1, (int64_t)COSMOS_IN_CHANNELS, (int64_t)latent_h, (int64_t)latent_w}, DType::BF16);
    FILE* f = fopen(latent_path.c_str(), "rb");
    std::vector<uint16_t> buf(numel);
    fread(buf.data(), 2, numel, f); fclose(f);
    latents.copy_from_host(buf.data(), numel * 2);
    fprintf(stderr, "[vae-test] loaded latents [%ld]\n", numel);
    
    Tensor decoded = vae_.decode(latents);
    
    int out_h = (int)decoded.dim(2), out_w = (int)decoded.dim(3);
    int64_t pixels = (int64_t)out_h * out_w;
    std::vector<__nv_bfloat16> dec_host(decoded.numel());
    decoded.copy_to_host(dec_host.data(), decoded.size_bytes());
    std::vector<uint8_t> rgb(pixels * 3);
    for (int64_t i = 0; i < pixels; i++)
        for (int c = 0; c < 3; c++) {
            float v = __bfloat162float(dec_host[c * pixels + i]);
            v = std::max(0.0f, std::min(1.0f, v / 2.0f + 0.5f));
            rgb[i * 3 + c] = (uint8_t)(v * 255.0f);
        }
    stbi_write_png(opts.output_path.c_str(), out_w, out_h, 3, rgb.data(), out_w * 3);
    fprintf(stderr, "[vae-test] saved %s (%dx%d)\n", opts.output_path.c_str(), out_w, out_h);
}
