#include "pipeline.h"
#include "safetensors.h"
#include "kernels/kernels.h"
#include "cuda_utils.cuh"

#include <cstdio>
#include <cmath>
#include <vector>
#include <algorithm>
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

// ========================= ER-SDE Helpers =========================

// Default ER-SDE noise scaler: f(x) = x * (exp(x^0.3) + 10)
static float er_sde_noise_scaler(float x) {
    return x * (expf(powf(fabsf(x), 0.3f)) + 10.0f);
}

// er_lambda for flow matching (CONST): sigma / (1 - sigma)
static float er_lambda_from_sigma(float sigma) {
    float s = fminf(sigma, 1.0f - 1e-4f);  // clamp to avoid infinity
    return s / (1.0f - s);
}

// Compute numerical integrals s and s_u for ER-SDE stages 2-3
// s   = integral from er_lambda_t to er_lambda_s of 1/f(lambda) d_lambda
// s_u = integral from er_lambda_t to er_lambda_s of (lambda - er_lambda_s)/f(lambda) d_lambda
static void er_sde_compute_integrals(float er_lambda_s, float er_lambda_t,
                                      float& s_out, float& s_u_out) {
    const int N_POINTS = 200;
    float dt = er_lambda_t - er_lambda_s;
    float step_size = -dt / (float)N_POINTS;

    float sum_s = 0.0f;
    float sum_su = 0.0f;
    for (int k = 0; k < N_POINTS; k++) {
        float lambda_pos = er_lambda_t + (float)k * step_size;
        float scaled = er_sde_noise_scaler(lambda_pos);
        if (fabsf(scaled) > 1e-10f) {
            sum_s += 1.0f / scaled;
            sum_su += (lambda_pos - er_lambda_s) / scaled;
        }
    }
    s_out = sum_s * step_size;
    s_u_out = sum_su * step_size;
}

// ========================= Sampler =========================

Tensor AnimaPipeline::sample_euler(Tensor latents, const Tensor& pos_cond, const Tensor& neg_cond,
                                    const std::vector<float>& sigmas, float guidance_scale,
                                    int latent_h, int latent_w, int S_text) {
    int64_t numel = latents.numel();
    bool use_ancestral = (opts_.sampler == "euler_a_rf");
    bool use_er_sde = (opts_.sampler == "er_sde");
    bool needs_noise = use_ancestral || use_er_sde;
    const auto& spec = opts_.spectrum;

    // Pre-allocate batched output buffer for forward_batched_cfg() — [2, C, 1, H, W].
    // Batch 0 = positive conditioning, batch 1 = negative conditioning.
    int64_t single_numel = (int64_t)COSMOS_OUT_CHANNELS * 1 * latent_h * latent_w;
    Tensor noise_batched({2, (int64_t)COSMOS_OUT_CHANNELS, 1, (int64_t)latent_h, (int64_t)latent_w}, DType::BF16);
    auto* noise_pos_ptr = noise_batched.bf16_ptr();
    auto* noise_neg_ptr = noise_batched.bf16_ptr() + single_numel;

    // Pre-allocate noise buffers for stochastic samplers
    Tensor rand_noise;
    Tensor noise_f32;
    curandGenerator_t gen = nullptr;
    if (needs_noise) {
        rand_noise = Tensor({numel}, DType::BF16);
        noise_f32 = Tensor({numel}, DType::F32);
        curandCreateGenerator(&gen, CURAND_RNG_PSEUDO_DEFAULT);
        curandSetPseudoRandomGeneratorSeed(gen, opts_.seed + 1000);
    }

    // ---- Spectrum setup ----
    int pe_h = latent_h / COSMOS_PATCH_H;
    int pe_w = latent_w / COSMOS_PATCH_W;
    int S = pe_h * pe_w;
    SpectrumState spectrum_pos, spectrum_neg;
    float* hidden_capture_buf = nullptr;
    float* predicted_hidden = nullptr;
    int spectrum_cached_steps = 0;

    if (spec.enabled) {
        int max_buf = std::min(opts_.num_steps, 20);
        spectrum_init(spectrum_pos, COSMOS_HIDDEN, S, max_buf, spec.order, spec.residual);
        spectrum_init(spectrum_neg, COSMOS_HIDDEN, S, max_buf, spec.order, spec.residual);
        spectrum_pos.current_window = (float)spec.window;
        spectrum_neg.current_window = (float)spec.window;
        spectrum_pos.ridge_lambda = spec.lambda;
        spectrum_neg.ridge_lambda = spec.lambda;

        // Capture buffer: [2*S, D] FP32 (pos in first half, neg in second)
        CUDA_CHECK(cudaMalloc(&hidden_capture_buf,
                              (size_t)2 * S * COSMOS_HIDDEN * sizeof(float)));
        // Prediction buffer: [S, D] FP32 (reused for pos and neg)
        CUDA_CHECK(cudaMalloc(&predicted_hidden,
                              (size_t)S * COSMOS_HIDDEN * sizeof(float)));

        fprintf(stderr, "[spectrum] enabled: warmup=%d, window=%d, order=%d, blend=%.2f, lambda=%.3f, flex=%.2f, residual=%s, quality_gate=%.2f\n",
                spec.warmup, spec.window, spec.order, spec.blend, spec.lambda, spec.flex,
                spec.residual ? "yes" : "no", spec.quality_gate);
    }

    // ---- ER-SDE setup ----
    // Double-buffered denoised and denoised_d for multi-stage ER-SDE
    Tensor er_denoised_a, er_denoised_b, er_dd_a, er_dd_b;
    __nv_bfloat16 *er_cur_denoised = nullptr, *er_old_denoised = nullptr;
    __nv_bfloat16 *er_cur_dd = nullptr, *er_old_dd = nullptr;
    std::vector<float> er_lambdas;
    int er_max_stage = opts_.er_sde_stages;

    if (use_er_sde) {
        er_denoised_a = Tensor({numel}, DType::BF16);
        er_denoised_b = Tensor({numel}, DType::BF16);
        er_dd_a = Tensor({numel}, DType::BF16);
        er_dd_b = Tensor({numel}, DType::BF16);
        er_cur_denoised = er_denoised_a.bf16_ptr();
        er_old_denoised = er_denoised_b.bf16_ptr();
        er_cur_dd = er_dd_a.bf16_ptr();
        er_old_dd = er_dd_b.bf16_ptr();

        // Precompute er_lambda = sigma/(1-sigma) for all sigma values
        er_lambdas.resize(sigmas.size());
        for (size_t j = 0; j < sigmas.size(); j++) {
            er_lambdas[j] = er_lambda_from_sigma(sigmas[j]);
        }

        fprintf(stderr, "[er_sde] max_stage=%d, s_noise=%.2f\n", er_max_stage, opts_.s_noise);
    }

    for (int i = 0; i < (int)sigmas.size() - 1; i++) {
        float sigma = sigmas[i];
        float sigma_next = sigmas[i + 1];

        // ---- Spectrum decision ----
        bool use_cache = false;
        int total_steps = (int)sigmas.size() - 1;
        if (spec.enabled && i >= spec.warmup) {
            // Never cache the last 3 steps (critical for fine detail)
            bool is_final = (total_steps - i) <= 3;
            int steps_since = i - spectrum_pos.last_actual_step;
            int win = (int)floorf(spectrum_pos.current_window);
            if (win < 1) win = 1;
            bool has_min = spectrum_pos.buffer_size >= 2 && spectrum_neg.buffer_size >= 2;

            // Sigma-rate check: don't cache when sigma changes too rapidly
            float sigma_rate = (sigma - sigma_next) / fmaxf(sigma, 1e-6f);
            bool sigma_ok = sigma_rate < spec.sigma_rate_max;

            // Quality gate: force actual if previous prediction quality was low
            bool quality_ok = !spectrum_pos.force_actual && !spectrum_neg.force_actual;

            use_cache = has_min && !is_final && sigma_ok && quality_ok
                        && (steps_since % win) != 0;
        }

        if (use_cache) {
            fprintf(stderr, "[sampler] step %d/%d: sigma=%.4f -> %.4f [spectrum cached]\n",
                    i + 1, (int)sigmas.size() - 1, sigma, sigma_next);

            // Predict positive hidden state and run output-only
            spectrum_predict(spectrum_pos, predicted_hidden, i, spec.blend);
            transformer_.forward_output_only(predicted_hidden, sigma,
                                              1, S, latent_h, latent_w, noise_pos_ptr);

            // Predict negative hidden state and run output-only
            spectrum_predict(spectrum_neg, predicted_hidden, i, spec.blend);
            transformer_.forward_output_only(predicted_hidden, sigma,
                                              1, S, latent_h, latent_w, noise_neg_ptr);

            spectrum_pos.consecutive_cached++;
            spectrum_neg.consecutive_cached++;
            spectrum_cached_steps++;
        } else {
            fprintf(stderr, "[sampler] step %d/%d: sigma=%.4f -> %.4f%s\n",
                    i + 1, (int)sigmas.size() - 1, sigma, sigma_next,
                    spec.enabled ? " [actual]" : "");

            // Enable hidden state capture if Spectrum is active
            if (spec.enabled) {
                transformer_.set_hidden_capture(hidden_capture_buf);
            }

            transformer_.forward_batched_cfg(latents, sigma,
                                 pos_cond.bf16_ptr(), neg_cond.bf16_ptr(),
                                 S_text, latent_h, latent_w, noise_batched.bf16_ptr());

            if (spec.enabled) {
                transformer_.set_hidden_capture(nullptr);

                // Split captured hidden [2*S, D] into pos and neg halves
                float* pos_hidden = hidden_capture_buf;
                float* neg_hidden = hidden_capture_buf + (size_t)S * COSMOS_HIDDEN;
                int64_t F_hidden = (int64_t)S * COSMOS_HIDDEN;

                // Shadow prediction for quality gate (before update changes buffer)
                if (spec.quality_gate > 0.0f && i >= spec.warmup
                    && spectrum_pos.buffer_size >= 2) {
                    spectrum_predict(spectrum_pos, predicted_hidden, i, spec.blend);
                    float cos_pos = spectrum_cosine_similarity(
                        predicted_hidden, pos_hidden, F_hidden);

                    spectrum_predict(spectrum_neg, predicted_hidden, i, spec.blend);
                    float cos_neg = spectrum_cosine_similarity(
                        predicted_hidden, neg_hidden, F_hidden);

                    float min_cos = fminf(cos_pos, cos_neg);
                    spectrum_pos.last_quality = min_cos;
                    spectrum_neg.last_quality = min_cos;

                    bool gate = min_cos < spec.quality_gate;
                    spectrum_pos.force_actual = gate;
                    spectrum_neg.force_actual = gate;

                    if (gate) {
                        fprintf(stderr, "[spectrum] quality gate: cos_sim=%.4f < %.2f\n",
                                min_cos, spec.quality_gate);
                    }
                }

                spectrum_update(spectrum_pos, pos_hidden, i);
                spectrum_update(spectrum_neg, neg_hidden, i);

                // Adaptive window growth
                if (spec.flex > 0.0f && i > spec.warmup) {
                    spectrum_pos.current_window += spec.flex;
                    spectrum_neg.current_window += spec.flex;
                }
            }
        }

        // ---- CFG + sampler step (same for actual and cached) ----
        if (use_er_sde) {
            // Clamp sigmas for er_lambda computation (avoid infinity at sigma=1)
            float sigma_c = fminf(sigma, 1.0f - 1e-4f);

            if (sigma_next == 0.0f) {
                // Final step: x = denoised = x - sigma * velocity
                cfg_euler_step_bf16(latents.bf16_ptr(), noise_pos_ptr, noise_neg_ptr,
                                    latents.bf16_ptr(), guidance_scale, sigma_c, 0.0f, numel, 0);
            } else {
                float sigma_next_c = fminf(sigma_next, 1.0f - 1e-4f);
                float el_s = er_lambdas[i];
                float el_t = er_lambdas[i + 1];

                float alpha_s = 1.0f - sigma_c;
                float alpha_t = 1.0f - sigma_next_c;
                float r_alpha = alpha_t / alpha_s;

                float ns_s = er_sde_noise_scaler(el_s);
                float ns_t = er_sde_noise_scaler(el_t);
                float r = ns_t / ns_s;

                // Stage 1 coefficients
                float A = r_alpha * r;
                float B = alpha_t * (1.0f - r);

                // Stage 2-3 coefficients
                int stage_used = std::min(er_max_stage, i + 1);
                float d_scale = 0.0f, coeff2 = 0.0f;
                float u_scale = 0.0f, coeff3 = 0.0f;

                if (stage_used >= 2) {
                    float dt = el_t - el_s;
                    float s_int, s_u_int;
                    er_sde_compute_integrals(el_s, el_t, s_int, s_u_int);

                    d_scale = 1.0f / (el_s - er_lambdas[i - 1]);
                    coeff2 = alpha_t * (dt + s_int * ns_t);

                    if (stage_used >= 3) {
                        u_scale = 2.0f / (el_s - er_lambdas[i - 2]);
                        coeff3 = alpha_t * (dt * dt / 2.0f + s_u_int * ns_t);
                    }
                }

                // Noise coefficient: alpha_t * s_noise * sqrt(el_t^2 - el_s^2 * r^2)
                float noise_sq = el_t * el_t - el_s * el_s * r * r;
                float noise_coeff = (noise_sq > 0.0f && opts_.s_noise > 0.0f)
                    ? alpha_t * opts_.s_noise * sqrtf(noise_sq) : 0.0f;

                // Handle NaN from extreme noise_scaler values
                if (!isfinite(noise_coeff)) noise_coeff = 0.0f;

                // Generate random noise if needed
                if (noise_coeff > 0.0f) {
                    curandGenerateNormal(gen, noise_f32.f32_ptr(), numel, 0.0f, 1.0f);
                    f32_to_bf16(noise_f32.f32_ptr(), rand_noise.bf16_ptr(), numel);
                }

                // Determine which history buffers to pass
                __nv_bfloat16* old_den_ptr = (i > 0) ? er_old_denoised : nullptr;
                __nv_bfloat16* dd_out_ptr = (stage_used >= 2) ? er_cur_dd : nullptr;
                __nv_bfloat16* old_dd_ptr = (stage_used >= 3 && i >= 2) ? er_old_dd : nullptr;

                cfg_er_sde_full_step_bf16(
                    latents.bf16_ptr(), noise_pos_ptr, noise_neg_ptr,
                    er_cur_denoised, old_den_ptr,
                    dd_out_ptr, old_dd_ptr,
                    (noise_coeff > 0.0f) ? rand_noise.bf16_ptr() : nullptr,
                    guidance_scale, sigma_c,
                    A, B, d_scale, coeff2, u_scale, coeff3, noise_coeff,
                    numel, 0);

                // Swap double buffers
                std::swap(er_cur_denoised, er_old_denoised);
                if (stage_used >= 2) {
                    std::swap(er_cur_dd, er_old_dd);
                }
            }
        } else if (use_ancestral && sigma_next > 0.0f) {
            curandGenerateNormal(gen, noise_f32.f32_ptr(), numel, 0.0f, 1.0f);
            f32_to_bf16(noise_f32.f32_ptr(), rand_noise.bf16_ptr(), numel);

            cfg_euler_a_rf_step_bf16(
                latents.bf16_ptr(), noise_pos_ptr, noise_neg_ptr,
                rand_noise.bf16_ptr(), latents.bf16_ptr(),
                guidance_scale, sigma, sigma_next,
                opts_.eta, opts_.s_noise, numel, 0);
        } else {
            if (use_ancestral && sigma_next == 0.0f) {
                cfg_euler_step_bf16(latents.bf16_ptr(), noise_pos_ptr, noise_neg_ptr,
                                    latents.bf16_ptr(), guidance_scale, sigma, 0.0f, numel, 0);
            } else {
                cfg_euler_step_bf16(latents.bf16_ptr(), noise_pos_ptr, noise_neg_ptr,
                                    latents.bf16_ptr(), guidance_scale, sigma, sigma_next, numel, 0);
            }
        }
    }

    if (gen) curandDestroyGenerator(gen);

    // ---- Spectrum cleanup ----
    if (spec.enabled) {
        fprintf(stderr, "[spectrum] done: %d/%d steps cached (%.0f%% speedup potential)\n",
                spectrum_cached_steps, (int)sigmas.size() - 1,
                100.0f * spectrum_cached_steps / ((int)sigmas.size() - 1));
        spectrum_pos.free_all();
        spectrum_neg.free_all();
        if (hidden_capture_buf) { cudaFree(hidden_capture_buf); }
        if (predicted_hidden) { cudaFree(predicted_hidden); }
    }

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
