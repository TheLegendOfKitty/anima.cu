#pragma once

#include "anima.h"
#include "tensor.h"
#include "qwen3.h"
#include "llm_adapter.h"
#include "transformer.h"
#include "scheduler.h"
#include "vae.h"
#include "qwen3_tokenizer.h"
#include "t5_tokenizer.h"
#include "spectrum.h"
#include <string>

struct AnimaOptions {
    std::string prompt = "1girl, solo, smile, best quality";
    std::string negative_prompt = "worst quality, low quality";
    int width = 1024;
    int height = 1024;
    int num_steps = 30;
    float guidance_scale = 4.0f;
    uint64_t seed = 42;
    std::string sampler = "euler_a_rf";      // euler, euler_a_rf
    std::string sigma_schedule = "beta";     // beta, simple, normal, uniform
    float beta_alpha = FORGE_BETA_ALPHA;
    float beta_beta = FORGE_BETA_BETA;
    float eta = 1.0f;
    float s_noise = 1.0f;
    int er_sde_stages = 3;  // ER-SDE max stages (1-3)
    std::string output_path = "output.png";

    // Spectrum acceleration
    SpectrumConfig spectrum;

    // Tokenizer paths
    std::string qwen3_tokenizer_path;
    std::string t5_tokenizer_path;
};

class AnimaPipeline {
public:
    AnimaPipeline() = default;

    // Load all components from model directory
    void load(const std::string& model_dir, const AnimaOptions& opts);

    // Generate an image
    void generate(const AnimaOptions& opts);
    void generate_with_cond(const AnimaOptions& opts, const std::string& pos_path, const std::string& neg_path);
    void vae_decode_from_file(const AnimaOptions& opts, const std::string& latent_path);

private:
    Qwen3Encoder qwen3_;
    LLMAdapter adapter_;
    CosmosTransformer transformer_;
    VAEDecoder vae_;

    Qwen3Tokenizer qwen3_tok_;
    T5Tokenizer t5_tok_;
    bool has_qwen3_tok_ = false;
    bool has_t5_tok_ = false;
    AnimaOptions opts_;  // cached for sampler access

    // Tokenization
    std::pair<std::vector<int>, std::vector<int>> tokenize_qwen3(const std::string& text);
    std::vector<int> tokenize_t5(const std::string& text);

    // Encode prompt to conditioning. Returns (tensor, num_real_tokens).
    std::pair<Tensor, int> encode_prompt(const std::string& prompt);

    // Euler denoising loop
    Tensor sample_euler(Tensor latents, const Tensor& pos_cond, const Tensor& neg_cond,
                         const std::vector<float>& sigmas, float guidance_scale,
                         int latent_h, int latent_w, int S_text);
};
