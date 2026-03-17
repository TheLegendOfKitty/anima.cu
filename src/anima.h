#pragma once

#include <cstdint>
#include <cuda_bf16.h>

// Anima model constants

// Qwen3-0.6B text encoder
constexpr int QWEN3_VOCAB_SIZE       = 151936;
constexpr int QWEN3_HIDDEN           = 1024;
constexpr int QWEN3_INTERMEDIATE     = 3072;
constexpr int QWEN3_LAYERS           = 28;
constexpr int QWEN3_HEADS            = 16;
constexpr int QWEN3_KV_HEADS         = 8;
constexpr int QWEN3_HEAD_DIM         = 128;  // actual from weights: q_proj=[2048,1024]
constexpr float QWEN3_RMS_EPS        = 1e-6f;
constexpr float QWEN3_ROPE_THETA     = 1000000.0f;
constexpr int QWEN3_DEFAULT_PAD      = 151643;

// LLM Adapter
constexpr int ADAPTER_VOCAB_SIZE     = 32128;
constexpr int ADAPTER_DIM            = 1024;
constexpr int ADAPTER_LAYERS         = 6;
constexpr int ADAPTER_HEADS          = 16;
constexpr int ADAPTER_HEAD_DIM       = 64;   // 1024 / 16
constexpr float ADAPTER_ROPE_THETA   = 10000.0f;
constexpr int CONDITIONING_MAX_LEN   = 512;

// Cosmos Transformer
constexpr int COSMOS_HIDDEN          = 2048;  // 16 heads * 128 head_dim
constexpr int COSMOS_HEADS           = 16;
constexpr int COSMOS_HEAD_DIM        = 128;
constexpr int COSMOS_LAYERS          = 28;
constexpr float COSMOS_MLP_RATIO     = 4.0f;
constexpr int COSMOS_MLP_DIM         = 8192; // 2048 * 4
constexpr int COSMOS_ADALN_DIM       = 256;
constexpr int COSMOS_TEXT_DIM        = 1024;
constexpr int COSMOS_IN_CHANNELS     = 16;
constexpr int COSMOS_OUT_CHANNELS    = 16;
constexpr int COSMOS_PATCH_T         = 1;
constexpr int COSMOS_PATCH_H         = 2;
constexpr int COSMOS_PATCH_W         = 2;

// VAE
constexpr int VAE_Z_DIM              = 16;
constexpr int VAE_BASE_DIM           = 96;
constexpr int VAE_SCALE_FACTOR       = 8;

// Sampling
constexpr float ANIMA_SAMPLING_MULT  = 1.0f;
constexpr float FORGE_BETA_ALPHA     = 0.6f;
constexpr float FORGE_BETA_BETA      = 0.6f;

// VAE latent statistics (per-channel)
constexpr float VAE_LATENTS_MEAN[16] = {
    -0.7571f, -0.7089f, -0.9113f,  0.1075f, -0.1745f,  0.9653f, -0.1517f,  1.5508f,
     0.4134f, -0.0715f,  0.5517f, -0.3632f, -0.1922f, -0.9497f,  0.2503f, -0.2921f,
};
constexpr float VAE_LATENTS_STD[16] = {
     2.8184f,  1.4541f,  2.3275f,  2.6558f,  1.2196f,  1.7708f,  2.6052f,  2.0743f,
     3.2687f,  2.1526f,  2.8652f,  1.5579f,  1.6382f,  1.1253f,  2.8251f,  1.9160f,
};
