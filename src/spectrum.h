#pragma once

#include <cuda_runtime.h>
#include <cstdio>

struct SpectrumConfig {
    bool enabled = false;
    int warmup = 8;        // steps before caching starts (DiT: 8-10)
    int window = 2;        // skip interval (2 = every other step)
    int order = 3;         // Chebyshev polynomial order M
    float blend = 0.3f;    // 0=Taylor only, 1=Chebyshev only
    float lambda = 0.1f;   // ridge regularization
    float flex = 0.25f;    // adaptive window growth per actual step
    bool residual = true;       // predict deltas instead of absolute values
    float quality_gate = 0.95f; // min cosine similarity to allow caching (0 = disabled)
    float sigma_rate_max = 0.15f; // max relative sigma change rate for caching
};

struct SpectrumState {
    // Feature buffer: stores last K features for polynomial fitting
    float* feature_buffer = nullptr;  // [K, F] FP32  (F = spatial_size * feature_dim)
    float* step_buffer = nullptr;     // [K] FP32 step indices
    int buffer_size = 0;
    int max_buffer_size = 20;

    // Chebyshev coefficients: [P, F] FP32 where P = order + 1
    float* cheb_coeffs = nullptr;
    bool coeffs_valid = false;

    // Shape info
    int feature_dim = 0;    // D = 2048
    int spatial_size = 0;   // S = pe_h * pe_w
    int max_order = 4;      // M (Chebyshev polynomial order)

    // Step tracking
    int last_actual_step = -1;
    int consecutive_cached = 0;
    float current_window = 2.0f;
    float ridge_lambda = 0.1f;

    // Temporary buffers for fitting
    float* design_matrix = nullptr;   // [K, P] FP32
    float* tau_buffer = nullptr;      // [K] FP32

    // Residual prediction mode
    float* last_actual_feature = nullptr;  // [F] FP32 — last actual hidden state
    float* delta_tmp = nullptr;            // [F] FP32 — temp for delta computation
    bool has_last_actual = false;
    bool use_residual = false;

    // Quality gating
    float last_quality = 1.0f;
    bool force_actual = false;

    void free_all();
};

// Initialize Spectrum state — allocates all GPU buffers
void spectrum_init(SpectrumState& state, int feature_dim, int spatial_size,
                   int max_buffer_size, int max_order, bool use_residual = true,
                   cudaStream_t stream = 0);

// Append feature [F] FP32 to buffer after an actual forward pass
void spectrum_update(SpectrumState& state, const float* feature, int step,
                     cudaStream_t stream = 0);

// Predict feature at target_step using Chebyshev + Taylor blend
// output: [F] FP32 pre-allocated by caller
void spectrum_predict(SpectrumState& state, float* output, int target_step,
                      float blend_weight, cudaStream_t stream = 0);

// Compute cosine similarity between two FP32 vectors on GPU
float spectrum_cosine_similarity(const float* a, const float* b, int64_t N,
                                  cudaStream_t stream = 0);

// Reset state for new inference (does not free memory)
void spectrum_reset(SpectrumState& state);
