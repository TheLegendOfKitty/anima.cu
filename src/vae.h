#pragma once

#include "tensor.h"
#include "safetensors.h"
#include <cublas_v2.h>

// ========================= VAE components =========================
// All weights and activations are BF16.
// Convolutions use im2col + cuBLAS GEMM (causal_conv3d_forward).

// QwenImageRMS_norm: pixel_norm(x) * gamma
struct VAENorm {
    Tensor gamma;   // [C] BF16
    int channels = 0;

    void load(const SafeTensorsFile& f, const std::string& key);
    // in-place: norm the input, multiply by gamma
    void forward(Tensor& x) const;
};

// ResBlock: norm1 -> SiLU -> conv1 -> norm2 -> SiLU -> conv2 + shortcut
struct VAEResBlock {
    VAENorm norm1, norm2;
    Tensor conv1_w, conv1_b;  // [out, in, kT, 3, 3] BF16 (5D)
    Tensor conv2_w, conv2_b;  // [out, out, kT, 3, 3] BF16 (5D)
    Tensor skip_w, skip_b;    // [out, in, 1, 1, 1] BF16 (5D, if in != out)
    bool has_skip = false;

    void load(const SafeTensorsFile& f, const std::string& prefix);
    Tensor forward(const Tensor& x, cublasHandle_t cublas) const;
};

// Mid-block attention: channel-wise self-attention
struct VAEAttention {
    VAENorm norm;
    Tensor qkv_w, qkv_b;   // [3*C, C, 1, 1] BF16
    Tensor proj_w, proj_b;  // [C, C, 1, 1] BF16
    int channels = 0;

    void load(const SafeTensorsFile& f, const std::string& prefix);
    Tensor forward(const Tensor& x, cublasHandle_t cublas) const;
};

// Upsampler: nearest 2x + spatial conv, optional time_conv (as precomputed 1x1 for T=1)
struct VAEUpsampler {
    Tensor spatial_w, spatial_b;  // [C_out, C_in, 3, 3] BF16 4D
    Tensor time_w, time_b;        // [C_out, C_in, 1, 1] BF16 4D (precomputed for T=1 frame 0)
    bool has_time = false;

    void load(const SafeTensorsFile& f, const std::string& prefix, bool temporal);
    Tensor forward(const Tensor& x, cublasHandle_t cublas) const;
};

// ========================= Full VAE Decoder =========================

class VAEDecoder {
public:
    void load(const SafeTensorsFile& f);
    // Decode latents to RGB image
    // Input: [1, 16, H/8, W/8] BF16 (normalized latents from denoiser)
    // Output: [1, 3, H, W] BF16 (pixel values in [-1, 1])
    Tensor decode(const Tensor& latents);

private:
    cublasHandle_t cublas_ = nullptr;

    // post_quant_conv: conv2 in checkpoint (16->16, 1x1, 5D)
    Tensor pqc_w, pqc_b;

    // decoder.conv_in: (16->384, 3x3, 5D)
    Tensor cin_w, cin_b;

    // Mid block
    VAEResBlock mid_res0, mid_res1;
    VAEAttention mid_attn;

    // 15 ResBlocks (upsamples 0-14)
    VAEResBlock res[15];

    // 3 upsamplers (at positions 3, 7, 11)
    VAEUpsampler up[3];

    // Output
    VAENorm norm_out;
    Tensor cout_w, cout_b;  // (96->3, 3x3, 5D)
};
