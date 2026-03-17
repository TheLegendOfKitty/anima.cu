/*
 * VAE Decoder for Anima (AutoencoderKLQwenImage)
 *
 * Uses im2col + cuBLAS GEMM for causal 3D convolutions (causal_conv3d_forward).
 * All weights and activations are BF16. The GEMM uses F32 accumulation internally
 * (CUBLAS_COMPUTE_32F) with BF16 inputs/outputs, matching PyTorch behavior.
 *
 * For T=1 image generation the causal conv3d naturally handles the temporal
 * dimension (causal padding zeros out future frames).
 */

#include "vae.h"
#include "anima.h"
#include "cuda_utils.cuh"
#include "kernels/kernels.h"
#include <cstdio>
#include <cstring>
#include <cmath>
#include <vector>

// ========================= vae_conv3d: im2col-based causal conv3d =========================

// Wraps causal_conv3d_forward for VAE tensors.
// Input:  BF16 [B, C_in, T, H, W] or [B, C_in, H, W] (T=1 implicit)
// Weight: BF16 [C_out, C_in, kT, kH, kW] (5D) or [C_out, C_in, kH, kW] (4D, kT=1)
// Bias:   BF16 [C_out] or empty
// Output: BF16 [B, C_out, T_out, H_out, W_out] reshaped to [B, C_out, H_out, W_out] if T_out=1
static Tensor vae_conv3d(cublasHandle_t cublas, const Tensor& input,
                          const Tensor& weight, const Tensor& bias,
                          int pad_h, int pad_w) {
    // Parse input dims - support both 4D [B,C,H,W] and 5D [B,C,T,H,W]
    int B, C_in, T, H, W;
    if (input.ndim() == 5) {
        B = (int)input.dim(0); C_in = (int)input.dim(1);
        T = (int)input.dim(2); H = (int)input.dim(3); W = (int)input.dim(4);
    } else {
        B = (int)input.dim(0); C_in = (int)input.dim(1);
        T = 1; H = (int)input.dim(2); W = (int)input.dim(3);
    }

    // Parse weight dims - support both 4D and 5D
    int C_out, kT, kH, kW;
    if (weight.ndim() == 5) {
        C_out = (int)weight.dim(0);
        kT = (int)weight.dim(2); kH = (int)weight.dim(3); kW = (int)weight.dim(4);
    } else {
        C_out = (int)weight.dim(0);
        kT = 1; kH = (int)weight.dim(2); kW = (int)weight.dim(3);
    }

    // Compute output dims
    int causal_pad = kT - 1;
    int T_padded = T + causal_pad;
    int T_out = T_padded - kT + 1;  // = T for stride 1
    int H_out = H + 2 * pad_h - kH + 1;
    int W_out = W + 2 * pad_w - kW + 1;

    // Allocate output
    Tensor output;
    if (T_out == 1) {
        output = Tensor({(int64_t)B, (int64_t)C_out, (int64_t)H_out, (int64_t)W_out}, DType::BF16);
    } else {
        output = Tensor({(int64_t)B, (int64_t)C_out, (int64_t)T_out, (int64_t)H_out, (int64_t)W_out}, DType::BF16);
    }

    // Process each batch (batch=1 always for VAE, but handle generally)
    int64_t in_stride = (int64_t)C_in * T * H * W;
    int64_t out_stride = (int64_t)C_out * T_out * H_out * W_out;

    const __nv_bfloat16* bias_ptr = bias.empty() ? nullptr : bias.bf16_ptr();

    for (int b = 0; b < B; b++) {
        causal_conv3d_forward(
            cublas,
            input.bf16_ptr() + b * in_stride,
            weight.bf16_ptr(),
            bias_ptr,
            output.bf16_ptr() + b * out_stride,
            C_in, T, H, W,
            C_out, kT, kH, kW,
            pad_h, pad_w,
            0  // stream
        );
    }

    return output;
}

// ========================= Weight loading helpers =========================

// Extract even rows from BF16 2D weight for pixel shuffle frame 0
// Input: [2*C_out, C_in] BF16 -> Output: [C_out, C_in] BF16
static Tensor extract_even_rows_bf16(const Tensor& w, int out_dim) {
    int full_out = (int)w.dim(0);
    int in_dim = (int)w.dim(1);

    std::vector<__nv_bfloat16> host(w.numel());
    w.copy_to_host(host.data(), w.size_bytes());

    std::vector<__nv_bfloat16> even(out_dim * in_dim);
    for (int i = 0; i < out_dim; i++)
        memcpy(&even[(int64_t)i * in_dim], &host[(int64_t)(i * 2) * in_dim], in_dim * 2);

    Tensor out({(int64_t)out_dim, (int64_t)in_dim}, DType::BF16);
    out.copy_from_host(even.data(), even.size() * 2);
    return out;
}

// Extract even elements from BF16 bias for pixel shuffle frame 0
static Tensor extract_even_elements_bf16(const Tensor& b, int out_dim) {
    std::vector<__nv_bfloat16> host(b.numel());
    b.copy_to_host(host.data(), b.size_bytes());

    std::vector<__nv_bfloat16> even(out_dim);
    for (int i = 0; i < out_dim; i++)
        even[i] = host[i * 2];

    Tensor out({(int64_t)out_dim}, DType::BF16);
    out.copy_from_host(even.data(), out_dim * 2);
    return out;
}

// ========================= VAENorm =========================

void VAENorm::load(const SafeTensorsFile& f, const std::string& key) {
    Tensor raw = f.load(key);
    // Squeeze from [C,1,1,1] or [C,1,1] to [C]
    channels = (int)raw.dim(0);
    int64_t numel = raw.numel();

    if (numel == channels) {
        gamma = std::move(raw);
    } else {
        // Copy just the channel elements (they're strided with trailing 1s)
        gamma = Tensor({(int64_t)channels}, DType::BF16);
        std::vector<__nv_bfloat16> host(numel);
        raw.copy_to_host(host.data(), numel * 2);
        gamma.copy_from_host(host.data(), channels * 2);
    }
}

void VAENorm::forward(Tensor& x) const {
    // x: [B, C, H, W] BF16
    int B = (int)x.dim(0), C = (int)x.dim(1);
    int spatial = (int)(x.dim(2) * x.dim(3));

    // Fused RMS norm + gamma (single BF16 write, avoids roundtrip)
    rms_norm_channel_bf16(x.bf16_ptr(), gamma.bf16_ptr(), x.bf16_ptr(), B, C, spatial, 1e-6f, 0);
}

// ========================= VAEResBlock =========================

void VAEResBlock::load(const SafeTensorsFile& f, const std::string& prefix) {
    // norm1: prefix.residual.0.gamma
    norm1.load(f, prefix + ".residual.0.gamma");
    // conv1: prefix.residual.2.weight/bias (5D BF16, kept as-is)
    conv1_w = f.load(prefix + ".residual.2.weight");
    conv1_b = f.load(prefix + ".residual.2.bias");
    // norm2: prefix.residual.3.gamma
    norm2.load(f, prefix + ".residual.3.gamma");
    // conv2: prefix.residual.6.weight/bias
    conv2_w = f.load(prefix + ".residual.6.weight");
    conv2_b = f.load(prefix + ".residual.6.bias");

    // Shortcut (1x1 conv if in_ch != out_ch)
    has_skip = f.has(prefix + ".shortcut.weight");
    if (has_skip) {
        skip_w = f.load(prefix + ".shortcut.weight");  // [out, in, 1, 1, 1] BF16
        skip_b = f.load(prefix + ".shortcut.bias");
    }
}

Tensor VAEResBlock::forward(const Tensor& x, cublasHandle_t cublas) const {
    // Shortcut path
    Tensor skip;
    if (has_skip) {
        skip = vae_conv3d(cublas, x, skip_w, skip_b, 0, 0);
    }

    // Main path: norm1 -> SiLU -> conv1
    Tensor out(x.shape(), DType::BF16);
    CUDA_CHECK(cudaMemcpy(out.data_ptr(), x.data_ptr(), x.size_bytes(), cudaMemcpyDeviceToDevice));
    norm1.forward(out);
    silu_bf16(out.bf16_ptr(), out.bf16_ptr(), out.numel(), 0);
    out = vae_conv3d(cublas, out, conv1_w, conv1_b, 1, 1);

    // norm2 -> SiLU -> conv2
    norm2.forward(out);
    silu_bf16(out.bf16_ptr(), out.bf16_ptr(), out.numel(), 0);
    out = vae_conv3d(cublas, out, conv2_w, conv2_b, 1, 1);

    // Residual
    const Tensor& residual = has_skip ? skip : x;
    add_bf16(out.bf16_ptr(), residual.bf16_ptr(), out.bf16_ptr(), out.numel(), 0);

    return out;
}

// ========================= VAEAttention =========================

void VAEAttention::load(const SafeTensorsFile& f, const std::string& prefix) {
    // prefix = "decoder.middle.1"
    norm.load(f, prefix + ".norm.gamma");
    channels = norm.channels;

    // to_qkv: [3*C, C, 1, 1] BF16
    qkv_w = f.load(prefix + ".to_qkv.weight");
    qkv_b = f.load(prefix + ".to_qkv.bias");

    // proj: [C, C, 1, 1] BF16
    proj_w = f.load(prefix + ".proj.weight");
    proj_b = f.load(prefix + ".proj.bias");
}

Tensor VAEAttention::forward(const Tensor& x, cublasHandle_t cublas) const {
    int B = (int)x.dim(0), C = channels;
    int H = (int)x.dim(2), W = (int)x.dim(3);
    int S = H * W;  // spatial = number of "tokens" for channel attention

    // Norm
    Tensor normed(x.shape(), DType::BF16);
    CUDA_CHECK(cudaMemcpy(normed.data_ptr(), x.data_ptr(), x.size_bytes(), cudaMemcpyDeviceToDevice));
    norm.forward(normed);

    // QKV projection: 1x1 conv via vae_conv3d (handles bias correctly)
    Tensor qkv = vae_conv3d(cublas, normed, qkv_w, qkv_b, 0, 0);

    // Split Q, K, V - all BF16
    Tensor Q({(int64_t)B, (int64_t)C, (int64_t)S}, DType::BF16);
    Tensor K({(int64_t)B, (int64_t)C, (int64_t)S}, DType::BF16);
    Tensor V({(int64_t)B, (int64_t)C, (int64_t)S}, DType::BF16);

    {
        auto* src = qkv.bf16_ptr();
        size_t chunk = (size_t)C * S;
        for (int b = 0; b < B; b++) {
            CUDA_CHECK(cudaMemcpy(Q.bf16_ptr() + b * chunk, src + b * 3 * chunk, chunk * 2, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(K.bf16_ptr() + b * chunk, src + b * 3 * chunk + chunk, chunk * 2, cudaMemcpyDeviceToDevice));
            CUDA_CHECK(cudaMemcpy(V.bf16_ptr() + b * chunk, src + b * 3 * chunk + 2 * chunk, chunk * 2, cudaMemcpyDeviceToDevice));
        }
    }

    // For each batch: attn = softmax(Q @ K^T / sqrt(S)) @ V
    // Attention scores computed in F32, softmax in F32, then BF16 output
    float scale = 1.0f / sqrtf((float)S);
    Tensor scores({(int64_t)B, (int64_t)C, (int64_t)C}, DType::F32);
    Tensor attn_out({(int64_t)B, (int64_t)C, (int64_t)S}, DType::BF16);

    CUBLAS_CHECK(cublasSetStream(cublas, 0));

    for (int b = 0; b < B; b++) {
        auto* q = Q.bf16_ptr() + (int64_t)b * C * S;
        auto* k = K.bf16_ptr() + (int64_t)b * C * S;
        auto* v = V.bf16_ptr() + (int64_t)b * C * S;
        auto* sc = scores.f32_ptr() + (int64_t)b * C * C;
        auto* ao = attn_out.bf16_ptr() + (int64_t)b * C * S;

        // scores = Q @ K^T: BF16 inputs, F32 output
        float alpha_s = scale, beta_s = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(cublas,
            CUBLAS_OP_T, CUBLAS_OP_N,
            C, C, S, &alpha_s,
            k, CUDA_R_16BF, S,
            q, CUDA_R_16BF, S,
            &beta_s,
            sc, CUDA_R_32F, C,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT));

        // Softmax in F32
        softmax_f32(sc, sc, C, C);

        // attn_out = scores @ V: convert scores to BF16, then BF16 GEMM
        Tensor sc_bf16({(int64_t)C, (int64_t)C}, DType::BF16);
        f32_to_bf16(sc, sc_bf16.bf16_ptr(), (int64_t)C * C, 0);

        float one = 1.0f, zero = 0.0f;
        CUBLAS_CHECK(cublasGemmEx(cublas,
            CUBLAS_OP_N, CUBLAS_OP_N,
            S, C, C, &one,
            v, CUDA_R_16BF, S,
            sc_bf16.bf16_ptr(), CUDA_R_16BF, C,
            &zero,
            ao, CUDA_R_16BF, S,
            CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));
    }

    // Reshape attn_out [B, C, S] -> [B, C, H, W] and project via 1x1 conv
    attn_out.reshape({(int64_t)B, (int64_t)C, (int64_t)H, (int64_t)W});
    Tensor projected = vae_conv3d(cublas, attn_out, proj_w, proj_b, 0, 0);

    // Residual
    add_bf16(projected.bf16_ptr(), x.bf16_ptr(), projected.bf16_ptr(), projected.numel(), 0);
    return projected;
}

// ========================= VAEUpsampler =========================

void VAEUpsampler::load(const SafeTensorsFile& f, const std::string& prefix, bool temporal) {
    has_time = temporal;

    // Spatial conv: prefix.resample.1.weight [C_out, C_in, 3, 3] BF16 4D
    spatial_w = f.load(prefix + ".resample.1.weight");
    spatial_b = f.load(prefix + ".resample.1.bias");

    if (temporal) {
        // Time conv: [2*C, C, 3, 1, 1] -> extract last temporal slice -> [2*C, C]
        // Then extract even rows for frame 0 pixel shuffle: [C, C]
        Tensor tw5 = f.load(prefix + ".time_conv.weight");
        int full_out = (int)tw5.dim(0);
        int in_ch = (int)tw5.dim(1);
        int half_out = full_out / 2;

        // Extract last temporal slice: [2*C, C, 3, 1, 1] -> [2*C, C]
        Tensor tw2d({(int64_t)full_out, (int64_t)in_ch}, DType::BF16);
        {
            std::vector<__nv_bfloat16> host5(tw5.numel());
            tw5.copy_to_host(host5.data(), tw5.size_bytes());
            std::vector<__nv_bfloat16> host2(full_out * in_ch);
            for (int co = 0; co < full_out; co++)
                for (int ci = 0; ci < in_ch; ci++)
                    host2[(int64_t)co * in_ch + ci] = host5[((int64_t)co * in_ch + ci) * 3 + 2];
            tw2d.copy_from_host(host2.data(), host2.size() * 2);
        }

        // Extract even rows for pixel shuffle frame 0 (BF16)
        time_w = extract_even_rows_bf16(tw2d, half_out);
        time_w.reshape({(int64_t)half_out, (int64_t)in_ch, 1, 1});

        // Extract even bias elements (BF16)
        Tensor tb_full = f.load(prefix + ".time_conv.bias");
        time_b = extract_even_elements_bf16(tb_full, half_out);
    }
}

Tensor VAEUpsampler::forward(const Tensor& x, cublasHandle_t cublas) const {
    // Copy input
    Tensor current(x.shape(), DType::BF16);
    CUDA_CHECK(cudaMemcpy(current.data_ptr(), x.data_ptr(), x.size_bytes(), cudaMemcpyDeviceToDevice));

    // Time conv + pixel shuffle (for T=1 frame 0) - BF16
    if (false && has_time) { // DISABLED: skip time_conv for T=1 (match qwen-image.cu)
        // Precomputed: 1x1 conv with even-row weights gives frame 0 directly
        // time_w is 4D [C_out, C_in, 1, 1], kT=1
        current = vae_conv3d(cublas, current, time_w, time_b, 0, 0);
    }

    // Nearest-neighbor 2x spatial upsample - BF16
    int B = (int)current.dim(0), C = (int)current.dim(1);
    int H = (int)current.dim(2), W = (int)current.dim(3);
    Tensor upsampled({(int64_t)B, (int64_t)C, (int64_t)(H * 2), (int64_t)(W * 2)}, DType::BF16);
    nearest_upsample_2x_bf16(current.bf16_ptr(), upsampled.bf16_ptr(), B, C, H, W, 0);

    // Spatial conv (3x3, pad=1) - BF16, spatial_w is 4D [Co, Ci, 3, 3]
    return vae_conv3d(cublas, upsampled, spatial_w, spatial_b, 1, 1);
}

// ========================= VAEDecoder =========================

void VAEDecoder::load(const SafeTensorsFile& f) {
    // Init cuBLAS
    CUBLAS_CHECK(cublasCreate(&cublas_));

    // Post-quant conv: conv2 [16, 16, 1, 1, 1] BF16 5D
    pqc_w = f.load("conv2.weight");
    pqc_b = f.load("conv2.bias");

    // Conv in: [384, 16, 3, 3, 3] BF16 5D
    cin_w = f.load("decoder.conv1.weight");
    cin_b = f.load("decoder.conv1.bias");

    // Mid block
    mid_res0.load(f, "decoder.middle.0");
    mid_attn.load(f, "decoder.middle.1");
    mid_res1.load(f, "decoder.middle.2");

    // 15 ResBlocks (upsamples 0-14, skipping 3, 7, 11 which are upsamplers)
    for (int i = 0; i < 15; i++) {
        if (i == 3 || i == 7 || i == 11) continue;
        char key[128];
        snprintf(key, sizeof(key), "decoder.upsamples.%d", i);
        res[i].load(f, key);
    }

    // 3 upsamplers at positions 3, 7, 11
    up[0].load(f, "decoder.upsamples.3", true);   // temporal
    up[1].load(f, "decoder.upsamples.7", true);   // temporal
    up[2].load(f, "decoder.upsamples.11", false);  // spatial only

    // Output
    norm_out.load(f, "decoder.head.0.gamma");
    cout_w = f.load("decoder.head.2.weight");
    cout_b = f.load("decoder.head.2.bias");

    fprintf(stderr, "[vae] decoder loaded (BF16, im2col+GEMM)\n");
}

Tensor VAEDecoder::decode(const Tensor& latents) {
    // Input: [1, 16, H, W] BF16 normalized latents
    // Denormalize: z = z * std + mean (stay in BF16)
    Tensor z(latents.shape(), DType::BF16);
    {
        int64_t C = latents.dim(1), spatial = latents.dim(2) * latents.dim(3);
        std::vector<__nv_bfloat16> host(latents.numel());
        latents.copy_to_host(host.data(), latents.size_bytes());

        // Denormalize in BF16
        for (int64_t c = 0; c < C; c++) {
            float mean = VAE_LATENTS_MEAN[c];
            float std_val = VAE_LATENTS_STD[c];
            for (int64_t s = 0; s < spatial; s++) {
                int64_t idx = c * spatial + s;
                float v = __bfloat162float(host[idx]);
                host[idx] = __float2bfloat16(v * std_val + mean);
            }
        }
        z.copy_from_host(host.data(), latents.numel() * sizeof(__nv_bfloat16));
    }
    fprintf(stderr, "[vae] denormalized latents (BF16)\n");

    // Post-quant conv (1x1) - BF16
    z = vae_conv3d(cublas_, z, pqc_w, pqc_b, 0, 0);

    // Conv in (3x3, pad=1) - BF16
    Tensor x = vae_conv3d(cublas_, z, cin_w, cin_b, 1, 1);
    fprintf(stderr, "[vae] conv_in -> [%ld, %ld, %ld, %ld]\n",
            x.dim(0), x.dim(1), x.dim(2), x.dim(3));

    // Mid block
    x = mid_res0.forward(x, cublas_);
    x = mid_attn.forward(x, cublas_);
    x = mid_res1.forward(x, cublas_);
    fprintf(stderr, "[vae] mid_block done\n");

    // Up blocks
    // Group 0: res[0], res[1], res[2], up[0]
    for (int i = 0; i < 3; i++) {
        x = res[i].forward(x, cublas_);
    }
    x = up[0].forward(x, cublas_);
    fprintf(stderr, "[vae] up_block 0 -> [%ld, %ld, %ld, %ld]\n",
            x.dim(0), x.dim(1), x.dim(2), x.dim(3));

    // Group 1: res[4], res[5], res[6], up[1]
    for (int i = 4; i <= 6; i++) {
        x = res[i].forward(x, cublas_);
    }
    x = up[1].forward(x, cublas_);
    fprintf(stderr, "[vae] up_block 1 -> [%ld, %ld, %ld, %ld]\n",
            x.dim(0), x.dim(1), x.dim(2), x.dim(3));

    // Group 2: res[8], res[9], res[10], up[2]
    for (int i = 8; i <= 10; i++) {
        x = res[i].forward(x, cublas_);
    }
    x = up[2].forward(x, cublas_);
    fprintf(stderr, "[vae] up_block 2 -> [%ld, %ld, %ld, %ld]\n",
            x.dim(0), x.dim(1), x.dim(2), x.dim(3));

    // Group 3: res[12], res[13], res[14] (no upsampler)
    for (int i = 12; i <= 14; i++) {
        x = res[i].forward(x, cublas_);
    }
    fprintf(stderr, "[vae] up_block 3 -> [%ld, %ld, %ld, %ld]\n",
            x.dim(0), x.dim(1), x.dim(2), x.dim(3));

    // Output: norm -> SiLU -> conv (all BF16)
    norm_out.forward(x);
    silu_bf16(x.bf16_ptr(), x.bf16_ptr(), x.numel(), 0);
    x = vae_conv3d(cublas_, x, cout_w, cout_b, 1, 1);

    fprintf(stderr, "[vae] decoded -> [%ld, %ld, %ld, %ld] (BF16)\n",
            x.dim(0), x.dim(1), x.dim(2), x.dim(3));

    return x;  // BF16 output
}
