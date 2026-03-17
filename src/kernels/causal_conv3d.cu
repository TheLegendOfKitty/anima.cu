// Causal Conv3D via im2col + cuBLAS GEMM
// Ported from qwen-image.cu which produces correct VAE output
#include "kernels.h"
#include "../cuda_utils.cuh"
#include <cublas_v2.h>

// im2col for 3D convolution with causal temporal padding
__global__ void im2col_3d_kernel(const __nv_bfloat16* input, __nv_bfloat16* col,
                                  int C_in, int T, int H, int W,
                                  int kT, int kH, int kW,
                                  int padH, int padW, int causal_pad,
                                  int strideT, int strideH, int strideW,
                                  int T_out, int H_out, int W_out) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)C_in * kT * kH * kW * T_out * H_out * W_out;
    if (idx >= total) return;

    int w_out = idx % W_out;
    int64_t rest = idx / W_out;
    int h_out = rest % H_out;
    rest = rest / H_out;
    int t_out = rest % T_out;
    rest = rest / T_out;
    int kw = rest % kW;
    rest = rest / kW;
    int kh = rest % kH;
    rest = rest / kH;
    int kt = rest % kT;
    int c = rest / kT;

    int t_in = t_out * strideT - causal_pad + kt;
    int h_in = h_out * strideH - padH + kh;
    int w_in = w_out * strideW - padW + kw;

    __nv_bfloat16 val;
    if (t_in >= 0 && t_in < T && h_in >= 0 && h_in < H && w_in >= 0 && w_in < W) {
        val = input[((int64_t)c * T + t_in) * H * W + h_in * W + w_in];
    } else {
        val = __float2bfloat16(0.0f);
    }

    int col_row = c * kT * kH * kW + kt * kH * kW + kh * kW + kw;
    int col_col = t_out * H_out * W_out + h_out * W_out + w_out;
    col[(int64_t)col_row * (T_out * H_out * W_out) + col_col] = val;
}

__global__ void conv_bias_add_kernel(const __nv_bfloat16* bias, __nv_bfloat16* out,
                                      int C, int spatial) {
    int64_t idx = (int64_t)blockIdx.x * blockDim.x + threadIdx.x;
    int64_t total = (int64_t)C * spatial;
    if (idx >= total) return;
    int c = idx / spatial;
    float v = __bfloat162float(out[idx]) + __bfloat162float(bias[c]);
    out[idx] = __float2bfloat16(v);
}

void causal_conv3d_forward(
    cublasHandle_t cublas,
    const __nv_bfloat16* input, const __nv_bfloat16* weight, const __nv_bfloat16* bias,
    __nv_bfloat16* output,
    int C_in, int T, int H, int W,
    int C_out, int kT, int kH, int kW,
    int padH, int padW,
    cudaStream_t stream) {

    int causal_pad = kT - 1;
    int T_padded = T + causal_pad;
    int H_out = (H + 2 * padH - kH) + 1;
    int W_out = (W + 2 * padW - kW) + 1;
    int T_out = (T_padded - kT) + 1;

    // Reuse workspace across calls (high-water-mark allocation avoids per-call malloc/free)
    static __nv_bfloat16* s_workspace = nullptr;
    static int64_t s_workspace_elems = 0;

    int64_t col_size = (int64_t)C_in * kT * kH * kW * T_out * H_out * W_out;
    if (col_size > s_workspace_elems) {
        if (s_workspace) { CUDA_CHECK(cudaFree(s_workspace)); }
        CUDA_CHECK(cudaMalloc(&s_workspace, col_size * sizeof(__nv_bfloat16)));
        s_workspace_elems = col_size;
    }

    // im2col
    int block = 256;
    int grid = (int)((col_size + block - 1) / block);
    im2col_3d_kernel<<<grid, block, 0, stream>>>(input, s_workspace,
        C_in, T, H, W, kT, kH, kW, padH, padW, causal_pad,
        1, 1, 1, T_out, H_out, W_out);

    // GEMM: weight[C_out, C_in*kT*kH*kW] @ col[C_in*kT*kH*kW, T_out*H_out*W_out]
    float alpha = 1.0f, beta = 0.0f;
    int M = C_out;
    int K = C_in * kT * kH * kW;
    int N = T_out * H_out * W_out;

    CUBLAS_CHECK(cublasSetStream(cublas, stream));
    CUBLAS_CHECK(cublasGemmEx(
        cublas, CUBLAS_OP_N, CUBLAS_OP_N,
        N, M, K,
        &alpha,
        s_workspace, CUDA_R_16BF, N,
        weight, CUDA_R_16BF, K,
        &beta,
        output, CUDA_R_16BF, N,
        CUBLAS_COMPUTE_32F, CUBLAS_GEMM_DEFAULT_TENSOR_OP));

    // Add bias
    if (bias) {
        int spatial = T_out * H_out * W_out;
        int total = C_out * spatial;
        int bk = 256;
        int gd = (total + bk - 1) / bk;
        conv_bias_add_kernel<<<gd, bk, 0, stream>>>(bias, output, C_out, spatial);
    }
}
