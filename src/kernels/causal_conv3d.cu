// Causal Conv3D: cuDNN conv2d fast-path (kT=1) + im2col fallback (kT>1)
// The cuDNN path eliminates the massive im2col column matrix and lets cuDNN
// pick the fastest algorithm (typically implicit-GEMM / Winograd on tensor cores).
#include "kernels.h"
#include "../cuda_utils.cuh"
#include <cublas_v2.h>
#include <cudnn.h>

// ---- cuDNN error checking ----
#define CUDNN_CHECK(call)                                                      \
    do {                                                                        \
        cudnnStatus_t st = (call);                                              \
        if (st != CUDNN_STATUS_SUCCESS) {                                       \
            fprintf(stderr, "cuDNN error at %s:%d: %s\n", __FILE__, __LINE__,   \
                    cudnnGetErrorString(st));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ---- Lazy-init cuDNN handle (one per process, never destroyed) ----
static cudnnHandle_t get_cudnn_handle() {
    static cudnnHandle_t handle = nullptr;
    if (!handle) {
        CUDNN_CHECK(cudnnCreate(&handle));
    }
    return handle;
}

// ---- cuDNN workspace (high-water-mark, same pattern as the im2col workspace) ----
static void* s_cudnn_workspace = nullptr;
static size_t s_cudnn_workspace_bytes = 0;

static void ensure_cudnn_workspace(size_t needed) {
    if (needed > s_cudnn_workspace_bytes) {
        if (s_cudnn_workspace) { CUDA_CHECK(cudaFree(s_cudnn_workspace)); }
        CUDA_CHECK(cudaMalloc(&s_cudnn_workspace, needed));
        s_cudnn_workspace_bytes = needed;
    }
}

// ---- Algorithm cache: avoids repeating cuDNN algorithm search for same config ----
#include <unordered_map>
struct ConvKey {
    int C_in, H, W, C_out, kH, kW, padH, padW;
    bool operator==(const ConvKey& o) const {
        return C_in==o.C_in && H==o.H && W==o.W && C_out==o.C_out &&
               kH==o.kH && kW==o.kW && padH==o.padH && padW==o.padW;
    }
};
struct ConvKeyHash {
    size_t operator()(const ConvKey& k) const {
        size_t h = 0;
        auto mix = [&](int v) { h ^= std::hash<int>()(v) + 0x9e3779b9 + (h<<6) + (h>>2); };
        mix(k.C_in); mix(k.H); mix(k.W); mix(k.C_out);
        mix(k.kH); mix(k.kW); mix(k.padH); mix(k.padW);
        return h;
    }
};
struct CachedConvPlan {
    cudnnConvolutionFwdAlgo_t algo;
    size_t wsSize;
    cudnnTensorDescriptor_t xDesc, yDesc;
    cudnnFilterDescriptor_t wDesc;
    cudnnConvolutionDescriptor_t convDesc;
};
static std::unordered_map<ConvKey, CachedConvPlan, ConvKeyHash> s_conv_cache;

// ---- cuDNN conv2d forward (BF16 data, F32 compute) ----
// Input:  [1, C_in, H, W]   NCHW BF16
// Weight: [C_out, C_in, kH, kW]  NCHW BF16
// Output: [1, C_out, H_out, W_out] NCHW BF16
static void cudnn_conv2d_forward(
    const __nv_bfloat16* input, const __nv_bfloat16* weight,
    __nv_bfloat16* output,
    int C_in, int H, int W,
    int C_out, int kH, int kW,
    int padH, int padW,
    cudaStream_t stream)
{
    cudnnHandle_t cudnn = get_cudnn_handle();
    CUDNN_CHECK(cudnnSetStream(cudnn, stream));

    ConvKey key{C_in, H, W, C_out, kH, kW, padH, padW};
    auto it = s_conv_cache.find(key);

    if (it == s_conv_cache.end()) {
        // First time for this config: create descriptors and find algorithm
        CachedConvPlan plan;

        CUDNN_CHECK(cudnnCreateTensorDescriptor(&plan.xDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(plan.xDesc, CUDNN_TENSOR_NCHW,
                                                CUDNN_DATA_BFLOAT16, 1, C_in, H, W));

        CUDNN_CHECK(cudnnCreateFilterDescriptor(&plan.wDesc));
        CUDNN_CHECK(cudnnSetFilter4dDescriptor(plan.wDesc, CUDNN_DATA_BFLOAT16,
                                                CUDNN_TENSOR_NCHW, C_out, C_in, kH, kW));

        CUDNN_CHECK(cudnnCreateConvolutionDescriptor(&plan.convDesc));
        CUDNN_CHECK(cudnnSetConvolution2dDescriptor(plan.convDesc,
                                                     padH, padW, 1, 1, 1, 1,
                                                     CUDNN_CROSS_CORRELATION,
                                                     CUDNN_DATA_FLOAT));
        CUDNN_CHECK(cudnnSetConvolutionMathType(plan.convDesc, CUDNN_TENSOR_OP_MATH));

        int H_out = H + 2 * padH - kH + 1;
        int W_out = W + 2 * padW - kW + 1;
        CUDNN_CHECK(cudnnCreateTensorDescriptor(&plan.yDesc));
        CUDNN_CHECK(cudnnSetTensor4dDescriptor(plan.yDesc, CUDNN_TENSOR_NCHW,
                                                CUDNN_DATA_BFLOAT16, 1, C_out, H_out, W_out));

        int returnedAlgoCount = 0;
        cudnnConvolutionFwdAlgoPerf_t perfResults[8];
        CUDNN_CHECK(cudnnGetConvolutionForwardAlgorithm_v7(
            cudnn, plan.xDesc, plan.wDesc, plan.convDesc, plan.yDesc,
            8, &returnedAlgoCount, perfResults));

        plan.algo = CUDNN_CONVOLUTION_FWD_ALGO_IMPLICIT_PRECOMP_GEMM;
        for (int i = 0; i < returnedAlgoCount; i++) {
            if (perfResults[i].status == CUDNN_STATUS_SUCCESS) {
                plan.algo = perfResults[i].algo;
                break;
            }
        }

        plan.wsSize = 0;
        CUDNN_CHECK(cudnnGetConvolutionForwardWorkspaceSize(
            cudnn, plan.xDesc, plan.wDesc, plan.convDesc, plan.yDesc,
            plan.algo, &plan.wsSize));

        it = s_conv_cache.emplace(key, plan).first;
    }

    auto& plan = it->second;
    ensure_cudnn_workspace(plan.wsSize);

    float alpha = 1.0f, beta = 0.0f;
    CUDNN_CHECK(cudnnConvolutionForward(
        cudnn, &alpha,
        plan.xDesc, input,
        plan.wDesc, weight,
        plan.convDesc, plan.algo,
        s_cudnn_workspace, s_cudnn_workspace_bytes,
        &beta,
        plan.yDesc, output));
}

// ---- im2col for 3D convolution with causal temporal padding (kT>1 fallback) ----
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

    // ---- Fast path: T=1 => use cuDNN conv2d ----
    // When T=1, causal padding makes all temporal slices except the last (kt=kT-1)
    // contribute zero. So we extract that slice from the weight and run 2D conv.
    if (T == 1) {
        const __nv_bfloat16* w2d = weight;

        // High-water-mark buffer for extracted 2D weights (kT>1 case)
        static __nv_bfloat16* s_w2d = nullptr;
        static int64_t s_w2d_elems = 0;

        if (kT > 1) {
            // weight layout: [C_out, C_in, kT, kH, kW]
            // We need weight[:, :, kT-1, :, :] as contiguous [C_out, C_in, kH, kW].
            int64_t w2d_size = (int64_t)C_out * C_in * kH * kW;
            if (w2d_size > s_w2d_elems) {
                if (s_w2d) { CUDA_CHECK(cudaFree(s_w2d)); }
                CUDA_CHECK(cudaMalloc(&s_w2d, w2d_size * sizeof(__nv_bfloat16)));
                s_w2d_elems = w2d_size;
            }
            // Use cudaMemcpy2DAsync: extract last temporal slice from each (co, ci) pair.
            // Source: weight at offset (kT-1)*kH*kW, pitch = kT*kH*kW elements
            // Dest: contiguous, pitch = kH*kW elements
            int64_t slice_bytes = (int64_t)kH * kW * sizeof(__nv_bfloat16);
            CUDA_CHECK(cudaMemcpy2DAsync(
                s_w2d, slice_bytes,  // dst, dpitch
                weight + (int64_t)(kT - 1) * kH * kW, (int64_t)kT * kH * kW * sizeof(__nv_bfloat16),  // src, spitch
                slice_bytes, (size_t)(C_out * C_in),  // width, height
                cudaMemcpyDeviceToDevice, stream));
            w2d = s_w2d;
        }

        cudnn_conv2d_forward(input, w2d, output,
                             C_in, H, W, C_out, kH, kW,
                             padH, padW, stream);

        // Add bias
        if (bias) {
            int spatial = H_out * W_out;
            int total = C_out * spatial;
            int bk = 256;
            int gd = (total + bk - 1) / bk;
            conv_bias_add_kernel<<<gd, bk, 0, stream>>>(bias, output, C_out, spatial);
        }
        return;
    }

    // ---- Fallback: kT>1 => im2col + cuBLAS GEMM (original path) ----

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
