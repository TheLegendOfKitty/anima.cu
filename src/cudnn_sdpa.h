#pragma once

#include <cudnn.h>
#include <cuda_bf16.h>
#include <cstdint>

// Wrapper for cuDNN flash attention (SDPA).
// Builds a cudnn_frontend graph for scaled dot-product attention
// and caches it for repeated execution at the same sequence lengths.
//
// Supports strided layouts — Q/K/V/O can be in [B*S, H*HD] physical layout
// described with strides [S*H*HD, HD, H*HD, 1] for logical [B, H, S, HD].
// This eliminates the need for head transpose kernels.

class CudnnSDPA {
public:
    CudnnSDPA() = default;
    ~CudnnSDPA();

    void init(cudnnHandle_t handle);

    // Execute fused flash attention with explicit strides.
    // Q/K/V/O pointers point to BF16 data.
    // Strides are for logical [B, H, S, HD] layout (4 elements each).
    void forward(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
                 __nv_bfloat16* O,
                 int B, int H, int S_q, int S_kv, int HD,
                 const int64_t* q_stride,   // [4]: strides for [B, H, S_q, HD]
                 const int64_t* kv_stride,  // [4]: strides for [B, H, S_kv, HD]
                 const int64_t* o_stride,   // [4]: strides for [B, H, S_q, HD]
                 float attn_scale,
                 cudaStream_t stream);

private:
    cudnnHandle_t handle_ = nullptr;

    struct CachedPlan;
    struct PlanCache;
    PlanCache* cache_ = nullptr;

    CachedPlan* get_plan(int B, int H, int S_q, int S_kv, int HD,
                         const int64_t* q_stride, const int64_t* kv_stride,
                         const int64_t* o_stride, float attn_scale);

    void* workspace_ = nullptr;
    size_t workspace_bytes_ = 0;
    void ensure_workspace(size_t needed);
};
