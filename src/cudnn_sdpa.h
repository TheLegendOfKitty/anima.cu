#pragma once

#include <cudnn.h>
#include <cuda_bf16.h>
#include <cstdint>

// Wrapper for cuDNN flash attention (SDPA).
// Builds a cudnn_frontend graph for scaled dot-product attention
// and caches it for repeated execution at the same sequence lengths.
//
// Expected layout: Q [B, H, S_q, HD], K [B, H, S_kv, HD], V [B, H, S_kv, HD]
// All BF16, F32 compute, no causal mask, no dropout.

class CudnnSDPA {
public:
    CudnnSDPA() = default;
    ~CudnnSDPA();

    // Initialize with a cuDNN handle (creates internal state).
    void init(cudnnHandle_t handle);

    // Execute fused flash attention.
    // Q: [B, H, S_q, HD] BF16
    // K: [B, H, S_kv, HD] BF16
    // V: [B, H, S_kv, HD] BF16
    // O: [B, H, S_q, HD] BF16 (output)
    void forward(const __nv_bfloat16* Q, const __nv_bfloat16* K, const __nv_bfloat16* V,
                 __nv_bfloat16* O,
                 int B, int H, int S_q, int S_kv, int HD,
                 float attn_scale,
                 cudaStream_t stream);

private:
    cudnnHandle_t handle_ = nullptr;

    // Cached graph for a specific (B, H, S_q, S_kv, HD) configuration
    struct CachedPlan;
    CachedPlan* get_plan(int B, int H, int S_q, int S_kv, int HD, float attn_scale);

    // Cache of compiled plans keyed by dimensions
    struct PlanCache;
    PlanCache* cache_ = nullptr;

    // Workspace buffer (high-water-mark)
    void* workspace_ = nullptr;
    size_t workspace_bytes_ = 0;
    void ensure_workspace(size_t needed);
};
