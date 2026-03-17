#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <cstdio>
#include <cstdlib>

// ---- Error checking (usable from .cpp and .cu) ----

#define CUDA_CHECK(call)                                                       \
    do {                                                                        \
        cudaError_t err = (call);                                               \
        if (err != cudaSuccess) {                                               \
            fprintf(stderr, "CUDA error at %s:%d: %s\n", __FILE__, __LINE__,    \
                    cudaGetErrorString(err));                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

#define CUBLAS_CHECK(call)                                                     \
    do {                                                                        \
        cublasStatus_t st = (call);                                             \
        if (st != CUBLAS_STATUS_SUCCESS) {                                      \
            fprintf(stderr, "cuBLAS error at %s:%d: %d\n", __FILE__, __LINE__,  \
                    (int)st);                                                    \
            exit(1);                                                            \
        }                                                                       \
    } while (0)

// ---- Launch helpers (usable from .cpp and .cu) ----

inline int ceil_div(int a, int b) { return (a + b - 1) / b; }

// ---- Device-only helpers (only compiled in .cu files) ----

#ifdef __CUDACC__

__device__ __forceinline__ float bf16_to_float(__nv_bfloat16 x) {
    return __bfloat162float(x);
}

__device__ __forceinline__ __nv_bfloat16 float_to_bf16(float x) {
    return __float2bfloat16(x);
}

__device__ __forceinline__ float warp_reduce_sum(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val += __shfl_down_sync(0xffffffff, val, offset);
    return val;
}

__device__ __forceinline__ float warp_reduce_max(float val) {
    for (int offset = 16; offset > 0; offset >>= 1)
        val = fmaxf(val, __shfl_down_sync(0xffffffff, val, offset));
    return val;
}

#endif // __CUDACC__
