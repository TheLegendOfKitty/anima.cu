#pragma once

#include <cuda_runtime.h>
#include <cuda_bf16.h>
#include <vector>
#include <cstddef>
#include <cstdint>
#include <cassert>
#include <string>

enum class DType : uint8_t {
    BF16,
    F32,
};

inline size_t dtype_size(DType dt) {
    switch (dt) {
        case DType::BF16: return 2;
        case DType::F32:  return 4;
    }
    return 0;
}

inline const char* dtype_name(DType dt) {
    switch (dt) {
        case DType::BF16: return "bf16";
        case DType::F32:  return "f32";
    }
    return "?";
}

// GPU tensor with owned device memory.
// Move-only (no copies — avoids accidental double-free).
class Tensor {
public:
    Tensor() = default;

    // Allocate a new tensor on GPU.
    Tensor(std::vector<int64_t> shape, DType dtype);

    // Take ownership of an existing device pointer.
    Tensor(void* data, std::vector<int64_t> shape, DType dtype, bool owned = true);

    ~Tensor();

    // Move semantics
    Tensor(Tensor&& o) noexcept;
    Tensor& operator=(Tensor&& o) noexcept;

    // No copies
    Tensor(const Tensor&) = delete;
    Tensor& operator=(const Tensor&) = delete;

    // ---- Accessors ----
    void*           data_ptr()       { return data_; }
    const void*     data_ptr() const { return data_; }

    __nv_bfloat16*  bf16_ptr()       { assert(dtype_ == DType::BF16); return static_cast<__nv_bfloat16*>(data_); }
    float*          f32_ptr()        { assert(dtype_ == DType::F32);  return static_cast<float*>(data_); }
    const __nv_bfloat16* bf16_ptr() const { assert(dtype_ == DType::BF16); return static_cast<const __nv_bfloat16*>(data_); }
    const float*         f32_ptr()  const { assert(dtype_ == DType::F32);  return static_cast<const float*>(data_); }

    DType           dtype()  const { return dtype_; }
    int             ndim()   const { return (int)shape_.size(); }
    int64_t         dim(int i) const { return shape_[i]; }
    const std::vector<int64_t>& shape() const { return shape_; }

    int64_t numel() const;
    size_t  size_bytes() const { return numel() * dtype_size(dtype_); }
    bool    empty() const { return data_ == nullptr; }

    // Upload from host to this tensor (must be pre-allocated).
    void copy_from_host(const void* src, size_t bytes);

    // Download to host.
    void copy_to_host(void* dst, size_t bytes) const;

    // Reshape in-place (must preserve numel).
    void reshape(std::vector<int64_t> new_shape);

    // Debug: short description
    std::string desc() const;

private:
    void*               data_  = nullptr;
    std::vector<int64_t> shape_;
    DType               dtype_ = DType::BF16;
    bool                owned_ = false;
};

// Create a zero-filled tensor.
Tensor zeros(std::vector<int64_t> shape, DType dtype);

// Create a tensor filled with a constant.
Tensor full(std::vector<int64_t> shape, float val, DType dtype);
