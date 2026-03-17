#include "tensor.h"
#include "cuda_utils.cuh"
#include <numeric>
#include <sstream>
#include <cstring>

Tensor::Tensor(std::vector<int64_t> shape, DType dtype)
    : shape_(std::move(shape)), dtype_(dtype), owned_(true) {
    size_t bytes = numel() * dtype_size(dtype_);
    if (bytes > 0) {
        CUDA_CHECK(cudaMalloc(&data_, bytes));
    }
}

Tensor::Tensor(void* data, std::vector<int64_t> shape, DType dtype, bool owned)
    : data_(data), shape_(std::move(shape)), dtype_(dtype), owned_(owned) {}

Tensor::~Tensor() {
    if (owned_ && data_) {
        cudaFree(data_);
    }
}

Tensor::Tensor(Tensor&& o) noexcept
    : data_(o.data_), shape_(std::move(o.shape_)), dtype_(o.dtype_), owned_(o.owned_) {
    o.data_ = nullptr;
    o.owned_ = false;
}

Tensor& Tensor::operator=(Tensor&& o) noexcept {
    if (this != &o) {
        if (owned_ && data_) cudaFree(data_);
        data_  = o.data_;
        shape_ = std::move(o.shape_);
        dtype_ = o.dtype_;
        owned_ = o.owned_;
        o.data_ = nullptr;
        o.owned_ = false;
    }
    return *this;
}

int64_t Tensor::numel() const {
    if (shape_.empty()) return 0;
    return std::accumulate(shape_.begin(), shape_.end(), (int64_t)1, std::multiplies<>());
}

void Tensor::copy_from_host(const void* src, size_t bytes) {
    assert(bytes <= size_bytes());
    CUDA_CHECK(cudaMemcpy(data_, src, bytes, cudaMemcpyHostToDevice));
}

void Tensor::copy_to_host(void* dst, size_t bytes) const {
    assert(bytes <= size_bytes());
    CUDA_CHECK(cudaMemcpy(dst, data_, bytes, cudaMemcpyDeviceToHost));
}

void Tensor::reshape(std::vector<int64_t> new_shape) {
    int64_t old_n = numel();
    int64_t new_n = std::accumulate(new_shape.begin(), new_shape.end(), (int64_t)1, std::multiplies<>());
    assert(old_n == new_n && "reshape must preserve numel");
    shape_ = std::move(new_shape);
}

std::string Tensor::desc() const {
    std::ostringstream ss;
    ss << dtype_name(dtype_) << "[";
    for (size_t i = 0; i < shape_.size(); i++) {
        if (i) ss << ",";
        ss << shape_[i];
    }
    ss << "]";
    return ss.str();
}

Tensor zeros(std::vector<int64_t> shape, DType dtype) {
    Tensor t(shape, dtype);
    CUDA_CHECK(cudaMemset(t.data_ptr(), 0, t.size_bytes()));
    return t;
}

Tensor full(std::vector<int64_t> shape, float val, DType dtype) {
    Tensor t(shape, dtype);
    int64_t n = t.numel();
    if (dtype == DType::F32) {
        std::vector<float> host(n, val);
        t.copy_from_host(host.data(), n * sizeof(float));
    } else {
        __nv_bfloat16 bval = __float2bfloat16(val);
        std::vector<__nv_bfloat16> host(n, bval);
        t.copy_from_host(host.data(), n * sizeof(__nv_bfloat16));
    }
    return t;
}
