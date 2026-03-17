#pragma once

#include "tensor.h"
#include <string>
#include <unordered_map>
#include <vector>
#include <cstdint>

// Metadata for a single tensor inside a safetensors file.
struct SafeTensorInfo {
    std::string     name;
    DType           dtype;
    std::vector<int64_t> shape;
    size_t          offset_start;   // byte offset into data section
    size_t          offset_end;
};

// Memory-mapped safetensors file loader.
// Usage:
//   SafeTensorsFile f("model.safetensors");
//   Tensor w = f.load("model.diffusion_model.transformer_blocks.0.attn1.to_q.weight");
class SafeTensorsFile {
public:
    explicit SafeTensorsFile(const std::string& path);
    ~SafeTensorsFile();

    // No copies
    SafeTensorsFile(const SafeTensorsFile&) = delete;
    SafeTensorsFile& operator=(const SafeTensorsFile&) = delete;

    // Load a single tensor to GPU.
    Tensor load(const std::string& name) const;

    // Check if a tensor exists.
    bool has(const std::string& name) const;

    // List all tensor names.
    std::vector<std::string> keys() const;

    // Get info for a tensor.
    const SafeTensorInfo& info(const std::string& name) const;

    // Number of tensors.
    size_t count() const { return tensors_.size(); }

private:
    std::string path_;
    int         fd_       = -1;
    void*       mmap_ptr_ = nullptr;
    size_t      file_size_ = 0;
    size_t      header_size_ = 0;
    const char* data_base_ = nullptr;  // start of tensor data (after header)

    std::unordered_map<std::string, SafeTensorInfo> tensors_;
};
