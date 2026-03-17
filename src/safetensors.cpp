#include "safetensors.h"
#include "cuda_utils.cuh"
#include <json.hpp>

#include <sys/mman.h>
#include <sys/stat.h>
#include <fcntl.h>
#include <unistd.h>
#include <cstring>
#include <stdexcept>
#include <algorithm>

using json = nlohmann::json;

static DType parse_dtype(const std::string& s) {
    if (s == "BF16")   return DType::BF16;
    if (s == "F32")    return DType::F32;
    if (s == "F16")    return DType::BF16;  // treat F16 as BF16 for loading (convert if needed)
    throw std::runtime_error("unsupported dtype in safetensors: " + s);
}

SafeTensorsFile::SafeTensorsFile(const std::string& path) : path_(path) {
    // Open file
    fd_ = open(path.c_str(), O_RDONLY);
    if (fd_ < 0) throw std::runtime_error("cannot open: " + path);

    // Get file size
    struct stat st;
    if (fstat(fd_, &st) < 0) {
        close(fd_);
        throw std::runtime_error("cannot stat: " + path);
    }
    file_size_ = st.st_size;

    // Memory-map the entire file
    mmap_ptr_ = mmap(nullptr, file_size_, PROT_READ, MAP_PRIVATE, fd_, 0);
    if (mmap_ptr_ == MAP_FAILED) {
        close(fd_);
        throw std::runtime_error("mmap failed: " + path);
    }

    // Parse 8-byte header size (little-endian uint64)
    const uint8_t* base = static_cast<const uint8_t*>(mmap_ptr_);
    uint64_t hdr_sz = 0;
    memcpy(&hdr_sz, base, 8);
    header_size_ = hdr_sz;

    // Parse JSON header
    const char* hdr_start = reinterpret_cast<const char*>(base + 8);
    json hdr = json::parse(hdr_start, hdr_start + header_size_);

    // Data section starts after 8 + header_size bytes
    data_base_ = reinterpret_cast<const char*>(base + 8 + header_size_);

    // Extract tensor metadata
    for (auto& [key, val] : hdr.items()) {
        if (key == "__metadata__") continue;

        SafeTensorInfo info;
        info.name = key;
        info.dtype = parse_dtype(val.at("dtype").get<std::string>());

        for (auto& s : val.at("shape")) {
            info.shape.push_back(s.get<int64_t>());
        }

        auto offsets = val.at("data_offsets");
        info.offset_start = offsets[0].get<size_t>();
        info.offset_end   = offsets[1].get<size_t>();

        tensors_[key] = std::move(info);
    }

    fprintf(stderr, "[safetensors] loaded %s: %zu tensors\n", path.c_str(), tensors_.size());
}

SafeTensorsFile::~SafeTensorsFile() {
    if (mmap_ptr_ && mmap_ptr_ != MAP_FAILED) {
        munmap(mmap_ptr_, file_size_);
    }
    if (fd_ >= 0) {
        close(fd_);
    }
}

bool SafeTensorsFile::has(const std::string& name) const {
    return tensors_.count(name) > 0;
}

const SafeTensorInfo& SafeTensorsFile::info(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end())
        throw std::runtime_error("tensor not found: " + name);
    return it->second;
}

std::vector<std::string> SafeTensorsFile::keys() const {
    std::vector<std::string> result;
    result.reserve(tensors_.size());
    for (auto& [k, _] : tensors_) {
        result.push_back(k);
    }
    std::sort(result.begin(), result.end());
    return result;
}

Tensor SafeTensorsFile::load(const std::string& name) const {
    auto it = tensors_.find(name);
    if (it == tensors_.end())
        throw std::runtime_error("tensor not found in " + path_ + ": " + name);

    const auto& ti = it->second;
    size_t bytes = ti.offset_end - ti.offset_start;

    Tensor t(ti.shape, ti.dtype);
    assert(t.size_bytes() == bytes);
    t.copy_from_host(data_base_ + ti.offset_start, bytes);
    return t;
}
