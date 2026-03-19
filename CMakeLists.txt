cmake_minimum_required(VERSION 3.25)
project(anima_cu LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 20)
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_ARCHITECTURES 120)

set(CMAKE_CUDA_FLAGS "${CMAKE_CUDA_FLAGS} -O3")
set(CMAKE_CXX_FLAGS "${CMAKE_CXX_FLAGS} -O3")

# CUDA toolkit
find_package(CUDAToolkit REQUIRED)

# cuDNN
find_library(CUDNN_LIB cudnn HINTS /usr/lib/aarch64-linux-gnu /usr/local/lib)
find_path(CUDNN_INCLUDE cudnn.h HINTS /usr/include/aarch64-linux-gnu /usr/local/include)

# sentencepiece (static build)
set(SENTENCEPIECE_LIB /tmp/sentencepiece/build/src/libsentencepiece.a)
set(SENTENCEPIECE_INCLUDE /tmp/sentencepiece/src)

# Third-party headers
add_library(third_party INTERFACE)
target_include_directories(third_party INTERFACE
    ${CMAKE_SOURCE_DIR}/third_party/nlohmann
    ${CMAKE_SOURCE_DIR}/third_party/stb
)

# Kernel library
add_library(anima_kernels STATIC
    src/kernels/rms_norm.cu
    src/kernels/layer_norm.cu
    src/kernels/gelu.cu
    src/kernels/silu.cu
    src/kernels/rope.cu
    src/kernels/adaln.cu
    src/kernels/elementwise.cu
    src/kernels/softmax.cu
    src/kernels/patchify.cu
    src/kernels/group_norm.cu
    src/kernels/gemma_ops.cu
    src/kernels/vae_kernels.cu
    src/kernels/transpose.cu
    src/kernels/f32_ops.cu
    src/kernels/causal_conv3d.cu
    src/cuda_utils.cu
    src/spectrum.cu
)
target_include_directories(anima_kernels PUBLIC ${CMAKE_SOURCE_DIR}/src ${CUDNN_INCLUDE})
target_link_libraries(anima_kernels PUBLIC CUDA::cudart CUDA::cublas CUDA::cusolver ${CUDNN_LIB})

# Core library
add_library(anima_core STATIC
    src/tensor.cpp
    src/safetensors.cpp
    src/linear.cpp
    src/qwen3.cpp
    src/llm_adapter.cpp
    src/transformer.cpp
    src/scheduler.cpp
    src/pipeline.cpp
    src/vae.cpp
    src/qwen3_tokenizer.cpp
    src/t5_tokenizer.cpp
    src/cudnn_sdpa.cpp
)
target_include_directories(anima_core PUBLIC
    ${CMAKE_SOURCE_DIR}/src
    ${CMAKE_SOURCE_DIR}/third_party
    ${CMAKE_SOURCE_DIR}/third_party/cudnn-frontend/include
    /home/parsa/sglang/.venv/lib/python3.13/site-packages/include
    ${SENTENCEPIECE_INCLUDE}
    ${CUDNN_INCLUDE}
)
target_link_libraries(anima_core PUBLIC
    anima_kernels
    third_party
    CUDA::cublas
    CUDA::cublasLt
    CUDA::curand
    CUDA::cuda_driver
    CUDA::nvrtc
    ${CUDNN_LIB}
    ${SENTENCEPIECE_LIB}
)

# Main executable
add_executable(anima src/main.cpp)
target_link_libraries(anima PRIVATE anima_core)
