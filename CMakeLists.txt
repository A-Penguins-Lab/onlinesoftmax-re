cmake_minimum_required(VERSION 3.18)
project(OSoftmax LANGUAGES CXX CUDA)

set(CMAKE_CXX_STANDARD 14)
set(CMAKE_CUDA_STANDARD 14)

# Find CUDA
find_package(CUDAToolkit REQUIRED)

add_executable(onlinesoftmax-re
    src/main.cpp
    src/host_functions.cpp
    src/kernels/softmax.cu
    stc/kernels/onlinesoftmax.cu
)

target_include_directories(onlinesoftmax-re PRIVATE include)

# Link CUDA runtime to make cuda_runtime.h available
target_link_libraries(onlinesoftmax-re PRIVATE CUDA::cudart)

# Set CUDA architectures (adjust for your GPU)
set_target_properties(onlinesoftmax-re PROPERTIES CUDA_ARCHITECTURES "75;86")
