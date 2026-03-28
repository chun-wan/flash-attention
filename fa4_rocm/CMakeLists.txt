cmake_minimum_required(VERSION 3.20)
project(fa4_rocm LANGUAGES CXX HIP)

set(CMAKE_CXX_STANDARD 17)
set(CMAKE_CXX_STANDARD_REQUIRED ON)

# ROCm paths
list(APPEND CMAKE_PREFIX_PATH "/opt/rocm")
find_package(hip REQUIRED)

# GPU architecture
set(FA4_GPU_ARCH "gfx942" CACHE STRING "Target GPU architecture")

# Tile sizes (compile-time constants)
set(FA4_HEAD_DIM 128 CACHE STRING "Head dimension")
set(FA4_BLOCK_M 64 CACHE STRING "Query tile M dimension")
set(FA4_BLOCK_N 64 CACHE STRING "Key/Value tile N dimension")

# Kernel library
add_library(fa4_kernels SHARED
    kernels/flash_fwd_gfx942.hip
)

target_compile_options(fa4_kernels PRIVATE
    -O3
    --offload-arch=${FA4_GPU_ARCH}
    -DHEAD_DIM=${FA4_HEAD_DIM}
    -DBLOCK_M=${FA4_BLOCK_M}
    -DBLOCK_N=${FA4_BLOCK_N}
)

target_include_directories(fa4_kernels PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/kernels
    ${hip_INCLUDE_DIRS}
)

target_link_libraries(fa4_kernels PRIVATE
    hip::device
    hip::host
)

# Install
install(TARGETS fa4_kernels DESTINATION lib)
install(FILES
    kernels/flash_common.h
    kernels/online_softmax.h
    kernels/flash_fwd_gfx942.hip
    DESTINATION include/fa4_rocm
)
