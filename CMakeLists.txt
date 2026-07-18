cmake_minimum_required(VERSION 3.28)

# Target this machine only: RTX 5070, Blackwell GB205, compute capability 12.0.
# Set before project() so enable_language(CUDA) picks it up; project() would
# otherwise seed a default (sm_75 here) that a later set() cannot dislodge.
if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)
    set(CMAKE_CUDA_ARCHITECTURES 120)
endif()

# C and CXX are enabled alongside CUDA because the math library wrappers and
# some tests are plain host translation units.
project(cuda_kernel_lab
    VERSION 0.1.0
    DESCRIPTION "CUDA kernel library driven from naive to compute bound tensor core GEMM"
    LANGUAGES CXX CUDA)

# ---------------------------------------------------------------------------
# Language standards
# ---------------------------------------------------------------------------
set(CMAKE_CXX_STANDARD 23)
set(CMAKE_CXX_STANDARD_REQUIRED ON)
set(CMAKE_CXX_EXTENSIONS OFF)

# Device code standard: the newest the installed nvcc accepts. Probed at
# Phase 0; recorded in PROGRESS.md. CUDA 13.3 accepts C++20 for device code.
set(CMAKE_CUDA_STANDARD 20)
set(CMAKE_CUDA_STANDARD_REQUIRED ON)

set(CMAKE_CUDA_SEPARABLE_COMPILATION ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(NOT CMAKE_BUILD_TYPE AND NOT CMAKE_CONFIGURATION_TYPES)
    set(CMAKE_BUILD_TYPE Release CACHE STRING "" FORCE)
endif()

list(APPEND CMAKE_MODULE_PATH "${CMAKE_SOURCE_DIR}/cmake")
include(CompilerWarnings)

# Generate line info in Release so ncu can attribute stalls to source lines
# during the diagnostic rounds without paying the -G debug penalty.
string(APPEND CMAKE_CUDA_FLAGS " -lineinfo")

# ---------------------------------------------------------------------------
# Options
# ---------------------------------------------------------------------------
option(CKL_BUILD_TESTS "Build correctness tests (need a GPU to run)" ON)
option(CKL_BUILD_BENCH "Build benchmark binaries (need a GPU to run)" ON)

# ---------------------------------------------------------------------------
# Header-only interface for the ckl public headers
# ---------------------------------------------------------------------------
add_library(ckl_headers INTERFACE)
target_include_directories(ckl_headers INTERFACE "${CMAKE_SOURCE_DIR}/include")
target_link_libraries(ckl_headers INTERFACE CUDA::cudart)

find_package(CUDAToolkit REQUIRED)

# ---------------------------------------------------------------------------
# Phase 0 tool: device probe
# ---------------------------------------------------------------------------
add_executable(device_probe src/tools/device_probe.cu)
target_link_libraries(device_probe PRIVATE ckl_headers)
ckl_set_warnings(device_probe)

# ---------------------------------------------------------------------------
# Subsequent phases attach their libraries and tests here as they are built.
# The GEMM library, family libraries, tests, and benchmarks are added by
# include()d fragments so this top file stays readable. Guarded so a partial
# tree still configures.
# ---------------------------------------------------------------------------
if(EXISTS "${CMAKE_SOURCE_DIR}/src/gemm/CMakeLists.txt")
    add_subdirectory(src/gemm)
endif()

if(CKL_BUILD_TESTS AND EXISTS "${CMAKE_SOURCE_DIR}/tests/CMakeLists.txt")
    enable_testing()
    add_subdirectory(tests)
endif()

if(CKL_BUILD_BENCH AND EXISTS "${CMAKE_SOURCE_DIR}/benchmarks/CMakeLists.txt")
    add_subdirectory(benchmarks)
endif()
