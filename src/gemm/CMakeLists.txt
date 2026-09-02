# The GEMM library. Each ladder rung adds its translation unit here as it is
# built. The cuBLAS oracle lives alongside the kernels so tests and benchmarks
# link one target. No explicit STATIC: BUILD_SHARED_LIBS decides.

# ---------------------------------------------------------------------------
# CUTLASS, the reference line on the ladder
# ---------------------------------------------------------------------------
#
# Source fetch and nothing else. CUTLASS is header only for the one translation
# unit that uses it, so no CUTLASS target is configured, nothing of it is
# installed, and no CUTLASS type reaches a public header. SOURCE_SUBDIR names a
# directory that holds no CMakeLists.txt, which is how FetchContent_MakeAvailable
# is told to populate the source and stop there.
#
# The tag is exact, not a branch. docs/cutlass.md argues for this particular
# instantiation by citing what this release's SM120 collective builders
# static_assert on, and a moving pin would quietly invalidate that argument.
#
# GIT_SHALLOW keeps the clone to the one tagged commit. The full history is over
# a gigabyte, and a build tree under /mnt/c pays Windows filesystem latency on
# every object, so the difference is minutes rather than seconds. The shallow
# v4.7.1 tree is about 220 MB. Configure with -DFETCHCONTENT_QUIET=OFF to watch
# the fetch when it feels stuck.
include(FetchContent)

set(CKL_CUTLASS_TAG "v4.7.1" CACHE STRING
    "Exact CUTLASS release tag the reference rung is built against")

FetchContent_Declare(cutlass
    GIT_REPOSITORY https://github.com/NVIDIA/cutlass.git
    GIT_TAG ${CKL_CUTLASS_TAG}
    GIT_SHALLOW TRUE
    GIT_PROGRESS TRUE
    SOURCE_SUBDIR cmake_of_cutlass_is_deliberately_not_built)
FetchContent_MakeAvailable(cutlass)

add_library(ckl_gemm
    gemm_naive.cu
    gemm_tiled.cu
    gemm_register.cu
    gemm_cp_async.cu
    gemm_wmma_fp16.cu
    gemm_wmma_bf16.cu
    gemm_mma_ptx.cu
    gemm_mma_ldmatrix.cu
    gemm_mma_opt.cu
    gemm_tile_family.cu
    gemm_splitk.cu
    gemm_streamk.cu
    gemm_cutlass_ref.cu
    cublas_gemm.cpp
    cublas_gemm_ex.cpp
)

# Everything CUTLASS needs is scoped to the one translation unit that includes
# it, so the headers, the relaxed constexpr rule and the two warning
# suppressions never reach a kernel of mine.
#
#   --expt-relaxed-constexpr  CUTLASS calls constexpr std::get from
#                             __host__ __device__ code in its convolution
#                             problem size headers.
#   -diag-suppress=20012      nvcc reports the __host__ __device__ annotation on
#                             an explicitly defaulted constructor as ignored, in
#                             an SM100 tile scheduler this build never
#                             instantiates.
#   -Wno-unused-parameter     params_universal_base.h takes device_sms and
#                             sm_occupancy in a base hook that the non
#                             persistent kernel does not read.
#
# Source level options are appended after the target level ones, so these three
# override the warning gate here and only here; --Werror=all-warnings still
# applies to this file for every other diagnostic.
set_source_files_properties(gemm_cutlass_ref.cu PROPERTIES
    INCLUDE_DIRECTORIES "${cutlass_SOURCE_DIR}/include"
    COMPILE_OPTIONS "--expt-relaxed-constexpr;-diag-suppress=20012;-Xcompiler=-Wno-unused-parameter")

target_link_libraries(ckl_gemm
    PUBLIC ckl_headers
    PRIVATE ckl_core CUDA::cublas
)

ckl_configure_library(ckl_gemm gemm)

# One export header for the whole package. generate_export_header needs a
# compiled target, so it hangs off this one; every library sets DEFINE_SYMBOL to
# CKL_EXPORTS (see ckl_configure_library) so the same header is correct for all
# six. Part 03 annotates the public declarations with CKL_EXPORT.
generate_export_header(ckl_gemm
    BASE_NAME CKL
    EXPORT_FILE_NAME "${PROJECT_BINARY_DIR}/include/ckl/ckl_export.h")
