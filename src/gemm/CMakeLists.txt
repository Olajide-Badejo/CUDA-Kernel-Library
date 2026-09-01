# The GEMM library. Each ladder rung adds its translation unit here as it is
# built. The cuBLAS oracle lives alongside the kernels so tests and benchmarks
# link one target. No explicit STATIC: BUILD_SHARED_LIBS decides.

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
    cublas_gemm.cpp
    cublas_gemm_ex.cpp
)

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
