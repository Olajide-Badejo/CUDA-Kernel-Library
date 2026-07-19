# The GEMM library. Each ladder rung adds its translation unit here as it is
# built. The cuBLAS oracle lives alongside the kernels so tests and benchmarks
# link one target.

add_library(ckl_gemm STATIC
    gemm_naive.cu
    gemm_tiled.cu
    gemm_register.cu
    gemm_cp_async.cu
    gemm_wmma_fp16.cu
    gemm_wmma_bf16.cu
    gemm_mma_ptx.cu
    gemm_mma_ldmatrix.cu
    gemm_mma_opt.cu
    cublas_gemm.cpp
    cublas_gemm_ex.cpp
)

target_link_libraries(ckl_gemm
    PUBLIC ckl_headers
    PRIVATE CUDA::cublas
)

set_target_properties(ckl_gemm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

ckl_set_warnings(ckl_gemm)
