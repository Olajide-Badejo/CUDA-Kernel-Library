# GEMV family library.

add_library(ckl_gemv STATIC
    gemv_naive.cu
    gemv_warp.cu
    gemv_vectorized.cu
    cublas_gemv.cpp
)

target_link_libraries(ckl_gemv
    PUBLIC ckl_headers
    PRIVATE CUDA::cublas
)

set_target_properties(ckl_gemv PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

ckl_set_warnings(ckl_gemv)
