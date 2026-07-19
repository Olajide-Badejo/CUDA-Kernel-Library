# TRSM family library.

add_library(ckl_trsm STATIC
    trsm_naive.cu
    trsm_blocked.cu
    cublas_trsm.cpp
)

target_link_libraries(ckl_trsm
    PUBLIC ckl_headers
    PRIVATE CUDA::cublas
)

set_target_properties(ckl_trsm PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

ckl_set_warnings(ckl_trsm)
