# TRSM family library.

add_library(ckl_trsm
    trsm_naive.cu
    trsm_blocked.cu
    cublas_trsm.cpp
)

target_link_libraries(ckl_trsm
    PUBLIC ckl_headers
    PRIVATE ckl_core CUDA::cublas
)

ckl_configure_library(ckl_trsm trsm)
