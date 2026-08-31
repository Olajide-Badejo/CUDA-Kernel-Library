# CSR SpMV family library.

add_library(ckl_sparse
    spmv_csr_naive.cu
    spmv_csr_warp.cu
    cusparse_ref.cpp
)

target_link_libraries(ckl_sparse
    PUBLIC ckl_headers
    PRIVATE ckl_core CUDA::cusparse
)

ckl_configure_library(ckl_sparse sparse)
