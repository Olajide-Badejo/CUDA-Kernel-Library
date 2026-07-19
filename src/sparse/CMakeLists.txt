# CSR SpMV family library.

add_library(ckl_sparse STATIC
    spmv_csr_naive.cu
    spmv_csr_warp.cu
    cusparse_ref.cpp
)

target_link_libraries(ckl_sparse
    PUBLIC ckl_headers
    PRIVATE CUDA::cusparse
)

set_target_properties(ckl_sparse PROPERTIES
    CUDA_SEPARABLE_COMPILATION ON
    POSITION_INDEPENDENT_CODE ON
)

ckl_set_warnings(ckl_sparse)
