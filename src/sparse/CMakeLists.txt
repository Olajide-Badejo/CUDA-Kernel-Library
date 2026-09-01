# CSR SpMV family library.
#
# spmv_plan.cpp is host only: it owns the cuSPARSE descriptors, both vendor
# workspaces and the two host side conversions, and it calls the launchers in
# the .cu files rather than containing a kernel of its own.

add_library(ckl_sparse
    spmv_csr_naive.cu
    spmv_csr_warp.cu
    spmv_csr_vector.cu
    spmv_merge.cu
    spmv_sell_c_sigma.cu
    spmv_bsr.cu
    spmv_plan.cpp
    cusparse_ref.cpp
)

target_link_libraries(ckl_sparse
    PUBLIC ckl_headers
    PRIVATE ckl_core CUDA::cusparse
)

ckl_configure_library(ckl_sparse sparse)
