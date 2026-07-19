// cuSPARSE cusparseSpMV oracle and baseline for CSR SpMV (FP32). Uses the generic
// API: a CSR matrix descriptor and dense vector descriptors, a workspace sized by
// cusparseSpMV_bufferSize. The handle is cached; the descriptors and workspace are
// created per call, which is fine for the tests and the modest SpMV benchmark.

#include <stdexcept>
#include <string>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "ckl/cuda_check.hpp"
#include "ckl/sparse.hpp"

namespace ckl {

namespace {

void check(cusparseStatus_t s, const char* expr) {
    if (s != CUSPARSE_STATUS_SUCCESS) {
        throw std::runtime_error(std::string("cuSPARSE error ") + cusparseGetErrorString(s) + ": " +
                                 expr);
    }
}

cusparseHandle_t handle() {
    static cusparseHandle_t h = [] {
        cusparseHandle_t created = nullptr;
        check(cusparseCreate(&created), "cusparseCreate");
        return created;
    }();
    return h;
}

}  // namespace

void spmv_cusparse(const int* row_ptr, const int* col_idx, const float* values, const float* x,
                   float* y, int m, int n, int nnz, float alpha, float beta, cudaStream_t stream) {
    if (m <= 0) {
        return;
    }
    cusparseHandle_t h = handle();
    check(cusparseSetStream(h, stream), "cusparseSetStream");

    cusparseSpMatDescr_t mat = nullptr;
    cusparseDnVecDescr_t vec_x = nullptr;
    cusparseDnVecDescr_t vec_y = nullptr;
    // cuSPARSE takes non const pointers; SpMV does not modify the inputs.
    check(cusparseCreateCsr(&mat, m, n, nnz, const_cast<int*>(row_ptr), const_cast<int*>(col_idx),
                            const_cast<float*>(values), CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I,
                            CUSPARSE_INDEX_BASE_ZERO, CUDA_R_32F),
          "cusparseCreateCsr");
    check(cusparseCreateDnVec(&vec_x, n, const_cast<float*>(x), CUDA_R_32F),
          "cusparseCreateDnVec x");
    check(cusparseCreateDnVec(&vec_y, m, y, CUDA_R_32F), "cusparseCreateDnVec y");

    std::size_t buffer_size = 0;
    check(cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vec_x, &beta,
                                  vec_y, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, &buffer_size),
          "cusparseSpMV_bufferSize");

    void* buffer = nullptr;
    if (buffer_size > 0) {
        CKL_CUDA_CHECK(cudaMalloc(&buffer, buffer_size));
    }
    check(cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vec_x, &beta, vec_y,
                       CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer),
          "cusparseSpMV");
    if (buffer != nullptr) {
        cudaFree(buffer);
    }
    cusparseDestroyDnVec(vec_y);
    cusparseDestroyDnVec(vec_x);
    cusparseDestroySpMat(mat);
}

}  // namespace ckl
