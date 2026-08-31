// cuSPARSE cusparseSpMV oracle and baseline for CSR SpMV (FP32). Uses the generic
// API: a CSR matrix descriptor and dense vector descriptors, a workspace sized by
// cusparseSpMV_bufferSize.
//
// Setup is cached behind a one entry plan. The v1 version created all three
// descriptors, sized the workspace, cudaMalloc'd it and cudaFree'd it on every
// call, and the benchmark harness timed all of that as if it were the vendor's
// SpMV. Every "percent of cuSPARSE" figure produced that way flattered my own
// kernels; that is defect A2 in docs/CORRECTIONS.md. Here the descriptors and the
// workspace are built once for a given set of pointers and dimensions, and a
// repeated call on the same problem enqueues only cusparseSetStream plus
// cusparseSpMV. One entry is enough: the harness times one shape at a time and
// the plan rebuilds when the key changes.

#include <cstddef>
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

// What makes two calls the same problem: the buffers cuSPARSE was handed and the
// shape it was told they hold. alpha and beta are passed per call and never enter
// the descriptors, so they stay out of the key.
struct PlanKey {
    const int* row_ptr = nullptr;
    const int* col_idx = nullptr;
    const float* values = nullptr;
    const float* x = nullptr;
    const float* y = nullptr;
    int m = 0;
    int n = 0;
    int nnz = 0;

    bool operator==(const PlanKey& o) const {
        return row_ptr == o.row_ptr && col_idx == o.col_idx && values == o.values && x == o.x &&
               y == o.y && m == o.m && n == o.n && nnz == o.nnz;
    }
};

// Owns the three descriptors and the workspace. The destructor releases whatever
// was built, so a failure part way through construction leaks nothing.
class SpmvPlan {
public:
    SpmvPlan(cusparseHandle_t h, const PlanKey& key, float alpha, float beta) : key_(key) {
        try {
            build(h, alpha, beta);
        } catch (...) {
            release();
            throw;
        }
    }

    ~SpmvPlan() { release(); }

    SpmvPlan(const SpmvPlan&) = delete;
    SpmvPlan& operator=(const SpmvPlan&) = delete;

    const PlanKey& key() const { return key_; }

    // The only work a repeated timed call does.
    void run(cusparseHandle_t h, float alpha, float beta, cudaStream_t stream) const {
        check(cusparseSetStream(h, stream), "cusparseSetStream");
        check(cusparseSpMV(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_, vec_x_, &beta, vec_y_,
                           CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT, buffer_),
              "cusparseSpMV");
    }

private:
    void build(cusparseHandle_t h, float alpha, float beta) {
        // cuSPARSE takes non const pointers; SpMV does not modify the inputs.
        check(cusparseCreateCsr(&mat_, key_.m, key_.n, key_.nnz, const_cast<int*>(key_.row_ptr),
                                const_cast<int*>(key_.col_idx), const_cast<float*>(key_.values),
                                CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                                CUDA_R_32F),
              "cusparseCreateCsr");
        check(cusparseCreateDnVec(&vec_x_, key_.n, const_cast<float*>(key_.x), CUDA_R_32F),
              "cusparseCreateDnVec x");
        check(cusparseCreateDnVec(&vec_y_, key_.m, const_cast<float*>(key_.y), CUDA_R_32F),
              "cusparseCreateDnVec y");

        std::size_t buffer_size = 0;
        check(cusparseSpMV_bufferSize(h, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat_, vec_x_,
                                      &beta, vec_y_, CUDA_R_32F, CUSPARSE_SPMV_ALG_DEFAULT,
                                      &buffer_size),
              "cusparseSpMV_bufferSize");
        if (buffer_size > 0) {
            CKL_CUDA_CHECK(cudaMalloc(&buffer_, buffer_size));
        }
    }

    void release() {
        if (buffer_ != nullptr) {
            cudaFree(buffer_);
            buffer_ = nullptr;
        }
        if (vec_y_ != nullptr) {
            cusparseDestroyDnVec(vec_y_);
            vec_y_ = nullptr;
        }
        if (vec_x_ != nullptr) {
            cusparseDestroyDnVec(vec_x_);
            vec_x_ = nullptr;
        }
        if (mat_ != nullptr) {
            cusparseDestroySpMat(mat_);
            mat_ = nullptr;
        }
    }

    PlanKey key_;
    cusparseSpMatDescr_t mat_ = nullptr;
    cusparseDnVecDescr_t vec_x_ = nullptr;
    cusparseDnVecDescr_t vec_y_ = nullptr;
    void* buffer_ = nullptr;
};

}  // namespace

void spmv_cusparse(const int* row_ptr, const int* col_idx, const float* values, const float* x,
                   float* y, int m, int n, int nnz, float alpha, float beta, cudaStream_t stream) {
    if (m <= 0) {
        return;
    }
    cusparseHandle_t h = handle();

    const PlanKey key{row_ptr, col_idx, values, x, y, m, n, nnz};
    // One entry, rebuilt when the problem changes. Deliberately never destroyed
    // at process exit: the cuSPARSE handle above is a leaked static too, and
    // releasing descriptors after the CUDA context has been torn down is worse
    // than not releasing them. The old plan is destroyed only once the new one
    // exists, so a failed rebuild leaves the previous plan usable.
    static SpmvPlan* cached = nullptr;
    if (cached == nullptr || !(cached->key() == key)) {
        SpmvPlan* fresh = new SpmvPlan(h, key, alpha, beta);
        delete cached;
        cached = fresh;
    }
    cached->run(h, alpha, beta, stream);
}

}  // namespace ckl
