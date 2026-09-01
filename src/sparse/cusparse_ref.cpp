// The v1 free function face of the cuSPARSE baseline, now a thin layer over
// ckl::SpmvPlan.
//
// v1 created three descriptors, sized the SpMV workspace, cudaMalloc'd it and
// cudaFree'd it on every call, and the benchmark harness timed all of that as if
// it were the vendor's SpMV. That is defect A2 in docs/CORRECTIONS.md. The first
// fix cached the descriptors behind a one entry plan local to this file; the
// plan is now a public type with a histogram, two vendor workspaces and the
// converted layouts, and this file keeps only the cache that lets the old free
// function signature stay source compatible.
//
// A benchmark should not come through here. It should build its own SpmvPlan
// once per matrix and call ckl::spmv, which is what bench_spmv and bench_all do.
// This path exists for a caller that already had the v1 call and wants it to
// keep working, and it says so in the header.

#include <memory>
#include <mutex>

#include <cuda_runtime.h>

#include "ckl/context.hpp"
#include "ckl/sparse.hpp"
#include "ckl/status.hpp"

#include "spmv_launch.hpp"

namespace ckl {

namespace {

// What makes two calls the same matrix: the buffers and the shape. alpha, beta,
// x and y are per call and never enter the plan, so they stay out of the key.
struct PlanKey {
    const int* row_ptr = nullptr;
    const int* col_idx = nullptr;
    const float* values = nullptr;
    int m = 0;
    int n = 0;
    int nnz = 0;

    bool operator==(const PlanKey& o) const {
        return row_ptr == o.row_ptr && col_idx == o.col_idx && values == o.values && m == o.m &&
               n == o.n && nnz == o.nnz;
    }
};

}  // namespace

void spmv_cusparse(const int* row_ptr, const int* col_idx, const float* values, const float* x,
                   float* y, int m, int n, int nnz, float alpha, float beta, cudaStream_t stream) {
    if (m <= 0) {
        return;
    }
    const PlanKey key{row_ptr, col_idx, values, m, n, nnz};

    // One entry, rebuilt when the matrix changes. The cache is process wide, so
    // the whole sequence runs under the default Context's lock, exactly as the
    // other v1 free functions do.
    //
    // The last plan is deliberately released at process exit rather than at
    // destruction time: it lives in a function local static that outlives
    // nothing, and destroying cuSPARSE descriptors after the CUDA context has
    // begun tearing down is worse than holding them.
    static std::unique_ptr<SpmvPlan> cached;
    static PlanKey cached_key;
    const std::lock_guard<std::mutex> lock(detail::default_context_mutex());
    if (!cached || !(cached_key == key)) {
        SpmvCsr a;
        a.row_ptr = row_ptr;
        a.col_idx = col_idx;
        a.values = values;
        a.m = m;
        a.n = n;
        a.nnz = nnz;
        SpmvPlanOptions opt;
        // Only the vendor path runs through here, so the two conversions would
        // be pure cost.
        opt.build_sell = false;
        opt.build_bsr = false;
        // Built before the old one is released, so a failed rebuild leaves the
        // previous plan usable.
        auto fresh = std::make_unique<SpmvPlan>(a, opt);
        cached = std::move(fresh);
        cached_key = key;
    }
    // The plan's launcher rather than ckl::spmv: ckl::spmv is the dispatcher and
    // lives in ckl_api, which links this library, so calling it from here would
    // close a cycle in the link graph. The v1 contract was a throw on failure
    // and this keeps it.
    detail::spmv_launch(*cached, SpmvAlgo::kCusparseDefault, alpha, x, beta, y, stream);
}

}  // namespace ckl
