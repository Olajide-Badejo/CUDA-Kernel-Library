#pragma once

// The plan's execution entry point, shared between ckl_sparse and ckl_api.
//
// ckl::spmv lives in src/api/dispatch.cpp because what it does is dispatch:
// validate, resolve kAuto, write chosen, and turn a thrown ckl::Error into a
// Status with a thread local message. What it dispatches to lives here, because
// running a variant needs the plan's private state: the descriptors, the two
// workspaces, the converted SELL and BSR copies and the merge carry scratch.
//
// This header is not installed. It is an implementation detail of the split
// between the two libraries, not part of the public API.

#include <cuda_runtime.h>

#include "ckl/ckl_export.h"
#include "ckl/sparse.hpp"

namespace ckl {
namespace detail {

/**
 * @brief Runs one already resolved SpMV variant out of a plan.
 * @param plan Plan holding the matrix and every cached object.
 * @param algo The path to run; never SpmvAlgo::kAuto, which the caller resolves.
 * @param alpha Host scale on the product.
 * @param x Device pointer to x, length n.
 * @param beta Host scale on the incoming y.
 * @param y Device pointer to y, length m.
 * @param stream Stream to enqueue on.
 * @throws ckl::Error when the plan has no state for the requested variant or a
 *         launch fails. It never returns a status; the caller maps the throw.
 */
CKL_EXPORT void spmv_launch(SpmvPlan& plan, SpmvAlgo algo, float alpha, const float* x, float beta,
                            float* y, cudaStream_t stream);

}  // namespace detail
}  // namespace ckl
