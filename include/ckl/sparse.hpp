#pragma once

/**
 * @file sparse.hpp
 * @brief CSR sparse matrix vector product, y = alpha * (A * x) + beta * y.
 *
 * A is stored in compressed sparse row form: row_ptr has length m+1, col_idx and
 * values have length nnz. Six hand written variants plus cuSPARSE under two
 * algorithms as oracle and baseline, all reachable through ckl::SpmvPlan.
 *
 * The plan is the entry point that matters. It is built once per matrix and
 * owns the cuSPARSE descriptors, both workspaces, the nonzero per row
 * histogram, the lane width the histogram implies, and the SELL-C-sigma and BSR
 * conversions. Building it is the expensive part; running a variant through it
 * enqueues kernel launches and nothing else. v1 created three descriptors,
 * sized a workspace, allocated it and freed it inside every call and timed all
 * of that as if it were the vendor's SpMV, which is defect A2 in
 * docs/CORRECTIONS.md. There is no public path in 1.1.0 that can do that again.
 *
 * The v1 free functions stay, unchanged and source compatible, because they are
 * the first two rungs of the ladder and a benchmark row still names them.
 */

#include <cstddef>
#include <memory>

#include <cuda_runtime.h>

#include "ckl/ckl_export.h"
#include "ckl/status.hpp"

namespace ckl {

/**
 * @brief Every SpMV path the library can run, plus the two vendor algorithms.
 *
 * kCusparseDefault is CUSPARSE_SPMV_ALG_DEFAULT and kCusparseAlg2 is
 * CUSPARSE_SPMV_CSR_ALG2, which buys a deterministic reduction order at a cost.
 * Both are measured on every matrix, because quoting a percentage against
 * whichever of the two happened to be slower is a weak claim.
 */
enum class SpmvAlgo {
    kAuto = 0,         ///< Let the plan choose from its histogram and report the choice.
    kCsrNaive,         ///< One thread per row.
    kCsrWarp,          ///< One 32 lane warp per row with a shfl reduction.
    kCsrVector,        ///< A lane group of 2 to 32 per row, width chosen from the histogram.
    kMerge,            ///< Merge path after Merrill and Garland, SC 2016.
    kSellCSigma,       ///< Sliced ELLPACK, C = 32, sigma from the plan.
    kBsr,              ///< Block CSR over the plan's detected block dimension.
    kCusparseDefault,  ///< cusparseSpMV with CUSPARSE_SPMV_ALG_DEFAULT.
    kCusparseAlg2      ///< cusparseSpMV with CUSPARSE_SPMV_CSR_ALG2.
};

/**
 * @brief Symbolic name of an SpMV algorithm.
 * @param a Algorithm to name.
 * @return A static string such as "kMerge", never null.
 */
CKL_EXPORT const char* spmv_algo_name(SpmvAlgo a);

/**
 * @brief A CSR matrix already resident on the device.
 *
 * The plan does not own any of these buffers and never frees them. They have to
 * outlive the plan.
 */
struct CKL_EXPORT SpmvCsr {
    const int* row_ptr = nullptr;   ///< Device pointer to the row offsets, length m+1.
    const int* col_idx = nullptr;   ///< Device pointer to the column indices, length nnz.
    const float* values = nullptr;  ///< Device pointer to the values, length nnz.
    int m = 0;                      ///< Rows of A and length of y.
    int n = 0;                      ///< Columns of A and length of x.
    int nnz = 0;                    ///< Number of stored nonzeros.
};

/**
 * @brief Tuning knobs a caller can set before the plan is built.
 *
 * Every field defaults to the rule the plan would apply on its own, so a caller
 * that wants the library's answer passes the struct unchanged and reads back
 * what it decided through the plan's accessors.
 */
struct CKL_EXPORT SpmvPlanOptions {
    /// Rows per SELL-C-sigma sort window. Zero asks the plan to choose: 256 on a
    /// regular degree distribution and 8192 on a power law one.
    int sigma = 0;
    /// Lane group width for kCsrVector, one of 2, 4, 8, 16 or 32. Zero asks the
    /// plan to choose from the histogram.
    int vector_width = 0;
    /// Square block dimension the BSR conversion uses, one of 2, 4 and 8. Zero
    /// asks the plan to run its detection pass and choose. One disables the
    /// conversion, and kBsr then reports kNotSupported with the reason.
    int block_dim = 0;
    /// Build the SELL-C-sigma copy of the matrix. Off saves the padded arrays on
    /// a plan that will never run that variant.
    bool build_sell = true;
    /// Build the BSR copy of the matrix.
    bool build_bsr = true;
};

/**
 * @brief The per matrix state every SpMV variant runs out of.
 *
 * One plan per matrix, reused across every variant and every repeat. It holds
 * the cuSPARSE handle and descriptors, a workspace per vendor algorithm, the
 * nonzero per row histogram, the SELL-C-sigma permutation and padded arrays,
 * and the BSR conversion. Constructing it copies the row offsets to the host,
 * runs the two conversions and allocates their device buffers; calling
 * ckl::spmv on it does none of that.
 *
 * @note A plan is externally synchronized, like a Context: give each thread its
 *       own, or hold a lock across every call that touches a shared one.
 * @note The plan holds the matrix pointers it was built from. Freeing those
 *       buffers, or writing new values into them, invalidates it.
 */
class CKL_EXPORT SpmvPlan {
public:
    /**
     * @brief Builds the plan for one matrix.
     * @param a The device resident CSR matrix; its buffers must outlive the plan.
     * @param opt Tuning knobs; the default asks the plan to decide everything.
     * @throws ckl::Error when a descriptor, an allocation or a conversion fails.
     */
    explicit SpmvPlan(const SpmvCsr& a, const SpmvPlanOptions& opt = SpmvPlanOptions());

    /// @brief Releases the descriptors, the workspaces and the converted copies.
    ~SpmvPlan();

    /**
     * @brief Move constructor.
     * @param other Plan to move from; it holds nothing afterwards.
     */
    SpmvPlan(SpmvPlan&& other) noexcept;

    /**
     * @brief Move assignment.
     * @param other Plan to move from; it holds nothing afterwards.
     * @return This plan.
     */
    SpmvPlan& operator=(SpmvPlan&& other) noexcept;

    SpmvPlan(const SpmvPlan&) = delete;
    SpmvPlan& operator=(const SpmvPlan&) = delete;

    /**
     * @brief The matrix this plan was built for.
     * @return The descriptor handed to the constructor.
     */
    const SpmvCsr& matrix() const;

    /**
     * @brief Rows holding exactly this many nonzeros.
     * @param bucket Histogram bucket index, 0 to histogram_buckets() - 1.
     * @return Row count in the bucket. Bucket b holds rows whose nonzero count
     *         is in [2^(b-1), 2^b) for b above zero, and exactly zero for b = 0.
     */
    long long histogram(int bucket) const;

    /**
     * @brief Number of histogram buckets.
     * @return 32, which covers every row length an int nonzero index can hold.
     */
    static int histogram_buckets();

    /**
     * @brief Mean nonzeros per row.
     * @return nnz divided by m, or zero for an empty matrix.
     */
    double mean_nnz_per_row() const;

    /**
     * @brief Longest row.
     * @return The largest nonzero count of any row, or zero for an empty matrix.
     */
    int max_nnz_per_row() const;

    /**
     * @brief Whether the plan classified this matrix as power law.
     * @return True when the longest row is at least 32 times the mean, which is
     *         the provisional rule the sigma default and the kAuto choice use.
     */
    bool power_law() const;

    /**
     * @brief The lane group width kCsrVector runs at.
     * @return 2, 4, 8, 16 or 32.
     */
    int vector_width() const;

    /**
     * @brief The SELL-C-sigma sort window.
     * @return Rows per sort window; 1 means no sorting at all.
     */
    int sigma() const;

    /**
     * @brief Slice height of the SELL-C-sigma layout.
     * @return 32, one warp per slice and one lane per row.
     */
    static int sell_slice_height();

    /**
     * @brief Nonzeros the SELL-C-sigma layout stores, padding included.
     * @return nnz_padded, or zero when the conversion was not built.
     * @note A variant that wins ten percent while padding two to one is not a
     *       win, which is why this rides on every SELL benchmark row.
     */
    long long sell_nnz_padded() const;

    /**
     * @brief Padding ratio of the SELL-C-sigma layout.
     * @return nnz_padded over nnz, or zero when the conversion was not built.
     */
    double sell_padding_ratio() const;

    /**
     * @brief Square block dimension the BSR conversion used.
     * @return The block dimension, or zero when the conversion was not built.
     */
    int bsr_block_dim() const;

    /**
     * @brief Nonzeros the BSR layout stores, block padding included.
     * @return block count times block_dim squared, or zero when not built.
     */
    long long bsr_nnz_padded() const;

    /**
     * @brief Host side cost of the BSR block detection pass.
     * @return Wall clock milliseconds the conversion took, or zero when not built.
     * @note Reported separately because the stretch variant only earns its place
     *       if the kernel saves more than the detection costs.
     */
    double bsr_detect_ms() const;

    /**
     * @brief Lower end of the traffic model, in bytes per SpMV.
     * @param beta_nonzero Whether y is read as well as written.
     * @return 4*nnz values plus 4*nnz indices plus 4*(m+1) offsets plus 4*n for a
     *         perfectly L2 resident x plus 4*m for the y write, plus 4*m again
     *         when beta is nonzero.
     */
    long long model_bytes_low(bool beta_nonzero) const;

    /**
     * @brief Upper end of the traffic model, in bytes per SpMV.
     * @param beta_nonzero Whether y is read as well as written.
     * @return The same sum with the x term at 4*nnz, which is what no reuse at
     *         all would cost.
     * @note The truth sits between the two ends and depends on how local the
     *       column indices are, which is exactly what the matrix suite varies.
     *       The model explains a measurement; it does not replace one.
     */
    long long model_bytes_high(bool beta_nonzero) const;

    /**
     * @brief What kAuto would run on this matrix.
     * @return The algorithm the plan's rule picks; it launches nothing.
     */
    SpmvAlgo query() const;

    /// @cond INTERNAL
    // The plan's state, and the accessor ckl::detail::spmv_launch reaches it
    // through. Public because the dispatcher lives in another translation unit
    // in another library; Impl is only ever defined in src/sparse.
    struct Impl;
    Impl& live() const;
    /// @endcond

private:
    std::unique_ptr<Impl> impl_;
};

/**
 * @brief Runs one SpMV variant through a plan: y = alpha * (A * x) + beta * y.
 * @param plan Plan built for the matrix; it supplies A and every cached object.
 * @param algo Path to run, or SpmvAlgo::kAuto to let the plan choose.
 * @param alpha Host scale on the product.
 * @param x Device pointer to x, length n.
 * @param beta Host scale on the incoming y; when zero, y is not read.
 * @param y Device pointer to y, length m, updated in place.
 * @param chosen Optional out-param receiving the path taken; may be null. When
 *        non-null it is written on success and on failure alike.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @return Status::kSuccess, or the failure status; ckl_last_error carries the detail.
 * @note An explicitly named algorithm is never rerouted. Only kAuto chooses, and
 *       chosen reports where it landed.
 * @note Asynchronous. Synchronize on stream before reading y.
 */
CKL_EXPORT Status spmv(SpmvPlan& plan, SpmvAlgo algo, float alpha, const float* x, float beta,
                       float* y, SpmvAlgo* chosen = nullptr, cudaStream_t stream = nullptr);

/**
 * @brief One thread per row.
 * @param row_ptr Device pointer to the CSR row offsets, length m+1.
 * @param col_idx Device pointer to the CSR column indices, length nnz.
 * @param values Device pointer to the CSR values, length nnz.
 * @param x Device pointer to x, length n.
 * @param y Device pointer to y, length m, updated in place.
 * @param m Rows of A and length of y.
 * @param n Columns of A and length of x.
 * @param nnz Number of stored nonzeros.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming y; when zero, y is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Simple, but a long row stalls its whole warp.
 * @note Asynchronous. Synchronize on stream before reading y.
 */
CKL_EXPORT void spmv_csr_naive(const int* row_ptr, const int* col_idx, const float* values,
                               const float* x, float* y, int m, int n, int nnz, float alpha,
                               float beta, cudaStream_t stream = nullptr);

/**
 * @brief One warp per row with a shfl reduction.
 * @param row_ptr Device pointer to the CSR row offsets, length m+1.
 * @param col_idx Device pointer to the CSR column indices, length nnz.
 * @param values Device pointer to the CSR values, length nnz.
 * @param x Device pointer to x, length n.
 * @param y Device pointer to y, length m, updated in place.
 * @param m Rows of A and length of y.
 * @param n Columns of A and length of x.
 * @param nnz Number of stored nonzeros.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming y; when zero, y is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note The row's nonzeros are shared across 32 lanes, so a few long rows do not
 *       serialize.
 */
CKL_EXPORT void spmv_csr_warp(const int* row_ptr, const int* col_idx, const float* values,
                              const float* x, float* y, int m, int n, int nnz, float alpha,
                              float beta, cudaStream_t stream = nullptr);

/**
 * @brief A lane group of the given width per row, with a partial shfl reduction.
 * @param row_ptr Device pointer to the CSR row offsets, length m+1.
 * @param col_idx Device pointer to the CSR column indices, length nnz.
 * @param values Device pointer to the CSR values, length nnz.
 * @param x Device pointer to x, length n.
 * @param y Device pointer to y, length m, updated in place.
 * @param m Rows of A and length of y.
 * @param n Columns of A and length of x.
 * @param nnz Number of stored nonzeros.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming y; when zero, y is not read.
 * @param width Lanes per row: 2, 4, 8, 16 or 32. Anything else is rounded up
 *        into that set.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note A full warp per row idles 25 of its 32 lanes on a matrix averaging 7
 *       nonzeros per row and still runs five reduction steps to combine nothing.
 *       This is that fixed, with the width picked from the plan's histogram.
 */
CKL_EXPORT void spmv_csr_vector(const int* row_ptr, const int* col_idx, const float* values,
                                const float* x, float* y, int m, int n, int nnz, float alpha,
                                float beta, int width, cudaStream_t stream = nullptr);

/**
 * @brief Merge path CSR SpMV after Merrill and Garland, SC 2016.
 * @param row_ptr Device pointer to the CSR row offsets, length m+1.
 * @param col_idx Device pointer to the CSR column indices, length nnz.
 * @param values Device pointer to the CSR values, length nnz.
 * @param x Device pointer to x, length n.
 * @param y Device pointer to y, length m, updated in place.
 * @param m Rows of A and length of y.
 * @param n Columns of A and length of x.
 * @param nnz Number of stored nonzeros.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming y; when zero, y is not read.
 * @param carry_rows Device scratch of at least spmv_merge_carry_count() ints.
 * @param carry_values Device scratch of at least spmv_merge_carry_count() floats.
 * @param threads Threads the work was split across; must match the value the
 *        scratch was sized from.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Every thread does exactly ceil((m + nnz) / threads) merge units whatever
 *       the degree distribution, so one 4700 nonzero row no longer decides how
 *       long the launch takes. A second pass stitches the rows that straddle a
 *       segment boundary.
 * @note The scratch is a plan concern; ckl::spmv passes the plan's own.
 */
CKL_EXPORT void spmv_merge(const int* row_ptr, const int* col_idx, const float* values,
                           const float* x, float* y, int m, int n, int nnz, float alpha, float beta,
                           int* carry_rows, float* carry_values, int threads,
                           cudaStream_t stream = nullptr);

/**
 * @brief Threads the merge path driver splits a matrix across.
 * @param m Rows of A.
 * @param nnz Number of stored nonzeros.
 * @return The thread count, which is also the length the carry scratch needs.
 */
CKL_EXPORT int spmv_merge_carry_count(int m, int nnz);

/**
 * @brief Sliced ELLPACK SpMV, C = 32, one warp per slice and one lane per row.
 * @param slice_ptr Device pointer to the slice offsets, length slices+1.
 * @param col_idx Device pointer to the padded column indices, length nnz_padded.
 * @param values Device pointer to the padded values, length nnz_padded.
 * @param perm Device pointer to the row permutation, length slices*32.
 * @param row_length Device pointer to the true nonzero count per padded row.
 * @param x Device pointer to x, length n.
 * @param y Device pointer to y, length m, updated in place.
 * @param m Rows of A and length of y.
 * @param slices Number of C row slices.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming y; when zero, y is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note The inner loop reads a full 128 byte column of the slice per step with
 *       no row_ptr indirection, which is the traffic both CSR kernels waste
 *       walking across row boundaries.
 * @note The conversion belongs to the plan. This entry point takes the converted
 *       arrays because building them inside the call is what defect A2 was.
 */
CKL_EXPORT void spmv_sell_c_sigma(const int* slice_ptr, const int* col_idx, const float* values,
                                  const int* perm, const int* row_length, const float* x, float* y,
                                  int m, int slices, float alpha, float beta,
                                  cudaStream_t stream = nullptr);

/**
 * @brief Block CSR SpMV over square blocks.
 * @param block_row_ptr Device pointer to the block row offsets, length block_rows+1.
 * @param block_col_idx Device pointer to the block column indices, length blocks.
 * @param values Device pointer to the block values, row major inside a block.
 * @param x Device pointer to x, length n.
 * @param y Device pointer to y, length m, updated in place.
 * @param m Rows of A and length of y.
 * @param n Columns of A and length of x.
 * @param block_rows Number of block rows, ceil(m / block_dim).
 * @param block_dim Square block dimension.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming y; when zero, y is not read.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note One column index and one x load are shared by block_dim squared values,
 *       which is the whole of what the variant buys, and it only buys it where
 *       the block structure is real.
 */
CKL_EXPORT void spmv_bsr(const int* block_row_ptr, const int* block_col_idx, const float* values,
                         const float* x, float* y, int m, int n, int block_rows, int block_dim,
                         float alpha, float beta, cudaStream_t stream = nullptr);

/**
 * @brief cuSPARSE cusparseSpMV oracle and baseline, CSR and FP32, same result.
 * @param row_ptr Device pointer to the CSR row offsets, length m+1.
 * @param col_idx Device pointer to the CSR column indices, length nnz.
 * @param values Device pointer to the CSR values, length nnz.
 * @param x Device pointer to x, length n.
 * @param y Device pointer to y, length m, updated in place.
 * @param m Rows of A and length of y.
 * @param n Columns of A and length of x.
 * @param nnz Number of stored nonzeros.
 * @param alpha Scale on the product.
 * @param beta Scale on the incoming y.
 * @param stream Stream to enqueue on; nullptr means the default stream.
 * @note Runs CUSPARSE_SPMV_ALG_DEFAULT on the process wide default Context and
 *       takes its lock, through a one entry plan cache keyed on the buffers and
 *       the shape. A benchmark should build its own ckl::SpmvPlan instead.
 */
CKL_EXPORT void spmv_cusparse(const int* row_ptr, const int* col_idx, const float* values,
                              const float* x, float* y, int m, int n, int nnz, float alpha,
                              float beta, cudaStream_t stream = nullptr);

}  // namespace ckl
