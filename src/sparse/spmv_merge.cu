// Merge path CSR SpMV, after Merrill and Garland, SC 2016.
//
// The warp per row kernel balances the lanes inside a row and does nothing about
// the imbalance between rows. On webbase-1M one row holds about 4700 nonzeros
// while its neighbors hold four, so one warp runs for the length of that row
// while the warps beside it retire immediately, and the launch takes as long as
// the longest row.
//
// The merge path removes the row from the unit of work entirely. Think of two
// sorted lists, the row end offsets row_ptr[1..m] and the nonzero indices
// 0..nnz-1, and of the path a merge of the two would take through the
// (m+1) by (nnz+1) grid. That path has exactly m + nnz steps whatever the degree
// distribution. Cut it into equal length diagonal segments, one per thread, and
// every thread does exactly ceil((m + nnz) / threads) units: a thread either
// consumes a nonzero, which is a multiply add, or consumes a row boundary, which
// is a store. No degree distribution can make one segment longer than another.
//
// What that costs is the rows that straddle a segment boundary. The thread that
// consumes a row's boundary item writes the part of the row it covered itself;
// every earlier thread that held a piece of the same row leaves it in a carry
// slot, and a second pass adds the carries in. The carry pass is one thread per
// segment, so it is proportional to the thread count and not to the matrix.
//
// The binary search that finds a thread's starting point is the standard merge
// path search: the split point on the diagonal is the smallest i with
// row_ptr[i+1] > diagonal - i - 1, which a lower bound over the row offsets
// answers in log(m) steps.

#include "ckl/cuda_check.hpp"
#include "ckl/sparse.hpp"

namespace ckl {

namespace {

constexpr int kBlock = 128;

// Merge units per thread. This is the one tuning number in the file and it is a
// shape decision rather than a measurement.
//
// It sets both how much serial work a thread does and how far apart two
// neighbouring threads read, because thread t starts at path offset t * items.
// Small keeps neighbouring lanes reading a narrow contiguous window, which is
// most of the coalescing this kernel can get; large amortizes the log(m) binary
// search each thread pays to find its own starting point. Seven is the value
// Merrill and Garland use for the same trade, and the first thing an ncu round
// on this kernel should reopen.
//
// The thread count follows from it rather than being capped, so the work per
// thread stays constant as the matrix grows. A 69 million nonzero matrix gets
// ten million threads and eighty megabytes of carry scratch, which is the
// correct trade: capping the threads instead would put a thousand serial merge
// units in each one and scatter its reads across the whole matrix.
constexpr int kItemsPerThread = 7;

// The split point of the merge path on this diagonal. row_end[t] is
// row_ptr[t+1], and the second list is the identity, so b[k] is k.
__device__ inline void merge_path_search(int diagonal, const int* __restrict__ row_end, int rows,
                                         int nnz, int& row, int& entry) {
    int lo = max(0, diagonal - nnz);
    int hi = min(diagonal, rows);
    while (lo < hi) {
        const int mid = (lo + hi) >> 1;
        if (row_end[mid] <= diagonal - mid - 1) {
            lo = mid + 1;
        } else {
            hi = mid;
        }
    }
    row = lo;
    entry = diagonal - lo;
}

__global__ void spmv_merge_kernel(const int* __restrict__ row_ptr, const int* __restrict__ col_idx,
                                  const float* __restrict__ values, const float* __restrict__ x,
                                  float* __restrict__ y, int m, int nnz, float alpha, float beta,
                                  int* __restrict__ carry_rows, float* __restrict__ carry_values,
                                  int threads, int items) {
    const int tid = static_cast<int>(blockIdx.x) * kBlock + static_cast<int>(threadIdx.x);
    if (tid >= threads) {
        return;
    }
    // Written unconditionally: the fixup pass reads every slot, so a thread that
    // returns early still has to say it carries nothing.
    carry_rows[tid] = -1;
    carry_values[tid] = 0.0f;

    const int path = m + nnz;
    const int begin = tid * items;
    if (begin >= path) {
        return;
    }
    const int end = min(begin + items, path);

    const int* row_end = row_ptr + 1;
    int row = 0;
    int entry = 0;
    merge_path_search(begin, row_end, m, nnz, row, entry);

    float sum = 0.0f;
    for (int d = begin; d < end; ++d) {
        if (row < m && row_end[row] <= entry) {
            // The row boundary: this thread owns the store for row, and every
            // earlier thread that held part of it will add its carry later.
            y[row] = (beta == 0.0f) ? alpha * sum : alpha * sum + beta * y[row];
            sum = 0.0f;
            ++row;
        } else if (entry < nnz) {
            sum += values[entry] * x[col_idx[entry]];
            ++entry;
        } else {
            break;
        }
    }
    if (sum != 0.0f && row < m) {
        carry_rows[tid] = row;
        carry_values[tid] = sum;
    }
}

// One thread per segment. Several segments can carry into the same very long
// row, which is the case the whole kernel exists for, so the add is atomic.
__global__ void spmv_merge_fixup_kernel(float* __restrict__ y, const int* __restrict__ carry_rows,
                                        const float* __restrict__ carry_values, int threads,
                                        float alpha) {
    const int tid = static_cast<int>(blockIdx.x) * kBlock + static_cast<int>(threadIdx.x);
    if (tid >= threads) {
        return;
    }
    const int row = carry_rows[tid];
    if (row >= 0) {
        atomicAdd(&y[row], alpha * carry_values[tid]);
    }
}

}  // namespace

int spmv_merge_carry_count(int m, int nnz) {
    if (m <= 0) {
        return 0;
    }
    const long long path = static_cast<long long>(m) + nnz;
    const long long threads = (path + kItemsPerThread - 1) / kItemsPerThread;
    // Rounded to whole blocks so the launcher's grid covers exactly the slots
    // the caller allocated, and every slot the fixup reads was written.
    const long long blocks = (threads + kBlock - 1) / kBlock;
    return static_cast<int>(blocks * kBlock);
}

void spmv_merge(const int* row_ptr, const int* col_idx, const float* values, const float* x,
                float* y, int m, int n, int nnz, float alpha, float beta, int* carry_rows,
                float* carry_values, int threads, cudaStream_t stream) {
    (void)n;
    if (m <= 0) {
        return;
    }
    if (threads <= 0 || carry_rows == nullptr || carry_values == nullptr) {
        throw Error(Status::kInvalidValue,
                    "spmv_merge: the carry scratch is a plan concern and must be sized from "
                    "spmv_merge_carry_count before the call");
    }
    const int grid = (threads + kBlock - 1) / kBlock;
    spmv_merge_kernel<<<grid, kBlock, 0, stream>>>(row_ptr, col_idx, values, x, y, m, nnz, alpha,
                                                   beta, carry_rows, carry_values, threads,
                                                   kItemsPerThread);
    CKL_CUDA_LAST_ERROR(false);
    spmv_merge_fixup_kernel<<<grid, kBlock, 0, stream>>>(y, carry_rows, carry_values, threads,
                                                         alpha);
    CKL_CUDA_LAST_ERROR(false);
}

}  // namespace ckl
