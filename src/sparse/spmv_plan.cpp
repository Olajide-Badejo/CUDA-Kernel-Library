// SpmvPlan: everything an SpMV variant needs that does not change between calls.
//
// v1 had no plan. cusparse_ref.cpp created three descriptors, sized the SpMV
// workspace, cudaMalloc'd it and cudaFree'd it on every call, and bench_all
// timed all of that as if it were the vendor's SpMV. That is defect A2, and it
// is why the 131.8 percent of cuSPARSE in docs/sparse.md says nothing about
// cuSPARSE. The plan is the structural fix: the descriptors, both workspaces,
// the histogram, the SELL-C-sigma copy, the BSR copy and the merge carry scratch
// are all built here, once per matrix, and a call enqueues launches and nothing
// else. The CUPTI counter bench_spmv arms is what proves that claim rather than
// asserting it.
//
// Three decisions in this file are mine and none of them is a measurement.
//
//   The lane width for spmv_csr_vector is the smallest power of two at or above
//   the mean nonzeros per row, clamped to 2 to 32. That is the classic CSR
//   vector rule: below the mean the group loops, above it the group idles, and
//   the mean is where those two costs cross for a distribution with a light
//   tail. It is wrong for a heavy tail, which is what the merge variant is for.
//
//   A matrix is called power law when its longest row holds at least 32 times
//   the mean. 32 is the warp width, so the rule reads as "one row can occupy a
//   whole warp for as long as the mean row occupies one lane". webbase-1M is
//   about 1500 times the mean and parabolic_fem is about 2, so nothing in the
//   suite sits near the threshold and the exact constant does not decide a case.
//
//   The BSR block dimension is the largest of 2, 4 and 8 whose fill ratio
//   reaches one half, and 2 when none of them does. Below one half the padded
//   values cost more traffic than the shared column indices save, which is the
//   arithmetic, not a measurement. The detection pass is timed on the host and
//   reported, because a stretch variant whose setup costs more than its kernel
//   saves has not earned its place.
//
// Every one of the three is provisional and says so in docs/sparse.md.

#include "ckl/sparse.hpp"

#include <algorithm>
#include <chrono>
#include <cstddef>
#include <memory>
#include <numeric>
#include <string>
#include <vector>

#include <cuda_runtime.h>
#include <cusparse.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/status.hpp"

#include "spmv_launch.hpp"

namespace ckl {

namespace {

void check(cusparseStatus_t s, const char* expr) {
    if (s != CUSPARSE_STATUS_SUCCESS) {
        throw Error(
            s == CUSPARSE_STATUS_ARCH_MISMATCH ? Status::kArchMismatch : Status::kExecutionFailed,
            std::string("cuSPARSE error ") + cusparseGetErrorString(s) + ": " + expr);
    }
}

constexpr int kHistogramBuckets = 32;
constexpr int kSliceHeight = 32;

// Sigma defaults. A regular degree distribution gains little from a wide sort
// window and pays for the scatter it puts on the y write; a power law one needs
// the wide window to stop a single long row padding an entire slice.
constexpr int kSigmaRegular = 256;
constexpr int kSigmaPowerLaw = 8192;

// The ceiling on a converted copy, in entries. Roughly a gigabyte of values plus
// indices, which is a memory limit and nothing else. The padding ratio is not
// gated here on purpose: a bad ratio is a result the benchmark row has to quote,
// not a reason to refuse to build the layout, and a variant that could only ever
// be built where it wins would make the padding column meaningless.
constexpr long long kMaxPaddedEntries = 1LL << 27;

// The fill a block has to reach before its shared column index pays for the
// zeros it drags along.
constexpr double kMinBlockFill = 0.5;

int bucket_of(int nnz_row) {
    if (nnz_row <= 0) {
        return 0;
    }
    int bucket = 1;
    int bound = 2;
    while (bucket + 1 < kHistogramBuckets && nnz_row >= bound) {
        ++bucket;
        bound <<= 1;
    }
    return bucket;
}

int width_for(double mean) {
    if (mean <= 2.0) {
        return 2;
    }
    if (mean <= 4.0) {
        return 4;
    }
    if (mean <= 8.0) {
        return 8;
    }
    if (mean <= 16.0) {
        return 16;
    }
    return 32;
}

}  // namespace

const char* spmv_algo_name(SpmvAlgo a) {
    switch (a) {
        case SpmvAlgo::kAuto:
            return "kAuto";
        case SpmvAlgo::kCsrNaive:
            return "kCsrNaive";
        case SpmvAlgo::kCsrWarp:
            return "kCsrWarp";
        case SpmvAlgo::kCsrVector:
            return "kCsrVector";
        case SpmvAlgo::kMerge:
            return "kMerge";
        case SpmvAlgo::kSellCSigma:
            return "kSellCSigma";
        case SpmvAlgo::kBsr:
            return "kBsr";
        case SpmvAlgo::kCusparseDefault:
            return "kCusparseDefault";
        case SpmvAlgo::kCusparseAlg2:
            return "kCusparseAlg2";
    }
    return "unknown";
}

// ---------------------------------------------------------------------------

struct SpmvPlan::Impl {
    SpmvCsr a;
    SpmvPlanOptions opt;

    // Host copies of the structure. Kept only for the conversions; released
    // before the constructor returns.
    std::vector<int> row_ptr;

    cusparseHandle_t handle = nullptr;
    cusparseSpMatDescr_t mat = nullptr;
    cusparseDnVecDescr_t vec_x = nullptr;
    cusparseDnVecDescr_t vec_y = nullptr;
    DeviceBuffer<unsigned char> buffer_default;
    DeviceBuffer<unsigned char> buffer_alg2;
    // The descriptors need a real pointer at build time so the workspace sizing
    // and the ALG2 analysis can run before any call arrives. This is that
    // pointer; every call rebinds the descriptors to the caller's own buffers.
    DeviceBuffer<float> placeholder;

    long long histogram[kHistogramBuckets] = {0};
    double mean = 0.0;
    int max_row = 0;
    bool power_law = false;
    int vector_width = 32;

    int sigma = 0;
    int slices = 0;
    long long sell_padded = 0;
    std::string sell_reason;
    DeviceBuffer<int> sell_slice_ptr;
    DeviceBuffer<int> sell_col;
    DeviceBuffer<float> sell_val;
    DeviceBuffer<int> sell_perm;
    DeviceBuffer<int> sell_len;

    int block_dim = 0;
    int block_rows = 0;
    long long blocks = 0;
    long long bsr_padded = 0;
    double detect_ms = 0.0;
    std::string bsr_reason;
    DeviceBuffer<int> bsr_row_ptr;
    DeviceBuffer<int> bsr_col;
    DeviceBuffer<float> bsr_val;

    int merge_threads = 0;
    DeviceBuffer<int> carry_rows;
    DeviceBuffer<float> carry_values;

    ~Impl() { release(); }

    void release() {
        if (vec_y != nullptr) {
            cusparseDestroyDnVec(vec_y);
            vec_y = nullptr;
        }
        if (vec_x != nullptr) {
            cusparseDestroyDnVec(vec_x);
            vec_x = nullptr;
        }
        if (mat != nullptr) {
            cusparseDestroySpMat(mat);
            mat = nullptr;
        }
        if (handle != nullptr) {
            cusparseDestroy(handle);
            handle = nullptr;
        }
    }

    void build();
    void build_cusparse();
    void build_histogram();
    void build_sell(const std::vector<int>& host_col, const std::vector<float>& host_val);
    void build_bsr(const std::vector<int>& host_col, const std::vector<float>& host_val);
    void run(SpmvAlgo algo, float alpha, const float* x, float beta, float* y, cudaStream_t stream);
    void run_cusparse(cusparseSpMVAlg_t alg, void* buffer, float alpha, const float* x, float beta,
                      float* y, cudaStream_t stream);
};

void SpmvPlan::Impl::build_cusparse() {
    check(cusparseCreate(&handle), "cusparseCreate");
    // cuSPARSE takes non const pointers; SpMV does not modify the inputs.
    check(cusparseCreateCsr(&mat, a.m, a.n, a.nnz, const_cast<int*>(a.row_ptr),
                            const_cast<int*>(a.col_idx), const_cast<float*>(a.values),
                            CUSPARSE_INDEX_32I, CUSPARSE_INDEX_32I, CUSPARSE_INDEX_BASE_ZERO,
                            CUDA_R_32F),
          "cusparseCreateCsr");

    const std::size_t widest = static_cast<std::size_t>(std::max(a.m, a.n));
    placeholder = DeviceBuffer<float>(widest);
    placeholder.zero();
    check(cusparseCreateDnVec(&vec_x, a.n, placeholder.data(), CUDA_R_32F),
          "cusparseCreateDnVec x");
    check(cusparseCreateDnVec(&vec_y, a.m, placeholder.data(), CUDA_R_32F),
          "cusparseCreateDnVec y");

    // Both vendor algorithms, both sized here. The workspace size does not
    // depend on alpha or beta, so the pair used for sizing is not the pair a
    // call will pass.
    const float one = 1.0f;
    const float zero = 0.0f;
    const cusparseSpMVAlg_t algs[2] = {CUSPARSE_SPMV_ALG_DEFAULT, CUSPARSE_SPMV_CSR_ALG2};
    DeviceBuffer<unsigned char>* buffers[2] = {&buffer_default, &buffer_alg2};
    for (int i = 0; i < 2; ++i) {
        std::size_t bytes = 0;
        check(cusparseSpMV_bufferSize(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat, vec_x,
                                      &zero, vec_y, CUDA_R_32F, algs[i], &bytes),
              "cusparseSpMV_bufferSize");
        if (bytes > 0) {
            *buffers[i] = DeviceBuffer<unsigned char>(bytes);
        }
        // ALG2 buys a deterministic reduction order out of an analysis of the
        // sparsity pattern. Running that analysis here rather than on the first
        // call is the whole point of having a plan: it depends on row_ptr and
        // col_idx, not on the dense vectors, so the placeholder is enough.
        check(cusparseSpMV_preprocess(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &one, mat, vec_x,
                                      &zero, vec_y, CUDA_R_32F, algs[i], buffers[i]->data()),
              "cusparseSpMV_preprocess");
    }
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
}

void SpmvPlan::Impl::build_histogram() {
    for (int i = 0; i < a.m; ++i) {
        const int len =
            row_ptr[static_cast<std::size_t>(i) + 1] - row_ptr[static_cast<std::size_t>(i)];
        histogram[bucket_of(len)] += 1;
        max_row = std::max(max_row, len);
    }
    mean = a.m > 0 ? static_cast<double>(a.nnz) / static_cast<double>(a.m) : 0.0;
    // 32 is the warp width: a row this much longer than the mean occupies a
    // whole warp for as long as a mean row occupies one lane.
    power_law = mean > 0.0 && static_cast<double>(max_row) >= 32.0 * mean;
    vector_width = opt.vector_width > 0 ? opt.vector_width : width_for(mean);
}

void SpmvPlan::Impl::build_sell(const std::vector<int>& host_col,
                                const std::vector<float>& host_val) {
    sigma = opt.sigma > 0 ? opt.sigma : (power_law ? kSigmaPowerLaw : kSigmaRegular);
    slices = (a.m + kSliceHeight - 1) / kSliceHeight;

    // The permutation: rows sorted by descending length inside each window of
    // sigma rows, so a slice of 32 holds rows of similar length and pads little.
    // sigma of 1 is no sorting at all, which is the control case the sweep runs.
    std::vector<int> order(static_cast<std::size_t>(a.m));
    std::iota(order.begin(), order.end(), 0);
    if (sigma > 1) {
        for (int w = 0; w < a.m; w += sigma) {
            const int stop = std::min(w + sigma, a.m);
            std::stable_sort(order.begin() + w, order.begin() + stop, [&](int lhs, int rhs) {
                const int ll = row_ptr[static_cast<std::size_t>(lhs) + 1] -
                               row_ptr[static_cast<std::size_t>(lhs)];
                const int rl = row_ptr[static_cast<std::size_t>(rhs) + 1] -
                               row_ptr[static_cast<std::size_t>(rhs)];
                return ll > rl;
            });
        }
    }

    const std::size_t slots = static_cast<std::size_t>(slices) * kSliceHeight;
    std::vector<int> perm(slots, -1);
    std::vector<int> length(slots, 0);
    for (int i = 0; i < a.m; ++i) {
        const int row = order[static_cast<std::size_t>(i)];
        perm[static_cast<std::size_t>(i)] = row;
        length[static_cast<std::size_t>(i)] =
            row_ptr[static_cast<std::size_t>(row) + 1] - row_ptr[static_cast<std::size_t>(row)];
    }

    std::vector<int> slice_ptr(static_cast<std::size_t>(slices) + 1, 0);
    long long padded = 0;
    for (int s = 0; s < slices; ++s) {
        int longest = 0;
        for (int lane = 0; lane < kSliceHeight; ++lane) {
            longest = std::max(longest, length[static_cast<std::size_t>(s) * kSliceHeight + lane]);
        }
        padded += static_cast<long long>(longest) * kSliceHeight;
        slice_ptr[static_cast<std::size_t>(s) + 1] = static_cast<int>(padded);
    }

    if (padded > kMaxPaddedEntries) {
        sell_reason = "the SELL-C-sigma layout would hold " + std::to_string(padded) +
                      " entries for a matrix of " + std::to_string(a.nnz) + " nonzeros, past the " +
                      std::to_string(kMaxPaddedEntries) +
                      " entry ceiling; raise sigma or set build_sell to false";
        slices = 0;
        sell_padded = 0;
        return;
    }

    std::vector<int> cols(static_cast<std::size_t>(padded), 0);
    std::vector<float> vals(static_cast<std::size_t>(padded), 0.0f);
    for (int s = 0; s < slices; ++s) {
        const int base = slice_ptr[static_cast<std::size_t>(s)];
        for (int lane = 0; lane < kSliceHeight; ++lane) {
            const std::size_t slot = static_cast<std::size_t>(s) * kSliceHeight + lane;
            const int row = perm[slot];
            if (row < 0) {
                continue;
            }
            const int start = row_ptr[static_cast<std::size_t>(row)];
            for (int j = 0; j < length[slot]; ++j) {
                const std::size_t at = static_cast<std::size_t>(base) +
                                       static_cast<std::size_t>(j) * kSliceHeight +
                                       static_cast<std::size_t>(lane);
                const std::size_t from =
                    static_cast<std::size_t>(start) + static_cast<std::size_t>(j);
                cols[at] = host_col[from];
                vals[at] = host_val[from];
            }
        }
    }

    sell_padded = padded;
    sell_slice_ptr = DeviceBuffer<int>(slice_ptr.size());
    sell_slice_ptr.copy_from_host(slice_ptr);
    sell_perm = DeviceBuffer<int>(perm.size());
    sell_perm.copy_from_host(perm);
    sell_len = DeviceBuffer<int>(length.size());
    sell_len.copy_from_host(length);
    if (padded > 0) {
        sell_col = DeviceBuffer<int>(cols.size());
        sell_col.copy_from_host(cols);
        sell_val = DeviceBuffer<float>(vals.size());
        sell_val.copy_from_host(vals);
    }
}

void SpmvPlan::Impl::build_bsr(const std::vector<int>& host_col,
                               const std::vector<float>& host_val) {
    if (opt.block_dim > 1 && opt.block_dim != 2 && opt.block_dim != 4 && opt.block_dim != 8) {
        // The kernel is instantiated for 2, 4 and 8. Building a conversion the
        // kernel would then refuse is worse than refusing here and saying why.
        bsr_reason = "block_dim " + std::to_string(opt.block_dim) +
                     " is not one of the instantiated dimensions 2, 4 and 8";
        return;
    }
    const auto started = std::chrono::steady_clock::now();

    // The detection pass: count the blocks each candidate dimension would need,
    // without materializing any of them. One stamp per block column, reset by
    // writing the current block row rather than by clearing the array.
    int chosen_dim = 0;
    long long chosen_blocks = 0;
    double chosen_fill = 0.0;
    // A caller who named a block dimension gets that one and no detection pass:
    // the point of the option is to override the rule, and running the rule
    // anyway and then ignoring it would only cost time.
    const int only[1] = {opt.block_dim};
    const int candidates[3] = {8, 4, 2};
    const int* first = opt.block_dim > 1 ? only : candidates;
    const int tried = opt.block_dim > 1 ? 1 : 3;

    // Widest first. Fill is non increasing in the block dimension, so the first
    // candidate that reaches the threshold is the widest one that does, and if
    // none of them reaches it the loop ends holding the narrowest, which is the
    // one with the least padding to answer for.
    for (int candidate = 0; candidate < tried; ++candidate) {
        const int bd = first[candidate];
        if (a.n < bd) {
            continue;
        }
        const int brows = (a.m + bd - 1) / bd;
        const int bcols = (a.n + bd - 1) / bd;
        std::vector<int> stamp(static_cast<std::size_t>(bcols), -1);
        long long count = 0;
        for (int br = 0; br < brows; ++br) {
            const int row_lo = br * bd;
            const int row_hi = std::min(row_lo + bd, a.m);
            for (int row = row_lo; row < row_hi; ++row) {
                for (int k = row_ptr[static_cast<std::size_t>(row)];
                     k < row_ptr[static_cast<std::size_t>(row) + 1]; ++k) {
                    const int bc = host_col[static_cast<std::size_t>(k)] / bd;
                    if (stamp[static_cast<std::size_t>(bc)] != br) {
                        stamp[static_cast<std::size_t>(bc)] = br;
                        ++count;
                    }
                }
            }
        }
        const long long padded = count * bd * bd;
        const double fill =
            padded > 0 ? static_cast<double>(a.nnz) / static_cast<double>(padded) : 0.0;
        chosen_dim = bd;
        chosen_blocks = count;
        chosen_fill = fill;
        if (fill >= kMinBlockFill) {
            break;
        }
    }
    detect_ms =
        std::chrono::duration<double, std::milli>(std::chrono::steady_clock::now() - started)
            .count();

    if (chosen_dim == 0) {
        bsr_reason = "the matrix has fewer columns than the smallest block dimension";
        return;
    }
    const long long padded = chosen_blocks * chosen_dim * chosen_dim;
    if (padded > kMaxPaddedEntries) {
        bsr_reason = "the best block dimension is " + std::to_string(chosen_dim) +
                     " at a fill of " + std::to_string(chosen_fill) + ", which would store " +
                     std::to_string(padded) + " values for " + std::to_string(a.nnz) +
                     " nonzeros, past the " + std::to_string(kMaxPaddedEntries) + " entry ceiling";
        return;
    }

    const int brows = (a.m + chosen_dim - 1) / chosen_dim;
    const int bcols = (a.n + chosen_dim - 1) / chosen_dim;
    std::vector<int> block_row_ptr(static_cast<std::size_t>(brows) + 1, 0);
    std::vector<int> block_col;
    block_col.reserve(static_cast<std::size_t>(chosen_blocks));
    std::vector<float> block_val(static_cast<std::size_t>(padded), 0.0f);
    std::vector<int> stamp(static_cast<std::size_t>(bcols), -1);
    std::vector<int> where(static_cast<std::size_t>(bcols), 0);

    for (int br = 0; br < brows; ++br) {
        const int row_lo = br * chosen_dim;
        const int row_hi = std::min(row_lo + chosen_dim, a.m);
        for (int row = row_lo; row < row_hi; ++row) {
            for (int k = row_ptr[static_cast<std::size_t>(row)];
                 k < row_ptr[static_cast<std::size_t>(row) + 1]; ++k) {
                const int col = host_col[static_cast<std::size_t>(k)];
                const int bc = col / chosen_dim;
                if (stamp[static_cast<std::size_t>(bc)] != br) {
                    stamp[static_cast<std::size_t>(bc)] = br;
                    where[static_cast<std::size_t>(bc)] = static_cast<int>(block_col.size());
                    block_col.push_back(bc);
                }
                const std::size_t block =
                    static_cast<std::size_t>(where[static_cast<std::size_t>(bc)]);
                const std::size_t at = block * static_cast<std::size_t>(chosen_dim) * chosen_dim +
                                       static_cast<std::size_t>(row - row_lo) * chosen_dim +
                                       static_cast<std::size_t>(col - bc * chosen_dim);
                // Duplicate columns sum, exactly as they do in CSR.
                block_val[at] += host_val[static_cast<std::size_t>(k)];
            }
        }
        block_row_ptr[static_cast<std::size_t>(br) + 1] = static_cast<int>(block_col.size());
    }

    block_dim = chosen_dim;
    block_rows = brows;
    blocks = static_cast<long long>(block_col.size());
    bsr_padded = blocks * chosen_dim * chosen_dim;
    bsr_row_ptr = DeviceBuffer<int>(block_row_ptr.size());
    bsr_row_ptr.copy_from_host(block_row_ptr);
    if (!block_col.empty()) {
        bsr_col = DeviceBuffer<int>(block_col.size());
        bsr_col.copy_from_host(block_col);
        bsr_val = DeviceBuffer<float>(static_cast<std::size_t>(bsr_padded));
        bsr_val.copy_from_host(std::vector<float>(
            block_val.begin(), block_val.begin() + static_cast<std::ptrdiff_t>(bsr_padded)));
    }
}

void SpmvPlan::Impl::build() {
    if (a.m < 0 || a.n < 0 || a.nnz < 0) {
        throw Error(Status::kInvalidValue, "SpmvPlan: m, n and nnz must all be non negative");
    }
    if (a.nnz > 0 && (a.row_ptr == nullptr || a.col_idx == nullptr || a.values == nullptr)) {
        throw Error(Status::kInvalidValue, "SpmvPlan: a null buffer in a non empty matrix");
    }
    if (a.m == 0 || a.n == 0) {
        // A matrix with no rows or no columns has nothing to descriptor and
        // nothing to convert. Every variant returns immediately on it, which is
        // the y untouched contract the correctness suite checks.
        sell_reason = "the matrix has no rows or no columns";
        bsr_reason = sell_reason;
        return;
    }

    build_cusparse();

    row_ptr.resize(static_cast<std::size_t>(a.m) + 1);
    CKL_CUDA_CHECK(cudaMemcpy(row_ptr.data(), a.row_ptr, row_ptr.size() * sizeof(int),
                              cudaMemcpyDeviceToHost));
    build_histogram();

    merge_threads = spmv_merge_carry_count(a.m, a.nnz);
    if (merge_threads > 0) {
        carry_rows = DeviceBuffer<int>(static_cast<std::size_t>(merge_threads));
        carry_values = DeviceBuffer<float>(static_cast<std::size_t>(merge_threads));
    }

    if (opt.build_sell || opt.build_bsr) {
        std::vector<int> host_col(static_cast<std::size_t>(a.nnz));
        std::vector<float> host_val(static_cast<std::size_t>(a.nnz));
        if (a.nnz > 0) {
            CKL_CUDA_CHECK(cudaMemcpy(host_col.data(), a.col_idx, host_col.size() * sizeof(int),
                                      cudaMemcpyDeviceToHost));
            CKL_CUDA_CHECK(cudaMemcpy(host_val.data(), a.values, host_val.size() * sizeof(float),
                                      cudaMemcpyDeviceToHost));
        }
        if (opt.build_sell) {
            build_sell(host_col, host_val);
        } else {
            sell_reason = "the plan was built with build_sell false";
        }
        if (opt.build_bsr && opt.block_dim != 1) {
            build_bsr(host_col, host_val);
        } else {
            bsr_reason = opt.block_dim == 1 ? "the plan was built with block_dim 1"
                                            : "the plan was built with build_bsr false";
        }
    } else {
        sell_reason = "the plan was built with build_sell false";
        bsr_reason = "the plan was built with build_bsr false";
    }

    // The host copy of the offsets is only needed by the conversions above.
    row_ptr.shrink_to_fit();
}

void SpmvPlan::Impl::run_cusparse(cusparseSpMVAlg_t alg, void* buffer, float alpha, const float* x,
                                  float beta, float* y, cudaStream_t stream) {
    // Rebinding the dense descriptors is a host side pointer write. It replaces
    // the three descriptor creations, the buffer sizing and the malloc and free
    // pair that v1 ran on every call.
    check(cusparseDnVecSetValues(vec_x, const_cast<float*>(x)), "cusparseDnVecSetValues x");
    check(cusparseDnVecSetValues(vec_y, y), "cusparseDnVecSetValues y");
    check(cusparseSetStream(handle, stream), "cusparseSetStream");
    check(cusparseSpMV(handle, CUSPARSE_OPERATION_NON_TRANSPOSE, &alpha, mat, vec_x, &beta, vec_y,
                       CUDA_R_32F, alg, buffer),
          "cusparseSpMV");
}

void SpmvPlan::Impl::run(SpmvAlgo algo, float alpha, const float* x, float beta, float* y,
                         cudaStream_t stream) {
    if (a.m <= 0 || a.n <= 0) {
        return;  // no output elements, so nothing to write and nothing to launch
    }
    switch (algo) {
        case SpmvAlgo::kCsrNaive:
            spmv_csr_naive(a.row_ptr, a.col_idx, a.values, x, y, a.m, a.n, a.nnz, alpha, beta,
                           stream);
            return;
        case SpmvAlgo::kCsrWarp:
            spmv_csr_warp(a.row_ptr, a.col_idx, a.values, x, y, a.m, a.n, a.nnz, alpha, beta,
                          stream);
            return;
        case SpmvAlgo::kCsrVector:
            spmv_csr_vector(a.row_ptr, a.col_idx, a.values, x, y, a.m, a.n, a.nnz, alpha, beta,
                            vector_width, stream);
            return;
        case SpmvAlgo::kMerge:
            spmv_merge(a.row_ptr, a.col_idx, a.values, x, y, a.m, a.n, a.nnz, alpha, beta,
                       carry_rows.data(), carry_values.data(), merge_threads, stream);
            return;
        case SpmvAlgo::kSellCSigma:
            if (slices <= 0) {
                throw Error(Status::kNotSupported,
                            "SpmvPlan has no SELL-C-sigma copy of this matrix: " + sell_reason);
            }
            spmv_sell_c_sigma(sell_slice_ptr.data(), sell_col.data(), sell_val.data(),
                              sell_perm.data(), sell_len.data(), x, y, a.m, slices, alpha, beta,
                              stream);
            return;
        case SpmvAlgo::kBsr:
            if (block_dim <= 0) {
                throw Error(Status::kNotSupported,
                            "SpmvPlan has no BSR copy of this matrix: " + bsr_reason);
            }
            spmv_bsr(bsr_row_ptr.data(), bsr_col.data(), bsr_val.data(), x, y, a.m, a.n, block_rows,
                     block_dim, alpha, beta, stream);
            return;
        case SpmvAlgo::kCusparseDefault:
            run_cusparse(CUSPARSE_SPMV_ALG_DEFAULT, buffer_default.data(), alpha, x, beta, y,
                         stream);
            return;
        case SpmvAlgo::kCusparseAlg2:
            run_cusparse(CUSPARSE_SPMV_CSR_ALG2, buffer_alg2.data(), alpha, x, beta, y, stream);
            return;
        case SpmvAlgo::kAuto:
        default:
            throw Error(Status::kInternal, "SpmvPlan::run reached kAuto; dispatch resolves it");
    }
}

// ---------------------------------------------------------------------------

SpmvPlan::SpmvPlan(const SpmvCsr& a, const SpmvPlanOptions& opt) : impl_(std::make_unique<Impl>()) {
    impl_->a = a;
    impl_->opt = opt;
    impl_->build();
}

SpmvPlan::~SpmvPlan() = default;
SpmvPlan::SpmvPlan(SpmvPlan&&) noexcept = default;
SpmvPlan& SpmvPlan::operator=(SpmvPlan&&) noexcept = default;

SpmvPlan::Impl& SpmvPlan::live() const {
    if (!impl_) {
        throw Error(Status::kInvalidValue, "SpmvPlan: this plan has been moved from");
    }
    return *impl_;
}

const SpmvCsr& SpmvPlan::matrix() const {
    return live().a;
}

long long SpmvPlan::histogram(int bucket) const {
    if (bucket < 0 || bucket >= kHistogramBuckets) {
        return 0;
    }
    return live().histogram[bucket];
}

int SpmvPlan::histogram_buckets() {
    return kHistogramBuckets;
}

double SpmvPlan::mean_nnz_per_row() const {
    return live().mean;
}

int SpmvPlan::max_nnz_per_row() const {
    return live().max_row;
}

bool SpmvPlan::power_law() const {
    return live().power_law;
}

int SpmvPlan::vector_width() const {
    return live().vector_width;
}

int SpmvPlan::sigma() const {
    return live().sigma;
}

int SpmvPlan::sell_slice_height() {
    return kSliceHeight;
}

long long SpmvPlan::sell_nnz_padded() const {
    return live().sell_padded;
}

double SpmvPlan::sell_padding_ratio() const {
    const Impl& p = live();
    if (p.a.nnz <= 0 || p.sell_padded <= 0) {
        return 0.0;
    }
    return static_cast<double>(p.sell_padded) / static_cast<double>(p.a.nnz);
}

int SpmvPlan::bsr_block_dim() const {
    return live().block_dim;
}

long long SpmvPlan::bsr_nnz_padded() const {
    return live().bsr_padded;
}

double SpmvPlan::bsr_detect_ms() const {
    return live().detect_ms;
}

long long SpmvPlan::model_bytes_low(bool beta_nonzero) const {
    const Impl& p = live();
    long long bytes =
        8LL * p.a.nnz + 4LL * (static_cast<long long>(p.a.m) + 1) + 4LL * p.a.n + 4LL * p.a.m;
    if (beta_nonzero) {
        bytes += 4LL * p.a.m;
    }
    return bytes;
}

long long SpmvPlan::model_bytes_high(bool beta_nonzero) const {
    const Impl& p = live();
    long long bytes =
        8LL * p.a.nnz + 4LL * (static_cast<long long>(p.a.m) + 1) + 4LL * p.a.nnz + 4LL * p.a.m;
    if (beta_nonzero) {
        bytes += 4LL * p.a.m;
    }
    return bytes;
}

SpmvAlgo SpmvPlan::query() const {
    const Impl& p = live();
    if (p.a.nnz == 0) {
        // Nothing to balance and nothing to gather; the cheapest launch wins.
        return SpmvAlgo::kCsrVector;
    }
    if (p.power_law) {
        // The case the merge path exists for: one row long enough to decide how
        // long the launch takes if any variant hands it to a single group.
        return SpmvAlgo::kMerge;
    }
    if (p.mean >= 32.0) {
        // A full warp per row wastes nothing at this length, and it keeps the
        // reduction to one shfl tree rather than one per narrow group.
        return SpmvAlgo::kCsrWarp;
    }
    return SpmvAlgo::kCsrVector;
}

namespace detail {

void spmv_launch(SpmvPlan& plan, SpmvAlgo algo, float alpha, const float* x, float beta, float* y,
                 cudaStream_t stream) {
    plan.live().run(algo, alpha, x, beta, y, stream);
}

}  // namespace detail

}  // namespace ckl
