// Block CSR SpMV, the stretch variant of the family.
//
// A dense r by c block shares one column index and one span of x across r*c
// values, so where the block structure is real the index traffic falls by a
// factor of r*c and each x element is loaded once for r rows instead of r times.
// That is the whole of the argument, and it only applies to pdb1HYS and cant.
// On the graph classes the blocks are empty air: the conversion pads a 4 by 4
// block for a single stored nonzero and moves sixteen times the values it needs.
//
// One warp per block row, lanes striding the block columns of that row. Each
// lane keeps BlockDim accumulators, one per row of the block, and the warp
// reduces all of them at the end. The reduction is BlockDim shuffles rather than
// one, which is the price of holding a whole block row's partial results in
// registers instead of going back to memory for them.
//
// The conversion, and the decision of whether it is worth running at all,
// belongs to SpmvPlan. The plan also records what the block detection pass cost
// on the host, because a variant whose setup costs more than its kernel saves
// has not earned its place.

#include "ckl/cuda_check.hpp"
#include "ckl/sparse.hpp"
#include "ckl/status.hpp"

namespace ckl {

namespace {

constexpr int kWarpsPerBlock = 4;
constexpr int kBlock = kWarpsPerBlock * 32;

template <int BlockDim>
__global__ void spmv_bsr_kernel(const int* __restrict__ block_row_ptr,
                                const int* __restrict__ block_col_idx,
                                const float* __restrict__ values, const float* __restrict__ x,
                                float* __restrict__ y, int m, int n, int block_rows, float alpha,
                                float beta) {
    const int thread = static_cast<int>(blockIdx.x) * kBlock + static_cast<int>(threadIdx.x);
    const int brow = thread / 32;
    const int lane = thread % 32;
    if (brow >= block_rows) {
        return;
    }
    const int start = block_row_ptr[brow];
    const int end = block_row_ptr[brow + 1];

    float acc[BlockDim];
#pragma unroll
    for (int r = 0; r < BlockDim; ++r) {
        acc[r] = 0.0f;
    }

    for (int b = start + lane; b < end; b += 32) {
        const int col0 = block_col_idx[b] * BlockDim;
        const float* block = values + static_cast<long long>(b) * BlockDim * BlockDim;
#pragma unroll
        for (int c = 0; c < BlockDim; ++c) {
            const int col = col0 + c;
            // The last block column of a matrix whose n is not a multiple of
            // BlockDim reaches past x. Those values are zero by construction, so
            // the guard is about not reading x, not about the answer.
            if (col < n) {
                const float xv = x[col];
#pragma unroll
                for (int r = 0; r < BlockDim; ++r) {
                    acc[r] += block[r * BlockDim + c] * xv;
                }
            }
        }
    }

#pragma unroll
    for (int r = 0; r < BlockDim; ++r) {
        float v = acc[r];
#pragma unroll
        for (int offset = 16; offset > 0; offset >>= 1) {
            v += __shfl_down_sync(0xffffffffu, v, offset);
        }
        if (lane == 0) {
            const int row = brow * BlockDim + r;
            if (row < m) {
                // y is not read when beta is zero, per the BLAS contract.
                y[row] = (beta == 0.0f) ? alpha * v : alpha * v + beta * y[row];
            }
        }
    }
}

template <int BlockDim>
void launch(const int* block_row_ptr, const int* block_col_idx, const float* values, const float* x,
            float* y, int m, int n, int block_rows, float alpha, float beta, cudaStream_t stream) {
    const long long threads = static_cast<long long>(block_rows) * 32;
    const int grid = static_cast<int>((threads + kBlock - 1) / kBlock);
    spmv_bsr_kernel<BlockDim><<<grid, kBlock, 0, stream>>>(block_row_ptr, block_col_idx, values, x,
                                                           y, m, n, block_rows, alpha, beta);
    CKL_CUDA_LAST_ERROR(false);
}

}  // namespace

void spmv_bsr(const int* block_row_ptr, const int* block_col_idx, const float* values,
              const float* x, float* y, int m, int n, int block_rows, int block_dim, float alpha,
              float beta, cudaStream_t stream) {
    if (m <= 0 || block_rows <= 0) {
        return;
    }
    switch (block_dim) {
        case 2:
            launch<2>(block_row_ptr, block_col_idx, values, x, y, m, n, block_rows, alpha, beta,
                      stream);
            return;
        case 4:
            launch<4>(block_row_ptr, block_col_idx, values, x, y, m, n, block_rows, alpha, beta,
                      stream);
            return;
        case 8:
            launch<8>(block_row_ptr, block_col_idx, values, x, y, m, n, block_rows, alpha, beta,
                      stream);
            return;
        default:
            // A silent fallback to another block dimension would make the row's
            // reported block size a lie, so this refuses instead.
            throw Error(Status::kNotSupported,
                        "spmv_bsr: the instantiated block dimensions are 2, 4 and 8");
    }
}

}  // namespace ckl
