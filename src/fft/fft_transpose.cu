// The transpose study, and the reason it exists.
//
// A 2D FFT is row transforms, transpose, row transforms, transpose. The row
// transforms are shared resident and cost one read and one write each; so do the
// transposes. Half the traffic of a 2D transform is therefore transposition, and
// a transpose is pure bandwidth with no arithmetic to hide behind, so it decides
// the time. That is why three variants are measured on their own before they are
// measured inside the 2D driver.
//
//   kNaive is the control. One thread per element, indexed by the output, so the
//   writes are contiguous and the reads walk down a column one sector at a time.
//   Nothing is shared. This is the variant whose sectors per request the ncu round
//   watches move.
//
//   kTiledPadded stages a 32 by 32 tile through shared memory so both the read
//   and the write are contiguous, with one element of padding on the row so the
//   column read out of shared does not land 32 ways on one bank.
//
//   kStridedShared keeps the shared staging and drops both of those: the load
//   walks the column and the tile is unpadded. It is the same kernel with the
//   strided access moved to the load side, so the difference between it and
//   kTiledPadded is what coalescing the load and padding the row are worth,
//   separately from what staging through shared memory is worth at all.
//
// Inside the 2D driver kStridedShared means something related and stronger: the
// column pass reads its column straight into the shared transform buffer, so
// there is no transpose at all and the 2D transform runs in two passes rather
// than four. See detail::fft_shared_strided.
//
// The epilogue arguments exist so the last transpose of the four step form can
// carry a convolution's pointwise multiply. The index they are applied at is the
// index inside one output matrix, so one filter spectrum serves a whole batch.

#include <string>

#include "ckl/cuda_check.hpp"
#include "ckl/fft.hpp"

#include "fft_device.cuh"

namespace ckl {

namespace {

constexpr int kTile = 32;
constexpr int kTileRows = 8;
constexpr int kBlock = 256;

__device__ __forceinline__ float2 apply_epilogue(float2 v, const float2* __restrict__ mul, int idx,
                                                 float scale) {
    if (mul != nullptr) {
        v = detail::cmul(v, mul[idx]);
    }
    if (scale != 1.0f) {
        v = detail::cscale(v, scale);
    }
    return v;
}

__global__ void transpose_naive_kernel(const float2* __restrict__ in, float2* __restrict__ out,
                                       int rows, int cols, long long total,
                                       const float2* __restrict__ mul, float scale) {
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= total) {
        return;
    }
    const long long elems = static_cast<long long>(rows) * cols;
    const long long b = t / elems;
    const int idx = static_cast<int>(t - b * elems);
    // The output is cols by rows, so index idx sits at output row idx / rows and
    // output column idx % rows, which is input row idx % rows and input column
    // idx / rows.
    const int r = idx % rows;
    const int c = idx / rows;
    const float2 v = in[b * elems + static_cast<long long>(r) * cols + c];
    out[b * elems + idx] = apply_epilogue(v, mul, idx, scale);
}

__global__ void transpose_tiled_kernel(const float2* __restrict__ in, float2* __restrict__ out,
                                       int rows, int cols, const float2* __restrict__ mul,
                                       float scale) {
    // One element of padding on the row so the column read below spreads across
    // banks instead of landing on one.
    __shared__ float2 tile[kTile][kTile + 1];
    const long long elems = static_cast<long long>(rows) * cols;
    const long long b = blockIdx.z;
    const int col0 = static_cast<int>(blockIdx.x) * kTile;
    const int row0 = static_cast<int>(blockIdx.y) * kTile;
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);

    for (int i = 0; i < kTile; i += kTileRows) {
        const int r = row0 + ty + i;
        const int c = col0 + tx;
        if (r < rows && c < cols) {
            tile[ty + i][tx] = in[b * elems + static_cast<long long>(r) * cols + c];
        }
    }
    __syncthreads();

    for (int i = 0; i < kTile; i += kTileRows) {
        const int r = col0 + ty + i;  // row of the output, which is a column of the input
        const int c = row0 + tx;
        if (r < cols && c < rows) {
            const int idx = r * rows + c;
            out[b * elems + idx] = apply_epilogue(tile[tx][ty + i], mul, idx, scale);
        }
    }
}

__global__ void transpose_strided_kernel(const float2* __restrict__ in, float2* __restrict__ out,
                                         int rows, int cols, const float2* __restrict__ mul,
                                         float scale) {
    // No padding, on purpose: this variant is the one that shows what the padding
    // in the tiled kernel is worth.
    __shared__ float2 tile[kTile][kTile];
    const long long elems = static_cast<long long>(rows) * cols;
    const long long b = blockIdx.z;
    const int col0 = static_cast<int>(blockIdx.x) * kTile;
    const int row0 = static_cast<int>(blockIdx.y) * kTile;
    const int tx = static_cast<int>(threadIdx.x);
    const int ty = static_cast<int>(threadIdx.y);

    // The lane index walks down a column of the input, so the load is strided.
    for (int i = 0; i < kTile; i += kTileRows) {
        const int r = row0 + tx;
        const int c = col0 + ty + i;
        if (r < rows && c < cols) {
            tile[tx][ty + i] = in[b * elems + static_cast<long long>(r) * cols + c];
        }
    }
    __syncthreads();

    for (int i = 0; i < kTile; i += kTileRows) {
        const int r = col0 + ty + i;
        const int c = row0 + tx;
        if (r < cols && c < rows) {
            const int idx = r * rows + c;
            out[b * elems + idx] = apply_epilogue(tile[tx][ty + i], mul, idx, scale);
        }
    }
}

int grid_of(long long total, int block) {
    return static_cast<int>((total + block - 1) / block);
}

}  // namespace

void fft_transpose(const float2* in, float2* out, int rows, int cols, int batch,
                   FftTranspose variant, const float2* epilogue_mul, float epilogue_scale,
                   cudaStream_t stream) {
    if (rows <= 0 || cols <= 0 || batch <= 0) {
        return;
    }
    if (variant == FftTranspose::kNaive) {
        const long long total = static_cast<long long>(rows) * cols * batch;
        transpose_naive_kernel<<<grid_of(total, kBlock), kBlock, 0, stream>>>(
            in, out, rows, cols, total, epilogue_mul, epilogue_scale);
        CKL_CUDA_LAST_ERROR(false);
        return;
    }
    // The batch rides on grid.z, which the driver caps at 65535. Nothing in the
    // family reaches that (the four step form only runs above 2^12, where the
    // batch is small), and a silent wrap would be worse than a refusal.
    if (batch > 65535) {
        throw Error(Status::kNotSupported,
                    "fft_transpose: the tiled variants carry the batch on grid.z, which is "
                    "limited to 65535; batch is " +
                        std::to_string(batch));
    }
    const dim3 block(kTile, kTileRows, 1);
    const dim3 grid(static_cast<unsigned>((cols + kTile - 1) / kTile),
                    static_cast<unsigned>((rows + kTile - 1) / kTile),
                    static_cast<unsigned>(batch));
    if (variant == FftTranspose::kTiledPadded) {
        transpose_tiled_kernel<<<grid, block, 0, stream>>>(in, out, rows, cols, epilogue_mul,
                                                           epilogue_scale);
    } else {
        transpose_strided_kernel<<<grid, block, 0, stream>>>(in, out, rows, cols, epilogue_mul,
                                                             epilogue_scale);
    }
    CKL_CUDA_LAST_ERROR(false);
}

}  // namespace ckl
