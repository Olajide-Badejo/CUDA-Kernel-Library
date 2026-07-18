// Shared memory tiled FP32 GEMM. Each block computes one TILE by TILE output
// tile, streaming K in TILE wide steps through shared memory so every A and B
// element loaded from global memory is reused TILE times by the threads of the
// block. That reuse is the whole point of this rung: it cuts global traffic by
// the tile factor versus the naive kernel, which is the first attributable step
// up the ladder.
//
// The shared tiles carry one column of padding so the staging loads, which
// write column adjacent elements, do not collide on shared memory banks.

#include "ckl/gemm.hpp"

namespace ckl {

namespace {

constexpr int kTile = 32;

__global__ void gemm_tiled_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int m, int n, int k,
                                  float alpha, float beta) {
    __shared__ float as[kTile][kTile + 1];
    __shared__ float bs[kTile][kTile + 1];

    const int tx = threadIdx.x;
    const int ty = threadIdx.y;
    const int row = blockIdx.y * kTile + ty;
    const int col = blockIdx.x * kTile + tx;

    float acc = 0.0f;
    const int num_tiles = (k + kTile - 1) / kTile;
    for (int t = 0; t < num_tiles; ++t) {
        const int a_col = t * kTile + tx;
        const int b_row = t * kTile + ty;

        as[ty][tx] = (row < m && a_col < k)
            ? a[static_cast<long long>(row) * k + a_col] : 0.0f;
        bs[ty][tx] = (b_row < k && col < n)
            ? b[static_cast<long long>(b_row) * n + col] : 0.0f;
        __syncthreads();

#pragma unroll
        for (int p = 0; p < kTile; ++p) {
            acc += as[ty][p] * bs[p][tx];
        }
        __syncthreads();
    }

    if (row < m && col < n) {
        const long long idx = static_cast<long long>(row) * n + col;
        c[idx] = alpha * acc + beta * c[idx];
    }
}

}  // namespace

void gemm_tiled(const float* a, const float* b, float* c,
                int m, int n, int k, float alpha, float beta,
                cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }
    if (k <= 0) {
        // Empty contraction reduces to C = beta * C; the tiled loop would skip
        // all tiles and never scale C, so handle it with the naive kernel which
        // does apply beta.
        gemm_naive(a, b, c, m, n, k, alpha, beta, stream);
        return;
    }
    const dim3 block(kTile, kTile);
    const dim3 grid((n + kTile - 1) / kTile, (m + kTile - 1) / kTile);
    gemm_tiled_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k, alpha, beta);
}

}  // namespace ckl
