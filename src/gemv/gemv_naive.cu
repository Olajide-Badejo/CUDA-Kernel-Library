// Naive GEMV: one thread per output row. Each thread walks its whole row of A.
// The read pattern is the weak point on purpose: at a given step of the inner
// loop, the 32 threads of a warp are reading A elements one full row (n floats)
// apart, so the loads do not coalesce. This is the honest baseline the warp and
// vectorized variants improve on.

#include "ckl/gemv.hpp"

namespace ckl {

namespace {

__global__ void gemv_naive_kernel(const float* __restrict__ a, const float* __restrict__ x,
                                  float* __restrict__ y, int m, int n, float alpha, float beta) {
    const int row = blockIdx.x * blockDim.x + threadIdx.x;
    if (row >= m) {
        return;
    }
    float sum = 0.0f;
    const long long base = static_cast<long long>(row) * n;
    for (int j = 0; j < n; ++j) {
        sum += a[base + j] * x[j];
    }
    y[row] = alpha * sum + beta * y[row];
}

}  // namespace

void gemv_naive(const float* a, const float* x, float* y, int m, int n, float alpha, float beta,
                cudaStream_t stream) {
    if (m <= 0) {
        return;
    }
    constexpr int kBlock = 128;
    const int grid = (m + kBlock - 1) / kBlock;
    gemv_naive_kernel<<<grid, kBlock, 0, stream>>>(a, x, y, m, n, alpha, beta);
}

}  // namespace ckl
