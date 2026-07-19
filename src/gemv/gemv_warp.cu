// Warp per row GEMV. Each warp owns one output row; the 32 lanes stride across
// the row so adjacent lanes read adjacent A elements (a coalesced row read), then
// a shfl_down reduction sums the lane partials. This is the coalescing fix over
// the naive kernel and, being memory bound, it should track achievable bandwidth.

#include "ckl/gemv.hpp"

namespace ckl {

namespace {

__device__ inline float warp_reduce_sum(float v) {
#pragma unroll
    for (int offset = 16; offset > 0; offset >>= 1) {
        v += __shfl_down_sync(0xffffffffu, v, offset);
    }
    return v;
}

__global__ void gemv_warp_kernel(const float* __restrict__ a, const float* __restrict__ x,
                                 float* __restrict__ y, int m, int n, float alpha, float beta) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp_id >= m) {
        return;
    }
    const long long base = static_cast<long long>(warp_id) * n;
    float sum = 0.0f;
    for (int j = lane; j < n; j += 32) {
        sum += a[base + j] * x[j];
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        y[warp_id] = alpha * sum + beta * y[warp_id];
    }
}

}  // namespace

void gemv_warp(const float* a, const float* x, float* y, int m, int n, float alpha, float beta,
               cudaStream_t stream) {
    if (m <= 0) {
        return;
    }
    constexpr int kWarpsPerBlock = 4;
    constexpr int kBlock = kWarpsPerBlock * 32;
    const int grid = (m + kWarpsPerBlock - 1) / kWarpsPerBlock;
    gemv_warp_kernel<<<grid, kBlock, 0, stream>>>(a, x, y, m, n, alpha, beta);
}

}  // namespace ckl
