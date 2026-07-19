// Vectorized warp per row GEMV. Same warp per row structure as gemv_warp, but the
// lanes read A and x as float4, so each memory instruction moves 16 bytes and the
// row read issues a quarter of the loads. Since GEMV is memory bound, cutting the
// instruction count and moving wider transactions is the lever that matters.
// Requires n to be a multiple of 4; other n use the scalar warp kernel.

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

__global__ void gemv_vectorized_kernel(const float* __restrict__ a, const float* __restrict__ x,
                                       float* __restrict__ y, int m, int n,
                                       float alpha, float beta) {
    const int warp_id = (blockIdx.x * blockDim.x + threadIdx.x) / 32;
    const int lane = threadIdx.x % 32;
    if (warp_id >= m) {
        return;
    }
    const int n4 = n / 4;
    const auto* a4 = reinterpret_cast<const float4*>(a + static_cast<long long>(warp_id) * n);
    const auto* x4 = reinterpret_cast<const float4*>(x);
    float sum = 0.0f;
    for (int j = lane; j < n4; j += 32) {
        const float4 av = a4[j];
        const float4 xv = x4[j];
        sum += av.x * xv.x + av.y * xv.y + av.z * xv.z + av.w * xv.w;
    }
    sum = warp_reduce_sum(sum);
    if (lane == 0) {
        y[warp_id] = alpha * sum + beta * y[warp_id];
    }
}

}  // namespace

void gemv_vectorized(const float* a, const float* x, float* y,
                     int m, int n, float alpha, float beta, cudaStream_t stream) {
    if (m <= 0) {
        return;
    }
    if (n % 4 != 0) {
        gemv_warp(a, x, y, m, n, alpha, beta, stream);
        return;
    }
    constexpr int kWarpsPerBlock = 4;
    constexpr int kBlock = kWarpsPerBlock * 32;
    const int grid = (m + kWarpsPerBlock - 1) / kWarpsPerBlock;
    gemv_vectorized_kernel<<<grid, kBlock, 0, stream>>>(a, x, y, m, n, alpha, beta);
}

}  // namespace ckl
