// Naive FP32 GEMM: one thread computes one output element with the natural
// triple loop collapsed so the inner k loop runs per thread. Every A and B
// element is read from global memory with no reuse, which is the point: this is
// the honest baseline the ladder improves on, and its Nsight profile is the
// memory bound starting picture the diagnostic rounds move away from.

#include "ckl/gemm.hpp"

namespace ckl {

namespace {

__global__ void gemm_naive_kernel(const float* __restrict__ a,
                                  const float* __restrict__ b,
                                  float* __restrict__ c,
                                  int m, int n, int k,
                                  float alpha, float beta) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }
    float acc = 0.0f;
    for (int p = 0; p < k; ++p) {
        acc += a[static_cast<long long>(row) * k + p] * b[static_cast<long long>(p) * n + col];
    }
    const long long idx = static_cast<long long>(row) * n + col;
    c[idx] = alpha * acc + beta * c[idx];
}

}  // namespace

void gemm_naive(const float* a, const float* b, float* c,
                int m, int n, int k, float alpha, float beta,
                cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;  // nothing to write; k == 0 still applies beta below
    }
    constexpr int kBlock = 16;
    const dim3 block(kBlock, kBlock);
    const dim3 grid((n + kBlock - 1) / kBlock, (m + kBlock - 1) / kBlock);
    gemm_naive_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k, alpha, beta);
}

const char* gemm_variant_name(GemmVariant v) {
    switch (v) {
        case GemmVariant::kNaive: return "naive";
        case GemmVariant::kTiled: return "tiled";
        case GemmVariant::kRegister: return "register";
        case GemmVariant::kCpAsync: return "cp_async";
        case GemmVariant::kWmmaFp16: return "wmma_fp16";
        case GemmVariant::kWmmaBf16: return "wmma_bf16";
        case GemmVariant::kMmaPtx: return "mma_ptx";
        case GemmVariant::kCutlass: return "cutlass";
        case GemmVariant::kCublas: return "cublas";
    }
    return "unknown";
}

}  // namespace ckl
