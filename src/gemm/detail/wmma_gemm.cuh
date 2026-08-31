#pragma once

// Templated WMMA GEMM shared by the FP16 and BF16 rungs. The two only differ in
// the fragment storage type, so the kernel, the shared staging, and the aligned
// fast path plus scalar fallback live here once and each .cu instantiates it for
// its type. FP32 accumulate throughout; the output C is FP32.
//
// Block tile 64 by 64, K step 16, four warps. Each warp owns a 32 by 32 region,
// which is a 2 by 2 grid of 16 by 16 WMMA fragments. A and B tiles are staged
// into shared memory with 128 bit loads (eight halves per load) so the fragment
// loads read from shared, not global. The fast path needs m and n multiples of
// 64 and k a multiple of 16; anything else (odd, sub tile, zero k) uses a scalar
// half input kernel so correctness holds on every shape the tests throw at it.
//
// The tensor body is compiled for sm_80 and newer only. BF16 WMMA fragments do
// not exist below that, and the project's release architecture list starts at
// sm_80, so the FP16 instantiation carries the same floor rather than a second
// one nobody builds. The scalar fallback has no floor and runs anywhere, which
// is why the architecture check sits on the fast path in the launcher instead of
// at the top of it.

#include <mma.h>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ckl/context.hpp"
#include "ckl/cuda_check.hpp"
#include "ckl/gemm.hpp"

namespace ckl {
namespace detail {

using namespace nvcuda;

constexpr int kWmmaM = 16;
constexpr int kWmmaN = 16;
constexpr int kWmmaK = 16;
constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 16;
constexpr int kWarpsM = 2;
constexpr int kWarpsN = 2;
constexpr int kWarps = kWarpsM * kWarpsN;     // 4
constexpr int kThreads = kWarps * 32;         // 128
constexpr int kWarpTileM = kBM / kWarpsM;     // 32
constexpr int kWarpTileN = kBN / kWarpsN;     // 32
// Used only inside the guarded device body, so below the architecture floor
// they have no reader and nvcc's unreferenced variable diagnostic, which is an
// error under CKL_WERROR, would fire.
[[maybe_unused]] constexpr int kMFrags = kWarpTileM / kWmmaM;  // 2
[[maybe_unused]] constexpr int kNFrags = kWarpTileN / kWmmaN;  // 2
constexpr int kMinArch = 80;                  // bf16 WMMA fragments

__device__ inline float to_float(__half h) {
    return __half2float(h);
}
__device__ inline float to_float(__nv_bfloat16 h) {
    return __bfloat162float(h);
}

template <typename T>
__global__ __launch_bounds__(kThreads) void wmma_gemm_kernel(const T* __restrict__ a,
                                                             const T* __restrict__ b,
                                                             float* __restrict__ c, int m, int n,
                                                             int k, float alpha, float beta) {
#if __CUDA_ARCH__ >= 800
    __shared__ __align__(16) T as[kBM * kBK];
    __shared__ __align__(16) T bs[kBK * kBN];

    const int block_row = blockIdx.y * kBM;
    const int block_col = blockIdx.x * kBN;
    const int warp = threadIdx.x / 32;
    const int warp_m = warp / kWarpsN;
    const int warp_n = warp % kWarpsN;

    wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> acc[kMFrags][kNFrags];
#pragma unroll
    for (int mi = 0; mi < kMFrags; ++mi) {
#pragma unroll
        for (int ni = 0; ni < kNFrags; ++ni) {
            wmma::fill_fragment(acc[mi][ni], 0.0f);
        }
    }

    // Staging indices: one 128 bit (eight element) load per thread for each of
    // A (64 by 16) and B (16 by 64).
    const int tid = threadIdx.x;
    const int a_row = tid / 2;
    const int a_col = (tid % 2) * 8;
    const int b_row = tid / 8;
    const int b_col = (tid % 8) * 8;

    for (int kk = 0; kk < k; kk += kBK) {
        *reinterpret_cast<float4*>(&as[a_row * kBK + a_col]) = *reinterpret_cast<const float4*>(
            &a[static_cast<long long>(block_row + a_row) * k + kk + a_col]);
        *reinterpret_cast<float4*>(&bs[b_row * kBN + b_col]) = *reinterpret_cast<const float4*>(
            &b[static_cast<long long>(kk + b_row) * n + block_col + b_col]);
        __syncthreads();

#pragma unroll
        for (int mi = 0; mi < kMFrags; ++mi) {
            wmma::fragment<wmma::matrix_a, kWmmaM, kWmmaN, kWmmaK, T, wmma::row_major> a_frag;
            const int a_shared_row = warp_m * kWarpTileM + mi * kWmmaM;
            wmma::load_matrix_sync(a_frag, &as[a_shared_row * kBK], kBK);
#pragma unroll
            for (int ni = 0; ni < kNFrags; ++ni) {
                wmma::fragment<wmma::matrix_b, kWmmaM, kWmmaN, kWmmaK, T, wmma::row_major> b_frag;
                const int b_shared_col = warp_n * kWarpTileN + ni * kWmmaN;
                wmma::load_matrix_sync(b_frag, &bs[b_shared_col], kBN);
                wmma::mma_sync(acc[mi][ni], a_frag, b_frag, acc[mi][ni]);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < kMFrags; ++mi) {
#pragma unroll
        for (int ni = 0; ni < kNFrags; ++ni) {
            const int row = block_row + warp_m * kWarpTileM + mi * kWmmaM;
            const int col = block_col + warp_n * kWarpTileN + ni * kWmmaN;
            float* c_tile = &c[static_cast<long long>(row) * n + col];
            // When beta is zero C is not an input at all, so the old fragment is
            // never loaded. That is the BLAS contract, and it also saves a
            // 16 by 16 global read per fragment.
            if (beta != 0.0f) {
                wmma::fragment<wmma::accumulator, kWmmaM, kWmmaN, kWmmaK, float> c_old;
                wmma::load_matrix_sync(c_old, c_tile, n, wmma::mem_row_major);
#pragma unroll
                for (int t = 0; t < acc[mi][ni].num_elements; ++t) {
                    acc[mi][ni].x[t] = alpha * acc[mi][ni].x[t] + beta * c_old.x[t];
                }
            } else {
#pragma unroll
                for (int t = 0; t < acc[mi][ni].num_elements; ++t) {
                    acc[mi][ni].x[t] = alpha * acc[mi][ni].x[t];
                }
            }
            wmma::store_matrix_sync(c_tile, acc[mi][ni], n, wmma::mem_row_major);
        }
    }
#else
    // No WMMA body below sm_80. The launcher checks the running device before it
    // reaches the fast path, so this is never launched.
    (void)a;
    (void)b;
    (void)c;
    (void)m;
    (void)n;
    (void)k;
    (void)alpha;
    (void)beta;
#endif
}

// Scalar half input fallback for shapes the WMMA fast path cannot tile. One
// thread per output element, FP32 accumulate, so it matches the tensor kernel to
// within the tolerance while staying correct on odd and degenerate shapes. No
// tensor instructions, so no architecture floor.
template <typename T>
__global__ void half_naive_kernel(const T* __restrict__ a, const T* __restrict__ b,
                                  float* __restrict__ c, int m, int n, int k, float alpha,
                                  float beta) {
    const int col = blockIdx.x * blockDim.x + threadIdx.x;
    const int row = blockIdx.y * blockDim.y + threadIdx.y;
    if (row >= m || col >= n) {
        return;
    }
    float sum = 0.0f;
    for (int p = 0; p < k; ++p) {
        sum += to_float(a[static_cast<long long>(row) * k + p]) *
               to_float(b[static_cast<long long>(p) * n + col]);
    }
    const long long idx = static_cast<long long>(row) * n + col;
    c[idx] = (beta == 0.0f) ? alpha * sum : alpha * sum + beta * c[idx];
}

template <typename T>
void launch_wmma(const T* a, const T* b, float* c, int m, int n, int k, float alpha, float beta,
                 cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }
    const bool fast = (m % kBM == 0) && (n % kBN == 0) && (k % kBK == 0) && k > 0;
    if (fast) {
        require_arch(kMinArch, "the WMMA GEMM fast path");
        const dim3 block(kThreads);
        const dim3 grid(n / kBN, m / kBM);
        wmma_gemm_kernel<T><<<grid, block, 0, stream>>>(a, b, c, m, n, k, alpha, beta);
    } else {
        constexpr int kB = 16;
        const dim3 block(kB, kB);
        const dim3 grid((n + kB - 1) / kB, (m + kB - 1) / kB);
        half_naive_kernel<T><<<grid, block, 0, stream>>>(a, b, c, m, n, k, alpha, beta);
    }
    CKL_CUDA_LAST_ERROR(false);
}

}  // namespace detail
}  // namespace ckl
