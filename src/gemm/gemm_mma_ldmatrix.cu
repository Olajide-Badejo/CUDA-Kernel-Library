// mma.sync tensor core GEMM with ldmatrix fragment loads. Rounds 4 and 5 showed
// the tensor kernels bound by the shared read (L1 / TEX) pipe: the fragment
// assembly, not the mma instruction, is the limiter. ldmatrix is the specific
// relief. One ldmatrix.sync loads a full 16 by 16 (or transposed 16 by 8)
// fragment per warp in a single swizzled shared transaction, replacing the many
// scalar shared loads the manual mma kernel issues.
//
// This variant keeps the same 64 by 64 block tile, K step 16, and four warp
// layout as the manual mma kernel so the round isolates the effect of the load
// path. If it flips the Speed of Light from memory bound to tensor bound, the
// next steps (larger tile, cp.async double buffering) push toward the gate.
// FP16 storage, FP32 accumulate. Aligned fast path; other shapes reuse the WMMA
// scalar fallback.

#include <cstdint>

#include <cuda_fp16.h>

#include "ckl/gemm.hpp"

namespace ckl {

namespace {

constexpr int kBM = 64;
constexpr int kBN = 64;
constexpr int kBK = 16;
constexpr int kWarpsM = 2;
constexpr int kWarpsN = 2;
constexpr int kThreads = kWarpsM * kWarpsN * 32;  // 128
constexpr int kWarpM = kBM / kWarpsM;             // 32
constexpr int kWarpN = kBN / kWarpsN;             // 32
constexpr int kMTiles = kWarpM / 16;              // 2
constexpr int kNTiles = kWarpN / 8;               // 4

__device__ inline uint32_t smem_u32(const void* p) {
    return static_cast<uint32_t>(__cvta_generic_to_shared(p));
}

__device__ inline void ldmatrix_x4(uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(smem_u32(p)));
}

__device__ inline void ldmatrix_x2_trans(uint32_t (&r)[2], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x2.trans.shared.b16 {%0,%1}, [%2];\n"
                 : "=r"(r[0]), "=r"(r[1])
                 : "r"(smem_u32(p)));
}

__device__ inline void mma_m16n8k16(float (&d)[4], const uint32_t (&a)[4], const uint32_t (&b)[2]) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b[0]), "r"(b[1]));
}

__global__ __launch_bounds__(kThreads) void gemm_mma_ldmatrix_kernel(const __half* __restrict__ a,
                                                                     const __half* __restrict__ b,
                                                                     float* __restrict__ c, int m,
                                                                     int n, int k, float alpha,
                                                                     float beta) {
    __shared__ __align__(16) __half as[kBM * kBK];  // [row][kk]
    __shared__ __align__(16) __half bs[kBK * kBN];  // [kk][col]

    const int block_row = blockIdx.y * kBM;
    const int block_col = blockIdx.x * kBN;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / kWarpsN;
    const int warp_n = warp % kWarpsN;

    float acc[kMTiles][kNTiles][4] = {};

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

        uint32_t a_frag[kMTiles][4];
#pragma unroll
        for (int mi = 0; mi < kMTiles; ++mi) {
            const int row_base = warp_m * kWarpM + mi * 16;
            // Row major A tile: lane gives its row and which 8 wide K half.
            ldmatrix_x4(a_frag[mi], &as[(row_base + (lane % 16)) * kBK + (lane / 16) * 8]);
        }
#pragma unroll
        for (int ni = 0; ni < kNTiles; ++ni) {
            const int col_base = warp_n * kWarpN + ni * 8;
            // Row major B tile loaded transposed to the col major operand.
            uint32_t b_frag[2];
            ldmatrix_x2_trans(b_frag, &bs[(lane % 16) * kBN + col_base]);
#pragma unroll
            for (int mi = 0; mi < kMTiles; ++mi) {
                mma_m16n8k16(acc[mi][ni], a_frag[mi], b_frag);
            }
        }
        __syncthreads();
    }

    const int group = lane / 4;
    const int tpair = lane % 4;
#pragma unroll
    for (int mi = 0; mi < kMTiles; ++mi) {
        const int row_base = block_row + warp_m * kWarpM + mi * 16;
#pragma unroll
        for (int ni = 0; ni < kNTiles; ++ni) {
            const int col_base = block_col + warp_n * kWarpN + ni * 8;
            const int r0 = row_base + group;
            const int r1 = row_base + group + 8;
            const int c0 = col_base + 2 * tpair;
            const int c1 = col_base + 2 * tpair + 1;
            const long long i00 = static_cast<long long>(r0) * n + c0;
            const long long i01 = static_cast<long long>(r0) * n + c1;
            const long long i10 = static_cast<long long>(r1) * n + c0;
            const long long i11 = static_cast<long long>(r1) * n + c1;
            c[i00] = alpha * acc[mi][ni][0] + beta * c[i00];
            c[i01] = alpha * acc[mi][ni][1] + beta * c[i01];
            c[i10] = alpha * acc[mi][ni][2] + beta * c[i10];
            c[i11] = alpha * acc[mi][ni][3] + beta * c[i11];
        }
    }
}

bool aligned(int m, int n, int k) {
    return m % kBM == 0 && n % kBN == 0 && k % kBK == 0 && k > 0;
}

}  // namespace

void gemm_mma_ldm(const __half* a, const __half* b, float* c, int m, int n, int k, float alpha,
                  float beta, cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }
    if (!aligned(m, n, k)) {
        gemm_wmma_fp16(a, b, c, m, n, k, alpha, beta, stream);
        return;
    }
    const dim3 block(kThreads);
    const dim3 grid(n / kBN, m / kBM);
    gemm_mma_ldmatrix_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k, alpha, beta);
}

}  // namespace ckl
