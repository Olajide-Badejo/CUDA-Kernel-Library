// PTX level tensor core GEMM using mma.sync directly. Round 4 showed the WMMA
// kernel starving the tensor cores because the generic load_matrix_sync path
// spends most of its cycles on shared memory transactions. Dropping to
// mma.sync.aligned.m16n8k16 removes the WMMA abstraction and lets the kernel
// place exactly the fragment elements each lane needs, following the thread to
// element mapping in the PTX ISA. FP16 storage, FP32 accumulate.
//
// This first PTX version loads fragments from shared memory with direct indexed
// reads rather than ldmatrix; ldmatrix is the next lever if the round still
// shows the shared pipe as the limiter. Block tile 64 by 64, K step 16, four
// warps, so it is a like for like comparison against the WMMA kernel and
// isolates the effect of the instruction path. Aligned fast path (m by 64,
// n by 64, k by 16); other shapes reuse the WMMA kernel's scalar fallback.

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
constexpr int kMTiles = kWarpM / 16;              // 2 (m16 per mma)
constexpr int kNTiles = kWarpN / 8;               // 4 (n8 per mma)

__device__ inline uint32_t pack(__half lo, __half hi) {
    __half2 h = __halves2half2(lo, hi);
    return *reinterpret_cast<uint32_t*>(&h);
}

__device__ inline void mma_m16n8k16(float (&d)[4], uint32_t a0, uint32_t a1,
                                    uint32_t a2, uint32_t a3, uint32_t b0, uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a0), "r"(a1), "r"(a2), "r"(a3), "r"(b0), "r"(b1));
}

__global__ __launch_bounds__(kThreads) void gemm_mma_ptx_kernel(
    const __half* __restrict__ a, const __half* __restrict__ b, float* __restrict__ c,
    int m, int n, int k, float alpha, float beta) {
    __shared__ __align__(16) __half as[kBM * kBK];  // [row][kk]
    __shared__ __align__(16) __half bs[kBK * kBN];  // [kk][col]

    const int block_row = blockIdx.y * kBM;
    const int block_col = blockIdx.x * kBN;
    const int tid = threadIdx.x;
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / kWarpsN;
    const int warp_n = warp % kWarpsN;

    // Lane to fragment mapping for m16n8k16 (PTX ISA): row group and the pair
    // index within the group.
    const int group = lane / 4;
    const int tpair = lane % 4;

    float acc[kMTiles][kNTiles][4] = {};

    // Staging: one 128 bit (eight half) load per thread each for A (64x16) and
    // B (16x64).
    const int a_row = tid / 2;
    const int a_col = (tid % 2) * 8;
    const int b_row = tid / 8;
    const int b_col = (tid % 8) * 8;

    for (int kk = 0; kk < k; kk += kBK) {
        *reinterpret_cast<float4*>(&as[a_row * kBK + a_col]) =
            *reinterpret_cast<const float4*>(
                &a[static_cast<long long>(block_row + a_row) * k + kk + a_col]);
        *reinterpret_cast<float4*>(&bs[b_row * kBN + b_col]) =
            *reinterpret_cast<const float4*>(
                &b[static_cast<long long>(kk + b_row) * n + block_col + b_col]);
        __syncthreads();

#pragma unroll
        for (int mi = 0; mi < kMTiles; ++mi) {
            const int row_base = warp_m * kWarpM + mi * 16;
            // A fragment: adjacent K elements pack into one b32, so a uint32 read
            // of the shared row grabs the f16x2 pair directly.
            const uint32_t a0 = *reinterpret_cast<const uint32_t*>(&as[(row_base + group) * kBK + 2 * tpair]);
            const uint32_t a1 = *reinterpret_cast<const uint32_t*>(&as[(row_base + group + 8) * kBK + 2 * tpair]);
            const uint32_t a2 = *reinterpret_cast<const uint32_t*>(&as[(row_base + group) * kBK + 2 * tpair + 8]);
            const uint32_t a3 = *reinterpret_cast<const uint32_t*>(&as[(row_base + group + 8) * kBK + 2 * tpair + 8]);
#pragma unroll
            for (int ni = 0; ni < kNTiles; ++ni) {
                const int col_base = warp_n * kWarpN + ni * 8;
                // B fragment: the two K elements of each b32 are a row apart in
                // shared (stride kBN), so pack them explicitly.
                const uint32_t b0 = pack(bs[(2 * tpair + 0) * kBN + col_base + group],
                                         bs[(2 * tpair + 1) * kBN + col_base + group]);
                const uint32_t b1 = pack(bs[(2 * tpair + 8) * kBN + col_base + group],
                                         bs[(2 * tpair + 9) * kBN + col_base + group]);
                mma_m16n8k16(acc[mi][ni], a0, a1, a2, a3, b0, b1);
            }
        }
        __syncthreads();
    }

#pragma unroll
    for (int mi = 0; mi < kMTiles; ++mi) {
        const int row_base = block_row + warp_m * kWarpM + mi * 16;
#pragma unroll
        for (int ni = 0; ni < kNTiles; ++ni) {
            const int col_base = block_col + warp_n * kWarpN + ni * 8;
            // D fragment mapping for m16n8k16: rows {group, group+8}, cols
            // {2*tpair, 2*tpair+1}.
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

void gemm_mma_ptx(const __half* a, const __half* b, float* c,
                  int m, int n, int k, float alpha, float beta,
                  cudaStream_t stream) {
    if (m <= 0 || n <= 0) {
        return;
    }
    if (!aligned(m, n, k)) {
        // The WMMA launcher already carries a correct scalar fallback for any
        // shape, so reuse it rather than duplicating one here.
        gemm_wmma_fp16(a, b, c, m, n, k, alpha, beta, stream);
        return;
    }
    const dim3 block(kThreads);
    const dim3 grid(n / kBN, m / kBM);
    gemm_mma_ptx_kernel<<<grid, block, 0, stream>>>(a, b, c, m, n, k, alpha, beta);
}

}  // namespace ckl
