#pragma once

// The tile family mainloop: the round 10 kernel of gemm_mma_opt.cu, templated on
// (BM, BN, BK, warps in M, warps in N) and given predicated edges.
//
// Structure is unchanged from the shape it was written at. Three stages of
// cp.async on dynamic shared memory, one __syncthreads per K stage (the buffer
// written at iteration s was last read at s-1, which that barrier already
// fences), ldmatrix.x4 for A and ldmatrix.x4.trans for B with the B fragments
// double buffered in registers across the K substeps, and mma.sync.m16n8k16.
//
// Two things are new. The extents are template parameters, with the constraints
// the family has to respect written as static_asserts rather than left to a
// comment: the three stage budget inside the 99 KB per block opt-in, warp count
// inside 48 per SM, exact division of the staging work across the block, and a
// warp tile the mma shape divides. And every global access is predicated, so a
// shape that does not divide the tile runs this mainloop with masked edges
// rather than rerouting to a scalar kernel that costs twenty five to thirty
// times as much.
//
// The predication has two halves. Global to shared uses cp.async's own source
// size field: a chunk that is partly past an edge copies the bytes that exist
// and zero fills the rest, and a chunk wholly past an edge copies nothing and
// zero fills all sixteen bytes. Zeros in the staged tile contribute nothing to
// the dot product, so the mainloop itself needs no edge logic at all. The
// epilogue takes the other half: the aligned form keeps the float2 store the
// accumulator fragment allows, and the predicated form stores element by element
// under a bounds test.
//
// One shape of input defeats cp.async entirely. A sixteen byte copy needs a
// sixteen byte aligned source, and a row of A starts at a multiple of k, so an
// odd k puts every second row on an odd address. k = 127 is not an exotic case,
// it is one of the shapes in the correctness matrix. Those shapes stage through
// plain loads and shared stores instead, chosen by a flag the launcher computes
// from the leading dimensions and the operand base addresses. The stores are
// synchronous, but the pipeline's own barrier already separates the stage that
// writes a buffer from the iteration that reads it, so the loop structure is
// unchanged and only the copy instruction differs.
//
// The K range is a parameter rather than the whole contraction, which is what
// lets split-K and stream-K reuse the same loop for a slice of K.

#include <cstddef>
#include <cstdint>

#include <cuda_fp16.h>
#include <cuda_pipeline.h>

#include "gemm_tile_swizzle.hpp"

namespace ckl {
namespace detail {

/// One member of the tile family: the block tile, the warp layout it is cut
/// into, and everything derived from the two.
template <int BM, int BN, int BK, int WARPS_M, int WARPS_N>
struct TileShape {
    static constexpr int kBM = BM;
    static constexpr int kBN = BN;
    static constexpr int kBK = BK;
    static constexpr int kWarpsM = WARPS_M;
    static constexpr int kWarpsN = WARPS_N;
    static constexpr int kWarps = WARPS_M * WARPS_N;
    static constexpr int kThreads = kWarps * 32;
    static constexpr int kWarpM = BM / WARPS_M;
    static constexpr int kWarpN = BN / WARPS_N;
    static constexpr int kMTiles = kWarpM / 16;
    static constexpr int kNTiles = kWarpN / 8;
    // One ldmatrix.x4.trans covers two adjacent 16 by 8 output tiles.
    static constexpr int kNPairs = kNTiles / 2;
    static constexpr int kKSub = BK / 16;
    static constexpr int kStages = 3;
    static constexpr int kATileHalves = BM * BK;
    static constexpr int kBTileHalves = BK * BN;
    static constexpr std::size_t kSmemBytes =
        static_cast<std::size_t>(kStages) * (kATileHalves + kBTileHalves) * sizeof(__half);
    static constexpr int kARowChunks = BK / 8;
    static constexpr int kBRowChunks = BN / 8;
    static constexpr int kAChunks = kATileHalves / 8;
    static constexpr int kBChunks = kBTileHalves / 8;
    static constexpr int kACopies = kAChunks / kThreads;
    static constexpr int kBCopies = kBChunks / kThreads;

    // The opt-in ceiling on plain sm_120. A shape above it cannot be launched at
    // all, so it is a compile error rather than a run time refusal.
    static constexpr std::size_t kSmemCeiling = 101376;

    static_assert(BM % WARPS_M == 0, "the block tile has to divide across the warp rows");
    static_assert(BN % WARPS_N == 0, "the block tile has to divide across the warp columns");
    static_assert(kWarpM % 16 == 0, "a warp tile is a whole number of 16 row mma tiles");
    static_assert(kWarpN % 16 == 0, "an ldmatrix.x4.trans covers two 8 column mma tiles");
    static_assert(BK % 16 == 0, "the K step is a whole number of mma K substeps");
    static_assert(kAChunks % kThreads == 0, "A staging has to divide evenly across the block");
    static_assert(kBChunks % kThreads == 0, "B staging has to divide evenly across the block");
    static_assert(kThreads <= 1024, "a block cannot hold more than 1024 threads");
    static_assert(kWarps <= 48, "a block cannot hold more than the 48 warps an SM carries");
    static_assert(kSmemBytes <= kSmemCeiling,
                  "three stages of this tile do not fit the per block shared memory opt-in");
};

// cp.async, ldmatrix and mma.sync together: sm_80 is the floor. An unused device
// function still reaches ptxas, so the helpers sit inside the same guard as the
// bodies that call them.
#if !defined(__CUDA_ARCH__) || __CUDA_ARCH__ >= 800

__device__ inline std::uint32_t tile_smem_u32(const void* p) {
    return static_cast<std::uint32_t>(__cvta_generic_to_shared(p));
}

__device__ inline void tile_ldmatrix_x4(std::uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(tile_smem_u32(p)));
}

__device__ inline void tile_ldmatrix_x4_trans(std::uint32_t (&r)[4], const void* p) {
    asm volatile("ldmatrix.sync.aligned.m8n8.x4.trans.shared.b16 {%0,%1,%2,%3}, [%4];\n"
                 : "=r"(r[0]), "=r"(r[1]), "=r"(r[2]), "=r"(r[3])
                 : "r"(tile_smem_u32(p)));
}

__device__ inline void tile_mma_m16n8k16(float (&d)[4], const std::uint32_t (&a)[4],
                                         std::uint32_t b0, std::uint32_t b1) {
    asm volatile(
        "mma.sync.aligned.m16n8k16.row.col.f32.f16.f16.f32 "
        "{%0,%1,%2,%3}, {%4,%5,%6,%7}, {%8,%9}, {%0,%1,%2,%3};\n"
        : "+f"(d[0]), "+f"(d[1]), "+f"(d[2]), "+f"(d[3])
        : "r"(a[0]), "r"(a[1]), "r"(a[2]), "r"(a[3]), "r"(b0), "r"(b1));
}

// Bytes of a sixteen byte chunk that exist, given how many elements are left
// before the edge. Zero means the chunk is wholly out of range.
__device__ inline int tile_chunk_bytes(int remaining) {
    const int e = remaining < 0 ? 0 : (remaining > 8 ? 8 : remaining);
    return e * 2;
}

// One block tile of C, accumulated over the K range [k_begin, k_end).
//
// vec says whether a sixteen byte cp.async is legal for these operands, which
// needs both leading dimensions to be a multiple of eight halves and both base
// pointers sixteen byte aligned. It is uniform across the whole launch, so the
// branch on it costs a predicate and not a divergence. It is only read on the
// predicated path; the aligned path could not have been chosen without it.
//
// The caller owns the shared buffers and the barrier that separates one use of
// them from the next: a persistent CTA reuses them across tiles, and the loop
// below has no way to know that.
template <typename S, bool TAIL>
__device__ void tile_accumulate(const __half* __restrict__ a, const __half* __restrict__ b, int m,
                                int n, int k, int k_begin, int k_end, int block_row, int block_col,
                                bool vec, float (&acc)[S::kMTiles][S::kNTiles][4], __half* as,
                                __half* bs) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / S::kWarpsN;
    const int warp_n = warp % S::kWarpsN;

    const int span = k_end - k_begin;
    const int num_stages = (span + S::kBK - 1) / S::kBK;

    // Past the last K stage the call still commits, empty, so the number of
    // outstanding batches the wait below counts against stays constant to the
    // end of the loop.
    auto stage = [&](int buf, int stage_idx) {
        if (stage_idx < num_stages) {
            const int kk = k_begin + stage_idx * S::kBK;
#pragma unroll
            for (int i = 0; i < S::kACopies; ++i) {
                const int f = tid + i * S::kThreads;
                const int a_row = f / S::kARowChunks;
                const int a_col = (f % S::kARowChunks) * 8;
                __half* const dst =
                    &as[buf * S::kATileHalves + swizzle_a_tile<S::kBK>(a_row, a_col)];
                const long long src = static_cast<long long>(block_row + a_row) * k + kk + a_col;
                if constexpr (TAIL) {
                    const bool row_ok = block_row + a_row < m;
                    if (vec) {
                        const int bytes = row_ok ? tile_chunk_bytes(k_end - (kk + a_col)) : 0;
                        // A chunk with no bytes to copy reads nothing, so the
                        // source points at the base of A rather than past its
                        // end.
                        __pipeline_memcpy_async(dst, bytes > 0 ? (a + src) : a, sizeof(float4),
                                                static_cast<std::size_t>(16 - bytes));
                    } else {
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            dst[j] = (row_ok && kk + a_col + j < k_end) ? a[src + j]
                                                                        : __float2half(0.0f);
                        }
                    }
                } else {
                    __pipeline_memcpy_async(dst, a + src, sizeof(float4));
                }
            }
#pragma unroll
            for (int i = 0; i < S::kBCopies; ++i) {
                const int f = tid + i * S::kThreads;
                const int b_row = f / S::kBRowChunks;
                const int b_col = (f % S::kBRowChunks) * 8;
                __half* const dst =
                    &bs[buf * S::kBTileHalves + swizzle_b_tile<S::kBN>(b_row, b_col)];
                const long long src = static_cast<long long>(kk + b_row) * n + block_col + b_col;
                if constexpr (TAIL) {
                    const bool row_ok = kk + b_row < k_end;
                    if (vec) {
                        const int bytes = row_ok ? tile_chunk_bytes(n - (block_col + b_col)) : 0;
                        __pipeline_memcpy_async(dst, bytes > 0 ? (b + src) : b, sizeof(float4),
                                                static_cast<std::size_t>(16 - bytes));
                    } else {
#pragma unroll
                        for (int j = 0; j < 8; ++j) {
                            dst[j] = (row_ok && block_col + b_col + j < n) ? b[src + j]
                                                                           : __float2half(0.0f);
                        }
                    }
                } else {
                    __pipeline_memcpy_async(dst, b + src, sizeof(float4));
                }
            }
        }
        __pipeline_commit();
    };

#pragma unroll
    for (int s = 0; s < S::kStages - 1; ++s) {
        stage(s, s);
    }

    int read_buf = 0;
    int write_buf = S::kStages - 1;
    for (int s = 0; s < num_stages; ++s) {
        __pipeline_wait_prior(S::kStages - 2);
        __syncthreads();  // the only barrier in the K loop
        // Issued after the barrier: the buffer being filled here was last read at
        // iteration s-1, which that barrier has already fenced.
        stage(write_buf, s + S::kStages - 1);

        const __half* const a_tile = as + read_buf * S::kATileHalves;
        const __half* const b_tile = bs + read_buf * S::kBTileHalves;

        std::uint32_t b_frag[2][S::kNPairs][4];
        // The destination comes in by reference rather than by slot index so the
        // fragment array is never addressed dynamically and stays in registers.
        auto load_b = [&](std::uint32_t (&dst)[S::kNPairs][4], int k_off) {
#pragma unroll
            for (int nj = 0; nj < S::kNPairs; ++nj) {
                const int col_base = warp_n * S::kWarpN + nj * 16;
                tile_ldmatrix_x4_trans(dst[nj],
                                       &b_tile[swizzle_b_tile<S::kBN>(k_off + (lane % 16),
                                                                      col_base + (lane / 16) * 8)]);
            }
        };
        load_b(b_frag[0], 0);

#pragma unroll
        for (int ks = 0; ks < S::kKSub; ++ks) {
            const int k_off = ks * 16;
            std::uint32_t a_frag[S::kMTiles][4];
#pragma unroll
            for (int mi = 0; mi < S::kMTiles; ++mi) {
                const int row_base = warp_m * S::kWarpM + mi * 16;
                tile_ldmatrix_x4(a_frag[mi], &a_tile[swizzle_a_tile<S::kBK>(
                                                 row_base + (lane % 16), k_off + (lane / 16) * 8)]);
            }
            if (ks + 1 < S::kKSub) {
                load_b(b_frag[(ks + 1) & 1], (ks + 1) * 16);
            }
#pragma unroll
            for (int nj = 0; nj < S::kNPairs; ++nj) {
#pragma unroll
                for (int mi = 0; mi < S::kMTiles; ++mi) {
                    tile_mma_m16n8k16(acc[mi][2 * nj], a_frag[mi], b_frag[ks & 1][nj][0],
                                      b_frag[ks & 1][nj][1]);
                    tile_mma_m16n8k16(acc[mi][2 * nj + 1], a_frag[mi], b_frag[ks & 1][nj][2],
                                      b_frag[ks & 1][nj][3]);
                }
            }
        }

        read_buf = (read_buf + 1 == S::kStages) ? 0 : read_buf + 1;
        write_buf = (write_buf + 1 == S::kStages) ? 0 : write_buf + 1;
    }

    // The trailing commits are empty, but a caller that reuses these buffers has
    // to know every copy this thread issued has landed before it takes its
    // barrier.
    __pipeline_wait_prior(0);
}

// Where a lane's accumulator element (mi, ni, e) lands in the block tile.
struct TileCoord {
    int row;
    int col;
};

template <typename S>
__device__ inline TileCoord tile_coord(int warp_m, int warp_n, int lane, int mi, int ni, int e) {
    // The m16n8k16 accumulator hands each lane two adjacent columns of two rows
    // per tile: elements 0 and 1 at row group, 2 and 3 at row group + 8.
    const int group = lane / 4;
    const int tpair = lane % 4;
    return TileCoord{warp_m * S::kWarpM + mi * 16 + group + (e >= 2 ? 8 : 0),
                     warp_n * S::kWarpN + ni * 8 + 2 * tpair + (e & 1)};
}

/// Writes C = alpha * acc + beta * C for this block tile.
///
/// The aligned form keeps the float2 store the fragment layout allows, which is
/// the widest one available without exchanging accumulators through shared
/// memory first. The predicated form stores element by element under a bounds
/// test; it is the edge path, so the store width matters less than the tile it
/// keeps on the fast mainloop.
template <typename S, bool TAIL>
__device__ void tile_write_c(float* __restrict__ c, int m, int n, int block_row, int block_col,
                             const float (&acc)[S::kMTiles][S::kNTiles][4], float alpha,
                             float beta) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / S::kWarpsN;
    const int warp_n = warp % S::kWarpsN;

#pragma unroll
    for (int mi = 0; mi < S::kMTiles; ++mi) {
#pragma unroll
        for (int ni = 0; ni < S::kNTiles; ++ni) {
            if constexpr (TAIL) {
#pragma unroll
                for (int e = 0; e < 4; ++e) {
                    const TileCoord t = tile_coord<S>(warp_m, warp_n, lane, mi, ni, e);
                    const int row = block_row + t.row;
                    const int col = block_col + t.col;
                    if (row < m && col < n) {
                        float* const p = c + static_cast<long long>(row) * n + col;
                        const float v = alpha * acc[mi][ni][e];
                        // C is not read when beta is zero, per the BLAS contract,
                        // so an uninitialized or NaN C is legal input.
                        *p = (beta == 0.0f) ? v : v + beta * (*p);
                    }
                }
            } else {
                const TileCoord t = tile_coord<S>(warp_m, warp_n, lane, mi, ni, 0);
                const int row = block_row + t.row;
                const int col = block_col + t.col;
                float2* const p0 =
                    reinterpret_cast<float2*>(c + static_cast<long long>(row) * n + col);
                float2* const p1 =
                    reinterpret_cast<float2*>(c + static_cast<long long>(row + 8) * n + col);
                if (beta == 0.0f) {
                    *p0 = make_float2(alpha * acc[mi][ni][0], alpha * acc[mi][ni][1]);
                    *p1 = make_float2(alpha * acc[mi][ni][2], alpha * acc[mi][ni][3]);
                } else {
                    const float2 v0 = *p0;
                    const float2 v1 = *p1;
                    *p0 = make_float2(alpha * acc[mi][ni][0] + beta * v0.x,
                                      alpha * acc[mi][ni][1] + beta * v0.y);
                    *p1 = make_float2(alpha * acc[mi][ni][2] + beta * v1.x,
                                      alpha * acc[mi][ni][3] + beta * v1.y);
                }
                (void)m;  // the aligned form has no edge to test against
            }
        }
    }
}

/// Writes the raw accumulator into a dense BM by BN plane, no scaling applied.
/// This is a stream-K peer handing its slice of K to the CTA that owns the tile.
template <typename S>
__device__ void tile_store_partial(float* __restrict__ dst,
                                   const float (&acc)[S::kMTiles][S::kNTiles][4]) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / S::kWarpsN;
    const int warp_n = warp % S::kWarpsN;
#pragma unroll
    for (int mi = 0; mi < S::kMTiles; ++mi) {
#pragma unroll
        for (int ni = 0; ni < S::kNTiles; ++ni) {
            const TileCoord t = tile_coord<S>(warp_m, warp_n, lane, mi, ni, 0);
            float2* const p0 =
                reinterpret_cast<float2*>(dst + static_cast<long long>(t.row) * S::kBN + t.col);
            float2* const p1 =
                reinterpret_cast<float2*>(dst + static_cast<long long>(t.row + 8) * S::kBN + t.col);
            *p0 = make_float2(acc[mi][ni][0], acc[mi][ni][1]);
            *p1 = make_float2(acc[mi][ni][2], acc[mi][ni][3]);
        }
    }
}

/// Adds a peer's plane into this CTA's accumulator, element for element.
template <typename S>
__device__ void tile_add_partial(const float* __restrict__ src,
                                 float (&acc)[S::kMTiles][S::kNTiles][4]) {
    const int tid = static_cast<int>(threadIdx.x);
    const int warp = tid / 32;
    const int lane = tid % 32;
    const int warp_m = warp / S::kWarpsN;
    const int warp_n = warp % S::kWarpsN;
#pragma unroll
    for (int mi = 0; mi < S::kMTiles; ++mi) {
#pragma unroll
        for (int ni = 0; ni < S::kNTiles; ++ni) {
            const TileCoord t = tile_coord<S>(warp_m, warp_n, lane, mi, ni, 0);
            const float2 v0 = *reinterpret_cast<const float2*>(
                src + static_cast<long long>(t.row) * S::kBN + t.col);
            const float2 v1 = *reinterpret_cast<const float2*>(
                src + static_cast<long long>(t.row + 8) * S::kBN + t.col);
            acc[mi][ni][0] += v0.x;
            acc[mi][ni][1] += v0.y;
            acc[mi][ni][2] += v1.x;
            acc[mi][ni][3] += v1.y;
        }
    }
}

#endif  // __CUDA_ARCH__ >= 800

}  // namespace detail
}  // namespace ckl
