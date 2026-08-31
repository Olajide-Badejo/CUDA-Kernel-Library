#pragma once

// Shared memory address maps for the 128 by 128 by 32 mma mainloop.
//
// They live in a header rather than inside gemm_mma_opt.cu so the GEMM test
// suite can check them on the host. A swizzle is only correct if the same map is
// applied at the cp.async store and at the ldmatrix load, and only useful if it
// is a bijection whose lane bases land in distinct banks; both are properties of
// pure integer arithmetic, so they are cheaper to test on the CPU than to infer
// from a profiler counter.
//
// A shared bank window on this part is 128 bytes: 32 banks of 4 bytes. The B
// tile row is 128 halves = 256 bytes, so XOR-ing the 16 byte chunk index of a
// row with the row index spreads a column read across all 32 banks. The A tile
// row is only 32 halves = 64 bytes, half a window, so the same trick on a single
// row cannot separate eight lanes: it has four chunks to permute and needs
// eight. The A map therefore packs two logical rows into one 128 byte line
// first, then permutes the eight chunks of that line by the line index. The tile
// occupies the same 8 KB either way.

#if defined(__CUDACC__)
#define CKL_SWZ_FN __host__ __device__ inline
#else
#define CKL_SWZ_FN inline
#endif

namespace ckl {
namespace detail {

// The one tile shape this file describes. Part 10 owns the tile family sweep; if
// a second shape arrives these become template parameters.
inline constexpr int kSwzTileM = 128;
inline constexpr int kSwzTileN = 128;
inline constexpr int kSwzTileK = 32;

// A tile, viewed as kSwzTileM/2 lines of 64 halves. Row r of the logical
// 128 by 32 tile occupies chunks 0 to 3 of line r/2 when r is even and chunks
// 4 to 7 when r is odd, then the eight chunks of the line are rotated by the low
// three bits of the line index.
//
// Domain: r in [0, kSwzTileM), c in [0, kSwzTileK). Range: [0, 128 * 32).
CKL_SWZ_FN int swizzle_a(int r, int c) {
    const int chunk8 = ((r & 1) << 2) | (c >> 3);  // 0 to 7
    return (r >> 1) * 64 + ((chunk8 ^ ((r >> 1) & 7)) << 3) + (c & 7);
}

// B tile, row major with 128 halves per row: the row is already two bank
// windows wide, so the eight chunk index is XOR-ed with the low three bits of
// the row index directly.
//
// Domain: r in [0, kSwzTileK), c in [0, kSwzTileN). Range: [0, 32 * 128).
CKL_SWZ_FN int swizzle_b(int r, int c) {
    return r * kSwzTileN + ((((c >> 3) ^ (r & 7)) << 3) + (c & 7));
}

}  // namespace detail
}  // namespace ckl
