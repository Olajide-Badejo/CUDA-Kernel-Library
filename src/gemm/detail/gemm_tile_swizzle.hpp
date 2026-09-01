#pragma once

// Shared memory address maps for the tile family mainloop, parameterized on the
// tile extents.
//
// gemm_swizzle.hpp describes the single 128 by 128 by 32 shape the round 10
// mainloop was written against. The family needs the same two properties at five
// more shapes, so the maps move here as templates and the fixed shape header
// keeps its own copies; the GEMM tensor suite asserts that the two agree at
// 128 by 128 by 32, so the duplication cannot drift.
//
// The requirements are unchanged. A map is correct if the same one is applied at
// the cp.async store and at the ldmatrix load, useful if it is a bijection whose
// lane bases land in distinct banks, and usable at all only if it moves eight
// halves at a time as one unit, because that is the granularity of both
// instructions. All three are integer arithmetic, so the tests check them on the
// host.
//
// A bank window is 128 bytes: 32 banks of 4. A B row of BN halves is at least
// two windows wide for every BN in the family, so XOR-ing the eight chunk index
// of a row with the low three bits of the row index spreads a column read across
// the banks. An A row of 32 halves is only 64 bytes, half a window, so at BK 32
// the map first packs two logical rows into one 128 byte line and permutes the
// eight chunks of that line by the line index. At BK 64 an A row is already a
// full window and the plain B style map applies.

#if defined(__CUDACC__)
#define CKL_TILE_SWZ_FN __host__ __device__ inline
#else
#define CKL_TILE_SWZ_FN inline
#endif

namespace ckl {
namespace detail {

/// A tile of a mainloop with K step BK, viewed as BM rows of BK halves.
/// Domain: r in [0, BM), c in [0, BK). Range: [0, BM * BK).
template <int BK>
CKL_TILE_SWZ_FN int swizzle_a_tile(int r, int c) {
    static_assert(BK == 32 || BK % 64 == 0,
                  "the A map covers a 64 byte row (BK 32) or whole 128 byte rows (BK a multiple "
                  "of 64)");
    if constexpr (BK == 32) {
        const int chunk8 = ((r & 1) << 2) | (c >> 3);  // 0 to 7
        return (r >> 1) * 64 + ((chunk8 ^ ((r >> 1) & 7)) << 3) + (c & 7);
    } else {
        return r * BK + ((((c >> 3) ^ (r & 7)) << 3) + (c & 7));
    }
}

/// B tile of a mainloop with N step BN, row major with BN halves per row.
/// Domain: r in [0, BK), c in [0, BN). Range: [0, BK * BN).
template <int BN>
CKL_TILE_SWZ_FN int swizzle_b_tile(int r, int c) {
    static_assert(BN % 64 == 0, "a B row has to hold at least eight 16 byte chunks to permute");
    return r * BN + ((((c >> 3) ^ (r & 7)) << 3) + (c & 7));
}

}  // namespace detail
}  // namespace ckl
