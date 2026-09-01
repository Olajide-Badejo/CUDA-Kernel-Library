#pragma once

// Device side pieces every FFT kernel in this directory shares: complex
// arithmetic, the factored twiddle lookup, and the radix 2, 4 and 8 butterflies.
//
// Not installed and not part of the public API. It exists so the butterflies are
// written once: a radix 8 DFT written three times is a radix 8 DFT that will
// disagree with itself.
//
// Sign convention. The tables hold exp(-2 pi i t / n), which is the forward
// kernel. `dsign` is +1 for a forward transform and -1 for an inverse one; the
// twiddle lookup conjugates by multiplying the imaginary part by dsign, and the
// multiplication by i inside a butterfly uses the opposite sign, because
// exp(-2 pi i / 4) is -i.

#include <cuda_runtime.h>

namespace ckl {
namespace detail {

__device__ __forceinline__ float2 cadd(float2 a, float2 b) {
    return make_float2(a.x + b.x, a.y + b.y);
}

__device__ __forceinline__ float2 csub(float2 a, float2 b) {
    return make_float2(a.x - b.x, a.y - b.y);
}

__device__ __forceinline__ float2 cmul(float2 a, float2 b) {
    return make_float2(a.x * b.x - a.y * b.y, a.x * b.y + a.y * b.x);
}

__device__ __forceinline__ float2 cscale(float2 a, float s) {
    return make_float2(a.x * s, a.y * s);
}

// s * i * z, with s either +1 or -1.
__device__ __forceinline__ float2 cmul_i(float2 z, float s) {
    return make_float2(-s * z.y, s * z.x);
}

// exp(-2 pi i idx / n) for a forward transform, conjugated for an inverse one.
// idx must be in [0, n). The table is factored, so this is two loads and one
// complex multiply; see ckl::FftTwiddles for why.
__device__ __forceinline__ float2 twiddle(const float2* __restrict__ hi,
                                          const float2* __restrict__ lo, int lo_bits, int idx,
                                          float dsign) {
    const float2 a = hi[idx >> lo_bits];
    const float2 b = lo[idx & ((1 << lo_bits) - 1)];
    float2 w = cmul(a, b);
    w.y *= dsign;
    return w;
}

// X[m] = sum_r u[r] * exp(-2 pi i r m / 4), conjugated for an inverse transform.
__device__ __forceinline__ void dft4(float2* u, float dsign) {
    const float2 t0 = cadd(u[0], u[2]);
    const float2 t1 = csub(u[0], u[2]);
    const float2 t2 = cadd(u[1], u[3]);
    const float2 t3 = csub(u[1], u[3]);
    const float2 it3 = cmul_i(t3, -dsign);
    u[0] = cadd(t0, t2);
    u[1] = cadd(t1, it3);
    u[2] = csub(t0, t2);
    u[3] = csub(t1, it3);
}

// X[m] = sum_r u[r] * exp(-2 pi i r m / 8), by two radix 4 transforms and the
// four eighth roots between them. The constant is cos(pi/4) = sin(pi/4).
__device__ __forceinline__ void dft8(float2* u, float dsign) {
    constexpr float kRoot = 0.70710678118654752f;
    float2 e[4] = {u[0], u[2], u[4], u[6]};
    float2 o[4] = {u[1], u[3], u[5], u[7]};
    dft4(e, dsign);
    dft4(o, dsign);
    // exp(-2 pi i m / 8) for m = 0, 1, 2, 3, conjugated by dsign.
    const float2 w1 = make_float2(kRoot, -kRoot * dsign);
    const float2 w2 = make_float2(0.0f, -dsign);
    const float2 w3 = make_float2(-kRoot, -kRoot * dsign);
    const float2 p0 = o[0];
    const float2 p1 = cmul(o[1], w1);
    const float2 p2 = cmul(o[2], w2);
    const float2 p3 = cmul(o[3], w3);
    u[0] = cadd(e[0], p0);
    u[1] = cadd(e[1], p1);
    u[2] = cadd(e[2], p2);
    u[3] = cadd(e[3], p3);
    u[4] = csub(e[0], p0);
    u[5] = csub(e[1], p1);
    u[6] = csub(e[2], p2);
    u[7] = csub(e[3], p3);
}

// One Stockham stage of radix R, for a single transform.
//
// Thread j owns one butterfly of the n / R that a stage of radix R has. It reads
// elements j, j + n/R, ... of the source, twists all but the first by
// exp(-2 pi i r k / (R p)) with k = j mod p, runs the radix R DFT, and writes the
// R results to R (j - k) + k + r p in the destination. Both the reads and the
// writes are contiguous runs and the digit permutation is inside that output
// index, which is what autosort means and why no bit reversal pass exists here.
//
// The three are written once and called from two translation units, because a
// radix 8 DFT written twice is a radix 8 DFT that will disagree with itself.

__device__ __forceinline__ void stage_radix2(const float2* __restrict__ src,
                                             float2* __restrict__ dst, int j, int n, int p,
                                             const float2* __restrict__ hi,
                                             const float2* __restrict__ lo, int lo_bits,
                                             float dsign) {
    const int span = n >> 1;
    const int k = j & (p - 1);
    const int step = span / p;
    const float2 u0 = src[j];
    const float2 u1 = cmul(src[j + span], twiddle(hi, lo, lo_bits, k * step, dsign));
    const int j2 = ((j - k) << 1) + k;
    dst[j2] = cadd(u0, u1);
    dst[j2 + p] = csub(u0, u1);
}

__device__ __forceinline__ void stage_radix4(const float2* __restrict__ src,
                                             float2* __restrict__ dst, int j, int n, int p,
                                             const float2* __restrict__ hi,
                                             const float2* __restrict__ lo, int lo_bits,
                                             float dsign) {
    const int span = n >> 2;
    const int k = j & (p - 1);
    const int step = span / p;
    float2 u[4];
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        u[r] = src[j + r * span];
    }
#pragma unroll
    for (int r = 1; r < 4; ++r) {
        u[r] = cmul(u[r], twiddle(hi, lo, lo_bits, r * k * step, dsign));
    }
    dft4(u, dsign);
    const int j2 = ((j - k) << 2) + k;
#pragma unroll
    for (int r = 0; r < 4; ++r) {
        dst[j2 + r * p] = u[r];
    }
}

__device__ __forceinline__ void stage_radix8(const float2* __restrict__ src,
                                             float2* __restrict__ dst, int j, int n, int p,
                                             const float2* __restrict__ hi,
                                             const float2* __restrict__ lo, int lo_bits,
                                             float dsign) {
    const int span = n >> 3;
    const int k = j & (p - 1);
    const int step = span / p;
    float2 u[8];
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        u[r] = src[j + r * span];
    }
#pragma unroll
    for (int r = 1; r < 8; ++r) {
        u[r] = cmul(u[r], twiddle(hi, lo, lo_bits, r * k * step, dsign));
    }
    dft8(u, dsign);
    const int j2 = ((j - k) << 3) + k;
#pragma unroll
    for (int r = 0; r < 8; ++r) {
        dst[j2 + r * p] = u[r];
    }
}

}  // namespace detail
}  // namespace ckl
