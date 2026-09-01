// Rung 1: Stockham autosort radix 2, one kernel launch per stage, everything in
// global memory. And rung 5: the real to complex packing and its untangle pass,
// which live here because they are radix 2 bookkeeping rather than a butterfly.
//
// The stage kernel is the whole of rung 1. At stride p a thread owns one
// butterfly: it reads element j and element j + n/2 of the source buffer, twists
// the second by exp(-2 pi i k / 2p) with k = j mod p, and writes the sum and the
// difference to 2(j-k)+k and 2(j-k)+k+p of the destination buffer. Both reads and
// both writes are contiguous runs, and the digit permutation has been folded into
// that output index, which is the entire reason this project does not implement
// Cooley-Tukey with a separate bit reversal pass. That pass is a scatter across
// the whole array, one sector per element at large n, and it costs a full memory
// round trip on a family that is already memory bound.
//
// The price is the second buffer. log2(n) launches each read n and write n
// complex points, so this rung moves 16 n log2(n) bytes and every later rung is
// an attack on that factor of log2(n).

#include "ckl/cuda_check.hpp"
#include "ckl/fft.hpp"

#include "fft_device.cuh"
#include "fft_internal.hpp"

namespace ckl {

namespace {

constexpr int kBlock = 256;

int grid_of(long long total, int block) {
    const long long g = (total + block - 1) / block;
    return static_cast<int>(g);
}

int log2_exact(int v) {
    int b = 0;
    while ((1 << b) < v) {
        ++b;
    }
    return b;
}

__global__ void stockham_radix2_stage(const float2* __restrict__ in, float2* __restrict__ out,
                                      const float2* __restrict__ hi, const float2* __restrict__ lo,
                                      int lo_bits, int n, int log2_half, int p, float dsign,
                                      long long total) {
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= total) {
        return;
    }
    const int j = static_cast<int>(t & ((n >> 1) - 1));
    const long long b = t >> log2_half;
    detail::stage_radix2(in + b * n, out + b * n, j, n, p, hi, lo, lo_bits, dsign);
}

__global__ void scale_kernel(float2* data, long long count, float scale) {
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= count) {
        return;
    }
    data[t] = detail::cscale(data[t], scale);
}

__global__ void pack_real_kernel(const float* __restrict__ in, float2* __restrict__ z,
                                 long long count) {
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= count) {
        return;
    }
    z[t] = make_float2(in[2 * t], in[2 * t + 1]);
}

__global__ void unpack_real_kernel(const float2* __restrict__ z, float* __restrict__ out,
                                   long long count, float scale) {
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= count) {
        return;
    }
    const float2 v = z[t];
    out[2 * t] = v.x * scale;
    out[2 * t + 1] = v.y * scale;
}

// Rung 5, second half. Z is the length n transform of the 2n reals read as n
// complex points. Bin k of the real spectrum is
//   Xe[k] + exp(-2 pi i k / 2n) * Xo[k]
// with Xe[k] = (Z[k] + conj(Z[n-k])) / 2 the transform of the even samples and
// Xo[k] = -i (Z[k] - conj(Z[n-k])) / 2 the transform of the odd ones. The bin at
// k and the bin at n-k need each other, which is why this is a separate pass and
// not an epilogue on the transform.
__global__ void r2c_untangle_kernel(const float2* __restrict__ z, float2* __restrict__ out,
                                    const float2* __restrict__ hi, const float2* __restrict__ lo,
                                    int lo_bits, int n, long long total) {
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= total) {
        return;
    }
    const int bins = n + 1;
    const int k = static_cast<int>(t % bins);
    const long long b = t / bins;
    const float2* src = z + b * n;

    const float2 zk = src[k == n ? 0 : k];
    const float2 zn = src[(n - k) & (n - 1)];
    const float2 zc = make_float2(zn.x, -zn.y);
    const float2 even = detail::cscale(detail::cadd(zk, zc), 0.5f);
    const float2 diff = detail::cscale(detail::csub(zk, zc), 0.5f);
    // -i * diff, which is the forward convention.
    const float2 odd = detail::cmul_i(diff, -1.0f);
    const float2 w = detail::twiddle(hi, lo, lo_bits, k, 1.0f);
    out[b * bins + k] = detail::cadd(even, detail::cmul(w, odd));
}

// The inverse of the untangle. Xe and Xo come back out of X[k] and conj(X[n-k]),
// and the packed spectrum is Z[k] = Xe[k] + i Xo[k].
__global__ void c2r_pack_kernel(const float2* __restrict__ spectrum, float2* __restrict__ z,
                                const float2* __restrict__ hi, const float2* __restrict__ lo,
                                int lo_bits, int n, int log2n, long long total) {
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= total) {
        return;
    }
    const int bins = n + 1;
    const int k = static_cast<int>(t & (n - 1));
    const long long b = t >> log2n;
    const float2* src = spectrum + b * bins;

    const float2 xk = src[k];
    const float2 xn = src[n - k];
    const float2 xc = make_float2(xn.x, -xn.y);
    const float2 even = detail::cscale(detail::cadd(xk, xc), 0.5f);
    const float2 diff = detail::cscale(detail::csub(xk, xc), 0.5f);
    // diff is exp(-2 pi i k / 2n) * Xo[k], so undo the twist with the conjugate.
    const float2 w = detail::twiddle(hi, lo, lo_bits, k, -1.0f);
    const float2 odd = detail::cmul(diff, w);
    z[b * n + k] = detail::cadd(even, detail::cmul_i(odd, 1.0f));
}

// The fast_twiddles rung. Two tables filled with __sincosf, whose argument
// reduction gives roughly 2^-21 relative accuracy: about sixteen times worse than
// rounding a double precision sine to FP32, and visible in the round trip error.
// This is a labelled rung and never the default.
__global__ void fill_twiddles_fast_kernel(float2* __restrict__ hi, float2* __restrict__ lo,
                                          int hi_len, int lo_len, float step_hi, float step_lo) {
    const int t =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    float s = 0.0f;
    float c = 0.0f;
    if (t < hi_len) {
        __sincosf(step_hi * static_cast<float>(t), &s, &c);
        hi[t] = make_float2(c, s);
    }
    if (t < lo_len) {
        __sincosf(step_lo * static_cast<float>(t), &s, &c);
        lo[t] = make_float2(c, s);
    }
}

}  // namespace

namespace detail {

void fft_fill_twiddles_fast(float2* hi, float2* lo, int hi_len, int lo_len, int n,
                            cudaStream_t stream) {
    constexpr float kTwoPi = 6.28318530717958647692f;
    const int total = hi_len > lo_len ? hi_len : lo_len;
    if (total <= 0) {
        return;
    }
    // hi[q] is exp(-2 pi i q * lo_len / n) and lo[r] is exp(-2 pi i r / n).
    const float step_hi = -kTwoPi * static_cast<float>(lo_len) / static_cast<float>(n);
    const float step_lo = -kTwoPi / static_cast<float>(n);
    fill_twiddles_fast_kernel<<<grid_of(total, kBlock), kBlock, 0, stream>>>(hi, lo, hi_len, lo_len,
                                                                             step_hi, step_lo);
    CKL_CUDA_LAST_ERROR(false);
}

}  // namespace detail

void fft_radix2_global(const float2* in, float2* out, float2* work, const FftTwiddles& tw, int n,
                       int batch, FftDirection dir, cudaStream_t stream) {
    if (n <= 1 || batch <= 0) {
        return;
    }
    const int stages = log2_exact(n);
    const int log2_half = stages - 1;
    const long long total = (static_cast<long long>(n) >> 1) * batch;
    const float dsign = dir == FftDirection::kForward ? 1.0f : -1.0f;

    const float2* src = in;
    for (int i = 0, p = 1; i < stages; ++i, p <<= 1) {
        // The last stage has to land in out, so the destination of stage i is
        // decided by how many stages are left rather than by how many have run.
        float2* dst = (((stages - 1 - i) & 1) == 0) ? out : work;
        stockham_radix2_stage<<<grid_of(total, kBlock), kBlock, 0, stream>>>(
            src, dst, tw.hi, tw.lo, tw.lo_bits, n, log2_half, p, dsign, total);
        CKL_CUDA_LAST_ERROR(false);
        src = dst;
    }
}

void fft_scale(float2* data, long long count, float scale, cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    scale_kernel<<<grid_of(count, kBlock), kBlock, 0, stream>>>(data, count, scale);
    CKL_CUDA_LAST_ERROR(false);
}

void fft_pack_real(const float* in, float2* z, int n, int batch, cudaStream_t stream) {
    const long long count = static_cast<long long>(n) * batch;
    if (count <= 0) {
        return;
    }
    pack_real_kernel<<<grid_of(count, kBlock), kBlock, 0, stream>>>(in, z, count);
    CKL_CUDA_LAST_ERROR(false);
}

void fft_unpack_real(const float2* z, float* out, int n, int batch, float scale,
                     cudaStream_t stream) {
    const long long count = static_cast<long long>(n) * batch;
    if (count <= 0) {
        return;
    }
    unpack_real_kernel<<<grid_of(count, kBlock), kBlock, 0, stream>>>(z, out, count, scale);
    CKL_CUDA_LAST_ERROR(false);
}

void fft_r2c_untangle(const float2* z, float2* out, const FftTwiddles& tw_double, int n, int batch,
                      cudaStream_t stream) {
    const long long total = static_cast<long long>(n + 1) * batch;
    if (total <= 0) {
        return;
    }
    r2c_untangle_kernel<<<grid_of(total, kBlock), kBlock, 0, stream>>>(
        z, out, tw_double.hi, tw_double.lo, tw_double.lo_bits, n, total);
    CKL_CUDA_LAST_ERROR(false);
}

void fft_c2r_pack(const float2* spectrum, float2* z, const FftTwiddles& tw_double, int n, int batch,
                  cudaStream_t stream) {
    const long long total = static_cast<long long>(n) * batch;
    if (total <= 0) {
        return;
    }
    c2r_pack_kernel<<<grid_of(total, kBlock), kBlock, 0, stream>>>(
        spectrum, z, tw_double.hi, tw_double.lo, tw_double.lo_bits, n, log2_exact(n), total);
    CKL_CUDA_LAST_ERROR(false);
}

}  // namespace ckl
