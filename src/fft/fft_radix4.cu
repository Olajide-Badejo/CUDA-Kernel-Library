// Rung 3: wider butterflies. Radix 4 halves the stage count of radix 2 and
// radix 8 cuts it to a third, so both cut the global memory traffic by the same
// factor, and both pay for it in registers: a radix 8 butterfly holds eight
// complex values and seven twiddles live at once. Registers rather than shared
// memory are therefore expected to be what limits occupancy here, which is why
// docs/fft.md asks for radix 2, 4 and 8 in one table with registers per thread
// printed beside throughput.
//
// log2(n) is not always divisible by 2 or by 3, so both ladders start with one
// narrower stage when they have to. That leading stage is a real mixed radix
// step, not a special case bolted on: Stockham composes stages of any radix as
// long as the running stride p multiplies up to n.
//
//   radix 4: one radix 2 stage first when log2(n) is odd.
//   radix 8: one radix 2 stage when log2(n) mod 3 is 1, one radix 4 stage when
//            it is 2, and nothing when it is 0.

#include "ckl/cuda_check.hpp"
#include "ckl/fft.hpp"

#include "fft_device.cuh"
#include "fft_internal.hpp"

namespace ckl {

namespace {

constexpr int kBlock = 256;

int grid_of(long long total, int block) {
    return static_cast<int>((total + block - 1) / block);
}

int log2_exact(int v) {
    int b = 0;
    while ((1 << b) < v) {
        ++b;
    }
    return b;
}

__global__ void radix2_stage(const float2* __restrict__ in, float2* __restrict__ out,
                             const float2* __restrict__ hi, const float2* __restrict__ lo,
                             int lo_bits, int n, int log2_span, int p, float dsign,
                             long long total) {
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= total) {
        return;
    }
    const int j = static_cast<int>(t & ((n >> 1) - 1));
    const long long b = t >> log2_span;
    detail::stage_radix2(in + b * n, out + b * n, j, n, p, hi, lo, lo_bits, dsign);
}

__global__ void radix4_stage(const float2* __restrict__ in, float2* __restrict__ out,
                             const float2* __restrict__ hi, const float2* __restrict__ lo,
                             int lo_bits, int n, int log2_span, int p, float dsign,
                             long long total) {
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= total) {
        return;
    }
    const int j = static_cast<int>(t & ((n >> 2) - 1));
    const long long b = t >> log2_span;
    detail::stage_radix4(in + b * n, out + b * n, j, n, p, hi, lo, lo_bits, dsign);
}

__global__ void radix8_stage(const float2* __restrict__ in, float2* __restrict__ out,
                             const float2* __restrict__ hi, const float2* __restrict__ lo,
                             int lo_bits, int n, int log2_span, int p, float dsign,
                             long long total) {
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= total) {
        return;
    }
    const int j = static_cast<int>(t & ((n >> 3) - 1));
    const long long b = t >> log2_span;
    detail::stage_radix8(in + b * n, out + b * n, j, n, p, hi, lo, lo_bits, dsign);
}

void run_ladder(int max_radix, const float2* in, float2* out, float2* work, const FftTwiddles& tw,
                int n, int batch, FftDirection dir, cudaStream_t stream) {
    int radix_of[32];
    const int stages = detail::fft_ladder(log2_exact(n), max_radix, radix_of);
    const float dsign = dir == FftDirection::kForward ? 1.0f : -1.0f;
    const float2* src = in;
    int p = 1;
    for (int i = 0; i < stages; ++i) {
        // The last stage has to land in out, so the destination is decided by how
        // many stages are left rather than by how many have run.
        float2* dst = (((stages - 1 - i) & 1) == 0) ? out : work;
        const int radix = radix_of[i];
        const int span = n / radix;
        const long long total = static_cast<long long>(span) * batch;
        const int log2_span = log2_exact(span);
        if (radix == 2) {
            radix2_stage<<<grid_of(total, kBlock), kBlock, 0, stream>>>(
                src, dst, tw.hi, tw.lo, tw.lo_bits, n, log2_span, p, dsign, total);
        } else if (radix == 4) {
            radix4_stage<<<grid_of(total, kBlock), kBlock, 0, stream>>>(
                src, dst, tw.hi, tw.lo, tw.lo_bits, n, log2_span, p, dsign, total);
        } else {
            radix8_stage<<<grid_of(total, kBlock), kBlock, 0, stream>>>(
                src, dst, tw.hi, tw.lo, tw.lo_bits, n, log2_span, p, dsign, total);
        }
        CKL_CUDA_LAST_ERROR(false);
        src = dst;
        p *= radix;
    }
}

}  // namespace

namespace detail {

int fft_ladder(int log2n, int max_radix, int* radix_out) {
    int count = 0;
    int bits = log2n;
    const auto push = [&](int r) {
        if (radix_out != nullptr) {
            radix_out[count] = r;
        }
        ++count;
    };
    if (max_radix >= 8) {
        // A radix 8 ladder needs log2(n) divisible by 3, so it opens with the
        // narrower stage that makes the rest divisible.
        if (bits % 3 == 1) {
            push(2);
            bits -= 1;
        } else if (bits % 3 == 2) {
            push(4);
            bits -= 2;
        }
        for (int i = 0; i < bits / 3; ++i) {
            push(8);
        }
        return count;
    }
    if (max_radix >= 4) {
        if ((bits & 1) != 0) {
            push(2);
            bits -= 1;
        }
        for (int i = 0; i < bits / 2; ++i) {
            push(4);
        }
        return count;
    }
    for (int i = 0; i < bits; ++i) {
        push(2);
    }
    return count;
}

}  // namespace detail

void fft_radix4_global(const float2* in, float2* out, float2* work, const FftTwiddles& tw, int n,
                       int batch, FftDirection dir, cudaStream_t stream) {
    if (n <= 1 || batch <= 0) {
        return;
    }
    run_ladder(4, in, out, work, tw, n, batch, dir, stream);
}

void fft_radix8_global(const float2* in, float2* out, float2* work, const FftTwiddles& tw, int n,
                       int batch, FftDirection dir, cudaStream_t stream) {
    if (n <= 1 || batch <= 0) {
        return;
    }
    run_ladder(8, in, out, work, tw, n, batch, dir, stream);
}

}  // namespace ckl
