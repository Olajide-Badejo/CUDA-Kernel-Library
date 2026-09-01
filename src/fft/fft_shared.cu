// Rung 2, the whole transform resident in shared memory, and rung 4, the four
// step form built out of it.
//
// The budget, derived. A complex FP32 point is 8 bytes and Stockham needs two
// buffers, so a resident transform of n points costs 16n bytes of shared memory.
// 2^12 costs 65536 bytes and fits inside the 101376 byte per block opt in limit
// on this part; 2^13 costs 131072 and does not. Holding one radix stage in
// registers and reusing a single shared buffer would bring 2^13 back inside at
// 65536 bytes, and that trick is not implemented here: above 2^12 the four step
// form takes over. FftPlan::shared_resident_max is 4096 and says so.
//
// Either way a 64 KB block is one block per SM against the 100 KB per SM limit,
// so a benchmark of a small transform has to batch until all 48 SMs have work.
// A single unbatched 2^12 transform measures launch overhead, not this kernel.
//
// The four step form factors n = n1 * n2 with both factors at or below the
// resident bound and runs
//
//   transpose, batched size n2 transforms, transpose, batched size n1 transforms,
//   transpose
//
// with the cross twiddle exp(-2 pi i n1 k2 / n) folded into the store epilogue of
// the first batched transform, which is why it does not appear as a pass of its
// own. Five passes over the array in total, against log2(n) for rung 1: at 2^24
// that is 5 against 24.

#include <mutex>
#include <string>

#include "ckl/cuda_check.hpp"
#include "ckl/fft.hpp"

#include "fft_device.cuh"
#include "fft_internal.hpp"

namespace ckl {

namespace {

constexpr int kBlock = 256;
constexpr int kSharedMax = 4096;
// Below this a block gets its shared memory without asking. Above it, and up to
// the 101376 byte per block ceiling of this part, it has to opt in.
constexpr int kSharedNoOptIn = 48 * 1024;

// One block per transform. Strides let the same kernel serve a contiguous batch
// and a column pass, which is what removes the transposes from the 2D driver.
__global__ void shared_kernel(const float2* __restrict__ in, float2* __restrict__ out,
                              const float2* __restrict__ hi, const float2* __restrict__ lo,
                              int lo_bits, int n, float dsign, int in_stride,
                              long long in_batch_stride, int out_stride, long long out_batch_stride,
                              const float2* __restrict__ mul, float scale) {
    extern __shared__ float2 sh[];
    float2* a = sh;
    float2* b = sh + n;
    const long long block = blockIdx.x;
    const float2* src = in + block * in_batch_stride;
    float2* dst = out + block * out_batch_stride;

    for (int i = static_cast<int>(threadIdx.x); i < n; i += blockDim.x) {
        a[i] = src[static_cast<long long>(i) * in_stride];
    }
    __syncthreads();

    const int half = n >> 1;
    for (int p = 1; p < n; p <<= 1) {
        for (int j = static_cast<int>(threadIdx.x); j < half; j += blockDim.x) {
            detail::stage_radix2(a, b, j, n, p, hi, lo, lo_bits, dsign);
        }
        __syncthreads();
        float2* t = a;
        a = b;
        b = t;
    }

    for (int i = static_cast<int>(threadIdx.x); i < n; i += blockDim.x) {
        float2 v = a[i];
        if (mul != nullptr) {
            v = detail::cmul(v, mul[i]);
        }
        if (scale != 1.0f) {
            v = detail::cscale(v, scale);
        }
        dst[static_cast<long long>(i) * out_stride] = v;
    }
}

// The same transform with the four step cross twiddle in the store epilogue. The
// block index inside one outer transform is n1, the element index is k2, and the
// factor is exp(-2 pi i n1 k2 / n_full), which is under n_full because
// (n1 - 1) (n2 - 1) is.
__global__ void shared_cross_kernel(const float2* __restrict__ in, float2* __restrict__ out,
                                    const float2* __restrict__ hi, const float2* __restrict__ lo,
                                    int lo_bits, const float2* __restrict__ fhi,
                                    const float2* __restrict__ flo, int flo_bits, int n,
                                    int n1_count, float dsign) {
    extern __shared__ float2 sh[];
    float2* a = sh;
    float2* b = sh + n;
    const long long block = blockIdx.x;
    const float2* src = in + block * n;
    float2* dst = out + block * n;
    const int row = static_cast<int>(block % n1_count);

    for (int i = static_cast<int>(threadIdx.x); i < n; i += blockDim.x) {
        a[i] = src[i];
    }
    __syncthreads();

    const int half = n >> 1;
    for (int p = 1; p < n; p <<= 1) {
        for (int j = static_cast<int>(threadIdx.x); j < half; j += blockDim.x) {
            detail::stage_radix2(a, b, j, n, p, hi, lo, lo_bits, dsign);
        }
        __syncthreads();
        float2* t = a;
        a = b;
        b = t;
    }

    for (int i = static_cast<int>(threadIdx.x); i < n; i += blockDim.x) {
        dst[i] = detail::cmul(a[i], detail::twiddle(fhi, flo, flo_bits, row * i, dsign));
    }
}

// The dynamic shared memory opt in, granted once. A driver call on every launch
// would be setup inside a timed region, and the plans call fft_shared_prepare at
// construction so the compare below is all a launch pays.
std::mutex& grant_mutex() {
    static std::mutex m;
    return m;
}

int& granted_bytes() {
    static int granted = kSharedNoOptIn;
    return granted;
}

void grant(int bytes) {
    if (bytes <= granted_bytes()) {
        return;
    }
    const std::lock_guard<std::mutex> lock(grant_mutex());
    if (bytes <= granted_bytes()) {
        return;
    }
    CKL_CUDA_CHECK(
        cudaFuncSetAttribute(shared_kernel, cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    CKL_CUDA_CHECK(cudaFuncSetAttribute(shared_cross_kernel,
                                        cudaFuncAttributeMaxDynamicSharedMemorySize, bytes));
    granted_bytes() = bytes;
}

int shared_bytes_for(int n) {
    return 2 * n * static_cast<int>(sizeof(float2));
}

void require_resident(int n, const char* what) {
    if (n > kSharedMax) {
        throw Error(Status::kNotSupported,
                    std::string(what) + ": " + std::to_string(n) + " points need " +
                        std::to_string(shared_bytes_for(n)) +
                        " bytes of shared memory for two ping pong buffers, above the 101376 byte "
                        "per block limit of this part. The bound is 4096 points; above it the four "
                        "step form takes over.");
    }
}

}  // namespace

void fft_shared_resident(const float2* in, float2* out, const FftTwiddles& tw, int n, int batch,
                         FftDirection dir, const float2* epilogue_mul, float epilogue_scale,
                         cudaStream_t stream) {
    if (n <= 1 || batch <= 0) {
        return;
    }
    require_resident(n, "fft_shared_resident");
    const int bytes = shared_bytes_for(n);
    grant(bytes);
    const float dsign = dir == FftDirection::kForward ? 1.0f : -1.0f;
    shared_kernel<<<batch, kBlock, bytes, stream>>>(in, out, tw.hi, tw.lo, tw.lo_bits, n, dsign, 1,
                                                    n, 1, n, epilogue_mul, epilogue_scale);
    CKL_CUDA_LAST_ERROR(false);
}

namespace detail {

void fft_shared_prepare(int bytes) {
    grant(bytes);
}

void fft_shared_strided(const float2* in, float2* out, const FftTwiddles& tw, int n, int batch,
                        FftDirection dir, int in_stride, long long in_batch_stride, int out_stride,
                        long long out_batch_stride, cudaStream_t stream) {
    if (n <= 1 || batch <= 0) {
        return;
    }
    require_resident(n, "fft_shared_strided");
    const int bytes = shared_bytes_for(n);
    grant(bytes);
    const float dsign = dir == FftDirection::kForward ? 1.0f : -1.0f;
    shared_kernel<<<batch, kBlock, bytes, stream>>>(in, out, tw.hi, tw.lo, tw.lo_bits, n, dsign,
                                                    in_stride, in_batch_stride, out_stride,
                                                    out_batch_stride, nullptr, 1.0f);
    CKL_CUDA_LAST_ERROR(false);
}

}  // namespace detail

void fft_four_step(const float2* in, float2* out, float2* work, const FftTwiddles& tw_n1,
                   const FftTwiddles& tw_n2, const FftTwiddles& tw_full, int n1, int n2, int batch,
                   FftDirection dir, FftTranspose transpose, const float2* epilogue_mul,
                   float epilogue_scale, cudaStream_t stream) {
    if (n1 <= 0 || n2 <= 0 || batch <= 0) {
        return;
    }
    require_resident(n1, "fft_four_step n1");
    require_resident(n2, "fft_four_step n2");
    const int bytes = shared_bytes_for(n1 > n2 ? n1 : n2);
    grant(bytes);
    const float dsign = dir == FftDirection::kForward ? 1.0f : -1.0f;

    // The input holds x[n] with n = i1 + i2 * n1, which is an n2 by n1 matrix
    // read row major. The first transpose turns it into the n1 by n2 matrix whose
    // rows are the size n2 sub transforms.
    fft_transpose(in, out, n2, n1, batch, transpose, nullptr, 1.0f, stream);

    // Size n2 transforms, one per row, with the cross twiddle in the epilogue.
    shared_cross_kernel<<<batch * n1, kBlock, shared_bytes_for(n2), stream>>>(
        out, work, tw_n2.hi, tw_n2.lo, tw_n2.lo_bits, tw_full.hi, tw_full.lo, tw_full.lo_bits, n2,
        n1, dsign);
    CKL_CUDA_LAST_ERROR(false);

    // Transpose so the size n1 sub transforms become rows in their turn.
    fft_transpose(work, out, n1, n2, batch, transpose, nullptr, 1.0f, stream);

    shared_kernel<<<batch * n2, kBlock, shared_bytes_for(n1), stream>>>(
        out, work, tw_n1.hi, tw_n1.lo, tw_n1.lo_bits, n1, dsign, 1, n1, 1, n1, nullptr, 1.0f);
    CKL_CUDA_LAST_ERROR(false);

    // The result sits at k2 * n1 + k1 and the caller wants k1 * n2 + k2, so the
    // last transpose is part of the algorithm rather than an afterthought. Any
    // epilogue rides on it, which is why a fused convolution pays nothing for its
    // pointwise multiply.
    fft_transpose(work, out, n2, n1, batch, transpose, epilogue_mul, epilogue_scale, stream);
}

}  // namespace ckl
