// The pieces the frequency domain convolution is built out of: the zero padding
// pass, the standalone pointwise multiply, and the extract that takes the real
// part back out.
//
// Two variants share these. The separate pass form runs forward, then the
// pointwise kernel here, then inverse. That kernel reads two spectra and writes
// one, so it moves 24 bytes per point it does not have to. The fused form folds
// the filter spectrum multiply into the store epilogue of the last forward pass
// and the 1/L scaling into the store epilogue of the last inverse pass, which is
// the same epilogue fusion the GEMM bias and ReLU study uses, and neither
// epilogue costs a byte of extra traffic: the value is already in a register on
// its way out.
//
// What that is worth is arithmetic, and it is a hypothesis until the sweep runs.
// At L = 2^21 the transform runs on the four step form at five passes of 16L
// bytes each, so a convolution moves about 160L bytes in its two transforms, plus
// 8L for the pack, 24L for the separate pointwise pass and 8L for the extract.
// Removing the pointwise pass removes 24L out of about 200L, so 10 to 15 percent
// of the whole convolution. Measured as one diagnostic round; nothing here claims
// it has been.
//
// The filter spectrum is computed once, when the plan is built, and never inside
// a timed region.

#include "ckl/cuda_check.hpp"
#include "ckl/fft.hpp"

#include "fft_device.cuh"

namespace ckl {

namespace {

constexpr int kBlock = 256;

int grid_of(long long total, int block) {
    return static_cast<int>((total + block - 1) / block);
}

__global__ void pointwise_kernel(const float2* __restrict__ a, const float2* __restrict__ b,
                                 float2* __restrict__ out, long long count, float scale) {
    const long long t = static_cast<long long>(blockIdx.x) * blockDim.x + threadIdx.x;
    if (t >= count) {
        return;
    }
    float2 v = detail::cmul(a[t], b[t]);
    if (scale != 1.0f) {
        v = detail::cscale(v, scale);
    }
    out[t] = v;
}

__global__ void pack_kernel(const float* __restrict__ signal, float2* __restrict__ padded, int n,
                            int l) {
    const int t =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (t >= l) {
        return;
    }
    padded[t] = t < n ? make_float2(signal[t], 0.0f) : make_float2(0.0f, 0.0f);
}

__global__ void extract_kernel(const float2* __restrict__ spectrum, float* __restrict__ out,
                               int out_len, float scale) {
    const int t =
        static_cast<int>(blockIdx.x) * static_cast<int>(blockDim.x) + static_cast<int>(threadIdx.x);
    if (t >= out_len) {
        return;
    }
    out[t] = spectrum[t].x * scale;
}

}  // namespace

void conv_pointwise(const float2* a, const float2* b, float2* out, long long count, float scale,
                    cudaStream_t stream) {
    if (count <= 0) {
        return;
    }
    pointwise_kernel<<<grid_of(count, kBlock), kBlock, 0, stream>>>(a, b, out, count, scale);
    CKL_CUDA_LAST_ERROR(false);
}

void conv_pack(const float* signal, float2* padded, int n, int l, cudaStream_t stream) {
    if (l <= 0) {
        return;
    }
    pack_kernel<<<grid_of(l, kBlock), kBlock, 0, stream>>>(signal, padded, n, l);
    CKL_CUDA_LAST_ERROR(false);
}

void conv_extract(const float2* spectrum, float* out, int out_len, float scale,
                  cudaStream_t stream) {
    if (out_len <= 0) {
        return;
    }
    extract_kernel<<<grid_of(out_len, kBlock), kBlock, 0, stream>>>(spectrum, out, out_len, scale);
    CKL_CUDA_LAST_ERROR(false);
}

}  // namespace ckl
