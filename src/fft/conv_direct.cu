// Direct time domain convolution, the other half of the crossover.
//
// out[i] = sum_j signal[j] * filter[i - j], one output per thread, 2 N M FLOP and
// no transform anywhere. It wins at short filters, where the FFT path pays three
// transforms to save arithmetic it did not have much of, and it loses at long
// ones, where 2 N M grows without bound while the FFT path's traffic does not.
// Where the two cross is the headline figure of this family, and it comes out of
// a sweep rather than out of this comment.
//
// Tiling. A block owns a tile of 256 outputs. The obvious shape puts the signal
// tile plus a halo of M - 1 samples in shared memory, and that shape does not
// survive the top of the sweep: M = 16384 is a 64 KB halo before the tile. So the
// filter is walked in chunks instead. For each chunk of 256 taps the block loads
// the 256 taps and the 511 signal samples those taps need, accumulates, and moves
// on, which holds the shared footprint at about 3 KB whatever M is.
//
// The second rung reads the taps from constant memory rather than shared. The
// bank is 64 KB per module and the spec budgets 16 KB of it, so this rung takes
// filters up to 4096 taps. A constant memory read broadcasts to a whole warp in
// one cycle when every lane reads the same address, which is exactly what the
// inner loop does, so it should beat the shared load; whether it does is a
// measurement.
//
// The bank is one per module, not one per plan, so a plan binds its taps once at
// construction and the launcher refuses to run against another plan's binding
// rather than silently convolving with the wrong filter.

#include <mutex>
#include <string>

#include "ckl/cuda_check.hpp"
#include "ckl/fft.hpp"

namespace ckl {

namespace {

constexpr int kTile = 256;
constexpr int kChunk = 256;
constexpr int kMaxConstantTaps = 4096;

__constant__ float g_taps[kMaxConstantTaps];

std::mutex& bank_mutex() {
    static std::mutex m;
    return m;
}

const void*& bank_owner() {
    static const void* owner = nullptr;
    return owner;
}

int& bank_taps() {
    static int taps = 0;
    return taps;
}

// One output per thread, the taps staged through shared memory a chunk at a time.
__global__ void direct_shared_kernel(const float* __restrict__ signal,
                                     const float* __restrict__ filter, float* __restrict__ out,
                                     int n, int m, int out_len) {
    __shared__ float taps[kChunk];
    __shared__ float sig[kTile + kChunk - 1];

    const int tid = static_cast<int>(threadIdx.x);
    const int base = static_cast<int>(blockIdx.x) * kTile;
    const int i = base + tid;
    float acc = 0.0f;

    for (int c = 0; c < m; c += kChunk) {
        const int taps_here = m - c < kChunk ? m - c : kChunk;
        for (int t = tid; t < kChunk; t += kTile) {
            taps[t] = t < taps_here ? filter[c + t] : 0.0f;
        }
        // Outputs base to base + kTile - 1 need signal[i - t] for t in
        // [c, c + kChunk), so the window starts at base - c - kChunk + 1.
        const int start = base - c - kChunk + 1;
        for (int s = tid; s < kTile + kChunk - 1; s += kTile) {
            const int idx = start + s;
            sig[s] = (idx >= 0 && idx < n) ? signal[idx] : 0.0f;
        }
        __syncthreads();

        for (int t = 0; t < taps_here; ++t) {
            // signal[i - c - t] sits at sig[i - c - t - start].
            acc += taps[t] * sig[i - c - t - start];
        }
        __syncthreads();
    }

    if (i < out_len) {
        out[i] = acc;
    }
}

// The same tiling with the taps in the constant bank, so the inner loop is a
// broadcast read rather than a shared load.
__global__ void direct_constant_kernel(const float* __restrict__ signal, float* __restrict__ out,
                                       int n, int m, int out_len) {
    __shared__ float sig[kTile + kChunk - 1];

    const int tid = static_cast<int>(threadIdx.x);
    const int base = static_cast<int>(blockIdx.x) * kTile;
    const int i = base + tid;
    float acc = 0.0f;

    for (int c = 0; c < m; c += kChunk) {
        const int taps_here = m - c < kChunk ? m - c : kChunk;
        const int start = base - c - kChunk + 1;
        for (int s = tid; s < kTile + kChunk - 1; s += kTile) {
            const int idx = start + s;
            sig[s] = (idx >= 0 && idx < n) ? signal[idx] : 0.0f;
        }
        __syncthreads();

        for (int t = 0; t < taps_here; ++t) {
            acc += g_taps[c + t] * sig[i - c - t - start];
        }
        __syncthreads();
    }

    if (i < out_len) {
        out[i] = acc;
    }
}

}  // namespace

void conv_direct_shared(const float* signal, const float* filter, float* out, int n, int m,
                        cudaStream_t stream) {
    if (n <= 0 || m <= 0) {
        return;
    }
    const int out_len = n + m - 1;
    const int grid = (out_len + kTile - 1) / kTile;
    direct_shared_kernel<<<grid, kTile, 0, stream>>>(signal, filter, out, n, m, out_len);
    CKL_CUDA_LAST_ERROR(false);
}

int conv_direct_constant_max_taps() {
    return kMaxConstantTaps;
}

void conv_direct_constant_bind(const float* filter, int m, const void* owner, cudaStream_t stream) {
    if (m <= 0 || m > kMaxConstantTaps) {
        throw Error(Status::kNotSupported,
                    "conv_direct_constant_bind: the constant memory rung holds at most " +
                        std::to_string(kMaxConstantTaps) + " taps and was asked for " +
                        std::to_string(m));
    }
    const std::lock_guard<std::mutex> lock(bank_mutex());
    CKL_CUDA_CHECK(cudaMemcpyToSymbolAsync(g_taps, filter,
                                           sizeof(float) * static_cast<std::size_t>(m), 0,
                                           cudaMemcpyDeviceToDevice, stream));
    bank_owner() = owner;
    bank_taps() = m;
}

bool conv_direct_constant_bound(const void* owner) {
    const std::lock_guard<std::mutex> lock(bank_mutex());
    return owner != nullptr && bank_owner() == owner;
}

void conv_direct_constant(const float* signal, float* out, int n, int m, const void* owner,
                          cudaStream_t stream) {
    if (n <= 0 || m <= 0) {
        return;
    }
    if (m > kMaxConstantTaps) {
        throw Error(Status::kNotSupported,
                    "conv_direct_constant: the constant memory rung holds at most " +
                        std::to_string(kMaxConstantTaps) + " taps and was asked for " +
                        std::to_string(m));
    }
    if (!conv_direct_constant_bound(owner)) {
        throw Error(Status::kInvalidValue,
                    "conv_direct_constant: the constant bank holds another plan's taps. The bank "
                    "is one per module, so rebind this plan's filter before running this variant.");
    }
    const int out_len = n + m - 1;
    const int grid = (out_len + kTile - 1) / kTile;
    direct_constant_kernel<<<grid, kTile, 0, stream>>>(signal, out, n, m, out_len);
    CKL_CUDA_LAST_ERROR(false);
}

}  // namespace ckl
