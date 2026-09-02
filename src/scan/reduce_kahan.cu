// The compensated rung: every partial carries its own rounding error alongside
// it, and the error is folded back in at every combine.
//
// A float sum of N terms loses about log2(N) bits to rounding even with a
// perfect tree, and at 2^28 elements of similar magnitude that is most of the
// significand. Compensated summation buys almost all of it back for the price of
// three extra floating point operations per element and one extra value per
// partial. The hypothesis this rung exists to test is that those operations are
// free at large N, where the kernel is waiting on DRAM and has issue slots to
// spare, and are not free at small N, where it is not. The sweep decides, and
// the row is next to r3 and r4 so the comparison is against the rungs that do
// the same loads.
//
// Two term arithmetic rather than plain Kahan: each partial is a pair, the high
// word and the running compensation, and merging two pairs uses a Knuth two sum
// on the high words and folds both compensations into the result. Plain Kahan
// compensates a serial accumulation and has nothing to say about how two
// compensated partials should meet, which is exactly what a parallel reduction
// spends its time doing.
//
// Compensation only means anything for addition. The plan refuses the other
// operators on this rung with a message that says so, so nothing reaches the
// fallback below; it exists because the rung is instantiated over the same
// operator set as the rest of the ladder and a template has to compile.

#include <algorithm>

#include "ckl/cuda_check.hpp"

#include "scan_device.cuh"
#include "scan_rungs.hpp"

namespace ckl {
namespace detail {

namespace {

constexpr int kBlock = 256;

template <typename T, class Op>
struct Compensated {
    struct Pair {
        T hi;
        T lo;
    };

    static __device__ Pair init() { return Pair{Op::identity(), Op::identity()}; }
    static __device__ Pair add(Pair a, T x) { return Pair{Op::apply(a.hi, x), a.lo}; }
    static __device__ Pair merge(Pair a, Pair b) { return Pair{Op::apply(a.hi, b.hi), a.lo}; }
    static __device__ T finish(Pair a) { return a.hi; }
};

template <typename T>
struct Compensated<T, SumOp<T>> {
    struct Pair {
        T hi;
        T lo;
    };

    static __device__ Pair init() { return Pair{T(0), T(0)}; }

    // Knuth two sum: s is the rounded sum and err is exactly what rounding lost,
    // with no assumption about which operand is larger.
    static __device__ Pair two_sum(T a, T b) {
        const T s = a + b;
        const T bb = s - a;
        const T err = (a - (s - bb)) + (b - bb);
        return Pair{s, err};
    }

    static __device__ Pair add(Pair a, T x) {
        const Pair t = two_sum(a.hi, x);
        return Pair{t.hi, a.lo + t.lo};
    }

    static __device__ Pair merge(Pair a, Pair b) {
        const Pair t = two_sum(a.hi, b.hi);
        return Pair{t.hi, a.lo + b.lo + t.lo};
    }

    static __device__ T finish(Pair a) { return a.hi + a.lo; }
};

template <typename T, class Op>
__device__ inline typename Compensated<T, Op>::Pair block_merge(
    T* s_hi, T* s_lo, typename Compensated<T, Op>::Pair v) {
    using C = Compensated<T, Op>;
    const int tid = static_cast<int>(threadIdx.x);
    s_hi[tid] = v.hi;
    s_lo[tid] = v.lo;
    __syncthreads();
    for (int half = kBlock / 2; half > 0; half >>= 1) {
        if (tid < half) {
            const typename C::Pair a{s_hi[tid], s_lo[tid]};
            const typename C::Pair b{s_hi[tid + half], s_lo[tid + half]};
            const typename C::Pair m = C::merge(a, b);
            s_hi[tid] = m.hi;
            s_lo[tid] = m.lo;
        }
        __syncthreads();
    }
    const typename C::Pair result{s_hi[0], s_lo[0]};
    __syncthreads();
    return result;
}

// The partials array holds the high words in the first `grid` slots and the
// compensations in the next `grid`, so the second pass can merge pairs rather
// than throwing the compensation away between the two levels.
template <typename T, class Op>
__global__ void kahan_partials(const T* in, T* partials, long long n, int grid) {
    using C = Compensated<T, Op>;
    using Vec = typename Vec16<T>::type;
    constexpr int kLanes = Vec16<T>::kLanes;
    __shared__ T s_hi[kBlock];
    __shared__ T s_lo[kBlock];

    const long long vectors = n / kLanes;
    const long long stride = static_cast<long long>(gridDim.x) * kBlock;
    const long long start =
        static_cast<long long>(blockIdx.x) * kBlock + static_cast<long long>(threadIdx.x);

    typename C::Pair acc = C::init();
    const Vec* v = reinterpret_cast<const Vec*>(in);
    for (long long i = start; i < vectors; i += stride) {
        const Vec value = v[i];
#pragma unroll
        for (int j = 0; j < kLanes; ++j) {
            acc = C::add(acc, VecAccess<T>::get(value, j));
        }
    }
    for (long long i = vectors * kLanes + start; i < n; i += stride) {
        acc = C::add(acc, in[i]);
    }

    const typename C::Pair total = block_merge<T, Op>(s_hi, s_lo, acc);
    if (threadIdx.x == 0) {
        partials[blockIdx.x] = total.hi;
        partials[grid + blockIdx.x] = total.lo;
    }
}

template <typename T, class Op>
__global__ void kahan_combine(const T* partials, T* out, int count) {
    using C = Compensated<T, Op>;
    __shared__ T s_hi[kBlock];
    __shared__ T s_lo[kBlock];
    const int tid = static_cast<int>(threadIdx.x);
    typename C::Pair acc = C::init();
    for (int i = tid; i < count; i += kBlock) {
        const typename C::Pair p{partials[i], partials[count + i]};
        acc = C::merge(acc, p);
    }
    const typename C::Pair total = block_merge<T, Op>(s_hi, s_lo, acc);
    if (tid == 0) {
        *out = C::finish(total);
    }
}

}  // namespace

template <typename T, ScanOp OP>
void reduce_kahan_launch(const T* in, T* out, long long n, T* partials, int blocks,
                         cudaStream_t stream) {
    using Op = typename OpOf<T, OP>::type;
    if (n <= 0) {
        reduce_fill_identity<T, OP>(out, stream);
        return;
    }
    const long long wanted = (n + kBlock - 1) / kBlock;
    const int grid = static_cast<int>(std::min<long long>(blocks, wanted));
    kahan_partials<T, Op>
        <<<static_cast<unsigned int>(grid), kBlock, 0, stream>>>(in, partials, n, grid);
    CKL_CUDA_LAST_ERROR(false);
    kahan_combine<T, Op><<<1, kBlock, 0, stream>>>(partials, out, grid);
    CKL_CUDA_LAST_ERROR(false);
}

#define CKL_INSTANTIATE(T, OP) \
    template void reduce_kahan_launch<T, OP>(const T*, T*, long long, T*, int, cudaStream_t);
CKL_SCAN_REDUCE_INSTANCES(CKL_INSTANTIATE)
#undef CKL_INSTANTIATE

}  // namespace detail
}  // namespace ckl
