#pragma once

// The device side pieces both ladders are built from: the four operators, the
// per type limits their identities need, the 16 byte vector types, the warp
// primitives, the three block scan flavours, and the shared memory exchange that
// turns a coalesced striped load into the blocked arrangement a serial scan
// needs.
//
// Two rules run through the whole file. Every combine puts the earlier element
// on the left, because ScanOp::kLastNonZero is associative and not commutative
// and a reduction tree that combined the other way round would still pass every
// sum test. And inclusive and exclusive come out of one core: a thread scans its
// own items serially from a running value, and whether it writes the running
// value before or after the combine is a template parameter.
//
// This header is an implementation detail of src/scan. It is not installed and
// nothing outside src/scan includes it.

#include <cfloat>
#include <climits>
#include <cmath>

#include <cuda_runtime.h>

#include "ckl/scan.hpp"

namespace ckl {
namespace detail {

// ---------------------------------------------------------------------------
// Per type limits
// ---------------------------------------------------------------------------
//
// std::numeric_limits is not usable in device code, so the four types the family
// instantiates carry their own. Float and double use infinity rather than the
// lowest finite value, so a dataset that contains FLT_MAX still reduces
// correctly under kMax.

template <typename T>
struct ScanLimits;

template <>
struct ScanLimits<float> {
    static __device__ __host__ float lowest() { return -INFINITY; }
    static __device__ __host__ float highest() { return INFINITY; }
};

template <>
struct ScanLimits<double> {
    static __device__ __host__ double lowest() { return -static_cast<double>(INFINITY); }
    static __device__ __host__ double highest() { return static_cast<double>(INFINITY); }
};

template <>
struct ScanLimits<int> {
    static __device__ __host__ int lowest() { return INT_MIN; }
    static __device__ __host__ int highest() { return INT_MAX; }
};

template <>
struct ScanLimits<long long> {
    static __device__ __host__ long long lowest() { return LLONG_MIN; }
    static __device__ __host__ long long highest() { return LLONG_MAX; }
};

// ---------------------------------------------------------------------------
// The operators
// ---------------------------------------------------------------------------
//
// apply(a, b) always means "a came first". kLastNonZero is the rung that proves
// the rest of the family respects that: it is associative, so every algorithm
// here is entitled to use it, and it is not commutative, so any algorithm that
// swapped the operands gets a different answer.

template <typename T>
struct SumOp {
    static __device__ __host__ T identity() { return T(0); }
    static __device__ __host__ T apply(T a, T b) { return a + b; }
};

template <typename T>
struct MaxOp {
    static __device__ __host__ T identity() { return ScanLimits<T>::lowest(); }
    static __device__ __host__ T apply(T a, T b) { return a > b ? a : b; }
};

template <typename T>
struct MinOp {
    static __device__ __host__ T identity() { return ScanLimits<T>::highest(); }
    static __device__ __host__ T apply(T a, T b) { return a < b ? a : b; }
};

template <typename T>
struct LastNonZeroOp {
    static __device__ __host__ T identity() { return T(0); }
    static __device__ __host__ T apply(T a, T b) { return b != T(0) ? b : a; }
};

// Maps the runtime operator enum onto the functor, so a rung can be a template
// over ScanOp and the plan can dispatch on two enums without ever including a
// CUDA header.
template <typename T, ScanOp OP>
struct OpOf;

template <typename T>
struct OpOf<T, ScanOp::kSum> {
    using type = SumOp<T>;
};

template <typename T>
struct OpOf<T, ScanOp::kMax> {
    using type = MaxOp<T>;
};

template <typename T>
struct OpOf<T, ScanOp::kMin> {
    using type = MinOp<T>;
};

template <typename T>
struct OpOf<T, ScanOp::kLastNonZero> {
    using type = LastNonZeroOp<T>;
};

// The instantiation sets, in one place so no two rungs can drift apart.
//
// FP32 carries every operator the rung supports and the other three types carry
// sum, which is what recomputing the roof per dtype needs. The reduction set
// stops at the three commutative operators: every reduction rung on the ladder
// combines its block partials in an order that depends on how the launch was
// scheduled, and no amount of care inside a block fixes that, so a non
// commutative operator has no honest reduction rung to run on. It runs on the
// whole scan ladder instead, which is where the associativity-only claim lives.
#define CKL_SCAN_REDUCE_INSTANCES(MACRO) \
    MACRO(float, ::ckl::ScanOp::kSum)    \
    MACRO(float, ::ckl::ScanOp::kMax)    \
    MACRO(float, ::ckl::ScanOp::kMin)    \
    MACRO(double, ::ckl::ScanOp::kSum)   \
    MACRO(int, ::ckl::ScanOp::kSum)      \
    MACRO(long long, ::ckl::ScanOp::kSum)

#define CKL_SCAN_SCAN_INSTANCES(MACRO)        \
    MACRO(float, ::ckl::ScanOp::kSum)         \
    MACRO(float, ::ckl::ScanOp::kMax)         \
    MACRO(float, ::ckl::ScanOp::kMin)         \
    MACRO(float, ::ckl::ScanOp::kLastNonZero) \
    MACRO(double, ::ckl::ScanOp::kSum)        \
    MACRO(int, ::ckl::ScanOp::kSum)           \
    MACRO(long long, ::ckl::ScanOp::kSum)

// ---------------------------------------------------------------------------
// Global atomic combine
// ---------------------------------------------------------------------------
//
// r0 and r2 both land one atomic on one address. Sum has a native instruction
// for all four types; max and min on a float do not, so those go through a
// compare and swap loop on the bit pattern. Both serialize on the same address,
// which is the whole point of the rung.

template <typename T>
struct AtomicBits;

template <>
struct AtomicBits<float> {
    using U = unsigned int;
    static __device__ U to(float v) { return __float_as_uint(v); }
    static __device__ float from(U u) { return __uint_as_float(u); }
};

template <>
struct AtomicBits<int> {
    using U = unsigned int;
    static __device__ U to(int v) { return static_cast<U>(v); }
    static __device__ int from(U u) { return static_cast<int>(u); }
};

template <>
struct AtomicBits<double> {
    using U = unsigned long long;
    static __device__ U to(double v) { return static_cast<U>(__double_as_longlong(v)); }
    static __device__ double from(U u) { return __longlong_as_double(static_cast<long long>(u)); }
};

template <>
struct AtomicBits<long long> {
    using U = unsigned long long;
    static __device__ U to(long long v) { return static_cast<U>(v); }
    static __device__ long long from(U u) { return static_cast<long long>(u); }
};

template <typename T, class Op>
struct AtomicCombine {
    static __device__ void run(T* addr, T value) {
        using U = typename AtomicBits<T>::U;
        U* raw = reinterpret_cast<U*>(addr);
        U old = *raw;
        U assumed;
        do {
            assumed = old;
            const T combined = Op::apply(AtomicBits<T>::from(assumed), value);
            old = atomicCAS(raw, assumed, AtomicBits<T>::to(combined));
        } while (assumed != old);
    }
};

template <typename T>
struct AtomicCombine<T, SumOp<T>> {
    static __device__ void run(T* addr, T value) { atomicAdd(addr, value); }
};

template <>
struct AtomicCombine<long long, SumOp<long long>> {
    static __device__ void run(long long* addr, long long value) {
        atomicAdd(reinterpret_cast<unsigned long long*>(addr),
                  static_cast<unsigned long long>(value));
    }
};

// ---------------------------------------------------------------------------
// 16 byte vector loads
// ---------------------------------------------------------------------------
//
// One instruction per 16 bytes, which is four elements at 4 bytes and two at 8.
// The rung is named vec4 after the float case it was written for; the width in
// bytes is what actually matters and it is the same either way.

template <typename T>
struct Vec16;

template <>
struct Vec16<float> {
    using type = float4;
    static constexpr int kLanes = 4;
};

template <>
struct Vec16<int> {
    using type = int4;
    static constexpr int kLanes = 4;
};

template <>
struct Vec16<double> {
    using type = double2;
    static constexpr int kLanes = 2;
};

template <>
struct Vec16<long long> {
    using type = longlong2;
    static constexpr int kLanes = 2;
};

// Lane access on a 16 byte vector. Written as a chain of comparisons on a
// compile time index rather than a reinterpret_cast to T*, because taking the
// address of a vector held in registers is what forces it into local memory.
template <typename T>
struct VecAccess;

template <>
struct VecAccess<float> {
    static __device__ float get(const float4& v, int j) {
        return j == 0 ? v.x : (j == 1 ? v.y : (j == 2 ? v.z : v.w));
    }
    static __device__ void set(float4& v, int j, float x) {
        if (j == 0) {
            v.x = x;
        } else if (j == 1) {
            v.y = x;
        } else if (j == 2) {
            v.z = x;
        } else {
            v.w = x;
        }
    }
};

template <>
struct VecAccess<int> {
    static __device__ int get(const int4& v, int j) {
        return j == 0 ? v.x : (j == 1 ? v.y : (j == 2 ? v.z : v.w));
    }
    static __device__ void set(int4& v, int j, int x) {
        if (j == 0) {
            v.x = x;
        } else if (j == 1) {
            v.y = x;
        } else if (j == 2) {
            v.z = x;
        } else {
            v.w = x;
        }
    }
};

template <>
struct VecAccess<double> {
    static __device__ double get(const double2& v, int j) { return j == 0 ? v.x : v.y; }
    static __device__ void set(double2& v, int j, double x) {
        if (j == 0) {
            v.x = x;
        } else {
            v.y = x;
        }
    }
};

template <>
struct VecAccess<long long> {
    static __device__ long long get(const longlong2& v, int j) { return j == 0 ? v.x : v.y; }
    static __device__ void set(longlong2& v, int j, long long x) {
        if (j == 0) {
            v.x = x;
        } else {
            v.y = x;
        }
    }
};

// Items each thread carries in the production rungs. Held at 32 bytes per thread
// whatever the element size, so the two 16 byte loads per thread are the same
// instruction mix for every type.
template <typename T>
struct TileShape {
    static constexpr int kBlock = 256;
    static constexpr int kItems = 32 / static_cast<int>(sizeof(T));
    static constexpr int kTile = kBlock * kItems;
};

// ---------------------------------------------------------------------------
// Warp primitives
// ---------------------------------------------------------------------------

constexpr unsigned int kFullMask = 0xffffffffu;

// Ordered warp reduction: lane 0 ends with v[0] op v[1] op ... op v[31].
//
// The offsets go up, not down. The usual butterfly, offset 16 then 8 down to 1,
// gives lane 0 the grouping op(op(v0,v16), op(v8,v24)) and so on, which is the
// right answer only if the operator commutes. Doubling the offset instead gives
// op(op(v0,v1), op(v2,v3)) and upward, which is the operands in index order and
// therefore right for any associative operator. Lanes above zero end with a
// partial answer built partly from out of range shuffles, which is why only lane
// zero's result is used; lane zero never reads a lane above 31.
template <typename T, class Op>
__device__ inline T warp_reduce(T v) {
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const T other = __shfl_down_sync(kFullMask, v, offset);
        v = Op::apply(v, other);
    }
    return v;
}

// Ordered warp inclusive scan.
template <typename T, class Op>
__device__ inline T warp_inclusive_scan(T v) {
    const int lane = static_cast<int>(threadIdx.x) & 31;
#pragma unroll
    for (int offset = 1; offset < 32; offset <<= 1) {
        const T other = __shfl_up_sync(kFullMask, v, offset);
        if (lane >= offset) {
            v = Op::apply(other, v);
        }
    }
    return v;
}

// ---------------------------------------------------------------------------
// Block reduction
// ---------------------------------------------------------------------------

template <typename T, class Op, int BLOCK>
struct BlockReduce {
    struct TempStorage {
        T warp[BLOCK / 32];
    };

    // Every thread contributes one value; the answer lands in thread 0. Every
    // thread reaches the trailing barrier, so the storage is reusable and a grid
    // stride loop can call this once per pass.
    static __device__ T run(TempStorage& tmp, T v) {
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        v = warp_reduce<T, Op>(v);
        if (lane == 0) {
            tmp.warp[warp] = v;
        }
        __syncthreads();
        T total = Op::identity();
        if (warp == 0) {
            T w = lane < BLOCK / 32 ? tmp.warp[lane] : Op::identity();
            total = warp_reduce<T, Op>(w);
        }
        __syncthreads();
        return total;
    }
};

// ---------------------------------------------------------------------------
// Block scan flavours
// ---------------------------------------------------------------------------
//
// Each flavour takes one value per thread and returns that thread's exclusive
// prefix, writing the block total through a pointer. Exclusive rather than
// inclusive because that is what both the item level serial scan and the tile
// level recursion want; an inclusive answer is one more combine away and the
// two rungs that need it do it themselves.

// s0. The shortest correct block scan there is, and the most work: log2(BLOCK)
// passes over BLOCK elements, so O(N log N) work at block scope where the
// Blelloch rung does O(N). It stays on the ladder because it fixes the barrier
// and register floor the later rungs are measured against, and because naming a
// rung work inefficient is the point of having it.
template <typename T, class Op, int BLOCK>
struct HillisSteeleBlockScan {
    struct TempStorage {
        T buf[2 * BLOCK];
    };

    static __device__ T exclusive(TempStorage& tmp, T v, T* total) {
        const int tid = static_cast<int>(threadIdx.x);
        int in = 0;
        tmp.buf[tid] = v;
        __syncthreads();
        for (int offset = 1; offset < BLOCK; offset <<= 1) {
            const int out = 1 - in;
            const T mine = tmp.buf[in * BLOCK + tid];
            const T value =
                tid >= offset ? Op::apply(tmp.buf[in * BLOCK + tid - offset], mine) : mine;
            tmp.buf[out * BLOCK + tid] = value;
            in = out;
            __syncthreads();
        }
        *total = tmp.buf[in * BLOCK + BLOCK - 1];
        // The exclusive answer is the neighbour's inclusive one, which is
        // already in the buffer, so no second pass is needed.
        const T result = tid > 0 ? tmp.buf[in * BLOCK + tid - 1] : Op::identity();
        __syncthreads();
        return result;
    }
};

// s1. Upsweep and downsweep, O(BLOCK) combines. The shared indices are padded by
// i + i/32 so the strided accesses of the two sweeps do not collide on a bank;
// docs/scan.md carries the ncu counter page with and without the padding.
template <typename T, class Op, int BLOCK>
struct BlellochBlockScan {
    static constexpr int kPadded = BLOCK + BLOCK / 32;

    struct TempStorage {
        T buf[kPadded];
        T total;
    };

    static __device__ int pad(int i) { return i + (i >> 5); }

    static __device__ T exclusive(TempStorage& tmp, T v, T* total) {
        const int tid = static_cast<int>(threadIdx.x);
        tmp.buf[pad(tid)] = v;
        __syncthreads();

        // Upsweep: a balanced binary tree of combines, earlier element on the
        // left at every node.
        int offset = 1;
        for (int d = BLOCK >> 1; d > 0; d >>= 1) {
            __syncthreads();
            if (tid < d) {
                const int ai = offset * (2 * tid + 1) - 1;
                const int bi = offset * (2 * tid + 2) - 1;
                tmp.buf[pad(bi)] = Op::apply(tmp.buf[pad(ai)], tmp.buf[pad(bi)]);
            }
            offset <<= 1;
        }
        __syncthreads();
        if (tid == 0) {
            tmp.total = tmp.buf[pad(BLOCK - 1)];
            tmp.buf[pad(BLOCK - 1)] = Op::identity();
        }

        // Downsweep. The swap plus combine is what turns the reduction tree into
        // an exclusive scan without a second array. Node bi holds the parent's
        // exclusive prefix and node ai holds the left subtree's aggregate, so the
        // right child's prefix is apply(parent prefix, left aggregate) and not the
        // other way round. Writing that pair the wrong way round still passes
        // every sum test, which is what ScanLadder.FunctorOperatorsExactly with
        // kLastNonZero is on the suite to catch.
        for (int d = 1; d < BLOCK; d <<= 1) {
            offset >>= 1;
            __syncthreads();
            if (tid < d) {
                const int ai = offset * (2 * tid + 1) - 1;
                const int bi = offset * (2 * tid + 2) - 1;
                const T t = tmp.buf[pad(ai)];
                tmp.buf[pad(ai)] = tmp.buf[pad(bi)];
                tmp.buf[pad(bi)] = Op::apply(tmp.buf[pad(bi)], t);
            }
        }
        __syncthreads();
        const T result = tmp.buf[pad(tid)];
        *total = tmp.total;
        __syncthreads();
        return result;
    }
};

// s2 and above. A warp scan, a scan of the warp totals by warp 0, and one
// combine: two shuffle rounds and one barrier instead of log2(BLOCK) barriers.
template <typename T, class Op, int BLOCK>
struct WarpBlockScan {
    static constexpr int kWarps = BLOCK / 32;

    struct TempStorage {
        T warp[kWarps];
        T total;
    };

    static __device__ T exclusive(TempStorage& tmp, T v, T* total) {
        const int lane = static_cast<int>(threadIdx.x) & 31;
        const int warp = static_cast<int>(threadIdx.x) >> 5;
        const T inclusive = warp_inclusive_scan<T, Op>(v);
        if (lane == 31) {
            tmp.warp[warp] = inclusive;
        }
        __syncthreads();
        if (warp == 0) {
            const T w = lane < kWarps ? tmp.warp[lane] : Op::identity();
            const T scanned = warp_inclusive_scan<T, Op>(w);
            // Hoisted out of the lane predicate: a shuffle with the full mask
            // has to be reached by every lane of the warp.
            const T shifted = __shfl_up_sync(kFullMask, scanned, 1);
            if (lane < kWarps) {
                // The exclusive prefix of warp `lane`, which is the inclusive
                // scan of the warp totals shifted by one.
                tmp.warp[lane] = lane == 0 ? Op::identity() : shifted;
            }
            if (lane == kWarps - 1) {
                tmp.total = scanned;
            }
        }
        __syncthreads();
        const T warp_prefix = tmp.warp[warp];
        // The thread's own exclusive value inside its warp.
        const T within = __shfl_up_sync(kFullMask, inclusive, 1);
        const T mine = lane == 0 ? Op::identity() : within;
        *total = tmp.total;
        const T result = Op::apply(warp_prefix, mine);
        __syncthreads();
        return result;
    }
};

// ---------------------------------------------------------------------------
// The tile exchange
// ---------------------------------------------------------------------------
//
// A scan needs each thread to own a contiguous run of items, and a coalesced
// load needs consecutive threads to read consecutive addresses. Those two are
// not the same arrangement, so the tile goes through shared memory: 16 byte
// vector loads striped across the block on the way in, a padded read on the way
// out. Without the i + i/32 padding the blocked read is an ITEMS way bank
// conflict on every access.

template <typename T, int BLOCK, int ITEMS>
struct TileExchange {
    static constexpr int kTile = BLOCK * ITEMS;
    // One item per thread needs no transpose at all: the coalesced arrangement
    // and the blocked one are the same arrangement. The storage shrinks to a
    // placeholder so the two naive rungs do not pay a shared memory round trip
    // that belongs to the wide rungs.
    static constexpr int kPadded = ITEMS == 1 ? 1 : kTile + kTile / 32;
    using Vec = typename Vec16<T>::type;
    static constexpr int kLanes = Vec16<T>::kLanes;
    static constexpr int kVecs = kTile / kLanes;

    struct TempStorage {
        T buf[kPadded];
    };

    static __device__ int pad(int i) { return i + (i >> 5); }

    // Reads one tile into the blocked arrangement. `valid` is how many of the
    // tile's elements exist; the rest come back as the operator's identity so
    // the tail needs no separate code path anywhere above this.
    static __device__ void load(TempStorage& tmp, const T* in, long long base, int valid, T fill,
                                T (&items)[ITEMS]) {
        const int tid = static_cast<int>(threadIdx.x);
        if constexpr (ITEMS == 1) {
            (void)tmp;
            items[0] = tid < valid ? in[base + tid] : fill;
        } else {
            if (valid == kTile) {
                const Vec* v = reinterpret_cast<const Vec*>(in + base);
                for (int i = tid; i < kVecs; i += BLOCK) {
                    const Vec value = v[i];
#pragma unroll
                    for (int j = 0; j < kLanes; ++j) {
                        tmp.buf[pad(i * kLanes + j)] = VecAccess<T>::get(value, j);
                    }
                }
            } else {
                for (int i = tid; i < kTile; i += BLOCK) {
                    tmp.buf[pad(i)] = i < valid ? in[base + i] : fill;
                }
            }
            __syncthreads();
#pragma unroll
            for (int i = 0; i < ITEMS; ++i) {
                items[i] = tmp.buf[pad(tid * ITEMS + i)];
            }
            __syncthreads();
        }
    }

    // Writes one tile back out of the blocked arrangement.
    static __device__ void store(TempStorage& tmp, T* out, long long base, int valid,
                                 const T (&items)[ITEMS]) {
        const int tid = static_cast<int>(threadIdx.x);
        if constexpr (ITEMS == 1) {
            (void)tmp;
            if (tid < valid) {
                out[base + tid] = items[0];
            }
        } else {
#pragma unroll
            for (int i = 0; i < ITEMS; ++i) {
                tmp.buf[pad(tid * ITEMS + i)] = items[i];
            }
            __syncthreads();
            if (valid == kTile) {
                Vec* v = reinterpret_cast<Vec*>(out + base);
                for (int i = tid; i < kVecs; i += BLOCK) {
                    Vec value;
#pragma unroll
                    for (int j = 0; j < kLanes; ++j) {
                        VecAccess<T>::set(value, j, tmp.buf[pad(i * kLanes + j)]);
                    }
                    v[i] = value;
                }
            } else {
                for (int i = tid; i < kTile; i += BLOCK) {
                    if (i < valid) {
                        out[base + i] = tmp.buf[pad(i)];
                    }
                }
            }
            __syncthreads();
        }
    }
};

// The serial scan inside one thread. This is the whole of what separates an
// inclusive scan from an exclusive one, which is why there is one core and not
// two.
template <typename T, class Op, int ITEMS, bool EXCLUSIVE>
__device__ inline T serial_scan(T (&items)[ITEMS], T running) {
#pragma unroll
    for (int i = 0; i < ITEMS; ++i) {
        const T value = items[i];
        if (EXCLUSIVE) {
            items[i] = running;
            running = Op::apply(running, value);
        } else {
            running = Op::apply(running, value);
            items[i] = running;
        }
    }
    return running;
}

// The ordered reduction of one thread's items.
template <typename T, class Op, int ITEMS>
__device__ inline T serial_reduce(const T (&items)[ITEMS]) {
    T acc = items[0];
#pragma unroll
    for (int i = 1; i < ITEMS; ++i) {
        acc = Op::apply(acc, items[i]);
    }
    return acc;
}

}  // namespace detail
}  // namespace ckl
