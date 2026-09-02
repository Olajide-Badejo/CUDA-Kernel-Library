// The CUB baseline for both ladders.
//
// CUB is the vendor implementation for this family. It ships inside the CUDA
// toolkit, it is what Thrust dispatches to, and it implements decoupled
// look-back itself, so parity is the realistic goal and a wide margin is not the
// claim. cuBLAS has no scan at all; its only overlap is asum and dot, which
// reduce different things and appear as a secondary row that says so.
//
// The fairness rules of the measurement protocol are structural here rather than
// a matter of discipline. The temporary storage query and its allocation happen
// when the plan is built, once per length and type, and the buffer is reused by
// every call and by every variant compared against it. This file cannot allocate
// even if it wanted to: it takes the buffer and its size as arguments. Only the
// second call is timed, which the benchmark drivers arrange by verifying before
// they time.
//
// CUB_VERSION rides on every results row and is cited in docs/scan.md, because a
// percentage against CUB is a percentage against one particular CUB.

#include <algorithm>

#include <cub/cub.cuh>

#include "ckl/cuda_check.hpp"

#include "scan_device.cuh"
#include "scan_rungs.hpp"

namespace ckl {
namespace detail {

namespace {

// The family's operators in the shape CUB wants them.
template <typename T, class Op>
struct CubOp {
    __device__ T operator()(const T& a, const T& b) const { return Op::apply(a, b); }
};

template <typename T>
std::size_t temp_bytes_for(long long n) {
    using Op = SumOp<T>;
    std::size_t worst = 0;
    std::size_t bytes = 0;
    const T* in = nullptr;
    T* out = nullptr;

    CKL_CUDA_CHECK(cub::DeviceReduce::Sum(nullptr, bytes, in, out, n));
    worst = std::max(worst, bytes);

    bytes = 0;
    CKL_CUDA_CHECK(
        cub::DeviceReduce::Reduce(nullptr, bytes, in, out, n, CubOp<T, Op>(), Op::identity()));
    worst = std::max(worst, bytes);

    bytes = 0;
    CKL_CUDA_CHECK(cub::DeviceScan::InclusiveSum(nullptr, bytes, in, out, n));
    worst = std::max(worst, bytes);

    bytes = 0;
    CKL_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(nullptr, bytes, in, out, n));
    worst = std::max(worst, bytes);

    bytes = 0;
    CKL_CUDA_CHECK(cub::DeviceScan::InclusiveScan(nullptr, bytes, in, out, CubOp<T, Op>(), n));
    worst = std::max(worst, bytes);

    bytes = 0;
    CKL_CUDA_CHECK(
        cub::DeviceScan::ExclusiveScan(nullptr, bytes, in, out, CubOp<T, Op>(), Op::identity(), n));
    worst = std::max(worst, bytes);

    return worst;
}

}  // namespace

int cub_version_macro() {
    return CUB_VERSION;
}

std::size_t cub_temp_bytes(ScanDType dtype, long long n) {
    if (n <= 0) {
        return 0;
    }
    switch (dtype) {
        case ScanDType::kF64:
            return temp_bytes_for<double>(n);
        case ScanDType::kI32:
            return temp_bytes_for<int>(n);
        case ScanDType::kI64:
            return temp_bytes_for<long long>(n);
        case ScanDType::kF32:
        default:
            return temp_bytes_for<float>(n);
    }
}

template <typename T, ScanOp OP>
void reduce_cub_launch(const T* in, T* out, long long n, void* temp, std::size_t temp_bytes,
                       cudaStream_t stream) {
    using Op = typename OpOf<T, OP>::type;
    if (n <= 0) {
        reduce_fill_identity<T, OP>(out, stream);
        return;
    }
    std::size_t bytes = temp_bytes;
    if constexpr (OP == ScanOp::kSum) {
        // The entry point the protocol names as the baseline of record.
        CKL_CUDA_CHECK(cub::DeviceReduce::Sum(temp, bytes, in, out, n, stream));
    } else {
        CKL_CUDA_CHECK(cub::DeviceReduce::Reduce(temp, bytes, in, out, n, CubOp<T, Op>(),
                                                 Op::identity(), stream));
    }
}

template <typename T, ScanOp OP>
void scan_cub_launch(const T* in, T* out, long long n, bool exclusive, void* temp,
                     std::size_t temp_bytes, cudaStream_t stream) {
    using Op = typename OpOf<T, OP>::type;
    if (n <= 0) {
        return;
    }
    std::size_t bytes = temp_bytes;
    if constexpr (OP == ScanOp::kSum) {
        if (exclusive) {
            CKL_CUDA_CHECK(cub::DeviceScan::ExclusiveSum(temp, bytes, in, out, n, stream));
        } else {
            CKL_CUDA_CHECK(cub::DeviceScan::InclusiveSum(temp, bytes, in, out, n, stream));
        }
    } else {
        if (exclusive) {
            CKL_CUDA_CHECK(cub::DeviceScan::ExclusiveScan(temp, bytes, in, out, CubOp<T, Op>(),
                                                          Op::identity(), n, stream));
        } else {
            CKL_CUDA_CHECK(
                cub::DeviceScan::InclusiveScan(temp, bytes, in, out, CubOp<T, Op>(), n, stream));
        }
    }
}

#define CKL_INSTANTIATE_REDUCE(T, OP)                                                   \
    template void reduce_cub_launch<T, OP>(const T*, T*, long long, void*, std::size_t, \
                                           cudaStream_t);
CKL_SCAN_REDUCE_INSTANCES(CKL_INSTANTIATE_REDUCE)
#undef CKL_INSTANTIATE_REDUCE

#define CKL_INSTANTIATE_SCAN(T, OP)                                                         \
    template void scan_cub_launch<T, OP>(const T*, T*, long long, bool, void*, std::size_t, \
                                         cudaStream_t);
CKL_SCAN_SCAN_INSTANCES(CKL_INSTANTIATE_SCAN)
#undef CKL_INSTANTIATE_SCAN

}  // namespace detail
}  // namespace ckl
