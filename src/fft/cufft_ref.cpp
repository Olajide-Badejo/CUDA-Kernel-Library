// The free function face of the cuFFT baseline, a thin layer over ckl::FftPlan.
//
// The plan is the whole risk in this family, which is why the spec handles it the
// way the measurement protocol handles a cuBLAS handle: created once per size,
// outside the timed region, reused across every rep and across every variant
// compared against it. Plan creation is not a benchmark row.
//
// A benchmark should not come through here. It should build its own ckl::FftPlan
// once per size and call ckl::fft with FftAlgo::kCufft, which is what bench_fft,
// bench_conv and bench_all do. This path exists for a caller that wants one
// transform without holding a plan, and the one entry cache below is what keeps
// that from creating a cuFFT plan on every call. The cache holds exactly one
// plan, so alternating between two sizes rebuilds on every call and the header
// says so.
//
// Workspace policy. The plan runs on cuFFT's own auto allocated work area:
// cufftSetAutoAllocation is never called and neither is cufftSetWorkArea. A
// baseline running without the work area a caller would actually get is not the
// baseline, and docs/fft.md records that the default path is the one measured.

#include <memory>
#include <mutex>

#include <cuda_runtime.h>

#include "ckl/context.hpp"
#include "ckl/fft.hpp"
#include "ckl/status.hpp"

#include "fft_internal.hpp"

namespace ckl {

namespace {

// What makes two calls the same plan. The direction is not in the key: one
// cufftHandle serves both, because cufftExecC2C takes the direction as an
// argument.
struct PlanKey {
    int n = 0;
    int batch = 0;

    bool operator==(const PlanKey& o) const { return n == o.n && batch == o.batch; }
};

}  // namespace

void fft_cufft(const float2* in, float2* out, int n, int batch, FftDirection dir,
               cudaStream_t stream) {
    if (n <= 1 || batch <= 0) {
        return;
    }
    const PlanKey key{n, batch};

    // One entry, rebuilt when the size changes. The cache is process wide, so the
    // whole sequence runs under the default Context's lock, exactly as the other
    // free function baselines in this library do.
    //
    // The last plan is deliberately released at process exit rather than at
    // destruction time: it lives in a function local static that outlives
    // nothing, and destroying a cuFFT handle after the CUDA context has begun
    // tearing down is worse than holding it.
    static std::unique_ptr<FftPlan> cached;
    static PlanKey cached_key;
    const std::lock_guard<std::mutex> lock(detail::default_context_mutex());
    if (!cached || !(cached_key == key)) {
        FftPlanOptions opt;
        opt.batch = batch;
        // Built before the old one is released, so a failed rebuild leaves the
        // previous plan usable.
        auto fresh = std::make_unique<FftPlan>(n, opt);
        cached = std::move(fresh);
        cached_key = key;
    }
    // The plan's launcher rather than ckl::fft: ckl::fft is the dispatcher and
    // lives in ckl_api, which links this library, so calling it from here would
    // close a cycle in the link graph. The free function contract is a throw on
    // failure and this keeps it.
    detail::fft_launch(*cached, FftAlgo::kCufft, dir, in, out, stream);
}

}  // namespace ckl
