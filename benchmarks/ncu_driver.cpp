// Single kernel driver for Nsight Compute rounds. It runs exactly one variant of
// one family on one shape a fixed number of times so ncu can profile one settled
// launch with `--launch-skip N -c 1`. Keeping the driver this small means the ncu
// report contains just the kernel under study, not the harness: no verification
// pass, no vendor comparison, no L2 flush kernel, nothing to attribute a counter
// to by accident.
//
// Usage: ncu_driver <variant> <arg> [launches]
//   GEMM variants take a square size:
//     naive, tiled, register, cp_async, wmma_fp16, wmma_bf16, mma_ptx, mma_ldm,
//     mma_opt, cublas_fp16, cublas_fp32
//   SpMV variants take a matrix name from the suite cache, or synthetic[:rows]:
//     spmv_naive, spmv_warp, spmv_vector, spmv_merge, spmv_sell, spmv_bsr,
//     spmv_cusparse, spmv_cusparse_alg2
//   FFT variants take log2 of the transform length:
//     fft_radix2, fft_radix4, fft_radix8, fft_four_step, fft_shared, fft_cufft
//   Reduction and scan variants take log2 of the element count:
//     reduce_vec4, reduce_single_pass, reduce_cub,
//     scan_lookback, scan_blelloch, scan_cub
//   launches how many times to run (default 5; profile the last with skip 4)
//
// The cublas_fp16 variant exists so the Section 9.4 profile-cuBLAS round runs
// through the same launcher as the hand kernels: same shapes, same buffers, same
// launch count, so the only difference between the two pages is the kernel. The
// other three families are here for the same reason the GEMM ones are: a round
// that has to say which kernel moved `dram__bytes.sum` cannot be run against a
// benchmark binary that launches nine variants and a baseline in one process.

#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/status.hpp"
#include "reference.hpp"

#ifdef CKL_NCU_SPARSE
#include "ckl/sparse.hpp"
#include "matrix_cache.hpp"
#endif
#ifdef CKL_NCU_FFT
#include "ckl/fft.hpp"
#endif
#ifdef CKL_NCU_SCAN
#include "ckl/scan.hpp"
#endif

namespace {

template <typename T>
T from_float(float f);
template <>
float from_float<float>(float f) {
    return f;
}
template <>
__half from_float<__half>(float f) {
    return __float2half(f);
}
template <>
__nv_bfloat16 from_float<__nv_bfloat16>(float f) {
    return __float2bfloat16(f);
}

template <typename T, typename LaunchFn>
void run(LaunchFn launch, int n, int launches) {
    std::vector<T> ha(static_cast<std::size_t>(n) * n);
    std::vector<T> hb(static_cast<std::size_t>(n) * n);
    const auto fa = ckl::random_matrix(n, n, 101);
    const auto fb = ckl::random_matrix(n, n, 202);
    for (std::size_t i = 0; i < ha.size(); ++i) {
        ha[i] = from_float<T>(fa[i]);
        hb[i] = from_float<T>(fb[i]);
    }
    ckl::DeviceBuffer<T> da(ha.size());
    ckl::DeviceBuffer<T> db(hb.size());
    ckl::DeviceBuffer<float> dc(static_cast<std::size_t>(n) * n);
    da.copy_from_host(ha);
    db.copy_from_host(hb);
    dc.zero();
    for (int i = 0; i < launches; ++i) {
        launch(da.data(), db.data(), dc.data(), n, n, n, 1.0f, 0.0f, nullptr);
    }
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
}

#if defined(CKL_NCU_SPARSE) || defined(CKL_NCU_FFT) || defined(CKL_NCU_SCAN)
// A status that is not kSuccess means the plan refused the rung, which on this
// driver is a fatal setup error rather than something to profile around.
bool ok(ckl::Status s, const char* what) {
    if (s == ckl::Status::kSuccess) {
        return true;
    }
    std::fprintf(stderr, "%s returned %s\n", what, ckl::status_string(s));
    return false;
}
#endif

#ifdef CKL_NCU_SPARSE
int run_spmv(const std::string& variant, const std::string& matrix, int launches) {
    static const struct {
        const char* name;
        ckl::SpmvAlgo algo;
    } kTable[] = {{"spmv_naive", ckl::SpmvAlgo::kCsrNaive},
                  {"spmv_warp", ckl::SpmvAlgo::kCsrWarp},
                  {"spmv_vector", ckl::SpmvAlgo::kCsrVector},
                  {"spmv_merge", ckl::SpmvAlgo::kMerge},
                  {"spmv_sell", ckl::SpmvAlgo::kSellCSigma},
                  {"spmv_bsr", ckl::SpmvAlgo::kBsr},
                  {"spmv_cusparse", ckl::SpmvAlgo::kCusparseDefault},
                  {"spmv_cusparse_alg2", ckl::SpmvAlgo::kCusparseAlg2}};
    ckl::SpmvAlgo algo = ckl::SpmvAlgo::kAuto;
    bool found = false;
    for (const auto& e : kTable) {
        if (variant == e.name) {
            algo = e.algo;
            found = true;
        }
    }
    if (!found) {
        std::fprintf(stderr, "unknown spmv variant: %s\n", variant.c_str());
        return 2;
    }

    ckl::bench::HostCsr a;
    std::string error;
    if (!ckl::bench::load_matrix(matrix, 1, &a, &error)) {
        std::fprintf(stderr, "%s\n", error.c_str());
        return 2;
    }
    ckl::DeviceBuffer<int> drp(a.row_ptr.size());
    ckl::DeviceBuffer<int> dci(a.col_idx.size());
    ckl::DeviceBuffer<float> dv(a.values.size());
    drp.copy_from_host(a.row_ptr);
    if (a.nnz > 0) {
        dci.copy_from_host(a.col_idx);
        dv.copy_from_host(a.values);
    }
    ckl::SpmvCsr view;
    view.row_ptr = drp.data();
    view.col_idx = dci.data();
    view.values = dv.data();
    view.m = a.m;
    view.n = a.n;
    view.nnz = a.nnz;
    ckl::SpmvPlan plan(view);

    const auto hx = ckl::random_matrix(a.n, 1, 7);
    ckl::DeviceBuffer<float> dx(hx.size());
    ckl::DeviceBuffer<float> dy(static_cast<std::size_t>(a.m));
    dx.copy_from_host(hx);
    dy.zero();

    for (int i = 0; i < launches; ++i) {
        if (!ok(ckl::spmv(plan, algo, 1.0f, dx.data(), 0.0f, dy.data()), variant.c_str())) {
            return 3;
        }
    }
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("ran %s on %s (m=%d n=%d nnz=%d), %d launches\n", variant.c_str(), a.name.c_str(),
                a.m, a.n, a.nnz, launches);
    return 0;
}
#endif

#ifdef CKL_NCU_FFT
int run_fft(const std::string& variant, int log2n, int launches) {
    static const struct {
        const char* name;
        ckl::FftAlgo algo;
    } kTable[] = {
        {"fft_radix2", ckl::FftAlgo::kRadix2Global},   {"fft_radix4", ckl::FftAlgo::kRadix4Global},
        {"fft_radix8", ckl::FftAlgo::kRadix8Global},   {"fft_four_step", ckl::FftAlgo::kFourStep},
        {"fft_shared", ckl::FftAlgo::kSharedResident}, {"fft_cufft", ckl::FftAlgo::kCufft}};
    ckl::FftAlgo algo = ckl::FftAlgo::kAuto;
    bool found = false;
    for (const auto& e : kTable) {
        if (variant == e.name) {
            algo = e.algo;
            found = true;
        }
    }
    if (!found) {
        std::fprintf(stderr, "unknown fft variant: %s\n", variant.c_str());
        return 2;
    }
    if (log2n < 1 || log2n > 24) {
        std::fprintf(stderr, "fft log2n out of range: %d\n", log2n);
        return 2;
    }
    const int n = 1 << log2n;

    // The shared resident rung holds both ping pong buffers in shared memory, so
    // it only exists at the small sizes, and one transform of 2^13 points is a
    // single block on a 48 SM device. Batch it to two blocks per SM, which is
    // what the sweep row for that rung does, or the page measures a launch that
    // leaves 95 of 96 blocks idle.
    ckl::FftPlanOptions opt;
    opt.batch = algo == ckl::FftAlgo::kSharedResident ? 96 : 1;
    ckl::FftPlan plan(n, opt);

    const std::size_t count = static_cast<std::size_t>(n) * static_cast<std::size_t>(opt.batch);
    std::vector<float2> hin(count);
    const auto noise = ckl::random_matrix(static_cast<int>(count), 2, 303);
    for (std::size_t i = 0; i < count; ++i) {
        hin[i].x = noise[2 * i];
        hin[i].y = noise[2 * i + 1];
    }
    ckl::DeviceBuffer<float2> din(count);
    ckl::DeviceBuffer<float2> dout(count);
    din.copy_from_host(hin);
    dout.zero();

    for (int i = 0; i < launches; ++i) {
        if (!ok(ckl::fft(plan, algo, ckl::FftDirection::kForward, din.data(), dout.data()),
                variant.c_str())) {
            return 3;
        }
    }
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    std::printf("ran %s at n=2^%d batch %d, %d launches\n", variant.c_str(), log2n, opt.batch,
                launches);
    return 0;
}
#endif

#ifdef CKL_NCU_SCAN
int run_scan(const std::string& variant, int log2n, int launches) {
    static const struct {
        const char* name;
        ckl::ReduceAlgo algo;
    } kReduce[] = {{"reduce_vec4", ckl::ReduceAlgo::kVec4},
                   {"reduce_single_pass", ckl::ReduceAlgo::kSinglePass},
                   {"reduce_two_pass", ckl::ReduceAlgo::kTwoPass},
                   {"reduce_cub", ckl::ReduceAlgo::kCub}};
    static const struct {
        const char* name;
        ckl::ScanAlgo algo;
    } kScan[] = {{"scan_lookback", ckl::ScanAlgo::kLookback},
                 {"scan_blelloch", ckl::ScanAlgo::kBlelloch},
                 {"scan_three_kernel", ckl::ScanAlgo::kThreeKernel},
                 {"scan_cub", ckl::ScanAlgo::kCub}};
    if (log2n < 0 || log2n > 30) {
        std::fprintf(stderr, "scan log2n out of range: %d\n", log2n);
        return 2;
    }
    const long long n = 1LL << log2n;

    ckl::ScanPlan plan(n);
    std::vector<float> hin(static_cast<std::size_t>(n), 1.0f);
    ckl::DeviceBuffer<float> din(hin.size());
    din.copy_from_host(hin);

    for (const auto& e : kReduce) {
        if (variant == e.name) {
            ckl::DeviceBuffer<float> dout(1);
            dout.zero();
            for (int i = 0; i < launches; ++i) {
                if (!ok(ckl::reduce(plan, e.algo, ckl::ScanOp::kSum, din.data(), dout.data()),
                        variant.c_str())) {
                    return 3;
                }
            }
            CKL_CUDA_CHECK(cudaDeviceSynchronize());
            std::printf("ran %s at n=2^%d, %d launches\n", variant.c_str(), log2n, launches);
            return 0;
        }
    }
    for (const auto& e : kScan) {
        if (variant == e.name) {
            ckl::DeviceBuffer<float> dout(hin.size());
            dout.zero();
            for (int i = 0; i < launches; ++i) {
                if (!ok(ckl::scan(plan, e.algo, ckl::ScanOp::kSum, false, din.data(), dout.data()),
                        variant.c_str())) {
                    return 3;
                }
            }
            CKL_CUDA_CHECK(cudaDeviceSynchronize());
            std::printf("ran %s at n=2^%d, %d launches\n", variant.c_str(), log2n, launches);
            return 0;
        }
    }
    std::fprintf(stderr, "unknown scan variant: %s\n", variant.c_str());
    return 2;
}
#endif

#if !defined(CKL_NCU_SPARSE) || !defined(CKL_NCU_FFT) || !defined(CKL_NCU_SCAN)
// A family this driver was not built with is named, not ignored.
int not_built(const char* family, const std::string& variant) {
    std::fprintf(stderr, "%s is not in this build, so %s cannot run here\n", family,
                 variant.c_str());
    return 4;
}
#endif

}  // namespace

// NOLINTNEXTLINE(bugprone-exception-escape): a ckl::Error here is fatal by design
int main(int argc, char** argv) {
    if (argc < 3) {
        std::fprintf(stderr, "usage: %s <variant> <arg> [launches]\n", argv[0]);
        return 2;
    }
    const std::string variant = argv[1];
    const std::string arg = argv[2];
    const int launches = argc > 3 ? std::atoi(argv[3]) : 5;

    if (variant.rfind("spmv_", 0) == 0) {
#ifdef CKL_NCU_SPARSE
        return run_spmv(variant, arg, launches);
#else
        return not_built("src/sparse", variant);
#endif
    }
    if (variant.rfind("fft_", 0) == 0) {
#ifdef CKL_NCU_FFT
        return run_fft(variant, std::atoi(arg.c_str()), launches);
#else
        return not_built("src/fft", variant);
#endif
    }
    if (variant.rfind("scan_", 0) == 0 || variant.rfind("reduce_", 0) == 0) {
#ifdef CKL_NCU_SCAN
        return run_scan(variant, std::atoi(arg.c_str()), launches);
#else
        return not_built("src/scan", variant);
#endif
    }

    const int n = std::atoi(arg.c_str());
    if (variant == "naive") {
        run<float>(ckl::gemm_naive, n, launches);
    } else if (variant == "tiled") {
        run<float>(ckl::gemm_tiled, n, launches);
    } else if (variant == "register") {
        run<float>(ckl::gemm_register, n, launches);
    } else if (variant == "cp_async") {
        run<float>(ckl::gemm_cp_async, n, launches);
    } else if (variant == "wmma_fp16") {
        run<__half>(ckl::gemm_wmma_fp16, n, launches);
    } else if (variant == "wmma_bf16") {
        run<__nv_bfloat16>(ckl::gemm_wmma_bf16, n, launches);
    } else if (variant == "mma_ptx") {
        run<__half>(ckl::gemm_mma_ptx, n, launches);
    } else if (variant == "mma_ldm") {
        run<__half>(ckl::gemm_mma_ldm, n, launches);
    } else if (variant == "mma_opt") {
        run<__half>(ckl::gemm_mma_opt, n, launches);
    } else if (variant == "cublas_fp16") {
        run<__half>(ckl::gemm_cublas_fp16, n, launches);
    } else if (variant == "cublas_fp32") {
        run<float>(ckl::gemm_cublas, n, launches);
    } else {
        std::fprintf(stderr, "unknown variant: %s\n", variant.c_str());
        return 2;
    }
    std::printf("ran %s at %d cubed, %d launches\n", variant.c_str(), n, launches);
    return 0;
}
