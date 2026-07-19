// One configuration, one JSON row. bench_all runs a single (family, variant,
// dtype, shape) benchmark with NVML sampling on, times both the kernel and its
// vendor baseline in the same process, and prints one JSON object. sweep.py drives
// it across the matrix and appends the rows to the canonical JSONL. Keeping the
// measurement in a small binary that does exactly one config makes the sweep
// resumable: a killed sweep just reruns the configs that have no row yet.
//
// Usage: bench_all <family> <variant> <dtype> <m> <n> <k> [commit]
//   family in {gemm, gemv, spmv, trsm}; dtype in {fp32, fp16, bf16}

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <string>
#include <vector>

#include <cuda_bf16.h>
#include <cuda_fp16.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/event_timer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/gemv.hpp"
#include "ckl/nvml_monitor.hpp"
#include "ckl/reference.hpp"
#include "ckl/sparse.hpp"
#include "ckl/trsm.hpp"

namespace {

struct Result {
    double median_ms = 0.0;
    double iqr_ms = 0.0;
    double gflops = 0.0;
    double baseline_gflops = 0.0;
    bool ok = true;
};

template <typename T>
T from_float(float f);
template <>
float from_float<float>(float f) { return f; }
template <>
__half from_float<__half>(float f) { return __float2half(f); }
template <>
__nv_bfloat16 from_float<__nv_bfloat16>(float f) { return __float2bfloat16(f); }

template <typename T>
std::vector<T> convert(const std::vector<float>& src) {
    std::vector<T> out(src.size());
    for (std::size_t i = 0; i < src.size(); ++i) out[i] = from_float<T>(src[i]);
    return out;
}

double gflops_of(double flops, double ms) { return flops / (ms / 1000.0) / 1.0e9; }

// GEMM, any precision. Times the variant and its cuBLAS baseline of the same dtype.
template <typename T, typename KFn, typename OFn>
Result bench_gemm_typed(KFn kernel, OFn oracle, int m, int n, int k) {
    const auto fa = ckl::random_matrix(m, k, 11);
    const auto fb = ckl::random_matrix(k, n, 22);
    const auto a = convert<T>(fa);
    const auto b = convert<T>(fb);
    ckl::DeviceBuffer<T> da(a.size());
    ckl::DeviceBuffer<T> db(b.size());
    ckl::DeviceBuffer<float> dc(static_cast<std::size_t>(m) * n);
    da.copy_from_host(a);
    db.copy_from_host(b);
    dc.zero();
    const double flops = 2.0 * m * n * k;
    ckl::TimingStats ks = ckl::time_stream([&](cudaStream_t s) {
        kernel(da.data(), db.data(), dc.data(), m, n, k, 1.0f, 0.0f, s);
    });
    ckl::TimingStats os = ckl::time_stream([&](cudaStream_t s) {
        oracle(da.data(), db.data(), dc.data(), m, n, k, 1.0f, 0.0f, s);
    });
    return {ks.median_ms, ks.iqr_ms, gflops_of(flops, ks.median_ms),
            gflops_of(flops, os.median_ms), true};
}

Result bench_gemm(const std::string& v, const std::string& dtype, int m, int n, int k) {
    if (dtype == "fp32") {
        std::function<void(const float*, const float*, float*, int, int, int, float, float, cudaStream_t)> kf;
        if (v == "naive") kf = ckl::gemm_naive;
        else if (v == "tiled") kf = ckl::gemm_tiled;
        else if (v == "register") kf = ckl::gemm_register;
        else if (v == "cp_async") kf = ckl::gemm_cp_async;
        else return {0, 0, 0, 0, false};
        return bench_gemm_typed<float>(kf, ckl::gemm_cublas, m, n, k);
    }
    if (dtype == "fp16") {
        std::function<void(const __half*, const __half*, float*, int, int, int, float, float, cudaStream_t)> kf;
        if (v == "wmma") kf = ckl::gemm_wmma_fp16;
        else if (v == "mma_ptx") kf = ckl::gemm_mma_ptx;
        else if (v == "mma_ldm") kf = ckl::gemm_mma_ldm;
        else if (v == "mma_opt") kf = ckl::gemm_mma_opt;
        else return {0, 0, 0, 0, false};
        return bench_gemm_typed<__half>(kf, ckl::gemm_cublas_fp16, m, n, k);
    }
    if (dtype == "bf16") {
        if (v != "wmma") return {0, 0, 0, 0, false};
        return bench_gemm_typed<__nv_bfloat16>(ckl::gemm_wmma_bf16, ckl::gemm_cublas_bf16, m, n, k);
    }
    return {0, 0, 0, 0, false};
}

Result bench_gemv(const std::string& v, int m, int n) {
    std::function<void(const float*, const float*, float*, int, int, float, float, cudaStream_t)> kf;
    if (v == "naive") kf = ckl::gemv_naive;
    else if (v == "warp") kf = ckl::gemv_warp;
    else if (v == "vectorized") kf = ckl::gemv_vectorized;
    else return {0, 0, 0, 0, false};
    const auto a = ckl::random_matrix(m, n, 3);
    const auto x = ckl::random_matrix(n, 1, 5);
    ckl::DeviceBuffer<float> da(a.size()), dx(x.size()), dy(static_cast<std::size_t>(m));
    da.copy_from_host(a);
    dx.copy_from_host(x);
    dy.zero();
    const double flops = 2.0 * m * n;
    ckl::TimingStats ks = ckl::time_stream([&](cudaStream_t s) {
        kf(da.data(), dx.data(), dy.data(), m, n, 1.0f, 0.0f, s);
    });
    ckl::TimingStats os = ckl::time_stream([&](cudaStream_t s) {
        ckl::gemv_cublas(da.data(), dx.data(), dy.data(), m, n, 1.0f, 0.0f, s);
    });
    return {ks.median_ms, ks.iqr_ms, gflops_of(flops, ks.median_ms),
            gflops_of(flops, os.median_ms), true};
}

Result bench_spmv(const std::string& v, int m, int n) {
    std::mt19937_64 rng(2024);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<float> val(-1.0f, 1.0f);
    std::uniform_int_distribution<int> col(0, n - 1);
    std::vector<int> row_ptr(static_cast<std::size_t>(m) + 1, 0);
    std::vector<int> col_idx;
    std::vector<float> values;
    for (int i = 0; i < m; ++i) {
        int deg = 8 + static_cast<int>(unit(rng) * 16.0);
        if (unit(rng) < 0.02) deg = std::min(n, 400 + static_cast<int>(unit(rng) * 600.0));
        deg = std::min(deg, n);
        std::vector<int> cols;
        while (static_cast<int>(cols.size()) < deg) {
            int c = col(rng);
            if (std::find(cols.begin(), cols.end(), c) == cols.end()) cols.push_back(c);
        }
        std::sort(cols.begin(), cols.end());
        for (int c : cols) { col_idx.push_back(c); values.push_back(val(rng)); }
        row_ptr[static_cast<std::size_t>(i) + 1] = row_ptr[static_cast<std::size_t>(i)] + deg;
    }
    const int nnz = row_ptr.back();
    const auto x = ckl::random_matrix(n, 1, 7);
    ckl::DeviceBuffer<int> drp(row_ptr.size()), dci(col_idx.size());
    ckl::DeviceBuffer<float> dv(values.size()), dx(x.size()), dy(static_cast<std::size_t>(m));
    drp.copy_from_host(row_ptr);
    dci.copy_from_host(col_idx);
    dv.copy_from_host(values);
    dx.copy_from_host(x);
    dy.zero();
    std::function<void(const int*, const int*, const float*, const float*, float*, int, int, int, float, float, cudaStream_t)> kf;
    if (v == "naive") kf = ckl::spmv_csr_naive;
    else if (v == "warp") kf = ckl::spmv_csr_warp;
    else return {0, 0, 0, 0, false};
    const double flops = 2.0 * nnz;
    ckl::TimingStats ks = ckl::time_stream([&](cudaStream_t s) {
        kf(drp.data(), dci.data(), dv.data(), dx.data(), dy.data(), m, n, nnz, 1.0f, 0.0f, s);
    });
    ckl::TimingStats os = ckl::time_stream([&](cudaStream_t s) {
        ckl::spmv_cusparse(drp.data(), dci.data(), dv.data(), dx.data(), dy.data(), m, n, nnz, 1.0f, 0.0f, s);
    });
    return {ks.median_ms, ks.iqr_ms, gflops_of(flops, ks.median_ms),
            gflops_of(flops, os.median_ms), true};
}

Result bench_trsm(const std::string& v, int m, int n) {
    auto a = ckl::random_matrix(m, m, 71);
    for (int i = 0; i < m; ++i) {
        for (int j = i + 1; j < m; ++j) a[static_cast<std::size_t>(i) * m + j] = 0.0f;
        a[static_cast<std::size_t>(i) * m + i] = static_cast<float>(m + 1);
    }
    const auto b0 = ckl::random_matrix(m, n, 92);
    ckl::DeviceBuffer<float> da(a.size()), db(b0.size());
    da.copy_from_host(a);
    std::function<void(const float*, float*, int, int, float, cudaStream_t)> kf;
    if (v == "naive") kf = ckl::trsm_naive;
    else if (v == "blocked") kf = ckl::trsm_blocked;
    else return {0, 0, 0, 0, false};
    // TRSM flops approx m*m*n (triangular solve).
    const double flops = static_cast<double>(m) * m * n;
    ckl::TimingStats ks = ckl::time_stream([&](cudaStream_t s) {
        db.copy_from_host(b0);
        kf(da.data(), db.data(), m, n, 1.0f, s);
    });
    ckl::TimingStats os = ckl::time_stream([&](cudaStream_t s) {
        db.copy_from_host(b0);
        ckl::trsm_cublas(da.data(), db.data(), m, n, 1.0f, s);
    });
    return {ks.median_ms, ks.iqr_ms, gflops_of(flops, ks.median_ms),
            gflops_of(flops, os.median_ms), true};
}

}  // namespace

int main(int argc, char** argv) {
    if (argc < 7) {
        std::fprintf(stderr, "usage: %s <family> <variant> <dtype> <m> <n> <k> [commit]\n", argv[0]);
        return 2;
    }
    const std::string family = argv[1];
    const std::string variant = argv[2];
    const std::string dtype = argv[3];
    const int m = std::atoi(argv[4]);
    const int n = std::atoi(argv[5]);
    const int k = std::atoi(argv[6]);
    const std::string commit = argc > 7 ? argv[7] : "unknown";

    ckl::NvmlMonitor monitor(25);
    monitor.start();

    Result r;
    if (family == "gemm") r = bench_gemm(variant, dtype, m, n, k);
    else if (family == "gemv") r = bench_gemv(variant, m, n);
    else if (family == "spmv") r = bench_spmv(variant, m, n);
    else if (family == "trsm") r = bench_trsm(variant, m, n);
    else { std::fprintf(stderr, "unknown family %s\n", family.c_str()); return 2; }

    ckl::NvmlSummary nv = monitor.stop();
    if (!r.ok) {
        std::fprintf(stderr, "unknown variant %s for family %s dtype %s\n",
                     variant.c_str(), family.c_str(), dtype.c_str());
        return 3;
    }

    int rt = 0;
    int drv = 0;
    cudaRuntimeGetVersion(&rt);
    cudaDriverGetVersion(&drv);

    const double pct = r.baseline_gflops > 0.0 ? 100.0 * r.gflops / r.baseline_gflops : 0.0;
    std::printf(
        "{\"family\":\"%s\",\"variant\":\"%s\",\"dtype\":\"%s\",\"m\":%d,\"n\":%d,\"k\":%d,"
        "\"median_ms\":%.6f,\"iqr_ms\":%.6f,\"gflops\":%.3f,\"baseline_gflops\":%.3f,"
        "\"pct_baseline\":%.2f,\"nvml_available\":%s,\"median_sm_clock_mhz\":%.0f,"
        "\"max_temp_c\":%u,\"max_power_w\":%.1f,\"throttled\":%s,\"nvml_samples\":%d,"
        "\"cuda_runtime\":%d,\"cuda_driver\":%d,\"commit\":\"%s\"}\n",
        family.c_str(), variant.c_str(), dtype.c_str(), m, n, k,
        r.median_ms, r.iqr_ms, r.gflops, r.baseline_gflops, pct,
        nv.available ? "true" : "false", nv.median_sm_clock_mhz, nv.max_temperature_c,
        nv.max_power_w, nv.throttled ? "true" : "false", nv.samples, rt, drv, commit.c_str());
    return 0;
}
