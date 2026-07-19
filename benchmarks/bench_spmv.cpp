// SpMV benchmark on a skewed degree matrix, where the warp per row kernel should
// pull ahead of the thread per row kernel because a few long rows no longer
// serialize a single thread. Reports GFLOP/s (2 * nnz per SpMV) and percent of
// cuSPARSE.

#include <algorithm>
#include <cstdio>
#include <cstdlib>
#include <functional>
#include <random>
#include <vector>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/event_timer.hpp"
#include "ckl/reference.hpp"
#include "ckl/sparse.hpp"

namespace {

using LaunchFn = std::function<void(const int*, const int*, const float*, const float*,
                                    float*, int, int, int, float, float, cudaStream_t)>;

struct Csr {
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
    int nnz;
};

Csr make_skewed(int m, int n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<float> val(-1.0f, 1.0f);
    std::uniform_int_distribution<int> col_dist(0, n - 1);
    Csr a;
    a.row_ptr.assign(static_cast<std::size_t>(m) + 1, 0);
    for (int i = 0; i < m; ++i) {
        int degree = 8 + static_cast<int>(unit(rng) * 16.0);
        if (unit(rng) < 0.02) {
            degree = std::min(n, 400 + static_cast<int>(unit(rng) * 600.0));
        }
        degree = std::min(degree, n);
        std::vector<int> cols;
        while (static_cast<int>(cols.size()) < degree) {
            int c = col_dist(rng);
            if (std::find(cols.begin(), cols.end(), c) == cols.end()) cols.push_back(c);
        }
        std::sort(cols.begin(), cols.end());
        for (int c : cols) {
            a.col_idx.push_back(c);
            a.values.push_back(val(rng));
        }
        a.row_ptr[static_cast<std::size_t>(i) + 1] = a.row_ptr[static_cast<std::size_t>(i)] + degree;
    }
    a.nnz = a.row_ptr.back();
    return a;
}

}  // namespace

int main(int argc, char** argv) {
    const int m = argc > 1 ? std::atoi(argv[1]) : 65536;
    const int n = argc > 2 ? std::atoi(argv[2]) : 65536;
    const Csr a = make_skewed(m, n, 2024);
    std::printf("SpMV skewed matrix: m=%d n=%d nnz=%d\n", m, n, a.nnz);

    const auto x = ckl::random_matrix(n, 1, 1);
    ckl::DeviceBuffer<int> drp(a.row_ptr.size());
    ckl::DeviceBuffer<int> dci(a.col_idx.size());
    ckl::DeviceBuffer<float> dv(a.values.size());
    ckl::DeviceBuffer<float> dx(x.size());
    ckl::DeviceBuffer<float> dy(static_cast<std::size_t>(m));
    drp.copy_from_host(a.row_ptr);
    dci.copy_from_host(a.col_idx);
    dv.copy_from_host(a.values);
    dx.copy_from_host(x);
    dy.zero();

    const std::vector<std::pair<std::string, LaunchFn>> variants = {
        {"naive", ckl::spmv_csr_naive},
        {"warp", ckl::spmv_csr_warp},
        {"cusparse", ckl::spmv_cusparse},
    };
    const double flops = 2.0 * static_cast<double>(a.nnz);
    double base = 0.0;
    std::vector<std::pair<std::string, double>> rows;
    for (const auto& v : variants) {
        ckl::TimingStats st = ckl::time_stream([&](cudaStream_t s) {
            v.second(drp.data(), dci.data(), dv.data(), dx.data(), dy.data(),
                     m, n, a.nnz, 1.0f, 0.0f, s);
        });
        const double gflops = flops / (st.median_ms / 1000.0) / 1.0e9;
        if (v.first == "cusparse") base = gflops;
        rows.emplace_back(v.first, gflops);
    }
    std::printf("%-10s %12s %12s\n", "variant", "gflops", "pct_cusparse");
    for (const auto& r : rows) {
        const double pct = base > 0.0 ? 100.0 * r.second / base : 0.0;
        std::printf("%-10s %12.1f %11.1f%%\n", r.first.c_str(), r.second, pct);
    }
    return 0;
}
