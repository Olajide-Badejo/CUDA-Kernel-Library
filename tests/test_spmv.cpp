// CSR SpMV correctness against cuSPARSE and a CPU reference. The test matrix has
// a skewed row degree distribution: most rows are short but a few are very long,
// which is the case where the warp per row kernel earns its keep over the thread
// per row kernel. Correctness must hold for both regardless.

#include <algorithm>
#include <cstdio>
#include <functional>
#include <random>
#include <vector>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/sparse.hpp"
#include "ckl/reference.hpp"

namespace {

struct Csr {
    std::vector<int> row_ptr;
    std::vector<int> col_idx;
    std::vector<float> values;
    int m;
    int n;
    int nnz;
};

// Build a skewed CSR: row degree is small for most rows, large for a few. Column
// indices are unique and sorted per row so the matrix is a valid CSR.
Csr make_skewed_csr(int m, int n, std::uint64_t seed) {
    std::mt19937_64 rng(seed);
    std::uniform_real_distribution<double> unit(0.0, 1.0);
    std::uniform_real_distribution<float> val(-1.0f, 1.0f);

    Csr a;
    a.m = m;
    a.n = n;
    a.row_ptr.assign(static_cast<std::size_t>(m) + 1, 0);
    for (int i = 0; i < m; ++i) {
        int degree = 4 + static_cast<int>(unit(rng) * 8.0);  // most rows: 4 to 12
        if (unit(rng) < 0.02) {
            degree = std::min(n, 300 + static_cast<int>(unit(rng) * 400.0));  // rare long rows
        }
        degree = std::min(degree, n);
        std::vector<int> cols;
        cols.reserve(static_cast<std::size_t>(degree));
        // Sample unique columns.
        std::uniform_int_distribution<int> col_dist(0, n - 1);
        while (static_cast<int>(cols.size()) < degree) {
            int c = col_dist(rng);
            if (std::find(cols.begin(), cols.end(), c) == cols.end()) {
                cols.push_back(c);
            }
        }
        std::sort(cols.begin(), cols.end());
        for (int c : cols) {
            a.col_idx.push_back(c);
            a.values.push_back(val(rng));
        }
        a.row_ptr[static_cast<std::size_t>(i) + 1] =
            a.row_ptr[static_cast<std::size_t>(i)] + degree;
    }
    a.nnz = a.row_ptr.back();
    return a;
}

std::vector<double> spmv_reference(const Csr& a, const std::vector<float>& x,
                                   const std::vector<float>& y0, float alpha, float beta) {
    std::vector<double> y(static_cast<std::size_t>(a.m));
    for (int i = 0; i < a.m; ++i) {
        double acc = 0.0;
        for (int k = a.row_ptr[i]; k < a.row_ptr[i + 1]; ++k) {
            acc += static_cast<double>(a.values[k]) * static_cast<double>(x[a.col_idx[k]]);
        }
        y[i] = static_cast<double>(alpha) * acc + static_cast<double>(beta) * y0[i];
    }
    return y;
}

using LaunchFn = std::function<void(const int*, const int*, const float*, const float*,
                                    float*, int, int, int, float, float, cudaStream_t)>;

std::vector<float> run(const LaunchFn& launch, const Csr& a, const ckl::DeviceBuffer<int>& drp,
                       const ckl::DeviceBuffer<int>& dci, const ckl::DeviceBuffer<float>& dv,
                       const ckl::DeviceBuffer<float>& dx, const std::vector<float>& y0,
                       float alpha, float beta) {
    ckl::DeviceBuffer<float> dy(y0.size());
    dy.copy_from_host(y0);
    launch(drp.data(), dci.data(), dv.data(), dx.data(), dy.data(),
           a.m, a.n, a.nnz, alpha, beta, nullptr);
    CKL_CUDA_CHECK(cudaDeviceSynchronize());
    return dy.to_host();
}

}  // namespace

int main() {
    const int m = 4096;
    const int n = 4096;
    const float alpha = 1.1f;
    const float beta = 0.3f;
    const Csr a = make_skewed_csr(m, n, 12345);

    int max_deg = 0;
    for (int i = 0; i < m; ++i) {
        max_deg = std::max(max_deg, a.row_ptr[i + 1] - a.row_ptr[i]);
    }
    std::printf("CSR SpMV correctness: m=%d n=%d nnz=%d max_row_degree=%d (skewed)\n",
                m, n, a.nnz, max_deg);

    const auto x = ckl::random_matrix(n, 1, 999);
    const auto y0 = ckl::random_matrix(m, 1, 555);

    ckl::DeviceBuffer<int> drp(a.row_ptr.size());
    ckl::DeviceBuffer<int> dci(a.col_idx.size());
    ckl::DeviceBuffer<float> dv(a.values.size());
    ckl::DeviceBuffer<float> dx(x.size());
    drp.copy_from_host(a.row_ptr);
    dci.copy_from_host(a.col_idx);
    dv.copy_from_host(a.values);
    dx.copy_from_host(x);

    const auto ref = spmv_reference(a, x, y0, alpha, beta);
    const auto y_cusparse = run(ckl::spmv_cusparse, a, drp, dci, dv, dx, y0, alpha, beta);
    const std::vector<double> cusparse_ref(y_cusparse.begin(), y_cusparse.end());
    const double err_oracle = ckl::relative_frobenius_error(y_cusparse, ref);
    std::printf("  cusparse vs cpu = %.3e\n", err_oracle);

    bool ok = err_oracle < 1e-4;
    const std::vector<std::pair<const char*, LaunchFn>> variants = {
        {"naive", ckl::spmv_csr_naive},
        {"warp", ckl::spmv_csr_warp},
    };
    for (const auto& v : variants) {
        const auto y = run(v.second, a, drp, dci, dv, dx, y0, alpha, beta);
        const double err = ckl::relative_frobenius_error(y, cusparse_ref);
        const bool pass = err < 1e-4;
        ok = ok && pass;
        std::printf("      %-6s vs cuSPARSE = %.3e  %s\n", v.first, err, pass ? "PASS" : "FAIL");
    }
    std::printf("%s\n", ok ? "all SpMV cases passed" : "SpMV cases FAILED");
    return ok ? 0 : 1;
}
