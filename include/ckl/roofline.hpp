#pragma once

// Roofline model. Analytical mode gives the closed form FLOP and byte counts per
// operation, hence the arithmetic intensity; empirical mode takes a measured
// achieved throughput and places the operating point against the machine's
// measured ceilings. The ridge point is peak compute over peak bandwidth; an
// operation whose intensity is left of the ridge is memory bound, right of it is
// compute bound. Williams, Waterman, Patterson, CACM 2009.

#include <cstddef>
#include <string>

namespace ckl {

// Measured machine ceilings, from the Phase 0 device probe rather than the
// datasheet. Bandwidth in bytes per second, compute in FLOP per second.
struct MachineCeilings {
    double bandwidth_bytes_per_s;  // measured streaming bandwidth
    double fp32_flops_per_s;       // FP32 CUDA core peak
    double tensor_flops_per_s;     // tensor core peak (empirical, e.g. best cuBLAS FP16)
};

struct RooflinePoint {
    std::string label;
    double flops;                 // total FLOPs of the operation
    double bytes;                 // total bytes moved (min traffic model)
    double achieved_flops_per_s;  // measured, from a timed run
    bool tensor;                  // compare against the tensor ceiling if true

    double intensity() const { return bytes > 0.0 ? flops / bytes : 0.0; }
    double achieved_gflops() const { return achieved_flops_per_s / 1.0e9; }
};

// Ridge intensity (FLOP per byte) for a ceiling: where the compute roof meets the
// bandwidth diagonal.
inline double ridge_intensity(double flops_per_s, double bandwidth_bytes_per_s) {
    return bandwidth_bytes_per_s > 0.0 ? flops_per_s / bandwidth_bytes_per_s : 0.0;
}

// Roofline attainable performance at a given intensity: min(compute roof,
// bandwidth diagonal).
inline double roofline_flops(double intensity, double flops_per_s, double bandwidth_bytes_per_s) {
    const double mem_bound = intensity * bandwidth_bytes_per_s;
    return mem_bound < flops_per_s ? mem_bound : flops_per_s;
}

// Analytical FLOP and byte counts.
// GEMM C(m,n) = A(m,k) B(k,n): 2*m*n*k FLOPs; minimum traffic reads A, B and
// writes C once, so bytes = (m*k + k*n + m*n) * element_size.
inline double gemm_flops(int m, int n, int k) {
    return 2.0 * static_cast<double>(m) * n * k;
}
inline double gemm_bytes(int m, int n, int k, std::size_t element_size) {
    const double words =
        static_cast<double>(m) * k + static_cast<double>(k) * n + static_cast<double>(m) * n;
    return words * static_cast<double>(element_size);
}

// GEMV y(m) = A(m,n) x(n): 2*m*n FLOPs; traffic reads A once (dominant) plus x
// and y.
inline double gemv_flops(int m, int n) {
    return 2.0 * static_cast<double>(m) * n;
}
inline double gemv_bytes(int m, int n, std::size_t element_size) {
    const double words = static_cast<double>(m) * n + n + m;
    return words * static_cast<double>(element_size);
}

}  // namespace ckl
