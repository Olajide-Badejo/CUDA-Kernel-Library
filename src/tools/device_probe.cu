// Phase 0 device probe. Prints the properties the roofline and the sweep need
// as ground truth, then measures an effective device to device bandwidth so
// later phases quote a measured ceiling rather than the datasheet number.
//
// Run: device_probe            prints properties plus a bandwidth measurement
//      device_probe --json     same data as one JSON object on stdout
//
// The bandwidth number here is a floor on achievable HBM bandwidth from a
// simple streaming copy. The roofline profiler in Phase 8 refines it; this is
// enough to sanity check the datasheet 672 GB/s figure at Phase 0.

#include <cstdio>
#include <string>
#include <vector>

#include <cuda_runtime.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"

namespace {

// Grid stride copy of float4 elements. Reads N float4 and writes N float4, so
// the moved traffic is 2 * N * sizeof(float4) bytes.
__global__ void stream_copy(const float4* __restrict__ src, float4* __restrict__ dst,
                            std::size_t n) {
    std::size_t stride = static_cast<std::size_t>(blockDim.x) * gridDim.x;
    for (std::size_t i = blockIdx.x * blockDim.x + threadIdx.x; i < n; i += stride) {
        dst[i] = src[i];
    }
}

double measure_bandwidth_gbps(int sm_count) {
    const std::size_t n_vec = std::size_t{1} << 25;  // 32 M float4 = 512 MiB per buffer
    ckl::DeviceBuffer<float4> src(n_vec);
    ckl::DeviceBuffer<float4> dst(n_vec);
    src.zero();
    dst.zero();

    const int block = 256;
    const int grid = sm_count * 32;

    // Warm up.
    for (int i = 0; i < 3; ++i) {
        stream_copy<<<grid, block>>>(src.data(), dst.data(), n_vec);
    }
    CKL_CUDA_LAST_ERROR(true);

    cudaEvent_t start;
    cudaEvent_t stop;
    CKL_CUDA_CHECK(cudaEventCreate(&start));
    CKL_CUDA_CHECK(cudaEventCreate(&stop));

    const int reps = 20;
    CKL_CUDA_CHECK(cudaEventRecord(start));
    for (int i = 0; i < reps; ++i) {
        stream_copy<<<grid, block>>>(src.data(), dst.data(), n_vec);
    }
    CKL_CUDA_CHECK(cudaEventRecord(stop));
    CKL_CUDA_CHECK(cudaEventSynchronize(stop));

    float ms = 0.0f;
    CKL_CUDA_CHECK(cudaEventElapsedTime(&ms, start, stop));
    CKL_CUDA_CHECK(cudaEventDestroy(start));
    CKL_CUDA_CHECK(cudaEventDestroy(stop));

    const double bytes_per_rep = 2.0 * static_cast<double>(n_vec) * sizeof(float4);
    const double seconds = (ms / 1000.0) / reps;
    return bytes_per_rep / seconds / 1.0e9;
}

}  // namespace

int main(int argc, char** argv) {
    bool json = (argc > 1 && std::string(argv[1]) == "--json");

    int device = 0;
    CKL_CUDA_CHECK(cudaGetDevice(&device));
    cudaDeviceProp p{};
    CKL_CUDA_CHECK(cudaGetDeviceProperties(&p, device));

    int runtime_version = 0;
    int driver_version = 0;
    CKL_CUDA_CHECK(cudaRuntimeGetVersion(&runtime_version));
    CKL_CUDA_CHECK(cudaDriverGetVersion(&driver_version));

    // CUDA 13 removed clockRate and memoryClockRate from cudaDeviceProp (they
    // were deprecated in 12.x). The attribute query still returns them in kHz.
    int mem_clock_khz = 0;
    int core_clock_khz = 0;
    CKL_CUDA_CHECK(cudaDeviceGetAttribute(&mem_clock_khz, cudaDevAttrMemoryClockRate, device));
    CKL_CUDA_CHECK(cudaDeviceGetAttribute(&core_clock_khz, cudaDevAttrClockRate, device));

    // Theoretical bandwidth from memory clock and bus width. memoryBusWidth is
    // in bits. GDDR7 moves data on both edges so the effective rate multiplier
    // lives in the datasheet, not here; this stays a clock-times-bus figure and
    // the measured value is the one we trust.
    const double mem_clock_hz = static_cast<double>(mem_clock_khz) * 1000.0;
    const double bus_bytes = static_cast<double>(p.memoryBusWidth) / 8.0;
    const double theo_bw_gbps = mem_clock_hz * bus_bytes * 2.0 / 1.0e9;

    const double meas_bw_gbps = measure_bandwidth_gbps(p.multiProcessorCount);

    // Peak FP32: cores * 2 (FMA) * boost clock.
    const double core_clock_hz = static_cast<double>(core_clock_khz) * 1000.0;
    const int cores_per_sm = 128;  // Blackwell consumer SM; re-verify against whitepaper
    const double cuda_cores = static_cast<double>(p.multiProcessorCount) * cores_per_sm;
    const double peak_fp32_tflops = cuda_cores * 2.0 * core_clock_hz / 1.0e12;

    if (json) {
        std::printf("{\n");
        std::printf("  \"name\": \"%s\",\n", p.name);
        std::printf("  \"compute_capability\": \"%d.%d\",\n", p.major, p.minor);
        std::printf("  \"sm_count\": %d,\n", p.multiProcessorCount);
        std::printf("  \"global_mem_mib\": %.0f,\n",
                    static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0));
        std::printf("  \"mem_bus_width_bits\": %d,\n", p.memoryBusWidth);
        std::printf("  \"mem_clock_khz\": %d,\n", mem_clock_khz);
        std::printf("  \"core_clock_khz\": %d,\n", core_clock_khz);
        std::printf("  \"l2_cache_kib\": %d,\n", p.l2CacheSize / 1024);
        std::printf("  \"shared_mem_per_block_kib\": %zu,\n", p.sharedMemPerBlock / 1024);
        std::printf("  \"shared_mem_per_sm_kib\": %zu,\n", p.sharedMemPerMultiprocessor / 1024);
        std::printf("  \"regs_per_sm\": %d,\n", p.regsPerMultiprocessor);
        std::printf("  \"max_threads_per_sm\": %d,\n", p.maxThreadsPerMultiProcessor);
        std::printf("  \"theoretical_bw_gbps\": %.1f,\n", theo_bw_gbps);
        std::printf("  \"measured_bw_gbps\": %.1f,\n", meas_bw_gbps);
        std::printf("  \"peak_fp32_tflops_estimate\": %.1f,\n", peak_fp32_tflops);
        std::printf("  \"runtime_version\": %d,\n", runtime_version);
        std::printf("  \"driver_version\": %d\n", driver_version);
        std::printf("}\n");
        return 0;
    }

    std::printf("CUDA Kernel Lab device probe\n");
    std::printf("============================\n");
    std::printf("Device                 : %s\n", p.name);
    std::printf("Compute capability     : %d.%d (sm_%d%d)\n", p.major, p.minor, p.major, p.minor);
    std::printf("SM count               : %d\n", p.multiProcessorCount);
    std::printf("Global memory          : %.0f MiB\n",
                static_cast<double>(p.totalGlobalMem) / (1024.0 * 1024.0));
    std::printf("Memory bus width       : %d bits\n", p.memoryBusWidth);
    std::printf("Memory clock           : %.0f MHz\n", static_cast<double>(mem_clock_khz) / 1000.0);
    std::printf("Core boost clock       : %.0f MHz\n", core_clock_hz / 1.0e6);
    std::printf("L2 cache               : %d KiB\n", p.l2CacheSize / 1024);
    std::printf("Shared mem per block   : %zu KiB\n", p.sharedMemPerBlock / 1024);
    std::printf("Shared mem per SM      : %zu KiB\n", p.sharedMemPerMultiprocessor / 1024);
    std::printf("Registers per SM       : %d\n", p.regsPerMultiprocessor);
    std::printf("Max threads per SM     : %d\n", p.maxThreadsPerMultiProcessor);
    std::printf("Warp size              : %d\n", p.warpSize);
    std::printf("----------------------------\n");
    std::printf("Theoretical BW (clk*bus): %.1f GB/s\n", theo_bw_gbps);
    std::printf("Measured stream BW      : %.1f GB/s\n", meas_bw_gbps);
    std::printf("Peak FP32 (est, non-TC) : %.1f TFLOP/s\n", peak_fp32_tflops);
    std::printf("----------------------------\n");
    std::printf("CUDA runtime version   : %d\n", runtime_version);
    std::printf("CUDA driver version    : %d\n", driver_version);
    return 0;
}
