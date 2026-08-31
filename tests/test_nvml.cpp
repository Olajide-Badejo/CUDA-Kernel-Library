// NVML telemetry. The monitor samples clocks, temperature, power and the
// driver's throttle reasons on a background thread; the measurement protocol
// treats a throttled window as a result to rerun rather than report, so what
// this suite checks is that the monitor actually collects samples and that its
// lifetime rules hold.
//
// Whether a short burst throttles depends on the machine's thermal state, so the
// throttled flag is reported, never asserted. On a host where NVML cannot
// initialize (a container without the management library, for instance) the
// monitor says so through NvmlSummary::available and the tests skip rather than
// fail: that is a property of the host, not of the code.

#include <chrono>
#include <string>
#include <utility>
#include <vector>

#include <cuda_runtime.h>

#include <gtest/gtest.h>

#include "ckl/cuda_check.hpp"
#include "ckl/device_buffer.hpp"
#include "ckl/gemm.hpp"
#include "ckl/nvml_monitor.hpp"
#include "gpu_environment.hpp"
#include "reference.hpp"

namespace {

using ckl::test::GpuTest;

// Keeps the GPU busy for the requested window so the sampler has something to
// see. The work is real GEMM, not a spin, because idle clocks tell you nothing.
void busy_for(double seconds) {
    const int n = 1024;
    const auto ha = ckl::random_matrix(n, n, ckl::test::seed_stream(7101));
    const auto hb = ckl::random_matrix(n, n, ckl::test::seed_stream(7102));
    ckl::DeviceBuffer<float> da(ha.size());
    ckl::DeviceBuffer<float> db(hb.size());
    ckl::DeviceBuffer<float> dc(static_cast<std::size_t>(n) * static_cast<std::size_t>(n));
    da.copy_from_host(ha);
    db.copy_from_host(hb);
    dc.zero();

    const auto t0 = std::chrono::steady_clock::now();
    while (std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count() < seconds) {
        for (int i = 0; i < 20; ++i) {
            ckl::gemm_cublas(da.data(), db.data(), dc.data(), n, n, n, 1.0f, 0.0f, nullptr);
        }
        CKL_CUDA_CHECK(cudaDeviceSynchronize());
    }
}

class Nvml : public GpuTest {};

TEST_F(Nvml, CollectsSamplesAcrossAWindow) {
    ckl::NvmlMonitor monitor(20);
    monitor.start();
    busy_for(0.4);
    const ckl::NvmlSummary s = monitor.stop();

    if (!s.available) {
        GTEST_SKIP() << "NVML is not available on this host: " << s.note;
    }

    EXPECT_GT(s.samples, 0) << "a 400 ms window at a 20 ms interval has to produce samples";
    EXPECT_EQ(s.samples, static_cast<int>(monitor.samples().size()));
    EXPECT_GT(s.median_sm_clock_mhz, 0.0);
    EXPECT_GT(s.max_temperature_c, 0u);
    EXPECT_GT(s.max_power_w, 0.0);
    EXPECT_GE(s.mean_gpu_util_pct, 0.0);
    EXPECT_LE(s.mean_gpu_util_pct, 100.0);

    // Reported, not asserted: a throttled window is a reason to rerun a sweep
    // after cooldown, and the sweep is where that decision belongs.
    ::testing::Test::RecordProperty("throttled", s.throttled ? "yes" : "no");
    ::testing::Test::RecordProperty("median_sm_clock_mhz", ckl::test::sci(s.median_sm_clock_mhz));
    ::testing::Test::RecordProperty("max_power_w", ckl::test::sci(s.max_power_w));
}

TEST_F(Nvml, SamplesCarryTheFieldsTheSummaryIsBuiltFrom) {
    ckl::NvmlMonitor monitor(10);
    monitor.start();
    busy_for(0.2);
    const ckl::NvmlSummary s = monitor.stop();
    if (!s.available) {
        GTEST_SKIP() << "NVML is not available on this host: " << s.note;
    }
    ASSERT_FALSE(monitor.samples().empty());

    double last_time = -1.0;
    unsigned int max_temp = 0;
    for (const ckl::NvmlSample& sample : monitor.samples()) {
        EXPECT_GE(sample.time_s, last_time) << "sample timestamps have to be non decreasing";
        last_time = sample.time_s;
        EXPECT_LE(sample.gpu_util_pct, 100u);
        max_temp = sample.temperature_c > max_temp ? sample.temperature_c : max_temp;
    }
    EXPECT_EQ(s.max_temperature_c, max_temp) << "the summary has to agree with the samples";
}

TEST_F(Nvml, AMovedMonitorCarriesItsWindow) {
    ckl::NvmlMonitor first(20);
    first.start();
    busy_for(0.15);
    ckl::NvmlMonitor second(std::move(first));
    const ckl::NvmlSummary s = second.stop();
    if (!s.available) {
        GTEST_SKIP() << "NVML is not available on this host: " << s.note;
    }
    EXPECT_GT(s.samples, 0);
}

// A monitor that never ran still answers, and answers honestly: zero samples,
// not a fabricated clock.
TEST_F(Nvml, StoppingWithoutStartingReportsNothingRatherThanGuessing) {
    ckl::NvmlMonitor monitor(20);
    const ckl::NvmlSummary s = monitor.stop();
    EXPECT_EQ(s.samples, 0);
    EXPECT_EQ(s.median_sm_clock_mhz, 0.0);
    EXPECT_FALSE(s.throttled);
}

}  // namespace
