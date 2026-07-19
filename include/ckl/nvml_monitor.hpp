#pragma once

// NVML telemetry sampled on a background thread. Start it before a timed sweep,
// stop it after, and read the summary: median SM clock, peak temperature and
// power, and whether the driver flagged any throttling during the window. The
// build spec's rule is that a sweep whose samples show throttling is rerun after
// cooldown rather than reported, so the throttled flag gates whether a result is
// trusted.

#include <cstdint>
#include <string>
#include <vector>

namespace ckl {

struct NvmlSample {
    double time_s;
    unsigned int sm_clock_mhz;
    unsigned int temperature_c;
    unsigned int power_mw;
    unsigned int gpu_util_pct;
    unsigned long long throttle_reasons;
};

struct NvmlSummary {
    int samples = 0;
    double median_sm_clock_mhz = 0.0;
    unsigned int max_temperature_c = 0;
    double max_power_w = 0.0;
    double mean_gpu_util_pct = 0.0;
    bool throttled = false;                 // any thermal, power, or reliability throttle seen
    unsigned long long throttle_reasons = 0;  // union of reasons across samples
    bool available = false;                 // false if NVML could not be initialized
    std::string note;                       // human readable status or error
};

// Samples NVML on a background thread between start() and stop(). Non copyable;
// one monitor drives one window at a time.
class NvmlMonitor {
public:
    explicit NvmlMonitor(unsigned int sample_interval_ms = 25, int device_index = 0);
    ~NvmlMonitor();

    NvmlMonitor(const NvmlMonitor&) = delete;
    NvmlMonitor& operator=(const NvmlMonitor&) = delete;

    void start();
    NvmlSummary stop();

    const std::vector<NvmlSample>& samples() const;

private:
    struct Impl;
    Impl* impl_;
};

}  // namespace ckl
