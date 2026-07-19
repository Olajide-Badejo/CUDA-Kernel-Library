// NVML monitor implementation. A worker thread polls the device at a fixed
// interval and appends samples; stop() joins it and reduces the samples to a
// summary. NVML init failure is not fatal: the summary comes back with available
// false and a note, so a machine or container without NVML still runs the sweep
// (just without throttle checking, which the caller can decide how to treat).

#include "ckl/nvml_monitor.hpp"

#include <algorithm>
#include <atomic>
#include <chrono>
#include <thread>

#include <nvml.h>

namespace ckl {

namespace {

// Throttle reasons that mean the result should not be trusted. GfxIdle and the
// plain "none" bit are not real throttles.
constexpr unsigned long long kBadThrottle =
    nvmlClocksThrottleReasonSwPowerCap | nvmlClocksThrottleReasonHwSlowdown |
    nvmlClocksThrottleReasonHwThermalSlowdown | nvmlClocksThrottleReasonHwPowerBrakeSlowdown |
    nvmlClocksThrottleReasonSwThermalSlowdown;

}  // namespace

struct NvmlMonitor::Impl {
    unsigned int interval_ms;
    int device_index;
    nvmlDevice_t device = nullptr;
    bool nvml_ok = false;
    std::string note;
    std::atomic<bool> running{false};
    std::thread worker;
    std::vector<NvmlSample> samples;
    std::chrono::steady_clock::time_point t0;

    void poll_once() {
        NvmlSample s{};
        s.time_s = std::chrono::duration<double>(std::chrono::steady_clock::now() - t0).count();
        unsigned int v = 0;
        if (nvmlDeviceGetClockInfo(device, NVML_CLOCK_SM, &v) == NVML_SUCCESS)
            s.sm_clock_mhz = v;
        nvmlTemperature_t temp{};
        temp.version = nvmlTemperature_v1;
        temp.sensorType = NVML_TEMPERATURE_GPU;
        if (nvmlDeviceGetTemperatureV(device, &temp) == NVML_SUCCESS) {
            s.temperature_c = static_cast<unsigned int>(temp.temperature);
        }
        if (nvmlDeviceGetPowerUsage(device, &v) == NVML_SUCCESS)
            s.power_mw = v;
        nvmlUtilization_t util{};
        if (nvmlDeviceGetUtilizationRates(device, &util) == NVML_SUCCESS)
            s.gpu_util_pct = util.gpu;
        unsigned long long reasons = 0;
        if (nvmlDeviceGetCurrentClocksEventReasons(device, &reasons) == NVML_SUCCESS) {
            s.throttle_reasons = reasons;
        }
        samples.push_back(s);
    }
};

NvmlMonitor::NvmlMonitor(unsigned int sample_interval_ms, int device_index) : impl_(new Impl()) {
    impl_->interval_ms = sample_interval_ms;
    impl_->device_index = device_index;
    if (nvmlInit_v2() != NVML_SUCCESS) {
        impl_->note = "nvmlInit failed";
        return;
    }
    if (nvmlDeviceGetHandleByIndex_v2(static_cast<unsigned int>(device_index), &impl_->device) !=
        NVML_SUCCESS) {
        impl_->note = "nvmlDeviceGetHandleByIndex failed";
        nvmlShutdown();
        return;
    }
    impl_->nvml_ok = true;
}

NvmlMonitor::~NvmlMonitor() {
    if (impl_ != nullptr) {
        if (impl_->running.load()) {
            stop();
        }
        if (impl_->nvml_ok) {
            nvmlShutdown();
        }
        delete impl_;
    }
}

void NvmlMonitor::start() {
    if (!impl_->nvml_ok || impl_->running.load()) {
        return;
    }
    impl_->samples.clear();
    impl_->t0 = std::chrono::steady_clock::now();
    impl_->running.store(true);
    impl_->worker = std::thread([this] {
        while (impl_->running.load()) {
            impl_->poll_once();
            std::this_thread::sleep_for(std::chrono::milliseconds(impl_->interval_ms));
        }
    });
}

NvmlSummary NvmlMonitor::stop() {
    NvmlSummary out;
    if (!impl_->nvml_ok) {
        out.available = false;
        out.note = impl_->note.empty() ? "NVML unavailable" : impl_->note;
        return out;
    }
    if (impl_->running.load()) {
        impl_->running.store(false);
        if (impl_->worker.joinable()) {
            impl_->worker.join();
        }
    }
    out.available = true;
    out.samples = static_cast<int>(impl_->samples.size());
    if (impl_->samples.empty()) {
        out.note = "no samples";
        return out;
    }

    std::vector<double> clocks;
    clocks.reserve(impl_->samples.size());
    double util_sum = 0.0;
    for (const auto& s : impl_->samples) {
        clocks.push_back(static_cast<double>(s.sm_clock_mhz));
        out.max_temperature_c = std::max(out.max_temperature_c, s.temperature_c);
        out.max_power_w = std::max(out.max_power_w, s.power_mw / 1000.0);
        util_sum += s.gpu_util_pct;
        out.throttle_reasons |= s.throttle_reasons;
    }
    std::sort(clocks.begin(), clocks.end());
    out.median_sm_clock_mhz = clocks[clocks.size() / 2];
    out.mean_gpu_util_pct = util_sum / static_cast<double>(impl_->samples.size());
    out.throttled = (out.throttle_reasons & kBadThrottle) != 0;
    out.note = out.throttled ? "throttling detected" : "clean";
    return out;
}

const std::vector<NvmlSample>& NvmlMonitor::samples() const {
    return impl_->samples;
}

}  // namespace ckl
