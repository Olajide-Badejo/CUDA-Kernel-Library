#pragma once

/**
 * @file nvml_monitor.hpp
 * @brief NVML telemetry sampled on a background thread.
 *
 * Start it before a timed sweep, stop it after, and read the summary: median SM
 * clock, peak temperature and power, and whether the driver flagged any
 * throttling during the window. The build spec's rule is that a sweep whose
 * samples show throttling is rerun after cooldown rather than reported, so the
 * throttled flag gates whether a result is trusted.
 */

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

#include "ckl/ckl_export.h"

namespace ckl {

/// @brief One NVML reading, taken at time_s after start().
struct NvmlSample {
    double time_s;                        ///< Seconds since the monitor started.
    unsigned int sm_clock_mhz;            ///< SM clock in MHz at this instant.
    unsigned int temperature_c;           ///< GPU temperature in degrees Celsius.
    unsigned int power_mw;                ///< Board power draw in milliwatts.
    unsigned int gpu_util_pct;            ///< Percent of the sampling period the GPU was busy.
    unsigned long long throttle_reasons;  ///< NVML clock throttle reason bitmask at this instant.
};

/// @brief What the monitor saw across one start() to stop() window.
struct NvmlSummary {
    int samples = 0;                          ///< Number of readings taken.
    double median_sm_clock_mhz = 0.0;         ///< Median SM clock; the roofline denominator.
    unsigned int max_temperature_c = 0;       ///< Hottest reading in the window.
    double max_power_w = 0.0;                 ///< Highest board power in watts.
    double mean_gpu_util_pct = 0.0;           ///< Mean utilization across the readings.
    bool throttled = false;                   ///< Any thermal, power, or reliability throttle seen.
    unsigned long long throttle_reasons = 0;  ///< Union of reasons across samples.
    bool available = false;                   ///< False if NVML could not be initialized.
    std::string note;                         ///< Human readable status or error.
};

/**
 * @brief Samples NVML on a background thread between start() and stop().
 *
 * Non copyable; one monitor drives one window at a time.
 */
class CKL_EXPORT NvmlMonitor {
public:
    /**
     * @brief Prepares a monitor; NVML itself is initialized on start().
     * @param sample_interval_ms Milliseconds between readings.
     * @param device_index NVML device ordinal to watch.
     */
    explicit NvmlMonitor(unsigned int sample_interval_ms = 25, int device_index = 0);

    /// @brief Stops the sampling thread if it is still running.
    ~NvmlMonitor();

    NvmlMonitor(const NvmlMonitor&) = delete;
    NvmlMonitor& operator=(const NvmlMonitor&) = delete;

    /**
     * @brief Move constructor, so a monitor can be handed to whoever owns the timed window.
     * @param other Monitor to move from; it holds nothing afterwards and calling
     *        into it throws.
     */
    NvmlMonitor(NvmlMonitor&& other) noexcept;

    /**
     * @brief Move assignment.
     * @param other Monitor to move from; it holds nothing afterwards.
     * @return This monitor.
     */
    NvmlMonitor& operator=(NvmlMonitor&& other) noexcept;

    /**
     * @brief Starts the sampling thread and the clock the sample times are measured from.
     * @note If NVML is unavailable the monitor records that instead of throwing;
     *       the summary comes back with available false and a note saying why.
     */
    void start();

    /**
     * @brief Stops the sampling thread and reduces the samples to a summary.
     * @return The summary of the window just closed.
     */
    NvmlSummary stop();

    /**
     * @brief Every reading taken in the last window.
     * @return A reference to the sample vector, valid until the next start().
     */
    const std::vector<NvmlSample>& samples() const;

private:
    struct Impl;
    std::unique_ptr<Impl> impl_;
};

}  // namespace ckl
