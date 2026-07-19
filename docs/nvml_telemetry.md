# NVML telemetry

`ckl::NvmlMonitor` (`src/telemetry/nvml_monitor.cpp`) samples the GPU on a
background thread between `start()` and `stop()`. It reads SM clock, temperature,
power, utilization, and the current clocks event (throttle) reasons at a fixed
interval (25 ms by default), and reduces the window to a summary: median SM clock,
peak temperature and power, mean utilization, and whether any real throttle
(software power cap, hardware or software thermal, power brake) fired.

## Why it gates results

The build spec's rule is that a sweep whose samples show throttling is rerun after
cooldown rather than reported. The monitor makes that checkable: every sweep runs
with it on, and the throttle flag rides along in the results row. A number taken
while the card was thermally or power throttled is not a number about the kernel,
it is a number about the cooling, so it does not count.

## Notes

- NVML init failure is not fatal. On a host or container without NVML the summary
  comes back with `available = false` and a note, so the sweep still runs (without
  throttle checking). Under WSL2 the runtime library ships with the Windows driver
  under `/usr/lib/wsl/lib`; the header comes with the toolkit.
- The CUDA 13 API is used (`nvmlDeviceGetTemperatureV`, the versioned temperature
  struct, and `nvmlDeviceGetCurrentClocksEventReasons`), so the build is free of the
  deprecation warnings the older calls now raise.

A short self check (`tests/test_nvml.cpp`) runs a GEMM for half a second with the
monitor on and prints the summary; on this machine a typical window reports median
SM clock, max temperature about 42 C, max power about 124 W, and no throttle.
