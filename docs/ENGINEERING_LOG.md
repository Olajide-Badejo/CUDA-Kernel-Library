# Engineering log

Dated entries, one per real problem, each with symptom, root cause, options,
chosen fix and why, and how it was verified. At Phase 10 this becomes the debug
report. Failed diagnostic rounds and reverted changes belong here too; they are
an equal rank deliverable, not padding.

## 2026-07-19: CMake ignored CMAKE_CUDA_ARCHITECTURES, built sm_75

Symptom: the first `device_probe` build compiled with
`arch=compute_75,code=[compute_75,sm_75]` even though the top CMakeLists set
`CMAKE_CUDA_ARCHITECTURES 120`, and the code failed to build against the 5070.

Root cause: I set the architecture after `project(... LANGUAGES CUDA)` inside an
`if(NOT DEFINED CMAKE_CUDA_ARCHITECTURES)` guard. `project()` already seeds a
default value for that variable, so the guard was false and my set was skipped;
`enable_language(CUDA)` had already captured the default.

Options: (a) set the target property per target; (b) set the variable
unconditionally after project; (c) set the variable before `project()` so
`enable_language` reads it. Chose (c) as the single global source of truth.

Fix: moved the architecture set above the `project()` call. Verified: the rebuild
compiled `sm_120` and `device_probe` ran on the GPU.

## 2026-07-19: cudaDeviceProp lost clockRate and memoryClockRate in CUDA 13

Symptom: `device_probe.cu` failed to compile with "class cudaDeviceProp has no
member memoryClockRate" and the same for clockRate.

Root cause: both fields were deprecated in CUDA 12.x and removed from the struct
in CUDA 13. The build floor here is CUDA 13.3, so the struct members are gone.

Fix: query the same values through `cudaDeviceGetAttribute` with
`cudaDevAttrMemoryClockRate` and `cudaDevAttrClockRate`, which still return kHz.
Verified: probe prints 14001 MHz memory clock and 2625 MHz core clock, matching
the datasheet, and the theoretical bandwidth lands on 672.0 GB/s.

## 2026-07-19: -Wpedantic rejected nvcc generated stub files

Symptom: with `-Xcompiler=-Wpedantic` and `--Werror=all-warnings`, the build
failed on "style of line directive is a GCC extension" pointing at
`tmpxft_*.cudafe1.stub.c`, a file nvcc generates, not one I wrote.

Root cause: separable compilation (`-rdc=true`) makes nvcc emit stub
translation units that use GCC style line directives. `-Wpedantic` forwarded to
the host compiler flags those directives, and `-Werror` makes it fatal.

Fix: keep `-Wpedantic` on hand written host C++ only
(`$<COMPILE_LANGUAGE:CXX>`), and forward just `-Wall -Wextra` to the nvcc host
pass. Device code still gets `--Werror=all-warnings`. Verified: clean build with
zero warnings; the gate still catches real warnings in our own device code.

## 2026-07-19: ncu ERR_NVGPUCTRPERM under WSL2, root does not help

Symptom: `ncu` on a trivial kernel returns `ERR_NVGPUCTRPERM` ("user does not
have permission to access NVIDIA GPU Performance Counters"). Running ncu under
`sudo` (passwordless sudo confirmed working) returns the same error.

Root cause: under WSL2 the CUDA driver is the Windows NVIDIA driver reached
through the GPU paravirtualization layer. GPU performance counter access is
gated by the Windows driver policy, not by Linux user or root. So no Linux side
permission change or `sudo` clears it; the fix has to happen on Windows.

Options: (a) NVIDIA Control Panel then Manage GPU Performance Counters then allow
all users; (b) set registry DWORD `RmProfilingAdminOnly = 0` under
`HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak` and reboot. Both
need Windows administrator; the registry path needs a reboot.

Resolved 2026-07-19. The repo owner set `RmProfilingAdminOnly = 0` under the
nvlddmkm NVTweak key and rebooted. Verification: the ncu probe on `stream_copy`
now returns no `ERR_NVGPUCTRPERM` and a populated table (DRAM throughput 86
percent, SM throughput 1.76 percent, the expected memory bound signature of a
copy kernel). Round 1 (tiled versus naive) ran cleanly afterward; see
`docs/DIAGNOSTIC_LOG.md`.

## 2026-07-19: three stage cp.async pipeline made the top tensor kernel slower, reverted

Symptom: hypothesizing that the top tensor kernel's residual latency exposure
(warp cycles per issued 40.6 at 32.6 percent occupancy) came from too shallow a
software pipeline, I deepened the cp.async pipeline from two stages to three.
Correctness held, but performance dropped: 4096 cubed fell from 79.8 to 74.5
percent of cuBLAS, 8192 from 79.4 to 73.9 percent.

Root cause: the third shared buffer raised shared memory use from 32 KB to 48 KB
per block, which cut how many blocks fit per SM and lowered occupancy further. On
this kernel the occupancy loss outweighed the extra latency hiding, because the
two stage pipeline was already covering most of the global load latency (DRAM only
13 percent at 4096). The bottleneck is the shared read pipe and register pressure,
not global load latency, so adding pipeline depth spent shared memory on the wrong
problem.

Fix: reverted to the two stage kernel (`git checkout` of `gemm_mma_opt.cu`),
confirmed back at 79.7 percent with correctness intact. Recorded as a failed round
(Round 8 hypothesis) rather than kept. The genuine remaining levers are the ones
named in DIAGNOSTIC_LOG Round 7: a swizzled shared layout, more register reuse, or
split K for the large shapes, not a deeper pipeline.
