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
`sudo` returns the same error.

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

## 2026-08-31: nvcc 13.3 cannot parse GCC 15.2's libstdc++, host compiler pinned to g++-14

Symptom: a default configure on this machine fails during the first CUDA
compilation, inside a libstdc++ header, with a parse error on `if consteval`. The
error names a system header and no file of mine, so it reads like a broken
toolkit rather than a host compiler mismatch.

Root cause: the machine's default host compiler is GCC 15.2. Its libstdc++ uses
`if consteval` in headers that nvcc's frontend pulls in, and the nvcc 13.3
frontend does not accept that form. nvcc runs its own C++ frontend over host code
before handing anything to the host compiler, so "the host compiler supports it"
is not the relevant question.

Options: (a) wait for a toolkit whose frontend accepts it; (b) patch or shadow the
offending header; (c) pin the host compiler to a version whose libstdc++ nvcc 13.3
can parse. Chose (c): (a) blocks the build on somebody else's release schedule and
(b) means shipping a project that edits the user's system headers.

Fix: the build pins `CMAKE_CXX_COMPILER` and `CMAKE_CUDA_HOST_COMPILER` to
g++-14.3, and the Makefile carries a `CMAKE_EXTRA` hook so a checkout on a machine
with a different default can pass it in one variable. `docs/building.md` states
the pin and the reason in its host compiler section, because a user who hits this
sees a libstdc++ error and has no way to guess the cause. Verified: configure and
full build clean under g++-14.3 on this machine, and the CI containers, which ship
GCC 13, need no flag at all.

## 2026-08-31: audited my own published claims against my own data, A1 to A7

Symptom: not a build failure. Preparing the 1.1.0 release I read the shipped
report and README back against the results files they were supposed to have come
from, and they did not agree.

Root cause: seven distinct defects, each with its own mechanism, recorded in full
in `docs/CORRECTIONS.md` and summarized here so the log carries the incident.
**A1**, provenance: all 60 summary rows and all eight `round_meta.txt` files carry
commit hashes that no longer resolve, because I rewrote the repository history
after the data was produced. The numbers are real; their audit trail is not.
**A2**, timed regions: the cuSPARSE wrapper built descriptors and sized and
allocated its workspace inside every call and the harness timed all of it, and the
TRSM benchmark restored its right hand side inside the timed window. **A3**, the
roofline: the compute roof was set to a cuBLAS measurement, which made every
"percent of roof" a restatement of "percent of cuBLAS" drawn on log axes. **A4**,
clocks: the sweep never locked them, so rows taken at 1042 MHz sit beside rows
taken at 2880 MHz. **A5**, claims the data does not support, including a flat
percent quoted from the best shape. **A6**, two of the nine diagnostic rounds
point at no committed ncu page. **A7**, the pipeline could not reproduce itself:
the roofline CSV was gitignored, the asset generator skipped the figure and exited
zero, and a clean clone rebuilt the report around a stale committed PNG and
reported success.

Fix: one gate per defect rather than one edit per defect, because an edit fixes
the instance and a gate fixes the class. `scripts/check_provenance.py` resolves
every recorded hash and fails on a dead one, with the known dead v1 hashes listed
as a standing debt. The SpMV plan makes per call descriptor construction
impossible through the public path, and an allocation gate armed on CUPTI
callbacks fails any run that allocates inside a timed region. The roofline draws
hardware roofs and puts cuBLAS on a dashed attainable line. The sweep refuses to
run without a locked clock. Unsupported claims are withdrawn in the register
rather than quietly reworded. And `gen_report_assets.py` treats a missing input as
a hard error, so the report fails loudly instead of reusing a picture whose input
is gone.

Verified: `make check-style` runs the provenance gate and the dash gate on every
invocation; `make report` regenerates every figure and table from the committed
results and CI diffs them byte for byte; the allocation gate was watched failing
on purpose with `--gate-red` before it was trusted; and the corrections register
ships in the published tree rather than being deleted with the defects.
