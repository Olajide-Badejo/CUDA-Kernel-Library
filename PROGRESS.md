# PROGRESS

One entry per phase: what got built, what was verified with real output, what is
still open. Kept current from the first commit per Section 0 of the spec.

## Phase 0: toolchain, GPU probe, dash lint, skeleton

Status: in progress. Repo skeleton, build system, dash checker, and device probe
are built and verified on the target machine. One blocker remains (ncu counter
permission) that needs a Windows-side action before the Phase 2/3 diagnostic
rounds; it does not block Phases 1 and 2 kernel work.

### Verified on the machine

- GPU probe built and run. Measured ground truth used everywhere the roofline
  needs a ceiling, replacing the datasheet:
  - RTX 5070, compute capability 12.0 (sm_120), 48 SMs, 12227 MiB.
  - Memory: 192 bit bus, 14001 MHz, theoretical 672.0 GB/s (matches datasheet).
  - Measured stream copy bandwidth: about 540 GB/s (roughly 80 percent of
    theoretical, typical for a plain streaming copy).
  - L2 cache 48 MiB, shared memory 100 KiB per SM (48 KiB per block default),
    65536 registers per SM, 1536 max threads per SM.
  - Peak FP32 estimate about 32.3 TFLOP/s (non tensor core).
- CMake configure and Ninja build clean, targeting sm_120.
- Dash checker (`scripts/check_no_dashes.py`) runs clean over the tree.

### Toolchain, with substitutions against the spec floors

| Component | Spec floor | Installed | Action |
|---|---|---|---|
| CUDA Toolkit | 13.3 | 13.3.73 | matches |
| GPU driver (CUDA UMD) | current | 610.62 / CUDA 13.3 | matches |
| GCC / G++ | 16.1 | 15.2.0 | 16.1 not yet released; 15.2 is newest available, recorded here |
| CMake | 4.4 | 4.2.3 | 4.2.3 configures and builds cleanly; upgrade only if a 4.4 feature is needed |
| CUDA C++ device standard | newest accepted, target 23 | C++20 | nvcc 13.3 accepts C++20 for device code; host code stays C++23. Recorded per Section 3 rather than downgrading host silently |
| Ubuntu (WSL2) | Ubuntu | 26.04 LTS | matches intent |
| Python | (unspecified) | 3.14.4 | fine |
| ncu / nsys | on PATH | present in toolkit | present |
| cuBLAS / cuSPARSE / cuSOLVER | present | 13.6 / 12.8 / 12.2 | present |
| NVML | present | header plus runtime `libnvidia-ml.so.1` plus stub | present |

Still to install when their phase arrives (not blocking kernel work):
clang-format, clang-tidy (style gate, Phase 9), ruff (Python lint), and
texlive-latex-extra plus latexmk (reports, Phase 10).

### Open blocker: ncu GPU performance counter permission (WSL2)

`ncu` returns `ERR_NVGPUCTRPERM` for both the normal user and root. Under WSL2 the
counter permission is enforced by the Windows NVIDIA driver, so a Linux side or
`sudo` fix does not apply. The diagnostic rounds (Phases 2 to 5) are the core of
the project and depend on this, so it must be cleared before Phase 2's round.

Fix (needs Windows administrator and a reboot or driver restart); either path:

- GUI: NVIDIA Control Panel then Desktop or Developer menu then Manage GPU
  Performance Counters then "Allow access to the GPU performance counters to all
  users", apply.
- Registry (elevated): set DWORD `RmProfilingAdminOnly = 0` under
  `HKLM\SYSTEM\CurrentControlSet\Services\nvlddmkm\Global\NVTweak`, then reboot.
  The key currently has no such value.

Verification after the fix: rerun the ncu probe on `stream_copy` and confirm no
`ERR_NVGPUCTRPERM` and a populated metric table. See ENGINEERING_LOG entry 2026-07-19.

### Decisions taken

- Repo lives on the Windows filesystem (the working directory) for IDE
  visibility; builds run in WSL2 against the `/mnt/c` path through a
  `~/ckl` symlink. Revisit only if the CUTLASS fetch in Phase 5 makes `/mnt/c`
  I/O too slow, in which case relocate to a WSL native path.
- `--Werror=all-warnings` on device code; `-Wpedantic` applied to hand written
  host C++ only, not the nvcc host passthrough, because nvcc's generated
  `.stub.c` files under separable compilation use line directives that
  `-Wpedantic` rejects and those files are not ours.

## Phase 1: naive GEMM, harness, cuBLAS oracle

Status: complete and verified on the machine.

### Built

- `include/ckl/gemm.hpp`: shared row major GEMM interface for the whole ladder.
- `src/gemm/gemm_naive.cu`: honest baseline, one thread per output element,
  natural indexing, no reuse.
- `src/gemm/cublas_gemm.cpp`: cuBLAS SGEMM oracle producing the identical row
  major C through the column major transpose identity; caches one handle;
  handles the empty contraction (k == 0) by scaling rather than a zero leading
  dimension SGEMM.
- `include/ckl/event_timer.hpp`: CUDA event timing, 5 warmups then 20 reps,
  median and IQR.
- `include/ckl/reference.hpp`: seeded random fill, double precision CPU GEMM,
  relative Frobenius error.
- `tests/test_gemm.cpp`: correctness gate over 8 shapes.
- `benchmarks/bench_gemm.cpp`: the early harness (naive vs cuBLAS, GFLOP/s,
  percent of cuBLAS).

### Verified output

Correctness (relative Frobenius, tolerance 1e-4), all PASS including the
degenerate shapes; cuBLAS versus the CPU double reference near 1e-7 confirms the
oracle orientation:

| shape | naive vs cuBLAS | cuBLAS vs CPU |
|---|---|---|
| 256^3 square | 3.0e-7 | 1.5e-7 |
| 384 x 512 x 128 | 4.4e-8 | 2.1e-7 |
| 129 x 257 x 193 (non tile aligned) | 2.6e-7 | 1.5e-7 |
| 7 x 5 x 11 (sub tile) | 9.2e-8 | 6.7e-8 |
| 512 x 384 x 640 | 4.6e-7 | 2.7e-7 |
| zero m, zero n, zero k | 0 | 0 |

Baseline benchmark (this machine, commit recorded in the run; naive is the
honest first pass, not a strawman):

| size | naive GFLOP/s | cuBLAS GFLOP/s | naive percent of cuBLAS |
|---|---|---|---|
| 256 | 1245 | 1419 | 87.8 |
| 512 | 1930 | 5604 | 34.4 |
| 1024 | 2132 | 17465 | 12.2 |
| 2048 | 2045 | 23606 | 8.7 |
| 4096 | 1909 | 22005 | 8.7 |

Reading: the naive kernel plateaus near 2 TFLOP/s because every A and B element
is refetched from global memory with no reuse, so it is bandwidth and latency
bound the moment the matrices leave the caches. cuBLAS reaches about 22 to 24
TFLOP/s true FP32 (below the 32.3 TFLOP/s estimated peak). This gap is what the
ladder closes, one attributable technique at a time.

## Phase 2: shared memory tiled GEMM, first diagnostic round

Status: pending (kernel can be built now; the ncu round waits on the counter
permission fix noted in Phase 0).
