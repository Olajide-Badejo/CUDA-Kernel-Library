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

Status: kernel built and verified correct; wall clock measured; the ncu
diagnostic round is staged and waits on the counter permission fix from Phase 0.

### Built

- `src/gemm/gemm_tiled.cu`: 32 by 32 shared memory tiled kernel, one column of
  shared padding against bank conflicts, boundary guarded so non tile aligned
  and sub tile shapes stay correct. Added to the test and benchmark variant
  lists, which now iterate every hand written variant against cuBLAS.

### Verified output

Correctness: tiled passes the 1e-4 gate on all 8 shapes.

Wall clock (this machine), an honest and initially surprising result:

| size | naive GFLOP/s | tiled GFLOP/s | cuBLAS GFLOP/s | tiled percent of cuBLAS |
|---|---|---|---|---|
| 512 | 1943 | 1597 | 7170 | 22.3 |
| 1024 | 2137 | 1814 | 18431 | 9.8 |
| 2048 | 2046 | 1751 | 23167 | 7.6 |
| 4096 | 1910 | 1588 | 22099 | 7.2 |

Reading (hypothesis, to be confirmed by the ncu round): tiling alone does not
beat the naive kernel on this GPU. Two reasons stand out before profiling. First,
occupancy: the 32 by 32 tile is a 1024 thread block, and with 1536 threads per SM
only one block fits (about 66 percent occupancy), while the naive 16 by 16 block
reaches 100 percent, and both kernels are latency bound so occupancy dominates.
Second, the 48 MB L2 already services the naive kernel's repeated B column reads,
so the shared memory staging trades cache hits for shared memory latency plus two
syncthreads per K tile without raising arithmetic intensity per thread, which
stays at one output element per thread. This matches the design rationale that
register blocking, not tiling, is the decisive step (Volkov and Demmel, SC 2008),
and it is precisely the picture the diagnostic round is meant to make concrete.
Phase 3 (register blocking, many outputs per thread) is where the ladder is
expected to pull clearly ahead of naive.

### Diagnostic round 1 (done, counters enabled)

Counter permission cleared (owner set `RmProfilingAdminOnly = 0` and rebooted;
verified). Round 1 ran on naive and tiled at 2048 and 4096; artifacts in
`experiments/results/ncu/round01/`. Result confirms the hypothesis: the tiled
kernel is limited by the shared memory (MIO) pipe and by 66.6 percent occupancy
(1024 thread block, one block per SM), while the naive kernel rides an 87.5
percent L1 hit rate at 99.8 percent occupancy and was never DRAM bound (37
percent DRAM throughput). Full metric table and reasoning in
`docs/DIAGNOSTIC_LOG.md` Round 1. Next single change: register blocking (Phase 3).

## Phase 3: register blocked, vectorized GEMM, diagnostic round 2

Status: complete and verified.

### Built

- `src/gemm/gemm_register.cu`: 128 by 128 block, 8 by 8 register tile per thread
  (256 threads), float4 staging loads, A transposed into shared memory. Aligned
  fast path (m by 128, n by 128, k by 8); falls back to the tiled kernel for odd,
  sub tile, and zero k shapes so correctness holds everywhere.

### Verified output

Correctness: register passes the 1e-4 gate on all 8 shapes (aligned shapes use
the fast kernel; several correctness shapes, including 256 cubed and 384 by 512
by 128 and 512 by 384 by 640, are aligned and exercise it directly).

Benchmark (this machine):

| size | register GFLOP/s | register percent of cuBLAS | naive GFLOP/s |
|---|---|---|---|
| 512 | 2945 | 60.3 | 1960 |
| 1024 | 7041 | 38.1 | 2136 |
| 2048 | 9763 | 41.5 | 2045 |
| 4096 | 10381 | 47.4 | 1890 |

At 4096 the register kernel is 5.5 times naive and 6.5 times the tiled kernel, and
jumps from about 8 percent to 47 percent of cuBLAS. Round 2 (see DIAGNOSTIC_LOG)
names the new limiter: the L1 / TEX shared read pipe at 88.8 percent plus a 32.6
percent occupancy ceiling from 106 registers per thread. DRAM is only 18.5
percent, so it is not bandwidth bound. Next single change: cp.async double
buffering (Phase 4).

## Phase 4: cp.async double buffered GEMM, diagnostic round 3

Status: complete and verified.

### Built

- `src/gemm/gemm_cp_async.cu`: double buffered register blocked kernel using
  cp.async (`__pipeline_memcpy_async`) to overlap the next tile's global to
  shared copy with the current tile's math. A is stored non transposed so
  cp.async can copy contiguous bytes; its shared reads broadcast across the warp
  so that costs no bank conflicts. Same alignment contract and tiled fallback.

### Verified output

Correctness: cp_async passes the 1e-4 gate on all 8 shapes.

Benchmark (this machine):

| size | cp_async GFLOP/s | percent of cuBLAS | register GFLOP/s |
|---|---|---|---|
| 1024 | 10895 | 63.2 | 7010 |
| 2048 | 15081 | 65.9 | 9721 |
| 4096 | 16174 | 73.4 | 10380 |

Round 3 (see DIAGNOSTIC_LOG): cp.async overlapped the loads (warp cycles per
issued instruction 9.3 to 3.49, DRAM 18.5 to 29.0 percent, no pipe saturated).
The terminal FP32 limiter is occupancy: 149 registers per thread caps the kernel
at one block per SM and 16.7 percent occupancy. Accepted as the top hand written
FP32 variant at 73 percent of cuBLAS, gap diagnosed. Next: tensor cores (Phase 5)
to lift the compute ceiling toward the compute bound gate.

## Phase 5 to 11

Status: pending. Phase 5 (WMMA FP16 and BF16, mma.sync PTX, CUTLASS reference,
rounds to the compute bound gate) is next.
