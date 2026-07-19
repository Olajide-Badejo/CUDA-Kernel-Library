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

## Phase 5: tensor core GEMM (in progress)

Status: WMMA FP16 and BF16 built, correct, benchmarked, and profiled (Round 4).
mma.sync PTX and CUTLASS reference still to come, then rounds to the compute
bound gate.

### Built so far

- `include/ckl/detail/wmma_gemm.cuh`: templated WMMA kernel (64 by 64 block, K
  step 16, four warps, FP32 accumulate) plus a scalar half input fallback for
  non aligned shapes; instantiated by `gemm_wmma_fp16.cu` and `gemm_wmma_bf16.cu`.
- `src/gemm/cublas_gemm_ex.cpp`: cublasGemmEx tensor oracles (FP16 and BF16 in,
  FP32 accumulate) producing the same row major C.
- `tests/test_gemm_tensor.cpp`, `benchmarks/bench_gemm_tensor.cpp`.

### Verified output

Correctness: all shapes pass. Kernel versus cuBLAS tensor oracle about 1e-7;
kernel versus FP32 CPU reference about 2.6e-4 (FP16) and 2.1e-3 (BF16). Tolerances
recorded in `docs/gemm.md`.

Benchmark (this machine), WMMA versus the cuBLAS tensor oracle per precision:

| size | fp16 wmma GFLOP/s | fp16 percent of cuBLAS | bf16 wmma GFLOP/s | bf16 percent of cuBLAS |
|---|---|---|---|---|
| 2048 | 35815 | 56.1 | 33061 | 51.6 |
| 4096 | 36308 | 59.7 | 33582 | 53.6 |
| 8192 | 34149 | 52.4 | 31891 | 48.4 |

WMMA at 4096 (36.3 TFLOP/s FP16) is 2.2 times the top FP32 kernel (16.2 TFLOP/s),
confirming tensor cores lift the ceiling. Round 4 (see DIAGNOSTIC_LOG): the kernel
is shared read (L1 / TEX) pipe bound at 92 percent with the tensor pipe only 53.7
percent utilized, so the cores are starved by the generic load_matrix_sync
fragment loads.

- `src/gemm/gemm_mma_ptx.cu`: PTX level mma.sync.aligned.m16n8k16 (FP16, FP32
  accumulate) with the lane to fragment mapping done by hand. Correct on all
  shapes (matches the oracle to about 1e-7). Round 5: it matches the WMMA kernel
  within noise (about 36.4 TFLOP/s), which is the informative result. It isolates
  the instruction path from the load path and confirms that mma.sync was never
  the bottleneck; the scalar fragment loads are. The single change that would move
  the number is ldmatrix (plus a larger tile and cp.async double buffering).

- `src/gemm/gemm_mma_ldmatrix.cu`: ldmatrix at the 64 by 64 tile. Round 6: no
  change (still L1 / TEX bound at 95 percent), because the small tile's bytes to
  MAC ratio is the real limiter, not the load instruction.
- `src/gemm/gemm_mma_opt.cu`: the top tensor kernel. 128 by 128 tile, K step 32,
  ldmatrix, cp.async double buffering, and a shared memory swizzle. Correct on all
  shapes. Optimization loop across Rounds 7 to 9:
  - Round 7 (128 by 128 plus double buffering): 48.6 TFLOP/s at 4096 (79.8 percent
    of cuBLAS), Speed of Light co limited (compute 72.5, memory 75.6).
  - Round 8 (three stage pipeline): slower, reverted (occupancy loss). Failed round
    recorded in ENGINEERING_LOG.
  - Round 9 (shared swizzle): a bank conflict counter showed 219 million shared
    load conflicts driving the shared pipe; the swizzle cut them to 33.7 million,
    collapsing memory Speed of Light from 75.6 to 30.2 percent and lifting compute
    to 82.7 percent.

  Result: 54.9 TFLOP/s at 4096 (90.3 percent of cuBLAS) and 58.4 TFLOP/s at 8192
  (90.1 percent), 3.4 times the top FP32 kernel.

  COMPUTE BOUND GATE: PASSED. Compute utilization (82.7 percent) is clearly above
  memory utilization (30.2 percent), the Tensor pipe is the highest utilized
  pipeline, and the operating point is right of the roofline ridge. Section 8.1
  stop condition met: gate passes AND at least 90 percent of cuBLAS at both 4096
  and 8192 for FP16. Nine rounds, one change each, every step backed by an ncu
  artifact. Split K was considered and correctly not applied (these shapes already
  saturate the SMs; see DIAGNOSTIC_LOG Round 9).

### Ladder summary (this machine, 4096 cubed, versus same precision cuBLAS)

| variant | precision | GFLOP/s | percent of cuBLAS |
|---|---|---|---|
| naive | FP32 | 1905 | 8.6 |
| tiled | FP32 | 1589 | 7.2 |
| register | FP32 | 10380 | 47.1 |
| cp_async | FP32 | 16174 | 73.4 |
| wmma_fp16 | FP16 | 36308 | 59.7 |
| mma_ptx | FP16 | 36360 | 59.8 |
| mma_ldm | FP16 | 36416 | 60.2 |
| mma_opt (top) | FP16 | 54931 | 90.3 |

Still to do in Phase 5: the CUTLASS reference instantiation (open source upper
reference), and the roofline based gate figure (Phase 8).

## Phase 6: GEMV family

Status: complete and verified.

### Built

- `src/gemv/gemv_naive.cu` (thread per row), `gemv_warp.cu` (warp per row with
  shfl reduction), `gemv_vectorized.cu` (float4 warp per row), `cublas_gemv.cpp`
  (SGEMV oracle). Tests and benchmark added.

### Verified output

Correctness (1e-4 versus cuBLAS) passes on square, tall, wide, non multiple of 4,
single row, and zero n shapes. Effective bandwidth for the read of A:

| size | naive GB/s | warp GB/s | vectorized GB/s | cuBLAS GB/s |
|---|---|---|---|---|
| 4096 | 250 | 590 | 577 | 587 |
| 8192 | 325 | 616 | 610 | 613 |

The warp and vectorized kernels match cuBLAS and reach about 610 GB/s at 8192,
near the measured streaming ceiling; the naive kernel is coalescing limited.
Documented as memory bound with roofline evidence in `docs/gemv.md` (arithmetic
intensity 0.5 FLOP per byte, far left of the measured ridge of about 60).

## Phase 7: TRSM and CSR SpMV families

Status: complete and verified.

### Built

- SpMV: `src/sparse/spmv_csr_naive.cu` (thread per row), `spmv_csr_warp.cu` (warp
  per row, shfl reduction), `cusparse_ref.cpp` (cusparseSpMV oracle). Tested on a
  skewed degree matrix; benchmark included.
- TRSM: `src/trsm/trsm_naive.cu` (forward substitution), `trsm_blocked.cu` (32
  wide diagonal blocks with a GEMM style trailing update), `cublas_trsm.cpp`
  (STRSM oracle, row major left solve mapped to a right side upper solve).

### Verified output

- SpMV correctness (skewed matrix, max row degree about 700): naive and warp both
  match cuSPARSE to about 1e-7. Benchmark at 131072 squared, 3.9 M nonzeros: naive
  60.1 GFLOP/s, warp 94.8 (131.8 percent of cuSPARSE), cuSPARSE 71.9. The warp
  kernel is 1.6 times naive on the skewed load, the intended result. See
  `docs/sparse.md`.
- TRSM correctness (lower, diagonally dominant): naive and blocked match cuBLAS to
  about 1e-7 with residuals near 1e-7, across square, tall, and non block aligned
  shapes. See `docs/trsm.md`.

## Phases 8 to 11

Status: pending.
