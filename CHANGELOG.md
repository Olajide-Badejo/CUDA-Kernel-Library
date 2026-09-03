# Changelog

Notable changes per phase. Dates are ISO. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions are tagged
at the definition of done milestones.

## [1.1.0] - 2026-09-03

The integrity release, and the release that turns the ladder into a library.
Phase A below was written first and blocks everything else; every measured claim
this release adds is marked pending until the locked clock sweep replaces the v1
numbers.

### Added

- Packaging. `find_package(CKL)` plus `target_link_libraries(app PRIVATE
  ckl::ckl)` works from an install prefix, verified by the standalone
  `tests/consume` project; a config package with dependency resolution, a
  generated version header, a pkg-config file written at install time,
  per-component `ckl::` aliases, `BUILD_SHARED_LIBS` support with hidden
  visibility and `CKL_EXPORT` annotations, and CMake presets (default, release,
  multiarch, asan). Separable compilation is gone: nothing device links, and RDC
  blocked non CMake consumers outright.
- The public API. `ckl::Context` (lazy vendor handles, cached device properties,
  stream and workspace), `ckl::Status`, descriptor driven `ckl::gemm` with
  `gemm_query`, `gemm_plan`, and `gemm_workspace_size`, and family entry points
  `ckl::spmv`, `ckl::fft`, `ckl::conv`, `ckl::reduce`, `ckl::scan`. Every
  dispatcher writes a `chosen` out parameter on success and failure, an explicit
  algorithm is never rerouted, and a shape a path cannot take returns
  `kNotSupported` instead of silently falling back.
- A pure C99 ABI in `include/ckl/ckl.h`: 21 entry points, no CUDA header, device
  pointers and streams as `void*`, a thread local last error, device memory
  helpers so a toolkit free consumer can manage residency, and a catch all on
  every entry so nothing but C crosses the boundary.
- A Fortran binding. `fortran/ckl_mod.f90` binds the whole ABI through
  `ISO_C_BINDING`; `SGEMM`, `SGEMV`, and `STRSM` ship under the standard mangled
  names in a separate `ckl_blas` archive behind `CKL_BLAS_ALIASES`, taking host
  arrays and managing transfers; a conjugate gradient example validates against
  LAPACK and passes the residual gate.
- GEMM performance mechanisms: a corrected A tile swizzle (the old mask could
  not separate the lanes of an ldmatrix.x4 wavefront), a three stage single
  barrier cp.async pipeline on dynamic shared memory, a tile shape family of six
  instantiations templated on block and warp geometry, predicated tails so any
  shape runs the real mainloop with masked edges instead of a 25x slower scalar
  path, two pass split-K with a real workspace answer, stream-K persistent CTAs
  whose owners wait only downward in block index, and a waves based dispatch
  heuristic that reads the committed tile sweep when one exists and holds the
  reference tile when none does.
- The CUTLASS reference line, `cutlass::gemm::device::GemmUniversal` at a pinned
  v4.7.1, the honest FP16 instantiation on sm_120 since the SM120 collective
  builders accept only blockscaled narrow precision kinds.
- Three new kernel families, each with its own algo enum, plan object, C ABI
  entry, tests, and vendor baseline measured through a cached plan: SpMV
  (vector CSR, merge path after Merrill and Garland, SELL-C-sigma, BSR) over a
  seven matrix SuiteSparse suite fetched by checksum plus the v1 generator;
  FFT (Stockham radix 2 and 4, shared resident, four step, R2C, 2D with a
  transpose study) with FFT and direct convolution and a fused pointwise
  epilogue, against a cuFFT baseline whose callback path was actually built and
  validated; reduction and scan ladders up to single pass decoupled look-back,
  with a bit deterministic mode, a Kahan rung, and CUB as the named baseline.
- Instruction level evidence: normalized SASS committed for nineteen rungs, a
  diff gate shown red on a one line kernel perturbation, and a register
  allocation study that had to edit PTX directives because `-maxrregcount` is
  ignored under `__launch_bounds__`; shared memory, not registers, pins the top
  kernel at 2 blocks per SM.
- Testing: 844 GoogleTest cases replace seven hand rolled mains, with shape
  derived tolerances, alpha and beta parameterization, NaN poisoned beta zero
  inputs, dispatch honesty assertions, canary guarded zero dimension cases,
  calibrated tolerance constants, and all four compute sanitizer tools clean on
  the GEMM suites.
- Measurement: benchmarks verify their output against the oracle before timing
  anything, keep every sample, assert the dispatched path, record the cuBLAS
  math mode from the handle that ran, flush L2 where the working set is cache
  resident, isolate launch overhead with CUDA graphs at small shapes, and run
  five independent processes per configuration with bootstrap confidence
  intervals; vendor baselines are measured once and joined. The verification
  gate caught the first `CUBLAS_GEMM_AUTOTUNE` call per shape leaving garbage
  in C.
- CI: eight jobs with no `continue-on-error`, pinned containers by digest,
  actions by SHA, tools by version; a build matrix over CUDA 12.8 and 13.3
  across five architectures with warnings as errors; install and consume, SASS
  diff, byte stable report regeneration, Doxygen to Pages, and self hosted GPU
  and nightly perf regression jobs that stay queued or skipped, never falsely
  green, until a runner exists.
- Documentation: `docs/using.md` with the load bearing contracts,
  `docs/building.md` with the toolchain story (g++-14 is required; GCC 15's
  libstdc++ breaks the nvcc 13.3 frontend), four runnable examples that build in
  tree and against an installed prefix, Doxygen over every public declaration
  with zero warnings, per family mechanism docs, `CITATION.cff`, a code of
  conduct, `SECURITY.md`, `NOTICE`, and issue and pull request templates.
- The report: related work over a twenty record bibliography that prints every
  record, limitations and threats to validity, a reproducibility statement, a
  numerical accuracy model for the headline kernel, chapters for every family,
  and the shared memory tiling loss to naive promoted from a buried paragraph
  to a titled section, reported as a single GPU finding.

### Fixed

- Provenance (A1). `scripts/check_provenance.py` resolves every commit hash
  recorded in `summary.csv`, `sweep.jsonl`, and each `round_meta.txt` with
  `git cat-file -t`, and fails when one does not resolve. It runs as part of
  `make check-style`. None of the v1 hashes resolve, because I rewrote history
  after the data was produced; they are listed in
  `experiments/results/legacy_hashes.txt` as a known debt and that file goes away
  when the sweep and the ncu rounds are re-run.
- Contaminated timed regions (A2). `src/sparse/cusparse_ref.cpp` caches the
  cuSPARSE descriptors and the SpMV workspace in a plan, so a timed call runs only
  `cusparseSetStream` and `cusparseSpMV` instead of creating three descriptors and
  allocating a workspace every time. The TRSM benchmarks in
  `benchmarks/bench_all.cpp` restore the right hand side from a second device
  buffer with a device to device copy enqueued before the start event, replacing a
  blocking host to device copy that sat between the two event records. The percent
  of cuSPARSE figures in `docs/sparse.md` are retracted pending re-measurement.
- Circular roofline (A3). The compute roofs in `src/profiler/roofline.cpp` are now
  hardware: SM count times the NVML graphics clock times 512 FLOP per cycle for
  FP16 inputs with FP32 accumulate, 256 for FP32. cuBLAS is recorded as a separate
  `cublas_attainable` row and drawn as a dashed attainable line, not as the
  ceiling. The ceilings CSV carries the device name, SM count, clock, and clock
  source; `scripts/plot_roofline.py` titles the chart from the device name instead
  of a hard coded "RTX 5070". `src/tools/device_probe.cu` reads FP32 lanes per SM
  from a compute capability table and prints unknown rather than guessing on an
  architecture it does not know.
- Clocks never locked (A4). `benchmarks/sweep.py` enables persistence mode, locks
  the graphics clock, verifies the lock, stamps `clock_locked` and
  `locked_clock_mhz` into every row, and rejects any row whose measured median
  clock drifted more than 2 percent from the lock. Without a lock it refuses to
  run unless `--allow-unlocked` is passed.
- Overstated claims (A5). The README headline reports the whole percent of cuBLAS
  curve (63.5 at 1024, 80.5 at 2048, 89.9 at 4096, 88.9 at 8192, geometric mean
  79.9) instead of "about 90 percent"; the swizzle line says conflicts fell from
  219M to 33.7M rather than that 219M were removed; the families table carries
  GFLOP/s and percent of vendor columns, so TRSM reads 31.8 percent of cuBLAS
  instead of a correctness figure in a performance column. In `report/main.tex`
  the tiled kernel is 17 percent slower rather than "slightly slower", and the
  starved tensor core round cites its Nsight rows by name.
- Missing evidence (A6). `benchmarks/run_ncu_round.sh` asks for
  `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` and the matching
  wavefront counter, so a re-run commits the bank conflict metric page that the
  central causal finding never had. Rounds 5 and 8 have no committed artifacts;
  that is recorded rather than reconstructed.
- Reproducibility gaps (A7). `.gitignore` no longer excludes the roofline CSVs or
  the sweep JSONL, so the results the report is built from are in the tree. A
  missing `summary.csv`, `roofline.csv`, or `roofline_ceilings.csv` is a hard error
  in `scripts/gen_report_assets.py` instead of a silent skip that reused a
  committed PNG and exited 0.
- `scripts/check_no_dashes.py` scans `.f90` files, and matches `.clang-format`,
  `.clang-tidy`, `.editorconfig`, and `.gitignore` by name; they were listed as
  suffixes, which never matched anything.

- `docs/CORRECTIONS.md`, the corrections register: one dated entry per defect,
  what was claimed, what the repository actually shows, what changed, and what is
  still pending. `experiments/results/legacy_hashes.txt` lists the dead v1 commit
  hashes as a named debt, and `make provenance` runs inside `make check-style`.

### Removed

- `12_COMBINED_cuda-kernel-lab_spec.md` and `PROGRESS.md`. The first could not
  ship alongside the sole authorship claims in `LICENSE` and `README.md`; the
  second's content lives in `docs/DIAGNOSTIC_LOG.md` and `docs/ENGINEERING_LOG.md`.

### Changed

- The README states plainly that I built the project with AI agent tooling
  (Claude Code) driven by my own specifications.
- `docs/ENGINEERING_LOG.md` no longer documents the sudo configuration of my
  machine; it says only that running ncu under sudo returned the same error.

## [1.0.0] - 2026-07-19

### Added

- GEMM ladder: naive, shared tiled, register blocked, cp.async double buffered
  (FP32); WMMA, mma.sync PTX, ldmatrix, and a swizzled top kernel (FP16 and BF16).
  Nine Nsight Compute diagnostic rounds; the top FP16 kernel passes the compute
  bound gate.
- Supporting families: GEMV (naive, warp, vectorized), CSR SpMV (naive, warp per
  row), TRSM (naive, blocked), all versus the vendor library; RAII cuSOLVER LU and
  Cholesky with residual checks.
- NVML telemetry with throttle gating; roofline profiler (analytical plus
  empirical) with a pedagogical figure.
- Resumable full sweep (bench_all plus sweep.py) writing the canonical summary.csv;
  epilogue fusion study justified by the roofline.
- Main report and debug report PDFs built from live results; clang-format,
  clang-tidy, and ruff configs; GitHub Actions CI; CONTRIBUTING.
- Repository skeleton, MIT license, `.gitignore`.
- `scripts/check_no_dashes.py` dash gate, verified clean over the tree.
- CMake build targeting sm_120 with a zero warning policy; Ninja generator.
- `src/tools/device_probe.cu`: prints device properties and a measured
  streaming bandwidth used as the roofline ceiling.
- Public headers `ckl/cuda_check.hpp` and `ckl/device_buffer.hpp`.
- Documented toolchain substitutions and the WSL2 ncu counter permission
  blocker in `docs/ENGINEERING_LOG.md`.
