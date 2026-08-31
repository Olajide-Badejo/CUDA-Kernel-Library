# Changelog

Notable changes per phase. Dates are ISO. The format follows
[Keep a Changelog](https://keepachangelog.com/en/1.1.0/), and versions are tagged
at the definition of done milestones.

## [Unreleased]

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

### Added

- `docs/CORRECTIONS.md`, the corrections register: one dated entry per defect,
  what was claimed, what the repository actually shows, what changed, and what is
  still pending.
- `experiments/results/legacy_hashes.txt`, the allowlist of dead v1 commit hashes.
- `make provenance`, and `make check-style` now runs it alongside the dash gate.

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
