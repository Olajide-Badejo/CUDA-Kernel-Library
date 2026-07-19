# Changelog

Notable changes per phase. Dates are ISO. This project follows the phase
roadmap in the build specification; versions are tagged at the definition of
done milestones.

## Unreleased

### Phases 1 to 10 (2026-07-19)

- GEMM ladder: naive, shared tiled, register blocked, cp.async double buffered
  (FP32); WMMA, mma.sync PTX, ldmatrix, and a swizzled top kernel (FP16 and BF16).
  Nine Nsight Compute diagnostic rounds; the top FP16 kernel passes the compute
  bound gate at about 90 percent of cuBLAS.
- Supporting families: GEMV (naive, warp, vectorized), CSR SpMV (naive, warp per
  row), TRSM (naive, blocked), all versus the vendor library; RAII cuSOLVER LU and
  Cholesky with residual checks.
- NVML telemetry with throttle gating; roofline profiler (analytical plus
  empirical) with a pedagogical figure.
- Resumable full sweep (bench_all plus sweep.py) writing the canonical summary.csv;
  epilogue fusion study justified by the roofline.
- Main report and debug report PDFs built from live results; clang-format,
  clang-tidy, and ruff configs; GitHub Actions CI; CONTRIBUTING.

### Phase 0 (2026-07-19)

- Repository skeleton, MIT license, `.gitignore`, attribution disabled.
- `scripts/check_no_dashes.py` dash gate, verified clean over the tree.
- CMake build targeting sm_120 with a zero warning policy; Ninja generator.
- `src/tools/device_probe.cu`: prints device properties and a measured
  streaming bandwidth used as the roofline ceiling.
- Public headers `ckl/cuda_check.hpp` and `ckl/device_buffer.hpp`.
- Documented toolchain substitutions and the WSL2 ncu counter permission
  blocker in `PROGRESS.md` and `docs/ENGINEERING_LOG.md`.
