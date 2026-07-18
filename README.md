# CUDA Kernel Lab

A CUDA kernel library whose GEMM path is driven, through explicit optimization
steps and Nsight Compute diagnostic rounds, from a naive baseline to a compute
bound tensor core kernel that lands close to cuBLAS on one specific GPU. Every
performance number in this repository is my kernel measured against cuBLAS (or
cuSPARSE for SpMV) on the same RTX 5070, in the same process. There are no cross
GPU comparisons anywhere.

Alongside the GEMM ladder the library carries GEMV, TRSM, CSR SpMV, and an RAII
cuSOLVER wrapper, a roofline profiler in analytical and empirical modes, and
NVML telemetry sampled during every benchmark to catch thermal throttling before
a number is trusted.

Author: Olajide Badejo. License: MIT.

## Status

Under construction, built phase by phase (see the roadmap in the build spec and
`PROGRESS.md`). The headline results table is added last, filled only from real
runs on this machine; until then any pending value reads "pending".

## Target machine

RTX 5070 (Blackwell GB205, compute capability 12.0, sm_120, 48 SMs, 12 GB
GDDR7), CUDA Toolkit 13.3, built and run inside WSL2 Ubuntu. Measured ceilings
captured at Phase 0 (used by the roofline in place of the datasheet):

| Quantity | Value |
|---|---|
| Peak memory bandwidth (theoretical, clock times bus) | 672 GB/s |
| Measured streaming bandwidth | about 540 GB/s |
| Peak FP32 (estimate, non tensor core) | about 32.3 TFLOP/s |
| L2 cache | 48 MiB |
| Shared memory per SM | 100 KiB |

Regenerate with `./build/device_probe`.

## Quick start

Requires an NVIDIA GPU with driver support for CUDA 13.3 and, for the diagnostic
rounds, GPU performance counter access enabled for ncu (see `PROGRESS.md` for the
WSL2 note).

```sh
cmake -S . -B build -G Ninja -DCMAKE_BUILD_TYPE=Release
cmake --build build
ctest --test-dir build            # GPU tests; skip cleanly without a GPU
```

A `Makefile` wrapping `setup`, `all`, `report`, and `check-style` targets is
added as the build fills in. Full reproduction from a clean clone via
`make setup && make all` is a definition-of-done item.

## Layout

See the build specification for the full tree. In short: `include/ckl` public
headers, `src/<family>` kernels, `tests` correctness against library oracles,
`benchmarks` the sweep harness, `docs` per component notes plus the diagnostic
and engineering logs, `report` and `report_debug` the two PDFs.

## Documentation

- `PROGRESS.md`: per phase status with verified output.
- `docs/ENGINEERING_LOG.md`: dated problems and fixes.
- `docs/DIAGNOSTIC_LOG.md`: the round by round GEMM optimization record.
- `docs/DESIGN_DECISIONS.md`: why the design is what it is.
