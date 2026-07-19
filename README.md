# CUDA Kernel Library

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

## Headline results

Measured on this RTX 5070, every number my kernel versus the vendor library in the
same process. The GEMM ladder at 4096 cubed, each rung against cuBLAS of the same
precision:

| rung | precision | GFLOP/s | percent of cuBLAS |
|---|---|---|---|
| naive | FP32 | 1905 | 8.6 |
| shared tiled | FP32 | 1589 | 7.2 |
| register blocked | FP32 | 10380 | 47 |
| cp.async double buffered | FP32 | 16174 | 73 |
| WMMA | FP16 | 36308 | 60 |
| mma.sync plus ldmatrix plus swizzle (top) | FP16 | 54930 | 90 |

The top tensor kernel passes the compute bound gate (Nsight Speed of Light: tensor
pipe 83 percent versus memory 30 percent) and reaches 90 percent of cuBLAS at both
4096 and 8192 cubed for FP16. Each rung was profiled and its next change chosen from
the evidence; the nine round record is in `docs/DIAGNOSTIC_LOG.md`.

Supporting families, same device, versus the vendor library:

| family | best hand written | baseline | note |
|---|---|---|---|
| GEMV (warp) | about 610 GB/s at 8192 | cuBLAS SGEMV | memory bound, matches cuBLAS |
| CSR SpMV (warp per row) | 1.6 times the naive kernel | cuSPARSE | skewed degree matrix |
| TRSM (blocked) | matches cuBLAS to 1e-7 | cuBLAS STRSM | residual verified |
| cuSOLVER LU / Cholesky | residual 5.6e-7 / 4.3e-7 | (is the vendor path) | RAII wrapper |

Roofline (measured ceilings 579 GB/s, 23 TFLOP/s FP32, 65 TFLOP/s tensor):
`report/figures/roofline.png`. Full per shape data: `experiments/results/summary.csv`.

## Status

Kernel families, diagnostic rounds, sweep, and roofline are complete and verified
on the machine (see `PROGRESS.md`). Remaining: the CUTLASS reference and the two
report PDFs. Every number above is traceable to a results file and a commit.

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
