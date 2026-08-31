# CSR SpMV

Sparse matrix vector product in compressed sparse row form, `y = alpha * (A * x)
+ beta * y`. Two hand written variants against cuSPARSE `cusparseSpMV`.

## Why a skewed matrix

With a uniform row degree the thread per row and warp per row kernels perform
about the same, so a uniform test would hide the difference the warp variant is
meant to fix. The tests and benchmark use a skewed distribution: most rows have a
handful of nonzeros, but about two percent are hundreds of nonzeros long. That is
where load imbalance bites.

## Variants

- naive (`spmv_csr_naive.cu`): one thread per row. A long row makes its thread run
  far longer than its warp neighbors, and the warp retires at the speed of its
  slowest row.
- warp (`spmv_csr_warp.cu`): one warp per row, the row's nonzeros shared across 32
  lanes with a shfl reduction. A long row is now split 32 ways, and the strided
  reads of `col_idx` and `values` coalesce within a row.

## Measured (this machine)

**The percent of cuSPARSE column below is retracted.** The v1 cuSPARSE wrapper
created three descriptors, sized the SpMV workspace, and ran `cudaMalloc` and
`cudaFree` on every call, and the harness timed all of it. So the "cuSPARSE"
denominator was the vendor's SpMV plus a per call setup my own kernels never
paid. That is defect A2 in [`CORRECTIONS.md`](CORRECTIONS.md). The 131.8 percent
figure, and the 83.6 percent next to it, say nothing about cuSPARSE.

`src/sparse/cusparse_ref.cpp` now caches the descriptors and the workspace in a
plan, so a repeated timed call enqueues only `cusparseSetStream` and
`cusparseSpMV`. Re-measurement on the fixed wrapper is pending; it is part of the
full sweep re-run, which also has to happen under locked clocks (defect A4).

Skewed matrix, m = n = 131072, about 3.9 million nonzeros, GFLOP/s (2 per
nonzero):

| variant | GFLOP/s | percent of cuSPARSE |
|---|---|---|
| naive | 60.1 | retracted, pending re-measurement |
| warp | 94.8 | retracted, pending re-measurement |
| cuSPARSE | contaminated, pending re-measurement | 100.0 |

The kernel to kernel comparison survives, because both hand written variants were
timed the same way: the warp per row kernel is about 1.6 times the naive kernel on
this skewed matrix, which is the load balancing win it exists to show. Whether it
beats cuSPARSE is an open question until the sweep is re-run. Correctness for both
variants is checked against cuSPARSE and a CPU reference to about 1e-7
(`tests/test_spmv.cpp`), and that check was never affected.
