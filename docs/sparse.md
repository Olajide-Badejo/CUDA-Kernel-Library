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

Skewed matrix, m = n = 131072, about 3.9 million nonzeros, GFLOP/s (2 per
nonzero) and percent of cuSPARSE:

| variant | GFLOP/s | percent of cuSPARSE |
|---|---|---|
| naive | 60.1 | 83.6 |
| warp | 94.8 | 131.8 |
| cuSPARSE | 71.9 | 100.0 |

The warp per row kernel is about 1.6 times the naive kernel on this skewed matrix,
exactly the load balancing win it exists to show, and it edges past the default
cuSPARSE algorithm here. Correctness for both variants is checked against cuSPARSE
and a CPU reference to about 1e-7 (`tests/test_spmv.cpp`).
