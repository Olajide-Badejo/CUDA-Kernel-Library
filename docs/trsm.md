# TRSM

Triangular solve with multiple right hand sides: solve `L X = alpha B` for X, with
L lower triangular (m by m, non unit diagonal) and B the m by n right hand side,
solved in place. Row major, single precision. Measured against cuBLAS STRSM.

## Naive versus blocked

- naive (`trsm_naive.cu`): forward substitution, one thread per right hand side
  column, sequential down the rows. Correct but the parallelism is only n wide, so
  it leaves the machine idle for tall systems with few right hand sides.
- blocked (`trsm_blocked.cu`): sweep the matrix in 32 wide diagonal blocks. For
  each block, solve the small block system by substitution, then subtract that
  block's contribution from all trailing rows. That subtraction, `B_below -=
  L_below,k * X_k`, is a dense matrix multiply and holds almost all the flops. So
  the blocked solve is a thin substitution wrapper around a GEMM, which is how BLAS
  libraries make TRSM run near GEMM speed.

## Correctness

Verified against cuBLAS STRSM and a double precision CPU reference, plus a residual
check `||L X - alpha B|| / ||alpha B||`. L is built diagonally dominant so the
system is well conditioned and the float solve stays close to the double reference.
Across square, tall, and non block aligned shapes, both variants match cuBLAS to
about 1e-7 with residuals near 1e-7 (`tests/test_trsm.cpp`). The oracle orientation
(row major left solve mapped to a cuBLAS right side upper solve) is confirmed by
the CPU cross check at about 1e-8.

Timing for TRSM is collected by the full sweep (`benchmarks/sweep.py`) alongside
the other families.
