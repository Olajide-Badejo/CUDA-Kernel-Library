# cuSOLVER dense solver

`ckl::DenseSolver` (`src/solver/dense_solver.cpp`) is an RAII wrapper over cuSOLVER
dense factorizations. It owns the cuSOLVER handle and the scratch each solve needs
(workspace sized by the `_bufferSize` query, the pivot array, the device info
word), so a solve is one call and nothing leaks when an exception unwinds.

Two solvers, column major (LAPACK convention), single precision:

- LU with partial pivoting: `getrf` then `getrs`, for general systems.
- Cholesky: `potrf` then `potrs`, for symmetric positive definite systems.

Both solve `A X = B` in place over B and leave the factorization in A. The device
info word is checked after each factorization, so a singular LU or a non positive
definite Cholesky raises rather than silently returning garbage.

## Done means residual, not "no error"

The test (`tests/test_solver.cpp`) does not settle for the absence of a CUDA error.
It forms the residual `||A X - B|| / ||B||` in double precision against a saved copy
of the original A (the factorization overwrites A). On this machine, n = 512, 8
right hand sides: LU residual about 5.6e-7, Cholesky residual about 4.3e-7. The
Cholesky test matrix is built as `B_transpose B + n I` so it is genuinely positive
definite.
