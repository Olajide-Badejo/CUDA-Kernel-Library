# GEMM

The GEMM path is the spine of this project: a ladder of variants, each isolating
one optimization technique, driven from a naive baseline toward a compute bound
tensor core kernel and measured at every rung against cuBLAS on the same GPU.

## Convention

Row major, `C = alpha * (A * B) + beta * C`, with A of shape m by k, B of shape
k by n, C of shape m by n. Row major is the natural indexing for the hand
written kernels. The cuBLAS oracle is column major, so it is wrapped with the
transpose identity (a row major `A * B` shares memory with a column major
`B_transpose * A_transpose`), which makes every comparison same shape and same
process. See `src/gemm/cublas_gemm.cpp` for the exact call.

## Variants

The ladder, in optimization order (built incrementally):

1. naive: one thread per output element, natural indexing, no reuse. The honest
   baseline. `src/gemm/gemm_naive.cu`.
2. tiled: shared memory staging with bank conflict padding.
3. register blocked: 128x128x32 block, 8x8 thread tile, float4 loads; tile
   parameters templated and swept.
4. cp.async double buffered: overlap global loads with math.
5. WMMA fp16 and bf16: 16x16x16 fragments, FP32 accumulate.
6. mma.sync PTX with ldmatrix fragment loads.
7. CUTLASS reference instantiation for sm_120.

cuBLAS is the baseline for every rung.

## Tolerances

FP32 correctness gate: relative Frobenius error under 1e-4 versus cuBLAS,
verified on square, non square, non tile aligned, sub tile, and zero dimension
shapes (`tests/test_gemm.cpp`). The FP16 and BF16 tolerances are derived
empirically from the error distribution when the tensor variants land and are
written here then.

## Baseline result

Measured on this machine at Phase 1 (see `PROGRESS.md` for the table with the
commit hash). The naive kernel plateaus near 2 TFLOP/s and falls to roughly 9
percent of cuBLAS at 2048 and above, because it refetches every A and B element
from global memory with no reuse and is bandwidth and latency bound once the
data leaves cache. Closing that gap is what the rest of the ladder does, and the
diagnostic log records each step with Nsight evidence.
