# Roofline profiler

The roofline model places each kernel against the machine's limits: attainable
performance is `min(peak compute, peak bandwidth * arithmetic intensity)`. Where a
kernel's intensity sits relative to the ridge (peak compute over peak bandwidth)
tells you whether more work per byte could help (left of the ridge, memory bound)
or whether it is already limited by the math units (right of the ridge, compute
bound). Williams, Waterman, Patterson, CACM 2009.

## How to read the figure

`report/figures/roofline.png` shades the two regions. Left of the ridge is the
memory bound region: performance rides the sloped bandwidth ceiling, so a kernel
there is limited by how fast it can move bytes. Right of the ridge is the compute
bound region: performance is capped by a flat compute ceiling, so a kernel there is
limited by the math units. The sloped edge is the measured streaming bandwidth; the
two flat edges are the measured FP32 and tensor compute peaks; each point is a
kernel at its arithmetic intensity and achieved throughput. FP32 kernels are
circles, tensor kernels triangles, so the split reads in grayscale too.

## Two modes

- Analytical (`benchmarks/support/roofline.hpp`): closed form FLOP and byte counts per
  operation. GEMM does `2 m n k` FLOPs and, in the minimum traffic model, moves
  `(m k + k n + m n)` elements; GEMV does `2 m n` FLOPs and moves about `m n`
  elements (A dominates). These give the arithmetic intensity of each operation.
- Empirical (`tools/roofline.cpp`): the bandwidth ceiling is measured on this GPU
  with a large device to device copy. The compute roofs are hardware, not a vendor
  library: SM count times the graphics clock times 512 FLOP per cycle for FP16
  inputs with FP32 accumulate, 256 for FP32 (defect A3 in `docs/CORRECTIONS.md`
  retired the old cuBLAS-derived "roof"). cuBLAS is measured too, but it lands as
  a separate attainable line, because a competitor is not a ceiling. The tool then
  times the ladder and places each variant, writing
  `experiments/results/roofline.csv` and `roofline_ceilings.csv` with the device
  name, SM count, clock, and clock source recorded.
  `scripts/plot_roofline.py` renders the figure from those.

## What it shows on this machine

Measured ceilings: about 579 GB/s streaming bandwidth, 23 TFLOP/s FP32, 65 TFLOP/s
tensor. Ridge points at about 40 FLOP/byte (FP32) and 113 FLOP/byte (tensor). The
GEMM variants sit far to the right of both ridges (intensity in the hundreds to
thousands), confirming GEMM is compute bound and that the naive kernel's low
throughput is an implementation gap, not an algorithmic ceiling. The top tensor
kernel `gemm_mma_opt` lands at about 84 percent of the tensor roof. GEMV sits at
0.5 FLOP/byte, far to the left on the bandwidth diagonal: memory bound, as the
model predicts and the GEMV kernels confirm. This is the written roofline evidence
the design references for the tile choice and the compute bound claim.
