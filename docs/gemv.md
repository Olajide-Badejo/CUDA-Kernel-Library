# GEMV

Matrix vector product, `y = alpha * (A * x) + beta * y`, A row major of shape m by
n. Three variants, all measured against cuBLAS SGEMV on the same GPU.

## GEMV is memory bound, and the roofline says so

Every element of A is read exactly once and used in a single multiply add. The
arithmetic intensity is therefore fixed at about 2 FLOPs per 4 byte element read,
that is 0.5 FLOP per byte, regardless of how the work is tiled. The measured
roofline ridge on this machine is about 60 FLOP per byte (32.3 TFLOP/s over 540
GB/s), so GEMV sits far to the left of the ridge: it is bandwidth bound with no
tiling that can change that. This is why there is no tensor core GEMV here; it
would be a bandwidth benchmark wearing a tensor core costume.

The right yardstick is therefore effective bandwidth, not GFLOP/s, and the goal is
to get the read of A close to the measured streaming bandwidth.

## Variants

- naive (`gemv_naive.cu`): one thread per output row. At each inner step the 32
  threads of a warp read A elements one full row (n floats) apart, so the loads do
  not coalesce. This is the honest baseline.
- warp (`gemv_warp.cu`): one warp per row, lanes stride across the row so adjacent
  lanes read adjacent A elements (coalesced), then a `__shfl_down_sync` reduction
  sums the lane partials.
- vectorized (`gemv_vectorized.cu`): the warp kernel with float4 loads of A and x,
  a quarter of the memory instructions. Falls back to the warp kernel when n is
  not a multiple of 4.

## Measured (this machine)

Effective bandwidth for the read of A (the dominant traffic), versus cuBLAS:

| size | naive GB/s | warp GB/s | vectorized GB/s | cuBLAS GB/s |
|---|---|---|---|---|
| 2048 | 168 (15 percent) | 1314 (121 percent) | 1138 (104 percent) | 1091 |
| 4096 | 250 (43 percent) | 590 (101 percent) | 577 (98 percent) | 587 |
| 8192 | 325 (53 percent) | 616 (101 percent) | 610 (100 percent) | 613 |

Reading. The naive kernel is coalescing limited and never gets close to
bandwidth. The warp kernel matches cuBLAS and reaches about 610 GB/s at 8192,
close to the measured streaming ceiling, because the row read now coalesces. The
vectorized kernel is essentially tied with the warp kernel: once the read is
coalesced and the kernel is bandwidth bound, wider loads do not add much, which is
the honest result rather than a manufactured win. The 2048 numbers exceed DRAM
bandwidth because a 2048 by 2048 float matrix is 16 MB and fits in the 48 MB L2,
so that row is L2 bandwidth; at 4096 (64 MB) and 8192 (256 MB) the matrix spills
L2 and the numbers reflect true DRAM bandwidth.
