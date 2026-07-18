# Design decisions

Why the design is what it is. Short entries, added when a choice is made, so the
report's methodology chapter and any future reader can see the reasoning without
reverse engineering it from code.

## Ladder before library

Each GEMM variant exists to isolate one technique so the benchmark table can
attribute a gain to a cause: shared memory tiling cuts global traffic by the tile
factor, register blocking lifts arithmetic intensity per thread, cp.async double
buffering overlaps global loads with math, WMMA moves the inner product onto
tensor cores, and the mma.sync PTX variant removes the WMMA abstraction penalty
and exposes fragment layout. Register blocking as the decisive step over pure
tiling follows Volkov and Demmel, SC 2008.

## CUTLASS as reference, not crutch

A CUTLASS device GEMM tuned for sm_120 is an upper reference for what template
metaprogrammed open source reaches on this card and a source of technique for the
pipelined mainloop. The hand written kernels stay hand written; the report
compares mine, CUTLASS, and cuBLAS on identical shapes.

## Comparison policy: same device, always

Percent of cuBLAS on the same GPU is the only defensible headline for a hand
written kernel because it cancels the hardware out of the claim. No cross GPU
numbers appear anywhere.

## Measured ceilings over datasheet

The roofline uses measured bandwidth (about 540 GB/s streaming at Phase 0) and a
measured or empirically bounded compute ceiling rather than the 672 GB/s
datasheet peak, so the ridge point and the compute bound gate reflect what the
machine actually delivers.

## Build and environment

- Device standard is C++20 (newest nvcc 13.3 accepts for device code); host code
  is C++23. Recorded rather than downgrading host silently.
- `-Wpedantic` is applied to hand written host C++ only, because nvcc's generated
  stub files under separable compilation are not ours to keep pedantic clean.
- Repo on the Windows filesystem, built in WSL2 through a symlink, for IDE
  visibility; revisit if CUTLASS fetch I/O over `/mnt/c` becomes a bottleneck.
