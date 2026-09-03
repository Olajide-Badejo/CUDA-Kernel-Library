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

## Measured bandwidth, hardware compute roofs

The roofline uses measured streaming bandwidth rather than the datasheet peak.
The compute roofs changed in 1.1.0: v1 measured cuBLAS and called it the
ceiling, which made every percent-of-roof a percent-of-cuBLAS restated (defect
A3); the roofs are now SM count times clock times the per-cycle FLOP rate, with
cuBLAS drawn as a separate attainable line.

## Build and environment

- Device standard is C++20 (newest nvcc 13.3 accepts for device code); host code
  is C++23, with the public headers requiring only C++17 of a consumer.
- Separable compilation was dropped in 1.1.0: nothing device links, RDC blocked
  whole program ptxas work, and an un-device-linked archive could not be
  consumed outside CMake, which the Fortran binding needed.
- Repo lives on the Windows filesystem and builds in WSL2; the host compiler is
  g++-14 because GCC 15's libstdc++ breaks the nvcc 13.3 frontend (the story is
  in `docs/building.md`). CUTLASS is fetched shallow at a pinned commit.

## Where the 1.1.0 decisions live

The library-shaping choices of 1.1.0 are recorded next to what they shaped: the
packaging and visibility story in `docs/building.md`, the API contracts in
`docs/using.md`, the measurement protocol rules in `docs/benchmarking.md`, the
per family mechanism choices in `docs/gemm.md`, `docs/sparse.md`,
`docs/fft.md`, and `docs/scan.md`, and every retracted or corrected claim in
`docs/CORRECTIONS.md`.
