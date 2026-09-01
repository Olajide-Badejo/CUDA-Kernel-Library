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

## The tile family

Rung 6 is one block shape, 128 by 128 with a K step of 32. The family templates
that mainloop on `(BM, BN, BK, warps in M, warps in N)` and ships six
instantiations:

| Tile | Warps | Threads | Shared memory, three stages | Registers, data parallel | Registers, stream-K |
|---|---|---|---|---|---|
| 128x128x32 | 2 by 4 | 256 | 49152 B | 122 aligned, 120 predicated | 128 aligned, 126 predicated |
| 128x256x32 | 2 by 8 | 512 | 73728 B | 124 aligned, 122 predicated | 127 aligned, 128 predicated |
| 256x128x32 | 4 by 4 | 512 | 73728 B | 128 aligned, 122 predicated | 128 aligned, 127 predicated |
| 128x64x64 | 2 by 2 | 128 | 73728 B | 162 aligned, 160 predicated | 168 aligned, 132 predicated |
| 64x128x64 | 1 by 4 | 128 | 73728 B | 164 aligned, 122 predicated | 168 aligned, 166 predicated |
| 128x128x64 | 2 by 4 | 256 | 98304 B | 126 aligned, 124 predicated | 160 aligned, 128 predicated |

Every warp tile is 64 by 32, which is why the accumulator stays at 64 registers
across the family and the shapes differ in shared memory footprint and
quantization rather than in register pressure. Nothing spills: every
instantiation reports a zero byte stack frame and zero spill loads and stores.
The constraints are `static_assert`s on the shape, not comments: three stages
inside the 99 KB per block opt-in, warps inside the 48 an SM carries, the staging
work dividing evenly across the block, and a warp tile the `m16n8k16` shape
divides. The counts above come from `nvcc --resource-usage` at `sm_120`.
Relative throughput of the six shapes: pending the locked clock tile sweep.

## Predicated tails

Before this, a shape not divisible by 128, 128, 32 rerouted to the WMMA kernel,
and one not divisible by 64, 64, 16 rerouted again to a scalar kernel with one
thread per output element. Nothing warned, and no benchmark measured the
difference.

The family predicates its edges instead, so every shape runs the same mainloop
with masked edges. Global to shared uses `cp.async`'s source size field: a
sixteen byte chunk that is partly past an edge copies the bytes that exist and
zero fills the rest, and one wholly past an edge copies nothing and zero fills
all sixteen. Zeros in the staged tile contribute nothing to the dot product, so
the mainloop itself carries no edge logic. The epilogue keeps the `float2` store
on aligned shapes and stores element by element under a bounds test otherwise.

One case defeats `cp.async` outright. A sixteen byte copy needs a sixteen byte
aligned source, and a row of A starts at a multiple of k, so an odd k puts every
second row on an odd address. Those shapes stage through plain loads and shared
stores, chosen by a flag the launcher computes from the leading dimensions and
the operand base addresses. The pipeline structure is unchanged: the barrier that
already separates the stage writing a buffer from the iteration reading it covers
a synchronous store just as well as an asynchronous one.

The scalar half precision path is still reachable, but only by naming
`Algo::kWmmaFp16` explicitly. `Algo::kAuto` never routes to it, and `chosen`
reports whatever did run. Cost of the tails relative to the aligned path: pending
the locked clock tile sweep.

## Split-K

A 1024 by 1024 output at a 128 by 128 tile is 64 blocks on a 48 SM part: one
wave and a quarter, so most of the machine idles for most of the call. Split-K
cuts the contraction into slices and runs each as its own grid layer, then a
fixup pass sums the slices.

Two passes rather than atomics into C. Atomics would need no scratch, but they
would make the result depend on the order blocks happened to retire, which is
not something a correctness test can pin down, and they would have to read C to
apply beta from inside the mainloop. The first pass writes raw partial sums into
its own plane and never touches C; the second sums the planes, multiplies by
alpha once, and adds beta times C once, so beta zero still does not read C.

The scratch is `slices * m * n` floats. `ckl::gemm_workspace_size` answers with
it, and a Context that already owns a workspace that large keeps the allocation
out of the call. Without one the driver allocates and frees its own around the
launch, which is correct but costs an allocation a benchmark should not be
timing. The slice count a caller asks for is a request: each slice is rounded up
to a whole number of the tile's K step, so a short K yields fewer slices than
asked, and `ckl::gemm_split_k_slices` reports how many. Speedup at the shapes
where waves are scarce: pending the locked clock tile sweep.

## Stream-K

Split-K cuts every tile the same way and pays a full extra pass over C for it.
Stream-K only touches the tiles that would have left SMs idle. The grid is laid
out as a block of persistent CTAs followed by ordinary data parallel ones:

```
[ sk_blocks stream-K CTAs ][ dp_tiles data parallel CTAs ]
```

The data parallel CTAs take the tiles that fill whole waves, one each. The
stream-K CTAs share out the K iterations of what is left, which is at most two
waves of tiles: the remainder that does not fill a wave, plus one full wave to
give the split something to absorb into.

A CTA whose share stops before the end of a tile holds only part of the sum, so
it publishes its partial into its own plane of the workspace, fences, and raises
a flag. The CTA that covers the tile's last K iteration waits on the flag of
every peer whose share ended inside the tile, adds the planes into its
accumulator, and writes C once.

The direction of that wait is the safety argument, so it is deliberate: an owner
only ever waits on CTAs with a lower index than its own. Blocks are dispatched in
increasing index order, so a resident owner implies every peer it can wait on is
already resident or already retired. Waiting upward instead would need every
stream-K CTA co-resident, which is an assumption about occupancy that a
profiler's instrumentation or a future register change could quietly turn into a
hang. The data parallel CTAs never wait at all.

Three orderings hold the protocol together, and all three are explicit.
`__threadfence` between the partial stores and the flag makes the partial visible
device wide before the flag that advertises it. `__syncthreads` between them makes
that true for every thread of the block and not just the one that raises the
flag. And the persistent loop carries a `__syncthreads` of its own, because the
shared tile buffers are reused from one tile to the next and the next tile's
`cp.async` stores would otherwise race the previous tile's `ldmatrix` reads;
that is the barrier `compute-sanitizer --tool racecheck` reports when it is
removed. Speedup on small and odd shapes: pending the locked clock tile sweep.

## Dispatch, Section 9.6

`Algo::kAuto` on an FP16 input with an FP32 output now decides like this. For
every tile in the family it computes

```
waves = ceil(m / BM) * ceil(n / BN) / (SMs * blocksPerSM)
```

with `blocksPerSM` from the occupancy API rather than from an assumption about
registers, cached per shape per process. It keeps the tiles whose measured
throughput at the nearest swept shape is within five percent of the best, and
among those picks the one maximizing `waves / ceil(waves)`, the fraction of the
final wave that actually has work. When `waves` is below one the grid cannot fill
the machine at all, and the decision escalates: to split-K when the occupancy
gap and the K depth both allow at least two slices with enough K steps each for
the pipeline to fill, and to stream-K otherwise, since stream-K spreads the same
work without a second pass over C.

The measured half of that rule needs a committed tile sweep, and the sweep is
owner work at locked clocks. Until one exists the throughput filter has nothing
to filter with, so the untuned path decides in two bands. Above two waves of the
128 by 128 by 32 reference tile it keeps that tile, because quantization on its
own prefers the smallest shape in the family and that is the wrong trade where
the grid is already deep: quantization costs a couple of percent there either
way, while a smaller tile stages more bytes per multiply add for the whole call,
and the reference tile is the one shape with measured history behind it. Below
two waves the quantization rule decides which tile runs, and below one wave the
decision escalates as above.

Neither band is a measurement. Every answer says so: `ckl::GemmPlan::tuned` is
false for the whole untuned path, and the guard goes away the moment a sweep is
committed, because the five percent filter then has data to work with. Relative
throughput of the tiles at every shape: pending the locked clock sweep.
`ckl::gemm_query` returns the rung, `ckl::gemm_plan` returns the rung plus the
tile, the slice count and the tuned flag, and the `chosen` out-param of
`ckl::gemm` reports the rung that actually ran.

The sweep that fills the table is `benchmarks/tile_sweep.py`, which drives every
instantiation across the Gate D shape matrix through `bench_all` under the
variant name `tile_<BM>x<BN>x<BK>` and writes
`experiments/results/tile_sweep.csv`. It refuses to run without a locked clock
unless `--allow-unlocked` is passed, and an unlocked run stamps every row
`clock_locked=false`, which means the heuristic must not be tuned from it. The
dispatcher reads the CSV from `experiments/results/tile_sweep.csv` relative to
the working directory, or from wherever `CKL_TILE_SWEEP_CSV` points.

## Tolerances

FP32 correctness gate: relative Frobenius error under 1e-4 versus cuBLAS,
verified on square, non square, non tile aligned, sub tile, and zero dimension
shapes (`tests/test_gemm.cpp`).

Tensor tolerances, derived from the measured error distribution over the test
shapes (`tests/test_gemm_tensor.cpp`):

- Hand written kernel versus the cuBLAS tensor oracle of the same precision:
  about 1e-7 relative Frobenius. The two agree to FP32 accumulate rounding, so
  the kernel is numerically the same computation as cuBLAS, not an approximation
  of it.
- FP16 versus an FP32 CPU reference: about 2.6e-4. This is the honest accuracy of
  FP16 storage with FP32 accumulate on these shapes.
- BF16 versus an FP32 CPU reference: about 2.1e-3, larger than FP16 as expected
  from BF16's shorter mantissa (8 bits versus 10).

The correctness gate for the tensor kernels is 1e-5 relative Frobenius against
the oracle, which the measured 1e-7 clears by two orders of magnitude. The v1
gate was 5e-2, which a kernel that dropped its last K stage would have sailed
through; that is the number this line used to quote and the reason it no longer
does. Every tile family member is held to the same 1e-5, at the unaligned shapes
as well as the aligned ones.

## Baseline result

Measured on this machine at Phase 1 (the sweep rows in
`experiments/results/summary.csv` carry the commit hash). The naive kernel plateaus near 2 TFLOP/s and falls to roughly 9
percent of cuBLAS at 2048 and above, because it refetches every A and B element
from global memory with no reuse and is bandwidth and latency bound once the
data leaves cache. Closing that gap is what the rest of the ladder does, and the
diagnostic log records each step with Nsight evidence.
