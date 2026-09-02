# Reduction and scan

Device wide reduction and prefix scan, two ladders out of one plan, with CUB as
the baseline for both.

## Name the baseline before quoting anything

The vendor implementation for this family is CUB. It ships inside the CUDA
toolkit as part of CCCL, and it is what Thrust dispatches to, so
`cub::DeviceReduce` and `cub::DeviceScan` are the numbers to beat.

Thrust is not benchmarked separately. It is CUB behind a different call
signature: `thrust::reduce` and `thrust::inclusive_scan` on the device policy
call straight into the same `DeviceReduce` and `DeviceScan` entry points this
library measures. A Thrust row would be a second measurement of the same kernel
with a different amount of host side wrapper around it, and reporting it as if it
were an independent implementation would be misleading.

cuBLAS has no scan at all. Its only overlap with this family is `cublasSasum`
and `cublasSdot`, which are reductions of a different thing: `asum` reduces
absolute values and `dot` reads two arrays of N floats. They belong on a
secondary row that says which operation it is, never as the baseline.

CUB implements decoupled look-back itself, so parity with it on the top scan rung
is the realistic goal and beating it by a wide margin is not the claim. The
interesting deliverable is the explanation of whatever gap exists in either
direction.

**Pinned CUB version.** `CUB_VERSION` is `300304`, which is CUB 3.3.4, from CUDA
Toolkit 13.3.73. It is read from the header macro through
`ckl::ScanPlan::cub_version`, printed by `bench_scan`, and stamped into every
`bench_all` row as `cub_version`. A percentage against CUB is a percentage
against one CUB.

## The roof

Per Section 5 and the ground rule, percent of roof is against the measured
579 GB/s DRAM figure from the device probe on this 48 SM `sm_120` part, never
against CUB.

- A reduction reads `N * sizeof(T)` bytes and writes one element.
- Decoupled look-back scan moves `2 * N * sizeof(T)` plus the tile status
  traffic.
- Reduce then scan moves about `3 * N * sizeof(T)`.
- Scan then propagate moves about `4 * N * sizeof(T)`.

`ckl::ScanPlan::reduce_model_bytes` and `ckl::ScanPlan::scan_model_bytes` are the
one implementation of those models. Every effective GB/s this family quotes uses
the rung's own declared model bytes as the numerator, so a rung that moves more
data cannot look faster by moving it, and the same number is what a measured
`dram__bytes.sum` gets checked against.

## The reduction ladder

One rung per file under `src/scan/`, each isolating one technique.

| Rung | File | Technique | Expected limiter |
|---|---|---|---|
| r0 | `reduce_atomic.cu` | one atomic per element onto a single global accumulator | atomic serialization on one address, not DRAM |
| r1 | `reduce_shared_tree.cu` | shared memory tree per block, one partial per block | barriers and shared bandwidth |
| r2 | `reduce_shuffle.cu` | `__shfl_down_sync` in the warp, one shared slot per warp, one atomic per block | load issue rate |
| r3 | `reduce_vec4.cu` | 16 byte loads and a grid stride loop over persistent blocks | DRAM bandwidth |
| r4 | `reduce_single_pass.cu` | one launch; the block that draws the last ticket combines the partials, against a two launch arm | DRAM bandwidth at large N, launch overhead at small N |

Two more rungs sit beside the ladder with rows of their own.
`reduce_deterministic.cu` is the bitwise deterministic mode and
`reduce_kahan.cu` is the compensated one. They are separate files rather than
branches inside r3 for the same reason every other rung is: a file per technique
is what makes the ladder readable.

The limiter progression is the hypothesis the sweep tests, not a result. r0 is
contention bound and is expected to sit one to two orders below the roof. r1 and
r2 are instruction and barrier bound. r3 is the first rung that should touch DRAM
bandwidth. r4 changes nothing at large N and removes one launch at small N. Below
about `2^16` elements the kernel is shorter than the launch, so those rows are
reported both stream launched and CUDA graph launched and no bandwidth claim is
made from them.

Both arms of r4 run the identical partials kernel. Only the combine differs, so
the difference between the two rows is one kernel launch and nothing else, which
is the only way that question can be answered honestly.

## The scan ladder

| Rung | File | Technique | Notes |
|---|---|---|---|
| s0 | `scan_hillis_steele.cu` | Hillis-Steele block scan | work inefficient, `O(N log N)` work at block scope |
| s1 | `scan_blelloch.cu` | work efficient upsweep and downsweep, bank conflict padding `i + i/32` | `O(N)` work, twice the barriers |
| s2 | `scan_three_kernel.cu` | scan then propagate over the production block scan and 16 byte loads | about `4 * N * sizeof(T)` moved |
| s3 | `scan_reduce_then_scan.cu` | reduce then scan, two passes over the input | about `3 * N * sizeof(T)` moved |
| s4 | `scan_lookback.cu` | single pass decoupled look-back | about `2 * N * sizeof(T)` moved; the top rung |

**s0 and s1 are block primitives, and they reach device wide through the same
decomposition s2 uses.** That is a decision worth stating plainly, because it is
what makes the three rows comparable. A Hillis-Steele block scan is not by itself
a device wide scan; it has to be wrapped in something. Wrapping s0 and s1 in one
decomposition and s2 in another would mean the s0 to s2 difference was two
changes at once. So `scan_levels.cuh` holds the scan then propagate recursion
once, as a template over the block scan flavour and the items each thread
carries, and the three rung files supply nothing but the policy. What separates
the three rows is then exactly the block primitive and the tile width.

s0 and s1 carry one item per thread, so their tile is 256 elements and their load
needs no shared memory transpose: the coalesced arrangement and the blocked one
are the same arrangement at one item per thread. s2, s3 and s4 carry 32 bytes per
thread, which is eight elements at four bytes and four at eight, loaded as two 16
byte vectors and transposed through shared memory into the blocked arrangement a
serial scan needs.

The recursion is on the aggregates: one block scans one tile and publishes its
aggregate, the aggregates are scanned exclusively by the same driver one level
down, and a propagate pass folds each tile's exclusive prefix back in. At `2^28`
elements and a 256 element tile that is four levels and the deepest level is a
single tile. All of the level buffers come out of one plan owned allocation.

### s4, decoupled look-back

Merrill and Garland, "Single-pass Parallel Prefix Scan with Decoupled
Look-back", NVIDIA technical report NVR-2016-002, 2016.

Three implementation constraints are not optional and all three are in
`src/scan/scan_lookback.cu`.

**The tile index comes from an `atomicAdd` on a global counter, never from
`blockIdx.x`.** That is what makes the wait safe. A block that claimed index k did
so after every block holding a lower index had already started, so those blocks
are resident and can make progress. Indexing by `blockIdx.x` would let a resident
block wait on one the scheduler has not launched yet, and that deadlocks.

**The status word is published with release semantics and read with acquire
semantics,** through `cuda::atomic_ref<unsigned int, cuda::thread_scope_device>`.
The aggregate and the inclusive prefix sit beside the flag as plain stores, which
is exactly what release and acquire exist to order: the release store on the flag
publishes everything the same thread wrote before it, and the acquire load makes
those writes visible to the reader before it touches them.

**The status array is zeroed by a memset enqueued on the same stream ahead of the
kernel, never inside it.** A block cannot zero its own slot, because a predecessor
may already have read it, and it cannot zero anyone else's for the same reason.
That memset is real traffic and `ckl::ScanPlan::lookback_status_bytes` counts it
in the model rather than leaving it out to make the rung look cheaper.

The look-back window is 32 tiles, one per lane of the first warp, so a whole
window is inspected in one round of loads and one ballot. That is the width the
algorithm was published with and the width a ballot can summarize in one
instruction. The window's combine runs from the highest lane down, because lane
31 holds the earliest tile and the operator is not required to commute.

## Operators, types and what is instantiated

Inclusive and exclusive come out of one templated core. A thread scans its own
items serially from a running value, and whether it writes the running value
before or after the combine is a template parameter; that is the whole of the
difference.

Four operators: `kSum`, `kMax`, `kMin`, and `kLastNonZero`. The last one is the
non commutative associative operator the spec asks for. `a op b` is `b` unless
`b` is zero, in which case it is `a`, so an inclusive scan of it carries the most
recent nonzero element forward and zero is its identity. Associativity is easy to
check and it is the only property the algorithms use.

It earns its place on the first day. `ScanLadder.FunctorOperatorsExactly` with
`kLastNonZero` caught a real defect in the Blelloch downsweep: the right child's
prefix was being computed as `apply(left aggregate, parent prefix)` instead of
`apply(parent prefix, left aggregate)`. Every sum, max and min test passed with
that ordering, because all three commute. Nothing else in the suite would have
found it.

**The reduction ladder carries the three commutative operators only.** Every
reduction rung here combines its block partials in an order the launch decides:
r0 and r2 through an atomic, r3 and r4 through a grid whose size comes from the
occupancy API, r1 through a grid stride loop over the partials. No amount of care
inside a block fixes that, so a non commutative operator has no honest reduction
rung to run on. `ckl::reduce` returns `kNotSupported` and names the reason, and
the operator runs on the whole scan ladder instead, which is where the
associativity-only claim actually lives, because a scan cannot reorder.

**FP32 carries every operator; FP64, int32 and int64 carry sum.** That is what
recomputing the roof per element type needs, and it keeps the instantiation set
at seven pairs per rung instead of sixteen. `ckl::ScanPlan::scan_refusal` names
the reason for anything outside that set. Extending the cross product is a
mechanical change to two macros in `src/scan/scan_device.cuh`.

## Determinism

Floating point addition is not associative, so a reduction gives a different
answer when it combines its partials in a different order, and every fast rung on
this ladder lets the launch decide that order. That is not a defect; it is what a
fast reduction costs.

`ReduceAlgo::kDeterministic` and `ScanAlgo::kDeterministic` pay the cost the other
way round. Both are selected by `kAuto` when `CKL_REDUCE_DETERMINISTIC` is set in
the environment to anything other than `0`, or when
`ScanPlanOptions::deterministic` is passed, and `chosen` reports the rung so a
caller can see it happened. Three things make the answer bit identical.

- The block count is a pure function of the length and nothing else. It never
  asks `cudaOccupancyMaxActiveBlocksPerMultiprocessor`, so the same input gives
  the same grid on a busy device, on an idle one, and on a different card.
- The element to thread mapping follows from that grid, and each thread combines
  its own elements in index order.
- The two levels are combined by an explicit halving tree, so the shape of the
  tree is a property of the block size rather than of the schedule.

The deterministic scan rung is reduce then scan. That rung's tile decomposition
is already a pure function of the length: tile t always covers the same elements,
always reduces them in the same order, and always receives the same prefix. It is
a separate name rather than an alias so a caller asking for determinism gets a
promise rather than an implementation detail that could change. Decoupled
look-back cannot make that promise, because how many predecessors a tile has to
combine depends on which of them happened to finish first.

The cost of determinism is a benchmark row of its own next to r3, r4 and s4. It
is not hidden and it is not defended; the sweep says what it costs at each
length.

**The artifact.** `experiments/results/scan_determinism.txt` holds the output
hashes from ten consecutive runs at `N = 2^24` and `N = 2^28`. It is a
determinism artifact produced at unlocked clocks, not a performance artifact: the
value it records is a property of the arithmetic and has nothing to do with how
fast the machine was running.

## Compensated summation

`ReduceAlgo::kKahan` carries its rounding error alongside every partial and folds
it back in at every combine. It is two term arithmetic rather than plain Kahan:
each partial is a pair, the high word and the running compensation, and merging
two pairs uses a Knuth two sum on the high words and folds both compensations
into the result. Plain Kahan compensates a serial accumulation and has nothing to
say about how two compensated partials should meet, which is exactly what a
parallel reduction spends its time doing.

The hypothesis is that the extra operations are free at large N, where the kernel
is waiting on DRAM and has issue slots to spare, and are not free at small N. The
sweep decides. What is already on the record is the accuracy:
`ScanDeterminism.CompensationBeatsThePlainSumOnDataThatNeedsIt` sums one large
term and `2^20 - 1` small ones and asserts the compensated answer is closer to the
exact one.

Compensation only means anything for addition, so the plan refuses the other
operators on this rung and says so.

## Tolerance and its calibration

The model is

    tol(N) = c * sqrt(N) * FLT_EPSILON

on the residual `max_i |got_i - ref_i| / max_j |x_j|`, which is the error
measured in units of the input scale. For random signed data the partial sums do
a random walk of size `sqrt(N)` times that scale and the rounding error is
machine epsilon times that walk, times a slowly growing factor for the depth of
the reduction tree, so the ratio of the residual to `sqrt(N) * FLT_EPSILON` is
O(1) and c absorbs the depth term.

**Two rungs do not fit that model and are not made to.** `kAtomic` and `kShuffle`
both finish through a single global atomic, so their last combine is a serial
accumulation of however many terms reached that address: all N elements for r0,
one partial per 256 element block for r2. The textbook bound for a recursive sum
grows with the length of that chain and not with its square root, so both get

    tol_serial(L) = c_serial * L * FLT_EPSILON

on the chain length L their rung produces. That the two atomic rungs are less
accurate as well as slower is a result, not an exemption: one accumulator is one
long dependent chain, and a long dependent chain is where floating point error
comes from.

**Method.** `ScanCalibration.DISABLED_ToleranceConstant` in `tests/test_scan.cpp`
sweeps eight lengths from 1023 to `2^24`, eight seeds, and every rung of both
ladders, computes the residual above, and reports the worst ratio of that
residual to the model's unit. It is disabled by default because it is a
calibration and not a check. Run it with

```sh
./ckl_test_scan --gtest_also_run_disabled_tests \
    --gtest_filter=ScanCalibration.DISABLED_ToleranceConstant
```

**Result.** Across four runs on this machine the worst `sqrt(N)` ratio was 5.09
to 6.07, always on one of the two rungs whose own last combine is a device wide
tree (`kCub` and `kLookback` traded places between runs), and the worst chain
ratio was 5.17 to 6.71, always on `kShuffle`. The committed constants are

- `c = 12.0`, in `ckl::scan_tolerance_c`, about twice the worst observed ratio.
- `c_serial = 16.0`, in `tests/test_scan.cpp` and mirrored in `bench_scan.cpp`
  and `bench_all.cpp`, about 2.4 times the worst observed ratio.

The slack is there for seed to seed variation and for the two atomic rungs, whose
ratio moves from run to run because their combine order does. It is not there to
make the tolerance unfailable: a rung that dropped a tile, mixed up an order, or
lost its compensation misses by orders of magnitude, not by a factor of two.

**Adversarial data is checked differently, because the model above says nothing
about cancellation.** The mixed magnitude datasets of `1e30` and `1e-30` are
compared against the double precision reference divided by the running sum of
magnitudes, which is what bounds the rounding error whatever the data does.

## Tests

`tests/test_scan.cpp`, registered as `ckl_test_scan`.

Lengths: 0, 1, 2, 31, 32, 33, 1023, 1024, 1025, 4095, 100000, 1000003 (the non
power of two mid size), 1048573 (prime), and `2^28`. Zero, one and two catch a
launcher that computes a grid of zero blocks or reads one element past the end.
The three around 32 catch a warp primitive that assumed a full warp; the three
around 1024 catch a block scan that assumed a full block; the prime catches a
tile decomposition that assumed the length divided by anything; `2^28` catches the
level recursion running out of levels, which nothing smaller does.

A length of zero writes its output for a reduction, which is the operator's
identity, returns success, and launches no zero sized grid.

Datasets: all equal, all zero, a single nonzero at the LAST index, alternating
signs, random uniform, and the adversarial magnitude mix. The single nonzero at
the last index is in the list because it is the one that catches a dropped final
tile; every other dataset would still look right without it.

Seeds come from `--seed` rather than from the shape, so one flag moves every
dataset in the suite at once.

The `2^28` cases are in a `ScanSlow` suite and registered as their own ctest entry
under the labels `gpu;slow`, with the discovery of the fast tests excluding them,
so `ctest -L gpu -LE slow` is the fast suite and `ctest -L slow` is the large one
and neither measures the other twice. Their datasets repeat a small pattern so the
reference is a closed form rather than a two gigabyte host array.

## Gate R

**Sanitizers.** All four compute-sanitizer tools run clean on the scan suite.
`synccheck` needs `--num-cuda-barriers 65536`, the same flag the tensor GEMM suite
needs, for the same reason: the tool's default budget for tracked barrier objects
is too small for this many kernels in one process.

```sh
compute-sanitizer --tool memcheck   ./ckl_test_scan --gtest_filter=-ScanSlow.*
compute-sanitizer --tool racecheck  --num-cuda-barriers 65536 ./ckl_test_scan --gtest_filter=-ScanSlow.*
compute-sanitizer --tool initcheck  ./ckl_test_scan --gtest_filter=-ScanSlow.*
compute-sanitizer --tool synccheck  --num-cuda-barriers 65536 ./ckl_test_scan --gtest_filter=-ScanSlow.*
```

**The racecheck gate has been shown failing.** Per ground rule 8 a gate counts
only after it has been seen red. Removing the `__syncthreads()` that publishes the
looked-back prefix from the first warp to the rest of the block turns racecheck
red with 120 errors, naming the write at `scan_lookback.cu:128` and the read at
`scan_lookback.cu:186`, and the correctness test fails alongside it. Reverting the
barrier returns the run to zero hazards.

**One negative result, reported rather than worked around.** Downgrading the
release and acquire orderings on the status flag to relaxed does **not** turn
racecheck red: the run stays at zero hazards. That is not evidence that the
ordering is unnecessary; it is what the tool's scope is. `racecheck` is documented
as a shared memory hazard detector and does not model global memory ordering, so
the release and acquire contract on the tile status word is outside what it can
check. The global side of the same contract is covered by `initcheck` instead:
removing the status array memset that precedes the launch turns initcheck red with
753 errors, every one it printed reading `Uninitialized __global__ memory read of
size 4 bytes` on the flag word, and restoring the memset returns it to zero. So
both halves of the
tile status protocol have a gate that has been seen failing, but they are two
different tools and the report should not claim racecheck covers the one it does
not.

**Determinism.** Ten consecutive runs at `N = 2^24` and at `N = 2^28` produce
identical output hashes, for both the deterministic reduction and the
deterministic scan. See `experiments/results/scan_determinism.txt`.

**The allocation gate.** `bench_scan` arms the CUPTI counter from
`benchmarks/support/allocation_gate.hpp` around every timed region and exits
non-zero if it moves. `--gate-red` puts a `cudaMalloc` and a `cudaFree` back
inside the timed lambda so the counter can be seen failing; it reports 640
allocations and exit code 5, and the same run without the flag reports zero and
exit code 0.

**The bandwidth and parity targets are pending.** Gate R asks for the top
reduction rung at 90 percent of the measured roof and the top scan rung at 85
percent, both within 5 percent of CUB, at `2^26`, `2^27` and `2^28`, at locked
clocks, over five independent process repeats with L2 flushed between reps. That
is an owner sweep and it has not been run. Nothing in this document is a locked
clock measurement, and no percent of roof for this family appears in the report
until `experiments/results/sweep.jsonl` carries the rows.

## Benchmarks

`benchmarks/bench_scan.cpp` is the family driver: one length, every rung of both
ladders, the CUB baseline, the allocation counter, and the two paired
measurements the protocol asks for.

- The flush pair. Input plus output below about `2^22` elements is L2 resident on
  this 48 MB part, so every mid size row is measured twice, flushed and
  unflushed, and both are printed. Only the flushed number can carry a bandwidth
  claim.
- The graph pair. At or below `2^16` elements the kernel is shorter than the
  launch, so those lengths are measured stream launched and graph launched with
  the same number of inner launches, and the difference is launch overhead with
  nothing else in it.

The graph pair found something on its first run, which is what it is for. At
`N = 2^16`, twenty inner launches per event pair, L2 flushed, at unlocked clocks
and therefore report only, every rung drops by about a factor of five when the
loop is captured into a graph, except the two that enqueue a small
`cudaMemsetAsync` ahead of their kernel:

| rung | stream, ms per launch | graph, ms per launch |
|---|---|---|
| `vec4` | 0.012402 | 0.002511 |
| `two_pass` | 0.011883 | 0.002436 |
| `shared_tree` | 0.011858 | 0.003043 |
| `baseline_cub_reduce` | 0.011893 | 0.002331 |
| `single_pass` | 0.011240 | 0.011209 |

`single_pass` zeroes its four byte ticket counter with a `cudaMemsetAsync` on the
stream ahead of every launch, and `lookback` does the same for its counter and
its tile status array. Captured into a graph, that memset node appears to cost
about what a stream launch costs, so the rung keeps its launch overhead where
every other rung sheds it. The measurement is at unlocked clocks and is not a
number of record; what it is is a lead for the owner sweep, which will have the
locked clock rows to say whether the effect survives.

The obvious change is to let the last block reset the counter on its way out, the
way the classic threadFenceReduction sample does, which removes the memset
entirely. It is not made here, for two reasons. It trades a robustness property
for a small N artifact: a launch that dies part way through leaves a dirty
counter behind, and the next call then waits for a block that will never arrive.
And the look-back rung cannot take that change at all, because Section 12.3 fixes
the pre-launch zeroing as one of its three non optional constraints, so the
family would end up with two different answers to the same question. The effect
is confined to lengths below about `2^16`, where the protocol already refuses to
make a bandwidth claim.

`benchmarks/bench_all.cpp` carries the `reduce` and `scan` families for the sweep,
with `baseline_cub_reduce` and `baseline_cub_scan` as the baseline variants
`sweep.py` measures once per length and joins onto every other row. The element
count goes in the `m` position; a non zero `k` selects the exclusive scan.

The oracle for both families is a double precision host reference rather than a
vendor call, on purpose. CUB is the baseline of record here, and using the
baseline as the oracle would make a rung that agreed with CUB by being CUB
indistinguishable from a rung that was right.

`benchmarks/sweep.py` sweeps seven lengths: `2^12` and `2^16` for the launch
isolation pair, `2^20` and `2^22` for the flush pair, and `2^26`, `2^27` and
`2^28` for the bandwidth rows Gate R is stated at. Only the inclusive scan is
swept; exclusive is the same kernel with one template parameter flipped, and a row
for it would double the matrix without answering a new question.

See [`benchmarking.md`](benchmarking.md) for the measurement protocol every one of
those rows is produced under.

## What is not built

- **Segmented scan.** The stretch rung with head flags is not implemented. It is
  the rung that would feed the sparse family, because a segmented reduction is
  the CSR SpMV back end and the row offsets are the head flags: a segmented
  inclusive scan over the values with a flag at each `row_ptr[i]` computes every
  row's dot product in one pass, which is what the merge path variant in
  `src/sparse/spmv_merge.cu` does by hand. Building it means a second operator
  form, `(flag, value)` pairs with a combine that resets on a flag, and that pair
  has to travel through the block scan, the level recursion and the look-back
  status word. The plumbing is there; the pair type is not.
- **The operator by element type cross product.** FP32 carries four operators and
  the other three types carry sum. Widening it is a change to two macros.
- **An ncu round.** The bank conflict counter page for s1 with and without the
  `i + i/32` padding, and the `dram__bytes.sum` check against the declared model
  bytes, are both owner work at locked clocks and both are pending.
- **A committed sweep.** `ScanPlan::query_reduce` and `ScanPlan::query_scan` pick
  on the traffic model rather than on a measurement, and they say so: the
  reduction picks the single pass rung because it is the grid stride loop with one
  launch instead of two, and the scan picks decoupled look-back above one tile
  because it moves half the bytes of the alternative. Neither is tuned.
