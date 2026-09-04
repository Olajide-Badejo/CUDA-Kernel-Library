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

    tol(N, R) = (c_sqrt * sqrt(N) + c_mag * R) * FLT_EPSILON

on the residual `max_i |got_i - ref_i| / max_j |x_j|`, which is the error
measured in units of the input scale, with

    R = sum_j |x_j| / max_j |x_j|

measured from the data the kernel was given rather than assumed from its length.
`ckl::scan_tolerance(n, R)` is the whole model and both the suite and the sweep
verifier call it, so a benchmark cannot pass a residual the tests would fail.

**The first term is the random walk.** For random signed data the partial sums do
a walk of size `sqrt(N)` times the input scale, and a tree combine's rounding is
machine epsilon times that walk, times a slowly growing factor for the depth, so
the ratio of the residual to `sqrt(N) * FLT_EPSILON` is O(1) and `c_sqrt` absorbs
the depth term. A reduction is only ever this term, and the one argument
`ckl::scan_tolerance(n)` is that case: its combine is a tree of logarithmic depth
with no cross tile prefix chain in it.

**The second term is the accumulated magnitude, and 1.1.0 is where it was
added.** `kLookback` and `kCub` are single pass scans: tile `t` gets its prefix
by combining with the inclusive prefix of the tiles before it, so element `i`
sits at the end of a chain whose length grows with `i` and not with its
logarithm. The rounding along a chain is bounded by the magnitude the chain
accumulates, `sum_j |x_j|`, and not by the square root of its length, which is
why the residual of those two rungs grows about linearly in N while the tree
rungs' grows like `sqrt(N)`. Below about `2^23` elements the walk term swamps it;
above that it takes over.

### How the sqrt only model was found to be wrong

The first locked clock campaign is what found it. Five configurations wrote no
row, three of them scans: `lookback` at `2^28` and `2^27` and `baseline_cub_scan`
at `2^28`, all exiting 9 from the bench self verify. Run by hand at the locked
clock, with the old `tol(N) = 12 * sqrt(N) * FLT_EPSILON`:

| configuration | residual | old tolerance | worst index |
| --- | --- | --- | --- |
| `scan lookback fp32 2^28` | 5.273438e-02 | 2.343750e-02 | 237041807 |
| `scan baseline_cub_scan fp32 2^28` | 4.101562e-02 | 2.343750e-02 | 256237453 |
| `scan lookback fp32 2^27` | 9.8e-03 to 3.6e-02 over six runs | 1.657282e-02 | near the end |

Three things decided that this was the model and not a defect in a rung of mine.

1. **CUB fails the same gate at the same length.** `baseline_cub_scan` is NVIDIA's
   decoupled look-back, not mine, and it misses by the same factor. A tolerance
   that the vendor implementation of the same algorithm cannot meet is a
   tolerance, not a bug report.
2. **The shape of the growth matches the chain and not the walk.** Worst residual
   for `kLookback` over sizes, one seed: `4.3e-04` at `2^22`, `1.5e-03` at `2^23`,
   `2.0e-03` at `2^24`, `3.3e-03` at `2^25`, `2.7e-03` at `2^26`, `2.3e-02` at
   `2^27`, `6.3e-02` at `2^28`. The `sqrt(N)` model predicts a factor of 1.41 per
   doubling and the measurement gives about 2. Divided by `sum_j |x_j|` instead,
   the same numbers are flat to within a factor of four across the whole range,
   which is the definition of the right denominator.
3. **The tree rungs never crossed the line.** `kDeterministic` at `2^28`
   residual `9.8e-03` against the same `2.3e-02`, and `kHillisSteele`,
   `kBlelloch` and `kThreeKernel` sit lower still. Only the two rungs that carry
   a tile to tile chain need the second term, which is exactly what the
   derivation says.

The calibration is why nothing caught it earlier: it stopped at `2^24`, one and
two doublings short of where the second term takes over, and the ladder in
`ScanLadder` stops at 1048573. The campaign swept `2^27` and `2^28` and nothing
in the suite had ever run a float prefix sum there.
`ScanSlow.FloatScanAtTheSweptLengths` now does, at both lengths and both
directions, against the double reference and the committed model, so the sweep
cannot reach a length the suite has not.

`2^27` is worth a second look because it is what a tolerance with no headroom
looks like. Six consecutive runs of `scan lookback fp32 2^27` gave residuals of
9.765625e-03, 1.171875e-02, 3.613281e-02, 2.050781e-02, 2.050781e-02 and
2.246094e-02 against an old tolerance of 1.657282e-02: four red, two green, from
one binary on one clock. Decoupled look-back's combine order depends on tile
scheduling, so its residual is a draw from a distribution, and the sweep drew red.

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
sweeps twelve lengths from 1023 to `2^28`, eight seeds below `2^24` and three
above, both scan directions, five datasets below `2^22` and the two signed ones
above, and every rung of both ladders. For each row it computes the residual
above and reports the ratio the corresponding constant is calibrated against:
`sqrt(N)` ratio for the reduction rungs, chain ratio for the two atomic ones, and
for the scan rungs the value of `c_mag` the residual implies once the committed
`sqrt(N)` term is taken out. It is disabled by default because it is a
calibration and not a check. Run it with

```sh
./ckl_test_scan --gtest_also_run_disabled_tests \
    --gtest_filter=ScanCalibration.DISABLED_ToleranceConstant
```

**Result**, one run of the case above on this machine, 183 seconds:

    worst sqrt(n) ratio   9.459891e+00 at reduce kSharedTree random n=268435456 seed=1
    worst chain ratio     6.705521e+00 at reduce kShuffle random n=1024 seed=8
    worst implied c_mag   3.628572e-03 at scan kLookback random n=268435456 seed=1 exclusive
    worst scan headroom   4.442656e-01 of the committed tolerance at the same row

The committed constants are

- `c_sqrt = 12.0`, in `ckl::scan_tolerance_c`, 1.27 times the worst observed
  ratio.
- `c_mag = 0.01`, in `ckl::scan_magnitude_c`, 2.76 times the worst observed
  implied value. An independent six seed probe over `2^25` to `2^28` outside the
  suite put the worst at `3.70e-03`, so the ratio holds across runs.
- `c_serial = 16.0`, in `tests/test_scan.cpp` and mirrored in `bench_scan.cpp`
  and `bench_all.cpp`, 2.4 times the worst observed ratio.

The slack is there for seed to seed variation and for the rungs whose ratio moves
from run to run because their combine order does. It is not there to make the
tolerance unfailable: the worst scan row in the calibration sits at 44 percent of
its tolerance, so a rung that dropped a tile, mixed up an order or lost its
compensation misses by orders of magnitude and the gate says so.

`c_sqrt` is the one to watch. It was calibrated when the sweep stopped at `2^24`
and its worst ratio there was around 6; at `2^28` the reduction rungs reach 9.46,
which is 79 percent of the committed 12. The reduction model is sound, since a
tree's error really does grow like `sqrt(N)`, but the constant has about a
quarter of its headroom left and a ladder that grew past `2^28` should
recalibrate it rather than assume it.

**Adversarial data goes through the same model, which is the point of measuring R
rather than assuming it.** The mixed magnitude datasets of `1e30` and `1e-30`
cancel down to something far smaller than the terms they were built from, so
`sum_j |x_j| / max_j |x_j|` is large for them and the bound they earn is large
with it. Before 1.1.0 they had a measure and a model of their own, compared
against the double reference divided by the running sum of magnitudes; that is
now what the second term says directly, and there is one model for every dataset.

**Where the model does not apply, and it is left that way on purpose.** Data that
is all one sign stops being about rounding dispersion and starts being about
absorption. At `0.75` per element the running sum passes `2^24 * 0.75` somewhere
above `2^22`, after which adding `0.75` to it either does nothing or moves a full
ulp, every rounding goes the same way instead of cancelling, and the residual is
enormous: `kCub` on all equal data at `2^28` implies a `c_mag` of about 89, four
orders above the calibrated value. That is fp32 telling the truth about a serial sum of
a quarter of a billion equal terms, not a defect, and no constant should be
stretched to cover it. The calibration therefore runs the structured datasets
only below `2^22`, where they are exact, and the sweep uses signed random data at
every length, so the domain of the model and the domain of the campaign are the
same.

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
and neither measures the other twice. Three of the four use a repeating pattern
or an integer input so the reference is a closed form rather than a two gigabyte
host array. The fourth, `FloatScanAtTheSweptLengths`, pays for the host array,
because a float prefix sum of random data at the lengths the campaign sweeps is
exactly the thing no closed form covers and exactly the thing that went unchecked
until the first campaign found the tolerance model outgrown there.

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

**The bandwidth targets are met and the CUB parity clause is not.** Gate R asks
for the top reduction rung at 90 percent of the measured roof and the top scan rung
at 85 percent, both within 5 percent of CUB, at `2^26`, `2^27` and `2^28`, at
locked clocks, over five independent process repeats with L2 flushed between reps.
The sweep ran at a locked 2497 MHz and the rows are in
`experiments/results/summary.csv`. Computed with the byte counts the gate names,
`N*4` for reduction and `2*N*4` for scan:

| clause | 2^26 | 2^27 | 2^28 | verdict |
|---|---|---|---|---|
| top reduction rung at least 521 GB/s | 596.46, `vec4` | 608.56, `single_pass` | 614.80, `single_pass` | green |
| top scan rung at least 492 GB/s | 543.00, `lookback` | 546.09, `lookback` | 550.07, `lookback` | green |
| top reduction rung within 5 percent of CUB | 101.37 percent | 100.47 percent | 100.19 percent | green |
| top scan rung within 5 percent of CUB | 94.15 percent | 94.52 percent | 95.49 percent | **red at 2^26 and 2^27** |

The scan miss is 0.85 points at 2^26 and 0.48 at 2^27, and diagnostic round 17
names its cause from the committed pages under
`experiments/results/ncu/round17/`. It is not traffic: `lookback` moves 518.20 MB
against 516.21 MB for CUB at 2^26, 0.4 percent more, and both sit within 4 percent
of their declared models. It is tile size. `lookback_kernel` carries 2,048 elements
per tile over 32,768 blocks at 34 registers; `DeviceScanKernel` carries 7,904 over
8,491 blocks at 48 registers, so CUB pays the decoupled look back handshake 3.9
times less often. The counters follow: 5,368,287 shared load wavefronts against
5,788,563, 290,681 bank conflicts against 480,147, and DRAM Throughput of 86.98
percent against 82.04 percent. Raising items per thread on the look back rung is
the change to try, and it is tuning rather than a redesign. The target is not being
lowered, and the report states the miss with this cause beside it.

The same round measured the `s1` padding question. There is no pre padding page in
this repository, so there is no before and after to show; the committed padded
state is
`tile_scan_kernel<..., BlellochBlockScan, 0>` at 2^26 with 138,405 conflicts
against 17,177,765 shared load wavefronts, 0.81 percent. The page also shows why
the rung sits at 37.53 percent of CUB: 17.2 million shared load wavefronts against
the 5.8 million of the look back rung for the same N, and 1067.46 MB of DRAM
traffic against 518.20 MB.

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
