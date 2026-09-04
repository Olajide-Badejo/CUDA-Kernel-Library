# FFT and convolution

This is the family where the memory roof, not the arithmetic roof, decides the
answer at every size in scope. It carries the derivations the report chapter
cites, so the chapter can point at a document instead of restating them.

No measured number appears here. Timings live in
`experiments/results/sweep.jsonl` and in the summary generated from it, and the
round trip error table is printed by `ckl_test_fft` rather than copied into
prose. Two facts that are not measurements do appear: the cuFFT callback probe
result with the toolkit version it was run against, which Gate X asks for by
name, and the shared memory arithmetic that fixes the rungs.

## Scope

One dimensional complex to complex FP32, powers of two from 2^10 to 2^24,
forward and inverse both first class. Real to complex is rung 5 and 2D row
column is rung 6. Everything runs through `ckl::FftPlan`, `ckl::Fft2dPlan` or
`ckl::ConvPlan`, declared in `include/ckl/fft.hpp`.

| Rung | Where | What it adds |
|---|---|---|
| 1 | `fft_radix2.cu` | Stockham autosort, one kernel launch per stage, global memory |
| 2 | `fft_shared.cu` | Single block, the whole transform resident in shared memory |
| 3 | `fft_radix4.cu` | Radix 4 and mixed radix 8 butterflies; fewer stages, more registers |
| 4 | `fft_shared.cu` and `fft_transpose.cu` | Four step form for n above the shared bound |
| 5 | `fft_radix2.cu` | Real to complex: 2n reals packed as n complex plus an untangle pass |
| 6 | `fft_transpose.cu` | 2D row column with the transpose study |

## Stockham, not Cooley-Tukey plus a bit reversal pass

Stockham autosort writes each stage to a second buffer with the digit
permutation folded into the output address, so every stage reads and writes
contiguous runs. At stride p a thread owns one butterfly: it reads elements j
and j + n/2 of the source, twists the second by exp(-2 pi i k / 2p) with
k = j mod p, and writes the sum and the difference to 2(j - k) + k and
2(j - k) + k + p of the destination. The radix 4 and radix 8 stages are the same
statement with R in place of 2. `src/fft/fft_device.cuh` holds one
implementation of each, called from two translation units, because a radix 8 DFT
written twice is a radix 8 DFT that will disagree with itself.

The in place alternative needs a separate bit reversal pass whose access pattern
is a scatter across the whole array, one sector per element at large n. That pass
alone costs a full memory round trip on a family that is already memory bound,
which is a whole extra pass out of the five the four step form needs. The price
of avoiding it is the second buffer, and the budget below pays it. Cooley-Tukey
with a bit reversal pass is not implemented here and is not planned.

## Shared memory budget, derived

A complex FP32 point is 8 bytes and Stockham needs two ping pong buffers, so a
resident transform of n points costs 16n bytes of shared memory.

- n = 2^12 costs 65536 bytes and fits inside the 101376 byte per block opt in
  limit of this part.
- n = 2^13 costs 131072 bytes and does not.

Holding one radix stage in registers and reusing a single shared buffer would
bring 2^13 back inside at 65536 bytes. **That trick is not implemented in
1.1.0.** `ckl::FftPlan::shared_resident_max()` is 4096 and
`ckl::FftPlan::refusal` on a larger plan returns the arithmetic above rather than
a bare status. Above 2^12 the four step form takes over.

Either way a 64 KB block is one block per SM against the 100 KB per SM limit, so
a small size benchmark has to batch transforms until all 48 SMs have work.
`bench_fft` defaults to two blocks per SM at or below the shared bound, and every
benchmark row carries its batch count, because a single unbatched 2^12 transform
measures launch overhead and not the kernel.

## The four step form

For n above the shared bound the plan factors n = n1 * n2 with both factors at or
below 2^12 and runs

1. transpose, reading the input as an n2 by n1 matrix,
2. batched size n2 transforms, one per row, with the cross twiddle
   exp(-2 pi i n1 k2 / n) folded into the store epilogue,
3. transpose,
4. batched size n1 transforms, one per row,
5. transpose, which is what puts the output at k1 n2 + k2 where the caller wants
   it.

Five passes over the array, each one read and one write, so 5 times 16n bytes.
Against log2(n) passes for rung 1 that is 5 against 24 at 2^24. The cross twiddle
costs no pass of its own because it rides on the epilogue of pass 2, and a fused
convolution's pointwise multiply costs no pass of its own because it rides on the
epilogue of pass 5.

The factorization splits log2(n) as evenly as it can, larger half first. Both
factors then sit at or below 2^12 for every n up to 2^24, which is the condition
the shared resident sub transform needs, and an even split gives both batched
passes the most blocks to fill 48 SMs with. `FftPlanOptions::four_step_n1`
overrides it, and the plan validates that the override is a power of two that
divides n and leaves both factors inside the bound.

## Twiddles

Computed in double precision on the host and rounded to FP32 once, when the plan
is built, outside every timed region. **Recurrence generated twiddles are banned
outright**: their error grows with the stage index, which is exactly the failure
the tolerance model would then have to absorb.

The tables are stored factored. A direct table of exp(-2 pi i t / n) for every t
costs 8n bytes, which is 128 MB at 2^24 and comes from DRAM on the late stages,
where it would be a quarter of the traffic the model declares. Writing
t = q 2^b + r and holding one table of the coarse factors and one of the fine
ones costs 8 (n / 2^b + 2^b) bytes instead, which is under 64 KB at every size in
scope and stays in cache. The price is one complex multiply and one extra
rounding per twiddle, far inside the bound below. `ckl::FftTwiddles` is that
pair and `ckl::detail::twiddle` is the lookup.

`__sincosf` is a labelled rung and never the default.
`FftPlanOptions::fast_twiddles` fills both tables on the device with it, and
`FftPlan::fast_twiddles()` reports that it did, so every benchmark row that used
it says so. Its argument reduction gives roughly 2^-21 relative accuracy, about
sixteen times worse than rounding a double precision sine, and
`ckl_test_fft --gtest_filter=FftTwiddleRung.*` prints the two round trip errors
side by side so the cost of the rung is on the record rather than asserted.

## The transpose study

A 2D transform is a row pass, a transpose, a row pass and a transpose. The row
passes are shared resident and cost one read and one write each; so do the
transposes. Half the traffic of a 2D transform is therefore transposition, and a
transpose is pure bandwidth with no arithmetic to hide behind. `fft_transpose.cu`
carries three variants, measured on their own before they are measured inside the
2D driver:

- `kNaive` is the control. One thread per element, indexed by the output, so the
  writes are contiguous and the reads walk down a column one sector at a time.
  Nothing goes through shared memory. This is the variant whose sectors per
  request the ncu round watches move.
- `kTiledPadded` stages a 32 by 32 tile through shared memory so both the read
  and the write are contiguous, with one element of padding on the row so the
  column read out of shared does not land 32 ways on one bank.
- `kStridedShared` keeps the shared staging and drops both of those: the load
  walks the column and the tile is unpadded. It is the same kernel with the
  strided access moved to the load side, so the difference between it and
  `kTiledPadded` separates what coalescing the load and padding the row are worth
  from what staging through shared memory is worth at all.

Inside the 2D driver `kStridedShared` means something related and stronger. The
column pass reads its column straight into the shared transform buffer, through
`ckl::detail::fft_shared_strided`, so there is no transpose at all and the 2D
transform runs in two passes rather than four. `Fft2dPlan::passes()` returns 2
for that variant and 4 for the other two, and
`Fft2dPlan::transpose_model_bytes()` returns zero for it. `bench_fft` prints the
transpose share of total 2D time per variant, measured by timing the same
transpose on its own at the same shape and doubling it, because the driver runs
two.

## Convolution

Linear convolution of a length N signal with a length M filter, output length
N + M - 1. The FFT path zero pads both to L, the next power of two at or above
N + M - 1, transforms, multiplies pointwise and transforms back, scaling by 1/L.
The filter spectrum is computed once when `ckl::ConvPlan` is built and cached, so
no timed region ever contains a transform of the filter.

Four hand written variants:

- `kFftSeparate` runs forward, a standalone pointwise multiply kernel, then
  inverse. That kernel reads two spectra and writes one, so it moves 24 bytes per
  point.
- `kFftFused` folds the filter spectrum multiply into the store epilogue of the
  last forward pass and the 1/L into the store epilogue of the last inverse pass,
  the same epilogue fusion methodology the GEMM bias and ReLU study uses. Neither
  epilogue costs a byte of extra traffic: the value is already in a register on
  its way out, and the filter spectrum read replaces a 24L pass with an 8L one.
  `ConvPlan::model_bytes` differs between the two variants by exactly 16L, which
  `ckl_test_conv` asserts. Whether that shows up as time is a hypothesis and the
  sweep is what settles it.
- `kDirectShared` is the tiled time domain kernel: a 256 output tile per block,
  the filter walked in chunks of 256 taps with the taps and the signal window for
  that chunk staged through shared memory. The obvious shape, a signal tile plus
  a halo of M - 1 samples, does not survive the top of the crossover sweep:
  M = 16384 is a 64 KB halo before the tile is counted. Chunking holds the shared
  footprint at about 3 KB whatever M is.
- `kDirectConstant` is the same tiling with the taps read from constant memory. A
  constant read broadcasts to a whole warp in one cycle when every lane reads the
  same address, which is what the inner loop does. The bank is 64 KB per module
  and the spec budgets 16 KB of it, so this rung takes filters up to 4096 taps
  and refuses longer ones with the count in the message.

The constant bank is one per module and not one per plan, so a plan binds its
taps once at construction and records that it owns them. A call from a plan that
does not own the bank rebinds first, which is a 16 KB device to device copy that
only happens when two plans alternate; the steady state costs nothing.
`conv_direct_constant` called directly with an owner that never bound throws
rather than convolving with somebody else's filter.

### The crossover

The headline figure of this family. The FFT path's cost barely moves with M; the
direct path is 2 N M FLOP and does not stop growing. `bench_conv --sweep` runs M
in powers of two from 8 to 16384 at a fixed N, and `benchmarks/sweep.py` runs the
same grid at N = 2^20 and N = 2^22 so the crossing is inside the range whichever
way it falls. `scripts/gen_report_assets.py` builds the chart from the committed
sweep rows, reads the crossing out of the data by log interpolation between the
two filter lengths that bracket the sign change, and prints it in the caption.
Nothing about that chart is placed by hand, and when the summary holds no
convolution rows the script removes the figure and says the chart is pending
rather than rebuilding the report around a stale picture.

`ConvPlan::query` picks between the two families by dividing the two models by
the two roofs of this part: the direct model by the FP32 roof and the FFT model
by the measured DRAM roof. That is arithmetic and not a measurement, the crossing
it predicts is exactly what the sweep exists to measure, and the rule is
provisional until the sweep runs.

## The cuFFT baseline

The plan is the whole risk in this family, so it is handled the way the
measurement protocol handles a cuBLAS handle.

- `cufftPlanMany` is created once per size and batch when the `ckl::FftPlan` is
  built, outside every timed region, and reused across every rep and every
  variant compared against it. Plan creation is not a benchmark row.
- `cufftSetStream` is called on the benchmark stream, and the plan remembers the
  last stream it was set to so a repeated call on the same stream is a compare
  and not a driver call.
- **Workspace policy: the default.** `cufftSetAutoAllocation` is never called and
  neither is `cufftSetWorkArea`. cuFFT allocates and owns its work area, which is
  what a caller would actually get. Only that path is measured, and no row mixes
  it with a hand managed one.
- **Scaling: neither direction is scaled, on either side.** cuFFT does not scale
  its inverse and neither does `ckl::fft`, so the two compute the same work.
  `ckl::fft_scale` is a separate pass a caller applies when it wants the round
  trip identity, and `ckl::conv` folds the 1/L into an epilogue where it costs
  nothing. Folding the scale into `ckl::fft` would have made the inverse
  incomparable with the baseline.
- Forward and inverse are timed separately; nothing averages the two.
- No allocation, no plan creation and no host copy sits inside any timed region,
  for either side. `bench_fft` and `bench_conv` arm the same CUPTI allocation
  counter `bench_spmv` uses across every timed lambda and exit non zero if it
  moves, and both carry `--gate-red` so the gate can be seen failing.

### The callback probe

Phase 0 of this family, per the build spec: spend thirty minutes building one
`cufftXtSetCallback` example against the installed toolkit, and record the
outcome and the toolkit version here.

**Outcome: it links and it runs.** The probe was built against CUDA 13.3.73 with
cuFFT 12.3.0, host compiler g++-14.3.0, on an RTX 5070 (sm_120, driver 610.62).
The build line was separate device compilation (`-rdc=true`) with static linking
against `libcufft_static` plus `libculibos`. The probe is numeric and not a
status code check: it transforms a constant one signal of length 1024 with a
store callback that scales by 0.5, and a plan that really ran the callback
reports n/2 at bin zero while a plan that silently ignored it reports n. It
reported n/2.

Because the probe passed, the vendor fused convolution baseline is the callback
path and it is built:
`benchmarks/bench_conv_callback.cu`, target `bench_conv_callback`. That target
links `libcufft_static` and nothing from libckl, because libckl links the shared
cuFFT and one process must not hold two copies. It reports two vendor rows,
`baseline_cufft_separate` and `baseline_cufft_callback`, and verifies that the
two agree before timing either, which is also the proof that the callback ran:
without it the fused answer would be neither multiplied nor scaled.

The `kCufft` variant inside `ckl::ConvPlan` is cuFFT plus a separate pointwise
kernel, and `bench_conv` says so in its own output. A table comparing against a
vendor fused convolution means `baseline_cufft_callback`, and no table in this
project implies a vendor fused path that was never built.

## Accuracy

Three checks, all in `tests/test_fft.cpp` and `tests/test_conv.cpp`:

- Round trip, forward then inverse then the 1/n, at every size from 2^10 to 2^24,
  on every rung that size supports. The test prints the whole table with the
  bound and the margin beside each row.
- Forward and inverse against a long double CPU DFT at n up to 2^12, where the
  O(n^2) reference is affordable.
- Convolution against direct time domain computation in double, at several signal
  lengths up to 2^16 and filter lengths from 8 to 16384.

The second check is not redundant, and showing the first one red is what proved
it. Swapping two outputs of the radix 4 butterfly in `fft_device.cuh` leaves the
round trip green at every size whose stage count is even: the forward and the
inverse make the same mistake and it cancels. The same break fails the long
double DFT check at 1024, 2048 and 4096 in both directions, and fails the cuFFT
agreement check, because neither of those runs the wrong butterfly twice. A round
trip on its own tests that a transform is invertible, not that it is a DFT.

### The tolerance model

Error grows with the number of stages, so the bound has to as well:

```
rms_rel(n) <= 8 * log2(n) * FLT_EPSILON
```

`ckl::fft_tolerance` is the one implementation and every assertion goes through
it. A flat tolerance would either pass a broken 2^24 or fail a correct one, which
is the argument `ckl::tol(k)` already makes for GEMM.

One correction to the build spec. Section 12.2 states the formula above and then
says it is 1.1e-5 at 2^24. With `FLT_EPSILON` at 1.1920929e-7 the formula gives
8 * 24 * 1.1920929e-7 = 2.29e-5; the quoted 1.1e-5 is the same formula evaluated
at the unit roundoff `FLT_EPSILON / 2`. This library implements the formula as
written and `ckl_test_fft` asserts the evaluated values, so the two cannot drift
apart again. Every rung sits far inside either reading, which is why the table
the test prints carries the margin column: a regression that halved the accuracy
would still pass the bound and would be obvious in the margin.

## Traffic model and the roofline

The transform counts 5 n log2(n) FLOP by the standard model.
`FftPlan::flop_model` returns it and the tables carry it for continuity with the
literature. It is not the metric. Arithmetic intensity for one ideal pass is
5 n log2(n) / 16n, which is 6.25 FLOP per byte at 2^20 against an FP32 ridge
point of 30.7e12 / 579e9 = 53 FLOP per byte. Every size in scope sits at least
eight times to the left of the ridge.

The honest metric is effective GB/s: declared model bytes over measured time,
against the measured 579 GB/s DRAM roof. The declared model is
`FftPlan::model_bytes`, which is passes times 16n times batch:

| Variant | Passes |
|---|---|
| `kRadix2Global` | log2(n) |
| `kRadix4Global` | ceil(log2(n) / 2) counting the leading radix 2 stage when log2(n) is odd |
| `kRadix8Global` | ceil(log2(n) / 3) counting the leading narrower stage |
| `kSharedResident` | 1 |
| `kFourStep` | 5 |
| `kCufft` | not declared; the vendor's internal pass count is not ours to state |

The twiddle traffic is declared separately, as a band rather than a single
number, by `FftPlan::twiddle_bytes_low` and `twiddle_bytes_high`. The low end is
the whole factored table read once, which is what a perfectly cached table would
cost; the high end is two loads per lookup with no reuse at all. Because the
tables are factored the truth sits at the low end, and the band is here so the
ncu round can say so with a measurement instead of an assertion.

`ConvPlan::model_bytes` is derived per variant in the source and covers the pack,
the two transforms, the pointwise pass where it is a separate pass, and the
extract; a direct variant counts its signal windows, its tap reads and its
output.

## Provisional decisions

Three, and none of them is a measurement. All three are named here because a
provisional rule that does not say it is provisional is a claim.

1. `FftPlan::query` runs the shared resident kernel at or below 2^12 and the four
   step form above it. That is the shared memory budget and nothing else. The
   radix 4 and radix 8 rungs are never chosen automatically, because whether a
   wider butterfly wins depends on the occupancy its register count leaves, and
   there is no committed sweep yet.
2. The four step factorization splits log2(n) evenly, larger half first. Both
   factors then fit the shared bound for every n up to 2^24 and both batched
   passes get the most blocks.
3. `ConvPlan::query` compares the two traffic models against the two roofs of
   this part. It predicts a crossover; the crossover is what the sweep measures.

## Diagnostic rounds

Same protocol as GEMM: one change per round, one committed ncu text page per
round under `experiments/results/ncu/`, one dated `DIAGNOSTIC_LOG.md` entry,
locked clocks, no wall clock only reverts. The expected limiters are named in
advance so a round is not spent finding them.

- **Shared bank conflicts in the butterfly exchange.** A stride 2^s access into a
  `float2` array conflicts up to 32 ways at the late stages. Candidates in order:
  split real and imaginary into two float arrays, pad the stride, XOR swizzle.
  Capture `l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld` on every round in
  this family, not only the round that targets it.
- **Global coalescing in the multi pass and transpose kernels.** Capture sectors
  per request for global loads and stores. `kNaive` is the control that shows the
  metric moving.
- **Occupancy from register heavy butterflies.** A radix 8 butterfly holds eight
  complex values plus seven twiddles live, so registers rather than shared memory
  may be the block limiter for rung 3. Capture the occupancy limiter reason and
  put radix 2, 4 and 8 in one table with registers per thread beside throughput.

## Gate X status

| Check | State |
|---|---|
| Round trip inside the derived bound at every size 2^10 to 2^24, both the shared and the four step paths | green, printed by `ckl_test_fft` |
| Best hand rung at 55 percent of the DRAM roof and 70 percent of cuFFT at 2^20 to 2^24 | roof leg green on the committed rows; **cuFFT leg red**, `radix8` reaches 60.5 percent of cuFFT at 2^20 and 38.3 percent at 2^24 (round 16) |
| At 2^10 to 2^13, batched to fill 48 SMs, 85 percent of cuFFT | pending the owner sweep at locked clocks |
| Measured `dram__bytes.sum` within 15 percent of the declared model | **red at 2^20, green at 2^24** (round 16): 64.08 MB measured against a declared 117,440,512 is 45.4 percent low, and 1977.03 MB against 2,147,483,648 is 3.5 percent low |
| The 2D driver reports the transpose share per size, tiled transpose at 70 percent of roof | share is reported by `bench_fft`; the roof percentage is pending the owner sweep |
| The crossover chart generated from committed rows with the crossing read from the data | hook wired in `gen_report_assets.py`; pending the committed sweep rows |
| The callback probe result and the toolkit version recorded here | green, above |

**Two things round 16 settled, both worth reading before the table above is used.**

The model bytes failure at 2^20 is not a measurement error and it is not a kernel
defect. The ping pong buffers are 8 MB each at that length, so the whole working
set sits inside the 48 MB L2 and six of the seven stage boundaries never reach
DRAM. The declared model assumes every stage writes through to memory, which is
true from 2^22 upward and false below it. Until the model carries an L2 residency
term, the effective GB/s column for the small and mid lengths is not evidence: the
committed `radix8` row at 2^20 reads 1091.942 GB/s, which is above the measured
579 GB/s DRAM roof, and the excess is bytes that L2 served.

The loss to cuFFT is a traffic loss, measured on both sides at 2^24: the `radix8`
ladder moves 1977.03 MB where cuFFT moves 744.37 MB, a factor of 2.66, and the
committed medians differ by a factor of 2.61. Each `radix8_stage` runs at 83.47
percent DRAM Throughput, so the rung is already at 95 percent of the roof on every
pass it makes. It makes 8 passes where cuFFT makes 3. Nothing inside a stage is
worth tuning until the pass count comes down.

The bank conflicts Section 12.2 predicted in the butterfly exchange are not there.
`shared_kernel` at 2^12 with batch 96 measures 0 conflicts against 319,488 shared
load wavefronts (`experiments/results/ncu/round16/fft_shared_12.txt`). That rung is
limited by occupancy instead: 65.54 KB of dynamic shared memory per block is one
block per SM and 16.65 percent achieved occupancy.

## Running it

```sh
./build/tests/ckl_test_fft                       # the round trip table and the rungs
./build/tests/ckl_test_conv                      # every variant against a double reference
./build/benchmarks/bench_fft 20                  # one length, every rung, report only
./build/benchmarks/bench_conv 20 --sweep         # the crossover rows, report only
./build/benchmarks/bench_conv_callback 20        # the vendor fused baseline
python3 benchmarks/sweep.py --only fft: --only conv:
```
