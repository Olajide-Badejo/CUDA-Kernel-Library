# Benchmarking

This is the measurement protocol the numbers in this repository are produced
under. It exists because the v1 numbers were not produced under one: clocks
roamed from 1042 to 2880 MHz inside a single sweep, the vendor baseline was
re-measured inside every row and drifted 1.9x on identical calls, the sweep ran
naive first and the top kernel last on a hot GPU, throttled rows were printed to
stderr and kept, and the summary mixed commits. Every rule below removes one of
those.

No number appears in this document. The numbers live in
`experiments/results/sweep.jsonl`, which is committed, and in the summary
generated from it.

## The pieces

`benchmarks/support/event_timer.hpp` is the timer. `benchmarks/bench_all.cpp`
measures exactly one configuration and prints exactly one JSON object.
`benchmarks/sweep.py` drives it across the ladder matrix.
`benchmarks/tile_sweep.py` drives it across the tile family's parameter space and
imports its whole protocol from `sweep.py`, so there is one set of rules rather
than two that drift.

One process per configuration is what makes independent repeats possible at all:
a repeat that shares a process shares a warmed cuBLAS handle, a warmed allocator
and a warmed instruction cache with the run before it.

## Timing

CUDA events, on a stream the harness creates rather than the legacy default one,
because graph capture refuses that stream.

The stream is created blocking rather than non blocking, on purpose. Host
uploads go through plain `cudaMemcpy`, which for pageable memory returns once the
bytes reach the driver's staging buffer and leaves the transfer to finish on the
legacy default stream. A non blocking stream does not wait for that, so a kernel
enqueued right after an upload can read the operand's previous contents. That is
not a hypothetical: it made roughly one verification in twenty fail with a
handful of elements computed from stale data.

Warmups first, then timed reps. Adaptive mode, which is the default, runs to
whichever of 1000 reps or two seconds arrives first, so a 40 microsecond kernel
is not summarized from twenty samples and a 10 millisecond one does not spend a
minute. `--fixed-reps` runs exactly `--reps`.

Every sample is kept. `TimingStats` carries the whole vector, `bench_all` prints
it as the `samples` array, and the sweep keeps the array of the representative
process alongside every process median. A reader who disagrees with the median
can recompute whatever statistic they prefer instead of trusting one number
someone else reduced.

Setup never sits inside a timed region. A kernel that consumes its input, which
is TRSM, gets a device to device restore enqueued on the same stream ahead of the
start event, so stream ordering guarantees it has finished before the timed
region opens. v1 put a host to device copy inside the timed lambda and charged a
PCIe transfer to both the kernel and its baseline.

## Self verification

Nothing is timed before it is verified. Before the loop opens, `bench_all` runs
one launch of the kernel and one of the vendor oracle into separate buffers and
compares them. On a mismatch it prints a JSON error object naming the stage, the
residual, the tolerance and the index of the worst element, and exits non-zero.
The sweep treats that as a failed configuration rather than writing a row.

A refusal is not a mismatch. When the plan returns `kNotSupported`, `bench_all`
exits 7 at stage `dispatch` and hands back the plan's own reason. The sweep
writes a row for it with `status=skipped_not_supported`, the reason in
`status_reason` and every timing column empty, prints it as a skip, and does not
fail. That row goes into `summary.csv` like any other, so a reader who goes
looking for a number that is not there finds the sentence saying why instead of a
gap. `refresh_summary` puts a skip through the same one row per key check as a
measurement, because a refusal and a measurement of the same configuration are
still two answers to one question. Every other non zero exit is still a failure.

The comparison is the worst elementwise difference divided by the magnitude the
rounding error is actually bounded by, which for a dot product is the sum of
`|a| |b|` over the contraction. That sum is itself a GEMM, so it is computed on
the device from `|A|` and `|B|` rather than in an m by n by k host loop, and the
result is directly comparable with `ckl::tol(k)`, the same shape derived
tolerance the correctness suite uses. Dividing by `|c|` instead would make the
bound depend on how much the dot product happened to cancel.

A triangular solve is checked by its backward residual instead: multiply the
answer back through L and see whether B comes out, scaled by `|L| |X|`.
Comparing two solves elementwise would need `|L inverse|`, and the inverse of
`|L|` is not that.

This is not a substitute for the test suite. It is the thing that stops a kernel
that silently fell back, returned early, or wrote nothing from posting a number,
which is the failure mode a benchmark cannot otherwise see. It has already
caught one: the first `CUBLAS_GEMM_AUTOTUNE` call for a shape is the tuning run
itself and leaves something other than the answer in C, so the autotuned baseline
is primed once before it is verified or timed.

`--verify-perturb X` scales the oracle before the comparison. It exists so the
gate can be shown red on demand and has no other use; any value other than 1.0
is stamped into the row.

## Dispatch assertions

A GEMM variant name maps to a `ckl::Algo` through one table, the call goes
through `ckl::gemm` on a Context created once, and `chosen` is asserted equal to
the algorithm the row claims. A benchmark that measures a path its row does not
name is defect class A2, and an assertion is the only thing that stops it
recurring.

Two variants cannot take that route and say so through the row's `entry_point`
field. `tile_<BM>x<BN>x<BK>` names a member of the tile family, and `GemmDesc`
names a rung rather than a family index, so a tile row calls
`ckl::gemm_tile_family` with the index pinned; the dispatch probe still runs
first and still asserts that the family accepts the shape.
`baseline_cublas_autotune` calls `cublasGemmEx` directly, because the library
never passes anything but `CUBLAS_GEMM_DEFAULT`.

Every GEMM row also records what `Algo::kAuto` would have picked for the shape
(`plan_algo`, `tile`, `splits`) and whether a committed tile sweep informed that
choice (`plan_tuned`).

## Vendor baselines, measured once

The vendor baseline is a configuration of its own, not something a variant row
measures for itself. `sweep.py` runs the baselines first, in shuffled order with
the same cooldowns, and every variant row at that shape carries a reference to
that one measurement: `baseline_variant`, `baseline_median_ms`,
`baseline_gflops`. `refresh_summary` computes the percentage from the joined
baseline and never from a per row remeasure.

Two cuBLAS baselines are measured for GEMM. `baseline_cublas_default` is the
heuristic pick and is the baseline of record every percentage is against.
`baseline_cublas_autotune` is the autotuned selection, measured beside it and
quoted against it, because beating the heuristic pick is a weaker claim than
beating the autotuned pick and the report has to say which one it means.
`CUBLAS_GEMM_AUTOTUNE` is declared in cuBLAS 12 and 13 and documented as
experimental, so the sweep probes it against a real call before scheduling
anything: `bench_all --probe-autotune` answers, and a toolkit that refuses it
loses the second baseline rather than the whole sweep. The answer travels in
every row as `cublas_autotune_declared`.

The first autotuned call for a shape is the tuning run, and what it leaves in C
is not the answer, so that variant is primed once before it is verified or timed.
The self verification is what found that; without it the autotuned baseline would
have posted a number from a call whose output was wrong.

### cuBLAS fairness protocol

`cublasSetWorkspace` is never called, at any size. One call forfeits the default
pool, which is 32 MiB on sm_120, and a baseline running without its pool is not
the baseline a caller would get. If a workspace experiment is ever run,
`cublasSetStream` comes first, because setting the stream resets the workspace to
the default pool.

The math mode is set explicitly and read back into every row as
`cublas_math_mode`, because with FP16 in and `CUBLAS_COMPUTE_32F` cuBLAS may
reduce split-K partials in reduced precision, which moves both its speed and its
accuracy as an oracle.
`bench_all --disallow-reduced-precision-reduction` sets
`CUBLAS_MATH_DISALLOW_REDUCED_PRECISION_REDUCTION` and the row records that it
did.

`cublasGemmAlgo_t` is not swept beyond those two named variants. Algo selection
is a no-op on sm_80 and newer, so sweeping it would be inventing a parameter.

## Repeats and the interval

Every configuration runs `--process-reps` fresh `bench_all` processes, five by
default. The row of record carries all N process medians (`medians_ms`), the
median of those medians (`median_of_medians_ms`, which is also `median_ms`), and
a bootstrap 95 percent interval on the median of medians (`ci95_lo_ms`,
`ci95_hi_ms`) from 10000 seeded resamples. The seed and the resample count are in
the row, so the interval can be reproduced exactly.

The row's throughput is computed from the median of medians, not from whichever
process happened to be representative.

## Order and cooldowns

Configuration order is shuffled under a seed. The seed is chosen fresh unless
`--order-seed` sets it, and it lands in every row as `order_seed` along with
`order_index`, so the order a row was measured in is recoverable. There is a
cooldown between configurations, five seconds by default.

v1 ran sixty configurations back to back in matrix order, naive first and
`mma_opt` last, which is exactly the order that measures the fastest kernels on
the hottest GPU.

## Clocks

The preamble enables persistence mode, locks the graphics clock, and verifies the
lock by querying the GPU back. Every row carries `clock_locked` and
`locked_clock_mhz`, and after the sweep any row whose measured median SM clock
drifted more than 2 percent from the lock fails the run.

Locking needs privileges WSL2 may not have. Under WSL2 the lock has to come from
an elevated Windows side `nvidia-smi -lgc`. Without it the sweep refuses to run
unless `--allow-unlocked` is passed, and an unlocked run stamps
`clock_locked=false` on every row, which is what makes it impossible to quote
that data as a locked clock measurement later.

## Throttle gating

A configuration whose rows come back throttled is cooled for
`--throttle-cooldown-s` and re-run once. If it is still throttled the sweep exits
non-zero and names the configurations. v1 printed a warning and carried on, which
is how throttled rows reached the report.

`--throttle-test` marks the first row of each phase throttled. It exists so the
gate can be shown failing without waiting for a hot GPU, and it is the only thing
in the driver that fabricates a value.

## L2 flush

sm_120 has 48 MB of L2. A GEMV at 2048, a small GEMM, or an SpMV runs entirely
out of that cache across back to back reps, so the unflushed number measures the
cache and the flushed number measures the kernel. Reporting one of them without
the other hides which question was answered.

So both are measured. `--flush-l2 auto`, the default, turns the flush on for the
memory bound families and for any row whose working set fits in L2; `on` and
`off` force it. The sweep emits both rows at every affected shape and the row
says which it is through `l2_flushed`, `flush_bytes` and `working_set_bytes`.

The flush is a write of a scratch buffer twice the size of L2, enqueued on the
same stream ahead of the start event, so it is stream ordered before the timed
region and never inside it.

## Launch isolation with CUDA graphs

At 128 to 768 the kernel is short enough that per launch overhead is a visible
fraction of it. Those shapes are measured twice: once stream launched and once
with the same number of inner launches captured into a CUDA graph and replayed.
Both rows carry `launch_mode` and `inner_launches`, and the difference between
them is launch overhead with nothing else in it.

Graphs are a measurement tool here. They are not a library feature and nothing in
`ckl` requires them.

## One commit per summary

`refresh_summary` filters the JSONL to a single commit, `--commit` or HEAD, and
asserts exactly one row per key. The key is the family, variant, dtype and shape
plus the three protocol axes: `l2_flushed`, `launch_mode` and `inner_launches`.
Two rows that differ on any of those answer different questions and must not
collapse onto each other. A duplicate key or a commit with no rows fails loudly
rather than being resolved by taking whichever row came first, which is what v1
did.

A shape can therefore carry up to four rows. Exactly one of them is marked
`canonical`: stream launched, one launch per event pair, and the flush state
`auto` would have chosen. That is the row a consumer looking up a shape should
read, and the summary is ordered so it comes first.

### Filling a hole in a campaign without re-running it

One commit per summary is a hard rule, and it collides with a real situation: a
campaign completes, a handful of configurations wrote no row, the cause turns out
to be in the harness rather than in a kernel, and the fix lands after the sweep
commit. Re-running everything at the new commit costs the whole campaign, and
stamping only the re-run rows with the new commit is not an option, since
`refresh_summary` would then refuse to build a summary that contains both.

The rule for this is the one the 1.1.0 campaign followed for the three scan rows
that the tolerance model had cost it, and it turns on one question: **did the fix
change anything the measurement went through?**

- **If it did not**, the re-run rows carry the sweep commit, and the reason is
  written down where the rows are. The scan hotfix changed
  `ckl::scan_tolerance`, the verifier that decides whether a kernel is allowed to
  be timed, and the kernels, the plan, the launch path, the timing loop and the
  protocol are untouched by it. A row measured after the fix is therefore the row
  the machine would have produced at the sweep commit if the gate had let it
  through, and stamping it with the commit whose code produced the timings is the
  honest label. Confirm it before relying on it: rebuild only what changed, and
  check that nothing in the timing path did.
- **If it did**, there is no shortcut. Re-run the campaign at the new commit.

The re-run rows are otherwise ordinary: full protocol, same locked clock, same
process repeat count, same drift gate. What separates them from the rest is that
they were measured on a later date, and `experiments/results/` keeps the JSONL
append order, so the sequence is on the record rather than being smoothed over.

## Row schema

`bench_all` stamps `schema_version`, `sweep.py` adds `sweep_schema_version`, and
both are 2. A row carrying neither came from the v1 harness: it re-measured its
own baseline inside the row and kept no samples. `refresh_summary` still reads
those rows and marks them `baseline_source=per_row_remeasured` rather than
quoting them as if they had been measured under this protocol. Building a summary
from them needs `--commit` pointing at the v1 hash, since they do not carry the
current one.

Three columns say what kind of row this is rather than what it measured.
`status` is `ok` or `skipped_not_supported`; `status_reason` carries the plan's
sentence when it is the latter; and `row_note` is empty on a normal sweep and
carries whatever `--row-note` was given otherwise, which is how a row filled in
after the fact says so on its own face.

## The performance baseline

`experiments/results/perf_baseline.csv` is the comparison base for the nightly
perf-regression job. It is a copy of a summary produced under the protocol above,
in the same schema, and it is committed so the job has something to compare
against on a clean checkout. Without it `scripts/perf_regression.py` exits
non-zero rather than passing on no evidence.

The committed baseline is the 1.1.0 campaign summary: 984 rows at commit
`1155de4`, locked at 2497 MHz, five process repeats per configuration with a
bootstrap interval on every row. It was made by copying the file, not by a
separate run, so the baseline and the numbers in the README and the report are the
same measurements and cannot drift apart:

```sh
cp experiments/results/summary.csv experiments/results/perf_baseline.csv
```

A row counts as a regression only when both halves are true: tonight's median is
more than the tolerance slower (5 percent by default) **and** the two bootstrap
intervals do not overlap. Either test alone produces a job that cries wolf, which
is why the baseline has to carry intervals; a summary from a run without process
repeats cannot serve as one. A throttled row in tonight's sweep fails the run on
its own, because a throttled row is not evidence in either direction.

Refresh the baseline deliberately, by copying a new campaign summary over it in
its own commit, and never to make a red job go green.

## Running it

```sh
make sweep                       # the whole matrix, locked clocks, five process repeats
make sweep-quick                 # a representative subset
make tile-sweep                  # the tile family decision table
make summary                     # rebuild summary.csv from rows already on file
python3 benchmarks/sweep.py --list
python3 benchmarks/sweep.py --only gemm:mma_opt:fp16:2048x2048x2048 --process-reps 3
```

The sweep is resumable: a configuration whose key already has a row at this
commit is skipped, so a killed run continues rather than repeating finished work.
`--force` redoes them.

## Epilogue fusion study

`benchmarks/fusion_study.cpp` answers one fusion decision with measurement. It
computes `C = relu(alpha * A*B + bias)` two ways: unfused, where the top GEMM
writes C and a separate kernel then reads C, adds the column bias, applies ReLU
and writes C again; and fused, where the bias and the ReLU are folded into the
GEMM epilogue while C is still in registers, so C is written once.

The reason the answer comes out the way it does is on the roofline: the
standalone epilogue has an arithmetic intensity of 0.167 FLOP/byte, far to the
left of the ridge, so as a separate pass it is pure bandwidth. Folding it into a
compute bound GEMM removes that pass at almost no cost. The measured times are in
the results files and in the report; this driver predates the protocol above and
its numbers are being re-measured under it.
