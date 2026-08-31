# Corrections register

I audited this repository against its own results files and found claims that the
data does not support, measurements taken through contaminated timing, and a
roofline that was measuring cuBLAS against cuBLAS. This file is the record. One
entry per defect: what I claimed, what the repository actually shows, what I
changed, and what is still open.

It stays in the published tree. A project whose selling point is honest
measurement does not get to quietly delete the part where the measurement was
wrong.

Entries are dated by the day the correction landed, not the day the defect was
introduced.

---

## 2026-08-31: authorship and tooling disclosure

**What the repository showed.** `12_COMBINED_cuda-kernel-lab_spec.md` was tracked
in the tree. It reads as a direct instruction to an AI agent and it tells the agent
to disable attribution before the first commit. `LICENSE`, `README.md`, and
`CONTRIBUTING.md` all assert sole authorship. Those two things cannot both ship.

**What changed.** I deleted `12_COMBINED_cuda-kernel-lab_spec.md`, and the README
now says in plain words that I built this with AI agent tooling (Claude Code)
driven by specifications I wrote, and that the decisions and the claims are mine.
`CLAUDE.md` and `.claude/` are gitignored. I also deleted `PROGRESS.md`: its
content is in `docs/DIAGNOSTIC_LOG.md` and `docs/ENGINEERING_LOG.md`, and keeping a
third partial copy of the same history is how the copies drift apart.

**Open.** Nothing.

---

## 2026-08-31: machine configuration in the engineering log

**What the repository showed.** `docs/ENGINEERING_LOG.md` recorded that
passwordless sudo was confirmed working on my machine. That is a fact about my
security configuration and it has no business in a public log; the technical point
was only that root does not clear `ERR_NVGPUCTRPERM` under WSL2.

**What changed.** The line now says only that running ncu under sudo returned the
same error. The debug report's account of the same fight already said only that
root did not help, so it needed no edit.

**Open.** Nothing.

---

## A1, 2026-08-31: provenance, the commit hashes do not resolve

**What I claimed.** The README says every result carries the commit that produced
it. `CONTRIBUTING.md` says the same. The report has a traceability paragraph
saying it too.

**What the repository shows.** All 60 rows of `experiments/results/summary.csv`
and all 60 rows of `sweep.jsonl` carry commit
`42d47e0d90068bd48e3a4d17895beec6287579a6`. The eight `round_meta.txt` files carry
six more distinct hashes: `ae42365c` (round 1), `fc1f8f80` (round 2), `c2f8c6b4`
(round 3), `3075ebfe` (round 4), `e2f0f705` (rounds 6, 7 and 7b), `2e73d412`
(round 9). `git cat-file -t` resolves none of the seven. I rewrote the repository
history after the data was produced, and every recorded hash died with it. The
numbers are real; the audit trail attached to them is not.

**What changed.** `scripts/check_provenance.py` walks the summary CSV, the sweep
JSONL and every `round_meta.txt`, runs `git cat-file -t` on each recorded hash, and
exits 1 listing the offenders when one does not resolve. It runs from
`make check-style`. The seven dead hashes are listed in
`experiments/results/legacy_hashes.txt`, which explains why they are there and says
plainly that anything traceable only to one of them is provisional.

**Open.** The real fix is to re-run the full sweep and the ncu rounds on current
history and delete the allowlist. That is a 4 to 7 hour run on my machine and it is
mine to do. Until then the allowlist is a standing debt and the gate reports it on
every run.

---

## A2, 2026-08-31: setup work inside timed regions

**What I claimed.** Ground rule 4 of my own specification: vendor baselines are
measured with no setup work inside the timed region.

**What the repository shows.** `src/sparse/cusparse_ref.cpp` created a CSR
descriptor and two dense vector descriptors, called `cusparseSpMV_bufferSize`, then
`cudaMalloc` and `cudaFree`, every single call, and the benchmark harness timed the
whole function. So the cuSPARSE denominator carried a per call setup my kernels
never paid. The consequences are on the record in the committed data: SpMV rows at
283.44, 174.33 and 127.47 percent of cuSPARSE, and `gemv,warp,2048` at 208.61
percent of cuBLAS.

`benchmarks/bench_all.cpp` called `db.copy_from_host(b0)` inside both TRSM timed
lambdas, which is a blocking host to device copy sitting between the two event
records. TRSM overwrites its right hand side so it does need a fresh B each rep;
what it does not need is that restore inside the timed window, and both the kernel
and the cuBLAS baseline carried it.

**What changed.** The cuSPARSE wrapper now builds a plan (handle, CSR descriptor,
two vector descriptors, workspace) keyed on the pointers and the shape, keeps it,
and reuses it, so a repeated timed call enqueues `cusparseSetStream` and
`cusparseSpMV` and nothing else. Descriptors are released by a destructor, so a
failure part way through construction leaks nothing. The TRSM benchmark keeps a
second device buffer holding the pristine right hand side and restores from it with
`cudaMemcpyAsync` device to device, enqueued on the stream before the start event.
The kernel under test and the baseline both go through the same path.
`docs/sparse.md` marks the 131.8 percent of cuSPARSE figure retracted, along with
the 83.6 percent beside it.

**Open.** Re-measurement. Every percent of cuSPARSE in this repository, and the
TRSM percent of cuBLAS, is provisional until the sweep is re-run on the fixed
code. The kernel to kernel comparisons (warp per row is about 1.6 times naive) hold,
because both hand written variants were timed identically.

---

## A3, 2026-08-31: the roofline measured cuBLAS against cuBLAS

**What I claimed.** A measured roofline that places each kernel against the
machine's limits, and a "percent of roof" for the top kernel.

**What the repository shows.** `src/profiler/roofline.cpp` set
`tensor_peak = measure_gemm_fp16(8192)`, which calls the cuBLAS FP16 wrapper, and
then labelled it the tensor core peak. Every percent of roof in v1 was a percent of
cuBLAS restated on log axes. `scripts/plot_roofline.py` hard coded the chart title
"Roofline on RTX 5070", so the figure would have lied on any other GPU.
`src/tools/device_probe.cu` hard coded `cores_per_sm = 128`, which is right for
this part and wrong for Volta, Turing and A100.

**What changed.** The roofs are hardware. The tensor roof is SM count times the SM
clock times 512 FLOP per cycle, which is the FP16 input, FP32 accumulate rate on
this consumer part and half the FP16 accumulate rate; everything here runs
`CUBLAS_COMPUTE_32F`, so that is the rate that applies. The FP32 roof is the same
form with 256 FLOP per cycle. The clock is NVML's graphics clock sampled across the
measurement window, and it is written into both CSVs along with its source; if NVML
is unavailable the profiler falls back to the boost clock and says so on stdout and
in the file, because a roof built on boost is optimistic. cuBLAS is kept as a
`cublas_attainable` row and drawn as a dashed line under the roof. The chart title
comes from `device_name` in the ceilings CSV. The device probe reads FP32 lanes per
SM from a compute capability table and prints unknown rather than a confident wrong
number on an architecture it does not know.

**Measured on 2026-08-31**, 48 SMs at 2842 MHz: tensor roof 69.8 TFLOP/s, FP32 roof
34.9 TFLOP/s, measured streaming bandwidth 581.1 GB/s. cuBLAS FP16 attains 67.9
TFLOP/s, which is 97.2 percent of the tensor roof; that agreement is the best
evidence I have that 512 FLOP per cycle is the right figure. cuBLAS SGEMM attains
23.7 TFLOP/s, 67.8 percent of the FP32 roof. The top kernel lands at 84.1 percent
of the tensor roof.

**Note.** The GEMV point sits slightly above its own roof (305.6 against 290.5
GFLOP/s). That is a model artifact, not a timing defect: the bandwidth ceiling is
measured with a device to device copy, which both reads and writes, and a read
dominated kernel can exceed it. I am leaving the point where the measurement puts
it rather than tuning the model to hide it.

**Open.** Nothing blocking. A read only bandwidth ceiling would place the GEMV
point more fairly, and that is a V2 improvement rather than a correction.

---

## A4, 2026-08-31: the clocks were never locked

**What I claimed.** A ladder chart presenting nine GEMM rungs as one comparable
sequence.

**What the repository shows.** No `nvidia-smi -pm`, no `-lgc`, no
`nvmlDeviceSetApplicationsClocks` anywhere in the history.
`median_sm_clock_mhz` in `summary.csv` ranges from 1042 to 2880 MHz. The naive
kernel at 1024 cubed ran at 1042 MHz and the top kernel at 4096 cubed ran at 2865;
those two rows are on the same bar chart, and the ratio between them is not a pure
kernel result.

**What changed.** `benchmarks/sweep.py` has a preamble that runs before any
configuration: persistence mode on, graphics clock locked to `--lock-clock`
(default 2500 MHz), then the lock verified by querying `clocks.gr` back. The locked
value goes into every JSONL row as `locked_clock_mhz`, with `clock_locked` beside
it. After the sweep, any row whose `median_sm_clock_mhz` drifted more than 2 percent
from the lock is listed and the sweep exits non zero. If the lock fails, which it
may under WSL2 because clock control needs privileges on the Windows side, the
sweep refuses to run unless `--allow-unlocked` is passed, and an unlocked run
stamps `clock_locked=false` on every row so no locked clock claim can be built from
it. The lock is released at the end.

**Open.** The re-run. Every ladder number currently in this repository was measured
unlocked and will be replaced. The README says so at the results table.

---

## A5, 2026-08-31: claims the data does not support

**What the repository shows**, checked row by row against `summary.csv` and
`docs/DIAGNOSTIC_LOG.md`:

| Where | What I claimed | What the data says |
|---|---|---|
| README headline | about 90 percent of cuBLAS | 63.47 at 1024, 80.48 at 2048, 89.94 at 4096, 88.91 at 8192 |
| README | 90 percent again at 8192 | 88.91 percent |
| README | the swizzle removed 219M bank conflicts | it reduced them from 219M to 33.7M |
| README families table | TRSM matches cuBLAS to 1e-7, in a performance column | 31.78 percent of cuBLAS at 2048 by 256; the 1e-7 is a correctness result |
| `report/main.tex` | the tiled kernel is slightly slower | 17 percent slower, 1584 against 1908 GFLOP/s at 4096 cubed |
| `report/main.tex` | the shared read pipe is at 92 percent | 92.40 is `Memory Throughput` in the GPU Speed Of Light section; the shared read figure is `L1/TEX Cache Throughput` at 94.38 |

**What changed.** Every one of those is now the recorded figure. The README leads
with the curve rather than its best point, and reports the geometric mean over the
swept range. The families table carries GFLOP/s and percent of vendor columns, so
TRSM's real performance number is visible instead of being replaced by a
correctness one. The report states the tiled regression as 17 percent and says that
it is the most interesting result in the paper, because the optimization everyone
reaches for first is a regression on this card. The starved tensor core paragraph
cites the Nsight rows by their exact names and points at the file they came from.

**Two discrepancies with my own correction notes**, both resolved in favour of the
data:

1. The geometric mean over the four swept `mma_opt` shapes is **79.94 percent**,
   not 78.0. I had written 78.0 from an earlier arithmetic; the four recorded
   values give 79.94 and that is what the README says.
2. The 92.40 and 94.38 figures are in
   `experiments/results/ncu/round04/wmma_fp16_4096.txt`, not in the round 9 page.
   That is the right source: the claim in the report is about the WMMA rounds, not
   about the swizzled kernel. And 92.40 is the `Memory Throughput` row of the
   Speed Of Light section (it also appears as `Mem Busy` in Memory Workload
   Analysis); it is not the memory pipes figure, which is `Mem Pipes Busy` at
   14.37 percent.

**Open.** All of these figures carry the A1 provenance caveat and the A4 unlocked
clock caveat. The README says so where the numbers are.

---

## A6, 2026-08-31: missing evidence for the rounds that carry the story

**What the repository shows.** `experiments/results/ncu/` has rounds 01, 02, 03,
04, 06, 07, 07b and 09. There is no round 5 directory and no round 8 directory.
Round 8 is the reverted three stage pipeline, which is the centrepiece of the
honesty narrative in both the README and the debug report, and its numbers (74.5
percent of cuBLAS, down from 79.8) appear in no committed file. The bank conflict
result, 219 million conflicts down to 33.7 million, is the central causal finding
of the whole optimization story, and no committed text page contains it: the
metric was read out of the binary `.ncu-rep`, and those are gitignored as
regenerable.

**What changed.** Nothing has been reconstructed and no round has been invented.
`benchmarks/run_ncu_round.sh` now passes
`--metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` together with
the matching shared load wavefront counter, so a re-run of rounds 7 and 9 dumps the
conflict numbers into the committed `.txt` page alongside the sections. This entry
is the record for rounds 5 and 8.

**Open.**

- Re-run rounds 7 and 9 with the conflict metric and commit the text pages. Mine to
  do.
- Round 8 is not reconstructed. It gets reopened properly as part of the V2
  pipeline work, where the three stage experiment is redone under a locked clock
  with its artifacts committed.
- Round 5 cannot be reconstructed from what is in the tree. It stays a gap in the
  numbering, recorded here rather than renumbered away.

---

## A7, 2026-08-31: the pipeline could not reproduce itself

**What I claimed.** Reproducibility is a deliverable: a clean clone plus
`make setup && make all` reproduces every number, and nothing is hand copied into
any document.

**What the repository shows.** `.gitignore` excluded
`experiments/results/**/*.jsonl` and `experiments/results/roofline*.csv`, so the
sweep JSONL and both roofline CSVs were never committed. `scripts/gen_report_assets.py`
then skipped the roofline figure when its CSV was missing and exited 0. On a clean
clone the report therefore rebuilt around the committed PNG and reported success,
in CI included. The gate could not fail, which means it was not a gate.

**What changed.** Both ignore lines are gone, so `roofline.csv`,
`roofline_ceilings.csv` and `sweep.jsonl` are in the tree.
`scripts/gen_report_assets.py` checks its inputs up front and exits non zero naming
what is missing and which target produces it. A summary CSV with no data rows is
also an error. `summary_*.csv` scratch files and the binary Nsight reports stay
ignored, because those genuinely are regenerable.

**Open.** The CI report job should end with
`git diff --exit-code report/figures report/tables`, so a figure that does not
regenerate byte for byte from the committed results fails the build. That belongs
with the CI work, not here.

---

## What closes this register

Two runs on my machine, both mine to do:

1. The full sweep, under a locked clock, on current history, with the fixed
   cuSPARSE and TRSM timing. That closes A1 for the sweep rows, A2, A4, and turns
   every provisional percent in A5 into a measured one.
2. Nsight Compute rounds 7 and 9 re-run with the bank conflict metric. That closes
   A1 for the round metadata and the committed half of A6.

Round 5 stays a gap and round 8 is reopened as new work, not as a reconstruction.
