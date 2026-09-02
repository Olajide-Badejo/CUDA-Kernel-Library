# CUTLASS, the reference line

cuBLAS is a closed binary. When my kernel is slower than cuBLAS, the only thing
that tells me is that a gap exists. CUTLASS is source, so a gap against CUTLASS
can be read technique by technique: I can open the mainloop that beat mine and
see which of its choices I do not make. That is the whole reason this rung
exists. It is a reference line on the ladder, next to the vendor line, not a
kernel of mine and not something `Algo::kAuto` will ever pick.

## The pin

CUTLASS is fetched at configure time by `FetchContent` in
`src/gemm/CMakeLists.txt`, pinned to the exact release tag:

| | |
|---|---|
| Tag | `v4.7.1` |
| Commit | `cb4247394dd82148787aed73e5dc7cef33cbf862` |
| Repository | <https://github.com/NVIDIA/cutlass> |
| License | BSD 3-Clause, recorded in `NOTICE` |

The pin is a tag and not a branch on purpose: the argument in the next section is
about what this release's SM120 builders do, and a moving pin would quietly
invalidate it. Override it with `-DCKL_CUTLASS_TAG=` only to test a newer
release, and update this page when the answer changes.

The clone is shallow (`GIT_SHALLOW TRUE`), which matters here: the full history
is over a gigabyte and a build tree under `/mnt/c` pays Windows filesystem
latency on every object. The shallow v4.7.1 tree is about 220 MB and lands in a
few seconds on a warm network. Nothing of CUTLASS is vendored into this
repository, no CUTLASS target is configured, and no CUTLASS type appears in a
public header: it is headers, included by one translation unit,
`src/gemm/gemm_cutlass_ref.cu`.

## Why the 2.x device API and not a CollectiveBuilder

The modern way to ask CUTLASS for a kernel is `CollectiveBuilder`, which picks a
mainloop for the architecture you name. On SM120 it cannot give you an FP16 one.
In the pinned release,
`include/cutlass/gemm/collective/builders/sm120_mma_builder.inl` opens its
non blockscaled specialization at line 80 with

```text
static_assert(detail::is_sm10x_f8f6f4_element<ElementA>() &&
              detail::is_sm10x_f8f6f4_element<ElementB>(), ...)
```

and closes at line 115 with

```text
static_assert(UseF8f6f4, "Non-blockscaled collective builder only supports F8F6F4 MMA.\n");
```

Half precision is not an f8f6f4 element, so the request does not compile. The
other SM120 builders in that directory are blockscaled or sparse, which is to
say narrow precision by construction: every SM120 entry in the CUTLASS changelog
since 3.8 is fp4, fp6 or fp8. That is the same observation Section 1 of the
build spec makes about the platform generally, and it is the reason a hand
written FP16 kernel has a genuine shot at the top of this ladder: the FP16 path
on consumer Blackwell is not where upstream is spending its tuning effort.

So the rung uses the 2.x device API over an Ampere style multistage mainloop.
That is not a fallback to something slow. `cp.async`, `ldmatrix` and
`mma.sync.aligned.m16n8k16` all assemble on plain sm_120, `m16n8k16` is the
largest dense FP16 MMA shape this part has, and an Ampere style multistage
mainloop is effectively what cuBLAS ships for FP16 here too.

## The instantiation

From `src/gemm/gemm_cutlass_ref.cu`. The shapes deliberately match
`gemm_mma_opt`, so the comparison is between two implementations of one design
rather than between two different designs.

| Parameter | Value | Same as `gemm_mma_opt` |
|---|---|---|
| Device API | `cutlass::gemm::device::GemmUniversal` | not applicable |
| Element A, B | `cutlass::half_t` | yes |
| Element C, accumulator | `float` | yes |
| Layouts | row major A, B and C | yes |
| Operator class | `cutlass::arch::OpClassTensorOp` | yes |
| Architecture tag | `cutlass::arch::Sm80` | not applicable |
| Threadblock tile | 128 by 128 by 32 | yes |
| Warp tile | 64 by 64 by 32 | no: `mma_opt` uses 64 by 32, eight warps |
| Instruction shape | 16 by 8 by 16 | yes |
| Pipeline stages | 3 | yes |
| Alignment A, B | 8 halves, 16 bytes | yes |
| Epilogue | `LinearCombination`, 4 floats per access | comparable |
| Threads per block | 128 | no: `mma_opt` runs 256 |

The one structural difference is the warp decomposition: CUTLASS splits the
block tile into four 64 by 64 warp tiles over 128 threads, and `gemm_mma_opt`
splits it into eight 64 by 32 warp tiles over 256 threads. That shows up in the
register budget. On nvcc 13.3.73 for sm_120, ptxas gives the CUTLASS kernel 236
registers per thread with no spill, against 122 for `gemm_mma_opt`; twice the
output per thread, roughly twice the accumulator state, half the threads.

Both stage the same mainloop buffer: three stages of a 128 by 32 A tile plus a
32 by 128 B tile is 49152 bytes, over the 48 KB static ceiling, so both opt into
dynamic shared memory and the sm_120 per block ceiling of 101376 bytes leaves
room. CUTLASS's launch allocation is the union of that buffer with its epilogue
staging, so it is at least 49152 and possibly more; the exact figure is a
`sizeof` inside the template and is not read off a disassembly, so it is not
quoted here.

`GemmUniversal::get_workspace_size` is queried on every call and asserted to be
zero for the single K slice this rung runs, because a workspace allocated inside
a call would be an allocation inside a timed benchmark region. Split-K is a rung
of its own on this ladder, so the reference line does not use CUTLASS's.

## The shape rule

Row major A has leading dimension k; row major B and C have leading dimension n.
The mainloop moves 16 bytes per access along the contiguous axis, so the rung
takes shapes with n and k divisible by 8. m is free: CUTLASS predicates the M
edge, and the test suite runs m of 1, 7, 65, 127 and 129 against the oracle.

Anything else is refused. `ckl::gemm` returns `Status::kNotSupported` with
`chosen` still reporting `kCutlass`, and the free function `ckl::gemm_cutlass`
throws. It is never rerouted to another kernel, because a benchmark row labelled
CUTLASS that measured one of my kernels would be worse than no row at all. Query
the rule with `ckl::gemm_cutlass_supports(m, n, k)`.

## What is measured, and what is not

Correctness is measured now. The rung is held to the same three errors as every
other tensor rung in `tests/test_gemm_tensor.cpp`: 1e-5 relative Frobenius
against a same precision cuBLAS oracle, `tol(k)` against a double precision
reference over the rounded inputs, and a storage roundoff bound against the FP32
data those inputs came from. It also passes the beta zero rule (C is not read),
the empty output and empty contraction cases, and large K.

The sanitizers are clean on it: memcheck, racecheck, initcheck and synccheck all
run without an error on the tensor suite with the rung in it. One note for
whoever runs them next, because it costs an hour to rediscover. `synccheck`
tracks a fixed number of `cuda::barrier` structures and this suite overflows
that budget, which it reports as

```text
Warning: Detected overflow of tracked cuda::barrier structures.
Try using --num-cuda-barriers to fix the issue
```

and then the context goes bad and every later test fails with
`cudaErrorUnknown`. That is the tool running out of room, not a kernel
misbehaving, and it predates this rung: the same overflow happens with the
CUTLASS tests filtered out. `--num-cuda-barriers 1024` clears it and the suite
passes with zero errors.

Performance is not measured yet. It needs locked clocks and the protocol in
`docs/benchmarking.md`, and that is owner work. The sweep already knows how to
produce the rows: `cutlass` is a first class FP16 variant in
`benchmarks/bench_all.cpp` and in `benchmarks/sweep.py`, and the ladder chart in
`scripts/gen_report_assets.py` has a slot for it.

## Gap analysis

The structure below is the analysis this rung exists to support. Every number in
it is pending: the sweep at locked clocks and the profile-cuBLAS round from
Section 9.4 of the build spec are both owner actions, and nothing here will be
filled in from an estimate.

| Shape (m = n = k) | `mma_opt` GFLOP/s | `cutlass` GFLOP/s | cuBLAS GFLOP/s | `mma_opt` as percent of cutlass |
|---|---|---|---|---|
| 1024 | pending | pending | pending | pending |
| 2048 | pending | pending | pending | pending |
| 4096 | pending | pending | pending | pending |
| 8192 | pending | pending | pending | pending |

Once those rows exist, the gap gets attributed technique by technique. These are
the candidates, in the order I would test them, each stated as a hypothesis
rather than a finding:

1. **Warp decomposition.** CUTLASS runs 128 threads with 64 by 64 warp tiles;
   `mma_opt` runs 256 threads with 64 by 32. The wider warp tile amortizes each
   `ldmatrix` over more `mma.sync`, at the cost of more accumulator registers per
   thread. Evidence to collect: the HMMA to LDSM ratio in each capture under
   `experiments/sass/`, which is a compile time fact and already available, plus
   `smsp__inst_executed` from an ncu round.
2. **Epilogue path.** CUTLASS stages the accumulator through shared memory to
   produce coalesced global stores; `mma_opt` stores 64 bit pairs straight from
   registers in the `mma` fragment layout. Evidence: `l1tex__t_sectors_pipe_lsu`
   and the store efficiency counters on the epilogue.
3. **Software pipelining detail.** Both run three stages, but the placement of
   the barriers and the depth of the register prefetch differ. Evidence:
   `smsp__average_warp_latency_issue_stalled_barrier` and the stall reasons.
4. **Predication cost.** The reference line predicates every edge; `mma_opt`
   takes an aligned fast path and hands unaligned shapes to the WMMA kernel. On
   aligned shapes this should favour `mma_opt`, and if it does not, the
   predication is not what costs.
5. **Register allocation and residency.** The reference line takes 236 registers
   over 128 threads; `mma_opt` takes 122 over 256. Both spill nothing. Which of
   the two the register file lets sit resident alongside a 49152 byte shared
   memory buffer is the question, and
   `experiments/results/register_study.csv` answers it for `mma_opt` only: shared
   memory pins it at two blocks per SM whatever the register count. The same
   study has not been run on the CUTLASS instantiation.

## Reproducing

```sh
make build                 # fetches CUTLASS at the pinned tag and builds the rung
ctest --test-dir build -R Cutlass --output-on-failure
./build/benchmarks/bench_all gemm cutlass fp16 4096 4096 4096
```

The benchmark call above needs locked clocks to produce a number worth quoting;
see `docs/benchmarking.md` for what to do when the clock cannot be locked.
