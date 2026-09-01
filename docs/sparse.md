# CSR SpMV

Sparse matrix vector product in compressed sparse row form, `y = alpha * (A * x)
+ beta * y`. Six hand written variants against cuSPARSE under both of its CSR
algorithms, all reached through one plan.

## Why a matrix suite

v1 measured on one synthetic matrix with one hard coded seed, and that was
enough to confirm the winner I already had and nothing else. The warp per row
kernel beat the thread per row kernel on a skewed degree distribution, which is
what a skewed degree distribution is for. What one matrix could not do was
separate the variants that came next: merge based and warp per row run at the
same speed on a uniform degree distribution, so a uniform matrix says nothing
about load balance, and a matrix with a heavy tail says nothing about the lane
waste a narrow row costs a full warp.

So the suite is seven SuiteSparse matrices plus the generator, chosen so that
each variant has at least one case it should win and at least one it should
lose.

| Matrix | Group | Class | Rows | Nonzeros | Why it is here |
|---|---|---|---|---|---|
| `parabolic_fem` | Wissgott | regular stencil | 525,825 | 3,674,625 | near uniform 7 nonzeros per row; the case `spmv_csr_vector` targets |
| `mc2depi` | Williams | near diagonal | 525,825 | 2,100,225 | 4 nonzeros per row, tight bandwidth, near best case locality |
| `cant` | Williams | banded FEM | 62,451 | 4,007,383 | about 64 nonzeros per row, narrow band, natural blocks for BSR |
| `pdb1HYS` | Williams | blocked | 36,417 | 4,344,765 | dense sub blocks; the second BSR case |
| `scircuit` | Hamm | irregular circuit | 170,998 | 958,936 | short rows with a long tail, very low work per byte |
| `webbase-1M` | Williams | power law web | 1,000,005 | 3,105,536 | max row degree 4700; where merge based must win |
| `soc-LiveJournal1` | SNAP | power law social | 4,847,571 | 68,993,773 | large power law; exercises the plan at scale |
| `synthetic_skewed` | generated | skewed | 131,072 | varies | v1's generator, kept, now driven by seeds 1 to 5 |

Every row above is the confirmed shape read out of the downloaded MatrixMarket
file, not the SuiteSparse index value: `scripts/fetch_matrices.py` refuses to
write a manifest row whose converted CSR disagrees with what
`experiments/matrices.toml` claims. One correction came out of that check.
`scircuit` is listed under Williams in the build spec's table and SuiteSparse
serves it from Hamm; the group in `matrices.toml` is the one that resolves.

Nothing downloaded enters the repository. Archives and converted matrices live
under `$CKL_MATRIX_CACHE`, default `~/.cache/ckl/matrices`, and the tracked
artifact is `experiments/matrix_manifest.csv`, which records the group, the
confirmed shape and nonzero count, and the SHA-256 of both the archive and the
converted binary CSR. `--verify` rechecks all of it and exits non-zero on a
mismatch, so a benchmark row that names a matrix names a specific sequence of
bytes.

```sh
python3 scripts/fetch_matrices.py                 # the six small matrices
python3 scripts/fetch_matrices.py --include-large # adds soc-LiveJournal1, about 250 MB
python3 scripts/fetch_matrices.py --verify
python3 scripts/fetch_matrices.py --list
```

## The plan

`ckl::SpmvPlan` is built once per matrix and owns everything that does not
change between calls: a cuSPARSE handle, the CSR and dense vector descriptors, a
workspace for each of the two vendor algorithms, the analysis
`CUSPARSE_SPMV_CSR_ALG2` needs, the histogram of nonzeros per row, the lane
width that histogram implies, and the SELL-C-sigma and BSR conversions. Running
a variant through it enqueues kernel launches and nothing else.

That is the structural fix for defect A2. v1's cuSPARSE wrapper created three
descriptors, sized the SpMV workspace, ran `cudaMalloc` and `cudaFree`, and did
all of it inside every call, and the harness timed it as the vendor's SpMV. The
plan makes that impossible through the public path: there is no entry point in
1.1.0 that builds a descriptor per call. `ckl::spmv_cusparse` and
`ckl_spmv_csr` keep the old free function shapes and hold a one entry plan cache
behind them, and both say so in their documentation.

The claim is checked rather than asserted. `bench_spmv` arms a CUPTI runtime
callback on the `cudaMalloc`, `cudaMallocAsync`, `cudaFree` and `cudaFreeAsync`
CBIDs around its timed lambda and exits non-zero if the counter moves.
`--gate-red` restores the v1 per call allocation on purpose so the gate can be
seen failing; the numbers it prints with the allocation back in are about fifty
percent slower across every variant, which is the size of the effect that got
into the v1 report.

## Variants

Each one fixes a specific thing, and each is reachable through `SpmvAlgo` with
`chosen` reported.

- **naive** (`spmv_csr_naive.cu`): one thread per row. A long row makes its
  thread run far longer than its warp neighbours, and the warp retires at the
  speed of its slowest row.
- **warp** (`spmv_csr_warp.cu`): one 32 lane warp per row with a `shfl_down`
  reduction. A long row is split 32 ways, and the strided reads of `col_idx` and
  `values` coalesce within a row.
- **vector** (`spmv_csr_vector.cu`): a lane group of 2, 4, 8, 16 or 32 per row,
  the width chosen from the plan's histogram and the kernel templated on it.
  Fixes the lane waste the warp variant pays on the regular classes: at 7
  nonzeros per row a full warp idles 25 lanes and still runs five reduction
  steps to combine nothing.
- **merge** (`spmv_merge.cu`): the merge path of Merrill and Garland, SC 2016.
  The merge of the row end offsets with the nonzero indices has exactly
  `m + nnz` steps whatever the degree distribution, so cutting it into equal
  diagonal segments gives every thread exactly `ceil((m + nnz) / threads)` units
  of work. A second pass adds in the partial sums of the rows that straddle a
  segment boundary. Fixes the imbalance that survives the warp variant, which
  still hands one 4700 nonzero row to a single warp while its neighbours finish
  four element rows.
- **sell** (`spmv_sell_c_sigma.cu`): sliced ELLPACK, C = 32, one warp per slice
  and one lane per row. Rows are sorted by length inside a window of sigma rows
  and padded to the slice maximum, so the inner loop is a 128 byte column read
  with no `row_ptr` indirection at all. Fixes the uncoalesced walk both CSR
  kernels do across row boundaries, and pays for it in padding.
- **bsr** (`spmv_bsr.cu`): block CSR, the stretch item. An r by r dense block
  shares one column index and one span of `x` across r squared values. It exists
  for `pdb1HYS` and `cant` and does nothing for the graph classes; its baseline
  is `cusparseSbsrmv`, not `cusparseSpMV`.

## The decisions that are mine, and provisional

Three numbers in `spmv_plan.cpp` are shape decisions rather than measurements,
and each of them says so in the source.

**The lane width** is the smallest power of two at or above the mean nonzeros
per row, clamped to 2 through 32: 2 up to a mean of 2, then 4, 8, 16 and 32.
Below the mean the group loops, above it the group idles, and the mean is where
those two costs cross for a distribution with a light tail. It is the wrong rule
for a heavy tail, which is what the merge variant is for.

**A matrix is power law** when its longest row holds at least 32 times the mean.
32 is the warp width, so the rule reads as one row occupying a whole warp for as
long as a mean row occupies one lane. Nothing in the suite sits near the
threshold: `webbase-1M` is about 1500 times its mean and `parabolic_fem` about
2, so the exact constant does not decide a case. It does decide the sigma
default, 8192 on power law and 256 otherwise, and it is worth stating plainly
that v1's own generator lands on the power law side of it.

**The BSR block dimension** is the largest of 8, 4 and 2 whose fill ratio
reaches one half, and 2 when none of them does. Below one half the padded zeros
cost more traffic than the shared column indices save. The detection pass is
timed on the host and reported as `bsr_detect_ms`, because a stretch variant
whose setup costs more than its kernel saves has not earned its place; on
`soc-LiveJournal1` the pass takes about 600 ms and then declines to build
anything, because no block dimension gets near half fill on a social graph and
the padded arrays would be past the plan's memory ceiling. That is the drop the
spec asks to be recorded rather than deleted: the code path stays, the plan
reports why it did not run, and `kBsr` returns `kNotSupported` with the reason
instead of falling back to a CSR kernel.

**The merge path's seven units per thread** is the fourth. It sets both the
serial work a thread does and how far apart neighbouring threads read, since
thread t starts at path offset 7t. The thread count follows from it rather than
being capped, so a 69 million nonzero matrix gets ten million threads and eighty
megabytes of carry scratch. An earlier version capped the threads at 65536
instead, which put over a thousand serial units in each one and scattered its
reads across the whole matrix; that is the first thing an ncu round on this
kernel should look at again.

`kAuto` picks from the histogram: power law goes to the merge path, a mean row
of 32 or more goes to the warp kernel, and everything else goes to the vector
kernel at the width the histogram implies. SELL and BSR are never auto chosen,
because whether they win depends on a padding ratio and a block fill that only a
measurement settles.

## Traffic model

Bytes moved by one SpMV:

```
4 nnz (values) + 4 nnz (col_idx) + 4 (m+1) (row_ptr) + x reads + 4 m (y write)
```

plus another `4 m` for the y read when beta is nonzero. The x term is the honest
part. With no reuse at all it is `4 nnz`; with perfect L2 residency it is `4 n`;
the truth sits between and depends on how local the column indices are, which is
exactly what the suite varies. So the model is printed as a band, `model_bytes_low`
and `model_bytes_high`, on every benchmark row. It explains a measurement and
does not replace one: the number of record is the measured `dram__bytes.sum`
from ncu, and until that round runs the field on every row reads
`pending ncu round`.

## Roof and targets

SpMV is memory bound, so percent of cuSPARSE is a comparison and not a ceiling.
The roof is the DRAM bandwidth `ckl_roofline` measures on this device at locked
clocks, never the datasheet figure.

The following are hypotheses under ground rule 3, to be replaced by runs:
regular and banded reach 70 to 85 percent of the measured DRAM roof, near
diagonal 75 to 90, and power law 35 to 55 and gather bound rather than bandwidth
bound, which the sectors per request row has to show. Percent of cuSPARSE per
class, same status: at least 95 on regular, banded and near diagonal, and at
least 110 on power law.

## Measured

**Pending.** No number from the tuned family is in this document, because none
has been measured at locked clocks yet. The sweep is owner work and it produces
`experiments/results/spmv_suite.csv` alongside the canonical JSONL; every row
there carries the matrix name, both cuSPARSE algorithms, the SELL padding ratio,
the locked clock and a commit hash that resolves under `check_provenance.py`.

**The v1 percent of cuSPARSE column is retracted and stays retracted.** The v1
wrapper created three descriptors, sized the SpMV workspace, and ran `cudaMalloc`
and `cudaFree` on every call, and the harness timed all of it. So the cuSPARSE
denominator was the vendor's SpMV plus a per call setup my own kernels never
paid. That is defect A2 in [`CORRECTIONS.md`](CORRECTIONS.md). The 131.8 percent
figure, and the 83.6 percent next to it, say nothing about cuSPARSE.

| variant | GFLOP/s | percent of cuSPARSE |
|---|---|---|
| naive | 60.1 | retracted, pending re-measurement |
| warp | 94.8 | retracted, pending re-measurement |
| cuSPARSE | contaminated, pending re-measurement | 100.0 |

The kernel to kernel comparison in that table survives, because both hand
written variants were timed the same way. Correctness for every variant is
checked against a double precision CPU reference with a per row tolerance of
`8 sqrt(nnz_row) FLT_EPSILON` (`tests/test_spmv.cpp`), and that check was never
affected.

## Gate S

Mechanically checkable, and the full runs are owner actions at locked clocks.

- `spmv_merge` median at least 1.5 times `spmv_csr_warp` on both power law
  matrices, five independent process repeats, bootstrap 95 percent intervals not
  overlapping.
- Every variant passes the per row tolerance on all seven suite matrices and all
  five synthetic seeds, and the full pathological set is green for every
  variant.
- No allocation inside any timed region, counted through CUPTI by `bench_spmv`,
  and that gate shown red with the v1 per call allocation restored and green
  after revert.
- Both cuSPARSE algorithms present for every matrix in `spmv_suite.csv`, with
  every reported percent naming which one it is against.
- `scripts/fetch_matrices.py --verify` exits 0, and every row of
  `spmv_suite.csv` names a matrix present in `experiments/matrix_manifest.csv`.
- Every percent of roof row carries the measured `dram__bytes.sum`, the locked
  clock, and a commit hash that resolves under `check_provenance.py`.
- SELL rows carry `nnz_padded / nnz`, and no SELL configuration is reported as a
  win at a padding ratio above 1.5 without the ratio quoted beside it.

## Running the family

```sh
./build/benchmarks/bench_spmv scircuit          # every variant on one matrix
./build/benchmarks/bench_spmv synthetic:131072 --seed 3
./build/benchmarks/bench_spmv scircuit --gate-red   # the allocation gate, red
python3 benchmarks/sweep.py --only spmv         # the family under the full protocol
```

`bench_spmv` prints whatever the clocks happened to be doing; nothing it prints
belongs in a document. The row of record comes from `bench_all` under
`sweep.py` with the clock locked, five process repeats and a bootstrap interval,
per [`benchmarking.md`](benchmarking.md).
