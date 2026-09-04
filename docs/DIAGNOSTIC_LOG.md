# Diagnostic log

The round by round record of the GEMM optimization loop. One dated entry per
Nsight Compute diagnostic round: the metric snapshot, the named top limiter, the
hypothesis, the single change applied, and the measured delta. Every entry points
at the ncu report under `experiments/results/ncu/<round>/` that backs it.

This is the method of the project (Section 5 of the spec) and the source of the
main report's optimization chapter. Rounds that failed, where the hypothesis was
wrong and the change was reverted, are kept here and cross referenced from the
engineering log.

Rounds begin at Phase 2 (tiled GEMM), once the naive baseline exists and the ncu
counter permission is cleared.

## Round 1: tiled versus naive, why tiling alone does not win

Date: 2026-07-19. Artifacts: `experiments/results/ncu/round01/` (naive and tiled
at 2048 and 4096 cubed, `.ncu-rep` plus `.txt` detail pages, `round_meta.txt`
with the commit and toolkit).

Going in (wall clock, 20 reps, median, 4096 cubed): naive 1910 GFLOP/s, tiled
1588 GFLOP/s, so the 32 by 32 tiled kernel is about 17 percent slower than naive.
Both near 8 percent of cuBLAS. The question the round answers: why does staging
into shared memory lose here.

Metric snapshot at 4096 cubed (Nsight Compute):

| metric | naive | tiled |
|---|---|---|
| Duration (ms) | 81.9 | 97.7 |
| Speed of Light, compute and memory (percent) | 92.2 | 81.4 |
| DRAM throughput (percent) | 37.1 | 15.7 |
| L1 / TEX hit rate (percent) | 87.5 | 0.4 |
| L2 hit rate (percent) | 50.3 | 50.9 |
| Achieved occupancy (percent) | 99.8 | 66.6 |
| Active warps per SM | 47.9 | 32.0 |
| Registers per thread | 40 | 37 |
| Warp cycles per issued instruction | 43.8 | 45.5 |
| Top warp stall | long scoreboard on L1TEX loads, 23.1 cyc | MIO throttle, 28.1 cyc |

Top limiter, named: for the tiled kernel it is the MIO (memory input/output) pipe,
driven by shared memory instructions (Nsight's own note: "high in cases of extreme
utilization of the MIO pipelines, which include ... shared memory instructions"),
compounded by low occupancy. For the naive kernel the limiter is the L1TEX pipe,
but at 99.8 percent occupancy the machine hides it well.

What the numbers say. The naive kernel is not really going to DRAM: its DRAM
throughput is only 37 percent while its L1 hit rate is 87.5 percent. The 48 MB L2
and the L1 caches absorb its redundant B column reads almost entirely, so its
"redundant" loads are cheap and it runs at near full occupancy against a busy but
well hidden L1TEX pipe. The tiled kernel deliberately routes each global element
through shared memory exactly once, which drops its L1 hit rate to 0.4 percent (it
no longer reuses through L1 at all) and moves the pressure onto the MIO and shared
memory pipe. That would be a fine trade if occupancy stayed high, but the 32 by 32
tile is a 1024 thread block and the block limit is 1 per SM (registers, shared
memory, and warp limits all cap it at one), so achieved occupancy falls to 66.6
percent and there are fewer warps to hide the shared memory latency. Net result:
slower than naive.

Hypothesis from Phase 2 confirmed: tiling alone is limited by occupancy and the
shared memory (MIO) pipe, not by global bandwidth, and the large cache hierarchy
means naive was never bandwidth bound to begin with.

Single change to apply next (Phase 3): register blocking. Give each thread many
outputs (an 8 by 8 register tile) inside a smaller thread block, so the same
output tile uses far fewer threads (restoring occupancy), each shared value loaded
is reused many times from registers (raising arithmetic intensity per thread and
work per MIO instruction), and the loads become fewer and wider (float4). This is
the Volkov and Demmel decisive step, and the next round measures whether it lifts
compute throughput and pulls clearly ahead of naive.

## Round 2: register blocked kernel, the decisive step and the new limiter

Date: 2026-07-19. Artifacts: `experiments/results/ncu/round02/`
(register at 2048 and 4096 cubed).

Result (4096 cubed): the register blocked kernel runs in 14.9 ms at 10381
GFLOP/s, versus tiled at 97.7 ms and 1588 GFLOP/s and naive at 1890 GFLOP/s. That
is 6.5 times the tiled kernel and 5.5 times naive, and it moves from about 8
percent of cuBLAS to 47 percent in one step. The ladder's central claim holds:
register blocking, not shared tiling, is the decisive move on this GPU.

Metric snapshot at 4096 cubed, register versus the tiled kernel from Round 1:

| metric | tiled | register |
|---|---|---|
| Duration (ms) | 97.7 | 14.9 |
| Achieved GFLOP/s | 1588 | 10381 |
| Speed of Light, memory (percent) | 81.4 | 85.4 |
| L1 / TEX pipe throughput (percent) | n/a | 88.8 |
| Speed of Light, compute (percent) | 81.4 | 65.1 |
| DRAM throughput (percent) | 15.7 | 18.5 |
| L1 / TEX hit rate (percent) | 0.4 | 18.5 |
| L2 hit rate (percent) | 50.9 | 65.0 |
| Achieved occupancy (percent) | 66.6 | 32.6 |
| Registers per thread | 37 | 106 |
| Warp cycles per issued instruction | 45.5 | 9.3 |
| Top warp stall | MIO throttle, 28.1 cyc | MIO throttle, 4.4 cyc |

Why it is faster despite lower occupancy: warp cycles per issued instruction fell
from 45.5 to 9.3, so each warp makes far more forward progress per cycle. The 8 by
8 register tile turns 16 shared memory reads per contraction step into 64 fused
multiply adds, so arithmetic intensity per thread rose sharply and the machine no
longer needs high occupancy to stay busy. DRAM throughput is only 18.5 percent, so
this kernel is not close to global bandwidth bound.

New top limiter, named: the L1 / TEX pipe at 88.8 percent, which here is the
shared memory read path feeding registers, and a secondary occupancy ceiling. The
8 by 8 accumulators plus fragments cost 106 registers per thread, so only two
128 by 128 blocks fit per SM and achieved occupancy is 32.6 percent (about 15.6
active warps per SM). The inner loop issues eight scalar LDS for A and eight for B
per contraction step; those scalar shared loads are what saturate the L1 / TEX
pipe.

Two candidate single changes were considered: (a) vectorize the shared loads to
128 bit (LDS.128) to cut the shared load instruction count roughly fourfold and
relieve the L1 / TEX pipe, and (b) cp.async double buffering (Phase 4) to overlap
the global tile loads with the compute of the previous tile and remove the
serializing syncthreads. Per the one change per round rule the next rung applies
cp.async double buffering, the ladder's designated Phase 4 step; the vectorized
shared load idea is kept on the list for a later round on the top kernel if the
L1 / TEX pipe is still the limiter after double buffering.

## Round 3: cp.async double buffering, and the FP32 occupancy wall

Date: 2026-07-19. Artifacts: `experiments/results/ncu/round03/`
(cp_async and register at 4096 cubed).

Result (4096 cubed): the cp.async kernel runs at 16174 GFLOP/s, 73.4 percent of
cuBLAS, up from the register kernel's 10380 GFLOP/s and 47 percent. Correctness
holds on all shapes.

Metric snapshot at 4096 cubed, cp.async versus the register kernel:

| metric | register | cp.async |
|---|---|---|
| Achieved GFLOP/s | 10380 | 16174 |
| Speed of Light, memory (percent) | 85.4 | 62.7 |
| Speed of Light, compute (percent) | 65.1 | 55.1 |
| DRAM throughput (percent) | 18.5 | 29.0 |
| Warp cycles per issued instruction | 9.3 | 3.49 |
| Achieved occupancy (percent) | 32.6 | 16.7 |
| Registers per thread | 106 | 149 |
| Active warps per SM | 15.6 | 8.0 |

What cp.async bought: warp cycles per issued instruction fell from 9.3 to 3.49, so
the loads now overlap the math instead of stalling in front of it, DRAM
utilization rose from 18.5 to 29.0 percent, and no pipe is saturated any more
(memory 62.7, compute 55.1). The kernel has excellent instruction level
parallelism.

New and terminal limiter for the FP32 path, named: occupancy. The double buffered
8 by 8 tile now costs 149 registers per thread, so only one 128 by 128 block fits
per SM and achieved occupancy is 16.7 percent, about 8 active warps per SM. With so
few warps and no pipe near its ceiling, the SM is underpopulated and latency
exposed; this is why it sits at 73 percent of cuBLAS rather than higher. Two FP32
levers remain (shrink the tile or otherwise cut register pressure to raise
occupancy, or vectorize shared loads), but both trade against the arithmetic
intensity per thread that made the register kernel fast, and neither changes the
fundamental ceiling: cuBLAS SGEMM on this card is itself a well tuned CUDA core
kernel, and matching it on FP32 CUDA cores is a game of diminishing returns.

Single change to apply next (Phase 5): change the compute primitive. Move the
inner product onto the fifth generation tensor cores with WMMA (FP16 and BF16
storage, FP32 accumulate). This is the designated path to the compute bound gate
and the 90 percent of cuBLAS target, because the tensor cores raise the compute
ceiling far above the FP32 CUDA core peak that both this kernel and cuBLAS SGEMM
are bounded by. The FP32 cp.async kernel is accepted as the top hand written FP32
variant at 73 percent of cuBLAS, with its gap diagnosed here as an occupancy and
register pressure wall.

## Round 4: WMMA tensor core kernel, tensor cores present but starved

Date: 2026-07-19. Artifacts: `experiments/results/ncu/round04/`
(wmma_fp16 at 4096 cubed).

Result (4096 cubed, FP16): the WMMA kernel runs at 36308 GFLOP/s, versus the FP32
cp.async kernel at 16174 GFLOP/s, so moving the inner product onto tensor cores
more than doubles throughput in one step. Against the FP16 cuBLAS tensor oracle
(60810 GFLOP/s) it sits at about 60 percent; the compute bound gate and the 90
percent target are not met yet, and the round says why.

Metric snapshot at 4096 cubed:

| metric | value |
|---|---|
| Achieved GFLOP/s | 36308 |
| Percent of FP16 cuBLAS | 59.7 |
| Speed of Light, memory (percent) | 92.4 |
| L1 / TEX pipe throughput (percent) | 94.4 |
| Speed of Light, compute (percent) | 53.7 |
| Tensor pipe utilization (percent) | 53.7 |
| DRAM throughput (percent) | 10.0 |
| L2 hit rate (percent) | 96.2 |
| Achieved occupancy (percent) | 32.9 |
| Registers per thread | 114 |
| Warp cycles per issued instruction | 68.3 |
| Top warp stall | MIO throttle, 26.7 cyc |

Top limiter, named: the L1 / TEX (shared memory read) pipe at 94.4 percent, with
MIO throttle as the dominant stall. The tensor pipe is only 53.7 percent utilized,
so the tensor cores are not the bottleneck; they are starved. The generic WMMA
`load_matrix_sync` issues many shared memory transactions to assemble each 16 by
16 fragment, and at a 64 by 64 block tile there is not enough tensor work per
fragment load to hide them, so warp cycles per issued instruction balloon to 68.3.
DRAM is only 10 percent, so this is a shared feeding problem, not a global one.

Single change to apply next: the mma.sync PTX path with ldmatrix. ldmatrix loads a
full 16 by 16 fragment per warp with one wide, swizzled shared transaction instead
of the many scalar loads the WMMA abstraction emits, which directly relieves the
L1 / TEX pipe and feeds the tensor cores. Larger block tiles and cp.async double
buffering are the further levers for the top kernel on the way to the compute
bound gate.

## Round 5: mma.sync PTX, isolating the instruction path from the load path

Date: 2026-07-19. The mma.sync kernel uses the same 64 by 64 block tile as the
WMMA kernel and the same shared staging, but replaces the WMMA fragment API with
direct mma.sync.aligned.m16n8k16 and manual, indexed shared loads for the
fragments (no ldmatrix yet). This is deliberately a like for like swap so the
round isolates the instruction path.

Result (this machine, FP16): the mma.sync kernel matches the WMMA kernel within
noise, about 36.4 TFLOP/s at 4096 cubed (57 to 60 percent of the FP16 cuBLAS
oracle), the same as WMMA's 36.3 TFLOP/s.

Reading: this is the honest and informative outcome. Round 4 named the limiter as
the shared read (L1 / TEX) pipe, not the WMMA abstraction. Removing the WMMA
abstraction while still assembling fragments with scalar shared loads leaves that
limiter untouched, so performance does not move. The mma.sync instruction was
never the bottleneck; the fragment loads are. This confirms the Round 4 diagnosis
by controlled experiment: it is the load path, not the compute instruction, that
gates the tensor kernel.

Single change that would matter (ldmatrix): replace the scalar fragment loads with
`ldmatrix.sync.aligned.m8n8.x4`, which fetches a full 16 by 16 fragment per warp in
one swizzled shared transaction and is the specific relief for the L1 / TEX pipe.
Combined with a larger block tile and cp.async double buffering, this is the route
toward the compute bound gate. The mma.sync kernel is kept as the honest PTX rung
that proves the instruction path was not the problem.

## Round 6: ldmatrix at 64 by 64, why the load path fix needs a bigger tile

Date: 2026-07-19. Artifacts: `experiments/results/ncu/round06/` (mma_ldm at 4096).

Change from Round 5: swap the scalar fragment loads for ldmatrix.x4 (A) and
ldmatrix.x2.trans (B), keeping the 64 by 64 tile so the round isolates ldmatrix.
Correctness holds (kernel versus oracle about 1e-7 on every shape).

Result: essentially unchanged, about 36.4 TFLOP/s at 4096 (60 percent of cuBLAS),
the same as the manual mma kernel. The metric snapshot explains why: memory Speed
of Light 91.1 percent, L1 / TEX pipe 95.1 percent, compute 52.5 percent, DRAM 11
percent, occupancy 32.9 percent, warp cycles per issued 77.2.

Reading: ldmatrix cut the instruction count for fragment assembly, but the L1 / TEX
pipe is still at 95 percent, so the kernel did not move. At a 64 by 64 tile the
kernel stages about 2048 halves of A and B to do a 64 by 64 by 16 block of MACs;
the ratio of bytes moved to fused multiply adds is too high, so the shared and
staging traffic saturates the pipe regardless of how efficiently each fragment is
loaded. The fix is not a better load instruction, it is a larger tile that reuses
each staged value more before restaging.

Single change to apply next: a 128 by 128 tile with K step 32 and cp.async double
buffering. That stages about 8192 halves to do a 128 by 128 by 32 block of MACs,
roughly a fourfold better bytes to MAC ratio, and cp.async overlaps the next
stage's global load with the current stage's math.

## Round 7: top kernel, 128 by 128 tile with ldmatrix and double buffering

Date: 2026-07-19. Artifacts: `experiments/results/ncu/round07/` (mma_opt at 4096)
and `round07b/` (at 8192).

Change: 128 by 128 block tile, K step 32, eight warps in a 2 by 4 layout (each warp
a 4 by 4 grid of 16 by 8 mma tiles), ldmatrix fragment loads, cp.async double
buffering. Correctness holds on every shape (kernel versus oracle about 1e-7).

Result: a large step. At 4096 cubed the kernel reaches 48.6 TFLOP/s, 79.8 percent
of cuBLAS, up from 36.4 TFLOP/s and 60 percent for the 64 by 64 kernels. At 8192 it
reaches 51.3 TFLOP/s, 79.4 percent. The FP16 top kernel is now 3.0 times the top
FP32 kernel (16.2 TFLOP/s).

Metric snapshot at 4096 cubed, versus the 64 by 64 ldmatrix kernel from Round 6:

| metric | ldmatrix 64x64 | top 128x128 |
|---|---|---|
| Achieved GFLOP/s | 36411 | 48571 |
| Percent of FP16 cuBLAS | 60.2 | 79.8 |
| Speed of Light, memory (percent) | 91.1 | 75.6 |
| L1 / TEX pipe throughput (percent) | 95.1 | 79.9 |
| Speed of Light, compute (percent) | 52.5 | 72.5 |
| DRAM throughput (percent) | 11.0 | 13.2 |
| Achieved occupancy (percent) | 32.9 | 32.6 |
| Warp cycles per issued instruction | 77.2 | 40.6 |

Compute bound gate assessment (honest): the larger tile did what the roofline
predicted. The Speed of Light flipped from clearly memory bound (compute 52.5,
memory 91.1) to co limited (compute 72.5, memory 75.6), and Nsight now names the
Tensor pipe as the highest utilized pipeline. Warp cycles per issued instruction
nearly halved (77.2 to 40.6), so the double buffering is hiding much more latency.
But the gate as defined in Section 5 asks for SM or tensor utilization clearly
above memory system utilization, and at 72.5 versus 75.6 it is not clearly above:
the kernel is compute and memory co limited, not clearly compute bound. I report it
that way rather than overclaiming a passed gate.

Remaining gap to a clean gate pass and to 90 percent of cuBLAS, attributed to named
causes with evidence:

1. L1 / TEX pipe at about 80 percent is now the shared read (ldmatrix) traffic, not
   the global path (DRAM is 13 percent at 4096). Reducing it further needs more
   register reuse per shared read, which the 4 by 4 warp fragment grid already
   pushes near the register budget (104 registers per thread, 32.6 percent
   occupancy), or a swizzled shared layout to cut any residual bank conflicts.
2. Occupancy is 32.6 percent and warp cycles per issued is still 40.6, so there is
   residual latency exposure. A three or four stage cp.async pipeline (this kernel
   uses two) would hide more of the ldmatrix to mma dependency.
3. At 8192 cubed the FP16 operands (128 MB each) exceed the 48 MB L2, so DRAM rises
   to 64 percent (memory Speed of Light 79.6, compute 77.6); there the kernel is
   closer to bandwidth co limited and would benefit from L2 aware tiling or split K,
   which is where cuBLAS's per shape tuning pulls ahead.

Named levers for further rounds (multistage pipeline, shared swizzle, split K,
per shape tile selection) are the difference between this hand written kernel at 80
percent and cuBLAS. Rounds 8 and 9 test two of them.

## Round 8 (failed, reverted): three stage cp.async pipeline

Date: 2026-07-19. Hypothesis: the residual latency exposure (warp cycles per issued
40.6 at 32.6 percent occupancy) is a too shallow software pipeline, so deepen
cp.async from two stages to three. Result: slower, 74.5 percent of cuBLAS at 4096
(down from 79.8). Root cause: the third shared buffer raised shared use to 48 KB
and cut occupancy, and the two stage pipeline was already hiding the global latency
(DRAM only 13 percent), so depth was spent on the wrong bottleneck. Reverted. Full
write up in `docs/ENGINEERING_LOG.md`. This is why the next round targeted the
shared read pipe directly instead.

## Round 9: shared memory swizzle, the gate passes

Date: 2026-07-19. Artifacts: `experiments/results/ncu/round09/` (mma_opt at 4096).

Measurement that motivated it: a bank conflict counter on the Round 7 kernel showed
`l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum` at 219 million against
269 million shared load wavefronts, about 0.8 conflicts per wavefront. The plain
row major shared tile makes ldmatrix collide: consecutive rows are 64 bytes (A) or
256 bytes apart, so lanes reading a column of the tile pile onto the same banks.

Change: swizzle the eight half (sixteen byte) chunk position within each shared
row by the row index (XOR), applied identically to the cp.async store and the
ldmatrix load, so it is a per row bijection and correctness is unconditional. Only
the bank mapping changes.

Result (4096 cubed): 54.9 TFLOP/s, 90.3 percent of cuBLAS, up from 48.6 TFLOP/s and
79.8 percent. At 8192 cubed: 58.4 TFLOP/s, 90.1 percent. Correctness holds on all
shapes (kernel versus oracle about 1e-7).

Metric snapshot at 4096 cubed, before and after the swizzle:

| metric | Round 7 (row major) | Round 9 (swizzled) |
|---|---|---|
| Achieved GFLOP/s | 48571 | 54931 |
| Percent of FP16 cuBLAS | 79.8 | 90.3 |
| Shared load bank conflicts | 219 million | 33.7 million |
| Shared load wavefronts | 269 million | 84 million |
| Speed of Light, compute (percent) | 72.5 | 82.7 |
| Speed of Light, memory (percent) | 75.6 | 30.2 |
| L1 / TEX pipe throughput (percent) | 79.9 | 31.8 |
| DRAM throughput (percent) | 13.2 | 14.5 |
| Warp cycles per issued instruction | 40.6 | 31.2 |

Compute bound gate: PASSES. The swizzle cut both the conflicts (219 to 33.7
million) and the total shared load wavefronts (269 to 84 million), which collapsed
the L1 / TEX pipe from 79.9 to 31.8 percent and the memory Speed of Light from 75.6
to 30.2 percent, while compute rose to 82.7 percent. Compute utilization (82.7) is
now clearly above memory utilization (30.2), and Nsight names the Tensor pipe the
highest utilized pipeline. The kernel's operating point is firmly right of the
roofline ridge (arithmetic intensity far above the measured ridge of about 60
FLOP per byte, DRAM at 14.5 percent). This is unambiguously compute bound.

Stop condition (Section 8.1): met. The compute bound gate passes AND the kernel is
at least 90 percent of cuBLAS at both 4096 cubed (90.3) and 8192 cubed (90.1) for
FP16. The optimization loop for the top kernel stops here, at round 9, one change
per round throughout, every step backed by an ncu artifact.

On split K (the other candidate lever): it is not needed here and would not help
these shapes. Split K raises parallelism when there are too few output tiles to
fill the SMs; at 4096 and 8192 with 128 by 128 tiles there are 1024 and 4096
output blocks against 48 SMs, so the machine is already saturated and the kernel is
compute bound. Split K is the right tool for small output, large K shapes, and is
noted for that case rather than applied where it cannot help.

## Round 10: mainloop rework for 1.1.0, four changes, measurement pending

Date: 2026-09-01. Artifacts: pending (owner sweep). Nothing in this entry is a
measured result; the four changes below are structural, they are verified by
tests and by the sanitizers, and every performance claim about them is a
hypothesis until the locked clock ladder is re-run.

**A tile swizzle, replaced.** Round 9's map XOR-ed the four chunk index of a
shared A row with the row index. An A row is 32 halves, 64 bytes, half a bank
window, so four chunks cannot separate the eight lanes of an `ldmatrix.x4`
wavefront and the map was a two way conflict on every A read. The new map packs
two logical A rows into one 128 byte line and permutes the eight chunks of that
line by the line index, which is the same trick that already worked for B. The
tile is the same size and the same map is applied at the `cp.async` store and at
the `ldmatrix` load, so correctness is unconditional. Both maps now live in
`src/gemm/detail/gemm_swizzle.hpp` and the GEMM tensor suite checks them on the
host: bijection, 16 byte chunks preserved, and distinct banks across every
wavefront the mainloop issues. Effect on the conflict counter: measurement
pending (owner sweep).

**Three stage pipeline, one barrier per K stage.** Round 8 tried a third stage,
lost on wall clock, and was reverted with the diagnosis that shared memory had
cut occupancy. Round 9's ncu page says that was arithmetically wrong: the block
limit was already set by registers. The reason to want a third stage is not
deeper latency hiding, it is that the buffer written at iteration s was last read
at iteration s-1 and is therefore already fenced by the iteration-s barrier, so
the loop needs one `__syncthreads` per K stage instead of two. The loop now
waits, syncs, issues the stage that is two ahead, and runs the mma, with no
trailing sync. Three stages exceed the 48 KB static `__shared__` ceiling on plain
sm_120, so the tile moved to dynamic shared memory and the launcher raises
`cudaFuncAttributeMaxDynamicSharedMemorySize` once per process after checking
`sharedMemPerBlockOptin`; a device that cannot give the block its budget gets
`kNotSupported` rather than a quieter, slower path. Effect on runtime:
measurement pending (owner sweep).

**Epilogue store width.** The SASS for the round 9 epilogue was scalar `STG.E`,
one float per store. The `m16n8k16` accumulator hands each lane two adjacent
columns of two rows per tile, so a `float2` is the widest store the fragment
layout allows without exchanging accumulators through shared memory first; the
epilogue now issues that pair explicitly and the SASS is `STG.E.64` throughout.
The launcher checks that C is eight byte aligned and refuses otherwise, because
that is a caller error worth naming. A 128 bit epilogue would need a shared
memory exchange and belongs to the register and epilogue study, not here. Effect
on runtime: measurement pending (owner sweep).

**B fragments.** B was read with `ldmatrix.x2.trans`, one instruction per 16 by 8
mma tile. It is now `ldmatrix.x4.trans`, one instruction per two neighbouring
tiles, which halves the B instruction count per K substep, and the B fragments
are double buffered in registers so the loads for substep k+1 are in flight while
the mma of substep k runs. The register file absorbed both without spilling and
the block limit is unchanged. Effect on runtime: measurement pending (owner
sweep).

Verification for the round: the full `ctest -L gpu` set is green, including the
new host side swizzle tests and the existing alpha and beta, non finite C, and
fusion suites; `memcheck`, `racecheck`, `initcheck` and `synccheck` are all clean
on the tensor GEMM suite; and racecheck was shown to go red first, by deleting
the single barrier from the new loop, before it was shown to go green with the
barrier back.

## Round 11: tile family, predicated tails, split-K, stream-K, measurement pending

Date: 2026-09-01. Artifacts: pending (owner sweep). Nothing in this entry is a
measured throughput result. The four changes below are structural, they are
verified by the correctness suites and by the four sanitizers, and every
statement about which one is faster is a hypothesis until the locked clock tile
sweep is run.

**A tile family, not a tile.** Round 10's mainloop is one block shape. It is now
a template on `(BM, BN, BK, warps in M, warps in N)` with six instantiations:
128x128x32, 128x256x32, 256x128x32, 128x64x64, 64x128x64 and 128x128x64. Every
warp tile is 64 by 32 across the family, so the accumulator stays at 64 registers
per thread and the shapes differ in shared memory footprint and in how they
quantize against 48 SMs rather than in register pressure. `nvcc
--resource-usage` at sm_120 reports 122, 124, 128, 162, 164 and 126 registers for
the aligned data parallel instantiations and 120, 122, 122, 160, 122 and 124 for
the predicated ones, and 128, 127, 128, 168, 168 and 160 aligned against 126,
128, 127, 132, 166 and 128 predicated for the persistent stream-K forms of the
same six shapes. Every one reports a zero byte stack frame and zero spill loads
and stores. The constraints each shape has to respect are
`static_assert`s rather than comments: three stages inside the 99 KB per block
opt-in, warps inside the 48 an SM carries, the staging work dividing evenly
across the block, and a warp tile the `m16n8k16` shape divides. Which shape wins
at which aspect ratio: measurement pending (owner sweep).

**The fallback cliff, closed.** Any shape not divisible by 128, 128, 32 used to
reroute to the WMMA kernel, and anything not divisible by 64, 64, 16 rerouted
again to a scalar kernel with one thread per output element. Four of the seven
tensor test shapes missed the fast path, so those rows were re-testing the
fallback under different names. The family predicates its edges: global to shared
uses `cp.async`'s source size field, so a sixteen byte chunk partly past an edge
copies the bytes that exist and zero fills the rest and one wholly past an edge
copies nothing, and zeros in the staged tile contribute nothing to the dot
product. The mainloop itself carries no edge logic at all.

One case defeated `cp.async` outright and was found by the correctness matrix
rather than by reasoning: a sixteen byte copy needs a sixteen byte aligned
source, and a row of A begins at a multiple of k, so an odd k puts every second
row on an odd address. 127 cubed is one of the shapes Section 9.5 names. Those
shapes stage through plain loads and shared stores under a flag the launcher
computes from the leading dimensions and the operand base addresses; the pipeline
barrier that already separates the stage writing a buffer from the iteration
reading it covers a synchronous store just as well as an asynchronous one. Cost
of the predicated path against the aligned one: measurement pending (owner
sweep).

**Split-K, two pass.** The first pass writes raw partial sums, one m by n plane
per K slice, and never touches C; the second sums the planes, multiplies by alpha
once and adds beta times C once, so beta zero still does not read C and every
partial is scaled exactly once. Atomics into C would need no scratch but would
make the result depend on the order blocks retired, which is not something a
correctness test can pin down. `gemm_workspace_size` is real now: it answers with
the partial planes, and a Context holding a workspace that large keeps the
allocation out of the call. Speedup where waves are scarce: measurement pending
(owner sweep).

**Stream-K, hybrid.** A bounded number of persistent CTAs take an equal share of
the K iterations of the tiles that do not fill a whole wave, at most two waves of
tiles between them, and every remaining tile runs data parallel one CTA per tile.
A CTA whose share stops before the end of a tile publishes its partial through
the workspace, fences, and raises a flag; the CTA covering that tile's last K
iteration waits on every peer whose share ended inside it, sums them in, and
writes C once.

The first version of this had the direction of the wait the other way round: the
CTA owning the tile's *first* K iteration waited on higher indexed peers. It is
correct only if every stream-K CTA is co-resident, which is what the occupancy
API is supposed to guarantee, but the occupancy the API reports is the
uninstrumented kernel's, and a profiler that patches the kernel, or a later
change to its register count, would quietly turn that into a hang rather than a
wrong answer. Waiting downward instead needs no residency assumption at all:
blocks are dispatched in increasing index order, so a resident owner implies
every peer it can wait on is resident or already retired. Speedup on small and
odd shapes: measurement pending (owner sweep).

**Dispatch.** `Algo::kAuto` on FP16 in and FP32 out now goes to the family for
every shape it can address, and the Section 9.6 rule decides which tile: maximize
`waves / ceil(waves)` with `blocksPerSM` from the occupancy API, among the tiles
within five percent of the best measured throughput at the nearest swept shape,
escalating to split-K or stream-K when waves is below one. The measured half of
that rule needs a committed sweep and there is none, so the choice currently
falls back to quantization alone. That is a hypothesis, and the API says so:
`ckl::GemmPlan::tuned` is false for a quantization only decision.

Open question for the sweep, recorded here so it is not forgotten: the
quantization only fallback prefers the tile that leaves the least of the final
wave idle, which at large square shapes is the smallest tile in the family. That
is very likely the wrong answer on throughput, because a small tile stages more
bytes per multiply add, and it is exactly the tradeoff the throughput filter
exists to settle. Until the sweep is committed, `kAuto` on a large FP16 square
may pick a slower shape than the 128 by 128 tile the previous release used. The
first thing to check after the sweep runs is whether the committed table moves
those shapes back.

Verification for the round: `ctest -L gpu` fully green, including every tile
instantiation against the cuBLAS oracle at the Section 9.5 unaligned matrix
(127 cubed, 1000 cubed, 4096 by 4090 by 4096, 129 by 257 by 193, and 8192 by 64
by 4096), the alpha and beta matrix, a non finite C under beta zero, the
workspace contract, the templated shared memory maps checked on the host for
every family shape, and the heuristic checked against a fixture sweep. All four
compute-sanitizer tools clean on both GEMM suites. Racecheck was shown to go red
first, by deleting the barrier the stream-K persistent loop puts between one
tile's reads and the next tile's stores, before it was shown to go green with the
barrier back.

## Round 12: GEMM current state at the locked clock, and the conflict counter A6 owes

Date: 2026-09-04. Artifacts: `experiments/results/ncu/round12/` (`mma_opt_4096.txt`,
`mma_opt_8192.txt`, `round_meta.txt`).

Why the round exists: defect A6 in `docs/CORRECTIONS.md` records that the 219
million to 33.7 million bank conflict result, the causal centre of the whole GEMM
story, appears in no committed text page, and that rounds 5 and 8 have no
artifacts at all. Round 9's numbers also predate the round 10 mainloop rework,
which replaced the A swizzle outright. So this round profiles the kernel that is
in the tree now, at the clock the committed sweep ran at.

Method note, and it changes every page from here on. The ncu default
`--clock-control base` takes the clocks away from the external lock: it ran the SM
at 2.60 GHz while `nvidia-smi` held 2497 MHz. `benchmarks/run_ncu_round.sh` now
passes `--clock-control none`, so the profiled kernel runs at the clock every
committed sweep row was measured at, and ncu prints its unmodified clocks warning
on each page. The pages report 2.50 GHz at 4096 and 2.39 GHz at 8192; the second
is a droop under a 20 ms kernel, not a different lock. ncu serializes kernels, so
no duration on any page in rounds 12 to 17 is quoted as performance.

Bank conflicts on `gemm_mma_opt_kernel`:

| where | shared load bank conflicts | shared load wavefronts | conflicts per wavefront |
|---|---|---|---|
| round 7 at 4096 (v1, log only, no artifact) | 219,000,000 | 269,000,000 | 0.81 |
| round 9 at 4096 (v1, log only, no artifact) | 33,700,000 | 84,000,000 | 0.40 |
| round 12 at 4096 (committed page) | 106,296 | 50,437,944 | 0.0021 |
| round 12 at 8192 (committed page) | 465,929 | 403,119,113 | 0.0012 |

The round 10 map, two logical A rows packed into one 128 byte line with the eight
chunks of that line permuted by the line index, takes the conflict count at 4096
from round 9's 33.7 million to 106,296, a factor of 317, and it takes the shared
load wavefront count from 84 million to 50.4 million at the same time. The near
zero this round was expected to find is what the page shows: 0.21 percent of
shared load wavefronts conflict at 4096 and 0.12 percent at 8192.

Limiter at 4096: the tensor pipe. Compute (SM) Throughput 90.38 percent against
Memory Throughput 26.19 percent, DRAM Throughput 8.57 percent, L1/TEX Cache
Throughput 23.02 percent, and Nsight names Tensor the highest utilized pipeline at
90.4 percent. Measured `dram__bytes.sum` is 142.26 MB against 134.2 MB of
compulsory A, B and C traffic, so L2 absorbs essentially every tile re-read (L2
hit rate 95.71 percent). What is left is latency on the math pipe: 29.81 warp
cycles per issued instruction, 55.4 percent of that the math pipe throttle stall,
at 32.66 percent achieved occupancy that 122 registers per thread and 49.15 KB of
dynamic shared memory per block both pin at two blocks per SM.

Limiter at 8192: DRAM, and the cause is L2 residency rather than anything in the
mainloop. Compute (SM) Throughput 92.46 percent, but Memory Throughput is 70.91
percent and the L2 hit rate falls from 95.71 percent at 4096 to 48.46 percent.
Measured traffic is 9.21 GB against 536.9 MB compulsory, 17.2 times the minimum.
The stall picture is unchanged (30.34 warp cycles per issued instruction, 55.5
percent math pipe throttle, 33.15 percent achieved occupancy), so nothing about
the mainloop degraded; the tile stream stopped fitting in L2. That is item (c) of
Section 9.4 of the build spec, the L2 rasterization swizzle with a persistence
window, and it has not been implemented. It is the next GEMM round.

## Round 13: what cuBLAS actually runs on this part

Date: 2026-09-04. Artifacts: `experiments/results/ncu/round13/`
(`cublas_fp16_1024.txt`, `cublas_fp16_4096.txt`, `cublas_fp16_8192.txt`,
`round_meta.txt`). This is the profile-cuBLAS round Section 9.4 asks for, run
through `ncu_driver` so the shapes, the buffers and the launch count match the
hand kernel pages exactly.

The spec predicted Ampere era `cutlass_80_tensorop_h16816gemm` kernels. That is
not what is there. Every shape lands on an `nvjet` kernel compiled for this
architecture:

| m = n = k | selected kernel | tile | stages | warp tile | registers | threads | dynamic shared | Compute SOL |
|---|---|---|---|---|---|---|---|---|
| 1024 | `nvjet_sm120_hss_mma_128x176x64_2_32x88x64_tmaAB_alignCD4_bz_NNNN` | 128x176x64 | 2 | 32x88x64 | 255 | 256 | 78.85 KB | 86.53 |
| 4096 | `nvjet_sm120_hss_mma_128x80x64_3_32x40x64_tmaAB_alignCD4_bz_NNNN` | 128x80x64 | 3 | 32x40x64 | 255 | 256 | 80.90 KB | 94.69 |
| 8192 | `nvjet_sm120_hss_mma_256x128x64_2_64x64x64_tmaAB_alignCD4_bz_NNNN` | 256x128x64 | 2 | 64x64x64 | 255 | 256 | 99.33 KB | 97.91 |

Four things in those names and numbers are the gap, and none of them was visible
before this page existed.

1. `tmaAB`. cuBLAS stages both operands through TMA. `mma_opt` uses `cp.async`.
   Section 9.4 puts TMA mainloops out of scope for this release on the grounds
   that they are high effort for a mid tier hypothesis; the vendor kernel that
   beats it uses TMA on both operands at every shape measured.
2. Non power of two N tiles: 176 at 1024 and 80 at 4096. The tile family in this
   tree instantiates six power of two shapes. A 128 by 80 tile quantizes against
   4096 differently from anything the family can express.
3. 255 registers per thread against the 122 of `mma_opt`, at 16.67 percent
   theoretical occupancy against 33.33 percent. cuBLAS spends the whole register
   file on one block per SM and does not try to hide latency with warps at all.
4. Up to 99.33 KB of dynamic shared memory per block, which is the full
   `sharedMemPerBlockOptin` budget on this part. `mma_opt` takes 49.15 KB.

Limiter on the vendor side, for the record: the same one. Compute (SM) Throughput
86.53, 94.69 and 97.91 percent at 1024, 4096 and 8192, against Memory Throughput
24.07, 36.03 and 23.66 percent. cuBLAS at 8192 measures `dram__bytes.sum` of
2.87 GB where `mma_opt` measures 9.21 GB for the same arithmetic, and its L2 hit
rate is 78.49 percent against the 48.46 percent of `mma_opt`. The 8192 gap is a
scheduling and residency gap, and round 12 said the same thing from the other
side.

Bank conflicts on the vendor kernels, since the counter is on every page now:
33,792 at 1024, 527,360 at 4096 and 2,097,152 at 8192. `mma_opt` measures 106,296
at 4096 and 465,929 at 8192, so the hand kernel has the cleaner shared path of the
two by that counter. It is not where the remaining 5 percent lives.

`docs/cutlass.md` carries this evidence in its gap section; the prediction it was
written against is corrected there rather than deleted.

## Round 14: Gate D compute bound check, clause by clause

Date: 2026-09-04. Artifacts: no new capture. This round reads the round 12 pages
and the committed sweep rows in `experiments/results/summary.csv`, all of which
carry `clock_locked=true` and `median_sm_clock_mhz=2497`.

The Gate D compute bound definition, from Section 9.6 of the build spec: on the
committed ncu page for the top kernel at 4096 and 8192, Compute (SM) Throughput at
least 80 percent and at least 2 times Memory Throughput, and achieved GFLOP/s at
least 75 percent of `48 * f_locked * 512`. With the lock at 2.497 GHz that roof is
48 x 2.497 x 512 = 61,366 GFLOP/s and the bar is 46,025 GFLOP/s.

| clause | at 4096 | verdict | at 8192 | verdict |
|---|---|---|---|---|
| Compute (SM) Throughput at least 80 percent | 90.38 | pass | 92.46 | pass |
| Compute at least 2x Memory Throughput | 90.38 against 26.19, ratio 3.45 | pass | 92.46 against 70.91, ratio 1.30 | fail |
| Achieved GFLOP/s at least 75 percent of 61,366 | 54,602, which is 88.98 percent | pass | 56,375, which is 91.87 percent | pass |

Five of six clauses pass. The one that fails is the 2x memory clause at 8192, and
round 12 names the cause with numbers rather than leaving it as a shrug: the L2
hit rate at 8192 is 48.46 percent against 95.71 percent at 4096, so measured DRAM
traffic is 9.21 GB against 536.9 MB compulsory and Memory Throughput rises to
70.91 percent. Compute is still the higher of the two and still above 80 percent;
what the kernel does not have at 8192 is the 2 times headroom the gate asks for.

I am recording the miss rather than restating the gate. The fix is the change
Section 9.4 lists as item (c) and this release did not ship: an L2 rasterization
swizzle over groups of tiles plus a `cudaAccessPolicyWindow` persistence window,
aimed at exactly the hit rate that fell. Until that round runs, Gate D is one
clause short at 8192, and the README and the report say so.

One further note on the same clause. The 2x rule is a proxy for compute bound and
it is a weak one at large shapes: a kernel can be tensor pipe limited, as this one
is at 90 percent Compute SOL with Tensor named the top pipeline, and still show
high Memory Throughput because its footprint left L2. I am not amending the gate
to say that. The clause is failed as written, and the argument for a better clause
belongs to whoever proposes one in writing.

## Round 15: SpMV, where merge wins and what it pays for it

Date: 2026-09-04. Artifacts: `experiments/results/ncu/round15/`
(`spmv_merge_parabolic_fem.txt`, `spmv_vector_parabolic_fem.txt`,
`spmv_merge_webbase-1M.txt`, `spmv_vector_webbase-1M.txt`, `round_meta.txt`).
`parabolic_fem` is the regular matrix of the pair (m = n = 525,825, nnz =
3,674,625, `power_law=false`); `webbase-1M` is power law (m = n = 1,000,005,
nnz = 3,105,536).

The four numbers Gate S wants, read off the committed pages. Warp execution
efficiency is `smsp__thread_inst_executed_per_inst_executed.ratio` over 32; the
tail ratio is `sm__cycles_active.max` over `sm__cycles_active.avg`.

| kernel | matrix | warp exec efficiency | tail ratio | sectors per global load request | dram__bytes.sum |
|---|---|---|---|---|---|
| `spmv_merge_kernel` | parabolic_fem | 30.34 of 32, 94.8 percent | 245,183 / 236,944 = 1.035 | 11.07 | 46.20 MB |
| `spmv_merge_fixup_kernel` | parabolic_fem | 30.36 of 32, 94.9 percent | 28,486 / 27,282 = 1.044 | 4.00 | 8.09 MB |
| `spmv_csr_vector_kernel<8>` | parabolic_fem | 25.75 of 32, 80.5 percent | 205,513 / 204,296 = 1.006 | 4.40 | 33.65 MB |
| `spmv_merge_kernel` | webbase-1M | 29.71 of 32, 92.8 percent | 196,039 / 192,352 = 1.019 | 11.05 | 34.23 MB |
| `spmv_merge_fixup_kernel` | webbase-1M | 28.81 of 32, 90.0 percent | 68,178 / 65,107 = 1.047 | 4.00 | 11.16 MB |
| `spmv_csr_vector_kernel<4>` | webbase-1M | 19.98 of 32, 62.4 percent | 212,490 / 206,873 = 1.027 | 2.70 | 32.87 MB |

Limiter, both matrices and both kernels: DRAM. Memory Throughput is 65.96 and
64.85 percent on parabolic_fem and 67.38 and 56.79 percent on webbase-1M, against
Compute (SM) Throughput of 19.90, 28.42, 25.52 and 27.57 percent. Achieved
occupancy is 84 to 94 percent everywhere, so this is not an occupancy story.

The finding worth the round: on the power law matrix the vector kernel throws away
37.6 percent of its lanes (19.98 of 32) while merge throws away 7.2 percent, and
merge pays for that with a gather costing 11.05 sectors per global load request
against the 2.70 of the vector kernel. Merge trades coalescing for lane occupancy,
and on webbase-1M that trade wins by a wide margin in the committed sweep: merge
median 0.116256 ms against 0.595328 ms for `spmv_csr_warp`, a factor of 5.12. The
tail ratio, which is the metric merge exists to move, is 1.019 to 1.047 on every
kernel here, so load balance is not the differentiator at these two matrices. Lane
utilization is.

Model bytes. The plan's band is printed in every sweep row. Measured against it:

| variant | matrix | model band, bytes | measured total | position |
|---|---|---|---|---|
| merge | parabolic_fem | 35,706,904 to 48,302,104 | 54.29 MB | 12.4 percent above the high end |
| vector | parabolic_fem | 35,706,904 to 48,302,104 | 33.65 MB | 5.8 percent below the low end |
| merge | webbase-1M | 36,844,352 to 45,266,476 | 45.39 MB | 0.3 percent above the high end |
| vector | webbase-1M | 36,844,352 to 45,266,476 | 32.87 MB | 10.8 percent below the low end |

Both vector rows land under the low end of the band, which is the end that assumes
perfect x residency, so on a first reading the kernel moved less than the
compulsory minimum. It did not. The gap is the y write: 4m is 2,103,300 bytes on
parabolic_fem and 4,000,020 bytes on webbase-1M, and the measured shortfalls are
2.06 MB and 3.97 MB. The y plane is still dirty in L2 when the kernel retires, so
those writes have not reached DRAM inside the window the counter covers. The model
counts them because they will be written eventually. That is a real difference
between the model and `dram__bytes.sum` and it belongs in `docs/sparse.md` rather
than being smoothed over: at these sizes the band brackets the measurement to
within one y plane.

Merge sits above the high end on both matrices because the model has no term for
the carry arrays or for the second pass: the fixup kernel alone moves 8.09 MB and
11.16 MB. That is a real cost of the merge path and the model does not describe it.

Gate S clauses this round can settle:

| clause | evidence | verdict |
|---|---|---|
| every percent of roof row carries a measured `dram__bytes.sum` | the six rows above; `summary.csv` still reads `pending ncu round` in `dram_bytes_sum`, so the sweep has to be re-summarized to carry them | open: the numbers exist, the column does not hold them yet |
| `spmv_merge` at least 1.5 times `spmv_csr_warp` on both power law matrices, intervals disjoint | webbase-1M 0.595328 / 0.116256 = 5.12x, intervals [0.594928, 0.595456] and [0.116256, 0.118240] | pass on this matrix |
| the same clause on soc-LiveJournal1 | 2.984416 / 2.320832 = 1.286x, intervals [2.984256, 2.984512] and [2.320800, 2.320896] | fail |

The soc-LiveJournal1 miss is the risk the board recorded before the locked run, and
the locked run confirms it at 1.286x rather than the 1.33x measured unlocked. The
cause is not load balance: the merge tail ratio is already near one on the matrices
profiled here and its lane utilization is already near 30 of 32. It is the gather,
at 11 sectors per request, over a matrix whose column indices have no locality left
to exploit. Block level tile staging of x is the change to try, and it is a kernel
change rather than a gate amendment.

## Round 16: FFT model bytes, and why the ladder loses to cuFFT

Date: 2026-09-04. Artifacts: `experiments/results/ncu/round16/`
(`fft_radix8_20.txt`, `fft_cufft_20.txt`, `fft_radix8_24.txt`, `fft_cufft_24.txt`,
`fft_shared_12.txt`, `round_meta.txt`). The best hand rung at both sizes is
`radix8` by the committed medians (0.107552 ms at 2^20 and 3.721280 ms at 2^24,
flushed), so that is the rung profiled. Every launch of one settled call is on the
page, so the total traffic of a rung is the sum of its stages rather than one stage
generalized.

The Gate X 15 percent clause, measured against the model each row declares:

| variant | length | launches profiled | declared model bytes | measured dram__bytes.sum | deviation | clause |
|---|---|---|---|---|---|---|
| radix8 | 2^20 | 7, one radix 4 stage and six radix 8 | 117,440,512 | 64.08 MB | 45.4 percent low | fail |
| radix8 | 2^24 | 8 radix 8 stages | 2,147,483,648 | 1977.03 MB | 3.5 percent low | pass |
| cuFFT | 2^20 | 2 | not declared | 16.87 MB | compulsory is 16.78 MB | reference |
| cuFFT | 2^24 | 3 | not declared | 744.37 MB | compulsory is 268.44 MB | reference |

The 2^20 failure has one cause and it is L2. The ping pong buffers of the rung are
8 MB each at 2^20, so the whole working set sits inside the 48 MB L2 and six of the
seven stage boundaries never reach DRAM: 64.08 MB measured against a model that
assumes all seven do. At 2^24 the buffers are 128 MB each, the assumption of the
model holds, and the deviation drops to 3.5 percent. This is the clause doing its
job. Any effective GB/s quoted for this rung at 2^20 is inflated by the same
factor: the committed row reads 1091.942 GB/s, above the measured 579 GB/s DRAM
roof, and the reason it is above the roof is that 45 percent of the bytes in its
numerator were served by L2.

Limiter, radix8 at 2^24: DRAM, and the loss to cuFFT is a traffic loss and nothing
else. Each of the eight stages runs at 83.47 percent DRAM Throughput and 552.35
GB/s against 10.06 percent Compute (SM) Throughput, so the rung is at 95 percent of
the measured 579 GB/s roof on every pass it makes. It makes 8 passes where cuFFT
makes 3: 1977.03 MB against 744.37 MB, a factor of 2.66, and the committed medians
are 3.721280 ms against 1.426944 ms, a factor of 2.61. The two ratios agree to
within 2 percent, and that is the whole diagnosis. Radix 8 is the wrong lever here.
The number of global passes is the lever, which is what the four step form and a
larger resident transform address.

Shared bank conflicts on the resident rung, which Section 12.2 named as the
expected limiter for this family: `shared_kernel` at 2^12 with batch 96 measures 0
conflicts against 319,488 shared load wavefronts. The predicted stride 2^s conflict
in the butterfly exchange is not there. The real constraint on that rung is
occupancy: 65.54 KB of dynamic shared memory per block gives one block per SM and
16.65 percent achieved occupancy, at 34.04 percent Memory Throughput and 26.41
percent Compute SOL, so it is latency bound with nothing to hide the latency
behind. For contrast, `regular_fft_factor` from cuFFT at 2^24 measures 27,073
conflicts against 1,075,649 shared load wavefronts, 2.5 percent, at 64 registers
and 48.64 percent achieved occupancy.

Gate X status after this round:

| clause | verdict |
|---|---|
| measured `dram__bytes.sum` within 15 percent of the declared model | fail at 2^20, 45.4 percent low on an L2 resident working set; pass at 2^24, 3.5 percent low |
| best hand rung at 55 percent of the measured 579 GB/s roof | passes on the committed rows, but the effective GB/s of the 2^20 row is not usable evidence until the model accounts for L2 residency |
| best hand rung at 70 percent of cuFFT at 2^20 to 2^24 | fail: 60.5 percent at 2^20 and 38.3 percent at 2^24, cause measured above |
| shared bank conflicts in the butterfly exchange | measured: 0 on the resident rung, so the expected limiter is absent |
| 2D transpose share and 70 percent of roof | not in this round, still pending |
| crossover chart from committed rows | not in this round, still pending |
| callback probe and toolkit version recorded | already green in `docs/fft.md` |

## Round 17: scan and reduction traffic, and the 5 percent CUB clause

Date: 2026-09-04. Artifacts: `experiments/results/ncu/round17/`
(`reduce_vec4_26.txt`, `reduce_single_pass_26.txt`, `scan_lookback_26.txt`,
`scan_cub_26.txt`, `reduce_cub_26.txt`, `scan_blelloch_26.txt`, `round_meta.txt`),
all at N of 2^26.

Traffic against the declared model:

| variant | launches | declared model bytes | measured dram__bytes.sum | deviation |
|---|---|---|---|---|
| reduce vec4 | 2 | 268,437,760 | 270.67 MB | 0.8 percent high |
| reduce single_pass | 1 | 268,437,760 | 273.95 MB | 2.1 percent high |
| reduce CUB | 2 | 268,435,456 | 276.99 MB | 3.2 percent high |
| scan lookback | 1 | 538,443,776 | 518.20 MB | 3.8 percent low |
| scan CUB | 2 | 536,870,912 | 516.21 MB | 3.8 percent low |
| scan blelloch | 7 | 1,073,741,824 | 1067.46 MB | 0.6 percent low |

Every row is inside 4 percent of its model, so unlike the FFT family this one has
no traffic accounting problem at the size the gate is stated at. The two scan rows
sit under their model by the same 3.8 percent, for the hand kernel and for CUB
alike, which is the output plane still partly dirty in L2 at kernel exit, the same
effect round 15 found on the SpMV y plane.

Gate R bandwidth clauses, from the committed sweep rows at locked clocks with L2
flushed, computed with the byte counts the gate names (`N*4` for reduction,
`2*N*4` for scan):

| clause | 2^26 | 2^27 | 2^28 | verdict |
|---|---|---|---|---|
| top reduction rung at least 521 GB/s | 596.46, vec4 | 608.56, single_pass | 614.80, single_pass | pass |
| top scan rung at least 492 GB/s | 543.00, lookback | 546.09, lookback | 550.07, lookback | pass |
| top reduction rung within 5 percent of CUB | 101.37 percent | 100.47 percent | 100.19 percent | pass |
| top scan rung within 5 percent of CUB | 94.15 percent | 94.52 percent | 95.49 percent | fail at 2^26 and 2^27, pass at 2^28 |

The scan miss is 0.85 points at 2^26 and 0.48 at 2^27, and this round says what it
is made of. It is not traffic: lookback moves 518.20 MB against 516.21 MB for CUB,
0.4 percent more. It is tile size. `lookback_kernel` runs 256 threads at 8 items
each, so 2,048 elements per tile over 32,768 blocks, at 34 registers and 98.42
percent achieved occupancy. `DeviceScanKernel` from CUB runs 416 threads over 8,491
blocks, which is 7,904 elements per tile, at 48 registers and 78.78 percent
achieved occupancy. CUB carries 3.9 times more data per tile, so it pays the
decoupled look back handshake 3.9 times less often, and the counters show it:
5,368,287 shared load wavefronts against 5,788,563, and 290,681 bank conflicts
against 480,147. DRAM Throughput lands at 86.98 percent for CUB and 82.04 percent
for lookback. Raising items per thread on the look back rung is the change to try,
and it is a tuning change rather than a redesign.

The s1 padding page. The bank conflict question Section 12.3 asks of the Blelloch
rung has no pre padding page committed anywhere in this repository, so there is no
before and after to show; what is committed now is the padded state.
`tile_scan_kernel<..., BlellochBlockScan, 0>` at 2^26 measures 138,405 conflicts
against 17,177,765 shared load wavefronts, 0.81 percent. The padding works. What
the page also shows is the cost of the rung itself: 17.2 million shared load
wavefronts against 5.8 million for the look back rung at the same N, and 1067.46 MB
of DRAM traffic against 518.20 MB, which is why blelloch sits at 37.53 percent of
CUB in the committed rows. The work efficient formulation is not paying for itself
on this part, and that is a measured statement now rather than a suspicion.

Two Gate R clauses are outside this round and unchanged: the sanitizer and
determinism clauses are green and recorded in `docs/scan.md`, and the tolerance
clauses are green in the test suite.
