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
