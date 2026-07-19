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
percent and cuBLAS. The kernel is accepted here as the top tensor variant at about
80 percent of cuBLAS with a co limited roofline; the gap to 90 percent and to a
clean compute bound gate is documented above with metric evidence, per the Section
8.1 stop condition's diagnosis branch.
