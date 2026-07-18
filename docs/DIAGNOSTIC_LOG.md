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
