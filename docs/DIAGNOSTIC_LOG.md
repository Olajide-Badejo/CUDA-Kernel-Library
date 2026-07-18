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

## Round 1 (staged): tiled versus naive, why tiling alone does not win

Date: staged 2026-07-19, ncu capture pending the Windows side counter permission.

Wall clock going in (this machine, 20 reps, median): at 4096 cubed the naive
kernel reaches 1910 GFLOP/s and the 32 by 32 tiled kernel reaches 1588 GFLOP/s,
so tiling is about 17 percent slower than naive, not faster. Both sit near 8
percent of cuBLAS.

Hypothesis to test with ncu: the tiled kernel is limited by occupancy and shared
memory latency, not by global bandwidth. The 32 by 32 tile is a 1024 thread
block, so only one block fits per SM (1536 threads per SM), about 66 percent
occupancy, against 100 percent for the naive 16 by 16 block; both kernels are
latency bound, so the lower occupancy hides less latency. Meanwhile the 48 MB L2
already absorbs the naive kernel's repeated B reads, so shared staging adds
latency and two syncthreads per K step without lifting arithmetic intensity per
thread (still one output per thread).

Metrics to capture when counters are available: Speed of Light (SM vs memory),
`sm__warps_active.avg.pct_of_peak_sustained_active` (achieved occupancy),
`smsp__average_warps_issue_stalled_*` (top stall reason, expecting a barrier or
MIO/short scoreboard signature), and `l1tex`/`lts` hit rates to confirm the L2
already serves naive. Planned single change after the round: move to register
blocking (Phase 3), many outputs per thread, which raises arithmetic intensity
per thread and is the expected decisive step.

Report artifact will land in `experiments/results/ncu/round01/` and be linked
here once captured.
