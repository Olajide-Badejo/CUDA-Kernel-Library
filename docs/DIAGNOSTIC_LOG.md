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
counter permission is cleared. No rounds recorded yet.
