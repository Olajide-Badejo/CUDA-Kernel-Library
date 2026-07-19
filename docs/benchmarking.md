# Benchmarking

## Timing protocol

Every timed measurement uses CUDA events with the same protocol
(`include/ckl/event_timer.hpp`): five warmup launches, then twenty timed reps,
reported as the median with the interquartile range so a single slow rep does not
move the headline. GEMM throughput is `2 m n k` FLOPs over the median time; GEMV
and SpMV report the same way, and GEMV is also read as effective bandwidth because
it is memory bound.

## The sweep

`benchmarks/bench_all.cpp` runs one configuration (family, variant, dtype, shape),
times both the kernel and its vendor baseline in the same process with NVML
sampling on, and prints one JSON row: timing, GFLOP/s, percent of baseline, the
NVML summary (median clock, peak temperature and power, throttle flag), toolkit and
driver versions, and the git commit.

`benchmarks/sweep.py` drives bench_all across the declared matrix (60
configurations across the four families), appends rows to
`experiments/results/sweep.jsonl`, and refreshes the one committed canonical
`experiments/results/summary.csv`. It is resumable: a configuration whose
(family, variant, dtype, shape, commit) row already exists is skipped, so a killed
sweep continues without repeating finished work. `--force` redoes, `--quick` runs a
representative subset, `--list` prints the matrix. Run it with `make sweep`.

Every row carries the commit that produced it, and every throttled row is flagged;
per the build rule a throttled row is rerun after cooldown rather than trusted. The
last full sweep completed with no throttled rows.

## Epilogue fusion study

`benchmarks/fusion_study.cpp` answers one fusion decision with measurement. It
computes `C = relu(alpha * A*B + bias)` two ways: unfused (the top GEMM writes C,
then a separate kernel reads C, adds the column bias, applies ReLU, and writes C
again) and fused (the bias and ReLU are folded into the GEMM epilogue while C is
still in registers, so C is written once).

Measured on this machine at 4096 cubed: GEMM alone 2.50 ms, the separate epilogue
pass 0.19 ms, so the unfused total is 2.70 ms; the fused kernel is 2.47 ms. Fusion
saves about 0.23 ms, 8.4 percent of the unfused time. The reason is on the
roofline: the standalone epilogue has an arithmetic intensity of 0.167 FLOP/byte,
far to the left of the ridge (about 113), so it is memory bound and, as a separate
pass, is pure bandwidth overhead. Folding it into the compute bound GEMM removes
that pass at almost no cost, so fusion is the right call. This is the epilogue
fusion decision the roofline is used to justify.
