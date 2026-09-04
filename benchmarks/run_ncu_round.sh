#!/usr/bin/env bash
# Run one Nsight Compute diagnostic round and save its artifacts where the
# diagnostic log can point at them. Each argument after the round name is a
# "variant:arg" pair; the matching kernel is profiled on one settled call (the
# driver runs the variant several times and we skip to the last).
#
# Usage: run_ncu_round.sh <round_name> <variant:arg[:kernels_per_call]> ...
#   e.g. run_ncu_round.sh round01 naive:2048 tiled:2048 naive:4096 tiled:4096
#        run_ncu_round.sh round15 spmv_merge:parabolic_fem spmv_vector:webbase-1M
#        run_ncu_round.sh round16 fft_radix8:20:7 fft_cufft:20:2
#
# The argument is a square dimension for the GEMM variants, a suite matrix name
# for the SpMV ones, and log2 of the length for FFT, scan and reduction. The
# third field says how many kernel launches one call of that variant issues; it
# is 1 for most rungs and the table below carries the exceptions, but a rung
# whose launch count depends on the size (every FFT ladder) has to be told.
#
# Output: experiments/results/ncu/<round_name>/<variant>_<arg>.ncu-rep and a
# matching .txt with the detail page, plus round_meta.txt recording the commit,
# toolkit, and driver.

set -euo pipefail

export PATH=/usr/local/cuda/bin:$PATH

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <round_name> <variant:arg[:kernels_per_call]> ..." >&2
    exit 2
fi

round="$1"; shift
out_dir="experiments/results/ncu/${round}"
mkdir -p "$out_dir"

driver="./build/benchmarks/ncu_driver"
if [[ ! -x "$driver" ]]; then
    echo "building ncu_driver first" >&2
    cmake --build build --target ncu_driver >/dev/null
fi

# Curated section set: enough to name the top limiter without the full replay
# cost. SpeedOfLight for the memory vs compute split, Occupancy for the achieved
# warps, MemoryWorkloadAnalysis for the cache hit picture and the sectors per
# request rows, WarpStateStats for the top stall reason, SchedulerStats and
# LaunchStats for context.
sections=(
    --section SpeedOfLight
    --section Occupancy
    --section MemoryWorkloadAnalysis
    --section WarpStateStats
    --section SchedulerStats
    --section LaunchStats
    --section ComputeWorkloadAnalysis
)

# Explicit metrics on top of the sections, one list for every family, because a
# page that carries a counter nobody asked for costs a replay pass and a page
# that is missing one costs a re-run.
#
# The shared load bank conflict counter is the one that named the cause of the
# round 9 swizzle win, and in v1 it was only ever read interactively out of the
# binary .ncu-rep, which is gitignored. No text page in the repository carried
# it, so the central causal finding of the whole optimization story had no
# committed evidence: defect A6 in docs/CORRECTIONS.md.
#
# dram__bytes.sum is the number of record for every bandwidth claim this project
# makes. Gate X invalidates an FFT round whose measured traffic is more than 15
# percent off the declared model, and the Gate S rows want it beside every
# percent of roof, so it belongs on every page rather than on the FFT ones.
#
# The last four are the SpMV and gather questions: warp execution efficiency
# (predicated-off lanes in a ragged row distribution), the tail ratio
# sm__cycles_active.max over .avg (one SM still working while the rest have
# retired), and sectors per request on both sides of global memory.
metrics=(
    --metrics l1tex__data_bank_conflicts_pipe_lsu_mem_shared_op_ld.sum,l1tex__data_pipe_lsu_wavefronts_mem_shared_op_ld.sum,dram__bytes.sum,smsp__thread_inst_executed_per_inst_executed.ratio,sm__cycles_active.max,sm__cycles_active.avg,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_ld.ratio,l1tex__average_t_sectors_per_request_pipe_lsu_mem_global_op_st.ratio
)

# ncu's default --clock-control base takes the clocks away from whatever the
# machine had them set to. On this box that moved the SM clock to 2.60 GHz while
# nvidia-smi held the lock at 2497 MHz, so the profiled kernel was not running at
# the clock every committed sweep row was measured at. "none" leaves the external
# lock alone; ncu prints a warning about unmodified clocks, and that warning is
# the correct outcome here.
clock=(--clock-control none)

{
    echo "round: ${round}"
    echo "date: $(date -Is)"
    echo "commit: $(git rev-parse HEAD)"
    echo "nvcc: $(nvcc --version | tail -1)"
    echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo unknown)"
    echo "locked_clock_mhz: $(nvidia-smi --query-gpu=clocks.gr --format=csv,noheader 2>/dev/null || echo unknown)"
    echo "ncu: $(ncu --version | tail -1)"
    echo "clock_control: none (the external nvidia-smi lock is left in place)"
} > "${out_dir}/round_meta.txt"

for pair in "$@"; do
    variant="${pair%%:*}"
    rest="${pair#*:}"
    arg="${rest%%:*}"
    per_call=""
    if [[ "$rest" == *:* ]]; then
        per_call="${rest##*:}"
    fi

    # The tensor kernels share one templated implementation, wmma_gemm_kernel;
    # the FP32 rungs each have their own gemm_<variant>_kernel. An empty regex
    # means no kernel filter, which is how a vendor call gets profiled: the
    # driver launches nothing else, so there is nothing to filter out.
    default_per_call=1
    case "$variant" in
        wmma_fp16|wmma_bf16) kernel="wmma_gemm_kernel" ;;
        mma_ldm) kernel="gemm_mma_ldmatrix_kernel" ;;
        cublas_fp16|cublas_fp32) kernel="" ;;
        fft_cufft) kernel="" ;;
        spmv_cusparse|spmv_cusparse_alg2) kernel="" ;;
        spmv_merge) kernel="spmv_merge_kernel|spmv_merge_fixup_kernel"; default_per_call=2 ;;
        spmv_vector) kernel="spmv_csr_vector_kernel" ;;
        spmv_warp) kernel="spmv_csr_warp_kernel" ;;
        spmv_naive) kernel="spmv_csr_naive_kernel" ;;
        spmv_sell) kernel="spmv_sell_kernel" ;;
        spmv_bsr) kernel="spmv_bsr_kernel" ;;
        fft_radix2) kernel="stockham_radix2_stage" ;;
        fft_radix4|fft_radix8) kernel="radix[248]_stage" ;;
        fft_four_step) kernel="shared_cross_kernel|transpose_[a-z]*_kernel" ;;
        fft_shared) kernel="shared_kernel" ;;
        reduce_vec4) kernel="vec4_partials|combine_partials"; default_per_call=2 ;;
        reduce_single_pass) kernel="single_pass" ;;
        reduce_two_pass) kernel="partials_only|combine_only"; default_per_call=2 ;;
        reduce_cub) kernel="DeviceReduce"; default_per_call=2 ;;
        scan_lookback) kernel="lookback_kernel" ;;
        scan_three_kernel) kernel="tile_scan_kernel|propagate_kernel"; default_per_call=2 ;;
        scan_blelloch) kernel="tile_scan_kernel|propagate_kernel"; default_per_call=2 ;;
        scan_cub) kernel="DeviceScan"; default_per_call=2 ;;
        *) kernel="gemm_${variant}_kernel" ;;
    esac
    if [[ -z "$per_call" ]]; then
        per_call="$default_per_call"
    fi

    launches=5
    skip=$(( (launches - 1) * per_call ))
    base="${out_dir}/${variant}_${arg}"
    echo "profiling ${variant} at ${arg} (${per_call} launch(es) per call, kernel ${kernel:-any})"

    filter=()
    if [[ -n "$kernel" ]]; then
        filter=(--kernel-name "regex:${kernel}")
    fi

    ncu "${sections[@]}" "${metrics[@]}" "${clock[@]}" "${filter[@]}" \
        --launch-skip "${skip}" --launch-count "${per_call}" \
        --force-overwrite \
        --export "${base}" \
        "$driver" "$variant" "$arg" "$launches" >/dev/null

    # Human readable detail page for the diagnostic log.
    ncu --import "${base}.ncu-rep" --page details > "${base}.txt"
    echo "  wrote ${base}.ncu-rep and ${base}.txt"
done

echo "round ${round} complete; artifacts in ${out_dir}"
