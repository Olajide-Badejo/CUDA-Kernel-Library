#!/usr/bin/env bash
# Run one Nsight Compute diagnostic round and save its artifacts where the
# diagnostic log can point at them. Each argument after the round name is a
# "variant:size" pair; the matching kernel is profiled on one settled launch
# (the driver runs it several times, we skip to the last).
#
# Usage: run_ncu_round.sh <round_name> <variant:size> [variant:size ...]
#   e.g. run_ncu_round.sh round01 naive:2048 tiled:2048 naive:4096 tiled:4096
#
# Output: experiments/results/ncu/<round_name>/<variant>_<size>.ncu-rep and a
# matching .txt with the detail page, plus round_meta.txt recording the commit,
# toolkit, and driver.

set -euo pipefail

export PATH=/usr/local/cuda/bin:$PATH

repo_root="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$repo_root"

if [[ $# -lt 2 ]]; then
    echo "usage: $0 <round_name> <variant:size> [variant:size ...]" >&2
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
# warps, MemoryWorkloadAnalysis for the cache hit picture, WarpStateStats for the
# top stall reason, SchedulerStats and LaunchStats for context.
sections=(
    --section SpeedOfLight
    --section Occupancy
    --section MemoryWorkloadAnalysis
    --section WarpStateStats
    --section SchedulerStats
    --section LaunchStats
    --section ComputeWorkloadAnalysis
)

{
    echo "round: ${round}"
    echo "date: $(date -Is)"
    echo "commit: $(git rev-parse HEAD)"
    echo "nvcc: $(nvcc --version | tail -1)"
    echo "driver: $(nvidia-smi --query-gpu=driver_version --format=csv,noheader 2>/dev/null || echo unknown)"
} > "${out_dir}/round_meta.txt"

for pair in "$@"; do
    variant="${pair%%:*}"
    size="${pair##*:}"
    kernel="gemm_${variant}_kernel"
    base="${out_dir}/${variant}_${size}"
    echo "profiling ${variant} at ${size} cubed (kernel ${kernel})"

    ncu "${sections[@]}" \
        --kernel-name "regex:${kernel}" \
        --launch-skip 4 --launch-count 1 \
        --force-overwrite \
        --export "${base}" \
        "$driver" "$variant" "$size" 5 >/dev/null

    # Human readable detail page for the diagnostic log.
    ncu --import "${base}.ncu-rep" --page details > "${base}.txt"
    echo "  wrote ${base}.ncu-rep and ${base}.txt"
done

echo "round ${round} complete; artifacts in ${out_dir}"
