#!/usr/bin/env python3
"""Drive the full benchmark sweep and write the canonical results.

This is the measurement protocol of the build spec's Section 13, in code. What
it adds over the v1 driver is not polish; each item below changed a number, not
only an error bar.

Locked clocks. The v1 sweep did not lock: there is no nvidia-smi -pm and no -lgc
anywhere in that history, and median_sm_clock_mhz in the resulting summary.csv
ranges from 1042 to 2880 MHz, which the ladder chart then presented as one apples
to apples sequence. That is defect A4. The preamble now enables persistence mode,
locks the graphics clock, verifies the lock by querying the GPU, stamps the
locked value into every row, and rejects any row whose measured median clock
drifted more than 2 percent. Locking needs privileges WSL2 may not have; if it
fails the sweep refuses to run unless --allow-unlocked is passed, and an unlocked
run stamps clock_locked=false on every row so no locked clock claim can be made
from it.

Independent process repeats. Every configuration runs --process-reps fresh
bench_all processes, five by default. The row of record carries all N medians,
the median of medians, and a seeded bootstrap 95 percent confidence interval, so
a reader can see the spread instead of trusting one number.

Vendor baselines measured once. v1 re-measured the baseline inside every variant
row and saw a 1.9x spread on identical cuBLAS calls, which is how percentages
above 100 of a vendor that was never beaten got into the report (defect A2). The
baseline is now its own configuration, run in its own process before the variants
at that shape, and every variant row references that one measurement.

Randomized order and cooldowns. v1 ran 60 configurations back to back, naive
first and mma_opt last, so the fastest kernels met the hottest GPU. The order is
now shuffled under a recorded seed and there is a cooldown between
configurations.

Throttle gating that fails. A throttled configuration is re-run once after a
longer cooldown. If it is still throttled the sweep exits non-zero and names the
rows. v1 printed a warning and carried on.

One commit per summary. refresh_summary filters the JSONL to a single commit and
asserts one row per key. v1 mixed commits and the report took the first match.

Run it with `make sweep`. --list prints the matrix without running anything,
--quick runs a representative subset, --force redoes finished work, and
--refresh-only rebuilds summary.csv from rows that already exist.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
import subprocess
import sys
import time
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
JSONL = RESULTS / "sweep.jsonl"
SUMMARY = RESULTS / "summary.csv"
BENCH_ALL = REPO / "build" / "benchmarks" / "bench_all"

# Version of the row this driver writes. A row without it came from the v1
# harness: it re-measured its own baseline and kept no samples, so it cannot be
# compared with a row that carries this field. refresh_summary reads both and
# says which is which through the baseline_source column.
SWEEP_SCHEMA_VERSION = 2

DEFAULT_LOCK_MHZ = 2500
# A row whose measured median clock drifts further than this from the lock is not
# comparable with the rest of the sweep.
CLOCK_TOLERANCE = 0.02

DEFAULT_PROCESS_REPS = 5
DEFAULT_COOLDOWN_S = 5.0
BOOTSTRAP_RESAMPLES = 10000

# sm_120 carries 48 MB of L2. A working set below that is resident across
# back to back reps, so those rows are measured both flushed and unflushed and
# the pair is what gets reported. See docs/benchmarking.md.
L2_BYTES = 48 * 1024 * 1024

# Shapes small enough that launch overhead is a visible fraction of the kernel
# time get a CUDA graph row next to the stream row.
GRAPH_SHAPE_MAX = 768
GRAPH_INNER = 20

# The vendor baseline for each family, as bench_all names the variant.
BASELINE_VARIANT = {
    "gemm": "baseline_cublas_default",
    "gemv": "baseline_cublas",
    "spmv": "baseline_cusparse",
    "trsm": "baseline_cublas",
}
# Measured alongside the baseline of record and reported next to it. Beating the
# heuristic pick is a weaker claim than beating the autotuned pick, so both are
# on the record and every percentage says which one it is against.
EXTRA_BASELINES = {"gemm": ["baseline_cublas_autotune"]}

DTYPE_BYTES = {"fp32": 4, "fp16": 2, "bf16": 2}

# Sweep matrix. Each entry: (family, variant, dtype, [shape (m, n, k), ...]).
GEMM_FP32_SIZES = [512, 1024, 2048, 4096]
GEMM_TENSOR_SIZES = [1024, 2048, 4096, 8192]
GEMM_SMALL_SIZES = [128, 256, 512, 768]
GEMV_SIZES = [1024, 2048, 4096, 8192]
SPMV_SIZES = [16384, 65536, 131072]
TRSM_SIZES = [512, 1024, 2048]


class Config:
    """One bench_all invocation, minus the process repeat index."""

    def __init__(self, family, variant, dtype, shape, flush="auto",
                 launch_mode="stream", inner=1):
        self.family = family
        self.variant = variant
        self.dtype = dtype
        self.m, self.n, self.k = shape
        self.flush = flush
        self.launch_mode = launch_mode
        self.inner = inner

    @property
    def baseline(self) -> bool:
        return self.variant.startswith("baseline_")

    def argv(self, bench: Path, commit: str) -> list[str]:
        return [str(bench), self.family, self.variant, self.dtype,
                str(self.m), str(self.n), str(self.k), commit,
                "--flush-l2", self.flush,
                "--launch-mode", self.launch_mode,
                "--inner", str(self.inner)]

    def spec(self) -> str:
        """Compact identity, which is what --only matches against."""
        return f"{self.family}:{self.variant}:{self.dtype}:{self.m}x{self.n}x{self.k}"

    def label(self) -> str:
        extra = []
        if self.flush != "auto":
            extra.append(f"flush={self.flush}")
        if self.launch_mode != "stream":
            extra.append(self.launch_mode)
        if self.inner != 1:
            extra.append(f"inner={self.inner}")
        tail = (" " + " ".join(extra)) if extra else ""
        return (f"{self.family:5s} {self.variant:24s} {self.dtype:5s} "
                f"{self.m}x{self.n}x{self.k}{tail}")

    def flush_state(self) -> bool:
        """What bench_all will decide, so the join key can be built here too."""
        if self.flush == "on":
            return True
        if self.flush == "off":
            return False
        return self.family in ("gemv", "spmv") or working_set(self) <= L2_BYTES

    def key(self) -> tuple:
        return (self.family, self.variant, self.dtype, self.m, self.n, self.k,
                self.flush_state(), self.launch_mode, self.inner)

    def baseline_key(self) -> tuple:
        return (self.family, self.dtype, self.m, self.n, self.k,
                self.flush_state(), self.launch_mode, self.inner)


def working_set(cfg: Config) -> int:
    """Bytes the configuration touches, for the L2 residency decision."""
    if cfg.family == "gemm":
        e = DTYPE_BYTES[cfg.dtype]
        return (cfg.m * cfg.k + cfg.k * cfg.n) * e + cfg.m * cfg.n * 4
    if cfg.family == "gemv":
        return (cfg.m * cfg.n + cfg.n + cfg.m) * 4
    if cfg.family == "trsm":
        return (cfg.m * cfg.m + cfg.m * cfg.n) * 4
    # SpMV builds its matrix inside bench_all, so the size is not known here.
    # The family is memory bound either way and always gets the flushed pair.
    return 0


def cube(sizes):
    return [(s, s, s) for s in sizes]


def variant_matrix(quick: bool) -> list[Config]:
    """The kernels under test, before the baselines and the paired rows."""
    rows: list[Config] = []
    fp32 = ["naive", "tiled", "register", "cp_async"] if not quick else ["naive", "cp_async"]
    for v in fp32:
        for shape in cube(GEMM_FP32_SIZES if not quick else [1024, 2048]):
            rows.append(Config("gemm", v, "fp32", shape))
    fp16 = ["wmma", "mma_ptx", "mma_ldm", "mma_opt"] if not quick else ["wmma", "mma_opt"]
    for v in fp16:
        for shape in cube(GEMM_TENSOR_SIZES if not quick else [2048, 4096]):
            rows.append(Config("gemm", v, "fp16", shape))
    # The tile family, split-K and stream-K at the small shapes they exist for:
    # those are the ones where the grid does not fill the machine.
    for v in ["mma_opt", "splitk", "streamk"]:
        for shape in cube(GEMM_SMALL_SIZES if not quick else [256]):
            rows.append(Config("gemm", v, "fp16", shape))
    for shape in cube(GEMM_TENSOR_SIZES if not quick else [2048]):
        rows.append(Config("gemm", "wmma", "bf16", shape))
    for v in ["naive", "warp", "vectorized"]:
        for shape in [(s, s, 0) for s in (GEMV_SIZES if not quick else [2048, 4096])]:
            rows.append(Config("gemv", v, "fp32", shape))
    for v in ["naive", "warp"]:
        for shape in [(s, s, 0) for s in (SPMV_SIZES if not quick else [65536])]:
            rows.append(Config("spmv", v, "fp32", shape))
    for v in ["naive", "blocked"]:
        for shape in [(s, 256, 0) for s in (TRSM_SIZES if not quick else [1024])]:
            rows.append(Config("trsm", v, "fp32", shape))
    return rows


def expand_protocol(rows: list[Config]) -> list[Config]:
    """Add the paired rows the protocol asks for.

    A row whose working set fits in L2 is measured twice, flushed and unflushed,
    because the unflushed number is a measurement of the cache and the flushed
    one is a measurement of the kernel; reporting only one of them hides which.
    A small GEMM is measured twice again, stream launched and graph launched, so
    launch overhead comes out as a difference rather than sitting inside the
    kernel time.
    """
    out: list[Config] = []
    for cfg in rows:
        resident = cfg.family in ("gemv", "spmv") or working_set(cfg) <= L2_BYTES
        if resident:
            out.append(Config(cfg.family, cfg.variant, cfg.dtype, (cfg.m, cfg.n, cfg.k),
                              flush="on"))
            out.append(Config(cfg.family, cfg.variant, cfg.dtype, (cfg.m, cfg.n, cfg.k),
                              flush="off"))
        else:
            out.append(cfg)
        if cfg.family == "gemm" and max(cfg.m, cfg.n, cfg.k) <= GRAPH_SHAPE_MAX:
            flush = "on" if resident else "auto"
            out.append(Config(cfg.family, cfg.variant, cfg.dtype, (cfg.m, cfg.n, cfg.k),
                              flush=flush, launch_mode="stream", inner=GRAPH_INNER))
            out.append(Config(cfg.family, cfg.variant, cfg.dtype, (cfg.m, cfg.n, cfg.k),
                              flush=flush, launch_mode="graph", inner=GRAPH_INNER))
    return out


def with_baselines(rows: list[Config]) -> tuple[list[Config], list[Config]]:
    """Split into (baseline configs, variant configs).

    One baseline per (family, dtype, shape, flush state, launch mode): exactly
    the join key, so every variant row at a shape is quoted against the same
    measurement rather than against a fresh one.
    """
    seen: set[tuple] = set()
    baselines: list[Config] = []
    for cfg in rows:
        names = [BASELINE_VARIANT[cfg.family]] + EXTRA_BASELINES.get(cfg.family, [])
        for name in names:
            base = Config(cfg.family, name, cfg.dtype, (cfg.m, cfg.n, cfg.k),
                          flush=cfg.flush, launch_mode=cfg.launch_mode, inner=cfg.inner)
            if base.key() in seen:
                continue
            seen.add(base.key())
            baselines.append(base)
    return baselines, rows


# ---------------------------------------------------------------------------
# Clock preamble. tile_sweep.py imports this outright, so the locking rule is
# one piece of code and cannot drift between the two drivers.
# ---------------------------------------------------------------------------


def nvidia_smi(*args: str) -> subprocess.CompletedProcess:
    return subprocess.run(["nvidia-smi", *args], capture_output=True, text=True, check=False)


def query_graphics_clock() -> int | None:
    """Current graphics clock in MHz, or None if nvidia-smi cannot be reached."""
    proc = nvidia_smi("--query-gpu=clocks.gr", "--format=csv,noheader,nounits")
    if proc.returncode != 0:
        return None
    try:
        return int(proc.stdout.strip().splitlines()[0].strip())
    except (ValueError, IndexError):
        return None


def lock_clocks(target_mhz: int) -> tuple[bool, str]:
    """Enable persistence mode and lock the graphics clock. Returns (locked, why)."""
    pm = nvidia_smi("-pm", "1")
    if pm.returncode != 0:
        return False, f"nvidia-smi -pm 1 failed: {(pm.stderr or pm.stdout).strip()}"
    lgc = nvidia_smi("-lgc", str(target_mhz))
    if lgc.returncode != 0:
        return False, f"nvidia-smi -lgc {target_mhz} failed: {(lgc.stderr or lgc.stdout).strip()}"
    observed = query_graphics_clock()
    if observed is None:
        return False, "clocks.gr could not be queried back, so the lock is unverified"
    if abs(observed - target_mhz) > target_mhz * CLOCK_TOLERANCE:
        return False, (f"asked for {target_mhz} MHz, the GPU reports {observed} MHz; "
                       f"the lock did not take")
    return True, f"graphics clock locked at {target_mhz} MHz, GPU reports {observed} MHz"


def unlock_clocks() -> None:
    nvidia_smi("-rgc")


def check_clock_drift(rows: list[dict], locked_mhz: int) -> list[str]:
    """Rows whose measured median clock strayed too far from the lock."""
    bad = []
    for r in rows:
        measured = r.get("median_sm_clock_mhz")
        if not measured:
            continue
        drift = abs(float(measured) - locked_mhz) / locked_mhz
        if drift > CLOCK_TOLERANCE:
            bad.append(f"{r['family']} {r['variant']} {r['dtype']} "
                       f"{r['m']}x{r['n']}x{r['k']}: median {measured} MHz against a "
                       f"{locked_mhz} MHz lock ({drift * 100:.1f}% off)")
    return bad


def git_commit() -> str:
    try:
        return subprocess.run(["git", "rev-parse", "HEAD"], cwd=REPO,
                              capture_output=True, text=True, check=True).stdout.strip()
    except (subprocess.CalledProcessError, FileNotFoundError):
        return "unknown"


def clock_preamble(lock_mhz: int, allow_unlocked: bool) -> tuple[bool, int | None, int]:
    """Lock before a single configuration runs, or refuse.

    Returns (locked, locked_mhz, exit_code). A non zero exit code means the
    caller should stop: locking failed and --allow-unlocked was not passed.
    """
    locked, why = lock_clocks(lock_mhz)
    if locked:
        print(f"clock: {why}")
        return True, lock_mhz, 0
    print(f"clock: NOT locked. {why}", file=sys.stderr)
    if not allow_unlocked:
        print("refusing to run. Locking needs persistence mode and clock control, which "
              "under WSL2 means an elevated Windows side nvidia-smi, or run with "
              "--allow-unlocked and accept that the rows carry no locked clock.",
              file=sys.stderr)
        return False, None, 2
    print("running unlocked because --allow-unlocked was passed; every row will be "
          "stamped clock_locked=false and no locked clock claim can be made from this "
          "data.", file=sys.stderr)
    return False, None, 0


# ---------------------------------------------------------------------------
# Running one configuration
# ---------------------------------------------------------------------------


class BenchFailure(RuntimeError):
    def __init__(self, cfg: Config, returncode: int, payload: dict | None, raw: str):
        self.cfg = cfg
        self.returncode = returncode
        self.payload = payload or {}
        self.raw = raw
        stage = self.payload.get("stage", "?")
        message = self.payload.get("message", raw.strip()[:400])
        super().__init__(f"{cfg.label()}: exit {returncode} at stage {stage}: {message}")


def run_once(bench: Path, cfg: Config, commit: str, extra: list[str]) -> dict:
    proc = subprocess.run(cfg.argv(bench, commit) + extra,
                          capture_output=True, text=True, check=False)
    payload = None
    text = proc.stdout.strip()
    if text:
        try:
            payload = json.loads(text.splitlines()[-1])
        except json.JSONDecodeError:
            payload = None
    if proc.returncode != 0 or payload is None or payload.get("error"):
        raise BenchFailure(cfg, proc.returncode, payload, proc.stdout + proc.stderr)
    return payload


def probe_autotune(bench: Path) -> tuple[bool, str]:
    """Ask bench_all whether this toolkit's CUBLAS_GEMM_AUTOTUNE actually runs.

    The value is declared in cuBLAS 12 and 13 and documented as experimental, so
    the protocol probes it against a real call rather than assuming. A toolkit
    that refuses it loses the second baseline and keeps the first; it does not
    lose the sweep.
    """
    proc = subprocess.run([str(bench), "--probe-autotune"],
                          capture_output=True, text=True, check=False)
    try:
        payload = json.loads(proc.stdout.strip().splitlines()[-1])
    except (json.JSONDecodeError, IndexError):
        return False, f"the probe printed nothing usable: {proc.stdout.strip()[:200]}"
    if payload.get("cublas_autotune_available"):
        return True, "CUBLAS_GEMM_AUTOTUNE runs on this toolkit"
    return False, (f"CUBLAS_GEMM_AUTOTUNE declared={payload.get('cublas_autotune_declared')}, "
                   f"status={payload.get('status', payload.get('note', '?'))}")


def bootstrap_ci(values: list[float], resamples: int, seed: int) -> tuple[float, float]:
    """Seeded bootstrap 95 percent interval on the median of the process medians."""
    import numpy as np

    rng = np.random.default_rng(seed)
    data = np.asarray(values, dtype=float)
    draws = rng.integers(0, data.size, size=(resamples, data.size))
    medians = np.median(data[draws], axis=1)
    lo, hi = np.percentile(medians, [2.5, 97.5])
    return float(lo), float(hi)


def median(values: list[float]) -> float:
    ordered = sorted(values)
    mid = len(ordered) // 2
    if len(ordered) % 2:
        return ordered[mid]
    return 0.5 * (ordered[mid - 1] + ordered[mid])


def measure_config(bench: Path, cfg: Config, commit: str, args) -> dict:
    """N independent processes, reduced to the one row of record.

    The row keeps every process median, so the reader can see the spread the
    interval was computed from, and the full sample array of the process whose
    median is the median of medians, so the reduction can be redone.
    """
    runs = []
    for rep in range(args.process_reps):
        if rep > 0 and args.cooldown_s > 0:
            time.sleep(args.cooldown_s)
        runs.append(run_once(bench, cfg, commit, args.bench_flags))
    medians = [float(r["median_ms"]) for r in runs]
    mid = median(medians)
    # The representative process is the one whose median is closest to the
    # median of medians; its samples are the ones that go into the row.
    rep_row = min(runs, key=lambda r: abs(float(r["median_ms"]) - mid))

    row = dict(rep_row)
    row["sweep_schema_version"] = SWEEP_SCHEMA_VERSION
    row["process_reps"] = args.process_reps
    row["medians_ms"] = medians
    row["median_of_medians_ms"] = mid
    row["median_ms"] = mid
    row["min_ms"] = min(float(r["min_ms"]) for r in runs)
    row["throttled"] = any(bool(r.get("throttled")) for r in runs)
    row["median_sm_clock_mhz"] = median([float(r["median_sm_clock_mhz"]) for r in runs])
    row["max_temp_c"] = max(int(r["max_temp_c"]) for r in runs)
    row["max_power_w"] = max(float(r["max_power_w"]) for r in runs)
    lo, hi = bootstrap_ci(medians, args.bootstrap_resamples, args.bootstrap_seed)
    row["ci95_lo_ms"] = lo
    row["ci95_hi_ms"] = hi
    row["bootstrap_resamples"] = args.bootstrap_resamples
    row["bootstrap_seed"] = args.bootstrap_seed
    row["cooldown_s"] = args.cooldown_s
    # The row of record's throughput comes from the median of medians, not from
    # whichever process happened to be representative.
    row["gflops"] = gflops_from(rep_row, mid)
    return row


def gflops_from(row: dict, ms: float) -> float:
    """Recompute throughput from a new time, reusing the row's own flop count."""
    if ms <= 0:
        return 0.0
    ref_ms = float(row["median_ms"])
    ref_gflops = float(row["gflops"])
    return ref_gflops * ref_ms / ms


# ---------------------------------------------------------------------------
# The summary
# ---------------------------------------------------------------------------

SUMMARY_COLUMNS = [
    "family", "variant", "dtype", "m", "n", "k",
    "median_ms", "iqr_ms", "min_ms", "reps",
    "ci95_lo_ms", "ci95_hi_ms", "process_reps",
    "gflops", "baseline_variant", "baseline_gflops", "pct_baseline", "baseline_source",
    "l2_flushed", "launch_mode", "inner_launches",
    "chosen", "entry_point", "tile", "splits", "plan_tuned",
    "verify_residual", "verify_tol",
    "cublas_math_mode", "cublas_gemm_algo",
    "throttled", "median_sm_clock_mhz", "clock_locked", "locked_clock_mhz",
    "max_temp_c", "max_power_w", "cuda_runtime", "cuda_driver",
    "canonical", "schema_version", "commit",
]


def auto_flush(row: dict) -> bool:
    """What --flush-l2 auto decides for this row, recomputed from the row itself."""
    if row.get("family") in ("gemv", "spmv"):
        return True
    return int(row.get("working_set_bytes", 0)) <= L2_BYTES


def is_canonical(row: dict) -> bool:
    """The one row per measurement that answers the plain question.

    A shape can carry up to four rows: flushed and unflushed, stream launched and
    graph launched. Three of them exist to expose an effect, and exactly one is
    the number the protocol would quote on its own. A v1 row, which has none of
    these fields, is canonical by default.
    """
    if "schema_version" not in row:
        return True
    return (row.get("launch_mode", "stream") == "stream"
            and int(row.get("inner_launches", 1)) == 1
            and bool(row.get("l2_flushed", False)) == auto_flush(row))


# The identity of a measurement. Cache state, launch path and the number of
# launches inside one event pair are part of it: two rows that differ in any of
# them are answers to different questions and must not collapse onto each other.
def summary_key(row: dict) -> tuple:
    return (row["family"], row["variant"], row["dtype"], row["m"], row["n"], row["k"],
            bool(row.get("l2_flushed", False)), row.get("launch_mode", "stream"),
            int(row.get("inner_launches", 1)))


def baseline_join_key(row: dict) -> tuple:
    return (row["family"], row["dtype"], row["m"], row["n"], row["k"],
            bool(row.get("l2_flushed", False)), row.get("launch_mode", "stream"),
            int(row.get("inner_launches", 1)))


def rel(path: Path) -> str:
    """Path relative to the repo when it is inside it, absolute otherwise."""
    try:
        return str(path.relative_to(REPO))
    except ValueError:
        return str(path)


def read_jsonl(path: Path) -> list[dict]:
    rows = []
    if not path.exists():
        return rows
    with path.open() as f:
        for line in f:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def attach_baseline(row: dict, baselines: dict[tuple, dict]) -> dict:
    """Fill in the baseline reference and the percentage from the joined row.

    A v1 row (no schema_version) measured its own baseline inside the row. It is
    read through unchanged, with baseline_source saying so, because the
    alternative is silently quoting it as if it had been measured under this
    protocol.
    """
    out = dict(row)
    if "schema_version" not in row:
        out["baseline_source"] = "per_row_remeasured"
        out.setdefault("baseline_variant", "")
        return out
    base = baselines.get(baseline_join_key(row))
    if row.get("variant") == BASELINE_VARIANT.get(row.get("family"), ""):
        out["baseline_source"] = "self"
        out["baseline_variant"] = row["variant"]
        out["baseline_gflops"] = row["gflops"]
        out["pct_baseline"] = 100.0
        return out
    if base is None:
        out["baseline_source"] = "missing"
        out["baseline_variant"] = ""
        out["baseline_gflops"] = ""
        out["pct_baseline"] = ""
        return out
    out["baseline_source"] = "joined"
    out["baseline_variant"] = base["variant"]
    out["baseline_gflops"] = base["gflops"]
    out["baseline_median_ms"] = base["median_ms"]
    out["pct_baseline"] = (100.0 * float(row["gflops"]) / float(base["gflops"])
                           if float(base["gflops"]) > 0 else "")
    return out


def refresh_summary(commit: str, jsonl: Path = JSONL, summary: Path = SUMMARY) -> int:
    """Rebuild summary.csv from exactly one commit's rows.

    Mixing commits is how v1's report ended up quoting whichever row matched
    first. One commit, one row per key, and a loud failure otherwise.
    """
    rows = read_jsonl(jsonl)
    if not rows:
        print(f"{jsonl} has no rows", file=sys.stderr)
        return 0
    mine = [r for r in rows if str(r.get("commit", "")) == commit]
    if not mine:
        seen = sorted({str(r.get("commit", ""))[:12] for r in rows})
        print(f"no row in {rel(jsonl)} carries commit {commit[:12]}. "
              f"The file holds: {', '.join(seen)}. Run the sweep at this commit, or pass "
              f"--commit with one of those.", file=sys.stderr)
        raise SystemExit(1)

    by_key: dict[tuple, dict] = {}
    duplicates: list[str] = []
    for r in mine:
        key = summary_key(r)
        if key in by_key:
            duplicates.append(" ".join(str(x) for x in key))
        by_key[key] = r
    if duplicates:
        print(f"{rel(jsonl)} has more than one row per key at commit "
              f"{commit[:12]}:", file=sys.stderr)
        for d in sorted(set(duplicates)):
            print("  " + d, file=sys.stderr)
        print("A summary built from this would quote whichever row came first. Deduplicate "
              "the JSONL or re-run the sweep with --force.", file=sys.stderr)
        raise SystemExit(1)

    baselines = {baseline_join_key(r): r for r in by_key.values()
                 if r.get("variant", "") == BASELINE_VARIANT.get(r.get("family", ""), "")}

    joined = [attach_baseline(r, baselines) for r in by_key.values()]
    for r in joined:
        r["canonical"] = is_canonical(r)
        for field in ("gflops", "baseline_gflops"):
            if isinstance(r.get(field), float):
                r[field] = round(r[field], 3)
        if isinstance(r.get("pct_baseline"), float):
            r["pct_baseline"] = round(r["pct_baseline"], 2)
        for field in ("ci95_lo_ms", "ci95_hi_ms", "median_ms", "iqr_ms", "min_ms",
                      "verify_residual", "verify_tol"):
            if isinstance(r.get(field), float):
                r[field] = round(r[field], 9)
    # Canonical rows come first inside a shape, so a consumer that looks a row up
    # by (family, variant, dtype, m) and takes the first match gets the row the
    # protocol calls the answer rather than a flush or launch mode study.
    joined.sort(key=lambda r: (r["family"], r["dtype"], r["variant"], r["m"], r["n"], r["k"],
                               0 if r["canonical"] else 1,
                               str(r.get("launch_mode", "")), int(r.get("inner_launches", 1)),
                               str(r.get("l2_flushed", ""))))
    summary.parent.mkdir(parents=True, exist_ok=True)
    with summary.open("w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=SUMMARY_COLUMNS, extrasaction="ignore")
        w.writeheader()
        for r in joined:
            w.writerow(r)
    missing = [r for r in joined if r.get("baseline_source") == "missing"]
    if missing:
        print(f"warning: {len(missing)} row(s) have no baseline row at their shape, so they "
              f"carry no percentage. Run the baseline configurations before quoting them.",
              file=sys.stderr)
    return len(joined)


# ---------------------------------------------------------------------------
# Driver
# ---------------------------------------------------------------------------


def done_keys(commit: str) -> set:
    keys = set()
    for r in read_jsonl(JSONL):
        if str(r.get("commit", "")) != commit:
            continue
        keys.add(summary_key(r))
    return keys


def run_phase(name: str, configs: list[Config], bench: Path, commit: str, args,
              out, measured: list[dict], throttled_rows: list[str],
              baselines: dict[tuple, dict]) -> int:
    """Measure one phase of the sweep. Returns the number of failed configurations."""
    failures = 0
    start = time.time()
    for i, cfg in enumerate(configs, start=1):
        if i > 1 and args.cooldown_s > 0:
            time.sleep(args.cooldown_s)
        try:
            row = measure_config(bench, cfg, commit, args)
        except BenchFailure as exc:
            print(f"  FAILED {exc}", file=sys.stderr)
            failures += 1
            continue

        # Test only hook, so the throttle gate can be shown failing without
        # waiting for a hot GPU. It marks the row, nothing else.
        if args.throttle_test and i == 1:
            row["throttled"] = True
            row["throttle_test_hook"] = True

        reruns = 0
        while row.get("throttled") and reruns < 1:
            reruns += 1
            print(f"  throttled: {cfg.label()}; cooling for {args.throttle_cooldown_s:.0f}s "
                  f"and re-running", file=sys.stderr)
            time.sleep(args.throttle_cooldown_s)
            try:
                row = measure_config(bench, cfg, commit, args)
            except BenchFailure as exc:
                print(f"  FAILED {exc}", file=sys.stderr)
                failures += 1
                row = None
                break
            if args.throttle_test and i == 1:
                row["throttled"] = True
                row["throttle_test_hook"] = True
        if row is None:
            continue
        row["throttle_reruns"] = reruns
        if row.get("throttled"):
            throttled_rows.append(cfg.label())

        row["clock_locked"] = args.locked
        row["locked_clock_mhz"] = args.locked_mhz if args.locked else ""
        row["order_seed"] = args.order_seed
        row["order_index"] = i
        row["order_phase"] = name
        if cfg.variant == BASELINE_VARIANT.get(cfg.family, ""):
            # Only the baseline of record is a join target. The autotuned pick is
            # measured next to it and quoted against it, because it answers a
            # different question: beating the heuristic is the weaker claim.
            baselines.setdefault(cfg.baseline_key(), row)
        if not cfg.baseline or cfg.variant != BASELINE_VARIANT.get(cfg.family, ""):
            base = baselines.get(cfg.baseline_key())
            if base is not None:
                row["baseline_variant"] = base["variant"]
                row["baseline_median_ms"] = base["median_ms"]
                row["baseline_gflops"] = base["gflops"]
                row["pct_baseline"] = (100.0 * float(row["gflops"]) / float(base["gflops"])
                                       if float(base["gflops"]) > 0 else 0.0)
        out.write(json.dumps(row) + "\n")
        out.flush()
        measured.append(row)

        elapsed = time.time() - start
        eta = elapsed / i * (len(configs) - i)
        pct = row.get("pct_baseline")
        pct_text = f" ({float(pct):.0f}% base)" if isinstance(pct, (int, float)) else ""
        print(f"  [{name} {i}/{len(configs)}] {cfg.label()} "
              f"{row['gflops']:.0f} GFLOP/s{pct_text} eta {eta:.0f}s")
    return failures


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--force", action="store_true", help="rerun configs that already have a row")
    ap.add_argument("--quick", action="store_true", help="representative subset")
    ap.add_argument("--list", action="store_true", help="print the matrix and exit")
    ap.add_argument("--bench", type=Path, default=BENCH_ALL, help="path to the bench_all binary")
    ap.add_argument("--lock-clock", type=int, default=DEFAULT_LOCK_MHZ, metavar="MHZ",
                    help=f"graphics clock to lock before measuring (default {DEFAULT_LOCK_MHZ})")
    ap.add_argument("--allow-unlocked", action="store_true",
                    help="run even though the clock could not be locked; every row is "
                         "stamped clock_locked=false and carries no locked clock")
    ap.add_argument("--process-reps", type=int, default=DEFAULT_PROCESS_REPS, metavar="N",
                    help=f"independent bench_all processes per configuration "
                         f"(default {DEFAULT_PROCESS_REPS})")
    ap.add_argument("--cooldown-s", type=float, default=DEFAULT_COOLDOWN_S, metavar="S",
                    help=f"sleep between runs (default {DEFAULT_COOLDOWN_S})")
    ap.add_argument("--throttle-cooldown-s", type=float, default=60.0, metavar="S",
                    help="sleep before re-running a throttled configuration (default 60)")
    ap.add_argument("--order-seed", type=int, default=None, metavar="N",
                    help="seed for the configuration order; a fresh random one by default, "
                         "and whichever is used is recorded in every row")
    ap.add_argument("--bootstrap-seed", type=int, default=20250901, metavar="N")
    ap.add_argument("--bootstrap-resamples", type=int, default=BOOTSTRAP_RESAMPLES, metavar="N")
    ap.add_argument("--commit", default=None, metavar="HASH",
                    help="commit to stamp on new rows and to build the summary from "
                         "(default HEAD)")
    ap.add_argument("--refresh-only", action="store_true",
                    help="rebuild summary.csv from rows that already exist, measure nothing")
    ap.add_argument("--jsonl", type=Path, default=JSONL, help="results file to append to")
    ap.add_argument("--summary", type=Path, default=SUMMARY, help="summary to write")
    ap.add_argument("--bench-flag", action="append", default=[], dest="bench_flags",
                    metavar="FLAG",
                    help="extra argument passed through to bench_all; repeatable, and "
                         "recorded in the row through the fields bench_all stamps itself")
    ap.add_argument("--only", action="append", default=[], metavar="TEXT",
                    help="restrict to configurations whose "
                         "family:variant:dtype:mxnxk contains this text; repeatable. The "
                         "baselines for the surviving shapes come along.")
    ap.add_argument("--limit", type=int, default=0, metavar="N",
                    help="run only the first N variant configurations after the shuffle "
                         "(0 means all)")
    ap.add_argument("--throttle-test", action="store_true",
                    help="TEST ONLY: mark the first row of each phase throttled, to show the "
                         "throttle gate failing")
    args = ap.parse_args()

    commit = args.commit or git_commit()

    if args.refresh_only:
        n = refresh_summary(commit, args.jsonl, args.summary)
        print(f"wrote {args.summary} ({n} rows at commit {commit[:12]})")
        return 0

    wanted = variant_matrix(args.quick)
    if args.only:
        wanted = [c for c in wanted if any(t in c.spec() for t in args.only)]
        if not wanted:
            print(f"no configuration matches {args.only}", file=sys.stderr)
            return 2
    baseline_cfgs, variant_cfgs = with_baselines(expand_protocol(wanted))
    if args.limit > 0:
        variant_cfgs = variant_cfgs[:args.limit]
        keep = {c.baseline_key() for c in variant_cfgs}
        baseline_cfgs = [b for b in baseline_cfgs if b.baseline_key() in keep]

    if args.list:
        for cfg in baseline_cfgs + variant_cfgs:
            print(cfg.label())
        print(f"{len(baseline_cfgs)} baseline and {len(variant_cfgs)} variant configurations, "
              f"{args.process_reps} processes each")
        return 0

    if not args.bench.exists():
        print(f"bench_all not found at {args.bench}; build first (make build)", file=sys.stderr)
        return 1
    try:
        import numpy  # noqa: F401
    except ImportError:
        print("numpy is needed for the bootstrap confidence interval; install it or the rows "
              "would carry a median with no interval, which is what Section 13 is trying to "
              "stop.", file=sys.stderr)
        return 1

    autotune_ok, why = probe_autotune(args.bench)
    if autotune_ok:
        print(f"cublas: {why}")
    else:
        print(f"cublas: {why}; dropping the autotuned baseline from this sweep",
              file=sys.stderr)
        baseline_cfgs = [c for c in baseline_cfgs if c.variant != "baseline_cublas_autotune"]

    locked, locked_mhz, code = clock_preamble(args.lock_clock, args.allow_unlocked)
    if code != 0:
        return code
    args.locked = locked
    args.locked_mhz = locked_mhz

    if args.order_seed is None:
        args.order_seed = random.randrange(1 << 30)
    rng = random.Random(args.order_seed)
    rng.shuffle(baseline_cfgs)
    rng.shuffle(variant_cfgs)
    print(f"order: shuffled under seed {args.order_seed}, cooldown {args.cooldown_s:.0f}s "
          f"between runs")

    args.jsonl.parent.mkdir(parents=True, exist_ok=True)
    done = set() if args.force else done_keys(commit)
    baseline_cfgs = [c for c in baseline_cfgs if c.key() not in done]
    variant_cfgs = [c for c in variant_cfgs if c.key() not in done]
    print(f"sweep: {len(baseline_cfgs)} baseline and {len(variant_cfgs)} variant configurations "
          f"to run at commit {commit[:12]}, {args.process_reps} processes each")

    # Baselines already on file at this commit are reused, so a resumed sweep
    # still joins every variant onto the same measurement.
    baselines: dict[tuple, dict] = {}
    for r in read_jsonl(args.jsonl):
        if str(r.get("commit", "")) == commit and \
                r.get("variant") == BASELINE_VARIANT.get(r.get("family"), ""):
            baselines[baseline_join_key(r)] = r

    measured: list[dict] = []
    throttled_rows: list[str] = []
    failures = 0
    with args.jsonl.open("a") as out:
        # Baselines first: a variant row cannot be written until the measurement
        # it will be quoted against exists.
        failures += run_phase("baseline", baseline_cfgs, args.bench, commit, args, out,
                              measured, throttled_rows, baselines)
        failures += run_phase("variant", variant_cfgs, args.bench, commit, args, out,
                              measured, throttled_rows, baselines)

    if locked:
        unlock_clocks()
        print("clock: lock released (nvidia-smi -rgc)")

    n_rows = refresh_summary(commit, args.jsonl, args.summary)
    print(f"wrote {args.summary} ({n_rows} rows at commit {commit[:12]})")

    status = 0
    if failures:
        print(f"{failures} configuration(s) failed to produce a row", file=sys.stderr)
        status = 1
    if throttled_rows:
        print(f"throttle gate FAILED: {len(throttled_rows)} configuration(s) were still "
              f"throttled after a cooldown and a re-run. These rows are not comparable with "
              f"the rest of the sweep:", file=sys.stderr)
        for line in throttled_rows:
            print("  " + line, file=sys.stderr)
        status = 1
    elif measured:
        print("no throttled rows")

    if locked:
        drifted = check_clock_drift(measured, locked_mhz)
        if drifted:
            print(f"clock drift: {len(drifted)} row(s) measured more than "
                  f"{CLOCK_TOLERANCE * 100:.0f} percent off the {locked_mhz} MHz lock. "
                  f"These are not comparable with the rest of the sweep:", file=sys.stderr)
            for line in drifted:
                print("  " + line, file=sys.stderr)
            status = 1
        else:
            print(f"clock drift: every row within {CLOCK_TOLERANCE * 100:.0f} percent of the "
                  f"{locked_mhz} MHz lock")
    return status


if __name__ == "__main__":
    raise SystemExit(main())
