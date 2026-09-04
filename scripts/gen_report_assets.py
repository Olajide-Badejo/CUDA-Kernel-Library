#!/usr/bin/env python3
"""Generate the report's tables and figures from the canonical results.

Idempotent: the same inputs produce the same booktabs tables and the same
figures, byte for byte. Reads experiments/results/, writes LaTeX into
report/tables/ and figures into report/figures/. The roofline figure is rendered
by plot_roofline.py (from the profiler's CSVs); everything else is here. Nothing
is hand copied into the report; it all comes from this script.

Two rules run through the whole file.

A missing input is a hard error. The v1 script skipped the roofline when the CSV
was absent and exited 0, so a clean clone (where the CSV was gitignored) rebuilt
the report around the committed PNG and reported success, in CI included. That is
defect A7 in docs/CORRECTIONS.md: a report that cannot be regenerated from the
tree has to fail loudly, not quietly reuse a stale picture.

A number that has not been measured is written "pending", never estimated. The
1.1.0 families are implemented but not swept, so their throughput columns say so,
and the tables are wired to their CSVs already: when the owner sweep lands, the
numbers appear without anybody editing a .tex file.
"""

from __future__ import annotations

import csv
import subprocess
import sys
from pathlib import Path

REPO = Path(__file__).resolve().parent.parent
RESULTS = REPO / "experiments" / "results"
TABLES = REPO / "report" / "tables"
FIGURES = REPO / "report" / "figures"

PENDING = "pending"
# A configuration the plan declined to run, as opposed to one nobody has measured
# yet. sweep.py writes those rows with status=skipped_not_supported and the
# plan's own reason; the reason itself is too long for a table cell and lives in
# summary.csv and in docs/sparse.md.
REFUSED = "refused"

# The GEMM ladder, in the order the rungs were built. Each entry is the key into
# the summary plus the label the report uses.
LADDER_ORDER = [
    ("gemm", "naive", "fp32", "naive"),
    ("gemm", "tiled", "fp32", "shared tiled"),
    ("gemm", "register", "fp32", "register blocked"),
    ("gemm", "cp_async", "fp32", "cp.async double buffered"),
    ("gemm", "wmma", "fp16", "WMMA"),
    ("gemm", "mma_ptx", "fp16", "mma.sync PTX"),
    ("gemm", "mma_ldm", "fp16", "mma.sync + ldmatrix"),
    ("gemm", "mma_opt", "fp16", "mma.sync + ldmatrix + swizzle"),
    ("gemm", "cutlass", "fp16", "CUTLASS reference line"),
    ("gemm", "wmma", "bf16", "WMMA (BF16)"),
]

TOP_VARIANT = ("gemm", "mma_opt", "fp16")

# The tile family. Shape and warp arrangement are structural facts about what is
# instantiated; the shared memory column is arithmetic on the shape (two operand
# tiles, three stages, two bytes an element) and is derived here rather than
# copied. Registers per thread and relative throughput come from a sweep that has
# not run, so they are pending. docs/gemm.md carries the resource usage report.
TILE_FAMILY = [
    # (BM, BN, BK, warps in M, warps in N)
    (128, 128, 32, 2, 4),
    (128, 256, 32, 2, 8),
    (256, 128, 32, 4, 4),
    (128, 64, 64, 2, 2),
    (64, 128, 64, 1, 4),
    (128, 128, 64, 2, 4),
]
TILE_STAGES = 3
TILE_BYTES_PER_ELEMENT = 2

# The decomposition mechanisms: what each one is for, and what the sweep measured.
# The fourth element is the summary variant the speedup column is computed from,
# against the data parallel mainloop at the same shape; None means the row is not a
# variant of its own and the last column is filled by hand from structure rather
# than from a timing.
DECOMPOSITIONS = [
    ("data parallel", "one CTA per output tile", "the baseline decomposition",
     None, "1.00x by definition"),
    ("predicated tails", "shapes the tile does not divide",
     "one mainloop for every shape, no silent reroute", None, None),
    ("split-K", "few waves, long contraction",
     "idle SMs when the output is smaller than one wave", "splitk", None),
    ("stream-K", "few waves, any contraction",
     "the same, without a full extra pass over C", "streamk", None),
]

# The decomposition variants are measured against this one, at the same shape.
DECOMP_BASELINE = ("gemm", "mma_opt", "fp16")

# The FFT, convolution, reduction and scan rungs. Each entry names the family and
# the variant exactly as benchmarks/sweep.py writes them, so the measured columns
# fill in from the summary the moment those rows land. The third column is derived
# from the algorithm (passes over the array, bytes moved per element) and is not a
# measurement.
FFT_RUNGS = [
    ("fft", "radix2", "radix 2, one kernel per stage",
     "a launch per butterfly stage", "log2(n) passes"),
    ("fft", "shared", "shared resident",
     "one block per transform, all stages in shared", "1 pass"),
    ("fft", "radix4", "radix 4 shared resident", "half the stages of radix 2", "1 pass"),
    ("fft", "radix8", "radix 8 shared resident", "a third of the stages of radix 2",
     "1 pass"),
    ("fft", "four_step", "four step",
     "transpose, batched, transpose, batched, transpose", "5 passes"),
    ("conv", "fft_separate", "convolution, FFT separate",
     "standalone pointwise multiply kernel", "24 bytes per point"),
    ("conv", "fft_fused", "convolution, FFT fused",
     "multiply folded into a store epilogue", "8 bytes per point"),
    ("conv", "direct_shared", "convolution, direct shared",
     "tiled time domain, filter chunked through shared", "2 N M FLOP"),
    ("conv", "direct_constant", "convolution, direct constant",
     "the same tiling, taps in constant memory", "2 N M FLOP"),
]

SCAN_RUNGS = [
    ("scan", "hillis_steele", "s0 Hillis-Steele", "block scan, work inefficient",
     "N log N work at block scope"),
    ("scan", "blelloch", "s1 Blelloch",
     "upsweep and downsweep, bank conflict padding", "N work, twice the barriers"),
    ("scan", "three_kernel", "s2 three kernel", "scan then propagate",
     "about 4 N bytes"),
    ("scan", "reduce_then_scan", "s3 reduce then scan", "two passes over the input",
     "about 3 N bytes"),
    ("scan", "lookback", "s4 decoupled look-back",
     "single pass, device scope handshake", "about 2 N bytes"),
    ("scan", "deterministic", "deterministic scan",
     "fixed grid, index order, explicit halving tree", "about 3 N bytes"),
]

REQUIRED_INPUTS = [
    (RESULTS / "summary.csv", "the benchmark sweep (make sweep)"),
    (RESULTS / "roofline.csv", "the roofline profiler (make roofline)"),
    (RESULTS / "roofline_ceilings.csv", "the roofline profiler (make roofline)"),
    (RESULTS / "register_study.csv", "the register allocation study (make register-study)"),
]

# Optional inputs. Each one, when it lands, fills in a table that currently reads
# pending. Nothing here is required, and nothing here is faked when it is absent.
OPTIONAL_INPUTS = [
    (RESULTS / "tile_sweep.csv", "the tile family decision table (make tile-sweep)"),
]


def require_inputs() -> None:
    """Stop with a message naming what is missing and how to produce it."""
    missing = [(p, how) for p, how in REQUIRED_INPUTS if not p.exists()]
    if not missing:
        return
    print("cannot generate report assets: required results are missing.", file=sys.stderr)
    for path, how in missing:
        print(f"  {path.relative_to(REPO)} is not there; produce it with {how}", file=sys.stderr)
    print("The report is generated from the results in the tree. Reusing a committed "
          "figure whose input is gone would make the report unreproducible.", file=sys.stderr)
    raise SystemExit(1)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        rows = list(csv.DictReader(f))
    if not rows:
        print(f"{path.relative_to(REPO)} has no data rows", file=sys.stderr)
        raise SystemExit(1)
    return rows


def read_optional(path: Path) -> list[dict[str, str]]:
    if not path.exists():
        return []
    with path.open() as f:
        return list(csv.DictReader(f))


# ---------------------------------------------------------------------------
# Row lookup
#
# A summary row is identified by family, variant, dtype and either a shape or a
# matrix name. The v1 schema had no matrix column, so (family, variant, dtype, m)
# was unique; it stops being unique the moment the SpMV suite writes one row per
# matrix at the same nominal m, and a lookup that ignores the matrix would then
# return whichever row happened to be first. Every lookup below therefore carries
# the matrix, and the sort that breaks a remaining tie is explicit rather than
# incidental so the tables stay byte stable.
# ---------------------------------------------------------------------------


def row_matrix(r: dict[str, str]) -> str:
    return str(r.get("matrix", "") or "")


def is_canonical_row(r: dict[str, str]) -> bool:
    """True when the row is the one the protocol would quote on its own.

    A v1 row has no canonical column at all and is canonical by default, which is
    the same rule benchmarks/sweep.py applies when it writes the summary.
    """
    raw = r.get("canonical")
    if raw is None or raw == "":
        return True
    return str(raw).strip().lower() in ("true", "1", "yes")


def find_all(rows, family, variant=None, dtype=None, m=None, matrix=None):
    """Every row matching the given fields, canonical rows first, then sorted."""
    out = []
    for r in rows:
        if r.get("family") != family:
            continue
        if variant is not None and r.get("variant") != variant:
            continue
        if dtype is not None and r.get("dtype") != dtype:
            continue
        if m is not None and int(r["m"]) != int(m):
            continue
        if matrix is not None and row_matrix(r) != matrix:
            continue
        out.append(r)
    out.sort(key=lambda r: (0 if is_canonical_row(r) else 1, row_matrix(r), int(r["m"]),
                            r.get("variant", ""), r.get("dtype", "")))
    return out


def find(rows, family, variant=None, dtype=None, m=None, matrix=None):
    """The single row of record for a lookup, or None."""
    hits = find_all(rows, family, variant, dtype, m, matrix)
    return hits[0] if hits else None


def fnum(r: dict[str, str], key: str) -> float | None:
    raw = r.get(key)
    if raw is None or str(raw).strip() == "":
        return None
    try:
        return float(raw)
    except ValueError:
        return None


def has_interval(r: dict[str, str]) -> bool:
    return fnum(r, "ci95_lo_ms") is not None and fnum(r, "ci95_hi_ms") is not None


def latex_escape(s: str) -> str:
    return (str(s).replace("\\", "").replace("_", r"\_").replace("%", r"\%")
            .replace("&", r"\&").replace("#", r"\#"))


def tabular(colspec: str, header: list[str], body: list[list[str]]) -> str:
    lines = [r"\begin{tabular}{" + colspec + "}", r"\toprule",
             " & ".join(header) + r" \\", r"\midrule"]
    for row in body:
        lines.append(" & ".join(row) + r" \\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    return "\n".join(lines)


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")


# ---------------------------------------------------------------------------
# GEMM
# ---------------------------------------------------------------------------


def write_gemm_ladder(rows, size: int) -> None:
    body = []
    for family, variant, dtype, label in LADDER_ORDER:
        r = find(rows, family, variant, dtype, size)
        if r is None:
            continue
        body.append([latex_escape(label), dtype, f"{float(r['gflops']):.0f}",
                     f"{float(r['pct_baseline']):.1f}"])
    write(TABLES / "gemm_ladder.tex",
          tabular("llrr", ["rung", "precision", "GFLOP/s", "percent of cuBLAS"], body))


def ladder_series(rows) -> tuple[list[dict], list[int], bool]:
    """One series per rung across every shape the summary carries for it."""
    sizes = sorted({int(r["m"]) for r in rows if r.get("family") == "gemm"})
    series = []
    any_interval = False
    for family, variant, dtype, label in LADDER_ORDER:
        points = []
        for m in sizes:
            r = find(rows, family, variant, dtype, m)
            if r is None:
                continue
            pct = fnum(r, "pct_baseline")
            if pct is None:
                continue
            lo = hi = None
            median = fnum(r, "median_ms")
            if has_interval(r) and median:
                # Throughput is inversely proportional to time, so the fast end of
                # the time interval is the high end of the percent interval.
                lo = pct * median / fnum(r, "ci95_hi_ms")
                hi = pct * median / fnum(r, "ci95_lo_ms")
                any_interval = True
            points.append((m, pct, lo, hi))
        if points:
            series.append({"label": label, "dtype": dtype,
                           "top": (family, variant, dtype) == TOP_VARIANT,
                           "points": points})
    return series, sizes, any_interval


def write_ladder_caption(series, sizes, any_interval: bool) -> None:
    """The half of the ladder caption that depends on what the data carries.

    Written as a macro definition rather than as bare text. An \\input inside a
    \\caption breaks when hyperref writes the caption out to the list of figures,
    so main.tex loads this file in the preamble and the figure uses the macro.
    """
    parts = [f"Drawn from {len(series)} rungs over "
             f"{len(sizes)} swept shapes in the committed summary."]
    if any_interval:
        parts.append("Error bars are the bootstrap 95 percent interval over the "
                     "independent process repeats recorded in each row.")
    else:
        parts.append("The rows in this build carry no confidence interval, because "
                     "they predate the protocol that records one, so the series are "
                     "drawn without error bars.")
    missing = [label for _f, _v, _d, label in LADDER_ORDER
               if label not in {s["label"] for s in series}]
    if missing:
        parts.append("Rungs with no rows in the committed summary are absent from the "
                     "chart rather than drawn flat: " + ", ".join(missing) + ".")
    write(TABLES / "ladder_caption.tex",
          "\\newcommand{\\laddercaption}{" + " ".join(parts) + "}\n")


def write_gemm_gap(rows) -> None:
    """The gap to cuBLAS, split into wave quantization and everything else.

    The quantization column is arithmetic on the tile geometry, not a measurement:
    at a 128 by 128 tile the output is a whole number of tiles competing for the
    resident block slots the shared memory budget allows, and the loss is the
    fraction of the last wave that runs empty. Stating it separately is the point
    of the table, because it is the part of the gap that a different decomposition
    fixes rather than a better mainloop.
    """
    tile_m = tile_n = 128
    slots = 48 * 2  # 48 SMs, two resident blocks each at 49152 bytes of shared
    family, variant, dtype = TOP_VARIANT
    body = []
    for r in find_all(rows, family, variant, dtype):
        # Canonical only. The protocol writes a graph launch row and an unflushed
        # row beside the row of record at the smaller shapes, and all three carry
        # the same shape; listing them here put "128 cubed" in the table four
        # times with four different percentages and no column to tell them apart.
        if not is_canonical_row(r):
            continue
        m, n = int(r["m"]), int(r["n"])
        pct = fnum(r, "pct_baseline")
        if pct is None:
            continue
        tiles = -(-m // tile_m) * -(-n // tile_n)
        waves = tiles / slots
        full_waves = -(-tiles // slots)
        quant = 100.0 * (1.0 - waves / full_waves)
        residual = 100.0 - pct - quant
        body.append([f"{m} cubed", f"{float(r['gflops']):,.0f}", f"{pct:.1f}",
                     f"{tiles}, {waves:.2f} waves", f"{quant:.1f}",
                     f"{residual:.1f}"])
    write(TABLES / "gemm_gap.tex",
          tabular("lrrlrr",
                  ["shape", "GFLOP/s", "percent of cuBLAS",
                   f"CTAs into {slots} slots", "quantization loss", "residual"],
                  body))


def tile_table_shape(tile_rows) -> int | None:
    """The largest square shape at which every tile in the family has a row.

    A decision table compares tiles, so every throughput cell in it has to come
    from the same shape. Taking the first row that happens to name each tile
    reads one tile at 8192 by 64 by 4096 and the next at 128 cubed, which puts
    303 and 15,135 in the same column and compares nothing. The sweep visits
    both square and rectangular shapes, so the square subset is selected here and
    the largest shape common to the whole family is the one quoted; the caller
    puts that shape in the header so the reader knows what was held fixed.
    """
    names = {f"{bm}x{bn}x{bk}" for bm, bn, bk, _wm, _wn in TILE_FAMILY}
    by_shape: dict[int, set[str]] = {}
    for r in tile_rows:
        tile = r.get("tile") or ""
        if tile not in names:
            continue
        try:
            m, n, k = int(r["m"]), int(r["n"]), int(r["k"])
        except (KeyError, ValueError):
            continue
        if not (m == n == k):
            continue
        by_shape.setdefault(m, set()).add(tile)
    complete = [m for m, seen in by_shape.items() if seen == names]
    return max(complete) if complete else None


def write_gemm_tiles(tile_rows) -> None:
    """The tile family: structure, derived shared memory, and measured throughput.

    Shape and warp arrangement are structural facts about what is instantiated;
    the shared memory column is arithmetic on the shape. The last two columns are
    the tile sweep, all at one shape, and they read pending when that sweep has
    not run.
    """
    shape = tile_table_shape(tile_rows)
    at = f"{shape} cubed" if shape else "the swept shape"
    body = []
    for bm, bn, bk, wm, wn in TILE_FAMILY:
        threads = wm * wn * 32
        shared = (bm * bk + bn * bk) * TILE_BYTES_PER_ELEMENT * TILE_STAGES
        name = f"{bm}x{bn}x{bk}"
        r = None
        if shape is not None:
            for candidate in tile_rows:
                if candidate.get("tile") != name:
                    continue
                try:
                    m, n, k = int(candidate["m"]), int(candidate["n"]), int(candidate["k"])
                except (KeyError, ValueError):
                    continue
                if m == n == k == shape:
                    r = candidate
                    break
        gflops = f"{float(r['gflops']):,.0f}" if r and fnum(r, "gflops") else PENDING
        pct = (f"{float(r['pct_baseline']):.1f}"
               if r and fnum(r, "pct_baseline") is not None else PENDING)
        body.append([name, f"{wm} by {wn}", str(threads), f"{shared:,}", gflops, pct])
    write(TABLES / "gemm_tiles.tex",
          tabular("llrrrr",
                  ["tile", "warps", "threads", "shared bytes, 3 stages",
                   f"GFLOP/s at {at}", "percent of cuBLAS"],
                  body))


def decomp_speedup(rows, variant: str) -> str:
    """Each shape's ratio against the data parallel mainloop, as a measured range.

    A single number would be a lie here: both alternative decompositions win at
    some shapes and lose at others, and that spread is the result. The cell
    therefore carries the range and the shapes it was measured over, and reads
    pending only when the sweep has no rows for the variant at all.
    """
    family, base_variant, dtype = DECOMP_BASELINE
    base = {}
    for r in find_all(rows, family, base_variant, dtype):
        if is_canonical_row(r) and fnum(r, "gflops"):
            base[int(r["m"])] = float(r["gflops"])
    ratios = []
    for r in find_all(rows, family, variant, dtype):
        if not is_canonical_row(r):
            continue
        gflops = fnum(r, "gflops")
        m = int(r["m"])
        if gflops and base.get(m):
            ratios.append((m, gflops / base[m]))
    if not ratios:
        return PENDING
    ratios.sort()
    lo = min(x for _m, x in ratios)
    hi = max(x for _m, x in ratios)
    shapes = f"{ratios[0][0]} to {ratios[-1][0]}"
    return f"{lo:.2f}x to {hi:.2f}x over {shapes} cubed"


def odd_shape_count(tile_rows) -> tuple[int, int]:
    """Rows, and distinct shapes, with an extent no tile in the family divides.

    128 is the largest extent any tile in TILE_FAMILY has, so a shape with a
    dimension that 128 does not divide is one where some tile in the family must
    predicate its edge. That is the condition the tails exist for.
    """
    shapes = set()
    count = 0
    for r in tile_rows:
        if not (r.get("tile") or ""):
            continue
        try:
            m, n, k = int(r["m"]), int(r["n"]), int(r["k"])
        except (KeyError, ValueError):
            continue
        if m % 128 == 0 and n % 128 == 0 and k % 128 == 0:
            continue
        shapes.add((m, n, k))
        count += 1
    return count, len(shapes)


def write_gemm_decomp(rows, tile_rows) -> None:
    rows_odd, shapes_odd = odd_shape_count(tile_rows)
    body = []
    for label, targets, fixes, variant, fixed in DECOMPOSITIONS:
        if fixed is not None:
            measured = fixed
        elif variant is not None:
            measured = decomp_speedup(rows, variant)
        elif label == "predicated tails":
            measured = (f"{rows_odd} rows at {shapes_odd} shapes ran the real mainloop"
                        if rows_odd else PENDING)
        else:
            measured = PENDING
        body.append([latex_escape(label), latex_escape(targets), latex_escape(fixes),
                     latex_escape(measured)])
    write(TABLES / "gemm_decomp.tex",
          tabular("lllr", ["decomposition", "shapes it targets", "what it fixes",
                           "measured"], body))


def write_cutlass_gap(rows) -> None:
    """The reference line comparison. Absent rows read pending, never estimated."""
    sizes = sorted({int(r["m"]) for r in rows
                    if r.get("family") == "gemm" and r.get("dtype") == "fp16"
                    and r.get("variant") in ("mma_opt", "cutlass")})
    body = []
    for m in sizes:
        mine = find(rows, "gemm", "mma_opt", "fp16", m)
        ref = find(rows, "gemm", "cutlass", "fp16", m)
        vendor = mine or ref
        mine_g = f"{float(mine['gflops']):,.0f}" if mine else PENDING
        ref_g = f"{float(ref['gflops']):,.0f}" if ref else PENDING
        cublas = (f"{float(vendor['baseline_gflops']):,.0f}"
                  if vendor and fnum(vendor, "baseline_gflops") else PENDING)
        if mine and ref and float(ref["gflops"]) > 0.0:
            ratio = f"{100.0 * float(mine['gflops']) / float(ref['gflops']):.1f}"
        else:
            ratio = PENDING
        body.append([f"{m} cubed", mine_g, ref_g, cublas, ratio])
    write(TABLES / "cutlass_gap.tex",
          tabular("lrrrr",
                  ["shape", "mma\\_opt GFLOP/s", "cutlass GFLOP/s",
                   "cuBLAS GFLOP/s", "mma\\_opt as percent of cutlass"],
                  body))


# ---------------------------------------------------------------------------
# The other families
# ---------------------------------------------------------------------------

# Families that have a chapter and a table of their own. Everything else in the
# summary lands in the supporting families table, so a family that appears in the
# data can never go unreported just because nobody added it to a list here.
DEDICATED_FAMILIES = {"gemm", "spmv", "fft", "conv", "scan", "reduce"}

FAMILY_LABELS = {
    "gemv": ("GEMV", "cuBLAS SGEMV"),
    "trsm": ("TRSM", "cuBLAS STRSM"),
}


def shape_label(r: dict[str, str]) -> str:
    matrix = row_matrix(r)
    if matrix:
        return latex_escape(matrix)
    m, n = int(r["m"]), int(r["n"])
    return f"{m} sq" if m == n else f"{m} by {n}"


def write_families(rows) -> None:
    """One row per (family, variant, shape or matrix) for whatever is on file.

    The lookup carries the matrix as part of the key. Without it the SpMV suite
    rows, which share a nominal m and differ only in the matrix they name, would
    collapse onto whichever row the file happened to list first.
    """
    seen = []
    for r in rows:
        fam = r.get("family", "")
        if fam and fam not in DEDICATED_FAMILIES and fam not in seen:
            seen.append(fam)
    body = []
    for family in sorted(seen):
        label, vendor = FAMILY_LABELS.get(family, (family.upper(), "the vendor"))
        for r in find_all(rows, family):
            if not is_canonical_row(r):
                continue
            pct = fnum(r, "pct_baseline")
            gflops = fnum(r, "gflops")
            body.append([
                f"{label} {latex_escape(r['variant'])}",
                shape_label(r),
                f"{gflops:.1f}" if gflops is not None else PENDING,
                f"{pct:.1f}" if pct is not None else PENDING,
                vendor,
            ])
    write(TABLES / "families.tex",
          tabular("llrrl", ["kernel", "shape", "GFLOP/s", "percent of vendor",
                            "baseline"], body))


def write_spmv_suite(rows) -> None:
    """The SpMV suite, keyed on the matrix, with the v1 percentages withdrawn.

    Correction A2: the v1 cuSPARSE wrapper built its descriptors and sized and
    allocated its workspace inside the timed region, so the denominator carried
    setup my kernels never paid. That cannot be subtracted after the fact, so the
    percent is printed as withdrawn rather than as a number. A row that names a
    matrix comes from the 1.1.0 schema and its percent stands.

    A row the plan refused reads "refused", not "pending". Pending means nobody
    has measured it yet; refused means the plan looked at this matrix and said
    no, and the two are not the same statement. docs/sparse.md carries the fill
    arithmetic behind the one refusal in the suite.
    """
    body = []
    for r in find_all(rows, "spmv"):
        if not is_canonical_row(r):
            continue
        matrix = row_matrix(r)
        gflops = fnum(r, "gflops")
        pct = fnum(r, "pct_baseline")
        refused = str(r.get("status", "")) == "skipped_not_supported"
        if refused:
            pct_cell = REFUSED
        elif not matrix:
            pct_cell = "withdrawn (A2)"
        elif pct is None:
            pct_cell = PENDING
        else:
            pct_cell = f"{pct:.1f}"
        nnz = r.get("nnz") or ""
        body.append([
            f"CSR SpMV {latex_escape(r['variant'])}",
            latex_escape(matrix) if matrix else f"synthetic, {int(r['m'])} rows",
            f"{int(nnz):,}" if str(nnz).strip().isdigit() else PENDING,
            REFUSED if refused else (f"{gflops:.1f}" if gflops is not None else PENDING),
            pct_cell,
        ])
    if not body:
        body = [["CSR SpMV suite", PENDING, PENDING, PENDING, PENDING]]
    write(TABLES / "spmv_suite.tex",
          tabular("lllrr", ["kernel", "matrix", "nonzeros", "GFLOP/s",
                            "percent of cuSPARSE"], body))


# Families whose rungs compute the same answer by algorithms with different FLOP
# counts. For these the sweep's pct_baseline, which is a ratio of throughputs,
# compares two different models and is not a statement about speed; the percent is
# recomputed here from the medians instead. See time_percent below.
TIME_RATIO_FAMILIES = {"conv"}


def time_percent(rows, r: dict[str, str]) -> float | None:
    """Percent of the vendor baseline computed from wall time, not throughput.

    Convolution is the case that forces this. A direct time domain rung does
    2 N M FLOP and an FFT rung does about 5 N log N, so dividing one rung's
    GFLOP/s by the other's compares two different models rather than two speeds,
    and on the committed rows it inverts the answer: at M = 8 the direct constant
    rung runs 0.0814 ms against cuFFT's 1.8603 ms, 23 times faster, while its
    pct_baseline reads 77.46; at M = 4096 it runs 9.1418 ms against 1.8588 ms,
    five times slower, while pct_baseline reads 352.91. Both kernels produce the
    same convolution, so the honest comparison is the time one, and higher is
    still better because this is a percent of the baseline's speed.
    """
    median = fnum(r, "median_ms")
    if not median:
        return None
    family = r.get("family")
    for candidate in rows:
        if candidate.get("family") != family:
            continue
        if not str(candidate.get("variant", "")).startswith("baseline_"):
            continue
        if not is_canonical_row(candidate):
            continue
        if candidate.get("m") != r.get("m") or candidate.get("n") != r.get("n"):
            continue
        base = fnum(candidate, "median_ms")
        if base:
            return 100.0 * base / median
    return None


def write_rung_table(name: str, rungs, rows, vendor: str, third_header: str,
                     metric: str = "gflops", metric_header: str = "GFLOP/s") -> None:
    """A mechanism table whose measured columns fill in when the rows land.

    Each rung names its family and variant exactly as the sweep writes them, so
    the join is on the row identity and not on a guess at the label. A family
    swept over more than one size contributes its canonical row at the largest
    one, ordered on the whole shape rather than on `m` alone: the convolution grid
    varies the filter length in `n` at a fixed signal length, so ordering on `m`
    leaves a tie that file order would otherwise break, which is not a decision a
    generated table should be making by accident.

    The throughput column is named by the caller because the families do not share
    a unit. Scan and reduction do a fixed two or three FLOP per element and the
    sweep writes their `gflops` as zero on purpose: the quantity that means
    anything for them is bytes moved per second, and the gate is stated in GB/s.
    Printing that zero under a GFLOP/s header would be a wrong number rather than
    a missing one.
    """
    body = []
    for family, variant, label, mechanism, derived in rungs:
        hits = [r for r in find_all(rows, family, variant) if is_canonical_row(r)]
        r = max(hits, key=lambda x: (int(x["m"]), int(x["n"]))) if hits else None
        value = fnum(r, metric) if r else None
        if r is None:
            pct = None
        elif family in TIME_RATIO_FAMILIES:
            pct = time_percent(rows, r)
        else:
            pct = fnum(r, "pct_baseline")
        body.append([
            latex_escape(label), latex_escape(mechanism), latex_escape(derived),
            f"{value:.1f}" if value is not None else PENDING,
            f"{pct:.1f}" if pct is not None else PENDING,
        ])
    write(TABLES / name,
          tabular("lllrr", ["rung", "mechanism", third_header,
                            f"{metric_header} at the largest swept size",
                            f"percent of {vendor}"], body))


def write_register_study(study_rows) -> None:
    """The register allocation study, straight out of its CSV."""
    def words(value: str) -> str:
        """A CSV enum reads better as words in a table than as an identifier."""
        return latex_escape(str(value).replace("_", " "))

    body = []
    for r in study_rows:
        body.append([
            words(r.get("study_axis", "")),
            latex_escape(r.get("maxrregcount", "")),
            latex_escape(r.get("minnctapersm", "")),
            latex_escape(r.get("registers", "")),
            latex_escape(r.get("spill_store_bytes", "")),
            latex_escape(r.get("spill_load_bytes", "")),
            latex_escape(r.get("blocks_by_registers", "")),
            latex_escape(r.get("blocks_per_sm", "")),
            words(r.get("limiter", "")),
            latex_escape(r.get("gflops", PENDING) or PENDING),
        ])
    write(TABLES / "register_study.tex",
          tabular("llrrrrrrll",
                  ["axis", "ceiling", "blocks/SM floor", "registers",
                   "spill st.", "spill ld.", "blocks by reg.", "blocks/SM",
                   "limiter", "GFLOP/s"],
                  body))


# ---------------------------------------------------------------------------
# Figures
# ---------------------------------------------------------------------------

INK = "#222222"
MUTED = "#6A6A6A"
GRID = "#E6E6E6"
FP32_C = "#0072B2"
TENSOR_C = "#E69F00"
TOP_C = "#009E73"

# One color and one marker per rung, fixed by label so a rung keeps its identity
# whatever else is on the chart. Colors are the Okabe and Ito colorblind safe set;
# the marker is a second channel so the lines stay separable in grayscale. The
# precision group is carried by the line style, not by the color.
RUNG_STYLE = {
    "naive": ("#0072B2", "o"),
    "shared tiled": ("#56B4E9", "s"),
    "register blocked": ("#CC79A7", "^"),
    "cp.async double buffered": ("#000000", "D"),
    "WMMA": ("#E69F00", "v"),
    "mma.sync PTX": ("#D55E00", "P"),
    "mma.sync + ldmatrix": ("#7A5C00", "X"),
    "mma.sync + ldmatrix + swizzle": (TOP_C, "*"),
    "CUTLASS reference line": ("#666666", "h"),
    "WMMA (BF16)": ("#B07AA1", "<"),
}


def plot_ladder(series, sizes, any_interval: bool) -> None:
    """Percent of cuBLAS against shape, every rung, every swept size.

    The v1 chart was a bar chart at one hard coded shape, which hid the thing the
    data actually says: the top kernel is not a flat 90 percent, it climbs with
    the shape because the small shapes cannot fill 48 SMs. Percent against shape
    puts that on the page.

    Line style carries the precision group and color carries the rung. Nine lines
    in two colors would be nine lines nobody can tell apart, and the group is the
    thing a reader must not get wrong, because the percent is of cuBLAS at the
    same precision and the two groups are not comparable as absolute speed. The
    colors are Okabe and Ito, and each rung keeps its own whatever else is on the
    chart, so a rung that gains or loses rows does not repaint the whole picture.
    """
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(9.6, 6.0))
    fig.subplots_adjust(left=0.085, right=0.70, top=0.82, bottom=0.13)

    for s in series:
        xs = [p[0] for p in s["points"]]
        ys = [p[1] for p in s["points"]]
        color, marker = RUNG_STYLE.get(s["label"], (INK, "o"))
        width, zorder = (2.6, 6) if s["top"] else (1.6, 4)
        style = "-" if s["dtype"] in ("fp16", "bf16") else "--"
        if any_interval and all(p[2] is not None for p in s["points"]):
            # A row whose five process repeats agreed exactly has a zero width
            # interval, and the reciprocal that converts time to percent then
            # lands a few parts in 1e15 either side of the point itself.
            # matplotlib rejects a negative bar, so the clamp is arithmetic
            # noise control, not a widening or a narrowing of any interval.
            lo = [max(0.0, p[1] - p[2]) for p in s["points"]]
            hi = [max(0.0, p[3] - p[1]) for p in s["points"]]
            ax.errorbar(xs, ys, yerr=[lo, hi], color=color, linestyle=style,
                        linewidth=width, marker=marker, markersize=5.5, capsize=3,
                        zorder=zorder, label=s["label"])
        else:
            ax.plot(xs, ys, style, color=color, linewidth=width, marker=marker,
                    markersize=5.5, zorder=zorder, label=s["label"])

    ax.set_xscale("log", base=2)
    ax.set_xticks(sizes)
    ax.set_xticklabels([f"${s}^3$" for s in sizes], fontsize=9.5)
    ax.set_xlabel("shape (m = n = k)", fontsize=10.5, fontweight="bold", color=INK)
    ax.set_ylabel("percent of cuBLAS at the same precision", fontsize=10.5,
                  fontweight="bold", color=INK)
    ax.set_ylim(0, 100)
    ax.grid(True, which="major", color=GRID, linewidth=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)

    fig.legend(loc="center left", bbox_to_anchor=(0.715, 0.5), frameon=True,
               framealpha=0.95, edgecolor="#DDDDDD", fontsize=8.5,
               title="rung (solid: tensor core, dashed: FP32)", title_fontsize=8.5)
    fig.suptitle("The GEMM ladder across shapes", x=0.5, y=0.955, fontsize=15,
                 fontweight="bold", color=INK)
    fig.text(0.5, 0.895,
             "percent is of cuBLAS at the same precision, so the solid and dashed "
             "groups are not comparable as absolute speed",
             ha="center", va="top", fontsize=9.5, color=MUTED)
    fig.savefig(FIGURES / "ladder.pdf")
    fig.savefig(FIGURES / "ladder.png", dpi=150)
    plt.close(fig)


def conv_crossover_rows(rows) -> dict[str, dict[int, dict[int, float]]]:
    """Canonical convolution rows as {signal length: {filter length: seconds}}.

    Two series come back keyed by the fastest FFT variant and the fastest direct
    variant at each filter length, because which member of each family wins is
    itself a result and the chart is about the two families crossing.
    """
    fft_variants = {"fft_separate", "fft_fused"}
    direct_variants = {"direct_shared", "direct_constant"}
    best_fft: dict[int, dict[int, float]] = {}
    best_direct: dict[int, dict[int, float]] = {}
    for r in rows:
        if r.get("family") != "conv" or not is_canonical_row(r):
            continue
        n = int(r["m"])
        m = int(r["n"])
        ms = float(r["median_ms"])
        target = None
        if r["variant"] in fft_variants:
            target = best_fft.setdefault(n, {})
        elif r["variant"] in direct_variants:
            target = best_direct.setdefault(n, {})
        if target is None:
            continue
        if m not in target or ms < target[m]:
            target[m] = ms
    return {"fft": best_fft, "direct": best_direct}


def crossing_filter_length(fft_series: dict[int, float],
                           direct_series: dict[int, float]) -> float | None:
    """The M where the two curves cross, read out of the data.

    Log-linear interpolation between the two filter lengths that bracket the sign
    change of (direct - fft). None when the sweep does not contain a crossing, in
    which case the caption says so rather than inventing one.
    """
    import math

    shared = sorted(set(fft_series) & set(direct_series))
    for lo, hi in zip(shared, shared[1:], strict=False):
        d_lo = direct_series[lo] - fft_series[lo]
        d_hi = direct_series[hi] - fft_series[hi]
        if d_lo <= 0.0 <= d_hi and d_hi != d_lo:
            frac = -d_lo / (d_hi - d_lo)
            return math.exp(math.log(lo) + frac * (math.log(hi) - math.log(lo)))
    return None


def plot_conv_crossover(rows) -> None:
    """The convolution crossover chart, generated from committed rows only.

    The chart has to come out of the sweep data with the crossing read by the
    script and printed in the caption, never placed by hand. When there are no
    convolution rows yet the figure is removed rather than left stale, and the
    block the FFT chapter inputs becomes a paragraph saying the chart is pending:
    a report rebuilt around a picture whose input is gone is defect A7.
    """
    series = conv_crossover_rows(rows)
    signals = sorted(set(series["fft"]) & set(series["direct"]))
    targets = [FIGURES / "conv_crossover.pdf", FIGURES / "conv_crossover.png"]
    block = TABLES / "conv_crossover_block.tex"
    if not signals:
        for path in targets:
            if path.exists():
                path.unlink()
        write(block,
              "The convolution crossover chart is pending: the committed summary "
              "carries no convolution rows, so there is nothing to plot and nothing "
              "is drawn. It appears here, with the crossing filter length read out "
              "of the data, once the sweep has run the convolution grid.\n")
        print("conv crossover: no convolution rows in summary.csv, so no chart. "
              "Run the conv family through benchmarks/sweep.py first.")
        return

    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    fft_c, direct_c = FP32_C, TENSOR_C

    fig, ax = plt.subplots(figsize=(9.0, 5.6))
    fig.subplots_adjust(left=0.11, right=0.97, top=0.84, bottom=0.16)
    caption_lines = []
    for i, n in enumerate(signals):
        fft_series = series["fft"][n]
        direct_series = series["direct"][n]
        ms = sorted(set(fft_series) & set(direct_series))
        style = "-" if i == 0 else "--"
        ax.plot(ms, [fft_series[m] for m in ms], style, color=fft_c, marker="o", markersize=4,
                label=f"FFT path, N = {n}")
        ax.plot(ms, [direct_series[m] for m in ms], style, color=direct_c, marker="s",
                markersize=4, label=f"direct path, N = {n}")
        crossing = crossing_filter_length(fft_series, direct_series)
        if crossing is None:
            caption_lines.append(
                f"At N = {n} the sweep contains no crossing: one path is faster at every "
                f"filter length measured.")
            continue
        ax.axvline(crossing, color=MUTED, linestyle=":", linewidth=1.2)
        ax.annotate(f"crossing at M = {crossing:.0f}", xy=(crossing, ax.get_ylim()[1]),
                    xytext=(4, -12), textcoords="offset points", fontsize=8.5, color=INK)
        caption_lines.append(f"At N = {n} the two paths cross at M = {crossing:.0f}.")

    ax.set_xscale("log", base=2)
    ax.set_yscale("log")
    ax.set_xlabel("filter length M (taps)", fontsize=10.5, fontweight="bold", color=INK)
    ax.set_ylabel("time per convolution (ms, lower is better)", fontsize=10.5,
                  fontweight="bold", color=INK)
    ax.grid(True, which="both", color=GRID, linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)
    ax.legend(frameon=True, framealpha=0.95, edgecolor="#DDDDDD", fontsize=8.5)
    fig.suptitle("Convolution crossover", x=0.5, y=0.96, fontsize=15, fontweight="bold",
                 color=INK)
    fig.text(0.5, 0.9,
             "the FFT path barely moves with M; the direct path is 2 N M and does not stop",
             ha="center", va="top", fontsize=9.5, color=MUTED)
    fig.savefig(FIGURES / "conv_crossover.pdf")
    fig.savefig(FIGURES / "conv_crossover.png", dpi=150)
    plt.close(fig)
    write(block,
          "\\begin{figure}[htbp]\n\\centering\n"
          "\\includegraphics[width=0.9\\textwidth]{conv_crossover.pdf}\n"
          "\\caption{The convolution crossover, generated from the committed sweep "
          "rows with the crossing read out of the data by log interpolation. "
          + " ".join(caption_lines) + "}\n\\label{fig:conv}\n\\end{figure}\n")
    print(f"conv crossover: {' '.join(caption_lines)}")


# ---------------------------------------------------------------------------


def main() -> int:
    require_inputs()
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = read_csv(RESULTS / "summary.csv")
    study = read_csv(RESULTS / "register_study.csv")
    tiles = read_optional(RESULTS / "tile_sweep.csv")

    for path, how in OPTIONAL_INPUTS:
        if not path.exists():
            print(f"note: {path.relative_to(REPO)} is absent, so the table it fills "
                  f"reads pending. Produce it with {how}.")

    steps = ["gemm ladder table", "gemm gap table", "tile family table",
             "decomposition table", "cutlass gap table", "families table",
             "spmv suite table", "fft table", "scan table", "register study table",
             "roofline figure", "ladder figure", "conv crossover figure"]
    try:
        from tqdm import tqdm
        it = tqdm(steps, unit="asset")
    except ImportError:
        it = steps

    series, sizes, any_interval = ladder_series(rows)
    for step in it:
        if step == "gemm ladder table":
            write_gemm_ladder(rows, 4096)
        elif step == "gemm gap table":
            write_gemm_gap(rows)
        elif step == "tile family table":
            write_gemm_tiles(tiles)
        elif step == "decomposition table":
            write_gemm_decomp(rows, tiles)
        elif step == "cutlass gap table":
            write_cutlass_gap(rows)
        elif step == "families table":
            write_families(rows)
        elif step == "spmv suite table":
            write_spmv_suite(rows)
        elif step == "fft table":
            write_rung_table("fft.tex", FFT_RUNGS, rows, "cuFFT", "traffic model")
        elif step == "scan table":
            write_rung_table("scan.tex", SCAN_RUNGS, rows, "CUB", "bytes moved",
                             metric="effective_gbs", metric_header="effective GB/s")
        elif step == "register study table":
            write_register_study(study)
        elif step == "roofline figure":
            subprocess.run([sys.executable, str(REPO / "scripts" / "plot_roofline.py")],
                           check=True)
        elif step == "ladder figure":
            plot_ladder(series, sizes, any_interval)
            write_ladder_caption(series, sizes, any_interval)
        elif step == "conv crossover figure":
            plot_conv_crossover(rows)

    print(f"wrote tables to {TABLES.relative_to(REPO)} and figures to {FIGURES.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
