#!/usr/bin/env python3
"""Generate the report's tables and figures from the canonical results.

Idempotent: the same summary data produces the same booktabs tables and the same
figures. Reads experiments/results/summary.csv, writes LaTeX tables into
report/tables/ and figures into report/figures/. The roofline figure is rendered
by plot_roofline.py (from the profiler's CSVs); this script also renders a GEMM
ladder bar chart. Nothing is hand copied into the report; it all comes from here.
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

LADDER_ORDER = [
    ("gemm", "naive", "fp32", "naive"),
    ("gemm", "tiled", "fp32", "shared tiled"),
    ("gemm", "register", "fp32", "register blocked"),
    ("gemm", "cp_async", "fp32", "cp.async double buffered"),
    ("gemm", "wmma", "fp16", "WMMA"),
    ("gemm", "mma_ptx", "fp16", "mma.sync PTX"),
    ("gemm", "mma_ldm", "fp16", "mma.sync + ldmatrix"),
    ("gemm", "mma_opt", "fp16", "mma.sync + ldmatrix + swizzle"),
    ("gemm", "wmma", "bf16", "WMMA (BF16)"),
]


def read_summary() -> list[dict[str, str]]:
    with (RESULTS / "summary.csv").open() as f:
        return list(csv.DictReader(f))


def find(rows, family, variant, dtype, m):
    for r in rows:
        if (r["family"] == family and r["variant"] == variant and r["dtype"] == dtype
                and int(r["m"]) == m):
            return r
    return None


def latex_escape(s: str) -> str:
    return s.replace("_", r"\_").replace("%", r"\%")


def write_gemm_ladder(rows, size: int) -> list[tuple[str, float, float]]:
    out = TABLES / "gemm_ladder.tex"
    plotted = []
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"rung & precision & GFLOP/s & percent of cuBLAS \\",
        r"\midrule",
    ]
    for family, variant, dtype, label in LADDER_ORDER:
        r = find(rows, family, variant, dtype, size)
        if r is None:
            continue
        g = float(r["gflops"])
        pct = float(r["pct_baseline"])
        lines.append(f"{latex_escape(label)} & {dtype} & {g:.0f} & {pct:.1f} \\\\")
        plotted.append((label, dtype, g, pct))
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")
    return plotted


def write_families(rows) -> None:
    out = TABLES / "families.tex"
    spec = [
        ("gemv", "warp", "fp32", 8192, "GEMV warp"),
        ("gemv", "vectorized", "fp32", 8192, "GEMV vectorized"),
        ("spmv", "warp", "fp32", 131072, "CSR SpMV warp"),
        ("spmv", "naive", "fp32", 131072, "CSR SpMV naive"),
        ("trsm", "blocked", "fp32", 2048, "TRSM blocked"),
        ("trsm", "naive", "fp32", 2048, "TRSM naive"),
    ]
    lines = [
        r"\begin{tabular}{llrr}",
        r"\toprule",
        r"kernel & shape & GFLOP/s & percent of vendor \\",
        r"\midrule",
    ]
    for family, variant, dtype, m, label in spec:
        r = find(rows, family, variant, dtype, m)
        if r is None:
            continue
        shape = f"{m}" if family != "gemv" else f"{m} sq"
        lines.append(f"{latex_escape(label)} & {shape} & {float(r['gflops']):.1f} & "
                     f"{float(r['pct_baseline']):.1f} \\\\")
    lines += [r"\bottomrule", r"\end{tabular}", ""]
    out.write_text("\n".join(lines), encoding="utf-8")


def plot_ladder(plotted: list[tuple[str, str, float, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    ink = "#222222"
    muted = "#6A6A6A"
    # Colors carry both the compute path and the baseline the percent is measured
    # against: FP32 kernels versus cuBLAS SGEMM, tensor kernels versus cuBLAS FP16
    # or BF16. The top kernel is called out in green.
    fp32_c, tensor_c, top_c = "#0072B2", "#E69F00", "#009E73"

    labels = [p[0] for p in plotted]
    gflops = [p[2] for p in plotted]
    colors = []
    fp32_count = 0
    for label, dtype, _g, _pct in plotted:
        if "swizzle" in label:
            colors.append(top_c)
        elif dtype in ("fp16", "bf16"):
            colors.append(tensor_c)
        else:
            colors.append(fp32_c)
            fp32_count += 1

    fig, ax = plt.subplots(figsize=(9.6, 6.4))
    # Fixed margins (not a tight bbox) so the figure centered title sits at the
    # center of the whole picture, and the legend has reserved space below.
    fig.subplots_adjust(left=0.235, right=0.965, top=0.82, bottom=0.26)

    y = list(range(len(labels)))
    ax.barh(y, gflops, color=colors, edgecolor="white", height=0.72, zorder=3)
    ax.set_yticks(y)
    ax.set_yticklabels(labels, fontsize=9.5, color=ink)
    ax.invert_yaxis()

    # Bar end labels: absolute throughput (the bar length) and the percent of the
    # same precision cuBLAS baseline (the color says which baseline).
    for i, (_l, _dt, g, pct) in enumerate(plotted):
        ax.text(g, i, f"  {g:,.0f} GFLOP/s   ({pct:.0f}% of cuBLAS)", va="center",
                fontsize=8.5, color=ink)

    # Dashed divider between the FP32 group (top) and the tensor core group
    # (bottom): the plot is really two grouped bar charts sharing one axis.
    ax.axhline(fp32_count - 0.5, color=muted, linestyle="--", linewidth=1.4, zorder=4)

    ax.set_xlabel("achieved throughput at $4096^3$  (bar length, higher is better)",
                  fontsize=10.5, fontweight="bold", color=ink)

    handles = [
        Patch(facecolor=fp32_c, edgecolor="white",
              label="FP32 kernel  (percent of cuBLAS SGEMM, about 23 TFLOP/s)"),
        Patch(facecolor=tensor_c, edgecolor="white",
              label="tensor core kernel  (percent of cuBLAS FP16 or BF16, about 65 TFLOP/s)"),
        Patch(facecolor=top_c, edgecolor="white", label="top kernel"),
    ]
    # Legend in its own reserved space at the bottom left, below the axis label.
    fig.legend(handles=handles, loc="lower left", bbox_to_anchor=(0.02, 0.015),
               frameon=True, framealpha=0.95, edgecolor="#DDDDDD", fontsize=8.5, ncol=1)

    # Title centered on the whole figure, with an explanatory subtitle under it.
    fig.suptitle("GEMM optimization ladder", x=0.5, y=0.955, fontsize=15,
                 fontweight="bold", color=ink)
    fig.text(0.5, 0.9,
             "each rung is one attributable step; percent is of cuBLAS at the same precision",
             ha="center", va="top", fontsize=9.5, color=muted)

    ax.grid(True, axis="x", color="#E6E6E6", linewidth=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(muted)
    ax.tick_params(colors=muted, labelcolor=ink)
    ax.set_xlim(0, max(gflops) * 1.62)
    fig.savefig(FIGURES / "ladder.pdf")
    fig.savefig(FIGURES / "ladder.png", dpi=150)


def main() -> int:
    TABLES.mkdir(parents=True, exist_ok=True)
    FIGURES.mkdir(parents=True, exist_ok=True)
    rows = read_summary()

    steps = ["gemm ladder table", "families table", "roofline figure", "ladder figure"]
    try:
        from tqdm import tqdm
        it = tqdm(steps, unit="asset")
    except ImportError:
        it = steps

    plotted = []
    for step in it:
        if step == "gemm ladder table":
            plotted = write_gemm_ladder(rows, 4096)
        elif step == "families table":
            write_families(rows)
        elif step == "roofline figure":
            if (RESULTS / "roofline.csv").exists():
                subprocess.run([sys.executable, str(REPO / "scripts" / "plot_roofline.py")],
                               check=True)
        elif step == "ladder figure":
            plot_ladder(plotted)

    print(f"wrote tables to {TABLES.relative_to(REPO)} and figures to {FIGURES.relative_to(REPO)}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
