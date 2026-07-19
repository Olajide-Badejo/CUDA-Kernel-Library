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
        plotted.append((label, g, pct))
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


def plot_ladder(plotted: list[tuple[str, float, float]]) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    labels = [p[0] for p in plotted]
    gflops = [p[1] for p in plotted]
    # Okabe-Ito, FP32 rungs cool, tensor rungs warm; the top kernel highlighted.
    colors = []
    for label, _g, _pct in plotted:
        if "swizzle" in label:
            colors.append("#009E73")
        elif any(t in label for t in ("WMMA", "mma")):
            colors.append("#E69F00")
        else:
            colors.append("#0072B2")

    fig, ax = plt.subplots(figsize=(8.4, 4.6))
    y = range(len(labels))
    ax.barh(list(y), gflops, color=colors, edgecolor="white", height=0.7)
    ax.set_yticks(list(y))
    ax.set_yticklabels(labels, fontsize=9)
    ax.invert_yaxis()
    ax.set_xlabel("GFLOP/s at 4096 cubed (higher is better)", fontsize=10)
    ax.set_title("GEMM ladder throughput", fontsize=12)
    for i, (_l, g, pct) in enumerate(plotted):
        ax.text(g, i, f"  {g:.0f}  ({pct:.0f}% cuBLAS)", va="center", fontsize=8.5)
    ax.grid(True, axis="x", color="#E2E2E2", linewidth=0.6)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    ax.set_xlim(0, max(gflops) * 1.35)
    fig.tight_layout()
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
