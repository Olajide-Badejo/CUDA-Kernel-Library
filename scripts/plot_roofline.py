#!/usr/bin/env python3
"""Render the roofline figure from the profiler's CSV output.

Reads experiments/results/roofline.csv (per variant operating points) and
experiments/results/roofline_ceilings.csv (bandwidth, compute roofs, device name,
clock), and writes a log log roofline PNG and PDF.

The figure is meant to be readable by someone who has not seen a roofline before:
the two regions (memory bound to the left of the ridge, compute bound to the
right) are shaded and labeled, the sloped edge is the measured bandwidth ceiling,
the flat edges are the hardware compute roofs, and each kernel is a labeled
point. A kernel under the sloped edge is limited by bandwidth; one under a flat
edge is limited by the math units.

The compute roofs are hardware (SMs times clock times FLOP per cycle), and cuBLAS
is drawn as a separate dashed attainable line rather than as the roof. Drawing the
cuBLAS measurement as the ceiling is what made every v1 "percent of roof" a
restatement of "percent of cuBLAS" (defect A3). The chart title comes from the
device name in the ceilings CSV, not from a hard coded model.

Colors are the Okabe and Ito colorblind safe set; FP32 and tensor kernels also
differ in marker shape, so identity survives grayscale and color vision
deficiency. Usage: plot_roofline.py [results_dir] [out_dir]
"""

from __future__ import annotations

import csv
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
from matplotlib.lines import Line2D

# Okabe and Ito qualitative palette, fixed order. Colorblind safe by construction.
OKABE_ITO = ["#0072B2", "#D55E00", "#009E73", "#CC79A7", "#E69F00", "#56B4E9", "#000000"]
INK = "#222222"
MUTED = "#7A7A7A"
GRID = "#E2E2E2"
MEM_TINT = "#4C78A8"      # cool: memory bound region
COMPUTE_TINT = "#E69F00"  # warm: compute bound region
ATTAINABLE_TINT = "#7A5C00"  # the cuBLAS attainable lines, distinct from the roofs

# Friendlier display names and the precision each kernel runs in. The SpMV, FFT
# and scan entries are the hooks for the 1.1.0 families: the profiler emits an
# operating point per variant and this plotter draws whatever the CSV carries, so
# those points join the chart when their rows land, with no code change here. A
# label that is not in this table is drawn under its own name rather than dropped.
DISPLAY = {
    "gemm_naive": ("GEMM naive", "fp32"),
    "gemm_cp_async": ("GEMM cp.async", "fp32"),
    "gemm_wmma_fp16": ("GEMM WMMA", "tensor"),
    "gemm_mma_opt": ("GEMM mma + swizzle (top)", "tensor"),
    "gemv_warp": ("GEMV warp", "fp32"),
    "spmv_warp": ("CSR SpMV warp", "fp32"),
    "spmv_merge": ("CSR SpMV merge", "fp32"),
    "spmv_sell": ("SpMV SELL-C-sigma", "fp32"),
    "fft_four_step": ("FFT four step", "fp32"),
    "fft_shared": ("FFT shared resident", "fp32"),
    "scan_lookback": ("scan look-back", "fp32"),
    "reduce_vec4": ("reduction vectorized", "fp32"),
    "cublas_attainable": ("cuBLAS FP16 (attainable)", "tensor"),
}

# Which family each label belongs to, for the generated caption. The caption has
# to name the families that are actually on the chart and the ones that are not,
# because "the roofline carries four families" is a claim and it has to be true of
# the picture rather than of the plan.
FAMILY_OF = {
    "gemm": "GEMM", "gemv": "GEMV", "spmv": "SpMV", "fft": "FFT",
    "conv": "convolution", "scan": "scan", "reduce": "reduction",
}
CAPTION_FAMILIES = ["gemm", "gemv", "spmv", "fft", "scan"]

# Drawn as a dashed attainable line, not as a kernel marker.
ATTAINABLE_LABEL = "cublas_attainable"


def read_ceilings(path: Path) -> tuple[dict[str, float], dict[str, str]]:
    """Split the ceilings file into its numeric entries and its text entries."""
    numbers: dict[str, float] = {}
    text: dict[str, str] = {}
    with path.open() as f:
        for row in csv.DictReader(f):
            key, raw = row["quantity"], row["value"]
            try:
                numbers[key] = float(raw)
            except ValueError:
                text[key] = raw
    return numbers, text


def read_points(path: Path) -> list[dict[str, str]]:
    with path.open() as f:
        return list(csv.DictReader(f))


def place_diagonal_label(ax, bw: float, x_at: float) -> None:
    """Label the bandwidth diagonal, rotated to match the line as it is drawn."""
    ax.figure.canvas.draw()
    p0 = ax.transData.transform((x_at, bw * x_at))
    p1 = ax.transData.transform((x_at * 10.0, bw * x_at * 10.0))
    angle = np.degrees(np.arctan2(p1[1] - p0[1], p1[0] - p0[0]))
    ax.text(x_at, bw * x_at * 1.28, f"bandwidth ceiling\n{bw:.0f} GB/s",
            rotation=angle, rotation_mode="anchor", fontsize=9.5, fontweight="bold",
            color=INK, va="bottom", ha="center",
            bbox=dict(boxstyle="round,pad=0.2", fc="white", ec="none", alpha=0.85))


def main() -> int:
    results = Path(sys.argv[1]) if len(sys.argv) > 1 else Path("experiments/results")
    out_dir = Path(sys.argv[2]) if len(sys.argv) > 2 else Path("report/figures")
    out_dir.mkdir(parents=True, exist_ok=True)

    ceilings, labels = read_ceilings(results / "roofline_ceilings.csv")
    points = read_points(results / "roofline.csv")

    bw = ceilings["bandwidth_gbps"]     # GB/s: bw * intensity gives GFLOP/s
    fp32 = ceilings["fp32_gflops"]
    tensor = ceilings["tensor_gflops"]
    cublas_tensor = ceilings.get("cublas_tensor_gflops")
    cublas_fp32 = ceilings.get("cublas_fp32_gflops")
    device_name = labels.get("device_name", "this GPU")
    clock_mhz = ceilings.get("clock_mhz")
    ridge = tensor / bw                 # outer envelope ridge (tensor)
    ridge_fp32 = fp32 / bw

    x_lo, x_hi = 0.1, 1.0e4
    y_lo, y_hi = 100.0, tensor * 1.7

    fig, ax = plt.subplots(figsize=(8.6, 5.8))
    ax.set_xscale("log")
    ax.set_yscale("log")
    ax.set_xlim(x_lo, x_hi)
    ax.set_ylim(y_lo, y_hi)

    # Region shading: left of the ridge is bandwidth limited, right is compute
    # limited. Very light tints so the data stays dominant.
    ax.axvspan(x_lo, ridge, color=MEM_TINT, alpha=0.07, zorder=0)
    ax.axvspan(ridge, x_hi, color=COMPUTE_TINT, alpha=0.07, zorder=0)
    ax.axvline(ridge, color=MUTED, linestyle=":", linewidth=1.3, zorder=1)
    # Point at the envelope corner with an arrow, so the caption stays clear of the
    # roof labels along the top.
    ax.annotate("ridge point\n(peak / bandwidth)", xy=(ridge, tensor), xytext=(300, 2600),
                textcoords="data", fontsize=8.5, color=MUTED, va="center", ha="center",
                arrowprops=dict(arrowstyle="->", color=MUTED, lw=1.1,
                                connectionstyle="arc3,rad=-0.2"))

    # Big region labels in the open space of each region.
    ax.text(0.55, 150, "MEMORY BOUND", fontsize=13, fontweight="bold",
            color=MEM_TINT, alpha=0.55, va="center", ha="left")
    ax.text(0.62, 118, "limited by bandwidth", fontsize=9, color=MEM_TINT, alpha=0.7,
            va="center", ha="left")
    ax.text(ridge * 1.25, 150, "COMPUTE BOUND", fontsize=13, fontweight="bold",
            color="#B4790B", alpha=0.7, va="center", ha="left")
    ax.text(ridge * 1.3, 118, "limited by the math units", fontsize=9, color="#B4790B",
            alpha=0.85, va="center", ha="left")

    x = np.logspace(np.log10(x_lo), np.log10(x_hi), 500)

    # Hardware compute roofs (flat), sharing the bandwidth diagonal. Labeled at
    # the shoulder just right of each ridge, clear of the data points on the far
    # right. These are SMs times clock times FLOP per cycle: the machine, not a
    # vendor library.
    ax.plot(x, np.minimum(tensor, bw * x), "-", color=INK, linewidth=2.2, zorder=3)
    ax.plot(x, np.minimum(fp32, bw * x), "-", color=INK, linewidth=1.6, zorder=3)
    ax.text(ridge * 1.2, tensor * 1.12, f"tensor roof  {tensor / 1000:.1f} TFLOP/s",
            fontsize=9, color=INK, va="bottom", ha="left")
    ax.text(ridge_fp32 * 1.2, fp32 * 1.12, f"FP32 roof  {fp32 / 1000:.1f} TFLOP/s",
            fontsize=9, color=INK, va="bottom", ha="left")

    # cuBLAS is what the vendor attains on this machine, drawn dashed and under
    # its roof so nobody reads it as the ceiling. Labels sit below the line at the
    # far right: the FP16 line runs close enough to the tensor roof that a label
    # above it would collide.
    # The third element is how far below its own line the label is dropped. The
    # FP16 line runs within a few percent of the tensor roof, so its label has to
    # clear the top kernel's marker as well as the roof.
    for value, name, drop in ((cublas_tensor, "cuBLAS FP16", 0.78),
                              (cublas_fp32, "cuBLAS SGEMM", 0.9)):
        if value is None:
            continue
        ax.plot([value / bw, x_hi], [value, value], "--", color=ATTAINABLE_TINT,
                linewidth=1.6, zorder=2)
        ax.text(x_hi * 0.95, value * drop, f"{name}  {value / 1000:.1f} TFLOP/s",
                fontsize=8.5, color=ATTAINABLE_TINT, va="top", ha="right")

    # Ridge tick marks on each roof.
    for peak in (fp32, tensor):
        ax.plot([peak / bw], [peak], marker="|", color=MUTED, markersize=11, zorder=4)

    # Kernel operating points. Circle = FP32, triangle = tensor. Direct labels.
    color_idx = 0
    for p in points:
        if p["label"] == ATTAINABLE_LABEL:
            continue  # already drawn as the dashed attainable line
        intensity = float(p["intensity_flop_per_byte"])
        gflops = float(p["achieved_gflops"])
        name, prec = DISPLAY.get(p["label"], (p["label"], p["ceiling"]))
        is_tensor = prec == "tensor"
        color = OKABE_ITO[color_idx % len(OKABE_ITO)]
        color_idx += 1
        ax.plot([intensity], [gflops], marker="^" if is_tensor else "o", color=color,
                markersize=12, markeredgecolor="white", markeredgewidth=1.6,
                linestyle="none", zorder=6)
        # Points far to the right get their label on the left, so it does not run
        # into the attainable line labels along the right edge.
        left_side = intensity > 100.0
        ax.annotate(name, (intensity, gflops), textcoords="offset points",
                    xytext=(-12 if left_side else 12, -2),
                    ha="right" if left_side else "left",
                    fontsize=9, color=INK, fontweight="medium", zorder=6)

    ax.set_xlabel("arithmetic intensity  (FLOP per byte moved)", fontsize=10.5, color=INK)
    ax.set_ylabel("performance  (GFLOP/s)", fontsize=10.5, color=INK)
    clock_note = f" at {clock_mhz:.0f} MHz" if clock_mhz else ""
    ax.set_title(f"Roofline on {device_name}{clock_note}", fontsize=12.5, color=INK)
    ax.grid(True, which="major", color=GRID, linewidth=0.7, zorder=0)
    ax.grid(True, which="minor", color=GRID, linewidth=0.4, alpha=0.6, zorder=0)
    for spine in ("top", "right"):
        ax.spines[spine].set_visible(False)
    for spine in ("left", "bottom"):
        ax.spines[spine].set_color(MUTED)
    ax.tick_params(colors=MUTED, labelcolor=INK)

    handles = [
        Line2D([0], [0], marker="o", color=INK, linestyle="none", markersize=9,
               markeredgecolor="white", label="FP32 kernel"),
        Line2D([0], [0], marker="^", color=INK, linestyle="none", markersize=9,
               markeredgecolor="white", label="tensor core kernel"),
        Line2D([0], [0], color=INK, linewidth=2.0, label="hardware roof (SM x clock x FLOP/cycle)"),
        Line2D([0], [0], color=ATTAINABLE_TINT, linewidth=1.6, linestyle="--",
               label="cuBLAS attainable (not a ceiling)"),
    ]
    ax.legend(handles=handles, loc="upper left", frameon=True, framealpha=0.95,
              edgecolor=GRID, fontsize=9)

    fig.tight_layout()
    place_diagonal_label(ax, bw, x_at=4.0)

    png = out_dir / "roofline.png"
    pdf = out_dir / "roofline.pdf"
    fig.savefig(png, dpi=150)
    fig.savefig(pdf)

    write_caption(out_dir.parent / "tables", points)
    print(f"wrote {png} and {pdf}")
    return 0


def write_caption(tables_dir: Path, points: list[dict[str, str]]) -> None:
    """Name the families that are on the chart, and the ones that are not.

    Generated rather than written, so the caption cannot claim a family whose
    points are absent. The 1.1.0 spec asks the roofline to carry GEMM, SpMV, FFT
    and scan; until those sweeps run it carries what it carries and says so.
    """
    drawn = set()
    for p in points:
        if p["label"] == ATTAINABLE_LABEL:
            continue
        drawn.add(p["label"].split("_")[0])
    present = [FAMILY_OF[f] for f in CAPTION_FAMILIES if f in drawn]
    absent = [FAMILY_OF[f] for f in CAPTION_FAMILIES if f not in drawn]
    parts = ["Families on the chart: " + ", ".join(present) + "."]
    if absent:
        parts.append("Pending, with no operating points in the committed profiler "
                     "output yet: " + ", ".join(absent) + ". The plotter draws "
                     "whatever the CSV carries, so those points appear here when "
                     "their sweep rows land.")
    tables_dir.mkdir(parents=True, exist_ok=True)
    # A macro definition, not bare text: an \input inside a \caption breaks when
    # hyperref writes the caption to the list of figures, so main.tex loads this
    # file in the preamble and the figure uses the macro.
    (tables_dir / "roofline_caption.tex").write_text(
        "\\newcommand{\\rooflinecaption}{" + " ".join(parts) + "}\n", encoding="utf-8")


if __name__ == "__main__":
    raise SystemExit(main())
