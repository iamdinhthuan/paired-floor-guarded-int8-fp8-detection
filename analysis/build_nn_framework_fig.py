#!/usr/bin/env python3
"""Redraw fig_framework.pdf (paired clean/corruption protocol schematic) as a
vector figure. The packaged figure was a rasterized JPEG carrying a comma
splice ("support interpretation, they do not"); this regenerates the same
layout as vector PDF/PNG with corrected text."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch, FancyArrowPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"

BLUE = "#2563EB"
DARK = "#1F2937"
GREY = "#475569"
RED = "#B91C1C"
PALE = "#F8FAFC"


def box(ax, x, y, w, h, label=None, fc="white", ec=DARK, lw=1.0,
        fontsize=8.6, dashed=False, weight="normal", tcolor=DARK):
    b = FancyBboxPatch((x, y), w, h,
                       boxstyle="round,pad=0.006,rounding_size=0.010",
                       facecolor=fc, edgecolor=ec, linewidth=lw,
                       linestyle="--" if dashed else "-")
    ax.add_patch(b)
    if label:
        ax.text(x + w / 2, y + h / 2, label, ha="center", va="center",
                fontsize=fontsize, color=tcolor, fontweight=weight)
    return b


def arrow(ax, a, b, color=DARK, lw=1.1, style="-|>", curve=0.0):
    ax.add_patch(FancyArrowPatch(
        a, b, arrowstyle=style, mutation_scale=10, linewidth=lw, color=color,
        connectionstyle=f"arc3,rad={curve}"))


def main() -> None:
    fig, ax = plt.subplots(figsize=(6.9, 3.3))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # ---- left: source and input-pair branch ----
    box(ax, 0.015, 0.56, 0.10, 0.16, "Source\nimage", fc=PALE, fontsize=8.8)
    box(ax, 0.165, 0.70, 0.165, 0.13,
        "Clean control\n(JPEG q95)", fc="white", fontsize=8.0)
    box(ax, 0.165, 0.44, 0.165, 0.13,
        "Corrupted input\n(severity s)", fc="white", fontsize=8.0)
    arrow(ax, (0.118, 0.66), (0.162, 0.765), curve=-0.22)
    arrow(ax, (0.118, 0.62), (0.162, 0.505), curve=0.22)
    ax.text(0.140, 0.615, "identical\nbytes", ha="center", va="center",
            fontsize=6.4, color=GREY)

    # ---- middle: precision branch, 4 stacked engine boxes ----
    box(ax, 0.365, 0.35, 0.165, 0.56, "", fc="none", ec=BLUE, lw=1.0,
        dashed=True)
    ax.text(0.4475, 0.935, "Precision branch", ha="center", va="center",
            fontsize=8.6, color=BLUE, fontweight="bold")
    engines = [(0.785, "TensorRT INT8", "clean input", BLUE),
               (0.665, "TensorRT FP8", "clean input", DARK),
               (0.475, "TensorRT INT8", "corrupted", BLUE),
               (0.355, "TensorRT FP8", "corrupted", DARK)]
    for y, label, cond, c in engines:
        box(ax, 0.372, y, 0.151, 0.095, f"{label}\n{cond}", fc=PALE, ec=c,
            fontsize=7.2, weight="bold" if "INT8" in label else "normal",
            tcolor=c)
    arrow(ax, (0.332, 0.765), (0.369, 0.832), curve=0.12)
    arrow(ax, (0.332, 0.505), (0.369, 0.522), curve=-0.12)

    # ---- right: 2x2 AP grid ----
    gx, gy, cw, ch = 0.60, 0.42, 0.125, 0.175
    for cidx, name in enumerate(("Clean", "Corrupted")):
        ax.text(gx + cidx * cw + cw / 2, 0.945, name, ha="center",
                fontsize=8.4, color=DARK, fontweight="bold")
        ax.text(gx + cidx * cw + cw / 2, 0.895,
                ("G$_{j95}$ = clean gap" if cidx == 0 else "G$_c$ = corrupted gap"),
                ha="center", fontsize=6.8, color=BLUE)
    for r, name in enumerate(("INT8", "FP8")):
        ax.text(gx - 0.015, gy + (1 - r) * ch + ch / 2, name, ha="right",
                va="center", fontsize=8.4, color=DARK, fontweight="bold")
    cells = [["AP$_{j95}$ INT8", "AP$_c$ INT8"],
             ["AP$_{j95}$ FP8", "AP$_c$ FP8"]]
    for r, row in enumerate(cells):
        for cidx, lab in enumerate(row):
            x, y = gx + cidx * cw, gy + (1 - r) * ch
            ax.add_patch(Rectangle((x, y), cw, ch, facecolor="white",
                                   edgecolor=DARK, linewidth=0.9))
            ax.text(x + cw / 2, y + ch / 2, lab, ha="center", va="center",
                    fontsize=7.8, color=DARK)

    # ---- output ----
    arrow(ax, (gx + 2 * cw + 0.012, 0.595), (0.925, 0.595))
    box(ax, 0.927, 0.50, 0.068, 0.19, "", fc=PALE, ec=DARK)
    ax.text(0.961, 0.595, "$\\Delta E$\n$=G_c$\n$-G_{j95}$", ha="center",
            va="center", fontsize=7.4, color=DARK)

    # ---- bottom provenance / guard notes ----
    box(ax, 0.015, 0.05, 0.97, 0.24, "", fc="none", ec=RED, lw=1.0)
    ax.text(0.032, 0.225, "Common-image bootstrap: both precisions are "
            "resampled on identical image samples", ha="left", va="center",
            fontsize=7.8, color=RED)
    ax.text(0.032, 0.115, "Accuracy-floor guard: $D$ and corrupted AP "
            "support interpretation; they do not transform or censor "
            "$\\Delta E$", ha="left", va="center", fontsize=7.8, color=RED)

    fig.subplots_adjust(left=0, right=1, top=1, bottom=0)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_framework.pdf")
    fig.savefig(OUT / "fig_framework.png", dpi=300)
    plt.close(fig)
    print("wrote", OUT / "fig_framework.pdf")


if __name__ == "__main__":
    main()
