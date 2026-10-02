#!/usr/bin/env python3
"""Draw fig_framework.pdf: the paired clean-control / corruption protocol.

Left to right: one source image yields a JPEG-95 clean control and a
corrupted input (also re-encoded at JPEG 95); both reach the INT8 and FP8
engines; the four AP cells form the 2x2 table from which Delta E is read.
Vector output (PDF) plus a 300-dpi PNG preview."""
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch, Rectangle

OUT = Path(__file__).resolve().parent.parent / "paper" / "figures"

INK = "#1F2937"
MUTED = "#64748B"
INT8_C = "#C2410C"
FP8_C = "#1D4ED8"
PALE = "#F1F5F9"
plt.rcParams.update({"font.family": "DejaVu Sans", "pdf.fonttype": 42})


def box(ax, x, y, w, h, text="", fc="white", ec=INK, lw=1.0, fs=8.5,
        color=INK, weight="normal"):
    ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.004,rounding_size=0.012",
                                facecolor=fc, edgecolor=ec, linewidth=lw))
    if text:
        ax.text(x + w / 2, y + h / 2, text, ha="center", va="center", fontsize=fs,
                color=color, fontweight=weight, linespacing=1.25)


def arrow(ax, a, b, color=INK, rad=0.0, lw=1.0):
    ax.add_patch(FancyArrowPatch(a, b, arrowstyle="-|>", mutation_scale=9, linewidth=lw,
                                 color=color, connectionstyle=f"arc3,rad={rad}",
                                 shrinkA=0, shrinkB=0))


def main() -> None:
    fig, ax = plt.subplots(figsize=(7.0, 2.55))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")

    # 1. inputs
    box(ax, 0.005, 0.47, 0.095, 0.20, "Source\nimage", fc=PALE, fs=8.5)
    box(ax, 0.150, 0.66, 0.165, 0.20, "Clean control\nJPEG q95", fs=8.2)
    box(ax, 0.150, 0.28, 0.165, 0.20, "Corrupted input\n+ JPEG q95", fs=8.2)
    arrow(ax, (0.100, 0.61), (0.150, 0.74), rad=-0.15)
    arrow(ax, (0.100, 0.53), (0.150, 0.40), rad=0.15)
    ax.text(0.355, 0.93, "identical bytes reach both engines", ha="center", fontsize=7.2,
            color=MUTED, style="italic")

    # 2. engines: each input reaches both engines
    ex, ew, eh = 0.395, 0.125, 0.17
    box(ax, ex, 0.62, ew, eh, "INT8 engine", ec=INT8_C, color=INT8_C, fs=8.2, weight="bold")
    box(ax, ex, 0.30, ew, eh, "FP8 engine", ec=FP8_C, color=FP8_C, fs=8.2, weight="bold")
    for ya in (0.76, 0.38):
        arrow(ax, (0.315, ya), (ex, 0.705))
        arrow(ax, (0.315, ya), (ex, 0.385))

    # 3. 2x2 AP table
    gx, gy, cw, ch = 0.605, 0.25, 0.130, 0.22
    ax.text(gx + cw / 2, gy + 2 * ch + 0.045, "clean (J95)", ha="center", fontsize=8.0, color=INK)
    ax.text(gx + 1.5 * cw, gy + 2 * ch + 0.045, "corrupted ($c$)", ha="center", fontsize=8.0,
            color=INK)
    labels = [[("AP$_{\\mathrm{J95}}^{\\mathrm{INT8}}$", INT8_C), ("AP$_{c}^{\\mathrm{INT8}}$", INT8_C)],
              [("AP$_{\\mathrm{J95}}^{\\mathrm{FP8}}$", FP8_C), ("AP$_{c}^{\\mathrm{FP8}}$", FP8_C)]]
    for r in range(2):
        for c in range(2):
            x, y = gx + c * cw, gy + (1 - r) * ch
            ax.add_patch(Rectangle((x, y), cw, ch, facecolor="white", edgecolor=INK, linewidth=0.9))
            lab, col = labels[r][c]
            ax.text(x + cw / 2, y + ch / 2, lab, ha="center", va="center", fontsize=9.0, color=col)
    arrow(ax, (ex + ew, 0.705), (gx, gy + 1.5 * ch), color=MUTED, lw=0.9)
    arrow(ax, (ex + ew, 0.385), (gx, gy + 0.5 * ch), color=MUTED, lw=0.9)
    ax.text(gx + cw / 2, gy - 0.075, "$G_0$: FP8 $-$ INT8", ha="center", fontsize=7.6, color=INK)
    ax.text(gx + 1.5 * cw, gy - 0.075, "$G_c$: FP8 $-$ INT8", ha="center", fontsize=7.6, color=INK)

    # 4. estimand
    arrow(ax, (gx + 2 * cw + 0.005, gy + ch), (0.893, gy + ch))
    box(ax, 0.896, gy + ch - 0.12, 0.099, 0.24, "$\\Delta E$\n$= G_c - G_0$", fc=PALE, fs=8.6)

    # 5. notes
    ax.text(0.005, 0.015, "All arms share one common-image bootstrap schedule ($B$ = 2,000).   "
            "Every $\\Delta E$ is reported with absolute corrupted AP (floor guard).",
            ha="left", va="bottom", fontsize=7.4, color=MUTED)

    fig.subplots_adjust(left=0.005, right=0.995, top=0.995, bottom=0.005)
    OUT.mkdir(parents=True, exist_ok=True)
    fig.savefig(OUT / "fig_framework.pdf")
    fig.savefig(OUT / "fig_framework.png", dpi=300)
    plt.close(fig)
    print("wrote", OUT / "fig_framework.pdf")


if __name__ == "__main__":
    main()
