"""Render publication diagnostics from unrounded, retained summary ledgers."""
import argparse
import hashlib
import json
from pathlib import Path

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.patches import FancyBboxPatch
import numpy as np
import pandas as pd


def holdout_points(record):
    p = float(record["delta_e_point"])
    q = [float(v) for v in record["delta_e_percentile95"]]
    if len(q) != 3 or not np.isfinite([p, *q]).all() or q != sorted(q):
        raise ValueError("invalid finite percentile record")
    return 100*p, 100*q[0], 100*q[-1]


def build(root, out):
    sources = [
        root/"paper/generated/direct_format_contrast_macro.csv",
        root/"paper/generated/direct_format_contrast_cells.csv",
        root/"paper/generated/direct_absolute_guardrail.csv",
        root/"paper/confirmatory_evidence/untouched_holdout_analysis.json",
    ]
    macro, cells, ap = [pd.read_csv(p) for p in sources[:3]]
    holdout = json.loads(sources[3].read_text())
    keys = ["dataset", "model", "corruption", "severity"]
    if len(cells) != 144 or cells.duplicated(keys).any():
        raise ValueError("expected 144 unique cells")
    lower = ap.pivot(index=keys, columns="precision", values="corrupt_ap_native")
    if len(lower) != 144 or lower[["int8-entropy", "fp8"]].isna().any().any():
        raise ValueError("missing paired AP arms")
    data = cells.merge(
        lower[["int8-entropy", "fp8"]].min(axis=1).rename("lower").reset_index(),
        on=keys, validate="one_to_one")
    plt.rcParams.update({"font.size": 9, "pdf.fonttype": 42, "ps.fonttype": 42})
    out.mkdir(parents=True, exist_ok=True)
    row = macro.loc[macro.endpoint == "four_dataset_macro_delta_e"].iloc[0]
    records = [
        ("Exploratory: four datasets", 100*row.point, 100*row.ci_low, 100*row.ci_high),
        ("Historical VOC/KITTI", 100*cells.loc[cells.dataset.isin(["voc", "kitti"]), "delta_e_all"].mean(), None, None),
        ("Final VOC/KITTI", *holdout_points(holdout["overall_balanced_equal_cell"])),
        ("Final VOC", *holdout_points(holdout["by_dataset"]["voc"])),
        ("Final KITTI", *holdout_points(holdout["by_dataset"]["kitti"])),
    ]
    fig, ax = plt.subplots(figsize=(4.6, 3.1), layout="constrained")
    for i, (label, point, low, high) in enumerate(records):
        color = "#126b79" if i >= 2 else "#6b7280"
        ax.plot(point, i, "o" if low is not None else "D", color=color)
        if low is not None:
            ax.plot([low, high], [i, i], color=color, lw=2)
        ax.annotate(f"{point:+.2f}", (point, i), xytext=(0, 8), textcoords="offset points",
                    ha="center", fontsize=8)
    ax.set_yticks(range(len(records)), [r[0] for r in records])
    ax.invert_yaxis()
    ax.axvline(0, color="0.55", ls="--", lw=.7)
    ax.set_xlabel(r"Interaction $\Delta E$ (AP points)")
    ax.set_xlim(-1.3, .3)
    ax.set_ylim(4.6, -.6)
    ax.spines[["top", "right"]].set_visible(False)
    fig.savefig(out/"fig_holdout_primary.pdf")
    plt.close(fig)
    fig, axes = plt.subplots(1, 2, figsize=(8.3, 3.45), layout="constrained")
    for ds, color, mark in zip(["coco", "voc", "kitti", "tt100k"],
                               ["#0072b2", "#009e73", "#d55e00", "#8b5ca3"],
                               ["o", "s", "^", "D"]):
        part = data.loc[data.dataset == ds]
        y = 100*part.delta_e_all
        axes[0].scatter(100*(part.delta_q_all+part.delta_e_all), y,
                        s=23, alpha=.75, color=color, marker=mark, label=ds.upper())
        axes[1].scatter(100*part.lower, y, s=23, alpha=.75, color=color, marker=mark)
    for ax in axes:
        ax.axhline(0, color=".4", lw=.7)
        ax.spines[["top", "right"]].set_visible(False)
        ax.grid(alpha=.12)
        ax.set_ylabel(r"$\Delta E$ (AP points)")
    axes[0].axvline(0, color=".4", lw=.7)
    axes[0].set_xlabel("Raw corrupted FP8 − INT8 gap (AP points)")
    axes[0].set_title("A  Raw gap versus clean-adjusted interaction", fontsize=9)
    axes[0].legend(frameon=False, fontsize=8, ncol=2, loc="upper left")
    axes[1].axvspan(0, 10, color="#eab676", alpha=.15)
    axes[1].axvline(5, color=".5", ls="--", lw=.8)
    axes[1].axvline(10, color=".5", ls=":", lw=.8)
    axes[1].set_title("B  Descriptive low-AP region (not a safety threshold)", fontsize=9)
    axes[1].set_xlabel("Lower corrupted AP of the two treatments")
    fig.savefig(out/"fig_decision_impact.pdf")
    plt.close(fig)
    fig, ax = plt.subplots(figsize=(8.3, 3.0), layout="constrained")
    ax.set_xlim(0, 12)
    ax.set_ylim(0, 4)
    ax.axis("off")

    def box(x, y, w, h, label, color="#eaf3f5"):
        ax.add_patch(FancyBboxPatch((x, y), w, h, boxstyle="round,pad=0.05",
                                   facecolor=color, edgecolor="#37616b", linewidth=.8))
        ax.text(x+w/2, y+h/2, label, ha="center", va="center", fontsize=9)

    def arrow(a, b):
        ax.annotate("", xy=b, xytext=a,
                    arrowprops={"arrowstyle": "->", "color": "#37616b", "lw": 1})

    box(.1, 1.6, 1.5, .7, "Source images\n+ annotations", "#f4f4f4")
    box(2.1, 2.6, 2.1, .7, "No named corruption\n→ terminal JPEG-95")
    box(2.1, 1.2, 2.1, .7, "Named corruption\n→ terminal JPEG-95")
    arrow((1.65, 2.1), (2.04, 2.9))
    arrow((1.65, 1.8), (2.04, 1.55))
    for y, condition in [(2.6, "clean"), (1.2, "corrupt")]:
        box(5.0, y, 2.15, .7, f"INT8 → AP ({condition})")
        box(7.75, y, 2.15, .7, f"FP8 → AP ({condition})")
        arrow((4.25, y+.35), (4.94, y+.35))
        ax.plot([4.55, 4.55, 8.82, 8.82],
                [y+.35, y+.91, y+.91, y+.76], color="#37616b", lw=.8)
        ax.text(6.9, y+.99, "Identical encoded bytes across formats",
                ha="center", fontsize=8, color="#126b79")
    box(10.35, 1.7, 1.5, 1.0, "Common image\ndraws across\nall four APs", "#f4f4f4")
    arrow((9.98, 2.95), (10.3, 2.4))
    arrow((9.98, 1.55), (10.3, 2.0))
    ax.text(6, .64, r"$\Delta E=(AP_{FP8}-AP_{INT8})_{\mathrm{corrupt}}"
            r"-(AP_{FP8}-AP_{INT8})_{\mathrm{clean}}$",
            ha="center", fontsize=11)
    ax.text(6, .23, "Report clean fidelity → absolute corrupted AP → interaction and scale → engine runtime",
            ha="center", fontsize=9, color="#126b79")
    fig.savefig(out/"fig_framework.pdf")
    plt.close(fig)
    return {str(p.relative_to(root)): hashlib.sha256(p.read_bytes()).hexdigest() for p in sources}


if __name__ == "__main__":
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path("."))
    p.add_argument("--output", type=Path, required=True)
    a = p.parse_args()
    print(json.dumps(build(a.root, a.output), indent=2))
