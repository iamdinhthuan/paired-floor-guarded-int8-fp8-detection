#!/usr/bin/env python3
"""Generate Neural-Networks manuscript tables + figures from frozen
Phase B / Phase C-D artifacts. Deterministic; writes paper/generated/nn_*.tex
and paper/figures/nn_*.pdf.
"""
from __future__ import annotations

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np

ROOT = Path(__file__).resolve().parent.parent
PHASE_B = ROOT / "submission_support_20260911" / "phase_b_results"
PHASE_CD = ROOT / "submission_support_20260911" / "phase_cd_results"
GEN = ROOT / "paper" / "generated"
FIG = ROOT / "paper" / "figures"


def load_contrasts(path: Path) -> dict:
    return json.loads(path.read_text())["contrasts"]


def ap_points(v):
    return v * 100.0


def write(path: Path, text: str) -> None:
    path.write_text(text, encoding="utf-8")
    print(f"wrote {path.name}")


def per_family_table(final: dict) -> str:
    """Supplement: per-corruption-family delta E and motion-blur-s5 LOO, YOLO blocks."""
    order = [("kitti", "KITTI"), ("voc", "VOC"), ("coco", "COCO-2k")]
    fams = [("fog", "Fog"), ("gaussian_noise", "Gaussian noise"), ("jpeg", "JPEG"),
            ("motion_blur", "Motion blur")]
    rows = []
    for ds, dsname in order:
        for model in ("yolo11n", "yolo11m", "yolo11x"):
            blk = next(b for b in final["yolo"]["blocks"]
                       if b["dataset"] == ds and b["model"] == model)
            cells = []
            for fam, _ in fams:
                d = blk["per_family_deltaE"][fam]
                sig = r"\sigstar" if d["ci95"][0] > 0 or d["ci95"][1] < 0 else ""
                cells.append(f"${d['point']:+.2f}${sig}")
            d = blk["deltaE_ex_mb5"]
            sig = r"\sigstar" if d["ci95"][0] > 0 or d["ci95"][1] < 0 else ""
            cells.append(f"${d['point']:+.2f}${sig} $[{d['ci95'][0]:+.2f},\\,{d['ci95'][1]:+.2f}]$")
            rows.append(f"{dsname} & {model.replace('yolo11','YOLO11')} & "
                        + " & ".join(cells) + r" \\")
    return ("\n".join([
        r"\begin{tabular}{lllllll}",
        r"\toprule",
        r"Dataset & Model & Fog & Gaussian noise & JPEG & Motion blur & "
        r"$\Delta E$ excl.\ mb-s5 \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}"]) + "\n")


# ---------------- YOLO deltaE table ----------------

def yolo_deltae_table(summary: dict, boot: dict, final: dict) -> str:
    rows = []
    order = [("kitti", "KITTI"), ("voc", "VOC"), ("coco", "COCO-2k")]
    holm = {(b["dataset"], b["model"]): b["p_holm"] for b in final["yolo"]["blocks"]}
    for ds, dsname in order:
        for model in ("yolo11n", "yolo11m", "yolo11x"):
            blk = next(b for b in summary["blocks"]
                       if b["dataset"] == ds and b["model"] == model)
            c = blk["contrasts"]["fp8_minus_int8"]
            bc = boot[f"{ds}__{model}"]["fp8_minus_int8"]
            de, ci = c["deltaE"], [v * 100 for v in bc["deltaE"]]
            sig = r"\sigstar" if ci[0] > 0 or ci[2] < 0 else ""
            ph = holm[(ds, model)]
            # smallest nonzero two-sided bootstrap p at B=2000 is 0.001;
            # an exact zero is bounded by the resolution, and the Holm
            # multiplier on the smallest of nine raw p's is 9 -> bound <0.009
            phs = "$<$0.009" if ph == 0 else f"{ph:.3f}"
            fp8c = blk["arms"]["fp8"]["corrupted_mean_ap"]
            i8c = blk["arms"]["int8"]["corrupted_mean_ap"]
            rows.append(
                f"{dsname} & {model.replace('yolo11','YOLO11')} & "
                f"{fp8c:.1f} & {i8c:.1f} & "
                f"{c['G0_q95']:+.2f} & {c['Gc']:+.2f} & "
                f"{de:+.2f}{sig} & $[{ci[0]:+.2f},\\,{ci[2]:+.2f}]$ & {phs} \\\\")
    return ("\n".join([
        r"\begin{tabular}{llrrrrllr}",
        r"\toprule",
        r"Dataset & Model & FP8 corr.\ AP & INT8 corr.\ AP & $G_0$ & $G_c$ & "
        r"$\Delta E$ & 95\% interval & $p_{\mathrm{Holm}}$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}"]) + "\n")


# ---------------- RetinaNet arm table ----------------

def load_points() -> dict:
    """Merge point estimates for intervention/control arms."""
    merged = {}
    for name in ("corruptcalib_points.json", "q95calib_points.json",
                 "cc2calib_points.json"):
        path = PHASE_CD / name
        if not path.is_file():
            continue
        data = json.loads(path.read_text())
        for ds, arms in data.items():
            merged.setdefault(ds, {}).update(arms)
    return merged


def retinanet_arms_table(summary: dict, report: dict) -> str:
    points = load_points()
    bpoints = json.loads((PHASE_B / "cell_points.json").read_text())
    corr_conds = [f"{c}-s{s}" for c in
                  ("fog", "gaussian_noise", "jpeg", "motion_blur")
                  for s in (1, 3, 5)]
    rows = []
    arm_names = [
        ("fp32", "FP32 reference"),
        ("fp8-matched512", "FP8-matched"),
        ("int8-legacy", "INT8-legacy (contract mismatch)"),
        ("int8-matched512", "INT8-matched"),
        ("int8-q95calib512", "INT8-q95calib (codec control)"),
        ("int8-selective512", "INT8-selective (reg-head FP32)"),
        ("int8-sel-q95calib512", "INT8-sel-q95calib"),
        ("int8-corruptcalib512", "INT8-corruptcalib"),
        ("int8-sel-corruptcalib512", "INT8-sel-corruptcalib"),
        ("int8-cc2calib512", "INT8-cc2calib (fold 2)"),
        ("int8-sel-cc2calib512", "INT8-sel-cc2calib (fold 2)"),
    ]
    for ds, dsname in (("kitti", "KITTI"), ("voc", "VOC")):
        blk = next(b for b in summary["blocks"]
                   if b["dataset"] == ds and b["model"] == "retinanet_r50_fpn_v2")
        first = True
        for key, label in arm_names:
            pa_b = bpoints.get(f"{ds}/retinanet_r50_fpn_v2", {}).get(key)
            if pa_b is not None:
                # Full-precision plug-in values from the per-cell metric records.
                clean = pa_b["clean-s0"]
                q95 = pa_b.get("codec-control-s0", float("nan"))
                corr = float(np.mean([pa_b[c] for c in corr_conds]))
            elif key in blk["arms"]:
                a = blk["arms"][key]
                clean = a["clean_orig_ap"]
                q95 = a.get("clean_q95_ap") or float("nan")
                corr = a["corrupted_mean_ap"]
            else:
                pa = points[ds].get(key)
                if pa is None:
                    continue
                clean = pa["clean"]
                q95 = pa["q95"]
                corr = pa["corr12"]
            ds_cell = dsname if first else ""
            first = False
            q95s = f"{q95:.2f}" if q95 == q95 else "--"
            rows.append(f"{ds_cell} & {label} & {clean:.2f} & {q95s} & {corr:.2f} \\\\")
        rows.append(r"\midrule")
    rows.pop()  # trailing midrule
    return ("\n".join([
        r"\begin{tabular}{llrrr}",
        r"\toprule",
        r"Dataset & Arm & Clean AP & $\Jclean$ AP & Corrupt mean AP \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}"]) + "\n")


# ---------------- Intervention table ----------------

INTERVENTION_ROWS = [
    ("int8-corruptcalib512_minus_int8-matched512", "INT8-corruptcalib $-$ INT8-matched"),
    ("int8-q95calib512_minus_int8-matched512", "INT8-q95calib $-$ INT8-matched"),
    ("int8-corruptcalib512_minus_int8-q95calib512", "INT8-corruptcalib $-$ INT8-q95calib"),
    ("int8-sel-corruptcalib512_minus_int8-selective512", "INT8-sel-corruptcalib $-$ INT8-selective"),
    ("int8-sel-q95calib512_minus_int8-selective512", "INT8-sel-q95calib $-$ INT8-selective"),
    ("int8-sel-corruptcalib512_minus_int8-sel-q95calib512",
     "INT8-sel-corruptcalib $-$ INT8-sel-q95calib"),
    ("int8-sel-corruptcalib512_minus_fp8-matched512", "INT8-sel-corruptcalib $-$ FP8-matched"),
    ("int8-corruptcalib512_minus_fp8-matched512", "INT8-corruptcalib $-$ FP8-matched"),
    ("int8-cc2calib512_minus_int8-matched512", "INT8-cc2calib $-$ INT8-matched (fold 2)"),
    ("int8-cc2calib512_minus_int8-q95calib512", "INT8-cc2calib $-$ INT8-q95calib (fold 2)"),
    ("int8-sel-cc2calib512_minus_int8-selective512",
     "INT8-sel-cc2calib $-$ INT8-selective (fold 2)"),
    ("int8-sel-cc2calib512_minus_int8-sel-q95calib512",
     "INT8-sel-cc2calib $-$ INT8-sel-q95calib (fold 2)"),
    ("int8-sel-cc2calib512_minus_fp8-matched512",
     "INT8-sel-cc2calib $-$ FP8-matched (fold 2)"),
    ("int8-cc2calib512_minus_fp8-matched512",
     "INT8-cc2calib $-$ FP8-matched (fold 2)"),
]


def fmt_ci(v: dict, stars: bool = True) -> str:
    lo, hi = v["ci95"]
    sig = r"\sigstar" if stars and (lo > 0 or hi < 0) else ""
    # a bound that rounds to +-0.00 inside a starred interval invites a
    # spurious "includes zero" reading -- show three decimals there
    if sig and (abs(lo) < 0.005 or abs(hi) < 0.005):
        return (f"${v['point']:+.2f}${sig} "
                f"$[{lo:+.3f},\\,{hi:+.3f}]$")
    return f"${v['point']:+.2f}${sig} $[{lo:+.2f},\\,{hi:+.2f}]$"


def intervention_table(final: dict) -> str:
    rows, emitted = [], 0
    for key, label in INTERVENTION_ROWS:
        group = []
        for ds, dsname in (("kitti", "KITTI"), ("voc", "VOC")):
            c = final["retinanet"][ds]["intervention"].get(key)
            if c is None:
                continue
            cells = [fmt_ci(c[m]) for m in ("j95", "corr12", "in_family", "held_out",
                                            "deltaE_held_out")]
            group.append(f"{label} & {dsname} & " + " & ".join(cells) + r" \\")
        if group:
            if emitted:
                rows.append(r"\addlinespace[3pt]")
            rows.extend(group)
            emitted += 1
    return ("\n".join([
        r"\begin{tabular}{lllllll}",
        r"\toprule",
        r"Contrast (A $-$ B) & Dataset & $\Jclean$ clean & Corrupt mean & In-family & "
        r"Held-out & $\Delta E_{\text{held-out}}$ \\",
        r"\midrule",
        *rows,
        r"\bottomrule",
        r"\end{tabular}"]) + "\n")


# ---------------- Phase A decomposition table ----------------

PHASE_A_DIR = ROOT / "submission_support_20260911" / "phase_a_arms"

# display label -> (summary file, treatment key) in the Phase-A ledgers
_DECOMPOSITION_ARMS = [
    ("INT8 legacy (mismatched calibration contract, 128-img)",
     "pilot_v1_summary.json", "int8_legacy_calibration"),
    ("INT8 matched preprocessing, 128-img calibration",
     "pilot_v1_summary.json", "int8_matched_calibration"),
    ("INT8 matched preprocessing, 512-img calibration",
     "recipe_v2_summary.json", "int8_matched_entropy_512cal"),
    ("INT8 max-estimator, 128-img calibration",
     "recipe_v1_summary.json", "int8_matched_max_calibration"),
    ("INT8 restricted to Conv+Add sites, 128-img calibration",
     "recipe_v1_summary.json", "int8_matched_convadd_calibration"),
    ("INT8 at exactly the FP8 compute sites (shared mask)",
     "recipe_v3_summary.json", "int8_matched_shared_mask_512cal"),
    ("INT8 backbone+FPN only (heads FP32)",
     "recipe_v4_summary.json", "int8_matched_backbone_only"),
    ("INT8 classification head only",
     "recipe_v5_summary.json", "int8_matched_cls_head_only"),
    ("INT8 regression head only",
     "recipe_v5_summary.json", "int8_matched_reg_head_only"),
    ("INT8 heads only (cls+reg)",
     "recipe_v4_summary.json", "int8_matched_heads_only"),
    ("INT8 all-except regression head",
     "recipe_v6_summary.json", "int8_matched_except_reg_head"),
    ("FP8 matched, 128-img calibration",
     "pilot_v1_summary.json", "fp8_matched_calibration"),
    ("FP32 reference (diagnostic build)",
     "pilot_v1_summary.json", "fp32"),
]


def _phase_a_ap(summary_file: str, treatment: str) -> float:
    s = json.loads((PHASE_A_DIR / summary_file).read_text())
    for row in s["rows"]:
        if row["treatment"] == treatment:
            return row["ap_points"]
    raise KeyError(f"{treatment} not in {summary_file}")


def decomposition_table() -> str:
    rows = [(label, f"{_phase_a_ap(sf, t):.2f}")
            for label, sf, t in _DECOMPOSITION_ARMS]
    return ("\n".join([
        r"\begin{tabular}{lr}",
        r"\toprule",
        r"Recipe arm (KITTI diagnostic, single-point) & $\Jclean$ AP \\",
        r"\midrule",
        *[f"{name} & {val} \\\\" for name, val in rows],
        r"\bottomrule",
        r"\end{tabular}"]) + "\n")


# ---------------- Figures ----------------

def deltae_forest(summary: dict, boot: dict, final: dict) -> None:
    entries = []
    labels = []
    for ds, dsname in (("kitti", "KITTI"), ("voc", "VOC"), ("coco", "COCO-2k")):
        for model in ("yolo11n", "yolo11m", "yolo11x"):
            blk = next(b for b in summary["blocks"]
                       if b["dataset"] == ds and b["model"] == model)
            c = blk["contrasts"]["fp8_minus_int8"]
            ci = boot[f"{ds}__{model}"]["fp8_minus_int8"]["deltaE"]
            entries.append((c["deltaE"], ci[0] * 100.0, ci[2] * 100.0))
            labels.append(f"{dsname} {model.replace('yolo11','')}")
    holm = {(b["dataset"], b["model"]): b["p_holm"] < 0.05
            for b in final["yolo"]["blocks"]}
    fig, ax = plt.subplots(figsize=(6.2, 4.2))
    ys = np.arange(len(entries))[::-1]
    palette = {"KITTI": "#1f77b4", "VOC": "#2ca02c", "COCO-2k": "#9467bd"}
    for y, (pt, lo, hi), lab in zip(ys, entries, labels):
        color = palette[lab.split()[0]]
        sig = lo > 0 or hi < 0
        ds_key = {"KITTI": "kitti", "VOC": "voc", "COCO-2k": "coco"}[lab.split()[0]]
        mo_key = "yolo11" + lab.split()[1]
        ax.plot([lo, hi], [y, y], color=color, lw=2.2,
                solid_capstyle="round", alpha=0.95 if sig else 0.45)
        ax.plot([pt], [y], "o", color=color, ms=5.5,
                alpha=0.95 if sig else 0.45)
        if holm.get((ds_key, mo_key)):
            ax.plot([pt], [y], "D", mfc="none", mec="k", ms=9, mew=0.9)
    ax.axvline(0, color="k", lw=0.8, ls="--")
    ax.set_yticks(ys)
    ax.set_yticklabels(labels, fontsize=8)
    ax.set_xlabel(r"$\Delta E$ (AP points), FP8$-$INT8")
    ax.set_title("Corruption interaction by block", fontsize=9)
    ax.grid(axis="x", alpha=0.25)
    fig.tight_layout()
    fig.savefig(FIG / "nn_deltae_forest.pdf")
    plt.close(fig)
    print("wrote nn_deltae_forest.pdf")


def retinanet_figure(summary: dict, report: dict) -> None:
    points = load_points()
    fig, axes = plt.subplots(1, 2, figsize=(9.2, 3.4), sharey=False)
    arms = [("fp32", "FP32", "#7f7f7f"),
            ("fp8-matched512", "FP8", "#1f77b4"),
            ("int8-matched512", "INT8-matched", "#d62728"),
            ("int8-q95calib512", "INT8-q95", "#8c564b"),
            ("int8-selective512", "INT8-sel", "#2ca02c"),
            ("int8-sel-q95calib512", "INT8-sel-q95", "#bcbd22"),
            ("int8-corruptcalib512", "INT8-cc", "#ff7f0e"),
            ("int8-sel-corruptcalib512", "INT8-sel-cc", "#9467bd"),
            ("int8-cc2calib512", "INT8-cc2", "#17becf"),
            ("int8-sel-cc2calib512", "INT8-sel-cc2", "#e377c2")]
    for ax, ds, dsname in zip(axes, ("kitti", "voc"), ("KITTI", "VOC")):
        blk = next(b for b in summary["blocks"]
                   if b["dataset"] == ds and b["model"] == "retinanet_r50_fpn_v2")
        clean, corr, labels, colors = [], [], [], []
        for key, label, color in arms:
            if key in blk["arms"]:
                clean.append(blk["arms"][key]["clean_orig_ap"])
                corr.append(blk["arms"][key]["corrupted_mean_ap"])
            elif ds in points and key in points[ds]:
                clean.append(points[ds][key]["clean"])
                corr.append(points[ds][key]["corr12"])
            else:
                continue
            labels.append(label)
            colors.append(color)
        x = np.arange(len(labels))
        ax.bar(x - 0.19, clean, 0.36, color=colors, alpha=0.95)
        ax.bar(x + 0.19, corr, 0.36,
               color=colors, alpha=0.45, hatch="//", edgecolor="white")
        ax.set_xticks(x)
        ax.set_xticklabels(labels, fontsize=7, rotation=28, ha="right")
        ax.set_title(dsname, fontsize=9)
        ax.set_ylabel("AP (points)" if ds == "kitti" else "")
        ax.grid(axis="y", alpha=0.25)
    handles = [plt.Rectangle((0, 0), 1, 1, facecolor="k", alpha=0.95),
               plt.Rectangle((0, 0), 1, 1, facecolor="k", alpha=0.45,
                             hatch="//", edgecolor="white")]
    fig.legend(handles, ["clean", "corrupt mean"], fontsize=8,
               frameon=False, ncol=2, loc="upper center",
               bbox_to_anchor=(0.5, 0.97))
    fig.suptitle("RetinaNet recipe arms: clean vs 12-cell corruption mean", fontsize=9, y=1.04)
    fig.tight_layout(rect=(0, 0, 1, 0.93))
    fig.savefig(FIG / "nn_retinanet_arms.pdf")
    plt.close(fig)
    print("wrote nn_retinanet_arms.pdf")


def intervention_figure(final: dict) -> None:
    specs = [
        ("int8-corruptcalib512_minus_int8-matched512", "full INT8\ncorrupt-calib"),
        ("int8-q95calib512_minus_int8-matched512", "full INT8\nq95-calib"),
        ("int8-sel-corruptcalib512_minus_int8-selective512", "selective\ncorrupt-calib"),
        ("int8-sel-q95calib512_minus_int8-selective512", "selective\nq95-calib"),
        ("int8-cc2calib512_minus_int8-matched512", "full INT8\ncc2-calib (f2)"),
        ("int8-sel-cc2calib512_minus_int8-selective512", "selective\ncc2-calib (f2)"),
    ]
    metrics = (("j95", r"$\mathrm{J95}$ clean", "#7f7f7f"),
               ("in_family", "in-family", "#1f77b4"),
               ("held_out", "held-out", "#d62728"))
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.0), sharey=True)
    for ax, (ds, dsname) in zip(axes, (("kitti", "KITTI"), ("voc", "VOC"))):
        con = final["retinanet"][ds]["intervention"]
        keep = [(k, lab) for k, lab in specs if k in con]
        x = np.arange(len(keep))
        width = 0.26
        for j, (m, mname, col) in enumerate(metrics):
            vals = [con[k][m] for k, _ in keep]
            pts = [v["point"] for v in vals]
            err = [[v["point"] - v["ci95"][0] for v in vals],
                   [v["ci95"][1] - v["point"] for v in vals]]
            ax.bar(x + (j - 1) * width, pts, width, yerr=err, capsize=2, color=col,
                   alpha=0.85, label=mname)
        ax.axhline(0, color="k", lw=0.8)
        ax.set_xticks(x)
        ax.set_xticklabels([lab for _, lab in keep], fontsize=6.5,
                           rotation=30, ha="right")
        ax.set_title(dsname, fontsize=9)
        ax.grid(axis="y", alpha=0.25)
    axes[0].set_ylabel(r"$\Delta$AP vs clean-calibrated counterpart")
    axes[0].legend(fontsize=7, frameon=False)
    fig.tight_layout()
    fig.savefig(FIG / "nn_intervention.pdf")
    plt.close(fig)
    print("wrote nn_intervention.pdf")


# ---------------- in-text number macros ----------------

def _ci(v: dict) -> str:
    lo, hi = v["ci95"]
    # a bound that rounds to +-0.00 invites a spurious "includes zero"
    # reading -- show three decimals there, as in fmt_ci
    if abs(lo) < 0.005 or abs(hi) < 0.005:
        return f"${v['point']:+.2f}$~$[{lo:+.3f},{hi:+.3f}]$"
    return f"${v['point']:+.2f}$~$[{lo:+.2f},{hi:+.2f}]$"


def sci(p: float) -> str:
    mant, exp = f"{p:.0e}".split("e")
    return f"{mant}\\times10^{{{int(exp)}}}"


# Diagnostic decomposition arm -> (summary file, treatment) for in-text macros
_DIAG_MACROS = {
    "DiagLegacyJ": ("pilot_v1_summary.json", "int8_legacy_calibration"),
    "DiagMatchedJ": ("pilot_v1_summary.json", "int8_matched_calibration"),
    "DiagMatchedFiveJ": ("recipe_v2_summary.json", "int8_matched_entropy_512cal"),
    "DiagMaxJ": ("recipe_v1_summary.json", "int8_matched_max_calibration"),
    "DiagConvAddJ": ("recipe_v1_summary.json", "int8_matched_convadd_calibration"),
    "DiagSharedMaskJ": ("recipe_v3_summary.json", "int8_matched_shared_mask_512cal"),
    "DiagBackboneJ": ("recipe_v4_summary.json", "int8_matched_backbone_only"),
    "DiagClsJ": ("recipe_v5_summary.json", "int8_matched_cls_head_only"),
    "DiagRegJ": ("recipe_v5_summary.json", "int8_matched_reg_head_only"),
    "DiagHeadsJ": ("recipe_v4_summary.json", "int8_matched_heads_only"),
    "DiagNoRegJ": ("recipe_v6_summary.json", "int8_matched_except_reg_head"),
    "DiagFpEightJ": ("pilot_v1_summary.json", "fp8_matched_calibration"),
    "DiagFpThirtyTwoJ": ("pilot_v1_summary.json", "fp32"),
}


def numbers_tex(final: dict) -> str:
    """Emit \\newcommand macros so every in-text number traces to nn_final_stats.json."""
    macros = {}
    for name, (sf, t) in _DIAG_MACROS.items():
        macros[name] = f"{_phase_a_ap(sf, t):.2f}"
    macros["DiagPreprocGain"] = f"{_phase_a_ap('pilot_v1_summary.json', 'int8_matched_calibration') - _phase_a_ap('pilot_v1_summary.json', 'int8_legacy_calibration'):.1f}"
    macros["DiagFpEightDeficitJ"] = f"{_phase_a_ap('pilot_v1_summary.json', 'fp8_matched_calibration') - _phase_a_ap('pilot_v1_summary.json', 'int8_legacy_calibration'):.1f}"
    bpts = json.loads((PHASE_B / "cell_points.json").read_text())
    for ds, D in (("kitti", "Kitti"), ("voc", "Voc")):
        cells = bpts[f"{ds}/retinanet_r50_fpn_v2"]
        for arm, A in (("fp32", "Fp"), ("fp8-matched512", "Fe"),
                       ("fp8-legacy", "FeLegacy"), ("int8-legacy", "Legacy"),
                       ("int8-matched512", "In"), ("int8-selective512", "Sel")):
            if arm in cells:
                macros[f"Arm{A}Clean{D}"] = f"{cells[arm]['clean-s0']:.2f}"
                if "codec-control-s0" in cells[arm]:
                    macros[f"Arm{A}J{D}"] = f"{cells[arm]['codec-control-s0']:.2f}"
        macros[f"HistDeficit{D}"] = f"{cells['fp8-matched512']['clean-s0'] - cells['int8-legacy']['clean-s0']:.1f}"
        macros[f"MatchedCleanDeficit{D}"] = f"{cells['fp8-matched512']['clean-s0'] - cells['int8-matched512']['clean-s0']:.1f}"
    y = final["yolo"]
    het = y["heterogeneity"]
    macros["HetQ"] = f"{het['Q']:.1f}"
    macros["HetDf"] = f"{het['df']}"
    macros["HetP"] = sci(het["p"])
    macros["HetIsq"] = f"{100 * het['I2']:.0f}"
    macros["HetTau"] = f"{het['tau']:.2f}"
    ex = het["ex_mb5"]
    macros["HetQEx"] = f"{ex['Q']:.1f}"
    macros["HetIsqEx"] = f"{100 * ex['I2']:.0f}"
    for ds, D in (("kitti", "Kitti"), ("voc", "Voc"), ("coco", "Coco")):
        w = het["within_dataset"][ds]
        macros[f"HetQ{D}"] = f"{w['Q']:.1f}"
        macros[f"HetP{D}"] = f"{w['p']:.2f}" if w["p"] >= 0.01 else sci(w["p"])
    macros["HolmSurvivors"] = str(sum(b["p_holm"] < 0.05 for b in y["blocks"]))
    _tau = het["tau"]
    _w = [1.0 / (b["se"] ** 2 + _tau ** 2) for b in y["blocks"]]
    _mu = sum(wi * b["point"] for wi, b in zip(_w, y["blocks"])) / sum(_w)
    _se_mu = (1.0 / sum(_w)) ** 0.5
    macros["ReMean"] = f"{_mu:+.2f}"
    macros["ReMeanCi"] = f"$[{_mu - 1.96 * _se_mu:+.2f},\\,{_mu + 1.96 * _se_mu:+.2f}]$"
    pw = y["pairwise"]["voc_n_minus_m"]
    macros["VocNmDiff"] = f"{pw['diff']:+.2f}"
    macros["VocNmZ"] = f"{pw['z']:.1f}"
    _SFX = {"yolo11n": "N", "yolo11m": "M", "yolo11x": "X"}
    g0, ret8, retf, fid = [], [], [], []
    for ds, D in (("kitti", "Kitti"), ("voc", "Voc"), ("coco", "Coco")):
        for mo in ("yolo11n", "yolo11m", "yolo11x"):
            cp = bpts[f"{ds}/{mo}"]
            g0.append(cp["fp8"]["codec-control-s0"] - cp["int8"]["codec-control-s0"])
            for a, lst in (("int8", ret8), ("fp8", retf)):
                co = [c for c in cp[a] if c not in ("clean-s0", "codec-control-s0")]
                lst.append(sum(cp[a][c] for c in co) / len(co) / cp[a]["codec-control-s0"])
                fid.append(abs(cp[a]["codec-control-s0"] - cp[a]["clean-s0"]))
    macros["GZeroMin"] = f"{min(g0):+.2f}"
    macros["GZeroMax"] = f"{max(g0):+.2f}"
    macros["RetMin"] = f"{min(min(ret8), min(retf)):.2f}"
    macros["RetMax"] = f"{max(max(ret8), max(retf)):.2f}"
    macros["RetDiff"] = f"{max(abs(a - b) for a, b in zip(ret8, retf)):.2f}"
    macros["JCleanFid"] = f"{max(fid):.1f}"
    for blk in y["blocks"]:
        K = f'{ {"kitti": "Kitti", "voc": "Voc", "coco": "Coco"}[blk["dataset"]] }{ _SFX[blk["model"]] }'.replace(" ", "")
        macros[f"DE{K}"] = f"{blk['point']:+.2f}"
        macros[f"DEci{K}"] = _ci(blk)
        macros[f"DEex{K}"] = f"{blk['deltaE_ex_mb5']['point']:+.2f}"
        if "per_family_deltaE" in blk:
            for fam, FN in (("fog", "Fog"), ("gaussian_noise", "Gn"),
                            ("jpeg", "Jpeg"), ("motion_blur", "Mb")):
                macros[f"Fam{K}{FN}"] = f"{blk['per_family_deltaE'][fam]['point']:+.2f}"
    import statistics as _st
    for fam, FN in (("fog", "Fog"), ("gaussian_noise", "Gn"),
                    ("jpeg", "Jpeg"), ("motion_blur", "Mb")):
        pts = [b["per_family_deltaE"][fam]["point"] for b in y["blocks"]
               if "per_family_deltaE" in b]
        macros[f"FamSDp{FN}"] = f"{_st.pstdev(pts):.1f}"
        macros[f"FamSDs{FN}"] = f"{_st.stdev(pts):.1f}"
    for ds, D in (("kitti", "Kitti"), ("voc", "Voc")):
        r = final["retinanet"][ds]
        pb = r["phase_b"]
        macros[f"MatchedGap{D}"] = f"{pb['fp8-matched512_minus_int8-matched512']['j95']['point']:.1f}"
        macros[f"MatchedGapJ{D}"] = _ci(pb["fp8-matched512_minus_int8-matched512"]["j95"])
        macros[f"MatchedDE{D}"] = _ci(pb["fp8-matched512_minus_int8-matched512"]["deltaE"])
        macros[f"MatchedDE{D}P"] = f"{pb['fp8-matched512_minus_int8-matched512']['deltaE']['point']:.1f}"
        for fam, FN in (("fog", "Fog"), ("gaussian_noise", "Gn"),
                        ("jpeg", "Jpeg"), ("motion_blur", "Mb")):
            macros[f"MatchedFam{FN}{D}"] = f"{pb['fp8-matched512_minus_int8-matched512']['per_family_deltaE'][fam]['point']:.1f}"
        cells = bpts[f"{ds}/retinanet_r50_fpn_v2"]
        macros[f"SelGainJ{D}"] = f"{cells['int8-selective512']['codec-control-s0'] - cells['int8-matched512']['codec-control-s0']:.1f}"
        co = [c for c in cells["fp8-matched512"] if c not in ("clean-s0", "codec-control-s0")]
        for a, A in (("int8-matched512", "In"), ("fp8-matched512", "Fe")):
            macros[f"Ret{D}{A}"] = f"{sum(cells[a][c] for c in co) / len(co) / cells[a]['codec-control-s0']:.2f}"
        s = pb["fp8-matched512_minus_int8-selective512"]
        macros[f"SelResJ{D}"] = _ci(s["j95"])
        macros[f"SelResJ{D}P"] = f"{s['j95']['point']:+.2f}"
        macros[f"SelResC{D}"] = _ci(s["corr12"])
        macros[f"SelResC{D}P"] = f"{s['corr12']['point']:+.2f}"
        macros[f"SelMinusFeC{D}"] = f"{-s['corr12']['point']:+.2f}"
        macros[f"SelResDE{D}"] = _ci(s["deltaE"])
        for fam, FN in (("fog", "Fog"), ("gaussian_noise", "Gn"),
                        ("jpeg", "Jpeg"), ("motion_blur", "Mb")):
            macros[f"SelFam{FN}{D}"] = _ci(s["per_family_deltaE"][fam])
        for arm, A in (("fp32", "Fp"), ("fp8-matched512", "Fe"),
                       ("int8-matched512", "In"), ("int8-selective512", "Sel")):
            macros[f"Wall{A}{D}"] = f"{r['mb5_wall'][arm]:.2f}"
        iv = r["intervention"]
        names = {"int8-corruptcalib512_minus_int8-matched512": "Cc",
                 "int8-q95calib512_minus_int8-matched512": "Qq",
                 "int8-corruptcalib512_minus_int8-q95calib512": "CcQ",
                 "int8-sel-corruptcalib512_minus_int8-selective512": "Scc",
                 "int8-sel-q95calib512_minus_int8-selective512": "Sqq",
                 "int8-sel-corruptcalib512_minus_fp8-matched512": "SccF",
                 "int8-corruptcalib512_minus_fp8-matched512": "CcF",
                 "int8-cc2calib512_minus_int8-matched512": "Ct",
                 "int8-cc2calib512_minus_int8-q95calib512": "CtQ",
                 "int8-sel-cc2calib512_minus_int8-selective512": "Sct",
                 "int8-sel-cc2calib512_minus_int8-sel-q95calib512": "SctQ",
                 "int8-sel-cc2calib512_minus_fp8-matched512": "SctF",
                 "int8-cc2calib512_minus_fp8-matched512": "CtF"}
        fams = {"j95": "J", "corr12": "C", "in_family": "In", "held_out": "H",
                "held_out_ex_mb5": "Hx", "deltaE_held_out": "DH", "deltaE_in_family": "DI"}
        for key, N in names.items():
            if key not in iv:
                for F in fams.values():
                    macros[f"{N}{F}{D}"] = macros[f"{N}{F}{D}P"] = r"\textbf{??}"
                continue
            for fam, F in fams.items():
                if fam not in iv[key]:
                    continue
                macros[f"{N}{F}{D}"] = _ci(iv[key][fam])
                macros[f"{N}{F}{D}P"] = f"{iv[key][fam]['point']:+.2f}"
    macros["NoRegFpGap"] = f"{_phase_a_ap('pilot_v1_summary.json', 'fp32') - _phase_a_ap('recipe_v6_summary.json', 'int8_matched_except_reg_head'):.2f}"
    _retd = []
    for _ph in ("phase_b_results", "phase_cd_results",
                "phase_f_fold2_results", "phase_e_q95_results"):
        _b = json.loads((PHASE_B.parent / _ph / "cell_points.json").read_text())
        for _k, _c in _b.items():
            if "retinanet" not in _k:
                continue
            for _v in _c.values():
                if "clean-s0" in _v and "codec-control-s0" in _v:
                    _retd.append(abs(_v["codec-control-s0"] - _v["clean-s0"]))
    macros["RetJCleanFid"] = f"{max(_retd):.1f}"
    lines = ["% Generated by analysis/build_nn_tables.py from nn_final_stats.json"]
    lines += [f"\\newcommand{{\\nn{k}}}{{{v}}}" for k, v in sorted(macros.items())]
    return "\n".join(lines) + "\n"


def main() -> None:
    summary = json.loads((PHASE_B / "summary.json").read_text())
    report = json.loads((PHASE_CD / "intervention_report.json").read_text())
    final = json.loads((ROOT / "submission_support_20260911" / "nn_final_stats.json").read_text())
    boot = {}
    for npz in (PHASE_B / "bootstrap").glob("*__bootstrap.json"):
        boot[npz.stem.replace("__bootstrap", "")] = load_contrasts(npz)
    GEN.mkdir(parents=True, exist_ok=True)
    FIG.mkdir(parents=True, exist_ok=True)
    write(GEN / "nn_yolo_deltae.tex", yolo_deltae_table(summary, boot, final))
    write(GEN / "nn_retinanet_arms.tex", retinanet_arms_table(summary, report))
    write(GEN / "nn_intervention.tex", intervention_table(final))
    write(GEN / "nn_decomposition.tex", decomposition_table())
    write(GEN / "nn_per_family.tex", per_family_table(final))
    write(GEN / "nn_numbers.tex", numbers_tex(final))
    deltae_forest(summary, boot, final)
    retinanet_figure(summary, report)
    intervention_figure(final)


if __name__ == "__main__":
    main()
