#!/usr/bin/env python3
"""Final manuscript statistics from plug-in points and paired bootstrap draws.

Writes ``submission_support_20260911/nn_final_stats.json`` with:
  * yolo: per-block FP8-INT8 Delta E (plug-in point, bootstrap CI/SE/p), Holm-
    adjusted p across the nine blocks, and a Cochran Q heterogeneity test;
  * retinanet: Phase-B selective/matched contrasts and all intervention and
    codec-control contrasts on J95-clean, corrupt-mean, in-family, held-out,
    held-out-excluding-motion-blur-s5, plus Delta E on the held-out family.
Point estimates are full-sample plug-in values; draws supply intervals only.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
from scipy import stats

from nn_contrasts import load_block

REPO = Path(__file__).resolve().parents[1]
SUPPORT = REPO / "submission_support_20260911"
PHASE_B = SUPPORT / "phase_b_results"
PHASE_CD = SUPPORT / "phase_cd_results"
PHASE_E = SUPPORT / "phase_e_q95_results"
PHASE_F = SUPPORT / "phase_f_fold2_results"
PHASE_G = SUPPORT / "phase_g_wa_results"
PHASE_H = SUPPORT / "phase_h_fcos_results"
RETINA = "retinanet_r50_fpn_v2"
FCOS = "fcos_r50_fpn"

INTERVENTION = [
    ("int8-corruptcalib512", "int8-matched512"),
    ("int8-q95calib512", "int8-matched512"),
    ("int8-corruptcalib512", "int8-q95calib512"),
    ("int8-sel-corruptcalib512", "int8-selective512"),
    ("int8-sel-q95calib512", "int8-selective512"),
    ("int8-sel-corruptcalib512", "int8-sel-q95calib512"),
    ("int8-sel-corruptcalib512", "fp8-matched512"),
    ("int8-corruptcalib512", "fp8-matched512"),
    ("fp8-matched512", "int8-selective512"),
    ("int8-cc2calib512", "int8-matched512"),
    ("int8-cc2calib512", "int8-q95calib512"),
    ("int8-sel-cc2calib512", "int8-selective512"),
    ("int8-sel-cc2calib512", "int8-sel-q95calib512"),
    ("int8-sel-cc2calib512", "fp8-matched512"),
    ("int8-cc2calib512", "fp8-matched512"),
]
FOLD2 = {"in_family": "held_out", "held_out": "in_family"}
FAMILIES = ("j95", "corr12", "in_family", "held_out", "held_out_ex_mb5")


def holm(p: list[float]) -> list[float]:
    order = np.argsort(p)
    adj, running = [0.0] * len(p), 0.0
    for k, i in enumerate(order):
        running = max(running, min(1.0, (len(p) - k) * p[i]))
        adj[i] = running
    return adj


def yolo_stats() -> dict:
    summary = {(b["dataset"], b["model"]): b
               for b in json.loads((PHASE_B / "summary.json").read_text())["blocks"]}
    rows = []
    for ds in ("kitti", "voc", "coco"):
        for model in ("yolo11n", "yolo11m", "yolo11x"):
            blk = load_block(PHASE_B, ds, model)
            d = blk.delta_e("fp8", "int8")
            # COCO cells are evaluated on the paired 2,000-image subset; the
            # subset plug-in point lives in summary.json (identical for
            # KITTI/VOC). The cell_points collector resolves COCO stems to the
            # subset metrics, so this is a consistency check, not a patch.
            head = summary[(ds, model)]["contrasts"]["fp8_minus_int8"]["deltaE"]
            if abs(d["point"] - head) > 0.05:
                raise SystemExit(
                    f"plug-in mismatch {ds}/{model}: cells {d['point']:.3f} "
                    f"vs summary {head:.3f}")
            d["point"] = head
            d["deltaE_ex_mb5"] = blk.delta_e("fp8", "int8", "corr11_ex_mb5")
            d["per_family_deltaE"] = {
                fam: blk.delta_e("fp8", "int8", f"fam_{fam}") for fam in
                ("fog", "gaussian_noise", "jpeg", "motion_blur")}
            fam_mean = float(np.mean([f["point"]
                                      for f in d["per_family_deltaE"].values()]))
            if abs(fam_mean - head) > 0.02:
                raise SystemExit(
                    f"family-mean inconsistency {ds}/{model}: {fam_mean:.3f} "
                    f"vs corr12 {head:.3f}")
            rows.append({"dataset": ds, "model": model, **d})
    for row, adj in zip(rows, holm([r["p"] for r in rows])):
        row["p_holm"] = adj
    x = np.array([r["point"] for r in rows])
    w = 1.0 / np.array([r["se"] for r in rows]) ** 2
    pooled = float((w * x).sum() / w.sum())
    q = float((w * (x - pooled) ** 2).sum())
    df = len(rows) - 1
    c = w.sum() - (w ** 2).sum() / w.sum()
    het = {"Q": q, "df": df, "p": float(stats.chi2.sf(q, df)),
           "I2": max(0.0, (q - df) / q), "tau": float(np.sqrt(max(0.0, (q - df) / c))),
           "pooled": pooled, "pooled_se": float(1.0 / np.sqrt(w.sum())),
           "note": "blocks treated as independent; within-dataset blocks share "
                   "images, so the independence assumption is an approximation "
                   "(the direction of any correlation bias is unverified)"}
    within = {}
    for ds in ("kitti", "voc", "coco"):
        sub = [r for r in rows if r["dataset"] == ds]
        xs = np.array([r["point"] for r in sub])
        ws = 1.0 / np.array([r["se"] for r in sub]) ** 2
        mu = (ws * xs).sum() / ws.sum()
        qs = float((ws * (xs - mu) ** 2).sum())
        within[ds] = {"Q": qs, "df": len(sub) - 1, "p": float(stats.chi2.sf(qs, len(sub) - 1))}
    het["within_dataset"] = within
    # Leave-one-out sensitivity: heterogeneity with the saturated
    # motion-blur-s5 cell excluded.
    x5 = np.array([r["deltaE_ex_mb5"]["point"] for r in rows])
    w5 = 1.0 / np.array([r["deltaE_ex_mb5"]["se"] for r in rows]) ** 2
    mu5 = float((w5 * x5).sum() / w5.sum())
    q5 = float((w5 * (x5 - mu5) ** 2).sum())
    het["ex_mb5"] = {"Q": q5, "df": df, "p": float(stats.chi2.sf(q5, df)),
                     "I2": max(0.0, (q5 - df) / q5)}
    pairs = {}
    for name, (i, j) in {"voc_n_minus_m": (3, 4), "coco_m_minus_x": (7, 8)}.items():
        a, b = rows[i], rows[j]
        z = (a["point"] - b["point"]) / np.hypot(a["se"], b["se"])
        pairs[name] = {"diff": a["point"] - b["point"], "z": float(z),
                       "p": float(2 * stats.norm.sf(abs(z)))}
    return {"blocks": rows, "heterogeneity": het, "pairwise": pairs}


def retina_stats() -> dict:
    out = {}
    for ds in ("kitti", "voc"):
        phase_b = load_block(PHASE_B, ds, RETINA)
        # Union of the C/D, E and F caches; shared arms must have identical draws.
        interv = load_block(PHASE_CD, ds, RETINA, (PHASE_E, PHASE_F))
        entry = {"intervention_source": "phase_cd+phase_e+phase_f (merged, paired)",
                 "phase_b": {}, "intervention": {},
                 "mb5_wall": {a: interv.p[a]["motion_blur-s5"]
                              for a in ("fp32", "fp8-matched512", "int8-matched512",
                                        "int8-selective512") if a in interv.p}}
        for a, b in (("fp8-matched512", "int8-matched512"),
                     ("fp8-matched512", "int8-selective512"),
                     ("int8-selective512", "int8-matched512")):
            entry["phase_b"][f"{a}_minus_{b}"] = {
                "j95": phase_b.diff(a, b, "j95"), "corr12": phase_b.diff(a, b, "corr12"),
                "deltaE": phase_b.delta_e(a, b),
                "per_family_deltaE": {
                    fam: phase_b.delta_e(a, b, f"fam_{fam}") for fam in
                    ("fog", "gaussian_noise", "jpeg", "motion_blur")}}
        for a, b in INTERVENTION:
            if not (interv.has(a, "j95") and interv.has(b, "j95")):
                continue
            # Fold-2 arms calibrate on fog+motion_blur, so their held-out family
            # is gaussian_noise+jpeg: swap the family labels for those rows.
            fam_of = (lambda f: FOLD2.get(f, f)) if "cc2" in a else (lambda f: f)
            res = {fam: interv.diff(a, b, fam_of(fam)) for fam in FAMILIES
                   if fam != "held_out_ex_mb5" or "cc2" not in a}
            res["deltaE_corr12"] = interv.delta_e(a, b, "corr12")
            res["deltaE_held_out"] = interv.delta_e(a, b, fam_of("held_out"))
            res["deltaE_in_family"] = interv.delta_e(a, b, fam_of("in_family"))
            entry["intervention"][f"{a}_minus_{b}"] = res
        out[ds] = entry
    return out


WA_CONTRASTS = [
    # Factorial decomposition of the regression-head localization: matched is
    # W8/A8, wonly is W8/Afp32 (activations restored), aonly is Wfp32/A8
    # (weights restored), selective is Wfp32/Afp32 (both restored).
    ("int8-wonly512", "int8-matched512"),
    ("int8-aonly512", "int8-matched512"),
    ("int8-selective512", "int8-wonly512"),
    ("int8-selective512", "int8-aonly512"),
    ("int8-wonly512", "int8-aonly512"),
    ("int8-selective512", "int8-matched512"),
]


def wa_stats() -> dict:
    out = {}
    for ds in ("kitti", "voc"):
        wa = load_block(PHASE_B, ds, RETINA, (PHASE_G,))
        entry = {"arms": {}, "contrasts": {}}
        for arm in ("int8-matched512", "int8-wonly512", "int8-aonly512",
                    "int8-selective512"):
            if wa.has(arm, "j95"):
                entry["arms"][arm] = {
                    fam: wa.level(arm, fam)[0]
                    for fam in ("j95", "corr12", "in_family", "held_out")}
        for a, b in WA_CONTRASTS:
            if not (wa.has(a, "j95") and wa.has(b, "j95")):
                continue
            entry["contrasts"][f"{a}_minus_{b}"] = {
                "j95": wa.diff(a, b, "j95"), "corr12": wa.diff(a, b, "corr12"),
                "deltaE": wa.delta_e(a, b),
                "per_family_deltaE": {
                    fam: wa.delta_e(a, b, f"fam_{fam}") for fam in
                    ("fog", "gaussian_noise", "jpeg", "motion_blur")}}
        out[ds] = entry
    return out


def fcos_stats() -> dict:
    out = {}
    for ds in ("kitti", "voc"):
        blk = load_block(PHASE_H, ds, FCOS)
        entry = {"arms": {}, "contrasts": {}}
        for arm in ("fp32", "fp8-matched512", "int8-matched512",
                    "int8-selective512"):
            if blk.has(arm, "j95"):
                entry["arms"][arm] = {
                    fam: blk.level(arm, fam)[0]
                    for fam in ("j95", "corr12", "in_family", "held_out")}
        for a, b in (("fp8-matched512", "int8-matched512"),
                     ("int8-selective512", "int8-matched512"),
                     ("fp8-matched512", "int8-selective512"),
                     ("fp32", "int8-selective512")):
            if not (blk.has(a, "j95") and blk.has(b, "j95")):
                continue
            entry["contrasts"][f"{a}_minus_{b}"] = {
                "j95": blk.diff(a, b, "j95"), "corr12": blk.diff(a, b, "corr12"),
                "deltaE": blk.delta_e(a, b),
                "per_family_deltaE": {
                    fam: blk.delta_e(a, b, f"fam_{fam}") for fam in
                    ("fog", "gaussian_noise", "jpeg", "motion_blur")}}
        out[ds] = entry
    return out


def main() -> None:
    result = {"units": "AP points; point = full-sample plug-in; ci95/se/p from "
                       "2,000 paired common-image bootstrap resamples",
              "yolo": yolo_stats(), "retinanet": retina_stats()}
    if (PHASE_G / "cell_points.json").is_file():
        result["retinanet_wa"] = wa_stats()
    if (PHASE_H / "cell_points.json").is_file():
        result["fcos"] = fcos_stats()
    path = SUPPORT / "nn_final_stats.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    het = result["yolo"]["heterogeneity"]
    print(f"wrote {path}; Q={het['Q']:.1f} p={het['p']:.1e} I2={het['I2']:.2f}")


if __name__ == "__main__":
    main()
