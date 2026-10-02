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

from nn_contrasts import load_block, summarize

REPO = Path(__file__).resolve().parents[1]
SUPPORT = REPO / "submission_support_20260911"
PHASE_B = SUPPORT / "phase_b_results"
PHASE_CD = SUPPORT / "phase_cd_results"
PHASE_E = SUPPORT / "phase_e_q95_results"
PHASE_F = SUPPORT / "phase_f_fold2_results"
PHASE_G = SUPPORT / "phase_g_wa_results"
PHASE_H = SUPPORT / "phase_h_fcos_results"
PHASE_I = SUPPORT / "phase_i_coco_pretrained_results"
PHASE_J = SUPPORT / "phase_j_maxcalib_results"
PHASE_K = SUPPORT / "phase_k_maxreg_results"
PHASE_L = SUPPORT / "phase_l_holdout_results"
PHASE_M = SUPPORT / "phase_m_cluster_bootstrap"
PHASE_N = SUPPORT / "phase_n_kitti_holdout_v2"
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
                   "images. A dataset-shared-schedule rerun "
                   "(shared_schedule/bootstrap) bounds the induced draw "
                   "correlation between model scales at |r|<=0.24, so the "
                   "independence approximation is at most mildly optimistic."}
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
    gls = gls_heterogeneity(rows)
    if gls:
        het["gls"] = gls
    return {"blocks": rows, "heterogeneity": het, "pairwise": pairs}


def gls_heterogeneity(rows: list[dict]) -> dict:
    """Covariance-aware (generalized) Cochran Q for the nine YOLO11 blocks.

    The phase-B caches seed each block separately, so the within-dataset
    covariance of DeltaE between model scales is taken from the
    dataset-shared-schedule rerun (shared_schedule/bootstrap, co-indexed
    draws); blocks on different datasets use disjoint images and are treated
    as independent. Point estimates are the frozen plug-in values.
    Q = r' S^-1 r with the GLS common mean; within-dataset tests use the 3x3
    sub-blocks and the VOC n-m contrast uses the full 2x2 covariance."""
    import sys as _sys
    _sys.path.insert(0, str(REPO / "analysis"))
    from nn_shared_schedule_correlation import delta_e_draws
    boot = SUPPORT / "shared_schedule" / "bootstrap"
    order = [(r["dataset"], r["model"]) for r in rows]
    if not all((boot / f"{d}__{m}__draws.npz").is_file() for d, m in order):
        return {}
    x = np.array([r["point"] for r in rows])
    cov = np.zeros((len(rows), len(rows)))
    n_draws = None
    for ds in ("kitti", "voc", "coco"):
        idx = [i for i, (d, _) in enumerate(order) if d == ds]
        # draw caches store AP as fractions; points/SEs are in AP points
        draws = 100.0 * np.stack([delta_e_draws(boot / f"{ds}__{order[i][1]}__draws.npz") for i in idx], axis=1)
        n_draws = draws.shape[0]
        sd, se = draws.std(axis=0, ddof=1), np.array([rows[i]["se"] for i in idx])
        if np.any(np.abs(sd / se - 1.0) > 0.35):
            raise SystemExit(f"shared-schedule draw SDs {sd} inconsistent with frozen SEs {se} ({ds})")
        cov[np.ix_(idx, idx)] = np.cov(draws, rowvar=False)

    def q_stat(sel: list[int]) -> tuple[float, int, float]:
        xs, inv = x[sel], np.linalg.inv(cov[np.ix_(sel, sel)])
        one = np.ones(len(sel))
        mu = float(one @ inv @ xs / (one @ inv @ one))
        r = xs - mu
        return float(r @ inv @ r), len(sel) - 1, mu

    q, df, mu = q_stat(list(range(len(rows))))
    out = {"Q": q, "df": df, "p": float(stats.chi2.sf(q, df)), "gls_mean": mu,
           "n_draws": int(n_draws), "within_dataset": {}}
    for ds in ("kitti", "voc", "coco"):
        sel = [i for i, (d, _) in enumerate(order) if d == ds]
        qs, dfs, _ = q_stat(sel)
        out["within_dataset"][ds] = {"Q": qs, "df": dfs, "p": float(stats.chi2.sf(qs, dfs))}
    i, j = 3, 4  # voc yolo11n, voc yolo11m
    var = cov[i, i] + cov[j, j] - 2 * cov[i, j]
    z = (x[i] - x[j]) / np.sqrt(var)
    out["voc_n_minus_m"] = {"diff": float(x[i] - x[j]), "z": float(z), "p": float(2 * stats.norm.sf(abs(z)))}
    return out


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
        if all(wa.has(a, "j95") for a in
               ("int8-matched512", "int8-wonly512", "int8-aonly512",
                "int8-selective512")):
            inter = {}
            for fam in ("j95", "corr12"):
                pts, drw = zip(*(wa.level(a, fam) for a in
                                 ("int8-selective512", "int8-wonly512",
                                  "int8-aonly512", "int8-matched512")))
                inter[fam] = summarize(
                    pts[0] - pts[1] - pts[2] + pts[3],
                    drw[0] - drw[1] - drw[2] + drw[3])
            entry["contrasts"]["operand_interaction"] = inter
        out[ds] = entry
    return out


def coco_pretrained_stats() -> dict:
    """Independent replication on torchvision pretrained COCO checkpoints."""
    out = {}
    for model in ("retinanet_pretrained", "fcos_pretrained"):
        blk = load_block(PHASE_I, "coco", model)
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
                     ("fp32", "int8-selective512"),
                     ("fp32", "int8-matched512")):
            if not (blk.has(a, "j95") and blk.has(b, "j95")):
                continue
            entry["contrasts"][f"{a}_minus_{b}"] = {
                "j95": blk.diff(a, b, "j95"), "corr12": blk.diff(a, b, "corr12"),
                "deltaE": blk.delta_e(a, b),
                "per_family_deltaE": {
                    fam: blk.delta_e(a, b, f"fam_{fam}") for fam in
                    ("fog", "gaussian_noise", "jpeg", "motion_blur")}}
        out[model] = entry
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


def maxcalib_stats() -> dict:
    """INT8 max-calibration counterfactual (B6): point estimates only.

    These arms were run as a post-hoc diagnostic without bootstrap draws, so
    the summary reports plug-in AP points per condition plus derived
    clean/corrupt-mean levels and the selective-minus-matched contrast as a
    bare point difference (no interval)."""
    src = PHASE_J / "cell_points.json"
    if not src.is_file():
        return {}
    pts = json.loads(src.read_text())
    corr = [f"{f}-s{s}" for f in ("fog", "gaussian_noise", "jpeg", "motion_blur")
            for s in (1, 3, 5)]
    out = {}
    for key, arms in pts.items():
        ds = key.split("/")[0]
        ent = {"arms": {}, "contrasts": {}}
        for arm, cells in arms.items():
            ent["arms"][arm] = {
                "clean": cells["clean-s0"],
                "codec_control": cells["codec-control-s0"],
                "corr12": float(np.mean([cells[c] for c in corr]))}
        a, b = "int8-maxsel512", "int8-max512"
        if a in arms and b in arms:
            ent["contrasts"]["maxsel_minus_max"] = {
                "clean": ent["arms"][a]["clean"] - ent["arms"][b]["clean"],
                "corr12": ent["arms"][a]["corr12"] - ent["arms"][b]["corr12"]}
        out[ds] = ent
    # head-local max-regression counterfactual (B7): max estimator applied to
    # the regression-head quantizers only, entropy retained elsewhere; same
    # point-estimate-only treatment as the graph-wide arms above.
    src_k = PHASE_K / "cell_points.json"
    if src_k.is_file():
        for key, arms in json.loads(src_k.read_text()).items():
            ds = key.split("/")[0]
            for arm, cells in arms.items():
                out.setdefault(ds, {"arms": {}, "contrasts": {}})["arms"][arm] = {
                    "clean": cells["clean-s0"],
                    "codec_control": cells["codec-control-s0"],
                    "corr12": float(np.mean([cells[c] for c in corr]))}
    return out


def holdout_stats() -> dict:
    """KITTI final-partition re-evaluation (B8): 1,197 images, point estimates.

    NOT a holdout: these images lie inside the RetinaNet checkpoint's
    5,985-image Ultralytics training partition (95 also in its calibration
    list); the "clean" key is the original clean image (clean-s0), not J95.

    The four factorial arms (matched/selective/W-only/A-only) re-evaluated on
    the later resplit's final partition; plug-in AP points per condition plus
    clean/corrupt-mean levels, no bootstrap draws."""
    src = PHASE_L / "cell_points.json"
    if not src.is_file():
        return {}
    pts = json.loads(src.read_text())
    corr = [f"{f}-s{s}" for f in ("fog", "gaussian_noise", "jpeg", "motion_blur")
            for s in (1, 3, 5)]
    out = {}
    for key, arms in pts.items():
        ent = {"arms": {}, "contrasts": {}}
        for arm, cells in arms.items():
            ent["arms"][arm] = {
                "clean": cells["clean-s0"],
                "codec_control": cells["codec-control-s0"],
                "corr12": float(np.mean([cells[c] for c in corr]))}
        a, b = "int8-selective512", "int8-matched512"
        if a in arms and b in arms:
            ent["contrasts"]["sel_minus_mat"] = {
                "clean": ent["arms"][a]["clean"] - ent["arms"][b]["clean"],
                "corr12": ent["arms"][a]["corr12"] - ent["arms"][b]["corr12"]}
        out[key] = ent
    return out


def _interaction(blk, fam: str) -> dict:
    pts, drw = zip(*(blk.level(a, fam) for a in
                     ("int8-selective512", "int8-wonly512", "int8-aonly512", "int8-matched512")))
    return summarize(pts[0] - pts[1] - pts[2] + pts[3], drw[0] - drw[1] - drw[2] + drw[3])


def _kitti_claims(yolo: dict, retina, fcos) -> list[tuple[str, dict]]:
    """The KITTI interval claims the manuscript makes, evaluated on any
    paired block set (image- or drive-resampled). ``yolo`` maps model ->
    Block; ``retina``/``fcos`` are Blocks holding every needed arm."""
    rows = []
    for m in ("yolo11n", "yolo11m", "yolo11x"):
        rows.append((f"YOLO11{m[-1]} DeltaE (FP8-INT8)", yolo[m].delta_e("fp8", "int8")))
    f, i, s_, w, a_ = ("fp8-matched512", "int8-matched512", "int8-selective512",
                       "int8-wonly512", "int8-aonly512")
    rows += [
        ("RetinaNet FP8-INT8 matched, J95 gap", retina.diff(f, i, "j95")),
        ("RetinaNet FP8-INT8 matched, DeltaE", retina.delta_e(f, i)),
        ("RetinaNet selective-matched, J95", retina.diff(s_, i, "j95")),
        ("RetinaNet FP8-selective, J95", retina.diff(f, s_, "j95")),
        ("RetinaNet FP8-selective, corrupt mean", retina.diff(f, s_, "corr12")),
        ("RetinaNet FP8-selective, motion-blur DeltaE", retina.delta_e(f, s_, "fam_motion_blur")),
        ("RetinaNet W-only-matched, J95", retina.diff(w, i, "j95")),
        ("RetinaNet A-only-matched, J95", retina.diff(a_, i, "j95")),
        ("RetinaNet selective-W-only, J95", retina.diff(s_, w, "j95")),
        ("RetinaNet operand interaction, corrupt mean", _interaction(retina, "corr12")),
        ("Fold 1 corrupt-calib - matched, held-out", retina.diff("int8-corruptcalib512", i, "held_out")),
        ("Fold 1 corrupt-calib - matched, held-out DeltaE",
         retina.delta_e("int8-corruptcalib512", i, "held_out")),
        ("Fold 2 corrupt-calib - matched, held-out", retina.diff("int8-cc2calib512", i, "in_family")),
        ("Fold 2 corrupt-calib - matched, held-out DeltaE",
         retina.delta_e("int8-cc2calib512", i, "in_family")),
        ("Fold 2 INT8-sel-cc2calib - INT8-selective, held-out",
         retina.diff("int8-sel-cc2calib512", s_, "in_family")),
        ("RetinaNet FP8-selective, JPEG DeltaE", retina.delta_e(f, s_, "fam_jpeg")),
        ("RetinaNet A-only-matched, DeltaE", retina.delta_e(a_, i)),
        ("RetinaNet operand interaction, J95", _interaction(retina, "j95")),
        ("INT8-q95calib - matched, J95", retina.diff("int8-q95calib512", i, "j95")),
        ("Fold 1 corrupt-calib - FP8, held-out", retina.diff("int8-corruptcalib512", f, "held_out")),
        ("Fold 1 INT8-sel-corruptcalib - FP8, held-out",
         retina.diff("int8-sel-corruptcalib512", f, "held_out")),
        ("Fold 1 INT8-sel-corruptcalib - INT8-selective, held-out",
         retina.diff("int8-sel-corruptcalib512", s_, "held_out")),
        ("Fold 2 corrupt-calib - q95calib, held-out DeltaE",
         retina.delta_e("int8-cc2calib512", "int8-q95calib512", "in_family")),
        ("Fold 2 INT8-sel-cc2calib - INT8-sel-q95calib, held-out",
         retina.diff("int8-sel-cc2calib512", "int8-sel-q95calib512", "in_family")),
        ("Fold 2 INT8-sel-cc2calib - FP8, held-out",
         retina.diff("int8-sel-cc2calib512", f, "in_family")),
        ("FCOS FP8-INT8 matched, J95", fcos.diff(f, i, "j95")),
        ("FCOS FP8-INT8 matched, corrupt mean", fcos.diff(f, i, "corr12")),
        ("FCOS FP8-INT8 matched, DeltaE", fcos.delta_e(f, i)),
        ("FCOS selective-matched, J95", fcos.diff(s_, i, "j95")),
        ("FCOS FP32-selective, J95", fcos.diff("fp32", s_, "j95")),
        ("FCOS FP32-selective, corrupt mean", fcos.diff("fp32", s_, "corr12")),
    ]
    return rows


def cluster_stats() -> dict:
    """Drive-clustered sensitivity for every KITTI interval claim (phase M).

    Image-level rows come from the frozen caches; drive rows from the
    nn_cluster_bootstrap_v1_20261002 caches (B=1,000; whole raw KITTI drives
    resampled, devkit mapping). Point estimates are identical plug-in values."""
    if not (PHASE_M / "cell_points.json").is_file():
        return {}
    image = _kitti_claims(
        {m: load_block(PHASE_B, "kitti", m) for m in ("yolo11n", "yolo11m", "yolo11x")},
        load_block(PHASE_B, "kitti", RETINA, (PHASE_CD, PHASE_E, PHASE_F, PHASE_G)),
        load_block(PHASE_H, "kitti", FCOS))
    drive = _kitti_claims(
        {m: load_block(PHASE_M, "kitti", m) for m in ("yolo11n", "yolo11m", "yolo11x")},
        load_block(PHASE_M, "kitti", RETINA), load_block(PHASE_M, "kitti", FCOS))
    rows = []
    for (name, im), (_, dr) in zip(image, drive):
        if abs(im["point"] - dr["point"]) > 1e-6:
            raise SystemExit(f"cluster plug-in mismatch for {name}: {im['point']} vs {dr['point']}")
        sig = lambda r: r["ci95"][0] > 0 or r["ci95"][1] < 0
        rows.append({"claim": name, "point": im["point"], "image": im, "drive": dr,
                     "image_excludes_zero": sig(im), "drive_excludes_zero": sig(dr),
                     "width_ratio": (dr["ci95"][1] - dr["ci95"][0]) / (im["ci95"][1] - im["ci95"][0])})
    clusters = json.loads((REPO / "manifests" / "clusters" /
                           "kitti_val_ultralytics_v1_drive_clusters.json").read_text())
    return {"rows": rows, "n_drives": clusters["n_clusters"], "n_images": clusters["n_images"],
            "n_changed": sum(r["image_excludes_zero"] != r["drive_excludes_zero"] for r in rows),
            "median_width_ratio": float(np.median([r["width_ratio"] for r in rows]))}


HOLDOUT_V2_CONTRASTS = [
    ("fp8-matched512", "int8-matched512"), ("int8-selective512", "int8-matched512"),
    ("fp8-matched512", "int8-selective512"), ("int8-wonly512", "int8-matched512"),
    ("int8-aonly512", "int8-matched512"), ("int8-selective512", "int8-wonly512"),
    ("fp32", "int8-selective512"),
]


def holdout_v2_stats() -> dict:
    """Genuine KITTI holdout (phase N): RetinaNet retrained on the 4,788-image
    resplit train list, calibrated on a disjoint 512-image train draw, and
    evaluated on the 1,197-image final partition it never saw."""
    out = {}
    for part in ("final_image", "final_drive", "val_image"):
        d = PHASE_N / part
        if not (d / "cell_points.json").is_file():
            continue
        blk = load_block(d, "kitti", RETINA)
        ent = {"arms": {}, "contrasts": {}}
        for arm in ("fp32", "fp8-matched512", "int8-matched512", "int8-selective512",
                    "int8-wonly512", "int8-aonly512"):
            if blk.has(arm, "j95"):
                ent["arms"][arm] = {"clean": blk.p[arm]["clean-s0"],
                                    "j95": blk.level(arm, "j95")[0],
                                    "corr12": blk.level(arm, "corr12")[0]}
        for a, b in HOLDOUT_V2_CONTRASTS:
            if blk.has(a, "j95") and blk.has(b, "j95"):
                ent["contrasts"][f"{a}_minus_{b}"] = {
                    "j95": blk.diff(a, b, "j95"), "corr12": blk.diff(a, b, "corr12"),
                    "deltaE": blk.delta_e(a, b)}
        if all(blk.has(a, "j95") for a in ("int8-matched512", "int8-wonly512",
                                            "int8-aonly512", "int8-selective512")):
            ent["contrasts"]["operand_interaction"] = {
                "j95": _interaction(blk, "j95"), "corr12": _interaction(blk, "corr12")}
        out[part] = ent
    if "final_image" in out and "final_drive" in out:
        for k, v in out["final_image"]["contrasts"].items():
            for fam in v:
                if abs(v[fam]["point"] - out["final_drive"]["contrasts"][k][fam]["point"]) > 1e-6:
                    raise SystemExit(f"holdout v2 plug-in mismatch {k}/{fam}")
    return out


def latency_stats() -> dict:
    """Idle-gated cudaEvent latency ledger (B4).

    Reads submission_support_20260911/nn_latency_summary.json produced by
    analysis/nn_latency_aggregate.py from per-rep benchmark records."""
    src = SUPPORT / "nn_latency_summary.json"
    if not src.is_file():
        return {}
    d = json.loads(src.read_text())
    out = {"timer": d.get("timer"), "conditions": {}, "ratio_vs_fp32": d.get("latency_ratio_vs_fp32", {})}
    for cid, e in sorted(d.get("conditions", {}).items()):
        out["conditions"][cid] = {
            "dataset": e["dataset"], "model": e["model"],
            "precision": e.get("precision"),
            "median_ms": e["median_ms"], "spread_ms": e["spread_ms"],
            "iqr_ms_typ": e["iqr_ms_typ"], "n_reps": e["n_reps"],
            "engine_sha256": e["engine_sha256"],
            "idle_allow_consistent": e["idle_allow_consistent"],
        }
    return out


def rebuild_variance_stats() -> dict:
    """TensorRT rebuild determinism audit (B3).

    Reads submission_support_20260911/nn_rebuild_variance_summary.json produced
    by analysis/nn_rebuild_variance_report.py (5 rebuilds per arm)."""
    src = SUPPORT / "nn_rebuild_variance_summary.json"
    if not src.is_file():
        return {}
    d = json.loads(src.read_text())
    return {
        "total_rebuilds": d.get("total_rebuilds"),
        "all_rebuilds_distinct": d.get("all_rebuilds_distinct"),
        "arms": {k: {"n_rebuilds": v["n_rebuilds"],
                     "distinct_engine_sha256": v["distinct_engine_sha256"],
                     "distinct_inspector_sha256": v["distinct_inspector_sha256"],
                     "n_layers": v.get("n_layers"),
                     "distinct_layer_counts": v.get("distinct_layer_counts")}
                 for k, v in d.get("arms", {}).items()},
        "ap_spread": d.get("ap_spread", {}),
    }


def actstats_stats() -> dict:
    """Head-region activation statistics (B5).

    Reads submission_support_20260911/nn_actstats_summary.json produced by
    analysis/nn_actstats_report.py from per-graph capture reports."""
    src = SUPPORT / "nn_actstats_summary.json"
    if not src.is_file():
        return {}
    return json.loads(src.read_text()).get("graphs", {})


def main() -> None:
    result = {"units": "AP points; point = full-sample plug-in; ci95/se/p from "
                       "2,000 paired common-image bootstrap resamples",
              "yolo": yolo_stats(), "retinanet": retina_stats()}
    if (PHASE_G / "cell_points.json").is_file():
        result["retinanet_wa"] = wa_stats()
    if (PHASE_H / "cell_points.json").is_file():
        result["fcos"] = fcos_stats()
    if (PHASE_I / "cell_points.json").is_file():
        result["coco_pretrained"] = coco_pretrained_stats()
    lat = latency_stats()
    if lat:
        result["latency"] = lat
    rb = rebuild_variance_stats()
    if rb:
        result["rebuild_variance"] = rb
    ac = actstats_stats()
    if ac:
        result["actstats"] = ac
    mc = maxcalib_stats()
    if mc:
        result["maxcalib"] = mc
    ho = holdout_stats()
    if ho:
        result["kitti_holdout"] = ho
    cl = cluster_stats()
    if cl:
        result["kitti_drive_cluster"] = cl
    h2 = holdout_v2_stats()
    if h2:
        result["kitti_holdout_v2"] = h2
    path = SUPPORT / "nn_final_stats.json"
    path.write_text(json.dumps(result, indent=2) + "\n")
    het = result["yolo"]["heterogeneity"]
    print(f"wrote {path}; Q={het['Q']:.1f} p={het['p']:.1e} I2={het['I2']:.2f}")


if __name__ == "__main__":
    main()
