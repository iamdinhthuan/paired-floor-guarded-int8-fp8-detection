#!/usr/bin/env python3
"""Phase C/D intervention report: held-out vs in-family corrupt-mean
contrasts from the paired per-cell bootstrap draws.

Reads {dataset}__retinanet_r50_fpn_v2__draws.npz produced by
run_nn_paired_bootstrap.py (8-arm intervention blocks) and emits a JSON
report with, per dataset:
  * per-arm clean / q95 / corrupt-mean(12) / held-out mean (fog+motion_blur)
    / in-family mean (gaussian_noise+jpeg) point + CI95
  * pre-registered contrasts on held-out and in-family means
All values are absolute AP points.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

HELD_OUT = ("fog", "motion_blur")
IN_FAMILY = ("gaussian_noise", "jpeg")
SEVERITIES = (1, 3, 5)

ARMS = ("fp32", "fp8-legacy", "int8-legacy", "fp8-matched512", "int8-matched512",
        "int8-selective512", "int8-corruptcalib512", "int8-sel-corruptcalib512",
        "int8-q95calib512", "int8-sel-q95calib512")

CONTRASTS = (
    ("int8-corruptcalib512", "int8-matched512"),      # primary: mechanism (full INT8)
    ("int8-q95calib512", "int8-matched512"),          # codec-only component (full INT8)
    ("int8-corruptcalib512", "int8-q95calib512"),     # corruption coverage beyond codec
    ("int8-sel-corruptcalib512", "int8-selective512"),  # calib effect on selective
    ("int8-sel-q95calib512", "int8-selective512"),    # codec-only on selective
    ("int8-sel-corruptcalib512", "int8-sel-q95calib512"),  # coverage beyond codec
    ("int8-sel-corruptcalib512", "fp8-matched512"),   # closes VOC residual gap?
    ("int8-corruptcalib512", "fp8-matched512"),       # intervention vs FP8 ceiling
    ("fp8-matched512", "int8-matched512"),            # Phase-B baseline contrast
    ("fp32", "int8-corruptcalib512"),                 # residual vs fp32
)


def ci95(draws: np.ndarray) -> list[float]:
    return [float(np.percentile(draws, 2.5)), float(np.percentile(draws, 97.5))]


def cell_key(arm: str, corruption: str, severity: int) -> str:
    return f"{arm}__cell__{corruption}-s{severity}"


def mean_draws(npz: np.lib.npyio.NpzFile, arm: str, corruptions) -> np.ndarray:
    keys = [cell_key(arm, c, s) for c in corruptions for s in SEVERITIES]
    return np.stack([npz[k] for k in keys], axis=1).mean(axis=1)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--bootstrap-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    out_path = Path(args.out)
    if out_path.exists():
        raise SystemExit(f"refusing to overwrite report: {out_path}")

    report = {"estimand": "AP points; held_out = fog+motion_blur s1/s3/s5; "
                          "in_family = gaussian_noise+jpeg s1/s3/s5",
              "datasets": {}}
    for npz_path in sorted(Path(args.bootstrap_dir).glob("*__retinanet_r50_fpn_v2__draws.npz")):
        dataset = npz_path.name.split("__")[0]
        npz = np.load(npz_path)
        arms = {}
        for arm in ARMS:
            if f"{arm}__clean" not in npz:
                continue
            arms[arm] = {
                "clean": {"point": float(np.mean(npz[f"{arm}__clean"])),
                          "ci95": ci95(npz[f"{arm}__clean"])},
                "q95": {"point": float(np.mean(npz[f"{arm}__q95"])),
                        "ci95": ci95(npz[f"{arm}__q95"])},
                "corr_mean12": {"point": float(np.mean(npz[f"{arm}__corr_mean"])),
                                "ci95": ci95(npz[f"{arm}__corr_mean"])},
            }
            held = mean_draws(npz, arm, HELD_OUT)
            infam = mean_draws(npz, arm, IN_FAMILY)
            arms[arm]["held_out_mean"] = {"point": float(np.mean(held)), "ci95": ci95(held)}
            arms[arm]["in_family_mean"] = {"point": float(np.mean(infam)), "ci95": ci95(infam)}
        contrasts = {}
        for a, b in CONTRASTS:
            if a not in arms or b not in arms:
                continue
            entry = {}
            for name, corruptions in (("held_out", HELD_OUT), ("in_family", IN_FAMILY)):
                diff = mean_draws(npz, a, corruptions) - mean_draws(npz, b, corruptions)
                entry[f"{name}_mean_diff"] = {"point": float(np.mean(diff)), "ci95": ci95(diff)}
            qa, qb = npz[f"{a}__q95"], npz[f"{b}__q95"]
            if np.isnan(qa).all() or np.isnan(qb).all():
                clean_diff = npz[f"{a}__clean"] - npz[f"{b}__clean"]
                basis = "orig"
            else:
                clean_diff = qa - qb
                basis = "q95"
            entry["clean_diff"] = {"point": float(np.mean(clean_diff)), "ci95": ci95(clean_diff),
                                   "basis": basis}
            corr_diff = npz[f"{a}__corr_mean"] - npz[f"{b}__corr_mean"]
            entry["corr_mean12_diff"] = {"point": float(np.mean(corr_diff)),
                                         "ci95": ci95(corr_diff)}
            contrasts[f"{a}_minus_{b}"] = entry
        report["datasets"][dataset] = {"arms": arms, "contrasts": contrasts}
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_path), "datasets": list(report["datasets"])}, indent=2))


if __name__ == "__main__":
    main()
