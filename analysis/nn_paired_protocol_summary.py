#!/usr/bin/env python3
"""Phase-B paired-protocol point-estimate report (no new inference).

Aggregates hash-bound metric records across attempts into per-block
tables: absolute clean AP (original and codec-control Q95), corrupted
mean AP, retention, FP8-INT8 gaps/interactions, and the RetinaNet
recipe contrast (legacy vs matched vs selective-precision).

Run on the evidence host: python analysis/nn_paired_protocol_summary.py
--root /home/thuan/topic_c_ivc --attempt nn_paired_protocol_v1_20260924
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

CORRUPTIONS = ("fog", "gaussian_noise", "jpeg", "motion_blur")
SEVERITIES = (1, 3, 5)
ATTEMPT_DIRS = [
    "nn_paired_protocol_v1_20260924",
    "kitti_pilot_117_v1",
    "voc_pilot_117_v1",
    "coco_uniform_p0_v1",
    "codec_control_p0_v1",
    "cross_family_v1",
]
YOLO_PRECISIONS = {"fp32": "fp32", "fp8": "fp8", "int8": "int8-entropy"}
BOOT_ALIASES = {
    "fp8matched_minus_int8matched": "fp8-matched512_minus_int8-matched512",
    "selective_minus_full_int8": "int8-selective512_minus_int8-matched512",
    "fp32_minus_selective": "fp32_minus_int8-selective512",
    "fp8matched_minus_selective": "fp8-matched512_minus_int8-selective512",
    "fp8legacy_minus_int8legacy": "fp8-legacy_minus_int8-legacy",
    "fp32_minus_int8legacy": "fp32_minus_int8-legacy",
}
RETINANET_ARMS = {
    "fp32": "fp32",
    "fp8-legacy": "fp8",
    "int8-legacy": "int8-entropy",
    "fp8-matched512": "fp8-matched512",
    "int8-matched512": "int8-matched512",
    "int8-selective512": "int8-selective512",
}


def load_json(path: Path) -> dict:
    return json.loads(path.read_text(encoding="utf-8"))


def metric_lookup(root: Path, dataset: str, model: str, arm: str, corruption: str, severity: int) -> dict | None:
    """Find the metric record for one cell, preferring the new attempt."""
    for attempt in ATTEMPT_DIRS:
        mdir = root / "outputs" / "metrics" / attempt
        if not mdir.is_dir():
            continue
        for path in sorted(mdir.glob(f"{dataset}_val__{model}__*__{corruption}-s{severity}*.json")):
            record = load_json(path)
            if record.get("corruption") == "codec-control" and corruption != "codec-control":
                continue
            precision = record.get("precision")
            condition = record.get("condition_id", "")
            if model.startswith("retinanet"):
                if attempt == "nn_paired_protocol_v1_20260924" and f"__{arm}__" in condition:
                    return record
                if attempt == "cross_family_v1" and precision == arm:
                    return record
            elif model.startswith("yolo11"):
                if precision in (arm, arm + "-entropy", arm + "-none") or f"__{arm}-" in condition or f"__{arm}__" in condition:
                    return record
    return None


def coco_subset_metric(root: Path, attempt: str, model: str, arm: str, corruption: str, severity: int) -> dict | None:
    mdir = root / "outputs" / "metrics" / attempt
    pattern = f"subset_coco_val2017__{model}__{arm}*__{corruption}-s{severity}*.json"
    for path in sorted(mdir.glob(pattern)):
        return load_json(path)
    return None


def collect_block(root: Path, attempt: str, dataset: str, model: str, arms: dict, universe: str) -> dict:
    cells: dict[str, dict[str, dict]] = {}
    for arm_label in arms:
        arm_cells = {}
        for corruption in ("clean", "codec-control", *CORRUPTIONS):
            for severity in ([0] if corruption in ("clean", "codec-control") else SEVERITIES):
                if universe == "coco-subset":
                    record = coco_subset_metric(root, attempt, model, arm_label, corruption, severity)
                else:
                    record = metric_lookup(root, dataset, model, arms[arm_label], corruption, severity)
                if record is not None:
                    arm_cells[f"{corruption}-s{severity}"] = record
        cells[arm_label] = arm_cells
    return cells


def summarize_arm(cells: dict) -> dict:
    clean = cells.get("clean-s0")
    q95 = cells.get("codec-control-s0")
    corrupted = [cells[f"{c}-s{s}"]["stats"]["AP"] * 100
                 for c in CORRUPTIONS for s in SEVERITIES if f"{c}-s{s}" in cells]
    return {
        "n_conditions": len(cells),
        "clean_orig_ap": round(clean["stats"]["AP"] * 100, 4) if clean else None,
        "clean_q95_ap": round(q95["stats"]["AP"] * 100, 4) if q95 else None,
        "corrupted_mean_ap": round(sum(corrupted) / len(corrupted), 4) if corrupted else None,
        "corrupted_cells": len(corrupted),
        "retention_vs_q95": round(sum(corrupted) / len(corrupted) / (q95["stats"]["AP"] * 100), 4)
        if corrupted and q95 and q95["stats"]["AP"] > 0 else None,
    }


def contrasts(summary: dict[str, dict], ref: str, arm: str) -> dict:
    a, b = summary.get(ref), summary.get(arm)
    if not a or not b or a["clean_q95_ap"] is None or b["clean_q95_ap"] is None:
        return {}
    if a["corrupted_mean_ap"] is None or b["corrupted_mean_ap"] is None:
        return {}
    g0 = a["clean_q95_ap"] - b["clean_q95_ap"]
    gc = a["corrupted_mean_ap"] - b["corrupted_mean_ap"]
    return {"G0_q95": round(g0, 4), "Gc": round(gc, 4), "deltaE": round(gc - g0, 4)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/thuan/topic_c_ivc"))
    parser.add_argument("--attempt", default="nn_paired_protocol_v1_20260924")
    parser.add_argument("--out", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    attempt = args.attempt
    out = args.out or root / "outputs" / "analysis" / attempt
    out.mkdir(parents=True, exist_ok=True)

    report = {"attempt": attempt, "blocks": []}
    lines = ["# Phase-B paired protocol — point-estimate summary",
             "",
             "All values are AP points. Clean basis: original clean and",
             "codec-control Q95. Corrupted mean = equal weight over the 12",
             "corruption-severity cells. Retention = corrupted / Q95-clean.",
             ""]

    yolo_arms = {"fp32": "fp32", "fp8": "fp8", "int8": "int8-entropy"}
    for dataset, universe in (("kitti", "kitti_val"), ("voc", "voc_val")):
        for model in ("yolo11n", "yolo11m", "yolo11x"):
            cells = collect_block(root, attempt, dataset, model, yolo_arms, universe)
            block = {"dataset": dataset, "model": model, "universe": universe,
                     "arms": {a: summarize_arm(c) for a, c in cells.items()}}
            block["contrasts"] = {"fp8_minus_int8": contrasts(block["arms"], "fp8", "int8"),
                                  "fp32_minus_int8": contrasts(block["arms"], "fp32", "int8")}
            report["blocks"].append(block)
    for model in ("yolo11n", "yolo11m", "yolo11x"):
        cells = collect_block(root, attempt, "coco", model, yolo_arms, "coco-subset")
        block = {"dataset": "coco", "model": model, "universe": "coco-subset-2000",
                 "arms": {a: summarize_arm(c) for a, c in cells.items()}}
        block["contrasts"] = {"fp8_minus_int8": contrasts(block["arms"], "fp8", "int8"),
                              "fp32_minus_int8": contrasts(block["arms"], "fp32", "int8")}
        report["blocks"].append(block)

    for dataset in ("kitti", "voc"):
        cells = collect_block(root, attempt, dataset, "retinanet_r50_fpn_v2", RETINANET_ARMS, f"{dataset}_val")
        block = {"dataset": dataset, "model": "retinanet_r50_fpn_v2", "universe": f"{dataset}_val",
                 "arms": {a: summarize_arm(c) for a, c in cells.items()}}
        block["contrasts"] = {
            "fp8matched_minus_int8matched": contrasts(block["arms"], "fp8-matched512", "int8-matched512"),
            "selective_minus_full_int8": contrasts(block["arms"], "int8-selective512", "int8-matched512"),
            "fp32_minus_selective": contrasts(block["arms"], "fp32", "int8-selective512"),
        }
        report["blocks"].append(block)

    boot_dir = root / "outputs" / "analysis" / attempt / "bootstrap"
    for block in report["blocks"]:
        boot_path = boot_dir / f"{block['dataset']}__{block['model']}__bootstrap.json"
        boot = load_json(boot_path) if boot_path.is_file() else None
        lines.append(f"## {block['dataset']} / {block['model']}  ({block['universe']})")
        lines.append("")
        lines.append("| arm | clean-orig | clean-Q95 | corrupt-mean | retention | n |")
        lines.append("|---|---:|---:|---:|---:|---:|")
        for arm, s in block["arms"].items():
            fmt = lambda v: "-" if v is None else f"{v:.4f}"
            lines.append(f"| {arm} | {fmt(s['clean_orig_ap'])} | {fmt(s['clean_q95_ap'])} "
                         f"| {fmt(s['corrupted_mean_ap'])} | {fmt(s['retention_vs_q95'])} | {s['n_conditions']} |")
        lines.append("")
        for name, c in block["contrasts"].items():
            if not c:
                continue
            line = f"- {name}: G0={c['G0_q95']:+.4f}  Gc={c['Gc']:+.4f}  ΔE={c['deltaE']:+.4f}"
            boot_name = BOOT_ALIASES.get(name, name)
            if boot and boot_name in boot.get("contrasts", {}):
                ci = boot["contrasts"][boot_name]
                dE = [v * 100 for v in ci["deltaE"]]
                line += f"  (B={boot['n_boot']} paired ΔE 95% CI [{dE[0]:+.4f}, {dE[2]:+.4f}], basis={ci['basis']})"
            lines.append(line)
        lines.append("")

    (out / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    (out / "summary.md").write_text("\n".join(lines))
    print(f"wrote {out / 'summary.md'}")
    print("\n".join(lines[:60]))


if __name__ == "__main__":
    main()
