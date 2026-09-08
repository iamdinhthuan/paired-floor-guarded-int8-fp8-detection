#!/usr/bin/env python3
"""Audit the clean/corrupt covariance contribution using retained paired draws.

This is an empirical variance ablation, not an independent four-arm bootstrap
or a confidence-coverage experiment. It leaves INT8/FP8 pairing within each
condition intact and analytically sets only clean/corrupt gap covariance to zero.
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from bootstrap_format_contrast import canonical_hash, load_draw_cache
from topic_c.manifest import sha256_file


def covariance_diagnostic(clean_gap: np.ndarray, direct: np.ndarray) -> dict:
    clean_gap, direct = np.asarray(clean_gap, dtype=float), np.asarray(direct, dtype=float)
    if (clean_gap.ndim != 1 or clean_gap.shape != direct.shape or clean_gap.size < 2
            or not np.isfinite(clean_gap).all() or not np.isfinite(direct).all()):
        raise ValueError("finite matched one-dimensional bootstrap vectors are required")
    corrupt_gap = direct + clean_gap
    paired = float(np.var(direct, ddof=1))
    zero_cov = float(np.var(corrupt_gap, ddof=1) + np.var(clean_gap, ddof=1))
    twice_cov = float(2 * np.cov(corrupt_gap, clean_gap, ddof=1)[0, 1])
    if not np.isclose(paired, zero_cov - twice_cov, rtol=1e-10, atol=1e-10):
        raise RuntimeError("bootstrap covariance identity failed")
    return dict(paired_variance=paired, zero_covariance_variance=zero_cov,
                twice_covariance=twice_cov, paired_sd=float(np.sqrt(paired)),
                zero_covariance_sd=float(np.sqrt(zero_cov)),
                sd_ratio=float(np.sqrt(zero_cov / paired)) if paired > 0 else None)


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    source = root / "outputs/bootstrap/ivc_format_contrast_v1"
    rows, bindings = [], {}
    for path in sorted(source.glob("*__*__*.json")):
        if path.stem.endswith("__clean-s0"):
            continue
        report = json.loads(path.read_text())
        if report["artifact_sha256"] != canonical_hash(report, "artifact_sha256"):
            raise RuntimeError(f"primary component hash mismatch: {path}")
        dataset, model, condition = path.stem.split("__")
        clean_ref, direct_ref = report["clean_arm_cache"], report["temporary_draw_cache"]
        clean_path, direct_path = root / clean_ref["path"], root / direct_ref["path"]
        if sha256_file(clean_path) != clean_ref["sha256"]:
            raise RuntimeError(f"clean cache hash mismatch: {clean_path}")
        with np.load(clean_path, allow_pickle=False) as clean:
            metadata = json.loads(str(clean["metadata"].item()))
            if (metadata["cache_identity_sha256"] != clean_ref["identity_sha256"]
                    or metadata["dataset"] != dataset or metadata["model"] != model
                    or metadata["bootstrap_schedule"]["sha256"] != report["bootstrap_schedule"]["sha256"]
                    or metadata["annotation_sha256"] != report["annotation"]["sha256"]
                    or any(metadata["input_hashes"][arm] != report["input_hashes"][arm]
                           for arm in ("int8_clean", "fp8_clean"))):
                raise RuntimeError(f"clean/direct schedule or source mismatch: {path}")
            clean_gap = 100 * (clean["fp8_clean"][:, 0] - clean["int8_clean"][:, 0])
        direct = load_draw_cache(direct_path, expected_sha256=direct_ref["sha256"],
                                 expected_identity_sha256=direct_ref["identity_sha256"],
                                 n_boot=report["n_boot"], schedule_sha256=report["bootstrap_schedule"]["sha256"])
        direct_ap = direct["delta_e_all"] * 100
        if not np.allclose(np.percentile(direct_ap, [2.5, 50, 97.5]),
                           np.asarray(report["percentile_intervals"]["delta_e"]["all"]) * 100,
                           rtol=1e-9, atol=1e-9):
            raise RuntimeError(f"primary interval mismatch: {path}")
        rows.append(dict(dataset=dataset, model=model, condition=condition,
                         n_boot=report["n_boot"], **covariance_diagnostic(clean_gap, direct_ap)))
        for item in (path, clean_path, direct_path):
            bindings[str(item.relative_to(root))] = sha256_file(item)
    expected = {(d, m, f"{c}-s{s}") for d in ("coco", "voc", "kitti", "tt100k")
                for m in ("yolo11n", "yolo11m", "yolo11x")
                for c in ("gaussian_noise", "motion_blur", "fog", "jpeg") for s in (1, 3, 5)}
    if len(rows) != 144 or {(r["dataset"], r["model"], r["condition"]) for r in rows} != expected:
        raise RuntimeError("covariance audit requires exactly the 144 primary cells")
    groups = {}
    for name in ("coco", "voc", "kitti", "tt100k", "all"):
        subset = [r for r in rows if name == "all" or r["dataset"] == name]
        ratios = [r["sd_ratio"] for r in subset if r["sd_ratio"] is not None]
        groups[name] = dict(n=len(subset), median_paired_sd=float(np.median([r["paired_sd"] for r in subset])),
                            median_zero_covariance_sd=float(np.median([r["zero_covariance_sd"] for r in subset])),
                            median_sd_ratio=float(np.median(ratios)), min_sd_ratio=min(ratios), max_sd_ratio=max(ratios),
                            n_increased=sum(r["twice_covariance"] > 1e-12 for r in subset),
                            n_decreased=sum(r["twice_covariance"] < -1e-12 for r in subset))
    args.output.mkdir(parents=True, exist_ok=True)
    with (args.output / "cells.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader(); writer.writerows(rows)
    report = dict(schema_version=1, scope=__doc__, groups=groups, source_sha256=bindings,
                  implementation_sha256=sha256_file(__file__), cells_sha256=sha256_file(args.output / "cells.csv"))
    report["summary_sha256"] = canonical_hash(report, "summary_sha256")
    (args.output / "summary.json").write_text(json.dumps(report, indent=2) + "\n")
    tex = [r"\begin{tabular}{lrrrrr}", r"\toprule",
           r"Dataset & $n$ & Paired SD & Zero-cov. SD & SD ratio & Increased\\", r"\midrule"]
    for name, group in groups.items():
        label = {"coco": "COCO", "voc": "VOC", "kitti": "KITTI", "tt100k": "TT100K", "all": "All"}[name]
        tex.append(f"{label} & {group['n']} & {group['median_paired_sd']:.3f} & {group['median_zero_covariance_sd']:.3f} & {group['median_sd_ratio']:.3f} & {group['n_increased']}" + r"\\")
    tex += [r"\bottomrule", r"\end{tabular}"]
    args.table.parent.mkdir(parents=True, exist_ok=True)
    args.table.write_text("\n".join(tex) + "\n")
    print(json.dumps(groups, indent=2))


if __name__ == "__main__":
    main()
