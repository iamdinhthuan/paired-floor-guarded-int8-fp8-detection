#!/usr/bin/env python3
"""Cross-block draw correlation under a shared dataset resample schedule.

The phase-B draw caches seed their image resamples by (namespace|dataset|model),
so model blocks within a dataset do not share resampling draws and cross-block
covariance is not identifiable. The ``nn_shared_schedule_v1_20260930`` rerun
seeds by (namespace|dataset) only, making per-draw blocks within a dataset
co-indexed; this script reports the Pearson correlation of the per-draw
DeltaE (fp8 - int8) vectors across the three YOLO11 scales on each dataset.

A correlation materially different from zero means the independence
approximation in the pooled heterogeneity statistics biases the pooled
standard error (positive correlation overestimates effective sample size);
the manuscript limitation note reports the observed range.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np

MODELS = ("yolo11n", "yolo11m", "yolo11x")
DATASETS = ("kitti", "voc", "coco")
CORRUPT = [f"{c}-s{s}" for c in ("fog", "gaussian_noise", "jpeg", "motion_blur")
           for s in (1, 3, 5)]


def delta_e_draws(npz_path: Path) -> np.ndarray:
    """Per-draw DeltaE (fp8 - int8): corr12 mean gap minus j95 gap."""
    z = np.load(npz_path)
    def corr_mean(arm: str) -> np.ndarray:
        return np.mean([z[f"{arm}__cell__{c}"] for c in CORRUPT], axis=0)
    return (corr_mean("fp8") - corr_mean("int8")) - (
        z["fp8__cell__codec-control-s0"] - z["int8__cell__codec-control-s0"])


def main() -> None:
    boot_dir = Path(sys.argv[1]) if len(sys.argv) > 1 else Path(
        "submission_support_20260911/shared_schedule/bootstrap")
    report = {}
    for ds in DATASETS:
        draws = {}
        for model in MODELS:
            path = boot_dir / f"{ds}__{model}__draws.npz"
            if not path.is_file():
                print(f"missing {path}", file=sys.stderr)
                break
            draws[model] = delta_e_draws(path)
        else:
            n = len(next(iter(draws.values())))
            pairs = {}
            for i, a in enumerate(MODELS):
                for b in MODELS[i + 1:]:
                    r = float(np.corrcoef(draws[a], draws[b])[0, 1])
                    pairs[f"{a}~{b}"] = r
            report[ds] = {"n_draws": n, "pearson_r": pairs}
            print(ds, {k: round(v, 3) for k, v in pairs.items()})
    out = boot_dir / "shared_schedule_correlation.json"
    out.write_text(json.dumps(report, indent=2) + "\n")
    print("wrote", out)


if __name__ == "__main__":
    main()
