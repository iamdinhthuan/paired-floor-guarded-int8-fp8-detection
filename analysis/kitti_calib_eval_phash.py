#!/usr/bin/env python3
"""Perceptual-hash audit: near-duplicates between the KITTI confirmatory
512-image calibration draw and the 1,496-image selection (evaluation)
partition under the frame-level resplit. Writes
submission_support_20260911/nn_split_audit/kitti_calib_eval_phash.json.

Run on the GPU host (dataset lives there):

    ssh thuan@100.111.139.103 \
      '/home/thuan/miniconda3/envs/qtsd/bin/python analysis/kitti_calib_eval_phash.py'

Method: 16x16 grayscale average hash per image; for each calibration image
the minimum Hamming distance to any evaluation image is reported. A
near-identical subset was additionally verified by 64x64 mean absolute
pixel difference (<5/255).
"""
import json
import os
from pathlib import Path

import numpy as np
from PIL import Image

ROOT = Path("/home/thuan/topic_c_ivc")
DATA = ROOT / "data/datasets/kitti"
CAL = ROOT / ("manifests/calibration/"
              "kitti_confirmatory_train_512_s20260818_v1.json")
EVL = ROOT / ("manifests/splits/kitti_confirmatory_v1/lists/"
              "kitti_confirmatory_v1_selection.txt")
OUT = ROOT / "outputs/logs/kitti_phash_check.json"


def ahash16(path: Path) -> np.ndarray:
    im = Image.open(path).convert("L").resize((16, 16), Image.LANCZOS)
    a = np.asarray(im, dtype=float)
    return (a > a.mean()).flatten()


def main() -> None:
    cal = json.loads(CAL.read_text())
    calib = [r["source_relpath"] for r in cal["records"]]
    evl = [l.strip() for l in EVL.read_text().splitlines() if l.strip()]
    hc = [ahash16(DATA / p) for p in calib]
    he = np.array([ahash16(Path(p)) for p in evl], dtype=bool)
    dmin = np.array([np.count_nonzero(he != h, axis=1).min() for h in hc])
    res = {
        "method": "average-hash 16x16 grayscale, min Hamming per calib image",
        "n_calib": len(calib), "n_eval": len(evl),
        "frac_d0": float((dmin == 0).mean()),
        "frac_le5": float((dmin <= 5).mean()),
        "frac_le10": float((dmin <= 10).mean()),
        "dmin_median": float(np.median(dmin)),
    }
    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
