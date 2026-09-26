#!/usr/bin/env python3
"""Compute clean / q95 / corrupt-mean-12 point AP for an attempt's arms.

Reads metric JSONs under outputs/metrics/{attempt}/ on the remote host and
writes a points JSON compatible with build_nn_tables.load_points().
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

CONDITIONS = [("clean", 0), ("codec-control", 0)] + [
    (c, s) for c in ("fog", "gaussian_noise", "jpeg", "motion_blur") for s in (1, 3, 5)
]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", default="/home/thuan/topic_c_ivc")
    parser.add_argument("--attempt", required=True)
    parser.add_argument("--model", default="retinanet_r50_fpn_v2")
    parser.add_argument("--arms", required=True, help="comma-separated arm names")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    metrics_dir = Path(args.root) / "outputs" / "metrics" / args.attempt
    out = {}
    for arm in args.arms.split(","):
        for dataset in ("kitti", "voc"):
            vals = {}
            for corruption, sev in CONDITIONS:
                prefix = (f"{dataset}_val__{args.model}__{arm}__"
                          f"{corruption}-s{sev}")
                path = metrics_dir / f"{prefix}.json"
                if not path.is_file():
                    continue
                metric = json.loads(path.read_text())
                vals[f"{corruption}-s{sev}"] = metric["stats"]["AP"] * 100.0
            corrupt = [v for k, v in vals.items()
                       if k.split("-s")[0] not in ("clean", "codec-control")]
            if not corrupt:
                continue
            out.setdefault(dataset, {})[arm] = {
                "clean": vals.get("clean-s0"),
                "q95": vals.get("codec-control-s0"),
                "corr12": sum(corrupt) / len(corrupt),
                "n_corrupt_cells": len(corrupt),
            }
    out_path = Path(args.out)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    out_path.write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({ds: {a: {k: round(v, 3) if isinstance(v, float) else v
                               for k, v in m.items()}
                           for a, m in arms.items()}
                      for ds, arms in out.items()}, indent=2))


if __name__ == "__main__":
    main()
