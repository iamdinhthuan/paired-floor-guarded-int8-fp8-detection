#!/usr/bin/env python3
"""Collect full-sample plug-in AP (AP points) for every bootstrap cell.

For each ``*__bootstrap.json`` in --bootstrap-dir, resolve each cell's metric
record by stem across the frozen attempt directories and write
{dataset: {arm: {condition: AP_points}}}. These plug-in values are the point
estimates reported in the manuscript; bootstrap draws supply intervals only.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

ATTEMPTS = ("nn_maxcalib_v1_20260930", "nn_coco_pretrained_v1_20260930",
            "nn_fcos_replication_v1_20260927",
            "nn_corruptcalib_fold2_v1_20260925",
            "nn_q95calib_v1_20260924",
            "nn_paired_protocol_v1_20260924",
            "nn_corruptcalib_v1_20260924", "kitti_pilot_117_v1", "voc_pilot_117_v1",
            "coco_uniform_p0_v1", "codec_control_p0_v1", "cross_family_v1")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/home/thuan/topic_c_ivc")
    ap.add_argument("--bootstrap-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    root = Path(args.root)
    out: dict = {}
    for path in sorted(Path(args.bootstrap_dir).glob("*__bootstrap.json")):
        block = json.loads(path.read_text())
        key = f"{block['dataset']}/{block['model']}"
        for cell in block["cells"]:
            stem = cell["stem"]
            # COCO confirmatory cells were evaluated on the frozen 2,000-image
            # subset; their metric records carry a "subset_" dataset prefix.
            # Without this, the stem resolves to the 5,000-image full-val
            # records in the exploratory attempts.
            if block["dataset"] == "coco" and not stem.startswith("coco_paired2000"):
                stem = f"subset_{stem}"
            hit = next((root / "outputs" / "metrics" / a / f"{stem}.json"
                        for a in ATTEMPTS
                        if (root / "outputs" / "metrics" / a / f"{stem}.json").is_file()),
                       None)
            if hit is None:
                raise SystemExit(f"missing metric for {cell['stem']}")
            metric = json.loads(hit.read_text())
            n_img = metric.get("n_images")
            if n_img is not None and n_img != block["n_images"]:
                raise SystemExit(
                    f"n_images mismatch for {stem}: metric={n_img} "
                    f"draws={block['n_images']} ({hit})")
            ap_pts = metric["stats"]["AP"] * 100.0
            out.setdefault(key, {}).setdefault(cell["arm"], {})[cell["condition"]] = ap_pts
    Path(args.out).write_text(json.dumps(out, indent=2) + "\n")
    print(json.dumps({k: sorted(v) for k, v in out.items()}, indent=1))


if __name__ == "__main__":
    main()
