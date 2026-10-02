#!/usr/bin/env python3
"""Build the frozen COCO-2000 pretrained-detector eval assets:

* per-condition image manifests restricted to the frozen paired 2,000-image
  subset (codec-control + 12 corruption cells + clean),
* an eval annotation JSON restricted to the subset (real COCO categories),
* two index-mapping category files for raw torchvision label indices
  (RetinaNet: channel k -> category k via label = channel - 1;
   FCOS: label = channel directly).
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from topic_c.manifest import manifest_sha256, sha256_file

CONDS = (
    ["codec_control_q95"]
    + [f"{fam}_s{sev}" for fam in
       ("fog", "gaussian_noise", "jpeg", "motion_blur") for sev in (1, 3, 5)]
)


def write_manifest(document: dict, path: Path) -> None:
    document["manifest_sha256"] = manifest_sha256(document)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document) + "\n", encoding="utf-8")
    path.with_suffix(path.suffix + ".complete").write_text(
        document["manifest_sha256"], encoding="utf-8")


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--root", default="/home/thuan/topic_c_ivc")
    args = p.parse_args()
    root = Path(args.root)

    subset = json.loads((root / "manifests/subsets/nn_paired_protocol_v1_20260924"
                        "/coco_val2017_paired_2000_s20260924.json").read_text())
    keep = set(int(i) for i in subset["image_ids"])
    assert len(keep) == 2000

    attempt = "nn_coco_pretrained_v1_20260930"
    out_dir = root / "manifests/images" / attempt
    out_dir.mkdir(parents=True, exist_ok=True)

    for cond in ["clean"] + list(CONDS):
        if cond == "clean":
            src = root / "manifests/images/coco_val2017_clean_p0_v1.json"
            suffix = "clean"
        elif cond == "codec_control_q95":
            src = root / "manifests/images/coco_val2017_codec_control_q95_p0_v1.json"
            suffix = "codec_control_q95"
        else:
            src = root / f"manifests/images/coco_val2017_full_{cond}.json"
            suffix = cond
        full = json.loads(src.read_text())
        records = [r for r in full["records"] if int(r["image_id"]) in keep]
        assert len(records) == 2000, (cond, len(records))
        cache_base = {"clean": "data/datasets/coco_pilot_v1_20260923/images/val2017",
                      "codec_control_q95": "data/codec_control/coco"}.get(cond, "data/coco_c")
        for r in records:
            cached = root / cache_base / r["output_relpath"]
            if not cached.is_file():
                raise SystemExit(f"missing cache bytes: {cached}")
        doc = {
            "schema_version": 1,
            "dataset": "coco", "split": "val2017_paired2000",
            "created_at_utc": datetime.now(timezone.utc).isoformat(),
            "attempt": attempt,
            "parent_manifest": str(src),
            "parent_manifest_sha256": full.get("manifest_sha256"),
            "subset": "coco_val2017_paired_2000_s20260924",
            "expected_image_ids": [int(r["image_id"]) for r in records],
            "records": records,
        }
        write_manifest(doc, out_dir / f"coco_paired2000_{suffix}.json")
        print("WROTE", suffix, len(records))

    # eval annotations: real COCO categories restricted to subset
    ann = json.loads((root / "data/datasets/coco_pilot_v1_20260923/annotations"
                      "/instances_val2017.json").read_text())
    keep_imgs = [i for i in ann["images"] if int(i["id"]) in keep]
    keep_anns = [a for a in ann["annotations"] if int(a["image_id"]) in keep]
    eval_doc = {k: ann[k] for k in ("info", "licenses", "categories")}
    eval_doc["images"] = keep_imgs
    eval_doc["annotations"] = keep_anns
    out_ann = root / "manifests/annotations/coco_val2017_paired2000_eval.json"
    out_ann.write_text(json.dumps(eval_doc) + "\n", encoding="utf-8")
    print("EVAL ANN", len(keep_imgs), "images", len(keep_anns), "anns",
          "sha", sha256_file(out_ann))

    # index-mapping category files for raw torchvision label channels
    cat_dir = root / "manifests/annotations"
    retinanet_map = {"categories": [{"id": i + 1, "name": f"ch{i+1}"}
                                    for i in range(91)]}
    fcos_map = {"categories": [{"id": i, "name": f"ch{i}"}
                               for i in range(91)]}
    (cat_dir / "coco_torchvision_map_retinanet91.json").write_text(
        json.dumps(retinanet_map) + "\n")
    (cat_dir / "coco_torchvision_map_fcos91.json").write_text(
        json.dumps(fcos_map) + "\n")
    print("WROTE category maps")


if __name__ == "__main__":
    main()
