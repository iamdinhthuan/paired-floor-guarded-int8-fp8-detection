#!/usr/bin/env python3
"""Official COCOeval with metric provenance embedded in every output."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from topic_c.manifest import sha256_file

STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--input-record", required=True)
    parser.add_argument("--run-record", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit(f"refusing to overwrite metric: {output}")
    record = json.loads(Path(args.input_record).read_text(encoding="utf-8"))
    run = json.loads(Path(args.run_record).read_text(encoding="utf-8"))
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    if sha256_file(args.predictions) != run.get("prediction_sha256"):
        raise SystemExit("prediction SHA-256 disagrees with run record")
    coco_gt = COCO(args.annotations)
    coco_dt = coco_gt.loadRes(predictions)
    evaluation = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluation.params.imgIds = record["image_ids"]
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    result = {"schema_version": 1, "condition_id": run["condition_id"], "dataset": run["dataset"], "split": run["split"], "model": run["model"], "precision": run["precision"], "corruption": run["corruption"], "severity": run["severity"], "n_images": len(record["image_ids"]), "stats": {name: float(value) for name, value in zip(STAT_NAMES, evaluation.stats)}, "prediction_sha256": run["prediction_sha256"], "input_manifest_sha256": run["input_manifest_sha256"], "input_image_ids_sha256": record["image_ids_sha256"], "run_record_sha256": sha256_file(args.run_record)}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"METRIC VALID -> {output}")


if __name__ == "__main__":
    main()
