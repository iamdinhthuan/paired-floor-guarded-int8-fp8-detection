#!/usr/bin/env python3
"""Evaluate an existing hash-bound prediction set on a frozen image subset.

Produces a metric record with explicit subset provenance: the parent run
record, the parent prediction bytes, and the frozen selection record are
all cited by SHA-256.  No new inference is performed; subset AP is a
deterministic restriction of an existing prediction file.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ModuleNotFoundError:  # Guards remain importable without the evaluator extra.
    COCO = None
    COCOeval = None

from topic_c.manifest import sha256_file

STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]


def load_marked(path: Path, field: str) -> dict:
    marker = Path(str(path) + ".complete")
    document = json.loads(path.read_text(encoding="utf-8"))
    if not marker.is_file() or marker.read_text().strip() != document.get(field):
        raise SystemExit(f"SUBSET EVAL REFUSED: completion marker invalid: {path}")
    return document


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--input-record", required=True)
    parser.add_argument("--run-record", required=True)
    parser.add_argument("--subset", required=True, help="frozen selection record from build_paired_subset.py")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit(f"SUBSET EVAL REFUSED: output exists: {output}")
    subset = load_marked(Path(args.subset), "selection_sha256")
    subset_ids = [int(v) for v in subset["image_ids"]]
    run = json.loads(Path(args.run_record).read_text(encoding="utf-8"))
    inputs = json.loads(Path(args.input_record).read_text(encoding="utf-8"))
    parent_ids = [int(v) for v in inputs["image_ids"]]
    if set(subset_ids) - set(parent_ids):
        raise SystemExit("SUBSET EVAL REFUSED: subset escapes parent image universe")
    if sha256_file(args.predictions) != run.get("prediction_sha256"):
        raise SystemExit("SUBSET EVAL REFUSED: prediction SHA-256 disagrees with run record")
    if sha256_file(args.annotations) != run.get("annotation_sha256"):
        raise SystemExit("SUBSET EVAL REFUSED: annotation SHA-256 disagrees with run record")
    if COCO is None or COCOeval is None:
        raise SystemExit("SUBSET EVAL REFUSED: pycocotools is required")
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    coco_gt = COCO(args.annotations)
    coco_dt = coco_gt.loadRes(predictions)
    evaluation = COCOeval(coco_gt, coco_dt, iouType="bbox")
    evaluation.params.imgIds = subset_ids
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    result = {
        "schema_version": 1,
        "condition_id": run["condition_id"] + "__subset-" + subset["selection_sha256"][:8],
        "dataset": run["dataset"], "split": run["split"], "model": run["model"],
        "precision": run["precision"], "corruption": run["corruption"],
        "severity": run["severity"], "n_images": len(subset_ids),
        "stats": {name: float(value) for name, value in zip(STAT_NAMES, evaluation.stats)},
        "prediction_sha256": run["prediction_sha256"],
        "input_manifest_sha256": run["input_manifest_sha256"],
        "input_image_ids_sha256": hashlib.sha256(
            json.dumps(subset_ids, separators=(",", ":")).encode()).hexdigest(),
        "run_record_sha256": sha256_file(args.run_record),
        "subset_provenance": {
            "kind": "deterministic_subset_restriction_of_existing_predictions",
            "selection_sha256": subset["selection_sha256"],
            "parent_condition_id": run["condition_id"],
            "parent_n_images": len(parent_ids),
        },
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"SUBSET METRIC VALID -> {output}")


if __name__ == "__main__":
    main()
