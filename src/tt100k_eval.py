#!/usr/bin/env python3
"""COCO-style TT100K evaluation with original-pixel bbox-height strata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval

from topic_c.manifest import sha256_file
from topic_c.tt100k_height import HEIGHT_BINS, accumulated_height_stats, height_evaluation


STAT_NAMES = ["AP", "AP50", "AP75", "AP_small", "AP_medium", "AP_large", "AR1", "AR10", "AR100", "AR_small", "AR_medium", "AR_large"]
def coco_from_dataset(dataset: dict) -> COCO:
    result = COCO()
    result.dataset = dataset
    result.createIndex()
    return result


def evaluate(ground_truth: dict, predictions: list[dict], image_ids: list[int]) -> dict[str, float]:
    gt = coco_from_dataset(ground_truth)
    dt = gt.loadRes(predictions)
    evaluation = COCOeval(gt, dt, iouType="bbox")
    evaluation.params.imgIds = image_ids
    evaluation.evaluate()
    evaluation.accumulate()
    evaluation.summarize()
    return {name: float(value) for name, value in zip(STAT_NAMES, evaluation.stats)}


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
        raise SystemExit(f"TT100K EVALUATION REFUSED: metric output already exists: {output}")
    annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    input_record = json.loads(Path(args.input_record).read_text(encoding="utf-8"))
    run = json.loads(Path(args.run_record).read_text(encoding="utf-8"))
    if run.get("dataset") != "tt100k" or run.get("split") != "test":
        raise SystemExit("TT100K EVALUATION REFUSED: run record is not a TT100K test condition")
    if sha256_file(args.predictions) != run.get("prediction_sha256"):
        raise SystemExit("TT100K EVALUATION REFUSED: prediction hash mismatch")
    if sha256_file(args.annotations) != run.get("annotation_sha256"):
        raise SystemExit("TT100K EVALUATION REFUSED: annotation hash mismatch")
    image_ids = input_record.get("image_ids")
    if not isinstance(image_ids, list) or not image_ids or len(image_ids) != len(set(image_ids)):
        raise SystemExit("TT100K EVALUATION REFUSED: input IDs are invalid")
    strata = accumulated_height_stats(height_evaluation(annotations, predictions, image_ids, HEIGHT_BINS))
    result = {
        "schema_version": 1, "condition_id": run["condition_id"], "model": run["model"],
        "dataset": run["dataset"], "split": run["split"], "precision": run["precision"], "corruption": run["corruption"], "severity": run["severity"],
        "n_images": len(image_ids), "stats": evaluate(annotations, predictions, image_ids), "height_bins_px":
            [{"name": name, "min": lower, "max": None if upper == float("inf") else upper} for name, lower, upper in HEIGHT_BINS],
        "height_strata_stats": strata, "prediction_sha256": run["prediction_sha256"],
        "input_manifest_sha256": run["input_manifest_sha256"], "input_image_ids_sha256": input_record["image_ids_sha256"],
        "run_record_sha256": sha256_file(args.run_record),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(f"TT100K METRIC VALID -> {output}")


if __name__ == "__main__":
    main()
