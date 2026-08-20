#!/usr/bin/env python3
"""Evaluate a complete frozen 117-run VOC, KITTI, or TT100K pilot attempt."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from execute_dataset_pilot import load_plan, output_paths
from topic_c.manifest import sha256_file


def cpu_environment() -> dict[str, str]:
    result = dict(os.environ)
    result.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    return result


def metric_path(root: Path, attempt: str, condition_id: str) -> Path:
    return root / "outputs" / "metrics" / attempt / f"{condition_id}.json"


def valid_metric(run: dict, run_record: Path, metric: Path, expected_images: int) -> None:
    data, record = json.loads(metric.read_text(encoding="utf-8")), json.loads(run_record.read_text(encoding="utf-8"))
    fields = ("condition_id", "dataset", "split", "model", "precision", "corruption", "severity", "prediction_sha256", "input_manifest_sha256")
    if (any(data.get(field) != (record.get(field) if field == "prediction_sha256" else run.get(field)) for field in fields)
            or data.get("run_record_sha256") != sha256_file(run_record) or data.get("n_images") != expected_images):
        raise SystemExit(f"PILOT EVALUATION REFUSED: metric provenance mismatch: {metric}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True); parser.add_argument("--project-root", required=True)
    parser.add_argument("--dataset", choices=("coco", "voc", "kitti", "tt100k"), required=True); parser.add_argument("--split", required=True)
    parser.add_argument("--annotations", required=True); parser.add_argument("--attempt", required=True); parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--execute", action="store_true"); parser.add_argument("--resume-verified", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    plan = load_plan(Path(args.plan), args.dataset, args.split)
    receipt = root / "manifests" / "approvals" / f"{args.attempt}.json"
    if not receipt.is_file() or json.loads(receipt.read_text(encoding="utf-8")).get("plan_sha256") != plan["plan_sha256"]:
        raise SystemExit("PILOT EVALUATION REFUSED: matching approval receipt is required")
    pending, completed = [], []
    for run in plan["runs"]:
        prediction, input_record, run_record = output_paths(root, args.attempt, run["condition_id"])
        if not all(path.is_file() for path in (prediction, input_record, run_record)):
            raise SystemExit(f"PILOT EVALUATION REFUSED: incomplete inference output: {run['condition_id']}")
        record, inputs = json.loads(run_record.read_text(encoding="utf-8")), json.loads(input_record.read_text(encoding="utf-8"))
        if record.get("prediction_sha256") != sha256_file(prediction) or record.get("n_images") != args.expected_images or len(inputs.get("image_ids", [])) != args.expected_images:
            raise SystemExit(f"PILOT EVALUATION REFUSED: invalid inference provenance: {run['condition_id']}")
        metric = metric_path(root, args.attempt, run["condition_id"])
        if metric.exists():
            if not args.resume_verified:
                raise SystemExit(f"PILOT EVALUATION REFUSED: metric already exists: {metric}")
            valid_metric(run, run_record, metric, args.expected_images); completed.append(run)
        else:
            pending.append(run)
    if not args.execute:
        print(json.dumps({"dry_run": True, "runs": 117, "completed": len(completed), "pending": len(pending)}, indent=2)); return
    evaluator = root / "src" / ("tt100k_eval.py" if args.dataset == "tt100k" else "coco_eval.py")
    for ordinal, run in enumerate(pending, start=1):
        prediction, input_record, run_record = output_paths(root, args.attempt, run["condition_id"])
        subprocess.run([sys.executable, str(root / "src" / "validate_predictions.py"), "--predictions", str(prediction),
                        "--input-record", str(input_record), "--annotations", args.annotations], check=True)
        print(f"DATASET COCOEVAL {ordinal}/{len(pending)} {run['condition_id']}", flush=True)
        subprocess.run(["taskset", "-c", "0", sys.executable, str(evaluator), "--annotations", args.annotations,
                        "--predictions", str(prediction), "--input-record", str(input_record), "--run-record", str(run_record),
                        "--out", str(metric_path(root, args.attempt, run["condition_id"]))], check=True, env=cpu_environment())
    records = [output_paths(root, args.attempt, run["condition_id"])[2] for run in plan["runs"]]
    subprocess.run([sys.executable, str(root / "src" / "validate_run_registry.py"),
                    *[item for record in records for item in ("--run-record", str(record))]], check=True)
    report = root / "outputs" / "reports" / f"{args.attempt}_evaluation_complete.json"
    if report.exists():
        raise SystemExit(f"PILOT EVALUATION REFUSED: report already exists: {report}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "dataset": args.dataset, "split": args.split, "attempt": args.attempt, "runs": 117,
                                  "plan_sha256": plan["plan_sha256"], "approval_receipt_sha256": sha256_file(receipt),
                                  "metric_sha256": {run["condition_id"]: sha256_file(metric_path(root, args.attempt, run["condition_id"])) for run in plan["runs"]}}, indent=2) + "\n", encoding="utf-8")
    print(f"DATASET PILOT EVALUATION COMPLETE dataset={args.dataset} runs=117")


if __name__ == "__main__":
    main()
