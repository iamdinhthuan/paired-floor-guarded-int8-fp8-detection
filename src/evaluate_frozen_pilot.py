#!/usr/bin/env python3
"""Validate and evaluate every completed condition in one frozen pilot attempt."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from execute_frozen_pilot import paths
from pilot_registry import canonical_hash
from topic_c.manifest import sha256_file


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != canonical_hash(plan, "plan_sha256") or len(plan.get("runs", [])) != 117:
        raise SystemExit("PILOT EVALUATION REFUSED: frozen plan is invalid")
    return plan


def metric_path(root: Path, attempt: str, condition_id: str) -> Path:
    return root / "outputs" / "metrics" / attempt / f"{condition_id}.json"


def validate_run(run: dict, prediction: Path, input_record: Path, run_record: Path) -> None:
    record = json.loads(run_record.read_text(encoding="utf-8"))
    input_data = json.loads(input_record.read_text(encoding="utf-8"))
    required = ("condition_id", "model", "precision", "calibrator", "corruption", "severity", "engine_sha256", "input_manifest_sha256")
    if any(record.get(field) != run.get(field) for field in required):
        raise SystemExit(f"PILOT EVALUATION REFUSED: run-record disagrees with frozen plan: {run['condition_id']}")
    if input_data.get("condition_id") != run["condition_id"] or input_data.get("input_manifest_sha256") != run["input_manifest_sha256"]:
        raise SystemExit(f"PILOT EVALUATION REFUSED: input-record disagrees with frozen plan: {run['condition_id']}")
    if record.get("prediction_sha256") != sha256_file(prediction):
        raise SystemExit(f"PILOT EVALUATION REFUSED: prediction hash mismatch: {run['condition_id']}")
    if record.get("n_images") != 5000 or len(input_data.get("image_ids", [])) != 5000:
        raise SystemExit(f"PILOT EVALUATION REFUSED: non-full input set: {run['condition_id']}")


def validate_metric(run: dict, run_record: Path, metric_path: Path) -> None:
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    record = json.loads(run_record.read_text(encoding="utf-8"))
    if (metric.get("condition_id") != run["condition_id"] or metric.get("prediction_sha256") != record["prediction_sha256"]
            or metric.get("input_manifest_sha256") != run["input_manifest_sha256"]
            or metric.get("run_record_sha256") != sha256_file(run_record)):
        raise SystemExit(f"PILOT EVALUATION REFUSED: existing metric provenance mismatch: {run['condition_id']}")


def serial_cpu_environment() -> dict[str, str]:
    environment = dict(os.environ)
    environment.update({"OPENBLAS_NUM_THREADS": "1", "OMP_NUM_THREADS": "1", "MKL_NUM_THREADS": "1", "NUMEXPR_NUM_THREADS": "1"})
    return environment


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--attempt", default="pilot_117_v1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-verified", action="store_true", help="skip only existing metrics that match frozen provenance")
    args = parser.parse_args()
    root, annotations = Path(args.project_root).resolve(), Path(args.annotations).resolve()
    plan = load_plan(Path(args.plan))
    receipt = root / "manifests" / "approvals" / f"{args.attempt}.json"
    receipt_data = json.loads(receipt.read_text(encoding="utf-8")) if receipt.is_file() else None
    if receipt_data and receipt_data.get("plan_sha256") != plan["plan_sha256"]:
        raise SystemExit("PILOT EVALUATION REFUSED: approval receipt cites another plan")
    missing = []
    for run in plan["runs"]:
        prediction, input_record, run_record = paths(root, args.attempt, run["condition_id"])
        if not all(path.is_file() for path in (prediction, input_record, run_record)):
            missing.append(run["condition_id"])
    if not args.execute:
        print(json.dumps({"dry_run": True, "attempt": args.attempt, "runs": len(plan["runs"]),
                          "approval_receipt_present": bool(receipt_data), "missing_inference_outputs": missing}, indent=2))
        return
    if not receipt_data:
        raise SystemExit("PILOT EVALUATION REFUSED: no human-review approval receipt")
    if missing:
        raise SystemExit(f"PILOT EVALUATION REFUSED: {len(missing)} conditions lack complete inference outputs")
    completed, pending = [], []
    for run in plan["runs"]:
        prediction, input_record, run_record = paths(root, args.attempt, run["condition_id"])
        validate_run(run, prediction, input_record, run_record)
        metric = metric_path(root, args.attempt, run["condition_id"])
        if metric.exists():
            if not args.resume_verified:
                raise SystemExit(f"PILOT EVALUATION REFUSED: metric output already exists: {metric}")
            validate_metric(run, run_record, metric)
            completed.append(run)
        else:
            pending.append(run)
    run_records = []
    serial_environment = serial_cpu_environment()
    for index, run in enumerate(pending, start=1):
        prediction, input_record, run_record = paths(root, args.attempt, run["condition_id"])
        subprocess.run([sys.executable, str(root / "src" / "validate_predictions.py"), "--predictions", str(prediction),
                        "--input-record", str(input_record), "--annotations", str(annotations)], check=True)
        print(f"COCOEVAL RESUME {index}/{len(pending)} (completed={len(completed)}) {run['condition_id']}", flush=True)
        subprocess.run(["taskset", "-c", "0", sys.executable, str(root / "src" / "coco_eval.py"), "--annotations", str(annotations),
                        "--predictions", str(prediction), "--input-record", str(input_record), "--run-record", str(run_record),
                        "--out", str(metric_path(root, args.attempt, run["condition_id"]))], check=True, env=serial_environment)
    run_records = [str(paths(root, args.attempt, run["condition_id"])[2]) for run in plan["runs"]]
    subprocess.run([sys.executable, str(root / "src" / "validate_run_registry.py"),
                    *[argument for run_record in run_records for argument in ("--run-record", run_record)]], check=True)
    report = root / "outputs" / "reports" / f"{args.attempt}_evaluation_complete.json"
    if report.exists():
        raise SystemExit(f"PILOT EVALUATION REFUSED: completion report already exists: {report}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "attempt": args.attempt, "plan_sha256": plan["plan_sha256"], "runs": len(plan["runs"]),
                                  "approval_receipt_sha256": sha256_file(receipt),
                                  "metric_sha256": {run["condition_id"]: sha256_file(metric_path(root, args.attempt, run["condition_id"])) for run in plan["runs"]}}, indent=2) + "\n", encoding="utf-8")
    print(f"PILOT EVALUATION COMPLETE runs={len(plan['runs'])} attempt={args.attempt}")


if __name__ == "__main__":
    main()
