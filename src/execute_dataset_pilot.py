#!/usr/bin/env python3
"""Fail-closed transfer pilot executor with per-model FP16 parity admission."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pilot_registry import calibration_sha256, canonical_hash
from topic_c.manifest import read_manifest, sha256_file


def output_paths(root: Path, attempt: str, condition_id: str) -> tuple[Path, Path, Path]:
    return (root / "outputs" / "predictions" / attempt / f"{condition_id}.json",
            root / "outputs" / "inputs" / attempt / f"{condition_id}.json",
            root / "manifests" / "runs" / attempt / f"{condition_id}.json")


def load_plan(path: Path, dataset: str, split: str) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != canonical_hash(plan, "plan_sha256") or len(plan.get("runs", [])) != 117:
        raise SystemExit("PILOT EXECUTION REFUSED: invalid frozen plan")
    top_dataset, top_split = plan.get("dataset"), plan.get("split")
    if ((top_dataset is not None and top_dataset != dataset)
            or (top_split is not None and top_split != split)
            or any(run.get("dataset") != dataset or run.get("split") != split for run in plan["runs"])):
        raise SystemExit("PILOT EXECUTION REFUSED: dataset/split plan mismatch")
    return plan


def plan_calibration_sha256(plan: dict) -> str:
    calibrated = [
        run for run in plan["runs"] if run.get("precision") in {"int8-entropy", "fp8"}
    ]
    digests = {
        run.get("calibration_sha256")
        for run in calibrated
    }
    formats = {run.get("precision") for run in calibrated}
    if len(digests) != 1 or None in digests or formats != {"int8-entropy", "fp8"}:
        raise SystemExit("PILOT EXECUTION REFUSED: inconsistent INT8/FP8 calibration provenance")
    return digests.pop()


def validate_parity(paths: list[Path], dataset: str) -> None:
    if len(paths) != 3:
        raise SystemExit("PILOT EXECUTION REFUSED: exactly three FP16 parity reports are required")
    models = set()
    for path in paths:
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("pass"):
            raise SystemExit(f"PILOT EXECUTION REFUSED: FP16 parity failed: {path}")
        artifacts = report.get("artifacts")
        if isinstance(artifacts, dict):
            marker = path.with_suffix(path.suffix + ".complete")
            if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != sha256_file(path):
                raise SystemExit(f"PILOT EXECUTION REFUSED: incomplete four-backend parity report: {path}")
            if set(artifacts) != {"pytorch", "onnxruntime", "trt-fp32", "trt-fp16"}:
                raise SystemExit(f"PILOT EXECUTION REFUSED: incomplete four-backend parity grid: {path}")
            loaded = {}
            for label, item in artifacts.items():
                if not isinstance(item, dict):
                    raise SystemExit(f"PILOT EXECUTION REFUSED: malformed parity artifact: {path}")
                metric_path, run_path = Path(item.get("metric", "")), Path(item.get("run_record", ""))
                if (not metric_path.is_file() or not run_path.is_file()
                        or sha256_file(metric_path) != item.get("metric_sha256")
                        or sha256_file(run_path) != item.get("run_record_sha256")):
                    raise SystemExit(f"PILOT EXECUTION REFUSED: parity artifact hash mismatch: {path}/{label}")
                metric, run = (json.loads(value.read_text(encoding="utf-8")) for value in (metric_path, run_path))
                if metric.get("run_record_sha256") != sha256_file(run_path):
                    raise SystemExit(f"PILOT EXECUTION REFUSED: parity metric/run mismatch: {path}/{label}")
                loaded[label] = (metric, run)
            fp32, fp16 = loaded["trt-fp32"][0], loaded["trt-fp16"][0]
            for label in ("pytorch", "onnxruntime"):
                metric, run = loaded[label]
                if (metric.get("dataset") != dataset or run.get("dataset") != dataset
                        or metric.get("model") != report.get("model") or run.get("model") != report.get("model")
                        or run.get("backend") != label or run.get("precision") != "fp32-reference"):
                    raise SystemExit(f"PILOT EXECUTION REFUSED: invalid source parity provenance: {path}/{label}")
        else:
            fp32 = json.loads(Path(report["fp32_metric"]).read_text(encoding="utf-8"))
            fp16 = json.loads(Path(report["fp16_metric"]).read_text(encoding="utf-8"))
        if fp32.get("dataset") != dataset or fp32.get("model") != fp16.get("model") or fp32.get("precision") != "fp32" or fp16.get("precision") != "fp16":
            raise SystemExit(f"PILOT EXECUTION REFUSED: invalid FP16 parity provenance: {path}")
        models.add(fp32["model"])
    if models != {"yolo11n", "yolo11m", "yolo11x"}:
        raise SystemExit("PILOT EXECUTION REFUSED: parity reports do not cover YOLO11 n/m/x")


def command(root: Path, args: argparse.Namespace, run: dict) -> list[str]:
    prediction, input_record, run_record = output_paths(root, args.attempt, run["condition_id"])
    cache_root = args.clean_root if run["corruption"] == "clean" else args.corruption_root
    command = [sys.executable, str(root / "src" / "coco_infer_trt.py"), "--engine", run["engine_path"],
               "--annotations", args.annotations, "--image-manifest", str(root / run["input_manifest_path"]),
               "--manifest-cache-root", cache_root, "--out", str(prediction), "--input-record", str(input_record),
               "--run-record", str(run_record), "--condition-id", run["condition_id"], "--dataset", args.dataset,
               "--split", args.split, "--imgsz", str(args.imgsz), "--model", run["model"],
               "--precision", run["precision"], "--calibrator", run["calibrator"], "--corruption", run["corruption"],
               "--severity", str(run["severity"])]
    if args.class_map:
        command += ["--class-map", args.class_map]
    if run["precision"] in {"int8-entropy", "fp8"}:
        command += ["--calibration-list", args.calibration_list, "--calibration-method", "entropy",
                    "--calibration-provenance", "verified"]
    return command


def complete(run: dict, paths: tuple[Path, Path, Path], expected_images: int) -> bool:
    prediction, input_record, run_record = paths
    if not all(path.is_file() for path in paths):
        return False
    record, inputs = json.loads(run_record.read_text(encoding="utf-8")), json.loads(input_record.read_text(encoding="utf-8"))
    fields = ("condition_id", "dataset", "split", "model", "precision", "calibrator", "corruption", "severity", "engine_sha256", "input_manifest_sha256")
    if (any(record.get(field) != run.get(field) for field in fields) or inputs.get("condition_id") != run["condition_id"]
            or inputs.get("input_manifest_sha256") != run.get("input_manifest_sha256")):
        raise SystemExit(f"PILOT RESUME REFUSED: provenance mismatch: {run['condition_id']}")
    if record.get("prediction_sha256") != sha256_file(prediction) or len(inputs.get("image_ids", [])) != expected_images or record.get("n_images") != expected_images:
        raise SystemExit(f"PILOT RESUME REFUSED: malformed output: {run['condition_id']}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True); parser.add_argument("--project-root", required=True)
    parser.add_argument("--dataset", choices=("coco", "voc", "kitti", "tt100k"), required=True); parser.add_argument("--split", required=True)
    parser.add_argument("--annotations", required=True); parser.add_argument("--class-map")
    parser.add_argument("--clean-root", required=True); parser.add_argument("--corruption-root", required=True)
    parser.add_argument("--calibration-list", required=True); parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--expected-images", type=int, required=True); parser.add_argument("--fp16-parity", action="append", default=[])
    parser.add_argument("--attempt", required=True); parser.add_argument("--execute", action="store_true"); parser.add_argument("--resume-verified", action="store_true")
    parser.add_argument("--reviewed-by")
    args = parser.parse_args()
    root = Path(args.project_root).resolve(); plan = load_plan(Path(args.plan), args.dataset, args.split)
    try:
        actual_calibration_sha = calibration_sha256(args.calibration_list)
    except ValueError as exc:
        raise SystemExit(f"PILOT EXECUTION REFUSED: {exc}") from exc
    if actual_calibration_sha != plan_calibration_sha256(plan):
        raise SystemExit("PILOT EXECUTION REFUSED: calibration list hash mismatch")
    validate_parity([Path(path) for path in args.fp16_parity], args.dataset)
    pending, completed = [], []
    for run in plan["runs"]:
        manifest_path = root / run["input_manifest_path"]; manifest = read_manifest(manifest_path)
        marker = manifest_path.with_suffix(manifest_path.suffix + ".complete")
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != manifest.get("manifest_sha256") or manifest.get("manifest_sha256") != run.get("input_manifest_sha256"):
            raise SystemExit(f"PILOT EXECUTION REFUSED: unvalidated manifest: {run['condition_id']}")
        state = output_paths(root, args.attempt, run["condition_id"])
        existing = [path.exists() for path in state]
        if any(existing) and not all(existing):
            raise SystemExit(f"PILOT EXECUTION REFUSED: partial output triple: {run['condition_id']}")
        if all(existing):
            if not args.resume_verified:
                raise SystemExit(f"PILOT EXECUTION REFUSED: existing output triple: {run['condition_id']}")
            complete(run, state, args.expected_images); completed.append(run)
        else:
            pending.append(run)
    if not args.execute:
        print(json.dumps({"dry_run": True, "runs": 117, "completed": len(completed), "pending": len(pending), "first_command": command(root, args, plan["runs"][0])}, indent=2)); return
    if not args.reviewed_by:
        raise SystemExit("PILOT EXECUTION REFUSED: explicit human review acknowledgement is required")
    receipt = root / "manifests" / "approvals" / f"{args.attempt}.json"
    if receipt.exists():
        old = json.loads(receipt.read_text(encoding="utf-8"))
        if old.get("plan_sha256") != plan["plan_sha256"] or not args.resume_verified:
            raise SystemExit("PILOT EXECUTION REFUSED: approval receipt cannot be reused")
    else:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"schema_version": 1, "approved_at_utc": datetime.now(timezone.utc).isoformat(), "reviewed_by": args.reviewed_by, "plan_sha256": plan["plan_sha256"], "attempt": args.attempt, "runs": 117}, indent=2) + "\n", encoding="utf-8")
    for ordinal, run in enumerate(pending, start=1):
        print(f"DATASET PILOT {ordinal}/{len(pending)} {run['condition_id']}", flush=True)
        subprocess.run(command(root, args, run), check=True)
        prediction, input_record, _ = output_paths(root, args.attempt, run["condition_id"])
        subprocess.run([sys.executable, str(root / "src" / "validate_predictions.py"), "--predictions", str(prediction), "--input-record", str(input_record), "--annotations", args.annotations], check=True)
    print(f"DATASET PILOT INFERENCE COMPLETE dataset={args.dataset} runs=117")


if __name__ == "__main__":
    main()
