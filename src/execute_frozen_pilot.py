#!/usr/bin/env python3
"""Fail-closed launcher for the reviewed, frozen 117-run COCO pilot.

It intentionally defaults to a non-mutating dry run.  Execution requires a
human-review acknowledgement and creates an immutable approval receipt before
the first inference.  Each condition is refused if any of its three outputs
already exists, preventing accidental mixing of attempts.
"""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pilot_registry import canonical_hash
from topic_c.manifest import read_manifest, sha256_file


def refuse_existing(paths: list[Path]) -> None:
    existing = [str(path) for path in paths if path.exists()]
    if existing:
        raise SystemExit("PILOT EXECUTION REFUSED: outputs already exist:\n" + "\n".join(existing[:20]))


def ready_runs(plan: dict, root: Path, calibration_list: Path) -> list[dict]:
    if plan.get("plan_sha256") != canonical_hash(plan, "plan_sha256"):
        raise SystemExit("PILOT EXECUTION REFUSED: frozen plan hash mismatch")
    if len(plan.get("runs", [])) != 117:
        raise SystemExit("PILOT EXECUTION REFUSED: frozen plan must contain exactly 117 runs")
    if not calibration_list.is_file():
        raise SystemExit("PILOT EXECUTION REFUSED: calibration list is missing")
    actual_calibration_hash = sha256_file(calibration_list)
    seen = set()
    for run in plan["runs"]:
        condition_id = run.get("condition_id")
        if not condition_id or condition_id in seen:
            raise SystemExit("PILOT EXECUTION REFUSED: condition IDs must be present and unique")
        seen.add(condition_id)
        if sha256_file(run["engine_path"]) != run["engine_sha256"]:
            raise SystemExit(f"PILOT EXECUTION REFUSED: engine hash mismatch: {condition_id}")
        manifest_path = root / run["input_manifest_path"]
        manifest = read_manifest(manifest_path)
        marker = manifest_path.with_suffix(manifest_path.suffix + ".complete")
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != manifest["manifest_sha256"]:
            raise SystemExit(f"PILOT EXECUTION REFUSED: manifest lacks matching marker: {condition_id}")
        if run.get("input_manifest_sha256") != manifest["manifest_sha256"]:
            raise SystemExit(f"PILOT EXECUTION REFUSED: frozen manifest hash mismatch: {condition_id}")
        expected_id = run["condition_id_template"].replace("{manifest8}", manifest["manifest_sha256"][:8])
        if condition_id != expected_id:
            raise SystemExit(f"PILOT EXECUTION REFUSED: frozen condition ID mismatch: {condition_id}")
        if run["precision"] == "int8-entropy" and run.get("calibration_sha256") != actual_calibration_hash:
            raise SystemExit(f"PILOT EXECUTION REFUSED: calibration-list hash mismatch: {condition_id}")
    return plan["runs"]


def paths(root: Path, attempt: str, condition_id: str) -> tuple[Path, Path, Path]:
    return (
        root / "outputs" / "predictions" / attempt / f"{condition_id}.json",
        root / "outputs" / "inputs" / attempt / f"{condition_id}.json",
        root / "manifests" / "runs" / attempt / f"{condition_id}.json",
    )


def verify_completed_run(run: dict, output_paths: tuple[Path, Path, Path]) -> bool:
    prediction, input_record, run_record = output_paths
    if not all(path.is_file() for path in output_paths):
        return False
    record = json.loads(run_record.read_text(encoding="utf-8"))
    inputs = json.loads(input_record.read_text(encoding="utf-8"))
    fields = ("condition_id", "model", "precision", "calibrator", "corruption", "severity", "engine_sha256", "input_manifest_sha256")
    if any(record.get(field) != run.get(field) for field in fields):
        raise SystemExit(f"PILOT RESUME REFUSED: run-record differs from frozen plan: {run['condition_id']}")
    if inputs.get("condition_id") != run["condition_id"] or inputs.get("input_manifest_sha256") != run["input_manifest_sha256"]:
        raise SystemExit(f"PILOT RESUME REFUSED: input-record differs from frozen plan: {run['condition_id']}")
    if record.get("prediction_sha256") != sha256_file(prediction):
        raise SystemExit(f"PILOT RESUME REFUSED: prediction hash mismatch: {run['condition_id']}")
    if record.get("n_images") != 5000 or len(inputs.get("image_ids", [])) != 5000:
        raise SystemExit(f"PILOT RESUME REFUSED: incomplete evaluated ID set: {run['condition_id']}")
    return True


def command(root: Path, annotations: Path, calibration_list: Path, attempt: str, run: dict) -> list[str]:
    prediction, input_record, run_record = paths(root, attempt, run["condition_id"])
    cache_root = (Path("/home/thuan/coco_journal/data/coco/images/val2017") if run["corruption"] == "clean"
                  else root / "data" / "coco_c")
    cmd = [
        sys.executable, str(root / "src" / "coco_infer_trt.py"),
        "--engine", run["engine_path"], "--annotations", str(annotations),
        "--image-manifest", str(root / run["input_manifest_path"]),
        "--manifest-cache-root", str(cache_root),
        "--out", str(prediction), "--input-record", str(input_record), "--run-record", str(run_record),
        "--condition-id", run["condition_id"], "--model", run["model"],
        "--precision", run["precision"], "--calibrator", run["calibrator"],
        "--corruption", run["corruption"], "--severity", str(run["severity"]),
    ]
    if run["precision"] == "int8-entropy":
        cmd += ["--calibration-list", str(calibration_list), "--calibration-provenance", "verified"]
    return cmd


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--calibration-list", required=True)
    parser.add_argument("--attempt", default="pilot_117_v1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-verified", action="store_true", help="skip only output triples that fully match the frozen plan")
    parser.add_argument("--reviewed-by", help="required with --execute; stored in the approval receipt")
    args = parser.parse_args()
    root, annotations, calibration_list = Path(args.project_root).resolve(), Path(args.annotations).resolve(), Path(args.calibration_list).resolve()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    runs = ready_runs(plan, root, calibration_list)
    completed, pending = [], []
    for run in runs:
        output_paths = paths(root, args.attempt, run["condition_id"])
        present = [path.exists() for path in output_paths]
        if any(present) and not all(present):
            raise SystemExit(f"PILOT EXECUTION REFUSED: partial output triple: {run['condition_id']}")
        if all(present):
            if not args.resume_verified:
                refuse_existing(list(output_paths))
            verify_completed_run(run, output_paths)
            completed.append(run)
        else:
            pending.append(run)
    if not args.execute:
        print(json.dumps({"dry_run": True, "runs": len(runs), "completed": len(completed), "pending": len(pending), "attempt": args.attempt,
                          "plan_sha256": plan["plan_sha256"], "first_command": command(root, annotations, calibration_list, args.attempt, runs[0])}, indent=2))
        return
    if not args.reviewed_by or not args.reviewed_by.strip():
        raise SystemExit("PILOT EXECUTION REFUSED: --reviewed-by is required after human review")
    receipt = root / "manifests" / "approvals" / f"{args.attempt}.json"
    if receipt.exists():
        old_receipt = json.loads(receipt.read_text(encoding="utf-8"))
        if not args.resume_verified or old_receipt.get("plan_sha256") != plan["plan_sha256"] or old_receipt.get("attempt") != args.attempt:
            raise SystemExit("PILOT EXECUTION REFUSED: approval receipt already exists and cannot be reused")
    else:
        receipt.parent.mkdir(parents=True, exist_ok=True)
        receipt.write_text(json.dumps({"schema_version": 1, "approved_at_utc": datetime.now(timezone.utc).isoformat(),
                                       "reviewed_by": args.reviewed_by, "plan_sha256": plan["plan_sha256"],
                                       "runs": len(runs), "attempt": args.attempt}, indent=2) + "\n", encoding="utf-8")
    for index, run in enumerate(pending, start=1):
        print(f"PILOT RESUME {index}/{len(pending)} (completed={len(completed)}) {run['condition_id']}", flush=True)
        subprocess.run(command(root, annotations, calibration_list, args.attempt, run), check=True)
        prediction, input_record, _ = paths(root, args.attempt, run["condition_id"])
        subprocess.run([sys.executable, str(root / "src" / "validate_predictions.py"), "--predictions", str(prediction),
                        "--input-record", str(input_record), "--annotations", str(annotations)], check=True)
    print(f"PILOT INFERENCE COMPLETE runs={len(runs)} attempt={args.attempt}")


if __name__ == "__main__":
    main()
