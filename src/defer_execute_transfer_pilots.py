#!/usr/bin/env python3
"""Execute and evaluate the explicitly approved transfer pilots after all admission gates."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from topic_c.manifest import sha256_file


def canonical_hash(document: dict, excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def valid(path: Path, field: str) -> bool:
    if not path.is_file():
        return False
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return document.get(field) == canonical_hash(document, field)


def valid_evaluation(path: Path, plan_sha: str) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("plan_sha256") == plan_sha and data.get("runs") == 117 and len(data.get("metric_sha256", {})) == 117


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True); parser.add_argument("--wait-for-prep-report", required=True)
    parser.add_argument("--config", required=True); parser.add_argument("--report-out", required=True); parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root, prerequisite, config_path, report = (Path(args.project_root).resolve(), Path(args.wait_for_prep_report).resolve(),
                                                Path(args.config).resolve(), Path(args.report_out).resolve())
    if report.exists():
        raise SystemExit(f"TRANSFER EXECUTION REFUSED: report already exists: {report}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    datasets = config.get("datasets", [])
    if [item.get("dataset") for item in datasets] != ["voc", "kitti", "tt100k"]:
        raise SystemExit("TRANSFER EXECUTION REFUSED: unexpected frozen dataset sequence")
    while not valid(prerequisite, "transfer_prep_report_sha256"):
        print(f"TRANSFER EXECUTION waiting for hash-valid prep report: {prerequisite}", flush=True)
        time.sleep(args.poll_seconds)
    records = []
    for data in datasets:
        dataset, split = data["dataset"], data["split"]
        plan_path = root / "manifests" / "plans" / f"{dataset}_yolo11_nmx_117_frozen_v1.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if plan.get("plan_sha256") != canonical_hash(plan, "plan_sha256") or len(plan.get("runs", [])) != 117:
            raise SystemExit(f"TRANSFER EXECUTION REFUSED: frozen plan invalid: {plan_path}")
        attempt = f"{dataset}_pilot_117_v1"
        receipt = root / "manifests" / "approvals" / f"{attempt}.json"
        evaluation = root / "outputs" / "reports" / f"{attempt}_evaluation_complete.json"
        if not valid_evaluation(evaluation, plan["plan_sha256"]):
            parity = [root / "outputs" / "reports" / f"{dataset}_{model}_fp16_parity_v1.json" for model in ("yolo11n", "yolo11m", "yolo11x")]
            command = [sys.executable, str(root / "src" / "execute_dataset_pilot.py"), "--plan", str(plan_path), "--project-root", str(root),
                       "--dataset", dataset, "--split", split, "--annotations", str(root / data["annotations"]), "--class-map", str(root / data["class_map"]),
                       "--clean-root", str(root / data["clean_root"]), "--corruption-root", str(root / data["corruption_root"]),
                       "--calibration-list", str(root / data["calibration_list"]), "--imgsz", str(data["imgsz"]),
                       "--expected-images", str(data["expected_images"]), "--attempt", attempt, "--execute", "--resume-verified",
                       "--reviewed-by", "project-owner-explicit-pilot-approval-in-active-session"]
            for parity_path in parity:
                command += ["--fp16-parity", str(parity_path)]
            subprocess.run(command, check=True)
            subprocess.run([sys.executable, str(root / "src" / "evaluate_dataset_pilot.py"), "--plan", str(plan_path), "--project-root", str(root),
                            "--dataset", dataset, "--split", split, "--annotations", str(root / data["annotations"]), "--attempt", attempt,
                            "--expected-images", str(data["expected_images"]), "--execute", "--resume-verified"], check=True)
        if not valid_evaluation(evaluation, plan["plan_sha256"]):
            raise SystemExit(f"TRANSFER EXECUTION REFUSED: evaluation not complete: {dataset}")
        if not receipt.is_file() or json.loads(receipt.read_text(encoding="utf-8")).get("plan_sha256") != plan["plan_sha256"]:
            raise SystemExit(f"TRANSFER EXECUTION REFUSED: approval receipt invalid: {dataset}")
        records.append({"dataset": dataset, "attempt": attempt, "plan_sha256": plan["plan_sha256"],
                        "evaluation_report": str(evaluation), "evaluation_report_sha256": sha256_file(evaluation)})
    document = {"schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "config": str(config_path),
                "config_sha256": sha256_file(config_path), "prep_report": str(prerequisite), "prep_report_sha256": sha256_file(prerequisite), "datasets": records}
    document["transfer_execution_report_sha256"] = canonical_hash(document, "transfer_execution_report_sha256")
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"TRANSFER EXECUTION COMPLETE datasets={len(records)} report={report}")


if __name__ == "__main__":
    main()
