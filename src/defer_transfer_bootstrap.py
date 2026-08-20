#!/usr/bin/env python3
"""Run all transfer paired-bootstrap cells after their complete pilot evaluations."""
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


def valid_bootstrap(path: Path, plan_sha: str) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("plan_sha256") == plan_sha and data.get("cells") == 72 and data.get("n_boot") == 500 and len(data.get("artifacts_sha256", {})) == 72


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True); parser.add_argument("--wait-for-execution-report", required=True)
    parser.add_argument("--config", required=True); parser.add_argument("--report-out", required=True); parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root, prerequisite, config_path, report = (Path(args.project_root).resolve(), Path(args.wait_for_execution_report).resolve(),
                                                Path(args.config).resolve(), Path(args.report_out).resolve())
    if report.exists():
        raise SystemExit(f"TRANSFER BOOTSTRAP REFUSED: report already exists: {report}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if [item.get("dataset") for item in config.get("datasets", [])] != ["voc", "kitti", "tt100k"]:
        raise SystemExit("TRANSFER BOOTSTRAP REFUSED: unexpected frozen dataset sequence")
    while not valid(prerequisite, "transfer_execution_report_sha256"):
        print(f"TRANSFER BOOTSTRAP waiting for hash-valid execution report: {prerequisite}", flush=True)
        time.sleep(args.poll_seconds)
    records = []
    for data in config["datasets"]:
        dataset, attempt = data["dataset"], f"{data['dataset']}_pilot_117_v1"
        plan_path = root / "manifests" / "plans" / f"{dataset}_yolo11_nmx_117_frozen_v1.json"
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        bootstrap = root / "outputs" / "reports" / f"{attempt}_bootstrap_complete.json"
        if not valid_bootstrap(bootstrap, plan.get("plan_sha256")):
            command = [sys.executable, str(root / "src" / "bootstrap_dataset_pilot.py"), "--plan", str(plan_path), "--project-root", str(root),
                       "--annotations", str(root / data["annotations"]), "--attempt", attempt, "--expected-images", str(data["expected_images"]),
                       "--execute", "--resume-verified", "--jobs", "6"]
            if dataset == "tt100k":
                command += ["--bootstrap-runner", str(root / "src" / "paired_bootstrap_tt100k.py")]
            subprocess.run(command, check=True)
        if not valid_bootstrap(bootstrap, plan.get("plan_sha256")):
            raise SystemExit(f"TRANSFER BOOTSTRAP REFUSED: incomplete output: {dataset}")
        records.append({"dataset": dataset, "attempt": attempt, "bootstrap_report": str(bootstrap), "sha256": sha256_file(bootstrap)})
    document = {"schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "config": str(config_path),
                "config_sha256": sha256_file(config_path), "execution_report": str(prerequisite), "execution_report_sha256": sha256_file(prerequisite), "datasets": records}
    document["transfer_bootstrap_report_sha256"] = canonical_hash(document, "transfer_bootstrap_report_sha256")
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"TRANSFER BOOTSTRAP COMPLETE datasets={len(records)} report={report}")


if __name__ == "__main__":
    main()
