#!/usr/bin/env python3
"""Start all three transfer FP16 parity gates only after the engine ladders are complete."""
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


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True); parser.add_argument("--wait-for-ladder-report", required=True)
    parser.add_argument("--config", required=True); parser.add_argument("--report-out", required=True); parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root, prerequisite, config_path, report = (Path(args.project_root).resolve(), Path(args.wait_for_ladder_report).resolve(),
                                                Path(args.config).resolve(), Path(args.report_out).resolve())
    if report.exists():
        raise SystemExit(f"FP16 PARITY QUEUE REFUSED: report already exists: {report}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    datasets = [item.get("dataset") for item in config.get("datasets", [])]
    if datasets != ["voc", "kitti", "tt100k"]:
        raise SystemExit("FP16 PARITY QUEUE REFUSED: unexpected frozen dataset sequence")
    while not valid(prerequisite, "ladder_report_sha256"):
        print(f"FP16 PARITY QUEUE waiting for hash-valid ladder report: {prerequisite}", flush=True)
        time.sleep(args.poll_seconds)
    records = []
    for dataset in datasets:
        completion = root / "outputs" / "reports" / f"{dataset}_clean_fp16_parity_complete_v1.json"
        if not valid(completion, "parity_completion_sha256"):
            subprocess.run([sys.executable, str(root / "src" / "run_clean_fp16_parity.py"), "--project-root", str(root),
                            "--config", str(config_path), "--dataset", dataset, "--resume-verified"], check=True)
        if not valid(completion, "parity_completion_sha256"):
            raise SystemExit(f"FP16 PARITY QUEUE REFUSED: invalid completion artifact: {completion}")
        records.append({"dataset": dataset, "completion": str(completion), "sha256": sha256_file(completion)})
    document = {"schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "config": str(config_path),
                "config_sha256": sha256_file(config_path), "ladder_report": str(prerequisite), "ladder_report_sha256": sha256_file(prerequisite),
                "datasets": records}
    document["parity_queue_report_sha256"] = canonical_hash(document, "parity_queue_report_sha256")
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"FP16 PARITY QUEUE COMPLETE datasets={len(records)} report={report}")


if __name__ == "__main__":
    main()
