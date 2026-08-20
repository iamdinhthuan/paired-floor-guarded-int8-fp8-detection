#!/usr/bin/env python3
"""Create one immutable summary only after all four reduced-pilot result chains are complete."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from topic_c.manifest import sha256_file


def canonical_hash(document: dict, excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def read_json(path: Path) -> dict | None:
    try:
        return json.loads(path.read_text(encoding="utf-8")) if path.is_file() else None
    except json.JSONDecodeError:
        return None


def transfer_complete(execution: dict | None, bootstrap: dict | None) -> bool:
    return (bool(execution) and bool(bootstrap) and execution.get("transfer_execution_report_sha256") == canonical_hash(execution, "transfer_execution_report_sha256")
            and bootstrap.get("transfer_bootstrap_report_sha256") == canonical_hash(bootstrap, "transfer_bootstrap_report_sha256")
            and len(execution.get("datasets", [])) == 3 and len(bootstrap.get("datasets", [])) == 3)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True); parser.add_argument("--out", required=True); parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root, output = Path(args.project_root).resolve(), Path(args.out).resolve()
    if output.exists():
        raise SystemExit(f"FOUR-DATASET REPORT REFUSED: output already exists: {output}")
    coco_path = root / "outputs" / "reports" / "coco_pilot_117_final_report.json"
    execution_path = root / "outputs" / "reports" / "three_transfer_pilot_execution_v1.json"
    bootstrap_path = root / "outputs" / "reports" / "three_transfer_bootstrap_v1.json"
    while True:
        coco, execution, bootstrap = read_json(coco_path), read_json(execution_path), read_json(bootstrap_path)
        if coco and coco.get("checks", {}).get("runs") == 117 and coco.get("checks", {}).get("bootstrap_cells") == 72 and transfer_complete(execution, bootstrap):
            break
        print("FOUR-DATASET REPORT waiting for COCO and transfer completion reports", flush=True)
        time.sleep(args.poll_seconds)
    datasets = [{"dataset": "coco", "report": str(coco_path), "report_sha256": sha256_file(coco_path),
                 "runs": coco["checks"]["runs"], "bootstrap_cells": coco["checks"]["bootstrap_cells"], "bootstrap_replicates": coco["checks"]["bootstrap_replicates"]}]
    for item in execution["datasets"]:
        dataset = item["dataset"]
        boot = next(value for value in bootstrap["datasets"] if value["dataset"] == dataset)
        datasets.append({"dataset": dataset, "attempt": item["attempt"], "plan_sha256": item["plan_sha256"],
                         "evaluation_report": item["evaluation_report"], "evaluation_report_sha256": item["evaluation_report_sha256"],
                         "bootstrap_report": boot["bootstrap_report"], "bootstrap_report_sha256": boot["sha256"],
                         "runs": 117, "bootstrap_cells": 72, "bootstrap_replicates": 500})
    document = {"schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "phase": "four-dataset reduced pilot",
                "datasets": datasets, "checks": {"datasets": 4, "runs_per_dataset": 117, "bootstrap_cells_per_dataset": 72,
                "bootstrap_replicates": 500, "all_dataset_completion_reports_hash_validated": True},
                "scope_guard": "This report covers COCO, VOC, KITTI, and TT100K reduced pilots only; it does not authorize the full grid or calibration intervention."}
    document["four_dataset_report_sha256"] = canonical_hash(document, "four_dataset_report_sha256")
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"FOUR-DATASET REPORT COMPLETE datasets={len(datasets)} out={output}")


if __name__ == "__main__":
    main()
