#!/usr/bin/env python3
"""Separate diagnostic runs and finalize the canonical 18x3 deployment grid."""
from __future__ import annotations

import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

from topic_c.manifest import sha256_file


def main() -> None:
    root = Path(".").resolve(); attempt = "cross_family_deployment_v1"
    base = root / "outputs" / "deployment" / attempt
    records, logs = base / "records", base / "logs"
    diagnostic = root / "outputs" / "deployment" / "cross_family_diagnostic_v1"
    for source_dir, suffix in ((records, ".json"), (logs, ".log")):
        destination = diagnostic / source_dir.name; destination.mkdir(parents=True, exist_ok=True)
        for path in source_dir.glob(f"*int8-conv-v2*{suffix}"):
            target = destination / path.name
            if target.exists(): raise SystemExit(f"diagnostic target exists: {target}")
            os.replace(path, target)
    record_paths, log_paths = sorted(records.glob("*.json")), sorted(logs.glob("*.log"))
    if len(record_paths) != 54 or len(log_paths) != 54:
        raise SystemExit(f"canonical deployment grid is not 54+54: {len(record_paths)}+{len(log_paths)}")
    hashes, conditions, repetitions = {}, set(), set()
    for path in record_paths:
        document = json.loads(path.read_text())
        log = logs / path.with_suffix(".log").name
        if not log.is_file() or document.get("raw_log_sha256") != sha256_file(log):
            raise SystemExit(f"raw-log binding mismatch: {path}")
        condition = document.get("condition"); repetition = document.get("repetition")
        if "int8-conv-v2" in str(condition) or repetition not in (1, 2, 3):
            raise SystemExit(f"noncanonical record: {path}")
        conditions.add(condition); repetitions.add((condition, repetition))
        hashes[f"records/{path.name}"] = sha256_file(path); hashes[f"logs/{log.name}"] = sha256_file(log)
    if len(conditions) != 18 or len(repetitions) != 54:
        raise SystemExit("deployment condition/repetition coverage mismatch")
    report = root / "outputs" / "reports" / f"{attempt}_complete.json"
    if report.exists(): raise SystemExit("completion report already exists")
    result = {"schema_version": 1, "attempt": attempt, "created_at_utc": datetime.now(timezone.utc).isoformat(),
              "conditions": 18, "repetitions_per_condition": 3, "records": 54,
              "scope": "TensorRT engine-only; generated input; excludes preprocessing, decode, NMS and transfers",
              "diagnostic_runs_excluded": 3, "artifact_sha256": dict(sorted(hashes.items()))}
    result["report_sha256"] = hashlib.sha256(json.dumps(result, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text(json.dumps(result, indent=2) + "\n")
    print(json.dumps({"report": str(report), "conditions": 18, "records": 54, "artifacts": 108}))


if __name__ == "__main__": main()
