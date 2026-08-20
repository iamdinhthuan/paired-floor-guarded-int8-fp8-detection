#!/usr/bin/env python3
"""Run three immutable engine-only trtexec repetitions for 18 cross-family engines."""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import uuid
from datetime import datetime, timezone
from pathlib import Path

from benchmark_trt_engines import parse_trtexec_latency
from topic_c.manifest import sha256_file


def main() -> None:
    root = Path(".").resolve(); attempt = "cross_family_deployment_v1"
    records = root / "outputs" / "deployment" / attempt / "records"
    logs = root / "outputs" / "deployment" / attempt / "logs"
    report = root / "outputs" / "reports" / f"{attempt}_complete.json"
    if records.exists() or logs.exists() or report.exists():
        raise SystemExit("deployment benchmark refused: output already exists")
    records.mkdir(parents=True); logs.mkdir(parents=True)
    trt = Path("/home/thuan/traffic/third_party/TensorRT-11.1.0.106")
    executable, library = trt / "bin" / "trtexec", trt / "lib"
    environment = dict(os.environ); environment["LD_LIBRARY_PATH"] = str(library)
    hashes = {}; conditions = 0
    for registry in sorted((root / "manifests/engines/cross_family_v1").glob("*.json")):
        marker = registry.with_suffix(registry.suffix + ".complete")
        if marker.read_text().strip() != sha256_file(registry): raise RuntimeError(f"invalid registry: {registry}")
        engine_record = json.loads(registry.read_text()); engine = Path(engine_record["engine"])
        if sha256_file(engine) != engine_record["engine_sha256"]: raise RuntimeError(f"invalid engine: {engine}")
        condition = registry.stem; conditions += 1
        for repetition in (1, 2, 3):
            stem = f"{condition}__r{repetition}"
            command = [str(executable), f"--loadEngine={engine}", "--warmUp=5000", "--duration=30",
                       "--useCudaGraph", "--noDataTransfers", "--useSpinWait"]
            started = datetime.now(timezone.utc).isoformat()
            process = subprocess.run(command, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                     text=True, env=environment, check=False)
            ended = datetime.now(timezone.utc).isoformat()
            if process.returncode != 0: raise RuntimeError(f"trtexec failed: {stem}\n{process.stdout[-2000:]}")
            metrics = parse_trtexec_latency(process.stdout)
            log = logs / f"{stem}.log"; log.write_text(process.stdout)
            record = {"schema_version": 1, "run_id": uuid.uuid4().hex, "condition": condition,
                      "dataset": engine_record["dataset"], "model": engine_record["model"],
                      "precision": engine_record["precision"], "repetition": repetition,
                      "engine_registry_sha256": sha256_file(registry), "engine_sha256": sha256_file(engine),
                      "trtexec_sha256": sha256_file(executable), "command": command,
                      "started_at_utc": started, "ended_at_utc": ended, "return_code": process.returncode,
                      "raw_log_sha256": sha256_file(log), **metrics}
            path = records / f"{stem}.json"; path.write_text(json.dumps(record, indent=2) + "\n")
            hashes[f"records/{path.name}"] = sha256_file(path); hashes[f"logs/{log.name}"] = sha256_file(log)
            print(f"DEPLOYMENT {len(hashes)//2}/54 {stem}", flush=True)
    if conditions != 18 or len(hashes) != 108: raise RuntimeError("deployment grid is incomplete")
    document = {"schema_version": 1, "attempt": attempt, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "conditions": conditions, "repetitions_per_condition": 3, "records": 54,
                "scope": "TensorRT engine-only; generated input; excludes preprocessing, decode, NMS and transfers",
                "artifact_sha256": dict(sorted(hashes.items()))}
    document["report_sha256"] = hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text(json.dumps(document, indent=2) + "\n")
    print("CROSS-FAMILY DEPLOYMENT COMPLETE")


if __name__ == "__main__": main()
