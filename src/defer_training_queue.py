#!/usr/bin/env python3
"""Start one frozen training queue only after another hash-valid queue completes."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def canonical_hash(document: dict, excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def complete(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return document.get("queue_report_sha256") == canonical_hash(document, "queue_report_sha256")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--wait-for-report", required=True)
    parser.add_argument("--queue", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root, prerequisite, queue, report = (Path(args.project_root).resolve(), Path(args.wait_for_report).resolve(),
                                         Path(args.queue).resolve(), Path(args.report_out).resolve())
    if report.exists():
        raise SystemExit("DEFERRED TRAINING REFUSED: target queue report already exists")
    if not queue.is_file():
        raise SystemExit("DEFERRED TRAINING REFUSED: queue file is absent")
    while not complete(prerequisite):
        print(f"DEFERRED TRAINING waiting for hash-valid prerequisite: {prerequisite}", flush=True)
        time.sleep(args.poll_seconds)
    print(f"DEFERRED TRAINING prerequisite complete: {prerequisite}", flush=True)
    subprocess.run([sys.executable, str(root / "src" / "run_training_queue.py"), "--project-root", str(root),
                    "--queue", str(queue), "--report-out", str(report), "--poll-seconds", str(args.poll_seconds)], check=True)


if __name__ == "__main__":
    main()
