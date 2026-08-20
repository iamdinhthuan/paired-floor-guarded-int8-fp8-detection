#!/usr/bin/env python3
"""Retain best.pt only after a completed training registry validates both weights."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path

from topic_c.manifest import sha256_file


def compact_completed_run(registry_path: Path, output: Path) -> dict:
    registry_path, output = registry_path.resolve(), output.resolve()
    marker = registry_path.with_suffix(registry_path.suffix + ".complete")
    if (
        not registry_path.is_file()
        or not marker.is_file()
        or marker.read_text(encoding="utf-8").strip() != sha256_file(registry_path)
    ):
        raise SystemExit("CHECKPOINT COMPACTION REFUSED: training registry is incomplete")
    if output.exists() or output.with_suffix(output.suffix + ".complete").exists():
        raise SystemExit("CHECKPOINT COMPACTION REFUSED: compaction report already exists")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    best = Path(registry.get("best_weights", "")).resolve()
    last = Path(registry.get("last_weights", "")).resolve()
    if best.name != "best.pt" or last.name != "last.pt" or best.parent != last.parent:
        raise SystemExit("CHECKPOINT COMPACTION REFUSED: unexpected checkpoint layout")
    if not best.is_file() or sha256_file(best) != registry.get("best_weights_sha256"):
        raise SystemExit("CHECKPOINT COMPACTION REFUSED: best checkpoint hash mismatch")
    if not last.is_file() or sha256_file(last) != registry.get("last_weights_sha256"):
        raise SystemExit("CHECKPOINT COMPACTION REFUSED: last checkpoint hash mismatch")
    best_sha, last_sha = sha256_file(best), sha256_file(last)
    last.unlink()
    if last.exists() or not best.is_file() or sha256_file(best) != best_sha:
        raise SystemExit("CHECKPOINT COMPACTION REFUSED: post-delete verification failed")
    report = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "run_id": registry.get("run_id"),
        "training_registry": str(registry_path),
        "training_registry_sha256": sha256_file(registry_path),
        "retained_best_weights": str(best),
        "retained_best_weights_sha256": best_sha,
        "deleted_last_weights": str(last),
        "deleted_last_weights_sha256": last_sha,
        "recovery": "last.pt was intentionally deleted after hash validation; best.pt remains",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".complete").write_text(sha256_file(output) + "\n", encoding="utf-8")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-registry", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    report = compact_completed_run(Path(args.training_registry), Path(args.out))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
