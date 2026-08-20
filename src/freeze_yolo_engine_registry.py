#!/usr/bin/env python3
"""Freeze a complete, hash-verified YOLO precision ladder for one dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from topic_c.manifest import sha256_file


MODELS = ("yolo11n", "yolo11m", "yolo11x")
PRECISIONS = ("fp32", "fp16", "int8-entropy", "fp8")


def canonical_hash(document: dict, excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def complete(path: Path) -> dict:
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.is_file() or not marker.is_file() or marker.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise SystemExit(f"ENGINE REGISTRY REFUSED: incomplete engine record: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--dataset", choices=("coco", "voc", "kitti", "tt100k"), required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, output = Path(args.project_root).resolve(), Path(args.out).resolve()
    if output.exists():
        raise SystemExit(f"ENGINE REGISTRY REFUSED: refusing to overwrite: {output}")
    engines: dict[str, dict[str, dict]] = {}
    for model in MODELS:
        engines[model] = {}
        for precision in PRECISIONS:
            record_path = root / "manifests" / "engines" / f"{args.dataset}_{model}_{precision}_v1.json"
            record = complete(record_path)
            if record.get("dataset") != args.dataset or record.get("model") != model or record.get("precision") != precision:
                raise SystemExit(f"ENGINE REGISTRY REFUSED: provenance mismatch: {record_path}")
            engine = Path(record.get("engine", "")).resolve()
            if not engine.is_file() or sha256_file(engine) != record.get("engine_sha256"):
                raise SystemExit(f"ENGINE REGISTRY REFUSED: engine hash mismatch: {record_path}")
            engines[model][precision] = {
                "path": str(engine), "sha256": record["engine_sha256"],
                "engine_registry_sha256": sha256_file(record_path),
                "calibration_sha256": record.get("calibration_sha256"),
                "imgsz": record.get("imgsz"),
            }
    document = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset, "models": list(MODELS), "precisions": list(PRECISIONS), "engines": engines,
    }
    document["registry_sha256"] = canonical_hash(document, "registry_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".complete").write_text(sha256_file(output) + "\n", encoding="utf-8")
    print(json.dumps({"ENGINE REGISTRY COMPLETE": str(output), "registry_sha256": document["registry_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
