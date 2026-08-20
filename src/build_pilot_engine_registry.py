#!/usr/bin/env python3
"""Freeze legacy engine/calibration provenance in the Topic C project."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pilot_registry import canonical_hash
from topic_c.manifest import sha256_file


ENGINE_NAMES = {
    "yolo11n": {"fp32": "yolo11n_fp32.plan", "int8-entropy": "yolo11n_int8_entropy.plan", "fp8": "yolo11n_fp8.plan"},
    "yolo11m": {"fp32": "yolo11m_fp32.plan", "int8-entropy": "yolo11m_int8_entropy.plan", "fp8": "yolo11m_fp8.plan"},
    # The x-rung predates the underscore naming convention but is the entropy sweep artifact.
    "yolo11x": {"fp32": "yolo11x_fp32.plan", "int8-entropy": "yolo11x_int8ent.plan", "fp8": "yolo11x_fp8.plan"},
}


def asset(path: Path) -> dict[str, object]:
    if not path.is_file():
        raise SystemExit(f"missing provenance asset: {path}")
    return {"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--legacy-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, output = Path(args.legacy_root), Path(args.out)
    if output.exists():
        raise SystemExit(f"refusing to overwrite engine registry: {output}")
    calibration_list = root / "data/coco/calib/calib_list.json"
    evidence_paths = [
        root / "tools/coco_common.py", root / "tools/coco_quantize_onnx.py",
        root / "tools/calib_ablation.sh", root / "tools/review_experiments2.sh",
        root / "tools/entropy_finish.sh", root / "entropy_finish.log", root / "run_review2.log",
    ]
    engines = {}
    for model, mapping in ENGINE_NAMES.items():
        engines[model] = {}
        for precision, name in mapping.items():
            engine = asset(root / "exports/coco_pilot" / name)
            engines[model][precision] = {
                **engine,
                "calibrator": "entropy" if precision == "int8-entropy" else "not_applicable",
                "calibration_sha256": sha256_file(calibration_list) if precision == "int8-entropy" else None,
                "calibration_provenance": "verified" if precision == "int8-entropy" else "not_applicable",
            }
    document = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "purpose": "Topic C COCO pilot engine registry; legacy assets are immutable references",
        "calibration": {"list": asset(calibration_list), "method": "entropy", "n_images": 512,
                        "preprocessing": "COCO letterbox 640; source is tools/coco_common.py"},
        "evidence": [asset(path) for path in evidence_paths],
        "engines": engines,
    }
    document["registry_sha256"] = canonical_hash(document, "registry_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"ENGINE REGISTRY VALID models={len(engines)} sha256={document['registry_sha256']}")


if __name__ == "__main__":
    main()
