#!/usr/bin/env python3
"""Export a hash-verified trained YOLO checkpoint to a fixed-shape FP32 ONNX file."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

import onnx
import ultralytics
from ultralytics import YOLO

from topic_c.manifest import sha256_file


def complete_record(path: Path) -> dict:
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.is_file() or not marker.is_file() or marker.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise SystemExit(f"ONNX EXPORT REFUSED: incomplete registry: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-registry", required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--out", required=True)
    parser.add_argument("--registry-out", required=True)
    args = parser.parse_args()
    training_path, output, record_out = Path(args.training_registry).resolve(), Path(args.out).resolve(), Path(args.registry_out).resolve()
    if output.exists() or record_out.exists() or output.with_suffix(".pt").exists():
        raise SystemExit("ONNX EXPORT REFUSED: output, source copy, or registry already exists")
    training = complete_record(training_path)
    best = Path(training.get("best_weights", "")).resolve()
    if not best.is_file() or sha256_file(best) != training.get("best_weights_sha256"):
        raise SystemExit("ONNX EXPORT REFUSED: best checkpoint hash mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    source_copy = output.with_suffix(".pt")
    shutil.copy2(best, source_copy)
    if sha256_file(source_copy) != sha256_file(best):
        raise SystemExit("ONNX EXPORT REFUSED: checkpoint copy hash mismatch")
    model = YOLO(str(source_copy))
    produced = Path(str(model.export(format="onnx", imgsz=args.imgsz, batch=1, dynamic=False, simplify=True,
                                     opset=19, nms=False, half=False, device=0))).resolve()
    if produced != output or not output.is_file():
        raise SystemExit(f"ONNX EXPORT REFUSED: exporter produced unexpected path: {produced}")
    graph = onnx.load(output, load_external_data=False).graph
    if len(graph.input) != 1 or len(graph.output) != 1:
        raise SystemExit("ONNX EXPORT REFUSED: controlled runner requires one input and one output")
    record = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": training["dataset"], "model": training["model"], "training_registry_sha256": sha256_file(training_path),
        "source_checkpoint": str(source_copy), "source_checkpoint_sha256": sha256_file(source_copy),
        "onnx": str(output), "onnx_sha256": sha256_file(output), "imgsz": args.imgsz,
        "opset": 19, "dynamic": False, "nms_embedded": False, "half": False,
        "input_name": graph.input[0].name, "output_name": graph.output[0].name,
        "ultralytics_version": ultralytics.__version__,
    }
    record_out.parent.mkdir(parents=True, exist_ok=True)
    record_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record_out.with_suffix(record_out.suffix + ".complete").write_text(sha256_file(record_out) + "\n", encoding="utf-8")
    print(json.dumps({"ONNX EXPORT COMPLETE": str(output), "onnx_sha256": record["onnx_sha256"],
                      "registry": str(record_out)}, indent=2))


if __name__ == "__main__":
    main()
