#!/usr/bin/env python3
"""Create FP16, INT8-entropy, or FP8 ModelOpt ONNX from frozen inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx

from topic_c.coco_data import preprocess
from topic_c.manifest import sha256_file


@contextmanager
def isolated_onnx_source(source: Path):
    """Shield a frozen source graph from in-place ModelOpt normalization."""
    source = Path(source).resolve()
    before = sha256_file(source)
    with tempfile.TemporaryDirectory(prefix="modelopt-source-") as directory:
        working = Path(directory) / source.name
        shutil.copy2(source, working)
        if sha256_file(working) != before:
            raise RuntimeError("isolated ONNX copy hash mismatch")
        yield working
    if sha256_file(source) != before:
        raise RuntimeError("ModelOpt mutated the frozen source ONNX")


def canonical_hash(document: dict, field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_complete(path: Path, marker_field: str | None = None) -> dict:
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.is_file() or not marker.is_file():
        raise SystemExit(f"ONNX QUANTIZATION REFUSED: incomplete input: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = document.get(marker_field) if marker_field else sha256_file(path)
    if marker.read_text(encoding="utf-8").strip() != expected:
        raise SystemExit(f"ONNX QUANTIZATION REFUSED: completion marker mismatch: {path}")
    return document


def calibration_tensor(document: dict, imgsz: int) -> np.ndarray:
    root = Path(document["dataset_root"]).resolve()
    tensors = []
    for record in document["records"]:
        path = root / record["source_relpath"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise SystemExit(f"ONNX QUANTIZATION REFUSED: calibration image hash mismatch: {path}")
        tensors.append(preprocess(str(path), imgsz)[0][0])
    values = np.stack(tensors).astype(np.float32)
    if values.shape != (document["n_images"], 3, imgsz, imgsz):
        raise SystemExit(f"ONNX QUANTIZATION REFUSED: unexpected calibration tensor shape: {values.shape}")
    return values


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-registry", required=True)
    parser.add_argument("--mode", choices=("fp16", "int8-entropy", "fp8"), required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--calibration-list")
    parser.add_argument("--quantize-op-types", help="comma-separated audited operator allowlist")
    parser.add_argument("--out", required=True)
    parser.add_argument("--registry-out", required=True)
    args = parser.parse_args()
    onnx_registry, output, record_out = Path(args.onnx_registry).resolve(), Path(args.out).resolve(), Path(args.registry_out).resolve()
    if output.exists() or record_out.exists():
        raise SystemExit("ONNX QUANTIZATION REFUSED: output or registry already exists")
    source = load_complete(onnx_registry)
    onnx_path = Path(source.get("onnx", "")).resolve()
    if not onnx_path.is_file() or sha256_file(onnx_path) != source.get("onnx_sha256"):
        raise SystemExit("ONNX QUANTIZATION REFUSED: source FP32 ONNX hash mismatch")
    if args.mode == "fp16" and args.calibration_list:
        raise SystemExit("ONNX QUANTIZATION REFUSED: FP16 conversion has no calibration list")
    calibration = None
    quantize_op_types = [value.strip() for value in args.quantize_op_types.split(",") if value.strip()] if args.quantize_op_types else None
    calibration_path = Path(args.calibration_list).resolve() if args.calibration_list else None
    if args.mode != "fp16":
        if calibration_path is None:
            raise SystemExit("ONNX QUANTIZATION REFUSED: INT8/FP8 requires a train-only calibration list")
        calibration = load_complete(calibration_path, "calibration_sha256")
        if calibration.get("dataset") != source.get("dataset") or calibration.get("split") != "train":
            raise SystemExit("ONNX QUANTIZATION REFUSED: calibration list dataset/split mismatch")
    graph = onnx.load(onnx_path, load_external_data=False).graph
    if len(graph.input) != 1:
        raise SystemExit("ONNX QUANTIZATION REFUSED: expected exactly one model input")
    output.parent.mkdir(parents=True, exist_ok=True)
    with isolated_onnx_source(onnx_path) as modelopt_input:
        if args.mode == "fp16":
            import modelopt.onnx.autocast as autocast
            random_input = np.random.default_rng(20260807).normal(size=(1, 3, args.imgsz, args.imgsz)).astype(np.float32)
            onnx.save(autocast.convert_to_mixed_precision(modelopt_input, low_precision_type="fp16", keep_io_types=True,
                                                           calibration_data={graph.input[0].name: random_input}), output)
            calibration_sha = None
        else:
            from modelopt.onnx.quantization import quantize
            # ModelOpt may normalize its input graph in place.  It receives a
            # disposable byte-identical copy so INT8 cannot change FP8's source.
            options = {"quantize_mode": "fp8" if args.mode == "fp8" else "int8",
                       "calibration_data": {graph.input[0].name: calibration_tensor(calibration, args.imgsz)},
                       "calibration_method": "entropy", "calibration_eps": ["cpu"],
                       "op_types_to_exclude": ["Sigmoid"], "output_path": str(output)}
            if quantize_op_types is not None:
                options["op_types_to_quantize"] = quantize_op_types
            quantize(str(modelopt_input), **options)
            calibration_sha = calibration["calibration_sha256"]
    if not output.is_file():
        raise SystemExit("ONNX QUANTIZATION REFUSED: ModelOpt produced no ONNX output")
    record = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": source["dataset"],
        "model": source["model"], "precision": args.mode, "source_onnx_registry_sha256": sha256_file(onnx_registry),
        "source_onnx_sha256": sha256_file(onnx_path), "output_onnx": str(output), "output_onnx_sha256": sha256_file(output),
        "calibration_list": str(calibration_path) if calibration_path else None, "calibration_sha256": calibration_sha,
        "calibration_method": "entropy" if args.mode != "fp16" else "not_applicable", "imgsz": args.imgsz,
        "quantize_mode": "fp8" if args.mode == "fp8" else ("int8" if args.mode == "int8-entropy" else "fp16_autocast"),
        "calibration_eps": ["cpu"] if args.mode != "fp16" else [],
        "op_types_to_exclude": ["Sigmoid"] if args.mode != "fp16" else [],
        "op_types_to_quantize": quantize_op_types,
        "input_names": source.get("input_names", [value.name for value in graph.input]),
        "output_names": source.get("output_names", [value.name for value in graph.output]),
        "decoder": source.get("decoder"),
        "num_classes_excluding_background": source.get("num_classes_excluding_background"),
        "modelopt_version": importlib.metadata.version("nvidia-modelopt"),
        "onnx_version": onnx.__version__,
        "command": list(__import__("sys").argv),
    }
    record["registry_sha256"] = canonical_hash(record, "registry_sha256")
    record_out.parent.mkdir(parents=True, exist_ok=True)
    record_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record_out.with_suffix(record_out.suffix + ".complete").write_text(sha256_file(record_out) + "\n", encoding="utf-8")
    print(json.dumps({"ONNX QUANTIZATION COMPLETE": args.mode, "onnx_sha256": record["output_onnx_sha256"],
                      "registry": str(record_out)}, indent=2))


if __name__ == "__main__":
    main()
