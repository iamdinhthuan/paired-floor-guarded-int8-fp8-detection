#!/usr/bin/env python3
"""TensorRT COCO inference that cannot silently substitute clean image bytes."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path

import numpy as np

from topic_c.coco_data import load_coco_images, preprocess
from pilot_registry import calibration_sha256
from topic_c.manifest import read_manifest, sha256_file
from topic_c.yolo_decode import COCO80_TO_91, CONF_FLOOR, decode


def refuse_existing(*paths: str) -> None:
    existing = [path for path in paths if Path(path).exists()]
    if existing:
        raise SystemExit("refusing to overwrite: " + ", ".join(existing))


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def runtime_environment(tensorrt_version: str) -> dict[str, object]:
    """Capture non-secret, condition-level runtime provenance once per inference."""
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,uuid", "--format=csv,noheader"],
            text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True,
        )
        gpu = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError):
        gpu = []
    return {"python_executable": os.sys.executable, "python_version": platform.python_version(),
            "tensorrt_version": tensorrt_version, "gpu": gpu}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine", required=True)
    parser.add_argument("--annotations", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-root", help="root containing official COCO filenames")
    source.add_argument("--image-manifest", help="validated corruption manifest")
    parser.add_argument("--manifest-cache-root", help="required with --image-manifest")
    parser.add_argument("--out", required=True)
    parser.add_argument("--input-record", required=True, help="sidecar with every evaluated official ID")
    parser.add_argument("--run-record", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--dataset", default="coco", help="dataset namespace embedded in run provenance")
    parser.add_argument("--split", default="val2017", help="dataset split embedded in run provenance")
    parser.add_argument("--model", required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--calibrator", default="none")
    parser.add_argument("--calibration-list", help="immutable build-calibration list for a quantized engine")
    parser.add_argument("--calibration-method", choices=("entropy", "max", "not_applicable"), default="not_applicable")
    parser.add_argument("--calibration-provenance", choices=("verified", "candidate_unverified", "not_applicable"), default="not_applicable")
    parser.add_argument("--corruption", required=True)
    parser.add_argument("--severity", type=int, required=True)
    parser.add_argument("--class-map", help="immutable JSON with class_to_category_id for non-COCO label spaces")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=CONF_FLOOR)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    refuse_existing(args.out, args.input_record, args.run_record)
    if args.image_manifest and not args.manifest_cache_root:
        raise SystemExit("--manifest-cache-root is required with --image-manifest")
    if args.calibration_list and not Path(args.calibration_list).is_file():
        raise SystemExit("--calibration-list is not a file")
    if args.precision.startswith("int8") and not args.calibration_list:
        raise SystemExit("INT8 inference requires --calibration-list provenance")
    if args.calibration_list and args.calibration_method == "not_applicable":
        raise SystemExit("a supplied calibration list requires --calibration-method")
    if args.class_map:
        class_map_path = Path(args.class_map)
        class_map_data = json.loads(class_map_path.read_text(encoding="utf-8"))
        class_to_category = class_map_data.get("class_to_category_id")
        if not isinstance(class_to_category, list) or not all(isinstance(item, int) and item > 0 for item in class_to_category):
            raise SystemExit("--class-map must contain a positive integer class_to_category_id list")
        declared_hash = class_map_data.get("class_map_sha256")
        map_payload = {key: value for key, value in class_map_data.items() if key != "class_map_sha256"}
        map_hash = hashlib.sha256(json.dumps(map_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
        if declared_hash != map_hash:
            raise SystemExit("--class-map canonical SHA-256 mismatch")
        if (class_map_data.get("dataset") != args.dataset or class_map_data.get("split") != args.split
                or class_map_data.get("annotation_sha256") != sha256_file(args.annotations)):
            raise SystemExit("--class-map dataset, split, or annotation provenance mismatch")
    else:
        class_map_path, class_to_category, map_hash = None, COCO80_TO_91, "builtin:COCO80_TO_91"
    manifest_hash = None
    if args.image_manifest:
        manifest = read_manifest(args.image_manifest)
        marker = Path(args.image_manifest + ".complete")
        if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != manifest["manifest_sha256"]:
            raise SystemExit("corruption manifest lacks a matching validation completion marker")
        manifest_hash = manifest["manifest_sha256"]
        if manifest.get("dataset") != args.dataset or manifest.get("split") != args.split:
            raise SystemExit("input manifest dataset/split disagrees with the condition")
        if manifest.get("records") and (manifest["records"][0]["corruption"] != args.corruption or manifest["records"][0]["severity"] != args.severity):
            raise SystemExit("condition label disagrees with the supplied image manifest")
    else:
        manifest_hash = f"clean-root:{sha256_file(args.annotations)}"
    images = load_coco_images(args.annotations, image_root=args.image_root, image_manifest=args.image_manifest, manifest_cache_root=args.manifest_cache_root, limit=args.limit)
    import tensorrt as trt
    try:
        import cuda.bindings.runtime as cudart
    except ImportError:
        import cuda.cudart as cudart
    logger = trt.Logger(trt.Logger.WARNING)
    with open(args.engine, "rb") as handle, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(handle.read())
    if engine is None:
        raise SystemExit("TensorRT could not deserialize engine")
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    inputs = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
    outputs = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
    if len(inputs) != 1 or len(outputs) != 1:
        raise SystemExit("this controlled YOLO runner requires one input and one output tensor")
    input_name, output_name = inputs[0], outputs[0]
    context.set_input_shape(input_name, (1, 3, args.imgsz, args.imgsz))
    host, device = {}, {}
    for name in names:
        shape = tuple(context.get_tensor_shape(name))
        if any(dim < 1 for dim in shape):
            raise SystemExit(f"unresolved TensorRT tensor shape for {name}: {shape}")
        host[name] = np.empty(shape, dtype=trt.nptype(engine.get_tensor_dtype(name)))
        error, device[name] = cudart.cudaMalloc(host[name].nbytes)
        if error != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc failed for {name}: {error}")
        context.set_tensor_address(name, device[name])
    error, stream = cudart.cudaStreamCreate()
    if error != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError(f"cudaStreamCreate failed: {error}")
    predictions, started = [], time.time()
    try:
        for number, (image_id, image_path) in enumerate(images, start=1):
            tensor, gain, padx, pady, width, height = preprocess(image_path, args.imgsz)
            np.copyto(host[input_name], tensor, casting="same_kind")
            for pointer, size, kind in ((device[input_name], host[input_name].nbytes, cudart.cudaMemcpyKind.cudaMemcpyHostToDevice),):
                error, = cudart.cudaMemcpyAsync(pointer, host[input_name].ctypes.data, size, kind, stream)
                if error != cudart.cudaError_t.cudaSuccess:
                    raise RuntimeError(f"H2D copy failed: {error}")
            if not context.execute_async_v3(stream_handle=stream):
                raise RuntimeError("TensorRT execute_async_v3 returned false")
            error, = cudart.cudaMemcpyAsync(host[output_name].ctypes.data, device[output_name], host[output_name].nbytes, cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
            if error != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError(f"D2H copy failed: {error}")
            cudart.cudaStreamSynchronize(stream)
            for x1, y1, x2, y2, score, cls in decode(host[output_name], args.conf, gain, padx, pady, width, height):
                if not 0 <= cls < len(class_to_category):
                    raise SystemExit(f"decoder class {cls} is outside the immutable class map")
                predictions.append({"image_id": image_id, "category_id": class_to_category[cls], "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)], "score": score})
            if number % 100 == 0:
                print(f"{number}/{len(images)} images; {number / (time.time() - started):.2f} image/s", flush=True)
    finally:
        cudart.cudaStreamDestroy(stream)
        for pointer in device.values():
            cudart.cudaFree(pointer)
    ordered_ids = [item[0] for item in images]
    for path in (args.out, args.input_record, args.run_record):
        Path(path).parent.mkdir(parents=True, exist_ok=True)
    Path(args.out).write_text(json.dumps(predictions) + "\n", encoding="utf-8")
    input_record = {"schema_version": 1, "condition_id": args.condition_id, "image_ids": ordered_ids, "image_ids_sha256": ids_sha256(ordered_ids), "input_manifest_sha256": manifest_hash}
    Path(args.input_record).write_text(json.dumps(input_record, indent=2) + "\n", encoding="utf-8")
    source_root = Path(__file__).resolve().parent / "topic_c"
    run_record = {"schema_version": 2, "condition_id": args.condition_id, "dataset": args.dataset, "split": args.split, "model": args.model, "precision": args.precision, "calibrator": args.calibrator, "calibration_list_path": str(Path(args.calibration_list).resolve()) if args.calibration_list else None, "calibration_sha256": calibration_sha256(args.calibration_list) if args.calibration_list else None, "calibration_method": args.calibration_method, "calibration_provenance": args.calibration_provenance, "corruption": args.corruption, "severity": args.severity, "engine_path": str(Path(args.engine).resolve()), "engine_sha256": sha256_file(args.engine), "runner_sha256": sha256_file(__file__), "preprocess_sha256": sha256_file(source_root / "coco_data.py"), "decoder_sha256": sha256_file(source_root / "yolo_decode.py"), "class_map_path": str(class_map_path.resolve()) if class_map_path else None, "class_map_sha256": sha256_file(class_map_path) if class_map_path else map_hash, "annotation_sha256": sha256_file(args.annotations), "input_manifest_sha256": manifest_hash, "input_image_ids_sha256": input_record["image_ids_sha256"], "prediction_sha256": sha256_file(args.out), "n_images": len(images), "n_detections": len(predictions), "runtime_seconds": time.time() - started, "runtime_environment": runtime_environment(trt.__version__), "command": " ".join(os.sys.argv)}
    Path(args.run_record).write_text(json.dumps(run_record, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"predictions": len(predictions), "images": len(images), "manifest_sha256": manifest_hash}, indent=2))


if __name__ == "__main__":
    main()
