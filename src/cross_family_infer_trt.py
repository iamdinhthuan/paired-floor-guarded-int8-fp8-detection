#!/usr/bin/env python3
"""Hash-bound TensorRT inference for RT-DETR and raw-head RetinaNet engines."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from pathlib import Path

import numpy as np

from topic_c.coco_data import load_coco_images, preprocess
from topic_c.cross_family import decode_retinanet, decode_rtdetr, preprocess_retinanet
from topic_c.manifest import read_manifest, sha256_file


def complete_json(path: Path) -> dict:
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.is_file() or not marker.is_file() or marker.read_text().strip() != sha256_file(path):
        raise SystemExit(f"CROSS-FAMILY INFERENCE REFUSED: incomplete registry: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def ids_sha256(values: list[int]) -> str:
    return hashlib.sha256(json.dumps(values, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--engine-registry", required=True)
    parser.add_argument("--annotations", required=True)
    source = parser.add_mutually_exclusive_group(required=True)
    source.add_argument("--image-root")
    source.add_argument("--image-manifest")
    parser.add_argument("--manifest-cache-root")
    parser.add_argument("--out", required=True); parser.add_argument("--input-record", required=True)
    parser.add_argument("--run-record", required=True); parser.add_argument("--condition-id", required=True)
    parser.add_argument("--dataset", required=True); parser.add_argument("--split", required=True)
    parser.add_argument("--corruption", default="clean"); parser.add_argument("--severity", type=int, default=0)
    parser.add_argument("--confidence", type=float); parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    destinations = [Path(args.out), Path(args.input_record), Path(args.run_record)]
    if any(path.exists() for path in destinations):
        raise SystemExit("CROSS-FAMILY INFERENCE REFUSED: output already exists")
    registry_path = Path(args.engine_registry).resolve(); record = complete_json(registry_path)
    engine_path = Path(record["engine"]).resolve()
    if not engine_path.is_file() or sha256_file(engine_path) != record["engine_sha256"]:
        raise SystemExit("CROSS-FAMILY INFERENCE REFUSED: engine hash mismatch")
    decoder = record.get("decoder")
    if decoder not in {"ultralytics_rtdetr_raw_v1", "torchvision_retinanet_raw_v1"}:
        raise SystemExit("CROSS-FAMILY INFERENCE REFUSED: unsupported decoder contract")
    imgsz = int(record["imgsz"]); confidence = args.confidence
    if confidence is None: confidence = 0.001 if decoder.startswith("ultralytics") else 0.05
    if args.image_manifest and not args.manifest_cache_root:
        raise SystemExit("--manifest-cache-root is required with --image-manifest")
    manifest_sha = f"clean-root:{sha256_file(args.annotations)}"
    if args.image_manifest:
        manifest = read_manifest(args.image_manifest)
        marker = Path(args.image_manifest + ".complete")
        if not marker.is_file() or marker.read_text().strip() != manifest["manifest_sha256"]:
            raise SystemExit("CROSS-FAMILY INFERENCE REFUSED: invalid image manifest")
        manifest_sha = manifest["manifest_sha256"]
    images = load_coco_images(args.annotations, image_root=args.image_root, image_manifest=args.image_manifest,
                              manifest_cache_root=args.manifest_cache_root, limit=args.limit)
    annotations = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    categories = sorted(annotations["categories"], key=lambda value: value["id"])
    class_to_category = [value["id"] for value in categories]
    import tensorrt as trt
    try: import cuda.bindings.runtime as cudart
    except ImportError: import cuda.cudart as cudart
    logger = trt.Logger(trt.Logger.ERROR)
    with engine_path.open("rb") as handle, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(handle.read())
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(index) for index in range(engine.num_io_tensors)]
    inputs = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.INPUT]
    outputs = [name for name in names if engine.get_tensor_mode(name) == trt.TensorIOMode.OUTPUT]
    if len(inputs) != 1 or len(outputs) != len(record["output_names"]):
        raise SystemExit("CROSS-FAMILY INFERENCE REFUSED: engine IO mismatch")
    input_name = inputs[0]; context.set_input_shape(input_name, (1, 3, imgsz, imgsz))
    host, device = {}, {}
    for name in names:
        shape = tuple(context.get_tensor_shape(name))
        host[name] = np.empty(shape, dtype=trt.nptype(engine.get_tensor_dtype(name)))
        error, device[name] = cudart.cudaMalloc(host[name].nbytes)
        if error != cudart.cudaError_t.cudaSuccess: raise RuntimeError(f"cudaMalloc failed: {name}")
        context.set_tensor_address(name, device[name])
    error, stream = cudart.cudaStreamCreate()
    if error != cudart.cudaError_t.cudaSuccess: raise RuntimeError("cudaStreamCreate failed")
    predictions, started = [], time.time()
    try:
        for ordinal, (image_id, path) in enumerate(images, start=1):
            if decoder.startswith("ultralytics"):
                tensor, gain, padx, pady, width, height = preprocess(path, imgsz)
                decode_args = (gain, padx, pady, width, height)
            else:
                tensor, scale, resized_w, resized_h, width, height = preprocess_retinanet(path, imgsz)
                decode_args = (scale, resized_w, resized_h, width, height)
            np.copyto(host[input_name], tensor, casting="same_kind")
            cudart.cudaMemcpyAsync(device[input_name], host[input_name].ctypes.data, host[input_name].nbytes,
                                   cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
            if not context.execute_async_v3(stream_handle=stream): raise RuntimeError("TensorRT execution failed")
            for name in outputs:
                cudart.cudaMemcpyAsync(host[name].ctypes.data, device[name], host[name].nbytes,
                                       cudart.cudaMemcpyKind.cudaMemcpyDeviceToHost, stream)
            cudart.cudaStreamSynchronize(stream)
            if decoder.startswith("ultralytics"):
                rows = decode_rtdetr(host[outputs[0]], confidence, *decode_args)
            else:
                by_name = {name: host[name] for name in outputs}
                rows = decode_retinanet(by_name["cls_logits"], by_name["bbox_regression"], by_name["anchors"],
                                        confidence, *decode_args)
            for x1, y1, x2, y2, score, label in rows:
                if not 0 <= label < len(class_to_category): raise RuntimeError(f"class out of range: {label}")
                predictions.append({"image_id": image_id, "category_id": class_to_category[label],
                                    "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
            if ordinal % 100 == 0: print(f"{ordinal}/{len(images)}", flush=True)
    finally:
        cudart.cudaStreamDestroy(stream)
        for pointer in device.values(): cudart.cudaFree(pointer)
    ordered_ids = [value[0] for value in images]
    for path in destinations: path.parent.mkdir(parents=True, exist_ok=True)
    destinations[0].write_text(json.dumps(predictions) + "\n")
    input_record = {"schema_version": 1, "condition_id": args.condition_id, "image_ids": ordered_ids,
                    "image_ids_sha256": ids_sha256(ordered_ids), "input_manifest_sha256": manifest_sha}
    destinations[1].write_text(json.dumps(input_record, indent=2) + "\n")
    run = {"schema_version": 1, "condition_id": args.condition_id, "dataset": args.dataset, "split": args.split,
           "model": record["model"], "precision": record["precision"], "decoder": decoder,
           "decoder_sha256": sha256_file(Path(__file__).parent / "topic_c" / "cross_family.py"),
           "engine_registry": str(registry_path), "engine_registry_sha256": sha256_file(registry_path),
           "engine_sha256": record["engine_sha256"], "corruption": args.corruption, "severity": args.severity,
           "annotation_sha256": sha256_file(args.annotations),
           "confidence": confidence, "input_manifest_sha256": manifest_sha,
           "input_image_ids_sha256": input_record["image_ids_sha256"], "prediction_sha256": sha256_file(destinations[0]),
           "n_images": len(images), "n_detections": len(predictions), "runtime_seconds": time.time() - started}
    destinations[2].write_text(json.dumps(run, indent=2) + "\n")
    print(json.dumps({"images": len(images), "detections": len(predictions), "runtime_seconds": run["runtime_seconds"]}))


if __name__ == "__main__": main()
