#!/usr/bin/env python3
"""Run clean YOLO reference inference through PyTorch or ONNX Runtime CUDA.

Both backends use the same manifest bytes, letterbox preprocessing, decoder,
class map, and output schema as the TensorRT runner. ONNX Runtime is pinned to
CUDAExecutionProvider as its primary provider and runtime fallback is disabled.
ORT may still register CPUExecutionProvider for shape-only bookkeeping; the
registered-provider list is retained in the run record.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import platform
import subprocess
import time
from pathlib import Path
from typing import Callable

import numpy as np

from topic_c.coco_data import load_coco_images, preprocess
from topic_c.manifest import read_manifest, sha256_file
from topic_c.yolo_decode import CONF_FLOOR, decode


def ids_sha256(ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()


def normalize_yolo_output(output: object) -> np.ndarray:
    if isinstance(output, (tuple, list)):
        if not output:
            raise RuntimeError("backend did not return a canonical YOLO output")
        output = output[0]
    if isinstance(output, dict):
        raise RuntimeError("backend did not return a canonical YOLO output")
    if hasattr(output, "detach"):
        output = output.detach()
    if hasattr(output, "cpu"):
        output = output.cpu()
    if hasattr(output, "numpy"):
        output = output.numpy()
    values = np.asarray(output)
    if values.ndim != 3 or values.shape[0] != 1 or values.shape[1] <= 4:
        raise RuntimeError(f"backend did not return a canonical YOLO output: {values.shape}")
    if not np.issubdtype(values.dtype, np.floating) or not np.isfinite(values).all():
        raise RuntimeError("backend did not return a finite canonical YOLO output")
    return values


def ort_providers(available: list[str]) -> list[str]:
    if "CUDAExecutionProvider" not in available:
        raise SystemExit("REFERENCE INFERENCE REFUSED: ONNX Runtime CUDAExecutionProvider is unavailable")
    return ["CUDAExecutionProvider"]


def validate_ort_session_providers(assigned: list[str]) -> list[str]:
    if not assigned or assigned[0] != "CUDAExecutionProvider":
        raise SystemExit(
            "REFERENCE INFERENCE REFUSED: CUDA is not the primary ONNX Runtime provider"
        )
    return assigned


def load_export_registry(path: Path, dataset: str, model: str) -> tuple[dict, dict[str, Path]]:
    path = path.resolve()
    marker = path.with_suffix(path.suffix + ".complete")
    if (
        not path.is_file()
        or not marker.is_file()
        or marker.read_text(encoding="utf-8").strip() != sha256_file(path)
    ):
        raise SystemExit("REFERENCE INFERENCE REFUSED: incomplete ONNX export registry")
    document = json.loads(path.read_text(encoding="utf-8"))
    if document.get("dataset") != dataset or document.get("model") != model:
        raise SystemExit("REFERENCE INFERENCE REFUSED: export registry identity mismatch")
    checkpoint = Path(document.get("source_checkpoint", "")).resolve()
    onnx = Path(document.get("onnx", "")).resolve()
    if not checkpoint.is_file() or sha256_file(checkpoint) != document.get("source_checkpoint_sha256"):
        raise SystemExit("REFERENCE INFERENCE REFUSED: checkpoint hash mismatch")
    if not onnx.is_file() or sha256_file(onnx) != document.get("onnx_sha256"):
        raise SystemExit("REFERENCE INFERENCE REFUSED: ONNX hash mismatch")
    return document, {"pytorch": checkpoint, "onnxruntime": onnx}


def build_run_record(
    *,
    condition_id: str,
    dataset: str,
    split: str,
    model: str,
    backend: str,
    model_artifact: Path,
    export_registry: Path,
    annotation_sha256: str,
    manifest_sha256: str,
    image_ids_sha256: str,
    prediction_sha256: str,
    n_images: int,
    n_detections: int,
    runtime_seconds: float,
    runtime: dict,
    class_map_sha256: str,
) -> dict:
    return {
        "schema_version": 1,
        "condition_id": condition_id,
        "dataset": dataset,
        "split": split,
        "model": model,
        "precision": "fp32-reference",
        "backend": backend,
        "calibrator": "none",
        "corruption": "clean",
        "severity": 0,
        "model_artifact": str(model_artifact.resolve()),
        "model_artifact_sha256": sha256_file(model_artifact),
        "export_registry": str(export_registry.resolve()),
        "export_registry_sha256": sha256_file(export_registry),
        "annotation_sha256": annotation_sha256,
        "input_manifest_sha256": manifest_sha256,
        "input_image_ids_sha256": image_ids_sha256,
        "prediction_sha256": prediction_sha256,
        "class_map_sha256": class_map_sha256,
        "n_images": n_images,
        "n_detections": n_detections,
        "runtime_seconds": runtime_seconds,
        "runtime_environment": runtime,
    }


def class_mapping(path: Path, dataset: str, split: str, annotation: Path) -> tuple[list[int], str]:
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in document.items() if key != "class_map_sha256"}
    canonical = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    mapping = document.get("class_to_category_id")
    if (
        document.get("class_map_sha256") != canonical
        or document.get("dataset") != dataset
        or document.get("split") != split
        or document.get("annotation_sha256") != sha256_file(annotation)
        or not isinstance(mapping, list)
        or not mapping
        or not all(isinstance(item, int) and item > 0 for item in mapping)
    ):
        raise SystemExit("REFERENCE INFERENCE REFUSED: invalid class map")
    return mapping, sha256_file(path)


def runtime_environment(backend: str, version: str, detail: dict) -> dict:
    try:
        query = subprocess.run(
            ["nvidia-smi", "--query-gpu=name,driver_version,uuid", "--format=csv,noheader"],
            check=True,
            capture_output=True,
            text=True,
        )
        gpu = [line.strip() for line in query.stdout.splitlines() if line.strip()]
    except (OSError, subprocess.CalledProcessError):
        gpu = []
    return {
        "python_executable": os.sys.executable,
        "python_version": platform.python_version(),
        "backend": backend,
        "backend_version": version,
        "gpu": gpu,
        **detail,
    }


def pytorch_backend(path: Path) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    import torch
    import ultralytics
    from ultralytics import YOLO

    if not torch.cuda.is_available():
        raise SystemExit("REFERENCE INFERENCE REFUSED: PyTorch CUDA is unavailable")
    model = YOLO(str(path)).model.to("cuda:0").eval()

    def infer(tensor: np.ndarray) -> np.ndarray:
        values = torch.from_numpy(tensor).to("cuda:0")
        with torch.inference_mode():
            return normalize_yolo_output(model(values))

    detail = runtime_environment(
        "pytorch",
        torch.__version__,
        {"ultralytics_version": ultralytics.__version__, "device": "cuda:0"},
    )
    return infer, detail


def onnxruntime_backend(path: Path, input_name: str, output_name: str) -> tuple[Callable[[np.ndarray], np.ndarray], dict]:
    import onnxruntime as ort

    providers = ort_providers(ort.get_available_providers())
    session = ort.InferenceSession(str(path), providers=providers)
    session.disable_fallback()
    assigned = validate_ort_session_providers(session.get_providers())
    if [item.name for item in session.get_inputs()] != [input_name] or [item.name for item in session.get_outputs()] != [output_name]:
        raise SystemExit("REFERENCE INFERENCE REFUSED: ONNX Runtime IO differs from export registry")

    def infer(tensor: np.ndarray) -> np.ndarray:
        return normalize_yolo_output(session.run([output_name], {input_name: tensor}))

    detail = runtime_environment(
        "onnxruntime",
        ort.__version__,
        {
            "requested_providers": providers,
            "registered_providers": assigned,
            "runtime_fallback_disabled": True,
            "input_name": input_name,
            "output_name": output_name,
        },
    )
    return infer, detail


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--backend", choices=("pytorch", "onnxruntime"), required=True)
    parser.add_argument("--onnx-registry", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image-manifest", required=True)
    parser.add_argument("--manifest-cache-root", required=True)
    parser.add_argument("--class-map", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--input-record", required=True)
    parser.add_argument("--run-record", required=True)
    parser.add_argument("--condition-id", required=True)
    parser.add_argument("--dataset", choices=("voc", "kitti", "tt100k"), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--model", choices=("yolo11n", "yolo11m", "yolo11x"), required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--conf", type=float, default=CONF_FLOOR)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    output, input_record, run_record = map(
        lambda value: Path(value).resolve(), (args.out, args.input_record, args.run_record)
    )
    existing = [str(path) for path in (output, input_record, run_record) if path.exists()]
    if existing:
        raise SystemExit("REFERENCE INFERENCE REFUSED: refusing to overwrite: " + ", ".join(existing))
    annotation = Path(args.annotations).resolve()
    manifest_path = Path(args.image_manifest).resolve()
    manifest = read_manifest(manifest_path)
    marker = manifest_path.with_suffix(manifest_path.suffix + ".complete")
    if (
        not marker.is_file()
        or marker.read_text(encoding="utf-8").strip() != manifest.get("manifest_sha256")
        or manifest.get("dataset") != args.dataset
        or manifest.get("split") != args.split
        or any(
            record.get("corruption") != "clean" or record.get("severity") != 0
            for record in manifest.get("records", [])
        )
    ):
        raise SystemExit("REFERENCE INFERENCE REFUSED: invalid clean manifest")
    mapping, class_map_sha = class_mapping(
        Path(args.class_map).resolve(), args.dataset, args.split, annotation
    )
    export_path = Path(args.onnx_registry).resolve()
    export, artifacts = load_export_registry(export_path, args.dataset, args.model)
    artifact = artifacts[args.backend]
    if args.backend == "pytorch":
        infer, runtime = pytorch_backend(artifact)
    else:
        infer, runtime = onnxruntime_backend(
            artifact,
            export.get("input_name", "images"),
            export.get("output_name", "output0"),
        )
    images = load_coco_images(
        annotation,
        image_manifest=manifest_path,
        manifest_cache_root=args.manifest_cache_root,
        limit=args.limit,
    )
    predictions: list[dict] = []
    started = time.time()
    for number, (image_id, image_path) in enumerate(images, start=1):
        tensor, gain, padx, pady, width, height = preprocess(image_path, args.imgsz)
        raw = infer(tensor)
        for x1, y1, x2, y2, score, class_id in decode(
            raw, args.conf, gain, padx, pady, width, height
        ):
            if not 0 <= class_id < len(mapping):
                raise SystemExit("REFERENCE INFERENCE REFUSED: decoded class outside class map")
            predictions.append(
                {
                    "image_id": image_id,
                    "category_id": mapping[class_id],
                    "bbox": [float(x1), float(y1), float(x2 - x1), float(y2 - y1)],
                    "score": score,
                }
            )
        if number % 100 == 0:
            elapsed = max(time.time() - started, 1e-9)
            print(f"REFERENCE {args.backend} {number}/{len(images)} {number / elapsed:.2f} image/s", flush=True)
    elapsed = time.time() - started
    ids = [image_id for image_id, _ in images]
    for path in (output, input_record, run_record):
        path.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(predictions) + "\n", encoding="utf-8")
    input_document = {
        "schema_version": 1,
        "condition_id": args.condition_id,
        "image_ids": ids,
        "image_ids_sha256": ids_sha256(ids),
        "input_manifest_sha256": manifest["manifest_sha256"],
    }
    input_record.write_text(json.dumps(input_document, indent=2) + "\n", encoding="utf-8")
    record = build_run_record(
        condition_id=args.condition_id,
        dataset=args.dataset,
        split=args.split,
        model=args.model,
        backend=args.backend,
        model_artifact=artifact,
        export_registry=export_path,
        annotation_sha256=sha256_file(annotation),
        manifest_sha256=manifest["manifest_sha256"],
        image_ids_sha256=input_document["image_ids_sha256"],
        prediction_sha256=sha256_file(output),
        n_images=len(images),
        n_detections=len(predictions),
        runtime_seconds=elapsed,
        runtime=runtime,
        class_map_sha256=class_map_sha,
    )
    source_root = Path(__file__).resolve().parent / "topic_c"
    record.update(
        {
            "runner_sha256": sha256_file(__file__),
            "preprocess_sha256": sha256_file(source_root / "coco_data.py"),
            "decoder_sha256": sha256_file(source_root / "yolo_decode.py"),
            "command": list(os.sys.argv),
        }
    )
    run_record.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "backend": args.backend,
                "images": len(images),
                "predictions": len(predictions),
                "prediction_sha256": record["prediction_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
