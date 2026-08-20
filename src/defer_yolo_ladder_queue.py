#!/usr/bin/env python3
"""Wait for all training, then build hash-bound ONNX and TensorRT ladders serially."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from topic_c.manifest import sha256_file


def canonical_hash(document: dict, excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def valid_training_queue(path: Path) -> bool:
    if not path.is_file():
        return False
    try:
        data = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return data.get("queue_report_sha256") == canonical_hash(data, "queue_report_sha256")


def valid_complete(path: Path) -> bool:
    marker = path.with_suffix(path.suffix + ".complete")
    return path.is_file() and marker.is_file() and marker.read_text(encoding="utf-8").strip() == sha256_file(path)


def valid_calibration(path: Path, dataset: str) -> bool:
    """Validate the self-hashed calibration-list contract.

    Calibration lists embed ``calibration_sha256`` over their canonical JSON
    payload and store that same digest in the completion marker.  Other ladder
    registries use a whole-file digest, so they must continue through
    ``valid_complete`` instead.
    """
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.is_file() or not marker.is_file():
        return False
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    records = document.get("records")
    digest = document.get("calibration_sha256")
    if not isinstance(records, list) or len(records) != 512:
        return False
    source_paths = [record.get("source_relpath") for record in records if isinstance(record, dict)]
    if len(source_paths) != 512 or len(set(source_paths)) != 512 or any(not path for path in source_paths):
        return False
    return (
        document.get("schema_version") == 1
        and document.get("dataset") == dataset
        and document.get("split") == "train"
        and document.get("n_images") == 512
        and isinstance(digest, str)
        and len(digest) == 64
        and digest == canonical_hash(document, "calibration_sha256")
        and marker.read_text(encoding="utf-8").strip() == digest
    )


def run(command: list[str]) -> None:
    print("LADDER COMMAND " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def stage(root: Path, config: dict, dataset: str, model: str, precision: str, calibration: Path, imgsz: int) -> tuple[Path, Path]:
    onnx_dir = root / "engines" / dataset / model
    onnx_registry_dir = root / "manifests" / "onnx"
    engine_registry_dir = root / "manifests" / "engines"
    training = root / "manifests" / "training" / f"{dataset}_{model}_train_v1.json"
    fp32_onnx = onnx_dir / "fp32.onnx"
    fp32_registry = onnx_registry_dir / f"{dataset}_{model}_fp32_v1.json"
    if not valid_complete(fp32_registry):
        if fp32_registry.exists() or fp32_onnx.exists() or fp32_onnx.with_suffix(".pt").exists():
            raise SystemExit(f"LADDER REFUSED: partial FP32 export: {dataset}/{model}")
        run([sys.executable, str(root / "src" / "export_yolo_onnx.py"), "--training-registry", str(training),
             "--imgsz", str(imgsz), "--out", str(fp32_onnx), "--registry-out", str(fp32_registry)])
    if not valid_complete(fp32_registry):
        raise SystemExit(f"LADDER REFUSED: FP32 registry invalid: {fp32_registry}")
    if precision == "fp32":
        source_registry = fp32_registry
    else:
        quant_onnx = onnx_dir / f"{precision}.onnx"
        source_registry = onnx_registry_dir / f"{dataset}_{model}_{precision}_v1.json"
        if not valid_complete(source_registry):
            if source_registry.exists() or quant_onnx.exists():
                raise SystemExit(f"LADDER REFUSED: partial {precision} quantization: {dataset}/{model}")
            command = [sys.executable, str(root / "src" / "quantize_yolo_onnx.py"), "--onnx-registry", str(fp32_registry),
                       "--mode", precision, "--imgsz", str(imgsz), "--out", str(quant_onnx), "--registry-out", str(source_registry)]
            if precision != "fp16":
                command += ["--calibration-list", str(calibration)]
            run(command)
    if not valid_complete(source_registry):
        raise SystemExit(f"LADDER REFUSED: ONNX registry invalid: {source_registry}")
    engine = onnx_dir / f"{precision}.plan"
    engine_registry = engine_registry_dir / f"{dataset}_{model}_{precision}_v1.json"
    if not valid_complete(engine_registry):
        if engine_registry.exists() or engine.exists():
            raise SystemExit(f"LADDER REFUSED: partial {precision} engine build: {dataset}/{model}")
        run([sys.executable, str(root / "src" / "build_yolo_trt_engine.py"), "--onnx-registry", str(source_registry),
             "--precision", precision, "--trt-root", config["trt_root"], "--engine", str(engine),
             "--build-log", str(root / "outputs" / "logs" / "engines" / f"{dataset}_{model}_{precision}_v1.log"),
             "--registry-out", str(engine_registry), "--workspace", config["workspace"]])
    if not valid_complete(engine_registry):
        raise SystemExit(f"LADDER REFUSED: engine registry invalid: {engine_registry}")
    return source_registry, engine_registry


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--wait-for-training-report", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root, prerequisite, config_path, report = (Path(args.project_root).resolve(), Path(args.wait_for_training_report).resolve(),
                                                Path(args.config).resolve(), Path(args.report_out).resolve())
    if report.exists():
        raise SystemExit(f"LADDER REFUSED: report already exists: {report}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    if config.get("models") != ["yolo11n", "yolo11m", "yolo11x"] or config.get("precisions") != ["fp32", "fp16", "int8-entropy", "fp8"]:
        raise SystemExit("LADDER REFUSED: unexpected frozen ladder matrix")
    while not valid_training_queue(prerequisite):
        print(f"LADDER waiting for hash-valid training report: {prerequisite}", flush=True)
        time.sleep(args.poll_seconds)
    records = []
    for data in config["datasets"]:
        dataset, imgsz = data["dataset"], int(data["imgsz"])
        calibration = (root / data["calibration_list"]).resolve()
        if not valid_calibration(calibration, dataset):
            raise SystemExit(f"LADDER REFUSED: calibration list incomplete: {calibration}")
        for model in config["models"]:
            training = root / "manifests" / "training" / f"{dataset}_{model}_train_v1.json"
            if not valid_complete(training):
                raise SystemExit(f"LADDER REFUSED: training registry incomplete: {training}")
            for precision in config["precisions"]:
                onnx_registry, engine_registry = stage(root, config, dataset, model, precision, calibration, imgsz)
                records.append({"dataset": dataset, "model": model, "precision": precision,
                                "onnx_registry_sha256": sha256_file(onnx_registry), "engine_registry_sha256": sha256_file(engine_registry)})
        registry = root / "manifests" / "engines" / f"{dataset}_yolo11_nmx_ladder_v1.json"
        if not valid_complete(registry):
            if registry.exists():
                raise SystemExit(f"LADDER REFUSED: partial frozen dataset engine registry: {registry}")
            run([sys.executable, str(root / "src" / "freeze_yolo_engine_registry.py"), "--project-root", str(root),
                 "--dataset", dataset, "--out", str(registry)])
    document = {"schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                "config": str(config_path), "config_sha256": sha256_file(config_path), "training_report": str(prerequisite),
                "training_report_sha256": sha256_file(prerequisite), "artifacts": records}
    document["ladder_report_sha256"] = canonical_hash(document, "ladder_report_sha256")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"LADDER COMPLETE artifacts={len(records)} report={report}")


if __name__ == "__main__":
    main()
