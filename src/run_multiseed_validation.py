#!/usr/bin/env python3
"""Derive and execute the frozen YOLO11m training/calibration seed validation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import shutil
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor
from datetime import datetime, timezone
from itertools import product
from pathlib import Path
from typing import Any, Callable


ATTEMPT = "ivc_multiseed_yolo11m_s5_v1"
TRAINING_SEEDS = [20260807, 20260813, 20260814]
NEW_TRAINING_SEEDS = [20260813, 20260814]
CALIBRATION_SEEDS = [20260807, 20260813, 20260814]
DATASET_BATCH = {"voc": 16, "kitti": 16, "tt100k": 4}
FORMATS = ["int8-entropy", "fp8"]
CORRUPTIONS = ["gaussian_noise", "motion_blur", "fog", "jpeg"]
TRTEXEC_SHA256 = "68b061d276601b7c6d8aafa4d8d75319f32382c85673360407a6d7ee6411aa4d"
MAX_EVIDENCE_WORKERS = 8


def canonical_hash(document: dict[str, Any], excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _relative_path(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value:
        raise ValueError(f"{field} must be a non-empty relative path")
    path = Path(value)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{field} escapes the project root")
    return path.as_posix()


def load_config(path: Path) -> dict[str, Any]:
    document = json.loads(Path(path).read_text(encoding="utf-8"))
    if not isinstance(document, dict) or document.get("schema_version") != 1:
        raise ValueError("multi-seed configuration schema is invalid")
    if document.get("config_sha256") != canonical_hash(document, "config_sha256"):
        raise ValueError("multi-seed configuration SHA-256 mismatch")
    if document.get("attempt") != ATTEMPT or document.get("model") != "yolo11m":
        raise ValueError("multi-seed attempt/model is not frozen")
    if document.get("training_seeds") != TRAINING_SEEDS:
        raise ValueError("multi-seed training seeds are not frozen")
    if document.get("new_training_seeds") != NEW_TRAINING_SEEDS:
        raise ValueError("multi-seed new training seeds are not frozen")
    if document.get("calibration_seeds") != CALIBRATION_SEEDS:
        raise ValueError("multi-seed calibration seeds are not frozen")
    if document.get("formats") != FORMATS or document.get("corruptions") != CORRUPTIONS:
        raise ValueError("multi-seed format/corruption grid is not frozen")
    if document.get("severity") != 5 or document.get("calibration_images") != 512:
        raise ValueError("multi-seed severity/calibration count is not frozen")
    datasets = document.get("datasets")
    if not isinstance(datasets, list) or [item.get("dataset") for item in datasets] != ["voc", "kitti", "tt100k"]:
        raise ValueError("multi-seed dataset order is not frozen")
    required_paths = (
        "annotations", "class_map", "data_yaml", "acquisition_registry",
        "baseline_training_registry", "baseline_calibration_manifest",
        "clean_manifest", "clean_cache_root", "corruption_cache_root",
    )
    for item in datasets:
        dataset = item["dataset"]
        if item.get("batch") != DATASET_BATCH[dataset] or item["batch"] <= 0:
            raise ValueError(f"multi-seed batch is not frozen for {dataset}")
        if not isinstance(item.get("imgsz"), int) or item["imgsz"] <= 0:
            raise ValueError(f"multi-seed image size is invalid for {dataset}")
        if not isinstance(item.get("expected_images"), int) or item["expected_images"] <= 0:
            raise ValueError(f"multi-seed image count is invalid for {dataset}")
        for field in required_paths:
            item[field] = _relative_path(item.get(field), f"{dataset}.{field}")
    if not isinstance(document.get("minimum_free_gib"), int) or document["minimum_free_gib"] < 20:
        raise ValueError("multi-seed free-disk gate must be at least 20 GiB")
    return document


def _training_registry(config: dict[str, Any], dataset: dict[str, Any], seed: int) -> str:
    if seed == TRAINING_SEEDS[0]:
        return dataset["baseline_training_registry"]
    run_id = f"{config['attempt']}__{dataset['dataset']}__yolo11m__ts{seed}"
    return f"manifests/training/{run_id}.json"


def _calibration_manifest(config: dict[str, Any], dataset: dict[str, Any], seed: int) -> str:
    if seed == CALIBRATION_SEEDS[0]:
        return dataset["baseline_calibration_manifest"]
    return (
        f"manifests/calibration/{dataset['dataset']}_train_clean_512_"
        f"s{seed}_{config['attempt']}.json"
    )


def derive_jobs(config: dict[str, Any], project_root: Path) -> dict[str, list[dict[str, Any]]]:
    root = Path(project_root).resolve()
    jobs: dict[str, list[dict[str, Any]]] = {
        stage: []
        for stage in (
            "training", "calibration", "export", "quantization", "engine",
            "inference", "metric", "direct_cell",
        )
    }
    attempt = config["attempt"]
    for dataset in config["datasets"]:
        name = dataset["dataset"]
        for seed in config["new_training_seeds"]:
            run_id = f"{attempt}__{name}__yolo11m__ts{seed}"
            jobs["training"].append({
                "job_id": f"training__{name}__ts{seed}",
                "dataset": name,
                "training_seed": seed,
                "batch": dataset["batch"],
                "run_id": run_id,
                "registry": str(root / "manifests" / "training" / f"{run_id}.json"),
            })
        for calibration_seed in config["calibration_seeds"]:
            jobs["calibration"].append({
                "job_id": f"calibration__{name}__cs{calibration_seed}",
                "dataset": name,
                "calibration_seed": calibration_seed,
                "manifest": str(root / _calibration_manifest(config, dataset, calibration_seed)),
                "existing": calibration_seed == CALIBRATION_SEEDS[0],
            })
        for training_seed in config["training_seeds"]:
            training_registry = _training_registry(config, dataset, training_seed)
            export_id = f"{name}__yolo11m__ts{training_seed}"
            fp32_registry = f"manifests/onnx/{attempt}/{export_id}__fp32.json"
            jobs["export"].append({
                "job_id": f"export__{name}__ts{training_seed}",
                "dataset": name,
                "training_seed": training_seed,
                "training_registry": str(root / training_registry),
                "registry": str(root / fp32_registry),
            })
            for calibration_seed, precision in product(config["calibration_seeds"], config["formats"]):
                stem = f"{export_id}__cs{calibration_seed}__{precision}"
                calibration_manifest = _calibration_manifest(config, dataset, calibration_seed)
                quant_registry = f"manifests/onnx/{attempt}/{stem}.json"
                engine_registry = f"manifests/engines/{attempt}/{stem}.json"
                common = {
                    "dataset": name,
                    "training_seed": training_seed,
                    "calibration_seed": calibration_seed,
                    "precision": precision,
                    "calibration_manifest": str(root / calibration_manifest),
                }
                jobs["quantization"].append({
                    "job_id": f"quantization__{name}__ts{training_seed}__cs{calibration_seed}__{precision}",
                    **common,
                    "source_registry": str(root / fp32_registry),
                    "registry": str(root / quant_registry),
                })
                jobs["engine"].append({
                    "job_id": f"engine__{name}__ts{training_seed}__cs{calibration_seed}__{precision}",
                    **common,
                    "source_registry": str(root / quant_registry),
                    "registry": str(root / engine_registry),
                })
                inputs = [("codec_control", 0, dataset["clean_manifest"])] + [
                    (
                        corruption,
                        config["severity"],
                        f"manifests/images/{dataset['manifest_prefix']}_full_{corruption}_s{config['severity']}.json",
                    )
                    for corruption in config["corruptions"]
                ]
                for corruption, severity, manifest in inputs:
                    condition = (
                        f"{name}__yolo11m__ts{training_seed}__cs{calibration_seed}__"
                        f"{precision}__{corruption}-s{severity}"
                    )
                    infer = {
                        "job_id": f"inference__{condition}",
                        **common,
                        "corruption": corruption,
                        "severity": severity,
                        "condition_id": condition,
                        "input_manifest": str(root / manifest),
                        "engine_registry": str(root / engine_registry),
                    }
                    jobs["inference"].append(infer)
                    jobs["metric"].append({
                        **infer,
                        "job_id": f"metric__{condition}",
                    })
        for training_seed, calibration_seed, corruption in product(
            config["training_seeds"], config["calibration_seeds"], config["corruptions"]
        ):
            jobs["direct_cell"].append({
                "job_id": f"direct_cell__{name}__ts{training_seed}__cs{calibration_seed}__{corruption}-s5",
                "dataset": name,
                "training_seed": training_seed,
                "calibration_seed": calibration_seed,
                "corruption": corruption,
                "severity": 5,
            })
    identifiers = [job["job_id"] for items in jobs.values() for job in items]
    if len(identifiers) != len(set(identifiers)):
        raise ValueError("multi-seed derived job IDs are not unique")
    expected = {
        "training": 6, "calibration": 9, "export": 9, "quantization": 54,
        "engine": 54, "inference": 270, "metric": 270, "direct_cell": 108,
    }
    if {stage: len(items) for stage, items in jobs.items()} != expected:
        raise ValueError("multi-seed derived grid has unexpected cardinality")
    return jobs


def validate_resource_snapshot(snapshot: dict[str, Any], *, minimum_free_gib: int) -> None:
    if snapshot.get("gpu_name") != "NVIDIA GeForce RTX 5090":
        raise RuntimeError("multi-seed preflight requires NVIDIA GeForce RTX 5090")
    minimum = minimum_free_gib * 1024**3
    if not isinstance(snapshot.get("disk_free_bytes"), int) or snapshot["disk_free_bytes"] < minimum:
        raise RuntimeError(f"multi-seed preflight requires at least {minimum_free_gib} GiB free disk")
    processes = snapshot.get("compute_processes")
    if not isinstance(processes, list):
        raise RuntimeError("multi-seed GPU compute process snapshot is invalid")
    if processes:
        rendered = ", ".join(f"{item.get('pid')}:{item.get('name')}" for item in processes)
        raise RuntimeError(f"multi-seed preflight found an unapproved GPU compute process: {rendered}")


def validate_trt_runtime(
    trt_root: Path, *, expected_sha256: str = TRTEXEC_SHA256
) -> dict[str, str]:
    root = Path(trt_root).resolve()
    executable = root / "bin" / "trtexec"
    library = root / "lib"
    if not executable.is_file() or not library.is_dir():
        raise RuntimeError(f"multi-seed TensorRT bin/lib layout unavailable: {root}")
    observed_sha256 = _sha256(executable)
    if observed_sha256 != expected_sha256:
        raise RuntimeError(
            "multi-seed trtexec SHA-256 mismatch: "
            f"expected {expected_sha256}, observed {observed_sha256}"
        )
    return {
        "trtexec": str(executable),
        "trtexec_sha256": observed_sha256,
        "trt_lib": str(library),
    }


def training_command(
    job: dict[str, Any], config: dict[str, Any], project_root: Path
) -> list[str]:
    root = Path(project_root).resolve()
    dataset = next(item for item in config["datasets"] if item["dataset"] == job["dataset"])
    seed = job["training_seed"]
    profile = root / "configs" / "training" / f"ivc_multiseed_yolo11m_s{seed}_v1.json"
    return [
        sys.executable,
        str(root / "src" / "train_yolo_dataset.py"),
        "--project-root", str(root),
        "--profile", str(profile),
        "--dataset", job["dataset"],
        "--model", "yolo11m",
        "--data-yaml", str(root / dataset["data_yaml"]),
        "--acquisition-registry", str(root / dataset["acquisition_registry"]),
        "--run-id", job["run_id"],
        "--registry-out", job["registry"],
    ]


def _dataset_config(config: dict[str, Any], dataset: str) -> dict[str, Any]:
    return next(item for item in config["datasets"] if item["dataset"] == dataset)


def calibration_command(
    job: dict[str, Any], config: dict[str, Any], project_root: Path
) -> list[str]:
    if job.get("existing"):
        raise ValueError("baseline calibration manifest has no creation command")
    root = Path(project_root).resolve()
    dataset = _dataset_config(config, job["dataset"])
    return [
        sys.executable,
        str(root / "src" / "build_train_calibration_list.py"),
        "--dataset", job["dataset"],
        "--data-yaml", str(root / dataset["data_yaml"]),
        "--acquisition-registry", str(root / dataset["acquisition_registry"]),
        "--n-images", str(config["calibration_images"]),
        "--seed", str(job["calibration_seed"]),
        "--out", job["manifest"],
    ]


def _export_onnx_path(job: dict[str, Any], config: dict[str, Any], root: Path) -> Path:
    return root / "outputs" / "onnx" / config["attempt"] / f"{Path(job['registry']).stem}.onnx"


def export_command(
    job: dict[str, Any], config: dict[str, Any], project_root: Path
) -> list[str]:
    root = Path(project_root).resolve()
    dataset = _dataset_config(config, job["dataset"])
    return [
        sys.executable,
        str(root / "src" / "export_yolo_onnx.py"),
        "--training-registry", job["training_registry"],
        "--imgsz", str(dataset["imgsz"]),
        "--out", str(_export_onnx_path(job, config, root)),
        "--registry-out", job["registry"],
    ]


def _quantized_onnx_path(job: dict[str, Any], config: dict[str, Any], root: Path) -> Path:
    return root / "outputs" / "onnx" / config["attempt"] / f"{Path(job['registry']).stem}.onnx"


def quantization_command(
    job: dict[str, Any], config: dict[str, Any], project_root: Path
) -> list[str]:
    root = Path(project_root).resolve()
    dataset = _dataset_config(config, job["dataset"])
    return [
        sys.executable,
        str(root / "src" / "quantize_yolo_onnx.py"),
        "--onnx-registry", job["source_registry"],
        "--mode", job["precision"],
        "--imgsz", str(dataset["imgsz"]),
        "--calibration-list", job["calibration_manifest"],
        "--out", str(_quantized_onnx_path(job, config, root)),
        "--registry-out", job["registry"],
    ]


def _engine_paths(
    job: dict[str, Any], config: dict[str, Any], root: Path
) -> tuple[Path, Path, Path]:
    stem = Path(job.get("source_registry", job["registry"])).stem
    engine = root / "outputs" / "engines" / config["attempt"] / f"{stem}.plan"
    log = root / "outputs" / "logs" / config["attempt"] / f"{stem}.build.log"
    registry = root / "manifests" / "engines" / config["attempt"] / f"{stem}.json"
    return engine, log, registry


def engine_command(
    job: dict[str, Any], config: dict[str, Any], project_root: Path
) -> list[str]:
    root = Path(project_root).resolve()
    source_registry = job.get("source_registry", job["registry"])
    engine, log, registry = _engine_paths(job, config, root)
    return [
        sys.executable,
        str(root / "src" / "build_yolo_trt_engine.py"),
        "--onnx-registry", source_registry,
        "--precision", job["precision"],
        "--trt-root", config["trt_root"],
        "--engine", str(engine),
        "--build-log", str(log),
        "--registry-out", str(registry),
        "--workspace", "4096M",
    ]


def _condition_paths(root: Path, attempt: str, condition_id: str) -> tuple[Path, Path, Path, Path]:
    return (
        root / "outputs" / "predictions" / attempt / f"{condition_id}.json",
        root / "outputs" / "inputs" / attempt / f"{condition_id}.json",
        root / "manifests" / "runs" / attempt / f"{condition_id}.json",
        root / "outputs" / "metrics" / attempt / f"{condition_id}.json",
    )


def inference_command(
    job: dict[str, Any], config: dict[str, Any], project_root: Path
) -> list[str]:
    root = Path(project_root).resolve()
    dataset = _dataset_config(config, job["dataset"])
    engine, _, _ = _engine_paths(
        {"registry": job["engine_registry"], "precision": job["precision"]},
        config,
        root,
    )
    prediction, inputs, run_record, _ = _condition_paths(
        root, config["attempt"], job["condition_id"]
    )
    cache = (
        dataset["clean_cache_root"]
        if job["corruption"] == "codec_control"
        else dataset["corruption_cache_root"]
    )
    command = [
        sys.executable,
        str(root / "src" / "coco_infer_trt.py"),
        "--engine", str(engine),
        "--annotations", str(root / dataset["annotations"]),
        "--image-manifest", job["input_manifest"],
        "--manifest-cache-root", str(root / cache),
        "--out", str(prediction),
        "--input-record", str(inputs),
        "--run-record", str(run_record),
        "--condition-id", job["condition_id"],
        "--dataset", job["dataset"],
        "--split", dataset["split"],
        "--model", "yolo11m",
        "--precision", job["precision"],
        "--calibrator", "entropy",
        "--calibration-list", job["calibration_manifest"],
        "--calibration-method", "entropy",
        "--calibration-provenance", "verified",
        "--corruption", job["corruption"],
        "--severity", str(job["severity"]),
        "--class-map", str(root / dataset["class_map"]),
        "--imgsz", str(dataset["imgsz"]),
    ]
    return command


def metric_command(
    job: dict[str, Any], config: dict[str, Any], project_root: Path
) -> list[str]:
    root = Path(project_root).resolve()
    dataset = _dataset_config(config, job["dataset"])
    prediction, inputs, run_record, metric = _condition_paths(
        root, config["attempt"], job["condition_id"]
    )
    evaluator = "tt100k_eval.py" if job["dataset"] == "tt100k" else "coco_eval.py"
    return [
        sys.executable,
        str(root / "src" / evaluator),
        "--annotations", str(root / dataset["annotations"]),
        "--predictions", str(prediction),
        "--input-record", str(inputs),
        "--run-record", str(run_record),
        "--out", str(metric),
    ]


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _complete_file(path: Path) -> bool:
    marker = path.with_suffix(path.suffix + ".complete")
    return path.is_file() and marker.is_file() and marker.read_text(encoding="utf-8").strip() == _sha256(path)


def _read_sha_complete(path: Path, label: str) -> dict[str, Any] | None:
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.exists() and not marker.exists():
        return None
    if not path.is_file() or not marker.is_file():
        raise RuntimeError(f"multi-seed {label} is partial: {path}")
    if marker.read_text(encoding="utf-8").strip() != _sha256(path):
        raise RuntimeError(f"multi-seed {label} completion hash mismatch: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"multi-seed {label} JSON is invalid: {path}") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"multi-seed {label} is not an object: {path}")
    return document


def validate_calibration_job(
    job: dict[str, Any],
    config: dict[str, Any],
    root: Path,
    *,
    verify_images: bool = True,
) -> bool:
    path = Path(job["manifest"])
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.exists() and not marker.exists():
        return False
    if not path.is_file() or not marker.is_file():
        raise RuntimeError(f"multi-seed calibration manifest is partial: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError(f"multi-seed calibration JSON is invalid: {path}") from error
    declared = document.get("calibration_sha256")
    if (
        not isinstance(declared, str)
        or len(declared) != 64
        or declared != canonical_hash(document, "calibration_sha256")
        or marker.read_text(encoding="utf-8").strip() != declared
    ):
        raise RuntimeError(f"multi-seed calibration canonical hash mismatch: {path}")
    if document.get("dataset") != job["dataset"]:
        raise RuntimeError(f"multi-seed calibration dataset mismatch: {path}")
    if document.get("seed") != job["calibration_seed"]:
        raise RuntimeError(f"multi-seed calibration seed mismatch: {path}")
    records = document.get("records")
    if (
        document.get("split") != "train"
        or document.get("n_images") != config["calibration_images"]
        or not isinstance(records, list)
        or len(records) != config["calibration_images"]
    ):
        raise RuntimeError(f"multi-seed calibration record grid mismatch: {path}")
    if verify_images:
        dataset_root = Path(document.get("dataset_root", ""))
        if not dataset_root.is_dir():
            raise RuntimeError(f"multi-seed calibration dataset root is absent: {path}")
        for record in records:
            relative = record.get("source_relpath") if isinstance(record, dict) else None
            expected = record.get("sha256") if isinstance(record, dict) else None
            if not isinstance(relative, str) or not isinstance(expected, str):
                raise RuntimeError(f"multi-seed calibration record is malformed: {path}")
            image = (dataset_root / relative).resolve()
            if dataset_root.resolve() not in image.parents or not image.is_file() or _sha256(image) != expected:
                raise RuntimeError(f"multi-seed calibration image hash mismatch: {image}")
    return True


def write_stage_report(
    path: Path,
    *,
    attempt: str,
    stage: str,
    config_sha256: str,
    artifacts: list[Path],
    root: Path,
) -> dict[str, Any]:
    path = Path(path)
    root = Path(root).resolve()
    if path.exists():
        raise RuntimeError(f"multi-seed stage report already exists: {path}")
    bindings: dict[str, str] = {}
    for artifact in artifacts:
        resolved = Path(artifact).resolve()
        if root not in resolved.parents or not resolved.is_file():
            raise RuntimeError(f"multi-seed stage artifact is absent or outside root: {artifact}")
        bindings[str(resolved.relative_to(root))] = _sha256(resolved)
    document = {
        "schema_version": 1,
        "status": "complete",
        "attempt": attempt,
        "stage": stage,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha256,
        "artifacts_sha256": dict(sorted(bindings.items())),
    }
    document["report_sha256"] = canonical_hash(document, "report_sha256")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def validate_stage_report(
    path: Path,
    *,
    attempt: str,
    stage: str,
    config_sha256: str,
    artifacts: list[Path],
    root: Path,
) -> dict[str, Any]:
    path = Path(path)
    root = Path(root).resolve()
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"multi-seed {stage} report is absent or invalid: {path}") from error
    if (
        document.get("status") != "complete"
        or document.get("attempt") != attempt
        or document.get("stage") != stage
        or document.get("config_sha256") != config_sha256
        or document.get("report_sha256") != canonical_hash(document, "report_sha256")
    ):
        raise RuntimeError(f"multi-seed {stage} report identity/hash mismatch: {path}")
    expected_paths = {str(Path(item).resolve().relative_to(root)) for item in artifacts}
    bindings = document.get("artifacts_sha256")
    if not isinstance(bindings, dict) or set(bindings) != expected_paths:
        raise RuntimeError(f"multi-seed {stage} report artifact grid mismatch: {path}")
    for relative, expected in bindings.items():
        artifact = root / relative
        if not artifact.is_file() or _sha256(artifact) != expected:
            raise RuntimeError(f"multi-seed {stage} artifact hash mismatch: {relative}")
    return document


def _required_preflight_paths(config: dict[str, Any], root: Path) -> list[Path]:
    paths = [
        root / "src" / name
        for name in (
            "run_multiseed_validation.py", "train_yolo_dataset.py",
            "build_train_calibration_list.py", "export_yolo_onnx.py",
            "quantize_yolo_onnx.py", "build_yolo_trt_engine.py",
            "coco_infer_trt.py", "validate_predictions.py", "coco_eval.py",
            "tt100k_eval.py",
        )
    ]
    for seed in NEW_TRAINING_SEEDS:
        paths.append(root / "configs" / "training" / f"ivc_multiseed_yolo11m_s{seed}_v1.json")
    for dataset in config["datasets"]:
        paths.extend(
            root / dataset[field]
            for field in (
                "annotations", "class_map", "data_yaml", "acquisition_registry",
                "baseline_training_registry", "baseline_calibration_manifest",
                "clean_manifest",
            )
        )
        paths.extend(
            root / "manifests" / "images" / (
                f"{dataset['manifest_prefix']}_full_{corruption}_s5.json"
            )
            for corruption in CORRUPTIONS
        )
    return paths


def capture_resource_snapshot(root: Path) -> dict[str, Any]:
    gpu = subprocess.run(
        ["nvidia-smi", "--query-gpu=name,memory.total,memory.free,utilization.gpu,temperature.gpu",
         "--format=csv,noheader,nounits"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    ).stdout.strip().split(",")
    if len(gpu) != 5:
        raise RuntimeError("multi-seed GPU snapshot is malformed")
    process_query = subprocess.run(
        ["nvidia-smi", "--query-compute-apps=pid,process_name,used_gpu_memory",
         "--format=csv,noheader,nounits"],
        check=True, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE,
    )
    processes = []
    for line in process_query.stdout.splitlines():
        if not line.strip():
            continue
        parts = [part.strip() for part in line.split(",", 2)]
        if len(parts) != 3:
            raise RuntimeError("multi-seed GPU process snapshot is malformed")
        processes.append({"pid": int(parts[0]), "name": parts[1], "used_memory_mib": int(parts[2])})
    disk = shutil.disk_usage(root)
    return {
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "gpu_name": gpu[0].strip(),
        "gpu_memory_total_mib": int(gpu[1]),
        "gpu_memory_free_mib": int(gpu[2]),
        "gpu_utilization_percent": int(gpu[3]),
        "gpu_temperature_c": int(gpu[4]),
        "disk_free_bytes": disk.free,
        "compute_processes": processes,
        "conda_default_env": os.environ.get("CONDA_DEFAULT_ENV"),
        "python_executable": sys.executable,
    }


def preflight(config: dict[str, Any], root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    if os.environ.get("CONDA_DEFAULT_ENV") != "qtsd":
        raise RuntimeError("multi-seed preflight requires conda environment qtsd")
    missing = [path for path in _required_preflight_paths(config, root) if not path.is_file()]
    if missing:
        raise RuntimeError(f"multi-seed preflight missing required path: {missing[0]}")
    incomplete = []
    for dataset in config["datasets"]:
        for field in ("acquisition_registry", "baseline_training_registry"):
            path = root / dataset[field]
            if not _complete_file(path):
                incomplete.append(path)
        calibration = root / dataset["baseline_calibration_manifest"]
        marker = calibration.with_suffix(calibration.suffix + ".complete")
        document = json.loads(calibration.read_text(encoding="utf-8"))
        if (
            not marker.is_file()
            or marker.read_text(encoding="utf-8").strip() != document.get("calibration_sha256")
        ):
            incomplete.append(calibration)
    if incomplete:
        raise RuntimeError(f"multi-seed preflight found incomplete input: {incomplete[0]}")
    snapshot = capture_resource_snapshot(root)
    validate_resource_snapshot(snapshot, minimum_free_gib=config["minimum_free_gib"])
    jobs = derive_jobs(config, root)
    return {
        "status": "pass",
        "attempt": config["attempt"],
        "config_sha256": config["config_sha256"],
        "resource_snapshot": snapshot,
        "job_counts": {stage: len(items) for stage, items in jobs.items()},
        "source_sha256": {
            str(path.relative_to(root)): _sha256(path)
            for path in _required_preflight_paths(config, root)
            if path.suffix == ".py"
        },
    }


def _validate_training_registry(job: dict[str, Any]) -> bool:
    path = Path(job["registry"])
    if not _complete_file(path):
        return False
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("run_id") != job["run_id"]
        or document.get("dataset") != job["dataset"]
        or document.get("model") != "yolo11m"
        or document.get("resolved_batch") != job["batch"]
    ):
        raise RuntimeError(f"multi-seed completed training registry mismatch: {path}")
    for field in ("best_weights", "last_weights", "ultralytics_args"):
        artifact = Path(document.get(field, ""))
        expected = document.get(f"{field}_sha256")
        if not artifact.is_file() or _sha256(artifact) != expected:
            raise RuntimeError(f"multi-seed completed training artifact mismatch: {artifact}")
    return True


def execute_training(config: dict[str, Any], root: Path, *, resume_verified: bool) -> dict[str, Any]:
    root = Path(root).resolve()
    jobs = derive_jobs(config, root)["training"]
    completed = []
    for ordinal, job in enumerate(jobs, start=1):
        if _validate_training_registry(job):
            if not resume_verified:
                raise RuntimeError(f"multi-seed training output already exists: {job['run_id']}")
            print(f"MULTISEED TRAIN {ordinal}/{len(jobs)} verified {job['run_id']}", flush=True)
            completed.append(job)
            continue
        output = root / "outputs" / "training" / job["dataset"] / job["run_id"]
        command = training_command(job, config, root)
        if output.exists():
            if not resume_verified:
                raise RuntimeError(f"multi-seed partial training requires resume: {output}")
            checkpoint = output / "weights" / "last.pt"
            if not checkpoint.is_file():
                raise RuntimeError(f"multi-seed partial training lacks last checkpoint: {output}")
            command.extend(["--resume-from", str(checkpoint)])
        print(f"MULTISEED TRAIN {ordinal}/{len(jobs)} starting {job['run_id']}", flush=True)
        subprocess.run(command, check=True)
        if not _validate_training_registry(job):
            raise RuntimeError(f"multi-seed training did not complete registry: {job['run_id']}")
        completed.append(job)
    report = root / "outputs" / "reports" / f"{config['attempt']}_training_complete.json"
    if report.exists():
        raise RuntimeError(f"multi-seed training report already exists: {report}")
    document = {
        "schema_version": 1,
        "status": "complete",
        "attempt": config["attempt"],
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config["config_sha256"],
        "training_registries_sha256": {
            str(Path(job["registry"]).relative_to(root)): _sha256(Path(job["registry"]))
            for job in completed
        },
    }
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return document


def validate_training_completion(config: dict[str, Any], root: Path) -> dict[str, Any]:
    root = Path(root).resolve()
    jobs = derive_jobs(config, root)["training"]
    for job in jobs:
        if not _validate_training_registry(job):
            raise RuntimeError(f"multi-seed training registry is incomplete: {job['run_id']}")
    report = root / "outputs" / "reports" / f"{config['attempt']}_training_complete.json"
    try:
        document = json.loads(report.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError("multi-seed training completion report is absent or invalid") from error
    expected = {
        str(Path(job["registry"]).relative_to(root)): _sha256(Path(job["registry"]))
        for job in jobs
    }
    if (
        document.get("status") != "complete"
        or document.get("attempt") != config["attempt"]
        or document.get("config_sha256") != config["config_sha256"]
        or document.get("training_registries_sha256") != expected
    ):
        raise RuntimeError("multi-seed training completion report provenance mismatch")
    return document


def validate_export_job(job: dict[str, Any], config: dict[str, Any], root: Path) -> bool:
    registry_path = Path(job["registry"])
    output = _export_onnx_path(job, config, Path(root).resolve())
    source_copy = output.with_suffix(".pt")
    document = _read_sha_complete(registry_path, "FP32 ONNX registry")
    if document is None:
        if output.exists() or source_copy.exists():
            raise RuntimeError(f"multi-seed FP32 ONNX output lacks registry: {output}")
        return False
    dataset = _dataset_config(config, job["dataset"])
    training = Path(job["training_registry"])
    if (
        document.get("dataset") != job["dataset"]
        or document.get("model") != "yolo11m"
        or document.get("imgsz") != dataset["imgsz"]
        or document.get("training_registry_sha256") != _sha256(training)
        or Path(document.get("onnx", "")).resolve() != output.resolve()
        or document.get("onnx_sha256") != (_sha256(output) if output.is_file() else None)
        or Path(document.get("source_checkpoint", "")).resolve() != source_copy.resolve()
        or document.get("source_checkpoint_sha256") != (_sha256(source_copy) if source_copy.is_file() else None)
    ):
        raise RuntimeError(f"multi-seed FP32 ONNX registry provenance mismatch: {registry_path}")
    return True


def validate_quantization_job(job: dict[str, Any], config: dict[str, Any], root: Path) -> bool:
    registry_path = Path(job["registry"])
    output = _quantized_onnx_path(job, config, Path(root).resolve())
    document = _read_sha_complete(registry_path, "quantized ONNX registry")
    if document is None:
        if output.exists():
            raise RuntimeError(f"multi-seed quantized ONNX lacks registry: {output}")
        return False
    dataset = _dataset_config(config, job["dataset"])
    calibration_path = Path(job["calibration_manifest"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (
        document.get("dataset") != job["dataset"]
        or document.get("model") != "yolo11m"
        or document.get("precision") != job["precision"]
        or document.get("imgsz") != dataset["imgsz"]
        or document.get("source_onnx_registry_sha256") != _sha256(Path(job["source_registry"]))
        or Path(document.get("calibration_list", "")).resolve() != calibration_path.resolve()
        or document.get("calibration_sha256") != calibration.get("calibration_sha256")
        or document.get("calibration_method") != "entropy"
        or document.get("calibration_eps") != ["cpu"]
        or document.get("op_types_to_exclude") != ["Sigmoid"]
        or Path(document.get("output_onnx", "")).resolve() != output.resolve()
        or document.get("output_onnx_sha256") != (_sha256(output) if output.is_file() else None)
    ):
        raise RuntimeError(f"multi-seed quantized ONNX provenance mismatch: {registry_path}")
    return True


def validate_engine_job(job: dict[str, Any], config: dict[str, Any], root: Path) -> bool:
    root = Path(root).resolve()
    engine, log, registry_path = _engine_paths(job, config, root)
    if registry_path.resolve() != Path(job["registry"]).resolve():
        raise RuntimeError(f"multi-seed engine registry path derivation mismatch: {registry_path}")
    document = _read_sha_complete(registry_path, "engine registry")
    if document is None:
        if engine.exists() or log.exists():
            raise RuntimeError(f"multi-seed engine/log lacks registry: {engine}")
        return False
    calibration_path = Path(job["calibration_manifest"])
    calibration = json.loads(calibration_path.read_text(encoding="utf-8"))
    if (
        document.get("dataset") != job["dataset"]
        or document.get("model") != "yolo11m"
        or document.get("precision") != job["precision"]
        or document.get("source_onnx_registry_sha256") != _sha256(Path(job["source_registry"]))
        or Path(document.get("engine", "")).resolve() != engine.resolve()
        or document.get("engine_sha256") != (_sha256(engine) if engine.is_file() else None)
        or Path(document.get("build_log", "")).resolve() != log.resolve()
        or document.get("build_log_sha256") != (_sha256(log) if log.is_file() else None)
        or document.get("calibration_sha256") != calibration.get("calibration_sha256")
        or document.get("calibration_method") != "entropy"
    ):
        raise RuntimeError(f"multi-seed engine provenance mismatch: {registry_path}")
    return True


def _read_json(path: Path, label: str) -> Any:
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"multi-seed {label} JSON is absent or invalid: {path}") from error


def validate_inference_job(
    job: dict[str, Any], config: dict[str, Any], root: Path
) -> bool:
    root = Path(root).resolve()
    prediction, inputs_path, run_path, _ = _condition_paths(
        root, config["attempt"], job["condition_id"]
    )
    dataset = _dataset_config(config, job["dataset"])
    manifest_path = Path(job["input_manifest"])
    manifest = _read_json(manifest_path, "input manifest")
    marker = manifest_path.with_suffix(manifest_path.suffix + ".complete")
    if not isinstance(manifest, dict):
        raise RuntimeError(f"multi-seed input manifest is not an object: {manifest_path}")
    manifest_sha = manifest.get("manifest_sha256")
    records = manifest.get("records")
    expected_ids = manifest.get("expected_image_ids")
    if (
        not isinstance(manifest_sha, str)
        or manifest_sha != canonical_hash(manifest, "manifest_sha256")
        or not marker.is_file()
        or marker.read_text(encoding="utf-8").strip() != manifest_sha
        or not isinstance(records, list)
        or not isinstance(expected_ids, list)
        or len(records) != dataset["expected_images"]
        or [record.get("image_id") for record in records if isinstance(record, dict)] != expected_ids
    ):
        raise RuntimeError(f"multi-seed input manifest provenance mismatch: {manifest_path}")
    image_ids_sha = hashlib.sha256(
        json.dumps(expected_ids, separators=(",", ":")).encode()
    ).hexdigest()
    engine_registry_path = Path(job["engine_registry"])
    engine_registry = _read_sha_complete(engine_registry_path, "engine registry")
    if engine_registry is None:
        raise RuntimeError(f"multi-seed engine registry is absent: {engine_registry_path}")
    engine, _, derived_registry = _engine_paths(
        {"registry": job["engine_registry"], "precision": job["precision"]},
        config,
        root,
    )
    calibration_path = Path(job["calibration_manifest"])
    calibration = _read_json(calibration_path, "calibration manifest")
    if not isinstance(calibration, dict):
        raise RuntimeError(f"multi-seed calibration manifest is not an object: {calibration_path}")
    calibration_sha = calibration.get("calibration_sha256")
    calibration_marker = calibration_path.with_suffix(
        calibration_path.suffix + ".complete"
    )
    if (
        not isinstance(calibration_sha, str)
        or calibration_sha != canonical_hash(calibration, "calibration_sha256")
        or not calibration_marker.is_file()
        or calibration_marker.read_text(encoding="utf-8").strip() != calibration_sha
        or derived_registry.resolve() != engine_registry_path.resolve()
        or engine_registry.get("dataset") != job["dataset"]
        or engine_registry.get("model") != "yolo11m"
        or engine_registry.get("precision") != job["precision"]
        or Path(engine_registry.get("engine", "")).resolve() != engine.resolve()
        or not engine.is_file()
        or engine_registry.get("engine_sha256") != _sha256(engine)
        or engine_registry.get("calibration_sha256") != calibration_sha
        or engine_registry.get("calibration_method") != "entropy"
    ):
        raise RuntimeError(f"multi-seed inference engine provenance mismatch: {engine_registry_path}")

    states = [path.is_file() for path in (prediction, inputs_path, run_path)]
    if not any(states):
        return False
    if not all(states):
        raise RuntimeError(
            f"multi-seed inference has a partial output triple: {job['condition_id']}"
        )
    predictions = _read_json(prediction, "prediction")
    inputs = _read_json(inputs_path, "input record")
    run = _read_json(run_path, "run record")
    if not isinstance(predictions, list) or not isinstance(inputs, dict) or not isinstance(run, dict):
        raise RuntimeError(f"multi-seed inference output shape mismatch: {job['condition_id']}")
    if (
        inputs.get("condition_id") != job["condition_id"]
        or inputs.get("image_ids") != expected_ids
        or inputs.get("image_ids_sha256") != image_ids_sha
        or inputs.get("input_manifest_sha256") != manifest_sha
    ):
        raise RuntimeError(f"multi-seed input record provenance mismatch: {inputs_path}")

    class_map = root / dataset["class_map"]
    annotations = root / dataset["annotations"]
    source_hashes = {
        "runner_sha256": _sha256(root / "src" / "coco_infer_trt.py"),
        "preprocess_sha256": _sha256(root / "src" / "topic_c" / "coco_data.py"),
        "decoder_sha256": _sha256(root / "src" / "topic_c" / "yolo_decode.py"),
    }
    if any(run.get(field) != digest for field, digest in source_hashes.items()):
        raise RuntimeError(f"multi-seed inference source provenance mismatch: {run_path}")
    prediction_sha = _sha256(prediction)
    exact = {
        "condition_id": job["condition_id"],
        "dataset": job["dataset"],
        "split": dataset["split"],
        "model": "yolo11m",
        "precision": job["precision"],
        "calibrator": "entropy",
        "calibration_sha256": calibration_sha,
        "calibration_method": "entropy",
        "calibration_provenance": "verified",
        "corruption": job["corruption"],
        "severity": job["severity"],
        "engine_sha256": _sha256(engine),
        "class_map_sha256": _sha256(class_map),
        "annotation_sha256": _sha256(annotations),
        "input_manifest_sha256": manifest_sha,
        "input_image_ids_sha256": image_ids_sha,
        "prediction_sha256": prediction_sha,
        "n_images": dataset["expected_images"],
        "n_detections": len(predictions),
    }
    if any(run.get(field) != value for field, value in exact.items()):
        if run.get("prediction_sha256") != prediction_sha:
            raise RuntimeError(f"multi-seed inference prediction hash mismatch: {prediction}")
        raise RuntimeError(f"multi-seed inference run provenance mismatch: {run_path}")
    if (
        Path(run.get("engine_path", "")).resolve() != engine.resolve()
        or Path(run.get("calibration_list_path", "")).resolve() != calibration_path.resolve()
        or Path(run.get("class_map_path", "")).resolve() != class_map.resolve()
    ):
        raise RuntimeError(f"multi-seed inference path provenance mismatch: {run_path}")
    return True


def validate_metric_job(
    job: dict[str, Any], config: dict[str, Any], root: Path
) -> bool:
    root = Path(root).resolve()
    if not validate_inference_job(job, config, root):
        raise RuntimeError(f"multi-seed metric lacks validated inference: {job['condition_id']}")
    prediction, inputs_path, run_path, metric_path = _condition_paths(
        root, config["attempt"], job["condition_id"]
    )
    if not metric_path.exists():
        return False
    if not metric_path.is_file():
        raise RuntimeError(f"multi-seed metric path is not a file: {metric_path}")
    metric = _read_json(metric_path, "metric")
    inputs = _read_json(inputs_path, "input record")
    run = _read_json(run_path, "run record")
    dataset = _dataset_config(config, job["dataset"])
    if not isinstance(metric, dict) or not isinstance(inputs, dict) or not isinstance(run, dict):
        raise RuntimeError(f"multi-seed metric record shape mismatch: {metric_path}")
    exact = {
        "condition_id": job["condition_id"],
        "dataset": job["dataset"],
        "split": dataset["split"],
        "model": "yolo11m",
        "precision": job["precision"],
        "corruption": job["corruption"],
        "severity": job["severity"],
        "n_images": dataset["expected_images"],
        "prediction_sha256": _sha256(prediction),
        "input_manifest_sha256": run.get("input_manifest_sha256"),
        "input_image_ids_sha256": inputs.get("image_ids_sha256"),
        "run_record_sha256": _sha256(run_path),
    }
    stats = metric.get("stats")
    ap = stats.get("AP") if isinstance(stats, dict) else None
    if (
        any(metric.get(field) != value for field, value in exact.items())
        or not isinstance(ap, (int, float))
        or isinstance(ap, bool)
        or not math.isfinite(ap)
        or not 0.0 <= ap <= 1.0
    ):
        raise RuntimeError(f"multi-seed metric provenance/semantics mismatch: {metric_path}")
    return True


def run_evidence_child(
    command: list[str], *, phase: str, condition_id: str, root: Path
) -> None:
    root = Path(root).resolve()
    try:
        subprocess.run(command, check=True)
    except subprocess.CalledProcessError as error:
        ledger = (
            root
            / "outputs"
            / "logs"
            / f"{ATTEMPT}.evidence.failures.jsonl"
        )
        ledger.parent.mkdir(parents=True, exist_ok=True)
        event = {
            "schema_version": 1,
            "failed_at_utc": datetime.now(timezone.utc).isoformat(),
            "attempt": ATTEMPT,
            "phase": phase,
            "condition_id": condition_id,
            "returncode": error.returncode,
            "command": command,
        }
        with ledger.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(event, sort_keys=True) + "\n")
            handle.flush()
            os.fsync(handle.fileno())
        raise RuntimeError(
            f"multi-seed {phase} child failed for {condition_id} "
            f"with return code {error.returncode}"
        ) from error


def evidence_artifact_paths(
    config: dict[str, Any], root: Path
) -> dict[str, Any]:
    root = Path(root).resolve()
    jobs = derive_jobs(config, root)
    inference_outputs = [
        path
        for job in jobs["inference"]
        for path in _condition_paths(root, config["attempt"], job["condition_id"])[0:3]
    ]
    metrics = [
        _condition_paths(root, config["attempt"], job["condition_id"])[3]
        for job in jobs["metric"]
    ]
    input_manifests = sorted(
        {Path(job["input_manifest"]) for job in jobs["inference"]},
        key=lambda path: path.as_posix(),
    )
    report_root = root / "outputs" / "reports"
    return {
        "inference_outputs": inference_outputs,
        "input_manifests": input_manifests,
        "metrics": metrics,
        "engine_report": report_root / f"{config['attempt']}_engine_complete.json",
        "inference_report": report_root / f"{config['attempt']}_inference_complete.json",
    }


def evidence_execution_order(
    jobs: dict[str, list[dict[str, Any]]]
) -> list[tuple[dict[str, Any], dict[str, Any]]]:
    inference = jobs.get("inference", [])
    metrics = jobs.get("metric", [])
    if len(inference) != 270 or len(metrics) != 270:
        raise RuntimeError("multi-seed evidence execution grid is not exactly 270 pairs")
    pairs = list(zip(inference, metrics, strict=True))
    if any(left.get("condition_id") != right.get("condition_id") for left, right in pairs):
        raise RuntimeError("multi-seed inference/metric pairing mismatch")
    condition_ids = [left["condition_id"] for left, _ in pairs]
    if len(set(condition_ids)) != 270:
        raise RuntimeError("multi-seed evidence condition IDs are not unique")
    return pairs


def run_bounded_evidence_states(
    states: list[tuple[dict[str, Any], dict[str, Any], bool, bool]],
    *,
    workers: int,
    work: Callable[[tuple[dict[str, Any], dict[str, Any], bool, bool]], str],
) -> list[str]:
    if (
        type(workers) is not int
        or workers < 1
        or workers > MAX_EVIDENCE_WORKERS
    ):
        raise ValueError(
            f"evidence workers must be an integer in [1, {MAX_EVIDENCE_WORKERS}]"
        )
    if workers == 1:
        return [work(state) for state in states]
    with ThreadPoolExecutor(
        max_workers=workers, thread_name_prefix="multiseed-evidence"
    ) as executor:
        return list(executor.map(work, states))


def execute_evidence(
    config: dict[str, Any], root: Path, *, resume_verified: bool, workers: int = 1
) -> dict[str, Any]:
    root = Path(root).resolve()
    jobs = derive_jobs(config, root)
    pairs = evidence_execution_order(jobs)
    paths = evidence_artifact_paths(config, root)

    engine_artifacts = [
        artifact
        for job in jobs["engine"]
        for artifact in _engine_paths(job, config, root)
    ]
    validate_stage_report(
        paths["engine_report"],
        attempt=config["attempt"],
        stage="engine",
        config_sha256=config["config_sha256"],
        artifacts=engine_artifacts,
        root=root,
    )

    states: list[tuple[dict[str, Any], dict[str, Any], bool, bool]] = []
    for inference_job, metric_job in pairs:
        inference_complete = validate_inference_job(inference_job, config, root)
        metric_path = _condition_paths(
            root, config["attempt"], metric_job["condition_id"]
        )[3]
        metric_complete = (
            validate_metric_job(metric_job, config, root) if metric_path.exists() else False
        )
        if (inference_complete or metric_complete) and not resume_verified:
            raise RuntimeError(
                "multi-seed evidence output already exists: "
                f"{inference_job['condition_id']}"
            )
        states.append(
            (inference_job, metric_job, inference_complete, metric_complete)
        )

    ordinal_by_condition = {
        state[0]["condition_id"]: ordinal for ordinal, state in enumerate(states, start=1)
    }

    def execute_state(
        state: tuple[dict[str, Any], dict[str, Any], bool, bool]
    ) -> str:
        inference_job, metric_job, inference_complete, metric_complete = state
        condition_id = inference_job["condition_id"]
        ordinal = ordinal_by_condition[condition_id]
        prediction, inputs, _, _ = _condition_paths(
            root, config["attempt"], condition_id
        )
        dataset = _dataset_config(config, inference_job["dataset"])
        if inference_complete:
            print(f"MULTISEED INFERENCE {ordinal}/270 verified {condition_id}", flush=True)
        else:
            print(f"MULTISEED INFERENCE {ordinal}/270 starting {condition_id}", flush=True)
            run_evidence_child(
                inference_command(inference_job, config, root),
                phase="inference",
                condition_id=condition_id,
                root=root,
            )
            run_evidence_child(
                [
                    sys.executable,
                    str(root / "src" / "validate_predictions.py"),
                    "--predictions",
                    str(prediction),
                    "--input-record",
                    str(inputs),
                    "--annotations",
                    str(root / dataset["annotations"]),
                ],
                phase="prediction-validation",
                condition_id=condition_id,
                root=root,
            )
            if not validate_inference_job(inference_job, config, root):
                raise RuntimeError(
                    f"multi-seed inference did not complete: {condition_id}"
                )
        if metric_complete:
            print(f"MULTISEED METRIC {ordinal}/270 verified {condition_id}", flush=True)
        else:
            print(f"MULTISEED METRIC {ordinal}/270 starting {condition_id}", flush=True)
            run_evidence_child(
                metric_command(metric_job, config, root),
                phase="metric",
                condition_id=condition_id,
                root=root,
            )
            if not validate_metric_job(metric_job, config, root):
                raise RuntimeError(f"multi-seed metric did not complete: {condition_id}")
        return condition_id

    run_bounded_evidence_states(states, workers=workers, work=execute_state)

    inference_report = _finish_stage(
        config,
        root,
        "inference",
        [
            paths["engine_report"],
            *paths["input_manifests"],
            *paths["inference_outputs"],
        ],
        resume_verified=resume_verified,
    )
    metric_report = _finish_stage(
        config,
        root,
        "metric",
        [paths["inference_report"], *paths["metrics"]],
        resume_verified=resume_verified,
    )
    return {
        "status": "complete",
        "conditions": 270,
        "inference_report_sha256": _sha256(paths["inference_report"]),
        "metric_report_sha256": _sha256(
            root
            / "outputs"
            / "reports"
            / f"{config['attempt']}_metric_complete.json"
        ),
        "reports": [inference_report, metric_report],
    }


def _finish_stage(
    config: dict[str, Any],
    root: Path,
    stage: str,
    artifacts: list[Path],
    *,
    resume_verified: bool,
) -> dict[str, Any]:
    report = root / "outputs" / "reports" / f"{config['attempt']}_{stage}_complete.json"
    if report.exists():
        if not resume_verified:
            raise RuntimeError(f"multi-seed {stage} completion report already exists")
        return validate_stage_report(
            report,
            attempt=config["attempt"],
            stage=stage,
            config_sha256=config["config_sha256"],
            artifacts=artifacts,
            root=root,
        )
    return write_stage_report(
        report,
        attempt=config["attempt"],
        stage=stage,
        config_sha256=config["config_sha256"],
        artifacts=artifacts,
        root=root,
    )


def execute_build(
    config: dict[str, Any], root: Path, *, resume_verified: bool
) -> dict[str, Any]:
    root = Path(root).resolve()
    validate_trt_runtime(Path(config["trt_root"]))
    validate_training_completion(config, root)
    jobs = derive_jobs(config, root)

    calibration_artifacts: list[Path] = []
    for ordinal, job in enumerate(jobs["calibration"], start=1):
        valid = validate_calibration_job(job, config, root)
        if valid:
            if not job["existing"] and not resume_verified:
                raise RuntimeError(f"multi-seed calibration output already exists: {job['manifest']}")
            state = "baseline" if job["existing"] else "verified"
            print(f"MULTISEED CALIBRATION {ordinal}/9 {state} {job['job_id']}", flush=True)
        else:
            print(f"MULTISEED CALIBRATION {ordinal}/9 starting {job['job_id']}", flush=True)
            subprocess.run(calibration_command(job, config, root), check=True)
            if not validate_calibration_job(job, config, root):
                raise RuntimeError(f"multi-seed calibration did not complete: {job['job_id']}")
        calibration_artifacts.append(Path(job["manifest"]))
    calibration_report = _finish_stage(
        config, root, "calibration", calibration_artifacts, resume_verified=resume_verified
    )

    export_artifacts: list[Path] = []
    for ordinal, job in enumerate(jobs["export"], start=1):
        if validate_export_job(job, config, root):
            if not resume_verified:
                raise RuntimeError(f"multi-seed export output already exists: {job['registry']}")
            print(f"MULTISEED EXPORT {ordinal}/9 verified {job['job_id']}", flush=True)
        else:
            print(f"MULTISEED EXPORT {ordinal}/9 starting {job['job_id']}", flush=True)
            subprocess.run(export_command(job, config, root), check=True)
            if not validate_export_job(job, config, root):
                raise RuntimeError(f"multi-seed export did not complete: {job['job_id']}")
        registry = Path(job["registry"])
        onnx_path = _export_onnx_path(job, config, root)
        export_artifacts.extend([registry, onnx_path, onnx_path.with_suffix(".pt")])
    export_report = _finish_stage(
        config, root, "export", export_artifacts, resume_verified=resume_verified
    )

    quantization_artifacts: list[Path] = []
    for ordinal, job in enumerate(jobs["quantization"], start=1):
        if validate_quantization_job(job, config, root):
            if not resume_verified:
                raise RuntimeError(f"multi-seed quantization output already exists: {job['registry']}")
            print(f"MULTISEED QUANTIZE {ordinal}/54 verified {job['job_id']}", flush=True)
        else:
            print(f"MULTISEED QUANTIZE {ordinal}/54 starting {job['job_id']}", flush=True)
            subprocess.run(quantization_command(job, config, root), check=True)
            if not validate_quantization_job(job, config, root):
                raise RuntimeError(f"multi-seed quantization did not complete: {job['job_id']}")
        quantization_artifacts.extend(
            [Path(job["registry"]), _quantized_onnx_path(job, config, root)]
        )
    quantization_report = _finish_stage(
        config, root, "quantization", quantization_artifacts,
        resume_verified=resume_verified,
    )

    engine_artifacts: list[Path] = []
    for ordinal, job in enumerate(jobs["engine"], start=1):
        if validate_engine_job(job, config, root):
            if not resume_verified:
                raise RuntimeError(f"multi-seed engine output already exists: {job['registry']}")
            print(f"MULTISEED ENGINE {ordinal}/54 verified {job['job_id']}", flush=True)
        else:
            print(f"MULTISEED ENGINE {ordinal}/54 starting {job['job_id']}", flush=True)
            subprocess.run(engine_command(job, config, root), check=True)
            if not validate_engine_job(job, config, root):
                raise RuntimeError(f"multi-seed engine did not complete: {job['job_id']}")
        engine_artifacts.extend(_engine_paths(job, config, root))
    engine_report = _finish_stage(
        config, root, "engine", engine_artifacts, resume_verified=resume_verified
    )
    return {
        "status": "complete",
        "calibration_report_sha256": _sha256(
            root / "outputs" / "reports" / f"{config['attempt']}_calibration_complete.json"
        ),
        "export_report_sha256": _sha256(
            root / "outputs" / "reports" / f"{config['attempt']}_export_complete.json"
        ),
        "quantization_report_sha256": _sha256(
            root / "outputs" / "reports" / f"{config['attempt']}_quantization_complete.json"
        ),
        "engine_report_sha256": _sha256(
            root / "outputs" / "reports" / f"{config['attempt']}_engine_complete.json"
        ),
        "reports": [calibration_report, export_report, quantization_report, engine_report],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    action = parser.add_mutually_exclusive_group(required=True)
    action.add_argument("--preflight", action="store_true")
    action.add_argument("--execute-training", action="store_true")
    action.add_argument("--execute-build", action="store_true")
    action.add_argument("--execute-evidence", action="store_true")
    parser.add_argument("--resume-verified", action="store_true")
    parser.add_argument("--evidence-workers", type=int, default=1)
    args = parser.parse_args()
    root = args.project_root.resolve()
    config = load_config(args.config)
    snapshot = preflight(config, root)
    print(json.dumps(snapshot, indent=2, sort_keys=True), flush=True)
    if args.execute_training:
        report = execute_training(config, root, resume_verified=args.resume_verified)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    elif args.execute_build:
        report = execute_build(config, root, resume_verified=args.resume_verified)
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)
    elif args.execute_evidence:
        report = execute_evidence(
            config,
            root,
            resume_verified=args.resume_verified,
            workers=args.evidence_workers,
        )
        print(json.dumps(report, indent=2, sort_keys=True), flush=True)


if __name__ == "__main__":
    main()
