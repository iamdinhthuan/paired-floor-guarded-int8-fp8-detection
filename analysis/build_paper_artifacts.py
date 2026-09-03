#!/usr/bin/env python3
"""Validate the completed P0 four-dataset study and generate paper-ready artifacts.

The script is deliberately fail-closed.  It refuses to produce tables or figures
unless the P0 completion chain, its bound artifacts, and the intended analysis
grids validate.  Legacy metric directories remain canonical for the three
transfer datasets, while COCO and all matched-codec evidence come from P0.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import math
import re
import shlex
from datetime import datetime
from itertools import product
from pathlib import Path
from typing import Any

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns
from matplotlib.patches import FancyArrowPatch, FancyBboxPatch


DATASETS = ("coco", "voc", "kitti", "tt100k")
ATTEMPTS = {
    "coco": "coco_uniform_p0_v1",
    "voc": "voc_pilot_117_v1",
    "kitti": "kitti_pilot_117_v1",
    "tt100k": "tt100k_pilot_117_v1",
}
EVAL_REPORTS = {
    "coco": "coco_uniform_p0_v1_evaluation_complete.json",
    "voc": "voc_pilot_117_v1_evaluation_complete.json",
    "kitti": "kitti_pilot_117_v1_evaluation_complete.json",
    "tt100k": "tt100k_pilot_117_v1_evaluation_complete.json",
}
LEGACY_ARTIFACT_ROOT = Path("artifacts/four_dataset_pilot_v1")
P0_REPORT = "p0_strengthening_complete_v1.json"
HISTORICAL_LADDER_CONFIG_SHA256 = "96afb13f82b89cbdefa2284d22676ad5593104768424836d00477e4a2d44e812"
FINAL_P0_CONFIG_SHA256 = "ab044caa9bd85622eeb291dc2c04d0ee82c787b69c0d9fc01c121344f03e2c15"
P0_COMPONENT_REPORTS = {
    "coco_uniform_ladder_p0_v1.json",
    "coco_clean_fp16_parity_complete_v1.json",
    "coco_uniform_p0_v1_evaluation_complete.json",
    "codec_control_p0_v1_complete.json",
    "matched_codec_p0_v1_complete.json",
}
DATASET_META = [
    ("COCO", "val2017", 5000, 36781, "80/80", 640, r"area: $<32^2$ / $[32^2,96^2)$ / $\geq96^2$"),
    ("VOC", "VOC2007 test", 4952, 12032, "20/20", 640, r"area: $<32^2$ / $[32^2,96^2)$ / $\geq96^2$"),
    ("KITTI", "validation", 1496, 8128, "8/8", 640, r"area: $<32^2$ / $[32^2,96^2)$ / $\geq96^2$"),
    ("TT100K", "test", 3067, 8181, "221/136", 1280, "height: XS/S/M/L/XL"),
]
MODEL_ORDER = ["yolo11n", "yolo11m", "yolo11x"]
PRECISION_ORDER = ["fp32", "int8-entropy", "fp8"]
QUANT_ORDER = ["int8-entropy", "fp8"]
CORRUPTION_ORDER = ["gaussian_noise", "motion_blur", "fog", "jpeg"]
DISPLAY = {
    "coco": "COCO",
    "voc": "VOC",
    "kitti": "KITTI",
    "tt100k": "TT100K",
    "yolo11n": "YOLO11n",
    "yolo11m": "YOLO11m",
    "yolo11x": "YOLO11x",
    "fp32": "FP32",
    "int8-entropy": "INT8",
    "fp8": "FP8",
    "gaussian_noise": "Gaussian noise",
    "motion_blur": "Motion blur",
    "fog": "Fog",
    "jpeg": "JPEG",
}

FORMAT_CONTRAST_ATTEMPT = "ivc_format_contrast_v1"
DEPLOYMENT_ATTEMPT = "ivc_deployment_benchmark_v1"
FORMAT_ARM_NAMES = ("int8_clean", "fp8_clean", "int8_corrupt", "fp8_corrupt")
FORMAT_METHOD = "paired image bootstrap; one shared resample across INT8/FP8 × clean/corrupt"
FORMAT_SIGN_CONVENTION = {
    "delta_q": "AP(fp8, clean) - AP(int8, clean); positive means INT8 has lower clean AP",
    "delta_e": "[AP(int8, clean)-AP(int8, corrupt)] - [AP(fp8, clean)-AP(fp8, corrupt)]; positive means INT8 amplifies corruption more",
    "delta_psi": "delta_e(small) - delta_e(large), or small_like - large_like for TT100K height",
}
DEPLOYMENT_PRECISIONS = ("fp32", "int8-entropy", "fp8")
DEPLOYMENT_RUNTIME_FIELDS = {
    "hostname",
    "project_root",
    "platform",
    "python_version",
    "gpu_name",
    "gpu_uuid",
    "driver_version",
    "cuda_version",
    "tensorrt_python_version",
    "trtexec_version_output",
    "trtexec_executable",
    "trtexec_sha256",
}
BENCHMARK_LOG_PREFIX = "IVC_TRTEXEC_LOG_V1 "
BENCHMARK_LOG_FORMAT = "ivc_trtexec_log_v1"
BENCHMARK_NUMBER = r"[+-]?(?:nan|inf(?:inity)?|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"
GPU_IDLE_COMMAND = ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
ENGINE_TREATMENT_FIELDS = {
    "dataset",
    "model",
    "precision",
    "onnx_sha256",
    "engine_sha256",
    "engine_bytes",
    "quantize_mode",
    "calibration_list",
    "calibration_sha256",
    "calibration_method",
    "calibration_eps",
    "op_types_to_exclude",
    "modelopt_version",
    "onnx_version",
    "onnx_command",
    "tensorrt_python_version",
    "trtexec_sha256",
    "engine_command",
    "fp8_encoding",
    "fp8_granularity",
}
FORMAT_EXECUTION_PACKAGE_FILES = {
    f"configs/{FORMAT_CONTRAST_ATTEMPT}.json",
    "src/run_format_contrast_p0.py",
    "src/format_contrast_scheduler.py",
    "src/bootstrap_format_contrast.py",
    "src/bootstrap_format_contrast_macro.py",
    "src/validate_format_contrast_evidence.py",
    "src/paired_bootstrap.py",
    "src/topic_c/manifest.py",
    "src/topic_c/tt100k_height.py",
}


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def canonical_hash(document: dict[str, Any], field: str) -> str:
    clone = dict(document)
    clone.pop(field, None)
    raw = json.dumps(clone, sort_keys=True, separators=(",", ":"), ensure_ascii=False)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _producer_canonical_hash(document: dict[str, Any], field: str) -> str:
    """Match the source runners' ASCII-safe canonical JSON hash."""
    clone = dict(document)
    clone.pop(field, None)
    raw = json.dumps(
        clone,
        sort_keys=True,
        separators=(",", ":"),
        ensure_ascii=True,
    )
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _require_sha256(value: Any, label: str) -> str:
    if not isinstance(value, str) or re.fullmatch(r"[0-9a-f]{64}", value) is None:
        raise RuntimeError(f"{label} is not a lowercase SHA-256")
    return value


def _strict_relative_file(root: Path, relative: Any, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise RuntimeError(f"{label} must be a safe relative path")
    raw = Path(relative)
    if raw.is_absolute() or ".." in raw.parts:
        raise RuntimeError(f"{label} must be a safe relative path: {relative}")
    candidate = (root / raw).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"{label} path escapes evidence root: {relative}") from error
    if not candidate.is_file():
        raise RuntimeError(f"bound artifact is missing: {relative}")
    return candidate


def _bound_local_file(root: Path, reported: Any, label: str) -> Path:
    if not isinstance(reported, str) or not reported:
        raise RuntimeError(f"{label} path is missing")
    raw = Path(reported)
    if ".." in raw.parts:
        raise RuntimeError(f"{label} must be a safe relative path: {reported}")
    if raw.is_absolute():
        anchors = [
            index
            for index, part in enumerate(raw.parts)
            if part in {"outputs", "configs", "manifests"}
        ]
        if not anchors:
            raise RuntimeError(f"{label} has no portable local-root anchor: {reported}")
        raw = Path(*raw.parts[anchors[-1] :])
    return _strict_relative_file(root, str(raw), label)


def _read_object(path: Path, label: str) -> dict[str, Any]:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise RuntimeError(f"invalid {label}: {path}") from error
    if not isinstance(document, dict):
        raise RuntimeError(f"{label} must be a JSON object: {path}")
    return document


def _validate_file_binding(root: Path, relative: Any, expected: Any, label: str) -> Path:
    path = _strict_relative_file(root, relative, label)
    if sha256_file(path) != _require_sha256(expected, f"{label} SHA-256"):
        raise RuntimeError(f"{label} SHA-256 mismatch: {relative}")
    return path


def _format_component_grid() -> tuple[set[str], set[str], set[str]]:
    clean: set[str] = set()
    corrupted: set[str] = set()
    for dataset, model in product(DATASETS, MODEL_ORDER):
        prefix = f"outputs/bootstrap/{FORMAT_CONTRAST_ATTEMPT}/{dataset}__{model}__"
        clean.add(prefix + "clean-s0.json")
        for corruption, severity in product(CORRUPTION_ORDER, (1, 3, 5)):
            corrupted.add(prefix + f"{corruption}-s{severity}.json")
    return clean | corrupted, clean, corrupted


def _command_path_below(value: Any, root: Path) -> bool:
    if not isinstance(value, str):
        return False
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts:
        return False
    try:
        path.relative_to(root)
    except ValueError:
        return False
    return True


def _consume_command_option(
    command: list[str], index: int, option: str, expected: str | None = None
) -> tuple[int, str]:
    if index + 1 >= len(command) or command[index] != option:
        raise RuntimeError(f"expected scheduler command option {option}")
    value = command[index + 1]
    if expected is not None and value != expected:
        raise RuntimeError(f"scheduler command option {option} is invalid")
    return index + 2, value


def _validate_format_scheduler_command(
    command: list[str], phase: str, relative: str, component: dict[str, Any], execution_root: str
) -> None:
    remote_root = Path(execution_root)
    python_path = Path(command[0]) if command else None
    if python_path is None or not python_path.is_absolute() or ".." in python_path.parts:
        raise RuntimeError(f"format-contrast {phase} scheduler command is invalid: {relative}")
    dataset, model = Path(relative).stem.split("__")[:2]
    if phase == "clean":
        expected = [
            command[0],
            str(remote_root / "src" / "run_format_contrast_p0.py"),
            "--project-root",
            execution_root,
            "--config",
            str(remote_root / "configs" / f"{FORMAT_CONTRAST_ATTEMPT}.json"),
            "--execute-clean-cell",
            "--dataset",
            dataset,
            "--model",
            model,
        ]
        if command != expected:
            raise RuntimeError(f"format-contrast clean scheduler command is invalid: {relative}")
        return
    if len(command) < 2 or command[1] != str(remote_root / "src" / "bootstrap_format_contrast.py"):
        raise RuntimeError(f"format-contrast corrupt scheduler command is invalid: {relative}")
    try:
        index, _ = _consume_command_option(command, 2, "--endpoint", component["endpoint_type"])
        index, _ = _consume_command_option(
            command, index, "--annotations", component["annotation"]["path"]
        )
        index, _ = _consume_command_option(
            command, index, "--expected-images", str(component["n_images"])
        )
        index, _ = _consume_command_option(
            command, index, "--annotation-sha256", component["annotation"]["sha256"]
        )
        for arm in FORMAT_ARM_NAMES:
            option = "--" + arm.replace("_", "-")
            for arm_option in (option, option + "-input", option + "-run"):
                index, source = _consume_command_option(command, index, arm_option)
                if not _command_path_below(source, remote_root):
                    raise RuntimeError(f"scheduler command option {arm_option} escapes execution root")
        index, _ = _consume_command_option(command, index, "--n-boot", "2000")
        index, _ = _consume_command_option(command, index, "--seed", str(component["seed"]))
        index, _ = _consume_command_option(command, index, "--workers", "1")
        index, _ = _consume_command_option(command, index, "--out", str(remote_root / relative))
        index, _ = _consume_command_option(command, index, "--evidence-root", execution_root)
        index, _ = _consume_command_option(
            command,
            index,
            "--draw-cache",
            str(remote_root / component["temporary_draw_cache"]["path"]),
        )
        index, _ = _consume_command_option(
            command,
            index,
            "--schedule",
            str(remote_root / component["bootstrap_schedule"]["path"]),
        )
        index, _ = _consume_command_option(
            command,
            index,
            "--clean-arm-cache",
            str(remote_root / component["clean_arm_cache"]["path"]),
        )
        index, _ = _consume_command_option(
            command,
            index,
            "--clean-arm-cache-sha256",
            component["clean_arm_cache"]["sha256"],
        )
        index, _ = _consume_command_option(
            command,
            index,
            "--clean-arm-cache-identity-sha256",
            component["clean_arm_cache"]["identity_sha256"],
        )
        index, _ = _consume_command_option(command, index, "--dataset", dataset)
        index, _ = _consume_command_option(command, index, "--model", model)
    except (KeyError, TypeError) as error:
        raise RuntimeError(
            f"format-contrast corrupt scheduler command is invalid: {relative}"
        ) from error
    if index != len(command):
        raise RuntimeError(f"format-contrast corrupt scheduler command is invalid: {relative}")


def _finite_number(value: Any) -> bool:
    return (
        not isinstance(value, bool)
        and isinstance(value, (int, float))
        and math.isfinite(float(value))
    )


def _format_dataset_seed(namespace: str, dataset: str) -> int:
    value = f"{namespace}|paired-format-dataset|{dataset}"
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % (2**32)


def validate_format_contrast(artifact_root: Path) -> dict[str, Any]:
    """Validate a complete Task 2 chain without wiring it into the paper build."""
    root = Path(artifact_root).resolve()
    report_path = root / "outputs" / "reports" / f"{FORMAT_CONTRAST_ATTEMPT}_complete.json"
    report = _read_object(report_path, "format-contrast completion report")
    if report.get("report_sha256") != _producer_canonical_hash(report, "report_sha256"):
        raise RuntimeError("format-contrast completion report has an invalid canonical SHA-256")
    if (
        report.get("schema_version") != 1
        or report.get("attempt") != FORMAT_CONTRAST_ATTEMPT
        or report.get("clean_contrasts") != 12
        or report.get("corrupted_contrasts") != 144
        or report.get("n_boot") != 2000
    ):
        raise RuntimeError("format-contrast completion grid/bootstrap contract is invalid")

    config_path = _bound_local_file(root, report.get("config"), "format-contrast config")
    if config_path != (root / "configs" / f"{FORMAT_CONTRAST_ATTEMPT}.json").resolve():
        raise RuntimeError("format-contrast report binds a noncanonical config path")
    if sha256_file(config_path) != _require_sha256(
        report.get("config_sha256"), "format-contrast config SHA-256"
    ):
        raise RuntimeError("format-contrast config SHA-256 mismatch")
    config = _read_object(config_path, "format-contrast config")
    config_datasets = config.get("datasets")
    if (
        config.get("schema_version") != 1
        or config.get("attempt") != FORMAT_CONTRAST_ATTEMPT
        or config.get("n_boot") != 2000
        or config.get("bootstrap_workers") != 1
        or config.get("cell_workers") != 4
        or not isinstance(config.get("seed_namespace"), str)
        or not config["seed_namespace"]
        or tuple(config.get("models", ())) != tuple(MODEL_ORDER)
        or tuple(config.get("precisions", ())) != tuple(QUANT_ORDER)
        or tuple(config.get("corruptions", ())) != tuple(CORRUPTION_ORDER)
        or tuple(config.get("severities", ())) != (1, 3, 5)
        or not isinstance(config_datasets, list)
        or len(config_datasets) != 4
        or any(not isinstance(row, dict) for row in config_datasets)
        or tuple(row.get("dataset") for row in config_datasets) != tuple(DATASETS)
        or any(
            set(row)
            != {
                "dataset",
                "annotations",
                "expected_images",
                "clean_source_attempt",
                "clean_corruption",
                "corruption_source_attempt",
            }
            for row in config_datasets
        )
    ):
        raise RuntimeError("format-contrast config scientific grid is invalid")
    config_by_dataset = {row["dataset"]: row for row in config_datasets}
    expected_images = {"coco": 5000, "voc": 4952, "kitti": 1496, "tt100k": 3067}
    expected_attempts = {
        "coco": "coco_uniform_p0_v1",
        "voc": "voc_pilot_117_v1",
        "kitti": "kitti_pilot_117_v1",
        "tt100k": "tt100k_pilot_117_v1",
    }
    if any(
        row.get("expected_images") != expected_images[dataset]
        or row.get("clean_source_attempt") != "codec_control_p0_v1"
        or row.get("clean_corruption") != "codec_control"
        or row.get("corruption_source_attempt") != expected_attempts[dataset]
        or not isinstance(row.get("annotations"), str)
        or not row["annotations"]
        for dataset, row in config_by_dataset.items()
    ):
        raise RuntimeError("format-contrast config dataset bindings are invalid")

    package_binding = report.get("execution_package")
    if not isinstance(package_binding, dict) or set(package_binding) != {
        "path",
        "sha256",
        "manifest_sha256",
    }:
        raise RuntimeError("format-contrast execution package binding is invalid")
    package_path = _validate_file_binding(
        root, package_binding["path"], package_binding["sha256"], "execution package"
    )
    package = _read_object(package_path, "execution package")
    execution_root = package.get("execution_root")
    if (
        package.get("schema_version") != 1
        or package.get("attempt") != FORMAT_CONTRAST_ATTEMPT
        or not isinstance(execution_root, str)
        or not Path(execution_root).is_absolute()
        or ".." in Path(execution_root).parts
        or package.get("manifest_sha256")
        != _producer_canonical_hash(package, "manifest_sha256")
        or package.get("manifest_sha256") != package_binding["manifest_sha256"]
        or not isinstance(package.get("files_sha256"), dict)
        or set(package["files_sha256"]) != FORMAT_EXECUTION_PACKAGE_FILES
    ):
        raise RuntimeError("format-contrast execution package integrity is invalid")
    for relative, expected in package["files_sha256"].items():
        _validate_file_binding(root, relative, expected, "execution package source")

    expected_components, expected_clean, expected_corrupted = _format_component_grid()
    ledgers = report.get("scheduler_ledgers")
    if not isinstance(ledgers, dict) or set(ledgers) != {"clean", "corrupt"}:
        raise RuntimeError("format-contrast scheduler ledger bindings are invalid")
    ledger_records: dict[str, list[dict[str, Any]]] = {}
    for phase, count in (("clean", 12), ("corrupt", 144)):
        binding = ledgers[phase]
        if not isinstance(binding, dict) or set(binding) != {"path", "sha256", "ledger_sha256"}:
            raise RuntimeError(f"format-contrast {phase} scheduler binding is invalid")
        ledger_path = _validate_file_binding(
            root, binding["path"], binding["sha256"], f"{phase} scheduler ledger"
        )
        ledger = _read_object(ledger_path, f"{phase} scheduler ledger")
        if (
            ledger.get("schema_version") != 1
            or ledger.get("status") != "complete"
            or ledger.get("ledger_sha256") != _producer_canonical_hash(ledger, "ledger_sha256")
            or ledger.get("ledger_sha256") != binding["ledger_sha256"]
            or ledger.get("scheduler") != "bounded independent immutable format-contrast cells"
            or ledger.get("execution_package") != package_binding
            or type(ledger.get("cell_workers")) is not int
            or ledger.get("cell_workers") != 4
            or type(ledger.get("peak_active")) is not int
            or not 1 <= ledger["peak_active"] <= 4
            or ledger.get("failure") is not None
            or not isinstance(ledger.get("records"), list)
            or len(ledger["records"]) != count
        ):
            raise RuntimeError(f"format-contrast {phase} scheduler ledger is invalid")
        ledger_records[phase] = ledger["records"]
        for record in ledger["records"]:
            if (
                not isinstance(record, dict)
                or not isinstance(record.get("key"), str)
                or not isinstance(record.get("command"), list)
                or not record["command"]
                or not all(isinstance(value, str) for value in record["command"])
                or type(record.get("pid")) is not int
                or record["pid"] <= 0
                or not isinstance(record.get("log_path"), str)
                or not isinstance(record.get("output_paths"), list)
                or len(record["output_paths"]) != 2
                or record.get("exit_status") != 0
                or record.get("missing_output_paths") != []
                or not isinstance(record.get("started_at_utc"), str)
                or not isinstance(record.get("ended_at_utc"), str)
                or not isinstance(record.get("output_sha256"), dict)
            ):
                raise RuntimeError(f"format-contrast {phase} scheduler record is invalid")
            log_path = _strict_relative_file(root, record["log_path"], "scheduler task log")
            if sha256_file(log_path) != _require_sha256(
                record.get("log_sha256"), "scheduler task log SHA-256"
            ):
                raise RuntimeError(
                    f"scheduler task log SHA-256 mismatch: {record['log_path']}"
                )
            if set(record["output_sha256"]) != set(record["output_paths"]):
                raise RuntimeError(f"scheduler output binding is incomplete: {record['key']}")
            for output_relative in record["output_paths"]:
                _validate_file_binding(
                    root,
                    output_relative,
                    record["output_sha256"][output_relative],
                    "scheduler output",
                )

    macro_relative = (
        f"outputs/bootstrap/{FORMAT_CONTRAST_ATTEMPT}/"
        f"{FORMAT_CONTRAST_ATTEMPT}_joint_macro.json"
    )
    component_hashes = report.get("component_artifacts_sha256")
    artifact_hashes = report.get("artifacts_sha256")
    supporting_hashes = report.get("supporting_evidence_sha256")
    evidence_hashes = report.get("evidence_files_sha256")
    if not isinstance(component_hashes, dict) or set(component_hashes) != expected_components:
        raise RuntimeError("format-contrast component grid is not the exact 12 clean/144 corrupt grid")
    if (
        not isinstance(artifact_hashes, dict)
        or set(artifact_hashes) != expected_components | {macro_relative}
        or any(artifact_hashes.get(path) != digest for path, digest in component_hashes.items())
        or artifact_hashes.get(macro_relative) != report.get("joint_macro_artifact_sha256")
    ):
        raise RuntimeError("format-contrast artifact binding must contain 156 components and one macro")
    schedules = report.get("bootstrap_schedules")
    clean_caches = report.get("clean_arm_caches")
    draw_caches = report.get("draw_caches")
    if (
        not isinstance(schedules, dict)
        or set(schedules) != set(DATASETS)
        or not isinstance(clean_caches, dict)
        or len(clean_caches) != 12
        or not isinstance(draw_caches, dict)
        or set(draw_caches) != expected_corrupted
        or not isinstance(supporting_hashes, dict)
        or len(supporting_hashes) != 160
        or not isinstance(evidence_hashes, dict)
        or len(evidence_hashes) != 317
        or evidence_hashes != {**artifact_hashes, **supporting_hashes}
    ):
        raise RuntimeError("format-contrast supporting evidence chain is incomplete")
    for relative, expected in evidence_hashes.items():
        _validate_file_binding(root, relative, expected, "format-contrast evidence")

    documents = {
        relative: _read_object(root / relative, "format-contrast component")
        for relative in expected_components
    }
    run_hashes: set[str] = set()
    for relative, document in documents.items():
        parts = Path(relative).stem.split("__")
        dataset, model = parts[0], parts[1]
        expected_endpoint = "tt100k-height" if dataset == "tt100k" else "area"
        expected_labels = (
            {"all", "small_like", "large_like"}
            if dataset == "tt100k"
            else {"all", "small", "medium", "large"}
        )
        expected_fields = {
            "schema_version",
            "method",
            "endpoint_type",
            "annotation",
            "n_images",
            "n_boot",
            "seed",
            "point",
            "percentile_intervals",
            "input_hashes",
            "bootstrap_schedule",
            "clean_arm_cache",
            "sign_convention",
            "artifact_sha256",
        }
        if dataset == "tt100k":
            expected_fields.add("height_bins_px")
        if relative in expected_corrupted:
            expected_fields.add("temporary_draw_cache")
        if (
            set(document) != expected_fields
            or document.get("artifact_sha256")
            != _producer_canonical_hash(document, "artifact_sha256")
            or document.get("schema_version") != 1
            or document.get("method") != FORMAT_METHOD
            or document.get("n_boot") != 2000
            or document.get("endpoint_type") != expected_endpoint
            or not isinstance(document.get("n_images"), int)
            or document["n_images"] <= 0
            or set(document.get("point", {})) != {"delta_q", "delta_e", "delta_psi"}
            or set(document.get("percentile_intervals", {}))
            != {"percentiles", "delta_q", "delta_e", "delta_psi"}
            or document.get("sign_convention") != FORMAT_SIGN_CONVENTION
        ):
            raise RuntimeError(f"format-contrast component schema/grid is invalid: {relative}")
        point = document["point"]
        intervals = document["percentile_intervals"]
        if (
            not isinstance(point["delta_q"], dict)
            or set(point["delta_q"]) != expected_labels
            or not isinstance(point["delta_e"], dict)
            or set(point["delta_e"]) != expected_labels
            or any(not _finite_number(value) for value in point["delta_q"].values())
            or any(not _finite_number(value) for value in point["delta_e"].values())
            or not _finite_number(point["delta_psi"])
            or intervals.get("percentiles") != [2.5, 50.0, 97.5]
            or not isinstance(intervals["delta_q"], dict)
            or set(intervals["delta_q"]) != expected_labels
            or not isinstance(intervals["delta_e"], dict)
            or set(intervals["delta_e"]) != expected_labels
            or any(
                not isinstance(values, list)
                or len(values) != 3
                or any(not _finite_number(value) for value in values)
                for group in (intervals["delta_q"], intervals["delta_e"])
                for values in group.values()
            )
            or not isinstance(intervals["delta_psi"], list)
            or len(intervals["delta_psi"]) != 3
            or any(not _finite_number(value) for value in intervals["delta_psi"])
        ):
            raise RuntimeError(f"format-contrast component statistics are invalid: {relative}")
        small_label, large_label = (
            ("small_like", "large_like") if dataset == "tt100k" else ("small", "large")
        )
        if not math.isclose(
            float(point["delta_psi"]),
            float(point["delta_e"][small_label]) - float(point["delta_e"][large_label]),
            rel_tol=0.0,
            abs_tol=1e-15,
        ):
            raise RuntimeError(f"format-contrast delta_psi algebra is invalid: {relative}")
        if dataset == "tt100k" and document.get("height_bins_px") != {
            "small_like": {"min": 0.0, "max": 24.0},
            "large_like": {"min": 48.0, "max": None},
        }:
            raise RuntimeError(f"TT100K height endpoint bins are invalid: {relative}")
        annotation = document.get("annotation")
        configured_annotation = Path(config_by_dataset[dataset]["annotations"])
        if not configured_annotation.is_absolute():
            configured_annotation = Path(execution_root) / configured_annotation
        if (
            not isinstance(annotation, dict)
            or set(annotation) != {"path", "sha256"}
            or annotation.get("sha256") != report.get("annotations_sha256", {}).get(dataset)
            or not isinstance(annotation.get("path"), str)
            or not Path(annotation["path"]).is_absolute()
            or ".." in Path(annotation["path"]).parts
            or Path(annotation["path"]) != configured_annotation
            or document["n_images"] != config_by_dataset[dataset]["expected_images"]
        ):
            raise RuntimeError(f"format-contrast annotation binding is invalid: {relative}")
        schedule = document.get("bootstrap_schedule")
        if (
            not isinstance(schedule, dict)
            or set(schedule)
            != {"path", "sha256", "n_boot", "seed", "n_images", "image_ids_sha256"}
            or schedule != schedules[dataset]
            or schedule.get("n_boot") != 2000
            or schedule.get("n_images") != document["n_images"]
            or schedule.get("seed") != document.get("seed")
            or schedule.get("seed") != _format_dataset_seed(config["seed_namespace"], dataset)
        ):
            raise RuntimeError(f"format-contrast schedule binding is invalid: {relative}")
        inputs = document.get("input_hashes")
        if not isinstance(inputs, dict) or set(inputs) != set(FORMAT_ARM_NAMES):
            raise RuntimeError(f"format-contrast input binding is incomplete: {relative}")
        for binding in inputs.values():
            if not isinstance(binding, dict) or set(binding) != {
                "prediction_sha256",
                "input_record_sha256",
                "input_manifest_sha256",
                "image_ids_sha256",
                "run_record_sha256",
            }:
                raise RuntimeError(f"format-contrast input binding is invalid: {relative}")
            for field, value in binding.items():
                _require_sha256(value, f"format-contrast {field}")
            if binding["image_ids_sha256"] != schedule.get("image_ids_sha256"):
                raise RuntimeError(f"format-contrast image universe mismatch: {relative}")
            run_hashes.add(binding["run_record_sha256"])
        clean_reference = document.get("clean_arm_cache")
        if (
            not isinstance(clean_reference, dict)
            or set(clean_reference) != {"path", "sha256", "identity_sha256"}
            or clean_caches.get(clean_reference["path"])
            != {
                "sha256": clean_reference["sha256"],
                "identity_sha256": clean_reference["identity_sha256"],
            }
        ):
            raise RuntimeError(f"format-contrast clean-cache binding is invalid: {relative}")
        _require_sha256(clean_reference["sha256"], "format-contrast clean-cache SHA-256")
        _require_sha256(
            clean_reference["identity_sha256"], "format-contrast clean-cache identity SHA-256"
        )
        temporary = document.get("temporary_draw_cache")
        if relative in expected_clean:
            if temporary is not None:
                raise RuntimeError(f"clean format-contrast component binds a draw cache: {relative}")
            if (
                any(value != 0.0 for value in point["delta_e"].values())
                or point["delta_psi"] != 0.0
                or any(
                    value != 0.0
                    for values in intervals["delta_e"].values()
                    for value in values
                )
                or any(value != 0.0 for value in intervals["delta_psi"])
                or inputs["int8_corrupt"] != inputs["int8_clean"]
                or inputs["fp8_corrupt"] != inputs["fp8_clean"]
            ):
                raise RuntimeError(f"clean format-contrast identity is invalid: {relative}")
        else:
            if (
                not isinstance(temporary, dict)
                or set(temporary) != {"path", "sha256", "identity_sha256"}
                or draw_caches.get(relative) != temporary
            ):
                raise RuntimeError(f"corrupt format-contrast draw-cache binding is invalid: {relative}")
            _require_sha256(temporary["sha256"], "format-contrast draw-cache SHA-256")
            _require_sha256(
                temporary["identity_sha256"], "format-contrast draw-cache identity SHA-256"
            )

    for phase, relatives in (("clean", expected_clean), ("corrupt", expected_corrupted)):
        by_key = {
            record.get("key"): record
            for record in ledger_records[phase]
            if isinstance(record, dict)
        }
        expected_keys = {f"{phase}:{Path(relative).stem}" for relative in relatives}
        if len(by_key) != len(ledger_records[phase]) or set(by_key) != expected_keys:
            raise RuntimeError(f"format-contrast {phase} scheduler task grid is invalid")
        for relative in relatives:
            record = by_key[f"{phase}:{Path(relative).stem}"]
            auxiliary = (
                documents[relative]["clean_arm_cache"]["path"]
                if phase == "clean"
                else documents[relative]["temporary_draw_cache"]["path"]
            )
            expected_outputs = [relative, auxiliary]
            expected_log = (
                f"outputs/logs/{FORMAT_CONTRAST_ATTEMPT}/format_contrast_cells/"
                f"{phase}__{Path(relative).stem}.log"
            )
            if record.get("output_paths") != expected_outputs or record.get("log_path") != expected_log:
                raise RuntimeError(f"format-contrast {phase} scheduler task binding is invalid")
            _validate_format_scheduler_command(
                record["command"], phase, relative, documents[relative], execution_root
            )

    for dataset, model in product(DATASETS, MODEL_ORDER):
        clean_relative = (
            f"outputs/bootstrap/{FORMAT_CONTRAST_ATTEMPT}/{dataset}__{model}__clean-s0.json"
        )
        clean_inputs = documents[clean_relative]["input_hashes"]
        prefix = f"outputs/bootstrap/{FORMAT_CONTRAST_ATTEMPT}/{dataset}__{model}__"
        for relative in expected_corrupted:
            if not relative.startswith(prefix):
                continue
            inputs = documents[relative]["input_hashes"]
            if (
                inputs["int8_clean"] != clean_inputs["int8_clean"]
                or inputs["fp8_clean"] != clean_inputs["fp8_clean"]
                or documents[relative]["clean_arm_cache"]
                != documents[clean_relative]["clean_arm_cache"]
            ):
                raise RuntimeError(
                    f"format-contrast components do not share clean input/cache bindings: {relative}"
                )

    local_run_hashes = {
        sha256_file(path)
        for path in (root / "manifests" / "runs").glob("**/*.json")
        if path.is_file()
    }
    if not run_hashes or not run_hashes.issubset(local_run_hashes):
        raise RuntimeError("format-contrast run-record SHA-256 witnesses are incomplete")

    macro = _read_object(root / macro_relative, "format-contrast joint macro")
    macro_components = macro.get("component_artifacts")
    if (
        macro.get("artifact_sha256") != _producer_canonical_hash(macro, "artifact_sha256")
        or macro.get("schema_version") != 1
        or macro.get("method")
        != "joint paired image bootstrap; one shared resample per dataset and replicate across all models, corruptions, severities, and formats"
        or macro.get("n_boot") != 2000
        or macro.get("seed_namespace") != config.get("seed_namespace")
        or macro.get("component_cells") != 144
        or macro.get("annotations_sha256") != report.get("annotations_sha256")
        or macro.get("bootstrap_schedules") != schedules
        or not isinstance(macro_components, dict)
        or set(macro_components) != expected_corrupted
    ):
        raise RuntimeError("format-contrast joint macro grid/hash is invalid")
    for relative, binding in macro_components.items():
        component = documents[relative]
        if (
            not isinstance(binding, dict)
            or set(binding)
            != {
                "artifact_sha256",
                "annotation_sha256",
                "input_hashes",
                "bootstrap_schedule",
                "clean_arm_cache",
                "temporary_draw_cache",
            }
            or binding.get("artifact_sha256") != component_hashes[relative]
            or binding.get("annotation_sha256") != component.get("annotation", {}).get("sha256")
            or binding.get("input_hashes") != component.get("input_hashes")
            or binding.get("bootstrap_schedule") != component.get("bootstrap_schedule")
            or binding.get("clean_arm_cache") != component.get("clean_arm_cache")
            or binding.get("temporary_draw_cache") != component.get("temporary_draw_cache")
        ):
            raise RuntimeError(f"format-contrast joint macro binding is invalid: {relative}")
    coverage = macro.get("coverage")
    macro_points = macro.get("point", {})
    macro_intervals = macro.get("percentile_intervals", {})
    expected_coverage = {
        "four_dataset_macro_delta_e": 144,
        "area_macro_delta_psi": 108,
        "tt100k_height_macro_delta_psi": 36,
    }
    if (
        not isinstance(coverage, dict)
        or set(coverage) != set(expected_coverage)
        or {
            name: values.get("planned_component_count")
            for name, values in coverage.items()
            if isinstance(values, dict)
        }
        != expected_coverage
        or any(coverage[name].get("total_replicates") != 2000 for name in expected_coverage)
        or any(
            type(coverage[name].get("finite_complete_replicates")) is not int
            or not 0 <= coverage[name]["finite_complete_replicates"] <= 2000
            for name in expected_coverage
        )
        or set(macro.get("point", {})) != set(expected_coverage)
        or set(macro.get("percentile_intervals", {})) != set(expected_coverage)
    ):
        raise RuntimeError("format-contrast joint macro coverage is invalid")
    if (
        any(not _finite_number(value) for value in macro_points.values())
        or any(
            not isinstance(values, list)
            or len(values) != 3
            or any(not _finite_number(value) for value in values)
            for values in macro_intervals.values()
        )
    ):
        raise RuntimeError("format-contrast joint macro statistics are invalid")
    corrupted_documents = [documents[relative] for relative in sorted(expected_corrupted)]
    expected_macro_points = {
        "four_dataset_macro_delta_e": float(
            np.mean([document["point"]["delta_e"]["all"] for document in corrupted_documents])
        ),
        "area_macro_delta_psi": float(
            np.mean(
                [
                    document["point"]["delta_psi"]
                    for document in corrupted_documents
                    if document["endpoint_type"] == "area"
                ]
            )
        ),
        "tt100k_height_macro_delta_psi": float(
            np.mean(
                [
                    document["point"]["delta_psi"]
                    for document in corrupted_documents
                    if document["endpoint_type"] == "tt100k-height"
                ]
            )
        ),
    }
    if any(
        not math.isclose(
            float(macro_points[name]), expected, rel_tol=1e-12, abs_tol=1e-15
        )
        for name, expected in expected_macro_points.items()
    ):
        raise RuntimeError("format-contrast joint macro point recomputation mismatch")
    return report


def _parse_benchmark_log(document: str, label: str) -> tuple[dict[str, Any], str]:
    header, separator, raw = document.partition("\n")
    if not separator or not header.startswith(BENCHMARK_LOG_PREFIX):
        raise RuntimeError(f"invalid benchmark log envelope: {label}")
    try:
        metadata = json.loads(header[len(BENCHMARK_LOG_PREFIX) :])
    except json.JSONDecodeError as error:
        raise RuntimeError(f"invalid benchmark log envelope: {label}") from error
    if (
        not isinstance(metadata, dict)
        or metadata.get("schema_version") != 1
        or metadata.get("log_format") != BENCHMARK_LOG_FORMAT
        or metadata.get("raw_output_bytes") != len(raw.encode("utf-8"))
        or metadata.get("raw_output_sha256") != hashlib.sha256(raw.encode("utf-8")).hexdigest()
    ):
        raise RuntimeError(f"invalid benchmark log envelope: {label}")
    return metadata, raw


def _parse_benchmark_metrics(raw: str, command: list[str], label: str) -> dict[str, float]:
    running = re.findall(
        r"^&&&& RUNNING TensorRT\.trtexec(?:\s+\[[^\]\r\n]*\])*\s+#\s+(.+?)\s*$",
        raw,
        flags=re.MULTILINE,
    )
    passed = list(
        re.finditer(
            r"^&&&& PASSED TensorRT\.trtexec(?:\s+\[[^\]\r\n]*\])*\s+#\s+(.+?)\s*$",
            raw,
            flags=re.MULTILINE,
        )
    )
    failed = re.findall(r"^&&&& FAILED TensorRT\.trtexec\b[^\r\n]*$", raw, flags=re.MULTILINE)
    try:
        running_command = shlex.split(running[0]) if len(running) == 1 else None
        passed_command = shlex.split(passed[0].group(1)) if len(passed) == 1 else None
    except ValueError as error:
        raise RuntimeError(f"benchmark log successful command envelope is invalid: {label}") from error
    if running_command != command or passed_command != command or failed:
        raise RuntimeError(f"benchmark log successful command envelope is invalid: {label}")
    log_prefix = r"^[ \t]*(?:\[[^\]\r\n]+\][ \t]*)*"
    starts = list(
        re.finditer(
            log_prefix + r"===\s*Performance summary\s*===\s*$",
            raw,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )
    if len(starts) != 1 or len(passed) != 1 or passed[0].start() <= starts[0].end():
        raise RuntimeError(f"benchmark log metrics are incomplete: {label}")
    block = raw[starts[0].end() : passed[0].start()]
    throughput_pattern = log_prefix + rf"Throughput:[ \t]*({BENCHMARK_NUMBER})[ \t]*qps[ \t]*$"
    latency_pattern = (
        log_prefix
        + rf"Latency:[ \t]*min[ \t]*=[ \t]*({BENCHMARK_NUMBER})[ \t]*ms,[ \t]*"
        rf"max[ \t]*=[ \t]*({BENCHMARK_NUMBER})[ \t]*ms,[ \t]*"
        rf"mean[ \t]*=[ \t]*({BENCHMARK_NUMBER})[ \t]*ms,[ \t]*"
        rf"median[ \t]*=[ \t]*({BENCHMARK_NUMBER})[ \t]*ms,[ \t]*"
        rf"percentile\(90%\)[ \t]*=[ \t]*({BENCHMARK_NUMBER})[ \t]*ms,[ \t]*"
        rf"percentile\(95%\)[ \t]*=[ \t]*({BENCHMARK_NUMBER})[ \t]*ms,[ \t]*"
        rf"percentile\(99%\)[ \t]*=[ \t]*({BENCHMARK_NUMBER})[ \t]*ms[ \t]*$"
    )
    flags = re.IGNORECASE | re.MULTILINE
    throughput = re.findall(throughput_pattern, block, flags=flags)
    latency = re.findall(
        latency_pattern,
        block,
        flags=flags,
    )
    if (
        len(throughput) != 1
        or len(latency) != 1
        or len(re.findall(throughput_pattern, raw, flags=flags)) != 1
        or len(re.findall(latency_pattern, raw, flags=flags)) != 1
    ):
        raise RuntimeError(f"benchmark log metrics are incomplete: {label}")
    try:
        throughput_value = float(throughput[0])
        minimum, maximum, mean, median, p90, p95, p99 = map(float, latency[0])
    except ValueError as error:
        raise RuntimeError(f"benchmark log metrics are invalid: {label}") from error
    values = (throughput_value, minimum, maximum, mean, median, p90, p95, p99)
    if (
        any(not math.isfinite(value) or value <= 0 for value in values)
        or not minimum <= median <= p90 <= p95 <= p99 <= maximum
        or not minimum <= mean <= maximum
    ):
        raise RuntimeError(f"benchmark log metrics are invalid: {label}")
    return {
        "throughput_qps": throughput_value,
        "latency_mean_ms": mean,
        "latency_median_ms": median,
        "latency_p99_ms": p99,
    }


def validate_deployment_benchmark(artifact_root: Path) -> dict[str, Any]:
    """Validate a complete Task 3 chain without wiring it into the paper build."""
    root = Path(artifact_root).resolve()
    config_path = root / "configs" / f"{DEPLOYMENT_ATTEMPT}.json"
    config = _read_object(config_path, "deployment benchmark config")
    expected_output = f"outputs/benchmarks/{DEPLOYMENT_ATTEMPT}"
    expected_report = f"outputs/reports/{DEPLOYMENT_ATTEMPT}_complete.json"
    expected_treatments = f"{expected_output}/engine_treatments.json"
    registry_sets = config.get("registry_sets")
    if (
        config.get("schema_version") != 1
        or config.get("attempt") != DEPLOYMENT_ATTEMPT
        or tuple(config.get("models", ())) != tuple(MODEL_ORDER)
        or tuple(config.get("precisions", ())) != DEPLOYMENT_PRECISIONS
        or config.get("repetitions") != 3
        or config.get("warmup_ms") != 5000
        or config.get("duration_s") != 30
        or not isinstance(registry_sets, list)
        or len(registry_sets) != 4
        or any(not isinstance(row, dict) for row in registry_sets)
        or tuple(row.get("dataset") for row in registry_sets) != tuple(DATASETS)
        or any(
            not all(
                isinstance(row.get(field), str) and row[field]
                for field in ("ladder_registry", "engine_registry_dir", "onnx_registry_dir")
            )
            for row in registry_sets
        )
        or config.get("output_dir") != expected_output
        or config.get("report") != expected_report
        or config.get("engine_treatments") != expected_treatments
    ):
        raise RuntimeError("deployment benchmark config/grid contract is invalid")
    for field in ("required_gpu_name", "required_hostname", "required_project_root"):
        value = config.get(field)
        if not isinstance(value, str) or not value.strip() or value.strip().lower() in {
            "unknown",
            "unrecorded",
            "tbd",
        }:
            raise RuntimeError(f"deployment benchmark config {field} is invalid")
    for field in ("required_project_root", "required_engine_root", "required_trtexec"):
        value = config.get(field)
        path = Path(value) if isinstance(value, str) else None
        if path is None or not path.is_absolute() or ".." in path.parts:
            raise RuntimeError(f"deployment benchmark config {field} path is invalid")
    report_path = _strict_relative_file(root, config["report"], "deployment completion report")
    report = _read_object(report_path, "deployment completion report")
    if report.get("report_sha256") != _producer_canonical_hash(report, "report_sha256"):
        raise RuntimeError("deployment completion report has an invalid canonical SHA-256")
    config_sha = sha256_file(config_path)
    if (
        report.get("schema_version") != 1
        or report.get("attempt") != DEPLOYMENT_ATTEMPT
        or report.get("config_sha256") != config_sha
        or report.get("engine_conditions") != 36
        or report.get("repetitions_per_engine") != 3
        or report.get("repetition_records") != 108
        or report.get("raw_logs") != 108
        or report.get("limitations") != ["No memory claim.", "No power or energy claim."]
    ):
        raise RuntimeError("deployment completion report grid/count contract is invalid")

    expected_conditions = set(product(DATASETS, MODEL_ORDER, DEPLOYMENT_PRECISIONS))
    expected_records = {
        f"{dataset}__{model}__{precision}__rep-{repetition:02d}.json"
        for dataset, model, precision in expected_conditions
        for repetition in (1, 2, 3)
    }
    expected_logs = {Path(name).with_suffix(".log").name for name in expected_records}
    records_sha = report.get("records_sha256")
    logs_sha = report.get("logs_sha256")
    if (
        not isinstance(records_sha, dict)
        or set(records_sha) != expected_records
        or not isinstance(logs_sha, dict)
        or set(logs_sha) != expected_logs
        or any(Path(name).name != name or ".." in Path(name).parts for name in records_sha)
        or any(Path(name).name != name or ".." in Path(name).parts for name in logs_sha)
    ):
        raise RuntimeError("deployment completion report has noncanonical record/log paths")

    records_by_condition: dict[tuple[str, str, str], list[dict[str, Any]]] = {}
    run_ids: set[str] = set()
    raw_hashes: set[str] = set()
    log_hashes: set[str] = set()
    common_runtime: dict[str, Any] | None = None
    record_dir = root / expected_output
    log_dir = record_dir / "logs"
    required_record_fields = {
        "schema_version",
        "attempt",
        "config_sha256",
        "condition_id",
        "dataset",
        "model",
        "precision",
        "repetition",
        "run_id",
        "log_format",
        "trtexec_return_code",
        "raw_output_sha256",
        "started_at_utc",
        "ended_at_utc",
        "engine",
        "engine_sha256",
        "engine_bytes",
        "ladder_registry",
        "ladder_registry_sha256",
        "engine_registry",
        "engine_registry_sha256",
        "onnx_registry",
        "onnx_registry_sha256",
        "input_binding_registry",
        "input_binding_registry_sha256",
        "input_onnx",
        "input_onnx_sha256",
        "input_name",
        "input_shape",
        "input_dynamic",
        "shape_flag_required",
        "trtexec",
        "trtexec_sha256",
        "gpu_runtime_identity",
        "gpu_idle_query_command",
        "gpu_idle_query_output",
        "trtexec_command",
        "log_sha256",
        "throughput_qps",
        "latency_mean_ms",
        "latency_median_ms",
        "latency_p99_ms",
    }
    for record_name in sorted(expected_records):
        log_name = Path(record_name).with_suffix(".log").name
        record_path = record_dir / record_name
        log_path = log_dir / log_name
        if not record_path.is_file() or sha256_file(record_path) != _require_sha256(
            records_sha[record_name], "benchmark record SHA-256"
        ):
            raise RuntimeError(f"benchmark record SHA-256 mismatch: {record_name}")
        if not log_path.is_file() or sha256_file(log_path) != _require_sha256(
            logs_sha[log_name], "benchmark log SHA-256"
        ):
            raise RuntimeError(f"benchmark log SHA-256 mismatch: {log_name}")
        record = _read_object(record_path, "benchmark record")
        if set(record) != required_record_fields:
            raise RuntimeError(f"benchmark record field set is invalid: {record_name}")
        stem_parts = record_name.removesuffix(".json").rsplit("__rep-", 1)
        identity_parts = stem_parts[0].split("__")
        if len(identity_parts) != 3:
            raise RuntimeError(f"benchmark record condition identity is invalid: {record_name}")
        dataset, model, precision = identity_parts
        repetition = int(stem_parts[1])
        condition = (dataset, model, precision)
        condition_id = "__".join(condition)
        if (
            condition not in expected_conditions
            or repetition not in {1, 2, 3}
            or record.get("schema_version") != 1
            or record.get("attempt") != DEPLOYMENT_ATTEMPT
            or record.get("config_sha256") != config_sha
            or record.get("condition_id") != condition_id
            or (record.get("dataset"), record.get("model"), record.get("precision")) != condition
            or record.get("repetition") != repetition
            or record.get("log_format") != BENCHMARK_LOG_FORMAT
            or record.get("trtexec_return_code") != 0
            or type(record.get("engine_bytes")) is not int
            or record["engine_bytes"] <= 0
            or not isinstance(record.get("input_name"), str)
            or not record["input_name"]
            or not isinstance(record.get("input_shape"), list)
            or len(record["input_shape"]) != 4
            or any(type(value) is not int or value <= 0 for value in record["input_shape"])
            or not isinstance(record.get("input_dynamic"), bool)
            or not isinstance(record.get("shape_flag_required"), bool)
            or record.get("gpu_idle_query_command") != GPU_IDLE_COMMAND
            or not isinstance(record.get("gpu_idle_query_output"), str)
        ):
            raise RuntimeError(f"benchmark record grid/provenance is invalid: {record_name}")
        try:
            started = datetime.fromisoformat(record["started_at_utc"])
            ended = datetime.fromisoformat(record["ended_at_utc"])
        except (TypeError, ValueError) as error:
            raise RuntimeError(f"benchmark record timestamp is invalid: {record_name}") from error
        if started.tzinfo is None or ended.tzinfo is None or ended < started:
            raise RuntimeError(f"benchmark record timestamp is invalid: {record_name}")
        for field in (
            "engine_sha256",
            "ladder_registry_sha256",
            "engine_registry_sha256",
            "onnx_registry_sha256",
            "input_binding_registry_sha256",
            "input_onnx_sha256",
            "trtexec_sha256",
            "raw_output_sha256",
            "log_sha256",
        ):
            _require_sha256(record.get(field), f"benchmark record {field}")
        for field in (
            "engine",
            "ladder_registry",
            "engine_registry",
            "onnx_registry",
            "input_binding_registry",
            "input_onnx",
            "trtexec",
        ):
            value = record.get(field)
            path = Path(value) if isinstance(value, str) else None
            if path is None or not path.is_absolute() or ".." in path.parts:
                raise RuntimeError(f"benchmark record {field} path is invalid: {record_name}")
        try:
            Path(record["engine"]).relative_to(Path(config["required_engine_root"]))
            for field in (
                "ladder_registry",
                "engine_registry",
                "onnx_registry",
                "input_binding_registry",
                "input_onnx",
            ):
                Path(record[field]).relative_to(Path(config["required_project_root"]))
        except ValueError as error:
            raise RuntimeError(f"benchmark record path escapes configured roots: {record_name}") from error
        if record["log_sha256"] != logs_sha[log_name]:
            raise RuntimeError(f"benchmark record/log SHA-256 binding mismatch: {record_name}")
        if (
            record.get("trtexec") != config.get("required_trtexec")
            or record.get("trtexec_sha256")
            != record.get("gpu_runtime_identity", {}).get("trtexec_sha256")
        ):
            raise RuntimeError(f"benchmark trtexec binding mismatch: {record_name}")
        command = [
            record["trtexec"],
            f"--loadEngine={record['engine']}",
            "--warmUp=5000",
            "--duration=30",
        ]
        if record.get("shape_flag_required"):
            shape = "x".join(str(value) for value in record.get("input_shape", ()))
            command.append(f"--shapes={record.get('input_name')}:{shape}")
        if record.get("trtexec_command") != command:
            raise RuntimeError(f"benchmark command binding mismatch: {record_name}")
        runtime = record.get("gpu_runtime_identity")
        if (
            not isinstance(runtime, dict)
            or set(runtime) != DEPLOYMENT_RUNTIME_FIELDS
            or any(not isinstance(value, str) or not value.strip() for value in runtime.values())
            or runtime.get("hostname") != config.get("required_hostname")
            or runtime.get("project_root") != config.get("required_project_root")
            or runtime.get("gpu_name") != config.get("required_gpu_name")
            or runtime.get("trtexec_executable") != config.get("required_trtexec")
            or runtime.get("trtexec_sha256") != record.get("trtexec_sha256")
        ):
            raise RuntimeError(f"benchmark runtime binding mismatch: {record_name}")
        runtime_patterns = {
            "gpu_uuid": r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
            "driver_version": r"[0-9]+(?:\.[0-9]+){1,3}",
            "cuda_version": r"[0-9]+(?:\.[0-9]+){1,2}",
            "python_version": r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+._a-zA-Z0-9]*)?",
        }
        if (
            any(re.fullmatch(pattern, runtime[field]) is None for field, pattern in runtime_patterns.items())
            or runtime["platform"].strip().lower() in {"unknown", "unrecorded", "tbd"}
            or re.search(
                rf"(?<![0-9.]){re.escape(runtime['tensorrt_python_version'])}(?![0-9.])",
                runtime["trtexec_version_output"],
            )
            is None
        ):
            raise RuntimeError(f"benchmark runtime identity is invalid: {record_name}")
        for line in record["gpu_idle_query_output"].splitlines():
            value = line.strip()
            if not value or value.lower() in {"n/a", "[n/a]", "no running processes found"}:
                continue
            if not value.isdigit():
                raise RuntimeError(f"benchmark GPU-idle output is malformed: {record_name}")
            raise RuntimeError(f"benchmark GPU was not idle: {record_name}")
        if common_runtime is None:
            common_runtime = runtime
        elif runtime != common_runtime:
            raise RuntimeError("benchmark records do not share one runtime identity")
        metadata, raw = _parse_benchmark_log(log_path.read_text(encoding="utf-8"), log_name)
        expected_metadata = {
            "attempt": DEPLOYMENT_ATTEMPT,
            "config_sha256": config_sha,
            "condition_id": condition_id,
            "dataset": dataset,
            "model": model,
            "precision": precision,
            "repetition": repetition,
            "run_id": record.get("run_id"),
            "started_at_utc": record.get("started_at_utc"),
            "ended_at_utc": record.get("ended_at_utc"),
            "return_code": 0,
            "engine_sha256": record.get("engine_sha256"),
            "trtexec_command": command,
        }
        if any(metadata.get(field) != value for field, value in expected_metadata.items()):
            raise RuntimeError(f"benchmark log metadata binding mismatch: {log_name}")
        raw_sha = hashlib.sha256(raw.encode("utf-8")).hexdigest()
        if record.get("raw_output_sha256") != raw_sha:
            raise RuntimeError(f"benchmark raw-output SHA-256 mismatch: {record_name}")
        metrics = _parse_benchmark_metrics(raw, command, log_name)
        for field, expected in metrics.items():
            value = record.get(field)
            if isinstance(value, bool) or not isinstance(value, (int, float)) or value != expected:
                raise RuntimeError(f"raw log metric mismatch for {field}: {record_name}")
        run_id = record.get("run_id")
        if not isinstance(run_id, str) or re.fullmatch(r"[0-9a-f]{32}", run_id) is None:
            raise RuntimeError(f"benchmark run_id is invalid: {record_name}")
        if run_id in run_ids or raw_sha in raw_hashes or record["log_sha256"] in log_hashes:
            raise RuntimeError("benchmark completion chain contains duplicate run evidence")
        run_ids.add(run_id)
        raw_hashes.add(raw_sha)
        log_hashes.add(record["log_sha256"])
        records_by_condition.setdefault(condition, []).append(record)

    if (
        set(records_by_condition) != expected_conditions
        or any({row["repetition"] for row in rows} != {1, 2, 3} for rows in records_by_condition.values())
    ):
        raise RuntimeError("deployment benchmark repetitions are not the exact 36 x {1,2,3} grid")

    treatments_path = _strict_relative_file(root, config["engine_treatments"], "engine treatments")
    if sha256_file(treatments_path) != _require_sha256(
        report.get("engine_treatments_sha256"), "engine treatments SHA-256"
    ):
        raise RuntimeError("engine treatment report binding mismatch")
    treatments = _read_object(treatments_path, "engine treatments")
    if (
        treatments.get("treatments_sha256")
        != _producer_canonical_hash(treatments, "treatments_sha256")
        or treatments.get("schema_version") != 1
        or treatments.get("attempt") != DEPLOYMENT_ATTEMPT
        or treatments.get("engine_conditions") != 36
        or not isinstance(treatments.get("records"), list)
        or len(treatments["records"]) != 36
    ):
        raise RuntimeError("engine treatment report integrity is invalid")
    treatment_by_condition: dict[tuple[Any, Any, Any], dict[str, Any]] = {}
    for treatment in treatments["records"]:
        if not isinstance(treatment, dict) or set(treatment) != ENGINE_TREATMENT_FIELDS:
            raise RuntimeError("engine treatment record is invalid")
        _require_sha256(treatment.get("onnx_sha256"), "engine treatment ONNX SHA-256")
        _require_sha256(treatment.get("engine_sha256"), "engine treatment engine SHA-256")
        _require_sha256(treatment.get("trtexec_sha256"), "engine treatment trtexec SHA-256")
        condition = (
            treatment.get("dataset"),
            treatment.get("model"),
            treatment.get("precision"),
        )
        if condition in treatment_by_condition:
            raise RuntimeError("engine treatment report contains a duplicate condition")
        treatment_by_condition[condition] = treatment
    if set(treatment_by_condition) != expected_conditions:
        raise RuntimeError("engine treatment report does not bind the exact 36-condition grid")
    for condition, treatment in treatment_by_condition.items():
        rows = records_by_condition[condition]
        reference = rows[0]
        invariant_fields = {
            "attempt",
            "condition_id",
            "dataset",
            "model",
            "precision",
            "engine",
            "engine_sha256",
            "engine_bytes",
            "ladder_registry",
            "ladder_registry_sha256",
            "engine_registry",
            "engine_registry_sha256",
            "onnx_registry",
            "onnx_registry_sha256",
            "input_binding_registry",
            "input_binding_registry_sha256",
            "input_onnx",
            "input_onnx_sha256",
            "input_name",
            "input_shape",
            "input_dynamic",
            "shape_flag_required",
            "trtexec",
            "trtexec_sha256",
        }
        if any(
            any(row.get(field) != reference.get(field) for field in invariant_fields)
            for row in rows[1:]
        ):
            raise RuntimeError(f"benchmark repetition provenance mismatch: {condition}")
        if (
            treatment.get("onnx_sha256") != reference.get("input_onnx_sha256")
            or treatment.get("engine_sha256") != reference.get("engine_sha256")
            or treatment.get("engine_bytes") != reference.get("engine_bytes")
            or treatment.get("tensorrt_python_version")
            != reference["gpu_runtime_identity"].get("tensorrt_python_version")
            or treatment.get("trtexec_sha256") != reference.get("trtexec_sha256")
            or any(
                row.get("engine_sha256") != reference.get("engine_sha256")
                or row.get("engine_bytes") != reference.get("engine_bytes")
                for row in rows
            )
        ):
            if treatment.get("onnx_sha256") != reference.get("input_onnx_sha256"):
                raise RuntimeError(f"ONNX treatment/benchmark binding mismatch: {condition}")
            raise RuntimeError(f"engine treatment/benchmark binding mismatch: {condition}")
    return report


def escape_tex(value: object) -> str:
    return str(value).replace("_", r"\_").replace("%", r"\%")


def fmt_ap(value: float) -> str:
    return f"{100.0 * value:.2f}"


def fmt_effect(value: float) -> str:
    return f"{100.0 * value:+.2f}"


def fmt_points(value: float) -> str:
    """Format an absolute AP-point quantity without a misleading sign."""
    return f"{100.0 * value:.3f}"


def write_booktabs(path: Path, frame: pd.DataFrame, columns: list[str], align: str) -> None:
    lines = [r"\begin{tabular}{" + align + "}", r"\toprule"]
    lines.append(" & ".join(escape_tex(c) for c in columns) + r" \\")
    lines.append(r"\midrule")
    for row in frame[columns].itertuples(index=False, name=None):
        lines.append(" & ".join(escape_tex(x) for x in row) + r" \\")
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    path.write_text("\n".join(lines) + "\n", encoding="utf-8")


def validate_final_report(reports: Path) -> dict[str, Any]:
    path = reports / "four_dataset_pilot_final_report.json"
    doc = read_json(path)
    if canonical_hash(doc, "four_dataset_report_sha256") != doc.get("four_dataset_report_sha256"):
        raise RuntimeError("four-dataset report has an invalid canonical SHA-256")
    checks = doc.get("checks", {})
    expected = {
        "datasets": 4,
        "runs_per_dataset": 117,
        "bootstrap_cells_per_dataset": 72,
        "bootstrap_replicates": 500,
        "all_dataset_completion_reports_hash_validated": True,
    }
    for key, value in expected.items():
        if checks.get(key) != value:
            raise RuntimeError(f"four-dataset report failed {key}: {checks.get(key)!r}")
    return doc


def expected_metric_hashes(dataset: str, report: dict[str, Any]) -> dict[str, str]:
    return dict(report["metric_sha256"])


def resolve_local_report_path(artifact_root: Path, reported_path: str) -> Path:
    """Map a report path to a file below the local artifact root, without escape."""
    root = artifact_root.resolve()
    raw = Path(reported_path)
    if ".." in raw.parts:
        raise RuntimeError(f"unsafe artifact path in report: {reported_path}")
    if raw.is_absolute():
        anchors = [index for index, part in enumerate(raw.parts) if part in {"outputs", "configs", "manifests"}]
        if not anchors:
            raise RuntimeError(f"remote artifact path has no local-root anchor: {reported_path}")
        relative = Path(*raw.parts[anchors[-1] :])
    else:
        relative = raw
    if not relative.parts or relative.parts[0] not in {"outputs", "configs", "manifests"}:
        raise RuntimeError(f"artifact path is not root-relative: {reported_path}")
    candidate = (root / relative).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as error:
        raise RuntimeError(f"artifact path escapes local root: {reported_path}") from error
    if not candidate.is_file():
        raise RuntimeError(f"bound artifact is missing: {relative}")
    return candidate


def validate_bound_file(artifact_root: Path, reported_path: str, expected_sha256: str) -> Path:
    path = resolve_local_report_path(artifact_root, reported_path)
    actual = sha256_file(path)
    if actual != expected_sha256:
        raise RuntimeError(f"artifact hash mismatch for {path.relative_to(artifact_root.resolve())}")
    return path


def validate_canonical_field(document: dict[str, Any], field: str, label: str) -> None:
    if canonical_hash(document, field) != document.get(field):
        raise RuntimeError(f"{label} has an invalid canonical SHA-256")


def validate_historical_ladder_config(
    *,
    ladder_config_sha256: str,
    current_config_sha256: str,
    master_config_sha256: str,
) -> None:
    """Allow only the audited pre-codec-control ladder config transition."""
    if ladder_config_sha256 != HISTORICAL_LADDER_CONFIG_SHA256:
        raise RuntimeError("unrecognized historical ladder config SHA-256")
    if current_config_sha256 != FINAL_P0_CONFIG_SHA256:
        raise RuntimeError("current P0 config does not match the audited finalized SHA-256")
    if master_config_sha256 != FINAL_P0_CONFIG_SHA256:
        raise RuntimeError("master report does not bind the audited finalized P0 config")


def validate_p0_completion(artifact_root: Path) -> dict[str, dict[str, Any]]:
    """Validate the master P0 report, every component, and bound analysis artifacts."""
    report_path = artifact_root / "outputs" / "reports" / P0_REPORT
    master = read_json(report_path)
    validate_canonical_field(master, "report_sha256", "P0 completion report")
    if master.get("status") != "complete":
        raise RuntimeError(f"P0 completion status is not complete: {master.get('status')!r}")
    if set(Path(name).name for name in master.get("reports", {})) != P0_COMPONENT_REPORTS:
        raise RuntimeError("P0 completion report does not bind the exact component-report set")
    current_config_path = validate_bound_file(
        artifact_root, master["config"], master["config_sha256"]
    )

    documents: dict[str, dict[str, Any]] = {P0_REPORT: master}
    for reported, expected_hash in master["reports"].items():
        path = validate_bound_file(artifact_root, reported, expected_hash)
        documents[path.name] = read_json(path)

    ladder = documents["coco_uniform_ladder_p0_v1.json"]
    validate_canonical_field(ladder, "report_sha256", "COCO uniform ladder report")
    # The ladder was completed before codec-control settings were appended to
    # the shared P0 config.  The master report binds the final config, while
    # its raw hash binds this immutable historical ladder report.  Validate
    # every concrete ladder artifact below rather than comparing the stale
    # intermediate config-file digest to the finalized config bytes.
    validate_historical_ladder_config(
        ladder_config_sha256=ladder.get("config_sha256", ""),
        current_config_sha256=sha256_file(current_config_path),
        master_config_sha256=master.get("config_sha256", ""),
    )
    calibration_path = resolve_local_report_path(artifact_root, ladder["calibration_list"])
    calibration = read_json(calibration_path)
    validate_canonical_field(calibration, "calibration_sha256", "COCO calibration manifest")
    if calibration["calibration_sha256"] != ladder["calibration_sha256"]:
        raise RuntimeError("COCO ladder calibration binding mismatch")
    validate_bound_file(
        artifact_root,
        ladder["combined_engine_registry"],
        ladder["combined_engine_registry_sha256"],
    )
    if len(ladder.get("artifacts", [])) != 12:
        raise RuntimeError("COCO uniform ladder report does not bind 12 engine conditions")
    for row in ladder["artifacts"]:
        validate_bound_file(artifact_root, row["onnx_registry"], row["onnx_registry_sha256"])
        validate_bound_file(artifact_root, row["engine_registry"], row["engine_registry_sha256"])

    parity = documents["coco_clean_fp16_parity_complete_v1.json"]
    validate_canonical_field(parity, "parity_completion_sha256", "COCO FP16 parity report")
    if len(parity.get("reports", [])) != 3:
        raise RuntimeError("COCO FP16 parity completion report does not bind three models")
    for row in parity["reports"]:
        nested_path = validate_bound_file(artifact_root, row["report"], row["sha256"])
        nested = read_json(nested_path)
        if not nested.get("pass") or nested.get("absolute_ap_gap", math.inf) > nested.get("tolerance", -math.inf):
            raise RuntimeError(f"COCO FP16 parity gate failed: {nested_path.name}")

    codec = documents["codec_control_p0_v1_complete.json"]
    validate_canonical_field(codec, "report_sha256", "codec-control completion report")
    validate_bound_file(artifact_root, codec["config"], codec["config_sha256"])
    if codec.get("attempt") != "codec_control_p0_v1" or len(codec.get("conditions", [])) != 36:
        raise RuntimeError("codec-control completion report does not bind exactly 36 conditions")
    for row in codec["conditions"]:
        validate_bound_file(artifact_root, row["metric"], row["sha256"])

    matched = documents["matched_codec_p0_v1_complete.json"]
    validate_canonical_field(matched, "report_sha256", "matched-codec completion report")
    validate_bound_file(artifact_root, matched["config"], matched["config_sha256"])
    if matched.get("attempt") != "matched_codec_p0_v1" or matched.get("cells") != 288 or matched.get("n_boot") != 500:
        raise RuntimeError("matched-codec completion report must bind 288 cells at B=500")
    if len(matched.get("artifacts_sha256", {})) != 288:
        raise RuntimeError("matched-codec completion report does not bind exactly 288 artifacts")
    for reported, expected_hash in matched["artifacts_sha256"].items():
        validate_bound_file(artifact_root, reported, expected_hash)

    evaluation = documents["coco_uniform_p0_v1_evaluation_complete.json"]
    if evaluation.get("dataset") != "coco" or evaluation.get("attempt") != "coco_uniform_p0_v1":
        raise RuntimeError("COCO P0 evaluation report has the wrong dataset or attempt")
    if evaluation.get("runs") != 117 or len(evaluation.get("metric_sha256", {})) != 117:
        raise RuntimeError("COCO P0 evaluation report does not bind exactly 117 metrics")
    plan_path = artifact_root / "manifests" / "plans" / "coco_uniform_p0_frozen_v1.json"
    plan = read_json(plan_path)
    validate_canonical_field(plan, "plan_sha256", "COCO frozen execution plan")
    if plan.get("plan_sha256") != evaluation.get("plan_sha256") or len(plan.get("runs", [])) != 117:
        raise RuntimeError("COCO evaluation report does not bind the 117-run frozen plan")
    planned_conditions = {row["condition_id"] for row in plan["runs"]}
    if planned_conditions != set(evaluation["metric_sha256"]):
        raise RuntimeError("COCO evaluated conditions differ from the frozen execution plan")
    approval_path = artifact_root / "manifests" / "approvals" / "coco_uniform_p0_v1.json"
    if sha256_file(approval_path) != evaluation.get("approval_receipt_sha256"):
        raise RuntimeError("COCO evaluation approval-receipt binding mismatch")
    return documents


def metric_row(document: dict[str, Any], dataset: str) -> dict[str, Any]:
    row = {
        "dataset": dataset,
        "condition_id": document["condition_id"],
        "model": document["model"],
        "precision": document["precision"],
        "corruption": document["corruption"],
        "severity": int(document["severity"]),
        "n_images": int(document["n_images"]),
    }
    row.update(document["stats"])
    if dataset == "tt100k":
        for label, values in document.get("height_strata_stats", {}).items():
            row[f"AP_height_{label}"] = values["AP"]
    return row


def validate_primary_grids(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    codec_controls: pd.DataFrame,
) -> None:
    """Require the primary inputs to contain every intended semantic cell once."""
    metric_conditions = {("clean", 0)} | {
        (corruption, severity)
        for corruption in CORRUPTION_ORDER
        for severity in (1, 3, 5)
    }
    grids = (
        (
            "metric",
            metrics,
            ["dataset", "model", "precision", "corruption", "severity"],
            {
                (dataset, model, precision, corruption, severity)
                for dataset, model, precision in product(DATASETS, MODEL_ORDER, PRECISION_ORDER)
                for corruption, severity in metric_conditions
            },
        ),
        (
            "matched-codec",
            bootstrap,
            ["dataset", "model", "precision", "corruption", "severity"],
            set(product(DATASETS, MODEL_ORDER, QUANT_ORDER, CORRUPTION_ORDER, (1, 3, 5))),
        ),
        (
            "codec-control",
            codec_controls,
            ["dataset", "model", "precision"],
            set(product(DATASETS, MODEL_ORDER, PRECISION_ORDER)),
        ),
    )
    for label, frame, columns, expected in grids:
        try:
            cells = [tuple(row) for row in frame[columns].itertuples(index=False, name=None)]
        except KeyError as error:
            raise RuntimeError(f"{label} semantic grid is incomplete") from error
        if len(cells) != len(expected) or len(set(cells)) != len(cells) or set(cells) != expected:
            raise RuntimeError(f"{label} semantic grid is incomplete")


def validate_and_load(
    artifact_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, pd.DataFrame]:
    artifact_root = artifact_root.resolve()
    p0_outputs = artifact_root / "outputs"
    p0_reports = p0_outputs / "reports"
    legacy_root = artifact_root / LEGACY_ARTIFACT_ROOT
    legacy_outputs = legacy_root / "outputs"
    legacy_reports = legacy_outputs / "reports"
    p0 = validate_p0_completion(artifact_root)
    final = validate_final_report(legacy_reports)
    final_by_dataset = {row["dataset"]: row for row in final["datasets"]}
    metric_rows: list[dict[str, Any]] = []
    boot_rows: list[dict[str, Any]] = []
    codec_rows: list[dict[str, Any]] = []
    integrity_rows: list[dict[str, Any]] = []

    for dataset in DATASETS:
        attempt = ATTEMPTS[dataset]
        metric_outputs = p0_outputs if dataset == "coco" else legacy_outputs
        metric_reports = p0_reports if dataset == "coco" else legacy_reports
        metric_dir = metric_outputs / "metrics" / attempt
        metric_paths = sorted(metric_dir.glob("*.json"))
        if len(metric_paths) != 117:
            raise RuntimeError(f"{dataset}: expected 117 metrics, found {len(metric_paths)}")

        eval_path = metric_reports / EVAL_REPORTS[dataset]
        eval_doc = read_json(eval_path)
        metric_hashes = expected_metric_hashes(dataset, eval_doc)
        if len(metric_hashes) != 117:
            raise RuntimeError(f"{dataset}: evaluation report does not bind 117 metrics")
        for path in metric_paths:
            doc = read_json(path)
            condition = doc["condition_id"]
            if metric_hashes.get(condition) != sha256_file(path):
                raise RuntimeError(f"{dataset}: metric hash mismatch for {path.name}")
            metric_rows.append(metric_row(doc, dataset))

        boot_report = p0["matched_codec_p0_v1_complete.json"]
        boot_dir = p0_outputs / "bootstrap" / "matched_codec_p0_v1"
        boot_paths = sorted(boot_dir.glob(f"{dataset}__*.json"))
        if len(boot_paths) != 72:
            raise RuntimeError(f"{dataset}: expected 72 matched-codec bootstraps, found {len(boot_paths)}")
        bound = {Path(key).name: value for key, value in boot_report["artifacts_sha256"].items()}
        for path in boot_paths:
            if bound.get(path.name) != sha256_file(path):
                raise RuntimeError(f"{dataset}: bootstrap hash mismatch for {path.name}")
            doc = read_json(path)
            if doc.get("n_boot") != 500:
                raise RuntimeError(f"{dataset}: bootstrap cell is not B=500: {path.name}")
            stem = path.stem
            file_dataset, model, quant, corr_sev = stem.split("__", 3)
            if file_dataset != dataset:
                raise RuntimeError(f"bootstrap filename dataset mismatch: {path.name}")
            corruption, severity_text = corr_sev.rsplit("-s", 1)
            point = doc["point"]
            is_tt = dataset == "tt100k"
            small_key = "small_like" if is_tt else "small"
            large_key = "large_like" if is_tt else "large"
            psi_key = "psi_small_like_minus_large_like" if is_tt else "psi_small_minus_large"
            ci_key = "ci95_psi_small_like_minus_large_like" if is_tt else "ci95_psi_small_minus_large"
            ci = doc[ci_key]
            if is_tt:
                q_clean_all = math.nan
                q_corrupt_all = math.nan
                e_all = math.nan
                e_all_ci = (math.nan, math.nan, math.nan)
                e_small = point[small_key]["excess"]
                e_large = point[large_key]["excess"]
            else:
                q_clean_all = point["q_clean"]["all"]
                q_corrupt_all = point["q_corrupt"]["all"]
                e_all = point["excess"]["all"]
                e_all_ci = doc["ci95_excess"]["all"]
                e_small = point["excess"][small_key]
                e_large = point["excess"][large_key]
            boot_rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "precision": quant,
                    "corruption": corruption,
                    "severity": int(severity_text),
                    "n_images": int(doc["n_images"]),
                    "n_boot": int(doc["n_boot"]),
                    "q_clean_all": q_clean_all,
                    "q_corrupt_all": q_corrupt_all,
                    "e_all": e_all,
                    "e_all_ci_low": e_all_ci[0],
                    "e_all_ci_median": e_all_ci[1],
                    "e_all_ci_high": e_all_ci[2],
                    "e_small": e_small,
                    "e_large": e_large,
                    "psi": point[psi_key],
                    "psi_ci_low": ci[0],
                    "psi_ci_median": ci[1],
                    "psi_ci_high": ci[2],
                    "size_definition": "height" if is_tt else "area",
                }
            )

        if dataset != "coco":
            final_row = final_by_dataset[dataset]
            if final_row.get("runs") != 117 or final_row.get("bootstrap_cells") != 72:
                raise RuntimeError(f"{dataset}: final report count mismatch")
            if final_row.get("evaluation_report_sha256") != sha256_file(eval_path):
                raise RuntimeError(f"{dataset}: final report evaluation binding mismatch")
        integrity_rows.append(
            {
                "Dataset": DISPLAY[dataset],
                "Metrics": len(metric_paths),
                "Bootstrap cells": len(boot_paths),
                "Replicates/cell": 500,
                "Metric hashes": "pass",
                "Bootstrap hashes": "pass",
                "Size endpoint": "height" if dataset == "tt100k" else "area",
            }
        )

    codec_report = p0["codec_control_p0_v1_complete.json"]
    for item in codec_report["conditions"]:
        path = resolve_local_report_path(artifact_root, item["metric"])
        document = read_json(path)
        if document.get("corruption") != "codec_control" or document.get("severity") != 0:
            raise RuntimeError(f"malformed codec-control metric: {path.name}")
        codec_rows.append(metric_row(document, document["dataset"]))

    metrics = pd.DataFrame(metric_rows)
    bootstrap = pd.DataFrame(boot_rows)
    codec_controls = pd.DataFrame(codec_rows)
    integrity = pd.DataFrame(integrity_rows)
    if len(codec_controls) != 36 or codec_controls.groupby("dataset").size().to_dict() != {
        dataset: 9 for dataset in DATASETS
    }:
        raise RuntimeError("codec-control grid is incomplete or unbalanced")

    # TT100K's matched bootstrap stores native height endpoints only.  Its
    # overall point interaction is reconstructed from the same JPEG-95 clean
    # controls and the matching corrupted metric cells.
    metric_index = metrics.set_index(["dataset", "model", "precision", "corruption", "severity"])
    codec_index = codec_controls.set_index(["dataset", "model", "precision"])
    for index in bootstrap.index[bootstrap.dataset == "tt100k"]:
        row = bootstrap.loc[index]
        clean_fp32 = codec_index.loc[("tt100k", row.model, "fp32"), "AP"]
        clean_quant = codec_index.loc[("tt100k", row.model, row.precision), "AP"]
        corrupt_fp32 = metric_index.loc[("tt100k", row.model, "fp32", row.corruption, row.severity), "AP"]
        corrupt_quant = metric_index.loc[("tt100k", row.model, row.precision, row.corruption, row.severity), "AP"]
        q_clean = clean_fp32 - clean_quant
        q_corrupt = corrupt_fp32 - corrupt_quant
        bootstrap.loc[index, ["q_clean_all", "q_corrupt_all", "e_all"]] = [
            q_clean,
            q_corrupt,
            q_corrupt - q_clean,
        ]
    validate_primary_grids(metrics, bootstrap, codec_controls)
    tt = bootstrap[bootstrap.dataset == "tt100k"]
    if np.allclose(tt.psi.to_numpy(), 0.0) or tt.psi.nunique() < 10:
        raise RuntimeError("TT100K height endpoint is degenerate; refusing stale analysis")
    return metrics, bootstrap, integrity, codec_controls


def generate_codec_sensitivity(
    metrics: pd.DataFrame, codec_controls: pd.DataFrame, out: Path
) -> pd.DataFrame:
    """Compare original-clean and deterministic JPEG-95-clean AP."""
    original = metrics[(metrics.corruption == "clean") & (metrics.severity == 0)][
        ["dataset", "model", "precision", "AP"]
    ].rename(columns={"AP": "original_clean_ap"})
    codec = codec_controls[["dataset", "model", "precision", "AP"]].rename(
        columns={"AP": "codec_clean_ap"}
    )
    sensitivity = original.merge(
        codec, on=["dataset", "model", "precision"], validate="one_to_one"
    )
    if len(sensitivity) != 36:
        raise RuntimeError("codec sensitivity requires exactly 36 matched clean conditions")
    sensitivity["codec_minus_original"] = 100 * (
        sensitivity.codec_clean_ap - sensitivity.original_clean_ap
    )
    sensitivity = sensitivity.sort_values(
        ["dataset", "model", "precision"],
        key=lambda column: column.map(
            {
                **{value: index for index, value in enumerate(DATASETS)},
                **{value: index for index, value in enumerate(MODEL_ORDER)},
                **{value: index for index, value in enumerate(PRECISION_ORDER)},
            }
        ),
    ).reset_index(drop=True)
    sensitivity.to_csv(out / "codec_sensitivity.csv", index=False)
    shown = sensitivity.copy()
    shown["Dataset"] = shown.dataset.map(DISPLAY)
    shown["Model"] = shown.model.map(DISPLAY)
    shown["Precision"] = shown.precision.map(DISPLAY)
    shown["Original clean AP (points)"] = shown.original_clean_ap.map(fmt_ap)
    shown["JPEG-95 clean AP (points)"] = shown.codec_clean_ap.map(fmt_ap)
    shown["Codec minus original (AP points)"] = shown.codec_minus_original.map(
        lambda value: f"{value:+.3f}"
    )
    write_booktabs(
        out / "codec_sensitivity.tex",
        shown,
        [
            "Dataset",
            "Model",
            "Precision",
            "Original clean AP (points)",
            "JPEG-95 clean AP (points)",
            "Codec minus original (AP points)",
        ],
        "lllrrr",
    )
    return sensitivity


def replace_original_clean_with_codec(
    metrics: pd.DataFrame, codec_controls: pd.DataFrame
) -> pd.DataFrame:
    """Return the primary metric grid with JPEG-95 controls as clean cells."""
    clean = codec_controls.copy()
    clean["corruption"] = "clean"
    clean["severity"] = 0
    primary = pd.concat([metrics[metrics.corruption != "clean"], clean], ignore_index=True)
    if len(primary) != len(metrics):
        raise RuntimeError("primary metric grid changed size while replacing clean controls")
    return primary


def generate_tables(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    integrity: pd.DataFrame,
    codec_controls: pd.DataFrame,
    out: Path,
) -> dict[str, Any]:
    out.mkdir(parents=True, exist_ok=True)
    metrics.to_csv(out / "all_metrics.csv", index=False)
    bootstrap.to_csv(out / "interaction_cells.csv", index=False)
    integrity.to_csv(out / "integrity.csv", index=False)
    sensitivity = generate_codec_sensitivity(metrics, codec_controls, out)

    clean = codec_controls.copy()
    clean["Dataset"] = clean.dataset.map(DISPLAY)
    clean["Model"] = clean.model.map(DISPLAY)
    clean["Precision"] = clean.precision.map(DISPLAY)
    clean["AP (%)"] = clean.AP.map(fmt_ap)
    clean["AP small (%)"] = clean.AP_small.map(fmt_ap)
    clean["AP large (%)"] = clean.AP_large.map(fmt_ap)
    clean = clean.sort_values(
        ["dataset", "model", "precision"],
        key=lambda col: col.map(
            {**{x: i for i, x in enumerate(DATASETS)}, **{x: i for i, x in enumerate(MODEL_ORDER)}, **{x: i for i, x in enumerate(PRECISION_ORDER)}}
        ),
    )
    clean[["dataset", "model", "precision", "AP", "AP_small", "AP_large"]].to_csv(
        out / "clean_ap.csv", index=False
    )
    write_booktabs(
        out / "clean_ap.tex",
        clean,
        ["Dataset", "Model", "Precision", "AP (%)", "AP small (%)", "AP large (%)"],
        "lllrrr",
    )

    summary = (
        bootstrap.groupby(["dataset", "precision"], sort=False)
        .agg(
            cells=("psi", "size"),
            mean_e_all=("e_all", "mean"),
            sd_e_all=("e_all", "std"),
            mean_e_small=("e_small", "mean"),
            mean_e_large=("e_large", "mean"),
            mean_psi=("psi", "mean"),
            sd_psi=("psi", "std"),
            positive_psi=("psi", lambda x: int((x > 0).sum())),
        )
        .reset_index()
    )
    ci_summary = (
        bootstrap.groupby(["dataset", "precision"], sort=False)
        .apply(
            lambda group: pd.Series(
                {
                    "ci_positive": int((group.psi_ci_low > 0).sum()),
                    "ci_negative": int((group.psi_ci_high < 0).sum()),
                    "ci_crossing": int(((group.psi_ci_low <= 0) & (group.psi_ci_high >= 0)).sum()),
                    "psi_min": group.psi.min(),
                    "psi_max": group.psi.max(),
                }
            ),
            include_groups=False,
        )
        .reset_index()
    )
    summary = summary.merge(ci_summary, on=["dataset", "precision"], how="left")
    summary.to_csv(out / "interaction_summary.csv", index=False)
    shown = summary.copy()
    shown["Dataset"] = shown.dataset.map(DISPLAY)
    shown["Format"] = shown.precision.map(DISPLAY)
    shown["Mean E all (AP pt)"] = shown.mean_e_all.map(fmt_effect)
    shown["E small area / small-like height"] = shown.mean_e_small.map(fmt_effect)
    shown["E large area / large-like height"] = shown.mean_e_large.map(fmt_effect)
    shown["Mean Psi (AP pt)"] = shown.mean_psi.map(fmt_effect)
    shown["Positive Psi"] = shown.apply(lambda r: f"{int(r.positive_psi)}/{int(r.cells)}", axis=1)
    shown["CI sign counts"] = shown.apply(
        lambda r: f"+{int(r.ci_positive)} / -{int(r.ci_negative)} / {int(r.ci_crossing)}", axis=1
    )
    write_booktabs(
        out / "interaction_summary.tex",
        shown,
        ["Dataset", "Format", "Mean E all (AP pt)", "E small area / small-like height", "E large area / large-like height", "Mean Psi (AP pt)", "Positive Psi", "CI sign counts"],
        "llrp{2.35cm}p{2.35cm}rrp{2.0cm}",
    )

    clean_gap = []
    for (dataset, model), group in clean.groupby(["dataset", "model"]):
        by_precision = group.set_index("precision")
        for quant in QUANT_ORDER:
            clean_gap.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "precision": quant,
                    "q_clean_all": by_precision.loc["fp32", "AP"] - by_precision.loc[quant, "AP"],
                    "q_clean_small": by_precision.loc["fp32", "AP_small"] - by_precision.loc[quant, "AP_small"],
                    "q_clean_large": by_precision.loc["fp32", "AP_large"] - by_precision.loc[quant, "AP_large"],
                }
            )
    clean_gaps = pd.DataFrame(clean_gap)
    clean_gaps.to_csv(out / "clean_quantization_gaps.csv", index=False)

    dataset_meta = pd.DataFrame(
        DATASET_META,
        columns=["Dataset", "Split", "Images", "Objects", "Declared/observed classes", "Input px", "Size strata"],
    )
    dataset_meta.to_csv(out / "datasets.csv", index=False)
    write_booktabs(out / "datasets.tex", dataset_meta, list(dataset_meta.columns), "llrrrrp{5.2cm}")
    write_booktabs(out / "integrity.tex", integrity, list(integrity.columns), "lrrrrll")

    return {
        "clean": clean,
        "summary": summary,
        "clean_gaps": clean_gaps,
        "codec_sensitivity": sensitivity,
    }


def generate_parity_table(artifact_root: Path, out: Path) -> None:
    """Collect the 12 clean FP16 parity gates (3 COCO + 9 transfer)."""
    reports = artifact_root / "outputs" / "reports"
    legacy_reports = artifact_root / LEGACY_ARTIFACT_ROOT / "outputs" / "reports"
    rows: list[dict[str, Any]] = []
    for model in MODEL_ORDER:
        item = read_json(reports / f"coco_{model}_fp16_parity_v1.json")
        rows.append({"Dataset": "COCO", "Model": DISPLAY[model], "FP16 AP gap (points)": fmt_points(item["absolute_ap_gap"]), "Tolerance (points)": fmt_points(item["tolerance"]), "Pass": "yes" if item["pass"] else "no"})
    for dataset in ("voc", "kitti", "tt100k"):
        for model in MODEL_ORDER:
            path = legacy_reports / f"{dataset}_yolo11{model[-1]}_fp16_parity_v1.json"
            if not path.is_file():
                raise RuntimeError(f"missing parity report: {path}")
            item = read_json(path)
            rows.append({"Dataset": DISPLAY[dataset], "Model": DISPLAY[model], "FP16 AP gap (points)": fmt_points(item["absolute_ap_gap"]), "Tolerance (points)": fmt_points(item["tolerance"]), "Pass": "yes" if item["pass"] else "no"})
    frame = pd.DataFrame(rows)
    if len(frame) != 12 or (frame["Pass"] != "yes").any():
        raise RuntimeError("FP16 parity table is incomplete or contains a failed gate")
    frame.to_csv(out / "fp16_parity.csv", index=False)
    write_booktabs(out / "fp16_parity.tex", frame, list(frame.columns), "llrrl")


def compute_absolute_corruption(metrics: pd.DataFrame) -> pd.DataFrame:
    """Attach each corrupted AP result to its matched clean reference."""
    clean = metrics[(metrics.corruption == "clean") & (metrics.severity == 0)][
        ["dataset", "model", "precision", "AP", "AP_small", "AP_large"]
    ].rename(
        columns={
            "AP": "clean_AP",
            "AP_small": "clean_AP_small",
            "AP_large": "clean_AP_large",
        }
    )
    corrupted = metrics[metrics.corruption != "clean"].merge(
        clean, on=["dataset", "model", "precision"], validate="many_to_one"
    )
    corrupted["ap_drop"] = corrupted.clean_AP - corrupted.AP
    corrupted["ap_small_drop"] = corrupted.clean_AP_small - corrupted.AP_small
    corrupted["ap_large_drop"] = corrupted.clean_AP_large - corrupted.AP_large
    corrupted["ap_retention"] = np.where(
        corrupted.clean_AP > 0, corrupted.AP / corrupted.clean_AP, np.nan
    )
    return corrupted


def generate_extended_tables(
    metrics: pd.DataFrame,
    bootstrap: pd.DataFrame,
    derived: dict[str, Any],
    out: Path,
) -> dict[str, pd.DataFrame]:
    """Generate paper and appendix tables that expose the full factorial design."""
    absolute = compute_absolute_corruption(metrics)
    absolute.to_csv(out / "absolute_corruption_cells.csv", index=False)

    # Overall E uses the full matched-codec grid.  Area Psi uses COCO/VOC/KITTI;
    # TT100K height Psi remains visible only in dataset-specific summaries.
    complete_grid = bootstrap.copy()
    area_grid = bootstrap[bootstrap.dataset.isin(["coco", "voc", "kitti"])].copy()
    corruption_e = (
        complete_grid.groupby(["corruption", "precision"], sort=False)
        .agg(
            e_cells=("e_all", "size"),
            mean_e_all=("e_all", "mean"),
        )
        .reset_index()
    )
    corruption_psi = (
        area_grid.groupby(["corruption", "precision"], sort=False)
        .agg(
            psi_cells=("psi", "size"),
            mean_e_small=("e_small", "mean"),
            mean_e_large=("e_large", "mean"),
            mean_psi=("psi", "mean"),
            ci_positive=("psi_ci_low", lambda x: int((x > 0).sum())),
            ci_negative=("psi_ci_high", lambda x: int((x < 0).sum())),
        )
        .reset_index()
    )
    corruption_summary = corruption_e.merge(
        corruption_psi, on=["corruption", "precision"], validate="one_to_one"
    )
    corruption_summary["ci_crossing"] = (
        corruption_summary.psi_cells
        - corruption_summary.ci_positive
        - corruption_summary.ci_negative
    )
    corruption_summary = corruption_summary.sort_values(
        ["corruption", "precision"],
        key=lambda column: column.map(
            {
                **{value: index for index, value in enumerate(CORRUPTION_ORDER)},
                **{value: index for index, value in enumerate(QUANT_ORDER)},
            }
        ),
    )
    corruption_summary.to_csv(out / "corruption_summary.csv", index=False)
    shown = corruption_summary.copy()
    shown["Corruption"] = shown.corruption.map(DISPLAY)
    shown["Format"] = shown.precision.map(DISPLAY)
    shown["Mean E"] = shown.mean_e_all.map(fmt_effect)
    shown["Mean E small"] = shown.mean_e_small.map(fmt_effect)
    shown["Mean E large"] = shown.mean_e_large.map(fmt_effect)
    shown["Mean Psi"] = shown.mean_psi.map(fmt_effect)
    shown["CI + / - / 0"] = shown.apply(
        lambda r: f"{int(r.ci_positive)} / {int(r.ci_negative)} / {int(r.ci_crossing)}",
        axis=1,
    )
    write_booktabs(
        out / "corruption_summary.tex",
        shown,
        ["Corruption", "Format", "Mean E", "Mean E small", "Mean E large", "Mean Psi", "CI + / - / 0"],
        "llrrrrr",
    )

    clean_capacity = (
        derived["clean_gaps"]
        .groupby(["model", "precision"], sort=False)
        .q_clean_all.mean()
        .rename("mean_clean_gap")
        .reset_index()
    )
    capacity_e = (
        complete_grid.groupby(["model", "precision"], sort=False)
        .agg(
            e_cells=("e_all", "size"),
            mean_e_all=("e_all", "mean"),
        )
        .reset_index()
    )
    capacity_psi = (
        area_grid.groupby(["model", "precision"], sort=False)
        .agg(
            psi_cells=("psi", "size"),
            mean_psi=("psi", "mean"),
            psi_sd=("psi", "std"),
            ci_positive=("psi_ci_low", lambda x: int((x > 0).sum())),
            ci_negative=("psi_ci_high", lambda x: int((x < 0).sum())),
        )
        .reset_index()
    )
    capacity_summary = (
        capacity_e.merge(capacity_psi, on=["model", "precision"], validate="one_to_one")
        .merge(clean_capacity, on=["model", "precision"], validate="one_to_one")
    )
    capacity_summary["ci_crossing"] = (
        capacity_summary.psi_cells - capacity_summary.ci_positive - capacity_summary.ci_negative
    )
    capacity_summary = capacity_summary.sort_values(
        ["model", "precision"],
        key=lambda column: column.map(
            {
                **{value: index for index, value in enumerate(MODEL_ORDER)},
                **{value: index for index, value in enumerate(QUANT_ORDER)},
            }
        ),
    )
    capacity_summary.to_csv(out / "capacity_summary.csv", index=False)
    cap_show = capacity_summary.copy()
    cap_show["Model"] = cap_show.model.map(DISPLAY)
    cap_show["Format"] = cap_show.precision.map(DISPLAY)
    cap_show["Clean gap"] = cap_show.mean_clean_gap.map(fmt_effect)
    cap_show["Mean E"] = cap_show.mean_e_all.map(fmt_effect)
    cap_show["Mean Psi"] = cap_show.mean_psi.map(fmt_effect)
    cap_show["SD Psi"] = cap_show.psi_sd.map(lambda value: f"{100 * value:.2f}")
    cap_show["CI + / - / 0"] = cap_show.apply(
        lambda r: f"{int(r.ci_positive)} / {int(r.ci_negative)} / {int(r.ci_crossing)}",
        axis=1,
    )
    write_booktabs(
        out / "capacity_summary.tex",
        cap_show,
        ["Model", "Format", "Clean gap", "Mean E", "Mean Psi", "SD Psi", "CI + / - / 0"],
        "llrrrrr",
    )

    severe = (
        absolute[absolute.severity == 5]
        .groupby(["dataset", "corruption", "precision"], sort=False)
        .ap_drop.mean()
        .mul(100)
        .unstack("precision")
        .reset_index()
    )
    severe = severe[["dataset", "corruption", "fp32", "int8-entropy", "fp8"]]
    severe.to_csv(out / "severity5_absolute_ap_drop.csv", index=False)
    severe_show = severe.copy()
    severe_show["Dataset"] = severe_show.dataset.map(DISPLAY)
    severe_show["Corruption"] = severe_show.corruption.map(DISPLAY)
    for source, target in (("fp32", "FP32"), ("int8-entropy", "INT8"), ("fp8", "FP8")):
        severe_show[target] = severe_show[source].map(lambda value: f"{value:.2f}")
    write_booktabs(
        out / "severity5_absolute_ap_drop.tex",
        severe_show,
        ["Dataset", "Corruption", "FP32", "INT8", "FP8"],
        "llrrr",
    )

    endpoint_extremes: list[pd.DataFrame] = []
    for size_definition, endpoint_data in bootstrap.groupby("size_definition", sort=False):
        significant = endpoint_data[
            (endpoint_data.psi_ci_low > 0) | (endpoint_data.psi_ci_high < 0)
        ].copy()
        negative = significant[significant.psi < 0].nsmallest(6, "psi")
        positive = significant[significant.psi > 0].nlargest(6, "psi")
        selected = pd.concat([negative, positive], ignore_index=True)
        selected["endpoint"] = size_definition
        endpoint_extremes.append(selected)
    extremes = pd.concat(endpoint_extremes, ignore_index=True).sort_values(
        ["endpoint", "psi"]
    )
    if extremes.empty:
        raise RuntimeError("no interaction interval excludes zero")
    extremes["label"] = extremes.apply(
        lambda r: (
            f"{r.endpoint} | {DISPLAY[r.dataset]} | {DISPLAY[r.model]} | {DISPLAY[r.precision]} | "
            f"{DISPLAY[r.corruption]} s{int(r.severity)}"
        ),
        axis=1,
    )
    extremes.to_csv(out / "extreme_interactions.csv", index=False)
    ext_show = extremes.copy()
    ext_show["Condition"] = ext_show.label
    ext_show["Psi"] = ext_show.psi.map(fmt_effect)
    ext_show["95% interval"] = ext_show.apply(
        lambda r: f"[{100*r.psi_ci_low:+.2f}, {100*r.psi_ci_high:+.2f}]", axis=1
    )
    write_booktabs(
        out / "extreme_interactions.tex",
        ext_show,
        ["Condition", "Psi", "95% interval"],
        "p{10.4cm}rr",
    )

    return {
        "absolute": absolute,
        "corruption_summary": corruption_summary,
        "capacity_summary": capacity_summary,
        "extremes": extremes,
    }


def setup_plotting() -> None:
    sns.set_theme(style="whitegrid", context="paper")
    plt.rcParams.update(
        {
            "font.size": 8.5,
            "axes.titlesize": 9.5,
            "axes.labelsize": 9,
            "legend.fontsize": 8,
            "pdf.fonttype": 42,
            "ps.fonttype": 42,
            "savefig.bbox": "tight",
        }
    )


def save_figure(fig: plt.Figure, base: Path) -> None:
    fig.savefig(base.with_suffix(".pdf"))
    fig.savefig(base.with_suffix(".png"), dpi=300)
    plt.close(fig)


def generate_framework_figure(out: Path) -> None:
    """Draw the final-width paired-excess-gap framework as a vector figure."""
    fig, ax = plt.subplots(figsize=(6.89, 4.20))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")

    stages = [
        (
            0.04,
            "1",
            "Materialize inputs",
            "JPEG-95 clean control\ncorrupt bytes frozen once",
            "#FFF3DF",
            "#A66B24",
        ),
        (
            0.365,
            "2",
            "Evaluate matched engines",
            "FP32 typed · INT8 · FP8\nsame IDs, geometry, decoder",
            "#E8F1FA",
            "#35648C",
        ),
        (
            0.69,
            "3",
            "Pair the four cells",
            "$Q_{95}$ → $E$ → $\\Psi$\ncommon image-bootstrap draws",
            "#F0EEF8",
            "#62558F",
        ),
    ]
    width, height, y = 0.27, 0.245, 0.55
    for x, number, title, body, fill, edge in stages:
        box = FancyBboxPatch(
            (x, y),
            width,
            height,
            boxstyle="round,pad=0.012,rounding_size=0.025",
            linewidth=1.25,
            edgecolor=edge,
            facecolor=fill,
        )
        ax.add_patch(box)
        ax.add_patch(plt.Circle((x + 0.035, y + height - 0.035), 0.018, color=edge))
        ax.text(x + 0.035, y + height - 0.035, number, ha="center", va="center", color="white", fontsize=8.3, fontweight="bold")
        ax.text(x + width / 2, y + 0.157, title, ha="center", va="center", fontsize=9.5, fontweight="bold", color="#1F2937")
        ax.text(x + width / 2, y + 0.077, body, ha="center", va="center", fontsize=8.5, linespacing=1.20, color="#334155")
    for index in range(len(stages) - 1):
        x_start = stages[index][0] + width + 0.008
        x_end = stages[index + 1][0] - 0.008
        ax.add_patch(
            FancyArrowPatch(
                (x_start, y + height / 2), (x_end, y + height / 2),
                arrowstyle="-|>",
                mutation_scale=11,
                linewidth=1.10,
                color="#64748B",
            )
        )

    ax.text(
        0.5,
        0.93,
        "Paired excess-gap evaluation framework",
        ha="center",
        va="center",
        fontsize=12.8,
        fontweight="bold",
        color="#172033",
    )
    direct = FancyBboxPatch(
        (0.095, 0.275), 0.81, 0.14,
        boxstyle="round,pad=0.012,rounding_size=0.02",
        linewidth=1.15, edgecolor="#34756C", facecolor="#EBF5F3",
    )
    ax.add_patch(direct)
    ax.text(
        0.5, 0.345,
        r"Direct format contrast: $\Delta E=E_{\mathrm{INT8}}-E_{\mathrm{FP8}}$; FP32 cancels  |  "
        r"$\Delta\Psi=\Psi_{\mathrm{INT8}}-\Psi_{\mathrm{FP8}}$",
        ha="center",
        va="center",
        fontsize=8.45,
        color="#334155",
    )
    provenance = FancyBboxPatch(
        (0.035, 0.075), 0.93, 0.105,
        boxstyle="round,pad=0.010,rounding_size=0.02",
        linewidth=1.0, edgecolor="#64748B", facecolor="#F8FAFC",
    )
    ax.add_patch(provenance)
    ax.text(
        0.5, 0.127,
        "Shared bytes and image resamples; SHA-256: checkpoint → engine → input → metric → paired-bootstrap artifact",
        ha="center", va="center", fontsize=8.3, color="#334155",
    )
    fig.subplots_adjust(left=0, right=1, bottom=0, top=1)
    fig.savefig((out / "fig_framework").with_suffix(".pdf"), bbox_inches=None)
    fig.savefig((out / "fig_framework").with_suffix(".png"), dpi=300, bbox_inches=None)
    plt.close(fig)


def generate_method_graphical_abstract(out: Path) -> None:
    """Render an Elsevier-sized graphical abstract without result-bearing values."""
    out.mkdir(parents=True, exist_ok=True)
    fig = plt.figure(figsize=(10.0, 4.0), dpi=300)
    ax = fig.add_axes((0, 0, 1, 1))
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.axis("off")
    fig.patch.set_facecolor("white")
    fig.text(0.5, 0.93, "Paired, floor-guarded evaluation under image corruptions", ha="center", va="center", fontsize=17, fontweight="bold", color="#172033")
    fig.text(0.5, 0.875, "Subtract the matched-clean INT8--FP8 gap before interpreting the corrupted-input gap", ha="center", va="center", fontsize=9.5, color="#475569")

    cards = [
        (0.035, "1", "Match four treatment cells", "same ordered images\nmatched JPEG-95 bytes", "#EAF3FB", "#4F86B4"),
        (0.362, "2", "Estimate the interaction", "subtract the matched-clean\ntreatment discrepancy", "#FFF4E5", "#B7791F"),
        (0.690, "3", "Guard the conclusion", "check task accuracy before\ninterpreting gap contraction", "#EAF7F0", "#2F855A"),
    ]
    for x, number, title, subtitle, fill, edge in cards:
        ax.add_patch(FancyBboxPatch((x, 0.14), 0.275, 0.64, boxstyle="round,pad=0.012,rounding_size=0.022", facecolor=fill, edgecolor=edge, linewidth=1.2))
        ax.text(x + 0.035, 0.71, number, ha="center", va="center", fontsize=10, fontweight="bold", color="white", bbox={"boxstyle": "circle,pad=0.27", "fc": edge, "ec": edge})
        ax.text(x + 0.075, 0.71, title, ha="left", va="center", fontsize=10.5, fontweight="bold", color="#172033")
        ax.text(x + 0.1375, 0.635, subtitle, ha="center", va="center", fontsize=8, color="#475569")

    for index, (precision, condition, color) in enumerate((
        ("INT8", "J95 clean", "#4F86B4"), ("FP8", "J95 clean", "#D97706"),
        ("INT8", "corrupted", "#4F86B4"), ("FP8", "corrupted", "#D97706"),
    )):
        col, row = index % 2, index // 2
        x, y = 0.073 + col * 0.105, 0.455 - row * 0.145
        ax.add_patch(FancyBboxPatch((x, y), 0.093, 0.12, boxstyle="round,pad=0.006", facecolor="white", edgecolor=color, linewidth=1.0))
        ax.text(x + 0.0465, y + 0.073, precision, ha="center", va="center", fontsize=7.3, fontweight="bold", color=color)
        ax.text(x + 0.0465, y + 0.034, condition, ha="center", va="center", fontsize=6.7, color="#475569")
    ax.text(0.172, 0.245, "one checkpoint · one decoder · identical bytes", ha="center", va="center", fontsize=7.1, color="#475569")

    ax.text(0.500, 0.550, r"$\Delta E=(AP_{FP8}-AP_{INT8})_{corrupt}$", ha="center", va="center", fontsize=11.5, color="#172033")
    ax.text(0.500, 0.455, r"$\quad -(AP_{FP8}-AP_{INT8})_{J95}$", ha="center", va="center", fontsize=11.5, fontweight="bold", color="#A65B08")
    ax.text(0.500, 0.350, "common-image bootstrap preserves all four dependencies", ha="center", va="center", fontsize=7.7, color="#475569")
    ax.text(0.500, 0.270, "FP32 supports treatment diagnostics but cancels\nfrom the direct paired interaction", ha="center", va="center", fontsize=7.2, color="#475569")

    for index, (label, detail) in enumerate((("1", "matched-clean\nfidelity"), ("2", "absolute corrupted\naccuracy"), ("3", "interaction, then\nruntime"))):
        y = 0.57 - index * 0.125
        ax.text(0.730, y, label, ha="left", va="center", fontsize=9.5, fontweight="bold", color="#24734B")
        ax.text(0.820, y, detail, ha="left", va="center", fontsize=7.2, color="#475569", linespacing=1.15)
    ax.text(0.828, 0.217, "Gap contraction near a shared accuracy floor\nis not retained robustness.", ha="center", va="center", fontsize=7.0, color="#475569")
    for x_start, x_end in ((0.317, 0.352), (0.644, 0.680)):
        ax.add_patch(FancyArrowPatch((x_start, 0.46), (x_end, 0.46), arrowstyle="-|>", mutation_scale=13, linewidth=1.2, color="#64748B"))
    original_bbox = plt.rcParams["savefig.bbox"]
    original_pad = plt.rcParams["savefig.pad_inches"]
    try:
        plt.rcParams["savefig.bbox"] = None
        plt.rcParams["savefig.pad_inches"] = 0.0
        fig.savefig(out / "graphical_abstract.png", dpi=300, bbox_inches=None, pad_inches=0.0)
        fig.savefig(out / "graphical_abstract.pdf", bbox_inches=None, pad_inches=0.0)
    finally:
        plt.rcParams["savefig.bbox"] = original_bbox
        plt.rcParams["savefig.pad_inches"] = original_pad
    plt.close(fig)


def generate_absolute_robustness_figure(metrics: pd.DataFrame, out: Path) -> None:
    """Plot absolute AP trajectories so interaction effects are not mistaken for robustness."""
    clean = metrics[(metrics.corruption == "clean") & (metrics.severity == 0)].copy()
    expanded_clean = pd.concat(
        [clean.assign(corruption=corruption) for corruption in CORRUPTION_ORDER],
        ignore_index=True,
    )
    curves = pd.concat([expanded_clean, metrics[metrics.corruption != "clean"]], ignore_index=True)
    curves = (
        curves.groupby(["dataset", "corruption", "precision", "severity"], sort=False)
        .AP.mean()
        .mul(100)
        .reset_index()
    )
    colors = {"fp32": "#2F343B", "int8-entropy": "#C44E52", "fp8": "#4C72B0"}
    markers = {"fp32": "o", "int8-entropy": "s", "fp8": "^"}
    fig, axes = plt.subplots(4, 4, figsize=(11.6, 8.3), sharex=True, sharey="row")
    for i, dataset in enumerate(DATASETS):
        for j, corruption in enumerate(CORRUPTION_ORDER):
            ax = axes[i, j]
            subset = curves[(curves.dataset == dataset) & (curves.corruption == corruption)]
            for precision in PRECISION_ORDER:
                line = subset[subset.precision == precision].sort_values("severity")
                ax.plot(
                    line.severity,
                    line.AP,
                    marker=markers[precision],
                    markersize=3.4,
                    linewidth=1.2,
                    color=colors[precision],
                    label=DISPLAY[precision],
                )
            ax.set_xticks([0, 1, 3, 5], ["C", "1", "3", "5"])
            ax.grid(True, alpha=0.22)
            if i == 0:
                ax.set_title(DISPLAY[corruption])
            if j == 0:
                ax.set_ylabel(f"{DISPLAY[dataset]}\nAP (%)")
            if i == len(DATASETS) - 1:
                ax.set_xlabel("Condition / severity")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.005))
    fig.suptitle("Absolute AP trajectories averaged over YOLO11n/m/x", y=1.035, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out / "fig_absolute_robustness")


def generate_interaction_distribution_figure(bootstrap: pd.DataFrame, out: Path) -> None:
    """Show cell-level distributions rather than relying on grand means."""
    e_data = bootstrap.copy()
    area_psi = bootstrap[bootstrap.dataset.isin(["coco", "voc", "kitti"])].copy()
    height_psi = bootstrap[bootstrap.dataset == "tt100k"].copy()
    for plot_data in (e_data, area_psi, height_psi):
        plot_data["Corruption"] = plot_data.corruption.map(DISPLAY)
        plot_data["Format"] = plot_data.precision.map(DISPLAY)
        plot_data["E (AP points)"] = 100 * plot_data.e_all
        plot_data["Psi (AP points)"] = 100 * plot_data.psi
    palette = {"INT8": "#C44E52", "FP8": "#4C72B0"}
    fig, axes = plt.subplots(1, 3, figsize=(13.2, 3.55), sharex=True)
    for ax, data, value, title in (
        (axes[0], e_data, "E (AP points)", "Excess overall gap, E — all datasets"),
        (axes[1], area_psi, "Psi (AP points)", "Area interaction, Psi — COCO/VOC/KITTI"),
        (axes[2], height_psi, "Psi (AP points)", "Height interaction, Psi — TT100K"),
    ):
        sns.boxplot(
            data=data,
            x="Corruption",
            y=value,
            hue="Format",
            order=[DISPLAY[x] for x in CORRUPTION_ORDER],
            hue_order=["INT8", "FP8"],
            palette=palette,
            showfliers=False,
            width=0.66,
            linewidth=0.9,
            ax=ax,
        )
        sns.stripplot(
            data=data,
            x="Corruption",
            y=value,
            hue="Format",
            order=[DISPLAY[x] for x in CORRUPTION_ORDER],
            hue_order=["INT8", "FP8"],
            dodge=True,
            palette=palette,
            alpha=0.34,
            size=2.2,
            jitter=0.18,
            linewidth=0,
            ax=ax,
        )
        ax.axhline(0, color="#111827", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("")
        ax.tick_params(axis="x", rotation=20)
        if ax.legend_:
            ax.legend_.remove()
    handles, labels = axes[2].get_legend_handles_labels()
    fig.legend(handles[:2], labels[:2], loc="upper center", ncol=2, frameon=False, bbox_to_anchor=(0.5, 1.03))
    fig.tight_layout()
    save_figure(fig, out / "fig_interaction_distributions")


def generate_interaction_atlas_figure(bootstrap: pd.DataFrame, out: Path) -> None:
    """Render every one of the 288 Psi point estimates in one auditable atlas."""
    atlas = bootstrap.copy()
    atlas["psi_points"] = 100 * atlas.psi
    atlas["condition"] = atlas.apply(
        lambda r: f"{DISPLAY[r.corruption].split()[0][:3]}{int(r.severity)}", axis=1
    )
    ordered_conditions = [
        f"{DISPLAY[corruption].split()[0][:3]}{severity}"
        for corruption in CORRUPTION_ORDER
        for severity in (1, 3, 5)
    ]
    limit = max(abs(atlas.psi_points.min()), abs(atlas.psi_points.max()))
    fig, axes = plt.subplots(2, 4, figsize=(12.1, 5.5), sharex=True, sharey=True)
    for i, precision in enumerate(QUANT_ORDER):
        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]
            subset = atlas[(atlas.precision == precision) & (atlas.dataset == dataset)]
            matrix = (
                subset.pivot(index="model", columns="condition", values="psi_points")
                .reindex(index=MODEL_ORDER, columns=ordered_conditions)
            )
            matrix.index = [DISPLAY[x] for x in matrix.index]
            sns.heatmap(
                matrix,
                cmap="vlag",
                center=0,
                vmin=-limit,
                vmax=limit,
                annot=True,
                fmt="+.1f",
                annot_kws={"fontsize": 5.2},
                linewidths=0.25,
                linecolor="white",
                cbar=(j == len(DATASETS) - 1),
                cbar_kws={"label": "Psi (AP points)", "shrink": 0.75},
                ax=ax,
            )
            endpoint = "height" if dataset == "tt100k" else "area"
            ax.set_title(f"{DISPLAY[dataset]} ({endpoint})")
            ax.set_xlabel(DISPLAY[precision])
            ax.set_ylabel("")
            ax.tick_params(axis="x", labelrotation=45, labelsize=6.4)
            ax.tick_params(axis="y", labelrotation=0, labelsize=7.2)
    fig.text(
        0.5,
        0.015,
        "Column abbreviations: Gau = Gaussian noise, Mot = motion blur, Fog = fog, JPE = JPEG; suffix is severity.",
        ha="center",
        fontsize=7.3,
        color="#475569",
    )
    fig.suptitle("Complete size-interaction atlas: all 288 model × format × corruption × severity cells", y=1.02, fontsize=11.5, fontweight="bold")
    fig.tight_layout(rect=(0, 0.04, 1, 0.98))
    save_figure(fig, out / "fig_interaction_atlas")


def generate_extreme_forest_figure(extremes: pd.DataFrame, out: Path) -> None:
    """Forest plots separated by area and TT100K height endpoints."""
    endpoint_order = [endpoint for endpoint in ("area", "height") if endpoint in set(extremes.endpoint)]
    max_rows = max(int((extremes.endpoint == endpoint).sum()) for endpoint in endpoint_order)
    fig, axes = plt.subplots(
        1, len(endpoint_order), figsize=(13.0, max(3.5, 0.42 * max_rows + 1.2)), squeeze=False
    )
    for ax, endpoint in zip(axes[0], endpoint_order):
        data = extremes[extremes.endpoint == endpoint].sort_values("psi").reset_index(drop=True)
        y = np.arange(len(data))
        point = 100 * data.psi.to_numpy()
        low = 100 * data.psi_ci_low.to_numpy()
        high = 100 * data.psi_ci_high.to_numpy()
        colors = np.where(point < 0, "#C44E52", "#4C72B0")
        for idx in range(len(data)):
            ax.errorbar(
                point[idx],
                y[idx],
                xerr=[[point[idx] - low[idx]], [high[idx] - point[idx]]],
                fmt="o",
                markersize=5.2,
                capsize=2.8,
                color=colors[idx],
                ecolor=colors[idx],
                linewidth=1.2,
            )
        ax.axvline(0, color="#111827", linewidth=0.85)
        ax.set_yticks(y, data.label)
        ax.set_xlabel("Psi (AP points)")
        ax.set_title(f"{endpoint.title()} endpoint")
        ax.grid(axis="x", alpha=0.25)
        ax.grid(axis="y", visible=False)
    fig.suptitle("Largest endpoint-specific interactions with intervals excluding zero")
    fig.tight_layout()
    save_figure(fig, out / "fig_extreme_forest")


def generate_tt100k_height_figure(metrics: pd.DataFrame, out: Path) -> None:
    """Expose the native TT100K height endpoint instead of hiding it in a pooled score."""
    strata = ["XS", "S", "M", "L", "XL"]
    columns = [f"AP_height_{label}" for label in strata]
    tt = metrics[metrics.dataset == "tt100k"].copy()
    clean = tt[(tt.corruption == "clean") & (tt.severity == 0)]
    severe = tt[(tt.corruption != "clean") & (tt.severity == 5)]
    colors = {"fp32": "#2F343B", "int8-entropy": "#C44E52", "fp8": "#4C72B0"}
    markers = {"fp32": "o", "int8-entropy": "s", "fp8": "^"}
    fig, axes = plt.subplots(2, 3, figsize=(10.8, 5.5), sharex=True, sharey=True)
    for j, model in enumerate(MODEL_ORDER):
        for i, (frame, row_title) in enumerate(((clean, "Clean"), (severe, "Severity 5 mean"))):
            ax = axes[i, j]
            for precision in PRECISION_ORDER:
                subset = frame[(frame.model == model) & (frame.precision == precision)]
                values = subset[columns].mean(axis=0).to_numpy(dtype=float) * 100
                ax.plot(
                    strata,
                    values,
                    marker=markers[precision],
                    color=colors[precision],
                    linewidth=1.3,
                    markersize=4.0,
                    label=DISPLAY[precision],
                )
            ax.set_title(f"{DISPLAY[model]} — {row_title}")
            ax.grid(True, alpha=0.24)
            if j == 0:
                ax.set_ylabel("AP (%)")
            if i == 1:
                ax.set_xlabel("Original bounding-box height stratum")
    handles, labels = axes[0, 0].get_legend_handles_labels()
    fig.legend(handles, labels, loc="upper center", ncol=3, frameon=False, bbox_to_anchor=(0.5, 1.02))
    fig.suptitle("TT100K native height-stratified performance", y=1.055, fontsize=11.5, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out / "fig_tt100k_height")


def generate_figures(metrics: pd.DataFrame, bootstrap: pd.DataFrame, derived: dict[str, Any], out: Path) -> None:
    out.mkdir(parents=True, exist_ok=True)
    setup_plotting()
    palette = {"int8-entropy": "#C44E52", "fp8": "#4C72B0"}

    generate_framework_figure(out)
    generate_absolute_robustness_figure(metrics, out)
    generate_interaction_distribution_figure(bootstrap, out)
    generate_interaction_atlas_figure(bootstrap, out)
    generate_extreme_forest_figure(derived["extremes"], out)
    generate_tt100k_height_figure(metrics, out)

    gaps = derived["clean_gaps"].copy()
    gaps["Dataset"] = gaps.dataset.map(DISPLAY)
    gaps["Model"] = gaps.model.map(DISPLAY)
    gaps["Format"] = gaps.precision.map(DISPLAY)
    gaps["Clean AP loss (points)"] = 100 * gaps.q_clean_all
    fig, axes = plt.subplots(1, 4, figsize=(11.2, 2.65), sharey=True)
    for ax, dataset in zip(axes, DATASETS):
        subset = gaps[gaps.dataset == dataset]
        sns.barplot(
            data=subset,
            x="Model",
            y="Clean AP loss (points)",
            hue="precision",
            hue_order=QUANT_ORDER,
            palette=palette,
            ax=ax,
        )
        ax.set_title(DISPLAY[dataset])
        ax.axhline(0, color="black", linewidth=0.7)
        ax.set_xlabel("")
        if ax is not axes[0]:
            ax.set_ylabel("")
        if ax is axes[-1]:
            handles, _ = ax.get_legend_handles_labels()
            ax.legend(handles, ["INT8", "FP8"], title="")
        elif ax.legend_:
            ax.legend_.remove()
    fig.suptitle("Clean quantization gap relative to the matched FP32 engine", y=1.02)
    save_figure(fig, out / "fig_clean_gap")

    summary = derived["summary"].copy()
    summary["Dataset"] = summary.dataset.map(DISPLAY)
    summary["Format"] = summary.precision.map(DISPLAY)
    summary["Mean E (AP points)"] = 100 * summary.mean_e_all
    summary["Mean Psi (AP points)"] = 100 * summary.mean_psi
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 2.85))
    for ax, y, title in [
        (axes[0], "Mean E (AP points)", "Excess overall quantization gap, E"),
        (axes[1], "Mean Psi (AP points)", "Area Psi; TT100K shown as height Psi"),
    ]:
        sns.barplot(
            data=summary,
            x="Dataset",
            y=y,
            hue="precision",
            hue_order=QUANT_ORDER,
            palette=palette,
            ax=ax,
        )
        ax.axhline(0, color="black", linewidth=0.8)
        ax.set_title(title)
        ax.set_xlabel("")
    axes[0].legend_.remove()
    handles, _ = axes[1].get_legend_handles_labels()
    axes[1].legend(handles, ["INT8", "FP8"], title="")
    save_figure(fig, out / "fig_dataset_interactions")

    severity = (
        bootstrap[bootstrap.dataset.isin(["coco", "voc", "kitti"])]
        .groupby(["precision", "corruption", "severity"], sort=False)
        .agg(e_small=("e_small", "mean"), e_large=("e_large", "mean"))
        .reset_index()
        .melt(
            id_vars=["precision", "corruption", "severity"],
            value_vars=["e_small", "e_large"],
            var_name="stratum",
            value_name="effect",
        )
    )
    severity["effect"] *= 100
    fig, axes = plt.subplots(2, 4, figsize=(11.2, 5.0), sharex=True, sharey=True)
    for i, precision in enumerate(QUANT_ORDER):
        for j, corruption in enumerate(CORRUPTION_ORDER):
            ax = axes[i, j]
            subset = severity[(severity.precision == precision) & (severity.corruption == corruption)]
            sns.lineplot(
                data=subset,
                x="severity",
                y="effect",
                hue="stratum",
                style="stratum",
                markers=True,
                dashes=False,
                palette={"e_small": "#DD8452", "e_large": "#55A868"},
                ax=ax,
            )
            ax.axhline(0, color="black", linewidth=0.6)
            ax.set_title(f"{DISPLAY[precision]} — {DISPLAY[corruption]}")
            ax.set_xlabel("Severity")
            ax.set_ylabel("E (AP points)" if j == 0 else "")
            if not (i == 0 and j == 3):
                ax.legend_.remove()
            else:
                handles, _ = ax.get_legend_handles_labels()
                ax.legend(handles, ["Small-like", "Large-like"], title="")
    save_figure(fig, out / "fig_severity_profiles")

    heat = (
        bootstrap.groupby(["dataset", "precision", "corruption"], sort=False).psi.mean().mul(100).reset_index()
    )
    fig, axes = plt.subplots(2, 4, figsize=(10.4, 4.9), sharex=True, sharey=True)
    limit = max(abs(heat.psi.min()), abs(heat.psi.max()))
    for i, precision in enumerate(QUANT_ORDER):
        for j, dataset in enumerate(DATASETS):
            ax = axes[i, j]
            sub = heat[(heat.precision == precision) & (heat.dataset == dataset)]
            matrix = sub.set_index("corruption").reindex(CORRUPTION_ORDER)[["psi"]]
            matrix.index = [DISPLAY[x] for x in matrix.index]
            sns.heatmap(
                matrix,
                cmap="vlag",
                center=0,
                vmin=-limit,
                vmax=limit,
                annot=True,
                fmt="+.2f",
                cbar=(j == 3),
                cbar_kws={"label": "Mean Psi (AP points)"},
                ax=ax,
            )
            endpoint = "height" if dataset == "tt100k" else "area"
            ax.set_title(f"{DISPLAY[dataset]} ({endpoint})")
            ax.set_xlabel(DISPLAY[precision])
            ax.set_ylabel("")
    save_figure(fig, out / "fig_interaction_heatmap")


def make_macros(metrics: pd.DataFrame, bootstrap: pd.DataFrame, out: Path) -> dict[str, float]:
    summary = bootstrap.groupby(["dataset", "precision"])[["e_all", "e_small", "e_large", "psi"]].mean()
    area = bootstrap[bootstrap.dataset.isin(["coco", "voc", "kitti"])]
    complete_grid = bootstrap
    tt_height = bootstrap[bootstrap.dataset == "tt100k"]
    coco = bootstrap[bootstrap.dataset == "coco"]
    values: dict[str, float] = {
        "MeanINTCleanGap": complete_grid[complete_grid.precision == "int8-entropy"].q_clean_all.mean(),
        "MeanFPTEightCleanGap": complete_grid[complete_grid.precision == "fp8"].q_clean_all.mean(),
        "AreaINTMeanPsi": area[area.precision == "int8-entropy"].psi.mean(),
        "AreaFPTEightMeanPsi": area[area.precision == "fp8"].psi.mean(),
        "TransferINTMeanPsi": area[area.precision == "int8-entropy"].psi.mean(),
        "TransferFPTEightMeanPsi": area[area.precision == "fp8"].psi.mean(),
        "TTINTMeanPsi": summary.loc[("tt100k", "int8-entropy"), "psi"],
        "TTFPTEightMeanPsi": summary.loc[("tt100k", "fp8"), "psi"],
        "AllINTMeanE": complete_grid[complete_grid.precision == "int8-entropy"].e_all.mean(),
        "AllFPTEightMeanE": complete_grid[complete_grid.precision == "fp8"].e_all.mean(),
        "COCOINTMeanE": coco[coco.precision == "int8-entropy"].e_all.mean(),
        "COCOFPTEightMeanE": coco[coco.precision == "fp8"].e_all.mean(),
        "MostNegativePsi": area.psi.min(),
        "MostPositivePsi": area.psi.max(),
        "TTMostNegativePsi": tt_height.psi.min(),
        "TTMostPositivePsi": tt_height.psi.max(),
    }
    lines = ["% Generated by analysis/build_paper_artifacts.py; values are AP points."]
    for name, value in values.items():
        lines.append(rf"\newcommand{{\{name}}}{{{100 * value:+.2f}}}")
    ci_positive = int((area.psi_ci_low > 0).sum())
    ci_negative = int((area.psi_ci_high < 0).sum())
    ci_crossing = int(((area.psi_ci_low <= 0) & (area.psi_ci_high >= 0)).sum())
    e_ci_positive = int((area.e_all_ci_low > 0).sum())
    e_ci_negative = int((area.e_all_ci_high < 0).sum())
    e_ci_crossing = int(
        ((area.e_all_ci_low <= 0) & (area.e_all_ci_high >= 0)).sum()
    )
    tt_ci_positive = int((tt_height.psi_ci_low > 0).sum())
    tt_ci_negative = int((tt_height.psi_ci_high < 0).sum())
    tt_ci_crossing = int(
        ((tt_height.psi_ci_low <= 0) & (tt_height.psi_ci_high >= 0)).sum()
    )
    area_fp32 = metrics[
        metrics.dataset.isin(["coco", "voc", "kitti"])
        & metrics.precision.eq("fp32")
        & metrics.corruption.ne("clean")
    ][["dataset", "model", "corruption", "severity", "AP"]]
    area_floor = area.merge(
        area_fp32,
        on=["dataset", "model", "corruption", "severity"],
        validate="many_to_one",
    )
    below_five = area_floor[area_floor.AP < 0.05]
    below_ten = area_floor[area_floor.AP < 0.10]
    lines.extend(
        [
            rf"\newcommand{{\AreaECIPositive}}{{{e_ci_positive}}}",
            rf"\newcommand{{\AreaECINegative}}{{{e_ci_negative}}}",
            rf"\newcommand{{\AreaECICrossing}}{{{e_ci_crossing}}}",
            rf"\newcommand{{\PsiCIPositive}}{{{ci_positive}}}",
            rf"\newcommand{{\PsiCINegative}}{{{ci_negative}}}",
            rf"\newcommand{{\PsiCICrossing}}{{{ci_crossing}}}",
            rf"\newcommand{{\TTPsiCIPositive}}{{{tt_ci_positive}}}",
            rf"\newcommand{{\TTPsiCINegative}}{{{tt_ci_negative}}}",
            rf"\newcommand{{\TTPsiCICrossing}}{{{tt_ci_crossing}}}",
            rf"\newcommand{{\AreaBelowFiveAPNegativeE}}{{{int((below_five.e_all < 0).sum())}}}",
            rf"\newcommand{{\AreaBelowFiveAPCells}}{{{len(below_five)}}}",
            rf"\newcommand{{\AreaBelowTenAPNegativeE}}{{{int((below_ten.e_all < 0).sum())}}}",
            rf"\newcommand{{\AreaBelowTenAPCells}}{{{len(below_ten)}}}",
        ]
    )
    out.write_text("\n".join(lines) + "\n", encoding="utf-8")
    return values


def load_direct_evidence(
    artifact_root: Path,
) -> tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame, dict[str, Any], dict[str, Any]]:
    """Load only the fully validated direct INT8--FP8 evidence chain.

    This deliberately has no fallback to the historical per-format P0 analysis:
    direct paired tables can be produced only after both completion reports pass
    their byte-level/provenance validators.
    """
    root = Path(artifact_root).resolve()
    contrast_report = validate_format_contrast(root)
    deployment_report = validate_deployment_benchmark(root)
    _, _, corrupted = _format_component_grid()

    contrast_rows: list[dict[str, Any]] = []
    for relative in sorted(corrupted):
        document = _read_object(root / relative, "direct paired component")
        dataset, model, condition = Path(relative).stem.split("__")
        corruption, severity_text = condition.rsplit("-s", 1)
        point = document["point"]
        intervals = document["percentile_intervals"]
        labels = ("small_like", "large_like") if dataset == "tt100k" else ("small", "large")
        contrast_rows.append(
            {
                "dataset": dataset,
                "model": model,
                "corruption": corruption,
                "severity": int(severity_text),
                "endpoint_type": document["endpoint_type"],
                "n_images": document["n_images"],
                "delta_q_all": float(point["delta_q"]["all"]),
                "delta_e_all": float(point["delta_e"]["all"]),
                "delta_e_ci_low": float(intervals["delta_e"]["all"][0]),
                "delta_e_ci_median": float(intervals["delta_e"]["all"][1]),
                "delta_e_ci_high": float(intervals["delta_e"]["all"][2]),
                "delta_e_small": float(point["delta_e"][labels[0]]),
                "delta_e_large": float(point["delta_e"][labels[1]]),
                "delta_psi": float(point["delta_psi"]),
                "delta_psi_ci_low": float(intervals["delta_psi"][0]),
                "delta_psi_ci_median": float(intervals["delta_psi"][1]),
                "delta_psi_ci_high": float(intervals["delta_psi"][2]),
            }
        )
    contrast = pd.DataFrame(contrast_rows)
    if len(contrast) != 144 or contrast.duplicated(
        ["dataset", "model", "corruption", "severity"]
    ).any():
        raise RuntimeError("direct paired contrast table does not have the exact 144-cell grid")

    macro_relative = (
        f"outputs/bootstrap/{FORMAT_CONTRAST_ATTEMPT}/"
        f"{FORMAT_CONTRAST_ATTEMPT}_joint_macro.json"
    )
    macro = _read_object(root / macro_relative, "direct paired macro")
    macro_rows = pd.DataFrame(
        [
            {
                "endpoint": endpoint,
                "point": float(macro["point"][endpoint]),
                "ci_low": float(macro["percentile_intervals"][endpoint][0]),
                "ci_median": float(macro["percentile_intervals"][endpoint][1]),
                "ci_high": float(macro["percentile_intervals"][endpoint][2]),
                "planned_component_count": int(
                    macro["coverage"][endpoint]["planned_component_count"]
                ),
                "finite_complete_replicates": int(
                    macro["coverage"][endpoint]["finite_complete_replicates"]
                ),
            }
            for endpoint in (
                "four_dataset_macro_delta_e",
                "area_macro_delta_psi",
                "tt100k_height_macro_delta_psi",
            )
        ]
    )

    record_dir = root / "outputs" / "benchmarks" / DEPLOYMENT_ATTEMPT
    deployment_rows: list[dict[str, Any]] = []
    for record_name in sorted(deployment_report["records_sha256"]):
        record = _read_object(record_dir / record_name, "deployment benchmark record")
        deployment_rows.append(
            {
                "dataset": record["dataset"],
                "model": record["model"],
                "precision": record["precision"],
                "repetition": int(record["repetition"]),
                "engine_bytes": int(record["engine_bytes"]),
                "latency_mean_ms": float(record["latency_mean_ms"]),
                "latency_median_ms": float(record["latency_median_ms"]),
                "latency_p99_ms": float(record["latency_p99_ms"]),
                "throughput_qps": float(record["throughput_qps"]),
            }
        )
    repetitions = pd.DataFrame(deployment_rows)
    if len(repetitions) != 108:
        raise RuntimeError("direct deployment table does not have 108 repetitions")
    condition_columns = ["dataset", "model", "precision"]
    if repetitions.groupby(condition_columns).size().ne(3).any():
        raise RuntimeError("direct deployment repetitions are not balanced")
    if repetitions.groupby(condition_columns)["engine_bytes"].nunique().ne(1).any():
        raise RuntimeError("direct deployment engine size changed across repetitions")
    deployment = (
        repetitions.groupby(condition_columns, as_index=False, sort=False)
        .agg(
            engine_bytes=("engine_bytes", "first"),
            latency_mean_ms=("latency_mean_ms", "median"),
            latency_median_ms=("latency_median_ms", "median"),
            latency_p99_ms=("latency_p99_ms", "median"),
            throughput_qps=("throughput_qps", "median"),
        )
        .sort_values(condition_columns, key=lambda column: column.map({
            **{value: index for index, value in enumerate(DATASETS)},
            **{value: index for index, value in enumerate(MODEL_ORDER)},
            **{value: index for index, value in enumerate(DEPLOYMENT_PRECISIONS)},
        }))
        .reset_index(drop=True)
    )
    if len(deployment) != 36:
        raise RuntimeError("direct deployment condition grid does not have 36 cells")
    return contrast, macro_rows, deployment, contrast_report, deployment_report


def _validate_direct_metric_binding(
    source: dict[str, Any],
    metric: dict[str, Any],
    *,
    component_name: str,
    arm: str,
    dataset: str,
    model: str,
    corruption: str,
    severity: int,
) -> None:
    """Reject any provenance or treatment mismatch for one direct arm."""
    for source_field, metric_field in (
        ("prediction_sha256", "prediction_sha256"),
        ("input_manifest_sha256", "input_manifest_sha256"),
        ("image_ids_sha256", "input_image_ids_sha256"),
        ("run_record_sha256", "run_record_sha256"),
    ):
        if source.get(source_field) != metric.get(metric_field):
            raise RuntimeError(
                f"direct/metric provenance mismatch for "
                f"{component_name}:{arm}:{source_field}"
            )
    precision = arm.split("_", 1)[0]
    expected_precision = {"int8": "int8-entropy", "fp8": "fp8"}[precision]
    expected_corruption = "codec_control" if arm.endswith("_clean") else corruption
    expected_severity = 0 if arm.endswith("_clean") else severity
    if (
        metric.get("dataset") != dataset
        or metric.get("model") != model
        or metric.get("precision") != expected_precision
        or metric.get("corruption") != expected_corruption
        or int(metric.get("severity", -1)) != expected_severity
    ):
        raise RuntimeError(f"direct/metric semantic mismatch for {component_name}:{arm}")


def load_direct_accuracy_guardrail(
    artifact_root: Path,
) -> tuple[pd.DataFrame, dict[str, str], dict[str, str]]:
    """Bind absolute INT8/FP8 AP and loss to the exact direct prediction arms.

    The direct bootstrap deliberately stores only format contrasts.  This
    guardrail reuses the independently completed metric chain, but accepts a
    metric only when prediction, input-manifest, ordered-image, and run-record
    hashes all equal the corresponding direct arm.  It therefore adds absolute
    interpretation without rerunning inference or weakening the evidence
    boundary.
    """
    root = Path(artifact_root).resolve()
    # This validates the full 468-metric/36-codec-control P0 completion chain
    # before any metric document is indexed below.
    validate_and_load(root)
    metric_paths = [
        *sorted((root / "outputs" / "metrics" / ATTEMPTS["coco"]).glob("*.json")),
        *[
            path
            for dataset in ("voc", "kitti", "tt100k")
            for path in sorted(
                (
                    root
                    / LEGACY_ARTIFACT_ROOT
                    / "outputs"
                    / "metrics"
                    / ATTEMPTS[dataset]
                ).glob("*.json")
            )
        ],
        *sorted((root / "outputs" / "metrics" / "codec_control_p0_v1").glob("*.json")),
    ]
    metric_index: dict[str, tuple[Path, dict[str, Any]]] = {}
    for path in metric_paths:
        document = _read_object(path, "absolute-accuracy metric")
        prediction_sha = document.get("prediction_sha256")
        if not isinstance(prediction_sha, str) or len(prediction_sha) != 64:
            raise RuntimeError(f"absolute-accuracy metric lacks prediction hash: {path}")
        if prediction_sha in metric_index:
            raise RuntimeError("absolute-accuracy prediction hash is not unique")
        metric_index[prediction_sha] = (path, document)

    _, _, corrupted = _format_component_grid()
    rows: list[dict[str, Any]] = []
    used_metric_paths: dict[str, str] = {}
    precision_by_arm = {"int8": "int8-entropy", "fp8": "fp8"}
    for relative in sorted(corrupted):
        component = _read_object(root / relative, "direct paired component")
        dataset, model, condition = Path(relative).stem.split("__")
        corruption, severity_text = condition.rsplit("-s", 1)
        severity = int(severity_text)
        metrics: dict[str, dict[str, Any]] = {}
        for arm, source in component["input_hashes"].items():
            prediction_sha = source["prediction_sha256"]
            if prediction_sha not in metric_index:
                raise RuntimeError(f"direct prediction has no validated metric record: {arm}")
            path, metric = metric_index[prediction_sha]
            _validate_direct_metric_binding(
                source,
                metric,
                component_name=Path(relative).name,
                arm=arm,
                dataset=dataset,
                model=model,
                corruption=corruption,
                severity=severity,
            )
            metrics[arm] = metric
            used_metric_paths[str(path.relative_to(root))] = sha256_file(path)

        for precision in ("int8", "fp8"):
            clean = metrics[f"{precision}_clean"]
            corrupt = metrics[f"{precision}_corrupt"]
            clean_ap = float(clean["stats"]["AP"])
            corrupt_ap = float(corrupt["stats"]["AP"])
            rows.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "precision": precision_by_arm[precision],
                    "corruption": corruption,
                    "severity": severity,
                    "clean_ap_native": clean_ap,
                    "corrupt_ap_native": corrupt_ap,
                    "d_native": clean_ap - corrupt_ap,
                    "clean_metric_sha256": sha256_file(metric_index[clean["prediction_sha256"]][0]),
                    "corrupt_metric_sha256": sha256_file(metric_index[corrupt["prediction_sha256"]][0]),
                }
            )
        cell_rows = rows[-2:]
        reconstructed = cell_rows[0]["d_native"] - cell_rows[1]["d_native"]
        recorded = float(component["point"]["delta_e"]["all"])
        if not math.isclose(reconstructed, recorded, rel_tol=0.0, abs_tol=1e-12):
            raise RuntimeError(
                f"absolute-accuracy records do not reconstruct direct delta_e: {Path(relative).name}"
            )

    guardrail = pd.DataFrame(rows)
    expected = set(product(DATASETS, MODEL_ORDER, QUANT_ORDER, CORRUPTION_ORDER, (1, 3, 5)))
    observed = set(
        guardrail[["dataset", "model", "precision", "corruption", "severity"]]
        .itertuples(index=False, name=None)
    )
    if len(guardrail) != 288 or observed != expected:
        raise RuntimeError("absolute-accuracy guardrail does not have the exact 288-arm grid")
    if len(used_metric_paths) != 312:
        raise RuntimeError("absolute-accuracy guardrail does not bind exactly 312 metric records")
    completion_paths = {
        "outputs/reports/p0_strengthening_complete_v1.json": root
        / "outputs"
        / "reports"
        / "p0_strengthening_complete_v1.json",
        "artifacts/four_dataset_pilot_v1/outputs/reports/four_dataset_pilot_final_report.json": root
        / LEGACY_ARTIFACT_ROOT
        / "outputs"
        / "reports"
        / "four_dataset_pilot_final_report.json",
    }
    completion_hashes = {relative: sha256_file(path) for relative, path in completion_paths.items()}
    return guardrail, dict(sorted(used_metric_paths.items())), completion_hashes


def load_direct_size_accuracy_guardrail(
    artifact_root: Path, guardrail: pd.DataFrame
) -> pd.DataFrame:
    """Recover absolute area/height AP from the exact metrics bound by the direct arms."""
    root = Path(artifact_root).resolve()
    validate_and_load(root)
    metric_paths = [
        *sorted((root / "outputs" / "metrics" / ATTEMPTS["coco"]).glob("*.json")),
        *[
            path
            for dataset in ("voc", "kitti", "tt100k")
            for path in sorted(
                (root / LEGACY_ARTIFACT_ROOT / "outputs" / "metrics" / ATTEMPTS[dataset]).glob("*.json")
            )
        ],
        *sorted((root / "outputs" / "metrics" / "codec_control_p0_v1").glob("*.json")),
    ]
    by_file_sha = {
        sha256_file(path): _read_object(path, "size-accuracy metric") for path in metric_paths
    }
    required_hashes = set(guardrail.clean_metric_sha256) | set(guardrail.corrupt_metric_sha256)
    if not required_hashes.issubset(by_file_sha):
        raise RuntimeError("size guardrail cannot resolve every bound metric hash")

    rows: list[dict[str, Any]] = []
    for arm in guardrail.itertuples(index=False):
        clean = by_file_sha[arm.clean_metric_sha256]
        corrupt = by_file_sha[arm.corrupt_metric_sha256]
        if arm.dataset == "tt100k":
            endpoint_type = "original-height"
            endpoints = ("XS", "S", "L", "XL")
            clean_values = clean.get("height_strata_stats", {})
            corrupt_values = corrupt.get("height_strata_stats", {})
            values = [
                (endpoint, clean_values.get(endpoint, {}).get("AP"), corrupt_values.get(endpoint, {}).get("AP"))
                for endpoint in endpoints
            ]
        else:
            endpoint_type = "coco-area"
            values = [
                ("small", clean.get("stats", {}).get("AP_small"), corrupt.get("stats", {}).get("AP_small")),
                ("large", clean.get("stats", {}).get("AP_large"), corrupt.get("stats", {}).get("AP_large")),
            ]
        for endpoint, clean_ap, corrupt_ap in values:
            if not isinstance(clean_ap, (int, float)) or not isinstance(corrupt_ap, (int, float)):
                raise RuntimeError(
                    f"size guardrail lacks {endpoint_type}/{endpoint} AP for "
                    f"{arm.dataset}/{arm.model}/{arm.precision}/{arm.corruption}-s{arm.severity}"
                )
            rows.append(
                {
                    "dataset": arm.dataset,
                    "model": arm.model,
                    "precision": arm.precision,
                    "corruption": arm.corruption,
                    "severity": int(arm.severity),
                    "endpoint_type": endpoint_type,
                    "endpoint": endpoint,
                    "clean_ap_native": float(clean_ap),
                    "corrupt_ap_native": float(corrupt_ap),
                    "d_native": float(clean_ap) - float(corrupt_ap),
                    "clean_metric_sha256": arm.clean_metric_sha256,
                    "corrupt_metric_sha256": arm.corrupt_metric_sha256,
                }
            )
    result = pd.DataFrame(rows)
    expected_endpoints = {
        **{dataset: {"small", "large"} for dataset in ("coco", "voc", "kitti")},
        "tt100k": {"XS", "S", "L", "XL"},
    }
    if len(result) != 720 or any(
        set(result.loc[result.dataset == dataset, "endpoint"]) != endpoints
        for dataset, endpoints in expected_endpoints.items()
    ):
        raise RuntimeError("size guardrail does not have the exact endpoint grid")
    return result


def direct_heterogeneity_sensitivity(
    contrast: pd.DataFrame,
    *,
    image_counts: dict[str, int],
    object_counts: dict[str, int],
) -> pd.DataFrame:
    """Compute descriptive weighting and leave-one-factor-out summaries."""
    required = {"dataset", "corruption", "delta_e_all"}
    if not required.issubset(contrast.columns) or contrast.empty:
        raise RuntimeError("heterogeneity sensitivity requires non-empty direct cells")
    datasets = set(contrast.dataset)
    if set(image_counts) != datasets or set(object_counts) != datasets:
        raise RuntimeError("heterogeneity sensitivity weights do not match datasets")
    if any(value <= 0 for value in [*image_counts.values(), *object_counts.values()]):
        raise RuntimeError("heterogeneity sensitivity weights must be positive")

    values = contrast.delta_e_all.to_numpy(dtype=float)
    rows: list[dict[str, Any]] = [
        {
            "analysis": "distribution",
            "level": "all-cells",
            "cells": len(contrast),
            "point_native": float(values.mean()),
            "sd_native": float(values.std(ddof=1)) if len(values) > 1 else 0.0,
            "q25_native": float(np.quantile(values, 0.25)),
            "median_native": float(np.median(values)),
            "q75_native": float(np.quantile(values, 0.75)),
        }
    ]

    def add_point(analysis: str, level: str, frame: pd.DataFrame, point: float) -> None:
        rows.append(
            {
                "analysis": analysis,
                "level": level,
                "cells": len(frame),
                "point_native": float(point),
                "sd_native": float("nan"),
                "q25_native": float("nan"),
                "median_native": float("nan"),
                "q75_native": float("nan"),
            }
        )

    add_point("weighting", "equal-cell", contrast, float(values.mean()))
    for level, counts in (("image-weighted", image_counts), ("object-weighted", object_counts)):
        weights = contrast.dataset.map(counts).to_numpy(dtype=float)
        add_point("weighting", level, contrast, float(np.average(values, weights=weights)))
    for dataset in sorted(datasets):
        retained = contrast[contrast.dataset != dataset]
        add_point(
            "leave-one-dataset-out", dataset, retained, float(retained.delta_e_all.mean())
        )
    for corruption in sorted(set(contrast.corruption)):
        retained = contrast[contrast.corruption != corruption]
        add_point(
            "leave-one-corruption-out",
            corruption,
            retained,
            float(retained.delta_e_all.mean()),
        )
    return pd.DataFrame(rows)


def direct_runtime_sensitivity(
    deployment: pd.DataFrame, *, image_sizes: dict[str, int]
) -> pd.DataFrame:
    """Summarize matched latency ratios without pooling input geometries blindly."""
    required = {"dataset", "model", "precision", "latency_median_ms"}
    if not required.issubset(deployment.columns) or deployment.empty:
        raise RuntimeError("runtime sensitivity requires matched deployment conditions")
    if set(deployment.dataset) != set(image_sizes):
        raise RuntimeError("runtime sensitivity image sizes do not match datasets")
    pivot = deployment.pivot(
        index=["dataset", "model"], columns="precision", values="latency_median_ms"
    ).reset_index()
    if not set(DEPLOYMENT_PRECISIONS).issubset(pivot.columns):
        raise RuntimeError("runtime sensitivity lacks a complete precision ladder")
    pivot["input_size"] = pivot.dataset.map(image_sizes).astype(int)
    pivot["fp32_over_int8"] = pivot.fp32 / pivot["int8-entropy"]
    pivot["fp32_over_fp8"] = pivot.fp32 / pivot.fp8
    pivot["int8_over_fp8"] = pivot["int8-entropy"] / pivot.fp8
    ratio_columns = ("fp32_over_int8", "fp32_over_fp8", "int8_over_fp8")
    rows: list[dict[str, Any]] = []
    for stratum, column in (("input-size", "input_size"), ("capacity", "model")):
        for level, frame in pivot.groupby(column, sort=True):
            row: dict[str, Any] = {
                "stratum": stratum,
                "level": str(level),
                "conditions": len(frame),
            }
            for ratio in ratio_columns:
                values = frame[ratio].to_numpy(dtype=float)
                row[ratio] = float(np.median(values))
                row[f"{ratio}_q25"] = float(np.quantile(values, 0.25))
                row[f"{ratio}_q75"] = float(np.quantile(values, 0.75))
            rows.append(row)
    return pd.DataFrame(rows)


def _write_direct_sensitivity_artifacts(
    contrast: pd.DataFrame, deployment: pd.DataFrame, out: Path
) -> tuple[pd.DataFrame, pd.DataFrame]:
    dataset_rows = {row[0].casefold(): row for row in DATASET_META}
    image_counts = {name: int(values[2]) for name, values in dataset_rows.items()}
    object_counts = {name: int(values[3]) for name, values in dataset_rows.items()}
    image_sizes = {name: int(values[5]) for name, values in dataset_rows.items()}
    heterogeneity = direct_heterogeneity_sensitivity(
        contrast, image_counts=image_counts, object_counts=object_counts
    )
    runtime = direct_runtime_sensitivity(deployment, image_sizes=image_sizes)
    heterogeneity.to_csv(out / "direct_heterogeneity_sensitivity.csv", index=False)
    runtime.to_csv(out / "direct_runtime_sensitivity.csv", index=False)

    shown = heterogeneity.copy()
    shown["Analysis"] = shown.analysis.map(
        {
            "distribution": "Cell distribution",
            "weighting": "Weighting",
            "leave-one-dataset-out": "Leave-one-dataset-out",
            "leave-one-corruption-out": "Leave-one-corruption-out",
        }
    )
    shown["Level"] = shown.level.map(lambda value: DISPLAY.get(value, value.replace("-", " ")))
    shown["Cells"] = shown.cells.astype(str)
    shown[r"Mean $\Delta E$ (AP pt)"] = shown.point_native.map(fmt_effect)
    shown["SD / IQR (AP pt)"] = shown.apply(
        lambda row: (
            f"{100 * row.sd_native:.2f} / "
            f"[{100 * row.q25_native:+.2f}, {100 * row.q75_native:+.2f}]"
            if math.isfinite(row.sd_native)
            else "--"
        ),
        axis=1,
    )
    write_booktabs(
        out / "direct_heterogeneity_sensitivity.tex",
        shown,
        ["Analysis", "Level", "Cells", r"Mean $\Delta E$ (AP pt)", "SD / IQR (AP pt)"],
        "llrrl",
    )

    shown_runtime = runtime.copy()
    shown_runtime["Stratum"] = shown_runtime.stratum.map(
        {"input-size": "Input size", "capacity": "Capacity"}
    )
    shown_runtime["Level"] = shown_runtime.level.map(lambda value: DISPLAY.get(value, value))
    shown_runtime["Conditions"] = shown_runtime.conditions.astype(str)
    for source, target in (
        ("fp32_over_int8", "FP32/INT8"),
        ("fp32_over_fp8", "FP32/FP8"),
        ("int8_over_fp8", "INT8/FP8"),
    ):
        shown_runtime[target] = shown_runtime.apply(
            lambda row: (
                f"{row[source]:.2f} "
                f"[{row[source + '_q25']:.2f}, {row[source + '_q75']:.2f}]"
            ),
            axis=1,
        )
    write_booktabs(
        out / "direct_runtime_sensitivity.tex",
        shown_runtime,
        ["Stratum", "Level", "Conditions", "FP32/INT8", "FP32/FP8", "INT8/FP8"],
        "llrlll",
    )
    weighting = heterogeneity[heterogeneity.analysis == "weighting"].set_index("level")
    leave_dataset = heterogeneity[
        heterogeneity.analysis == "leave-one-dataset-out"
    ].point_native
    leave_corruption = heterogeneity[
        heterogeneity.analysis == "leave-one-corruption-out"
    ].point_native
    distribution = heterogeneity[
        heterogeneity.analysis == "distribution"
    ].iloc[0]
    narrative = (
        "% Generated from validated direct cells; do not edit manually.\n"
        "The near-zero macro was also weighting-sensitive but remained small relative to "
        "the between-cell dispersion. Equal-cell, image-count-weighted, and "
        f"object-count-weighted means were {fmt_effect(weighting.loc['equal-cell', 'point_native'])}, "
        f"{fmt_effect(weighting.loc['image-weighted', 'point_native'])}, and "
        f"{fmt_effect(weighting.loc['object-weighted', 'point_native'])} AP points, respectively, "
        f"whereas the empirical cell SD was {100 * distribution.sd_native:.2f} points. "
        f"Leave-one-dataset-out means ranged from {fmt_effect(leave_dataset.min())} to "
        f"{fmt_effect(leave_dataset.max())} points, and leave-one-corruption-out means ranged "
        f"from {fmt_effect(leave_corruption.min())} to {fmt_effect(leave_corruption.max())}. "
        "These sign changes around zero reinforce conditional heterogeneity rather than a "
        "format-wide interaction.\n"
    )
    (out / "direct_sensitivity_narrative.tex").write_text(narrative, encoding="utf-8")
    return heterogeneity, runtime


def _direct_size_guardrail_summary(size_guardrail: pd.DataFrame, out: Path) -> None:
    size_guardrail.to_csv(out / "direct_size_guardrail.csv", index=False)
    summary = (
        size_guardrail.groupby(
            ["dataset", "precision", "endpoint_type", "endpoint"],
            as_index=False,
            sort=False,
        )
        .agg(
            cells=("d_native", "size"),
            mean_clean_ap=("clean_ap_native", "mean"),
            mean_corrupt_ap=("corrupt_ap_native", "mean"),
            mean_d=("d_native", "mean"),
            corrupt_below_5=("corrupt_ap_native", lambda values: int((values < 0.05).sum())),
        )
    )
    if len(summary) != 20 or set(summary.cells) != {36}:
        raise RuntimeError("size guardrail summary does not have the exact 20-row grid")
    shown = summary.copy()
    shown["Dataset"] = shown.dataset.map(DISPLAY)
    shown["Format"] = shown.precision.map(DISPLAY)
    shown["Endpoint"] = shown.endpoint_type + ":" + shown.endpoint
    shown["Cells"] = shown.cells.astype(str)
    shown["Clean AP"] = shown.mean_clean_ap.map(fmt_ap)
    shown["Corrupted AP"] = shown.mean_corrupt_ap.map(fmt_ap)
    shown[r"Mean $D$ (AP pt)"] = shown.mean_d.map(fmt_effect)
    shown["AP<5"] = shown.corrupt_below_5.astype(str)
    write_booktabs(
        out / "direct_size_guardrail_summary.tex",
        shown,
        ["Dataset", "Format", "Endpoint", "Cells", "Clean AP", "Corrupted AP", r"Mean $D$ (AP pt)", "AP<5"],
        "lllrrrrr",
    )


def _direct_tex_summary(contrast: pd.DataFrame, deployment: pd.DataFrame, out: Path) -> None:
    dataset_summary = (
        contrast.groupby("dataset", as_index=False, sort=False)
        .agg(
            cells=("delta_e_all", "size"),
            mean_delta_e=("delta_e_all", "mean"),
            ci_positive=("delta_e_ci_low", lambda values: int((values > 0).sum())),
            ci_negative=("delta_e_ci_high", lambda values: int((values < 0).sum())),
            mean_delta_psi=("delta_psi", "mean"),
        )
        .reindex(columns=["dataset", "cells", "mean_delta_e", "ci_positive", "ci_negative", "mean_delta_psi"])
        .set_index("dataset")
        .reindex(DATASETS)
        .reset_index()
    )
    dataset_summary["ci_crossing"] = (
        dataset_summary.cells - dataset_summary.ci_positive - dataset_summary.ci_negative
    )
    shown = dataset_summary.copy()
    shown["Dataset"] = shown.dataset.map(DISPLAY)
    shown["Cells"] = shown.cells.astype(str)
    shown[r"Mean $\Delta E$ (AP pt)"] = shown.mean_delta_e.map(fmt_effect)
    shown["95% percentile sign (+/-/cross)"] = shown.apply(
        lambda row: f"{int(row.ci_positive)} / {int(row.ci_negative)} / {int(row.ci_crossing)}",
        axis=1,
    )
    shown[r"Mean $\Delta\Psi$ (AP pt)"] = shown.mean_delta_psi.map(fmt_effect)
    write_booktabs(
        out / "direct_format_contrast_summary.tex",
        shown,
        ["Dataset", "Cells", r"Mean $\Delta E$ (AP pt)", "95% percentile sign (+/-/cross)", r"Mean $\Delta\Psi$ (AP pt)"],
        "lrrrr",
    )

    deploy_summary = (
        deployment.groupby("precision", as_index=False, sort=False)
        .agg(
            conditions=("engine_bytes", "size"),
            median_engine_mib=("engine_bytes", lambda values: float(np.median(values)) / (1024**2)),
            median_latency_ms=("latency_median_ms", "median"),
            median_throughput_qps=("throughput_qps", "median"),
        )
        .sort_values("precision", key=lambda column: column.map({value: index for index, value in enumerate(DEPLOYMENT_PRECISIONS)}))
    )
    shown_deploy = deploy_summary.copy()
    shown_deploy["Precision"] = shown_deploy.precision.map(DISPLAY)
    shown_deploy["Conditions"] = shown_deploy.conditions.astype(str)
    shown_deploy["Median engine (MiB)"] = shown_deploy.median_engine_mib.map(lambda value: f"{value:.2f}")
    shown_deploy["Median latency (ms)"] = shown_deploy.median_latency_ms.map(lambda value: f"{value:.3f}")
    shown_deploy["Median throughput (qps)"] = shown_deploy.median_throughput_qps.map(lambda value: f"{value:.1f}")
    write_booktabs(
        out / "direct_deployment_summary.tex",
        shown_deploy,
        ["Precision", "Conditions", "Median engine (MiB)", "Median latency (ms)", "Median throughput (qps)"],
        "lrrrr",
    )


def _direct_heterogeneity_summary(contrast: pd.DataFrame, out: Path) -> None:
    """Summarize all prespecified design axes without treating margins as tests."""
    frames: list[pd.DataFrame] = []
    for factor, column, order in (
        ("Checkpoint rung", "model", MODEL_ORDER),
        ("Corruption", "corruption", CORRUPTION_ORDER),
        ("Severity", "severity", (1, 3, 5)),
    ):
        grouped = (
            contrast.groupby(column, as_index=False, sort=False)
            .agg(
                cells=("delta_e_all", "size"),
                mean_delta_e=("delta_e_all", "mean"),
                min_delta_e=("delta_e_all", "min"),
                max_delta_e=("delta_e_all", "max"),
                interval_positive=("delta_e_ci_low", lambda values: int((values > 0).sum())),
                interval_negative=("delta_e_ci_high", lambda values: int((values < 0).sum())),
            )
            .set_index(column)
            .reindex(order)
            .reset_index()
        )
        grouped["interval_crossing"] = (
            grouped.cells - grouped.interval_positive - grouped.interval_negative
        )
        grouped.insert(0, "factor", factor)
        grouped = grouped.rename(columns={column: "level"})
        frames.append(grouped)
    summary = pd.concat(frames, ignore_index=True)
    summary.to_csv(out / "direct_heterogeneity_summary.csv", index=False)

    shown = pd.DataFrame(
        {
            "Factor": summary.factor,
            "Level": summary.level.map(lambda value: DISPLAY.get(value, str(value))),
            "Cells": summary.cells.astype(str),
            r"Mean $\Delta E$": summary.mean_delta_e.map(fmt_effect),
            r"Range $\Delta E$": summary.apply(
                lambda row: f"{fmt_effect(row.min_delta_e)} to {fmt_effect(row.max_delta_e)}",
                axis=1,
            ),
            "95% percentile sign (+/-/cross)": summary.apply(
                lambda row: (
                    f"{int(row.interval_positive)} / {int(row.interval_negative)} / "
                    f"{int(row.interval_crossing)}"
                ),
                axis=1,
            ),
        }
    )
    write_booktabs(
        out / "direct_heterogeneity_summary.tex",
        shown,
        [
            "Factor",
            "Level",
            "Cells",
            r"Mean $\Delta E$",
            r"Range $\Delta E$",
            "95% percentile sign (+/-/cross)",
        ],
        "llrrrr",
    )


def _direct_accuracy_guardrail_summary(guardrail: pd.DataFrame, out: Path) -> None:
    """Write a compact absolute-accuracy table from the hash-matched arms."""
    severe = (
        guardrail[guardrail.severity.eq(5)]
        .groupby(["dataset", "precision"], as_index=False)
        .agg(severe_ap=("corrupt_ap_native", "mean"), severe_d=("d_native", "mean"))
    )
    summary = (
        guardrail.groupby(["dataset", "precision"], as_index=False)
        .agg(
            clean_ap=("clean_ap_native", "mean"),
            corrupt_ap=("corrupt_ap_native", "mean"),
            mean_d=("d_native", "mean"),
        )
        .merge(severe, on=["dataset", "precision"], validate="one_to_one")
        .sort_values(
            ["dataset", "precision"],
            key=lambda column: column.map(
                {
                    **{value: index for index, value in enumerate(DATASETS)},
                    "int8-entropy": 0,
                    "fp8": 1,
                }
            ),
        )
    )
    shown = pd.DataFrame(
        {
            "Dataset": summary.dataset.map(DISPLAY),
            "Format": summary.precision.map(DISPLAY),
            "Clean AP": summary.clean_ap.map(fmt_points),
            "All-corrupt AP": summary.corrupt_ap.map(fmt_points),
            "Mean $D$": summary.mean_d.map(fmt_points),
            "Severity-5 AP": summary.severe_ap.map(fmt_points),
            "Severity-5 $D$": summary.severe_d.map(fmt_points),
        }
    )
    write_booktabs(
        out / "direct_absolute_guardrail_summary.tex",
        shown,
        [
            "Dataset",
            "Format",
            "Clean AP",
            "All-corrupt AP",
            "Mean $D$",
            "Severity-5 AP",
            "Severity-5 $D$",
        ],
        "llrrrrr",
    )
    narrative = (
        "% Generated from 312 hash-matched validated metric records; do not edit manually.\n"
        f"Across the 288 format-specific corrupted arms, {int((guardrail.d_native > 0).sum())} "
        f"had lower AP than their matched JPEG-95 clean arm, while "
        f"{int((guardrail.d_native < 0).sum())} had a small gain. The median signed "
        f"clean-to-corrupted AP loss was {100 * guardrail.d_native.median():.2f} AP points and the mean was "
        f"{100 * guardrail.d_native.mean():.2f} AP points, exposing the right-skewed "
        "severity of the loss distribution. Moreover, "
        f"{int((guardrail.corrupt_ap_native < 0.10).sum())} arms had corrupted AP below 10 AP points and "
        f"{int((guardrail.corrupt_ap_native < 0.05).sum())} below 5 AP points. Thus a "
        "contracting FP8--INT8 discrepancy can coexist with severe absolute degradation; "
        "negative $\\Delta E$ is not evidence of robustness.\n"
    )
    (out / "direct_absolute_guardrail_narrative.tex").write_text(
        narrative, encoding="utf-8"
    )


def _direct_paired_figure(contrast: pd.DataFrame, macro: pd.DataFrame, out: Path) -> None:
    """Make a final-column-width direct-contrast figure with interval provenance."""
    setup_plotting()
    fig, axes = plt.subplots(1, 2, figsize=(6.89, 3.25), gridspec_kw={"width_ratios": [1.22, 1.0]})
    data = contrast.copy()
    data["dataset_label"] = data.dataset.map(DISPLAY)
    data["delta_e_ap_points"] = 100.0 * data["delta_e_all"]
    sns.stripplot(
        data=data,
        x="delta_e_ap_points",
        y="dataset_label",
        order=[DISPLAY[item] for item in DATASETS],
        color="#3B82A0",
        alpha=0.52,
        jitter=0.22,
        size=2.7,
        ax=axes[0],
    )
    means = data.groupby("dataset", sort=False).delta_e_ap_points.mean().reindex(DATASETS)
    for y, value in enumerate(means):
        axes[0].plot(value, y, marker="D", color="#B45309", markersize=4.7, zorder=4)
    axes[0].axvline(0, color="#111827", linewidth=0.8)
    axes[0].set_xlabel(r"Direct paired $\Delta E$ (AP points)")
    axes[0].set_ylabel("")
    axes[0].set_title("144 matched corruption cells")
    axes[0].grid(axis="x", alpha=0.25)
    axes[0].grid(axis="y", visible=False)

    labels = {
        "four_dataset_macro_delta_e": r"Four-dataset $\Delta E$",
        "area_macro_delta_psi": r"Area $\Delta\Psi$",
        "tt100k_height_macro_delta_psi": r"TT100K height $\Delta\Psi$",
    }
    macro = macro.iloc[::-1].reset_index(drop=True)
    for y, row in macro.iterrows():
        point, low, high = 100 * row[["point", "ci_low", "ci_high"]].to_numpy(dtype=float)
        axes[1].errorbar(
            point, y, xerr=[[point - low], [high - point]], fmt="o", color="#B45309",
            ecolor="#B45309", capsize=2.6, markersize=4.6, linewidth=1.15,
        )
    axes[1].axvline(0, color="#111827", linewidth=0.8)
    axes[1].set_yticks(range(len(macro)), [labels[item] for item in macro.endpoint])
    # The caption carries the full interval terminology; a compact axis label
    # prevents clipping in the two-column CAS layout.
    axes[1].set_xlabel("Paired AP points")
    axes[1].set_title("Joint bootstrap summaries")
    axes[1].grid(axis="x", alpha=0.25)
    axes[1].grid(axis="y", visible=False)
    fig.suptitle("Direct INT8–FP8 paired excess-gap evidence", y=1.02, fontsize=10.5, fontweight="bold")
    fig.tight_layout()
    save_figure(fig, out / "fig_paired_excess_gap")


def generate_direct_evidence_artifacts(
    artifact_root: Path, generated: Path, figures: Path
) -> dict[str, Any]:
    """Emit direct paired/RTX-5090 artifacts only after both chains validate.

    Validation is deliberately completed before either destination directory is
    created, so a failed completion chain cannot leave plausible-looking output.
    """
    root = Path(artifact_root).resolve()
    contrast, macro, deployment, contrast_report, deployment_report = load_direct_evidence(root)
    guardrail, guardrail_metric_hashes, guardrail_completion_hashes = (
        load_direct_accuracy_guardrail(root)
    )
    size_guardrail = load_direct_size_accuracy_guardrail(root, guardrail)
    canonical_metrics, _, _, codec_controls = validate_and_load(root)

    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)
    contrast.to_csv(generated / "direct_format_contrast_cells.csv", index=False)
    macro.to_csv(generated / "direct_format_contrast_macro.csv", index=False)
    deployment.to_csv(generated / "direct_deployment_conditions.csv", index=False)
    guardrail.to_csv(generated / "direct_absolute_guardrail.csv", index=False)
    _direct_tex_summary(contrast, deployment, generated)
    _direct_heterogeneity_summary(contrast, generated)
    _direct_accuracy_guardrail_summary(guardrail, generated)
    _direct_size_guardrail_summary(size_guardrail, generated)
    codec_sensitivity = generate_codec_sensitivity(
        canonical_metrics, codec_controls, generated
    )
    heterogeneity_sensitivity, runtime_sensitivity = (
        _write_direct_sensitivity_artifacts(contrast, deployment, generated)
    )
    _direct_paired_figure(contrast, macro, figures)
    values = {
        "DirectFourDatasetDeltaE": float(
            macro.loc[macro.endpoint.eq("four_dataset_macro_delta_e"), "point"].iloc[0]
        ),
        "DirectAreaDeltaPsi": float(
            macro.loc[macro.endpoint.eq("area_macro_delta_psi"), "point"].iloc[0]
        ),
        "DirectTTHeightDeltaPsi": float(
            macro.loc[macro.endpoint.eq("tt100k_height_macro_delta_psi"), "point"].iloc[0]
        ),
    }
    (generated / "numbers_direct.tex").write_text(
        "% Generated from validated direct paired evidence; values are AP points.\n"
        + "\n".join(
            rf"\newcommand{{\{name}}}{{{100 * value:+.2f}}}" for name, value in values.items()
        )
        + "\n",
        encoding="utf-8",
    )
    macro_index = macro.set_index("endpoint")
    four_dataset = macro_index.loc["four_dataset_macro_delta_e"]
    area_psi = macro_index.loc["area_macro_delta_psi"]
    tt_height_psi = macro_index.loc["tt100k_height_macro_delta_psi"]
    positive = int((contrast.delta_e_ci_low > 0).sum())
    negative = int((contrast.delta_e_ci_high < 0).sum())
    crossing = int(len(contrast) - positive - negative)
    minimum = contrast.loc[contrast.delta_e_all.idxmin()]
    maximum = contrast.loc[contrast.delta_e_all.idxmax()]
    dataset_means = contrast.groupby("dataset").delta_e_all.mean()
    narrative = (
        "% Generated from validated direct paired evidence; do not edit manually.\n"
        "\\paragraph{Direct paired INT8--FP8 contrast.} "
        "Using the common image-bootstrap draws, the balanced four-dataset macro "
        f"$\\Delta E$ was {100 * four_dataset.point:+.2f} AP points "
        f"(95\\% percentile interval {100 * four_dataset.ci_low:+.2f} to "
        f"{100 * four_dataset.ci_high:+.2f}). "
        f"Among the 144 condition-level $\\Delta E$ intervals, {positive} were positive, "
        f"{negative} were negative, and {crossing} crossed zero. "
        f"Condition-level point estimates ranged from {100 * minimum.delta_e_all:+.2f} "
        f"AP points ({DISPLAY[minimum.dataset]}, {DISPLAY[minimum.model]}, "
        f"{DISPLAY[minimum.corruption]}, severity {int(minimum.severity)}) to "
        f"{100 * maximum.delta_e_all:+.2f} AP points ({DISPLAY[maximum.dataset]}, "
        f"{DISPLAY[maximum.model]}, {DISPLAY[maximum.corruption]}, severity "
        f"{int(maximum.severity)}). Dataset means opposed one another---for example, "
        f"TT100K was {100 * dataset_means['tt100k']:+.2f} AP points whereas KITTI was "
        f"{100 * dataset_means['kitti']:+.2f}---so the near-zero macro is not evidence "
        "of homogeneous condition effects. These extrema are descriptive regime "
        "indicators, not multiplicity-adjusted discoveries. "
        f"The separate area $\\Delta\\Psi$ macro was {100 * area_psi.point:+.2f} "
        f"({100 * area_psi.ci_low:+.2f} to {100 * area_psi.ci_high:+.2f}) AP points; "
        f"the separately reported TT100K height $\\Delta\\Psi$ macro was "
        f"{100 * tt_height_psi.point:+.2f} "
        f"({100 * tt_height_psi.ci_low:+.2f} to {100 * tt_height_psi.ci_high:+.2f}) AP points. "
        "These image-paired percentile summaries are exploratory and conditional on the "
        "frozen engines, calibration, decoder, and evaluation sets.\n"
    )
    (generated / "direct_results_narrative.tex").write_text(narrative, encoding="utf-8")
    pivot = deployment.pivot(index=["dataset", "model"], columns="precision", values="latency_median_ms")
    int8_speedup = pivot["fp32"] / pivot["int8-entropy"]
    fp8_speedup = pivot["fp32"] / pivot["fp8"]
    fp8_vs_int8 = pivot["int8-entropy"] / pivot["fp8"]
    deployment_narrative = (
        "% Generated from validated RTX 5090 deployment evidence; do not edit manually.\n"
        "Across the 12 matched dataset--model conditions, the median FP32-to-INT8 "
        f"latency ratio was {int8_speedup.median():.2f}$\\times$ "
        f"(range {int8_speedup.min():.2f}--{int8_speedup.max():.2f}$\\times$), and the "
        f"FP32-to-FP8 ratio was {fp8_speedup.median():.2f}$\\times$ "
        f"({fp8_speedup.min():.2f}--{fp8_speedup.max():.2f}$\\times$). FP8 versus INT8 "
        f"had a median INT8-to-FP8 latency ratio of {fp8_vs_int8.median():.2f}$\\times$ but ranged from "
        f"{fp8_vs_int8.min():.2f} to {fp8_vs_int8.max():.2f}$\\times$; FP8 was therefore "
        "not faster in every matched condition.\n"
    )
    (generated / "direct_deployment_narrative.tex").write_text(
        deployment_narrative, encoding="utf-8"
    )
    data_dictionary = {
        "schema_version": 1,
        "direct_format_contrast_cells.csv": {
            "unit": "native AP fraction; multiply by 100 for AP points",
            "ap_columns": [column for column in contrast.columns if column.startswith("delta_")],
        },
        "direct_format_contrast_macro.csv": {
            "unit": "native AP fraction; multiply by 100 for AP points",
            "ap_columns": ["point", "ci_low", "ci_median", "ci_high"],
        },
        "direct_deployment_conditions.csv": {
            "engine_bytes": "bytes",
            "latency_columns": "milliseconds",
            "throughput_qps": "queries per second",
        },
        "direct_absolute_guardrail.csv": {
            "clean_ap_native": "native AP fraction; multiply by 100 for AP points",
            "corrupt_ap_native": "native AP fraction; multiply by 100 for AP points",
            "d_native": "clean AP minus corrupted AP in the native AP fraction",
            "metric_sha256_columns": "SHA-256 of the exact validated clean/corrupted metric records",
        },
        "direct_size_guardrail.csv": {
            "unit": "native AP fraction; multiply by 100 for AP points",
            "endpoint_types": "COCO area for COCO/VOC/KITTI; original-box height for TT100K",
            "metric_sha256_columns": "same exact validated metrics as the overall absolute guardrail",
        },
        "codec_sensitivity.csv": {
            "original_clean_ap": "native AP fraction; multiply by 100 for AP points",
            "codec_clean_ap": "native AP fraction; multiply by 100 for AP points",
            "codec_minus_original": "JPEG-95 minus original-source clean AP in AP points",
        },
        "direct_heterogeneity_sensitivity.csv": {
            "point_native": "native AP fraction; multiply by 100 for AP points",
            "weighting": "equal-cell, image-count, or object-count descriptive weights",
            "leave_one_out": "mean of retained direct cells after the named exclusion",
        },
        "direct_runtime_sensitivity.csv": {
            "ratios": "matched-condition latency ratios; table entries are median [Q1, Q3]",
            "strata": "fixed input size or YOLO11 capacity",
        },
    }
    (generated / "direct_data_dictionary.json").write_text(
        json.dumps(data_dictionary, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    clean_by_block = (
        guardrail.groupby(["dataset", "model", "precision"], as_index=False)
        .clean_ap_native.first()
        .pivot(index=["dataset", "model"], columns="precision", values="clean_ap_native")
    )
    clean_fp8_advantage = float(
        (clean_by_block["fp8"] - clean_by_block["int8-entropy"]).mean()
    )
    abstract = (
        "% Generated from validated direct paired evidence; do not edit manually.\n"
        "Post-training quantization is commonly evaluated on clean images, leaving unclear "
        "whether corruption changes the performance gap between deployment formats. We present "
        "a paired, floor-aware evaluation of executable INT8 and FP8 YOLO11 TensorRT engines. "
        "For each condition, a deterministic JPEG-95 clean control and a corrupted counterpart "
        "use identical encoded images, and the format interaction is measured as "
        "$\\Delta E=(\\mathrm{FP8}-\\mathrm{INT8})_{\\mathrm{corrupt}}-"
        "(\\mathrm{FP8}-\\mathrm{INT8})_{\\mathrm{clean}}$ using common image-bootstrap draws. "
        "The primary grid covers 144 corruption cells spanning COCO, PASCAL VOC, KITTI, and "
        "TT100K; three YOLO11 capacities; four corruptions; and three severities. FP8 exceeded "
        f"INT8 matched-clean AP by {100 * clean_fp8_advantage:.2f} points on average. The balanced "
        f"four-dataset $\\Delta E$ was {100 * four_dataset.point:+.2f} AP points "
        f"(95\\% percentile interval {100 * four_dataset.ci_low:+.2f} to "
        f"{100 * four_dataset.ci_high:+.2f}), indicating no universal change in the format gap. "
        f"Condition effects were heterogeneous, while the hash-matched absolute guardrail found "
        f"median and mean clean-to-corrupted losses of "
        f"{100 * guardrail.d_native.median():.2f} and "
        f"{100 * guardrail.d_native.mean():.2f} AP points across 288 "
        f"format-specific corrupted arms; "
        f"{int((guardrail.corrupt_ap_native < 0.10).sum())} arms had corrupted AP below 10 AP points. "
        "The results show that matched-clean fidelity, absolute corrupted accuracy, and the "
        "format--corruption interaction answer different deployment questions: gap contraction "
        "under a common accuracy floor is not robustness evidence.\n"
    )
    (generated / "direct_abstract.tex").write_text(abstract, encoding="utf-8")
    conclusion = (
        "% Generated from validated direct paired evidence; do not edit manually.\n"
        "The paired excess-gap design makes the direct INT8--FP8 question explicit: it removes "
        "the matched-clean format discrepancy before comparing corrupted inputs and reuses the "
        "same image-bootstrap draws. The balanced four-dataset $\\Delta E$ was "
        f"{100 * four_dataset.point:+.2f} AP points (95\\% percentile interval "
        f"{100 * four_dataset.ci_low:+.2f} to {100 * four_dataset.ci_high:+.2f}), with "
        f"{positive} positive, {negative} negative, and {crossing} crossing condition-level "
        "intervals. Area and TT100K-height $\\Delta\\Psi$ remain separate by construction. "
        f"The 312-record absolute guardrail found median and mean losses of "
        f"{100 * guardrail.d_native.median():.2f} and "
        f"{100 * guardrail.d_native.mean():.2f} AP points across 288 corrupted arms, with "
        f"{int((guardrail.corrupt_ap_native < 0.10).sum())} arms having corrupted AP below 10 AP points. "
        "The mixture of signs and these absolute losses mean that this exploratory evidence "
        "does not establish format superiority or a universal small-object amplification law. "
        "Its contribution is a reproducible, byte-matched and hash-bound measurement protocol "
        "for testing format-specific corruption interaction without conflating it with clean "
        "quantization loss.\n"
    )
    (generated / "direct_conclusion.tex").write_text(conclusion, encoding="utf-8")
    contrast_completion = root / "outputs" / "reports" / f"{FORMAT_CONTRAST_ATTEMPT}_complete.json"
    deployment_completion = root / "outputs" / "reports" / f"{DEPLOYMENT_ATTEMPT}_complete.json"
    direct_generated_paths = (
        generated / "direct_format_contrast_cells.csv",
        generated / "direct_format_contrast_macro.csv",
        generated / "direct_deployment_conditions.csv",
        generated / "direct_absolute_guardrail.csv",
        generated / "direct_format_contrast_summary.tex",
        generated / "direct_heterogeneity_summary.csv",
        generated / "direct_heterogeneity_summary.tex",
        generated / "direct_deployment_summary.tex",
        generated / "direct_absolute_guardrail_summary.tex",
        generated / "direct_absolute_guardrail_narrative.tex",
        generated / "direct_size_guardrail.csv",
        generated / "direct_size_guardrail_summary.tex",
        generated / "codec_sensitivity.csv",
        generated / "codec_sensitivity.tex",
        generated / "direct_heterogeneity_sensitivity.csv",
        generated / "direct_heterogeneity_sensitivity.tex",
        generated / "direct_runtime_sensitivity.csv",
        generated / "direct_runtime_sensitivity.tex",
        generated / "direct_sensitivity_narrative.tex",
        generated / "numbers_direct.tex",
        generated / "direct_results_narrative.tex",
        generated / "direct_deployment_narrative.tex",
        generated / "direct_data_dictionary.json",
        generated / "direct_abstract.tex",
        generated / "direct_conclusion.tex",
        figures / "fig_paired_excess_gap.pdf",
        figures / "fig_paired_excess_gap.png",
    )
    if any(not path.is_file() for path in direct_generated_paths):
        raise RuntimeError("direct evidence generator did not create its complete artifact set")
    audit = {
        "schema_version": 1,
        "status": "valid",
        "scope": "direct paired INT8--FP8 excess-gap evidence and RTX 5090 deployment ledger",
        "counts": {
            "corrupted_paired_components": int(len(contrast)),
            "joint_macro_endpoints": int(len(macro)),
            "deployment_conditions": int(len(deployment)),
            "deployment_repetitions": 108,
            "absolute_guardrail_arms": int(len(guardrail)),
            "absolute_guardrail_metric_records": int(len(guardrail_metric_hashes)),
            "codec_sensitivity_conditions": int(len(codec_sensitivity)),
            "heterogeneity_sensitivity_rows": int(len(heterogeneity_sensitivity)),
            "runtime_sensitivity_rows": int(len(runtime_sensitivity)),
            "size_guardrail_rows": int(len(size_guardrail)),
        },
        "completion_reports_sha256": {
            str(contrast_completion.relative_to(root)): sha256_file(contrast_completion),
            str(deployment_completion.relative_to(root)): sha256_file(deployment_completion),
            **guardrail_completion_hashes,
        },
        "validated_report_hashes": {
            "format_contrast_report_sha256": contrast_report["report_sha256"],
            "deployment_report_sha256": deployment_report["report_sha256"],
        },
        "macro_values_ap": values,
        "absolute_guardrail_metric_files_sha256": guardrail_metric_hashes,
        "generated_artifacts_sha256": {
            str(path.relative_to(root)): sha256_file(path)
            for path in direct_generated_paths
        },
    }
    (generated / "direct_evidence_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--artifact-root",
        type=Path,
        default=Path("."),
    )
    parser.add_argument("--paper-root", type=Path, default=Path("paper"))
    parser.add_argument(
        "--direct-evidence",
        action="store_true",
        help="generate direct paired INT8--FP8 and deployment artifacts after their completion chains validate",
    )
    args = parser.parse_args()

    artifact_root = args.artifact_root.resolve()
    paper_root = args.paper_root.resolve()
    generated = paper_root / "generated"
    figures = paper_root / "figures"

    if args.direct_evidence:
        audit = generate_direct_evidence_artifacts(artifact_root, generated, figures)
        print(json.dumps(audit["counts"], sort_keys=True))
        return

    generated.mkdir(parents=True, exist_ok=True)
    figures.mkdir(parents=True, exist_ok=True)

    metrics, bootstrap, integrity, codec_controls = validate_and_load(artifact_root)
    primary_metrics = replace_original_clean_with_codec(metrics, codec_controls)
    derived = generate_tables(metrics, bootstrap, integrity, codec_controls, generated)
    generate_parity_table(artifact_root, generated)
    derived.update(generate_extended_tables(primary_metrics, bootstrap, derived, generated))
    generate_figures(primary_metrics, bootstrap, derived, figures)
    macro_values = make_macros(primary_metrics, bootstrap, generated / "numbers.tex")
    outputs = artifact_root / "outputs"
    legacy_outputs = artifact_root / LEGACY_ARTIFACT_ROOT / "outputs"
    canonical_metric_paths = sorted(
        [*list((outputs / "metrics" / ATTEMPTS["coco"]).glob("*.json"))]
        + [
            path
            for dataset in ("voc", "kitti", "tt100k")
            for path in (legacy_outputs / "metrics" / ATTEMPTS[dataset]).glob("*.json")
        ]
    )
    codec_metric_paths = sorted((outputs / "metrics" / "codec_control_p0_v1").glob("*.json"))
    canonical_bootstrap_paths = sorted((outputs / "bootstrap" / "matched_codec_p0_v1").glob("*.json"))
    canonical_report_paths = sorted(
        {
            outputs / "reports" / P0_REPORT,
            *(outputs / "reports" / name for name in P0_COMPONENT_REPORTS),
            *(outputs / "reports" / f"coco_{model}_fp16_parity_v1.json" for model in MODEL_ORDER),
            legacy_outputs / "reports" / "four_dataset_pilot_final_report.json",
            *(legacy_outputs / "reports" / EVAL_REPORTS[dataset] for dataset in ("voc", "kitti", "tt100k")),
        }
    )
    source_hashes = {
        "canonical_metrics": {
            str(path.relative_to(artifact_root)): sha256_file(path) for path in canonical_metric_paths
        },
        "canonical_bootstrap": {
            str(path.relative_to(artifact_root)): sha256_file(path) for path in canonical_bootstrap_paths
        },
        "codec_control_metrics": {
            str(path.relative_to(artifact_root)): sha256_file(path) for path in codec_metric_paths
        },
        "canonical_reports": {
            str(path.relative_to(artifact_root)): sha256_file(path) for path in canonical_report_paths
        },
    }
    all_output_json = set(outputs.rglob("*.json"))
    canonical_json = set(
        canonical_metric_paths + codec_metric_paths + canonical_bootstrap_paths + canonical_report_paths
    )
    excluded_json = sorted(all_output_json - canonical_json)
    repo_root = Path(__file__).resolve().parents[1]
    code_hashes = {
        str(path.relative_to(repo_root)): sha256_file(path)
        for base, patterns in (
            (repo_root / "src", ("*.py",)),
            (repo_root / "configs", ("*.json", "*.yaml", "*.yml")),
            (repo_root / "tests", ("*.py",)),
        )
        if base.exists()
        for pattern in patterns
        for path in sorted(base.rglob(pattern))
    }
    audit_sources = [
        *sorted((repo_root / "analysis").glob("*.py")),
        repo_root / "pytest.ini",
        repo_root / "paper" / "Makefile",
        repo_root / "paper" / "main.tex",
        repo_root / "paper" / "highlights.txt",
        repo_root / "paper" / "references.bib",
    ]
    for path in audit_sources:
        if path.is_file():
            code_hashes[str(path.relative_to(repo_root))] = sha256_file(path)
    audit = {
        "schema_version": 2,
        "status": "valid",
        "scope": "P0 matched-codec four-dataset study; exploratory B=500 percentile intervals",
        "counts": {
            "datasets": 4,
            "metric_records": int(len(metrics)),
            "codec_control_records": int(len(codec_controls)),
            "bootstrap_cells": int(len(bootstrap)),
            "bootstrap_replicates_per_cell": 500,
            "canonical_report_files": int(len(canonical_report_paths)),
            "excluded_noncanonical_output_json": int(len(excluded_json)),
        },
        "tt100k_height_endpoint_non_degenerate": True,
        "audit_exceptions": {
            "historical_engine_build_config_unarchived": {
                "status": "accepted_narrow_exception",
                "historical_ladder_config_sha256": HISTORICAL_LADDER_CONFIG_SHA256,
                "final_p0_config_sha256": FINAL_P0_CONFIG_SHA256,
                "limitation": "The byte-identical historical config file used for the COCO ladder was not archived; the master-bound ladder report and every calibration/ONNX/engine artifact binding validate.",
            }
        },
        "macro_values_ap": macro_values,
        "source_hashes": source_hashes,
        "excluded_artifacts": {
            "reason": "not referenced by the validated four-dataset canonical completion chain",
            "json_paths": [str(path.relative_to(artifact_root)) for path in excluded_json],
        },
        "code_config_test_hashes": code_hashes,
    }
    (generated / "artifact_audit.json").write_text(
        json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    print(json.dumps(audit["counts"], sort_keys=True))


if __name__ == "__main__":
    main()
