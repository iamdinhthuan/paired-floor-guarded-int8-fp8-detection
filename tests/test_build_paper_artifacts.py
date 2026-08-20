from __future__ import annotations

import copy
import hashlib
import inspect
import json
from itertools import product
from pathlib import Path
import shlex
import sys

import pytest
import pandas as pd
from PIL import Image


PROJECT_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))

import build_paper_artifacts  # noqa: E402


validate_and_load = build_paper_artifacts.validate_and_load


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def _canonical_sha(document: dict[str, object], field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True)
    return hashlib.sha256(raw.encode("utf-8")).hexdigest()


def _write_json(path: Path, document: dict[str, object]) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return _sha256(path)


def _write_bytes(path: Path, payload: bytes) -> str:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_bytes(payload)
    return _sha256(path)


def _format_contrast_fixture(root: Path) -> Path:
    attempt = "ivc_format_contrast_v1"
    datasets = ("coco", "voc", "kitti", "tt100k")
    models = ("yolo11n", "yolo11m", "yolo11x")
    corruptions = ("gaussian_noise", "motion_blur", "fog", "jpeg")
    severities = (1, 3, 5)
    expected_images = {"coco": 5000, "voc": 4952, "kitti": 1496, "tt100k": 3067}
    corruption_attempts = {
        "coco": "coco_uniform_p0_v1",
        "voc": "voc_pilot_117_v1",
        "kitti": "kitti_pilot_117_v1",
        "tt100k": "tt100k_pilot_117_v1",
    }
    config = {
        "schema_version": 1,
        "attempt": attempt,
        "n_boot": 2000,
        "bootstrap_workers": 1,
        "cell_workers": 4,
        "seed_namespace": "ivc-format-contrast-v1-test",
        "models": list(models),
        "precisions": ["int8-entropy", "fp8"],
        "corruptions": list(corruptions),
        "severities": list(severities),
        "datasets": [
            {
                "dataset": dataset,
                "annotations": str(root / "annotations" / f"{dataset}.json"),
                "expected_images": expected_images[dataset],
                "clean_source_attempt": "codec_control_p0_v1",
                "clean_corruption": "codec_control",
                "corruption_source_attempt": corruption_attempts[dataset],
            }
            for dataset in datasets
        ],
    }
    config_path = root / "configs" / f"{attempt}.json"
    config_sha = _write_json(config_path, config)

    run_witness = root / "manifests" / "runs" / "fixture.json"
    run_sha = _write_json(run_witness, {"schema_version": 1, "fixture": True})
    schedules: dict[str, dict[str, object]] = {}
    supporting: dict[str, str] = {}
    for dataset in datasets:
        relative = f"outputs/work/{attempt}/schedules/{dataset}.npz"
        digest = _write_bytes(root / relative, f"schedule:{dataset}".encode())
        schedules[dataset] = {
            "path": relative,
            "sha256": digest,
            "n_boot": 2000,
            "seed": int.from_bytes(
                hashlib.sha256(
                    f"{config['seed_namespace']}|paired-format-dataset|{dataset}".encode()
                ).digest()[:8],
                "big",
            )
            % (2**32),
            "n_images": expected_images[dataset],
            "image_ids_sha256": hashlib.sha256(dataset.encode()).hexdigest(),
        }
        supporting[relative] = digest

    clean_caches: dict[str, dict[str, str]] = {}
    for dataset, model in product(datasets, models):
        relative = f"outputs/work/{attempt}/clean_arm_aps/{dataset}__{model}.npz"
        digest = _write_bytes(root / relative, f"clean:{dataset}:{model}".encode())
        clean_caches[relative] = {"sha256": digest, "identity_sha256": "c" * 64}
        supporting[relative] = digest

    component_hashes: dict[str, str] = {}
    component_documents: dict[str, dict[str, object]] = {}
    draw_caches: dict[str, dict[str, str]] = {}
    clean_relatives: list[str] = []
    corrupted_relatives: list[str] = []
    for dataset, model in product(datasets, models):
        conditions = [("clean", 0), *product(corruptions, severities)]
        for corruption, severity in conditions:
            relative = (
                f"outputs/bootstrap/{attempt}/"
                f"{dataset}__{model}__{corruption}-s{severity}.json"
            )
            is_clean = corruption == "clean"
            (clean_relatives if is_clean else corrupted_relatives).append(relative)
            schedule = schedules[dataset]
            image_sha = str(schedule["image_ids_sha256"])
            endpoint_labels = (
                ("all", "small_like", "large_like")
                if dataset == "tt100k"
                else ("all", "small", "medium", "large")
            )
            input_hashes = {
                arm: {
                    "prediction_sha256": hashlib.sha256(f"pred:{arm}".encode()).hexdigest(),
                    "input_record_sha256": hashlib.sha256(f"input:{arm}".encode()).hexdigest(),
                    "input_manifest_sha256": "a" * 64,
                    "image_ids_sha256": image_sha,
                    "run_record_sha256": run_sha,
                }
                for arm in ("int8_clean", "fp8_clean", "int8_corrupt", "fp8_corrupt")
            }
            if is_clean:
                input_hashes["int8_corrupt"] = dict(input_hashes["int8_clean"])
                input_hashes["fp8_corrupt"] = dict(input_hashes["fp8_clean"])
            clean_path = (
                f"outputs/work/{attempt}/clean_arm_aps/{dataset}__{model}.npz"
            )
            document: dict[str, object] = {
                "schema_version": 1,
                "method": "paired image bootstrap; one shared resample across INT8/FP8 × clean/corrupt",
                "endpoint_type": "tt100k-height" if dataset == "tt100k" else "area",
                "annotation": {"path": str(root / "annotations" / f"{dataset}.json"), "sha256": "b" * 64},
                "n_images": expected_images[dataset],
                "n_boot": 2000,
                "seed": schedule["seed"],
                "point": {
                    "delta_q": {label: 0.01 for label in endpoint_labels},
                    "delta_e": {label: 0.0 if is_clean else 0.02 for label in endpoint_labels},
                    "delta_psi": 0.0,
                },
                "percentile_intervals": {
                    "percentiles": [2.5, 50.0, 97.5],
                    "delta_q": {label: [0.0, 0.01, 0.02] for label in endpoint_labels},
                    "delta_e": {
                        label: ([0.0, 0.0, 0.0] if is_clean else [0.01, 0.02, 0.03])
                        for label in endpoint_labels
                    },
                    "delta_psi": [0.0, 0.0, 0.0] if is_clean else [-0.01, 0.0, 0.01],
                },
                "input_hashes": input_hashes,
                "bootstrap_schedule": schedule,
                "clean_arm_cache": {"path": clean_path, **clean_caches[clean_path]},
                "sign_convention": {
                    "delta_q": "AP(fp8, clean) - AP(int8, clean); positive means INT8 has lower clean AP",
                    "delta_e": "[AP(int8, clean)-AP(int8, corrupt)] - [AP(fp8, clean)-AP(fp8, corrupt)]; positive means INT8 amplifies corruption more",
                    "delta_psi": "delta_e(small) - delta_e(large), or small_like - large_like for TT100K height",
                },
            }
            if dataset == "tt100k":
                document["height_bins_px"] = {
                    "small_like": {"min": 0.0, "max": 24.0},
                    "large_like": {"min": 48.0, "max": None},
                }
            if not is_clean:
                draw_path = (
                    f"outputs/work/{attempt}/format_contrast_draws/"
                    f"{Path(relative).stem}.npz"
                )
                draw_sha = _write_bytes(root / draw_path, f"draw:{relative}".encode())
                draw = {"path": draw_path, "sha256": draw_sha, "identity_sha256": "d" * 64}
                document["temporary_draw_cache"] = draw
                draw_caches[relative] = draw
                supporting[draw_path] = draw_sha
            document["artifact_sha256"] = _canonical_sha(document, "artifact_sha256")
            component_hashes[relative] = _write_json(root / relative, document)
            component_documents[relative] = document

    macro_relative = f"outputs/bootstrap/{attempt}/{attempt}_joint_macro.json"
    macro = {
        "schema_version": 1,
        "method": "joint paired image bootstrap; one shared resample per dataset and replicate across all models, corruptions, severities, and formats",
        "n_boot": 2000,
        "seed_namespace": config["seed_namespace"],
        "component_cells": 144,
        "component_artifacts": {
            relative: {
                "artifact_sha256": component_hashes[relative],
                "annotation_sha256": component_documents[relative]["annotation"]["sha256"],
                "input_hashes": component_documents[relative]["input_hashes"],
                "bootstrap_schedule": component_documents[relative]["bootstrap_schedule"],
                "clean_arm_cache": component_documents[relative]["clean_arm_cache"],
                "temporary_draw_cache": component_documents[relative]["temporary_draw_cache"],
            }
            for relative in corrupted_relatives
        },
        "point": {
            "four_dataset_macro_delta_e": 0.02,
            "area_macro_delta_psi": 0.0,
            "tt100k_height_macro_delta_psi": 0.0,
        },
        "percentile_intervals": {
            "four_dataset_macro_delta_e": [0.01, 0.02, 0.03],
            "area_macro_delta_psi": [-0.01, 0.0, 0.01],
            "tt100k_height_macro_delta_psi": [-0.01, 0.0, 0.01],
        },
        "coverage": {
            "four_dataset_macro_delta_e": {"planned_component_count": 144, "finite_complete_replicates": 2000, "total_replicates": 2000},
            "area_macro_delta_psi": {"planned_component_count": 108, "finite_complete_replicates": 2000, "total_replicates": 2000},
            "tt100k_height_macro_delta_psi": {"planned_component_count": 36, "finite_complete_replicates": 2000, "total_replicates": 2000},
        },
        "annotations_sha256": {dataset: "b" * 64 for dataset in datasets},
        "bootstrap_schedules": schedules,
    }
    macro["artifact_sha256"] = _canonical_sha(macro, "artifact_sha256")
    macro_file_sha = _write_json(root / macro_relative, macro)
    artifacts = {**component_hashes, macro_relative: macro_file_sha}

    package_relative = f"outputs/reports/{attempt}_execution_package.json"
    package_files = (
        f"configs/{attempt}.json",
        "src/run_format_contrast_p0.py",
        "src/format_contrast_scheduler.py",
        "src/bootstrap_format_contrast.py",
        "src/bootstrap_format_contrast_macro.py",
        "src/validate_format_contrast_evidence.py",
        "src/paired_bootstrap.py",
        "src/topic_c/manifest.py",
        "src/topic_c/tt100k_height.py",
    )
    package_hashes = {package_files[0]: config_sha}
    for relative in package_files[1:]:
        package_hashes[relative] = _write_bytes(root / relative, relative.encode())
    package = {
        "schema_version": 1,
        "attempt": attempt,
        "execution_root": str(root),
        "files_sha256": package_hashes,
    }
    package["manifest_sha256"] = _canonical_sha(package, "manifest_sha256")
    package_file_sha = _write_json(root / package_relative, package)
    package_binding = {
        "path": package_relative,
        "sha256": package_file_sha,
        "manifest_sha256": package["manifest_sha256"],
    }
    ledgers: dict[str, dict[str, str]] = {}
    for phase, relatives in (("clean", clean_relatives), ("corrupt", corrupted_relatives)):
        ledger_relative = f"outputs/reports/{attempt}_{phase}_scheduler.json"
        records: list[dict[str, object]] = []
        for index, relative in enumerate(relatives, start=1):
            log_relative = (
                f"outputs/logs/{attempt}/format_contrast_cells/"
                f"{phase}__{Path(relative).stem}.log"
            )
            log_sha = _write_bytes(root / log_relative, f"log:{phase}:{relative}".encode())
            component = component_documents[relative]
            auxiliary = (
                component["clean_arm_cache"]["path"]
                if phase == "clean"
                else component["temporary_draw_cache"]["path"]
            )
            output_paths = [relative, auxiliary]
            dataset, model = Path(relative).stem.split("__")[:2]
            if phase == "clean":
                command = [
                    str(root / "python"),
                    str(root / "src" / "run_format_contrast_p0.py"),
                    "--project-root",
                    str(root),
                    "--config",
                    str(config_path),
                    "--execute-clean-cell",
                    "--dataset",
                    dataset,
                    "--model",
                    model,
                ]
            else:
                command = [
                    str(root / "python"),
                    str(root / "src" / "bootstrap_format_contrast.py"),
                    "--endpoint",
                    component["endpoint_type"],
                    "--annotations",
                    component["annotation"]["path"],
                    "--expected-images",
                    str(component["n_images"]),
                    "--annotation-sha256",
                    component["annotation"]["sha256"],
                ]
                for arm in ("int8_clean", "fp8_clean", "int8_corrupt", "fp8_corrupt"):
                    option = "--" + arm.replace("_", "-")
                    command.extend(
                        [
                            option,
                            str(root / "inputs" / f"{arm}.json"),
                            option + "-input",
                            str(root / "inputs" / f"{arm}-input.json"),
                            option + "-run",
                            str(root / "manifests" / "runs" / "fixture.json"),
                        ]
                    )
                command.extend(
                    [
                        "--n-boot",
                        "2000",
                        "--seed",
                        str(component["seed"]),
                        "--workers",
                        "1",
                        "--out",
                        str(root / relative),
                        "--evidence-root",
                        str(root),
                        "--draw-cache",
                        str(root / component["temporary_draw_cache"]["path"]),
                        "--schedule",
                        str(root / component["bootstrap_schedule"]["path"]),
                        "--clean-arm-cache",
                        str(root / component["clean_arm_cache"]["path"]),
                        "--clean-arm-cache-sha256",
                        component["clean_arm_cache"]["sha256"],
                        "--clean-arm-cache-identity-sha256",
                        component["clean_arm_cache"]["identity_sha256"],
                        "--dataset",
                        dataset,
                        "--model",
                        model,
                    ]
                )
            records.append(
                {
                    "key": f"{phase}:{Path(relative).stem}",
                    "command": command,
                    "pid": index,
                    "log_path": log_relative,
                    "output_paths": output_paths,
                    "started_at_utc": "2026-08-12T00:00:00+00:00",
                    "ended_at_utc": "2026-08-12T00:01:00+00:00",
                    "exit_status": 0,
                    "verification": {"status": "valid"},
                    "missing_output_paths": [],
                    "output_sha256": {path: _sha256(root / path) for path in output_paths},
                    "log_sha256": log_sha,
                }
            )
        ledger = {
            "schema_version": 1,
            "scheduler": "bounded independent immutable format-contrast cells",
            "execution_package": package_binding,
            "cell_workers": 4,
            "peak_active": 4,
            "status": "complete",
            "failure": None,
            "records": records,
        }
        ledger["ledger_sha256"] = _canonical_sha(ledger, "ledger_sha256")
        ledger_file_sha = _write_json(root / ledger_relative, ledger)
        ledgers[phase] = {
            "path": ledger_relative,
            "sha256": ledger_file_sha,
            "ledger_sha256": ledger["ledger_sha256"],
        }

    report = {
        "schema_version": 1,
        "completed_at_utc": "2026-08-12T00:00:00+00:00",
        "attempt": attempt,
        "clean_contrasts": 12,
        "corrupted_contrasts": 144,
        "n_boot": 2000,
        "config": str(config_path),
        "config_sha256": config_sha,
        "execution_package": package_binding,
        "scheduler_ledgers": ledgers,
        "component_artifacts_sha256": component_hashes,
        "joint_macro_artifact_sha256": macro_file_sha,
        "artifacts_sha256": artifacts,
        "bootstrap_schedules": schedules,
        "clean_arm_caches": clean_caches,
        "draw_caches": draw_caches,
        "supporting_evidence_sha256": supporting,
        "evidence_files_sha256": {**artifacts, **supporting},
        "annotations_sha256": {dataset: "b" * 64 for dataset in datasets},
    }
    report["report_sha256"] = _canonical_sha(report, "report_sha256")
    report_path = root / "outputs" / "reports" / f"{attempt}_complete.json"
    _write_json(report_path, report)
    return report_path


def _deployment_fixture(root: Path) -> Path:
    attempt = "ivc_deployment_benchmark_v1"
    datasets = ("coco", "voc", "kitti", "tt100k")
    models = ("yolo11n", "yolo11m", "yolo11x")
    precisions = ("fp32", "int8-entropy", "fp8")
    benchmark_relative = f"outputs/benchmarks/{attempt}"
    treatment_relative = f"{benchmark_relative}/engine_treatments.json"
    report_relative = f"outputs/reports/{attempt}_complete.json"
    trtexec = root / "runtime" / "trtexec"
    config = {
        "schema_version": 1,
        "attempt": attempt,
        "models": list(models),
        "precisions": list(precisions),
        "repetitions": 3,
        "warmup_ms": 5000,
        "duration_s": 30,
        "required_gpu_name": "NVIDIA GeForce RTX 5090",
        "required_hostname": "fixture-host",
        "required_project_root": str(root),
        "required_engine_root": str(root / "engines"),
        "required_trtexec": str(trtexec),
        "output_dir": benchmark_relative,
        "report": report_relative,
        "engine_treatments": treatment_relative,
        "registry_sets": [
            {
                "dataset": dataset,
                "ladder_registry": str(root / "manifests" / f"{dataset}_ladder.json"),
                "engine_registry_dir": str(root / "manifests"),
                "onnx_registry_dir": str(root / "manifests"),
            }
            for dataset in datasets
        ],
    }
    config_path = root / "configs" / f"{attempt}.json"
    config_sha = _write_json(config_path, config)
    trtexec_sha = hashlib.sha256(b"trtexec").hexdigest()
    runtime_identity = {
        "hostname": "fixture-host",
        "project_root": str(root),
        "platform": "Linux-fixture",
        "python_version": "3.11.0",
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "driver_version": "590.1",
        "cuda_version": "13.0",
        "tensorrt_python_version": "11.1.0.106",
        "trtexec_version_output": "TensorRT version 11.1.0.106",
        "trtexec_executable": str(trtexec),
        "trtexec_sha256": trtexec_sha,
    }
    treatment_records: list[dict[str, object]] = []
    records_sha: dict[str, str] = {}
    logs_sha: dict[str, str] = {}
    for condition_index, (dataset, model, precision) in enumerate(
        product(datasets, models, precisions), start=1
    ):
        condition_id = f"{dataset}__{model}__{precision}"
        engine = root / "engines" / f"{condition_id}.engine"
        engine_sha = hashlib.sha256(condition_id.encode()).hexdigest()
        onnx_sha = hashlib.sha256(f"onnx:{condition_id}".encode()).hexdigest()
        treatment_records.append(
            {
                "dataset": dataset,
                "model": model,
                "precision": precision,
                "onnx_sha256": onnx_sha,
                "engine_sha256": engine_sha,
                "engine_bytes": 1000 + condition_index,
                "quantize_mode": "unrecorded",
                "calibration_list": "unrecorded",
                "calibration_sha256": "unrecorded",
                "calibration_method": "unrecorded",
                "calibration_eps": "unrecorded",
                "op_types_to_exclude": "unrecorded",
                "modelopt_version": "unrecorded",
                "onnx_version": "unrecorded",
                "onnx_command": "unrecorded",
                "tensorrt_python_version": "11.1.0.106",
                "trtexec_sha256": trtexec_sha,
                "engine_command": ["fixture"],
                "fp8_encoding": "unrecorded",
                "fp8_granularity": "unrecorded",
            }
        )
        for repetition in (1, 2, 3):
            stem = f"{condition_id}__rep-{repetition:02d}"
            command = [str(trtexec), f"--loadEngine={engine}", "--warmUp=5000", "--duration=30"]
            throughput = float(300 + condition_index + repetition / 10)
            raw = (
                f"&&&& RUNNING TensorRT.trtexec [TensorRT v11.1.0.106] [b48] # {shlex.join(command)}\n"
                "[08/12/2026-12:34:56] [I] === Performance summary ===\n"
                f"[08/12/2026-12:34:56] [I] Throughput: {throughput:.2f} qps\n"
                "[08/12/2026-12:34:56] [I] Latency: min = 1.20 ms, max = 2.50 ms, mean = 1.60 ms, "
                "median = 1.55 ms, percentile(90%) = 1.90 ms, "
                "percentile(95%) = 2.00 ms, percentile(99%) = 2.10 ms\n"
                f"&&&& PASSED TensorRT.trtexec [TensorRT v11.1.0.106] [b48] # {shlex.join(command)}\n"
            )
            raw_sha = hashlib.sha256(raw.encode()).hexdigest()
            run_id = hashlib.md5(stem.encode()).hexdigest()
            metadata = {
                "schema_version": 1,
                "log_format": "ivc_trtexec_log_v1",
                "attempt": attempt,
                "config_sha256": config_sha,
                "condition_id": condition_id,
                "dataset": dataset,
                "model": model,
                "precision": precision,
                "repetition": repetition,
                "run_id": run_id,
                "started_at_utc": "2026-08-12T00:00:00+00:00",
                "ended_at_utc": "2026-08-12T00:01:00+00:00",
                "return_code": 0,
                "engine_sha256": engine_sha,
                "trtexec_command": command,
                "raw_output_bytes": len(raw.encode()),
                "raw_output_sha256": raw_sha,
            }
            header = json.dumps(metadata, sort_keys=True, separators=(",", ":"))
            log_path = root / benchmark_relative / "logs" / f"{stem}.log"
            log_sha = _write_bytes(log_path, f"IVC_TRTEXEC_LOG_V1 {header}\n{raw}".encode())
            record = {
                "schema_version": 1,
                "attempt": attempt,
                "config_sha256": config_sha,
                "condition_id": condition_id,
                "dataset": dataset,
                "model": model,
                "precision": precision,
                "repetition": repetition,
                "run_id": run_id,
                "log_format": "ivc_trtexec_log_v1",
                "trtexec_return_code": 0,
                "raw_output_sha256": raw_sha,
                "started_at_utc": metadata["started_at_utc"],
                "ended_at_utc": metadata["ended_at_utc"],
                "engine": str(engine),
                "engine_sha256": engine_sha,
                "engine_bytes": 1000 + condition_index,
                "ladder_registry": str(root / "manifests" / f"{dataset}_ladder.json"),
                "ladder_registry_sha256": "1" * 64,
                "engine_registry": str(root / "manifests" / f"{condition_id}_engine.json"),
                "engine_registry_sha256": "2" * 64,
                "onnx_registry": str(root / "manifests" / f"{condition_id}_onnx.json"),
                "onnx_registry_sha256": "3" * 64,
                "input_binding_registry": str(root / "manifests" / f"{dataset}__{model}_fp32.json"),
                "input_binding_registry_sha256": "4" * 64,
                "input_onnx": str(root / "onnx" / f"{dataset}__{model}_fp32.onnx"),
                "input_onnx_sha256": onnx_sha,
                "input_name": "images",
                "input_shape": [1, 3, 640, 640],
                "input_dynamic": False,
                "shape_flag_required": False,
                "trtexec": str(trtexec),
                "trtexec_sha256": trtexec_sha,
                "gpu_runtime_identity": runtime_identity,
                "gpu_idle_query_command": ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"],
                "gpu_idle_query_output": "",
                "trtexec_command": command,
                "log_sha256": log_sha,
                "throughput_qps": throughput,
                "latency_mean_ms": 1.60,
                "latency_median_ms": 1.55,
                "latency_p99_ms": 2.10,
            }
            record_path = root / benchmark_relative / f"{stem}.json"
            records_sha[f"{stem}.json"] = _write_json(record_path, record)
            logs_sha[f"{stem}.log"] = log_sha

    treatment = {
        "schema_version": 1,
        "attempt": attempt,
        "records": treatment_records,
        "engine_conditions": 36,
    }
    treatment["treatments_sha256"] = _canonical_sha(treatment, "treatments_sha256")
    treatment_file_sha = _write_json(root / treatment_relative, treatment)
    report = {
        "schema_version": 1,
        "attempt": attempt,
        "created_at_utc": "2026-08-12T00:02:00+00:00",
        "config_sha256": config_sha,
        "engine_conditions": 36,
        "repetitions_per_engine": 3,
        "repetition_records": 108,
        "raw_logs": 108,
        "records_sha256": records_sha,
        "logs_sha256": logs_sha,
        "limitations": ["No memory claim.", "No power or energy claim."],
        "engine_treatments_sha256": treatment_file_sha,
    }
    report["report_sha256"] = _canonical_sha(report, "report_sha256")
    report_path = root / report_relative
    _write_json(report_path, report)
    return report_path


def test_primary_loader_uses_complete_p0_evidence_grid() -> None:
    metrics, bootstrap, integrity, codec_controls = validate_and_load(PROJECT_ROOT)

    assert len(metrics) == 468
    assert len(codec_controls) == 36
    assert len(bootstrap) == 288
    assert bootstrap.groupby("dataset").size().to_dict() == {
        "coco": 72,
        "kitti": 72,
        "tt100k": 72,
        "voc": 72,
    }
    assert bootstrap.n_boot.eq(500).all()

    coco_clean = metrics.query(
        "dataset == 'coco' and model == 'yolo11n' and "
        "precision == 'fp32' and corruption == 'clean' and severity == 0"
    ).iloc[0]
    assert coco_clean.AP == pytest.approx(0.3873194381840302)

    matched_cell = bootstrap.query(
        "dataset == 'coco' and model == 'yolo11n' and "
        "precision == 'int8-entropy' and corruption == 'gaussian_noise' and severity == 1"
    ).iloc[0]
    assert matched_cell.q_clean_all == pytest.approx(0.020435754364369785)

    assert integrity["Bootstrap cells"].eq(72).all()


def test_codec_sensitivity_reports_signed_ap_point_difference(tmp_path: Path) -> None:
    metrics, _, _, codec_controls = validate_and_load(PROJECT_ROOT)

    sensitivity = build_paper_artifacts.generate_codec_sensitivity(
        metrics, codec_controls, tmp_path
    )

    assert len(sensitivity) == 36
    coco_n_fp32 = sensitivity.query(
        "dataset == 'coco' and model == 'yolo11n' and precision == 'fp32'"
    ).iloc[0]
    assert coco_n_fp32.original_clean_ap == pytest.approx(0.3873194381840302)
    assert coco_n_fp32.codec_clean_ap == pytest.approx(0.38696779143553606)
    assert coco_n_fp32.codec_minus_original == pytest.approx(-0.03516467484941499)
    assert (tmp_path / "codec_sensitivity.csv").is_file()
    assert (tmp_path / "codec_sensitivity.tex").is_file()


def test_only_documented_historical_ladder_config_hash_is_accepted() -> None:
    with pytest.raises(RuntimeError, match="unrecognized historical ladder config"):
        build_paper_artifacts.validate_historical_ladder_config(
            ladder_config_sha256="0" * 64,
            current_config_sha256="ab044caa9bd85622eeb291dc2c04d0ee82c787b69c0d9fc01c121344f03e2c15",
            master_config_sha256="ab044caa9bd85622eeb291dc2c04d0ee82c787b69c0d9fc01c121344f03e2c15",
        )


def test_semantically_incomplete_matched_codec_grid_fails_closed() -> None:
    metrics, bootstrap, _, codec_controls = validate_and_load(PROJECT_ROOT)
    malformed = bootstrap.copy()
    semantic_columns = ["dataset", "model", "precision", "corruption", "severity"]
    malformed.loc[malformed.index[0], semantic_columns] = malformed.loc[
        malformed.index[1], semantic_columns
    ].to_numpy()

    with pytest.raises(RuntimeError, match="matched-codec semantic grid is incomplete"):
        build_paper_artifacts.validate_primary_grids(metrics, malformed, codec_controls)


def test_primary_loader_exposes_overall_area_excess_gap_intervals() -> None:
    _, bootstrap, _, _ = validate_and_load(PROJECT_ROOT)

    interval_columns = {"e_all_ci_low", "e_all_ci_median", "e_all_ci_high"}
    assert interval_columns <= set(bootstrap.columns)

    area = bootstrap.query("dataset != 'tt100k'")
    tt100k = bootstrap.query("dataset == 'tt100k'")
    assert area[list(interval_columns)].notna().all().all()
    assert tt100k[list(interval_columns)].isna().all().all()

    assert int((area.e_all_ci_low > 0).sum()) == 38
    assert int((area.e_all_ci_high < 0).sum()) == 45
    assert int(
        ((area.e_all_ci_low <= 0) & (area.e_all_ci_high >= 0)).sum()
    ) == 133


def test_generated_macros_include_overall_e_and_floor_diagnostics(
    tmp_path: Path,
) -> None:
    metrics, bootstrap, _, codec_controls = validate_and_load(PROJECT_ROOT)
    primary_metrics = build_paper_artifacts.replace_original_clean_with_codec(
        metrics, codec_controls
    )

    macro_path = tmp_path / "numbers.tex"
    build_paper_artifacts.make_macros(primary_metrics, bootstrap, macro_path)
    macros = macro_path.read_text(encoding="utf-8")

    for expected in (
        r"\newcommand{\AreaECIPositive}{38}",
        r"\newcommand{\AreaECINegative}{45}",
        r"\newcommand{\AreaECICrossing}{133}",
        r"\newcommand{\AreaBelowFiveAPNegativeE}{6}",
        r"\newcommand{\AreaBelowFiveAPCells}{6}",
        r"\newcommand{\AreaBelowTenAPNegativeE}{17}",
        r"\newcommand{\AreaBelowTenAPCells}{20}",
    ):
        assert expected in macros


def test_interaction_table_headers_distinguish_area_and_tt100k_height(
    tmp_path: Path,
) -> None:
    metrics, bootstrap, integrity, codec_controls = validate_and_load(PROJECT_ROOT)

    build_paper_artifacts.generate_tables(
        metrics, bootstrap, integrity, codec_controls, tmp_path
    )
    table = (tmp_path / "interaction_summary.tex").read_text(encoding="utf-8")

    assert "small area / small-like height" in table
    assert "large area / large-like height" in table


def test_format_contrast_validator_accepts_exact_complete_synthetic_chain(
    tmp_path: Path,
) -> None:
    _format_contrast_fixture(tmp_path)

    report = build_paper_artifacts.validate_format_contrast(tmp_path)

    assert report["clean_contrasts"] == 12
    assert report["corrupted_contrasts"] == 144
    assert report["n_boot"] == 2000
    assert len(report["component_artifacts_sha256"]) == 156
    assert len(report["artifacts_sha256"]) == 157
    assert len(report["evidence_files_sha256"]) == 317


def test_format_contrast_validator_rejects_mutated_condition_grid(tmp_path: Path) -> None:
    report_path = _format_contrast_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    old = next(iter(report["component_artifacts_sha256"]))
    forged = old.replace("coco__yolo11n__clean-s0", "coco__yolo11n__clean-s1")
    digest = report["component_artifacts_sha256"].pop(old)
    report["component_artifacts_sha256"][forged] = digest
    report["report_sha256"] = _canonical_sha(report, "report_sha256")
    _write_json(report_path, report)

    with pytest.raises(RuntimeError, match="component grid"):
        build_paper_artifacts.validate_format_contrast(tmp_path)


def test_format_contrast_validator_rejects_contrast_file_sha_mutation(tmp_path: Path) -> None:
    report_path = _format_contrast_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    relative = next(iter(report["component_artifacts_sha256"]))
    (tmp_path / relative).write_text("{}\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        build_paper_artifacts.validate_format_contrast(tmp_path)


def test_format_contrast_validator_rejects_traversal_in_bound_path(tmp_path: Path) -> None:
    report_path = _format_contrast_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    report["execution_package"]["path"] = "../execution-package.json"
    report["report_sha256"] = _canonical_sha(report, "report_sha256")
    _write_json(report_path, report)

    with pytest.raises(RuntimeError, match="safe relative path"):
        build_paper_artifacts.validate_format_contrast(tmp_path)


def test_format_contrast_validator_rejects_scheduler_log_sha_mutation(tmp_path: Path) -> None:
    report_path = _format_contrast_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    ledger_path = tmp_path / report["scheduler_ledgers"]["clean"]["path"]
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    log_path = tmp_path / ledger["records"][0]["log_path"]
    log_path.write_bytes(b"mutated scheduler log")

    with pytest.raises(RuntimeError, match="scheduler task log SHA-256 mismatch"):
        build_paper_artifacts.validate_format_contrast(tmp_path)


def test_format_contrast_validator_rejects_rehashed_macro_input_binding(tmp_path: Path) -> None:
    report_path = _format_contrast_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    macro_relative = next(
        relative for relative in report["artifacts_sha256"] if relative.endswith("_joint_macro.json")
    )
    macro_path = tmp_path / macro_relative
    macro = json.loads(macro_path.read_text(encoding="utf-8"))
    component = next(iter(macro["component_artifacts"].values()))
    component["input_hashes"]["int8_clean"]["prediction_sha256"] = "f" * 64
    macro["artifact_sha256"] = _canonical_sha(macro, "artifact_sha256")
    macro_file_sha = _write_json(macro_path, macro)
    report["joint_macro_artifact_sha256"] = macro_file_sha
    report["artifacts_sha256"][macro_relative] = macro_file_sha
    report["evidence_files_sha256"][macro_relative] = macro_file_sha
    report["report_sha256"] = _canonical_sha(report, "report_sha256")
    _write_json(report_path, report)

    with pytest.raises(RuntimeError, match="joint macro binding"):
        build_paper_artifacts.validate_format_contrast(tmp_path)


def test_format_contrast_validator_rejects_rehashed_scheduler_command(tmp_path: Path) -> None:
    report_path = _format_contrast_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    binding = report["scheduler_ledgers"]["clean"]
    ledger_path = tmp_path / binding["path"]
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    ledger["records"][0]["command"] = [str(tmp_path / "forged-runner")]
    ledger["ledger_sha256"] = _canonical_sha(ledger, "ledger_sha256")
    binding["ledger_sha256"] = ledger["ledger_sha256"]
    binding["sha256"] = _write_json(ledger_path, ledger)
    report["report_sha256"] = _canonical_sha(report, "report_sha256")
    _write_json(report_path, report)

    with pytest.raises(RuntimeError, match="scheduler command"):
        build_paper_artifacts.validate_format_contrast(tmp_path)


def test_format_contrast_validator_rejects_rehashed_nonnumeric_macro(tmp_path: Path) -> None:
    report_path = _format_contrast_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    macro_relative = next(
        relative for relative in report["artifacts_sha256"] if relative.endswith("_joint_macro.json")
    )
    macro_path = tmp_path / macro_relative
    macro = json.loads(macro_path.read_text(encoding="utf-8"))
    macro["point"]["four_dataset_macro_delta_e"] = "fabricated"
    macro["percentile_intervals"]["four_dataset_macro_delta_e"] = ["fabricated"]
    macro["artifact_sha256"] = _canonical_sha(macro, "artifact_sha256")
    macro_file_sha = _write_json(macro_path, macro)
    report["joint_macro_artifact_sha256"] = macro_file_sha
    report["artifacts_sha256"][macro_relative] = macro_file_sha
    report["evidence_files_sha256"][macro_relative] = macro_file_sha
    report["report_sha256"] = _canonical_sha(report, "report_sha256")
    _write_json(report_path, report)

    with pytest.raises(RuntimeError, match="joint macro statistics"):
        build_paper_artifacts.validate_format_contrast(tmp_path)


def test_direct_evidence_builder_emits_only_validated_paired_and_deployment_summaries(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch,
) -> None:
    _format_contrast_fixture(tmp_path)
    _deployment_fixture(tmp_path)
    guardrail = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "model": model,
                "precision": precision,
                "corruption": corruption,
                "severity": severity,
                "clean_ap_native": 0.5,
                "corrupt_ap_native": 0.4,
                "d_native": 0.1,
                "clean_metric_sha256": "a" * 64,
                "corrupt_metric_sha256": "b" * 64,
            }
            for dataset, model, precision, corruption, severity in product(
                build_paper_artifacts.DATASETS,
                build_paper_artifacts.MODEL_ORDER,
                build_paper_artifacts.QUANT_ORDER,
                build_paper_artifacts.CORRUPTION_ORDER,
                (1, 3, 5),
            )
        ]
    )
    monkeypatch.setattr(
        build_paper_artifacts,
        "load_direct_accuracy_guardrail",
        lambda root: (
            guardrail,
            {f"metric-{index}.json": "c" * 64 for index in range(312)},
            {"p0-completion.json": "d" * 64},
        ),
    )
    size_guardrail = pd.DataFrame(
        [
            {
                **row._asdict(),
                "endpoint_type": "original-height" if row.dataset == "tt100k" else "coco-area",
                "endpoint": endpoint,
                "clean_ap_native": 0.5,
                "corrupt_ap_native": 0.4,
                "d_native": 0.1,
            }
            for row in guardrail.itertuples(index=False)
            for endpoint in (("XS", "S", "L", "XL") if row.dataset == "tt100k" else ("small", "large"))
        ]
    )
    monkeypatch.setattr(
        build_paper_artifacts,
        "load_direct_size_accuracy_guardrail",
        lambda root, direct_guardrail: size_guardrail,
    )
    clean_conditions = list(
        product(
            build_paper_artifacts.DATASETS,
            build_paper_artifacts.MODEL_ORDER,
            build_paper_artifacts.PRECISION_ORDER,
        )
    )
    original_clean = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "model": model,
                "precision": precision,
                "corruption": "clean",
                "severity": 0,
                "AP": 0.50,
            }
            for dataset, model, precision in clean_conditions
        ]
    )
    codec_clean = pd.DataFrame(
        [
            {
                "dataset": dataset,
                "model": model,
                "precision": precision,
                "AP": 0.49,
            }
            for dataset, model, precision in clean_conditions
        ]
    )
    monkeypatch.setattr(
        build_paper_artifacts,
        "validate_and_load",
        lambda root: (original_clean, pd.DataFrame(), pd.DataFrame(), codec_clean),
    )

    generated = tmp_path / "paper" / "generated"
    figures = tmp_path / "paper" / "figures"
    audit = build_paper_artifacts.generate_direct_evidence_artifacts(
        tmp_path, generated, figures
    )

    cells = pd.read_csv(generated / "direct_format_contrast_cells.csv")
    deployment = pd.read_csv(generated / "direct_deployment_conditions.csv")
    assert audit["status"] == "valid"
    assert audit["counts"] == {
        "corrupted_paired_components": 144,
        "joint_macro_endpoints": 3,
        "deployment_conditions": 36,
        "deployment_repetitions": 108,
        "absolute_guardrail_arms": 288,
        "absolute_guardrail_metric_records": 312,
        "codec_sensitivity_conditions": 36,
        "heterogeneity_sensitivity_rows": 12,
        "runtime_sensitivity_rows": 5,
        "size_guardrail_rows": 720,
    }
    assert set(audit["generated_artifacts_sha256"]) == {
        "paper/generated/direct_format_contrast_cells.csv",
        "paper/generated/direct_format_contrast_macro.csv",
        "paper/generated/direct_deployment_conditions.csv",
        "paper/generated/direct_absolute_guardrail.csv",
        "paper/generated/direct_format_contrast_summary.tex",
        "paper/generated/direct_heterogeneity_summary.csv",
        "paper/generated/direct_heterogeneity_summary.tex",
        "paper/generated/direct_deployment_summary.tex",
        "paper/generated/direct_absolute_guardrail_summary.tex",
        "paper/generated/direct_absolute_guardrail_narrative.tex",
        "paper/generated/codec_sensitivity.csv",
        "paper/generated/codec_sensitivity.tex",
        "paper/generated/direct_heterogeneity_sensitivity.csv",
        "paper/generated/direct_heterogeneity_sensitivity.tex",
        "paper/generated/direct_runtime_sensitivity.csv",
        "paper/generated/direct_runtime_sensitivity.tex",
        "paper/generated/direct_sensitivity_narrative.tex",
        "paper/generated/direct_size_guardrail.csv",
        "paper/generated/direct_size_guardrail_summary.tex",
        "paper/generated/numbers_direct.tex",
            "paper/generated/direct_results_narrative.tex",
            "paper/generated/direct_deployment_narrative.tex",
            "paper/generated/direct_data_dictionary.json",
        "paper/generated/direct_abstract.tex",
        "paper/generated/direct_conclusion.tex",
        "paper/figures/fig_paired_excess_gap.pdf",
        "paper/figures/fig_paired_excess_gap.png",
    }
    assert len(cells) == 144
    assert set(cells) >= {
        "dataset",
        "model",
        "corruption",
        "severity",
        "delta_e_all",
        "delta_e_ci_low",
        "delta_e_ci_high",
        "delta_psi",
    }
    assert len(deployment) == 36
    assert set(deployment) >= {
        "dataset",
        "model",
        "precision",
        "engine_bytes",
        "latency_median_ms",
        "throughput_qps",
    }
    for path in (
        generated / "direct_format_contrast_summary.tex",
        generated / "direct_heterogeneity_summary.csv",
        generated / "direct_heterogeneity_summary.tex",
        generated / "direct_deployment_summary.tex",
        generated / "direct_absolute_guardrail_summary.tex",
        generated / "direct_absolute_guardrail_narrative.tex",
        generated / "codec_sensitivity.csv",
        generated / "codec_sensitivity.tex",
        generated / "direct_heterogeneity_sensitivity.csv",
        generated / "direct_heterogeneity_sensitivity.tex",
        generated / "direct_runtime_sensitivity.csv",
        generated / "direct_runtime_sensitivity.tex",
        generated / "direct_sensitivity_narrative.tex",
        generated / "direct_size_guardrail.csv",
        generated / "direct_size_guardrail_summary.tex",
        generated / "numbers_direct.tex",
        generated / "direct_abstract.tex",
        generated / "direct_conclusion.tex",
        generated / "direct_results_narrative.tex",
        generated / "direct_evidence_audit.json",
        figures / "fig_paired_excess_gap.pdf",
    ):
        assert path.is_file(), path
    narrative = (generated / "direct_results_narrative.tex").read_text(encoding="utf-8")
    assert "144 condition-level" in narrative
    assert "exploratory" in narrative
    heterogeneity = pd.read_csv(generated / "direct_heterogeneity_summary.csv")
    assert set(heterogeneity.factor) == {"Checkpoint rung", "Corruption", "Severity"}
    assert len(heterogeneity) == 10
    assert set(heterogeneity.cells) == {36, 48}
    guardrail_text = (generated / "direct_absolute_guardrail_narrative.tex").read_text(
        encoding="utf-8"
    )
    assert "median" in guardrail_text and "mean" in guardrail_text
    assert "arms had corrupted AP below 10 AP points" in guardrail_text
    deployment_text = (generated / "direct_deployment_narrative.tex").read_text(
        encoding="utf-8"
    )
    assert "INT8-to-FP8 latency ratio" in deployment_text
    dataset_table = (generated / "direct_format_contrast_summary.tex").read_text(
        encoding="utf-8"
    )
    assert dataset_table.index("COCO") < dataset_table.index("VOC")
    assert dataset_table.index("VOC") < dataset_table.index("KITTI")
    assert dataset_table.index("KITTI") < dataset_table.index("TT100K")
    abstract = (generated / "direct_abstract.tex").read_text(encoding="utf-8")
    conclusion = (generated / "direct_conclusion.tex").read_text(encoding="utf-8")
    assert "common image-bootstrap draws" in abstract
    assert "144" in abstract
    assert "does not establish format superiority" in conclusion


def test_direct_evidence_builder_fails_closed_before_writing_on_invalid_component(
    tmp_path: Path,
) -> None:
    report_path = _format_contrast_fixture(tmp_path)
    _deployment_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    relative = next(iter(report["component_artifacts_sha256"]))
    (tmp_path / relative).write_text("{}\n", encoding="utf-8")

    generated = tmp_path / "paper" / "generated"
    with pytest.raises(RuntimeError, match="SHA-256 mismatch"):
        build_paper_artifacts.generate_direct_evidence_artifacts(
            tmp_path, generated, tmp_path / "paper" / "figures"
        )
    assert not generated.exists()


def test_direct_absolute_guardrail_binds_exact_metric_arms_and_reconstructs_delta_e() -> None:
    guardrail, metric_hashes, completion_hashes = (
        build_paper_artifacts.load_direct_accuracy_guardrail(PROJECT_ROOT)
    )

    assert len(guardrail) == 288
    assert len(metric_hashes) == 312
    assert set(completion_hashes) == {
        "outputs/reports/p0_strengthening_complete_v1.json",
        "artifacts/four_dataset_pilot_v1/outputs/reports/four_dataset_pilot_final_report.json",
    }
    assert guardrail[["dataset", "model", "precision", "corruption", "severity"]].duplicated().sum() == 0
    assert guardrail.d_native.median() == pytest.approx(0.055125, abs=1e-6)
    assert int((guardrail.corrupt_ap_native < 0.10).sum()) == 35


def test_direct_size_guardrail_keeps_area_and_tt100k_height_endpoints_separate() -> None:
    """Catches interpreting a size interaction without absolute size-specific AP."""
    guardrail, metric_hashes, _ = build_paper_artifacts.load_direct_accuracy_guardrail(
        PROJECT_ROOT
    )
    size_guardrail = build_paper_artifacts.load_direct_size_accuracy_guardrail(
        PROJECT_ROOT, guardrail
    )

    assert len(size_guardrail) == 720
    assert set(size_guardrail.query("dataset != 'tt100k'").endpoint) == {"small", "large"}
    assert set(size_guardrail.query("dataset == 'tt100k'").endpoint) == {"XS", "S", "L", "XL"}
    assert size_guardrail[["clean_ap_native", "corrupt_ap_native", "d_native"]].notna().all().all()
    assert set(size_guardrail.clean_metric_sha256) <= set(metric_hashes.values())
    assert set(size_guardrail.corrupt_metric_sha256) <= set(metric_hashes.values())


def test_direct_metric_binding_rejects_a_changed_provenance_hash() -> None:
    source = {
        "prediction_sha256": "1" * 64,
        "input_manifest_sha256": "2" * 64,
        "image_ids_sha256": "3" * 64,
        "run_record_sha256": "4" * 64,
    }
    metric = {
        "prediction_sha256": "1" * 64,
        "input_manifest_sha256": "2" * 64,
        "input_image_ids_sha256": "3" * 64,
        "run_record_sha256": "5" * 64,
        "model": "yolo11n",
        "precision": "int8-entropy",
        "corruption": "gaussian_noise",
        "severity": 3,
    }

    with pytest.raises(RuntimeError, match="run_record_sha256"):
        build_paper_artifacts._validate_direct_metric_binding(
            source,
            metric,
            component_name="coco__yolo11n__gaussian_noise-s3",
            arm="int8_corrupt",
            dataset="coco",
            model="yolo11n",
            corruption="gaussian_noise",
            severity=3,
        )


def test_direct_metric_binding_rejects_a_semantically_wrong_arm() -> None:
    source = {
        "prediction_sha256": "1" * 64,
        "input_manifest_sha256": "2" * 64,
        "image_ids_sha256": "3" * 64,
        "run_record_sha256": "4" * 64,
    }
    metric = {
        "prediction_sha256": "1" * 64,
        "input_manifest_sha256": "2" * 64,
        "input_image_ids_sha256": "3" * 64,
        "run_record_sha256": "4" * 64,
        "model": "yolo11n",
        "precision": "fp8",
        "corruption": "gaussian_noise",
        "severity": 3,
    }

    with pytest.raises(RuntimeError, match="semantic mismatch"):
        build_paper_artifacts._validate_direct_metric_binding(
            source,
            metric,
            component_name="coco__yolo11n__gaussian_noise-s3",
            arm="int8_corrupt",
            dataset="coco",
            model="yolo11n",
            corruption="gaussian_noise",
            severity=3,
        )


def test_direct_results_template_references_only_direct_generated_artifacts() -> None:
    template = (PROJECT_ROOT / "paper" / "direct_results_template.tex").read_text(
        encoding="utf-8"
    )

    for expected in (
        r"\input{generated/direct_results_narrative.tex}",
        r"\input{generated/direct_format_contrast_summary.tex}",
        r"\input{generated/direct_heterogeneity_summary.tex}",
        r"\input{generated/direct_deployment_summary.tex}",
        r"\input{generated/direct_deployment_narrative.tex}",
        r"\input{generated/direct_absolute_guardrail_narrative.tex}",
        r"\input{generated/direct_absolute_guardrail_summary.tex}",
        r"\includegraphics[width=\textwidth]{fig_paired_excess_gap.pdf}",
        "do not measure memory, power, or energy",
    ):
        assert expected in template
    assert "interaction_summary.tex" not in template
    assert "numbers.tex" not in template
    for label in (
        "tab:direct-absolute-guardrail",
        "fig:direct-paired-excess-gap",
        "tab:direct-format-summary",
        "tab:direct-heterogeneity",
        "tab:direct-deployment-summary",
    ):
        assert rf"\ref{{{label}}}" in template


def test_active_supplement_exposes_codec_control_sensitivity() -> None:
    """Catches dropping the validated original-source/JPEG-95 control comparison."""
    supplement = (PROJECT_ROOT / "paper" / "supplement.tex").read_text(
        encoding="utf-8"
    )

    assert r"\input{generated/codec_sensitivity.tex}" in supplement
    assert r"\input{generated/direct_heterogeneity_sensitivity.tex}" in supplement
    assert r"\input{generated/direct_runtime_sensitivity.tex}" in supplement
    assert r"\input{generated/direct_size_guardrail_summary.tex}" in supplement
    assert "original-source clean" in supplement
    assert "point sensitivity analysis" in supplement


def test_direct_results_consumes_generated_sensitivity_narrative_once() -> None:
    """Keep quantitative sensitivity in Results rather than introducing it in Discussion."""
    results = (PROJECT_ROOT / "paper" / "direct_results_template.tex").read_text(
        encoding="utf-8"
    )
    discussion = (PROJECT_ROOT / "paper" / "direct_discussion_template.tex").read_text(
        encoding="utf-8"
    )

    marker = r"\input{generated/direct_sensitivity_narrative.tex}"
    assert results.count(marker) == 1
    assert marker not in discussion


def test_direct_heterogeneity_sensitivity_recomputes_weighting_and_leave_one_out() -> None:
    """Catches a grand mean that silently hides domain/corruption dependence."""
    cells = pd.DataFrame(
        {
            "dataset": ["a", "a", "b", "b"],
            "corruption": ["fog", "jpeg", "fog", "jpeg"],
            "delta_e_all": [0.00, 0.02, 0.04, 0.06],
        }
    )

    summary = build_paper_artifacts.direct_heterogeneity_sensitivity(
        cells,
        image_counts={"a": 1, "b": 3},
        object_counts={"a": 2, "b": 1},
    ).set_index(["analysis", "level"])

    assert summary.loc[("weighting", "equal-cell"), "point_native"] == pytest.approx(0.03)
    assert summary.loc[("weighting", "image-weighted"), "point_native"] == pytest.approx(0.04)
    assert summary.loc[("weighting", "object-weighted"), "point_native"] == pytest.approx(0.14 / 6)
    assert summary.loc[("leave-one-dataset-out", "a"), "point_native"] == pytest.approx(0.05)
    assert summary.loc[("leave-one-dataset-out", "b"), "point_native"] == pytest.approx(0.01)
    assert summary.loc[("leave-one-corruption-out", "fog"), "point_native"] == pytest.approx(0.04)
    assert summary.loc[("leave-one-corruption-out", "jpeg"), "point_native"] == pytest.approx(0.02)


def test_direct_runtime_sensitivity_stratifies_matched_ratios() -> None:
    """Catches pooling latency across incompatible input geometries."""
    deployment = pd.DataFrame(
        [
            {"dataset": dataset, "model": "yolo11n", "precision": precision, "latency_median_ms": latency}
            for dataset, values in {
                "coco": {"fp32": 4.0, "int8-entropy": 2.0, "fp8": 1.0},
                "tt100k": {"fp32": 8.0, "int8-entropy": 4.0, "fp8": 4.0},
            }.items()
            for precision, latency in values.items()
        ]
    )

    summary = build_paper_artifacts.direct_runtime_sensitivity(
        deployment, image_sizes={"coco": 640, "tt100k": 1280}
    ).set_index(["stratum", "level"])

    assert summary.loc[("input-size", "640"), "fp32_over_int8"] == pytest.approx(2.0)
    assert summary.loc[("input-size", "640"), "fp32_over_fp8"] == pytest.approx(4.0)
    assert summary.loc[("input-size", "1280"), "int8_over_fp8"] == pytest.approx(1.0)
    assert summary.loc[("capacity", "yolo11n"), "conditions"] == 2


def test_main_source_switches_every_result_bearing_block_to_direct_evidence() -> None:
    """A completed direct chain must not leave stale P0 claims in the final PDF."""
    main = (PROJECT_ROOT / "paper" / "main.tex").read_text(encoding="utf-8")

    for expected in (
        r"\IfFileExists{generated/direct_evidence_audit.json}",
        r"\input{generated/numbers_direct.tex}",
        r"\PackageError{ivc-direct}{Validated direct-evidence audit is missing}",
        r"\Delta E=(\mathrm{FP8}-\mathrm{INT8})_{\mathrm{corrupt}}",
        r"\input{direct_methods_tail.tex}",
        r"\input{direct_results_template.tex}",
        r"\input{direct_discussion_template.tex}",
        r"\input{generated/direct_conclusion.tex}",
        r"\input{direct_availability_template.tex}",
    ):
        assert expected in main

    direct_methods = (PROJECT_ROOT / "paper" / "direct_methods_tail.tex").read_text(
        encoding="utf-8"
    )
    assert "2,000" in direct_methods
    assert "144 corrupted" in direct_methods
    assert "500" not in direct_methods
    assert "136 observed classes" in direct_methods
    direct_results = (PROJECT_ROOT / "paper" / "direct_results_template.tex").read_text(
        encoding="utf-8"
    )
    supplement = (PROJECT_ROOT / "paper" / "supplement.tex").read_text(encoding="utf-8")
    assert "All 12 prespecified clean FP16 consistency gates passed" in direct_results
    assert "historical COCO configuration digest is not byte-identically recoverable" in supplement
    assert r"Table~\ref{tab:measurement-contract}" in main
    assert (
        r"\ifdirectevidence Original-source clean files are outside the direct paired analysis."
        in main
    )


def test_direct_abstract_and_conclusion_disambiguate_guardrail_thresholds() -> None:
    abstract = (PROJECT_ROOT / "paper" / "generated" / "direct_abstract.tex").read_text(
        encoding="utf-8"
    )
    conclusion = (
        PROJECT_ROOT / "paper" / "generated" / "direct_conclusion.tex"
    ).read_text(encoding="utf-8")
    for text in (abstract, conclusion):
        assert "median" in text and "mean" in text
        assert "corrupted AP below 10 AP points" in text
    assert "matched-clean AP" in abstract
    assert "architecture-dependent" not in abstract


def test_direct_paired_figure_converts_cell_values_to_ap_points() -> None:
    source = inspect.getsource(build_paper_artifacts._direct_paired_figure)
    assert 'data["delta_e_ap_points"] = 100.0 * data["delta_e_all"]' in source
    assert 'x="delta_e_ap_points"' in source
    assert 'set_xlabel(r"Direct paired $\\Delta E$ (AP points)")' in source
    assert 'suptitle("Direct INT8–FP8 paired excess-gap evidence"' in source



def test_method_graphical_abstract_has_elsevier_25_to_1_raster_contract(
    tmp_path: Path,
) -> None:
    build_paper_artifacts.generate_method_graphical_abstract(tmp_path)

    png = tmp_path / "graphical_abstract.png"
    pdf = tmp_path / "graphical_abstract.pdf"
    assert png.is_file()
    assert pdf.is_file()
    with Image.open(png) as image:
        assert image.size == (3000, 1200)


def test_deployment_validator_accepts_exact_complete_synthetic_chain(tmp_path: Path) -> None:
    _deployment_fixture(tmp_path)

    report = build_paper_artifacts.validate_deployment_benchmark(tmp_path)

    assert report["engine_conditions"] == 36
    assert report["repetitions_per_engine"] == 3
    assert report["repetition_records"] == 108
    assert report["raw_logs"] == 108


def test_deployment_validator_rejects_rehashed_latency_record_mutation(tmp_path: Path) -> None:
    report_path = _deployment_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    record_name = next(iter(report["records_sha256"]))
    record_path = tmp_path / "outputs" / "benchmarks" / "ivc_deployment_benchmark_v1" / record_name
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["latency_median_ms"] = 9.99
    report["records_sha256"][record_name] = _write_json(record_path, record)
    report["report_sha256"] = _canonical_sha(report, "report_sha256")
    _write_json(report_path, report)

    with pytest.raises(RuntimeError, match="raw log metric mismatch"):
        build_paper_artifacts.validate_deployment_benchmark(tmp_path)


def test_deployment_validator_rejects_rehashed_nonidle_gpu_record(tmp_path: Path) -> None:
    report_path = _deployment_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    record_name = next(iter(report["records_sha256"]))
    record_path = tmp_path / "outputs" / "benchmarks" / "ivc_deployment_benchmark_v1" / record_name
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["gpu_idle_query_output"] = "4815162342\n"
    report["records_sha256"][record_name] = _write_json(record_path, record)
    report["report_sha256"] = _canonical_sha(report, "report_sha256")
    _write_json(report_path, report)

    with pytest.raises(RuntimeError, match="GPU was not idle"):
        build_paper_artifacts.validate_deployment_benchmark(tmp_path)


def test_deployment_validator_rejects_rehashed_treatment_onnx_mutation(tmp_path: Path) -> None:
    report_path = _deployment_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    treatment_path = (
        tmp_path
        / "outputs"
        / "benchmarks"
        / "ivc_deployment_benchmark_v1"
        / "engine_treatments.json"
    )
    treatment = json.loads(treatment_path.read_text(encoding="utf-8"))
    treatment["records"][0]["onnx_sha256"] = "f" * 64
    treatment["treatments_sha256"] = _canonical_sha(treatment, "treatments_sha256")
    report["engine_treatments_sha256"] = _write_json(treatment_path, treatment)
    report["report_sha256"] = _canonical_sha(report, "report_sha256")
    _write_json(report_path, report)

    with pytest.raises(RuntimeError, match="ONNX treatment/benchmark binding"):
        build_paper_artifacts.validate_deployment_benchmark(tmp_path)


def test_deployment_validator_rejects_rehashed_repetition_provenance_mutation(
    tmp_path: Path,
) -> None:
    report_path = _deployment_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    record_name = next(name for name in report["records_sha256"] if name.endswith("rep-02.json"))
    record_path = tmp_path / "outputs" / "benchmarks" / "ivc_deployment_benchmark_v1" / record_name
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["input_name"] = "forged_images"
    report["records_sha256"][record_name] = _write_json(record_path, record)
    report["report_sha256"] = _canonical_sha(report, "report_sha256")
    _write_json(report_path, report)

    with pytest.raises(RuntimeError, match="repetition provenance"):
        build_paper_artifacts.validate_deployment_benchmark(tmp_path)


def test_deployment_validator_rejects_traversal_in_record_binding(tmp_path: Path) -> None:
    report_path = _deployment_fixture(tmp_path)
    report = json.loads(report_path.read_text(encoding="utf-8"))
    record_name = next(iter(report["records_sha256"]))
    digest = report["records_sha256"].pop(record_name)
    report["records_sha256"][f"../{record_name}"] = digest
    report["report_sha256"] = _canonical_sha(report, "report_sha256")
    _write_json(report_path, report)

    with pytest.raises(RuntimeError, match="canonical record/log paths"):
        build_paper_artifacts.validate_deployment_benchmark(tmp_path)
