from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import verify_reference_parity as parity


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def write_pair(root: Path, label: str, ap: float) -> tuple[Path, Path]:
    run = {
        "condition_id": f"voc_final__yolo11n__{label}__clean-s0",
        "dataset": "voc",
        "split": "final",
        "model": "yolo11n",
        "corruption": "clean",
        "severity": 0,
        "input_manifest_sha256": "a" * 64,
        "input_image_ids_sha256": "b" * 64,
        "annotation_sha256": "c" * 64,
        "prediction_sha256": "d" * 64,
        "n_images": 12,
    }
    if label in {"pytorch", "onnxruntime"}:
        run.update({"backend": label, "precision": "fp32-reference"})
    else:
        run.update({"precision": "fp32" if label == "trt-fp32" else "fp16"})
    run_path = root / f"{label}.run.json"
    run_path.write_text(json.dumps(run) + "\n", encoding="utf-8")
    metric = {
        "condition_id": run["condition_id"],
        "dataset": "voc",
        "split": "final",
        "model": "yolo11n",
        "corruption": "clean",
        "severity": 0,
        "n_images": 12,
        "input_manifest_sha256": "a" * 64,
        "input_image_ids_sha256": "b" * 64,
        "prediction_sha256": "d" * 64,
        "run_record_sha256": digest(run_path),
        "stats": {"AP": ap},
    }
    metric_path = root / f"{label}.metric.json"
    metric_path.write_text(json.dumps(metric) + "\n", encoding="utf-8")
    return metric_path, run_path


def complete_grid(tmp_path: Path, aps: dict[str, float] | None = None) -> dict[str, tuple[Path, Path]]:
    values = aps or {
        "pytorch": 0.5000,
        "onnxruntime": 0.5002,
        "trt-fp32": 0.4998,
        "trt-fp16": 0.4950,
    }
    return {label: write_pair(tmp_path, label, ap) for label, ap in values.items()}


def test_parity_accepts_exact_shared_inputs_and_locked_tolerances(tmp_path: Path) -> None:
    report = parity.verify_reference_metrics(
        complete_grid(tmp_path), source_tolerance=0.001, fp16_tolerance=0.01
    )

    assert report["pass"] is True
    assert report["ap"] == {
        "pytorch": 0.5,
        "onnxruntime": 0.5002,
        "trt-fp32": 0.4998,
        "trt-fp16": 0.495,
    }
    assert report["absolute_ap_gaps"]["pytorch_to_trt_fp32"] == pytest.approx(0.0002)


def test_parity_rejects_source_or_fp16_gap_beyond_threshold(tmp_path: Path) -> None:
    report = parity.verify_reference_metrics(
        complete_grid(
            tmp_path,
            {
                "pytorch": 0.50,
                "onnxruntime": 0.503,
                "trt-fp32": 0.499,
                "trt-fp16": 0.48,
            },
        ),
        source_tolerance=0.001,
        fp16_tolerance=0.01,
    )

    assert report["pass"] is False
    assert any("PyTorch--ONNX Runtime" in error for error in report["errors"])
    assert any("FP32--FP16" in error for error in report["errors"])


def test_parity_rejects_input_universe_mismatch(tmp_path: Path) -> None:
    pairs = complete_grid(tmp_path)
    metric_path, _ = pairs["onnxruntime"]
    metric = json.loads(metric_path.read_text())
    metric["input_image_ids_sha256"] = "f" * 64
    metric_path.write_text(json.dumps(metric) + "\n", encoding="utf-8")

    report = parity.verify_reference_metrics(
        pairs, source_tolerance=0.001, fp16_tolerance=0.01
    )

    assert report["pass"] is False
    assert any("input_image_ids_sha256" in error for error in report["errors"])


def test_parity_rejects_wrong_backend_identity(tmp_path: Path) -> None:
    pairs = complete_grid(tmp_path)
    _, run_path = pairs["pytorch"]
    run = json.loads(run_path.read_text())
    run["backend"] = "onnxruntime"
    run_path.write_text(json.dumps(run) + "\n", encoding="utf-8")
    metric_path, _ = pairs["pytorch"]
    metric = json.loads(metric_path.read_text())
    metric["run_record_sha256"] = digest(run_path)
    metric_path.write_text(json.dumps(metric) + "\n", encoding="utf-8")

    report = parity.verify_reference_metrics(
        pairs, source_tolerance=0.001, fp16_tolerance=0.01
    )

    assert report["pass"] is False
    assert any("backend identity" in error for error in report["errors"])
