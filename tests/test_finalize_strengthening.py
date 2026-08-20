from __future__ import annotations

import json
from pathlib import Path

import pytest

import finalize_strengthening as finalizer
from pilot_registry import canonical_hash
from topic_c.manifest import sha256_file


def _write_completion(path: Path, document: dict, key: str) -> None:
    document[key] = canonical_hash(document, key)
    path.write_text(json.dumps(document), encoding="utf-8")
    path.with_suffix(path.suffix + ".complete").write_text(
        sha256_file(path) + "\n", encoding="utf-8"
    )


def test_validate_self_hashed_completion_rejects_mutated_bytes(tmp_path: Path) -> None:
    path = tmp_path / "analysis.json"
    _write_completion(path, {"schema_version": 1, "cells": 72}, "analysis_sha256")

    assert finalizer.validate_self_hashed_completion(path, "analysis_sha256")["cells"] == 72

    path.write_text(path.read_text(encoding="utf-8") + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="completion marker"):
        finalizer.validate_self_hashed_completion(path, "analysis_sha256")


def test_validate_parity_requires_exact_six_passing_dataset_model_reports(tmp_path: Path) -> None:
    rows = []
    for dataset in finalizer.CONFIRMATORY_DATASETS:
        for model in finalizer.MODELS:
            report = tmp_path / f"{dataset}_{model}.json"
            report.write_text(json.dumps({"dataset": dataset, "model": model, "pass": True, "errors": []}), encoding="utf-8")
            rows.append({"dataset": dataset, "model": model, "report": str(report), "report_sha256": sha256_file(report)})
    execution = tmp_path / "execution.json"
    execution.write_text(
        json.dumps(
            {
                "reference": {
                    "tf32_enabled": False,
                    "required_parity_chain": [
                        "PyTorch", "FP32 ONNX Runtime", "FP32 TensorRT", "FP16 TensorRT"
                    ],
                }
            }
        ),
        encoding="utf-8",
    )
    completion = tmp_path / "parity.json"
    _write_completion(
        completion,
        {
            "schema_version": 1,
            "attempt": "voc_kitti_confirmatory_v1",
            "execution_config": str(execution),
            "execution_config_sha256": sha256_file(execution),
            "reports": rows,
        },
        "parity_completion_sha256",
    )

    validated = finalizer.validate_parity_completion(completion)
    assert len(validated["reports"]) == 6

    first = Path(rows[0]["report"])
    first.write_text(json.dumps({"dataset": "voc", "model": "yolo11n", "pass": False, "errors": ["bad"]}), encoding="utf-8")
    with pytest.raises(SystemExit, match="parity report hash mismatch"):
        finalizer.validate_parity_completion(completion)
