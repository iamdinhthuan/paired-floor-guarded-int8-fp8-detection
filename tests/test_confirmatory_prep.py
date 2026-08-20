from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import run_confirmatory_prep as prep


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(document: dict, field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_corruption_manifest_path_is_attempt_namespaced(tmp_path: Path) -> None:
    path = prep.corruption_manifest_path(
        tmp_path,
        "voc_kitti_confirmatory_v1",
        "voc",
        "gaussian_noise",
        3,
    )

    assert path == (
        tmp_path
        / "manifests"
        / "images"
        / "voc_kitti_confirmatory_v1"
        / "voc_final_gaussian_noise_s3.json"
    )


def test_prep_config_rejects_mutated_matrix_or_source(tmp_path: Path) -> None:
    for relative, content in (
        ("configs/execution.json", "{}\n"),
        ("configs/parity.json", "{}\n"),
        ("configs/matrix.json", "{}\n"),
        ("src/generator.py", "source\n"),
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(content, encoding="utf-8")
    config = {
        "schema_version": 1,
        "attempt": "voc_kitti_confirmatory_v1",
        "execution_config": "configs/execution.json",
        "execution_config_sha256": digest(tmp_path / "configs/execution.json"),
        "parity_config": "configs/parity.json",
        "parity_config_sha256": digest(tmp_path / "configs/parity.json"),
        "matrix": "configs/matrix.json",
        "matrix_sha256": digest(tmp_path / "configs/matrix.json"),
        "minimum_free_gib_after_estimate": 50,
        "source_manifest": [
            {"path": "src/generator.py", "sha256": digest(tmp_path / "src/generator.py")}
        ],
    }
    config["config_sha256"] = canonical(config, "config_sha256")
    path = tmp_path / "configs/prep.json"
    path.write_text(json.dumps(config) + "\n", encoding="utf-8")

    prep.load_prep_config(tmp_path, path)

    (tmp_path / "configs/matrix.json").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="matrix hash mismatch"):
        prep.load_prep_config(tmp_path, path)


def test_parity_completion_requires_exact_six_passed_bound_reports(tmp_path: Path) -> None:
    reports = []
    for dataset in ("voc", "kitti"):
        for model in ("yolo11n", "yolo11m", "yolo11x"):
            path = tmp_path / "reports" / f"{dataset}_{model}.json"
            path.parent.mkdir(parents=True, exist_ok=True)
            path.write_text(
                json.dumps({"dataset": dataset, "model": model, "pass": True}) + "\n",
                encoding="utf-8",
            )
            path.with_suffix(path.suffix + ".complete").write_text(
                digest(path) + "\n", encoding="utf-8"
            )
            reports.append(
                {"dataset": dataset, "model": model, "report": str(path), "report_sha256": digest(path)}
            )
    completion = {
        "attempt": "attempt",
        "parity_config_sha256": "a" * 64,
        "ladder_report_sha256": "b" * 64,
        "reports": reports,
    }
    completion["parity_completion_sha256"] = canonical(
        completion, "parity_completion_sha256"
    )
    path = tmp_path / "completion.json"
    path.write_text(json.dumps(completion) + "\n", encoding="utf-8")

    prep.validate_parity_completion(
        path,
        attempt="attempt",
        parity_config_sha256="a" * 64,
        ladder_report_sha256="b" * 64,
    )

    report_path = Path(reports[0]["report"])
    report_path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="nested parity report hash mismatch"):
        prep.validate_parity_completion(
            path,
            attempt="attempt",
            parity_config_sha256="a" * 64,
            ladder_report_sha256="b" * 64,
        )


def test_matrix_machine_split_must_match_execution_and_preserve_final_role() -> None:
    matrix = {
        "expected_runs_per_dataset": 117,
        "models": ["yolo11n", "yolo11m", "yolo11x"],
        "precisions": ["fp32", "int8-entropy", "fp8"],
        "conditions": [{"corruption": "clean", "severity": 0}] * 13,
        "dataset_protocols": {
            "voc": {"split": "test", "partition_role": "final"},
            "kitti": {"split": "test", "partition_role": "final"},
        },
    }
    execution = {
        "datasets": {
            "voc": {"split": "test", "partition_role": "final"},
            "kitti": {"split": "test", "partition_role": "final"},
        }
    }

    prep.validate_matrix(matrix, execution)

    matrix["dataset_protocols"]["voc"]["split"] = "final"
    with pytest.raises(SystemExit, match="matrix machine split"):
        prep.validate_matrix(matrix, execution)
