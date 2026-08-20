from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import run_confirmatory_inference as inference


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(document: dict, field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_dataset_attempt_names_are_final_role_and_collision_free() -> None:
    assert inference.dataset_attempt("voc_kitti_confirmatory_v1", "voc") == (
        "voc_confirmatory_final_117_v1"
    )
    assert inference.dataset_attempt("voc_kitti_confirmatory_v1", "kitti") == (
        "kitti_confirmatory_final_117_v1"
    )
    with pytest.raises(ValueError, match="invalid dataset"):
        inference.dataset_attempt("voc_kitti_confirmatory_v1", "coco")


def test_inference_config_rejects_mutated_bound_source(tmp_path: Path) -> None:
    for relative in ("configs/execution.json", "configs/prep.json", "src/executor.py"):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(relative + "\n", encoding="utf-8")
    config = {
        "schema_version": 1,
        "attempt": "voc_kitti_confirmatory_v1",
        "execution_config": "configs/execution.json",
        "execution_config_sha256": digest(tmp_path / "configs/execution.json"),
        "prep_config": "configs/prep.json",
        "prep_config_sha256": digest(tmp_path / "configs/prep.json"),
        "reviewed_by": "project-owner-explicit-confirmatory-approval-2026-08-18",
        "source_manifest": [
            {"path": "src/executor.py", "sha256": digest(tmp_path / "src/executor.py")}
        ],
    }
    config["config_sha256"] = canonical(config, "config_sha256")
    path = tmp_path / "configs/inference.json"
    path.write_text(json.dumps(config) + "\n", encoding="utf-8")

    inference.load_inference_config(tmp_path, path)

    (tmp_path / "src/executor.py").write_text("mutated\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="source manifest hash mismatch"):
        inference.load_inference_config(tmp_path, path)


def test_prep_completion_requires_two_hash_bound_117_run_plans(tmp_path: Path) -> None:
    datasets = []
    for dataset in ("voc", "kitti"):
        plan = {
            "runs": [{"condition_id": f"{dataset}-{index}"} for index in range(117)]
        }
        plan["plan_sha256"] = canonical(plan, "plan_sha256")
        plan_path = tmp_path / f"{dataset}_plan.json"
        plan_path.write_text(json.dumps(plan) + "\n", encoding="utf-8")
        datasets.append(
            {
                "dataset": dataset,
                "frozen_plan": str(plan_path),
                "frozen_plan_sha256": digest(plan_path),
                "plan_sha256": plan["plan_sha256"],
            }
        )
    report = {
        "attempt": "attempt",
        "config_sha256": "a" * 64,
        "parity_completion_sha256": "b" * 64,
        "datasets": datasets,
    }
    report["prep_report_sha256"] = canonical(report, "prep_report_sha256")
    path = tmp_path / "prep_report.json"
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    result = inference.validate_prep_completion(
        path,
        attempt="attempt",
        prep_config_sha256="a" * 64,
        parity_completion_sha256="b" * 64,
    )
    assert set(result) == {"voc", "kitti"}

    Path(datasets[0]["frozen_plan"]).write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="frozen plan hash mismatch"):
        inference.validate_prep_completion(
            path,
            attempt="attempt",
            prep_config_sha256="a" * 64,
            parity_completion_sha256="b" * 64,
        )


def test_parity_paths_for_dataset_require_three_models_and_passed_reports(tmp_path: Path) -> None:
    rows = []
    for model in ("yolo11n", "yolo11m", "yolo11x"):
        report = tmp_path / f"voc_{model}.json"
        report.write_text(
            json.dumps({"dataset": "voc", "model": model, "pass": True}) + "\n",
            encoding="utf-8",
        )
        report.with_suffix(report.suffix + ".complete").write_text(
            digest(report) + "\n", encoding="utf-8"
        )
        rows.append(
            {"dataset": "voc", "model": model, "report": str(report), "report_sha256": digest(report)}
        )

    paths = inference.parity_paths_for_dataset(rows, "voc")
    assert [path.name for path in paths] == [
        "voc_yolo11n.json",
        "voc_yolo11m.json",
        "voc_yolo11x.json",
    ]

    rows.pop()
    with pytest.raises(SystemExit, match="three parity reports"):
        inference.parity_paths_for_dataset(rows, "voc")
