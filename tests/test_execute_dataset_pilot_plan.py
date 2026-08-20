from __future__ import annotations

import json
from pathlib import Path

import pytest

from execute_dataset_pilot import load_plan, plan_calibration_sha256, validate_parity
from pilot_registry import canonical_hash


CALIBRATION_SHA = "a" * 64


def frozen_plan(*, top_level: bool = False, run_dataset: str = "voc") -> dict:
    runs = []
    for index in range(117):
        runs.append({
            "condition_id": f"condition-{index}",
            "dataset": run_dataset,
            "split": "val",
            "precision": "int8-entropy" if index < 13 else ("fp8" if index < 26 else "fp32"),
            "calibration_sha256": CALIBRATION_SHA if index < 26 else None,
        })
    document = {"schema_version": 1, "runs": runs}
    if top_level:
        document.update({"dataset": "voc", "split": "val", "calibration_sha256": CALIBRATION_SHA})
    document["plan_sha256"] = canonical_hash(document, "plan_sha256")
    return document


def write_plan(path: Path, document: dict) -> Path:
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def test_load_plan_accepts_hash_bound_run_provenance_without_top_level_fields(tmp_path: Path) -> None:
    plan = load_plan(write_plan(tmp_path / "plan.json", frozen_plan()), "voc", "val")

    assert plan_calibration_sha256(plan) == CALIBRATION_SHA


def test_load_plan_accepts_matching_top_level_fields(tmp_path: Path) -> None:
    load_plan(write_plan(tmp_path / "plan.json", frozen_plan(top_level=True)), "voc", "val")


def test_load_plan_rejects_run_level_dataset_mismatch(tmp_path: Path) -> None:
    path = write_plan(tmp_path / "plan.json", frozen_plan(run_dataset="kitti"))

    with pytest.raises(SystemExit, match="dataset/split plan mismatch"):
        load_plan(path, "voc", "val")


def test_plan_calibration_sha256_rejects_inconsistent_int8_provenance() -> None:
    plan = frozen_plan()
    plan["runs"][0]["calibration_sha256"] = "b" * 64

    with pytest.raises(SystemExit, match="inconsistent INT8/FP8 calibration provenance"):
        plan_calibration_sha256(plan)


def test_plan_calibration_sha256_rejects_inconsistent_fp8_provenance() -> None:
    plan = frozen_plan()
    plan["runs"][13]["calibration_sha256"] = "b" * 64

    with pytest.raises(SystemExit, match="INT8/FP8 calibration provenance"):
        plan_calibration_sha256(plan)


def test_validate_parity_accepts_six_passed_four_backend_reports(tmp_path: Path) -> None:
    reports = []
    for model in ("yolo11n", "yolo11m", "yolo11x"):
        artifacts = {}
        for label, precision in (
            ("pytorch", "fp32-reference"),
            ("onnxruntime", "fp32-reference"),
            ("trt-fp32", "fp32"),
            ("trt-fp16", "fp16"),
        ):
            run = tmp_path / f"{model}_{label}.run.json"
            run.write_text(
                json.dumps(
                    {
                        "dataset": "voc",
                        "model": model,
                        "backend": label if label in {"pytorch", "onnxruntime"} else None,
                        "precision": precision,
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            metric = tmp_path / f"{model}_{label}.metric.json"
            metric.write_text(
                json.dumps(
                    {
                        "dataset": "voc",
                        "model": model,
                        "precision": precision,
                        "run_record_sha256": __import__("hashlib").sha256(run.read_bytes()).hexdigest(),
                    }
                )
                + "\n",
                encoding="utf-8",
            )
            artifacts[label] = {
                "metric": str(metric),
                "metric_sha256": __import__("hashlib").sha256(metric.read_bytes()).hexdigest(),
                "run_record": str(run),
                "run_record_sha256": __import__("hashlib").sha256(run.read_bytes()).hexdigest(),
            }
        report = tmp_path / f"{model}_parity.json"
        report.write_text(
            json.dumps(
                {
                    "pass": True,
                    "dataset": "voc",
                    "model": model,
                    "artifacts": artifacts,
                }
            )
            + "\n",
            encoding="utf-8",
        )
        report.with_suffix(report.suffix + ".complete").write_text(
            __import__("hashlib").sha256(report.read_bytes()).hexdigest() + "\n",
            encoding="utf-8",
        )
        reports.append(report)

    validate_parity(reports, "voc")

    metric = Path(json.loads(reports[0].read_text())["artifacts"]["trt-fp16"]["metric"])
    metric.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="artifact hash mismatch"):
        validate_parity(reports, "voc")
