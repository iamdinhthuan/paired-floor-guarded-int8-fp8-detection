from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import run_confirmatory_parity as parity


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(document: dict, field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_parity_paths_are_namespaced_per_backend(tmp_path: Path) -> None:
    paths = parity.parity_paths(
        tmp_path, "voc_kitti_confirmatory_v1", "voc", "yolo11m", "onnxruntime"
    )

    base = (
        tmp_path
        / "outputs"
        / "reference_parity"
        / "voc_kitti_confirmatory_v1"
        / "voc"
        / "yolo11m"
    )
    assert paths.prediction == base / "onnxruntime.predictions.json"
    assert paths.input_record == base / "onnxruntime.inputs.json"
    assert paths.run_record == base / "onnxruntime.run.json"
    assert paths.metric == base / "onnxruntime.metric.json"


def test_parity_config_rejects_mutated_source_package(tmp_path: Path) -> None:
    source = tmp_path / "src" / "runner.py"
    source.parent.mkdir(parents=True)
    source.write_text("original\n", encoding="utf-8")
    execution = tmp_path / "configs" / "execution.json"
    execution.parent.mkdir(parents=True)
    execution.write_text("{}\n", encoding="utf-8")
    config = {
        "schema_version": 1,
        "attempt": "voc_kitti_confirmatory_v1",
        "execution_config": "configs/execution.json",
        "execution_config_sha256": digest(execution),
        "backends": ["pytorch", "onnxruntime", "trt-fp32", "trt-fp16"],
        "source_ap_tolerance_native": 0.001,
        "fp16_ap_tolerance_native": 0.01,
        "source_manifest": [{"path": "src/runner.py", "sha256": digest(source)}],
    }
    config["config_sha256"] = canonical(config, "config_sha256")
    path = tmp_path / "configs" / "parity.json"
    path.write_text(json.dumps(config) + "\n", encoding="utf-8")

    validated = parity.load_parity_config(tmp_path, path)
    assert validated["source_ap_tolerance_native"] == 0.001

    source.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="source manifest hash mismatch"):
        parity.load_parity_config(tmp_path, path)


def test_complete_backend_requires_prediction_input_run_and_metric_hash_closure(tmp_path: Path) -> None:
    paths = parity.parity_paths(tmp_path, "attempt", "voc", "yolo11n", "pytorch")
    paths.prediction.parent.mkdir(parents=True)
    paths.prediction.write_text("[]\n", encoding="utf-8")
    paths.input_record.write_text(
        json.dumps(
            {
                "condition_id": "condition",
                "image_ids": [1, 2],
                "image_ids_sha256": hashlib.sha256(b"[1,2]").hexdigest(),
                "input_manifest_sha256": "a" * 64,
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths.run_record.write_text(
        json.dumps(
            {
                "condition_id": "condition",
                "backend": "pytorch",
                "precision": "fp32-reference",
                "dataset": "voc",
                "split": "final",
                "model": "yolo11n",
                "corruption": "clean",
                "severity": 0,
                "n_images": 2,
                "prediction_sha256": digest(paths.prediction),
                "input_manifest_sha256": "a" * 64,
                "input_image_ids_sha256": hashlib.sha256(b"[1,2]").hexdigest(),
            }
        )
        + "\n",
        encoding="utf-8",
    )
    paths.metric.write_text(
        json.dumps(
            {
                "condition_id": "condition",
                "dataset": "voc",
                "split": "final",
                "model": "yolo11n",
                "corruption": "clean",
                "severity": 0,
                "n_images": 2,
                "prediction_sha256": digest(paths.prediction),
                "input_manifest_sha256": "a" * 64,
                "input_image_ids_sha256": hashlib.sha256(b"[1,2]").hexdigest(),
                "run_record_sha256": digest(paths.run_record),
                "stats": {"AP": 0.5},
            }
        )
        + "\n",
        encoding="utf-8",
    )

    parity.validate_backend_output(
        paths,
        label="pytorch",
        dataset="voc",
        split="final",
        model="yolo11n",
        expected_images=2,
        manifest_sha256="a" * 64,
    )

    paths.prediction.write_text("[{}]\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="prediction hash mismatch"):
        parity.validate_backend_output(
            paths,
            label="pytorch",
            dataset="voc",
            split="final",
            model="yolo11n",
            expected_images=2,
            manifest_sha256="a" * 64,
        )


def test_ladder_report_requires_exact_24_artifacts_and_bound_execution_config(tmp_path: Path) -> None:
    execution = tmp_path / "execution.json"
    execution.write_text("{}\n", encoding="utf-8")
    artifacts = [
        {"dataset": dataset, "model": model, "precision": precision}
        for dataset in ("voc", "kitti")
        for model in ("yolo11n", "yolo11m", "yolo11x")
        for precision in ("fp32", "fp16", "int8-entropy", "fp8")
    ]
    report = {
        "attempt": "attempt",
        "config_sha256": digest(execution),
        "artifacts": artifacts,
        "dataset_registries": [
            {"dataset": "voc", "path": "voc.json", "sha256": "a" * 64},
            {"dataset": "kitti", "path": "kitti.json", "sha256": "b" * 64},
        ],
    }
    report["ladder_report_sha256"] = canonical(report, "ladder_report_sha256")
    path = tmp_path / "ladder.json"
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    parity.validate_ladder_report(path, attempt="attempt", execution_config=execution)

    report["artifacts"].pop()
    report["ladder_report_sha256"] = canonical(report, "ladder_report_sha256")
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="24-artifact grid"):
        parity.validate_ladder_report(path, attempt="attempt", execution_config=execution)


def test_trt_command_has_exactly_one_input_record_argument(tmp_path: Path) -> None:
    command = parity.trt_backend_command(
        python="python",
        runner=tmp_path / "coco_infer_trt.py",
        engine=tmp_path / "model.engine",
        annotation=tmp_path / "instances.json",
        manifest=tmp_path / "images.json",
        clean_root=tmp_path / "images",
        prediction=tmp_path / "predictions.json",
        input_record=tmp_path / "inputs.json",
        run_record=tmp_path / "run.json",
        class_map=tmp_path / "classes.json",
        condition="voc_final__yolo11n__trt-fp32__clean-s0",
        dataset="voc",
        split="test",
        model="yolo11n",
        imgsz=640,
        precision="fp32",
    )

    assert command.count("--input-record") == 1
    index = command.index("--input-record")
    assert command[index + 1] == str(tmp_path / "inputs.json")
    assert command[index + 2] == "--run-record"
