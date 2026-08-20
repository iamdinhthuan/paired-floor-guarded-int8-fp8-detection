from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import run_confirmatory_ladder as ladder


MODELS = ("yolo11n", "yolo11m", "yolo11x")
PRECISIONS = ("fp32", "fp16", "int8-entropy", "fp8")


def write_json(path: Path, document: dict) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return path


def physical_sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(document: dict, field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def frozen_inputs(root: Path) -> tuple[Path, Path, Path, dict]:
    protocol = {
        "schema_version": 1,
        "attempt": "voc_kitti_confirmatory_v1",
        "protocol": "prospectively_locked_confirmatory_resplit",
        "models": list(MODELS),
        "formats": list(PRECISIONS),
        "datasets": {"voc": {}, "kitti": {}},
    }
    protocol["config_sha256"] = canonical_sha(protocol, "config_sha256")
    protocol_path = write_json(root / "configs" / "confirmatory_voc_kitti_v1.json", protocol)

    jobs = [
        {
            "dataset": dataset,
            "model": model,
            "run_id": f"{dataset}_{model}_confirmatory_s20260818_v1",
        }
        for dataset in ("voc", "kitti")
        for model in MODELS
    ]
    queue = {
        "schema_version": 1,
        "queue_id": "voc_kitti_confirmatory_yolo11_nmx_queue_v1",
        "jobs": jobs,
    }
    queue_path = write_json(root / "configs" / "training" / "queue.json", queue)

    dataset_configs = {}
    for dataset in ("voc", "kitti"):
        clean_manifest = write_json(
            root / "manifests" / "images" / f"{dataset}_clean.json",
            {"dataset": dataset, "split": "test", "records": []},
        )
        class_map = write_json(
            root / "manifests" / "classes" / f"{dataset}.json",
            {"dataset": dataset, "split": "test"},
        )
        annotation = write_json(root / "manifests" / "annotations" / f"{dataset}.json", {})
        dataset_configs[dataset] = {
            "split": "test",
            "partition_role": "final",
            "imgsz": 640,
            "expected_images": 1,
            "annotation": str(annotation.relative_to(root)),
            "clean_manifest": str(clean_manifest.relative_to(root)),
            "class_map": str(class_map.relative_to(root)),
            "calibration": f"manifests/calibration/{dataset}.json",
            "clean_root": f"data/{dataset}",
            "corruption_root": f"data/corrupt/{dataset}",
        }
    execution = {
        "schema_version": 1,
        "attempt": "voc_kitti_confirmatory_v1",
        "protocol_config": str(protocol_path.relative_to(root)),
        "protocol_config_sha256": physical_sha(protocol_path),
        "training_queue": str(queue_path.relative_to(root)),
        "training_queue_sha256": physical_sha(queue_path),
        "models": list(MODELS),
        "precisions": list(PRECISIONS),
        "trt_root": "/opt/TensorRT-11.1.0.106",
        "workspace": "4096M",
        "datasets": dataset_configs,
    }
    execution["config_sha256"] = canonical_sha(execution, "config_sha256")
    execution_path = write_json(root / "configs" / "confirmatory_execution_v1.json", execution)
    return execution_path, protocol_path, queue_path, queue


def test_execution_config_binds_protocol_queue_and_exact_grid(tmp_path: Path) -> None:
    execution_path, _, _, queue = frozen_inputs(tmp_path)

    config, jobs = ladder.load_execution_config(tmp_path, execution_path)

    assert config["attempt"] == "voc_kitti_confirmatory_v1"
    assert set(jobs) == {
        (dataset, model) for dataset in ("voc", "kitti") for model in MODELS
    }
    assert jobs[("voc", "yolo11m")] == "voc_yolo11m_confirmatory_s20260818_v1"
    assert len(queue["jobs"]) == 6


def test_execution_config_rejects_mutated_bound_queue(tmp_path: Path) -> None:
    execution_path, _, queue_path, queue = frozen_inputs(tmp_path)
    queue["jobs"][0]["run_id"] = "forged"
    write_json(queue_path, queue)

    with pytest.raises(SystemExit, match="training queue hash mismatch"):
        ladder.load_execution_config(tmp_path, execution_path)


def test_execution_config_rejects_partition_role_used_as_literal_manifest_split(tmp_path: Path) -> None:
    execution_path, _, _, _ = frozen_inputs(tmp_path)
    execution = json.loads(execution_path.read_text())
    execution["datasets"]["voc"]["split"] = "final"
    execution["config_sha256"] = canonical_sha(execution, "config_sha256")
    write_json(execution_path, execution)

    with pytest.raises(SystemExit, match="machine split differs from frozen assets"):
        ladder.load_execution_config(tmp_path, execution_path)


def test_confirmatory_artifacts_are_namespaced_away_from_exploratory_ladder(tmp_path: Path) -> None:
    paths = ladder.artifact_paths(
        tmp_path, "voc_kitti_confirmatory_v1", "voc", "yolo11n", "int8-entropy"
    )

    assert paths.onnx == (
        tmp_path
        / "engines"
        / "voc_kitti_confirmatory_v1"
        / "voc"
        / "yolo11n"
        / "int8-entropy.onnx"
    )
    assert paths.onnx_registry == (
        tmp_path
        / "manifests"
        / "onnx"
        / "voc_kitti_confirmatory_v1"
        / "voc_yolo11n_int8-entropy.json"
    )
    assert paths.engine_registry == (
        tmp_path
        / "manifests"
        / "engines"
        / "voc_kitti_confirmatory_v1"
        / "voc_yolo11n_int8-entropy.json"
    )
    assert tmp_path / "engines" / "voc" / "yolo11n" not in paths.onnx.parents


def test_training_completion_requires_all_six_hash_bound_registries(tmp_path: Path) -> None:
    execution_path, _, _, _ = frozen_inputs(tmp_path)
    _, jobs = ladder.load_execution_config(tmp_path, execution_path)
    records = []
    for (dataset, model), run_id in jobs.items():
        weights = tmp_path / "outputs" / "training" / dataset / run_id / "weights"
        weights.mkdir(parents=True)
        best = weights / "best.pt"
        best.write_bytes(f"best:{run_id}".encode())
        last = weights / "last.pt"
        registry = write_json(
            tmp_path / "manifests" / "training" / f"{run_id}.json",
            {
                "dataset": dataset,
                "model": model,
                "run_id": run_id,
                "best_weights": str(best),
                "best_weights_sha256": physical_sha(best),
                "last_weights": str(last),
            },
        )
        registry.with_suffix(registry.suffix + ".complete").write_text(
            physical_sha(registry) + "\n", encoding="utf-8"
        )
        compaction = write_json(
            tmp_path / "manifests" / "training" / f"{run_id}_best_only_v1.json",
            {
                "run_id": run_id,
                "training_registry": str(registry),
                "training_registry_sha256": physical_sha(registry),
                "retained_best_weights": str(best),
                "retained_best_weights_sha256": physical_sha(best),
                "deleted_last_weights": str(last),
            },
        )
        compaction.with_suffix(compaction.suffix + ".complete").write_text(
            physical_sha(compaction) + "\n", encoding="utf-8"
        )
        records.append(
            {
                "run_id": run_id,
                "registry_sha256": physical_sha(registry),
                "compaction_report_sha256": physical_sha(compaction),
            }
        )
    report = {"schema_version": 1, "jobs": records}
    report["queue_report_sha256"] = canonical_sha(report, "queue_report_sha256")
    report_path = write_json(tmp_path / "outputs" / "reports" / "training.json", report)

    validated = ladder.validate_training_completion(tmp_path, report_path, jobs)

    assert len(validated) == 6
    first = next(iter(validated.values()))
    first.write_text("{}\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="training registry hash mismatch"):
        ladder.validate_training_completion(tmp_path, report_path, jobs)


def test_training_completion_rejects_missing_compaction_proof(tmp_path: Path) -> None:
    execution_path, _, _, _ = frozen_inputs(tmp_path)
    _, jobs = ladder.load_execution_config(tmp_path, execution_path)
    report = {"schema_version": 1, "jobs": []}
    for (dataset, model), run_id in jobs.items():
        registry = write_json(
            tmp_path / "manifests" / "training" / f"{run_id}.json",
            {"dataset": dataset, "model": model, "run_id": run_id},
        )
        registry.with_suffix(registry.suffix + ".complete").write_text(
            physical_sha(registry) + "\n", encoding="utf-8"
        )
        report["jobs"].append(
            {"run_id": run_id, "registry_sha256": physical_sha(registry)}
        )
    report["queue_report_sha256"] = canonical_sha(report, "queue_report_sha256")
    report_path = write_json(tmp_path / "outputs" / "reports" / "training.json", report)

    with pytest.raises(SystemExit, match="compaction proof"):
        ladder.validate_training_completion(tmp_path, report_path, jobs)


def test_dataset_registry_requires_exact_24_engine_records(tmp_path: Path) -> None:
    records: dict[tuple[str, str], Path] = {}
    for model in MODELS:
        for precision in PRECISIONS:
            engine = tmp_path / "engines" / "attempt" / "voc" / model / f"{precision}.plan"
            engine.parent.mkdir(parents=True, exist_ok=True)
            engine.write_bytes(f"{model}:{precision}".encode())
            record = {
                "dataset": "voc",
                "model": model,
                "precision": precision,
                "engine": str(engine),
                "engine_sha256": physical_sha(engine),
                "calibration_sha256": None,
                "imgsz": 640,
            }
            path = write_json(tmp_path / "records" / f"{model}_{precision}.json", record)
            path.with_suffix(path.suffix + ".complete").write_text(
                physical_sha(path) + "\n", encoding="utf-8"
            )
            records[(model, precision)] = path

    output = tmp_path / "manifests" / "engines" / "attempt" / "voc_ladder.json"
    ladder.freeze_dataset_registry(
        dataset="voc", attempt="attempt", records=records, output=output
    )

    document = json.loads(output.read_text(encoding="utf-8"))
    assert document["registry_sha256"] == canonical_sha(document, "registry_sha256")
    assert sum(len(item) for item in document["engines"].values()) == 12
    assert output.with_suffix(output.suffix + ".complete").read_text().strip() == physical_sha(output)

    records.pop(("yolo11x", "fp8"))
    with pytest.raises(SystemExit, match="exact engine grid"):
        ladder.freeze_dataset_registry(
            dataset="voc",
            attempt="attempt",
            records=records,
            output=tmp_path / "incomplete.json",
        )
