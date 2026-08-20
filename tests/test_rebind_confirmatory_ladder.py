from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

import rebind_confirmatory_ladder as rebind


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def configs() -> tuple[dict, dict]:
    old = {
        "schema_version": 1,
        "attempt": "attempt",
        "models": ["yolo11n", "yolo11m", "yolo11x"],
        "precisions": ["fp32", "fp16", "int8-entropy", "fp8"],
        "trt_root": "/opt/trt",
        "workspace": "4096M",
        "datasets": {
            "voc": {"split": "final", "imgsz": 640, "calibration": "voc.json"},
            "kitti": {"split": "final", "imgsz": 640, "calibration": "kitti.json"},
        },
        "config_sha256": "old",
    }
    new = json.loads(json.dumps(old))
    new["schema_version"] = 2
    new["supersedes_execution_config"] = "old.json"
    new["supersedes_execution_config_sha256"] = "a" * 64
    new["correction"] = "machine split key correction"
    new["config_sha256"] = "new"
    for item in new["datasets"].values():
        item["split"] = "test"
        item["partition_role"] = "final"
    return old, new


def test_metadata_correction_allows_only_split_key_and_partition_role() -> None:
    old, new = configs()

    rebind.validate_metadata_only_correction(old, new)

    new["workspace"] = "8192M"
    with pytest.raises(SystemExit, match="build-affecting"):
        rebind.validate_metadata_only_correction(old, new)


def test_source_ladder_artifacts_require_all_24_registry_hashes(tmp_path: Path) -> None:
    artifacts = []
    for dataset in ("voc", "kitti"):
        for model in ("yolo11n", "yolo11m", "yolo11x"):
            for precision in ("fp32", "fp16", "int8-entropy", "fp8"):
                onnx = tmp_path / f"{dataset}_{model}_{precision}.onnx.json"
                engine = tmp_path / f"{dataset}_{model}_{precision}.engine.json"
                onnx.write_text("{}\n", encoding="utf-8")
                engine.write_text("{}\n", encoding="utf-8")
                artifacts.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "precision": precision,
                        "onnx_registry": str(onnx),
                        "onnx_registry_sha256": digest(onnx),
                        "engine_registry": str(engine),
                        "engine_registry_sha256": digest(engine),
                    }
                )
    registries = []
    for dataset in ("voc", "kitti"):
        path = tmp_path / f"{dataset}_ladder.json"
        path.write_text("{}\n", encoding="utf-8")
        path.with_suffix(path.suffix + ".complete").write_text(
            digest(path) + "\n", encoding="utf-8"
        )
        registries.append({"dataset": dataset, "path": str(path), "sha256": digest(path)})
    report = {"artifacts": artifacts, "dataset_registries": registries}

    rebind.validate_source_artifacts(report)

    Path(artifacts[0]["engine_registry"]).write_text("mutated\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="artifact hash mismatch"):
        rebind.validate_source_artifacts(report)
