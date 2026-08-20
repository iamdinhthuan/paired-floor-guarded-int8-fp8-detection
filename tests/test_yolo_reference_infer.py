from __future__ import annotations

import hashlib
import json
from pathlib import Path

import numpy as np
import pytest

import yolo_reference_infer as reference


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


class FakeTensor:
    def __init__(self, values: np.ndarray) -> None:
        self.values = values

    def detach(self) -> "FakeTensor":
        return self

    def cpu(self) -> "FakeTensor":
        return self

    def numpy(self) -> np.ndarray:
        return self.values


def test_normalize_yolo_output_accepts_pytorch_tuple_and_preserves_values() -> None:
    values = np.arange(24 * 10, dtype=np.float32).reshape(1, 24, 10)

    normalized = reference.normalize_yolo_output((FakeTensor(values), ["aux"]))

    assert normalized.dtype == np.float32
    assert np.array_equal(normalized, values)


@pytest.mark.parametrize(
    "output",
    [
        np.zeros((2, 24, 10), dtype=np.float32),
        np.zeros((1, 4, 10), dtype=np.float32),
        np.zeros((1, 24, 10, 1), dtype=np.float32),
        {"prediction": np.zeros((1, 24, 10), dtype=np.float32)},
    ],
)
def test_normalize_yolo_output_rejects_noncanonical_shapes(output: object) -> None:
    with pytest.raises(RuntimeError, match="canonical YOLO output"):
        reference.normalize_yolo_output(output)


def test_onnxruntime_provider_is_cuda_only_and_never_silently_falls_back() -> None:
    assert reference.ort_providers(["CPUExecutionProvider", "CUDAExecutionProvider"]) == [
        "CUDAExecutionProvider"
    ]
    with pytest.raises(SystemExit, match="CUDAExecutionProvider"):
        reference.ort_providers(["CPUExecutionProvider"])


def test_onnxruntime_session_accepts_registered_cpu_ep_but_requires_cuda_first() -> None:
    # ORT automatically registers CPUExecutionProvider after the requested CUDA
    # provider. This is not evidence that numerical graph nodes fell back.
    assert reference.validate_ort_session_providers(
        ["CUDAExecutionProvider", "CPUExecutionProvider"]
    ) == ["CUDAExecutionProvider", "CPUExecutionProvider"]
    with pytest.raises(SystemExit, match="CUDA is not the primary"):
        reference.validate_ort_session_providers(
            ["CPUExecutionProvider", "CUDAExecutionProvider"]
        )


def test_complete_export_registry_binds_checkpoint_and_onnx_bytes(tmp_path: Path) -> None:
    checkpoint, onnx = tmp_path / "model.pt", tmp_path / "model.onnx"
    checkpoint.write_bytes(b"checkpoint")
    onnx.write_bytes(b"onnx")
    registry = tmp_path / "export.json"
    registry.write_text(
        json.dumps(
            {
                "dataset": "voc",
                "model": "yolo11n",
                "source_checkpoint": str(checkpoint),
                "source_checkpoint_sha256": digest(checkpoint),
                "onnx": str(onnx),
                "onnx_sha256": digest(onnx),
                "input_name": "images",
                "output_name": "output0",
            }
        )
        + "\n",
        encoding="utf-8",
    )
    registry.with_suffix(".json.complete").write_text(digest(registry) + "\n", encoding="utf-8")

    document, artifacts = reference.load_export_registry(registry, "voc", "yolo11n")

    assert document["input_name"] == "images"
    assert artifacts == {"pytorch": checkpoint.resolve(), "onnxruntime": onnx.resolve()}
    onnx.write_bytes(b"mutated")
    with pytest.raises(SystemExit, match="ONNX hash mismatch"):
        reference.load_export_registry(registry, "voc", "yolo11n")


def test_reference_record_binds_backend_model_and_common_input_hashes(tmp_path: Path) -> None:
    model = tmp_path / "model.onnx"
    model.write_bytes(b"model")
    registry = tmp_path / "registry.json"
    registry.write_text("{}\n", encoding="utf-8")

    record = reference.build_run_record(
        condition_id="voc_final__yolo11n__onnxruntime__clean-s0",
        dataset="voc",
        split="final",
        model="yolo11n",
        backend="onnxruntime",
        model_artifact=model,
        export_registry=registry,
        annotation_sha256="a" * 64,
        manifest_sha256="b" * 64,
        image_ids_sha256="c" * 64,
        prediction_sha256="d" * 64,
        n_images=12,
        n_detections=34,
        runtime_seconds=1.5,
        runtime={"provider": "CUDAExecutionProvider"},
        class_map_sha256="e" * 64,
    )

    assert record["precision"] == "fp32-reference"
    assert record["backend"] == "onnxruntime"
    assert record["model_artifact_sha256"] == digest(model)
    assert record["export_registry_sha256"] == digest(registry)
    assert record["input_manifest_sha256"] == "b" * 64
