from __future__ import annotations

import json
from pathlib import Path

from defer_yolo_ladder_queue import canonical_hash, valid_calibration
from topic_c.manifest import sha256_file


def calibration_document(dataset: str = "voc") -> dict:
    document = {
        "schema_version": 1,
        "dataset": dataset,
        "split": "train",
        "n_images": 512,
        "records": [
            {"source_relpath": f"images/train/{index:06d}.jpg", "sha256": f"{index:064x}"}
            for index in range(512)
        ],
    }
    document["calibration_sha256"] = canonical_hash(document, "calibration_sha256")
    return document


def write_artifact(root: Path, document: dict, marker: str | None = None) -> Path:
    output = root / "calibration.json"
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(".json.complete").write_text(
        (marker if marker is not None else document["calibration_sha256"]) + "\n",
        encoding="utf-8",
    )
    return output


def test_accepts_canonical_calibration_marker(tmp_path: Path) -> None:
    output = write_artifact(tmp_path, calibration_document())
    assert valid_calibration(output, "voc")


def test_rejects_whole_file_marker_for_self_hashed_calibration(tmp_path: Path) -> None:
    document = calibration_document()
    output = write_artifact(tmp_path, document)
    output.with_suffix(".json.complete").write_text(sha256_file(output) + "\n", encoding="utf-8")
    assert not valid_calibration(output, "voc")


def test_rejects_mutated_or_wrong_dataset_calibration(tmp_path: Path) -> None:
    document = calibration_document()
    output = write_artifact(tmp_path, document)
    document["records"][0]["source_relpath"] = "images/train/changed.jpg"
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    assert not valid_calibration(output, "voc")
    assert not valid_calibration(output, "kitti")
