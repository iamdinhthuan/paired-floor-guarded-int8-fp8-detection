from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest

from pilot_registry import calibration_sha256, canonical_hash


def test_calibration_sha256_accepts_valid_self_hashed_document(tmp_path: Path) -> None:
    document = {"schema_version": 1, "dataset": "voc", "records": [{"source_relpath": "image.jpg"}]}
    document["calibration_sha256"] = canonical_hash(document, "calibration_sha256")
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")

    assert calibration_sha256(path) == document["calibration_sha256"]
    assert calibration_sha256(path) != hashlib.sha256(path.read_bytes()).hexdigest()


def test_calibration_sha256_rejects_mutated_self_hashed_document(tmp_path: Path) -> None:
    document = {"schema_version": 1, "dataset": "voc", "records": []}
    document["calibration_sha256"] = canonical_hash(document, "calibration_sha256")
    document["dataset"] = "kitti"
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="calibration SHA-256 mismatch"):
        calibration_sha256(path)


def test_calibration_sha256_keeps_legacy_whole_file_fallback(tmp_path: Path) -> None:
    path = tmp_path / "legacy-calibration.txt"
    path.write_text("/data/images/one.jpg\n/data/images/two.jpg\n", encoding="utf-8")

    assert calibration_sha256(path) == hashlib.sha256(path.read_bytes()).hexdigest()
