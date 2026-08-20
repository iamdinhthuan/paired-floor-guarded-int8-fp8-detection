from __future__ import annotations

import hashlib
import json
from pathlib import Path

import rebind_subset_class_map as rebind


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_rebound_map_preserves_mapping_and_binds_subset_annotation(tmp_path: Path) -> None:
    annotation = tmp_path / "subset.json"
    annotation.write_text("{}\n", encoding="utf-8")
    source = {
        "schema_version": 1,
        "dataset": "voc",
        "split": "test",
        "annotation_sha256": "old",
        "class_to_category_id": [1, 2, 3],
    }
    source["class_map_sha256"] = rebind.canonical(source)

    result = rebind.rebound_document(source, annotation)

    assert result["class_to_category_id"] == [1, 2, 3]
    assert result["annotation_sha256"] == digest(annotation)
    assert result["class_map_sha256"] == rebind.canonical(result)
