import hashlib
import json
import os
import subprocess
import sys
from pathlib import Path

import pytest
import numpy as np

from topic_c.coco_data import load_coco_images
from topic_c.manifest import ManifestError, manifest_sha256, validate_manifest, validate_shared_input_bytes
from topic_c.yolo_decode import decode


def _write_manifest(path, records):
    document = {"schema_version": 1, "dataset": "coco", "split": "val2017", "expected_image_ids": [record["image_id"] for record in records], "records": records}
    document["manifest_sha256"] = manifest_sha256(document)
    path.write_text(json.dumps(document))
    return document


def test_manifest_loader_preserves_official_ids_with_alternate_bytes(tmp_path):
    annotations = tmp_path / "instances.json"
    annotations.write_text(json.dumps({"images": [{"id": 7, "file_name": "a.jpg"}, {"id": 11, "file_name": "b.jpg"}]}))
    cache = tmp_path / "cache"
    cache.mkdir()
    records = []
    for image_id, name in [(7, "alternate/a.jpg"), (11, "alternate/b.jpg")]:
        destination = cache / name
        destination.parent.mkdir(exist_ok=True)
        destination.write_bytes(str(image_id).encode())
        records.append({"dataset": "coco", "image_id": image_id, "source_relpath": "a.jpg" if image_id == 7 else "b.jpg", "source_sha256": "unused_by_loader", "corruption": "fog", "severity": 3, "seed": image_id, "output_relpath": name, "sha256": hashlib.sha256(destination.read_bytes()).hexdigest(), "generator": "test", "generator_version": "1"})
    manifest = tmp_path / "fog.json"
    _write_manifest(manifest, records)
    assert load_coco_images(str(annotations), image_manifest=str(manifest), manifest_cache_root=str(cache)) == [(7, str((cache / "alternate/a.jpg").resolve())), (11, str((cache / "alternate/b.jpg").resolve()))]


def test_loader_rejects_ambiguous_image_source(tmp_path):
    annotations = tmp_path / "instances.json"
    annotations.write_text(json.dumps({"images": [{"id": 7, "file_name": "a.jpg"}]}))
    with pytest.raises(ManifestError, match="exactly one"):
        load_coco_images(str(annotations), image_root=str(tmp_path), image_manifest="x", manifest_cache_root=str(tmp_path))


def test_validation_catches_duplicate_ids_and_hash_mismatch(tmp_path):
    annotations = tmp_path / "instances.json"
    annotations.write_text(json.dumps({"images": [{"id": 7, "file_name": "a.jpg"}]}))
    cache = tmp_path / "cache"
    cache.mkdir()
    record = {"dataset": "coco", "image_id": 7, "source_relpath": "a.jpg", "source_sha256": "bad", "corruption": "clean", "severity": 0, "seed": 0, "output_relpath": "a.jpg", "sha256": "bad", "generator": "test", "generator_version": "1"}
    document = {"records": [record, dict(record)], "expected_image_ids": [7, 7]}
    document["manifest_sha256"] = manifest_sha256(document)
    failures = validate_manifest(document, annotations, cache, cache, require_pixels_changed=False)
    assert any("duplicate image_id" in failure for failure in failures)
    assert any("cached file absent" in failure for failure in failures)


def test_validation_binds_each_manifest_record_to_clean_source_bytes(tmp_path):
    annotations = tmp_path / "instances.json"
    annotations.write_text(json.dumps({"images": [{"id": 7, "file_name": "a.jpg"}]}))
    clean, cache = tmp_path / "clean", tmp_path / "cache"
    clean.mkdir(); cache.mkdir()
    (clean / "a.jpg").write_bytes(b"clean-bytes")
    (cache / "a.jpg").write_bytes(b"clean-bytes")
    record = {"dataset": "coco", "image_id": 7, "source_relpath": "a.jpg",
              "source_sha256": hashlib.sha256(b"clean-bytes").hexdigest(), "corruption": "clean", "severity": 0,
              "seed": 0, "output_relpath": "a.jpg", "sha256": hashlib.sha256(b"clean-bytes").hexdigest(),
              "generator": "test", "generator_version": "1"}
    document = {"records": [record], "expected_image_ids": [7]}
    document["manifest_sha256"] = manifest_sha256(document)
    assert validate_manifest(document, annotations, cache, clean, require_pixels_changed=False) == []
    record["source_sha256"] = "0" * 64
    document["manifest_sha256"] = manifest_sha256(document)
    assert any("clean source SHA-256 mismatch" in failure for failure in validate_manifest(document, annotations, cache, clean, require_pixels_changed=False))


def test_registry_requires_one_input_manifest_per_condition():
    records = [
        {"dataset": "coco", "split": "val2017", "corruption": "fog", "severity": 3, "input_manifest_sha256": "a"},
        {"dataset": "coco", "split": "val2017", "corruption": "fog", "severity": 3, "input_manifest_sha256": "b"},
    ]
    assert validate_shared_input_bytes(records)


def test_registry_accepts_clean_severity_zero():
    record = {"dataset": "coco", "split": "val2017", "corruption": "clean", "severity": 0, "input_manifest_sha256": "clean-hash"}
    assert validate_shared_input_bytes([record]) == []


def test_decoder_drops_box_inverted_by_frame_clipping():
    # Raw xywh is wholly left of the image; clipping would otherwise make x2 < x1.
    output = np.array([[[-10.0], [10.0], [1.0], [2.0], [0.9]]], dtype=np.float32)
    assert decode(output, confidence=0.001, gain=1.0, padx=0.0, pady=0.0, width=100, height=100) == []


def test_tt100k_paired_bootstrap_uses_original_height_small_and_large_like_bins(tmp_path):
    pytest.importorskip("pycocotools")
    annotations = tmp_path / "annotations.json"
    annotations.write_text(json.dumps({"images": [{"id": 1, "file_name": "a.jpg", "width": 100, "height": 100}, {"id": 2, "file_name": "b.jpg", "width": 100, "height": 120}],
                                       "categories": [{"id": 1, "name": "sign"}],
                                       "annotations": [{"id": 1, "image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 10], "area": 200, "iscrowd": 0},
                                                       {"id": 2, "image_id": 2, "category_id": 1, "bbox": [10, 10, 20, 100], "area": 2000, "iscrowd": 0}]}))
    ids = [1, 2]
    ids_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
    inputs = []
    for name in ("fp32_clean", "quant_clean", "fp32_corrupt", "quant_corrupt"):
        path = tmp_path / f"{name}_input.json"
        path.write_text(json.dumps({"image_ids": ids, "image_ids_sha256": ids_hash, "input_manifest_sha256": f"manifest-{name}"}))
        inputs.append(path)
    predictions = []
    payloads = [
        [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 10], "score": 0.9}, {"image_id": 2, "category_id": 1, "bbox": [10, 10, 20, 100], "score": 0.9}],
        [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 10], "score": 0.9}, {"image_id": 2, "category_id": 1, "bbox": [10, 10, 20, 100], "score": 0.9}],
        [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 10], "score": 0.9}, {"image_id": 2, "category_id": 1, "bbox": [10, 10, 20, 100], "score": 0.9}],
        [{"image_id": 1, "category_id": 1, "bbox": [10, 10, 20, 10], "score": 0.9}, {"image_id": 2, "category_id": 1, "bbox": [10, 10, 20, 100], "score": 0.9}],
    ]
    for name, payload in zip(("fp32_clean", "quant_clean", "fp32_corrupt", "quant_corrupt"), payloads):
        path = tmp_path / f"{name}.json"; path.write_text(json.dumps(payload)); predictions.append(path)
    output = tmp_path / "bootstrap.json"
    environment = dict(os.environ); environment["PYTHONPATH"] = str(Path(__file__).parents[1] / "src")
    subprocess.run([sys.executable, str(Path(__file__).parents[1] / "src" / "paired_bootstrap_tt100k.py"), "--annotations", str(annotations),
                    "--fp32-clean", str(predictions[0]), "--quant-clean", str(predictions[1]), "--fp32-corrupt", str(predictions[2]), "--quant-corrupt", str(predictions[3]),
                    "--fp32-clean-input", str(inputs[0]), "--quant-clean-input", str(inputs[1]), "--fp32-corrupt-input", str(inputs[2]), "--quant-corrupt-input", str(inputs[3]),
                    "--quant-label", "fp8", "--n-boot", "2", "--seed", "7", "--out", str(output)], check=True, env=environment, capture_output=True, text=True)
    result = json.loads(output.read_text())
    assert result["n_images"] == 2 and result["n_boot"] == 2
    assert result["height_bins_px"]["small_like"] == {"min": 0.0, "max": 24.0}
    assert result["height_bins_px"]["large_like"] == {"min": 48.0, "max": None}
