"""Behavioral regression tests for the read-only TT100K annotation audit."""
import importlib.util
import json
from pathlib import Path

import pytest


MODULE = Path(__file__).resolve().parents[1] / "analysis/audit_v4_tt100k_annotations.py"


def audit_module():
    assert MODULE.is_file(), "TT100K audit implementation is missing"
    spec = importlib.util.spec_from_file_location("v4_tt100k_audit", MODULE)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def record(objects):
    return {"id": 7, "path": "other/7.jpg", "objects": objects}


def obj(category="keep", bbox=None):
    return {"category": category, "bbox": bbox or {"xmin": 10, "ymin": 20, "xmax": 30, "ymax": 60}}


@pytest.mark.parametrize("source,label,want", [
    (record([]), None, "valid_negative"),
    (record([obj()]), "0 0.200000 0.200000 0.200000 0.200000\n", "positive_retained_verified"),
    (record([obj("excluded")]), None, "explicitly_excluded_classes"),
    (record([obj()]), None, "positive_missing_converted_label"),
    ({"id": 8, "path": "other/7.jpg", "objects": []}, None, "invalid_image_id"),
    (None, None, "missing_original_annotation"),
    ({"id": 7, "path": "other/7.jpg"}, None, "missing_object_list"),
    (record([obj()]), "0 0.200000 0.300000 0.200000 0.200000\n", "annotation_conversion_mismatch"),
    (record([]), "0 0.2 0.2 0.2 0.2\n", "annotation_conversion_mismatch"),
])
def test_classifies_evidence_without_treating_absent_annotation_as_negative(source, label, want):
    result = audit_module().classify_image("7", source, {"keep": 0}, 100, 200, label)
    assert result["classification"] == want


def test_converter_clips_center_and_size_separately_then_rounds_six_decimals():
    source = record([obj(bbox={"xmin": -10, "ymin": 20, "xmax": 30, "ymax": 60})])
    result = audit_module().classify_image("7", source, {"keep": 0}, 100, 200,
                                         "0 0.100000 0.200000 0.400000 0.200000\n")
    assert result["classification"] == "positive_retained_verified"
    assert result["expected_rows"] == [[0, 0.1, 0.2, 0.4, 0.2]]


def test_coco_conversion_uses_clipped_edges_area_and_one_based_category():
    result = audit_module().coco_rows([[0, .1, .2, .4, .2]], 100, 200)
    assert result[0]["category_id"] == 1
    assert result[0]["bbox"] == pytest.approx([0, 20, 30, 40])
    assert result[0]["area"] == pytest.approx(1200)
    assert result[0]["iscrowd"] == 0


def test_invalid_label_is_a_reported_failure_not_an_exception():
    result = audit_module().classify_image("7", record([obj()]), {"keep": 0}, 100, 200, "0 nan .2 .2 .2\n")
    assert result["classification"] == "annotation_conversion_mismatch"


def test_end_to_end_audit_keeps_native_selection_and_final_distinct(tmp_path):
    from PIL import Image
    import yaml
    root = tmp_path
    data = root / "data/datasets/TT100K"
    for split, original, number in [("train", "train", 1), ("val", "other", 7), ("test", "test", 9)]:
        for directory in [data / "data" / original, data / "images" / split, data / "labels" / split]:
            directory.mkdir(parents=True, exist_ok=True)
        Image.new("RGB", (100, 200)).save(data / "data" / original / f"{number}.jpg")
        (data / "images" / split / f"{number}.jpg").write_bytes((data / "data" / original / f"{number}.jpg").read_bytes())
    original = {"imgs": {"1": {"id": 1, "path": "train/1.jpg", "objects": [obj()]},
                         "7": record([]), "9": {"id": 9, "path": "test/9.jpg", "objects": [obj()]}}, "types": ["keep"]}
    (data / "data/annotations.json").write_text(json.dumps(original))
    for split, number in [("train", 1), ("test", 9)]:
        (data / "labels" / split / f"{number}.txt").write_text("0 .2 .2 .2 .2\n")
    config = root / "configs/datasets/tt100k_ultralytics_v1.yaml"
    config.parent.mkdir(parents=True)
    config.write_text(yaml.safe_dump({"path": str(data), "names": {0: "keep"}, "train": "images/train", "val": "images/val", "test": "images/test", "download": "# fixture"}))
    report = audit_module().run_audit(root)
    assert report["splits"]["val"]["classifications"] == {"valid_negative": 1}
    assert report["splits"]["train"]["classifications"] == {"positive_retained_verified": 1}
    assert report["splits"]["test"]["classifications"] == {"positive_retained_verified": 1}
    assert report["ledger"][0]["source_sha256"] == report["ledger"][0]["converted_image_sha256"]
    assert report["final_evaluation"]["status"] == "missing"


@pytest.mark.parametrize("mutation", [None, "hash", "area", "class", "id", "extra_image"])
def test_final_audit_checks_frozen_identity_geometry_class_and_membership(tmp_path, mutation):
    import hashlib
    module = audit_module()
    anno = {"images": [{"id": 1, "file_name": "images/test/7.jpg", "width": 100, "height": 200}],
            "annotations": [{"image_id": 1, "category_id": 1, "bbox": [25, 50, 50, 100], "area": 5000, "iscrowd": 0}],
            "categories": [{"id": 1, "name": "keep"}]}
    manifest = {"expected_image_ids": [1], "records": [{"image_id": 1, "source_relpath": "images/test/7.jpg",
                                                        "source_sha256": "abc", "sha256": "abc"}]}
    if mutation == "hash":
        manifest["records"][0]["source_sha256"] = "wrong"
    if mutation == "area":
        anno["annotations"][0]["area"] = 10000
    if mutation == "class":
        anno["annotations"][0]["category_id"] = 2
    if mutation == "id":
        manifest["expected_image_ids"] = [2]
    ledger = [{"split": "test", "converted_relpath": "images/test/7.jpg", "width": 100, "height": 200,
               "expected_rows": [[0, .5, .5, .5, .5]], "converted_image_sha256": "abc"}]
    if mutation == "extra_image":
        ledger.append({**ledger[0], "converted_relpath": "images/test/99.jpg"})
    anno_bytes = json.dumps(anno).encode()
    class_map = {"class_names": ["keep"], "class_to_category_id": [1], "annotation_sha256": hashlib.sha256(anno_bytes).hexdigest()}
    for name, document in [("annotations/tt100k_test_ultralytics_v1_coco.json", anno),
                           ("images/tt100k_test_clean_ultralytics_v1.json", manifest),
                           ("classes/tt100k_test_ultralytics_v1.json", class_map)]:
        path = tmp_path / "manifests" / name
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(document))
    report = module.audit_final(tmp_path, ledger, ["keep"])
    assert report["status"] == ("verified" if mutation is None else "mismatch")


def test_source_split_lists_do_not_classify_unannotated_jpegs_as_negatives(tmp_path):
    module = audit_module()
    assert hasattr(module, "audit_source_lists"), "Source split membership audit missing"
    directory = tmp_path / "data/train"
    directory.mkdir(parents=True)
    (directory / "ids.txt").write_text("1\n2\n")
    (directory / "2.jpg").write_bytes(b"source-only image")
    original = {"imgs": {"1": {"id": 1, "path": "train/1.jpg", "objects": []}}}
    result = module.audit_source_lists(tmp_path, original)
    assert result["train"]["listed_images"] == 2
    assert result["train"]["listed_without_original_annotation"][0]["original_image_id"] == "2"
    assert result["train"]["listed_without_original_annotation"][0]["classification"] == "missing_original_annotation_not_converted"
    assert result["train"]["listed_without_original_annotation"][0]["source_sha256"] is not None
    assert result["test"]["status"] == "missing"
