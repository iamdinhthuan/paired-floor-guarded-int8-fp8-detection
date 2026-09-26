import copy

import numpy as np
import pytest

from run_nn_preprocessing_pilot import (
    calibration_subset,
    clean_gate,
    treatment_quantize_options,
    write_once,
)
from topic_c.manifest import sha256_file


def test_calibration_subset_is_seeded_unique_and_train_only():
    source = {"dataset": "kitti", "split": "train", "n_images": 20,
              "records": [{"source_relpath": str(i), "sha256": str(i)} for i in range(20)],
              "calibration_sha256": "old"}
    original = copy.deepcopy(source)
    first = calibration_subset(source, 8, 20260923, "parent")
    assert first == calibration_subset(source, 8, 20260923, "parent")
    assert source == original
    assert len(first["records"]) == first["n_images"] == 8
    assert len({r["source_relpath"] for r in first["records"]}) == 8
    assert first["calibration_sha256"] != "old"
    assert first["parent_manifest_sha256"] == "parent"
    source["split"] = "val"
    with pytest.raises(ValueError, match="train"):
        calibration_subset(source, 8, 1, "parent")


@pytest.mark.parametrize("count", [0, 21])
def test_calibration_subset_rejects_invalid_count(count):
    with pytest.raises(ValueError):
        calibration_subset({"split": "train", "n_images": 20,
                            "records": [{"source_relpath": str(i)} for i in range(20)]}, count, 1, "p")


def test_clean_gate_requires_both_formats_and_usable_reference():
    policy = {"maximum_clean_reference_loss_ap_points": 2., "minimum_reference_ap_points": 30.}
    values = {"fp32": 50., "int8_legacy_calibration": 7.,
              "int8_matched_calibration": 49., "fp8_matched_calibration": 49.5}
    assert clean_gate(values, policy)["passed"]
    values["int8_matched_calibration"] = 14.
    assert not clean_gate(values, policy)["passed"]
    values.update(fp32=2., int8_matched_calibration=1., fp8_matched_calibration=1.)
    assert not clean_gate(values, policy)["passed"]
    values["fp32"] = np.nan
    with pytest.raises(ValueError):
        clean_gate(values, policy)


def test_treatment_quantize_options_are_explicit_and_ordered(tmp_path):
    assert treatment_quantize_options({
        "preprocessing": "retinanet_normalized",
        "quantize_op_types": ["Conv", "Add"],
    }, tmp_path) == ["--quantize-op-types", "Conv,Add"]
    assert treatment_quantize_options({
        "preprocessing": "yolo_letterbox",
        "quantize_op_types": "Conv",
    }, tmp_path) == ["--quantize-op-types", "Conv", "--allow-preprocessing-mismatch"]
    assert treatment_quantize_options({"preprocessing": "retinanet_normalized"}, tmp_path) == []


def test_treatment_quantize_options_bind_node_mask_by_hash(tmp_path):
    mask = tmp_path / "mask.json"
    mask.write_bytes(b"frozen-mask")
    digest = sha256_file(mask)
    treatment = {"preprocessing": "retinanet_normalized",
                 "node_mask": "mask.json", "node_mask_sha256": digest}
    assert treatment_quantize_options(treatment, tmp_path) == ["--node-mask", str(mask)]
    treatment["node_mask_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="changed"):
        treatment_quantize_options(treatment, tmp_path)


def test_outputs_cannot_overwrite_existing_evidence(tmp_path):
    path = tmp_path / "result.json"
    write_once(path, {"value": 1})
    original = path.read_bytes()
    with pytest.raises(FileExistsError):
        write_once(path, {"value": 2})
    assert path.read_bytes() == original
