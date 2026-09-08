import copy
import json
import math

import pytest

import build_codec_interaction as audit


DATASETS = ("coco", "voc", "kitti", "tt100k")
MODELS = ("yolo11n", "yolo11m", "yolo11x")
CORRUPTIONS = ("gaussian_noise", "motion_blur", "fog", "jpeg")


def synthetic_rows():
    codec, cells = [], []
    effects = {
        "gaussian_noise": 0.01,
        "motion_blur": 0.04,
        "fog": 0.03,
        "jpeg": 0.0,
    }
    for dataset in DATASETS:
        for model in MODELS:
            codec.extend([
                dict(dataset=dataset, model=model, precision="fp32",
                     original_clean_ap=0.60, codec_clean_ap=0.60,
                     codec_minus_original=0.0),
                dict(dataset=dataset, model=model, precision="int8-entropy",
                     original_clean_ap=0.50, codec_clean_ap=0.51,
                     codec_minus_original=1.0),
                dict(dataset=dataset, model=model, precision="fp8",
                     original_clean_ap=0.55, codec_clean_ap=0.53,
                     codec_minus_original=-2.0),
            ])
            for corruption in CORRUPTIONS:
                for severity in (1, 3, 5):
                    cells.append(dict(
                        dataset=dataset, model=model, corruption=corruption,
                        severity=severity, endpoint_type="area", n_images=100,
                        delta_q_all=0.02, delta_e_all=effects[corruption],
                    ))
    return codec, cells


def test_hand_derived_control_substitution_and_sign_counts():
    codec, direct = synthetic_rows()
    summary, blocks, cells = audit.analyze(codec, direct)

    assert len(blocks) == 12
    assert len(cells) == 144
    assert all(row["delta_e_j95_minus_original_ap"] == pytest.approx(3.0)
               for row in blocks)
    assert summary["macro"]["delta_e_j95_mean_ap"] == pytest.approx(2.0)
    assert summary["macro"]["delta_e_original_mean_ap"] == pytest.approx(-1.0)
    assert summary["macro"]["j95_minus_original_mean_ap"] == pytest.approx(3.0)
    assert summary["sign_inventory"] == {
        "cells": 144,
        "j95_positive": 108,
        "j95_negative": 0,
        "j95_zero": 36,
        "original_positive": 36,
        "original_negative": 72,
        "original_zero": 36,
        "strict_nonzero_both": 72,
        "strict_sign_disagreements": 36,
        "j95_positive_to_original_negative": 36,
        "j95_negative_to_original_positive": 0,
    }
    gaussian = next(row for row in cells if row["dataset"] == "coco"
                    and row["model"] == "yolo11n"
                    and row["corruption"] == "gaussian_noise"
                    and row["severity"] == 1)
    assert gaussian["delta_e_j95_ap"] == pytest.approx(1.0)
    assert gaussian["delta_e_original_ap"] == pytest.approx(-2.0)
    assert gaussian["corrupted_gap_reconstructed_ap"] == pytest.approx(3.0)
    assert gaussian["four_arm_identity_error_ap"] == pytest.approx(0.0)


@pytest.mark.parametrize("which", ["codec", "direct"])
def test_complete_unique_grids_are_required(which):
    codec, direct = synthetic_rows()
    rows = codec if which == "codec" else direct
    rows.pop()
    with pytest.raises(ValueError, match="complete unique"):
        audit.analyze(codec, direct)

    codec, direct = synthetic_rows()
    rows = codec if which == "codec" else direct
    rows[-1] = copy.deepcopy(rows[0])
    with pytest.raises(ValueError, match="complete unique"):
        audit.analyze(codec, direct)


@pytest.mark.parametrize(
    ("target", "field", "value", "message"),
    [
        ("codec", "original_clean_ap", math.nan, "finite"),
        ("codec", "codec_clean_ap", 1.01, "native AP"),
        ("direct", "delta_e_all", math.inf, "finite"),
        ("direct", "delta_e_all", 1.01, "native AP"),
    ],
)
def test_nonfinite_and_scale_errors_are_rejected(target, field, value, message):
    codec, direct = synthetic_rows()
    rows = codec if target == "codec" else direct
    rows[0][field] = value
    with pytest.raises(ValueError, match=message):
        audit.analyze(codec, direct)


def test_codec_delta_identity_error_is_rejected():
    codec, direct = synthetic_rows()
    codec[1]["codec_minus_original"] = 0.5
    with pytest.raises(ValueError, match="codec-minus-original"):
        audit.analyze(codec, direct)


def test_j95_clean_gap_identity_error_is_rejected():
    codec, direct = synthetic_rows()
    direct[0]["delta_q_all"] = 0.03
    with pytest.raises(ValueError, match="J95 clean-gap"):
        audit.analyze(codec, direct)


def test_component_status_never_turns_missing_or_mismatch_into_verified():
    assert audit.identity_status("same", "same") == "verified_same"
    assert audit.identity_status("same", "different") == "verified_mismatch"
    assert audit.identity_status("same", None) == "unknown"


def test_metric_index_filters_a_shared_directory_by_dataset(tmp_path):
    common = {
        "model": "yolo11n", "precision": "int8-entropy",
        "corruption": "codec_control", "severity": 0,
    }
    (tmp_path / "coco.json").write_text(json.dumps({**common, "dataset": "coco"}))
    (tmp_path / "voc.json").write_text(json.dumps({**common, "dataset": "voc"}))

    indexed = audit._metric_index(tmp_path, "coco", "codec_control")

    assert list(indexed) == [("yolo11n", "int8-entropy")]
    assert indexed[("yolo11n", "int8-entropy")][1]["dataset"] == "coco"
