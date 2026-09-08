from pathlib import Path
import json

import pytest
import run_tide_error_decomposition as tide_module

from run_tide_error_decomposition import (
    bind_source_condition,
    expected_primary_identities,
    normalize_precision,
    paired_interactions,
    parse_prediction_name,
    summarize_interactions,
)


def test_prediction_filename_parser_normalizes_precision() -> None:
    parsed = parse_prediction_name(
        Path("voc_val__yolo11m__int8-entropy__motion_blur-s5__abc__def.json")
    )
    assert parsed == {
        "image_universe": "voc_val",
        "model": "yolo11m",
        "precision": "int8",
        "corruption": "motion_blur",
        "severity": 5,
        "precision_token": "int8-entropy",
    }
    assert normalize_precision("fp8-none") == "fp8"


def test_prediction_filename_parser_rejects_invalid_clean_severity() -> None:
    with pytest.raises(ValueError, match="clean/severity"):
        parse_prediction_name(Path("voc_val__yolo11m__fp8-none__clean-s3__abc__def.json"))


def test_prediction_filename_parser_accepts_codec_matched_control() -> None:
    parsed = parse_prediction_name(
        Path("voc_val__yolo11m__fp8__codec-control-s0__abc__def.json")
    )
    assert parsed["corruption"] == "codec-control"
    assert parsed["severity"] == 0


def test_source_binding_uses_codec_control_and_rejects_original_clean() -> None:
    codec = parse_prediction_name(
        Path("voc_val__yolo11m__fp8__codec-control-s0__abc__def.json")
    )
    bound = bind_source_condition(
        codec, source_kind="matched_clean", corruptions={"fog"}
    )
    assert bound is not None
    assert bound["corruption"] == "clean"
    assert bound["source_corruption"] == "codec_control"
    assert bound["control_materialization"] == "deterministic_jpeg95"

    source_clean = parse_prediction_name(
        Path("voc_val__yolo11m__fp8-none__clean-s0__abc__def.json")
    )
    assert bind_source_condition(
        source_clean, source_kind="corrupted", corruptions={"fog"}
    ) is None


def _record(precision: str, corruption: str, severity: int, ap: float, offset: float):
    errors = {name: offset + index for index, name in enumerate(("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss"))}
    return {
        "dataset": "voc",
        "model": "yolo11m",
        "precision": precision,
        "corruption": corruption,
        "severity": severity,
        "tide_ap50": ap,
        "cocoeval_ap50": ap + 0.25,
        "main_error_dap50": errors,
        "main_error_count_per_100_gt": errors,
        "input_image_ids_sha256": "1" * 64,
        "annotations_sha256": "2" * 64,
        "engine_sha256": ("a" if precision == "int8" else "b") * 64,
        "runner_sha256": "c" * 64,
        "preprocess_sha256": "d" * 64,
        "decoder_sha256": "e" * 64,
        "class_map_sha256": "builtin:test",
        "input_manifest_sha256": ("3" if corruption == "clean" else "4") * 64,
        "n_images": 10,
        "n_ground_truth": 20,
    }


def test_paired_error_interactions_follow_paper_sign_convention() -> None:
    records = [
        _record("int8", "clean", 0, 60.0, 3.0),
        _record("fp8", "clean", 0, 62.0, 2.0),
        _record("int8", "fog", 5, 30.0, 8.0),
        _record("fp8", "fog", 5, 35.0, 4.0),
    ]
    rows = paired_interactions(records)
    assert len(rows) == 1
    row = rows[0]
    assert row["fp8_minus_int8_tide_ap50_clean"] == pytest.approx(2.0)
    assert row["fp8_minus_int8_tide_ap50_corrupt"] == pytest.approx(5.0)
    assert row["delta_e_tide_ap50"] == pytest.approx(3.0)
    assert row["delta_e_cocoeval_ap50"] == pytest.approx(3.0)
    # INT8 has 1 dAP more burden on clean and 4 dAP more under fog.
    assert row["delta_error_burden_dap50_Cls"] == pytest.approx(3.0)


def test_summary_marks_tide_components_as_nonadditive() -> None:
    records = [
        _record("int8", "clean", 0, 60.0, 3.0),
        _record("fp8", "clean", 0, 62.0, 2.0),
        _record("int8", "fog", 5, 30.0, 8.0),
        _record("fp8", "fog", 5, 35.0, 4.0),
    ]
    summary = summarize_interactions(paired_interactions(records))
    assert summary["n_direct_cells"] == 1
    assert "not an additive decomposition" in summary["interpretation"]
    assert summary["groups"]["overall"]["all"]["delta_e_cocoeval_ap50"] == pytest.approx(3.0)


def test_declared_primary_grid_is_exactly_312_arms() -> None:
    config = {
        "tide": {
            "datasets": [{"name": name} for name in ("coco", "voc", "kitti", "tt100k")],
            "models": ["yolo11n", "yolo11m", "yolo11x"],
            "precisions": ["int8", "fp8"],
            "corruptions": ["gaussian_noise", "motion_blur", "fog", "jpeg"],
            "severities": [1, 3, 5],
        }
    }
    identities = expected_primary_identities(config)
    assert len(identities) == 312
    assert ("voc", "yolo11m", "fp8", "clean", 0) in identities
    assert ("tt100k", "yolo11x", "int8", "motion_blur", 5) in identities


def test_paired_analysis_rejects_duplicate_semantic_records() -> None:
    records = [
        _record("int8", "clean", 0, 60.0, 3.0),
        _record("int8", "clean", 0, 60.0, 3.0),
    ]
    with pytest.raises(RuntimeError, match="duplicate semantic"):
        paired_interactions(records)


@pytest.mark.parametrize("field", ["engine_sha256", "preprocess_sha256", "decoder_sha256", "class_map_sha256"])
def test_paired_analysis_rejects_changed_execution_treatment(field: str) -> None:
    records = [
        _record("int8", "clean", 0, 60.0, 3.0),
        _record("fp8", "clean", 0, 62.0, 2.0),
        _record("int8", "fog", 5, 30.0, 8.0),
        _record("fp8", "fog", 5, 35.0, 4.0),
    ]
    records[2][field] = "f" * 64
    with pytest.raises(RuntimeError, match="treatment"):
        paired_interactions(records)


def test_paired_analysis_rejects_missing_engine_identity() -> None:
    records = [
        _record("int8", "clean", 0, 60.0, 3.0),
        _record("fp8", "clean", 0, 62.0, 2.0),
        _record("int8", "fog", 5, 30.0, 8.0),
        _record("fp8", "fog", 5, 35.0, 4.0),
    ]
    for record in records:
        del record["engine_sha256"]
    with pytest.raises(RuntimeError, match="treatment"):
        paired_interactions(records)


def test_paired_analysis_rejects_different_encoded_bytes_between_formats() -> None:
    records = [
        _record("int8", "clean", 0, 60.0, 3.0),
        _record("fp8", "clean", 0, 62.0, 2.0),
        _record("int8", "fog", 5, 30.0, 8.0),
        _record("fp8", "fog", 5, 35.0, 4.0),
    ]
    records[3]["input_manifest_sha256"] = "5" * 64
    with pytest.raises(RuntimeError, match="encoded"):
        paired_interactions(records)


def _legacy_cache_fixture(tmp_path):
    record = _record("int8", "clean", 0, 60.0, 3.0)
    record.update({
        "schema_version": 2, "condition_id": "test", "split": "val",
        "filename_corruption": "codec-control", "source_corruption": "codec_control",
        "control_materialization": "deterministic_jpeg95", "n_predictions": 30,
        "predictions_sha256": "6" * 64, "run_record_sha256": "7" * 64,
        "input_record_sha256": "8" * 64, "metric_record_sha256": "9" * 64,
        "positive_iou": 0.5, "background_iou": 0.1,
        "max_detections_per_image": 100, "tidecv_version": "1.0.1",
        "implementation_sha256": "f2b6a5a8abd9c385f26978fd918b06b9a7d6885305aa1fe99b3b922a0381f703",
        "config_sha256": "old-config",
    })
    job = dict(record, predictions="test.json", prediction_sha256="6" * 64,
               annotation_sha256="2" * 64)
    for key in ("engine_sha256", "runner_sha256", "preprocess_sha256", "decoder_sha256", "class_map_sha256"):
        del record[key]
    record["record_sha256"] = tide_module.canonical_hash(record, "record_sha256")
    path = tmp_path / "test.json"
    path.write_text(json.dumps(record))
    parameters = dict(positive_iou=0.5, background_iou=0.1, tidecv_version="1.0.1",
                      implementation_sha256="new-implementation", config_sha256="new-config")
    return path, job, parameters


def test_verified_legacy_cache_retains_original_computation_provenance(tmp_path):
    path, job, parameters = _legacy_cache_fixture(tmp_path)
    record = tide_module.revalidate_cached_record(path, job, parameters)
    assert record is not None
    assert record["tide_ap50"] == 60.0
    assert record["engine_sha256"] == "a" * 64
    assert record["cache_source_implementation_sha256"] == "f2b6a5a8abd9c385f26978fd918b06b9a7d6885305aa1fe99b3b922a0381f703"
    assert json.loads(path.read_text())["schema_version"] == 2  # never mutate old evidence


@pytest.mark.parametrize("field", ["prediction_sha256", "annotation_sha256", "run_record_sha256", "input_manifest_sha256", "cocoeval_ap50"])
def test_cached_diagnostics_are_rejected_when_upstream_evidence_changes(tmp_path, field):
    path, job, parameters = _legacy_cache_fixture(tmp_path)
    job[field] = "changed"
    assert tide_module.revalidate_cached_record(path, job, parameters) is None


def test_cached_diagnostics_reject_unknown_algorithm_or_tampered_payload(tmp_path):
    path, job, parameters = _legacy_cache_fixture(tmp_path)
    record = json.loads(path.read_text())
    record["tide_ap50"] = 99.0
    path.write_text(json.dumps(record))
    assert tide_module.revalidate_cached_record(path, job, parameters) is None
    record["implementation_sha256"] = "unknown-algorithm"
    record["record_sha256"] = tide_module.canonical_hash(record, "record_sha256")
    path.write_text(json.dumps(record))
    assert tide_module.revalidate_cached_record(path, job, parameters) is None


def test_wrapper_version_difference_is_reported_not_hidden_as_full_source_parity():
    records = [
        _record("int8", "clean", 0, 60.0, 3.0),
        _record("fp8", "clean", 0, 62.0, 2.0),
        _record("int8", "fog", 5, 30.0, 8.0),
        _record("fp8", "fog", 5, 35.0, 4.0),
    ]
    records[2]["runner_sha256"] = "f" * 64
    row = paired_interactions(records)[0]
    assert row["wrapper_source_identity_matched"] is False
    assert row["delta_e_tide_ap50"] == 3.0
