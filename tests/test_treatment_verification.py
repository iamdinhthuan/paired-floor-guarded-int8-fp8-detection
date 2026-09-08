"""Evidence validation must fail closed without promoting partial provenance."""
import copy
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path
import shutil

import pytest


def implementation():
    assert importlib.util.find_spec("build_treatment_verification") is not None, "builder not implemented"
    return importlib.import_module("build_treatment_verification")


def canonical(document, field="calibration_sha256"):
    return hashlib.sha256(json.dumps({k: v for k, v in document.items() if k != field},
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def calibration_file(tmp_path):
    doc = {"dataset": "voc", "split": "train", "n_images": 512,
           "records": [{"source_relpath": f"images/{i}.jpg", "sha256": "a" * 64}
                       for i in range(512)]}
    doc["calibration_sha256"] = canonical(doc)
    path = tmp_path / "calibration.json"
    path.write_text(json.dumps(doc, indent=2))
    Path(str(path) + ".complete").write_text(doc["calibration_sha256"] + "\n")
    return path, doc


def test_canonical_calibration_is_not_file_byte_hash(tmp_path):
    mod = implementation()
    path, doc = calibration_file(tmp_path)
    assert mod.sha256_file(path) != doc["calibration_sha256"]
    result = mod.verify_calibration(path, "voc")
    assert result["status"] == "canonical_identity_verified"
    assert result["image_payload_status"] == "not_assessed"
    path.write_text(json.dumps(doc, separators=(",", ":")))
    assert mod.verify_calibration(path, "voc")["canonical_sha256"] == doc["calibration_sha256"]


def test_tampered_calibration_rejected_despite_matching_self_and_marker(tmp_path):
    mod = implementation()
    path, doc = calibration_file(tmp_path)
    doc["records"][0]["sha256"] = "b" * 64
    path.write_text(json.dumps(doc))
    with pytest.raises(ValueError, match="canonical"):
        mod.verify_calibration(path, "voc")


def test_calibration_missing_marker_and_wrong_dataset_rejected(tmp_path):
    mod = implementation()
    path, _ = calibration_file(tmp_path)
    with pytest.raises(ValueError, match="dataset"):
        mod.verify_calibration(path, "kitti")
    Path(str(path) + ".complete").unlink()
    with pytest.raises(ValueError, match="marker"):
        mod.verify_calibration(path, "voc")


def test_payload_absence_distinguished_from_hash_failure(tmp_path):
    mod = implementation()
    path = tmp_path / "engine.plan"
    assert mod.verify_payload(path, "a" * 64)["status"] == "recorded_only_payload_absent"
    path.write_bytes(b"engine bytes")
    with pytest.raises(ValueError, match="payload hash"):
        mod.verify_payload(path, "a" * 64)
    assert mod.verify_payload(path, mod.sha256_file(path))["status"] == "artifact_bytes_verified"


def test_registry_link_uses_file_bytes_not_canonical_json(tmp_path):
    mod = implementation()
    path = tmp_path / "onnx.json"
    path.write_text(json.dumps({"output_onnx_sha256": "a" * 64}, indent=2))
    mod.verify_byte_link(path, mod.sha256_file(path))
    with pytest.raises(ValueError, match="file-byte"):
        mod.verify_byte_link(path, canonical(json.loads(path.read_text())))


def test_factorial_missing_and_duplicate_rows_rejected():
    mod = implementation()
    rows = [{"layer": "primary", "dataset": "voc", "model": "yolo11n", "precision": p}
            for p in ("fp8", "int8-entropy")]
    expected = {("primary", "voc", "yolo11n", p) for p in ("fp8", "int8-entropy")}
    mod.verify_universe(rows, expected)
    with pytest.raises(ValueError, match="missing"):
        mod.verify_universe(rows[:1], expected)
    with pytest.raises(ValueError, match="duplicate"):
        mod.verify_universe(rows + rows[:1], expected)


def run_records():
    return [{"dataset": "voc", "model": "yolo11n", "precision": "fp8",
             "corruption": c, "severity": s, "engine_sha256": "a" * 64,
             "calibration_sha256": None, "condition_id": f"voc_val__yolo11n__fp8-none__{c}-s{s}__aaaaaaaa__bbbbbbbb",
             "preprocess_sha256": "c" * 64, "decoder_sha256": "d" * 64,
             "class_map_sha256": "e" * 64, "annotation_sha256": "f" * 64,
             "input_image_ids_sha256": "1" * 64, "input_manifest_sha256": "b" * 64,
             "prediction_sha256": "2" * 64, "runner_sha256": "3" * 64,
             "n_images": 10, "runtime_environment": {"python_version": "3.11"}}
            for c, s in [("codec_control", 0), ("fog", 1)]]


def test_run_grid_rejects_missing_duplicate_and_contradictory_engine():
    mod = implementation()
    records = run_records()
    args = (("voc", "yolo11n", "fp8"), "a" * 64, "9" * 64,
            {("codec_control", 0), ("fog", 1)}, 10)
    result = mod.verify_runs(records, *args)
    assert result["engine_binding_status"] == "record_binding_verified"
    assert result["calibration_hash_present_count"] == 0
    with pytest.raises(ValueError, match="missing"):
        mod.verify_runs(records[:1], *args)
    with pytest.raises(ValueError, match="duplicate"):
        mod.verify_runs(records + records[:1], *args)
    changed = copy.deepcopy(records)
    changed[1]["engine_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="engine identity"):
        mod.verify_runs(changed, *args)


def test_recorded_calibration_mismatch_is_not_treated_as_missing():
    mod = implementation()
    records = run_records()
    records[1]["calibration_sha256"] = "0" * 64
    with pytest.raises(ValueError, match="calibration binding"):
        mod.verify_runs(records, ("voc", "yolo11n", "fp8"), "a" * 64, "9" * 64,
                        {("codec_control", 0), ("fog", 1)}, 10)


def test_condition_id_must_agree_with_structured_treatment_and_condition():
    mod = implementation()
    records = run_records()
    records[1]["condition_id"] = records[1]["condition_id"].replace("yolo11n", "yolo11x")
    with pytest.raises(ValueError, match="condition treatment"):
        mod.verify_runs(records, ("voc", "yolo11n", "fp8"), "a" * 64, "9" * 64,
                        {("codec_control", 0), ("fog", 1)}, 10)


def test_clean_int8_shorthand_requires_structured_entropy_identity():
    mod = implementation()
    records = run_records()[:1]
    records[0]["precision"] = "int8-entropy"
    records[0]["condition_id"] = records[0]["condition_id"].replace("fp8-none", "int8")
    result = mod.verify_runs(records, ("voc", "yolo11n", "int8-entropy"), "a" * 64, "9" * 64,
                             {("codec_control", 0)}, 10)
    assert result["source_precision_tokens"] == ["int8"]


def test_holdout_metric_hash_never_promoted_to_full_engine_digest():
    mod = implementation()
    report = {"dataset": "voc", "split": "test", "metric_sha256": {
        "voc_test__yolo11n__fp8-entropy__clean-s0__1234abcd__9876abcd": "e" * 64,
        "voc_test__yolo11n__fp8-entropy__fog-s1__1234abcd__9876abcd": "f" * 64}}
    result = mod.holdout_identity(report, "yolo11n", "fp8", {("clean", 0), ("fog", 1)})
    assert result["engine_sha256"] is None
    assert result["engine_digest_prefix"] == "1234abcd"
    assert result["status"] == "digest_prefix_only"
    assert result["metric_payload_status"] == "not_assessed"
    assert result["source_precision_tokens"] == ["fp8-entropy"]
    changed = copy.deepcopy(report)
    key = next(k for k in changed["metric_sha256"] if "fog" in k)
    changed["metric_sha256"][key.replace("1234abcd", "deadbeef")] = changed["metric_sha256"].pop(key)
    with pytest.raises(ValueError, match="prefix"):
        mod.holdout_identity(changed, "yolo11n", "fp8", {("clean", 0), ("fog", 1)})


def test_explicit_path_mapping_cannot_substitute_legacy_coco(tmp_path):
    mod = implementation()
    path = mod.map_historical_path(tmp_path, "coco", "/home/thuan/topic_c_ivc/engines/coco/yolo11n/fp8.plan")
    assert path == tmp_path / "engines/coco/yolo11n/fp8.plan"
    with pytest.raises(ValueError, match="path"):
        mod.map_historical_path(tmp_path, "coco", "/unexpected/engines/coco/yolo11n/fp8.plan")


def test_table_uses_verified_counts_not_assumed_local_payload_presence():
    mod = implementation()
    summary = {"primary_run_records_bound": 312,
               "groups": [{"label": "Primary COCO", "treatments": 6, "full_run_binding": 6, "engine_bytes": 0, "graph_reports": 0},
                          {"label": "Primary transfer", "treatments": 18, "full_run_binding": 18, "engine_bytes": 0, "graph_reports": 18},
                          {"label": "Holdout (partial)", "treatments": 12, "full_run_binding": 0, "engine_bytes": 0, "graph_reports": 0}]}
    table = mod.render_table(summary)
    assert "Primary transfer & 18 & 18 & 0 & 18" in table
    assert "Primary transfer & 18 & 18 & 18 & 18" not in table


def test_current_inventory_retains_24_primary_and_12_partial_holdout(tmp_path):
    mod = implementation()
    root = Path(__file__).resolve().parents[1]
    manifest, summary = mod.build(root)
    assert summary["primary_treatments"] == 24
    assert summary["holdout_treatments"] == 12
    assert summary["primary_run_records_bound"] == 312
    payload_count = sum((root / r["engine_payload"]["mapped_local_path"]).is_file()
                        for r in manifest["treatments"] if r["layer"] == "primary")
    assert summary["primary_engine_payloads_verified"] == payload_count
    assert summary["primary_engine_payloads_absent"] == 24 - payload_count
    assert summary["primary_graph_reports_bound"] == 18
    assert summary["primary_graph_reports_absent"] == 6
    assert summary["primary_paired_calibration_blocks_verified"] == 12
    assert all(r["identity"]["engine_sha256"] is None for r in manifest["treatments"] if r["layer"] == "holdout")
    assert all(r["kernel_precision_status"] == "not_assessed" for r in manifest["treatments"])
    assert all(not s["path"].endswith((".plan", ".onnx")) for s in manifest["compact_evidence_sources"])
    for source in manifest["compact_evidence_sources"]:
        path = root / source["path"]
        assert mod.sha256_file(path) == source["sha256"]
        destination = tmp_path / source["path"]
        destination.parent.mkdir(parents=True, exist_ok=True)
        shutil.copy2(path, destination)
    compact_manifest, compact_summary = mod.build(tmp_path)
    assert compact_summary["primary_run_records_bound"] == 312
    assert compact_summary["primary_engine_payloads_verified"] == 0
    assert compact_summary["primary_engine_payloads_absent"] == 24
    assert compact_summary["primary_graph_reports_bound"] == 18
    assert compact_summary["primary_onnx_payloads_verified"] == 0
    assert all(r["graph"]["graph_payload_status"] == "recorded_only_payload_absent"
               for r in compact_manifest["treatments"]
               if r["graph"]["status"] == "report_bound_to_registered_onnx")
    assert "Primary transfer & 18 & 18 & 0 & 18" in mod.render_table(compact_summary)
