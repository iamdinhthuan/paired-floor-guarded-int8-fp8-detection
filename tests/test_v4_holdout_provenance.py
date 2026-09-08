"""Retained files must not manufacture historical execution evidence."""
import hashlib
import importlib
import importlib.util
import json
from pathlib import Path

import pytest


def impl():
    assert importlib.util.find_spec("audit_v4_holdout_provenance") is not None, "audit is not implemented"
    return importlib.import_module("audit_v4_holdout_provenance")


def digest(value):
    return hashlib.sha256(value).hexdigest()


def test_current_engine_matching_prefix_cannot_verify_absent_run_binding():
    module = impl()
    result = module.classify_lineage({"engine_bytes": "verified", "historical_prefix": "verified",
                                     "run_prediction_binding": "missing"})
    assert result == "partial"


def test_only_all_verified_links_are_verified_and_empty_evidence_is_missing():
    module = impl()
    assert module.classify_lineage({"engine": "verified", "run": "verified"}) == "verified"
    assert module.classify_lineage({"engine": "missing", "run": "missing"}) == "missing"
    assert module.classify_lineage({}) == "missing"
    assert module.classify_lineage({"engine": "verified", "run": "partial"}) == "partial"


def test_payload_full_hash_is_required_and_mismatch_is_not_a_missing_file(tmp_path):
    module = impl()
    payload = tmp_path / "engine.plan"
    payload.write_bytes(b"retained engine")
    auditor = module.Auditor(tmp_path)
    assert auditor.payload(payload, digest(b"retained engine"))["status"] == "verified"
    assert auditor.payload(payload, digest(b"retained engine")[:8])["status"] == "partial"
    wrong = auditor.payload(payload, "a" * 64)
    assert wrong["status"] == "partial"
    assert wrong["reason"] == "hash_mismatch"
    assert auditor.payload(tmp_path / "absent", "a" * 64)["status"] == "missing"


def test_canonical_identity_does_not_use_json_file_bytes(tmp_path):
    module = impl()
    document = {"records": [{"image_id": 1}], "expected_image_ids": [1]}
    document["manifest_sha256"] = digest(json.dumps(document, sort_keys=True, separators=(",", ":")).encode())
    path = tmp_path / "manifest.json"
    path.write_text(json.dumps(document, indent=2))
    auditor = module.Auditor(tmp_path)
    assert auditor.canonical(path, "manifest_sha256")["status"] == "verified"
    document["records"][0]["image_id"] = 2
    path.write_text(json.dumps(document))
    assert module.Auditor(tmp_path).canonical(path, "manifest_sha256")["status"] == "partial"


def test_run_binding_requires_metric_to_hash_actual_run_and_prediction(tmp_path):
    module = impl()
    prediction = tmp_path / "pred.json"
    prediction.write_text("[]")
    run = {"condition_id": "example", "engine_sha256": "a" * 64,
           "prediction_sha256": digest(b"[]"), "input_manifest_sha256": "b" * 64,
           "input_image_ids_sha256": "c" * 64}
    run_path = tmp_path / "run.json"
    run_path.write_text(json.dumps(run))
    metric = {**run, "run_record_sha256": digest(run_path.read_bytes())}
    auditor = module.Auditor(tmp_path)
    assert module.verify_run_binding(auditor, run_path, run, prediction, metric, "a" * 64)["status"] == "verified"
    metric["run_record_sha256"] = "d" * 64
    assert module.verify_run_binding(auditor, run_path, run, prediction, metric, "a" * 64)["status"] == "partial"
    metric.pop("run_record_sha256")
    assert module.verify_run_binding(auditor, run_path, run, prediction, metric, "a" * 64)["status"] == "partial"


def test_missing_repository_produces_twelve_explicit_missing_treatments(tmp_path):
    report = impl().audit(tmp_path)
    assert report["counts"] == {"treatments": 12, "verified": 0, "partial": 0, "missing": 12,
                                "expected_runs": 156, "retained_runs": 0}
    assert len({(row["dataset"], row["model"], row["precision"]) for row in report["treatments"]}) == 12


def retained_run_fixture(tmp_path):
    def write(relative, value):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(value))
        return str(path)

    manifest = {"records": [{"image_id": 1}, {"image_id": 2}], "expected_image_ids": [1, 2]}
    manifest["manifest_sha256"] = digest(json.dumps(manifest, sort_keys=True, separators=(",", ":")).encode())
    manifest_path = write("manifest.json", manifest)
    annotation = write("annotations.json", {"images": [{"id": 1}, {"id": 2}]})
    class_map = write("classes.json", {"classes": ["car"]})
    config = {"annotation": annotation, "class_map": class_map, "clean_manifest": manifest_path}
    cid = "kitti_test__yolo11m__int8-entropy__fog-s1__aaaaaaaa__" + manifest["manifest_sha256"][:8]
    subpath = "kitti_confirmatory_final_117_v1/" + cid + ".json"
    prediction_path = write("outputs/predictions/" + subpath, [])
    ids_hash = digest(b"[1,2]")
    run = {"condition_id": cid, "dataset": "kitti", "model": "yolo11m", "precision": "int8-entropy",
           "corruption": "fog", "severity": 1, "engine_path": "retained.plan", "engine_sha256": "a" * 64,
           "calibration_sha256": "b" * 64, "input_manifest_sha256": manifest["manifest_sha256"],
           "input_image_ids_sha256": ids_hash, "n_images": 2,
           "prediction_sha256": digest(Path(prediction_path).read_bytes()),
           "annotation_sha256": digest(Path(annotation).read_bytes()), "class_map_sha256": digest(Path(class_map).read_bytes()),
           "command": "runner --image-manifest " + manifest_path}
    for field, path in (("runner_sha256", "src/coco_infer_trt.py"), ("preprocess_sha256", "src/topic_c/coco_data.py"),
                        ("decoder_sha256", "src/topic_c/yolo_decode.py")):
        module_path = write(path, "bound historical source")
        run[field] = digest(Path(module_path).read_bytes())
    run_path = write("manifests/runs/" + subpath, run)
    write("outputs/inputs/" + subpath, {"condition_id": cid, "image_ids": [1, 2], "image_ids_sha256": ids_hash,
                                       "input_manifest_sha256": manifest["manifest_sha256"]})
    metric = {**run, "run_record_sha256": digest(Path(run_path).read_bytes())}
    metric_path = write("outputs/metrics/" + subpath, metric)
    report = {"metric_sha256": {cid: digest(Path(metric_path).read_bytes())}}
    engine = {"engine": "retained.plan", "engine_sha256": "a" * 64, "calibration_sha256": "b" * 64}
    return config, report, engine, run_path, metric_path


def test_complete_run_links_verify_and_annotation_universe_mismatch_fails(tmp_path, monkeypatch):
    module = impl()
    monkeypatch.setitem(module.DATASETS, "kitti", 2)
    config, report, engine, _, metric_path = retained_run_fixture(tmp_path)
    result = module.audit_run(module.Auditor(tmp_path), "kitti", "yolo11m", "int8-entropy", ("fog", 1), report, engine, config)
    assert result["status"] == "verified"
    # Keep the count at two, but change the official image universe and update
    # the recorded annotation hash. An internally consistent wrong universe
    # still must not pass the input identity gate.
    Path(config["annotation"]).write_text(json.dumps({"images": [{"id": 7}, {"id": 8}]}))
    result = module.audit_run(module.Auditor(tmp_path), "kitti", "yolo11m", "int8-entropy", ("fog", 1), report, engine, config)
    assert result["checks"]["annotation_image_ids"]["status"] == "partial"


def test_missing_run_record_with_retained_matching_engine_stays_partial(tmp_path, monkeypatch):
    module = impl()
    monkeypatch.setitem(module.DATASETS, "kitti", 2)
    config, report, engine, run_path, _ = retained_run_fixture(tmp_path)
    Path(run_path).unlink()
    result = module.audit_run(module.Auditor(tmp_path), "kitti", "yolo11m", "int8-entropy", ("fog", 1), report, engine, config)
    assert result["status"] == "partial"
    assert result["checks"]["run_prediction_metric"]["status"] == "missing"


def test_fp8_entropy_condition_token_selects_record_without_rewriting_precision(tmp_path, monkeypatch):
    module = impl()
    monkeypatch.setitem(module.DATASETS, "kitti", 2)
    config, report, engine, _, _ = retained_run_fixture(tmp_path)
    old_cid = next(iter(report["metric_sha256"]))
    cid = old_cid.replace("__int8-entropy__", "__fp8-entropy__")
    # Historical filenames encode calibrator; structured records keep fp8.
    report["metric_sha256"] = {cid: report["metric_sha256"][old_cid]}
    result = module.audit_run(module.Auditor(tmp_path), "kitti", "yolo11m", "fp8", ("fog", 1), report, engine, config)
    assert result.get("condition_id") == cid
    assert result["status"] == "partial"  # Selection alone cannot verify absent payloads.


def test_original_clean_requires_every_record_to_reuse_source_identity():
    module = impl()
    record = {"source_relpath": "image.png", "output_relpath": "image.png", "source_sha256": "a" * 64,
              "sha256": "a" * 64, "generator": "clean_reference", "corruption": "clean", "severity": 0}
    result = module.clean_manifest_semantics({"records": [record]})
    assert result == {"records": 1, "original_source_identity_records": 1, "control": "original_source"}
    altered = {**record, "sha256": "b" * 64, "output_relpath": "q95.jpg"}
    assert module.clean_manifest_semantics({"records": [record, altered]})["control"] == "not_verified_as_original_source"
    assert module.clean_manifest_semantics({})["control"] == "missing"
