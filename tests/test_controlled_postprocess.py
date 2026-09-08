from __future__ import annotations

import importlib
import importlib.util
import json
from pathlib import Path

import numpy as np
import pytest


def module():
    assert importlib.util.find_spec("run_controlled_postprocess"), "automatic postprocessor missing"
    return importlib.import_module("run_controlled_postprocess")


def test_direct_and_omega_preserve_shared_fp8_cancellation():
    m = module()
    clean = {"default": np.array([0.1, 0.2]), "aligned": np.array([0.2, 0.3]),
             "fp8": np.array([0.6, 0.7])}
    corrupt = {"default": np.array([0.05, 0.1]), "aligned": np.array([0.07, 0.15]),
               "fp8": np.array([0.2, 0.3])}
    result = m.contrasts(clean, corrupt)
    np.testing.assert_allclose(result["delta_e_default"], [-0.35, -0.3])
    np.testing.assert_allclose(result["delta_e_aligned"], [-0.27, -0.25])
    np.testing.assert_allclose(result["omega"], [-0.08, -0.05])


def test_joint_summary_does_not_average_interval_endpoints():
    m = module()
    x = np.linspace(-1, 1, 2000)
    rows = [x if i % 2 == 0 else -x for i in range(12)]
    assert m.joint_interval(rows) == pytest.approx([0, 0, 0])
    with pytest.raises(ValueError):
        m.joint_interval(rows[:-1])
    rows[0][0] = np.nan
    with pytest.raises(ValueError):
        m.joint_interval(rows)


def test_committed_text_can_be_reused_but_not_overwritten(tmp_path):
    m = module()
    p = tmp_path / "report.md"
    m.exact_text(p, "valid evidence\n")
    before = p.stat().st_mtime_ns
    m.exact_text(p, "valid evidence\n")
    assert p.stat().st_mtime_ns == before
    with pytest.raises(ValueError):
        m.exact_text(p, "different evidence\n")
    assert p.read_text() == "valid evidence\n"


def test_csv_newlines_are_reusable_without_false_mutation(tmp_path):
    m = module()
    path = tmp_path / "cells.csv"
    m.exact_text(path, "condition,AP\r\nclean,1.0\r\n")
    m.exact_text(path, "condition,AP\r\nclean,1.0\r\n")
    assert path.read_bytes() == b"condition,AP\r\nclean,1.0\r\n"


def test_final_report_is_complete_and_idempotent(tmp_path):
    m = module()
    records = {}
    draws = {}
    for condition in ["clean-s0"] + [f"{c}-s{s}" for c, s in m.CONDITIONS]:
        records[condition] = {}
        for arm, value in (("default", 0.1), ("aligned", 0.2), ("fp8", 0.6)):
            records[condition][arm] = {"point": value}
            draws[f"{arm}__{condition}"] = np.full(2000, value)
    records["clean-s0"]["fp32"] = {"point": 0.7}
    manifest = {"records": records, "manifest_sha256": "fixture", "source_sha256": {}, "scope": "test"}
    (tmp_path / "manifest.json").write_text(json.dumps(manifest))
    (tmp_path / "schedule.npz").write_bytes(b"fixture")
    m.finish(tmp_path, manifest, [], draws)
    summary = json.loads((tmp_path / "summary.json").read_text())
    assert summary["macro"]["omega"]["point"] == pytest.approx(0)
    assert len(summary["cells"]) == 12
    timestamp = (tmp_path / "complete.json").stat().st_mtime_ns
    m.finish(tmp_path, manifest, [], draws)
    assert (tmp_path / "complete.json").stat().st_mtime_ns == timestamp
    completion = json.loads((tmp_path / "complete.json").read_text())
    for name, digest in completion.items():
        assert m.sha256_file(tmp_path / name) == digest


def test_bootstrap_task_preserves_duplicate_positions_and_reuses_valid_cache(tmp_path):
    m = module()
    pytest.importorskip("pycocotools")
    annotation = tmp_path / "gt.json"
    prediction = tmp_path / "pred.json"
    annotation.write_text(json.dumps({"info": {}, "images": [{"id": 1}, {"id": 2}],
        "categories": [{"id": 1, "name": "object"}], "annotations": [
            {"id": i, "image_id": i, "category_id": 1, "bbox": [0, 0, 10, 10],
             "area": 100, "iscrowd": 0} for i in [1, 2]]}))
    prediction.write_text(json.dumps([
        {"image_id": 1, "category_id": 1, "bbox": [0, 0, 10, 10], "score": 0.9},
        {"image_id": 2, "category_id": 1, "bbox": [20, 20, 10, 10], "score": 0.8}]))
    schedule = tmp_path / "schedule.npz"
    np.savez_compressed(schedule, samples=np.array([[0, 0], [1, 1]], dtype=np.int32))
    payload = {"name": "toy", "annotations": str(annotation), "prediction": str(prediction),
        "annotation_sha256": m.sha256_file(annotation), "prediction_sha256": m.sha256_file(prediction),
        "schedule": str(schedule), "schedule_sha256": m.sha256_file(schedule),
        "image_ids": [1, 2], "n_boot": 2, "point": 51 / 101,
        "manifest_sha256": "test_manifest", "out": str(tmp_path / "toy.npz")}
    first = m.bootstrap_task(payload)
    np.testing.assert_allclose(first, [1, 0], atol=1e-12)
    timestamp = Path(payload["out"]).stat().st_mtime_ns
    np.testing.assert_array_equal(m.bootstrap_task(payload), first)
    assert Path(payload["out"]).stat().st_mtime_ns == timestamp
    Path(payload["out"]).write_bytes(b"damaged")
    with pytest.raises(ValueError):
        m.bootstrap_task(payload)


def test_build_policy_rejects_tf32_or_other_build_confounds():
    m = module()
    record = {"tf32_enabled": False, "workspace": "4096M", "trtexec_sha256": "binary",
              "tensorrt_python_version": "11", "command": ["trtexec", "--onnx=a", "--saveEngine=b", "--noTF32"]}
    other = dict(record, command=["trtexec", "--onnx=c", "--saveEngine=d", "--noTF32"])
    m.validate_build_policy([record, other])
    with pytest.raises(ValueError):
        m.validate_build_policy([record, dict(other, tf32_enabled=True)])
    with pytest.raises(ValueError):
        m.validate_build_policy([record, dict(other, workspace="8192M")])
