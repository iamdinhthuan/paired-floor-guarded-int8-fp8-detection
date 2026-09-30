"""Integration guards for the max-calibration counterfactual (phase J / B6).

The counterfactual runs INT8-matched and INT8-selective RetinaNet arms under
the max range estimator on the frozen 512-image calibration manifests and the
same 14-condition schedule. These are post-hoc single-build diagnostics with
plug-in points only (no bootstrap draws); the tests pin the evidence layout
and the generated manuscript artifacts so the block cannot silently drop out.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PHASE_J = ROOT / "submission_support_20260911" / "phase_j_maxcalib_results"
SUPP = ROOT / "paper" / "supplement.tex"
GENERATED = ROOT / "paper" / "generated"
STATS = ROOT / "submission_support_20260911" / "nn_final_stats.json"

ARMS = ("int8-max512", "int8-maxsel512")


def _cells() -> dict:
    path = PHASE_J / "cell_points.json"
    if not path.is_file():
        pytest.skip("phase-J cell points not yet collected")
    return json.loads(path.read_text())


def test_cell_points_cover_both_datasets_and_arms() -> None:
    cells = _cells()
    for ds in ("kitti", "voc"):
        block = cells.get(f"{ds}/retinanet_maxcalib")
        assert block is not None, f"missing block {ds}/retinanet_maxcalib"
        for arm in ARMS:
            assert arm in block, f"{ds} missing arm {arm}"
            conds = block[arm]
            assert "clean-s0" in conds and "codec-control-s0" in conds
            corrupt = [c for c in conds
                       if c not in ("clean-s0", "codec-control-s0")]
            assert len(corrupt) == 12, f"{ds}/{arm} has {len(corrupt)} corrupt"
            for cond, ap in conds.items():
                assert 0.0 <= ap <= 100.0, f"{ds}/{arm}/{cond}={ap}"


def test_metric_records_are_complete() -> None:
    metrics = PHASE_J / "metrics"
    records = list(metrics.glob("*.json"))
    if len(records) < 2 * len(ARMS) * 14:
        pytest.skip("phase-J metric records not yet fetched")
    for record in records:
        data = json.loads(record.read_text())
        assert data.get("stats", {}).get("AP") is not None, record.name
        assert data.get("input_image_ids_sha256"), record.name
        assert data.get("prediction_sha256"), record.name


def test_final_stats_carry_maxcalib_block() -> None:
    if not STATS.is_file():
        pytest.skip("nn_final_stats.json not generated")
    final = json.loads(STATS.read_text())
    mc = final.get("maxcalib")
    if not mc:
        pytest.skip("maxcalib block not yet integrated")
    for ds in ("kitti", "voc"):
        ent = mc.get(ds)
        assert ent is not None and set(ent["arms"]) == set(ARMS)
        assert "maxsel_minus_max" in ent["contrasts"]
    # The counterfactual's scientific content: max calibration must not
    # exceed the entropy-calibrated matched level on KITTI clean.
    assert mc["kitti"]["arms"]["int8-max512"]["j95"] < 28.0


def test_supplement_references_maxcalib_table() -> None:
    text = SUPP.read_text(encoding="utf-8")
    assert "\\input{generated/nn_maxcalib.tex}" in text
    assert "sec:maxcalib" in text
    table_path = GENERATED / "nn_maxcalib.tex"
    if table_path.is_file():
        table = table_path.read_text(encoding="utf-8")
        assert table.count("INT8-matched (max)") == 2
        assert table.count("INT8-selective (max)") == 2
