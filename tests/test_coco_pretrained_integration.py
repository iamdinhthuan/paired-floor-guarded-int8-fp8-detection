"""Integration guards for the pretrained-COCO replication block (phase I).

The block replicates the RetinaNet/FCOS arm set on off-the-shelf torchvision
checkpoints over a paired 2,000-image COCO val2017 subset. These tests pin the
evidence layout, the generated manuscript artifacts, and the wiring in
``paper/main_nn.tex`` so the replication cannot silently drop out of the
paper or regress to a partial arm set.
"""
from __future__ import annotations

import json
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
PHASE_I = ROOT / "submission_support_20260911" / "phase_i_coco_pretrained_results"
MAIN = ROOT / "paper" / "main_nn.tex"
GENERATED = ROOT / "paper" / "generated"
STATS = ROOT / "submission_support_20260911" / "nn_final_stats.json"

MODELS = ("retinanet_pretrained", "fcos_pretrained")
ARMS = ("fp32", "fp8-matched512", "int8-matched512", "int8-selective512")
CONDITIONS = (
    ("clean", "codec-control")
    + tuple(c for c in ("fog", "gaussian_noise", "jpeg", "motion_blur"))
)


def _cells() -> dict:
    path = PHASE_I / "cell_points.json"
    if not path.is_file():
        pytest.skip("phase-I cell points not yet collected")
    return json.loads(path.read_text())


def test_cell_points_cover_both_detectors_and_all_arms() -> None:
    cells = _cells()
    for model in MODELS:
        block = cells.get(f"coco/{model}")
        assert block is not None, f"missing block coco/{model}"
        for arm in ARMS:
            assert arm in block, f"{model} missing arm {arm}"
            conds = block[arm]
            for cond in ("clean-s0", "codec-control-s0"):
                assert cond in conds, f"{model}/{arm} missing {cond}"
            corrupt = [c for c in conds if c not in ("clean-s0", "codec-control-s0")]
            assert len(corrupt) == 12, f"{model}/{arm} has {len(corrupt)} corrupt cells"
            for cond, ap in conds.items():
                assert 0.0 < ap <= 100.0, f"{model}/{arm}/{cond}={ap}"


def test_metric_records_are_complete_2000_image_cells() -> None:
    metrics = PHASE_I / "metrics"
    records = list(metrics.glob("*.json"))
    assert len(records) == 2 * len(ARMS) * 14
    for record in records:
        data = json.loads(record.read_text())
        assert data["n_images"] == 2000, record.name


def test_bootstrap_draw_caches_exist_when_integrated() -> None:
    boot = PHASE_I / "bootstrap"
    npz = list(boot.glob("*__draws.npz"))
    if len(npz) < len(MODELS):
        pytest.skip("bootstrap draws not yet fetched")
    for model in MODELS:
        cache = boot / f"coco__{model}__draws.npz"
        summary = boot / f"coco__{model}__bootstrap.json"
        assert cache.is_file(), f"missing draw cache for {model}"
        assert summary.is_file(), f"missing bootstrap summary for {model}"


def test_final_stats_carry_coco_pretrained_block() -> None:
    if not STATS.is_file():
        pytest.skip("nn_final_stats.json not generated")
    final = json.loads(STATS.read_text())
    cp = final.get("coco_pretrained")
    complete = cp and all(
        cp.get(model, {}).get("arms") for model in MODELS)
    if not complete:
        pytest.skip("coco_pretrained block not yet integrated "
                    "(arms populate once bootstrap draw caches are fetched)")
    for model in MODELS:
        entry = cp.get(model)
        assert entry is not None and set(entry["arms"]) == set(ARMS)
        fm = entry["contrasts"]["fp8-matched512_minus_int8-matched512"]
        sm = entry["contrasts"]["int8-selective512_minus_int8-matched512"]
        for key in ("j95", "corr12", "deltaE"):
            for est in (fm, sm):
                assert "ci95" in est[key] and "p" in est[key]


def test_manuscript_references_pretrained_table() -> None:
    text = MAIN.read_text(encoding="utf-8")
    assert "\\input{generated/nn_coco_pretrained.tex}" in text
    assert "sec:coco-pretrained" in text
    table_path = GENERATED / "nn_coco_pretrained.tex"
    draws_ready = len(list((PHASE_I / "bootstrap").glob("*__draws.npz"))) >= 2
    if table_path.is_file() and draws_ready:
        table = table_path.read_text(encoding="utf-8")
        assert table.count("RetinaNet") == 1 and table.count("FCOS") == 1
