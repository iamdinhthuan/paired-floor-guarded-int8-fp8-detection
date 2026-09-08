"""Holdout tables preserve the unrounded estimand and paired draw covariance."""
import importlib
import importlib.util
import json

import numpy as np
import pytest


def module():
    assert importlib.util.find_spec("build_v4_holdout_table") is not None, "holdout table builder missing"
    return importlib.import_module("build_v4_holdout_table")


def cells(dataset="voc", model="yolo11n"):
    return [dict(dataset=dataset, model=model, corruption=c, severity=s,
                 int8_clean_ap=.4, fp8_clean_ap=.5, int8_corrupted_ap=.2,
                 fp8_corrupted_ap=.25, delta_e=-.05)
            for c in ("gaussian_noise", "motion_blur", "fog", "jpeg") for s in (1, 3, 5)]


def test_block_uses_unrounded_four_ap_and_never_invents_corrupted_gap_ci():
    result = module().summarize_block(cells(), np.full((12, 2000), -.05))
    assert result["int8_clean_ap_points"] == pytest.approx(40)
    assert result["fp8_corrupted_mean_ap_points"] == pytest.approx(25)
    assert result["clean_gap_ap_points"] == pytest.approx(10)
    assert result["corrupted_gap_ap_points"] == pytest.approx(5)
    assert result["delta_e_ap_points"] == pytest.approx(-5)
    assert result["delta_e_percentile95_ap_points"] == pytest.approx([-5, -5, -5])
    assert result["corrupted_gap_percentile95_ap_points"] is None
    assert result["corrupted_gap_interval_status"] == "missing_component_ap_draws"
    assert result["clean_control"] == "original_source"


def test_correlated_cell_draws_are_averaged_before_percentiles():
    draws = np.tile(np.linspace(-.1, .1, 2000), (12, 1))
    draws[1::2] *= -1
    result = module().summarize_block(cells(), draws, corrupted_gap_draws=draws)
    assert result["delta_e_percentile95_ap_points"] == pytest.approx([0, 0, 0], abs=1e-13)
    assert result["corrupted_gap_percentile95_ap_points"] == pytest.approx([0, 0, 0], abs=1e-13)


def test_block_rejects_duplicate_missing_conditions_and_changed_clean_ap():
    m = module()
    original = cells()
    for broken in (original[:-1], original[:-1] + [original[0]],
                   [{**original[0], "fp8_clean_ap": .6}] + original[1:]):
        with pytest.raises(ValueError):
            m.summarize_block(broken, np.zeros((12, 2000)))


def test_block_rejects_nonfinite_or_wrong_draw_count_and_wrong_identity():
    m = module()
    for draws in (np.zeros((12, 1999)), np.full((12, 2000), np.nan)):
        with pytest.raises(ValueError):
            m.summarize_block(cells(), draws)
    bad = cells()
    bad[0]["delta_e"] = 1
    with pytest.raises(ValueError):
        m.summarize_block(bad, np.zeros((12, 2000)))


def test_macro_six_blocks_average_paired_draws_and_reject_extra_block():
    m = module()
    blocks, draws = [], []
    for dataset in ("voc", "kitti"):
        for model in ("yolo11n", "yolo11m", "yolo11x"):
            draw = np.tile(np.linspace(-.1, .1, 2000), (12, 1)) * (-1 if len(blocks) % 2 else 1)
            blocks.append(m.summarize_block(cells(dataset, model), draw))
            draws.append(draw.mean(axis=0))
    macro = m.summarize_macro(blocks, np.asarray(draws))
    assert macro["delta_e_ap_points"] == pytest.approx(-5)
    assert macro["delta_e_percentile95_ap_points"] == pytest.approx([0, 0, 0], abs=1e-13)
    with pytest.raises(ValueError):
        m.summarize_macro(blocks + [blocks[0]], np.asarray(draws))


def test_same_seed_is_insufficient_if_ordered_image_identity_differs():
    m = module()
    reference = {"seed": 19, "n_boot": 2000, "n_images": 2,
                 "input_hashes": {"a": {"image_ids_sha256": "a" * 64}}}
    changed = {**reference, "input_hashes": {"a": {"image_ids_sha256": "b" * 64}}}
    with pytest.raises(ValueError):
        m.validate_pairing([reference, changed], 19, 2)
    m.validate_pairing([reference, reference], 19, 2)


def test_historical_schedule_uses_choice_calls_not_v4_integer_sampler():
    m = module()
    actual = m.historical_schedule([1, 4, 9], seed=7, n_boot=4)
    np.testing.assert_array_equal(actual, [[2, 1, 2], [2, 1, 2], [2, 0, 0], [0, 0, 2]])
    with pytest.raises(ValueError):
        m.historical_schedule([1, 1], seed=7)


def test_gap_recovery_adds_paired_draw_vectors_not_interval_endpoints():
    m = module()
    blocks, values, clean_draws = [], [], {}
    x = np.linspace(-.1, .1, 2000)
    for dataset in ("voc", "kitti"):
        for model in ("yolo11n", "yolo11m", "yolo11x"):
            blocks.append(m.summarize_block(cells(dataset, model), np.tile(x, (12, 1))))
            values.append(x)
            clean_draws[dataset + "_" + model] = {"int8-entropy": np.full(2000, .5), "fp8": .5 - x}
    result = {"blocks": blocks, "primary_macro": m.summarize_macro(blocks, values), "limits": []}
    recovered, gap_draws = m.recover_intervals(result, values, clean_draws)
    assert recovered["primary_macro"]["corrupted_gap_percentile95_ap_points"] == pytest.approx([0, 0, 0], abs=1e-13)
    assert all(b["corrupted_gap_interval_status"] == "recovered_paired_historical_schedule" for b in recovered["blocks"])
    np.testing.assert_allclose(gap_draws, 0, atol=1e-15)
    assert result["blocks"][0]["corrupted_gap_percentile95_ap_points"] is None  # Base artifact untouched.


def test_gap_recovery_refuses_missing_clean_vector_or_nonfinite_ap():
    m = module()
    result = {"blocks": [{"dataset": "voc", "model": "yolo11n"}], "primary_macro": {}, "limits": []}
    with pytest.raises(ValueError):
        m.recover_intervals(result, np.zeros((6, 2000)), {})


def test_tail_gate_requires_hash_valid_completed_clean_control(tmp_path):
    m = module()
    marker = tmp_path / "complete.json"
    assert not m.completion_ready(marker)
    metric = tmp_path / "summary.json"
    metric.write_text("{}")
    marker.write_text(json.dumps({"summary.json": "a" * 64}))
    with pytest.raises(ValueError):
        m.completion_ready(marker)
    marker.write_text(json.dumps({"summary.json": m.sha256_file(metric)}))
    assert m.completion_ready(marker)


def test_clean_recovery_requires_all_twelve_cells_to_bind_same_prediction_and_order():
    m = module()
    binding = {"prediction_sha256": "a" * 64, "input_record_sha256": "b" * 64,
               "input_manifest_sha256": "c" * 64, "image_ids_sha256": "d" * 64}
    docs = [{"input_hashes": {"quant_clean": dict(binding)}} for _ in range(12)]
    m.validate_clean_binding(docs, binding)
    docs[-1]["input_hashes"]["quant_clean"]["prediction_sha256"] = "e" * 64
    with pytest.raises(ValueError):
        m.validate_clean_binding(docs, binding)
    with pytest.raises(ValueError):
        m.validate_clean_binding(docs[:1], binding)


def test_recovery_output_table_displays_recovered_gap_intervals(tmp_path):
    m = module()
    blocks, arrays, clean = [], [], {}
    for dataset in ("voc", "kitti"):
        for model in ("yolo11n", "yolo11m", "yolo11x"):
            blocks.append(m.summarize_block(cells(dataset, model), np.full((12, 2000), -.05)))
            arrays.append(np.full(2000, -.05))
            clean[dataset + "_" + model] = {"int8-entropy": np.full(2000, .4), "fp8": np.full(2000, .5)}
    base = {"blocks": blocks, "primary_macro": m.summarize_macro(blocks, arrays), "limits": [],
            "historical_display_macro": "-5.00", "sources": []}
    recovered, _ = m.recover_intervals(base, arrays, clean)
    completion = m.write_outputs(tmp_path, recovered, np.asarray(arrays), list(clean))
    assert completion["corrupted_gap_intervals_missing"] == 0
    assert "[5.00, 5.00]" in (tmp_path / "holdout_synthesis.tex").read_text()


def test_explicit_ledger_path_is_verified_even_without_remote_paper_tree(tmp_path):
    m = module()
    ledger = tmp_path / "retained_ledger.json"
    ledger.write_text(json.dumps({"analysis_sha256": "a" * 64}))
    with pytest.raises(ValueError, match="ledger canonical hash"):
        m.build(tmp_path / "remote_root", tmp_path / "sources", ledger_path=ledger)
