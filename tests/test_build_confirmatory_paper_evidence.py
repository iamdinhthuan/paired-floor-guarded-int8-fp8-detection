from __future__ import annotations

import json
from pathlib import Path

import pytest

import build_confirmatory_paper_evidence as builder
from pilot_registry import canonical_hash
from topic_c.manifest import sha256_file


def _confirmatory() -> dict:
    return {
        "scope": {"direct_cells": 72, "partition_role": "untouched final holdout"},
        "overall_balanced_equal_cell": {
            "delta_e_point": -0.0012,
            "delta_e_percentile95": [-0.0030, -0.0011, 0.0004],
            "delta_psi_point": 0.0025,
            "delta_psi_percentile95": [-0.0010, 0.0024, 0.0060],
        },
        "by_dataset": {
            "voc": {
                "n_cells": 36,
                "delta_e_point": -0.0010,
                "delta_e_percentile95": [-0.0020, -0.0011, 0.0001],
            },
            "kitti": {
                "n_cells": 36,
                "delta_e_point": -0.0014,
                "delta_e_percentile95": [-0.0040, -0.0012, 0.0010],
            },
        },
        "matched_clean": {
            "blocks": 6,
            "fp8_minus_int8_mean": 0.014,
            "positive_blocks": 6,
        },
        "absolute_corrupted_ap_guardrail": {
            "format_arms": 144,
            "below_5_ap_points": 9,
            "below_10_ap_points": 17,
        },
    }


def _realization() -> dict:
    return {
        "n_boot": 2000,
        "realization_seeds": [11, 22, 33],
        "overall_delta_e": {
            "n_realization_cells": 108,
            "n_base_conditions": 36,
            "mean_delta_e": -0.0007,
            "nested_percentile95": [-0.0020, -0.0006, 0.0010],
            "mean_between_realization_variance": 0.000004,
            "mean_within_image_variance": 0.000001,
        },
        "overall_delta_psi": {
            "mean_delta_psi": 0.0015,
            "nested_percentile95": [-0.0010, 0.0014, 0.0040],
            "mean_between_realization_variance": 0.000002,
            "mean_within_image_variance": 0.000003,
        },
    }


def _fixed_universe() -> dict:
    return {
        "n_images": 3067,
        "n_boot": 10000,
        "point": {"delta_e_all": -0.0025, "delta_psi_height": -0.0392},
        "fixed_universe": {
            "delta_e_all": {
                "percentile_interval": [-0.0119, -0.0017, 0.0088],
                "max_checkpoint_percentile_shift_native_ap": 0.00066,
            },
            "delta_psi_height": {
                "percentile_interval": [-0.0569, -0.0408, -0.0248],
                "max_checkpoint_percentile_shift_native_ap": 0.00140,
            },
        },
        "ordinary_reference": {
            "n_boot": 2000,
            "delta_e_all": [-0.0179, -0.0020, 0.0134],
            "delta_psi_height": [-0.0722, -0.0448, -0.0181],
        },
    }


def test_render_confirmatory_results_converts_native_ap_to_points() -> None:
    rendered = builder.render_confirmatory_results(_confirmatory())

    assert "untouched final partitions" in rendered
    assert "$-0.12$ AP points" in rendered
    assert "$[-0.30,\,-0.11,\,+0.04]$" in rendered
    assert "1.40 AP points" in rendered
    assert "6/6" in rendered
    assert r"\label{tab:holdout-by-dataset}" in rendered
    assert r"\resizebox{\columnwidth}{!}{%" in rendered
    assert "VOC & 2,510 & 5,823 & 36" in rendered
    assert "KITTI & 1,496 & 1,197 & 36" in rendered


def test_render_confirmatory_methods_states_selection_final_separation_and_joint_draws() -> None:
    rendered = builder.render_confirmatory_methods(_confirmatory())

    assert "checkpoint-selection" in rendered
    assert "untouched final partition" in rendered
    assert "5,823" in rendered and "1,197" in rendered
    assert "seed 20260818" in rendered
    assert "TF32 disabled" in rendered
    assert "2,000" in rendered
    assert "same dataset-level draw schedule" in rendered


def test_render_realization_results_reports_both_variance_sources() -> None:
    rendered = builder.render_realization_results(_realization())

    assert "108 realization cells" in rendered
    assert "36 base conditions" in rendered
    assert "$-0.07$ AP points" in rendered
    assert "0.0400" in rendered
    assert "0.0100" in rendered
    assert "three fixed realization seeds" in rendered


def test_render_conclusion_and_supplement_preserve_conditional_scope() -> None:
    conclusion = builder.render_confirmatory_conclusion(_confirmatory(), _realization())
    supplement = builder.render_confirmatory_supplement(_confirmatory(), _realization())

    assert "untouched VOC/KITTI final partitions" in conclusion
    assert "provide bounded checks against" in conclusion
    assert "reduce evaluation-selection" not in conclusion
    assert "does not establish a population-level" in conclusion
    assert "3 datasets $\\times$ 3 realization seeds" in supplement
    assert "117 conditions per dataset (234 total)" in supplement
    assert "untouched-holdout validation branch" in supplement
    assert "nested across severity" in supplement
    assert "fixed class-covering subset" in supplement


def test_render_fixed_universe_reports_ordinary_comparison_and_monte_carlo_shift() -> None:
    rendered = builder.render_fixed_universe_results(_fixed_universe())

    assert "10,000" in rendered and "3,067" in rendered
    assert "fixed category universe" in rendered
    assert "$[-1.19,\,-0.17,\,+0.88]$" in rendered
    assert "$[-1.79,\,-0.20,\,+1.34]$" in rendered
    assert "0.066" in rendered and "0.140" in rendered
    assert "median sign and zero-crossing status" in rendered
    assert "sign interpretation therefore survives" not in rendered


def test_render_combined_abstract_adds_confirmatory_checks_under_250_words() -> None:
    rendered = builder.render_combined_abstract(_confirmatory(), _realization())
    plain = rendered.replace("\\%", "%")

    assert "primary exploratory grid" in rendered
    assert "untouched-holdout validation" in rendered
    assert "three-realization sensitivity" in rendered
    assert "distinct conditional scopes" in rendered
    assert "$-0.12$ AP points" in rendered
    assert len(plain.split()) <= 250


def test_render_combined_highlights_has_five_lines_within_elsevier_limit() -> None:
    rendered = builder.render_combined_highlights(_confirmatory(), _realization())
    lines = [line.removeprefix("- ") for line in rendered.splitlines() if line]

    assert len(lines) == 5
    assert max(map(len, lines)) <= 85
    assert any("Untouched VOC/KITTI" in line for line in lines)
    assert any("realization" in line for line in lines)


def test_validate_report_requires_internal_hash_and_physical_completion_marker(tmp_path: Path) -> None:
    report = {"schema_version": 1, "value": 7}
    report["analysis_sha256"] = canonical_hash(report, "analysis_sha256")
    path = tmp_path / "report.json"
    path.write_text(json.dumps(report), encoding="utf-8")
    path.with_suffix(".json.complete").write_text(sha256_file(path) + "\n", encoding="utf-8")

    assert builder.validate_report(path, hash_key="analysis_sha256")["value"] == 7

    path.write_text(json.dumps({**report, "value": 8}), encoding="utf-8")
    with pytest.raises(ValueError, match="completion marker"):
        builder.validate_report(path, hash_key="analysis_sha256")


def test_master_completion_must_bind_both_analysis_report_bytes(tmp_path: Path) -> None:
    confirm = tmp_path / "confirm.json"
    realization = tmp_path / "realization.json"
    confirm.write_text("confirm\n", encoding="utf-8")
    realization.write_text("realization\n", encoding="utf-8")
    master = {
        "status": "complete",
        "validated_components": {
            "confirmatory_analysis": {"path": "/remote/original/confirm.json", "sha256": sha256_file(confirm)},
            "realization_analysis": {"path": "/remote/original/realization.json", "sha256": sha256_file(realization)},
        },
    }

    builder.validate_master_bindings(master, confirm, realization)

    realization.write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="master completion binding mismatch"):
        builder.validate_master_bindings(master, confirm, realization)
