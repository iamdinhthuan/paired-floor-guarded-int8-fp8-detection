from __future__ import annotations

import importlib.util
import csv
import json
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
ANALYZER_PATH = PROJECT_ROOT / "src" / "analyze_multiseed_validation.py"


def load_analyzer():
    assert ANALYZER_PATH.is_file(), "multi-seed analyzer has not been implemented"
    spec = importlib.util.spec_from_file_location("analyze_multiseed_validation", ANALYZER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_direct_cell_uses_loss_difference_and_keeps_absolute_ap() -> None:
    analyzer = load_analyzer()

    row = analyzer.direct_cell(
        dataset="voc",
        training_seed=20260813,
        calibration_seed=20260814,
        corruption="fog",
        int8_clean_ap=0.60,
        int8_corrupt_ap=0.40,
        fp8_clean_ap=0.59,
        fp8_corrupt_ap=0.41,
    )

    assert row["d_int8_native_ap"] == pytest.approx(0.20)
    assert row["d_fp8_native_ap"] == pytest.approx(0.18)
    assert row["delta_e_native_ap"] == pytest.approx(0.02)
    assert row["delta_e_ap_points"] == pytest.approx(2.0)
    assert row["int8_corrupt_ap"] == pytest.approx(0.40)
    assert row["fp8_corrupt_ap"] == pytest.approx(0.41)


def test_two_way_seed_decomposition_matches_hand_derived_additive_grid() -> None:
    analyzer = load_analyzer()
    training_seeds = [20260807, 20260813, 20260814]
    calibration_seeds = [20260807, 20260813, 20260814]
    values = [
        [0.0, 1.0, 2.0],
        [1.0, 2.0, 3.0],
        [2.0, 3.0, 4.0],
    ]
    rows = [
        {
            "dataset": "voc",
            "corruption": "fog",
            "training_seed": training_seed,
            "calibration_seed": calibration_seed,
            "delta_e_ap_points": values[train_index][calibration_index],
        }
        for train_index, training_seed in enumerate(training_seeds)
        for calibration_index, calibration_seed in enumerate(calibration_seeds)
    ]

    result = analyzer.two_way_seed_decomposition(rows)

    assert result["n"] == 9
    assert result["grand_mean_ap_points"] == pytest.approx(2.0)
    assert result["ss_training"] == pytest.approx(6.0)
    assert result["ss_calibration"] == pytest.approx(6.0)
    assert result["ss_interaction"] == pytest.approx(0.0)
    assert result["ss_total"] == pytest.approx(12.0)
    assert result["share_training"] == pytest.approx(0.5)
    assert result["share_calibration"] == pytest.approx(0.5)
    assert result["share_interaction"] == pytest.approx(0.0)


def test_two_way_seed_decomposition_rejects_missing_or_duplicate_cells() -> None:
    analyzer = load_analyzer()
    rows = [
        {
            "dataset": "voc",
            "corruption": "fog",
            "training_seed": training_seed,
            "calibration_seed": calibration_seed,
            "delta_e_ap_points": float(index),
        }
        for index, (training_seed, calibration_seed) in enumerate(
            (training_seed, calibration_seed)
            for training_seed in [20260807, 20260813, 20260814]
            for calibration_seed in [20260807, 20260813, 20260814]
        )
    ]

    with pytest.raises(RuntimeError, match="3x3"):
        analyzer.two_way_seed_decomposition(rows[:-1])
    with pytest.raises(RuntimeError, match="duplicate"):
        analyzer.two_way_seed_decomposition([*rows[:-1], rows[0]])


def synthetic_metric_rows() -> list[dict]:
    rows = []
    for dataset_index, dataset in enumerate(["voc", "kitti", "tt100k"]):
        for training_index, training_seed in enumerate([20260807, 20260813, 20260814]):
            for calibration_index, calibration_seed in enumerate([20260807, 20260813, 20260814]):
                clean = {
                    "int8-entropy": 0.60 + 0.001 * training_index,
                    "fp8": 0.59 + 0.001 * calibration_index,
                }
                for precision, ap in clean.items():
                    rows.append(
                        {
                            "dataset": dataset,
                            "training_seed": training_seed,
                            "calibration_seed": calibration_seed,
                            "precision": precision,
                            "corruption": "codec_control",
                            "severity": 0,
                            "AP": ap,
                            "metric_sha256": f"clean-{dataset_index}-{training_index}-{calibration_index}-{precision}",
                        }
                    )
                for corruption_index, corruption in enumerate(
                    ["gaussian_noise", "motion_blur", "fog", "jpeg"], start=1
                ):
                    for precision, clean_ap in clean.items():
                        loss = 0.10 + 0.01 * corruption_index
                        if precision == "fp8":
                            loss -= 0.002
                        rows.append(
                            {
                                "dataset": dataset,
                                "training_seed": training_seed,
                                "calibration_seed": calibration_seed,
                                "precision": precision,
                                "corruption": corruption,
                                "severity": 5,
                                "AP": clean_ap - loss,
                                "metric_sha256": f"corrupt-{dataset_index}-{training_index}-{calibration_index}-{corruption}-{precision}",
                            }
                        )
    return rows


def test_build_direct_cells_requires_exact_270_metrics_and_emits_108_cells() -> None:
    analyzer = load_analyzer()
    metrics = synthetic_metric_rows()

    cells = analyzer.build_direct_cells(metrics)

    assert len(metrics) == 270
    assert len(cells) == 108
    assert len(
        {
            (
                row["dataset"],
                row["training_seed"],
                row["calibration_seed"],
                row["corruption"],
            )
            for row in cells
        }
    ) == 108
    assert all(row["delta_e_ap_points"] == pytest.approx(0.2) for row in cells)

    with pytest.raises(RuntimeError, match="270"):
        analyzer.build_direct_cells(metrics[:-1])
    with pytest.raises(RuntimeError, match="duplicate"):
        analyzer.build_direct_cells([*metrics[:-1], metrics[0]])


def test_summary_reports_seed_marginals_signs_and_12_block_decompositions() -> None:
    analyzer = load_analyzer()
    cells = analyzer.build_direct_cells(synthetic_metric_rows())

    summary = analyzer.summarize_cells(cells)

    assert summary["overall"]["n"] == 108
    assert summary["overall"]["mean_delta_e_ap_points"] == pytest.approx(0.2)
    assert summary["sign_inventory"] == {"positive": 108, "negative": 0, "zero": 0}
    assert len(summary["marginals"]) == 14
    assert len(summary["variance_decomposition"]) == 12
    assert {
        (row["factor"], str(row["level"])) for row in summary["marginals"]
    } >= {
        ("training_seed", "20260807"),
        ("calibration_seed", "20260814"),
        ("dataset", "tt100k"),
        ("corruption", "motion_blur"),
    }


def test_analysis_outputs_are_self_hashed_and_refuse_overwrite(tmp_path: Path) -> None:
    analyzer = load_analyzer()
    cells = analyzer.build_direct_cells(synthetic_metric_rows())

    paths = analyzer.write_analysis_outputs(
        cells,
        output_dir=tmp_path,
        attempt="ivc_multiseed_yolo11m_s5_v1",
        config_sha256="a" * 64,
        metric_completion_report_sha256="b" * 64,
    )

    assert set(paths) == {
        "cells_csv",
        "marginals_csv",
        "decomposition_csv",
        "main_tex",
        "supplement_tex",
        "report_json",
    }
    with paths["cells_csv"].open(encoding="utf-8", newline="") as handle:
        assert len(list(csv.DictReader(handle))) == 108
    report = json.loads(paths["report_json"].read_text(encoding="utf-8"))
    assert report["report_sha256"] == analyzer.canonical_hash(report, "report_sha256")
    assert report["analyzer_sha256"] == analyzer.sha256_file(ANALYZER_PATH)
    assert report["counts"] == {
        "direct_cells": 108,
        "marginals": 14,
        "variance_blocks": 12,
    }
    main_tex = paths["main_tex"].read_text(encoding="utf-8")
    supplement_tex = paths["supplement_tex"].read_text(encoding="utf-8")
    assert "108 direct cells" in main_tex
    assert "YOLO11m" in main_tex and "severity~5" in main_tex
    assert "Training seed" in main_tex and "Calibration seed" in main_tex
    assert main_tex.count("2026080") + main_tex.count("2026081") == 6
    assert "+0.200" in main_tex
    assert r"\begin{table}[t]" in main_tex
    assert r"\resizebox{\columnwidth}{!}" in main_tex
    assert r"\begin{table*}" not in main_tex
    assert "12 dataset--corruption blocks" in supplement_tex
    assert supplement_tex.count(r"\\") >= 13
    assert "Gaussian noise" in supplement_tex
    assert "Motion blur" in supplement_tex
    assert "Train$\\times$cal. interaction (\\%)" in supplement_tex
    assert "interaction/residual" not in supplement_tex
    assert "Residual (\\%)" not in supplement_tex
    analyzer.validate_analysis_outputs(paths)
    paths["cells_csv"].write_text("changed\n", encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact hash"):
        analyzer.validate_analysis_outputs(paths)
    with pytest.raises(RuntimeError, match="overwrite"):
        analyzer.write_analysis_outputs(
            cells,
            output_dir=tmp_path,
            attempt="ivc_multiseed_yolo11m_s5_v1",
            config_sha256="a" * 64,
            metric_completion_report_sha256="b" * 64,
        )
