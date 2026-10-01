"""Integration guards for the NN activation and KITTI holdout extensions."""
from __future__ import annotations

import hashlib
import json
import shutil
import subprocess
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
SUPPORT = ROOT / "submission_support_20260911"
PAPER = ROOT / "paper"
PHASE_L = SUPPORT / "phase_l_holdout_results"


def _sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_fcos_activation_summaries_reach_generated_tables() -> None:
    stats = json.loads((SUPPORT / "nn_final_stats.json").read_text())
    assert {"kitti_fcos", "voc_fcos"} <= set(stats["actstats"])

    regional = (PAPER / "generated" / "nn_actstats.tex").read_text()
    per_level = (PAPER / "generated" / "nn_actstats_level.tex").read_text()
    macros = (PAPER / "generated" / "nn_numbers.tex").read_text()
    assert "KITTI FCOS (trained)" in regional
    assert "VOC FCOS (trained)" in regional
    assert "KITTI FCOS" in per_level and "VOC FCOS" in per_level
    assert r"\nnActRegKurtPThreeFcosKitti" in macros
    assert r"\nnActRegKurtPThreeFcosVoc" in macros


def test_holdout_has_four_complete_arms_and_is_in_supplement() -> None:
    cells = json.loads((PHASE_L / "cell_points.json").read_text())
    block = cells["kitti_holdout/retinanet_r50_fpn_v2"]
    expected = {
        "int8-matched512", "int8-selective512", "int8-wonly512", "int8-aonly512"
    }
    assert set(block) == expected
    for arm, conditions in block.items():
        assert len(conditions) == 14, arm
        assert "clean-s0" in conditions
        assert "codec-control-s0" in conditions

    supplement = (PAPER / "supplement.tex").read_text()
    table = (PAPER / "generated" / "nn_holdout.tex").read_text()
    main = (PAPER / "main_nn.tex").read_text()
    assert r"\input{generated/nn_holdout.tex}" in supplement
    assert r"\label{tab:nn-holdout}" in supplement
    assert "KITTI final-holdout check" in supplement
    assert "Table~S10" in main
    assert "34.70" in table and "65.25" in table


def test_holdout_metric_records_bind_inputs_runs_and_engines() -> None:
    metrics_dir = PHASE_L / "metrics"
    assert len(list(metrics_dir.glob("*.json"))) == 56
    points = json.loads((PHASE_L / "cell_points.json").read_text())[
        "kitti_holdout/retinanet_r50_fpn_v2"
    ]
    annotation_sha = _sha256(
        ROOT / "manifests" / "annotations" / "kitti_confirmatory_final_v1_coco.json"
    )
    calibration_marker = (
        PHASE_L / "manifests" / "calibration"
        / "kitti_train_clean_512_s20260807_v1.json.complete"
    ).read_text().strip()
    codec_marker = (
        PHASE_L / "manifests" / "images"
        / "kitti_final_codec_control_q95_p0_v1.json.complete"
    ).read_text().strip()

    for metric_path in metrics_dir.glob("*.json"):
        metric = json.loads(metric_path.read_text())
        remainder = metric_path.stem.removeprefix(
            "kitti_holdout__retinanet_r50_fpn_v2__"
        )
        arm, condition = remainder.split("__", 1)
        input_path = PHASE_L / "inputs" / metric_path.name
        run_path = PHASE_L / "runs" / metric_path.name
        input_record = json.loads(input_path.read_text())
        run_record = json.loads(run_path.read_text())

        assert metric["n_images"] == 1197
        assert input_record["image_ids_sha256"] == metric["input_image_ids_sha256"]
        assert run_record["input_image_ids_sha256"] == metric["input_image_ids_sha256"]
        assert input_record["input_manifest_sha256"] == metric["input_manifest_sha256"]
        assert run_record["input_manifest_sha256"] == metric["input_manifest_sha256"]
        assert _sha256(run_path) == metric["run_record_sha256"]
        assert run_record["prediction_sha256"] == metric["prediction_sha256"]
        assert abs(points[arm][condition] - 100 * metric["stats"]["AP"]) < 1e-9
        assert run_record["annotation_sha256"] == annotation_sha

        engine_dir = PHASE_L / "engines" / f"kitti_retinanet_{arm}"
        engine_path = engine_dir / "trt" / "engine.json"
        engine = json.loads(engine_path.read_text())
        onnx_path = engine_dir / "onnx.json"
        onnx_registry = json.loads(onnx_path.read_text())
        assert _sha256(engine_path) == run_record["engine_registry_sha256"]
        assert engine["engine_sha256"] == run_record["engine_sha256"]
        assert _sha256(onnx_path) == engine["source_onnx_registry_sha256"]
        assert engine["source_onnx_sha256"] == onnx_registry["output_onnx_sha256"]
        assert engine["calibration_sha256"] == calibration_marker

        if condition == "clean-s0":
            expected_input_manifest = f"clean-root:{annotation_sha}"
        elif condition == "codec-control-s0":
            expected_input_manifest = codec_marker
        else:
            corruption, severity = condition.rsplit("-s", 1)
            marker = (
                ROOT / "manifests" / "images" / "voc_kitti_confirmatory_v1"
                / f"kitti_final_{corruption}_s{severity}.json.complete"
            )
            expected_input_manifest = marker.read_text().strip()
        assert metric["input_manifest_sha256"] == expected_input_manifest

    assert not list(PHASE_L.rglob("*.engine"))
    assert not list(PHASE_L.rglob("*.onnx"))


def test_phase_l_manifest_covers_compact_evidence() -> None:
    manifest = PHASE_L / "MANIFEST.sha256"
    assert manifest.is_file()
    records = {}
    for line in manifest.read_text().splitlines():
        digest, relative = line.split("  ", 1)
        records[relative] = digest
    expected = {
        path.relative_to(PHASE_L).as_posix(): _sha256(path)
        for path in PHASE_L.rglob("*")
        if path.is_file() and path != manifest
    }
    assert records == expected


def test_built_pdf_text_layer_preserves_ligatures_and_punctuation(tmp_path: Path) -> None:
    pdf = PAPER / "main_nn.pdf"
    pdftotext = shutil.which("pdftotext")
    if not pdf.is_file() or not pdftotext:
        pytest.skip("built manuscript PDF or pdftotext is unavailable")

    text_path = tmp_path / "main.txt"
    subprocess.run(
        [pdftotext, "-layout", str(pdf), str(text_path)],
        check=True,
        capture_output=True,
        text=True,
    )
    text = text_path.read_text(encoding="utf-8")
    for token in ("deficit", "off-the-shelf", "—", "“", "”"):
        assert token in text
    for damaged in ("decit", "dened", "o-the-shelf", "contractthe"):
        assert damaged not in text
