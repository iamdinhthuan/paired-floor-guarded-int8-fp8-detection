from __future__ import annotations

import hashlib
import json
from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER = PROJECT_ROOT / "paper"


def test_main_integrates_targeted_multiseed_evidence_without_overgeneralizing() -> None:
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    discussion = main.split(r"\section{Discussion}", 1)[1].split(
        r"\section{Conclusion}", 1
    )[0]
    normalized_main = " ".join(main.split())

    assert r"$3\times3$ factorial design yields 54 TensorRT engines and 108" in normalized_main
    assert "YOLO11m on VOC, KITTI, and TT100K at severity level 5" in normalized_main
    assert "not treated as estimates of variance over a broader population" in normalized_main
    assert "The targeted seed analysis fixes YOLO11m" in discussion
    assert "We did not repeat engine builds" in discussion
    assert "does not cover COCO" in main
    assert "combination of the original seeds" in main
    assert "-1.21" in main
    assert "-1.11" in main
    assert "one main training seed" in main
    assert "All other elements of the experimental procedure remain fixed" in normalized_main
    assert "variation is restricted to the training seed and the calibration-list seed" in normalized_main


def test_supplement_integrates_hash_validated_seed_decomposition() -> None:
    supplement = (PAPER / "supplement.tex").read_text(encoding="utf-8")

    assert r"\section{Training--calibration seed sensitivity}" in supplement
    assert r"\label{tab:multiseed-decomposition}" in supplement
    assert "Across 108 cells" in supplement
    assert "do not extend to COCO" in supplement

    evidence = PAPER / "multiseed_evidence"
    analysis = json.loads((evidence / "analysis.json").read_text(encoding="utf-8"))
    assert analysis["status"] == "complete"
    assert analysis["counts"] == {
        "direct_cells": 108,
        "marginals": 14,
        "variance_blocks": 12,
    }
    names = {
        "cells_csv": "direct_cells.csv",
        "decomposition_csv": "variance_decomposition.csv",
        "marginals_csv": "marginals.csv",
    }
    for key, name in names.items():
        digest = hashlib.sha256((evidence / name).read_bytes()).hexdigest()
        assert digest == analysis["artifacts_sha256"][key]


def test_multiseed_public_evidence_is_separate_from_the_compact_cviu_source() -> None:
    creator = (PROJECT_ROOT / "analysis" / "build_cviu_submission_package.py").read_text(
        encoding="utf-8"
    )

    assert "publication_dependencies(root)" in creator
    assert "multiseed_evidence" not in creator
    for name in (
        "analysis.json",
        "analysis_complete.json",
        "direct_cells.csv",
        "marginals.csv",
        "variance_decomposition.csv",
    ):
        assert (PAPER / "multiseed_evidence" / name).is_file()
