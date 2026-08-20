from pathlib import Path


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER = PROJECT_ROOT / "paper"


def test_main_integrates_targeted_multiseed_evidence_without_overgeneralizing() -> None:
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    discussion = (PAPER / "direct_discussion_template.tex").read_text(
        encoding="utf-8"
    )
    fragment = (PAPER / "generated" / "multiseed_main.tex").read_text(
        encoding="utf-8"
    )
    normalized_main = " ".join(main.split())

    assert r"\input{generated/multiseed_main.tex}" in main
    assert "3\\times3\\times3\\times4=108" in normalized_main
    assert "YOLO11m" in fragment
    assert "severity~5" in fragment
    assert "VOC, KITTI, and TT100K" in fragment
    assert r"targeted $3\times3$ training--calibration seed experiment" in discussion
    assert "engine-build variability" in discussion
    assert "family-level replication" in discussion
    assert "COCO" in discussion
    assert "original-seed intersection" in main
    assert "-1.214" in main
    assert "-1.109" in main
    assert "primary grid checkpoints were trained once" in main
    assert "These transfer models were initialized from pretrained weights and trained once" not in main
    assert "without replacement from the same frozen training split" in normalized_main
    assert "subset overlap was permitted" in normalized_main
    assert "same hash-bound JPEG-95 clean and corrupted inputs" in normalized_main
    assert "Only the training seed and calibration-list seed varied" in normalized_main


def test_supplement_integrates_hash_validated_seed_decomposition() -> None:
    supplement = (PAPER / "supplement.tex").read_text(encoding="utf-8")

    assert r"\input{generated/multiseed_supplement.tex}" in supplement
    assert "do not cover training seeds" not in supplement


def test_multiseed_generated_fragments_are_packaged_for_overleaf() -> None:
    creator = (PROJECT_ROOT / "analysis" / "create_overleaf_upload.py").read_text(
        encoding="utf-8"
    )

    assert '"multiseed_main.tex"' in creator
    assert '"multiseed_supplement.tex"' in creator
    assert "_validate_multiseed_evidence(project_root)" in creator
    assert "_copy_multiseed_evidence(project_root, staging)" in creator
    assert '"analysis_complete.json"' in creator
    assert '"direct_cells.csv"' in creator
