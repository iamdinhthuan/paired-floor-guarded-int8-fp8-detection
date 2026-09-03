import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def _sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def test_cross_family_analysis_is_complete_and_binds_generated_tables() -> None:
    path = ROOT / "artifacts/cross_family_v1/analysis.json"
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in document.items() if key != "analysis_sha256"}
    digest = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    assert document["schema_version"] == 2
    assert document["metrics"] == document["runs"] == 234
    assert document["matched_clean_metrics"] == document["matched_clean_runs"] == 18
    assert document["deployment_records"] == 54
    assert document["analysis_sha256"] == digest
    assert len(document["source_artifacts_sha256"]) == 507
    for relative, expected in document["source_artifacts_sha256"].items():
        assert _sha(ROOT / relative) == expected

    portable_path = ROOT / "paper/cross_family_evidence/analysis.json"
    portable = json.loads(portable_path.read_text(encoding="utf-8"))
    assert portable["schema_version"] == 3
    assert portable["status"] == "valid"
    assert portable["rows"] == {"direct_interactions": 72, "quantized_interactions": 144}
    for relative, expected in portable["source_sha256"].items():
        assert _sha(ROOT / "paper" / relative) == expected
    direct_rows = (ROOT / "paper/generated/cross_family_direct_cells.csv").read_text(
        encoding="utf-8"
    ).splitlines()
    assert len(direct_rows) - 1 == 72


def test_cross_family_interaction_uses_matched_jpeg95_control() -> None:
    main = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    supplement = (ROOT / "paper/supplement.tex").read_text(encoding="utf-8")
    normalized = " ".join(supplement.split())
    assert "18 separate clean cells encoded at JPEG quality 95" in main
    assert "Matched JPEG-95 clean AP" in supplement
    assert r"E=Q_c-Q_{\Jclean}" in supplement
    assert "failure of recipe portability under these recorded treatments" in supplement
    assert "not an intrinsic limitation of INT8 or a universal advantage of FP8" in normalized


def test_cviu_builder_packages_the_active_cross_family_tables() -> None:
    builder = (ROOT / "analysis/build_cviu_submission_package.py").read_text(
        encoding="utf-8"
    )
    main = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    supplement = (ROOT / "paper/supplement.tex").read_text(encoding="utf-8")
    for required in (
        "cross_family_q95_summary.tex",
        "cross_family_interaction.tex",
        "cross_family_direct_summary.tex",
        "cross_family_runtime_by_dataset.tex",
    ):
        assert rf"\input{{generated/{required}}}" in supplement
    assert "publication_dependencies(root)" in builder
    assert '(stage / "generated").mkdir()' in builder
    assert "for path in inputs" in builder
    assert "cross_family_evidence_audit.json" not in main
