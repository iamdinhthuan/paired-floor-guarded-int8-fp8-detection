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
    assert len(document["generated_artifacts_sha256"]) == 5
    for relative, expected in document["generated_artifacts_sha256"].items():
        assert _sha(ROOT / relative) == expected
    audit = json.loads(
        (ROOT / "paper/generated/cross_family_evidence_audit.json").read_text(
            encoding="utf-8"
        )
    )
    audit_payload = {key: value for key, value in audit.items() if key != "audit_sha256"}
    assert audit["status"] == "valid"
    assert audit["analysis_file_sha256"] == _sha(path)
    assert audit["generated_artifacts_sha256"] == document["generated_artifacts_sha256"]
    assert audit["audit_sha256"] == hashlib.sha256(
        json.dumps(audit_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_cross_family_interaction_uses_matched_jpeg95_control() -> None:
    methods = (ROOT / "paper/cross_family_methods.tex").read_text(encoding="utf-8")
    results = (ROOT / "paper/cross_family_results.tex").read_text(encoding="utf-8")
    assert "Eighteen separately materialized JPEG-95 clean cells" in methods
    assert "original-source clean AP" in methods
    assert "dataset-level JPEG-95 matched-clean AP" in results
    assert "rather than an FP8 robustness gain" in " ".join(results.split())


def test_overleaf_builder_requires_compact_cross_family_evidence() -> None:
    builder = (ROOT / "analysis/create_overleaf_upload.py").read_text(encoding="utf-8")
    main = (ROOT / "paper/main.tex").read_text(encoding="utf-8")
    assert r"\IfFileExists{generated/cross_family_evidence_audit.json}" in main
    assert r"\PackageError{ivc-cross-family}" in main
    for required in (
        '"cross_family_methods.tex"',
        '"cross_family_results.tex"',
        '"cross_family_discussion.tex"',
        '"cross_family_q95_clean.csv"',
        '"cross_family_evidence_audit.json"',
        "_validate_cross_family_evidence(project_root)",
        "_copy_cross_family_evidence(project_root, staging)",
    ):
        assert required in builder
