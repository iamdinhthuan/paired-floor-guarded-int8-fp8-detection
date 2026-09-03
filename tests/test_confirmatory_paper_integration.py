from __future__ import annotations

import json
from pathlib import Path

import pytest

from pilot_registry import canonical_hash
from create_overleaf_upload import (
    _resolve_relocated_parity_child,
    _resolve_relocated_confirmatory_components,
    _write_portable_confirmatory_manifest,
)
from submission_audit import _normalize_pdf_text, validate_confirmatory_integration
from topic_c.manifest import sha256_file

ROOT = Path(__file__).resolve().parents[1]


def test_pdf_text_normalization_repairs_common_tex_ligature_extraction() -> None:
    extracted = "A prespeci\ufb01ed untouched-partition rerun near a common AP \ufb02oor"

    assert _normalize_pdf_text(extracted) == (
        "A prespecified untouched-partition rerun near a common AP floor"
    )


def test_confirmatory_components_resolve_local_bytes_not_remote_origin_paths(tmp_path: Path) -> None:
    reports = tmp_path / "outputs" / "reports"
    reports.mkdir(parents=True)
    local = {
        "strict_reference_parity": reports / "voc_kitti_confirmatory_reference_parity_v2.json",
        "confirmatory_analysis": reports / "voc_kitti_confirmatory_analysis_v1.json",
        "realization_analysis": reports / "corruption_realization_analysis_v1.json",
        "fixed_universe_sensitivity": tmp_path / "outputs" / "bootstrap" / "fixed.json",
    }
    local["fixed_universe_sensitivity"].parent.mkdir(parents=True)
    for name, path in local.items():
        path.write_text(name + "\n", encoding="utf-8")
    audit = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in local.items()
        if name != "strict_reference_parity"
    }
    master = {
        "validated_components": {
            name: {
                "path": f"/home/thuan/topic_c_ivc/original/{name}.json",
                "sha256": sha256_file(path),
            }
            for name, path in local.items()
        }
    }

    resolved = _resolve_relocated_confirmatory_components(tmp_path, audit, master)

    assert resolved == local


def test_parity_child_resolves_canonical_local_copy_by_hash(tmp_path: Path) -> None:
    child = (
        tmp_path
        / "outputs/reports/reference_parity/voc_kitti_confirmatory_v1/voc_yolo11m.json"
    )
    child.parent.mkdir(parents=True)
    child.write_text("parity child\n", encoding="utf-8")
    row = {
        "dataset": "voc",
        "model": "yolo11m",
        "report": "/home/thuan/topic_c_ivc/original/voc_yolo11m.json",
        "report_sha256": sha256_file(child),
    }

    assert _resolve_relocated_parity_child(tmp_path, row) == child.resolve()

    child.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(ValueError, match="strict parity child report mismatch"):
        _resolve_relocated_parity_child(tmp_path, row)


def test_main_paper_integrates_holdout_evidence_in_self_contained_cviu_source() -> None:
    main = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")

    assert r"\subsection{Training, checkpoint selection, and holdout rerun}" in main
    assert r"\subsection{Evidence from final VOC and KITTI holdouts}" in main
    assert r"\input{generated/conditionality_scope_summary.tex}" in main
    assert "72 cells" in main
    assert "-0.55 AP points" in main
    assert "final images excluded from checkpoint selection" in main
    assert "strongest post-selection holdout evidence" in main
    for stale in (
        r"\newif\ifconfirmatoryevidence",
        "confirmatory_methods.tex",
        "confirmatory_results.tex",
        "confirmatory_conclusion.tex",
    ):
        assert stale not in main


def test_overleaf_builder_requires_and_copies_compact_confirmatory_evidence() -> None:
    builder = (ROOT / "analysis" / "create_overleaf_upload.py").read_text(
        encoding="utf-8"
    )

    assert "_validate_confirmatory_evidence(project_root)" in builder
    assert "_copy_confirmatory_evidence(project_root, staging)" in builder
    for name in (
        "confirmatory_evidence_audit.json",
        "confirmatory_methods.tex",
        "confirmatory_results.tex",
        "corruption_realization_results.tex",
        "fixed_universe_results.tex",
        "confirmatory_conclusion.tex",
        "confirmatory_supplement.tex",
        "confirmatory_abstract.tex",
        "confirmatory_highlights.txt",
    ):
        assert f'"{name}"' in builder


def test_supplement_integrates_detailed_holdout_and_realization_evidence() -> None:
    supplement = (ROOT / "paper" / "supplement.tex").read_text(encoding="utf-8")
    normalized = " ".join(supplement.split())

    assert r"\section{Selection-disjoint holdout and corruption-realization sensitivities}" in supplement
    for artifact in (
        "holdout_split_summary.tex",
        "holdout_reference_parity.tex",
        "corruption_realization_seed_summary.tex",
        "corruption_realization_variance.tex",
        "conditionality_scope_summary.tex",
    ):
        assert rf"\input{{generated/{artifact}}}" in supplement
    assert "does not convert the broader exploratory grid into a prospective confirmatory study" in normalized


def test_discussion_distinguishes_exploratory_grid_from_final_holdout() -> None:
    main = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")
    discussion = main.split(r"\section{Discussion}", 1)[1].split(
        r"\section{Conclusion}", 1
    )[0]

    assert "different scopes" in discussion
    assert "retrains six checkpoints" in discussion
    assert "final image lists" in discussion
    assert "should therefore not be treated as a failed replication" in discussion
    assert "The final rerun covers only VOC and KITTI" in discussion


def test_submission_gate_rejects_changed_confirmatory_fragment(tmp_path: Path) -> None:
    generated = tmp_path / "generated"
    generated.mkdir()
    names = {
        "confirmatory_methods.tex",
        "confirmatory_results.tex",
        "corruption_realization_results.tex",
        "confirmatory_conclusion.tex",
        "confirmatory_supplement.tex",
        "fixed_universe_results.tex",
        "confirmatory_abstract.tex",
        "confirmatory_highlights.txt",
    }
    records = {}
    for name in names:
        path = generated / name
        path.write_text(name + "\n", encoding="utf-8")
        records[name] = {"path": str(path), "sha256": sha256_file(path)}
    audit = {"schema_version": 1, "generated": records}
    audit["audit_sha256"] = canonical_hash(audit, "audit_sha256")
    audit_path = generated / "confirmatory_evidence_audit.json"
    audit_path.write_text(json.dumps(audit), encoding="utf-8")
    abstract = (generated / "confirmatory_abstract.tex").read_text(encoding="utf-8")
    abstract_body = "\n".join(
        line for line in abstract.splitlines() if not line.lstrip().startswith("%")
    )
    (tmp_path / "highlights.txt").write_text(
        (generated / "confirmatory_highlights.txt").read_text(encoding="utf-8"),
        encoding="utf-8",
    )
    main = (
        r"\begin{abstract}" + abstract_body + r"\end{abstract}" + "\n"
        + "\n".join(
            rf"\input{{generated/{name}}}"
            for name in names
            if name
            not in {
                "confirmatory_supplement.tex",
                "confirmatory_abstract.tex",
                "confirmatory_highlights.txt",
            }
        )
    )
    supplement = r"\input{generated/confirmatory_supplement.tex}"

    validate_confirmatory_integration(tmp_path, main, supplement)

    (generated / "confirmatory_results.tex").write_text("changed\n", encoding="utf-8")
    with pytest.raises(ValueError, match="hash mismatch"):
        validate_confirmatory_integration(tmp_path, main, supplement)


def test_portable_confirmatory_manifest_maps_master_components_to_packaged_names(tmp_path: Path) -> None:
    destination = tmp_path / "confirmatory_evidence"
    destination.mkdir()
    sources = {}
    names = {
        "master_completion": "master_completion.json",
        "confirmatory_analysis": "untouched_holdout_analysis.json",
    }
    for key, name in names.items():
        source = tmp_path / f"source_{key}.json"
        source.write_text(key + "\n", encoding="utf-8")
        packaged = destination / name
        packaged.write_bytes(source.read_bytes())
        sources[key] = source

    path = _write_portable_confirmatory_manifest(destination, sources, names)
    document = json.loads(path.read_text(encoding="utf-8"))

    assert document["schema_version"] == 1
    assert document["files"]["confirmatory_analysis"]["packaged_path"] == "untouched_holdout_analysis.json"
    assert document["files"]["confirmatory_analysis"]["sha256"] == sha256_file(
        destination / "untouched_holdout_analysis.json"
    )
    assert document["manifest_sha256"] == canonical_hash(document, "manifest_sha256")
