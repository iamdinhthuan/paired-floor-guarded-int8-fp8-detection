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


def test_main_paper_gates_confirmatory_fragments_on_validated_audit() -> None:
    main = (ROOT / "paper" / "main.tex").read_text(encoding="utf-8")

    assert r"\newif\ifconfirmatoryevidence" in main
    assert r"\IfFileExists{generated/confirmatory_evidence_audit.json}" in main
    assert r"\input{generated/confirmatory_methods.tex}" in main
    assert r"\input{generated/confirmatory_results.tex}" in main
    assert r"\input{generated/corruption_realization_results.tex}" in main
    assert r"\input{generated/fixed_universe_results.tex}" in main
    assert r"\input{generated/confirmatory_conclusion.tex}" in main


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


def test_supplement_gates_detailed_confirmatory_design_on_same_audit() -> None:
    supplement = (ROOT / "paper" / "supplement.tex").read_text(encoding="utf-8")

    assert r"\newif\ifconfirmatoryevidence" in supplement
    assert r"\IfFileExists{generated/confirmatory_evidence_audit.json}" in supplement
    assert r"\input{generated/confirmatory_supplement.tex}" in supplement


def test_discussion_distinguishes_historical_primary_from_confirmatory_rerun() -> None:
    discussion = (ROOT / "paper" / "direct_discussion_template.tex").read_text(
        encoding="utf-8"
    )

    assert r"\ifconfirmatoryevidence" in discussion
    assert "historical exploratory primary grid" in discussion
    assert "untouched final partitions" in discussion
    assert "explicitly disabled TF32" in discussion
    assert "three fixed corruption realizations" in discussion


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
