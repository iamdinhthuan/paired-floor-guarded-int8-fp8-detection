#!/usr/bin/env python3
"""Create a self-contained, evidence-gated Overleaf upload folder.

The upload directory deliberately contains only editable submission sources and
their source-owned assets.  It excludes local build products and experiment
payloads.  A manifest makes the hand-off auditable, while a separate ZIP lets
the same directory be uploaded through Overleaf's ``New Project`` workflow.
"""

from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import tempfile
import zipfile
from pathlib import Path


_VENDORED_TEX = {
    Path("/data_nvme/texlive/2026/texmf-dist/tex/latex/els-cas-templates/cas-dc.cls"): "cas-dc.cls",
    Path("/data_nvme/texlive/2026/texmf-dist/tex/latex/els-cas-templates/cas-common.sty"): "cas-common.sty",
    Path("/data_nvme/texlive/2026/texmf-dist/bibtex/bst/els-cas-templates/cas-model2-names.bst"): "cas-model2-names.bst",
}

_DIRECT_BOUND_ASSETS = {
    "paper/generated/direct_format_contrast_cells.csv",
    "paper/generated/direct_format_contrast_macro.csv",
    "paper/generated/direct_deployment_conditions.csv",
    "paper/generated/direct_absolute_guardrail.csv",
    "paper/generated/direct_format_contrast_summary.tex",
    "paper/generated/direct_heterogeneity_summary.csv",
    "paper/generated/direct_heterogeneity_summary.tex",
    "paper/generated/direct_deployment_summary.tex",
    "paper/generated/direct_absolute_guardrail_summary.tex",
    "paper/generated/direct_absolute_guardrail_narrative.tex",
    "paper/generated/numbers_direct.tex",
    "paper/generated/direct_results_narrative.tex",
    "paper/generated/direct_deployment_narrative.tex",
    "paper/generated/direct_data_dictionary.json",
    "paper/generated/direct_abstract.tex",
    "paper/generated/direct_conclusion.tex",
    "paper/generated/codec_sensitivity.csv",
    "paper/generated/codec_sensitivity.tex",
    "paper/generated/direct_heterogeneity_sensitivity.csv",
    "paper/generated/direct_heterogeneity_sensitivity.tex",
    "paper/generated/direct_runtime_sensitivity.csv",
    "paper/generated/direct_runtime_sensitivity.tex",
    "paper/generated/direct_sensitivity_narrative.tex",
    "paper/generated/direct_size_guardrail.csv",
    "paper/generated/direct_size_guardrail_summary.tex",
    "paper/figures/fig_paired_excess_gap.pdf",
    "paper/figures/fig_paired_excess_gap.png",
}


def _sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _write_readme(destination: Path, *, direct_evidence_ready: bool) -> None:
    evidence_status = (
        """The exact direct paired INT8--FP8 completion chain has validated locally.
`main.tex` therefore selects the generated direct abstract, Results,
Discussion, Conclusion, and availability statement.  The direct
audit binds the generated prose/tables and paired figure by SHA-256; this
package includes those direct source assets.  The separate completed
multi-seed analysis binds a balanced 3 x 3 training--calibration seed
sensitivity for YOLO11m on VOC, KITTI, and TT100K at severity 5; its two
packaged LaTeX fragments are byte-checked against that analysis completion.
The cross-family extension adds RT-DETR-L and RetinaNet-R50-FPN-v2 on VOC,
KITTI, and TT100K. Its compact analysis validates 234 accuracy cells, 18
JPEG-95 matched-clean controls, and 54 engine-only timing repetitions before
the cross-family tables are copied into this package.
Author action required: the authors must still confirm
the funding statement, CRediT roles, archive creators, public data/code
location, and DOI-pending text before submission."""
        if direct_evidence_ready
        else """This snapshot is suitable for collaborative editing and layout review, not
for final submission.  Its historical P0 result tables remain in `main.tex`
only while the exact direct INT8--FP8 paired-contrast evidence chain is
running.  The final submission must replace that Results block with the
evidence-gated `direct_results_template.tex` output after the 12-clean,
144-corrupted, 2,000-replicate completion report validates locally.  The
manuscript automatically switches its abstract, Results, Discussion,
Conclusion, and availability statement to the direct-evidence blocks
only when the validated `generated/direct_evidence_audit.json` is present.
Before submission, authors must also confirm the funding statement, CRediT
roles, archive creators, public data/code location, and DOI-pending text."""
    )
    destination.write_text(
        """# IVC Overleaf upload

Set `main.tex` as the main document in Overleaf and compile with pdfLaTeX.
The local copies of `cas-dc.cls`, `cas-common.sty`, and
`cas-model2-names.bst` reproduce the Elsevier CAS layout used by the IVC
reference manuscript.  `supplement.tex` is a separately compilable
Supplementary File S1.  `graphical_abstract.png` is the 3000 x 1200 px
Elsevier graphical-abstract asset.

This is a source-only hand-off: it intentionally excludes local `.aux`,
`.log`, `.pdf`, raw predictions, datasets, engines, and credentials.

## Evidence status

"""
        + evidence_status
        + """

The `SOURCE_MANIFEST.sha256` file records the byte hashes of every packaged
file other than itself.
""",
        encoding="utf-8",
    )


def _write_vendor_notice(destination: Path) -> None:
    destination.write_text(
        """This directory contains unmodified local copies of the Elsevier CAS
class files `cas-dc.cls`, `cas-common.sty`, and `cas-model2-names.bst` so that
the Overleaf upload has its required class/style assets beside `main.tex`.
They originate from the installed TeX Live els-cas-templates distribution.
""",
        encoding="utf-8",
    )


def _write_manifest(root: Path) -> None:
    entries = []
    for path in sorted(item for item in root.rglob("*") if item.is_file()):
        relative = path.relative_to(root).as_posix()
        if relative == "SOURCE_MANIFEST.sha256":
            continue
        entries.append(f"{_sha256(path)}  {relative}")
    (root / "SOURCE_MANIFEST.sha256").write_text(
        "\n".join(entries) + "\n", encoding="utf-8"
    )


def _copy_tree(source: Path, destination: Path) -> None:
    if not source.is_dir():
        raise ValueError(f"required source directory is missing: {source}")
    shutil.copytree(source, destination)


def _copy_direct_assets(paper: Path, staging: Path) -> None:
    """Copy only files consumed by the fail-closed direct manuscript branch."""
    generated_names = {
        "numbers.tex",
        "datasets.tex",
        "direct_evidence_audit.json",
        "direct_format_contrast_cells.csv",
        "direct_format_contrast_macro.csv",
        "direct_deployment_conditions.csv",
        "direct_absolute_guardrail.csv",
        "direct_format_contrast_summary.tex",
        "direct_heterogeneity_summary.csv",
        "direct_heterogeneity_summary.tex",
        "direct_deployment_summary.tex",
        "direct_absolute_guardrail_summary.tex",
        "direct_absolute_guardrail_narrative.tex",
        "numbers_direct.tex",
        "direct_results_narrative.tex",
        "direct_deployment_narrative.tex",
        "direct_abstract.tex",
        "direct_conclusion.tex",
        "direct_data_dictionary.json",
        "codec_sensitivity.csv",
        "codec_sensitivity.tex",
        "direct_heterogeneity_sensitivity.csv",
        "direct_heterogeneity_sensitivity.tex",
        "direct_runtime_sensitivity.csv",
        "direct_runtime_sensitivity.tex",
        "direct_sensitivity_narrative.tex",
        "direct_size_guardrail.csv",
        "direct_size_guardrail_summary.tex",
        "multiseed_main.tex",
        "multiseed_supplement.tex",
        "cross_family_cells.csv",
        "cross_family_q95_clean.csv",
        "cross_family_clean.tex",
        "cross_family_interaction.tex",
        "cross_family_deployment.tex",
        "cross_family_evidence_audit.json",
        "cross_family_interaction_consistent.tex",
        "cross_family_q95_summary.tex",
        "cross_family_sign_consistency_audit.json",
        "quantization_graph_coverage.tex",
        "quantization_graph_coverage_audit.json",
        "confirmatory_evidence_audit.json",
        "confirmatory_methods.tex",
        "confirmatory_results.tex",
        "corruption_realization_results.tex",
        "fixed_universe_results.tex",
        "confirmatory_conclusion.tex",
        "confirmatory_supplement.tex",
        "confirmatory_abstract.tex",
        "confirmatory_highlights.txt",
    }
    figure_names = {
        "fig_framework.pdf",
        "fig_direct_heterogeneity_heatmap.png",
        "fig_cross_family_gates.png",
        "fig_paired_excess_gap.pdf",
        "fig_paired_excess_gap.png",
    }
    for directory, names in (("generated", generated_names), ("figures", figure_names)):
        destination = staging / directory
        destination.mkdir()
        for name in sorted(names):
            source = paper / directory / name
            if not source.is_file():
                raise ValueError(f"required direct source asset is missing: {source}")
            shutil.copy2(source, destination / name)


def _copy_multiseed_evidence(project_root: Path, staging: Path) -> None:
    """Copy the compact, hash-validated multi-seed audit surface."""
    attempt = "ivc_multiseed_yolo11m_s5_v1"
    analysis_root = project_root / "outputs" / "analysis" / attempt
    reports_root = project_root / "outputs" / "reports"
    sources = {
        project_root / "configs" / f"{attempt}.json": "config.json",
        reports_root / f"{attempt}_inference_complete.json":
            "inference_complete.json",
        reports_root / f"{attempt}_metric_complete.json": "metric_complete.json",
        reports_root / f"{attempt}_analysis_complete.json": "analysis_complete.json",
        analysis_root / f"{attempt}_analysis.json": "analysis.json",
        analysis_root / f"{attempt}_direct_cells.csv": "direct_cells.csv",
        analysis_root / f"{attempt}_marginals.csv": "marginals.csv",
        analysis_root / f"{attempt}_variance_decomposition.csv":
            "variance_decomposition.csv",
    }
    destination = staging / "multiseed_evidence"
    destination.mkdir()
    for source, name in sources.items():
        if not source.is_file():
            raise ValueError(f"required multi-seed evidence is missing: {source}")
        shutil.copy2(source, destination / name)
    (destination / "README.md").write_text(
        """# Multi-seed compact evidence

This directory records the balanced YOLO11m severity-5 sensitivity analysis
over VOC, KITTI, and TT100K. The grid contains three training seeds, three
calibration seeds, two formats, one JPEG-95 matched-clean control, and four
corruptions. The 270 metrics produce 108 direct format-contrast cells.

`analysis_complete.json` binds the compact analysis artifacts and the metric
completion report. `metric_complete.json` binds all 270 metric records and the
inference completion report. The latter records the remote prediction/input/run
artifact hashes; those 1.5 GiB prediction payloads are intentionally not copied
into this editable Overleaf source package. This is therefore a compact audit
surface, not a replacement for the future public evidence archive.
""",
        encoding="utf-8",
    )


def _copy_cross_family_evidence(project_root: Path, staging: Path) -> None:
    """Copy the compact cross-family audit without bulky prediction payloads."""
    sources = {
        project_root / "artifacts/cross_family_v1/analysis.json": "analysis.json",
        project_root / "artifacts/cross_family_v1/outputs/reports/cross_family_deployment_v1_complete.json": "deployment_complete.json",
        project_root / "artifacts/cross_family_v1/outputs/reports/cross_family_q95_clean_v1_complete.json": "q95_clean_complete.json",
        project_root / "artifacts/cross_family_v1/outputs/reports/cross_family_accuracy_v1_complete.json": "accuracy_complete.json",
    }
    destination = staging / "cross_family_evidence"
    destination.mkdir()
    for source, name in sources.items():
        if not source.is_file():
            raise ValueError(f"required cross-family evidence is missing: {source}")
        shutil.copy2(source, destination / name)
    (destination / "README.md").write_text(
        """# Cross-family compact evidence

This directory binds the descriptive RT-DETR-L and RetinaNet-R50-FPN-v2
extension on VOC, KITTI, and TT100K. `analysis.json` records the exact 234-cell
accuracy grid, 18 JPEG-95 matched-clean controls, 54 deployment repetitions,
generated-table hashes, and the local source-artifact hashes used to produce
them. The three completion reports close the accuracy, JPEG-95, and deployment ledgers.
Bulky predictions, datasets, engines, and checkpoints are intentionally not
part of this editable Overleaf source package.
""",
        encoding="utf-8",
    )


def _resolve_relocated_confirmatory_components(
    project_root: Path, audit: dict, master: dict
) -> dict[str, Path]:
    """Resolve local copies while preserving remote origin paths in the ledgers."""
    components = master.get("validated_components", {})
    candidates = {
        "strict_reference_parity": project_root
        / "outputs/reports/voc_kitti_confirmatory_reference_parity_v2.json",
        "confirmatory_analysis": Path(
            audit.get("confirmatory_analysis", {}).get("path", "")
        ),
        "realization_analysis": Path(
            audit.get("realization_analysis", {}).get("path", "")
        ),
        "fixed_universe_sensitivity": Path(
            audit.get("fixed_universe_sensitivity", {}).get("path", "")
        ),
    }
    resolved = {}
    for name, candidate in candidates.items():
        path = candidate.resolve()
        binding = components.get(name, {})
        if not path.is_file() or _sha256(path) != binding.get("sha256"):
            raise ValueError(f"confirmatory master component mismatch: {name}")
        audit_record = audit.get(name)
        if audit_record is not None and _sha256(path) != audit_record.get("sha256"):
            raise ValueError(f"confirmatory paper component mismatch: {name}")
        resolved[name] = path
    return resolved


def _validate_confirmatory_evidence(project_root: Path) -> dict[str, Path]:
    """Validate the compact master chain without copying bulky predictions/caches."""
    paper = project_root / "paper"
    audit_path = paper / "generated" / "confirmatory_evidence_audit.json"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("confirmatory paper audit is missing") from error
    payload = {key: value for key, value in audit.items() if key != "audit_sha256"}
    if hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest() != audit.get("audit_sha256"):
        raise ValueError("confirmatory paper audit self-hash is invalid")
    generated = audit.get("generated")
    expected_generated = {
        "confirmatory_methods.tex",
        "confirmatory_results.tex",
        "corruption_realization_results.tex",
        "fixed_universe_results.tex",
        "confirmatory_conclusion.tex",
        "confirmatory_supplement.tex",
        "confirmatory_abstract.tex",
        "confirmatory_highlights.txt",
    }
    if not isinstance(generated, dict) or set(generated) != expected_generated:
        raise ValueError("confirmatory generated fragment grid is invalid")
    for name, record in generated.items():
        path = (paper / "generated" / name).resolve()
        if Path(record.get("path", "")).resolve() != path or not path.is_file() or _sha256(path) != record.get("sha256"):
            raise ValueError(f"confirmatory generated fragment mismatch: {name}")
    master_record = audit.get("master_completion", {})
    master_path = Path(master_record.get("path", "")).resolve()
    master_marker = master_path.with_suffix(master_path.suffix + ".complete")
    if (
        not master_path.is_file()
        or not master_marker.is_file()
        or _sha256(master_path) != master_record.get("sha256")
        or master_marker.read_text(encoding="utf-8").strip() != _sha256(master_path)
    ):
        raise ValueError("confirmatory master completion bytes are invalid")
    master = json.loads(master_path.read_text(encoding="utf-8"))
    master_payload = {key: value for key, value in master.items() if key != "completion_sha256"}
    if (
        master.get("status") != "complete"
        or hashlib.sha256(
            json.dumps(master_payload, sort_keys=True, separators=(",", ":")).encode()
        ).hexdigest() != master.get("completion_sha256")
        or master.get("scope")
        != {
            "strict_reference_parity_blocks": 6,
            "untouched_holdout_direct_cells": 72,
            "corruption_realization_cells": 108,
            "corruption_realization_base_conditions": 36,
            "fixed_universe_bootstrap_draws": 10000,
        }
    ):
        raise ValueError("confirmatory master completion scope or self-hash is invalid")
    components = master.get("validated_components")
    expected_components = {
        "strict_reference_parity",
        "confirmatory_analysis",
        "realization_analysis",
        "fixed_universe_sensitivity",
    }
    if not isinstance(components, dict) or set(components) != expected_components:
        raise ValueError("confirmatory master component grid is invalid")
    paths = {"master_completion": master_path, "paper_audit": audit_path}
    paths.update(_resolve_relocated_confirmatory_components(project_root, audit, master))
    return paths


def _write_portable_confirmatory_manifest(
    destination: Path, sources: dict[str, Path], names: dict[str, str]
) -> Path:
    """Bind immutable source bytes to their root-portable packaged filenames."""
    if set(sources) != set(names) or not sources:
        raise ValueError("portable confirmatory source/name grid mismatch")
    files = {}
    for key in sorted(sources):
        source = sources[key].resolve()
        packaged = destination / names[key]
        if not source.is_file() or not packaged.is_file() or _sha256(source) != _sha256(packaged):
            raise ValueError(f"portable confirmatory byte mismatch: {key}")
        files[key] = {
            "packaged_path": names[key],
            "sha256": _sha256(packaged),
        }
    document = {
        "schema_version": 1,
        "purpose": "portable filename map for immutable confirmatory-strengthening evidence bytes",
        "files": files,
    }
    payload = json.dumps(document, sort_keys=True, separators=(",", ":")).encode()
    document["manifest_sha256"] = hashlib.sha256(payload).hexdigest()
    output = destination / "PORTABLE_MANIFEST.json"
    if output.exists():
        raise ValueError("portable confirmatory manifest already exists")
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return output


def _resolve_relocated_parity_child(project_root: Path, row: dict) -> Path:
    dataset = row.get("dataset")
    model = row.get("model")
    if dataset not in {"voc", "kitti"} or model not in {
        "yolo11n",
        "yolo11m",
        "yolo11x",
    }:
        raise ValueError("strict parity child identity is invalid")
    path = (
        project_root
        / "outputs/reports/reference_parity/voc_kitti_confirmatory_v1"
        / f"{dataset}_{model}.json"
    ).resolve()
    if not path.is_file() or _sha256(path) != row.get("report_sha256"):
        raise ValueError("strict parity child report mismatch")
    return path


def _copy_confirmatory_evidence(project_root: Path, staging: Path) -> None:
    """Copy the compact strengthening reports, never predictions or draw caches."""
    sources = _validate_confirmatory_evidence(project_root)
    destination = staging / "confirmatory_evidence"
    destination.mkdir()
    names = {
        "master_completion": "master_completion.json",
        "paper_audit": "paper_evidence_audit.json",
        "strict_reference_parity": "strict_reference_parity.json",
        "confirmatory_analysis": "untouched_holdout_analysis.json",
        "realization_analysis": "corruption_realization_analysis.json",
        "fixed_universe_sensitivity": "tt100k_fixed_universe_b10000.json",
    }
    for key, name in names.items():
        shutil.copy2(sources[key], destination / name)
    portable_sources = dict(sources)
    portable_names = dict(names)
    config = project_root / "configs" / "strengthening_completion_v1.json"
    if not config.is_file():
        raise ValueError("strengthening completion config is missing")
    shutil.copy2(config, destination / "strengthening_completion_config.json")
    portable_sources["strengthening_completion_config"] = config
    portable_names["strengthening_completion_config"] = "strengthening_completion_config.json"
    parity = json.loads(sources["strict_reference_parity"].read_text(encoding="utf-8"))
    parity_destination = destination / "strict_parity_blocks"
    parity_destination.mkdir()
    for row in parity.get("reports", []):
        source = _resolve_relocated_parity_child(project_root, row)
        packaged_name = f"strict_parity_blocks/{row['dataset']}_{row['model']}.json"
        shutil.copy2(source, destination / packaged_name)
        key = f"strict_parity_{row['dataset']}_{row['model']}"
        portable_sources[key] = source
        portable_names[key] = packaged_name
    _write_portable_confirmatory_manifest(
        destination, portable_sources, portable_names
    )
    (destination / "README.md").write_text(
        """# Confirmatory-strengthening compact evidence

This directory contains the master completion report and compact analyses for
the untouched VOC/KITTI final-partition rerun, three-realization corruption
sensitivity, strict four-backend parity, and the targeted TT100K fixed-category
B=10,000 bootstrap. The master report binds the full remote evidence chain by
SHA-256. Raw predictions, datasets, engines, schedules, and draw caches are
intentionally excluded from this editable Overleaf package and belong in the
persistent public archive.
""",
        encoding="utf-8",
    )


def _validate_cross_family_evidence(project_root: Path) -> None:
    analysis_path = project_root / "artifacts/cross_family_v1/analysis.json"
    try:
        analysis = json.loads(analysis_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("validated cross-family analysis is missing") from error
    payload = {key: value for key, value in analysis.items() if key != "analysis_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if (
        analysis.get("schema_version") != 2
        or analysis.get("metrics") != 234
        or analysis.get("matched_clean_metrics") != 18
        or analysis.get("deployment_records") != 54
        or hashlib.sha256(canonical).hexdigest() != analysis.get("analysis_sha256")
    ):
        raise ValueError("cross-family analysis identity/grid is invalid")
    bindings = analysis.get("generated_artifacts_sha256")
    if not isinstance(bindings, dict) or len(bindings) != 5:
        raise ValueError("cross-family generated-artifact set is invalid")
    for relative, expected in bindings.items():
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("cross-family generated-artifact path is unsafe")
        path = project_root / candidate
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"cross-family generated artifact mismatch: {relative}")
    audit_path = project_root / "paper/generated/cross_family_evidence_audit.json"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("cross-family evidence audit is missing") from error
    audit_payload = {key: value for key, value in audit.items() if key != "audit_sha256"}
    audit_digest = hashlib.sha256(
        json.dumps(audit_payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()
    if (
        audit.get("status") != "valid"
        or audit.get("audit_sha256") != audit_digest
        or audit.get("analysis_file_sha256") != _sha256(analysis_path)
        or audit.get("analysis_sha256") != analysis.get("analysis_sha256")
        or audit.get("generated_artifacts_sha256") != bindings
    ):
        raise ValueError("cross-family evidence audit binding is invalid")


def _validate_direct_evidence_audit(project_root: Path) -> None:
    """Confirm the direct package signal and generated artifact byte bindings."""
    audit_path = project_root / "paper" / "generated" / "direct_evidence_audit.json"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("validated direct-evidence audit is missing or invalid") from error
    bindings = audit.get("generated_artifacts_sha256") if isinstance(audit, dict) else None
    if (
        audit.get("status") != "valid"
        or not isinstance(bindings, dict)
        or set(bindings) != _DIRECT_BOUND_ASSETS
    ):
        raise ValueError("direct-evidence audit does not bind its exact generated artifact set")
    for relative, expected in bindings.items():
        candidate = Path(relative)
        if (
            not isinstance(relative, str)
            or not isinstance(expected, str)
            or len(expected) != 64
            or candidate.is_absolute()
            or ".." in candidate.parts
            or candidate.parts[:1] != ("paper",)
        ):
            raise ValueError("direct-evidence audit has an unsafe generated artifact binding")
        path = (project_root / candidate).resolve()
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"direct-evidence generated artifact hash mismatch: {relative}")


def _validate_multiseed_evidence(project_root: Path) -> None:
    """Validate the completed analysis chain and exact packaged TeX fragments."""
    attempt = "ivc_multiseed_yolo11m_s5_v1"
    completion_path = (
        project_root / "outputs" / "reports" / f"{attempt}_analysis_complete.json"
    )
    try:
        completion = json.loads(completion_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as error:
        raise ValueError("validated multi-seed analysis completion is missing") from error
    payload = {key: value for key, value in completion.items() if key != "report_sha256"}
    canonical = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    if (
        completion.get("status") != "complete"
        or completion.get("stage") != "analysis"
        or hashlib.sha256(canonical).hexdigest() != completion.get("report_sha256")
    ):
        raise ValueError("multi-seed analysis completion identity is invalid")
    bindings = completion.get("artifacts_sha256")
    if not isinstance(bindings, dict) or len(bindings) != 7:
        raise ValueError("multi-seed analysis completion grid is invalid")
    for relative, expected in bindings.items():
        candidate = Path(relative)
        if candidate.is_absolute() or ".." in candidate.parts:
            raise ValueError("multi-seed analysis completion path is unsafe")
        path = project_root / candidate
        if not path.is_file() or _sha256(path) != expected:
            raise ValueError(f"multi-seed analysis artifact hash mismatch: {relative}")
    analysis_root = project_root / "outputs" / "analysis" / attempt
    fragment_pairs = {
        project_root / "paper" / "generated" / "multiseed_main.tex":
            analysis_root / f"{attempt}_main.tex",
        project_root / "paper" / "generated" / "multiseed_supplement.tex":
            analysis_root / f"{attempt}_supplement.tex",
    }
    for packaged, analyzed in fragment_pairs.items():
        if not packaged.is_file() or _sha256(packaged) != _sha256(analyzed):
            raise ValueError(f"multi-seed packaged fragment mismatch: {packaged.name}")


def build_overleaf_upload(
    project_root: Path, output: Path, *, require_direct_evidence: bool = False
) -> tuple[Path, Path]:
    """Create a directory and root-flat ZIP without silently replacing either."""
    project_root = project_root.resolve()
    paper = project_root / "paper"
    output = output.resolve()
    archive = output.with_suffix(".zip")
    required_files = (
        "main.tex",
        "supplement.tex",
        "references.bib",
        "CITATION.cff",
        "zenodo.json",
        "highlights.txt",
        "abstract_p0_fallback.tex",
        "direct_results_template.tex",
        "direct_methods_tail.tex",
        "direct_discussion_template.tex",
        "direct_availability_template.tex",
        "cross_family_methods.tex",
        "cross_family_results.tex",
        "cross_family_discussion.tex",
        "graphical_abstract.png",
        "graphical_abstract.pdf",
    )
    missing = [str(paper / name) for name in required_files if not (paper / name).is_file()]
    missing.extend(str(path) for path in _VENDORED_TEX if not path.is_file())
    if missing:
        raise ValueError("cannot package missing source asset(s): " + ", ".join(missing))
    if output.exists() or archive.exists():
        raise ValueError(
            f"refusing to overwrite existing Overleaf hand-off: {output} or {archive}"
        )
    if output.parent != paper.resolve():
        raise ValueError("Overleaf hand-off must be created directly inside paper/")
    direct_audit_present = (paper / "generated" / "direct_evidence_audit.json").is_file()
    direct_evidence_ready = False
    if direct_audit_present:
        _validate_direct_evidence_audit(project_root)
        direct_evidence_ready = True
    if require_direct_evidence and not direct_evidence_ready:
        raise ValueError("validated direct-evidence audit is required for this package")
    _validate_multiseed_evidence(project_root)
    _validate_cross_family_evidence(project_root)
    _validate_confirmatory_evidence(project_root)

    with tempfile.TemporaryDirectory(prefix=f".{output.name}-", dir=output.parent) as temporary:
        staging = Path(temporary) / output.name
        staging.mkdir()
        for name in required_files:
            shutil.copy2(paper / name, staging / name)
        if direct_evidence_ready:
            _copy_direct_assets(paper, staging)
        else:
            _copy_tree(paper / "generated", staging / "generated")
            _copy_tree(paper / "figures", staging / "figures")
        _copy_multiseed_evidence(project_root, staging)
        _copy_cross_family_evidence(project_root, staging)
        _copy_confirmatory_evidence(project_root, staging)
        for source, name in _VENDORED_TEX.items():
            shutil.copy2(source, staging / name)
        _write_readme(staging / "README.md", direct_evidence_ready=direct_evidence_ready)
        _write_vendor_notice(staging / "CAS_TEMPLATE_NOTICE.txt")
        _write_manifest(staging)

        temporary_archive = Path(temporary) / archive.name
        with zipfile.ZipFile(temporary_archive, "w", compression=zipfile.ZIP_DEFLATED) as bundle:
            for path in sorted(item for item in staging.rglob("*") if item.is_file()):
                bundle.write(path, path.relative_to(staging).as_posix())

        staging.rename(output)
        temporary_archive.rename(archive)
    return output, archive


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/overleaf_upload_ivc_current_20260812"),
        help="new directory, directly below paper/ (default: %(default)s)",
    )
    parser.add_argument(
        "--require-direct-evidence",
        action="store_true",
        help="refuse package creation unless the direct audit and all 27 generated hashes validate",
    )
    arguments = parser.parse_args()
    project_root = arguments.project_root.resolve()
    output = arguments.output
    if not output.is_absolute():
        output = project_root / output
    directory, archive = build_overleaf_upload(
        project_root, output, require_direct_evidence=arguments.require_direct_evidence
    )
    print(f"Overleaf directory: {directory}")
    print(f"Overleaf ZIP: {archive}")


if __name__ == "__main__":
    main()
