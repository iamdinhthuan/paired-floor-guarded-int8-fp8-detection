"""Submission-facing citation and archive metadata contracts."""

from __future__ import annotations

import base64
import json
import os
import re
import shutil
import subprocess
import sys
import zipfile
from pathlib import Path

import pytest
import yaml


PROJECT_ROOT = Path(__file__).resolve().parents[1]
PAPER = PROJECT_ROOT / "paper"
sys.path.insert(0, str(PROJECT_ROOT / "analysis"))
import submission_audit  # noqa: E402
import submission_metadata  # noqa: E402
import submission_package  # noqa: E402
IVC_DOIS = {
    "10.1016/j.imavis.2023.104692",
    "10.1016/j.imavis.2024.105035",
    "10.1016/j.imavis.2024.105054",
    "10.1016/j.imavis.2024.105095",
    "10.1016/j.imavis.2025.105740",
}

_ONE_PIXEL_PNG = base64.b64decode(
    "iVBORw0KGgoAAAANSUhEUgAAAAEAAAABCAIAAACQd1PeAAAADElEQVR4nGP4z8AAAAMBAQDJ"
    "/pLvAAAAAElFTkSuQmCC"
)


def bibtex_entries(text: str) -> dict[str, str]:
    return submission_metadata.bibtex_entries(text)


def citation_keys(tex: str, *, root: Path | None = None) -> set[str]:
    return submission_metadata.citation_keys(tex, root=root)


def field(entry: str, name: str) -> str:
    match = re.search(rf"\b{name}\s*=\s*\{{([^}}]+)\}}", entry, flags=re.I)
    assert match, f"missing {name} in {entry.splitlines()[0]}"
    return match.group(1).strip()


def optional_field(entry: str, name: str) -> str:
    match = re.search(rf"\b{name}\s*=\s*\{{([^}}]+)\}}", entry, flags=re.I)
    return match.group(1).strip() if match else ""


def test_bibliography_is_recent_closed_and_has_exact_ivc_subset() -> None:
    """Catches stale entries, broken citations, padding, or an IVC DOI swap."""
    entries = bibtex_entries((PAPER / "references.bib").read_text(encoding="utf-8"))
    cited = citation_keys(
        (PAPER / "main.tex").read_text(encoding="utf-8"), root=PAPER
    )

    assert entries
    assert cited == set(entries), f"missing={sorted(cited - set(entries))}, orphaned={sorted(set(entries) - cited)}"
    assert all(2023 <= int(field(entry, "year")) <= 2026 for entry in entries.values())

    ivc_entries = [
        entry
        for entry in entries.values()
        if optional_field(entry, "journal").casefold() == "image and vision computing"
    ]
    assert len(ivc_entries) == 5
    assert {field(entry, "doi").casefold() for entry in ivc_entries} == IVC_DOIS


def test_citation_keys_resolves_owned_input_files(tmp_path: Path) -> None:
    """Catches a false orphan when an active citation lives in an owned TeX input."""
    (tmp_path / "included.tex").write_text(
        r"Included evidence \citep{included2026}." + "\n", encoding="utf-8"
    )
    tex = r"Main evidence \citep{main2026}.\input{included.tex}"

    assert citation_keys(tex, root=tmp_path) == {"main2026", "included2026"}


def test_citation_cff_uses_existing_manuscript_authors_pending_confirmation() -> None:
    """Catches an author placeholder or altered affiliation in the CFF record."""
    cff = yaml.safe_load((PAPER / "CITATION.cff").read_text(encoding="utf-8"))

    assert cff["cff-version"] == "1.2.0"
    title, authors = submission_metadata.manuscript_metadata(
        (PAPER / "main.tex").read_text(encoding="utf-8")
    )
    assert cff["title"] == title
    assert "author confirmation required" in cff["message"].casefold()
    assert cff["authors"] == authors


def test_active_manuscript_uses_the_requested_ivc_cas_double_column_format() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")

    assert r"\documentclass[a4paper,fleqn]{cas-dc}" in tex
    assert r"\shortauthors{Nguyen et al.}" in tex
    assert (
        r"\title[mode=title]{A Paired, Floor-Aware Evaluation of INT8 and FP8 "
        r"Object Detectors under Image Corruptions}"
    ) in tex
    assert r"\bibliographystyle{cas-model2-names}" in tex


def test_highlights_report_quantitative_confirmatory_evidence() -> None:
    lines = [
        line.removeprefix("- ").strip()
        for line in (PAPER / "highlights.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 5
    assert all(len(line) <= 85 for line in lines)
    joined = " ".join(lines)
    for value in ("144", "-0.02", "1.40", "-0.55"):
        assert value in joined
    assert "architecture-dependent" not in joined


def test_main_manuscript_states_the_evidence_hierarchy_explicitly() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")

    assert r"\label{tab:evidence-hierarchy}" in tex
    for phrase in (
        "Exploratory landscape",
        "Untouched-partition validation",
        "Seed sensitivity",
        "Corruption-realization sensitivity",
        "Fixed-universe sensitivity",
        "Architecture stress cases",
    ):
        assert phrase in tex
    assert "strongest holdout evidence" in tex


def test_final_main_source_does_not_retain_the_inactive_b500_protocol() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")

    assert "500 replicates" not in tex
    assert "500-replicate" not in tex
    assert "$B=500$" not in tex


def test_conclusion_keeps_format_and_architecture_claims_conditional() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")

    assert "Within the exploratory grid, FP8 retained 1.40 more matched-clean" in tex
    assert "recorded recipe--architecture stress cases" in tex
    assert "one checkpoint per dataset--family block" in tex


def test_main_methods_define_all_confirmatory_sensitivity_protocols() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8") + (PAPER / "generated" / "confirmatory_methods.tex").read_text(encoding="utf-8")
    normalized = " ".join(tex.split())

    for heading in (
        r"\subsection{Untouched-holdout validation protocol}",
        r"\subsection{Corruption-realization sensitivity design}",
        r"\subsection{Fixed-universe TT100K bootstrap sensitivity}",
    ):
        assert heading in tex
    for detail in (
        "seed 20260818",
        "5,823 final images",
            "1,197-image final set",
        "202608181, 202608182, and 202608183",
        "exponential mean-one image weights",
        "same four treatment arms",
    ):
        assert detail in normalized


def test_results_do_not_duplicate_the_multiseed_table_and_place_weighting_before_discussion() -> None:
    results = (PAPER / "direct_results_template.tex").read_text(encoding="utf-8")
    discussion = (PAPER / "direct_discussion_template.tex").read_text(encoding="utf-8")

    assert "Training--calibration seed sensitivity" not in results
    assert "20260807 & 36" not in results
    assert r"\input{generated/direct_sensitivity_narrative.tex}" in results
    assert r"\input{generated/direct_sensitivity_narrative.tex}" not in discussion


def test_front_matter_and_positioning_use_current_bounded_terminology() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    figure_source = (PAPER / "make_result_figures.py").read_text(encoding="utf-8")

    assert "Paired evaluation" in tex
    assert "Detection transformers" not in tex
    assert "Both recent real-world corruption benchmarks" not in tex
    assert "difference-in-differences contrast is therefore needed" not in tex
    assert "clean admissibility" not in figure_source
    assert "checking matched-clean fidelity" in figure_source


def test_cross_family_supplement_references_are_not_hard_coded() -> None:
    cross_family = (PAPER / "cross_family_results.tex").read_text(encoding="utf-8")
    assert "Supplementary Table~S2" not in cross_family
    assert "Supplementary Table~S3" not in cross_family
    assert "Supplementary File~S1" in cross_family


def test_primary_bootstrap_states_duplicate_handling_and_fixed_universe_scope() -> None:
    methods = (PAPER / "direct_methods_tail.tex").read_text(encoding="utf-8")
    assert "unique replicate-local identities" in methods
    assert "full-grid fixed-universe" in methods


def test_front_matter_has_at_most_seven_keywords_and_discloses_both_assistants() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    keywords = tex.split(r"\begin{keywords}", 1)[1].split(r"\end{keywords}", 1)[0]
    assert len([item for item in keywords.split(r"\sep") if item.strip()]) <= 7
    declaration = submission_metadata.extract_section(
        tex, "Declaration of generative AI and AI-assisted technologies"
    )
    assert "OpenAI Codex" in declaration
    assert "Anthropic Claude Code" in declaration
    assert r"\orcidlink" not in tex
    assert tex.count("[orcid=") == 5


def test_submission_audit_allows_only_the_empty_cas_titlebox_overfull() -> None:
    tex = r"""\documentclass[a4paper,fleqn]{cas-dc}
\begin{document}
\maketitle
\end{document}
"""
    benign_log = "Overfull \\hbox (123.62721pt too wide) detected at line 3\n[]\n []\n"
    real_log = "Overfull \\hbox (5.0pt too wide) detected at line 2\n[]\n"

    assert submission_audit.latex_log_issues(benign_log, tex) == []
    assert submission_audit.latex_log_issues(real_log, tex) == ["Overfull"]


def test_all_direct_results_floats_render_before_discussion() -> None:
    pdf = PROJECT_ROOT / "paper" / "main.pdf"
    info = subprocess.run(
        ["pdfinfo", str(pdf)], check=True, capture_output=True, text=True
    ).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE).group(1))
    page_texts = [
        subprocess.run(
            ["pdftotext", "-f", str(page), "-l", str(page), "-layout", str(pdf), "-"],
            check=True,
            capture_output=True,
            text=True,
        ).stdout
        for page in range(1, pages + 1)
    ]

    assert submission_audit.results_float_order_issues(page_texts) == []
    rendered = subprocess.run(
        ["pdftotext", str(pdf), "-"], check=True, capture_output=True, text=True
    ).stdout
    assert "--" not in rendered


def test_abstract_word_count_resolves_an_owned_input_file(tmp_path: Path) -> None:
    """Prevents a conditional evidence branch from evading the abstract limit."""
    (tmp_path / "direct_abstract.tex").write_text(
        "One two three four.", encoding="utf-8"
    )
    tex = r"\begin{abstract}\input{direct_abstract.tex}\end{abstract}"

    assert submission_audit.abstract_word_count(tex, root=tmp_path) == 4


def test_abstract_word_count_resolves_the_active_direct_generated_input(tmp_path: Path) -> None:
    """The direct branch must not evade the abstract limit via generated/ input."""
    generated = tmp_path / "generated"
    generated.mkdir()
    (generated / "direct_evidence_audit.json").write_text("{}\n", encoding="utf-8")
    (generated / "direct_abstract.tex").write_text(
        "One two three four five.", encoding="utf-8"
    )
    tex = r"\begin{abstract}\activeabstract\end{abstract}"

    assert submission_audit.abstract_word_count(tex, root=tmp_path) == 5


def test_literal_cas_abstract_is_token_identical_to_generated_evidence() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    generated = (PAPER / "generated" / "confirmatory_abstract.tex").read_text(
        encoding="utf-8"
    )

    submission_audit.validate_literal_direct_abstract(tex, generated)
    with pytest.raises(SystemExit, match="drift"):
        submission_audit.validate_literal_direct_abstract(
            tex.replace("untouched-holdout", "selected-partition", 1), generated
        )


def test_manuscript_metadata_extracts_cas_author_address_records() -> None:
    tex = r"""
\title[mode=title]{CAS metadata fixture}
\author[first]{Ada Lovelace\orcidlink{0000-0000-0000-0001}}
\ead{ada@example.test}
\author[second]{Grace Hopper\corref{corresponding}}
\address[first]{Analytical Engine Laboratory, London, United Kingdom}
\address[second]{Compiler Laboratory, Arlington, United States}
"""

    title, authors = submission_metadata.manuscript_metadata(tex)

    assert title == "CAS metadata fixture"
    assert authors == [
        {
            "family-names": "Lovelace",
            "given-names": "Ada",
            "affiliation": "Analytical Engine Laboratory, London, United Kingdom",
        },
        {
            "family-names": "Hopper",
            "given-names": "Grace",
            "affiliation": "Compiler Laboratory, Arlington, United States",
        },
    ]


def test_zenodo_metadata_uses_schema_and_author_confirmation_placeholder() -> None:
    """Catches a fabricated archive DOI, person, or malformed deposit shell."""
    zenodo = json.loads((PAPER / "zenodo.json").read_text(encoding="utf-8"))
    metadata = zenodo["metadata"]

    assert zenodo["$schema"] == "https://zenodo.org/schemas/deposits/metadata.json"
    title, _ = submission_metadata.manuscript_metadata(
        (PAPER / "main.tex").read_text(encoding="utf-8")
    )
    assert metadata["title"] == title
    assert metadata["description"]
    assert metadata["creators"] == [{"name": "AUTHOR CONFIRMATION REQUIRED"}]
    assert "doi pending" in metadata["description"].casefold()


def test_funding_is_an_author_action_and_data_availability_says_doi_pending() -> None:
    """Catches unsupported funding declarations and a fictitious public archive."""
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    reviewer_notes = (PAPER / "README.md").read_text(encoding="utf-8")

    funding = re.search(r"\\section\*\{Funding\}(.*?)(?=\\section\*\{|\\bibliography)", tex, flags=re.DOTALL)
    assert funding
    assert "author action required" in funding.group(1).casefold()
    assert "no specific funding" not in funding.group(1).casefold()
    assert "doi pending" in submission_metadata.extract_section(
        tex, "Data and code availability"
    ).casefold()
    assert "funding" in reviewer_notes.casefold()
    assert "author action required" in reviewer_notes.casefold()


def test_commented_citation_is_not_active_and_fails_audit_closure() -> None:
    """Catches a citation hidden in a TeX comment instead of active prose."""
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    mutated = tex.replace(r"\citep{kolf2023syper}", r"% \citep{kolf2023syper}")

    assert "kolf2023syper" not in citation_keys(mutated)
    with pytest.raises(SystemExit, match="closure"):
        submission_audit.validate_submission_metadata(mutated, bib)


def test_commented_pending_doi_is_not_an_availability_statement() -> None:
    """Catches an active availability section missing its pending-DOI wording."""
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    mutated = tex.replace("Archive DOI pending:", "% Archive DOI pending:")

    with pytest.raises(SystemExit, match="Data availability"):
        submission_audit.validate_submission_metadata(mutated, bib)


def test_duplicate_bibtex_key_cannot_hide_an_out_of_window_entry() -> None:
    """Catches a later duplicate key silently overwriting a 1900 record."""
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    duplicate = "@article{kolf2023syper,\n  year = {1900}\n}\n\n" + bib

    with pytest.raises(ValueError, match="duplicate"):
        bibtex_entries(duplicate)
    with pytest.raises(SystemExit, match="duplicate"):
        submission_audit.validate_submission_metadata(tex, duplicate)


def test_indented_duplicate_bibtex_key_cannot_hide_an_out_of_window_entry() -> None:
    """Catches a whitespace-prefixed later BibTeX entry outside the year window."""
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    duplicate = bib + "\n  @article{kolf2023syper,\n    year = {1900}\n  }\n"

    with pytest.raises(ValueError, match="duplicate"):
        submission_metadata.bibtex_entries(duplicate)
    with pytest.raises(SystemExit, match="duplicate"):
        submission_audit.validate_submission_metadata(tex, duplicate)


@pytest.mark.parametrize(
    "old,new",
    [
        (r"\author[mymainaddress]{Dinh Thuan Nguyen", r"\author[mymainaddress]{Changed Author"),
        (
            "Faculty of Electrical and Electronics Engineering, Ton Duc Thang University, Ho Chi Minh City, Vietnam",
            "Changed affiliation",
        ),
    ],
)
def test_manuscript_metadata_change_fails_when_cff_and_zenodo_are_unchanged(
    old: str, new: str
) -> None:
    """Catches drift from manuscript title, author order, or affiliation into metadata."""
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")

    with pytest.raises(SystemExit, match="manuscript metadata"):
        submission_audit.validate_submission_metadata(tex.replace(old, new, 1), bib)


def test_current_manuscript_title_change_fails_when_cff_and_zenodo_are_unchanged() -> None:
    """Derives the active title, so the drift guard survives a legitimate retitling."""
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    match = re.search(r"\\title(?:\[[^]]+\])?\{([^{}]+)\}", tex)
    assert match, "expected one active manuscript title"
    mutated = tex[: match.start(1)] + "Changed manuscript title" + tex[match.end(1) :]

    with pytest.raises(SystemExit, match="manuscript metadata"):
        submission_audit.validate_submission_metadata(mutated, bib)


def test_zenodo_title_must_match_the_manuscript_and_cff() -> None:
    """Catches a non-empty but mismatched Zenodo deposit title."""
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    zenodo = json.loads((PAPER / "zenodo.json").read_text(encoding="utf-8"))
    zenodo["metadata"]["title"] = "Different Zenodo title"

    with pytest.raises(SystemExit, match="Zenodo title"):
        submission_audit.validate_submission_metadata(
            tex, bib, zenodo_text=json.dumps(zenodo)
        )


def test_later_active_title_is_rejected_as_ambiguous() -> None:
    """Catches a second active title that would otherwise override manuscript metadata."""
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    mutated = tex.replace(
        r"\begin{document}",
        "\\title{Different active title}\n\\begin{document}",
        1,
    )

    with pytest.raises(SystemExit, match=r"multiple active \\title definitions"):
        submission_audit.validate_submission_metadata(mutated, bib)


def _write_flat_source_zip(
    archive: Path,
    *,
    tex: str | None = None,
    members: dict[str, bytes | str] | None = None,
) -> None:
    """Create a hand-checked flat LaTeX archive fixture."""
    source = {
        "main.tex": (
            tex
            or "\\documentclass{article}\n"
            "\\usepackage{graphicx}\n"
            "\\begin{document}\n"
            "\\input{body}\n"
            "% \\input{commented-out}\n"
            "\\includegraphics[width=1cm]{plot.png}\n"
            "\\end{document}\n"
        ),
        "references.bib": "@article{fixture, title={Fixture}}\n",
        "body.tex": "Fixture body.\n",
        "plot.png": _ONE_PIXEL_PNG,
    }
    source.update(members or {})
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, content in source.items():
            bundle.writestr(name, content)


def _latex_runner(returncode: int = 0, output: str = ""):
    """Return a deterministic stand-in for the external LaTeX executable."""
    def run(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        assert command == [
            "latexmk",
            "-pdf",
            "-recorder",
            "-interaction=nonstopmode",
            "-halt-on-error",
            "main.tex",
        ]
        assert (cwd / "main.tex").is_file()
        if returncode == 0:
            (cwd / "main.fls").write_text("INPUT main.tex\n", encoding="utf-8")
        return subprocess.CompletedProcess(command, returncode, stdout=output, stderr="")

    return run


def test_flat_source_zip_accepts_complete_flat_archive(tmp_path: Path) -> None:
    """Catches a verifier that rejects the minimum independently compilable source set."""
    archive = tmp_path / "source.zip"
    _write_flat_source_zip(archive)

    submission_package.verify_flat_source_archive(
        archive, latexmk_path="latexmk", runner=_latex_runner()
    )


@pytest.mark.skipif(shutil.which("latexmk") is None, reason="latexmk is unavailable")
def test_flat_source_zip_compiles_genuine_fixture_with_real_latexmk(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    """Catches a fixture that only constructs a command but cannot build after extraction."""
    tex_bin = "/data_nvme/texlive/2026/bin/x86_64-linux"
    monkeypatch.setenv("PATH", f"{tex_bin}{os.pathsep}{os.environ['PATH']}")
    archive = tmp_path / "genuine-source.zip"
    _write_flat_source_zip(archive)
    smoke = tmp_path / "smoke.tex"
    smoke.write_text("\\documentclass{article}\\begin{document}ok\\end{document}\n")
    available = subprocess.run(
        [str(shutil.which("latexmk")), "-pdf", "-halt-on-error", smoke.name],
        cwd=tmp_path,
        capture_output=True,
        text=True,
        check=False,
    )
    if available.returncode != 0:
        pytest.skip("latexmk is installed but its TeX toolchain is not configured")

    submission_package.verify_flat_source_archive(
        archive, latexmk_path=shutil.which("latexmk")
    )


def test_flat_source_zip_rejects_missing_input_target(tmp_path: Path) -> None:
    """Catches a copied main.tex whose extensionless input has no .tex member."""
    archive = tmp_path / "missing-input.zip"
    _write_flat_source_zip(archive, members={"body.tex": ""})
    with zipfile.ZipFile(archive, "a") as bundle:
        # Rebuild without body.tex because ZipFile cannot remove an entry.
        retained = {
            info.filename: bundle.read(info)
            for info in bundle.infolist()
            if info.filename != "body.tex"
        }
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, content in retained.items():
            bundle.writestr(name, content)

    with pytest.raises(ValueError, match=r"missing TeX input.*body\.tex"):
        submission_package.verify_flat_source_archive(archive, runner=_latex_runner())


def test_flat_source_zip_rejects_missing_graphic_target(tmp_path: Path) -> None:
    """Catches a copied main.tex whose extensionless graphic has no supported file."""
    archive = tmp_path / "missing-graphic.zip"
    _write_flat_source_zip(archive)
    with zipfile.ZipFile(archive, "a") as bundle:
        retained = {
            info.filename: bundle.read(info)
            for info in bundle.infolist()
            if info.filename != "plot.png"
        }
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, content in retained.items():
            bundle.writestr(name, content)

    with pytest.raises(ValueError, match=r"missing graphic.*plot"):
        submission_package.verify_flat_source_archive(archive, runner=_latex_runner())


@pytest.mark.parametrize(
    ("member", "contents", "error"),
    [
        ("generated/", b"", "directory"),
        ("generated/numbers.tex", "x", "flat"),
        ("/absolute.tex", "x", "absolute"),
        ("../escape.tex", "x", "traversal"),
    ],
)
def test_flat_source_zip_rejects_nonflat_or_unsafe_member(
    tmp_path: Path, member: str, contents: bytes | str, error: str
) -> None:
    """Catches archive names that could create a directory or escape extraction root."""
    archive = tmp_path / "unsafe-member.zip"
    _write_flat_source_zip(archive, members={member: contents})

    with pytest.raises(ValueError, match=error):
        submission_package.verify_flat_source_archive(archive, runner=_latex_runner())


def test_flat_source_zip_rejects_traversal_reference(tmp_path: Path) -> None:
    """Catches an input path in copied TeX that would escape the flat package."""
    archive = tmp_path / "traversal-reference.zip"
    _write_flat_source_zip(
        archive,
        tex="\\documentclass{article}\n\\input{../outside}\n\\begin{document}x\\end{document}\n",
    )

    with pytest.raises(ValueError, match="traversal"):
        submission_package.verify_flat_source_archive(archive, runner=_latex_runner())


def test_flat_source_zip_rejects_duplicate_member_name(tmp_path: Path) -> None:
    """Catches ZIP duplicate entries that would make copied source ambiguous."""
    archive = tmp_path / "duplicate.zip"
    _write_flat_source_zip(archive)
    with zipfile.ZipFile(archive, "a") as bundle:
        with pytest.warns(UserWarning, match="Duplicate name"):
            bundle.writestr("body.tex", "different body")

    with pytest.raises(ValueError, match="duplicate"):
        submission_package.verify_flat_source_archive(archive, runner=_latex_runner())


def test_flat_source_zip_rejects_symlink_like_member(tmp_path: Path) -> None:
    """Catches Unix-mode symlink metadata even when the member name is harmless."""
    archive = tmp_path / "symlink.zip"
    _write_flat_source_zip(archive)
    link = zipfile.ZipInfo("linked.tex")
    link.create_system = 3
    link.external_attr = 0o120777 << 16
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr(link, "body.tex")

    with pytest.raises(ValueError, match="symlink"):
        submission_package.verify_flat_source_archive(archive, runner=_latex_runner())


def test_flat_source_zip_rejects_latex_failure_output(tmp_path: Path) -> None:
    """Catches an archive that passes membership checks but fails its extracted build."""
    archive = tmp_path / "latex-failure.zip"
    _write_flat_source_zip(archive)

    with pytest.raises(ValueError, match=r"LaTeX compilation failed.*Undefined control sequence"):
        submission_package.verify_flat_source_archive(
            archive,
            latexmk_path="latexmk",
            runner=_latex_runner(returncode=1, output="Undefined control sequence"),
        )


@pytest.mark.parametrize("member", ["./main.tex", ".\\main.tex", "folder\\body.tex"])
def test_flat_source_zip_rejects_raw_member_path_aliases(
    tmp_path: Path, member: str
) -> None:
    """Catches raw ZIP spelling that maps to a flat destination after extraction."""
    archive = tmp_path / "raw-member-alias.zip"
    _write_flat_source_zip(archive, members={member: "unsafe replacement"})

    with pytest.raises(ValueError, match="flat"):
        submission_package.verify_flat_source_archive(archive, runner=_latex_runner())


@pytest.mark.parametrize("kind", ["unix-directory", "unix-fifo", "dos-directory"])
def test_flat_source_zip_rejects_nonregular_member_metadata(
    tmp_path: Path, kind: str
) -> None:
    """Catches directory and special-file metadata hidden behind a flat file name."""
    archive = tmp_path / f"{kind}.zip"
    _write_flat_source_zip(archive)
    info = zipfile.ZipInfo("metadata-entry")
    if kind == "unix-directory":
        info.create_system = 3
        info.external_attr = 0o040755 << 16
    elif kind == "unix-fifo":
        info.create_system = 3
        info.external_attr = 0o010644 << 16
    else:
        info.create_system = 0
        info.external_attr = 0x10
    with zipfile.ZipFile(archive, "a") as bundle:
        bundle.writestr(info, "not a regular file")

    with pytest.raises(ValueError, match="non-regular"):
        submission_package.verify_flat_source_archive(archive, runner=_latex_runner())


def test_flat_source_zip_rejects_recursive_absolute_input(tmp_path: Path) -> None:
    """Catches a safe main input whose reachable copied file escapes the package."""
    archive = tmp_path / "recursive-absolute-input.zip"
    _write_flat_source_zip(archive, members={"body.tex": "\\input{/outside.tex}\n"})

    with pytest.raises(ValueError, match="absolute TeX input"):
        submission_package.verify_flat_source_archive(archive, runner=_latex_runner())


@pytest.mark.parametrize(
    ("tex", "error"),
    [
        ("\\input /outside.tex\n", "absolute TeX input"),
        ("\\include /outside.tex\n", "unbraced TeX include"),
        ("\\includegraphics*{missing}\n", "missing graphic"),
    ],
)
def test_flat_source_zip_rejects_unbraced_or_starred_external_dependency(
    tmp_path: Path, tex: str, error: str
) -> None:
    """Catches legal TeX command forms that must not bypass member validation."""
    archive = tmp_path / "alternate-command-form.zip"
    _write_flat_source_zip(
        archive,
        tex="\\documentclass{article}\n\\begin{document}\n"
        + tex
        + "\\end{document}\n",
    )

    with pytest.raises(ValueError, match=error):
        submission_package.verify_flat_source_archive(archive, runner=_latex_runner())


def test_flat_source_zip_handles_recursive_input_cycle(tmp_path: Path) -> None:
    """Catches a recursive validator that loops on a valid archive-owned input cycle."""
    archive = tmp_path / "input-cycle.zip"
    _write_flat_source_zip(archive, members={"body.tex": "\\input{main}\n"})

    submission_package.verify_flat_source_archive(
        archive, latexmk_path="latexmk", runner=_latex_runner()
    )


def test_flat_source_zip_rejects_unbraced_include(tmp_path: Path) -> None:
    """Catches a parser that gives LaTeX include primitive-input semantics."""
    archive = tmp_path / "unbraced-include.zip"
    _write_flat_source_zip(
        archive,
        tex="\\documentclass{article}\n\\begin{document}\n"
        "\\include body\n\\end{document}\n",
    )

    with pytest.raises(ValueError, match="unbraced TeX include"):
        submission_package.verify_flat_source_archive(
            archive, latexmk_path="latexmk", runner=_latex_runner()
        )


def test_flat_source_zip_rejects_recursive_missing_graphic(tmp_path: Path) -> None:
    """Catches a reachable copied TeX file whose graphic is absent from the ZIP."""
    archive = tmp_path / "recursive-missing-graphic.zip"
    _write_flat_source_zip(
        archive, members={"body.tex": "\\includegraphics*{missing}\n"}
    )

    with pytest.raises(ValueError, match="missing graphic"):
        submission_package.verify_flat_source_archive(archive, runner=_latex_runner())


def test_flat_source_zip_rejects_non_utf8_reachable_input(tmp_path: Path) -> None:
    """Catches a copied input file whose bytes cannot be safely parsed for dependencies."""
    archive = tmp_path / "non-utf8-input.zip"
    _write_flat_source_zip(archive, members={"body.tex": b"\xff\xfe"})

    with pytest.raises(ValueError, match=r"body\.tex.*UTF-8"):
        submission_package.verify_flat_source_archive(archive, runner=_latex_runner())


@pytest.mark.parametrize("required", ["main.tex", "references.bib"])
def test_flat_source_zip_requires_main_and_references(tmp_path: Path, required: str) -> None:
    """Catches an archive that omits either mandatory root source member."""
    archive = tmp_path / f"missing-{required}.zip"
    _write_flat_source_zip(archive)
    with zipfile.ZipFile(archive) as bundle:
        retained = {
            info.filename: bundle.read(info)
            for info in bundle.infolist()
            if info.filename != required
        }
    with zipfile.ZipFile(archive, "w") as bundle:
        for name, content in retained.items():
            bundle.writestr(name, content)

    with pytest.raises(ValueError, match="required members"):
        submission_package.verify_flat_source_archive(archive, runner=_latex_runner())


def test_flat_source_zip_wraps_latex_launch_oserror(tmp_path: Path) -> None:
    """Catches compiler launch failures that violate the verifier's ValueError contract."""
    archive = tmp_path / "compiler-launch.zip"
    _write_flat_source_zip(archive)

    with pytest.raises(ValueError, match="could not start LaTeX compilation"):
        submission_package.verify_flat_source_archive(
            archive, latexmk_path=tmp_path / "does-not-exist"
        )


def test_flat_source_zip_rejects_external_recorder_input(tmp_path: Path) -> None:
    """Catches a successful compiler run that reads an input outside extraction root."""
    archive = tmp_path / "external-recorder-input.zip"
    _write_flat_source_zip(archive)

    def recorder_runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        (cwd / "main.fls").write_text("INPUT /outside/manuscript-input.tex\n")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(ValueError, match="recorder loaded input outside extraction root"):
        submission_package.verify_flat_source_archive(
            archive, latexmk_path="latexmk", runner=recorder_runner
        )


def test_flat_source_zip_rejects_missing_recorder_from_successful_runner(
    tmp_path: Path,
) -> None:
    """Catches a successful injected compiler runner that omits recorder evidence."""
    archive = tmp_path / "missing-recorder.zip"
    _write_flat_source_zip(archive)

    def no_recorder_runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(ValueError, match="did not produce recorder output"):
        submission_package.verify_flat_source_archive(
            archive, latexmk_path="latexmk", runner=no_recorder_runner
        )


def test_flat_source_zip_rejects_empty_recorder_from_successful_runner(
    tmp_path: Path,
) -> None:
    """Catches recorder evidence that contains no root source input record."""
    archive = tmp_path / "empty-recorder.zip"
    _write_flat_source_zip(archive)

    def empty_recorder_runner(command: list[str], cwd: Path) -> subprocess.CompletedProcess[str]:
        (cwd / "main.fls").write_text("", encoding="utf-8")
        return subprocess.CompletedProcess(command, 0, stdout="", stderr="")

    with pytest.raises(ValueError, match="must record root main.tex input"):
        submission_package.verify_flat_source_archive(
            archive, latexmk_path="latexmk", runner=empty_recorder_runner
        )
