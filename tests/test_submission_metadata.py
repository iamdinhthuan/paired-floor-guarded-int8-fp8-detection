"""Submission-facing citation and archive metadata contracts."""

from __future__ import annotations

import base64
import json
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
import submission_metadata  # noqa: E402
import submission_package  # noqa: E402
import validate_cviu_paper_package as cviu_validator  # noqa: E402

CVIU_TITLE = (
    "A Paired, Floor-Guarded Evaluation Protocol for INT8 and FP8 Object "
    "Detectors under Image Corruptions"
)
CVIU_DOIS = {
    "10.1016/j.cviu.2020.102907",
    "10.1016/j.cviu.2022.103445",
    "10.1016/j.cviu.2024.104252",
    "10.1016/j.cviu.2026.104735",
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


def test_bibliography_is_closed_and_has_foundational_and_cviu_sources() -> None:
    """Keep citation closure while allowing the foundational literature."""
    entries = bibtex_entries((PAPER / "references.bib").read_text(encoding="utf-8"))
    cited = citation_keys(
        (PAPER / "main.tex").read_text(encoding="utf-8"), root=PAPER
    )

    assert entries
    assert cited == set(entries), f"missing={sorted(cited - set(entries))}, orphaned={sorted(set(entries) - cited)}"
    years = {int(field(entry, "year")) for entry in entries.values()}
    assert min(years) <= 1981  # bootstrap foundations are intentionally retained
    assert max(years) == 2026

    cviu_entries = [
        entry
        for entry in entries.values()
        if optional_field(entry, "journal").casefold()
        == "computer vision and image understanding"
    ]
    assert len(cviu_entries) == 4
    assert {field(entry, "doi").casefold() for entry in cviu_entries} == CVIU_DOIS


def test_citation_keys_resolves_owned_input_files(tmp_path: Path) -> None:
    """Catches a false orphan when an active citation lives in an owned TeX input."""
    (tmp_path / "included.tex").write_text(
        r"Included evidence \citep{included2026}." + "\n", encoding="utf-8"
    )
    tex = r"Main evidence \citep{main2026}.\input{included.tex}"

    assert citation_keys(tex, root=tmp_path) == {"main2026", "included2026"}


def test_citation_cff_describes_the_frozen_cviu_reproducibility_package() -> None:
    """Keep the release record aligned with the paper and six human authors."""
    cff = yaml.safe_load((PAPER / "CITATION.cff").read_text(encoding="utf-8"))

    assert cff["cff-version"] == "1.2.0"
    assert cff["title"] == f"{CVIU_TITLE}: Reproducibility Package"
    assert cff["version"] == "2.1.0"
    assert cff["doi"] == "10.5281/zenodo.22275640"
    assert cff["license"] == "MIT"
    assert cviu_validator.cff_names(cff) == cviu_validator.AUTHORS
    serialized = json.dumps(cff)
    assert not cviu_validator.AI_NAME.search(serialized)


def test_active_manuscript_uses_the_requested_cviu_cas_double_column_format() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")

    assert r"\documentclass[a4paper,fleqn]{cas-dc}" in tex
    assert r"\shortauthors{Nguyen et al.}" in tex
    assert rf"\title[mode=title]{{{CVIU_TITLE}}}" in tex
    assert r"\bibliographystyle{elsarticle-num}" in tex


def test_highlights_meet_elsevier_contract_and_match_the_scientific_message() -> None:
    lines = [
        line.removeprefix("- ").strip()
        for line in (PAPER / "highlights.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 5
    assert all(len(line) <= 85 for line in lines)
    joined = " ".join(lines).casefold()
    for phrase in ("paired evaluation", "shared-image", "clean adjustment", "absolute ap"):
        assert phrase in joined
    assert "format superiority" not in joined


def test_main_manuscript_states_the_evidence_hierarchy_explicitly() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")

    assert r"\label{tab:evidence-hierarchy}" in tex
    for phrase in (
        "Exploratory landscape",
        "Final holdout",
        "Metric scale",
        "Training and calibration seeds",
        "Corruption realizations",
        "Fixed class set",
        "Transfer across architectures",
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

    assert "Across the 144-cell exploratory grid, FP8 outperformed INT8 by 1.40" in tex
    assert "The results do not establish a universal robustness ranking" in tex
    assert "one checkpoint per dataset--family block" in tex


def test_main_methods_define_all_holdout_and_sensitivity_protocols() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    normalized = " ".join(tex.split())

    for heading in (
        r"\subsection{Training, checkpoint selection, and holdout rerun}",
        r"\subsection{AP evaluation and paired uncertainty}",
        r"\subsection{Sensitivity designs}",
    ):
        assert heading in tex
    for detail in (
        "seed 20260818",
        "5,823 images in val2012",
        "1,197 images) for final evaluation",
        "202608181, 202608182, and 202608183",
        "independent exponential weights with unit rate",
        "reuses the same vector across all four treatment arms",
    ):
        assert detail in normalized


def test_results_place_quantitative_sensitivity_before_discussion() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    results = tex.split(r"\section{Results}", 1)[1].split(r"\section{Discussion}", 1)[0]
    discussion = tex.split(r"\section{Discussion}", 1)[1]

    for value in ("-0.23", "-1.11", "0.5318"):
        assert value in results
    assert results.count(r"\input{generated/conditionality_scope_summary.tex}") == 1
    assert r"\input{generated/conditionality_scope_summary.tex}" not in discussion


def test_front_matter_and_positioning_use_current_bounded_terminology() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")

    assert "paired evaluation protocol" in tex
    assert "Detection transformers" not in tex
    assert "Both recent real-world corruption benchmarks" not in tex
    assert "difference-in-differences contrast is therefore needed" not in tex
    assert "clean admissibility" not in tex
    assert "not interpreted as a causal effect" in tex


def test_cross_family_supplement_references_are_not_hard_coded() -> None:
    main = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert "Supplementary Table~S2" not in main
    assert "Supplementary Table~S3" not in main
    assert "Supplementary sections" in main or "Supplementary Information" in main


def test_primary_bootstrap_states_duplicate_handling_and_fixed_universe_scope() -> None:
    methods = (PAPER / "main.tex").read_text(encoding="utf-8")
    assert "unique identity within that bootstrap sample" in methods
    assert "does not validate the full 36-cell TT100K height macro" in methods


def test_front_matter_has_at_most_seven_keywords_and_discloses_both_assistants() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    keywords = tex.split(r"\begin{keywords}", 1)[1].split(r"\end{keywords}", 1)[0]
    assert len([item for item in keywords.split(r"\sep") if item.strip()]) <= 7
    declaration = submission_metadata.extract_section(
        tex,
        "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process",
    )
    assert "OpenAI Codex" in declaration
    assert "Anthropic Claude Code" in declaration
    assert "not listed as authors or contributors" in declaration
    assert "AI tools are not creators or contributors" in (PAPER / ".zenodo.json").read_text(
        encoding="utf-8"
    )
    assert tex.count(r"\orcidlink{") == 5


def test_cviu_package_validator_accepts_the_canonical_paper() -> None:
    errors, notes = cviu_validator.validate(PAPER)
    assert errors == []
    assert any(note.startswith("abstract:") for note in notes)
    assert any(note.startswith("metadata:") for note in notes)


def test_rendered_main_has_results_before_discussion_and_no_layout_warnings() -> None:
    pdf = PROJECT_ROOT / "paper" / "main.pdf"
    info = subprocess.run(
        ["pdfinfo", str(pdf)], check=True, capture_output=True, text=True
    ).stdout
    pages = int(re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE).group(1))
    assert 15 <= pages <= 25
    rendered = subprocess.run(
        ["pdftotext", str(pdf), "-"], check=True, capture_output=True, text=True
    ).stdout
    assert rendered.index("4. Results") < rendered.index("5. Discussion")
    assert "References" in rendered
    for log_name in ("main.log", "supplement.log"):
        log = (PAPER / log_name).read_text(encoding="utf-8")
        for pattern in cviu_validator.LOG_ERRORS.values():
            assert pattern.search(log) is None


def test_literal_cas_abstract_is_self_contained_and_within_cviu_limit() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    assert abstract
    assert r"\input{" not in abstract.group(1)
    words = re.findall(
        r"[A-Za-z0-9]+(?:[\u2010-\u2015'-][A-Za-z0-9]+)*",
        cviu_validator.plain_tex(abstract.group(1)),
    )
    assert len(words) == 228


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


def test_zenodo_metadata_is_release_ready_and_has_only_human_creators() -> None:
    """Catches creator drift, an old release, or AI attribution as authorship."""
    zenodo = json.loads((PAPER / ".zenodo.json").read_text(encoding="utf-8"))
    metadata, names = cviu_validator.zenodo_names(zenodo)

    assert metadata["title"] == f"{CVIU_TITLE}: Reproducibility Package"
    assert metadata["version"] == "2.1.0"
    assert metadata["license"] == "MIT"
    assert names == cviu_validator.AUTHORS
    assert metadata["related_identifiers"] == [
        {
            "identifier": "https://github.com/iamdinhthuan/paired-floor-guarded-int8-fp8-detection/tree/v2.1.0",
            "relation": "isSupplementTo",
            "scheme": "url",
            "resource_type": "software",
        }
    ]
    assert not cviu_validator.AI_NAME.search(" ".join(names))


def test_no_funding_and_public_archive_statements_are_final() -> None:
    """Catches reintroduced placeholders or unsupported funding."""
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")

    funding = re.search(r"\\section\*\{Funding\}(.*?)(?=\\section\*\{|\\bibliography)", tex, flags=re.DOTALL)
    assert funding
    assert "did not receive any specific grant" in funding.group(1).casefold()
    availability = submission_metadata.extract_section(tex, "Data and code availability")
    assert "10.5281/zenodo.22275640" in availability
    assert "10.5281/zenodo.22031663" in availability
    assert "github.com/iamdinhthuan/paired-floor-guarded-int8-fp8-detection" in availability
    assert "author action required" not in tex.casefold()
    assert "doi pending" not in tex.casefold()


def test_commented_citation_is_not_active_and_breaks_bibliography_closure() -> None:
    """Catches a citation hidden in a TeX comment instead of active prose."""
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    mutated = tex.replace(r"\citep{wen2020uadetrac}", r"% \citep{wen2020uadetrac}")

    entries = bibtex_entries(bib)
    assert "wen2020uadetrac" not in citation_keys(mutated)
    assert citation_keys(mutated) != set(entries)


def test_commented_version_doi_is_not_an_availability_statement() -> None:
    """Catches an active availability section missing the exact release DOI."""
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    mutated = tex.replace(
        "The CVIU-aligned release v2.1.0 is archived at",
        "% The CVIU-aligned release v2.1.0 is archived at",
    )

    availability = submission_metadata.extract_section(mutated, "Data and code availability")
    assert "10.5281/zenodo.22275640" not in availability


def test_duplicate_bibtex_key_cannot_hide_an_entry() -> None:
    """Catches a duplicate key silently overwriting a valid record."""
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    duplicate = "@article{wen2020uadetrac,\n  year = {1900}\n}\n\n" + bib

    with pytest.raises(ValueError, match="duplicate"):
        bibtex_entries(duplicate)


def test_indented_duplicate_bibtex_key_is_rejected() -> None:
    """Catches a whitespace-prefixed duplicate BibTeX entry."""
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    duplicate = bib + "\n  @article{wen2020uadetrac,\n    year = {1900}\n  }\n"

    with pytest.raises(ValueError, match="duplicate"):
        submission_metadata.bibtex_entries(duplicate)


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
    _, original = submission_metadata.manuscript_metadata(tex)
    _, changed = submission_metadata.manuscript_metadata(tex.replace(old, new, 1))
    assert changed != original


def test_current_manuscript_title_change_fails_when_cff_and_zenodo_are_unchanged() -> None:
    """Derives the active title, so the drift guard survives a legitimate retitling."""
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    match = re.search(r"\\title(?:\[[^]]+\])?\{([^{}]+)\}", tex)
    assert match, "expected one active manuscript title"
    mutated = tex[: match.start(1)] + "Changed manuscript title" + tex[match.end(1) :]

    title, _ = submission_metadata.manuscript_metadata(mutated)
    cff = yaml.safe_load((PAPER / "CITATION.cff").read_text(encoding="utf-8"))
    assert title not in cff["title"]


def test_zenodo_title_must_match_the_manuscript_and_cff() -> None:
    """Catches drift among manuscript, CFF, and Zenodo release titles."""
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    zenodo = json.loads((PAPER / ".zenodo.json").read_text(encoding="utf-8"))
    cff = yaml.safe_load((PAPER / "CITATION.cff").read_text(encoding="utf-8"))
    manuscript_title, _ = submission_metadata.manuscript_metadata(tex)
    expected = f"{manuscript_title}: Reproducibility Package"
    assert zenodo["title"] == cff["title"] == expected


def test_later_active_title_is_rejected_as_ambiguous() -> None:
    """Catches a second active title that would otherwise override manuscript metadata."""
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    mutated = tex.replace(
        r"\begin{document}",
        "\\title{Different active title}\n\\begin{document}",
        1,
    )

    with pytest.raises(ValueError, match=r"multiple active \\title definitions"):
        submission_metadata.manuscript_metadata(mutated)


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
    tmp_path: Path,
) -> None:
    """Catches a fixture that only constructs a command but cannot build after extraction."""
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
