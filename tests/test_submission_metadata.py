"""Submission-facing citation and archive metadata contracts (Neural Networks)."""

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

NN_TITLE = (
    "Quantization fragility under image corruption is recipe-dependent: "
    "paired evidence from INT8 and FP8 object detectors"
)
NN_MAIN = PAPER / "main_nn.tex"
VERSION = "3.0.3"
VERSION_DOI = "10.5281/zenodo.23082850"
CONCEPT_DOI = "10.5281/zenodo.22031663"

AUTHORS = [
    "Nguyen, Dinh Thuan",
    "Nguyen, Lam Phuong",
    "Nguyen, Vinh Huy",
    "Quang, Sy Vu",
    "Elara, Mohan Rajesh",
    "Le, Anh Vu",
]
AI_NAME = re.compile(r"devin|claude|codex|chatgpt|openai|anthropic|cognition", re.I)

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


def cff_names(cff: dict) -> list[str]:
    return [
        f"{a['family-names']}, {a['given-names']}" for a in cff["authors"]
    ]


def test_bibliography_is_closed_and_has_foundational_and_nn_sources() -> None:
    """Every citation in the NN manuscript resolves, with no orphan entries."""
    entries = bibtex_entries((PAPER / "references.bib").read_text(encoding="utf-8"))
    cited = citation_keys(NN_MAIN.read_text(encoding="utf-8"), root=PAPER) | citation_keys(
        (PAPER / "supplement.tex").read_text(encoding="utf-8"), root=PAPER
    )

    assert entries
    missing = cited - set(entries)
    assert not missing, f"cited but missing from references.bib: {sorted(missing)}"
    assert cited == set(entries), (
        f"uncited entries: {sorted(set(entries) - cited)}"
    )
    years = {int(field(entry, "year")) for entry in entries.values()}
    assert min(years) <= 1981  # bootstrap foundations are intentionally retained
    assert max(years) == 2026

    nn_entries = [
        entry
        for entry in entries.values()
        if optional_field(entry, "journal").casefold() == "neural networks"
    ]
    assert nn_entries


def test_citation_keys_resolves_owned_input_files(tmp_path: Path) -> None:
    """Catches a false orphan when an active citation lives in an owned TeX input."""
    (tmp_path / "included.tex").write_text(
        r"Included evidence \citep{included2026}." + "\n", encoding="utf-8"
    )
    tex = r"Main evidence \citep{main2026}.\input{included.tex}"

    assert citation_keys(tex, root=tmp_path) == {"main2026", "included2026"}


def test_citation_cff_describes_the_nn_reproducibility_package() -> None:
    """Keep the release record aligned with the NN paper and six human authors."""
    cff = yaml.safe_load((PAPER / "CITATION.cff").read_text(encoding="utf-8"))

    assert cff["cff-version"] == "1.2.0"
    assert cff["title"] == f"{NN_TITLE}: Reproducibility Package"
    assert cff["version"] == VERSION
    assert cff["doi"] == VERSION_DOI
    assert {"type": "doi", "value": CONCEPT_DOI,
            "description": "All-versions concept DOI"} in cff["identifiers"]
    assert cff["license"] == "MIT"
    assert cff_names(cff) == AUTHORS
    serialized = json.dumps(cff)
    assert not AI_NAME.search(serialized)


def test_root_metadata_matches_paper_metadata() -> None:
    """Root and paper release records must describe the same version."""
    root_cff = yaml.safe_load((PROJECT_ROOT / "CITATION.cff").read_text(encoding="utf-8"))
    root_zenodo = json.loads((PROJECT_ROOT / ".zenodo.json").read_text(encoding="utf-8"))

    assert root_cff["version"] == VERSION
    assert root_cff["doi"] == VERSION_DOI
    assert cff_names(root_cff) == AUTHORS
    assert root_zenodo["version"] == VERSION
    assert root_zenodo["related_identifiers"][0]["identifier"].endswith(f"/tree/v{VERSION}")


def test_highlights_meet_elsevier_contract_and_match_the_scientific_message() -> None:
    lines = [
        line.removeprefix("- ").strip()
        for line in (PAPER / "highlights.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    assert len(lines) == 5
    assert all(len(line) <= 85 for line in lines)
    joined = " ".join(lines).casefold()
    for phrase in ("corruption interaction", "yolo11", "regression head", "calibration"):
        assert phrase in joined
    assert "format superiority" not in joined
    assert "universal" not in joined


def test_rendered_main_has_results_before_discussion_and_no_layout_warnings() -> None:
    pdf = PROJECT_ROOT / "paper" / "main_nn.pdf"
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
    log_errors = (
        re.compile(r"Undefined control sequence"),
        re.compile(r"LaTeX Error"),
        re.compile(r"Citation .*undefined", re.I),
        re.compile(r"Reference .*undefined", re.I),
        re.compile(r"Emergency stop"),
    )
    for log_name in ("main_nn.log", "supplement.log"):
        log = (PAPER / log_name).read_text(encoding="utf-8")
        for pattern in log_errors:
            assert pattern.search(log) is None, f"{pattern.pattern} in {log_name}"


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
    metadata = zenodo["metadata"] if "metadata" in zenodo else zenodo
    names = [c["name"] for c in metadata["creators"]]

    assert metadata["title"] == f"{NN_TITLE}: Reproducibility Package"
    assert metadata["version"] == VERSION
    assert metadata["license"] == "MIT"
    assert names == AUTHORS
    assert metadata["related_identifiers"] == [
        {
            "identifier": f"https://github.com/iamdinhthuan/paired-floor-guarded-int8-fp8-detection/tree/v{VERSION}",
            "relation": "isSupplementTo",
            "scheme": "url",
            "resource_type": "software",
        }
    ]
    assert not AI_NAME.search(" ".join(names))


def test_no_funding_and_public_archive_statements_are_final() -> None:
    """Catches reintroduced placeholders or unsupported funding."""
    tex = NN_MAIN.read_text(encoding="utf-8")

    acknowledgments = submission_metadata.extract_section(tex, "Acknowledgments")
    assert "no specific grant" in acknowledgments.casefold()
    availability = submission_metadata.extract_section(tex, "Data and code availability")
    assert VERSION_DOI in availability
    assert CONCEPT_DOI in availability
    assert "author action required" not in tex.casefold()
    assert "doi pending" not in tex.casefold()


def test_ai_declaration_names_tools_and_keeps_authors_responsible() -> None:
    tex = NN_MAIN.read_text(encoding="utf-8")
    declaration = submission_metadata.extract_section(
        tex,
        "Declaration of generative AI and AI-assisted technologies in the manuscript preparation process",
    )
    assert "Cognition Devin" in declaration
    assert "Codex" in declaration or "ChatGPT" in declaration
    assert "Claude" in declaration
    assert "full responsibility" in declaration
    assert "AI tools are not creators or contributors" in (PAPER / ".zenodo.json").read_text(
        encoding="utf-8"
    )
    assert tex.count(r"\orcidlink{") == 5


def test_abstract_is_self_contained() -> None:
    tex = NN_MAIN.read_text(encoding="utf-8")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", tex, re.S)
    assert abstract
    assert r"\input{" not in abstract.group(1)
    assert r"\cite" not in abstract.group(1)


def test_duplicate_bibtex_key_cannot_hide_an_entry() -> None:
    """Catches a duplicate key silently overwriting a valid record."""
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    duplicate = "@article{dupfixture,\n  year = {1900}\n}\n\n@article{dupfixture,\n  year = {1901}\n}\n\n" + bib

    with pytest.raises(ValueError, match="duplicate"):
        bibtex_entries(duplicate)


def test_indented_duplicate_bibtex_key_is_rejected() -> None:
    """Catches a whitespace-prefixed duplicate BibTeX entry."""
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    duplicate = bib + "\n  @article{dupfixture,\n    year = {1900}\n  }\n@article{dupfixture,\n  year={1901}\n}\n"

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
    tex = NN_MAIN.read_text(encoding="utf-8")
    _, original = submission_metadata.manuscript_metadata(tex)
    _, changed = submission_metadata.manuscript_metadata(tex.replace(old, new, 1))
    assert changed != original


def test_current_manuscript_title_change_fails_when_cff_and_zenodo_are_unchanged() -> None:
    """Derives the active title, so the drift guard survives a legitimate retitling."""
    tex = NN_MAIN.read_text(encoding="utf-8")
    match = re.search(r"\\title(?:\[[^]]+\])?\{([^{}]+)\}", tex)
    assert match, "expected one active manuscript title"
    mutated = tex[: match.start(1)] + "Changed manuscript title" + tex[match.end(1) :]

    title, _ = submission_metadata.manuscript_metadata(mutated)
    cff = yaml.safe_load((PAPER / "CITATION.cff").read_text(encoding="utf-8"))
    assert title not in cff["title"]


def test_zenodo_title_must_match_the_manuscript_and_cff() -> None:
    """Catches drift among manuscript, CFF, and Zenodo release titles."""
    tex = NN_MAIN.read_text(encoding="utf-8")
    zenodo = json.loads((PAPER / ".zenodo.json").read_text(encoding="utf-8"))
    cff = yaml.safe_load((PAPER / "CITATION.cff").read_text(encoding="utf-8"))
    metadata = zenodo["metadata"] if "metadata" in zenodo else zenodo
    manuscript_title, _ = submission_metadata.manuscript_metadata(tex)
    expected = f"{manuscript_title}: Reproducibility Package"
    assert metadata["title"] == cff["title"] == expected


def test_later_active_title_is_rejected_as_ambiguous() -> None:
    """Catches a second active title that would otherwise override manuscript metadata."""
    tex = NN_MAIN.read_text(encoding="utf-8")
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
