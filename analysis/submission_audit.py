#!/usr/bin/env python3
"""Fail closed on lightweight manuscript packaging invariants."""

from __future__ import annotations

import json
import re
import subprocess
from pathlib import Path

from pilot_registry import canonical_hash
from topic_c.manifest import sha256_file

from submission_metadata import (
    IVC_DOIS,
    bibtex_entries,
    bibtex_field,
    citation_keys,
    extract_section,
    validate_metadata_documents,
)


ROOT = Path(__file__).resolve().parents[1]
PAPER = ROOT / "paper"


def _normalize_pdf_text(text: str) -> str:
    """Normalize whitespace and ligatures emitted by common pdftotext builds."""
    ligatures = {
        "\ufb00": "ff",
        "\ufb01": "fi",
        "\ufb02": "fl",
        "\ufb03": "ffi",
        "\ufb04": "ffl",
        # Some EC-font PDFs are decoded by pdftotext as C0 controls.
        "\x1c": "fi",
        "\x1d": "fl",
        "\x1e": "ffi",
        "\x1f": "ffl",
    }
    for encoded, plain in ligatures.items():
        text = text.replace(encoded, plain)
    return " ".join(text.split())


def validate_confirmatory_integration(paper: Path, tex: str, supplement: str) -> None:
    """Bind every optional confirmatory fragment to one validated paper audit."""
    audit_path = paper / "generated" / "confirmatory_evidence_audit.json"
    if not audit_path.is_file():
        return
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("confirmatory paper audit is invalid JSON") from exc
    expected = {
        "confirmatory_methods.tex",
        "confirmatory_results.tex",
        "corruption_realization_results.tex",
        "confirmatory_conclusion.tex",
        "confirmatory_supplement.tex",
        "fixed_universe_results.tex",
        "confirmatory_abstract.tex",
        "confirmatory_highlights.txt",
    }
    records = audit.get("generated")
    if (
        audit.get("audit_sha256") != canonical_hash(audit, "audit_sha256")
        or not isinstance(records, dict)
        or set(records) != expected
    ):
        raise ValueError("confirmatory paper audit hash or generated grid is invalid")
    for name in expected:
        record = records[name]
        path = Path(record.get("path", "")).resolve()
        required = (paper / "generated" / name).resolve()
        if path != required or not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise ValueError(f"confirmatory fragment hash mismatch: {name}")
        if name == "confirmatory_abstract.tex":
            try:
                literal = tex.split(r"\begin{abstract}", 1)[1].split(
                    r"\end{abstract}", 1
                )[0]
            except IndexError as error:
                raise ValueError("confirmatory literal abstract is missing") from error
            generated_body = "\n".join(
                line
                for line in path.read_text(encoding="utf-8").splitlines()
                if not line.lstrip().startswith("%")
            )
            if re.sub(r"\s+", "", literal) != re.sub(r"\s+", "", generated_body):
                raise ValueError("confirmatory literal abstract drift")
        elif name == "confirmatory_highlights.txt":
            highlights = paper / "highlights.txt"
            if not highlights.is_file() or highlights.read_bytes() != path.read_bytes():
                raise ValueError("confirmatory highlights drift")
        else:
            target = supplement if name == "confirmatory_supplement.tex" else tex
            if rf"\input{{generated/{name}}}" not in target:
                raise ValueError(f"confirmatory fragment is not integrated: {name}")


def latex_log_issues(log_text: str, tex: str) -> list[str]:
    """Return actionable LaTeX diagnostics, narrowly classifying one CAS artifact.

    ``cas-dc`` emits an empty full-width title-box overfull diagnostic at its
    own ``\\maketitle`` call.  It contains no manuscript content and is
    visually absent in the rendered page.  Every other overfull box remains a
    release-blocking layout error.
    """
    issues = [
        pattern
        for pattern in ("undefined citations", "undefined references")
        if pattern.lower() in log_text.lower()
    ]
    cas_format = r"\documentclass[a4paper,fleqn]{cas-dc}" in tex
    tex_lines = tex.splitlines()
    for match in re.finditer(
        r"Overfull \\hbox \([^\n]+\) detected at line (\d+)", log_text
    ):
        line_number = int(match.group(1))
        source_line = (
            tex_lines[line_number - 1].strip()
            if 0 < line_number <= len(tex_lines)
            else ""
        )
        following = log_text[match.end() : match.end() + 16]
        empty_cas_titlebox = (
            cas_format
            and source_line == r"\maketitle"
            and re.match(r"\n\[\]\n\s*\[\]", following) is not None
        )
        if not empty_cas_titlebox:
            issues.append("Overfull")
    return issues


def abstract_word_count(tex: str, *, root: Path | None = None) -> int:
    """Count words after resolving one owned abstract ``\\input`` source.

    The active CAS source selects an evidence-gated abstract by a macro.  The
    selected fallback is a normal flat input, and resolving it here keeps the
    submission word limit meaningful without treating arbitrary TeX paths as
    trusted files.
    """
    abstract = tex.split(r"\begin{abstract}", 1)[1].split(
        r"\end{abstract}", 1
    )[0]
    if abstract.strip() == r"\activeabstract":
        if root is None:
            root = PAPER
        source = (
            "generated/direct_abstract.tex"
            if (root / "generated" / "direct_evidence_audit.json").is_file()
            else "abstract_p0_fallback.tex"
        )
        abstract = rf"\input{{{source}}}"
    input_match = re.fullmatch(
        r"\s*\\input\{([A-Za-z0-9_.-]+(?:/[A-Za-z0-9_.-]+)*)\}\s*", abstract
    )
    if input_match is not None:
        if root is None:
            root = PAPER
        candidate = (root / input_match.group(1)).resolve()
        try:
            candidate.relative_to(root.resolve())
        except ValueError as error:
            raise SystemExit("abstract input escapes paper root") from error
        if not candidate.is_file():
            raise SystemExit(f"abstract input is missing: {input_match.group(1)}")
        abstract = candidate.read_text(encoding="utf-8")
    abstract = re.sub(r"\\[A-Za-z]+(?:\{\})?", "VALUE", abstract)
    abstract = re.sub(r"[{}~]", " ", abstract)
    return len(re.findall(r"\b[\w’-]+\b", abstract))


def validate_literal_direct_abstract(tex: str, generated: str) -> None:
    r"""Fail if CAS's literal abstract drifts from generated direct evidence.

    CAS captures this block through a verbatim-writing path, so ``\input``
    indirection leaves ``main.abs`` empty.  Ignoring layout whitespace and the
    generated-file comment preserves that required literal form while binding
    every substantive TeX token to the generated artifact.
    """
    try:
        literal = tex.split(r"\begin{abstract}", 1)[1].split(
            r"\end{abstract}", 1
        )[0]
    except IndexError as error:
        raise SystemExit("direct abstract is missing") from error
    generated_body = "\n".join(
        line for line in generated.splitlines() if not line.lstrip().startswith("%")
    )
    if re.sub(r"\s+", "", literal) != re.sub(r"\s+", "", generated_body):
        raise SystemExit("literal CAS abstract drift from generated direct evidence")


def validate_submission_metadata(
    tex: str,
    bib: str,
    *,
    cff_text: str | None = None,
    zenodo_text: str | None = None,
) -> tuple[int, int]:
    cited = citation_keys(tex, root=PAPER)
    try:
        entries = bibtex_entries(bib)
    except ValueError as error:
        raise SystemExit(str(error)) from error
    missing = sorted(cited - set(entries))
    orphaned = sorted(set(entries) - cited)
    if missing or orphaned:
        raise SystemExit(f"bibliography closure failure: missing={missing}, orphaned={orphaned}")

    try:
        old_entries = sorted(
            key
            for key, entry in entries.items()
            if not 2023 <= int(bibtex_field(entry, "year")) <= 2026
        )
    except (ValueError, TypeError) as error:
        raise SystemExit(str(error)) from error
    if old_entries:
        raise SystemExit(f"bibliography contains out-of-window entries: {old_entries}")

    ivc_entries = [
        entry
        for entry in entries.values()
        if bibtex_field(entry, "journal", required=False).casefold()
        == "image and vision computing"
    ]
    try:
        ivc_dois = {bibtex_field(entry, "doi").casefold() for entry in ivc_entries}
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if len(ivc_entries) != 5 or ivc_dois != IVC_DOIS:
        raise SystemExit(
            f"required IVC subset mismatch: entries={len(ivc_entries)}, dois={sorted(ivc_dois)}"
        )

    if cff_text is None:
        cff_text = (PAPER / "CITATION.cff").read_text(encoding="utf-8")
    if zenodo_text is None:
        zenodo_text = (PAPER / "zenodo.json").read_text(encoding="utf-8")
    try:
        validate_metadata_documents(tex, cff_text, zenodo_text)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    try:
        funding = extract_section(tex, "Funding")
        availability = extract_section(tex, "Data and code availability")
    except ValueError as error:
        raise SystemExit(str(error)) from error
    if "author action required" not in funding.casefold():
        raise SystemExit("Funding must remain an author-action gate")
    if "no specific funding" in funding.casefold():
        raise SystemExit("Funding statement is not supported by project records")
    if "doi pending" not in availability.casefold():
        raise SystemExit("Data availability must state that the DOI is pending")

    return len(entries), len(ivc_entries)


def results_float_order_issues(page_texts: list[str]) -> list[str]:
    """Require every direct-Results float to render before Discussion begins."""
    markers = {
        "absolute-accuracy guardrail table": "Hash-matched absolute-accuracy guardrail",
        "paired excess-gap figure": "Direct INT8-FP8 paired excess-gap evidence",
        "format-contrast table": "Dataset-level direct format contrasts.",
        "heterogeneity table": "Descriptive heterogeneity across checkpoint rung",
        "deployment table": "RTX 5090 TensorRT engine measurements after three-repetition",
    }
    normalized_pages = [" ".join(page.replace("–", "-").split()) for page in page_texts]
    discussion_page = next(
        (
            index
            for index, page in enumerate(normalized_pages)
            if re.search(r"\b\d+\.\s+Discussion\b", page)
        ),
        None,
    )
    if discussion_page is None:
        return ["rendered Discussion heading is missing"]
    issues: list[str] = []
    for label, marker in markers.items():
        normalized_marker = marker.replace("–", "-")
        marker_page = next(
            (
                index
                for index, page in enumerate(normalized_pages)
                if normalized_marker in page
            ),
            None,
        )
        if marker_page is None:
            issues.append(f"rendered {label} is missing")
        elif marker_page > discussion_page:
            issues.append(f"rendered {label} appears after Discussion begins")
        elif marker_page == discussion_page:
            page = normalized_pages[marker_page]
            marker_position = page.find(normalized_marker)
            discussion_match = re.search(r"\b\d+\.\s+Discussion\b", page)
            if discussion_match is not None and marker_position > discussion_match.start():
                issues.append(f"rendered {label} appears after Discussion begins")
    return issues


def main() -> None:
    tex = (PAPER / "main.tex").read_text(encoding="utf-8")
    bib = (PAPER / "references.bib").read_text(encoding="utf-8")
    supplement = (PAPER / "supplement.tex").read_text(encoding="utf-8")
    try:
        validate_confirmatory_integration(PAPER, tex, supplement)
    except ValueError as error:
        raise SystemExit(str(error)) from error

    words = abstract_word_count(tex, root=PAPER)
    if words > 250:
        raise SystemExit(f"abstract exceeds 250 words: {words}")
    direct_abstract = PAPER / "generated" / "direct_abstract.tex"
    confirmatory_audit = PAPER / "generated" / "confirmatory_evidence_audit.json"
    if direct_abstract.is_file() and not confirmatory_audit.is_file():
        validate_literal_direct_abstract(
            tex, direct_abstract.read_text(encoding="utf-8")
        )

    highlights = [
        line.removeprefix("- ").strip()
        for line in (PAPER / "highlights.txt").read_text(encoding="utf-8").splitlines()
        if line.strip()
    ]
    if not 3 <= len(highlights) <= 5:
        raise SystemExit(f"expected 3--5 highlights, found {len(highlights)}")
    too_long = [(i + 1, len(line)) for i, line in enumerate(highlights) if len(line) > 85]
    if too_long:
        raise SystemExit(f"highlights exceed 85 characters: {too_long}")

    bibliography_entries, ivc_entries = validate_submission_metadata(tex, bib)

    log = PAPER / "main.log"
    if log.exists():
        log_text = log.read_text(encoding="utf-8", errors="replace")
        bad = latex_log_issues(log_text, tex)
        if bad:
            raise SystemExit(f"LaTeX log contains: {bad}")

    pdf = PAPER / "main.pdf"
    if not pdf.exists() or pdf.stat().st_size == 0:
        raise SystemExit("paper/main.pdf is missing or empty")
    info = subprocess.run(
        ["pdfinfo", str(pdf)], check=True, capture_output=True, text=True
    ).stdout
    abstract_capture = PAPER / "main.abs"
    if not abstract_capture.is_file() or not abstract_capture.read_text(
        encoding="utf-8", errors="replace"
    ).strip():
        raise SystemExit("CAS abstract capture is empty")
    rendered_text = subprocess.run(
        ["pdftotext", str(pdf), "-"], check=True, capture_output=True, text=True
    ).stdout
    normalized_rendered = _normalize_pdf_text(rendered_text)
    if "--" in rendered_text:
        raise SystemExit("rendered PDF contains a literal TeX double hyphen")
    if confirmatory_audit.is_file():
        if (
            "An untouchedholdout validation on VOC and KITTI yielded"
            not in normalized_rendered
            or "A three-realization sensitivity yielded" not in normalized_rendered
        ):
            raise SystemExit("rendered PDF does not contain the confirmatory abstract")
    elif "The primary grid covers 144 corruption cells spanning COCO, PASCAL VOC, KITTI, and TT100K" not in normalized_rendered:
        raise SystemExit("rendered PDF does not contain the direct abstract")
    if (
        "gap contraction near a common AP floor" not in normalized_rendered
        or "is not robustness evidence" not in normalized_rendered
    ):
        raise SystemExit("rendered PDF does not contain the floor-aware abstract conclusion")
    layout_pages = subprocess.run(
        ["pdftotext", "-layout", str(pdf), "-"],
        check=True,
        capture_output=True,
        text=True,
    ).stdout.split("\f")
    float_issues = results_float_order_issues(layout_pages)
    if float_issues:
        raise SystemExit(f"Results float order failure: {float_issues}")
    pages = re.search(r"^Pages:\s+(\d+)", info, flags=re.MULTILINE)
    print(
        {
            "abstract_words": words,
            "highlights": len(highlights),
            "max_highlight_characters": max(map(len, highlights)),
            "citations": len(citation_keys(tex, root=PAPER)),
            "bibliography_entries": bibliography_entries,
            "ivc_entries": ivc_entries,
            "pdf_pages": int(pages.group(1)) if pages else None,
            "status": "pass",
        }
    )


if __name__ == "__main__":
    main()
