#!/usr/bin/env python3
"""Validate the compact canonical ``paper/`` package for CVIU submission."""

from __future__ import annotations

import argparse
import json
import re
import sys
from pathlib import Path

try:
    import yaml
    from PIL import Image
except ImportError as exc:  # pragma: no cover
    raise SystemExit("Validation requires PyYAML and Pillow") from exc


REPO = Path(__file__).resolve().parents[1]
TITLE = (
    "A Paired, Floor-Guarded Evaluation Protocol for INT8 and FP8 Object "
    "Detectors under Image Corruptions"
)
JOURNAL = "Computer Vision and Image Understanding"
AUTHORS = [
    "Dinh Thuan Nguyen",
    "Lam Phuong Nguyen",
    "Vinh Huy Nguyen",
    "Sy Vu Quang",
    "Mohan Rajesh Elara",
    "Anh Vu Le",
]
CORE = [
    "main.tex",
    "supplement.tex",
    "references.bib",
    "highlights.txt",
    "README.md",
    "CITATION.cff",
    ".zenodo.json",
    "cas-dc.cls",
    "cas-common.sty",
    "elsarticle-num.bst",
]
KEY_TABLES = [
    "datasets.tex",
    "decision_impact_summary.tex",
    "direct_absolute_guardrail_summary.tex",
    "direct_dataset_summary_compact.tex",
    "relative_retention_sensitivity.tex",
    "relative_retention_sign_disagreements.tex",
    "conditionality_scope_summary.tex",
    "holdout_split_summary.tex",
    "holdout_reference_parity.tex",
    "corruption_realization_seed_summary.tex",
    "corruption_realization_variance.tex",
    "tt100k_combined_endpoint_guardrail.tex",
    "training_realization_summary.tex",
    "direct_deployment_summary.tex",
    "cross_family_runtime_by_dataset.tex",
]
FRONT_MATTER = [
    "main.tex",
    "supplement.tex",
    "README.md",
    "highlights.txt",
    "cover_letter.txt",
    "CITATION.cff",
    ".zenodo.json",
    "zenodo.json",
]
AI_NAME = re.compile(r"\b(?:OpenAI|ChatGPT|Codex|Anthropic|Claude|Gemini|Copilot)\b", re.I)
BAD_TEXT = {
    "TODO/TBD": re.compile(r"\b(?:TODO|TBD)\b", re.I),
    "placeholder": re.compile(r"\bPLACEHOLDER\b|author action required", re.I),
    "pending DOI": re.compile(r"DOI\s*(?:is|:)\s*pending|<\s*DOI\s*>", re.I),
    "insert-field instruction": re.compile(
        r"(?:INSERT|ADD)\s+(?:THE\s+)?(?:DOI|REPOSITORY|URL|FUNDING)", re.I
    ),
    "old IVC acronym": re.compile(r"\bIVC\b"),
    "old IVC journal": re.compile(r"Image and Vision Computing", re.I),
    "old release": re.compile(r"\bv2\.0\.0\b|10\.5281/zenodo\.22031664"),
    "old title": re.compile(
        r"TensorRT INT8 and FP8 Object Detectors under Synthetic Image Corruptions", re.I
    ),
}
LOG_ERRORS = {
    "TeX error": re.compile(r"(?m)^!\s*(?:LaTeX|Package).*Error|Undefined control sequence"),
    "undefined citation/reference": re.compile(
        r"(?:Citation|Reference)[^\n]*(?:undefined|multiply defined)|"
        r"There were undefined references",
        re.I,
    ),
    "overfull box": re.compile(r"Overfull \\[hv]box", re.I),
    "underfull box": re.compile(r"Underfull \\[hv]box", re.I),
    "fatal TeX failure": re.compile(r"Emergency stop|Fatal error occurred", re.I),
}


def no_comments(text: str) -> str:
    """Remove unescaped LaTeX comments."""
    lines = []
    for line in text.splitlines():
        cut = len(line)
        for index, char in enumerate(line):
            if char == "%":
                slashes = 0
                cursor = index - 1
                while cursor >= 0 and line[cursor] == "\\":
                    slashes += 1
                    cursor -= 1
                if slashes % 2 == 0:
                    cut = index
                    break
        lines.append(line[:cut])
    return "\n".join(lines)


def plain_tex(text: str) -> str:
    text = no_comments(text)
    text = re.sub(r"\$[^$]*\$|\\\([^)]*\\\)|\\\[[^]]*\\\]", " ", text, flags=re.S)
    text = re.sub(r"\\(?:begin|end)\{[^}]+\}", " ", text)
    for _ in range(5):
        new = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?\{([^{}]*)\}", r" \1 ", text)
        if new == text:
            break
        text = new
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", text)
    return re.sub(r"[{}_^&]", " ", text.replace("~", " ").replace("--", "-"))


def local_file(root: Path, source: Path, token: str, suffixes: list[str], extras=()) -> Path | None:
    token = token.strip()
    if not token or "\\" in token or "#" in token:
        return None
    raw = Path(token)
    endings = [""] if raw.suffix else suffixes
    for base in (source.parent, root, *extras):
        for ending in endings:
            candidate = (base / f"{raw}{ending}").resolve()
            try:
                candidate.relative_to(root.resolve())
            except ValueError:
                continue
            if candidate.is_file():
                return candidate
    return None


def check_dependencies(root: Path, errors: list[str], notes: list[str]) -> None:
    missing = [name for name in CORE if not (root / name).is_file()]
    if missing:
        errors.append(f"missing core files: {missing}")
    if any(name in missing for name in ("main.tex", "supplement.tex")):
        return
    queue = [root / "main.tex", root / "supplement.tex"]
    seen: set[Path] = set()
    resolved_count = 0
    while queue:
        source = queue.pop().resolve()
        if source in seen:
            continue
        seen.add(source)
        text = no_comments(source.read_text(encoding="utf-8"))
        figure_dirs = [root / "figures", root / "generated"]
        for group in re.findall(r"\\graphicspath\s*\{((?:\{[^{}]*\})+)\}", text):
            figure_dirs.extend(source.parent / item for item in re.findall(r"\{([^{}]*)\}", group))

        for token in re.findall(r"\\(?:input|include)\s*\{([^{}]+)\}", text):
            path = local_file(root, source, token, [".tex"])
            if path:
                resolved_count += 1
                queue.append(path)
            else:
                errors.append(f"missing input from {source.relative_to(root)}: {token}")
        for token in re.findall(r"\\includegraphics\*?(?:\[[^]]*\])?\s*\{([^{}]+)\}", text):
            path = local_file(
                root,
                source,
                token,
                [".pdf", ".png", ".tif", ".tiff", ".jpg", ".jpeg", ".eps"],
                figure_dirs,
            )
            if path:
                resolved_count += 1
            else:
                errors.append(f"missing graphic from {source.relative_to(root)}: {token}")
        for group in re.findall(r"\\bibliography\s*\{([^{}]+)\}", text):
            for token in group.split(","):
                if local_file(root, source, token, [".bib"]):
                    resolved_count += 1
                else:
                    errors.append(f"missing bibliography: {token.strip()}")
        for token in re.findall(r"\\bibliographystyle\s*\{([^{}]+)\}", text):
            if local_file(root, source, token, [".bst"]):
                resolved_count += 1
            else:
                errors.append(f"bibliography style is not local: {token}")
    notes.append(f"dependencies: {len(seen)} TeX files, {resolved_count} local references")


def check_abstract_and_highlights(root: Path, errors: list[str], notes: list[str]) -> None:
    main = (root / "main.tex").read_text(encoding="utf-8")
    abstract = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.S)
    if not abstract:
        errors.append("main.tex has no abstract")
    else:
        words = re.findall(
            r"[A-Za-z0-9]+(?:[\u2010-\u2015'-][A-Za-z0-9]+)*", plain_tex(abstract.group(1))
        )
        if len(words) > 250:
            errors.append(f"abstract has {len(words)} words; maximum is 250")
        notes.append(f"abstract: {len(words)}/250 words")

    lines = [line.strip() for line in (root / "highlights.txt").read_text().splitlines() if line.strip()]
    if not 3 <= len(lines) <= 5:
        errors.append(f"highlights has {len(lines)} entries; expected 3--5")
    lengths = []
    for number, line in enumerate(lines, 1):
        if not line.startswith("- "):
            errors.append(f"highlight {number} must start with '- '")
        item = line[2:] if line.startswith("- ") else line
        lengths.append(len(item))
        if len(item) > 85:
            errors.append(f"highlight {number} has {len(item)} characters; maximum is 85")
    notes.append(f"highlights: {len(lines)} entries, lengths {lengths}")


def check_graphical_abstract(root: Path, errors: list[str], notes: list[str]) -> None:
    images = [
        root / name
        for name in ("graphical_abstract.png", "graphical_abstract.tif", "graphical_abstract.tiff")
        if (root / name).is_file()
    ]
    if not images:
        errors.append("missing graphical abstract PNG or TIFF")
    for path in images:
        try:
            with Image.open(path) as image:
                width, height = image.size
        except Exception as exc:  # pragma: no cover
            errors.append(f"cannot read {path.name}: {exc}")
            continue
        ratio = width / height if height else 0.0
        if width < 1328 or height < 531:
            errors.append(f"{path.name} is {width}x{height}; minimum is 1328x531")
        if not 2.35 <= ratio <= 2.65:
            errors.append(f"{path.name} ratio is {ratio:.3f}; expected roughly 2.5:1")
        notes.append(f"{path.name}: {width}x{height}, {ratio:.3f}:1")


def cff_names(data: dict) -> list[str]:
    return [
        f"{item.get('given-names', '')} {item.get('family-names', '')}".strip()
        for item in data.get("authors", [])
    ]


def zenodo_names(data: dict) -> tuple[dict, list[str]]:
    metadata = data.get("metadata", data)
    names = []
    for item in metadata.get("creators", []):
        raw = str(item.get("name", "")).strip()
        if "," in raw:
            family, given = (part.strip() for part in raw.split(",", 1))
            raw = f"{given} {family}".strip()
        names.append(raw)
    return metadata, names


def check_metadata(root: Path, errors: list[str], notes: list[str]) -> None:
    try:
        cff = yaml.safe_load((root / "CITATION.cff").read_text(encoding="utf-8"))
        if not isinstance(cff, dict):
            raise ValueError("top-level value is not a mapping")
    except Exception as exc:
        errors.append(f"invalid CITATION.cff YAML: {exc}")
        return
    zenodo_paths = [root / name for name in (".zenodo.json", "zenodo.json") if (root / name).is_file()]
    parsed = []
    for path in zenodo_paths:
        try:
            value = json.loads(path.read_text(encoding="utf-8"))
            if not isinstance(value, dict):
                raise ValueError("top-level value is not an object")
            parsed.append((path, value))
        except Exception as exc:
            errors.append(f"invalid {path.name} JSON: {exc}")

    main = (root / "main.tex").read_text(encoding="utf-8")
    author_region = main[main.find("\\title") : main.find("\\address")]
    main_names = []
    for line in author_region.splitlines():
        if re.match(r"^\s*\\author(?:\[[^]]*\])?\{", line):
            matches = [name for name in AUTHORS if name in line]
            main_names.append(matches[0] if len(matches) == 1 else "<unparsed>")
    if main_names != AUTHORS:
        errors.append(f"main.tex author list differs: {main_names!r}")
    names = cff_names(cff)
    if names != AUTHORS:
        errors.append(f"CITATION.cff author list differs: {names!r}")
    if any(AI_NAME.search(name) for name in names):
        errors.append("AI tool/provider appears among CITATION.cff authors")
    if cff.get("license") != "MIT":
        errors.append("CITATION.cff license is not MIT")
    if TITLE not in str(cff.get("title", "")) or TITLE not in main:
        errors.append("main/CITATION title does not match the CVIU title")

    for path, value in parsed:
        metadata, names = zenodo_names(value)
        if names != AUTHORS:
            errors.append(f"{path.name} creator list differs: {names!r}")
        if any(AI_NAME.search(name) for name in names):
            errors.append(f"AI tool/provider appears among {path.name} creators")
        if metadata.get("license") != "MIT":
            errors.append(f"{path.name} license is not MIT")
        if TITLE not in str(metadata.get("title", "")):
            errors.append(f"{path.name} title does not match the CVIU title")
    notes.append("metadata: valid YAML/JSON and six human authors cross-checked")


def check_text_tables_logs(root: Path, errors: list[str], notes: list[str]) -> None:
    target_seen = False
    for name in FRONT_MATTER:
        path = root / name
        if not path.is_file():
            continue
        text = path.read_text(encoding="utf-8")
        target_seen |= JOURNAL in text
        for label, pattern in BAD_TEXT.items():
            if pattern.search(text):
                errors.append(f"{label} found in {name}")
    if not target_seen:
        errors.append(f"target journal not found in publication files: {JOURNAL}")

    for name in KEY_TABLES:
        path = root / "generated" / name
        if not path.is_file():
            errors.append(f"missing key generated table: generated/{name}")
            continue
        text = path.read_text(encoding="utf-8")
        if path.stat().st_size <= 80 or not re.search(
            r"\\begin\{(?:tabular\*?|tabularx|longtable)\}", text
        ):
            errors.append(f"invalid generated table: generated/{name}")
    notes.append(f"key generated tables: {len(KEY_TABLES)} checked")

    log_count = 0
    for name in ("main.log", "supplement.log"):
        path = root / name
        if not path.is_file():
            continue
        log_count += 1
        text = path.read_text(encoding="utf-8", errors="replace")
        for label, pattern in LOG_ERRORS.items():
            if match := pattern.search(text):
                excerpt = " ".join(match.group(0).split())[:160]
                errors.append(f"{label} in {name}: {excerpt}")
    notes.append(f"final LaTeX logs: {log_count} checked" if log_count else "final LaTeX logs: absent")


def validate(root: Path) -> tuple[list[str], list[str]]:
    errors: list[str] = []
    notes: list[str] = []
    if not root.is_dir():
        return [f"paper directory does not exist: {root}"], notes
    check_dependencies(root, errors, notes)
    if (root / "main.tex").is_file() and (root / "highlights.txt").is_file():
        check_abstract_and_highlights(root, errors, notes)
    check_graphical_abstract(root, errors, notes)
    if all((root / name).is_file() for name in ("main.tex", "CITATION.cff", ".zenodo.json")):
        check_metadata(root, errors, notes)
    check_text_tables_logs(root, errors, notes)
    return errors, notes


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "--paper-dir",
        "--paper-root",
        dest="paper_dir",
        type=Path,
        default=REPO / "paper",
    )
    root = parser.parse_args().paper_dir.resolve()
    errors, notes = validate(root)
    print(f"CVIU package audit: {root}")
    for note in notes:
        print(f"  PASS  {note}")
    for error in errors:
        print(f"  FAIL  {error}")
    print("RESULT: PASS" if not errors else f"RESULT: FAIL ({len(errors)} issue(s))")
    return int(bool(errors))


if __name__ == "__main__":
    sys.exit(main())
