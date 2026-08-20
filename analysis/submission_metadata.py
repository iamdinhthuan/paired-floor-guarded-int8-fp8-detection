"""Fail-closed parsers and relational checks for submission metadata."""

from __future__ import annotations

import json
import re
from pathlib import Path

import yaml


IVC_DOIS = {
    "10.1016/j.imavis.2023.104692",
    "10.1016/j.imavis.2024.105035",
    "10.1016/j.imavis.2024.105054",
    "10.1016/j.imavis.2024.105095",
    "10.1016/j.imavis.2025.105740",
}
ZENODO_UPLOAD_TYPES = {
    "publication",
    "poster",
    "presentation",
    "dataset",
    "image",
    "video",
    "software",
    "lesson",
    "physicalobject",
    "other",
}
ZENODO_ACCESS_RIGHTS = {"open", "embargoed", "restricted", "closed"}


def strip_tex_comments(tex: str) -> str:
    """Remove true TeX comments while retaining percent signs escaped by ``\\``."""
    active_lines: list[str] = []
    for line in tex.splitlines(keepends=True):
        for index, character in enumerate(line):
            if character != "%":
                continue
            slashes = 0
            cursor = index - 1
            while cursor >= 0 and line[cursor] == "\\":
                slashes += 1
                cursor -= 1
            if slashes % 2 == 0:
                line = line[:index] + ("\n" if line.endswith("\n") else "")
                break
        active_lines.append(line)
    return "".join(active_lines)


def citation_keys(tex: str, *, root: Path | None = None) -> set[str]:
    """Return citations from TeX and, when owned, its recursively included files."""
    active = strip_tex_comments(tex)
    found: set[str] = set()
    for group in re.findall(r"\\cite\w*\{([^}]+)\}", active):
        found.update(key.strip() for key in group.split(",") if key.strip())
    if root is None:
        return found
    resolved_root = root.resolve()
    visited: set[Path] = set()

    def collect_inputs(source: str) -> None:
        for relative in re.findall(r"\\input\{([^}]+)\}", strip_tex_comments(source)):
            candidate = (resolved_root / relative).resolve()
            if candidate.suffix == "":
                candidate = candidate.with_suffix(".tex")
            try:
                candidate.relative_to(resolved_root)
            except ValueError as error:
                raise ValueError("TeX input escapes manuscript root") from error
            if candidate in visited:
                continue
            if not candidate.is_file():
                # Inactive evidence branches can reference generated files that
                # are intentionally absent from a compact submission package.
                # The LaTeX build validates the selected branch; citation
                # closure follows only owned inputs that are actually present.
                continue
            visited.add(candidate)
            included = candidate.read_text(encoding="utf-8")
            for group in re.findall(r"\\cite\w*\{([^}]+)\}", strip_tex_comments(included)):
                found.update(key.strip() for key in group.split(",") if key.strip())
            collect_inputs(included)

    collect_inputs(active)
    return found


def bibtex_entries(text: str) -> dict[str, str]:
    entries: dict[str, str] = {}
    for match in re.finditer(
        r"@\w+\s*\{\s*(?P<key>[^,\s]+)\s*,.*?(?=\n[ \t]*@|\Z)",
        text,
        flags=re.DOTALL,
    ):
        key = match.group("key")
        if key in entries:
            raise ValueError(f"duplicate BibTeX key: {key}")
        entries[key] = match.group(0)
    return entries


def bibtex_field(entry: str, field: str, *, required: bool = True) -> str:
    match = re.search(rf"\b{field}\s*=\s*\{{([^}}]+)\}}", entry, flags=re.I)
    if match:
        return match.group(1).strip()
    if required:
        raise ValueError(f"bibliography entry lacks {field}: {entry.splitlines()[0]}")
    return ""


def _command_calls(tex: str, command: str) -> list[tuple[str | None, str]]:
    """Return balanced TeX arguments and an optional single bracket argument."""
    calls: list[tuple[str | None, str]] = []
    for match in re.finditer(
        rf"\\{command}(?:\[(?P<option>[^\]]+)\])?\s*\{{", tex
    ):
        start = match.end()
        depth = 1
        for index in range(start, len(tex)):
            if tex[index] == "{":
                depth += 1
            elif tex[index] == "}":
                depth -= 1
                if depth == 0:
                    calls.append((match.group("option"), tex[start:index]))
                    break
        else:
            raise ValueError(f"unterminated \\{command} argument")
    return calls


def _command_argument(tex: str, command: str) -> str:
    calls = _command_calls(tex, command)
    if not calls:
        raise ValueError(f"missing \\{command} argument")
    if len(calls) != 1:
        raise ValueError(f"multiple active \\{command} definitions")
    return calls[0][1]


def _cas_manuscript_authors(tex: str) -> list[dict[str, str]]:
    author_calls = _command_calls(tex, "author")
    address_calls = _command_calls(tex, "address")
    addresses: dict[str, str] = {}
    for key, value in address_calls:
        if not key or key in addresses:
            raise ValueError("CAS manuscript address metadata is incomplete")
        addresses[key] = _plain_tex(value)
    authors: list[dict[str, str]] = []
    for address_keys, value in author_calls:
        if not address_keys:
            raise ValueError("CAS manuscript author lacks an address key")
        keys = [key.strip() for key in address_keys.split(",") if key.strip()]
        if len(keys) != 1 or keys[0] not in addresses:
            raise ValueError("CAS manuscript author metadata is incomplete")
        name = re.sub(r"\\(?:orcidlink|corref)\s*\{[^}]*\}", "", value)
        name = _plain_tex(name)
        parts = name.split()
        if len(parts) < 2:
            raise ValueError("CAS manuscript author metadata is incomplete")
        authors.append(
            {
                "family-names": parts[-1],
                "given-names": " ".join(parts[:-1]),
                "affiliation": addresses[keys[0]],
            }
        )
    if not authors:
        raise ValueError("CAS manuscript author metadata is empty")
    return authors


def _article_manuscript_authors(tex: str) -> list[dict[str, str]]:
    author_block = _command_argument(tex, "author")
    split = re.split(r"\\\\\s*\[[^]]+\]", author_block, maxsplit=1)
    if len(split) != 2:
        raise ValueError("author block lacks an affiliation separator")
    names_block, affiliations_block = split
    affiliations = {
        match.group("index"): _plain_tex(match.group("value"))
        for match in re.finditer(
            r"\\small\s*\$\^(?P<index>\d+)\$(?P<value>.*?)(?=\\\\|\Z)",
            affiliations_block,
            flags=re.DOTALL,
        )
    }
    authors: list[dict[str, str]] = []
    for match in re.finditer(
        r"(?P<name>.*?)\$\^\{(?P<index>\d+)(?:,[^}]*)?\}\$",
        names_block,
        flags=re.DOTALL,
    ):
        name = _plain_tex(match.group("name"))
        parts = name.split()
        if len(parts) < 2 or match.group("index") not in affiliations:
            raise ValueError("manuscript author metadata is incomplete")
        authors.append(
            {
                "family-names": parts[-1],
                "given-names": " ".join(parts[:-1]),
                "affiliation": affiliations[match.group("index")],
            }
        )
    if not authors:
        raise ValueError("manuscript author metadata is empty")
    return authors


def _plain_tex(value: str) -> str:
    value = re.sub(r"\\(?:small|textbf)\b", "", value)
    value = re.sub(r"\\\\(?:\[[^]]*\])?", " ", value)
    value = re.sub(r"\s+", " ", value)
    return value.strip(" ,\n")


def manuscript_metadata(tex: str) -> tuple[str, list[dict[str, str]]]:
    """Extract the active manuscript title and author/affiliation records."""
    active_tex = strip_tex_comments(tex)
    title = _plain_tex(_command_argument(active_tex, "title"))
    author_calls = _command_calls(active_tex, "author")
    if len(author_calls) > 1 or any(option is not None for option, _ in author_calls):
        authors = _cas_manuscript_authors(active_tex)
    else:
        authors = _article_manuscript_authors(active_tex)
    return title, authors


def extract_section(tex: str, heading: str) -> str:
    active_tex = strip_tex_comments(tex)
    start = re.search(
        rf"\\section\*?\s*\{{\s*{re.escape(heading)}\s*\}}",
        active_tex,
        flags=re.IGNORECASE,
    )
    if not start:
        raise ValueError(f"missing {heading} section")
    end = re.search(r"\\section\*?\s*\{", active_tex[start.end() :])
    return active_tex[start.end() : start.end() + end.start() if end else len(active_tex)]


def validate_metadata_documents(
    tex: str, cff_text: str, zenodo_text: str
) -> None:
    """Validate local CFF/Zenodo structure and exact active-manuscript linkage."""
    title, authors = manuscript_metadata(tex)
    try:
        cff = yaml.safe_load(cff_text)
    except yaml.YAMLError as error:
        raise ValueError(f"invalid CITATION.cff YAML: {error}") from error
    if not isinstance(cff, dict):
        raise ValueError("CITATION.cff must be a mapping")
    if cff.get("cff-version") != "1.2.0":
        raise ValueError("CITATION.cff must declare version 1.2.0")
    if cff.get("type") not in {"software", "dataset"}:
        raise ValueError("CITATION.cff type must be software or dataset")
    if not isinstance(cff.get("message"), str) or not cff["message"].strip():
        raise ValueError("CITATION.cff requires a non-empty message")
    if "author confirmation required" not in cff["message"].casefold():
        raise ValueError("CITATION.cff must require author confirmation")
    if cff.get("title") != title or cff.get("authors") != authors:
        raise ValueError("CFF manuscript metadata mismatch")

    try:
        zenodo = json.loads(zenodo_text)
    except json.JSONDecodeError as error:
        raise ValueError(f"invalid zenodo.json: {error}") from error
    metadata = zenodo.get("metadata") if isinstance(zenodo, dict) else None
    if zenodo.get("$schema") != "https://zenodo.org/schemas/deposits/metadata.json":
        raise ValueError("zenodo.json has an unexpected schema")
    if not isinstance(metadata, dict):
        raise ValueError("zenodo.json requires a metadata object")
    if metadata.get("title") != title:
        raise ValueError("Zenodo title must match manuscript metadata")
    if not isinstance(metadata.get("description"), str) or not metadata["description"].strip():
        raise ValueError("zenodo.json requires a non-empty description")
    if metadata.get("upload_type") not in ZENODO_UPLOAD_TYPES:
        raise ValueError("zenodo.json has an invalid upload_type")
    if metadata.get("access_right") not in ZENODO_ACCESS_RIGHTS:
        raise ValueError("zenodo.json has an invalid access_right")
    if metadata.get("creators") != [{"name": "AUTHOR CONFIRMATION REQUIRED"}]:
        raise ValueError("zenodo.json creators must remain author-confirmation placeholders")
    if not isinstance(metadata.get("keywords"), list) or not all(
        isinstance(keyword, str) and keyword.strip() for keyword in metadata["keywords"]
    ):
        raise ValueError("zenodo.json requires non-empty string keywords")
    if "doi pending" not in metadata["description"].casefold():
        raise ValueError("zenodo.json must state that the DOI is pending")
