#!/usr/bin/env python3
"""Apply validated confirmatory abstract/highlights to the canonical paper source."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pilot_registry import canonical_hash
from submission_audit import validate_confirmatory_integration
from topic_c.manifest import sha256_file


BEGIN = r"\begin{abstract}"
END = r"\end{abstract}"


def _generated_body(generated: str) -> str:
    body = "\n".join(
        line for line in generated.splitlines() if not line.lstrip().startswith("%")
    ).strip()
    if not body:
        raise ValueError("generated abstract body is empty")
    return body + "\n"


def replace_literal_abstract(source: str, generated: str) -> str:
    if source.count(BEGIN) != 1 or source.count(END) != 1:
        raise ValueError("main source must contain exactly one abstract block")
    before, remainder = source.split(BEGIN, 1)
    _, after = remainder.split(END, 1)
    return before + BEGIN + "\n" + _generated_body(generated) + END + after


def _validated_generated(paper: Path) -> tuple[Path, Path]:
    audit_path = paper / "generated" / "confirmatory_evidence_audit.json"
    try:
        audit = json.loads(audit_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError("confirmatory evidence audit is missing or invalid") from exc
    if audit.get("audit_sha256") != canonical_hash(audit, "audit_sha256"):
        raise ValueError("confirmatory evidence audit self-hash mismatch")
    result = []
    for name in ("confirmatory_abstract.tex", "confirmatory_highlights.txt"):
        record = audit.get("generated", {}).get(name, {})
        path = (paper / "generated" / name).resolve()
        if (
            Path(record.get("path", "")).resolve() != path
            or not path.is_file()
            or record.get("sha256") != sha256_file(path)
        ):
            raise ValueError(f"confirmatory generated binding mismatch: {name}")
        result.append(path)
    return result[0], result[1]


def _atomic_replace(path: Path, content: str) -> None:
    temporary = path.with_suffix(path.suffix + ".confirmatory-tmp")
    if temporary.exists():
        raise ValueError(f"temporary output already exists: {temporary}")
    temporary.write_text(content, encoding="utf-8")
    temporary.replace(path)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--paper-root", type=Path, required=True)
    args = parser.parse_args()
    paper = args.paper_root.resolve()
    main_path = paper / "main.tex"
    highlights_path = paper / "highlights.txt"
    abstract_path, generated_highlights = _validated_generated(paper)
    source = main_path.read_text(encoding="utf-8")
    updated = replace_literal_abstract(
        source, abstract_path.read_text(encoding="utf-8")
    )
    highlights = generated_highlights.read_text(encoding="utf-8")
    if source != updated:
        _atomic_replace(main_path, updated)
    if not highlights_path.is_file() or highlights_path.read_text(encoding="utf-8") != highlights:
        _atomic_replace(highlights_path, highlights)
    validate_confirmatory_integration(
        paper,
        main_path.read_text(encoding="utf-8"),
        (paper / "supplement.tex").read_text(encoding="utf-8"),
    )
    print("CONFIRMATORY PAPER LITERALS APPLIED abstract=1 highlights=5")


if __name__ == "__main__":
    main()
