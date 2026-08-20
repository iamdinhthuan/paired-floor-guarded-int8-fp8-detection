#!/usr/bin/env python3
"""Validate the revised IVC submission package.

The checks are deliberately package-level. They do not rerun training,
TensorRT inference, or upstream artifact production.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import defaultdict
from pathlib import Path
from typing import Iterable

try:
    from PIL import Image
except ImportError as exc:  # pragma: no cover - explicit dependency failure
    raise SystemExit("Pillow is required for graphical-abstract validation") from exc

ROOT = Path(__file__).resolve().parents[1]

EXPECTED_CROSS_MEANS = {
    ("rtdetr-l", "int8-entropy"): -6.811576541336124,
    ("rtdetr-l", "fp8"): -0.017908668770093106,
    ("retinanet-r50-fpn-v2", "int8-entropy"): -6.214841431638553,
    ("retinanet-r50-fpn-v2", "fp8"): 0.012380115006584319,
}

REQUIRED_FILES = [
    "main.tex",
    "supplement.tex",
    "references.bib",
    "highlights.txt",
    "graphical_abstract.png",
    "CITATION.cff",
    "README.md",
    "AUTHOR_CHECKLIST.md",
    "CHANGELOG.md",
    "cover_letter.txt",
    "build.sh",
    "verify.sh",
    "cas-dc.cls",
    "cas-common.sty",
    "cas-model2-names.bst",
    "sections/01_introduction.tex",
    "sections/02_related_work.tex",
    "sections/03_methods.tex",
    "sections/04_results.tex",
    "sections/05_discussion.tex",
    "sections/06_conclusion.tex",
    "figures/fig_decision_impact.pdf",
    "figures/fig_decision_impact.png",
    "generated/decision_impact_cells.csv",
    "generated/decision_impact_audit.json",
    "generated/cross_family_interaction_cells.csv",
    "generated/cross_family_direct_cells.csv",
    "generated/cross_family_evidence_audit.json",
    "scripts/make_decision_impact.py",
    "scripts/rebuild_cross_family_evidence.py",
    "scripts/build_s2.py",
    "scripts/make_manifest.py",
    "scripts/build_submission_zip.py",
]

STALE_FILES = [
    "abstract_p0_fallback.tex",
    "cross_family_discussion.tex",
    "cross_family_methods.tex",
    "cross_family_results.tex",
    "direct_availability_template.tex",
    "direct_discussion_template.tex",
    "direct_methods_tail.tex",
    "direct_results_template.tex",
    "zenodo.json",
    "generated/cross_family_interaction_consistent.tex",
    "generated/cross_family_sign_consistency_audit.json",
]

SOURCE_TEXT_FILES = [
    "main.tex",
    "supplement.tex",
    "sections/01_introduction.tex",
    "sections/02_related_work.tex",
    "sections/03_methods.tex",
    "sections/04_results.tex",
    "sections/05_discussion.tex",
    "sections/06_conclusion.tex",
    "README.md",
    "CITATION.cff",
    "highlights.txt",
    "cover_letter.txt",
]

FORBIDDEN_PATTERNS = {
    "author-action placeholder": re.compile(r"author action required", re.I),
    "confirmation placeholder": re.compile(r"author confirmation required", re.I),
    "DOI placeholder": re.compile(r"doi pending|<doi>|doi:\s*tbd", re.I),
    "generic TBD/TODO": re.compile(r"\b(?:TBD|TODO)\b"),
    "stale floor-aware framing": re.compile(r"floor-aware", re.I),
    "stale cross-family artifact": re.compile(r"cross_family_interaction_consistent", re.I),
}


def fail(message: str) -> None:
    raise AssertionError(message)


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8-sig") as handle:
        return list(csv.DictReader(handle))


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def strip_tex_for_word_count(text: str) -> str:
    text = re.sub(r"(?m)%.*$", " ", text)
    text = re.sub(r"\\begin\{[^}]+\}|\\end\{[^}]+\}", " ", text)
    text = re.sub(r"\$[^$]*\$", " ", text, flags=re.S)
    text = re.sub(r"\\\([^)]*\\\)", " ", text, flags=re.S)
    text = re.sub(r"\\\[[^]]*\\\]", " ", text, flags=re.S)
    # Preserve command arguments where possible, then remove remaining commands.
    for _ in range(4):
        text = re.sub(
            r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?\{([^{}]*)\}", r" \1 ", text
        )
    text = re.sub(r"\\[A-Za-z@]+\*?(?:\[[^]]*\])?", " ", text)
    text = text.replace("--", "-").replace("~", " ")
    text = re.sub(r"[{}_^&]", " ", text)
    return text


def extract_bib_keys(text: str) -> set[str]:
    keys: set[str] = set()
    for match in re.finditer(r"\\cite\w*\*?(?:\[[^]]*\]){0,2}\{([^}]+)\}", text):
        keys.update(k.strip() for k in match.group(1).split(",") if k.strip())
    return keys


def parse_bib_keys(text: str) -> set[str]:
    return set(re.findall(r"@[A-Za-z]+\s*\{\s*([^,\s]+)\s*,", text))


def validate_required_files() -> None:
    missing = [rel for rel in REQUIRED_FILES if not (ROOT / rel).is_file()]
    if missing:
        fail(f"Missing required files: {missing}")
    stale = [rel for rel in STALE_FILES if (ROOT / rel).exists()]
    if stale:
        fail(f"Stale or contradictory files remain: {stale}")


def validate_text_and_citations() -> int:
    combined = ""
    for rel in SOURCE_TEXT_FILES:
        path = ROOT / rel
        text = path.read_text(encoding="utf-8")
        combined += "\n" + text
        for label, pattern in FORBIDDEN_PATTERNS.items():
            if pattern.search(text):
                fail(f"{label} found in {rel}")

    main = (ROOT / "main.tex").read_text(encoding="utf-8")
    match = re.search(r"\\begin\{abstract\}(.*?)\\end\{abstract\}", main, re.S)
    if not match:
        fail("Could not locate abstract in main.tex")
    abstract_plain = strip_tex_for_word_count(match.group(1))
    abstract_words = re.findall(r"[A-Za-z0-9]+(?:[-'][A-Za-z0-9]+)*", abstract_plain)
    if len(abstract_words) > 250:
        fail(f"Abstract has {len(abstract_words)} words; maximum is 250")

    manuscript_text = "\n".join(
        (ROOT / rel).read_text(encoding="utf-8")
        for rel in [
            "main.tex",
            "sections/01_introduction.tex",
            "sections/02_related_work.tex",
            "sections/03_methods.tex",
            "sections/04_results.tex",
            "sections/05_discussion.tex",
            "sections/06_conclusion.tex",
        ]
    )
    cited = extract_bib_keys(manuscript_text)
    available = parse_bib_keys((ROOT / "references.bib").read_text(encoding="utf-8"))
    missing = sorted(cited - available)
    if missing:
        fail(f"Citation keys missing from references.bib: {missing}")
    return len(abstract_words)


def validate_highlights() -> list[int]:
    lines = [line.strip() for line in (ROOT / "highlights.txt").read_text().splitlines() if line.strip()]
    if not 3 <= len(lines) <= 5:
        fail(f"Highlights has {len(lines)} entries; expected 3--5")
    lengths = [len(line) for line in lines]
    too_long = [(i + 1, n, lines[i]) for i, n in enumerate(lengths) if n > 85]
    if too_long:
        fail(f"Highlight exceeds 85 characters: {too_long}")
    return lengths


def validate_graphical_abstract() -> tuple[int, int, float]:
    with Image.open(ROOT / "graphical_abstract.png") as image:
        width, height = image.size
    if width < 1328 or height < 531:
        fail(f"Graphical abstract is {width}x{height}; minimum is 1328x531")
    ratio = width / height
    if not 2.35 <= ratio <= 2.65:
        fail(f"Graphical abstract aspect ratio {ratio:.3f} is outside 2.35--2.65")
    return width, height, ratio


def validate_decision_impact() -> dict[str, int]:
    rows = read_csv(ROOT / "generated/decision_impact_cells.csv")
    if len(rows) != 144:
        fail(f"Decision-impact ledger has {len(rows)} rows; expected 144")
    counts = {
        "raw_positive": 0,
        "raw_negative": 0,
        "adjusted_positive": 0,
        "adjusted_negative": 0,
        "raw_positive_to_adjusted_negative": 0,
        "raw_negative_to_adjusted_positive": 0,
        "below_10_ap": 0,
        "below_5_ap": 0,
    }
    for row in rows:
        raw = float(row["raw_corrupt_gap_ap"])
        adjusted = float(row["adjusted_interaction_ap"])
        minimum = float(row["min_corrupt_ap"])
        if raw > 0:
            counts["raw_positive"] += 1
        elif raw < 0:
            counts["raw_negative"] += 1
        else:
            fail("Zero raw corrupted gap encountered; inventory assumes strict signs")
        if adjusted > 0:
            counts["adjusted_positive"] += 1
        elif adjusted < 0:
            counts["adjusted_negative"] += 1
        else:
            fail("Zero adjusted interaction encountered; inventory assumes strict signs")
        if raw > 0 and adjusted < 0:
            counts["raw_positive_to_adjusted_negative"] += 1
        if raw < 0 and adjusted > 0:
            counts["raw_negative_to_adjusted_positive"] += 1
        if minimum < 10:
            counts["below_10_ap"] += 1
        if minimum < 5:
            counts["below_5_ap"] += 1

    expected = {
        "raw_positive": 134,
        "raw_negative": 10,
        "adjusted_positive": 78,
        "adjusted_negative": 66,
        "raw_positive_to_adjusted_negative": 56,
        "raw_negative_to_adjusted_positive": 0,
        "below_10_ap": 18,
        "below_5_ap": 5,
    }
    if counts != expected:
        fail(f"Decision-impact inventory mismatch: {counts} != {expected}")

    audit = json.loads((ROOT / "generated/decision_impact_audit.json").read_text())
    if audit.get("status") != "valid" or audit.get("rows") != 144:
        fail("Decision-impact audit is not valid")
    for rel, digest in audit["generated_sha256"].items():
        actual = sha256(ROOT / rel)
        if actual != digest:
            fail(f"Decision-impact hash mismatch for {rel}")
    return counts


def validate_cross_family() -> dict[str, float]:
    rows = read_csv(ROOT / "generated/cross_family_interaction_cells.csv")
    if len(rows) != 144:
        fail(f"Cross-family interaction ledger has {len(rows)} rows; expected 144")
    grouped: dict[tuple[str, str], list[float]] = defaultdict(list)
    for row in rows:
        grouped[(row["model"], row["precision"])].append(float(row["e"]) * 100.0)
    means = {key: sum(values) / len(values) for key, values in grouped.items()}
    for key, expected in EXPECTED_CROSS_MEANS.items():
        actual = means.get(key)
        if actual is None or not math.isclose(actual, expected, rel_tol=0, abs_tol=1e-7):
            fail(f"Cross-family mean E mismatch for {key}: {actual} != {expected}")
    if means[("rtdetr-l", "int8-entropy")] >= 0 or means[("retinanet-r50-fpn-v2", "int8-entropy")] >= 0:
        fail("Canonical cross-family INT8 E signs must be negative")

    direct = read_csv(ROOT / "generated/cross_family_direct_cells.csv")
    if len(direct) != 72:
        fail(f"Cross-family direct ledger has {len(direct)} rows; expected 72")

    audit = json.loads((ROOT / "generated/cross_family_evidence_audit.json").read_text())
    if audit.get("status") != "valid" or audit.get("schema_version") != 2:
        fail("Cross-family evidence audit is not valid schema 2")
    for rel, digest in audit["generated_sha256"].items():
        actual = sha256(ROOT / rel)
        if actual != digest:
            fail(f"Cross-family hash mismatch for {rel}")
    return {f"{k[0]}/{k[1]}": v for k, v in means.items()}


def validate_logs() -> list[str]:
    checked: list[str] = []
    patterns = [
        re.compile(r"! LaTeX Error"),
        re.compile(r"Undefined control sequence"),
        re.compile(r"Citation .* undefined"),
        re.compile(r"Reference .* undefined"),
        re.compile(r"There were undefined references"),
        re.compile(r"Overfull \\hbox"),
        re.compile(r"Underfull \\hbox"),
    ]
    for rel in ["main.log", "supplement.log"]:
        path = ROOT / rel
        if not path.exists():
            continue
        checked.append(rel)
        text = path.read_text(encoding="utf-8", errors="replace")
        hits = [pattern.pattern for pattern in patterns if pattern.search(text)]
        if hits:
            fail(f"LaTeX log issues in {rel}: {hits}")
    return checked


def validate_manifest() -> int:
    path = ROOT / "SOURCE_MANIFEST.sha256"
    if not path.is_file():
        fail("SOURCE_MANIFEST.sha256 is missing")
    count = 0
    for lineno, line in enumerate(path.read_text().splitlines(), start=1):
        if not line.strip():
            continue
        match = re.fullmatch(r"([0-9a-f]{64})  (.+)", line)
        if not match:
            fail(f"Malformed manifest line {lineno}: {line!r}")
        expected, rel = match.groups()
        target = ROOT / rel
        if not target.is_file():
            fail(f"Manifest target missing: {rel}")
        actual = sha256(target)
        if actual != expected:
            fail(f"Manifest hash mismatch: {rel}")
        count += 1
    if count == 0:
        fail("Manifest contains no files")
    return count


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check-manifest", action="store_true")
    args = parser.parse_args()

    try:
        validate_required_files()
        abstract_words = validate_text_and_citations()
        highlight_lengths = validate_highlights()
        ga = validate_graphical_abstract()
        decision = validate_decision_impact()
        cross_means = validate_cross_family()
        logs = validate_logs()
        manifest_entries = validate_manifest() if args.check_manifest else None
    except (AssertionError, FileNotFoundError, KeyError, ValueError, json.JSONDecodeError) as exc:
        print(f"VALIDATION FAILED: {exc}", file=sys.stderr)
        return 1

    print("VALIDATION PASSED")
    print(f"  abstract words: {abstract_words}/250")
    print(f"  highlight lengths: {highlight_lengths}")
    print(f"  graphical abstract: {ga[0]}x{ga[1]} (ratio {ga[2]:.3f})")
    print(f"  decision-impact inventory: {decision}")
    print("  cross-family mean E (AP points):")
    for key in sorted(cross_means):
        print(f"    {key}: {cross_means[key]:+.6f}")
    print(f"  LaTeX logs checked: {logs or 'not present'}")
    if manifest_entries is not None:
        print(f"  manifest entries verified: {manifest_entries}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
