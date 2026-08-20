#!/usr/bin/env python3
"""Create a SHA-256 manifest for source and retained submission artifacts."""

from __future__ import annotations

import hashlib
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "SOURCE_MANIFEST.sha256"

EXCLUDED_DIRS = {
    ".git",
    "__pycache__",
    "_render_main",
    "_render_supplement",
    "_renders",
    "_contacts",
    "_package_test",
}
EXCLUDED_SUFFIXES = {
    ".aux",
    ".log",
    ".out",
    ".blg",
    ".bbl",
    ".abs",
    ".fls",
    ".fdb_latexmk",
    ".synctex.gz",
    ".toc",
}
EXCLUDED_NAMES = {
    "SOURCE_MANIFEST.sha256",
}


def included(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS or part.startswith("_render") for part in rel.parts[:-1]):
        return False
    if path.name in EXCLUDED_NAMES:
        return False
    if rel.as_posix() in {"main.pdf", "supplement.pdf"}:
        return False
    if any(path.name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        return False
    return path.is_file()


def digest(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def main() -> None:
    files = sorted((p for p in ROOT.rglob("*") if included(p)), key=lambda p: p.relative_to(ROOT).as_posix())
    lines = [f"{digest(path)}  {path.relative_to(ROOT).as_posix()}" for path in files]
    OUTPUT.write_text("\n".join(lines) + "\n", encoding="utf-8")
    print(f"Wrote {OUTPUT.name} with {len(lines)} entries")


if __name__ == "__main__":
    main()
