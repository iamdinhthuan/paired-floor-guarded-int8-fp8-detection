#!/usr/bin/env python3
"""Create a clean, deterministic ZIP of the revised IVC submission package."""

from __future__ import annotations

import argparse
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
FIXED_TIME = (2026, 8, 20, 0, 0, 0)
ARCHIVE_ROOT = "IVC_submission_revised_20260820"

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
EXCLUDED_ROOT_FILES = {"main.pdf", "supplement.pdf"}


def include(path: Path) -> bool:
    rel = path.relative_to(ROOT)
    if any(part in EXCLUDED_DIRS or part.startswith("_render") for part in rel.parts[:-1]):
        return False
    if rel.as_posix() in EXCLUDED_ROOT_FILES:
        return False
    if any(path.name.endswith(suffix) for suffix in EXCLUDED_SUFFIXES):
        return False
    return path.is_file()


def write_file(zf: zipfile.ZipFile, arcname: str, data: bytes, executable: bool) -> None:
    info = zipfile.ZipInfo(arcname, FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = (0o755 if executable else 0o644) << 16
    zf.writestr(info, data)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--output",
        type=Path,
        default=ROOT.parent / "IVC_submission_revised_20260820.zip",
    )
    args = parser.parse_args()
    output = args.output.resolve()

    required = [
        ROOT / "SOURCE_MANIFEST.sha256",
        ROOT / "preview/main.pdf",
        ROOT / "preview/supplement.pdf",
        ROOT / "supplementary_data_s2.zip",
    ]
    missing = [str(path.relative_to(ROOT)) for path in required if not path.is_file()]
    if missing:
        raise SystemExit(f"Cannot package; required files missing: {missing}")

    files = sorted((path for path in ROOT.rglob("*") if include(path)), key=lambda p: p.relative_to(ROOT).as_posix())
    output.parent.mkdir(parents=True, exist_ok=True)
    if output.exists():
        output.unlink()
    with zipfile.ZipFile(output, "w", allowZip64=True) as zf:
        for path in files:
            rel = path.relative_to(ROOT).as_posix()
            arcname = f"{ARCHIVE_ROOT}/{rel}"
            executable = path.suffix == ".sh" or path.parent.name == "scripts" and path.suffix == ".py"
            write_file(zf, arcname, path.read_bytes(), executable)
    print(f"Wrote {output} with {len(files)} files ({output.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
