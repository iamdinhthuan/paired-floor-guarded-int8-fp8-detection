#!/usr/bin/env python3
"""Write a compact, non-destructive inventory of the local experiment archive."""

from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ROOT = Path("artifacts/four_dataset_pilot_v1")
SECTIONS = (
    "configs",
    "outputs",
    "manifests",
    "engines",
    "legacy_coco",
    "pretrained",
    "execution_snapshot",
)


def scan(root: Path) -> dict[str, object]:
    sections: dict[str, object] = {}
    total_files = 0
    total_bytes = 0
    for name in SECTIONS:
        path = root / name
        files = [item for item in path.rglob("*") if item.is_file()] if path.exists() else []
        size = sum(item.stat().st_size for item in files)
        sections[name] = {
            "exists": path.exists(),
            "files": len(files),
            "bytes": size,
        }
        total_files += len(files)
        total_bytes += size

    output_root = root / "outputs"
    output_sections: dict[str, object] = {}
    if output_root.exists():
        for path in sorted(item for item in output_root.iterdir() if item.is_dir()):
            files = [item for item in path.rglob("*") if item.is_file()]
            output_sections[path.name] = {
                "files": len(files),
                "bytes": sum(item.stat().st_size for item in files),
            }

    return {
        "schema": "topic-c-local-artifact-inventory-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "total_files": total_files,
        "total_bytes": total_bytes,
        "sections": sections,
        "output_sections": output_sections,
        "integrity_note": (
            "This is a size/count inventory. Scientific integrity is enforced separately "
            "by paper/generated/artifact_audit.json and the SHA-256 completion reports."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or args.root / "artifact_inventory.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(scan(args.root), indent=2) + "\n", encoding="utf-8")
    print(output)


if __name__ == "__main__":
    main()
