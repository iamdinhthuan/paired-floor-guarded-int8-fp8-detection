#!/usr/bin/env python3
"""Rebind an immutable class-index map to a subset COCO annotation file."""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from topic_c.manifest import sha256_file


def canonical(document: dict) -> str:
    payload = {key: value for key, value in document.items() if key != "class_map_sha256"}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def rebound_document(source: dict, annotation: Path) -> dict:
    if source.get("class_map_sha256") != canonical(source):
        raise ValueError("source class map canonical hash mismatch")
    mapping = source.get("class_to_category_id")
    if not isinstance(mapping, list) or not mapping or not all(
        isinstance(item, int) and item > 0 for item in mapping
    ):
        raise ValueError("source class map is invalid")
    result = {
        **{key: value for key, value in source.items() if key != "class_map_sha256"},
        "annotation_sha256": sha256_file(annotation),
        "source_class_map_sha256": source["class_map_sha256"],
    }
    result["class_map_sha256"] = canonical(result)
    return result


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", required=True)
    parser.add_argument("--annotation", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    source_path = Path(args.source).resolve()
    annotation = Path(args.annotation).resolve()
    output = Path(args.out).resolve()
    marker = output.with_suffix(output.suffix + ".complete")
    if output.exists() or marker.exists():
        raise SystemExit("SUBSET CLASS MAP REFUSED: output exists")
    source = json.loads(source_path.read_text(encoding="utf-8"))
    result = rebound_document(source, annotation)
    result["source_class_map_path"] = str(source_path)
    result["class_map_sha256"] = canonical(result)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    marker.write_text(sha256_file(output) + "\n", encoding="utf-8")
    print(f"SUBSET CLASS MAP COMPLETE output={output}")


if __name__ == "__main__":
    main()
