#!/usr/bin/env python3
"""Freeze a paired image subset as a hash-bound selection record.

The subset record selects image IDs once (uniform without replacement
from a parent manifest's expected_image_ids) and binds the parent by
SHA-256.  Downstream subset evaluation cites this record; no image
bytes are duplicated.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from topic_c.manifest import read_manifest, sha256_file


def canonical(document: dict, field: str = "selection_sha256") -> str:
    payload = {key: value for key, value in document.items() if key != field}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def select_ids(universe_ids: list[int], count: int, seed: int) -> list[int]:
    if not 0 < count <= len(universe_ids):
        raise ValueError("subset count must be within the parent universe")
    if len(set(universe_ids)) != len(universe_ids) or universe_ids != sorted(universe_ids):
        raise ValueError("parent expected_image_ids must be unique and sorted")
    chosen = np.random.default_rng(seed).choice(len(universe_ids), count, replace=False)
    return sorted(int(universe_ids[index]) for index in chosen.tolist())


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--parent-manifest", type=Path, required=True,
                        help="hash-bound manifest whose expected_image_ids define the universe")
    parser.add_argument("--count", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--attempt", required=True,
                        help="attempt label recorded in the selection document")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    out = args.out.resolve()
    if out.exists() or Path(str(out) + ".complete").exists():
        raise SystemExit(f"SUBSET SELECTION REFUSED: output exists: {out}")
    parent_path = args.parent_manifest.resolve()
    parent = read_manifest(parent_path)
    marker = Path(str(parent_path) + ".complete")
    if not marker.is_file() or marker.read_text().strip() != parent["manifest_sha256"]:
        raise SystemExit("SUBSET SELECTION REFUSED: parent completion marker invalid")
    ids = select_ids([int(v) for v in parent["expected_image_ids"]], args.count, args.seed)
    if set(ids) - {int(r["image_id"]) for r in parent["records"]}:
        raise SystemExit("SUBSET SELECTION REFUSED: selected ID missing from parent records")
    document = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": args.attempt,
        "selection": "uniform_without_replacement_from_parent_expected_image_ids",
        "seed": args.seed,
        "count": args.count,
        "image_ids": ids,
        "parent_manifest": str(parent_path),
        "parent_manifest_sha256": parent["manifest_sha256"],
        "parent_dataset": parent.get("dataset"),
        "parent_split": parent.get("split"),
    }
    document["selection_sha256"] = canonical(document)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("x") as stream:
        json.dump(document, stream, indent=2)
        stream.write("\n")
    with Path(str(out) + ".complete").open("x") as stream:
        stream.write(document["selection_sha256"] + "\n")
    print(json.dumps({"subset": str(out), "count": len(ids),
                      "selection_sha256": document["selection_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
