#!/usr/bin/env python3
"""Compact a completed RetinaNet run to its inference checkpoint only."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from pathlib import Path

import torch


def digest(path: Path) -> str:
    value = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--registry", required=True)
    args = parser.parse_args()
    registry = Path(args.registry).resolve()
    marker = registry.with_suffix(registry.suffix + ".complete")
    if not registry.is_file() or not marker.is_file() or marker.read_text().strip() != digest(registry):
        raise SystemExit("refusing to compact an incomplete registry")
    record = json.loads(registry.read_text(encoding="utf-8"))
    best = Path(record["best_weights"]).resolve()
    last = Path(record["last_weights"]).resolve()
    if digest(best) != record["best_weights_sha256"] or digest(last) != record["last_weights_sha256"]:
        raise SystemExit("checkpoint digest mismatch")
    state = torch.load(best, map_location="cpu", weights_only=False)
    compact = {
        "schema_version": 1,
        "epoch": state["epoch"],
        "model": state["model"],
        "validation_loss": state["best_validation_loss"],
        "profile_sha256": state["profile_sha256"],
        "data_yaml_sha256": state["data_yaml_sha256"],
    }
    temporary = best.with_suffix(best.suffix + ".compact")
    torch.save(compact, temporary)
    os.replace(temporary, best)
    record["best_weights_sha256"] = digest(best)
    record.pop("last_weights", None)
    record.pop("last_weights_sha256", None)
    registry_tmp = registry.with_suffix(registry.suffix + ".compact")
    registry_tmp.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    os.replace(registry_tmp, registry)
    marker.write_text(digest(registry) + "\n", encoding="utf-8")
    last.unlink()
    print(json.dumps({"best": str(best), "best_sha256": digest(best), "registry_sha256": digest(registry)}))


if __name__ == "__main__":
    main()
