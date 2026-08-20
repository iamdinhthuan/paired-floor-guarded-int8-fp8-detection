#!/usr/bin/env python3
"""Freeze a deterministic, train-only image list for ModelOpt calibration."""
from __future__ import annotations

import argparse
import hashlib
import json
import random
from datetime import datetime, timezone
from pathlib import Path

from ultralytics.utils import YAML

from topic_c.manifest import sha256_file
from confirmatory_data import split_images as _split_images
from confirmatory_data import validate_dataset_provenance


def canonical_hash(document: dict) -> str:
    payload = {key: value for key, value in document.items() if key != "calibration_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def split_images(root: Path, configured_split: object) -> list[Path]:
    return _split_images(root, configured_split, prefix="CALIBRATION LIST")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("voc", "kitti", "tt100k"), required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--acquisition-registry", required=True)
    parser.add_argument("--split-registry")
    parser.add_argument("--n-images", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260807)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    data_yaml, acquisition, output = Path(args.data_yaml).resolve(), Path(args.acquisition_registry).resolve(), Path(args.out).resolve()
    if output.exists() or output.with_suffix(output.suffix + ".complete").exists():
        raise SystemExit(f"CALIBRATION LIST REFUSED: output already exists: {output}")
    provenance = validate_dataset_provenance(
        dataset=args.dataset, data_yaml=data_yaml, acquisition_registry=acquisition,
        split_registry=Path(args.split_registry) if args.split_registry else None,
    )
    registry = json.loads(acquisition.read_text(encoding="utf-8"))
    data = YAML.load(data_yaml)
    root = Path(registry["dataset_root"]).resolve()
    if Path(data.get("path", "")).resolve() != root:
        raise SystemExit("CALIBRATION LIST REFUSED: data root differs from acquisition registry")
    images = split_images(root, data.get("train"))
    if args.n_images <= 0 or args.n_images > len(images):
        raise SystemExit("CALIBRATION LIST REFUSED: requested calibration count is invalid")
    chosen = random.Random(args.seed).sample(images, args.n_images)
    chosen.sort(key=lambda path: path.relative_to(root).as_posix())
    document = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": args.dataset,
        "split": "train", "selection": "uniform_without_replacement_from_images_only", "n_images": args.n_images,
        "seed": args.seed, "dataset_root": str(root), "data_yaml_sha256": sha256_file(data_yaml),
        "acquisition_registry_sha256": sha256_file(acquisition),
        "data_provenance": provenance,
        "records": [{"source_relpath": path.relative_to(root).as_posix(), "sha256": sha256_file(path)} for path in chosen],
    }
    document["calibration_sha256"] = canonical_hash(document)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".complete").write_text(document["calibration_sha256"] + "\n", encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "n_images": args.n_images, "out": str(output),
                      "calibration_sha256": document["calibration_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
