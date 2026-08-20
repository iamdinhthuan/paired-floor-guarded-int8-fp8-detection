#!/usr/bin/env python3
"""Generate a corruption realization whose base RNG is nested across severity."""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

from PIL import Image

from generate_corruption import transform
from topic_c.manifest import (
    manifest_sha256,
    official_coco_images,
    sha256_file,
    validate_manifest,
)


def stable_realization_seed(
    dataset: str, image_id: int, corruption: str, realization_seed: int
) -> int:
    text = f"{dataset}:{image_id}:{corruption}:realization:{realization_seed}".encode()
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--corruption", required=True)
    parser.add_argument("--severity", type=int, choices=(1, 3, 5), required=True)
    parser.add_argument("--realization-seed", type=int, required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--resume-validated", action="store_true")
    args = parser.parse_args()
    manifest_path = Path(args.manifest_out).resolve()
    cache_root = Path(args.cache_root).resolve()
    clean_root = Path(args.clean_root).resolve()
    if manifest_path.exists() or manifest_path.with_suffix(manifest_path.suffix + ".complete").exists():
        raise SystemExit("REALIZATION CORRUPTION REFUSED: manifest exists")
    config_path = Path(args.config).resolve()
    config_bytes = config_path.read_bytes()
    config = json.loads(config_bytes)
    try:
        parameters = config["corruptions"][args.corruption]["severity"][str(args.severity)]
    except KeyError as exc:
        raise SystemExit(f"REALIZATION CORRUPTION REFUSED: unknown condition: {exc}") from exc
    records = []
    for item in official_coco_images(args.annotations):
        image_id = int(item["id"])
        relpath = item["file_name"]
        source = clean_root / relpath
        output_relpath = f"{args.corruption}/s{args.severity}/{Path(relpath).with_suffix('.jpg')}"
        destination = cache_root / output_relpath
        seed = stable_realization_seed(
            args.dataset, image_id, args.corruption, args.realization_seed
        )
        with Image.open(source) as image:
            result = transform(image, args.corruption, parameters, seed)
        encoded = io.BytesIO()
        result.save(
            encoded,
            format="JPEG",
            quality=int(config["output_encoding"]["quality"]),
            subsampling=int(config["output_encoding"]["subsampling"]),
        )
        expected = encoded.getvalue()
        expected_sha = hashlib.sha256(expected).hexdigest()
        if destination.exists():
            if not args.resume_validated or sha256_file(destination) != expected_sha:
                raise SystemExit(
                    f"REALIZATION CORRUPTION REFUSED: invalid existing cache: {destination}"
                )
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(
                dir=destination.parent, suffix=".tmp", delete=False
            ) as handle:
                temporary = Path(handle.name)
                handle.write(expected)
            try:
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        records.append(
            {
                "dataset": args.dataset,
                "image_id": image_id,
                "source_relpath": relpath,
                "source_sha256": sha256_file(source),
                "corruption": args.corruption,
                "severity": args.severity,
                "seed": seed,
                "realization_seed": args.realization_seed,
                "nested_across_severity": True,
                "output_relpath": output_relpath,
                "sha256": sha256_file(destination),
                "generator": config["generator"],
                "generator_version": config["generator_version"],
                "generator_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
            }
        )
    document = {
        "schema_version": 1,
        "dataset": args.dataset,
        "split": args.split,
        "realization_seed": args.realization_seed,
        "nested_across_severity": True,
        "expected_image_ids": [record["image_id"] for record in records],
        "records": records,
    }
    document["manifest_sha256"] = manifest_sha256(document)
    failures = validate_manifest(
        document,
        args.annotations,
        cache_root,
        clean_root,
        require_pixels_changed=True,
    )
    if failures:
        raise SystemExit("REALIZATION CORRUPTION REFUSED:\n" + "\n".join(failures[:20]))
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    manifest_path.with_suffix(manifest_path.suffix + ".complete").write_text(
        document["manifest_sha256"] + "\n", encoding="utf-8"
    )
    print(
        f"REALIZATION MANIFEST VALID records={len(records)} sha256={document['manifest_sha256']}"
    )


if __name__ == "__main__":
    main()
