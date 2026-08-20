#!/usr/bin/env python3
"""Materialize a deterministic JPEG-95 clean control for matched-codec analysis."""

from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from pathlib import Path

from PIL import Image

from topic_c.manifest import (
    manifest_sha256,
    official_coco_images,
    sha256_file,
    validate_manifest,
)


def encode_jpeg(image: Image.Image, quality: int = 95, subsampling: int = 0) -> bytes:
    """Return deterministic JPEG bytes for the matched clean treatment."""
    encoded = io.BytesIO()
    image.convert("RGB").save(
        encoded,
        format="JPEG",
        quality=quality,
        subsampling=subsampling,
    )
    return encoded.getvalue()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--quality", type=int, default=95)
    parser.add_argument("--subsampling", type=int, default=0)
    parser.add_argument("--resume-validated", action="store_true")
    args = parser.parse_args()

    if not 1 <= args.quality <= 100:
        raise SystemExit("CODEC CONTROL REFUSED: quality must be in [1, 100]")
    annotations = Path(args.annotations).resolve()
    clean_root = Path(args.clean_root).resolve()
    cache_root = Path(args.cache_root).resolve()
    manifest_out = Path(args.manifest_out).resolve()
    marker = manifest_out.with_suffix(manifest_out.suffix + ".complete")
    if manifest_out.exists() or marker.exists():
        raise SystemExit(f"CODEC CONTROL REFUSED: manifest already exists: {manifest_out}")

    records = []
    changed_bytes = 0
    for item in official_coco_images(annotations):
        image_id, relpath = int(item["id"]), item["file_name"]
        source = clean_root / relpath
        if not source.is_file():
            raise SystemExit(f"CODEC CONTROL REFUSED: clean source missing: {source}")
        output_relpath = f"jpeg95_clean/{Path(relpath).with_suffix('.jpg').as_posix()}"
        destination = cache_root / output_relpath
        with Image.open(source) as image:
            expected = encode_jpeg(image, args.quality, args.subsampling)
        expected_hash = hashlib.sha256(expected).hexdigest()
        source_hash = sha256_file(source)
        changed_bytes += int(expected_hash != source_hash)
        if destination.exists():
            if not args.resume_validated:
                raise SystemExit(f"CODEC CONTROL REFUSED: cached file exists: {destination}")
            if sha256_file(destination) != expected_hash:
                raise SystemExit(f"CODEC CONTROL REFUSED: cached hash mismatch: {destination}")
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
                "source_sha256": source_hash,
                "corruption": "codec_control",
                "severity": 0,
                "seed": 0,
                "output_relpath": output_relpath,
                "sha256": expected_hash,
                "generator": "topic_c_matched_codec_control",
                "generator_version": "1.0.0",
                "encoding": {
                    "format": "JPEG",
                    "quality": args.quality,
                    "subsampling": args.subsampling,
                },
            }
        )

    document = {
        "schema_version": 1,
        "dataset": args.dataset,
        "split": args.split,
        "treatment": "clean decode followed by deterministic JPEG materialization",
        "encoding": {
            "format": "JPEG",
            "quality": args.quality,
            "subsampling": args.subsampling,
        },
        "expected_image_ids": [record["image_id"] for record in records],
        "records": records,
    }
    document["manifest_sha256"] = manifest_sha256(document)
    failures = validate_manifest(
        document,
        annotations,
        cache_root,
        clean_root,
        require_pixels_changed=False,
    )
    if failures:
        raise SystemExit("CODEC CONTROL REFUSED:\n" + "\n".join(failures[:20]))
    if changed_bytes != len(records):
        raise SystemExit(
            f"CODEC CONTROL REFUSED: only {changed_bytes}/{len(records)} files changed bytes"
        )
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    marker.write_text(document["manifest_sha256"] + "\n", encoding="utf-8")
    print(
        json.dumps(
            {
                "dataset": args.dataset,
                "images": len(records),
                "byte_changed": changed_bytes,
                "manifest": str(manifest_out),
                "manifest_sha256": document["manifest_sha256"],
            },
            indent=2,
        )
    )


if __name__ == "__main__":
    main()
