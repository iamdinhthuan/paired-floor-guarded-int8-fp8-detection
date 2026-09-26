#!/usr/bin/env python3
"""Build a codec-matched clean calibration set + manifest (NN Phase E control).

Companion control to build_corrupt_calibration.py. The corruptcalib arm
re-encodes every calibration image at the frozen output_encoding
(JPEG q95, subsampling 0), while the Phase-B matched calibration manifests
point at original bytes (PNG for KITTI, original JPEG for VOC). This arm
applies *only* the codec -- no corruption -- to the same frozen 512-image
lists, isolating JPEG-codec adaptation from corruption coverage.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

import sys

sys.path.insert(0, str(Path(__file__).resolve().parent))
from topic_c.manifest import sha256_file  # noqa: E402


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-manifest", required=True,
                        help="frozen clean calibration manifest (records: source_relpath+sha256)")
    parser.add_argument("--config", required=True, help="frozen corruptions.json (output_encoding)")
    parser.add_argument("--schedule-id", required=True)
    parser.add_argument("--out-root", required=True, help="directory for re-encoded images")
    parser.add_argument("--manifest-out", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_out)
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite manifest: {manifest_path}")
    out_root = Path(args.out_root)
    config_bytes = Path(args.config).read_bytes()
    config = json.loads(config_bytes)
    source = json.loads(Path(args.calibration_manifest).read_bytes())
    source_sha256 = hashlib.sha256(Path(args.calibration_manifest).read_bytes()).hexdigest()
    clean_root = Path(source["dataset_root"]).resolve()
    encoding = config["output_encoding"]

    records = []
    for record in source["records"]:
        relpath = record["source_relpath"]
        clean_path = clean_root / relpath
        if sha256_file(clean_path) != record["sha256"]:
            raise SystemExit(f"calibration source hash mismatch: {clean_path}")
        with Image.open(clean_path) as image:
            decoded = image.convert("RGB")
        output_relpath = f"q95/{Path(relpath).with_suffix('.jpg')}"
        destination = out_root / output_relpath
        encoded = io.BytesIO()
        decoded.save(encoded, format=encoding["format"],
                     quality=int(encoding["quality"]), subsampling=int(encoding["subsampling"]))
        expected_bytes = encoded.getvalue()
        if destination.exists():
            if sha256_file(destination) != hashlib.sha256(expected_bytes).hexdigest():
                raise SystemExit(f"existing cache bytes fail validation: {destination}")
        else:
            destination.parent.mkdir(parents=True, exist_ok=True)
            with tempfile.NamedTemporaryFile(dir=destination.parent, suffix=".tmp", delete=False) as handle:
                handle.write(expected_bytes)
                temporary = Path(handle.name)
            try:
                os.replace(temporary, destination)
            finally:
                temporary.unlink(missing_ok=True)
        records.append({
            "source_relpath": output_relpath,
            "sha256": sha256_file(destination),
            "source_relpath_clean": relpath,
            "source_sha256": record["sha256"],
            "corruption": "jpeg95_encode",
            "severity": 0,
            "seed": 0,
        })

    document = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": source["dataset"],
        "split": source["split"],
        "selection": "codec_matched_clean_calibration_v1",
        "schedule_id": args.schedule_id,
        "in_family_corruptions": [],
        "held_out_corruptions": [],
        "severities": [],
        "n_images": len(records),
        "seed": source["seed"],
        "dataset_root": str(out_root.resolve()),
        "source_calibration_manifest": str(Path(args.calibration_manifest).resolve()),
        "source_calibration_manifest_sha256": source_sha256,
        "generator": config["generator"],
        "generator_version": config["generator_version"],
        "generator_config_sha256": hashlib.sha256(config_bytes).hexdigest(),
        "records": records,
    }
    for record in records:
        if sha256_file(out_root / record["source_relpath"]) != record["sha256"]:
            raise SystemExit(f"post-write hash validation failed: {record['source_relpath']}")
    payload = {key: value for key, value in document.items() if key != "calibration_sha256"}
    document["calibration_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    marker = manifest_path.with_suffix(manifest_path.suffix + ".complete")
    marker.write_text(document["calibration_sha256"] + "\n", encoding="utf-8")
    print(json.dumps({"manifest": str(manifest_path), "n_images": len(records)}, indent=2))


if __name__ == "__main__":
    main()
