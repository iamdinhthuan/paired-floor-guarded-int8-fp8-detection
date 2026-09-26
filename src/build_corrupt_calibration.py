#!/usr/bin/env python3
"""Build a corruption-aware calibration image set + manifest (NN Phase C/D).

Each image of a frozen clean calibration manifest is deterministically
assigned one in-family (corruption, severity) cell via sha256 of
``{schedule_id}:{dataset}:{source_relpath}``, transformed with the same
``transform``/frozen ``corruptions.json`` parameters as the evaluation
caches, JPEG-encoded with the same output_encoding, and emitted as a
calibration-schema manifest consumable by quantize_yolo_onnx.py.

Held-out corruptions are never assigned; the schedule id and the exact
in-family/held-out split are recorded in the manifest for audit.
"""
from __future__ import annotations

import argparse
import hashlib
import io
import json
import os
import sys
import tempfile
from datetime import datetime, timezone
from pathlib import Path

from PIL import Image

sys.path.insert(0, str(Path(__file__).resolve().parent))
from generate_corruption import transform  # noqa: E402
from topic_c.manifest import sha256_file  # noqa: E402


def calib_seed(dataset: str, source_relpath: str, corruption: str, severity: int) -> int:
    text = f"{dataset}:{source_relpath}:{corruption}:{severity}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(text).digest()[:8], "big")


def assign_condition(schedule_id: str, dataset: str, source_relpath: str,
                     corruptions: list[str], severities: list[int]) -> tuple[str, int]:
    text = f"{schedule_id}:{dataset}:{source_relpath}".encode("utf-8")
    index = int.from_bytes(hashlib.sha256(text).digest()[:8], "big")
    cells = [(c, s) for c in corruptions for s in severities]
    return cells[index % len(cells)]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--calibration-manifest", required=True,
                        help="frozen clean calibration manifest (records: source_relpath+sha256)")
    parser.add_argument("--config", required=True, help="frozen corruptions.json")
    parser.add_argument("--schedule-id", required=True)
    parser.add_argument("--in-family", required=True, help="comma list, e.g. gaussian_noise,jpeg")
    parser.add_argument("--severities", default="1,3,5")
    parser.add_argument("--held-out", required=True, help="comma list recorded for audit")
    parser.add_argument("--out-root", required=True, help="directory for corrupted images")
    parser.add_argument("--manifest-out", required=True)
    args = parser.parse_args()

    manifest_path = Path(args.manifest_out)
    if manifest_path.exists():
        raise SystemExit(f"refusing to overwrite manifest: {manifest_path}")
    out_root = Path(args.out_root)
    corruptions = [c.strip() for c in args.in_family.split(",") if c.strip()]
    held_out = [c.strip() for c in args.held_out.split(",") if c.strip()]
    severities = sorted(int(s) for s in args.severities.split(","))
    overlap = set(corruptions) & set(held_out)
    if overlap:
        raise SystemExit(f"in-family/held-out overlap refused: {sorted(overlap)}")

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
        corruption, severity = assign_condition(
            args.schedule_id, source["dataset"], relpath, corruptions, severities)
        parameters = config["corruptions"][corruption]["severity"][str(severity)]
        seed = calib_seed(source["dataset"], relpath, corruption, severity)
        with Image.open(clean_path) as image:
            result = transform(image, corruption, parameters, seed)
        output_relpath = f"{corruption}/s{severity}/{Path(relpath).with_suffix('.jpg')}"
        destination = out_root / output_relpath
        encoded = io.BytesIO()
        result.save(encoded, format=encoding["format"],
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
            "corruption": corruption,
            "severity": severity,
            "seed": seed,
        })

    document = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": source["dataset"],
        "split": source["split"],
        "selection": "deterministic_in_family_corruption_assignment_v1",
        "schedule_id": args.schedule_id,
        "in_family_corruptions": corruptions,
        "held_out_corruptions": held_out,
        "severities": severities,
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
    # canonical calibration_sha256 contract required by quantize_yolo_onnx.load_complete
    payload = {key: value for key, value in document.items() if key != "calibration_sha256"}
    document["calibration_sha256"] = hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()
    manifest_path.parent.mkdir(parents=True, exist_ok=True)
    manifest_path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    marker = manifest_path.with_suffix(manifest_path.suffix + ".complete")
    marker.write_text(document["calibration_sha256"] + "\n", encoding="utf-8")
    counts: dict[str, int] = {}
    for record in records:
        key = f"{record['corruption']}-s{record['severity']}"
        counts[key] = counts.get(key, 0) + 1
    print(json.dumps({"manifest": str(manifest_path), "n_images": len(records),
                      "cells": counts}, indent=2))


if __name__ == "__main__":
    main()
