"""Canonical corruption-manifest reading and validation."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path
from typing import Any, Iterable

REQUIRED_RECORD_FIELDS = {"dataset", "image_id", "source_relpath", "source_sha256", "corruption", "severity", "seed", "output_relpath", "sha256", "generator", "generator_version"}


class ManifestError(ValueError):
    pass


def sha256_file(path: str | Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_payload(document: dict[str, Any]) -> bytes:
    payload = {key: value for key, value in document.items() if key != "manifest_sha256"}
    return json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")


def manifest_sha256(document: dict[str, Any]) -> str:
    return hashlib.sha256(canonical_payload(document)).hexdigest()


def read_manifest(path: str | Path) -> dict[str, Any]:
    with Path(path).open(encoding="utf-8") as handle:
        document = json.load(handle)
    if not isinstance(document, dict) or not isinstance(document.get("records"), list):
        raise ManifestError("manifest must be a JSON object with a records list")
    actual = manifest_sha256(document)
    if document.get("manifest_sha256") != actual:
        raise ManifestError(f"manifest SHA-256 mismatch: calculated {actual}")
    return document


def official_coco_images(annotation_path: str | Path) -> list[dict[str, Any]]:
    with Path(annotation_path).open(encoding="utf-8") as handle:
        images = json.load(handle)["images"]
    if len({image["id"] for image in images}) != len(images):
        raise ManifestError("COCO annotations contain duplicate image IDs")
    return sorted(images, key=lambda image: image["id"])


def under_root(root: str | Path, relative_path: str) -> Path:
    root = Path(root).resolve()
    candidate = (root / relative_path).resolve()
    try:
        candidate.relative_to(root)
    except ValueError as exc:
        raise ManifestError(f"path escapes root: {relative_path}") from exc
    return candidate


def validate_manifest(document: dict[str, Any], annotation_path: str | Path, cache_root: str | Path | None, clean_root: str | Path | None, require_pixels_changed: bool = True) -> list[str]:
    """Return all failures. Empty result is the manifest admission gate."""
    failures: list[str] = []
    if document.get("manifest_sha256") != manifest_sha256(document):
        return ["manifest_sha256 does not cover canonical payload"]
    official = {int(item["id"]): item["file_name"] for item in official_coco_images(annotation_path)}
    ids, paths = [], set()
    for index, record in enumerate(document["records"]):
        missing = REQUIRED_RECORD_FIELDS - set(record)
        if missing:
            failures.append(f"record {index}: missing {sorted(missing)}")
            continue
        image_id = record["image_id"]
        if not isinstance(image_id, int) or image_id not in official:
            failures.append(f"record {index}: invalid COCO image_id {image_id!r}")
            continue
        ids.append(image_id)
        if record["source_relpath"] != official[image_id]:
            failures.append(f"record {index}: source_relpath differs from COCO annotation")
        if clean_root is not None:
            try:
                source = under_root(clean_root, record["source_relpath"])
                if not source.is_file():
                    failures.append(f"record {index}: clean source file absent")
                elif sha256_file(source) != record["source_sha256"]:
                    failures.append(f"record {index}: clean source SHA-256 mismatch")
            except ManifestError as exc:
                failures.append(f"record {index}: {exc}")
        output_relpath = record["output_relpath"]
        if not isinstance(output_relpath, str) or output_relpath in paths:
            failures.append(f"record {index}: duplicate/invalid output_relpath")
        paths.add(output_relpath)
        if cache_root is not None:
            try:
                output = under_root(cache_root, output_relpath)
                if not output.is_file():
                    failures.append(f"record {index}: cached file absent")
                elif sha256_file(output) != record["sha256"]:
                    failures.append(f"record {index}: cached SHA-256 mismatch")
            except ManifestError as exc:
                failures.append(f"record {index}: {exc}")
    if len(ids) != len(set(ids)):
        failures.append("duplicate image_id values")
    expected = document.get("expected_image_ids")
    if not isinstance(expected, list) or any(not isinstance(item, int) for item in expected):
        failures.append("expected_image_ids must be a list of integers")
    elif ids != expected:
        failures.append("records do not exactly preserve expected image IDs and order")
    if require_pixels_changed and any(item.get("corruption") != "clean" for item in document["records"]):
        if cache_root is None or clean_root is None:
            failures.append("pixel validation requires cache and clean roots")
        else:
            try:
                import numpy as np
                from PIL import Image
            except ImportError:
                failures.append("Pillow and numpy are required for pixel validation")
            else:
                for index, record in enumerate(document["records"]):
                    if record.get("corruption") == "clean":
                        continue
                    try:
                        with Image.open(under_root(clean_root, record["source_relpath"])) as image:
                            clean = np.asarray(image.convert("RGB"))
                        with Image.open(under_root(cache_root, record["output_relpath"])) as image:
                            corrupt = np.asarray(image.convert("RGB"))
                        if clean.shape != corrupt.shape:
                            failures.append(f"record {index}: image dimensions changed")
                        elif np.array_equal(clean, corrupt):
                            failures.append(f"record {index}: pixels equal clean image")
                    except (OSError, ManifestError) as exc:
                        failures.append(f"record {index}: pixel check failed: {exc}")
    return failures


def ordered_manifest_images(manifest_path: str | Path, cache_root: str | Path) -> list[tuple[int, str]]:
    document = read_manifest(manifest_path)
    return [(int(record["image_id"]), str(under_root(cache_root, record["output_relpath"]))) for record in document["records"]]


def validate_shared_input_bytes(run_records: Iterable[dict[str, Any]]) -> list[str]:
    """Every precision of one dataset/split/corruption/severity must cite one manifest hash."""
    groups: dict[tuple[Any, ...], set[str]] = {}
    for index, record in enumerate(run_records):
        fields = ("dataset", "split", "corruption", "severity", "input_manifest_sha256")
        if any(field not in record or record[field] is None for field in fields) or not record["input_manifest_sha256"]:
            return [f"run record {index}: missing required input-manifest provenance"]
        groups.setdefault(tuple(record[field] for field in fields[:-1]), set()).add(record["input_manifest_sha256"])
    return [f"{group}: multiple input manifests across precision formats" for group, hashes in groups.items() if len(hashes) != 1]
