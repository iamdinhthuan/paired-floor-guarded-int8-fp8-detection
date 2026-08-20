"""Shared fail-closed consumers for prospectively locked dataset bundles."""
from __future__ import annotations

import json
from pathlib import Path

from topic_c.manifest import sha256_file


IMAGE_SUFFIXES = {".jpg", ".jpeg", ".png", ".bmp", ".tif", ".tiff", ".webp"}


def split_images(root: Path, configured_split: object, *, prefix: str) -> list[Path]:
    root = root.resolve()
    values = configured_split if isinstance(configured_split, list) else [configured_split]
    images: list[Path] = []
    for configured in values:
        candidate = Path(str(configured))
        candidate = candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()
        if candidate.is_dir():
            images.extend(
                path.resolve() for path in candidate.iterdir()
                if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES
            )
        elif candidate.is_file():
            for line_number, line in enumerate(candidate.read_text(encoding="utf-8").splitlines(), start=1):
                if not line.strip():
                    continue
                image = Path(line.strip())
                image = image.resolve() if image.is_absolute() else (candidate.parent / image).resolve()
                if not image.is_file() or image.suffix.lower() not in IMAGE_SUFFIXES:
                    raise SystemExit(f"{prefix} REFUSED: invalid image-list entry {candidate}:{line_number}")
                images.append(image)
        else:
            raise SystemExit(f"{prefix} REFUSED: split source missing: {candidate}")
    images = sorted(images, key=lambda path: path.relative_to(root).as_posix())
    if not images or len(images) != len(set(images)):
        raise SystemExit(f"{prefix} REFUSED: empty or duplicate split images")
    if any(root not in image.parents for image in images):
        raise SystemExit(f"{prefix} REFUSED: image escapes dataset root")
    return images


def validate_dataset_provenance(
    *, dataset: str, data_yaml: Path, acquisition_registry: Path, split_registry: Path | None
) -> dict:
    acquisition_registry = acquisition_registry.resolve()
    data_yaml = data_yaml.resolve()
    marker = acquisition_registry.with_suffix(acquisition_registry.suffix + ".complete")
    if (
        not acquisition_registry.is_file()
        or not marker.is_file()
        or marker.read_text(encoding="utf-8").strip() != sha256_file(acquisition_registry)
    ):
        raise SystemExit("DATASET PROVENANCE REFUSED: acquisition registry is incomplete")
    acquired = json.loads(acquisition_registry.read_text(encoding="utf-8"))
    if acquired.get("dataset") != dataset:
        raise SystemExit("DATASET PROVENANCE REFUSED: acquisition dataset mismatch")
    if split_registry is None:
        if (
            Path(acquired.get("vendored_yaml", "")).resolve() != data_yaml
            or acquired.get("vendored_yaml_sha256") != sha256_file(data_yaml)
        ):
            raise SystemExit("DATASET PROVENANCE REFUSED: historical YAML binding mismatch")
        return {"mode": "historical_acquisition_yaml", "dataset_root": acquired["dataset_root"]}

    from freeze_confirmatory_splits import validate_split_bundle

    split_registry = split_registry.resolve()
    split = validate_split_bundle(split_registry)
    if (
        split.get("dataset") != dataset
        or Path(split.get("yaml_path", "")).resolve() != data_yaml
        or split.get("yaml_sha256") != sha256_file(data_yaml)
        or Path(split.get("acquisition_registry", "")).resolve() != acquisition_registry
        or split.get("acquisition_registry_sha256") != sha256_file(acquisition_registry)
    ):
        raise SystemExit("DATASET PROVENANCE REFUSED: confirmatory split binding mismatch")
    return {
        "mode": split["protocol"],
        "dataset_root": split["dataset_root"],
        "split_registry": str(split_registry),
        "split_registry_sha256": sha256_file(split_registry),
        "split_content_sha256": split["content_sha256"],
    }
