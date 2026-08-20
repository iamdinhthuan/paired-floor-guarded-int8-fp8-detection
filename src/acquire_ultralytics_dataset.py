#!/usr/bin/env python3
"""Acquire and validate a pinned Ultralytics detection dataset in Topic C space.

The script deliberately vendors the *installed* upstream dataset YAML before
asking Ultralytics to download.  This preserves the exact download recipe and
class mapping used for training while keeping all mutable data below the new
Topic C project root.  It never touches the legacy COCO or TT100K trees.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

import ultralytics
from ultralytics.data.utils import check_det_dataset
from ultralytics.utils import YAML
from ultralytics.utils.downloads import safe_download


DATASETS = {
    "voc": {
        "upstream_yaml": "VOC.yaml",
        "root_name": "VOC",
        "expected": {"train": 16551, "val": 4952},
    },
    "kitti": {
        "upstream_yaml": "kitti.yaml",
        "root_name": "kitti",
        "expected": {"train": 5985, "val": 1496},
    },
    "tt100k": {
        "upstream_yaml": "TT100K.yaml",
        "root_name": "TT100K",
        # The current official archive contains 16,811 source images, not
        # the 16,817 count stated in the upstream YAML comments/docs.  The
        # release's ``other`` (Ultralytics ``val``) split also includes images
        # without annotations, so absent labels there represent empty ground
        # truth rather than corruption of the download.
        "expected": {"train": 6103, "val": 7641, "test": 3067},
        "expected_label_files": {"train": 6103, "val": 1097, "test": 3067},
        "published_split_counts": {"train": 6105, "val": 7641, "test": 3071},
    },
}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def split_images(root: Path, split: object) -> list[Path]:
    values = split if isinstance(split, list) else [split]
    paths: list[Path] = []
    for value in values:
        directory = root / str(value)
        if not directory.is_dir():
            raise SystemExit(f"DATASET ACQUISITION REFUSED: missing split directory: {directory}")
        paths.extend(sorted(path for path in directory.iterdir() if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}))
    return paths


def label_for(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError as exc:
        raise SystemExit(f"DATASET ACQUISITION REFUSED: image is outside images/: {image}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def file_list_sha256(root: Path, images: list[Path], *, labels_required: bool) -> tuple[str, int]:
    digest = hashlib.sha256()
    labels_present = 0
    for image in images:
        label = label_for(image)
        if not label.is_file() and labels_required:
            raise SystemExit(f"DATASET ACQUISITION REFUSED: missing label: {label}")
        relative = image.relative_to(root).as_posix()
        digest.update(f"{relative}\t{image.stat().st_size}\t{sha256_file(image)}\n".encode("utf-8"))
        if label.is_file():
            labels_present += 1
            relative = label.relative_to(root).as_posix()
            digest.update(f"{relative}\t{label.stat().st_size}\t{sha256_file(label)}\n".encode("utf-8"))
        else:
            digest.update(f"{label.relative_to(root).as_posix()}\tEMPTY_LABEL\n".encode("utf-8"))
    return digest.hexdigest(), labels_present


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=tuple(DATASETS), required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--vendor-yaml", required=True)
    parser.add_argument("--registry-out", required=True)
    parser.add_argument("--download", action="store_true", help="perform the upstream download when data are absent")
    args = parser.parse_args()

    specification = DATASETS[args.dataset]
    project_root = Path(args.project_root).resolve()
    data_root = project_root / "data" / "datasets" / specification["root_name"]
    vendor_yaml, registry = Path(args.vendor_yaml).resolve(), Path(args.registry_out).resolve()
    if registry.exists():
        raise SystemExit(f"DATASET ACQUISITION REFUSED: registry already exists: {registry}")
    upstream = Path(ultralytics.__file__).resolve().parent / "cfg" / "datasets" / specification["upstream_yaml"]
    if not upstream.is_file():
        raise SystemExit(f"DATASET ACQUISITION REFUSED: installed upstream YAML missing: {upstream}")
    upstream_bytes = upstream.read_bytes()
    configuration = YAML.load(upstream)
    configuration["path"] = str(data_root)
    if vendor_yaml.exists():
        if YAML.load(vendor_yaml) != configuration:
            raise SystemExit(f"DATASET ACQUISITION REFUSED: existing vendored YAML differs: {vendor_yaml}")
    else:
        vendor_yaml.parent.mkdir(parents=True, exist_ok=True)
        YAML.save(vendor_yaml, configuration)

    download_recipe = configuration.get("download")
    # Ultralytics' URL-only helper always uses its global DATASETS_DIR.  That
    # location can point at a legacy experiment root, so route such archives
    # explicitly to this project's new data/datasets directory instead.
    if args.download and isinstance(download_recipe, str) and download_recipe.startswith("http") and download_recipe.endswith(".zip"):
        if not data_root.exists():
            safe_download(url=download_recipe, dir=data_root.parent, delete=True)
        checked = check_det_dataset(str(vendor_yaml), autodownload=False)
    elif args.download:
        checked = check_det_dataset(str(vendor_yaml), autodownload=True)
    else:
        checked = check_det_dataset(str(vendor_yaml), autodownload=False)
    resolved_root = Path(checked["path"]).resolve()
    if resolved_root != data_root:
        raise SystemExit(f"DATASET ACQUISITION REFUSED: resolved root differs: {resolved_root} != {data_root}")
    split_summary = {}
    aggregate = hashlib.sha256()
    for split_name, expected_count in specification["expected"].items():
        images = split_images(resolved_root, checked[split_name])
        if len(images) != expected_count:
            raise SystemExit(f"DATASET ACQUISITION REFUSED: {args.dataset} {split_name} count {len(images)} != {expected_count}")
        labels_required = args.dataset != "tt100k" or split_name != "val"
        file_hash, label_count = file_list_sha256(resolved_root, images, labels_required=labels_required)
        if (expected_labels := specification.get("expected_label_files", {}).get(split_name)) is not None and label_count != expected_labels:
            raise SystemExit(f"DATASET ACQUISITION REFUSED: {args.dataset} {split_name} label count {label_count} != {expected_labels}")
        split_summary[split_name] = {"images": len(images), "label_files": label_count,
                                     "empty_label_images": len(images) - label_count, "file_list_sha256": file_hash}
        aggregate.update(f"{split_name}\t{file_hash}\n".encode("utf-8"))
    document = {
        "schema_version": 1,
        "dataset": args.dataset,
        "acquired_at_utc": datetime.now(timezone.utc).isoformat(),
        "project_root": str(project_root),
        "dataset_root": str(resolved_root),
        "upstream_yaml": str(upstream),
        "upstream_yaml_sha256": hashlib.sha256(upstream_bytes).hexdigest(),
        "vendored_yaml": str(vendor_yaml),
        "vendored_yaml_sha256": sha256_file(vendor_yaml),
        "ultralytics_version": ultralytics.__version__,
        "class_names": checked["names"],
        "splits": split_summary,
        "published_split_counts": specification.get("published_split_counts"),
        "all_labeled_files_sha256": aggregate.hexdigest(),
    }
    registry.parent.mkdir(parents=True, exist_ok=True)
    registry.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    registry.with_suffix(registry.suffix + ".complete").write_text(sha256_file(registry) + "\n", encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "root": str(resolved_root), "splits": split_summary,
                      "registry": str(registry), "registry_sha256": sha256_file(registry)}, indent=2))


if __name__ == "__main__":
    main()
