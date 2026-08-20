#!/usr/bin/env python3
"""Freeze prospectively locked VOC/KITTI confirmatory partitions.

The generated final partitions are excluded from the *new* training,
checkpoint-selection, calibration, and decoder-tuning workflow.  They are not
described as historically untouched: all source data already existed in the
project before this confirmatory resplit was prespecified.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
from pathlib import Path
from typing import Iterable

import yaml


IMAGE_SUFFIXES = {".bmp", ".jpeg", ".jpg", ".png", ".tif", ".tiff", ".webp"}


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(document: dict) -> str:
    return hashlib.sha256(
        json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def discover_images(directory: Path) -> list[Path]:
    if not directory.is_dir():
        raise SystemExit(f"CONFIRMATORY SPLIT REFUSED: image directory missing: {directory}")
    images = sorted(
        (path.resolve() for path in directory.iterdir() if path.is_file() and path.suffix.lower() in IMAGE_SUFFIXES),
        key=str,
    )
    if not images:
        raise SystemExit(f"CONFIRMATORY SPLIT REFUSED: image directory empty: {directory}")
    return images


def label_path(dataset_root: Path, image: Path) -> Path:
    relative = image.resolve().relative_to((dataset_root / "images").resolve())
    return (dataset_root / "labels" / relative).with_suffix(".txt").resolve()


def image_classes(dataset_root: Path, image: Path, known_classes: set[int]) -> tuple[set[int], int]:
    label = label_path(dataset_root, image)
    if not label.is_file():
        raise SystemExit(f"CONFIRMATORY SPLIT REFUSED: label missing: {label}")
    classes: set[int] = set()
    objects = 0
    for line_number, row in enumerate(label.read_text(encoding="utf-8").splitlines(), start=1):
        fields = row.split()
        if len(fields) != 5:
            raise SystemExit(f"CONFIRMATORY SPLIT REFUSED: invalid label row: {label}:{line_number}")
        try:
            class_id = int(fields[0])
        except ValueError as error:
            raise SystemExit(f"CONFIRMATORY SPLIT REFUSED: invalid class id: {label}:{line_number}") from error
        if class_id not in known_classes:
            raise SystemExit(f"CONFIRMATORY SPLIT REFUSED: unknown class {class_id}: {label}:{line_number}")
        classes.add(class_id)
        objects += 1
    return classes, objects


def validate_partitions(partitions: dict[str, list[Path]]) -> None:
    expected = {"train", "selection", "final"}
    if set(partitions) != expected:
        raise SystemExit("CONFIRMATORY SPLIT REFUSED: expected train/selection/final partitions")
    resolved = {name: [path.resolve() for path in paths] for name, paths in partitions.items()}
    for name, paths in resolved.items():
        if not paths or len(paths) != len(set(paths)):
            raise SystemExit(f"CONFIRMATORY SPLIT REFUSED: empty or duplicate {name} partition")
    for left, right in (("train", "selection"), ("train", "final"), ("selection", "final")):
        if set(resolved[left]) & set(resolved[right]):
            raise SystemExit(f"CONFIRMATORY SPLIT REFUSED: partition overlap: {left}/{right}")


def deterministic_order(paths: Iterable[Path], dataset_root: Path, seed: int) -> list[Path]:
    def key(path: Path) -> tuple[str, str]:
        relative = path.resolve().relative_to(dataset_root.resolve()).as_posix()
        digest = hashlib.sha256(f"{seed}:{relative}".encode()).hexdigest()
        return digest, relative

    return sorted(paths, key=key)


def select_kitti_final(
    images: list[Path], dataset_root: Path, known_classes: set[int], seed: int, final_fraction: float
) -> tuple[list[Path], list[Path]]:
    if not 0.0 < final_fraction < 0.5:
        raise SystemExit("CONFIRMATORY SPLIT REFUSED: KITTI final fraction must be in (0, 0.5)")
    classes_by_image = {image: image_classes(dataset_root, image, known_classes)[0] for image in images}
    class_images = {class_id: [image for image in images if class_id in classes_by_image[image]] for class_id in known_classes}
    missing = [class_id for class_id, members in class_images.items() if len(members) < 2]
    if missing:
        raise SystemExit(f"CONFIRMATORY SPLIT REFUSED: classes cannot cover both KITTI train/final: {missing}")
    target = max(len(known_classes), math.ceil(len(images) * final_fraction))
    if target >= len(images):
        raise SystemExit("CONFIRMATORY SPLIT REFUSED: KITTI final partition would exhaust training")

    final: list[Path] = []
    selected: set[Path] = set()
    remaining_counts = {class_id: len(members) for class_id, members in class_images.items()}
    uncovered = set(known_classes)
    hash_order = deterministic_order(images, dataset_root, seed)
    hash_rank = {path: ordinal for ordinal, path in enumerate(hash_order)}

    while uncovered:
        class_id = min(uncovered, key=lambda item: (len(class_images[item]), item))
        candidates = [
            image for image in class_images[class_id]
            if image not in selected and all(remaining_counts[item] > 1 for item in classes_by_image[image])
        ]
        if not candidates:
            raise SystemExit(f"CONFIRMATORY SPLIT REFUSED: cannot preserve training coverage for class {class_id}")
        chosen = min(candidates, key=lambda image: (-len(classes_by_image[image] & uncovered), hash_rank[image]))
        selected.add(chosen)
        final.append(chosen)
        for item in classes_by_image[chosen]:
            remaining_counts[item] -= 1
        uncovered -= classes_by_image[chosen]

    for candidate in hash_order:
        if len(final) >= target:
            break
        if candidate in selected or any(remaining_counts[item] <= 1 for item in classes_by_image[candidate]):
            continue
        selected.add(candidate)
        final.append(candidate)
        for item in classes_by_image[candidate]:
            remaining_counts[item] -= 1
    if len(final) != target:
        raise SystemExit("CONFIRMATORY SPLIT REFUSED: unable to fill KITTI final partition safely")
    train = [image for image in images if image not in selected]
    return sorted(train, key=str), sorted(final, key=str)


def summarize_split(dataset_root: Path, images: list[Path], known_classes: set[int], list_path: Path) -> dict:
    class_image_counts = {class_id: 0 for class_id in known_classes}
    class_object_counts = {class_id: 0 for class_id in known_classes}
    objects = 0
    for image in images:
        classes, image_objects = image_classes(dataset_root, image, known_classes)
        objects += image_objects
        for class_id in classes:
            class_image_counts[class_id] += 1
        label = label_path(dataset_root, image)
        for row in label.read_text(encoding="utf-8").splitlines():
            class_object_counts[int(row.split()[0])] += 1
    absent = [class_id for class_id, count in class_image_counts.items() if count == 0]
    if absent:
        raise SystemExit(f"CONFIRMATORY SPLIT REFUSED: split lacks class coverage: {absent}")
    return {
        "images": len(images),
        "objects": objects,
        "class_image_counts": {str(key): value for key, value in sorted(class_image_counts.items())},
        "class_object_counts": {str(key): value for key, value in sorted(class_object_counts.items())},
        "list_path": str(list_path.resolve()),
        "list_sha256": sha256_file(list_path),
    }


def write_yaml(path: Path, dataset_root: Path, lists: dict[str, Path], names: dict[int, str]) -> None:
    document = {
        "path": str(dataset_root.resolve()),
        "train": str(lists["train"].resolve()),
        "val": str(lists["selection"].resolve()),
        "test": str(lists["final"].resolve()),
        "names": {int(key): value for key, value in sorted(names.items())},
    }
    path.write_text(yaml.safe_dump(document, sort_keys=False), encoding="utf-8")


def semantic_content(report: dict) -> dict:
    return {
        "schema_version": report["schema_version"],
        "dataset": report["dataset"],
        "protocol": report["protocol"],
        "seed": report["seed"],
        "final_fraction": report["final_fraction"],
        "dataset_root": report["dataset_root"],
        "acquisition_registry_sha256": report["acquisition_registry_sha256"],
        "partition_sources": report["partition_sources"],
        "names": report["names"],
        "splits": {
            name: {key: value for key, value in split.items() if key != "list_path"}
            for name, split in report["splits"].items()
        },
    }


def freeze_confirmatory_split(
    *,
    dataset: str,
    dataset_root: Path,
    names: dict[int, str],
    acquisition_registry: Path,
    output_root: Path,
    seed: int,
    final_fraction: float = 0.2,
) -> dict:
    dataset_root = dataset_root.resolve()
    acquisition_registry = acquisition_registry.resolve()
    output_root = output_root.resolve()
    if dataset not in {"voc", "kitti"}:
        raise SystemExit("CONFIRMATORY SPLIT REFUSED: dataset must be voc or kitti")
    if not acquisition_registry.is_file():
        raise SystemExit("CONFIRMATORY SPLIT REFUSED: acquisition registry missing")
    acquired = json.loads(acquisition_registry.read_text(encoding="utf-8"))
    if acquired.get("dataset") != dataset or Path(acquired.get("dataset_root", "")).resolve() != dataset_root:
        raise SystemExit("CONFIRMATORY SPLIT REFUSED: acquisition registry identity mismatch")
    if output_root.exists() and any(output_root.iterdir()):
        raise SystemExit("CONFIRMATORY SPLIT REFUSED: output root is not empty")
    output_root.mkdir(parents=True, exist_ok=True)
    lists_dir = output_root / "lists"
    lists_dir.mkdir()
    known_classes = set(names)
    if known_classes != set(range(len(known_classes))):
        raise SystemExit("CONFIRMATORY SPLIT REFUSED: class ids must be contiguous from zero")

    if dataset == "voc":
        sources = {
            "train": ["images/train2007", "images/train2012"],
            "selection": ["images/val2007"],
            "final": ["images/val2012"],
        }
        partitions = {
            name: sorted(
                [image for relative in relatives for image in discover_images(dataset_root / relative)], key=str
            )
            for name, relatives in sources.items()
        }
        effective_fraction = None
    else:
        sources = {
            "train": ["images/train (remainder after deterministic final selection)"],
            "selection": ["images/val"],
            "final": ["images/train (deterministic class-covered subset)"],
        }
        candidates = discover_images(dataset_root / "images" / "train")
        train, final = select_kitti_final(candidates, dataset_root, known_classes, seed, final_fraction)
        partitions = {
            "train": train,
            "selection": discover_images(dataset_root / "images" / "val"),
            "final": final,
        }
        effective_fraction = final_fraction
    validate_partitions(partitions)

    list_paths: dict[str, Path] = {}
    for name, images in partitions.items():
        path = lists_dir / f"{dataset}_confirmatory_v1_{name}.txt"
        path.write_text("".join(f"{image.resolve()}\n" for image in images), encoding="utf-8")
        list_paths[name] = path
    yaml_path = output_root / f"{dataset}_confirmatory_v1.yaml"
    write_yaml(yaml_path, dataset_root, list_paths, names)
    splits = {
        name: summarize_split(dataset_root, images, known_classes, list_paths[name])
        for name, images in partitions.items()
    }
    report_path = output_root / f"{dataset}_confirmatory_v1.json"
    report = {
        "schema_version": 1,
        "dataset": dataset,
        "protocol": "prospectively_locked_confirmatory_resplit",
        "scope_note": (
            "The final split is excluded from the new training, selection, calibration, and tuning workflow; "
            "it is not claimed to have been historically unseen before this prespecification."
        ),
        "seed": seed,
        "final_fraction": effective_fraction,
        "dataset_root": str(dataset_root),
        "acquisition_registry": str(acquisition_registry),
        "acquisition_registry_sha256": sha256_file(acquisition_registry),
        "partition_sources": sources,
        "names": {str(key): value for key, value in sorted(names.items())},
        "yaml_path": str(yaml_path),
        "yaml_sha256": sha256_file(yaml_path),
        "splits": splits,
        "report_path": str(report_path),
    }
    report["content_sha256"] = canonical_hash(semantic_content(report))
    report_path.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report_path.with_suffix(report_path.suffix + ".complete").write_text(sha256_file(report_path) + "\n", encoding="utf-8")
    return report


def validate_split_bundle(report_path: Path) -> dict:
    report_path = report_path.resolve()
    marker = report_path.with_suffix(report_path.suffix + ".complete")
    if not report_path.is_file() or not marker.is_file() or marker.read_text(encoding="utf-8").strip() != sha256_file(report_path):
        raise SystemExit("CONFIRMATORY SPLIT REFUSED: report completion hash mismatch")
    report = json.loads(report_path.read_text(encoding="utf-8"))
    if Path(report.get("report_path", "")).resolve() != report_path:
        raise SystemExit("CONFIRMATORY SPLIT REFUSED: report path mismatch")
    acquisition = Path(report["acquisition_registry"])
    if not acquisition.is_file() or sha256_file(acquisition) != report["acquisition_registry_sha256"]:
        raise SystemExit("CONFIRMATORY SPLIT REFUSED: acquisition registry hash mismatch")
    yaml_path = Path(report["yaml_path"])
    if not yaml_path.is_file() or sha256_file(yaml_path) != report["yaml_sha256"]:
        raise SystemExit("CONFIRMATORY SPLIT REFUSED: YAML hash mismatch")
    partitions: dict[str, list[Path]] = {}
    for name, split in report["splits"].items():
        list_path = Path(split["list_path"])
        if not list_path.is_file() or sha256_file(list_path) != split["list_sha256"]:
            raise SystemExit(f"CONFIRMATORY SPLIT REFUSED: list hash mismatch: {name}")
        partitions[name] = [Path(line) for line in list_path.read_text(encoding="utf-8").splitlines() if line]
        if len(partitions[name]) != split["images"]:
            raise SystemExit(f"CONFIRMATORY SPLIT REFUSED: list count mismatch: {name}")
    validate_partitions(partitions)
    if canonical_hash(semantic_content(report)) != report["content_sha256"]:
        raise SystemExit("CONFIRMATORY SPLIT REFUSED: semantic content hash mismatch")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("voc", "kitti"))
    parser.add_argument("--dataset-root")
    parser.add_argument("--source-yaml")
    parser.add_argument("--acquisition-registry")
    parser.add_argument("--output-root")
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--final-fraction", type=float, default=0.2)
    parser.add_argument("--validate-only")
    args = parser.parse_args()
    if args.validate_only:
        print(json.dumps(validate_split_bundle(Path(args.validate_only)), indent=2))
        return
    required = (args.dataset, args.dataset_root, args.source_yaml, args.acquisition_registry, args.output_root)
    if any(value is None for value in required):
        parser.error("generation requires --dataset, --dataset-root, --source-yaml, --acquisition-registry, and --output-root")
    source = yaml.safe_load(Path(args.source_yaml).read_text(encoding="utf-8"))
    names = {int(key): str(value) for key, value in source["names"].items()}
    report = freeze_confirmatory_split(
        dataset=args.dataset,
        dataset_root=Path(args.dataset_root),
        names=names,
        acquisition_registry=Path(args.acquisition_registry),
        output_root=Path(args.output_root),
        seed=args.seed,
        final_fraction=args.final_fraction,
    )
    print(json.dumps({"dataset": args.dataset, "content_sha256": report["content_sha256"], "splits": report["splits"]}, indent=2))


if __name__ == "__main__":
    main()
