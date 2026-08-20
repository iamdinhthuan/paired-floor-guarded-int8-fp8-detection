#!/usr/bin/env python3
"""Freeze a YOLO-label dataset split as COCO-style ground truth and clean input manifest.

VOC and KITTI are acquired in Ultralytics' YOLO-label layout.  This converter
creates immutable, COCO-API-compatible evaluation assets without modifying a
single upstream image or label.  Stable integer image IDs are assigned from
the sorted source-relative paths and are retained for all corruptions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

from PIL import Image
from ultralytics.utils import YAML

from topic_c.manifest import manifest_sha256, sha256_file, validate_manifest
from confirmatory_data import split_images as _split_images
from confirmatory_data import validate_dataset_provenance


def canonical_hash(document: dict) -> str:
    payload = {key: value for key, value in document.items() if key != "class_map_sha256"}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def label_for(image: Path) -> Path:
    parts = list(image.parts)
    try:
        index = parts.index("images")
    except ValueError as exc:
        raise SystemExit(f"EVAL ASSET REFUSED: source image outside images/: {image}") from exc
    parts[index] = "labels"
    return Path(*parts).with_suffix(".txt")


def split_images(root: Path, configured_split: object) -> list[Path]:
    return _split_images(root, configured_split, prefix="EVAL ASSET")


def load_labels(path: Path, width: int, height: int, class_count: int) -> list[tuple[int, list[float]]]:
    if not path.is_file():
        raise SystemExit(f"EVAL ASSET REFUSED: label missing: {path}")
    values = []
    for line_number, line in enumerate(path.read_text(encoding="utf-8").splitlines(), start=1):
        if not line.strip():
            continue
        fields = line.split()
        if len(fields) != 5:
            raise SystemExit(f"EVAL ASSET REFUSED: malformed YOLO label {path}:{line_number}")
        try:
            class_id = int(fields[0])
            xc, yc, bw, bh = (float(item) for item in fields[1:])
        except ValueError as exc:
            raise SystemExit(f"EVAL ASSET REFUSED: non-numeric YOLO label {path}:{line_number}") from exc
        if not 0 <= class_id < class_count or not (0 <= xc <= 1 and 0 <= yc <= 1 and 0 < bw <= 1 and 0 < bh <= 1):
            raise SystemExit(f"EVAL ASSET REFUSED: invalid normalized YOLO label {path}:{line_number}")
        left = max(0.0, (xc - bw / 2) * width)
        top = max(0.0, (yc - bh / 2) * height)
        right = min(float(width), (xc + bw / 2) * width)
        bottom = min(float(height), (yc + bh / 2) * height)
        if right <= left or bottom <= top:
            raise SystemExit(f"EVAL ASSET REFUSED: degenerate YOLO box {path}:{line_number}")
        values.append((class_id, [left, top, right - left, bottom - top]))
    return values


def write_new(path: Path, document: dict) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=("voc", "kitti", "tt100k"), required=True)
    parser.add_argument("--split", default="val")
    parser.add_argument("--acquisition-registry", required=True)
    parser.add_argument("--data-yaml")
    parser.add_argument("--split-registry")
    parser.add_argument("--out-annotations", required=True)
    parser.add_argument("--out-clean-manifest", required=True)
    parser.add_argument("--out-class-map", required=True)
    args = parser.parse_args()
    registry_path = Path(args.acquisition_registry).resolve()
    annotation_out = Path(args.out_annotations).resolve()
    manifest_out = Path(args.out_clean_manifest).resolve()
    class_map_out = Path(args.out_class_map).resolve()
    marker = registry_path.with_suffix(registry_path.suffix + ".complete")
    if not registry_path.is_file() or not marker.is_file() or marker.read_text(encoding="utf-8").strip() != sha256_file(registry_path):
        raise SystemExit("EVAL ASSET REFUSED: acquisition registry is absent or incomplete")
    if any(path.exists() for path in (annotation_out, manifest_out, class_map_out)):
        raise SystemExit("EVAL ASSET REFUSED: output already exists")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("dataset") != args.dataset:
        raise SystemExit("EVAL ASSET REFUSED: dataset differs from acquisition registry")
    root = Path(registry["dataset_root"]).resolve()
    yaml_path = Path(args.data_yaml).resolve() if args.data_yaml else Path(registry["vendored_yaml"]).resolve()
    provenance = validate_dataset_provenance(
        dataset=args.dataset, data_yaml=yaml_path, acquisition_registry=registry_path,
        split_registry=Path(args.split_registry) if args.split_registry else None,
    )
    configuration = YAML.load(yaml_path)
    names_dict = configuration.get("names")
    if not isinstance(names_dict, dict):
        raise SystemExit("EVAL ASSET REFUSED: data YAML has no class-name dictionary")
    names = [names_dict[index] if index in names_dict else names_dict[str(index)] for index in range(len(names_dict))]
    registered_names = {str(key): value for key, value in registry.get("class_names", {}).items()}
    configured_names = {str(key): value for key, value in names_dict.items()}
    if registered_names != configured_names:
        raise SystemExit("EVAL ASSET REFUSED: class map differs from acquisition registry")
    if args.split not in configuration:
        raise SystemExit(f"EVAL ASSET REFUSED: split absent from YAML: {args.split}")
    images = split_images(root, configuration[args.split])
    if args.split_registry:
        split_document = json.loads(Path(args.split_registry).read_text(encoding="utf-8"))
        mapping = {"train": "train", "val": "selection", "test": "final"}
        expected_count = split_document.get("splits", {}).get(mapping[args.split], {}).get("images")
    else:
        expected_count = registry.get("splits", {}).get(args.split, {}).get("images")
    if len(images) != expected_count:
        raise SystemExit(f"EVAL ASSET REFUSED: image count {len(images)} != registered {expected_count}")

    coco_images, annotations, records = [], [], []
    annotation_id = 1
    for image_id, image in enumerate(images, start=1):
        relative = image.relative_to(root).as_posix()
        with Image.open(image) as source:
            width, height = source.size
        coco_images.append({"id": image_id, "file_name": relative, "width": width, "height": height})
        for class_id, bbox in load_labels(label_for(image), width, height, len(names)):
            annotations.append({"id": annotation_id, "image_id": image_id, "category_id": class_id + 1,
                                "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0})
            annotation_id += 1
        records.append({"dataset": args.dataset, "image_id": image_id, "source_relpath": relative,
                        "source_sha256": sha256_file(image), "corruption": "clean", "severity": 0,
                        "seed": 0, "output_relpath": relative, "sha256": sha256_file(image),
                        "generator": "clean_reference", "generator_version": "1.0.0"})
    annotations_document = {
        "info": {"description": f"Topic C immutable {args.dataset} {args.split} conversion from Ultralytics YOLO labels",
                 "acquisition_registry_sha256": sha256_file(registry_path), "data_provenance": provenance},
        "licenses": [], "images": coco_images,
        "annotations": annotations,
        "categories": [{"id": index + 1, "name": name, "supercategory": "none"} for index, name in enumerate(names)],
    }
    manifest = {"schema_version": 1, "dataset": args.dataset, "split": args.split,
                "expected_image_ids": [item["id"] for item in coco_images], "records": records}
    manifest["manifest_sha256"] = manifest_sha256(manifest)
    failures = validate_manifest(manifest, annotation_out, root, root, require_pixels_changed=False) if annotation_out.exists() else []
    if failures:
        raise SystemExit("EVAL ASSET REFUSED: unexpected existing annotation validation path")
    # Validate against a temporary in-memory-equivalent file only after every source/label check above.
    write_new(annotation_out, annotations_document)
    failures = validate_manifest(manifest, annotation_out, root, root, require_pixels_changed=False)
    if failures:
        annotation_out.unlink()
        raise SystemExit("EVAL ASSET REFUSED: clean manifest validation failed:\n" + "\n".join(failures))
    class_map = {"schema_version": 1, "dataset": args.dataset, "split": args.split,
                 "annotation_sha256": sha256_file(annotation_out), "class_to_category_id": list(range(1, len(names) + 1)),
                 "class_names": names, "acquisition_registry_sha256": sha256_file(registry_path)}
    class_map["class_map_sha256"] = canonical_hash(class_map)
    write_new(manifest_out, manifest)
    manifest_out.with_suffix(manifest_out.suffix + ".complete").write_text(manifest["manifest_sha256"] + "\n", encoding="utf-8")
    write_new(class_map_out, class_map)
    class_map_out.with_suffix(class_map_out.suffix + ".complete").write_text(class_map["class_map_sha256"] + "\n", encoding="utf-8")
    print(json.dumps({"dataset": args.dataset, "split": args.split, "images": len(coco_images), "annotations": len(annotations),
                      "annotation_sha256": sha256_file(annotation_out), "manifest_sha256": manifest["manifest_sha256"],
                      "class_map_sha256": class_map["class_map_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
