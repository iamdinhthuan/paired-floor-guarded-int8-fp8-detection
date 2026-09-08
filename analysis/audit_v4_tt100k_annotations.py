#!/usr/bin/env python3
"""Read-only TT100K original → YOLO → frozen COCO annotation audit.

Remote mode streams this exact source over SSH stdin; it creates no remote file,
imports no training package, and writes only local report outputs. Pixel hashes
are current byte hashes, not retroactive historical execution attestations.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import math
from pathlib import Path
import shlex
import subprocess


def sha256(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def classify_image(image_id, original, class_to_idx, width, height, label_text):
    """Reproduce the pinned recipe, distinguishing absent evidence from negatives."""
    result = {"classification": None, "expected_rows": [], "excluded_categories": []}
    if original is None:
        result["classification"] = "missing_original_annotation"
        return result
    if str(original.get("id")) != str(image_id) or Path(original.get("path", "")).stem != str(image_id):
        result["classification"] = "invalid_image_id"
        return result
    if not isinstance(original.get("objects"), list):
        result["classification"] = "missing_object_list"
        return result
    result["original_object_count"] = len(original["objects"])
    try:
        for obj in original["objects"]:
            category = obj["category"]
            if category not in class_to_idx:
                result["excluded_categories"].append(category)
                continue
            box = obj["bbox"]
            xmin, ymin, xmax, ymax = (float(box[key]) for key in ("xmin", "ymin", "xmax", "ymax"))
            values = [((xmin + xmax) / 2) / width, ((ymin + ymax) / 2) / height,
                      (xmax - xmin) / width, (ymax - ymin) / height]
            if not all(math.isfinite(value) for value in values):
                raise ValueError("nonfinite source box")
            row = [class_to_idx[category]] + [float(f"{max(0, min(1, value)):.6f}") for value in values]
            result["expected_rows"].append(row)
    except (KeyError, ValueError, TypeError, ZeroDivisionError) as exc:
        result.update(classification="invalid_original_annotation", detail=str(exc))
        return result
    if result["expected_rows"] and label_text is None:
        result["classification"] = "positive_missing_converted_label"
        return result
    try:
        actual = []
        for line in (label_text or "").splitlines():
            if not line.strip():
                continue
            fields = line.split()
            if len(fields) != 5:
                raise ValueError("expected five YOLO fields")
            row = [int(fields[0])] + [float(value) for value in fields[1:]]
            if not all(math.isfinite(value) for value in row):
                raise ValueError("nonfinite label")
            actual.append(row)
        result["converted_object_count"] = len(actual)
        if actual != result["expected_rows"]:
            result["classification"] = "annotation_conversion_mismatch"
        elif result["expected_rows"]:
            result["classification"] = "positive_retained_verified"
        elif original["objects"]:
            result["classification"] = "explicitly_excluded_classes"
        else:
            result["classification"] = "valid_negative"
    except (ValueError, TypeError) as exc:
        result.update(classification="annotation_conversion_mismatch", detail=str(exc))
    return result


def coco_rows(rows, width, height):
    """Pinned build_yolo_dataset_eval_assets.load_labels edge clipping semantics."""
    converted = []
    for class_id, xc, yc, bw, bh in rows:
        left, top = max(0.0, (xc - bw / 2) * width), max(0.0, (yc - bh / 2) * height)
        right, bottom = min(float(width), (xc + bw / 2) * width), min(float(height), (yc + bh / 2) * height)
        bbox = [left, top, right - left, bottom - top]
        converted.append({"category_id": class_id + 1, "bbox": bbox, "area": bbox[2] * bbox[3], "iscrowd": 0})
    return converted


def audit_source_lists(dataset, original):
    """Audit archive ids.txt universes separately from JSON annotation universes."""
    result = {}
    for source_split, converted_split in (("train", "train"), ("other", "val"), ("test", "test")):
        path = dataset / "data" / source_split / "ids.txt"
        if not path.is_file():
            result[converted_split] = {"status": "missing", "path": str(path)}
            continue
        ids = path.read_text().split()
        annotated = {key for key, value in original["imgs"].items() if value["path"].startswith(source_split + "/")}
        absent = []
        for image_id in sorted(set(ids) - annotated):
            image = path.parent / (image_id + ".jpg")
            converted = dataset / "images" / converted_split / (image_id + ".jpg")
            absent.append({"original_image_id": image_id, "source_path": str(image),
                           "source_sha256": sha256(image) if image.is_file() else None,
                           "converted_image_exists": converted.is_file(),
                           "classification": "missing_original_annotation" if converted.is_file() else "missing_original_annotation_not_converted"})
        result[converted_split] = {"status": "present", "path": str(path), "sha256": sha256(path),
                                   "listed_images": len(ids), "duplicate_ids": len(ids) - len(set(ids)),
                                   "annotated_images": len(annotated), "listed_without_original_annotation": absent,
                                   "annotated_not_listed": sorted(annotated - set(ids))}
    return result


def audit_final(root, ledger, names):
    paths = {"annotations": root / "manifests/annotations/tt100k_test_ultralytics_v1_coco.json",
             "clean_manifest": root / "manifests/images/tt100k_test_clean_ultralytics_v1.json",
             "class_map": root / "manifests/classes/tt100k_test_ultralytics_v1.json"}
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        return {"status": "missing", "missing_paths": missing}
    docs = {key: json.loads(path.read_text()) for key, path in paths.items()}
    annotations, manifest, class_map = (docs[key] for key in paths)
    by_path = {row["converted_relpath"]: row for row in ledger if row["split"] == "test"}
    grouped = defaultdict(list)
    for ann in annotations["annotations"]:
        grouped[ann["image_id"]].append(ann)
    records = {row["image_id"]: row for row in manifest["records"]}
    issues = []
    ordered = [row["id"] for row in annotations["images"]]
    if ordered != manifest["expected_image_ids"] or len(ordered) != len(set(ordered)):
        issues.append("annotation/manifest ordered IDs differ or are duplicated")
    if len(records) != len(manifest["records"]) or set(records) != set(ordered):
        issues.append("manifest duplicate, missing or extra records")
    if set(grouped) - set(ordered):
        issues.append("orphan COCO annotation image IDs")
    if set(by_path) != {image["file_name"] for image in annotations["images"]}:
        issues.append("native test and final COCO source paths differ")
    expected_categories = [{"id": index + 1, "name": name} for index, name in enumerate(names)]
    if [{"id": c["id"], "name": c["name"]} for c in annotations["categories"]] != expected_categories:
        issues.append("COCO categories differ from native class map")
    if class_map.get("class_names") != names or class_map.get("class_to_category_id") != list(range(1, len(names) + 1)):
        issues.append("final class map differs from native class map")
    if class_map.get("annotation_sha256") != sha256(paths["annotations"]):
        issues.append("class map annotation byte hash differs")
    for image in annotations["images"]:
        row = by_path.get(image["file_name"])
        if row is None:
            continue
        row["final_evaluation_id"] = image["id"]
        failures = []
        if [image["width"], image["height"]] != [row.get("width"), row.get("height")]:
            failures.append("dimensions")
        expected = coco_rows(row.get("expected_rows", []), image["width"], image["height"])
        actual = [{key: ann[key] for key in ("category_id", "bbox", "area", "iscrowd")} for ann in grouped[image["id"]]]
        if actual != expected:
            failures.append("coco_boxes_categories_area")
        record = records.get(image["id"], {})
        if record.get("source_relpath") != image["file_name"] or record.get("source_sha256") != row["converted_image_sha256"] or record.get("sha256") != row["converted_image_sha256"]:
            failures.append("clean_manifest_path_or_byte_hash")
        row["final_evaluation_status"] = "verified" if not failures else "mismatch"
        if failures:
            issues.append({"image_id": image["id"], "failures": failures})
    return {"status": "verified" if not issues else "mismatch", "images": len(ordered),
            "objects": len(annotations["annotations"]), "issues": issues,
            "ordered_image_ids_sha256": hashlib.sha256(json.dumps(ordered, separators=(",", ":")).encode()).hexdigest(),
            "sources": {key: {"path": str(path), "sha256": sha256(path)} for key, path in paths.items()}}


def run_audit(root):
    from PIL import Image
    import yaml

    root = Path(root).resolve()
    dataset = root / "data/datasets/TT100K"
    annotation_path = dataset / "data/annotations.json"
    yaml_path = root / "configs/datasets/tt100k_ultralytics_v1.yaml"
    original = json.loads(annotation_path.read_text())
    configuration = yaml.safe_load(yaml_path.read_text())
    names = [configuration["names"][i] for i in range(len(configuration["names"]))]
    class_to_idx = {name: i for i, name in enumerate(names)}
    ledger, seen = [], set()
    for split in ("train", "val", "test"):
        images = sorted((dataset / configuration[split]).glob("*"))
        for image in images:
            if not image.is_file() or image.suffix.lower() not in {".jpg", ".jpeg", ".png"}:
                continue
            image_id = image.stem
            seen.add(image_id)
            source = original["imgs"].get(image_id)
            original_path = dataset / "data" / source["path"] if source else None
            label = dataset / "labels" / split / (image.stem + ".txt")
            row = {"original_image_id": image_id, "split": split,
                   "membership": {"train": "native_training", "val": "native_checkpoint_selection", "test": "historical_final_evaluation"}[split],
                   "original_relpath": source.get("path") if source else None,
                   "converted_relpath": image.relative_to(dataset).as_posix(),
                   "label_relpath": label.relative_to(dataset).as_posix(),
                   "label_present": label.is_file(), "label_sha256": sha256(label) if label.is_file() else None,
                   "source_sha256": sha256(original_path) if original_path and original_path.is_file() else None,
                   "converted_image_sha256": sha256(image)}
            row["image_copy_byte_match"] = row["source_sha256"] == row["converted_image_sha256"]
            try:
                with Image.open(image) as opened:
                    width, height = opened.size
                row.update(width=width, height=height)
                row.update(classify_image(image_id, source, class_to_idx, width, height,
                                          label.read_text() if label.is_file() else None))
                expected_split = "train" if source and "train" in source["path"] else "test" if source and "test" in source["path"] else "val"
                if expected_split != split:
                    row["classification"] = "split_conversion_mismatch"
            except (OSError, ValueError) as exc:
                row.update(classification="image_read_failure", detail=str(exc))
            ledger.append(row)
    original_not_converted = [{"original_image_id": key, "path": value.get("path"),
                               "original_file_exists": (dataset / "data" / value["path"]).is_file()}
                              for key, value in original["imgs"].items() if key not in seen]
    splits = {}
    for split in ("train", "val", "test"):
        rows = [row for row in ledger if row["split"] == split]
        splits[split] = {"images": len(rows), "label_files": sum(row["label_present"] for row in rows),
                         "original_objects": sum(row.get("original_object_count", 0) for row in rows),
                         "retained_objects": sum(len(row.get("expected_rows", [])) for row in rows),
                         "excluded_objects": sum(len(row.get("excluded_categories", [])) for row in rows),
                         "classifications": dict(Counter(row["classification"] for row in rows)),
                         "image_copy_hash_mismatches": sum(not row["image_copy_byte_match"] for row in rows)}
    source_paths = [annotation_path, yaml_path, root / "src/acquire_ultralytics_dataset.py",
                    root / "src/build_yolo_dataset_eval_assets.py", root / "src/topic_c/tt100k_height.py",
                    root / "src/tt100k_eval.py", root / "src/train_yolo_dataset.py",
                    root / "manifests/datasets/tt100k_acquisition_v1.json"]
    for model in ("n", "m", "x"):
        source_paths.extend([root / f"manifests/training/tt100k_yolo11{model}_train_v1.json",
                             root / f"outputs/training/tt100k/tt100k_yolo11{model}_train_v1/args.yaml"])
    sources = [{"path": str(path), "sha256": sha256(path) if path.is_file() else None,
                "status": "present" if path.is_file() else "missing"} for path in source_paths]
    final = audit_final(root, ledger, names)
    # Expected rows aid auditing but need not repeat every bounding box in ledger.
    for row in ledger:
        row["retained_object_count"] = len(row.pop("expected_rows", []))
    return {"schema_version": 1, "audit": "cviu_v4_tt100k_annotations",
            "audited_at_utc": datetime.now(timezone.utc).isoformat(), "project_root": str(root),
            "read_only_remote": True, "hash_semantics": "SHA-256 over exact current file bytes; per-image source, converted image and label hashes; not historical execution attestation",
            "class_count": len(names), "class_names": names,
            "original_annotation_images": len(original["imgs"]),
            "original_types_not_in_class_map": sorted(set(original.get("types", [])) - set(names)),
            "source_split_lists": audit_source_lists(dataset, original),
            "original_not_converted": original_not_converted, "splits": splits,
            "sources": sources, "conversion_recipe_sha256": hashlib.sha256(configuration.get("download", "").encode()).hexdigest(),
            "final_evaluation": final, "ledger": ledger,
            "limitations": ["Empty object lists are valid negatives relative to the supplied original annotations; visual completeness was not independently reannotated.",
                            "Current source/label agreement does not prove files were identical during historical training.",
                            "Archive split-list IDs absent from the original JSON are reported separately; they are not inferred negatives or proof of converter loss."]}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", required=True)
    parser.add_argument("--ssh")
    parser.add_argument("--remote-python", default="/home/thuan/miniconda3/envs/qtsd/bin/python")
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--stdout", action="store_true")
    args = parser.parse_args()
    if args.ssh:
        command = f"{shlex.quote(args.remote_python)} - --root {shlex.quote(args.root)} --stdout"
        proc = subprocess.run(["ssh", "-o", "BatchMode=yes", args.ssh, command],
                              input=Path(__file__).read_text(), text=True, capture_output=True, check=True)
        report = json.loads(proc.stdout)
    else:
        report = run_audit(args.root)
    if args.stdout:
        print(json.dumps(report, allow_nan=False))
        return
    if args.output_dir is None:
        parser.error("--output-dir or --stdout is required")
    args.output_dir.mkdir(parents=True, exist_ok=True)
    report["audit_script_sha256"] = sha256(Path(__file__))
    ledger = report.pop("ledger")
    ledger_path = args.output_dir / "image_ledger.jsonl"
    summary_path = args.output_dir / "summary.json"
    if ledger_path.exists() or summary_path.exists():
        raise SystemExit("Refusing to overwrite an existing audit report")
    ledger_path.write_text("".join(json.dumps(row, sort_keys=True, allow_nan=False) + "\n" for row in ledger))
    report["ledger"] = {"path": str(ledger_path), "rows": len(ledger), "sha256": sha256(ledger_path)}
    summary_path.write_text(json.dumps(report, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"summary": str(summary_path), "splits": report["splits"], "final_evaluation": report["final_evaluation"]["status"]}, indent=2))


if __name__ == "__main__":
    main()
