#!/usr/bin/env python3
"""Freeze a deterministic class-covering subset for corruption-seed sensitivity."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pilot_registry import canonical_hash
from topic_c.manifest import manifest_sha256, sha256_file


def rank(dataset: str, image_id: int, seed: int) -> int:
    return int.from_bytes(
        hashlib.sha256(f"{dataset}|{seed}|{image_id}".encode()).digest()[:8], "big"
    )


def select_images(
    *,
    dataset: str,
    ordered_image_ids: list[int],
    categories_by_image: dict[int, set[int]],
    target: int,
    seed: int,
) -> list[int]:
    if target < 1 or target > len(ordered_image_ids) or len(set(ordered_image_ids)) != len(ordered_image_ids):
        raise ValueError("invalid subset target or image universe")
    observed = set().union(*(categories_by_image.get(image_id, set()) for image_id in ordered_image_ids))
    uncovered = set(observed)
    selected: set[int] = set()
    while uncovered:
        candidates = [image_id for image_id in ordered_image_ids if image_id not in selected]
        best = max(
            candidates,
            key=lambda image_id: (
                len(categories_by_image.get(image_id, set()) & uncovered),
                -rank(dataset, image_id, seed),
            ),
        )
        gained = categories_by_image.get(best, set()) & uncovered
        if not gained:
            raise ValueError("category coverage cannot be satisfied")
        selected.add(best)
        uncovered -= gained
        if len(selected) > target:
            raise ValueError("target is too small to preserve category coverage")
    remaining = sorted(
        (image_id for image_id in ordered_image_ids if image_id not in selected),
        key=lambda image_id: rank(dataset, image_id, seed),
    )
    selected.update(remaining[: target - len(selected)])
    if len(selected) != target:
        raise ValueError("failed to materialize exact subset size")
    return [image_id for image_id in ordered_image_ids if image_id in selected]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--clean-manifest", required=True)
    parser.add_argument("--target", type=int, default=512)
    parser.add_argument("--seed", type=int, default=20260818)
    parser.add_argument("--annotation-out", required=True)
    parser.add_argument("--manifest-out", required=True)
    parser.add_argument("--registry-out", required=True)
    args = parser.parse_args()
    annotation_path = Path(args.annotations).resolve()
    manifest_path = Path(args.clean_manifest).resolve()
    annotation_out = Path(args.annotation_out).resolve()
    manifest_out = Path(args.manifest_out).resolve()
    registry_out = Path(args.registry_out).resolve()
    targets = (annotation_out, manifest_out, registry_out)
    if any(path.exists() or path.with_suffix(path.suffix + ".complete").exists() for path in targets):
        raise SystemExit("CORRUPTION SUBSET REFUSED: output exists")
    annotations = json.loads(annotation_path.read_text(encoding="utf-8"))
    manifest = json.loads(manifest_path.read_text(encoding="utf-8"))
    if manifest.get("manifest_sha256") != manifest_sha256(manifest):
        raise SystemExit("CORRUPTION SUBSET REFUSED: invalid clean manifest")
    records = manifest.get("records", [])
    ordered_ids = [int(record["image_id"]) for record in records]
    annotation_ids = [int(image["id"]) for image in annotations.get("images", [])]
    if set(ordered_ids) != set(annotation_ids):
        raise SystemExit("CORRUPTION SUBSET REFUSED: annotation/manifest universe mismatch")
    categories_by_image = {image_id: set() for image_id in ordered_ids}
    for annotation in annotations.get("annotations", []):
        image_id = int(annotation["image_id"])
        if image_id in categories_by_image and int(annotation.get("iscrowd", 0)) == 0:
            categories_by_image[image_id].add(int(annotation["category_id"]))
    selected = select_images(
        dataset=args.dataset,
        ordered_image_ids=ordered_ids,
        categories_by_image=categories_by_image,
        target=args.target,
        seed=args.seed,
    )
    selected_set = set(selected)
    image_by_id = {int(image["id"]): image for image in annotations["images"]}
    subset_annotations = {
        key: value
        for key, value in annotations.items()
        if key not in {"images", "annotations", "subset_provenance"}
    }
    subset_annotations["images"] = [image_by_id[image_id] for image_id in selected]
    subset_annotations["annotations"] = [
        annotation
        for annotation in annotations.get("annotations", [])
        if int(annotation["image_id"]) in selected_set
    ]
    subset_annotations["subset_provenance"] = {
        "dataset": args.dataset,
        "selection": "greedy observed-category cover then deterministic hash-rank fill",
        "target": args.target,
        "seed": args.seed,
        "source_annotation": str(annotation_path),
        "source_annotation_sha256": sha256_file(annotation_path),
        "source_manifest": str(manifest_path),
        "source_manifest_sha256": sha256_file(manifest_path),
    }
    selected_records = [record for record in records if int(record["image_id"]) in selected_set]
    subset_manifest = {
        key: value
        for key, value in manifest.items()
        if key not in {"records", "expected_image_ids", "manifest_sha256"}
    }
    subset_manifest["expected_image_ids"] = selected
    subset_manifest["records"] = selected_records
    subset_manifest["manifest_sha256"] = manifest_sha256(subset_manifest)
    annotation_out.parent.mkdir(parents=True, exist_ok=True)
    manifest_out.parent.mkdir(parents=True, exist_ok=True)
    registry_out.parent.mkdir(parents=True, exist_ok=True)
    annotation_out.write_text(json.dumps(subset_annotations, indent=2) + "\n", encoding="utf-8")
    manifest_out.write_text(json.dumps(subset_manifest, indent=2) + "\n", encoding="utf-8")
    annotation_out.with_suffix(annotation_out.suffix + ".complete").write_text(
        sha256_file(annotation_out) + "\n", encoding="utf-8"
    )
    manifest_out.with_suffix(manifest_out.suffix + ".complete").write_text(
        subset_manifest["manifest_sha256"] + "\n", encoding="utf-8"
    )
    observed = set().union(*(categories_by_image[image_id] for image_id in ordered_ids))
    covered = set().union(*(categories_by_image[image_id] for image_id in selected))
    registry = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": args.dataset,
        "target": args.target,
        "seed": args.seed,
        "images": len(selected),
        "annotations": len(subset_annotations["annotations"]),
        "observed_categories": len(observed),
        "covered_categories": len(covered),
        "all_observed_categories_preserved": covered == observed,
        "annotation": str(annotation_out),
        "annotation_sha256": sha256_file(annotation_out),
        "manifest": str(manifest_out),
        "manifest_file_sha256": sha256_file(manifest_out),
        "manifest_sha256": subset_manifest["manifest_sha256"],
    }
    registry["registry_sha256"] = canonical_hash(registry, "registry_sha256")
    registry_out.write_text(json.dumps(registry, indent=2) + "\n", encoding="utf-8")
    registry_out.with_suffix(registry_out.suffix + ".complete").write_text(
        sha256_file(registry_out) + "\n", encoding="utf-8"
    )
    print(f"CORRUPTION SUBSET COMPLETE dataset={args.dataset} images={len(selected)}")


if __name__ == "__main__":
    main()
