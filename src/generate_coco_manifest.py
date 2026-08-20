#!/usr/bin/env python3
"""Create a checksummed clean COCO manifest without copying or changing pixels."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from topic_c.manifest import manifest_sha256, official_coco_images, sha256_file, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--image-root", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()
    output, root = Path(args.out), Path(args.image_root)
    if output.exists():
        raise SystemExit(f"refusing to overwrite manifest: {output}")
    images = official_coco_images(args.annotations)
    if args.limit:
        images = images[:args.limit]
    records = []
    for image in images:
        file_path = root / image["file_name"]
        if not file_path.is_file():
            raise SystemExit(f"clean image missing: {file_path}")
        file_hash = sha256_file(file_path)
        records.append({"dataset": "coco", "image_id": int(image["id"]), "source_relpath": image["file_name"],
                        "source_sha256": file_hash, "corruption": "clean", "severity": 0, "seed": 0,
                        "output_relpath": image["file_name"], "sha256": file_hash,
                        "generator": "clean_reference", "generator_version": "2.0.0"})
    document = {"schema_version": 2, "dataset": "coco", "split": "val2017", "expected_image_ids": [record["image_id"] for record in records], "records": records}
    document["manifest_sha256"] = manifest_sha256(document)
    failures = validate_manifest(document, args.annotations, root, root, require_pixels_changed=False)
    if failures:
        raise SystemExit("clean manifest failed validation:\n" + "\n".join(failures))
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".complete").write_text(document["manifest_sha256"] + "\n", encoding="utf-8")
    print(f"{len(records)} clean COCO images -> {output}; sha256={document['manifest_sha256']}")


if __name__ == "__main__":
    main()
