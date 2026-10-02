#!/usr/bin/env python3
"""Map KITTI object-benchmark images to their raw driving sequence (drive).

Uses the official object devkit mapping (devkit_object.zip:
mapping/train_rand.txt + mapping/train_mapping.txt): train_rand[i] is the
1-based line of train_mapping.txt for object image i, and that line names the
raw date/drive/frame. The output is a cluster manifest
{"clusters": {"<annotation image id>": "<drive>"}} consumed by
run_nn_paired_bootstrap.py --cluster-manifest for a drive-clustered bootstrap.

The Ultralytics KITTI file stems (000000..007480) were verified to equal the
official object indices: all images of one capture date share that date's
camera resolution, and adjacent frames of one drive have a median 16x16
average-hash distance of 18 bits versus 106 for random pairs.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from collections import Counter
from pathlib import Path

DEVKIT_URL = "https://s3.eu-central-1.amazonaws.com/avg-kitti/devkit_object.zip"
TRAIN_RAND_SHA256 = "0976b8b4f876f538010a9690249b60c0766d852e23acd5e07b4c3e0bd70f305f"
TRAIN_MAPPING_SHA256 = "ca9a80524adbfbebc3179987e70a8dab1488a88ba368e77715d561c901196a54"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def drive_of_index(train_rand: Path, train_mapping: Path) -> dict[int, str]:
    if sha256_file(train_rand) != TRAIN_RAND_SHA256 or sha256_file(train_mapping) != TRAIN_MAPPING_SHA256:
        raise SystemExit("CLUSTER MAP REFUSED: devkit mapping files do not match the pinned hashes")
    rand = [int(x) for x in train_rand.read_text().strip().split(",")]
    rows = [line.split() for line in train_mapping.read_text().splitlines() if line.strip()]
    return {index: rows[line - 1][1] for index, line in enumerate(rand)}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--train-rand", type=Path, required=True)
    parser.add_argument("--train-mapping", type=Path, required=True)
    parser.add_argument("--annotations", type=Path, required=True, help="COCO-format KITTI annotations")
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    drives = drive_of_index(args.train_rand, args.train_mapping)
    images = json.loads(args.annotations.read_text(encoding="utf-8"))["images"]
    clusters = {str(image["id"]): drives[int(Path(image["file_name"]).stem)] for image in images}
    sizes = Counter(clusters.values())
    record = {"schema_version": 1, "source": DEVKIT_URL,
              "train_rand_sha256": TRAIN_RAND_SHA256, "train_mapping_sha256": TRAIN_MAPPING_SHA256,
              "annotations": str(args.annotations), "annotations_sha256": sha256_file(args.annotations),
              "n_images": len(clusters), "n_clusters": len(sizes),
              "max_cluster_size": max(sizes.values()), "clusters": clusters}
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(record, indent=1) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in record.items() if k != "clusters"}, indent=2))


if __name__ == "__main__":
    main()
