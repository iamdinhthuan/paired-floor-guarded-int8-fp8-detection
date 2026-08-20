#!/usr/bin/env python3
"""Validate COCO predictions against the exact evaluated-ID sidecar."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--predictions", required=True)
    parser.add_argument("--input-record", required=True)
    parser.add_argument("--annotations", required=True)
    args = parser.parse_args()
    predictions = json.loads(Path(args.predictions).read_text(encoding="utf-8"))
    input_record = json.loads(Path(args.input_record).read_text(encoding="utf-8"))
    images = {item["id"]: item for item in json.loads(Path(args.annotations).read_text(encoding="utf-8"))["images"]}
    categories = {item["id"] for item in json.loads(Path(args.annotations).read_text(encoding="utf-8"))["categories"]}
    allowed = set(input_record["image_ids"])
    errors = []
    for index, det in enumerate(predictions):
        if det.get("image_id") not in allowed:
            errors.append(f"detection {index}: image_id is outside input record")
        if det.get("category_id") not in categories:
            errors.append(f"detection {index}: invalid COCO category")
        values = [det.get("score"), *(det.get("bbox") or [])]
        if len(values) != 5 or not all(isinstance(value, (int, float)) and math.isfinite(value) for value in values):
            errors.append(f"detection {index}: non-finite/malformed score or bbox")
            continue
        x, y, w, h = det["bbox"]
        if w < 0 or h < 0 or x < 0 or y < 0:
            errors.append(f"detection {index}: negative/unclipped bbox")
        elif det["image_id"] in images and (x + w > images[det["image_id"]]["width"] + 1e-4 or y + h > images[det["image_id"]]["height"] + 1e-4):
            errors.append(f"detection {index}: bbox exceeds image frame")
    if errors:
        print("PREDICTIONS INVALID\n" + "\n".join(f"- {error}" for error in errors[:50]))
        raise SystemExit(2)
    print(f"PREDICTIONS VALID detections={len(predictions)} evaluated_images={len(allowed)}")


if __name__ == "__main__":
    main()
