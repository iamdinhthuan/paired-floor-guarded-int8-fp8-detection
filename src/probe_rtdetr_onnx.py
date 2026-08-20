#!/usr/bin/env python3
"""Small ONNX Runtime parity probe for frozen RT-DETR graphs."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import onnxruntime as ort

from topic_c.coco_data import load_coco_images, preprocess
from topic_c.cross_family import decode_rtdetr


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx", required=True); parser.add_argument("--annotations", required=True)
    parser.add_argument("--image-root", required=True); parser.add_argument("--out", required=True)
    parser.add_argument("--limit", type=int, default=10); parser.add_argument("--imgsz", type=int, default=640)
    args = parser.parse_args()
    destination = Path(args.out)
    if destination.exists(): raise SystemExit("probe output already exists")
    annotations = json.loads(Path(args.annotations).read_text())
    category_ids = [value["id"] for value in sorted(annotations["categories"], key=lambda value: value["id"])]
    images = load_coco_images(args.annotations, image_root=args.image_root, limit=args.limit)
    session = ort.InferenceSession(args.onnx, providers=["CPUExecutionProvider"])
    input_name = session.get_inputs()[0].name
    predictions = []
    for image_id, path in images:
        tensor, gain, padx, pady, width, height = preprocess(path, args.imgsz)
        output = session.run(None, {input_name: tensor})[0]
        for x1, y1, x2, y2, score, label in decode_rtdetr(output, 0.001, gain, padx, pady, width, height):
            predictions.append({"image_id": image_id, "category_id": category_ids[label],
                                "bbox": [x1, y1, x2 - x1, y2 - y1], "score": score})
    destination.parent.mkdir(parents=True, exist_ok=True)
    destination.write_text(json.dumps(predictions) + "\n")
    print(json.dumps({"images": len(images), "detections": len(predictions)}))


if __name__ == "__main__": main()
