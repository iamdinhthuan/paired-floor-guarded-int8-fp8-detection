#!/usr/bin/env python3
"""Export a completed RT-DETR or RetinaNet checkpoint to frozen-shape FP32 ONNX."""
from __future__ import annotations

import argparse
import json
import os
import shutil
import tempfile
from datetime import datetime, timezone
from pathlib import Path

import onnx
import torch
from torchvision.models.detection.image_list import ImageList
from ultralytics import RTDETR

from topic_c.manifest import sha256_file
from train_retinanet_dataset import build_model


def complete_record(path: Path) -> dict:
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.is_file() or not marker.is_file() or marker.read_text().strip() != sha256_file(path):
        raise SystemExit(f"CROSS-FAMILY EXPORT REFUSED: incomplete registry: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


class RetinaNetRaw(torch.nn.Module):
    def __init__(self, model: torch.nn.Module):
        super().__init__()
        self.model = model

    def forward(self, tensor: torch.Tensor):
        # Preprocessing is external and provenance-bound.  The graph consumes
        # the same normalized, aspect-resized, zero-padded square tensor that
        # GeneralizedRCNNTransform supplies to the backbone during training.
        images = ImageList(tensor, [(tensor.shape[-2], tensor.shape[-1])])
        features = self.model.backbone(tensor)
        if isinstance(features, torch.Tensor):
            features = {"0": features}
        feature_list = list(features.values())
        head = self.model.head(feature_list)
        anchors = self.model.anchor_generator(images, feature_list)
        return head["cls_logits"], head["bbox_regression"], anchors[0]


def export_rtdetr(checkpoint: Path, output: Path, imgsz: int) -> None:
    with tempfile.TemporaryDirectory(prefix="rtdetr-export-") as directory:
        copy = Path(directory) / checkpoint.name
        shutil.copy2(checkpoint, copy)
        produced = Path(str(RTDETR(str(copy)).export(
            format="onnx", imgsz=imgsz, batch=1, dynamic=False, simplify=True,
            opset=19, nms=False, half=False, device=0,
        ))).resolve()
        if not produced.is_file():
            raise RuntimeError("RT-DETR exporter produced no ONNX graph")
        os.replace(produced, output)


def export_retinanet(checkpoint: Path, output: Path, imgsz: int, num_classes: int) -> None:
    state = torch.load(checkpoint, map_location="cpu", weights_only=False)
    model = build_model(num_classes, imgsz)
    model.load_state_dict(state["model"], strict=True)
    wrapper = RetinaNetRaw(model.eval().cuda())
    example = torch.zeros((1, 3, imgsz, imgsz), dtype=torch.float32, device="cuda")
    torch.onnx.export(
        wrapper, example, output, input_names=["images"],
        output_names=["cls_logits", "bbox_regression", "anchors"],
        opset_version=19, dynamo=False, do_constant_folding=True,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--training-registry", required=True)
    parser.add_argument("--imgsz", type=int, required=True)
    parser.add_argument("--num-classes", type=int, required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--registry-out", required=True)
    args = parser.parse_args()
    training_path = Path(args.training_registry).resolve()
    output, registry_out = Path(args.out).resolve(), Path(args.registry_out).resolve()
    if output.exists() or registry_out.exists():
        raise SystemExit("CROSS-FAMILY EXPORT REFUSED: output already exists")
    training = complete_record(training_path)
    checkpoint = Path(training["best_weights"]).resolve()
    if not checkpoint.is_file() or sha256_file(checkpoint) != training["best_weights_sha256"]:
        raise SystemExit("CROSS-FAMILY EXPORT REFUSED: checkpoint hash mismatch")
    output.parent.mkdir(parents=True, exist_ok=True)
    model = training["model"]
    if model == "rtdetr-l":
        export_rtdetr(checkpoint, output, args.imgsz)
        decoder = "ultralytics_rtdetr_raw_v1"
    elif model == "retinanet-r50-fpn-v2":
        export_retinanet(checkpoint, output, args.imgsz, args.num_classes)
        decoder = "torchvision_retinanet_raw_v1"
    else:
        raise SystemExit(f"CROSS-FAMILY EXPORT REFUSED: unsupported model: {model}")
    graph = onnx.load(output, load_external_data=False).graph
    if len(graph.input) != 1 or len(graph.output) not in {1, 3}:
        raise SystemExit("CROSS-FAMILY EXPORT REFUSED: unexpected graph IO")
    record = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": training["dataset"], "model": model,
        "training_registry": str(training_path), "training_registry_sha256": sha256_file(training_path),
        "source_checkpoint": str(checkpoint), "source_checkpoint_sha256": sha256_file(checkpoint),
        "onnx": str(output), "onnx_sha256": sha256_file(output), "imgsz": args.imgsz,
        "num_classes_excluding_background": args.num_classes, "opset": 19, "dynamic": False,
        "input_names": [value.name for value in graph.input],
        "output_names": [value.name for value in graph.output], "decoder": decoder,
    }
    registry_out.parent.mkdir(parents=True, exist_ok=True)
    registry_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    registry_out.with_suffix(registry_out.suffix + ".complete").write_text(sha256_file(registry_out) + "\n")
    print(json.dumps({"ONNX EXPORT COMPLETE": str(output), "onnx_sha256": record["onnx_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
