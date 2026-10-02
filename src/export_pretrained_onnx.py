#!/usr/bin/env python3
"""Export torchvision-pretrained RetinaNet-R50-FPN-v2 / FCOS-R50-FPN (COCO_V1)
to frozen-shape FP32 ONNX with raw head outputs. No fine-tuning; the exported
weights are the stock torchvision COCO checkpoints."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

import onnx
import torch
from torchvision.models.detection import (
    FCOS_ResNet50_FPN_Weights, RetinaNet_ResNet50_FPN_V2_Weights,
    fcos_resnet50_fpn, retinanet_resnet50_fpn_v2)
from torchvision.models.detection.image_list import ImageList

from topic_c.manifest import sha256_file


class RetinaNetRaw(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tensor):
        images = ImageList(tensor, [(tensor.shape[-2], tensor.shape[-1])])
        features = self.model.backbone(tensor)
        feature_list = list(features.values())
        head = self.model.head(feature_list)
        anchors = self.model.anchor_generator(images, feature_list)
        return head["cls_logits"], head["bbox_regression"], anchors[0]


class FCOSRaw(torch.nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model

    def forward(self, tensor):
        images = ImageList(tensor, [(tensor.shape[-2], tensor.shape[-1])])
        features = self.model.backbone(tensor)
        feature_list = list(features.values())
        head = self.model.head(feature_list)
        anchors = self.model.anchor_generator(images, feature_list)
        return (head["cls_logits"][0], head["bbox_regression"][0],
                head["bbox_ctrness"][0], anchors[0])


def main() -> None:
    p = argparse.ArgumentParser()
    p.add_argument("--model", choices=["retinanet", "fcos"], required=True)
    p.add_argument("--imgsz", type=int, default=640)
    p.add_argument("--out", required=True)
    p.add_argument("--registry-out", required=True)
    args = p.parse_args()

    out = Path(args.out).resolve()
    reg = Path(args.registry_out).resolve()
    if out.exists() or reg.exists():
        raise SystemExit("PRETRAINED EXPORT REFUSED: output already exists")
    out.parent.mkdir(parents=True, exist_ok=True)
    reg.parent.mkdir(parents=True, exist_ok=True)

    if args.model == "retinanet":
        model = retinanet_resnet50_fpn_v2(
            weights=RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT,
            min_size=args.imgsz, max_size=args.imgsz)
        wrapper = RetinaNetRaw(model.eval())
        outputs = ["cls_logits", "bbox_regression", "anchors"]
        decoder = "torchvision_retinanet_raw_v1"
        model_name = "retinanet-r50-fpn-v2-coco-pretrained"
        weights_tag = str(RetinaNet_ResNet50_FPN_V2_Weights.DEFAULT)
    else:
        model = fcos_resnet50_fpn(
            weights=FCOS_ResNet50_FPN_Weights.DEFAULT,
            min_size=args.imgsz, max_size=args.imgsz)
        wrapper = FCOSRaw(model.eval())
        outputs = ["cls_logits", "bbox_regression", "bbox_ctrness", "anchors"]
        decoder = "torchvision_fcos_raw_v1"
        model_name = "fcos-r50-fpn-coco-pretrained"
        weights_tag = str(FCOS_ResNet50_FPN_Weights.DEFAULT)

    dev = "cuda" if torch.cuda.is_available() and \
        torch.cuda.mem_get_info()[0] > 4 << 30 else "cpu"
    wrapper = wrapper.to(dev).eval()
    example = torch.zeros((1, 3, args.imgsz, args.imgsz),
                          dtype=torch.float32, device=dev)
    torch.onnx.export(wrapper, example, out, input_names=["images"],
                      output_names=outputs, opset_version=19, dynamo=False,
                      do_constant_folding=True)

    graph = onnx.load(out, load_external_data=False).graph
    if len(graph.input) != 1 or len(graph.output) != len(outputs):
        raise SystemExit("PRETRAINED EXPORT REFUSED: unexpected graph IO")

    num_classes = wrapper.model.head.classification_head.num_classes
    record = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "coco",
        "model": model_name,
        "pretrained_weights": weights_tag,
        "training_registry": None,
        "source_checkpoint": f"torchvision:{weights_tag}",
        "onnx": str(out),
        "onnx_sha256": sha256_file(out),
        "imgsz": args.imgsz,
        "num_classes_excluding_background": int(num_classes),
        "opset": 19,
        "dynamic": False,
        "input_names": ["images"],
        "output_names": outputs,
        "decoder": decoder,
    }
    reg.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    print(f"EXPORTED {model_name} -> {out}")


if __name__ == "__main__":
    main()
