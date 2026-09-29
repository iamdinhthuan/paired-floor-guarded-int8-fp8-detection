#!/usr/bin/env python3
"""Validate decode_fcos against torchvision's FCOS reference post-processing.

Builds synthetic FCOS head outputs on the real FPN/anchor layout produced by
the torchvision model itself, then compares torchvision
``postprocess_detections`` (run under the study's RetinaNet decoder
constants) against the flat-array ``decode_fcos`` used by the TensorRT
inference path. Writes a JSON record for the evidence package.
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from topic_c.cross_family import decode_fcos  # noqa: E402

import torchvision  # noqa: E402
from torchvision.models.detection import fcos_resnet50_fpn  # noqa: E402
from torchvision.models.detection.image_list import ImageList  # noqa: E402

CONFIDENCE, NMS_IOU, TOPK, MAX_DETS = 0.05, 0.5, 1000, 300
N_IMAGES = 32
IMG = 640
SEED = 20260927
NUM_CLASSES = 8  # KITTI head; VOC differs only in channel count


def main() -> None:
    torch.manual_seed(SEED)
    rng = np.random.default_rng(SEED)
    model = fcos_resnet50_fpn(weights=None, weights_backbone=None,
                              num_classes=NUM_CLASSES)
    # the study runs the torchvision decode algorithm under the RetinaNet
    # decoder constants, which differ from torchvision's FCOS defaults
    model.score_thresh = CONFIDENCE
    model.nms_thresh = NMS_IOU
    model.detections_per_img = MAX_DETS
    model.topk_candidates = TOPK
    model.eval()

    record = {"constants": {"confidence": CONFIDENCE, "nms_iou": NMS_IOU,
                            "topk_per_level": TOPK, "max_detections": MAX_DETS},
              "torchvision": torchvision_version(),
              "n_images": N_IMAGES, "img_size": IMG, "num_classes": NUM_CLASSES,
              "seed": SEED, "matched_detection_sets": 0,
              "max_box_deviation_px": 0.0, "max_score_deviation": 0.0,
              "total_detections": 0}

    images = torch.randn(1, 3, IMG, IMG)
    with torch.no_grad():
        features = model.backbone(images)
        anchors = model.anchor_generator(ImageList(images, [(IMG, IMG)]),
                                         list(features.values()))
    level_shapes = [tuple(f.shape[-2:]) for f in features.values()]
    num_per_level = [h * w for h, w in level_shapes]
    anchor_cat = anchors[0].numpy()
    split_anchors = [list(anchors[0].split(num_per_level))]

    for i in range(N_IMAGES):
        cls_lv, reg_lv, ctr_lv = [], [], []
        for n in num_per_level:
            cls_lv.append(torch.from_numpy(rng.normal(0, 4, (1, n, NUM_CLASSES)).astype(np.float32)))
            reg_lv.append(torch.from_numpy(np.abs(rng.normal(0.5, 0.5, (1, n, 4)).astype(np.float32))))
            ctr_lv.append(torch.from_numpy(rng.normal(0, 1, (1, n, 1)).astype(np.float32)))
        head_outputs = {"cls_logits": cls_lv, "bbox_regression": reg_lv,
                        "bbox_ctrness": ctr_lv}
        ref = model.postprocess_detections(head_outputs, split_anchors,
                                           [(IMG, IMG)])[0]

        ours = decode_fcos(torch.cat(cls_lv, 1).numpy(),
                           torch.cat(reg_lv, 1).numpy(),
                           torch.cat(ctr_lv, 1).numpy(),
                           anchor_cat, CONFIDENCE, 1.0, IMG, IMG, IMG, IMG)
        ours = torch.tensor([[d[0], d[1], d[2], d[3], d[4], d[5]] for d in ours]) \
            if ours else torch.zeros(0, 6)

        ref_boxes, ref_scores, ref_labels = ref["boxes"], ref["scores"], ref["labels"]
        record["total_detections"] += len(ref_boxes)
        if len(ours) != len(ref_boxes):
            print(f"image {i}: COUNT MISMATCH {len(ours)} vs {len(ref_boxes)}")
            continue
        order = torch.argsort(ref_scores, descending=True)
        o_order = torch.argsort(ours[:, 4], descending=True)
        if not torch.equal(ref_labels[order], ours[o_order, 5].to(torch.long)):
            print(f"image {i}: label set differs")
            continue
        record["matched_detection_sets"] += 1
        record["max_box_deviation_px"] = max(
            record["max_box_deviation_px"],
            float((ref_boxes[order] - ours[o_order, :4]).abs().max()))
        record["max_score_deviation"] = max(
            record["max_score_deviation"],
            float((ref_scores[order] - ours[o_order, 4]).abs().max()))

    out = Path(__file__).resolve().parents[1] / "submission_support_20260911" / \
        "phase_h_fcos_results" / "decoder_validation.json"
    out.write_text(json.dumps(record, indent=2) + "\n")
    print(json.dumps(record, indent=1))


def torchvision_version() -> str:
    import torchvision
    return torchvision.__version__


if __name__ == "__main__":
    main()
