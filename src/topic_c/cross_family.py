"""Preprocessing and decoding contracts for cross-family TensorRT detectors."""
from __future__ import annotations

import math
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F
from PIL import Image
from torchvision.ops import batched_nms


RETINA_MEAN = (0.485, 0.456, 0.406)
RETINA_STD = (0.229, 0.224, 0.225)


def preprocess_retinanet(path: str | Path, imgsz: int):
    with Image.open(path) as opened:
        rgb = opened.convert("RGB")
    width, height = rgb.size
    tensor = torch.from_numpy(np.asarray(rgb, dtype=np.uint8).copy()).permute(2, 0, 1).float().div_(255.0)
    mean = torch.tensor(RETINA_MEAN)[:, None, None]
    std = torch.tensor(RETINA_STD)[:, None, None]
    tensor = (tensor - mean) / std
    scale = imgsz / max(height, width)
    tensor = F.interpolate(tensor[None], scale_factor=scale, mode="bilinear", align_corners=False,
                           recompute_scale_factor=True)[0]
    resized_h, resized_w = tensor.shape[-2:]
    padded = torch.zeros((3, imgsz, imgsz), dtype=torch.float32)
    padded[:, :resized_h, :resized_w] = tensor
    return padded[None].numpy(), scale, resized_w, resized_h, width, height


def decode_rtdetr(output: np.ndarray, confidence: float, gain: float, padx: float, pady: float,
                  width: int, height: int):
    rows = np.asarray(output).reshape(-1, 6)
    decoded = []
    canvas = max(width * gain + 2 * padx, height * gain + 2 * pady)
    for cx, cy, bw, bh, score, cls in rows:
        if float(score) < confidence:
            continue
        x1 = np.clip((float(cx - bw / 2) * canvas - padx) / gain, 0, width)
        y1 = np.clip((float(cy - bh / 2) * canvas - pady) / gain, 0, height)
        x2 = np.clip((float(cx + bw / 2) * canvas - padx) / gain, 0, width)
        y2 = np.clip((float(cy + bh / 2) * canvas - pady) / gain, 0, height)
        if x2 > x1 and y2 > y1:
            decoded.append((x1, y1, x2, y2, float(score), int(cls)))
    return decoded


def _retina_level_sizes(imgsz: int, anchors_per_location: int = 9) -> list[int]:
    return [math.ceil(imgsz / stride) ** 2 * anchors_per_location for stride in (8, 16, 32, 64, 128)]


def decode_retinanet(cls_logits: np.ndarray, bbox_regression: np.ndarray, anchors: np.ndarray,
                     confidence: float, scale: float, resized_width: int, resized_height: int,
                     width: int, height: int, topk_candidates: int = 1000,
                     nms_threshold: float = 0.5, detections_per_image: int = 300):
    logits = torch.from_numpy(np.asarray(cls_logits).reshape(-1, cls_logits.shape[-1])).float()
    deltas = torch.from_numpy(np.asarray(bbox_regression).reshape(-1, 4)).float()
    anchor_tensor = torch.from_numpy(np.asarray(anchors).reshape(-1, 4)).float()
    sizes = _retina_level_sizes(max(resized_width, resized_height, int(round(max(width, height) * scale))))
    if sum(sizes) != len(anchor_tensor) or len(logits) != len(anchor_tensor) or len(deltas) != len(anchor_tensor):
        raise ValueError("RetinaNet output does not match the frozen FPN anchor layout")
    all_boxes, all_scores, all_labels = [], [], []
    offset = 0
    for size in sizes:
        level_scores = logits[offset:offset + size].sigmoid().flatten()
        keep = torch.where(level_scores > confidence)[0]
        count = min(topk_candidates, int(keep.numel()))
        if count:
            scores, order = level_scores[keep].topk(count)
            indexes = keep[order]
            anchor_indexes = torch.div(indexes, logits.shape[1], rounding_mode="floor") + offset
            labels = indexes % logits.shape[1]
            foreground = labels > 0
            anchor_indexes, labels, scores = anchor_indexes[foreground], labels[foreground] - 1, scores[foreground]
            selected_anchors, selected_deltas = anchor_tensor[anchor_indexes], deltas[anchor_indexes]
            widths = selected_anchors[:, 2] - selected_anchors[:, 0]
            heights = selected_anchors[:, 3] - selected_anchors[:, 1]
            ctr_x = selected_anchors[:, 0] + 0.5 * widths
            ctr_y = selected_anchors[:, 1] + 0.5 * heights
            dx, dy = selected_deltas[:, 0], selected_deltas[:, 1]
            dw = selected_deltas[:, 2].clamp(max=math.log(1000.0 / 16))
            dh = selected_deltas[:, 3].clamp(max=math.log(1000.0 / 16))
            pred_ctr_x, pred_ctr_y = dx * widths + ctr_x, dy * heights + ctr_y
            pred_w, pred_h = dw.exp() * widths, dh.exp() * heights
            boxes = torch.stack((pred_ctr_x - 0.5 * pred_w, pred_ctr_y - 0.5 * pred_h,
                                 pred_ctr_x + 0.5 * pred_w, pred_ctr_y + 0.5 * pred_h), dim=1)
            boxes[:, 0::2].clamp_(0, resized_width)
            boxes[:, 1::2].clamp_(0, resized_height)
            nonempty = (boxes[:, 2] > boxes[:, 0]) & (boxes[:, 3] > boxes[:, 1])
            all_boxes.append(boxes[nonempty]); all_scores.append(scores[nonempty]); all_labels.append(labels[nonempty])
        offset += size
    if not all_boxes:
        return []
    boxes, scores, labels = torch.cat(all_boxes), torch.cat(all_scores), torch.cat(all_labels)
    keep = batched_nms(boxes, scores, labels, nms_threshold)[:detections_per_image]
    boxes = boxes[keep].div_(scale); scores, labels = scores[keep], labels[keep]
    boxes[:, 0::2].clamp_(0, width); boxes[:, 1::2].clamp_(0, height)
    return [(float(*box[0:1]), float(*box[1:2]), float(*box[2:3]), float(*box[3:4]), float(score), int(label))
            for box, score, label in zip(boxes, scores, labels)]
