"""Fixed YOLO COCO decoder: confidence 0.001, per-class NMS 0.7, top-300."""
from __future__ import annotations

import numpy as np

COCO80_TO_91 = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22, 23, 24, 25, 27, 28, 31, 32, 33, 34, 35, 36, 37, 38, 39, 40, 41, 42, 43, 44, 46, 47, 48, 49, 50, 51, 52, 53, 54, 55, 56, 57, 58, 59, 60, 61, 62, 63, 64, 65, 67, 70, 72, 73, 74, 75, 76, 77, 78, 79, 80, 81, 82, 84, 85, 86, 87, 88, 89, 90]
MAX_DET, CONF_FLOOR, NMS_IOU = 300, 1e-3, 0.7


def nms(boxes: np.ndarray, scores: np.ndarray, threshold: float) -> list[int]:
    x1, y1, x2, y2 = boxes.T
    areas, order, keep = (x2 - x1) * (y2 - y1), scores.argsort()[::-1], []
    while order.size:
        index = order[0]
        keep.append(int(index))
        if order.size == 1:
            break
        rest = order[1:]
        xx1, yy1 = np.maximum(x1[index], x1[rest]), np.maximum(y1[index], y1[rest])
        xx2, yy2 = np.minimum(x2[index], x2[rest]), np.minimum(y2[index], y2[rest])
        intersection = np.maximum(0, xx2 - xx1) * np.maximum(0, yy2 - yy1)
        order = rest[intersection / (areas[index] + areas[rest] - intersection + 1e-12) <= threshold]
    return keep


def decode(output: np.ndarray, confidence: float, gain: float, padx: float, pady: float, width: int, height: int) -> list[tuple[float, float, float, float, float, int]]:
    prediction = output[0].T
    classes = prediction[:, 4:]
    class_id, score = classes.argmax(axis=1), classes.max(axis=1)
    valid = score > confidence
    if not valid.any():
        return []
    values, class_id, score = prediction[valid, :4], class_id[valid], score[valid]
    boxes = np.column_stack((values[:, 0] - values[:, 2] / 2, values[:, 1] - values[:, 3] / 2, values[:, 0] + values[:, 2] / 2, values[:, 1] + values[:, 3] / 2))
    results = []
    for category in np.unique(class_id):
        selected = class_id == category
        for index in nms(boxes[selected], score[selected], NMS_IOU):
            x1, y1, x2, y2 = boxes[selected][index]
            x1, y1 = max((x1 - padx) / gain, 0.0), max((y1 - pady) / gain, 0.0)
            x2, y2 = min((x2 - padx) / gain, width), min((y2 - pady) / gain, height)
            if x2 <= x1 or y2 <= y1:
                continue
            results.append((x1, y1, x2, y2, float(score[selected][index]), int(category)))
    return sorted(results, key=lambda item: -item[4])[:MAX_DET]
