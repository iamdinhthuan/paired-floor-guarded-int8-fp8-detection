"""Height-stratified COCO evaluation helpers for TT100K.

COCOeval overwrites annotation ``ignore`` from ``iscrowd`` while preparing an
evaluation.  Height strata therefore cannot be implemented by setting custom
ignore flags.  Instead, map both ground-truth and detection areas to squared
bbox height and let COCOeval's native area-range machinery perform the
matched-GT and unmatched-detection filtering.
"""
from __future__ import annotations

import copy
from collections.abc import Iterable

from pycocotools.coco import COCO
from pycocotools.cocoeval import COCOeval


HEIGHT_BINS = (
    ("XS", 0.0, 12.0),
    ("S", 12.0, 24.0),
    ("M", 24.0, 48.0),
    ("L", 48.0, 96.0),
    ("XL", 96.0, float("inf")),
)
HEIGHT_GROUPS = {
    "small_like": (0.0, 24.0),
    "large_like": (48.0, float("inf")),
}


def _height_area(annotation: dict) -> float:
    height = max(float(annotation["bbox"][3]), 0.0)
    return height * height


def _area_range(lower: float, upper: float) -> list[float]:
    return [lower * lower, 1e10 if upper == float("inf") else upper * upper]


def height_evaluation(
    ground_truth: dict,
    predictions: list[dict],
    image_ids: list[int],
    bins: Iterable[tuple[str, float, float]],
) -> COCOeval:
    """Build and evaluate a COCOeval object whose area axis is bbox height."""
    document = copy.deepcopy(ground_truth)
    for annotation in document["annotations"]:
        annotation["area"] = _height_area(annotation)
    gt = COCO()
    gt.dataset = document
    gt.createIndex()
    dt = gt.loadRes(copy.deepcopy(predictions))
    for annotation in dt.dataset["annotations"]:
        annotation["area"] = _height_area(annotation)
    selected = tuple(bins)
    evaluation = COCOeval(gt, dt, iouType="bbox")
    evaluation.params.imgIds = image_ids
    evaluation.params.areaRng = [_area_range(lower, upper) for _, lower, upper in selected]
    evaluation.params.areaRngLbl = [name for name, _, _ in selected]
    evaluation.params.maxDets = [100]
    evaluation.evaluate()
    return evaluation


def accumulated_height_stats(evaluation: COCOeval) -> dict[str, dict[str, float]]:
    """Return AP/AP50/AP75/AR100 for every configured height range."""
    evaluation.accumulate()
    precision = evaluation.eval["precision"]
    recall = evaluation.eval["recall"]
    iou_thresholds = evaluation.params.iouThrs

    def mean_valid(values) -> float:
        valid = values[values > -1]
        return float(valid.mean()) if valid.size else float("nan")

    result: dict[str, dict[str, float]] = {}
    for area_index, label in enumerate(evaluation.params.areaRngLbl):
        p = precision[:, :, :, area_index, 0]
        result[label] = {
            "AP": mean_valid(p),
            "AP50": mean_valid(p[iou_thresholds == 0.5]),
            "AP75": mean_valid(p[iou_thresholds == 0.75]),
            "AR100": mean_valid(recall[:, :, area_index, 0]),
        }
    return result
