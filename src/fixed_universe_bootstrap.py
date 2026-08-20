"""Fixed-category-universe sensitivity estimator for COCO-style AP.

This module is intentionally separate from the hash-bound historical paired
bootstrap implementation. It defines a new Bayesian image-weight sensitivity
estimand and does not alter the completed ordinary-bootstrap evidence chain.
"""

from __future__ import annotations

from typing import Any

import numpy as np


def fixed_category_area_mask(evaluation: Any) -> np.ndarray:
    """Return category/area cells with positives in the full image universe."""
    params = evaluation.params
    categories, areas = len(params.catIds), len(params.areaRng)
    n_images = len(evaluation._paramsEval.imgIds)
    eligible = np.zeros((categories, areas), dtype=bool)
    for category in range(categories):
        base_category = category * areas * n_images
        for area in range(areas):
            base_area = base_category + area * n_images
            eligible[category, area] = any(
                entry is not None and np.count_nonzero(entry["gtIgnore"] == 0) > 0
                for entry in (
                    evaluation.evalImgs[base_area + position]
                    for position in range(n_images)
                )
            )
    return eligible


def accumulate_ap_weighted(
    evaluation: Any,
    image_weights: np.ndarray,
    eligible: np.ndarray | None = None,
) -> np.ndarray:
    """Compute COCO-style AP under strictly positive image weights.

    Detection and ground-truth contributions receive their source image's
    Bayesian-bootstrap weight. Stable score ties retain the frozen image order.
    This is a fixed-universe sensitivity estimand, not an identity claim with
    ordinary resampling by duplicated image index.
    """
    params = evaluation.params
    weights = np.asarray(image_weights, dtype=float)
    n_images = len(evaluation._paramsEval.imgIds)
    if (
        weights.shape != (n_images,)
        or not np.isfinite(weights).all()
        or np.any(weights <= 0)
    ):
        raise ValueError("fixed-universe image weights must be finite and strictly positive")
    categories, areas = len(params.catIds), len(params.areaRng)
    universe = fixed_category_area_mask(evaluation) if eligible is None else np.asarray(eligible)
    if universe.shape != (categories, areas) or universe.dtype != bool:
        raise ValueError("fixed category/area universe has the wrong shape or dtype")

    thresholds, recalls = len(params.iouThrs), len(params.recThrs)
    precision = -np.ones((thresholds, recalls, categories, areas))
    for category in range(categories):
        base_category = category * areas * n_images
        for area in range(areas):
            if not universe[category, area]:
                continue
            base_area = base_category + area * n_images
            weighted_entries = [
                (evaluation.evalImgs[base_area + position], weights[position])
                for position in range(n_images)
                if evaluation.evalImgs[base_area + position] is not None
            ]
            if not weighted_entries:
                continue
            scores = np.concatenate(
                [entry["dtScores"][:100] for entry, _ in weighted_entries]
            )
            order = np.argsort(-scores, kind="mergesort")
            matches = np.concatenate(
                [entry["dtMatches"][:, :100] for entry, _ in weighted_entries], axis=1
            )[:, order]
            ignored = np.concatenate(
                [entry["dtIgnore"][:, :100] for entry, _ in weighted_entries], axis=1
            )[:, order]
            detection_weights = np.concatenate(
                [
                    np.full(len(entry["dtScores"][:100]), weight, dtype=float)
                    for entry, weight in weighted_entries
                ]
            )[order]
            positives = float(
                sum(
                    weight * np.count_nonzero(entry["gtIgnore"] == 0)
                    for entry, weight in weighted_entries
                )
            )
            if positives <= 0:
                raise ValueError("fixed category universe lost all positive weight")
            true_positive = np.cumsum(
                np.logical_and(matches, np.logical_not(ignored)) * detection_weights,
                axis=1,
            )
            false_positive = np.cumsum(
                np.logical_and(np.logical_not(matches), np.logical_not(ignored))
                * detection_weights,
                axis=1,
            )
            for threshold in range(thresholds):
                tp, fp = true_positive[threshold], false_positive[threshold]
                recall = tp / positives
                curve = tp / (tp + fp + np.spacing(1))
                curve = np.maximum.accumulate(curve[::-1])[::-1]
                sampled = np.zeros(recalls)
                locations = np.searchsorted(recall, params.recThrs, side="left")
                valid = locations < len(curve)
                sampled[valid] = curve[locations[valid]]
                precision[threshold, :, category, area] = sampled
    result = []
    for area in range(areas):
        values = precision[:, :, :, area]
        result.append(
            float(values[values > -1].mean())
            if (values > -1).any()
            else float("nan")
        )
    return np.asarray(result)


def prepare_weighted_ap(evaluation: Any, eligible: np.ndarray | None = None) -> dict[str, Any]:
    """Precompute score ordering and image membership for repeated weight draws."""
    params = evaluation.params
    n_images = len(evaluation._paramsEval.imgIds)
    categories, areas = len(params.catIds), len(params.areaRng)
    universe = fixed_category_area_mask(evaluation) if eligible is None else np.asarray(eligible)
    if universe.shape != (categories, areas) or universe.dtype != bool:
        raise ValueError("fixed category/area universe has the wrong shape or dtype")
    streams: list[list[dict[str, np.ndarray] | None]] = []
    for category in range(categories):
        category_streams: list[dict[str, np.ndarray] | None] = []
        base_category = category * areas * n_images
        for area in range(areas):
            if not universe[category, area]:
                category_streams.append(None)
                continue
            base_area = base_category + area * n_images
            positioned = [
                (position, evaluation.evalImgs[base_area + position])
                for position in range(n_images)
                if evaluation.evalImgs[base_area + position] is not None
            ]
            scores = np.concatenate([entry["dtScores"][:100] for _, entry in positioned])
            order = np.argsort(-scores, kind="mergesort")
            matches = np.concatenate(
                [entry["dtMatches"][:, :100] for _, entry in positioned], axis=1
            )[:, order]
            ignored = np.concatenate(
                [entry["dtIgnore"][:, :100] for _, entry in positioned], axis=1
            )[:, order]
            detection_images = np.concatenate(
                [np.full(len(entry["dtScores"][:100]), position, dtype=np.int32) for position, entry in positioned]
            )[order]
            gt_counts = np.zeros(n_images, dtype=np.int32)
            for position, entry in positioned:
                gt_counts[position] = np.count_nonzero(entry["gtIgnore"] == 0)
            category_streams.append(
                {
                    "matches": matches,
                    "ignored": ignored,
                    "detection_images": detection_images,
                    "gt_counts": gt_counts,
                }
            )
        streams.append(category_streams)
    return {
        "n_images": n_images,
        "categories": categories,
        "areas": areas,
        "thresholds": len(params.iouThrs),
        "rec_thrs": np.asarray(params.recThrs),
        "streams": streams,
    }


def accumulate_prepared_ap(prepared: dict[str, Any], image_weights: np.ndarray) -> np.ndarray:
    """Evaluate one positive-weight draw using a prepared fixed-universe graph."""
    weights = np.asarray(image_weights, dtype=float)
    n_images = prepared["n_images"]
    if weights.shape != (n_images,) or not np.isfinite(weights).all() or np.any(weights <= 0):
        raise ValueError("fixed-universe image weights must be finite and strictly positive")
    thresholds = prepared["thresholds"]
    rec_thrs = prepared["rec_thrs"]
    recalls = len(rec_thrs)
    precision = -np.ones((thresholds, recalls, prepared["categories"], prepared["areas"]))
    for category, category_streams in enumerate(prepared["streams"]):
        for area, stream in enumerate(category_streams):
            if stream is None:
                continue
            positives = float(sum(weights[position] * count for position, count in enumerate(stream["gt_counts"])))
            if positives <= 0:
                raise ValueError("fixed category universe lost all positive weight")
            detection_weights = weights[stream["detection_images"]]
            true_positive = np.cumsum(
                np.logical_and(stream["matches"], np.logical_not(stream["ignored"])) * detection_weights,
                axis=1,
            )
            false_positive = np.cumsum(
                np.logical_and(np.logical_not(stream["matches"]), np.logical_not(stream["ignored"]))
                * detection_weights,
                axis=1,
            )
            for threshold in range(thresholds):
                tp, fp = true_positive[threshold], false_positive[threshold]
                recall = tp / positives
                curve = tp / (tp + fp + np.spacing(1))
                curve = np.maximum.accumulate(curve[::-1])[::-1]
                sampled = np.zeros(recalls)
                locations = np.searchsorted(recall, rec_thrs, side="left")
                valid = locations < len(curve)
                sampled[valid] = curve[locations[valid]]
                precision[threshold, :, category, area] = sampled
    result = []
    for area in range(prepared["areas"]):
        values = precision[:, :, :, area]
        result.append(float(values[values > -1].mean()) if (values > -1).any() else float("nan"))
    return np.asarray(result)


def bayesian_image_weights(n_images: int, *, n_boot: int, seed: int) -> np.ndarray:
    """Materialize deterministic positive weights normalized to mean one."""
    if n_images <= 0 or n_boot <= 0:
        raise ValueError("Bayesian bootstrap dimensions must be positive")
    rng = np.random.default_rng(seed)
    weights = rng.exponential(scale=1.0, size=(n_boot, n_images))
    weights *= n_images / weights.sum(axis=1, keepdims=True)
    return weights
