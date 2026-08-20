#!/usr/bin/env python3
"""Paired COCO bootstrap for Q, E, and Psi using shared image resamples."""
from __future__ import annotations

import argparse
import copy
import hashlib
import json
from pathlib import Path

import numpy as np
try:
    from pycocotools.coco import COCO
    from pycocotools.cocoeval import COCOeval
except ModuleNotFoundError:  # Draw-cache helpers remain importable without the evaluator extra.
    COCO = None
    COCOeval = None

from topic_c.manifest import sha256_file

BINS = [("all", 0, 1e10), ("small", 0, 32 ** 2), ("medium", 32 ** 2, 96 ** 2), ("large", 96 ** 2, 1e10)]


def build_eval(gt: COCO, predictions: list[dict], image_ids: list[int]) -> COCOeval:
    evaluation = COCOeval(gt, gt.loadRes(copy.deepcopy(predictions)), iouType="bbox")
    evaluation.params.imgIds = image_ids
    evaluation.params.areaRng = [[low, high] for _, low, high in BINS]
    evaluation.params.areaRngLbl = [name for name, _, _ in BINS]
    evaluation.params.maxDets = [100]
    evaluation.evaluate()
    return evaluation


def accumulate_ap(evaluation: COCOeval, image_positions: list[int]) -> np.ndarray:
    """Stock-equivalent AP over an image resample (duplicates are preserved)."""
    params = evaluation.params
    thresholds, recalls = len(params.iouThrs), len(params.recThrs)
    categories, areas, n_images = len(params.catIds), len(params.areaRng), len(evaluation._paramsEval.imgIds)
    precision = -np.ones((thresholds, recalls, categories, areas))
    for category in range(categories):
        base_category = category * areas * n_images
        for area in range(areas):
            base_area = base_category + area * n_images
            entries = [evaluation.evalImgs[base_area + position] for position in image_positions]
            entries = [entry for entry in entries if entry is not None]
            if not entries:
                continue
            scores = np.concatenate([entry["dtScores"][:100] for entry in entries])
            order = np.argsort(-scores, kind="mergesort")
            matches = np.concatenate([entry["dtMatches"][:, :100] for entry in entries], axis=1)[:, order]
            ignored_detections = np.concatenate([entry["dtIgnore"][:, :100] for entry in entries], axis=1)[:, order]
            ignored_gt = np.concatenate([entry["gtIgnore"] for entry in entries])
            positives = np.count_nonzero(ignored_gt == 0)
            if positives == 0:
                continue
            true_positive = np.cumsum(np.logical_and(matches, np.logical_not(ignored_detections)), axis=1)
            false_positive = np.cumsum(np.logical_and(np.logical_not(matches), np.logical_not(ignored_detections)), axis=1)
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
        result.append(float(values[values > -1].mean()) if (values > -1).any() else float("nan"))
    return np.asarray(result)


def read_input(path: str) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def percentile(values: np.ndarray) -> list[float]:
    values = values[np.isfinite(values)]
    return [float(item) for item in np.percentile(values, [2.5, 50, 97.5])] if len(values) else [float("nan")] * 3


def write_draw_cache(
    path: Path,
    *,
    excess: np.ndarray,
    psi: np.ndarray,
    n_boot: int,
    seed: int,
) -> dict:
    path = path.resolve()
    if path.exists():
        raise SystemExit(f"refusing to overwrite bootstrap draw cache: {path}")
    if excess.shape != (n_boot, len(BINS)) or psi.shape != (n_boot,):
        raise ValueError("bootstrap draw-cache shape mismatch")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        excess=np.asarray(excess, dtype=np.float64),
        psi=np.asarray(psi, dtype=np.float64),
        n_boot=np.asarray(n_boot, dtype=np.int64),
        seed=np.asarray(seed, dtype=np.int64),
    )
    return {
        "path": str(path),
        "sha256": sha256_file(path),
        "excess_shape": list(excess.shape),
        "psi_shape": list(psi.shape),
        "dtype": "float64",
    }


def main() -> None:
    if COCO is None or COCOeval is None:
        raise SystemExit("paired bootstrap requires pycocotools")
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--fp32-clean", required=True)
    parser.add_argument("--quant-clean", required=True)
    parser.add_argument("--fp32-corrupt", required=True)
    parser.add_argument("--quant-corrupt", required=True)
    parser.add_argument("--fp32-clean-input", required=True)
    parser.add_argument("--quant-clean-input", required=True)
    parser.add_argument("--fp32-corrupt-input", required=True)
    parser.add_argument("--quant-corrupt-input", required=True)
    parser.add_argument("--quant-label", required=True)
    parser.add_argument("--n-boot", type=int, default=100)
    parser.add_argument("--seed", type=int, default=20260806)
    parser.add_argument("--draw-cache")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit(f"refusing to overwrite bootstrap artifact: {output}")
    input_paths = [args.fp32_clean_input, args.quant_clean_input, args.fp32_corrupt_input, args.quant_corrupt_input]
    input_records = [read_input(path) for path in input_paths]
    ids = input_records[0]["image_ids"]
    if any(record["image_ids"] != ids for record in input_records[1:]):
        raise SystemExit("four linked cells do not have identical ordered image IDs")
    if len(ids) != len(set(ids)):
        raise SystemExit("bootstrap image IDs are duplicated before resampling")
    gt = COCO(args.annotations)
    prediction_paths = [args.fp32_clean, args.quant_clean, args.fp32_corrupt, args.quant_corrupt]
    predictions = [json.loads(Path(path).read_text(encoding="utf-8")) for path in prediction_paths]
    evaluations = [build_eval(gt, prediction, ids) for prediction in predictions]
    full = list(range(len(ids)))
    point = [accumulate_ap(evaluation, full) for evaluation in evaluations]
    rng = np.random.default_rng(args.seed)
    draws = []
    for draw_index in range(args.n_boot):
        sample = rng.choice(len(ids), size=len(ids), replace=True).tolist()
        fp32_clean, quant_clean, fp32_corrupt, quant_corrupt = [accumulate_ap(evaluation, sample) for evaluation in evaluations]
        q_clean, q_corrupt = fp32_clean - quant_clean, fp32_corrupt - quant_corrupt
        excess = q_corrupt - q_clean
        draws.append({"q_clean": q_clean, "q_corrupt": q_corrupt, "excess": excess, "psi": excess[1] - excess[3]})
        if (draw_index + 1) % 25 == 0 or draw_index + 1 == args.n_boot:
            print(f"PAIRED BOOTSTRAP {draw_index + 1}/{args.n_boot}", flush=True)
    def bins(values: np.ndarray) -> dict[str, list[float]]:
        return {name: percentile(values[:, index]) for index, (name, _, _) in enumerate(BINS)}
    draw_excess = np.asarray([draw["excess"] for draw in draws])
    draw_psi = np.asarray([draw["psi"] for draw in draws])
    point_q_clean, point_q_corrupt = point[0] - point[1], point[2] - point[3]
    point_excess = point_q_corrupt - point_q_clean
    draw_cache = None
    if args.draw_cache:
        draw_cache = write_draw_cache(
            Path(args.draw_cache),
            excess=draw_excess,
            psi=draw_psi,
            n_boot=args.n_boot,
            seed=args.seed,
        )
    result = {
        "schema_version": 1,
        "method": "paired image bootstrap; one shared resample across FP32/quant × clean/corrupt",
        "n_images": len(ids), "n_boot": args.n_boot, "seed": args.seed, "quant_label": args.quant_label,
        "input_hashes": {name: {"prediction_sha256": sha256_file(prediction), "input_record_sha256": sha256_file(input_record), "input_manifest_sha256": record["input_manifest_sha256"], "image_ids_sha256": record["image_ids_sha256"]} for name, prediction, input_record, record in zip(("fp32_clean", "quant_clean", "fp32_corrupt", "quant_corrupt"), prediction_paths, input_paths, input_records)},
        "point": {"q_clean": {name: float(point_q_clean[index]) for index, (name, _, _) in enumerate(BINS)}, "q_corrupt": {name: float(point_q_corrupt[index]) for index, (name, _, _) in enumerate(BINS)}, "excess": {name: float(point_excess[index]) for index, (name, _, _) in enumerate(BINS)}, "psi_small_minus_large": float(point_excess[1] - point_excess[3])},
        "ci95_excess": bins(draw_excess), "ci95_psi_small_minus_large": percentile(draw_psi),
    }
    if draw_cache is not None:
        result["draw_cache"] = draw_cache
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "psi": result["point"]["psi_small_minus_large"], "ci95": result["ci95_psi_small_minus_large"]}, indent=2))


if __name__ == "__main__":
    main()
