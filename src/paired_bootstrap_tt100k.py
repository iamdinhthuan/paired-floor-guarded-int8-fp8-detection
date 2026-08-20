#!/usr/bin/env python3
"""Paired TT100K bootstrap with original-pixel height small-like versus large-like strata."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np
from paired_bootstrap import accumulate_ap
from topic_c.manifest import sha256_file
from topic_c.tt100k_height import HEIGHT_GROUPS, height_evaluation


def percentile(values: np.ndarray) -> list[float]:
    values = values[np.isfinite(values)]
    return [float(value) for value in np.percentile(values, [2.5, 50, 97.5])] if len(values) else [float("nan")] * 3


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--annotations", required=True); parser.add_argument("--fp32-clean", required=True); parser.add_argument("--quant-clean", required=True)
    parser.add_argument("--fp32-corrupt", required=True); parser.add_argument("--quant-corrupt", required=True)
    parser.add_argument("--fp32-clean-input", required=True); parser.add_argument("--quant-clean-input", required=True)
    parser.add_argument("--fp32-corrupt-input", required=True); parser.add_argument("--quant-corrupt-input", required=True)
    parser.add_argument("--quant-label", required=True); parser.add_argument("--n-boot", type=int, default=500); parser.add_argument("--seed", type=int, required=True); parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit(f"TT100K BOOTSTRAP REFUSED: output already exists: {output}")
    input_paths = [args.fp32_clean_input, args.quant_clean_input, args.fp32_corrupt_input, args.quant_corrupt_input]
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in input_paths]
    image_ids = records[0].get("image_ids")
    if not image_ids or any(record.get("image_ids") != image_ids for record in records[1:]) or len(image_ids) != len(set(image_ids)):
        raise SystemExit("TT100K BOOTSTRAP REFUSED: four linked cells must contain the same ordered unique image IDs")
    annotation_data = json.loads(Path(args.annotations).read_text(encoding="utf-8"))
    prediction_paths = [args.fp32_clean, args.quant_clean, args.fp32_corrupt, args.quant_corrupt]
    predictions = [json.loads(Path(path).read_text(encoding="utf-8")) for path in prediction_paths]
    group_bins = [(name, lower, upper) for name, (lower, upper) in HEIGHT_GROUPS.items()]
    evaluations = [height_evaluation(annotation_data, prediction, image_ids, group_bins) for prediction in predictions]
    full = list(range(len(image_ids)))
    point, draws = {}, []
    point_ap = [accumulate_ap(value, full) for value in evaluations]
    for index, name in enumerate(HEIGHT_GROUPS):
        ap = [float(value[index]) for value in point_ap]
        q_clean, q_corrupt = ap[0] - ap[1], ap[2] - ap[3]
        point[name] = {"q_clean": q_clean, "q_corrupt": q_corrupt, "excess": q_corrupt - q_clean}
    rng = np.random.default_rng(args.seed)
    for draw_index in range(args.n_boot):
        sample = rng.choice(len(image_ids), size=len(image_ids), replace=True).tolist()
        sampled_ap = [accumulate_ap(value, sample) for value in evaluations]
        excess = {}
        for index, name in enumerate(HEIGHT_GROUPS):
            ap = [float(value[index]) for value in sampled_ap]
            excess[name] = (ap[2] - ap[3]) - (ap[0] - ap[1])
        draws.append(excess)
        if (draw_index + 1) % 25 == 0 or draw_index + 1 == args.n_boot:
            print(f"TT100K PAIRED BOOTSTRAP {draw_index + 1}/{args.n_boot}", flush=True)
    psi = np.asarray([draw["small_like"] - draw["large_like"] for draw in draws])
    result = {"schema_version": 1, "method": "paired image bootstrap; shared resample across FP32/quant × clean/corrupt; original bbox-height strata",
              "n_images": len(image_ids), "n_boot": args.n_boot, "seed": args.seed, "quant_label": args.quant_label,
              "height_bins_px": {name: {"min": bounds[0], "max": None if np.isinf(bounds[1]) else bounds[1]} for name, bounds in HEIGHT_GROUPS.items()},
              "input_hashes": {name: {"prediction_sha256": sha256_file(prediction), "input_record_sha256": sha256_file(input_path),
                                      "input_manifest_sha256": record["input_manifest_sha256"], "image_ids_sha256": record["image_ids_sha256"]}
                               for name, prediction, input_path, record in zip(("fp32_clean", "quant_clean", "fp32_corrupt", "quant_corrupt"), prediction_paths, input_paths, records)},
              "point": {**point, "psi_small_like_minus_large_like": point["small_like"]["excess"] - point["large_like"]["excess"]},
              "ci95_excess": {name: percentile(np.asarray([draw[name] for draw in draws])) for name in HEIGHT_GROUPS},
              "ci95_psi_small_like_minus_large_like": percentile(psi)}
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(result, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "psi": result["point"]["psi_small_like_minus_large_like"], "ci95": result["ci95_psi_small_like_minus_large_like"]}, indent=2))


if __name__ == "__main__":
    main()
