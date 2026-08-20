#!/usr/bin/env python3
"""Build the hash-bound untouched-holdout confirmatory analysis."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from pilot_registry import canonical_hash
from run_confirmatory_bootstrap import (
    DATASETS,
    dataset_seed,
    validate_bootstrap_completion,
    validate_inference_completion,
)
from topic_c.manifest import sha256_file


BINS = ("all", "small", "medium", "large")
MODELS = ("yolo11n", "yolo11m", "yolo11x")
CORRUPTIONS = ("gaussian_noise", "motion_blur", "fog", "jpeg")
SEVERITIES = (1, 3, 5)


def percentile(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 50, 97.5])]


def direct_contrast(
    *,
    int8_point: np.ndarray,
    fp8_point: np.ndarray,
    int8_draws: np.ndarray,
    fp8_draws: np.ndarray,
) -> dict:
    if int8_point.shape != (4,) or fp8_point.shape != (4,):
        raise ValueError("direct point endpoint shape mismatch")
    if int8_draws.shape != fp8_draws.shape or int8_draws.ndim != 2 or int8_draws.shape[1] != 4:
        raise ValueError("direct draw endpoint shape mismatch")
    point = int8_point - fp8_point
    draws = int8_draws - fp8_draws
    return {
        "point": point,
        "draws": draws,
        "psi_point": float(point[1] - point[3]),
        "psi_draws": draws[:, 1] - draws[:, 3],
    }


def summarize_cells(cells: list[dict]) -> dict:
    if not cells:
        raise ValueError("cannot summarize an empty cell set")
    delta_e = np.asarray([cell["delta_e"] for cell in cells], dtype=np.float64)
    delta_psi = np.asarray([cell["delta_psi"] for cell in cells], dtype=np.float64)
    delta_e_draws = np.stack([cell["delta_e_draws"] for cell in cells]).mean(axis=0)
    delta_psi_draws = np.stack([cell["delta_psi_draws"] for cell in cells]).mean(axis=0)
    return {
        "n_cells": len(cells),
        "delta_e_point": float(delta_e.mean()),
        "delta_e_percentile95": percentile(delta_e_draws),
        "delta_psi_point": float(delta_psi.mean()),
        "delta_psi_percentile95": percentile(delta_psi_draws),
        "delta_e_min": float(delta_e.min()),
        "delta_e_max": float(delta_e.max()),
        "delta_e_positive_cells": int(np.count_nonzero(delta_e > 0)),
        "delta_e_negative_cells": int(np.count_nonzero(delta_e < 0)),
        "delta_e_draws": delta_e_draws,
        "delta_psi_draws": delta_psi_draws,
    }


def artifact_index(root: Path, bootstrap_report: dict) -> dict[str, Path]:
    result = {}
    for relative in bootstrap_report["artifacts_sha256"]:
        path = (root / relative).resolve()
        result[path.name] = path
    if len(result) != 72:
        raise SystemExit("CONFIRMATORY ANALYSIS REFUSED: bootstrap basename collision")
    return result


def load_bootstrap_cell(path: Path, *, n_boot: int, seed: int) -> tuple[dict, np.ndarray]:
    document = json.loads(path.read_text(encoding="utf-8"))
    cache_record = document.get("draw_cache", {})
    cache = Path(cache_record.get("path", "")).resolve()
    if (
        document.get("n_boot") != n_boot
        or document.get("seed") != seed
        or not cache.is_file()
        or sha256_file(cache) != cache_record.get("sha256")
    ):
        raise SystemExit(f"CONFIRMATORY ANALYSIS REFUSED: invalid bootstrap cell: {path}")
    with np.load(cache, allow_pickle=False) as loaded:
        excess = np.asarray(loaded["excess"], dtype=np.float64)
        cached_n_boot = int(loaded["n_boot"])
        cached_seed = int(loaded["seed"])
    if excess.shape != (n_boot, 4) or cached_n_boot != n_boot or cached_seed != seed:
        raise SystemExit(f"CONFIRMATORY ANALYSIS REFUSED: invalid draw cache: {cache}")
    image_hashes = {
        record.get("image_ids_sha256")
        for record in document.get("input_hashes", {}).values()
        if isinstance(record, dict)
    }
    if len(image_hashes) != 1 or None in image_hashes:
        raise SystemExit(f"CONFIRMATORY ANALYSIS REFUSED: unpaired image universe: {path}")
    return document, excess


def metric_index(root: Path, inference_row: dict) -> dict[tuple, dict]:
    plan = json.loads(Path(inference_row["plan"]).read_text(encoding="utf-8"))
    evaluation = json.loads(
        Path(inference_row["evaluation_report"]).read_text(encoding="utf-8")
    )
    metric_hashes = evaluation.get("metric_sha256", {})
    if len(metric_hashes) != 117:
        raise SystemExit("CONFIRMATORY ANALYSIS REFUSED: incomplete metric ledger")
    result = {}
    for run in plan["runs"]:
        condition = run["condition_id"]
        metric_path = root / "outputs" / "metrics" / inference_row["attempt"] / f"{condition}.json"
        if not metric_path.is_file() or sha256_file(metric_path) != metric_hashes.get(condition):
            raise SystemExit(
                f"CONFIRMATORY ANALYSIS REFUSED: metric hash mismatch: {condition}"
            )
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        if metric.get("condition_id") != condition:
            raise SystemExit(
                f"CONFIRMATORY ANALYSIS REFUSED: metric identity mismatch: {condition}"
            )
        key = (run["model"], run["precision"], run["corruption"], run["severity"])
        if key in result:
            raise SystemExit("CONFIRMATORY ANALYSIS REFUSED: duplicate metric condition")
        result[key] = metric
    return result


def analyze_dataset(
    root: Path,
    *,
    dataset: str,
    inference_row: dict,
    bootstrap_report: dict,
    n_boot: int,
    seed: int,
) -> tuple[list[dict], list[dict]]:
    artifacts = artifact_index(root, bootstrap_report)
    metrics = metric_index(root, inference_row)
    cells = []
    public_rows = []
    clean_by_model = {}
    for model in MODELS:
        for corruption in CORRUPTIONS:
            for severity in SEVERITIES:
                documents = {}
                draws = {}
                for precision in ("int8-entropy", "fp8"):
                    name = f"{model}__{precision}__{corruption}-s{severity}.json"
                    documents[precision], draws[precision] = load_bootstrap_cell(
                        artifacts[name], n_boot=n_boot, seed=seed
                    )
                int8 = documents["int8-entropy"]
                fp8 = documents["fp8"]
                if {
                    item["image_ids_sha256"]
                    for document in (int8, fp8)
                    for item in document["input_hashes"].values()
                }.__len__() != 1:
                    raise SystemExit(
                        f"CONFIRMATORY ANALYSIS REFUSED: cross-format image mismatch: {dataset}/{model}"
                    )
                contrast = direct_contrast(
                    int8_point=np.asarray(
                        [int8["point"]["excess"][name] for name in BINS]
                    ),
                    fp8_point=np.asarray(
                        [fp8["point"]["excess"][name] for name in BINS]
                    ),
                    int8_draws=draws["int8-entropy"],
                    fp8_draws=draws["fp8"],
                )
                clean_gap = (
                    int8["point"]["q_clean"]["all"] - fp8["point"]["q_clean"]["all"]
                )
                clean_by_model.setdefault(model, []).append(clean_gap)
                int8_clean = metrics[(model, "int8-entropy", "clean", 0)]["stats"]["AP"]
                fp8_clean = metrics[(model, "fp8", "clean", 0)]["stats"]["AP"]
                int8_corrupt = metrics[(model, "int8-entropy", corruption, severity)]["stats"]["AP"]
                fp8_corrupt = metrics[(model, "fp8", corruption, severity)]["stats"]["AP"]
                internal = {
                    "dataset": dataset,
                    "model": model,
                    "corruption": corruption,
                    "severity": severity,
                    "delta_e": float(contrast["point"][0]),
                    "delta_psi": contrast["psi_point"],
                    "delta_e_draws": contrast["draws"][:, 0],
                    "delta_psi_draws": contrast["psi_draws"],
                }
                cells.append(internal)
                public_rows.append(
                    {
                        key: value
                        for key, value in internal.items()
                        if not key.endswith("_draws")
                    }
                    | {
                        "delta_e_percentile95": percentile(internal["delta_e_draws"]),
                        "delta_psi_percentile95": percentile(internal["delta_psi_draws"]),
                        "matched_clean_fp8_minus_int8": float(fp8_clean - int8_clean),
                        "int8_clean_ap": float(int8_clean),
                        "fp8_clean_ap": float(fp8_clean),
                        "int8_corrupted_ap": float(int8_corrupt),
                        "fp8_corrupted_ap": float(fp8_corrupt),
                    }
                )
    for model, values in clean_by_model.items():
        if max(values) - min(values) > 1e-12:
            raise SystemExit(
                f"CONFIRMATORY ANALYSIS REFUSED: clean gap changed across cells: {dataset}/{model}"
            )
    return cells, public_rows


def json_summary(summary: dict) -> dict:
    return {key: value for key, value in summary.items() if not key.endswith("_draws")}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--wait-for-bootstrap-report", required=True)
    parser.add_argument("--wait-for-inference-report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    bootstrap_completion_path = Path(args.wait_for_bootstrap_report).resolve()
    inference_path = Path(args.wait_for_inference_report).resolve()
    output = Path(args.out).resolve()
    if output.exists() or output.with_suffix(output.suffix + ".complete").exists():
        raise SystemExit(f"CONFIRMATORY ANALYSIS REFUSED: output exists: {output}")
    while not bootstrap_completion_path.is_file() or not inference_path.is_file():
        print("CONFIRMATORY ANALYSIS waiting for bootstrap and inference reports", flush=True)
        time.sleep(args.poll_seconds)
    bootstrap_completion = json.loads(
        bootstrap_completion_path.read_text(encoding="utf-8")
    )
    if bootstrap_completion.get("bootstrap_report_sha256") != canonical_hash(
        bootstrap_completion, "bootstrap_report_sha256"
    ):
        raise SystemExit("CONFIRMATORY ANALYSIS REFUSED: bootstrap completion hash mismatch")
    attempt = bootstrap_completion["attempt"]
    inference = validate_inference_completion(inference_path, attempt=attempt)
    bootstrap_rows = {row["dataset"]: row for row in bootstrap_completion["datasets"]}
    if set(bootstrap_rows) != set(DATASETS):
        raise SystemExit("CONFIRMATORY ANALYSIS REFUSED: bootstrap dataset grid mismatch")
    all_cells = []
    public_cells = []
    dataset_summaries = {}
    evidence = []
    for dataset in DATASETS:
        row = bootstrap_rows[dataset]
        report_path = Path(row["bootstrap_report"]).resolve()
        if sha256_file(report_path) != row["bootstrap_report_sha256"]:
            raise SystemExit(
                f"CONFIRMATORY ANALYSIS REFUSED: bootstrap report hash mismatch: {dataset}"
            )
        report = validate_bootstrap_completion(
            root,
            report_path,
            attempt=inference[dataset]["attempt"],
            plan_sha256=inference[dataset]["plan_sha256"],
            n_boot=row["n_boot"],
            shared_seed=row["shared_seed"],
        )
        cells, rows = analyze_dataset(
            root,
            dataset=dataset,
            inference_row=inference[dataset],
            bootstrap_report=report,
            n_boot=row["n_boot"],
            seed=row["shared_seed"],
        )
        all_cells.extend(cells)
        public_cells.extend(rows)
        dataset_summaries[dataset] = json_summary(summarize_cells(cells))
        evidence.append(
            {"dataset": dataset, "report": str(report_path), "sha256": sha256_file(report_path)}
        )
    if len(all_cells) != 72:
        raise SystemExit("CONFIRMATORY ANALYSIS REFUSED: exact 72-cell direct grid absent")
    overall = summarize_cells(all_cells)
    corrupted = [
        value
        for row in public_cells
        for value in (row["int8_corrupted_ap"], row["fp8_corrupted_ap"])
    ]
    clean_blocks = {
        (row["dataset"], row["model"]): row["matched_clean_fp8_minus_int8"]
        for row in public_cells
    }
    document = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": attempt,
        "estimand": "(FP8-INT8)_corrupt - (FP8-INT8)_matched-clean",
        "units": {"stored_ap": "native fraction", "ap_points_multiplier": 100},
        "scope": {
            "datasets": list(DATASETS),
            "models": list(MODELS),
            "corruptions": list(CORRUPTIONS),
            "severities": list(SEVERITIES),
            "direct_cells": 72,
            "partition_role": "untouched final holdout",
        },
        "overall_balanced_equal_cell": json_summary(overall),
        "by_dataset": dataset_summaries,
        "matched_clean": {
            "blocks": 6,
            "fp8_minus_int8_mean": float(np.mean(list(clean_blocks.values()))),
            "fp8_minus_int8_min": float(np.min(list(clean_blocks.values()))),
            "fp8_minus_int8_max": float(np.max(list(clean_blocks.values()))),
            "positive_blocks": int(np.count_nonzero(np.asarray(list(clean_blocks.values())) > 0)),
        },
        "absolute_corrupted_ap_guardrail": {
            "format_arms": len(corrupted),
            "median": float(np.median(corrupted)),
            "mean": float(np.mean(corrupted)),
            "below_5_ap_points": int(np.count_nonzero(np.asarray(corrupted) < 0.05)),
            "below_10_ap_points": int(np.count_nonzero(np.asarray(corrupted) < 0.10)),
        },
        "cells": public_cells,
        "evidence": evidence,
        "bootstrap_completion": str(bootstrap_completion_path),
        "bootstrap_completion_sha256": sha256_file(bootstrap_completion_path),
        "inference_completion": str(inference_path),
        "inference_completion_sha256": sha256_file(inference_path),
    }
    document["analysis_sha256"] = canonical_hash(document, "analysis_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".complete").write_text(
        sha256_file(output) + "\n", encoding="utf-8"
    )
    print(f"CONFIRMATORY ANALYSIS COMPLETE cells=72 output={output}")


if __name__ == "__main__":
    main()
