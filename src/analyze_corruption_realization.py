#!/usr/bin/env python3
"""Analyze paired image and corruption-realization uncertainty."""
from __future__ import annotations

import argparse
import json
import time
from collections.abc import Sequence
from datetime import datetime, timezone
from pathlib import Path

import numpy as np

from bootstrap_format_contrast import component_draw_cache_identity, load_draw_cache
from pilot_registry import canonical_hash
from topic_c.manifest import sha256_file


DATASETS = ("voc", "kitti", "tt100k")
CORRUPTIONS = ("gaussian_noise", "motion_blur", "fog", "jpeg")
SEVERITIES = (1, 3, 5)


def realization_schedule(*, n_boot: int, n_realizations: int, seed: int) -> np.ndarray:
    if n_boot < 1 or n_realizations < 2 or seed < 0:
        raise ValueError("invalid nested realization schedule parameters")
    return np.random.default_rng(seed).integers(
        0, n_realizations, size=(n_boot, n_realizations), dtype=np.int64
    )


def _ordered_cells(cells: Sequence[dict], expected_seeds: tuple[int, ...]) -> list[dict]:
    by_seed = {cell.get("realization_seed"): cell for cell in cells}
    if len(cells) != len(expected_seeds) or set(by_seed) != set(expected_seeds):
        raise ValueError("condition does not contain each expected realization seed exactly once")
    return [by_seed[seed] for seed in expected_seeds]


def decompose_condition(
    cells: Sequence[dict],
    *,
    expected_seeds: tuple[int, ...],
    point_key: str = "delta_e",
    draws_key: str = "delta_e_draws",
) -> dict:
    """Return a descriptive two-source variance decomposition for one base condition."""
    ordered = _ordered_cells(cells, expected_seeds)
    points = np.asarray([cell[point_key] for cell in ordered], dtype=np.float64)
    draws = np.stack(
        [np.asarray(cell[draws_key], dtype=np.float64) for cell in ordered]
    )
    if draws.ndim != 2 or draws.shape[1] < 2 or not np.isfinite(points).all() or not np.isfinite(draws).all():
        raise ValueError("condition contains invalid point estimates or image-bootstrap draws")
    between = float(np.var(points, ddof=1))
    within = float(np.mean(np.var(draws, axis=1, ddof=1)))
    total = between + within
    return {
        "n_realizations": len(expected_seeds),
        "n_image_bootstrap": int(draws.shape[1]),
        f"mean_{point_key}": float(points.mean()),
        "between_realization_variance": between,
        "within_image_variance": within,
        "descriptive_total_variance": total,
        "between_realization_share": float(between / total) if total > 0 else 0.0,
    }


def nested_condition_draws(
    cells: Sequence[dict],
    *,
    expected_seeds: tuple[int, ...],
    realization_indices: np.ndarray,
    draws_key: str = "delta_e_draws",
) -> np.ndarray:
    """Resample realizations while retaining each replicate's paired image draw index."""
    ordered = _ordered_cells(cells, expected_seeds)
    image_draws = np.stack(
        [np.asarray(cell[draws_key], dtype=np.float64) for cell in ordered]
    )
    indices = np.asarray(realization_indices)
    n_realizations, n_boot = image_draws.shape
    if (
        indices.shape != (n_boot, n_realizations)
        or not np.issubdtype(indices.dtype, np.integer)
        or np.any(indices < 0)
        or np.any(indices >= n_realizations)
    ):
        raise ValueError("realization resample schedule shape or index is invalid")
    return np.asarray(
        [image_draws[indices[b], b].mean() for b in range(n_boot)], dtype=np.float64
    )


def validate_exact_grid(
    cells: Sequence[dict], *, expected_seeds: tuple[int, ...], n_boot: int
) -> None:
    expected = {
        (dataset, corruption, severity, seed)
        for dataset in DATASETS
        for corruption in CORRUPTIONS
        for severity in SEVERITIES
        for seed in expected_seeds
    }
    observed = {
        (
            cell.get("dataset"),
            cell.get("corruption"),
            cell.get("severity"),
            cell.get("realization_seed"),
        )
        for cell in cells
    }
    if len(cells) != 108 or len(observed) != 108 or observed != expected:
        raise ValueError("exact 108-cell realization grid is absent")
    if any(np.asarray(cell.get("delta_e_draws")).shape != (n_boot,) for cell in cells):
        raise ValueError(f"every realization cell must contain exactly B={n_boot} image draws")


def summarize_grid(
    cells: Sequence[dict],
    *,
    expected_seeds: tuple[int, ...],
    realization_indices: np.ndarray,
    point_key: str = "delta_e",
    draws_key: str = "delta_e_draws",
) -> dict:
    """Equal-weight the 36 base conditions after nested two-source resampling."""
    indices = np.asarray(realization_indices)
    if indices.ndim != 2:
        raise ValueError("realization resample schedule must be two-dimensional")
    n_boot = indices.shape[0]
    validate_exact_grid(cells, expected_seeds=expected_seeds, n_boot=n_boot)
    groups: dict[tuple[str, str, int], list[dict]] = {}
    for cell in cells:
        key = (cell["dataset"], cell["corruption"], cell["severity"])
        groups.setdefault(key, []).append(cell)
    if len(groups) != 36:
        raise ValueError("exact 36-condition base grid is absent")
    decompositions = [
        decompose_condition(
            group,
            expected_seeds=expected_seeds,
            point_key=point_key,
            draws_key=draws_key,
        )
        for group in groups.values()
    ]
    nested = np.stack(
        [
            nested_condition_draws(
                group,
                expected_seeds=expected_seeds,
                realization_indices=indices,
                draws_key=draws_key,
            )
            for group in groups.values()
        ]
    ).mean(axis=0)
    return {
        "n_realization_cells": len(cells),
        "n_base_conditions": len(groups),
        f"mean_{point_key}": float(
            np.mean([item[f"mean_{point_key}"] for item in decompositions])
        ),
        "mean_between_realization_variance": float(
            np.mean([item["between_realization_variance"] for item in decompositions])
        ),
        "mean_within_image_variance": float(
            np.mean([item["within_image_variance"] for item in decompositions])
        ),
        "nested_macro_draws": nested,
    }


def _percentile(values: np.ndarray) -> list[float]:
    return [float(value) for value in np.percentile(values, [2.5, 50.0, 97.5])]


def _validated_cells(root: Path, completion: dict) -> list[dict]:
    n_boot = completion.get("n_boot")
    records = completion.get("artifacts")
    if n_boot != 2000 or not isinstance(records, list) or len(records) != 108:
        raise SystemExit("REALIZATION ANALYSIS REFUSED: incomplete bootstrap artifact grid")
    cells = []
    for record in records:
        component = Path(record.get("component", "")).resolve()
        cache = Path(record.get("draw_cache", "")).resolve()
        try:
            component.relative_to(root)
            cache.relative_to(root)
        except ValueError as exc:
            raise SystemExit("REALIZATION ANALYSIS REFUSED: evidence path escapes root") from exc
        if (
            not component.is_file()
            or not cache.is_file()
            or sha256_file(component) != record.get("component_sha256")
            or sha256_file(cache) != record.get("draw_cache_sha256")
        ):
            raise SystemExit(f"REALIZATION ANALYSIS REFUSED: artifact hash mismatch: {component}")
        document = json.loads(component.read_text(encoding="utf-8"))
        cache_binding = document.get("temporary_draw_cache", {})
        schedule = document.get("bootstrap_schedule", {})
        if (
            document.get("artifact_sha256") != canonical_hash(document, "artifact_sha256")
            or document.get("n_boot") != n_boot
            or cache_binding.get("sha256") != record["draw_cache_sha256"]
        ):
            raise SystemExit(f"REALIZATION ANALYSIS REFUSED: invalid component: {component}")
        draws = load_draw_cache(
            cache,
            expected_sha256=record["draw_cache_sha256"],
            expected_identity_sha256=component_draw_cache_identity(document),
            n_boot=n_boot,
            schedule_sha256=schedule.get("sha256"),
        )
        point = float(document["point"]["delta_e"]["all"])
        cells.append(
            {
                "dataset": record["dataset"],
                "corruption": record["corruption"],
                "severity": record["severity"],
                "realization_seed": record["realization_seed"],
                "delta_e": point,
                "delta_e_draws": draws["delta_e_all"],
                "delta_psi": float(document["point"]["delta_psi"]),
                "delta_psi_draws": draws["delta_psi"],
                "component_sha256": record["component_sha256"],
                "draw_cache_sha256": record["draw_cache_sha256"],
            }
        )
    return cells


def _dataset_summary(
    cells: list[dict], expected_seeds: tuple[int, ...], indices: np.ndarray,
    *, point_key: str = "delta_e", draws_key: str = "delta_e_draws",
) -> dict:
    groups: dict[tuple[str, int], list[dict]] = {}
    for cell in cells:
        groups.setdefault((cell["corruption"], cell["severity"]), []).append(cell)
    nested = np.stack(
        [
            nested_condition_draws(
                group, expected_seeds=expected_seeds, realization_indices=indices,
                draws_key=draws_key,
            )
            for group in groups.values()
        ]
    ).mean(axis=0)
    decompositions = [
        decompose_condition(
            group, expected_seeds=expected_seeds, point_key=point_key, draws_key=draws_key
        )
        for group in groups.values()
    ]
    return {
        "n_base_conditions": len(groups),
        f"mean_{point_key}": float(np.mean([row[f"mean_{point_key}"] for row in decompositions])),
        "nested_percentile95": _percentile(nested),
        "mean_between_realization_variance": float(
            np.mean([row["between_realization_variance"] for row in decompositions])
        ),
        "mean_within_image_variance": float(
            np.mean([row["within_image_variance"] for row in decompositions])
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--wait-for-bootstrap-report", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--nested-draw-cache", required=True)
    parser.add_argument("--realization-bootstrap-seed", type=int, default=20260819)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    prerequisite = Path(args.wait_for_bootstrap_report).resolve()
    output = Path(args.out).resolve()
    nested_path = Path(args.nested_draw_cache).resolve()
    if output.exists() or nested_path.exists() or output.with_suffix(output.suffix + ".complete").exists():
        raise SystemExit("REALIZATION ANALYSIS REFUSED: output already exists")
    while not prerequisite.is_file():
        print("REALIZATION ANALYSIS waiting for bootstrap completion", flush=True)
        time.sleep(args.poll_seconds)
    marker = prerequisite.with_suffix(prerequisite.suffix + ".complete")
    completion = json.loads(prerequisite.read_text(encoding="utf-8"))
    if (
        not marker.is_file()
        or marker.read_text(encoding="utf-8").strip() != sha256_file(prerequisite)
        or completion.get("completion_sha256") != canonical_hash(completion, "completion_sha256")
    ):
        raise SystemExit("REALIZATION ANALYSIS REFUSED: invalid bootstrap completion")
    seeds = tuple(completion.get("realization_seeds", []))
    if len(seeds) != 3 or len(set(seeds)) != 3:
        raise SystemExit("REALIZATION ANALYSIS REFUSED: invalid realization seeds")
    cells = _validated_cells(root, completion)
    n_boot = completion["n_boot"]
    validate_exact_grid(cells, expected_seeds=seeds, n_boot=n_boot)
    indices = realization_schedule(
        n_boot=n_boot, n_realizations=len(seeds), seed=args.realization_bootstrap_seed
    )
    overall_e = summarize_grid(cells, expected_seeds=seeds, realization_indices=indices)
    overall_psi = summarize_grid(
        cells,
        expected_seeds=seeds,
        realization_indices=indices,
        point_key="delta_psi",
        draws_key="delta_psi_draws",
    )
    nested_draws = overall_e.pop("nested_macro_draws")
    nested_psi_draws = overall_psi.pop("nested_macro_draws")
    nested_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        nested_path,
        schema_version=np.asarray(1),
        n_boot=np.asarray(n_boot),
        seed=np.asarray(args.realization_bootstrap_seed),
        realization_seeds=np.asarray(seeds, dtype=np.int64),
        realization_indices=indices,
        delta_e_macro=nested_draws,
        delta_psi_macro=nested_psi_draws,
    )
    conditions = []
    for dataset in DATASETS:
        for corruption in CORRUPTIONS:
            for severity in SEVERITIES:
                group = [
                    cell
                    for cell in cells
                    if cell["dataset"] == dataset
                    and cell["corruption"] == corruption
                    and cell["severity"] == severity
                ]
                delta_e = decompose_condition(group, expected_seeds=seeds)
                delta_e["nested_percentile95"] = _percentile(
                    nested_condition_draws(
                        group, expected_seeds=seeds, realization_indices=indices
                    )
                )
                delta_psi = decompose_condition(
                    group,
                    expected_seeds=seeds,
                    point_key="delta_psi",
                    draws_key="delta_psi_draws",
                )
                delta_psi["nested_percentile95"] = _percentile(
                    nested_condition_draws(
                        group,
                        expected_seeds=seeds,
                        realization_indices=indices,
                        draws_key="delta_psi_draws",
                    )
                )
                conditions.append(
                    {
                        "dataset": dataset,
                        "corruption": corruption,
                        "severity": severity,
                        "delta_e": delta_e,
                        "delta_psi": delta_psi,
                    }
                )
    public_cells = [
        {
            key: value
            for key, value in cell.items()
            if key not in {"delta_e_draws", "delta_psi_draws"}
        }
        | {"image_bootstrap_percentile95": _percentile(cell["delta_e_draws"])}
        for cell in cells
    ]
    report = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "descriptive nested sensitivity: shared paired image draws within each dataset and nonparametric resampling of three fixed corruption realizations",
        "scope_limit": "three prespecified realization seeds; not a population-level corruption-randomness claim",
        "n_boot": n_boot,
        "realization_bootstrap_seed": args.realization_bootstrap_seed,
        "realization_seeds": list(seeds),
        "bootstrap_completion": str(prerequisite),
        "bootstrap_completion_sha256": sha256_file(prerequisite),
        "nested_draw_cache": str(nested_path),
        "nested_draw_cache_sha256": sha256_file(nested_path),
        "overall_delta_e": {
            **overall_e,
            "nested_percentile95": _percentile(nested_draws),
        },
        "overall_delta_psi": {
            **overall_psi,
            "nested_percentile95": _percentile(nested_psi_draws),
        },
        "datasets": {
            dataset: {
                "delta_e": _dataset_summary(
                    [cell for cell in cells if cell["dataset"] == dataset], seeds, indices
                ),
                "delta_psi": _dataset_summary(
                    [cell for cell in cells if cell["dataset"] == dataset],
                    seeds,
                    indices,
                    point_key="delta_psi",
                    draws_key="delta_psi_draws",
                ),
            }
            for dataset in DATASETS
        },
        "conditions": conditions,
        "realization_cells": public_cells,
    }
    report["analysis_sha256"] = canonical_hash(report, "analysis_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".complete").write_text(
        sha256_file(output) + "\n", encoding="utf-8"
    )
    print(f"REALIZATION ANALYSIS COMPLETE cells=108 output={output}", flush=True)


if __name__ == "__main__":
    main()
