#!/usr/bin/env python3
"""Joint-resample macro inference for the format-contrast evidence grid."""
from __future__ import annotations

import json
from pathlib import Path
from typing import Any, Sequence

import numpy as np

from bootstrap_format_contrast import (
    ARM_NAMES,
    canonical_hash,
    component_draw_cache_identity,
    dataset_bootstrap_seed,
    load_clean_arm_cache,
    load_draw_cache,
    percentile,
    resolve_relative_evidence_path,
    validate_component_seed,
    validate_schedule_evidence,
)
from topic_c.manifest import sha256_file


ATTEMPT = "ivc_format_contrast_v1"


def _macro_output_path(root: Path, config: dict[str, Any]) -> Path:
    """Refuse macro writes outside the reviewed project root before directory creation."""
    if config.get("attempt") != ATTEMPT:
        raise ValueError(f"joint macro attempt must be exactly {ATTEMPT}")
    root = root.resolve()
    output = root / "outputs" / "bootstrap" / ATTEMPT / f"{ATTEMPT}_joint_macro.json"
    resolved = output.resolve()
    if root not in resolved.parents:
        raise ValueError(f"joint macro path escapes project root: {output}")
    return output


def joint_macro_draws(cell_draws: Sequence[np.ndarray]) -> np.ndarray:
    """Average aligned draws only when every planned component is finite."""
    if not cell_draws:
        raise ValueError("joint macro requires at least one cell draw")
    arrays = [np.asarray(draw, dtype=float) for draw in cell_draws]
    if any(array.ndim != 1 or len(array) != len(arrays[0]) for array in arrays):
        raise ValueError("joint macro draws must be aligned one-dimensional replicate arrays")
    values = np.vstack(arrays)
    complete = np.all(np.isfinite(values), axis=0)
    result = np.full(values.shape[1], np.nan)
    result[complete] = np.mean(values[:, complete], axis=0)
    return result


def macro_endpoint_plan(tasks: Sequence[dict[str, Any]]) -> dict[str, list[dict[str, Any]]]:
    """Return the fixed, unambiguous macro endpoint membership plan."""
    if len(tasks) != 144:
        raise ValueError(f"joint macro requires exactly 144 corrupted cells, found {len(tasks)}")
    if any(
        task.get("endpoint") != ("tt100k-height" if task.get("dataset") == "tt100k" else "area")
        for task in tasks
    ):
        raise ValueError("joint macro semantic endpoint membership mismatch")
    all_cells = list(tasks)
    area = [task for task in tasks if task.get("endpoint") == "area"]
    height = [task for task in tasks if task.get("endpoint") == "tt100k-height"]
    if len(area) != 108 or len(height) != 36:
        raise ValueError("joint macro requires 108 area cells and 36 TT100K height cells")
    datasets = {task.get("dataset") for task in tasks}
    if datasets != {"coco", "voc", "kitti", "tt100k"}:
        raise ValueError("joint macro requires the four planned datasets")
    if any(sum(task.get("dataset") == dataset for task in tasks) != 36 for dataset in datasets):
        raise ValueError("joint macro requires exactly 36 corrupted cells per dataset")
    return {
        "four_dataset_macro_delta_e": all_cells,
        "area_macro_delta_psi": area,
        "tt100k_height_macro_delta_psi": height,
    }


def macro_draws_from_components(
    tasks: Sequence[dict[str, Any]], components: Sequence[dict[str, np.ndarray]],
) -> dict[str, np.ndarray]:
    """Aggregate aligned, temporary component draw caches without re-evaluation."""
    plan = macro_endpoint_plan(tasks)
    if len(components) != len(tasks):
        raise ValueError("joint macro requires one draw cache for every corrupted component")
    for component in components:
        if set(component) != {"delta_e_all", "delta_psi"}:
            raise ValueError("component draw cache has an invalid metric set")
        if any(np.asarray(component[name]).ndim != 1 or len(component[name]) != 2000 for name in component):
            raise ValueError("component draw cache must contain 2,000 aligned draws per metric")
    by_task = {id(task): component for task, component in zip(tasks, components)}
    return {
        "four_dataset_macro_delta_e": joint_macro_draws([
            np.asarray(by_task[id(task)]["delta_e_all"], dtype=float)
            for task in plan["four_dataset_macro_delta_e"]
        ]),
        "area_macro_delta_psi": joint_macro_draws([
            np.asarray(by_task[id(task)]["delta_psi"], dtype=float)
            for task in plan["area_macro_delta_psi"]
        ]),
        "tt100k_height_macro_delta_psi": joint_macro_draws([
            np.asarray(by_task[id(task)]["delta_psi"], dtype=float)
            for task in plan["tt100k_height_macro_delta_psi"]
        ]),
    }


def validate_component_artifact(document: dict[str, Any]) -> None:
    """Reject any mutation of a hash-bound component before macro inference."""
    if document.get("artifact_sha256") != canonical_hash(document, "artifact_sha256"):
        raise ValueError("component artifact SHA-256 mismatch")


def _seed(namespace: str, dataset: str) -> int:
    return dataset_bootstrap_seed(namespace, dataset)


def _load_draw_cache(root: Path, component: dict[str, Any]) -> tuple[Path, dict[str, np.ndarray]]:
    """Read only a component-precommitted cache after every binding is checked."""
    reference = component.get("temporary_draw_cache")
    schedule = component.get("bootstrap_schedule", {})
    if (not isinstance(reference, dict) or set(reference) != {"path", "sha256", "identity_sha256"}
            or not isinstance(schedule, dict)):
        raise ValueError("component draw-cache binding is invalid")
    path = resolve_relative_evidence_path(root, reference["path"], "draw cache")
    identity = component_draw_cache_identity(component)
    if reference["identity_sha256"] != identity:
        raise ValueError(f"component draw-cache identity mismatch: {path}")
    return path, load_draw_cache(
        path, expected_sha256=reference["sha256"], expected_identity_sha256=identity,
        n_boot=int(component["n_boot"]), schedule_sha256=schedule.get("sha256"),
    )


def run_joint_macro(
    root: Path, config: dict[str, Any], tasks: Sequence[dict[str, Any]], draw_caches: dict[Path, Path],
) -> tuple[Path, dict[str, Any]]:
    """Create one immutable macro artifact from hash-bound component draw caches."""
    if int(config.get("n_boot", 0)) != 2000:
        raise ValueError("joint macro requires exactly 2000 replicates")
    plan = macro_endpoint_plan(tasks)
    output = _macro_output_path(root, config)
    if output.exists():
        raise ValueError(f"refusing to overwrite joint macro artifact: {output}")

    evidence: dict[str, dict[str, Any]] = {}
    caches: list[dict[str, np.ndarray]] = []
    points: list[dict[str, Any]] = []
    schedules: dict[str, dict[str, Any]] = {}
    for task in tasks:
        artifact = task["output"]
        if not artifact.is_file():
            raise ValueError(f"missing component contrast artifact: {artifact}")
        component = json.loads(artifact.read_text(encoding="utf-8"))
        validate_component_artifact(component)
        if (component.get("n_boot") != 2000 or component.get("endpoint_type") != task["endpoint"]
                or component.get("annotation", {}).get("sha256") != task["annotation_sha256"]
                or set(component.get("input_hashes", {})) != set(ARM_NAMES)):
            raise ValueError(f"invalid component contrast artifact: {artifact}")
        clean_cache = component.get("clean_arm_cache")
        if (not isinstance(clean_cache, dict) or set(clean_cache) != {"path", "sha256", "identity_sha256"}
                or not all(isinstance(clean_cache.get(key), str) and len(clean_cache[key]) == 64
                           for key in ("sha256", "identity_sha256"))):
            raise ValueError(f"invalid component clean-arm cache binding: {artifact}")
        schedule = component.get("bootstrap_schedule")
        if (not isinstance(schedule, dict) or any(key not in schedule for key in ("sha256", "n_boot", "seed", "n_images", "image_ids_sha256"))
                or schedule["n_boot"] != 2000 or schedule["seed"] != _seed(config["seed_namespace"], task["dataset"])):
            raise ValueError(f"invalid component bootstrap schedule binding: {artifact}")
        validate_component_seed(component, expected_seed=_seed(config["seed_namespace"], task["dataset"]))
        if task["dataset"] in schedules and schedules[task["dataset"]] != schedule:
            raise ValueError(f"components do not share one bootstrap schedule: {task['dataset']}")
        schedules[task["dataset"]] = schedule
        validate_schedule_evidence(root, schedule)
        clean_path = resolve_relative_evidence_path(root, clean_cache["path"], "clean-arm cache")
        load_clean_arm_cache(
            clean_path, endpoint=task["endpoint"], expected_images=task["expected_images"],
            annotation_sha256=task["annotation_sha256"], schedule=schedule,
            expected_sha256=clean_cache["sha256"], expected_identity_sha256=clean_cache["identity_sha256"],
            expected_dataset=task["dataset"], expected_model=task["model"],
            expected_input_hashes={name: component["input_hashes"][name] for name in ("int8_clean", "fp8_clean")},
        )
        relative = str(artifact.relative_to(root))
        evidence[relative] = {
            "artifact_sha256": sha256_file(artifact),
            "annotation_sha256": task["annotation_sha256"],
            "input_hashes": component["input_hashes"],
            "bootstrap_schedule": schedule,
            "clean_arm_cache": clean_cache,
        }
        if artifact not in draw_caches:
            raise ValueError(f"missing draw-cache mapping for component: {artifact}")
        cache_path = draw_caches[artifact]
        expected_cache_path, cache = _load_draw_cache(root, component)
        if expected_cache_path != cache_path.resolve():
            raise ValueError(f"component draw-cache path binding mismatch: {artifact}")
        caches.append(cache)
        evidence[relative]["temporary_draw_cache"] = component["temporary_draw_cache"]
        points.append(component["point"])

    metric_draws = macro_draws_from_components(tasks, caches)
    point_by_task = {id(task): values for task, values in zip(tasks, points)}
    point = {
        "four_dataset_macro_delta_e": float(np.mean([
            point_by_task[id(task)]["delta_e"]["all"] for task in plan["four_dataset_macro_delta_e"]
        ])),
        "area_macro_delta_psi": float(np.mean([
            point_by_task[id(task)]["delta_psi"] for task in plan["area_macro_delta_psi"]
        ])),
        "tt100k_height_macro_delta_psi": float(np.mean([
            point_by_task[id(task)]["delta_psi"] for task in plan["tt100k_height_macro_delta_psi"]
        ])),
    }
    document = {
        "schema_version": 1,
        "method": "joint paired image bootstrap; one shared resample per dataset and replicate across all models, corruptions, severities, and formats",
        "n_boot": 2000,
        "seed_namespace": config["seed_namespace"],
        "component_cells": 144,
        "component_artifacts": evidence,
        "annotations_sha256": {
            dataset: next(task["annotation_sha256"] for task in tasks if task["dataset"] == dataset)
            for dataset in ("coco", "voc", "kitti", "tt100k")
        },
        "bootstrap_schedules": schedules,
        "point": point,
        "percentile_intervals": {name: percentile(draws) for name, draws in metric_draws.items()},
        "coverage": {
            name: {
                "planned_component_count": len(plan[name]),
                "finite_complete_replicates": int(np.count_nonzero(np.isfinite(draws))),
                "total_replicates": 2000,
            }
            for name, draws in metric_draws.items()
        },
    }
    document["artifact_sha256"] = canonical_hash(document, "artifact_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return output, document
