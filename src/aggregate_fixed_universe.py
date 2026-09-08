#!/usr/bin/env python3
"""Validate and aggregate all 36 TT100K fixed-universe sensitivity cells.

No predictions are regenerated. Uniform-weight point estimates stay unchanged;
the positive-weight interval is a sensitivity, not a corrected ordinary CI.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from itertools import product
from pathlib import Path

import numpy as np

from bootstrap_format_contrast_macro import (
    _load_draw_cache,
    joint_macro_draws,
    validate_component_artifact,
)
from run_fixed_universe_sensitivity import (
    ARM_NAMES,
    canonical_hash,
    summarize_draws,
    validate_fixed_universe_artifact,
)
from topic_c.manifest import sha256_file


def summarize_macro(cells, points):
    arrays = [np.asarray(value, dtype=float) for value in cells]
    points = np.asarray(points, dtype=float)
    if (len(arrays) != 36 or points.shape != (36,) or not np.isfinite(points).all()
            or any(a.shape != (10000,) or not np.isfinite(a).all() for a in arrays)):
        raise ValueError("36 finite points and 36 aligned 10000-draw vectors required")
    return {"point_native_ap": float(points.mean()),
            **summarize_draws(joint_macro_draws(arrays), checkpoint=2000)}


def validate_joint_identity(reports):
    if not reports:
        raise ValueError("no reports")
    first = reports[0]
    images = {report["input_hashes"][arm]["image_ids_sha256"]
              for report in reports for arm in ARM_NAMES}
    if (len(images) != 1 or None in images
            or any(report["schedule"] != first["schedule"] for report in reports)
            or any(report["annotation"] != first["annotation"] for report in reports)):
        raise ValueError("joint schedule, annotation or ordered image identity mismatch")


def aggregate(root: Path, folder: Path, output: Path):
    root, folder, output = root.resolve(), folder.resolve(), output.resolve()
    if output.exists():
        raise FileExistsError(output)
    for path in (folder, output):
        path.relative_to(root)
    names = [f"tt100k__{model}__{corruption}-s{severity}"
             for model, corruption, severity in product(
                 ("yolo11n", "yolo11m", "yolo11x"),
                 ("fog", "gaussian_noise", "jpeg", "motion_blur"), (1, 3, 5))]
    observed = {path.stem for path in folder.glob("tt100k__*.json")}
    if observed != set(names):
        raise ValueError("exact 36-cell grid required; missing/extra report")
    reports, sources, ordinary_schedules = [], [], []
    fixed = {key: [] for key in ("delta_e_all", "delta_psi_height")}
    ordinary = {key: [] for key in fixed}
    points = {key: [] for key in fixed}
    for index, name in enumerate(names, 1):
        path = folder / f"{name}.json"
        report = validate_fixed_universe_artifact(root, path)
        expected_component = f"outputs/bootstrap/ivc_format_contrast_v1/{name}.json"
        if report["component"] != expected_component:
            raise ValueError(f"component membership mismatch: {name}")
        component = json.loads((root / expected_component).read_text())
        validate_component_artifact(component)
        if component.get("endpoint_type") != "tt100k-height" or component.get("n_boot") != 2000:
            raise ValueError("ordinary reference endpoint or replication count mismatch")
        _, old_draws = _load_draw_cache(root, component)
        ordinary_schedules.append(component["bootstrap_schedule"])
        with np.load(root / report["draws"]["path"], allow_pickle=False) as draws:
            for key in fixed:
                fixed[key].append(draws[key].copy())
                points[key].append(report["point"][key])
        ordinary["delta_e_all"].append(old_draws["delta_e_all"])
        ordinary["delta_psi_height"].append(old_draws["delta_psi"])
        reports.append(report)
        sources.append({"cell": name, "report_path": str(path.relative_to(root)),
                        "report_sha256": sha256_file(path), "draws": report["draws"],
                        "component": report["component"],
                        "component_sha256": report["component_sha256"],
                        "ordinary_draws": component["temporary_draw_cache"]})
        print(f"VALIDATED {index}/36 {name}", flush=True)
    validate_joint_identity(reports)
    if any(schedule != ordinary_schedules[0] for schedule in ordinary_schedules):
        raise ValueError("ordinary components do not share a joint schedule")
    result = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "method": "equal-cell mean within each common image-weight draw, then percentiles",
        "scope": "TT100K only; 36 historical cells, three capacities; fixed-universe sensitivity, not a replacement CI",
        "units": "native AP; multiply values and interval endpoints by 100 for AP points",
        "component_cells": 36, "n_images": 3067,
        "schedule": reports[0]["schedule"],
        "annotation": reports[0]["annotation"],
        "fixed_universe": {key: summarize_macro(fixed[key], points[key]) for key in fixed},
        "ordinary_reference": {},
        "source_bindings": sources,
        "aggregator_sha256": sha256_file(Path(__file__)),
        "validator_sha256": sha256_file(Path(__file__).with_name("run_fixed_universe_sensitivity.py")),
    }
    for key in ordinary:
        values = joint_macro_draws(ordinary[key])
        if not np.isfinite(values).all() or values.shape != (2000,):
            raise ValueError("ordinary macro has incomplete or invalid draws")
        result["ordinary_reference"][key] = {
            "point_native_ap": float(np.mean(points[key])),
            **summarize_draws(values, checkpoint=2000),
        }
    result["artifact_sha256"] = canonical_hash(result, "artifact_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    with output.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(result, indent=2, allow_nan=False) + "\n")
    print(json.dumps({"out": str(output), "fixed_universe": result["fixed_universe"],
                      "ordinary_reference": result["ordinary_reference"]}, indent=2))
    return result


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--folder", type=Path, required=True)
    parser.add_argument("--out", type=Path, required=True)
    args = parser.parse_args()
    aggregate(args.project_root, args.folder, args.out)
