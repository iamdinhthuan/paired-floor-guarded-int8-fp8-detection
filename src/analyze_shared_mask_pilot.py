#!/usr/bin/env python3
"""Validate and bootstrap AP/Omega for the shared-Q/DQ-mask pilot.

Omega is defined cell-wise as ``DeltaE_default - DeltaE_shared``.  Positive
Omega therefore means that the recorded default treatments produced a larger
corruption-associated INT8--FP8 gap change than the V2 coverage-aligned
treatments.  V2 keeps the recorded FP8 ONNX bytes fixed and normalizes only the
INT8 source-compute input topology.  This remains a descriptive policy
sensitivity: the INT8 and TensorRT engines are rebuilt, and historical default
builds may differ in TF32 policy.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np

from paired_bootstrap import accumulate_ap, build_eval
from topic_c.manifest import sha256_file
from topic_c.shared_mask_pilot import (
    ATTEMPT,
    CORRUPTIONS,
    FORMATS,
    SEVERITIES,
    EvidenceBundle,
    PilotError,
    locate_default_bundle,
    read_complete_json,
    same_input_identity,
    shared_bundle,
    shared_condition_id,
    validate_bundle,
    validate_config,
    validate_image_manifest,
    write_complete_json,
)


ARM_ORDER = ("default_int8", "default_fp8", "shared_int8", "shared_fp8")


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def percentile(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return [float("nan")] * 3
    return [float(value) for value in np.percentile(finite, [2.5, 50.0, 97.5])]


def contrast(
    *,
    default_int8_clean: float,
    default_fp8_clean: float,
    default_int8_corrupt: float,
    default_fp8_corrupt: float,
    shared_int8_clean: float,
    shared_fp8_clean: float,
    shared_int8_corrupt: float,
    shared_fp8_corrupt: float,
) -> dict[str, float]:
    """Calculate gaps, direct interactions and Omega in the caller's units."""
    values = (
        default_int8_clean, default_fp8_clean, default_int8_corrupt, default_fp8_corrupt,
        shared_int8_clean, shared_fp8_clean, shared_int8_corrupt, shared_fp8_corrupt,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise ValueError("contrast inputs must be finite")
    default_clean_gap = default_fp8_clean - default_int8_clean
    default_corrupt_gap = default_fp8_corrupt - default_int8_corrupt
    shared_clean_gap = shared_fp8_clean - shared_int8_clean
    shared_corrupt_gap = shared_fp8_corrupt - shared_int8_corrupt
    delta_default = default_corrupt_gap - default_clean_gap
    delta_shared = shared_corrupt_gap - shared_clean_gap
    return {
        "clean_gap_default": float(default_clean_gap),
        "corrupt_gap_default": float(default_corrupt_gap),
        "delta_e_default": float(delta_default),
        "clean_gap_shared": float(shared_clean_gap),
        "corrupt_gap_shared": float(shared_corrupt_gap),
        "delta_e_shared": float(delta_shared),
        "omega_default_minus_shared": float(delta_default - delta_shared),
    }


def dataset_seed(dataset: str) -> int:
    payload = f"{ATTEMPT}|common-image-bootstrap|{dataset}".encode("utf-8")
    return int.from_bytes(hashlib.sha256(payload).digest()[:8], "big") % (2**32)


def _record(bundle: EvidenceBundle, validated: dict[str, Any]) -> dict[str, Any]:
    return {
        "prediction": str(bundle.prediction),
        "input_record": str(bundle.input_record),
        "run_record": str(bundle.run_record),
        "metric": str(bundle.metric),
        "prediction_sha256": validated["prediction_sha256"],
        "input_record_sha256": validated["input_record_sha256"],
        "run_record_sha256": validated["run_record_sha256"],
        "metric_sha256": validated["metric_sha256"],
        "input_manifest_sha256": validated["input_manifest_sha256"],
        "input_image_ids_sha256": validated["input_image_ids_sha256"],
        "image_ids": validated["image_ids"],
        "ap": float(validated["stats"]["AP"]),
    }


def _validated_arm(
    root: Path,
    block: dict[str, Any],
    *,
    policy: str,
    precision: str,
    corruption: str,
    severity: int,
    manifest_sha: str,
) -> tuple[EvidenceBundle, dict[str, Any]]:
    if policy == "shared":
        # V3 defines the shared-policy FP8 arm as the exact frozen default
        # evidence, not a rebuilt/replayed approximation.  FP8 therefore
        # cancels algebraically in Omega while only aligned INT8 is new.
        if ATTEMPT == "shared_mask_pilot_v3" and precision == "fp8":
            bundle = locate_default_bundle(
                root, block, precision=precision, corruption=corruption, severity=severity
            )
        else:
            bundle = shared_bundle(root, shared_condition_id(block, precision, corruption, severity))
    elif policy == "default":
        bundle = locate_default_bundle(
            root, block, precision=precision, corruption=corruption, severity=severity
        )
    else:
        raise ValueError(f"invalid policy: {policy}")
    validated = validate_bundle(
        bundle,
        root=root,
        block=block,
        precision=precision,
        corruption=corruption,
        severity=severity,
        expected_manifest_sha256=manifest_sha,
    )
    return bundle, validated


def prepare_blocks(root: Path, config: dict[str, Any]) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    blocks, point_rows, clean_rows = [], [], []
    source_artifacts: dict[str, str] = {}
    for block in config["blocks"]:
        _, clean_manifest_sha = validate_image_manifest(root, block, corruption="clean", severity=0)
        clean: dict[str, dict[str, Any]] = {}
        for policy in ("default", "shared"):
            for precision in FORMATS:
                bundle, validated = _validated_arm(
                    root,
                    block,
                    policy=policy,
                    precision=precision,
                    corruption="clean",
                    severity=0,
                    manifest_sha=clean_manifest_sha,
                )
                clean[f"{policy}_{'int8' if precision == 'int8-entropy' else 'fp8'}"] = _record(bundle, validated)
                for path in bundle.paths:
                    source_artifacts[str(path.resolve().relative_to(root))] = sha256_file(path)
        clean_identity = same_input_identity(clean.values(), f"clean arms for {block['id']}")
        clean_rows.append(
            {
                "block_id": block["id"],
                "dataset": block["dataset"],
                "model": block["model"],
                "input_manifest_sha256": clean_identity[0],
                "input_image_ids_sha256": clean_identity[1],
                "default_int8_ap": clean["default_int8"]["ap"] * 100.0,
                "default_fp8_ap": clean["default_fp8"]["ap"] * 100.0,
                "shared_int8_ap": clean["shared_int8"]["ap"] * 100.0,
                "shared_fp8_ap": clean["shared_fp8"]["ap"] * 100.0,
                "default_fp8_minus_int8_ap_points": (clean["default_fp8"]["ap"] - clean["default_int8"]["ap"]) * 100.0,
                "shared_fp8_minus_int8_ap_points": (clean["shared_fp8"]["ap"] - clean["shared_int8"]["ap"]) * 100.0,
            }
        )
        conditions = []
        for corruption in CORRUPTIONS:
            for severity in SEVERITIES:
                _, manifest_sha = validate_image_manifest(
                    root, block, corruption=corruption, severity=severity
                )
                corrupt: dict[str, dict[str, Any]] = {}
                for policy in ("default", "shared"):
                    for precision in FORMATS:
                        bundle, validated = _validated_arm(
                            root,
                            block,
                            policy=policy,
                            precision=precision,
                            corruption=corruption,
                            severity=severity,
                            manifest_sha=manifest_sha,
                        )
                        key = f"{policy}_{'int8' if precision == 'int8-entropy' else 'fp8'}"
                        corrupt[key] = _record(bundle, validated)
                        for path in bundle.paths:
                            source_artifacts[str(path.resolve().relative_to(root))] = sha256_file(path)
                corrupt_identity = same_input_identity(corrupt.values(), f"corrupt arms for {block['id']}/{corruption}-s{severity}")
                if corrupt_identity[1] != clean_identity[1]:
                    raise PilotError(f"clean/corrupt ordered image IDs differ for {block['id']}/{corruption}-s{severity}")
                point = contrast(
                    default_int8_clean=clean["default_int8"]["ap"] * 100.0,
                    default_fp8_clean=clean["default_fp8"]["ap"] * 100.0,
                    default_int8_corrupt=corrupt["default_int8"]["ap"] * 100.0,
                    default_fp8_corrupt=corrupt["default_fp8"]["ap"] * 100.0,
                    shared_int8_clean=clean["shared_int8"]["ap"] * 100.0,
                    shared_fp8_clean=clean["shared_fp8"]["ap"] * 100.0,
                    shared_int8_corrupt=corrupt["shared_int8"]["ap"] * 100.0,
                    shared_fp8_corrupt=corrupt["shared_fp8"]["ap"] * 100.0,
                )
                row = {
                    "block_id": block["id"],
                    "dataset": block["dataset"],
                    "model": block["model"],
                    "corruption": corruption,
                    "severity": severity,
                    **point,
                    "default_int8_corrupt_ap": corrupt["default_int8"]["ap"] * 100.0,
                    "default_fp8_corrupt_ap": corrupt["default_fp8"]["ap"] * 100.0,
                    "shared_int8_corrupt_ap": corrupt["shared_int8"]["ap"] * 100.0,
                    "shared_fp8_corrupt_ap": corrupt["shared_fp8"]["ap"] * 100.0,
                }
                point_rows.append(row)
                conditions.append(
                    {
                        "corruption": corruption,
                        "severity": severity,
                        "input_manifest_sha256": corrupt_identity[0],
                        "input_image_ids_sha256": corrupt_identity[1],
                        "arms": corrupt,
                        "point": point,
                    }
                )
        annotation = (root / block["annotations"]).resolve()
        source_artifacts[str(annotation.relative_to(root))] = sha256_file(annotation)
        blocks.append(
            {
                "block_id": block["id"],
                "dataset": block["dataset"],
                "model": block["model"],
                "annotations": str(annotation),
                "annotation_sha256": sha256_file(annotation),
                "clean": clean,
                "conditions": conditions,
                "n_boot": int(config["bootstrap_replicates"]),
                "seed": dataset_seed(block["dataset"]),
                "config_sha256": config["config_sha256"],
                "project_root": str(root),
            }
        )
    if len(point_rows) != 36:
        raise PilotError(f"point grid must contain exactly 36 direct cells; found {len(point_rows)}")
    summary = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "attempt": ATTEMPT,
        "status": "complete",
        "metric": "COCO-style AP@[0.50:0.95] on a 0--100 scale",
        "omega_definition": "DeltaE_default - DeltaE_shared",
        "omega_interpretation": "positive means the default policy has the larger format-gap change",
        "shared_policy_definition": {
            "name": config["mask_policy"],
            "fp8_arm": (
                "exact frozen default FP8 engine/prediction/metric evidence; no rebuild or rerun"
                if ATTEMPT == "shared_mask_pilot_v3"
                else "exact frozen baseline FP8 ONNX bytes"
            ),
            "int8_arm": "INT8 aligned to the frozen FP8 source-compute input contract",
            "supersedes_failed_attempt": config["parent_failure"],
        },
        "scope": (
            "descriptive coverage-policy sensitivity with the baseline FP8 ONNX held byte-identical; "
            "not a pure causal effect of Q/DQ coverage"
        ),
        "config_sha256": config["config_sha256"],
        "clean_blocks": clean_rows,
        "cells": point_rows,
        "aggregate": {
            "cells": 36,
            "mean_delta_e_default": float(np.mean([row["delta_e_default"] for row in point_rows])),
            "mean_delta_e_shared": float(np.mean([row["delta_e_shared"] for row in point_rows])),
            "mean_omega_default_minus_shared": float(np.mean([row["omega_default_minus_shared"] for row in point_rows])),
        },
        "source_artifacts_sha256": dict(sorted(source_artifacts.items())),
        "limitations": [
            "The three blocks are a mechanistic pilot, not a detector-family population sample.",
            (
                "Omega is descriptive because aligned INT8 is a newly materialized ONNX/TensorRT treatment; "
                "the FP8 term is held to identical frozen evidence."
            ),
            "Historical default engines may permit TF32 whereas aligned INT8 engines explicitly disable TF32.",
            "Concurrent/shared-GPU runtime fields are inadmissible; this analysis uses accuracy evidence only.",
        ],
    }
    return blocks, summary


def _check_source(record: dict[str, Any]) -> None:
    for key in ("prediction", "input_record", "run_record", "metric"):
        path = Path(record[key])
        expected = record[f"{key}_sha256"] if key != "input_record" else record["input_record_sha256"]
        if not path.is_file() or sha256_file(path) != expected:
            raise PilotError(f"bootstrap source changed: {path}")


def _load_ids(record: dict[str, Any]) -> list[int]:
    _check_source(record)
    document = json.loads(Path(record["input_record"]).read_text(encoding="utf-8"))
    values = document.get("image_ids")
    if not isinstance(values, list) or values != record["image_ids"]:
        raise PilotError(f"bootstrap input IDs changed: {record['input_record']}")
    return values


def _load_predictions(record: dict[str, Any]) -> list[dict[str, Any]]:
    _check_source(record)
    value = json.loads(Path(record["prediction"]).read_text(encoding="utf-8"))
    if not isinstance(value, list):
        raise PilotError(f"prediction payload is not a list: {record['prediction']}")
    return value


def cell_paths(root: Path, block_id: str, corruption: str, severity: int) -> tuple[Path, Path]:
    stem = f"{block_id}__{corruption}-s{severity}"
    directory = root / "outputs" / "bootstrap" / ATTEMPT
    return directory / f"{stem}.npz", directory / f"{stem}.json"


def _cache_source_hashes(clean: dict[str, Any], corrupt: dict[str, Any], annotation: Path) -> dict[str, str]:
    result = {str(annotation): sha256_file(annotation)}
    for record in list(clean.values()) + list(corrupt.values()):
        for key in ("prediction", "input_record", "run_record", "metric"):
            result[record[key]] = sha256_file(Path(record[key]))
    return dict(sorted(result.items()))


def validate_cell_cache(
    npz_path: Path,
    record_path: Path,
    *,
    block_id: str,
    corruption: str,
    severity: int,
    n_boot: int,
    config_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    record = read_complete_json(record_path, self_hash_field="record_sha256")
    if (
        record.get("block_id") != block_id
        or record.get("corruption") != corruption
        or record.get("severity") != severity
        or record.get("n_boot") != n_boot
        or record.get("config_sha256") != config_sha256
        or record.get("draw_cache_sha256") != sha256_file(npz_path)
    ):
        raise PilotError(f"bootstrap cache identity mismatch: {record_path}")
    for path, expected in record.get("source_artifacts_sha256", {}).items():
        source = Path(path)
        if not source.is_file() or sha256_file(source) != expected:
            raise PilotError(f"bootstrap cache source changed: {source}")
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if set(data.files) != {
                "schema_version", "n_boot", "seed", "delta_e_default", "delta_e_shared", "omega"
            }:
                raise PilotError(f"bootstrap cache fields mismatch: {npz_path}")
            if int(data["schema_version"].item()) != 1 or int(data["n_boot"].item()) != n_boot:
                raise PilotError(f"bootstrap cache dimensions mismatch: {npz_path}")
            arrays = {
                "delta_e_default": np.asarray(data["delta_e_default"], dtype=float),
                "delta_e_shared": np.asarray(data["delta_e_shared"], dtype=float),
                "omega": np.asarray(data["omega"], dtype=float),
            }
    except OSError as exc:
        raise PilotError(f"invalid bootstrap cache: {npz_path}") from exc
    if any(values.shape != (n_boot,) or not np.all(np.isfinite(values)) for values in arrays.values()):
        raise PilotError(f"bootstrap cache replicate vector mismatch: {npz_path}")
    if not np.allclose(arrays["omega"], arrays["delta_e_default"] - arrays["delta_e_shared"], atol=1e-12, rtol=0):
        raise PilotError(f"bootstrap Omega identity mismatch: {npz_path}")
    return record, arrays


def _quarantine_partial(root: Path, paths: Iterable[Path], label: str) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = root / "outputs" / "quarantine" / ATTEMPT / f"{stamp}__bootstrap__{label}"
    directory.mkdir(parents=True, exist_ok=False)
    for ordinal, source in enumerate(existing):
        shutil.move(str(source), str(directory / f"{ordinal:02d}__{source.name}"))


def _build_arms(gt: Any, records: dict[str, dict[str, Any]], ids: list[int]) -> dict[str, Any]:
    return {name: build_eval(gt, _load_predictions(records[name]), ids) for name in ARM_ORDER}


def _ap_for_positions(evaluations: dict[str, Any], positions: list[int]) -> dict[str, float]:
    return {name: float(accumulate_ap(evaluations[name], positions)[0]) for name in ARM_ORDER}


def _bootstrap_block(payload: dict[str, Any]) -> list[str]:
    # Imported in the worker to keep the parent analyzer lightweight.
    from pycocotools.coco import COCO

    root = Path(payload["project_root"])
    annotation = Path(payload["annotations"])
    if sha256_file(annotation) != payload["annotation_sha256"]:
        raise PilotError(f"annotation changed before bootstrap: {annotation}")
    clean = payload["clean"]
    ids = _load_ids(clean["default_int8"])
    if any(_load_ids(clean[name]) != ids for name in ARM_ORDER[1:]):
        raise PilotError(f"clean bootstrap IDs differ: {payload['block_id']}")
    gt = COCO(str(annotation))
    clean_evaluations = _build_arms(gt, clean, ids)
    full = list(range(len(ids)))
    clean_point = _ap_for_positions(clean_evaluations, full)
    for name in ARM_ORDER:
        if abs(clean_point[name] - clean[name]["ap"]) > 5e-8:
            raise PilotError(f"custom AP accumulator disagrees with metric for {payload['block_id']}/{name}")
    n_boot, seed = int(payload["n_boot"]), int(payload["seed"])
    rng = np.random.default_rng(seed)
    samples = rng.integers(0, len(ids), size=(n_boot, len(ids)), dtype=np.int32)
    clean_draws = {name: np.empty(n_boot, dtype=np.float64) for name in ARM_ORDER}
    for index, sample in enumerate(samples):
        values = _ap_for_positions(clean_evaluations, sample.tolist())
        for name in ARM_ORDER:
            clean_draws[name][index] = values[name]
    outputs = []
    for ordinal, condition in enumerate(payload["conditions"], 1):
        corruption, severity = condition["corruption"], int(condition["severity"])
        npz_path, record_path = cell_paths(root, payload["block_id"], corruption, severity)
        marker = record_path.with_suffix(record_path.suffix + ".complete")
        if npz_path.exists() and record_path.exists() and marker.exists():
            validate_cell_cache(
                npz_path,
                record_path,
                block_id=payload["block_id"],
                corruption=corruption,
                severity=severity,
                n_boot=n_boot,
                config_sha256=payload["config_sha256"],
            )
            outputs.append(str(record_path))
            continue
        _quarantine_partial(root, (npz_path, record_path, marker), f"{payload['block_id']}__{corruption}-s{severity}")
        corrupt = condition["arms"]
        if any(_load_ids(corrupt[name]) != ids for name in ARM_ORDER):
            raise PilotError(f"corrupt bootstrap IDs differ: {payload['block_id']}/{corruption}-s{severity}")
        corrupt_evaluations = _build_arms(gt, corrupt, ids)
        corrupt_point = _ap_for_positions(corrupt_evaluations, full)
        for name in ARM_ORDER:
            if abs(corrupt_point[name] - corrupt[name]["ap"]) > 5e-8:
                raise PilotError(
                    f"custom AP accumulator disagrees with metric for {payload['block_id']}/{corruption}-s{severity}/{name}"
                )
        point = contrast(
            default_int8_clean=clean_point["default_int8"] * 100.0,
            default_fp8_clean=clean_point["default_fp8"] * 100.0,
            default_int8_corrupt=corrupt_point["default_int8"] * 100.0,
            default_fp8_corrupt=corrupt_point["default_fp8"] * 100.0,
            shared_int8_clean=clean_point["shared_int8"] * 100.0,
            shared_fp8_clean=clean_point["shared_fp8"] * 100.0,
            shared_int8_corrupt=corrupt_point["shared_int8"] * 100.0,
            shared_fp8_corrupt=corrupt_point["shared_fp8"] * 100.0,
        )
        if any(abs(point[key] - condition["point"][key]) > 5e-6 for key in point):
            raise PilotError(f"point metric/bootstrap identity mismatch: {payload['block_id']}/{corruption}-s{severity}")
        arrays = {
            "delta_e_default": np.empty(n_boot, dtype=np.float64),
            "delta_e_shared": np.empty(n_boot, dtype=np.float64),
            "omega": np.empty(n_boot, dtype=np.float64),
        }
        for index, sample in enumerate(samples):
            corrupt_values = _ap_for_positions(corrupt_evaluations, sample.tolist())
            value = contrast(
                default_int8_clean=clean_draws["default_int8"][index] * 100.0,
                default_fp8_clean=clean_draws["default_fp8"][index] * 100.0,
                default_int8_corrupt=corrupt_values["default_int8"] * 100.0,
                default_fp8_corrupt=corrupt_values["default_fp8"] * 100.0,
                shared_int8_clean=clean_draws["shared_int8"][index] * 100.0,
                shared_fp8_clean=clean_draws["shared_fp8"][index] * 100.0,
                shared_int8_corrupt=corrupt_values["shared_int8"] * 100.0,
                shared_fp8_corrupt=corrupt_values["shared_fp8"] * 100.0,
            )
            arrays["delta_e_default"][index] = value["delta_e_default"]
            arrays["delta_e_shared"][index] = value["delta_e_shared"]
            arrays["omega"][index] = value["omega_default_minus_shared"]
        npz_path.parent.mkdir(parents=True, exist_ok=True)
        temporary = npz_path.with_name(f".{npz_path.name}.{os.getpid()}.tmp.npz")
        np.savez_compressed(
            temporary,
            schema_version=np.asarray(1),
            n_boot=np.asarray(n_boot),
            seed=np.asarray(seed),
            delta_e_default=arrays["delta_e_default"],
            delta_e_shared=arrays["delta_e_shared"],
            omega=arrays["omega"],
        )
        os.replace(temporary, npz_path)
        record = {
            "schema_version": 1,
            "created_at_utc": utc_now(),
            "attempt": ATTEMPT,
            "block_id": payload["block_id"],
            "dataset": payload["dataset"],
            "model": payload["model"],
            "corruption": corruption,
            "severity": severity,
            "metric": "COCO-style AP@[0.50:0.95], AP points",
            "n_images": len(ids),
            "n_boot": n_boot,
            "seed": seed,
            "common_draw_contract": "same image-position resample across 8 default/shared x INT8/FP8 x clean/corrupt arms",
            "config_sha256": payload["config_sha256"],
            "point": point,
            "percentile_2.5_50_97.5": {name: percentile(values) for name, values in arrays.items()},
            "draw_cache": str(npz_path),
            "draw_cache_sha256": sha256_file(npz_path),
            "source_artifacts_sha256": _cache_source_hashes(clean, corrupt, annotation),
        }
        write_complete_json(record_path, record, self_hash_field="record_sha256")
        outputs.append(str(record_path))
        print(
            json.dumps(
                {
                    "BOOTSTRAP_CELL_COMPLETE": f"{payload['block_id']}/{corruption}-s{severity}",
                    "ordinal": ordinal,
                    "of": len(payload["conditions"]),
                    "omega": point["omega_default_minus_shared"],
                }
            ),
            flush=True,
        )
    return outputs


def _validate_point_summary(path: Path, config_sha256: str) -> dict[str, Any]:
    report = read_complete_json(path, self_hash_field="report_sha256")
    if report.get("status") != "complete" or report.get("config_sha256") != config_sha256 or len(report.get("cells", [])) != 36:
        raise PilotError(f"invalid point summary: {path}")
    for relative, expected in report.get("source_artifacts_sha256", {}).items():
        source = path.parents[3] / relative
        if not source.is_file() or sha256_file(source) != expected:
            raise PilotError(f"point-summary source changed: {source}")
    return report


def write_or_validate_point(root: Path, config: dict[str, Any], summary: dict[str, Any]) -> dict[str, Any]:
    path = root / "outputs" / "analysis" / ATTEMPT / "point_summary.json"
    if path.exists() or path.with_suffix(path.suffix + ".complete").exists():
        observed = _validate_point_summary(path, config["config_sha256"])
        # Recomputed point values must remain byte-for-byte canonical apart from creation/hash fields.
        keys = set(summary) - {"created_at_utc", "report_sha256"}
        if {key: observed.get(key) for key in keys} != {key: summary.get(key) for key in keys}:
            raise PilotError("existing point summary differs from the current validated evidence")
        return observed
    return write_complete_json(path, summary, self_hash_field="report_sha256")


def build_bootstrap_summary(root: Path, config: dict[str, Any], point: dict[str, Any]) -> dict[str, Any]:
    cells, arrays = [], {"delta_e_default": [], "delta_e_shared": [], "omega": []}
    source_artifacts = {}
    point_by_key = {
        (row["block_id"], row["corruption"], row["severity"]): row for row in point["cells"]
    }
    for block in config["blocks"]:
        for corruption in CORRUPTIONS:
            for severity in SEVERITIES:
                npz_path, record_path = cell_paths(root, block["id"], corruption, severity)
                record, values = validate_cell_cache(
                    npz_path,
                    record_path,
                    block_id=block["id"],
                    corruption=corruption,
                    severity=severity,
                    n_boot=int(config["bootstrap_replicates"]),
                    config_sha256=config["config_sha256"],
                )
                point_row = point_by_key[(block["id"], corruption, severity)]
                if any(
                    abs(record["point"][cache_name] - point_row[point_name]) > 5e-6
                    for cache_name, point_name in (
                        ("delta_e_default", "delta_e_default"),
                        ("delta_e_shared", "delta_e_shared"),
                        ("omega_default_minus_shared", "omega_default_minus_shared"),
                    )
                ):
                    raise PilotError(f"bootstrap/point summary mismatch: {record_path}")
                cells.append(
                    {
                        "block_id": block["id"],
                        "dataset": block["dataset"],
                        "model": block["model"],
                        "corruption": corruption,
                        "severity": severity,
                        "point": record["point"],
                        "percentile_2.5_50_97.5": record["percentile_2.5_50_97.5"],
                    }
                )
                for name in arrays:
                    arrays[name].append(values[name])
                source_artifacts[str(npz_path.relative_to(root))] = sha256_file(npz_path)
                source_artifacts[str(record_path.relative_to(root))] = sha256_file(record_path)
    if len(cells) != 36:
        raise PilotError("bootstrap summary requires exactly 36 cells")
    matrices = {name: np.stack(values, axis=0) for name, values in arrays.items()}
    aggregate_draws = {name: np.mean(values, axis=0) for name, values in matrices.items()}
    block_summaries = []
    for block in config["blocks"]:
        indices = [index for index, row in enumerate(cells) if row["block_id"] == block["id"]]
        block_summaries.append(
            {
                "block_id": block["id"],
                "cells": len(indices),
                **{
                    f"mean_{name}": float(np.mean([cells[index]["point"][
                        "omega_default_minus_shared" if name == "omega" else name
                    ] for index in indices]))
                    for name in arrays
                },
                **{
                    f"ci95_mean_{name}": percentile(np.mean(matrices[name][indices], axis=0))
                    for name in arrays
                },
            }
        )
    point_aggregate = point["aggregate"]
    checks = {
        "delta_e_default": point_aggregate["mean_delta_e_default"],
        "delta_e_shared": point_aggregate["mean_delta_e_shared"],
        "omega": point_aggregate["mean_omega_default_minus_shared"],
    }
    recomputed_points = {
        name: float(np.mean([
            cell["point"]["omega_default_minus_shared" if name == "omega" else name]
            for cell in cells
        ]))
        for name in arrays
    }
    if any(abs(recomputed_points[name] - checks[name]) > 5e-9 for name in arrays):
        raise PilotError("aggregate point identity failed")
    return {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "attempt": ATTEMPT,
        "status": "complete",
        "metric": "COCO-style AP@[0.50:0.95], AP points",
        "cells": 36,
        "n_boot": int(config["bootstrap_replicates"]),
        "common_draw_scope": "dataset image universe; identical draw index across all conditions sharing that universe",
        "omega_definition": "DeltaE_default - DeltaE_shared",
        "shared_policy_definition": {
            "name": config["mask_policy"],
            "fp8_arm": (
                "exact frozen default FP8 engine/prediction/metric evidence; no rebuild or rerun"
                if ATTEMPT == "shared_mask_pilot_v3"
                else "exact frozen baseline FP8 ONNX bytes"
            ),
            "int8_arm": "INT8 aligned to the frozen FP8 source-compute input contract",
            "supersedes_failed_attempt": config["parent_failure"],
        },
        "config_sha256": config["config_sha256"],
        "point_summary_sha256": sha256_file(root / "outputs" / "analysis" / ATTEMPT / "point_summary.json"),
        "aggregate": {
            "point": checks,
            "percentile_2.5_50_97.5": {name: percentile(values) for name, values in aggregate_draws.items()},
        },
        "blocks": block_summaries,
        "cell_records": cells,
        "source_artifacts_sha256": dict(sorted(source_artifacts.items())),
        "interpretation_boundary": (
            "Finite-image uncertainty conditional on checkpoints, calibration bytes, engine builds, "
            "corruption materializations and the three selected detector blocks."
        ),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default="configs/shared_mask_pilot_v2.json")
    parser.add_argument("--workers", type=int, default=3)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = (root / args.config).resolve() if not Path(args.config).is_absolute() else Path(args.config).resolve()
    try:
        if not 1 <= args.workers <= 4:
            raise PilotError("--workers must be in 1..4")
        config = validate_config(root, config_path)
        blocks, point_candidate = prepare_blocks(root, config)
        point = write_or_validate_point(root, config, point_candidate)
        print(json.dumps({"POINT_ANALYSIS_COMPLETE": 36, "aggregate": point["aggregate"]}), flush=True)

        pending = []
        for payload in blocks:
            needs_work = False
            for condition in payload["conditions"]:
                npz_path, record_path = cell_paths(
                    root, payload["block_id"], condition["corruption"], condition["severity"]
                )
                marker = record_path.with_suffix(record_path.suffix + ".complete")
                if npz_path.exists() and record_path.exists() and marker.exists():
                    validate_cell_cache(
                        npz_path,
                        record_path,
                        block_id=payload["block_id"],
                        corruption=condition["corruption"],
                        severity=condition["severity"],
                        n_boot=int(config["bootstrap_replicates"]),
                        config_sha256=config["config_sha256"],
                    )
                else:
                    _quarantine_partial(
                        root,
                        (npz_path, record_path, marker),
                        f"{payload['block_id']}__{condition['corruption']}-s{condition['severity']}",
                    )
                    needs_work = True
            if needs_work:
                pending.append(payload)
        if pending:
            with ProcessPoolExecutor(max_workers=min(args.workers, len(pending))) as executor:
                futures = {executor.submit(_bootstrap_block, payload): payload["block_id"] for payload in pending}
                for future in as_completed(futures):
                    outputs = future.result()
                    print(json.dumps({"BOOTSTRAP_BLOCK_COMPLETE": futures[future], "cells": len(outputs)}), flush=True)
        output = root / "outputs" / "analysis" / ATTEMPT / "bootstrap_summary.json"
        candidate = build_bootstrap_summary(root, config, point)
        if output.exists() or output.with_suffix(output.suffix + ".complete").exists():
            observed = read_complete_json(output, self_hash_field="report_sha256")
            keys = set(candidate) - {"created_at_utc", "report_sha256"}
            if {key: observed.get(key) for key in keys} != {key: candidate.get(key) for key in keys}:
                raise PilotError("existing bootstrap summary differs from current immutable caches")
            report = observed
        else:
            report = write_complete_json(output, candidate, self_hash_field="report_sha256")
        print(
            json.dumps(
                {
                    "SHARED_MASK_ANALYSIS_COMPLETE": report["cells"],
                    "aggregate": report["aggregate"],
                    "report": str(output),
                }
            ),
            flush=True,
        )
    except (PilotError, ValueError) as exc:
        raise SystemExit(f"SHARED-MASK ANALYSIS REFUSED: {exc}") from exc


if __name__ == "__main__":
    main()
