#!/usr/bin/env python3
"""Close the confirmatory-strengthening evidence chain after all analyses finish."""
from __future__ import annotations

import argparse
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from pilot_registry import canonical_hash
from run_fixed_universe_sensitivity import validate_fixed_universe_artifact
from topic_c.manifest import sha256_file


CONFIRMATORY_DATASETS = ("voc", "kitti")
MODELS = ("yolo11n", "yolo11m", "yolo11x")


def validate_self_hashed_completion(path: Path, hash_key: str) -> dict:
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.is_file() or not marker.is_file():
        raise SystemExit(f"STRENGTHENING FINALIZER REFUSED: report or completion marker missing: {path}")
    if marker.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise SystemExit(f"STRENGTHENING FINALIZER REFUSED: completion marker mismatch: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise SystemExit(f"STRENGTHENING FINALIZER REFUSED: invalid JSON: {path}") from exc
    if document.get(hash_key) != canonical_hash(document, hash_key):
        raise SystemExit(f"STRENGTHENING FINALIZER REFUSED: self-hash mismatch: {path}")
    return document


def validate_parity_completion(path: Path) -> dict:
    completion = validate_self_hashed_completion(path, "parity_completion_sha256")
    rows = completion.get("reports")
    expected = {(dataset, model) for dataset in CONFIRMATORY_DATASETS for model in MODELS}
    observed = {
        (row.get("dataset"), row.get("model"))
        for row in rows
        if isinstance(row, dict)
    } if isinstance(rows, list) else set()
    if len(rows or []) != 6 or observed != expected:
        raise SystemExit("STRENGTHENING FINALIZER REFUSED: exact six-report parity grid absent")
    for row in rows:
        report_path = Path(row.get("report", "")).resolve()
        if not report_path.is_file() or sha256_file(report_path) != row.get("report_sha256"):
            raise SystemExit(f"STRENGTHENING FINALIZER REFUSED: parity report hash mismatch: {report_path}")
        report = json.loads(report_path.read_text(encoding="utf-8"))
        if (
            report.get("dataset") != row["dataset"]
            or report.get("model") != row["model"]
            or report.get("pass") is not True
            or report.get("errors") != []
        ):
            raise SystemExit(f"STRENGTHENING FINALIZER REFUSED: parity report did not pass: {report_path}")
        for artifact in report.get("artifacts", {}).values():
            for key in ("metric", "run_record"):
                artifact_path = Path(artifact.get(key, "")).resolve()
                if not artifact_path.is_file() or sha256_file(artifact_path) != artifact.get(f"{key}_sha256"):
                    raise SystemExit(
                        f"STRENGTHENING FINALIZER REFUSED: parity artifact hash mismatch: {artifact_path}"
                    )
    execution_path = Path(completion.get("execution_config", "")).resolve()
    if not execution_path.is_file() or sha256_file(execution_path) != completion.get("execution_config_sha256"):
        raise SystemExit("STRENGTHENING FINALIZER REFUSED: parity execution config hash mismatch")
    execution = json.loads(execution_path.read_text(encoding="utf-8"))
    reference = execution.get("reference", {})
    if reference.get("tf32_enabled") is not False or reference.get("required_parity_chain") != [
        "PyTorch", "FP32 ONNX Runtime", "FP32 TensorRT", "FP16 TensorRT"
    ]:
        raise SystemExit("STRENGTHENING FINALIZER REFUSED: strict reference contract mismatch")
    return completion


def _validate_source_manifest(root: Path, config: dict) -> None:
    records = config.get("source_manifest")
    if not isinstance(records, list) or not records:
        raise SystemExit("STRENGTHENING FINALIZER REFUSED: source manifest absent")
    seen = set()
    for record in records:
        relative = record.get("path") if isinstance(record, dict) else None
        try:
            path = (root / relative).resolve()
            path.relative_to(root)
        except (TypeError, ValueError) as exc:
            raise SystemExit("STRENGTHENING FINALIZER REFUSED: unsafe source path") from exc
        if relative in seen or not path.is_file() or sha256_file(path) != record.get("sha256"):
            raise SystemExit(f"STRENGTHENING FINALIZER REFUSED: source hash mismatch: {relative}")
        seen.add(relative)


def load_config(root: Path, path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("STRENGTHENING FINALIZER REFUSED: invalid config") from exc
    if (
        config.get("config_sha256") != canonical_hash(config, "config_sha256")
        or config.get("attempt") != "ivc_confirmatory_strengthening_v1"
    ):
        raise SystemExit("STRENGTHENING FINALIZER REFUSED: config identity mismatch")
    _validate_source_manifest(root, config)
    return config


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = Path(args.config).resolve()
    output = Path(args.out).resolve()
    if output.exists() or output.with_suffix(output.suffix + ".complete").exists():
        raise SystemExit("STRENGTHENING FINALIZER REFUSED: output exists")
    config = load_config(root, config_path)
    paths = {name: (root / relative).resolve() for name, relative in config["reports"].items()}
    while any(not path.is_file() for path in paths.values()):
        print("STRENGTHENING FINALIZER waiting for analysis reports", flush=True)
        time.sleep(args.poll_seconds)
    parity = validate_parity_completion(paths["strict_reference_parity"])
    confirmatory = validate_self_hashed_completion(paths["confirmatory_analysis"], "analysis_sha256")
    realization = validate_self_hashed_completion(paths["realization_analysis"], "analysis_sha256")
    fixed = validate_fixed_universe_artifact(root, paths["fixed_universe_sensitivity"])
    if (
        confirmatory.get("scope", {}).get("direct_cells") != 72
        or confirmatory.get("scope", {}).get("partition_role") != "untouched final holdout"
        or len(confirmatory.get("cells", [])) != 72
        or realization.get("n_boot") != 2000
        or len(realization.get("realization_seeds", [])) != 3
        or len(realization.get("realization_cells", [])) != 108
        or len(realization.get("conditions", [])) != 36
        or fixed.get("n_boot") != 10_000
    ):
        raise SystemExit("STRENGTHENING FINALIZER REFUSED: final evidence scope mismatch")
    bindings = {
        name: {"path": str(path), "sha256": sha256_file(path)}
        for name, path in paths.items()
    }
    report = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": config["attempt"],
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "status": "complete",
        "validated_components": bindings,
        "scope": {
            "strict_reference_parity_blocks": len(parity["reports"]),
            "untouched_holdout_direct_cells": len(confirmatory["cells"]),
            "corruption_realization_cells": len(realization["realization_cells"]),
            "corruption_realization_base_conditions": len(realization["conditions"]),
            "fixed_universe_bootstrap_draws": fixed["n_boot"],
        },
        "headline_native_ap": {
            "confirmatory_delta_e": confirmatory["overall_balanced_equal_cell"]["delta_e_point"],
            "confirmatory_delta_e_percentile95": confirmatory["overall_balanced_equal_cell"]["delta_e_percentile95"],
            "realization_delta_e": realization["overall_delta_e"]["mean_delta_e"],
            "realization_delta_e_percentile95": realization["overall_delta_e"]["nested_percentile95"],
            "fixed_universe_delta_e_percentile95": fixed["fixed_universe"]["delta_e_all"]["percentile_interval"],
            "fixed_universe_delta_psi_percentile95": fixed["fixed_universe"]["delta_psi_height"]["percentile_interval"],
        },
    }
    report["completion_sha256"] = canonical_hash(report, "completion_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".complete").write_text(
        sha256_file(output) + "\n", encoding="utf-8"
    )
    print(f"STRENGTHENING FINALIZER COMPLETE output={output}", flush=True)


if __name__ == "__main__":
    main()
