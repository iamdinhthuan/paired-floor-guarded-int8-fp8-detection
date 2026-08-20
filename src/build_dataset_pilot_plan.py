#!/usr/bin/env python3
"""Create a hash-bound 117-condition transfer-dataset pilot proposal."""
from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path

from pilot_registry import calibration_sha256


def calibrator_for_precision(precision: str) -> str:
    if precision in {"int8-entropy", "fp8"}:
        return "entropy"
    if precision == "fp32":
        return "none"
    raise ValueError(f"unsupported precision: {precision}")


def canonical_hash(document: dict, excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--engine-registry", required=True)
    parser.add_argument("--dataset", choices=("coco", "voc", "kitti", "tt100k"), required=True)
    parser.add_argument("--split", required=True)
    parser.add_argument("--clean-manifest", required=True, help="path relative to the Topic C project")
    parser.add_argument("--corruption-manifest-template", required=True,
                        help="project-relative template with {corruption} and {severity} placeholders")
    parser.add_argument("--calibration-list", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    matrix_path, registry_path, calibration_path, output = (Path(args.matrix), Path(args.engine_registry),
                                                              Path(args.calibration_list), Path(args.out))
    if output.exists():
        raise SystemExit(f"PILOT PLAN REFUSED: refusing to overwrite: {output}")
    matrix, registry = json.loads(matrix_path.read_text(encoding="utf-8")), json.loads(registry_path.read_text(encoding="utf-8"))
    if matrix.get("expected_runs_per_dataset") != 117 or matrix.get("models") != ["yolo11n", "yolo11m", "yolo11x"]:
        raise SystemExit("PILOT PLAN REFUSED: unexpected frozen matrix")
    if matrix.get("dataset_protocols", {}).get(args.dataset, {}).get("split") != args.split:
        raise SystemExit("PILOT PLAN REFUSED: dataset split disagrees with frozen matrix")
    if registry.get("registry_sha256") != canonical_hash(registry, "registry_sha256") or registry.get("dataset") != args.dataset:
        raise SystemExit("PILOT PLAN REFUSED: engine registry hash or dataset mismatch")
    if not calibration_path.is_file():
        raise SystemExit("PILOT PLAN REFUSED: calibration list is absent")
    try:
        calibration_sha = calibration_sha256(calibration_path)
    except ValueError as exc:
        raise SystemExit(f"PILOT PLAN REFUSED: {exc}") from exc
    try:
        corruption_template = args.corruption_manifest_template.format(corruption="{corruption}", severity="{severity}")
    except (KeyError, IndexError) as exc:
        raise SystemExit("PILOT PLAN REFUSED: malformed corruption-manifest template") from exc
    runs = []
    for model in matrix["models"]:
        for precision in matrix["precisions"]:
            engine = registry.get("engines", {}).get(model, {}).get(precision)
            if not engine or not Path(engine.get("path", "")).is_file():
                raise SystemExit(f"PILOT PLAN REFUSED: missing engine: {model}/{precision}")
            actual_engine_sha = hashlib.sha256(Path(engine["path"]).read_bytes()).hexdigest()
            if actual_engine_sha != engine.get("sha256"):
                raise SystemExit(f"PILOT PLAN REFUSED: engine hash mismatch: {model}/{precision}")
            calibrator = calibrator_for_precision(precision)
            if precision in {"int8-entropy", "fp8"} and engine.get("calibration_sha256") != calibration_sha:
                raise SystemExit(f"PILOT PLAN REFUSED: INT8/FP8 calibration mismatch: {model}/{precision}")
            condition_precision = "int8" if precision == "int8-entropy" else precision
            for condition in matrix["conditions"]:
                corruption, severity = condition["corruption"], int(condition["severity"])
                manifest = args.clean_manifest if corruption == "clean" else corruption_template.format(corruption=corruption, severity=severity)
                runs.append({
                    "dataset": args.dataset, "split": args.split, "model": model, "precision": precision,
                    "calibrator": calibrator, "corruption": corruption, "severity": severity,
                    "engine_path": engine["path"], "engine_sha256": engine["sha256"],
                    "calibration_sha256": engine.get("calibration_sha256"), "input_manifest_path": manifest,
                    "condition_id_template": f"{args.dataset}_{args.split}__{model}__{condition_precision}-{calibrator}__{corruption}-s{severity}__{engine['sha256'][:8]}__{{manifest8}}",
                })
    if len(runs) != 117 or len({run["condition_id_template"] for run in runs}) != 117:
        raise SystemExit("PILOT PLAN REFUSED: expected 117 unique conditions")
    document = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": args.dataset,
                "split": args.split, "execution_policy": "requires three hash-valid clean FP32/FP16 parity gates before inference",
                "engine_registry_sha256": registry["registry_sha256"], "matrix_sha256": hashlib.sha256(matrix_path.read_bytes()).hexdigest(),
                "calibration_list": str(calibration_path.resolve()), "calibration_sha256": calibration_sha, "runs": runs}
    document["plan_sha256"] = canonical_hash(document, "plan_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"PILOT PLAN VALID dataset={args.dataset} runs=117 sha256={document['plan_sha256']}")


if __name__ == "__main__":
    main()
