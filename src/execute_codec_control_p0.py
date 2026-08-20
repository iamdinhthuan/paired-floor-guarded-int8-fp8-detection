#!/usr/bin/env python3
"""Run and evaluate the 36 matched JPEG-95 clean-control conditions."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pilot_registry import canonical_hash
from topic_c.manifest import read_manifest, sha256_file, validate_manifest


def resolved(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def output_paths(root: Path, attempt: str, condition: str) -> tuple[Path, Path, Path, Path]:
    return (
        root / "outputs" / "predictions" / attempt / f"{condition}.json",
        root / "outputs" / "inputs" / attempt / f"{condition}.json",
        root / "manifests" / "runs" / attempt / f"{condition}.json",
        root / "outputs" / "metrics" / attempt / f"{condition}.json",
    )


def completed_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    document = read_manifest(path)
    marker = path.with_suffix(path.suffix + ".complete")
    if not marker.is_file() or marker.read_text().strip() != document["manifest_sha256"]:
        return None
    return document


def validate_existing(paths: tuple[Path, Path, Path, Path], condition: str, images: int) -> bool:
    prediction, inputs, run, metric = paths
    present = [path.is_file() for path in paths]
    if not any(present):
        return False
    if not all(present):
        raise SystemExit(f"CODEC P0 REFUSED: partial condition: {condition}")
    run_data = json.loads(run.read_text())
    input_data = json.loads(inputs.read_text())
    metric_data = json.loads(metric.read_text())
    if (
        run_data.get("condition_id") != condition
        or run_data.get("prediction_sha256") != sha256_file(prediction)
        or run_data.get("n_images") != images
        or len(input_data.get("image_ids", [])) != images
        or metric_data.get("run_record_sha256") != sha256_file(run)
        or metric_data.get("prediction_sha256") != run_data["prediction_sha256"]
    ):
        raise SystemExit(f"CODEC P0 REFUSED: invalid existing condition: {condition}")
    return True


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    codec = config["codec_control"]
    attempt = codec["attempt"]
    records = []

    for data in codec["datasets"]:
        dataset, split = data["dataset"], data["split"]
        annotations = resolved(root, data["annotations"])
        clean_root = resolved(root, data["clean_root"])
        cache_root = resolved(root, data["cache_root"])
        manifest_path = resolved(root, data["manifest"])
        manifest = completed_manifest(manifest_path)
        if manifest is None:
            subprocess.run(
                [
                    sys.executable,
                    str(root / "src" / "materialize_codec_control.py"),
                    "--dataset",
                    dataset,
                    "--split",
                    split,
                    "--annotations",
                    str(annotations),
                    "--clean-root",
                    str(clean_root),
                    "--cache-root",
                    str(cache_root),
                    "--manifest-out",
                    str(manifest_path),
                    "--quality",
                    str(codec["quality"]),
                    "--subsampling",
                    str(codec["subsampling"]),
                    "--resume-validated",
                ],
                check=True,
            )
            manifest = completed_manifest(manifest_path)
        if manifest is None or len(manifest["records"]) != int(data["expected_images"]):
            raise SystemExit(f"CODEC P0 REFUSED: incomplete manifest: {dataset}")
        failures = validate_manifest(
            manifest, annotations, cache_root, clean_root, require_pixels_changed=False
        )
        if failures:
            raise SystemExit(
                f"CODEC P0 REFUSED: manifest validation failed: {dataset}\n"
                + "\n".join(failures[:20])
            )

        registry_path = resolved(root, data["engine_registry"])
        registry = json.loads(registry_path.read_text())
        if (
            registry.get("dataset") != dataset
            or registry.get("registry_sha256")
            != canonical_hash(registry, "registry_sha256")
        ):
            raise SystemExit(f"CODEC P0 REFUSED: engine registry invalid: {dataset}")
        calibration = resolved(root, data["calibration_list"])
        class_map = resolved(root, data["class_map"]) if data.get("class_map") else None
        evaluator = root / "src" / ("tt100k_eval.py" if dataset == "tt100k" else "coco_eval.py")

        for model in config["models"]:
            for precision in config["scientific_precisions"]:
                engine = registry["engines"][model][precision]
                engine_path = Path(engine["path"])
                if not engine_path.is_file() or sha256_file(engine_path) != engine["sha256"]:
                    raise SystemExit(f"CODEC P0 REFUSED: engine mismatch: {dataset}/{model}/{precision}")
                label = "int8" if precision == "int8-entropy" else precision
                condition = (
                    f"{dataset}_{split}__{model}__{label}__codec-control-s0__"
                    f"{engine['sha256'][:8]}__{manifest['manifest_sha256'][:8]}"
                )
                paths = output_paths(root, attempt, condition)
                if validate_existing(paths, condition, int(data["expected_images"])):
                    records.append({"dataset": dataset, "condition": condition, "metric": str(paths[3]), "sha256": sha256_file(paths[3])})
                    continue
                prediction, inputs, run, metric = paths
                command = [
                    sys.executable,
                    str(root / "src" / "coco_infer_trt.py"),
                    "--engine",
                    str(engine_path),
                    "--annotations",
                    str(annotations),
                    "--image-manifest",
                    str(manifest_path),
                    "--manifest-cache-root",
                    str(cache_root),
                    "--out",
                    str(prediction),
                    "--input-record",
                    str(inputs),
                    "--run-record",
                    str(run),
                    "--condition-id",
                    condition,
                    "--dataset",
                    dataset,
                    "--split",
                    split,
                    "--imgsz",
                    str(data["imgsz"]),
                    "--model",
                    model,
                    "--precision",
                    precision,
                    "--calibrator",
                    "entropy" if precision == "int8-entropy" else "none",
                    "--corruption",
                    "codec_control",
                    "--severity",
                    "0",
                ]
                if class_map:
                    command += ["--class-map", str(class_map)]
                if precision in {"int8-entropy", "fp8"}:
                    command += [
                        "--calibration-list",
                        str(calibration),
                        "--calibration-method",
                        "entropy",
                        "--calibration-provenance",
                        "verified",
                    ]
                print(f"CODEC P0 INFERENCE {dataset} {model} {precision}", flush=True)
                subprocess.run(command, check=True)
                subprocess.run(
                    [
                        sys.executable,
                        str(evaluator),
                        "--annotations",
                        str(annotations),
                        "--predictions",
                        str(prediction),
                        "--input-record",
                        str(inputs),
                        "--run-record",
                        str(run),
                        "--out",
                        str(metric),
                    ],
                    check=True,
                )
                if not validate_existing(paths, condition, int(data["expected_images"])):
                    raise SystemExit(f"CODEC P0 REFUSED: condition did not close: {condition}")
                records.append({"dataset": dataset, "condition": condition, "metric": str(metric), "sha256": sha256_file(metric)})

    report = root / "outputs" / "reports" / f"{attempt}_complete.json"
    if report.exists():
        old = json.loads(report.read_text())
        if old.get("report_sha256") != canonical_hash(old, "report_sha256"):
            raise SystemExit("CODEC P0 REFUSED: invalid existing report")
        print(f"CODEC P0 RESUME COMPLETE conditions={len(records)}")
        return
    document = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": attempt,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "conditions": records,
    }
    if len(records) != 36:
        raise SystemExit(f"CODEC P0 REFUSED: expected 36 conditions, found {len(records)}")
    document["report_sha256"] = canonical_hash(document, "report_sha256")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(document, indent=2) + "\n")
    print(f"CODEC P0 COMPLETE conditions=36 report={report}")


if __name__ == "__main__":
    main()
