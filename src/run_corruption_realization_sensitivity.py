#!/usr/bin/env python3
"""Run the prespecified three-seed corruption-realization sensitivity grid."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from generate_corruption_realization import stable_realization_seed
from pilot_registry import calibration_sha256, canonical_hash
from topic_c.manifest import read_manifest, sha256_file


DATASETS = ("voc", "kitti", "tt100k")
PRECISIONS = ("int8-entropy", "fp8")
CORRUPTIONS = ("gaussian_noise", "motion_blur", "fog", "jpeg")
SEVERITIES = (1, 3, 5)
SAFE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def manifest_tasks(*, datasets=DATASETS, realization_seeds: tuple[int, ...]) -> list[dict]:
    return [
        {
            "dataset": dataset,
            "realization_seed": realization_seed,
            "corruption": corruption,
            "severity": severity,
        }
        for dataset in datasets
        for realization_seed in realization_seeds
        for corruption in CORRUPTIONS
        for severity in SEVERITIES
    ]


def inference_tasks(*, datasets=DATASETS, realization_seeds: tuple[int, ...]) -> list[dict]:
    clean = [
        {
            "dataset": dataset,
            "realization_seed": None,
            "corruption": "clean",
            "severity": 0,
            "precision": precision,
        }
        for dataset in datasets
        for precision in PRECISIONS
    ]
    corrupt = [
        {**task, "precision": precision}
        for task in manifest_tasks(datasets=datasets, realization_seeds=realization_seeds)
        for precision in PRECISIONS
    ]
    return clean + corrupt


def task_identity(task: dict) -> str:
    seed = "clean" if task.get("realization_seed") is None else f"r{task['realization_seed']}"
    precision = task.get("precision", "manifest")
    return (
        f"{task['dataset']}__{seed}__yolo11m__{precision}__"
        f"{task['corruption']}-s{task['severity']}"
    )


def cache_root(root: Path, dataset: str, realization_seed: int, corruption: str) -> Path:
    suffix = "jpeg_deterministic" if corruption == "jpeg" else f"r{realization_seed}"
    return root / "data" / "corruption_realization_v1" / dataset / suffix


def resolve_inside(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SystemExit(f"REALIZATION SENSITIVITY REFUSED: missing {label}")
    root = root.resolve()
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise SystemExit(f"REALIZATION SENSITIVITY REFUSED: {label} escapes project root")
    return path


def complete(path: Path) -> bool:
    marker = path.with_suffix(path.suffix + ".complete")
    return (
        path.is_file()
        and marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == sha256_file(path)
    )


def load_config(root: Path, path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("REALIZATION SENSITIVITY REFUSED: unreadable config") from exc
    if config.get("config_sha256") != canonical_hash(config, "config_sha256"):
        raise SystemExit("REALIZATION SENSITIVITY REFUSED: config hash mismatch")
    seeds = config.get("realization_seeds")
    if (
        config.get("attempt") != "corruption_realization_v1"
        or not isinstance(seeds, list)
        or len(seeds) != 3
        or len(set(seeds)) != 3
        or not all(isinstance(seed, int) and seed >= 0 for seed in seeds)
    ):
        raise SystemExit("REALIZATION SENSITIVITY REFUSED: invalid attempt/seed grid")
    if (
        not isinstance(config.get("conservative_cache_estimate_gib"), int)
        or not 1 <= config["conservative_cache_estimate_gib"] <= 128
        or not isinstance(config.get("minimum_free_gib_after_estimate"), int)
        or config["minimum_free_gib_after_estimate"] < 20
    ):
        raise SystemExit("REALIZATION SENSITIVITY REFUSED: invalid disk envelope")
    rows = config.get("datasets")
    if (
        not isinstance(rows, list)
        or len(rows) != 3
        or [row.get("dataset") for row in rows if isinstance(row, dict)] != list(DATASETS)
    ):
        raise SystemExit("REALIZATION SENSITIVITY REFUSED: dataset sequence mismatch")
    corruption_config = resolve_inside(root, config.get("corruption_config"), "corruption config")
    if not corruption_config.is_file() or sha256_file(corruption_config) != config.get("corruption_config_sha256"):
        raise SystemExit("REALIZATION SENSITIVITY REFUSED: corruption config hash mismatch")
    manifest = config.get("source_manifest")
    if not isinstance(manifest, list) or not manifest:
        raise SystemExit("REALIZATION SENSITIVITY REFUSED: source manifest missing")
    for record in manifest:
        source = resolve_inside(root, record.get("path"), "source path")
        if not source.is_file() or sha256_file(source) != record.get("sha256"):
            raise SystemExit(
                f"REALIZATION SENSITIVITY REFUSED: source hash mismatch: {record.get('path')}"
            )
    return config


def load_subset(root: Path, row: dict) -> dict:
    registry_path = resolve_inside(root, row.get("subset_registry"), "subset registry")
    if not complete(registry_path) or sha256_file(registry_path) != row.get("subset_registry_sha256"):
        raise SystemExit(f"REALIZATION SENSITIVITY REFUSED: subset registry mismatch: {row.get('dataset')}")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if (
        registry.get("registry_sha256") != canonical_hash(registry, "registry_sha256")
        or registry.get("dataset") != row.get("dataset")
        or registry.get("images") != 512
        or registry.get("all_observed_categories_preserved") is not True
    ):
        raise SystemExit(f"REALIZATION SENSITIVITY REFUSED: invalid subset registry: {row.get('dataset')}")
    annotation = Path(registry["annotation"]).resolve()
    manifest = Path(registry["manifest"]).resolve()
    if (
        not annotation.is_file()
        or sha256_file(annotation) != registry["annotation_sha256"]
        or not manifest.is_file()
        or sha256_file(manifest) != registry["manifest_file_sha256"]
    ):
        raise SystemExit(f"REALIZATION SENSITIVITY REFUSED: subset artifact mismatch: {row.get('dataset')}")
    class_map = resolve_inside(root, row.get("class_map"), "subset class map")
    if not complete(class_map) or sha256_file(class_map) != row.get("class_map_sha256"):
        raise SystemExit(f"REALIZATION SENSITIVITY REFUSED: subset class-map mismatch: {row.get('dataset')}")
    class_document = json.loads(class_map.read_text(encoding="utf-8"))
    if (
        class_document.get("class_map_sha256") != canonical_hash(class_document, "class_map_sha256")
        or class_document.get("annotation_sha256") != sha256_file(annotation)
        or class_document.get("dataset") != row.get("dataset")
        or class_document.get("split") != row.get("split")
    ):
        raise SystemExit(f"REALIZATION SENSITIVITY REFUSED: invalid subset class map: {row.get('dataset')}")
    return {"registry": registry_path, "annotation": annotation, "manifest": manifest, "class_map": class_map}


def load_engines(root: Path, row: dict) -> dict[str, dict]:
    result = {}
    calibration = resolve_inside(root, row.get("calibration_list"), "calibration list")
    if not calibration.is_file() or sha256_file(calibration) != row.get("calibration_list_sha256"):
        raise SystemExit(f"REALIZATION SENSITIVITY REFUSED: calibration file mismatch: {row.get('dataset')}")
    canonical_calibration = calibration_sha256(str(calibration))
    for precision in PRECISIONS:
        binding = row.get("engines", {}).get(precision, {})
        registry_path = resolve_inside(root, binding.get("registry"), "engine registry")
        if not complete(registry_path) or sha256_file(registry_path) != binding.get("registry_sha256"):
            raise SystemExit(
                f"REALIZATION SENSITIVITY REFUSED: engine registry mismatch: {row.get('dataset')}/{precision}"
            )
        registry = json.loads(registry_path.read_text(encoding="utf-8"))
        engine = Path(registry.get("engine", "")).resolve()
        if (
            registry.get("dataset") != row.get("dataset")
            or registry.get("model") != "yolo11m"
            or registry.get("precision") != precision
            or registry.get("calibration_sha256") != canonical_calibration
            or not engine.is_file()
            or sha256_file(engine) != registry.get("engine_sha256")
        ):
            raise SystemExit(
                f"REALIZATION SENSITIVITY REFUSED: invalid engine treatment: {row.get('dataset')}/{precision}"
            )
        result[precision] = {
            "registry": registry_path,
            "engine": engine,
            "engine_sha256": registry["engine_sha256"],
            "calibration": calibration,
            "calibration_sha256": canonical_calibration,
        }
    return result


def manifest_path(root: Path, task: dict) -> Path:
    return (
        root
        / "manifests"
        / "images"
        / "corruption_realization_v1"
        / (
            f"{task['dataset']}_r{task['realization_seed']}_"
            f"{task['corruption']}_s{task['severity']}.json"
        )
    )


def condition_paths(root: Path, task: dict) -> tuple[Path, Path, Path, Path]:
    identity = task_identity(task)
    return (
        root / "outputs" / "predictions" / "corruption_realization_v1" / f"{identity}.json",
        root / "outputs" / "inputs" / "corruption_realization_v1" / f"{identity}.json",
        root / "manifests" / "runs" / "corruption_realization_v1" / f"{identity}.json",
        root / "outputs" / "metrics" / "corruption_realization_v1" / f"{identity}.json",
    )


def run(command: list[str]) -> None:
    print("REALIZATION SENSITIVITY COMMAND " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def validate_condition(root: Path, task: dict, manifest: Path, engine: dict, expected_images: int) -> dict:
    prediction, inputs_path, run_path, metric_path = condition_paths(root, task)
    if not all(path.is_file() for path in (prediction, inputs_path, run_path, metric_path)):
        raise SystemExit(f"REALIZATION SENSITIVITY REFUSED: incomplete condition: {task_identity(task)}")
    inputs = json.loads(inputs_path.read_text(encoding="utf-8"))
    record = json.loads(run_path.read_text(encoding="utf-8"))
    metric = json.loads(metric_path.read_text(encoding="utf-8"))
    manifest_document = read_manifest(manifest)
    ids = inputs.get("image_ids")
    ids_sha = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest() if isinstance(ids, list) else None
    if (
        len(ids or []) != expected_images
        or len(set(ids or [])) != expected_images
        or inputs.get("image_ids_sha256") != ids_sha
        or inputs.get("input_manifest_sha256") != manifest_document.get("manifest_sha256")
        or record.get("condition_id") != task_identity(task)
        or record.get("dataset") != task["dataset"]
        or record.get("model") != "yolo11m"
        or record.get("precision") != task["precision"]
        or record.get("corruption") != task["corruption"]
        or record.get("severity") != task["severity"]
        or record.get("engine_sha256") != engine["engine_sha256"]
        or record.get("calibration_sha256") != engine["calibration_sha256"]
        or record.get("prediction_sha256") != sha256_file(prediction)
        or record.get("input_image_ids_sha256") != ids_sha
        or metric.get("run_record_sha256") != sha256_file(run_path)
        or metric.get("prediction_sha256") != sha256_file(prediction)
        or metric.get("input_image_ids_sha256") != ids_sha
    ):
        raise SystemExit(f"REALIZATION SENSITIVITY REFUSED: condition provenance mismatch: {task_identity(task)}")
    return {
        "condition_id": task_identity(task),
        "prediction_sha256": sha256_file(prediction),
        "input_record_sha256": sha256_file(inputs_path),
        "run_record_sha256": sha256_file(run_path),
        "metric_sha256": sha256_file(metric_path),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--wait-for-confirmatory-analysis", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = Path(args.config).resolve()
    prerequisite = Path(args.wait_for_confirmatory_analysis).resolve()
    report_out = Path(args.report_out).resolve()
    if report_out.exists() or report_out.with_suffix(report_out.suffix + ".complete").exists():
        raise SystemExit("REALIZATION SENSITIVITY REFUSED: completion report exists")
    config = load_config(root, config_path)
    while not prerequisite.is_file():
        print("REALIZATION SENSITIVITY waiting for confirmatory analysis", flush=True)
        time.sleep(args.poll_seconds)
    upstream = json.loads(prerequisite.read_text(encoding="utf-8"))
    if upstream.get("analysis_sha256") != canonical_hash(upstream, "analysis_sha256"):
        raise SystemExit("REALIZATION SENSITIVITY REFUSED: invalid confirmatory analysis")
    rows = {row["dataset"]: row for row in config["datasets"]}
    subsets = {dataset: load_subset(root, rows[dataset]) for dataset in DATASETS}
    engines = {dataset: load_engines(root, rows[dataset]) for dataset in DATASETS}
    source_bytes = 0
    for dataset in DATASETS:
        clean_root = resolve_inside(root, rows[dataset]["clean_root"], "clean root")
        manifest = read_manifest(subsets[dataset]["manifest"])
        source_bytes += sum(
            (clean_root / record["source_relpath"]).stat().st_size
            for record in manifest["records"]
        )
    estimate = int(config["conservative_cache_estimate_gib"]) * 1024**3
    reserve = int(config["minimum_free_gib_after_estimate"]) * 1024**3
    free = shutil.disk_usage(root).free
    if free - estimate < reserve:
        raise SystemExit(
            f"REALIZATION SENSITIVITY REFUSED: free {free} - estimate {estimate} < reserve {reserve}"
        )
    manifest_records = []
    seeds = tuple(config["realization_seeds"])
    corruption_config = resolve_inside(root, config["corruption_config"], "corruption config")
    for task in manifest_tasks(realization_seeds=seeds):
        dataset = task["dataset"]
        row = rows[dataset]
        path = manifest_path(root, task)
        cache = cache_root(root, dataset, task["realization_seed"], task["corruption"])
        if not path.is_file():
            run(
                [
                    sys.executable,
                    str(root / "src" / "generate_corruption_realization.py"),
                    "--dataset", dataset,
                    "--split", row["split"],
                    "--annotations", str(subsets[dataset]["annotation"]),
                    "--clean-root", str(resolve_inside(root, row["clean_root"], "clean root")),
                    "--cache-root", str(cache),
                    "--config", str(corruption_config),
                    "--corruption", task["corruption"],
                    "--severity", str(task["severity"]),
                    "--realization-seed", str(task["realization_seed"]),
                    "--manifest-out", str(path),
                    "--resume-validated",
                ]
            )
        document = read_manifest(path)
        marker = path.with_suffix(path.suffix + ".complete")
        seeds_match = all(
            int(record["seed"])
            == stable_realization_seed(
                dataset,
                int(record["image_id"]),
                task["corruption"],
                task["realization_seed"],
            )
            for record in document.get("records", [])
        )
        if (
            not marker.is_file()
            or marker.read_text(encoding="utf-8").strip() != document.get("manifest_sha256")
            or len(document.get("records", [])) != 512
            or document.get("realization_seed") != task["realization_seed"]
            or document.get("nested_across_severity") is not True
            or not seeds_match
        ):
            raise SystemExit(f"REALIZATION SENSITIVITY REFUSED: invalid manifest: {path}")
        manifest_records.append(
            {**task, "path": str(path), "file_sha256": sha256_file(path), "manifest_sha256": document["manifest_sha256"]}
        )
    condition_records = []
    for task in inference_tasks(realization_seeds=seeds):
        dataset = task["dataset"]
        row = rows[dataset]
        engine = engines[dataset][task["precision"]]
        clean = task["corruption"] == "clean"
        manifest = subsets[dataset]["manifest"] if clean else manifest_path(root, task)
        cache = (
            resolve_inside(root, row["clean_root"], "clean root")
            if clean
            else cache_root(root, dataset, task["realization_seed"], task["corruption"])
        )
        paths = condition_paths(root, task)
        existing = [path.exists() for path in paths]
        if any(existing) and not all(existing):
            raise SystemExit(f"REALIZATION SENSITIVITY REFUSED: partial condition: {task_identity(task)}")
        if not all(existing):
            prediction, inputs_path, run_path, metric_path = paths
            run(
                [
                    sys.executable,
                    str(root / "src" / "coco_infer_trt.py"),
                    "--engine", str(engine["engine"]),
                    "--annotations", str(subsets[dataset]["annotation"]),
                    "--image-manifest", str(manifest),
                    "--manifest-cache-root", str(cache),
                    "--out", str(prediction),
                    "--input-record", str(inputs_path),
                    "--run-record", str(run_path),
                    "--condition-id", task_identity(task),
                    "--dataset", dataset,
                    "--split", row["split"],
                    "--model", "yolo11m",
                    "--precision", task["precision"],
                    "--calibrator", "entropy",
                    "--calibration-list", str(engine["calibration"]),
                    "--calibration-method", "entropy",
                    "--calibration-provenance", "verified",
                    "--corruption", task["corruption"],
                    "--severity", str(task["severity"]),
                    "--class-map", str(subsets[dataset]["class_map"]),
                    "--imgsz", str(row["imgsz"]),
                ]
            )
            run(
                [
                    sys.executable,
                    str(root / "src" / "coco_eval.py"),
                    "--annotations", str(subsets[dataset]["annotation"]),
                    "--predictions", str(prediction),
                    "--input-record", str(inputs_path),
                    "--run-record", str(run_path),
                    "--out", str(metric_path),
                ]
            )
        condition_records.append(validate_condition(root, task, manifest, engine, 512))
    if len(manifest_records) != 108 or len(condition_records) != 222:
        raise SystemExit("REALIZATION SENSITIVITY REFUSED: exact grid absent")
    report = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": config["attempt"],
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "confirmatory_analysis": str(prerequisite),
        "confirmatory_analysis_sha256": sha256_file(prerequisite),
        "realization_seeds": list(seeds),
        "manifests": manifest_records,
        "conditions": condition_records,
        "disk_preflight": {
            "source_bytes": source_bytes,
            "conservative_estimate_bytes": estimate,
            "free_bytes": free,
            "reserve_bytes": reserve,
        },
    }
    report["completion_sha256"] = canonical_hash(report, "completion_sha256")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report_out.with_suffix(report_out.suffix + ".complete").write_text(
        sha256_file(report_out) + "\n", encoding="utf-8"
    )
    print(f"REALIZATION SENSITIVITY COMPLETE conditions=222 output={report_out}")


if __name__ == "__main__":
    main()
