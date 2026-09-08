#!/usr/bin/env python3
"""Run a resumable TIDE error audit over frozen COCO-format predictions.

The runner deliberately treats TIDE's error-specific delta-AP values as
diagnostics, not as an additive decomposition of COCO AP.  It evaluates each
prediction payload at IoU=0.50, records error counts and correctable dAP, and
then forms the same clean-adjusted FP8--INT8 interaction used by the paper for
each diagnostic component.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import importlib.metadata
import json
import math
import os
import re
from concurrent.futures import ProcessPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from topic_c.manifest import sha256_file


FILENAME_RE = re.compile(
    r"^(?P<image_universe>.+?)__(?P<model>.+?)__(?P<precision>.+?)__"
    r"(?P<corruption>[a-z0-9_-]+)-s(?P<severity>[0-9]+)__.+\.json$"
)
MAIN_ERRORS = ("Cls", "Loc", "Both", "Dupe", "Bkg", "Miss")
SPECIAL_ERRORS = ("FalsePos", "FalseNeg")
DEFAULT_MODELS = ("yolo11n", "yolo11m", "yolo11x")
DEFAULT_CORRUPTIONS = ("gaussian_noise", "motion_blur", "fog", "jpeg")
DEFAULT_SEVERITIES = (1, 3, 5)
MAX_DETECTIONS = 100


def canonical_hash(document: dict[str, Any], excluded: str | None = None) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def normalize_precision(value: str) -> str:
    if value.startswith("int8"):
        return "int8"
    if value.startswith("fp8"):
        return "fp8"
    if value.startswith("fp32"):
        return "fp32"
    raise ValueError(f"unsupported precision token: {value}")


def normalize_corruption(value: str) -> str:
    return value.replace("-", "_")


def parse_prediction_name(path: Path) -> dict[str, Any]:
    match = FILENAME_RE.match(path.name)
    if match is None:
        raise ValueError(f"unrecognized prediction filename: {path.name}")
    parsed: dict[str, Any] = match.groupdict()
    parsed["precision_token"] = parsed["precision"]
    parsed["precision"] = normalize_precision(parsed["precision"])
    parsed["severity"] = int(parsed["severity"])
    zero_severity_tokens = {"clean", "codec-control", "codec_control"}
    if (parsed["corruption"] in zero_severity_tokens) != (parsed["severity"] == 0):
        raise ValueError(f"invalid clean/severity pairing: {path.name}")
    return parsed


def _read_json(path: Path, label: str) -> Any:
    if not path.is_file() or path.stat().st_size == 0:
        raise FileNotFoundError(f"missing/empty {label}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid {label}: {path}") from exc


def _resolve_dataset_path(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def expected_primary_identities(config: dict[str, Any]) -> set[tuple[str, str, str, str, int]]:
    tide = config["tide"]
    datasets = tuple(item["name"] for item in tide["datasets"])
    models = tuple(tide.get("models", DEFAULT_MODELS))
    precisions = tuple(tide["precisions"])
    corruptions = tuple(tide.get("corruptions", DEFAULT_CORRUPTIONS))
    severities = tuple(int(value) for value in tide.get("severities", DEFAULT_SEVERITIES))
    if (
        len(set(datasets)) != len(datasets)
        or len(set(models)) != len(models)
        or set(precisions) != {"int8", "fp8"}
        or len(precisions) != 2
        or len(set(corruptions)) != len(corruptions)
        or len(set(severities)) != len(severities)
        or not datasets
        or not models
        or not corruptions
        or not severities
    ):
        raise ValueError("invalid primary TIDE Cartesian-grid declaration")
    conditions = (("clean", 0),) + tuple(
        (corruption, severity)
        for corruption in corruptions
        for severity in severities
    )
    return {
        (dataset, model, precision, corruption, severity)
        for dataset in datasets
        for model in models
        for precision in precisions
        for corruption, severity in conditions
    }


def bind_source_condition(
    fields: dict[str, Any], *, source_kind: str, corruptions: set[str]
) -> dict[str, Any] | None:
    """Map the JPEG-95 filename token to clean, while rejecting source-clean arms."""
    bound = dict(fields)
    bound["filename_corruption"] = bound["corruption"]
    bound["source_corruption"] = normalize_corruption(bound["corruption"])
    if source_kind == "matched_clean":
        if bound["source_corruption"] != "codec_control" or bound["severity"] != 0:
            return None
        bound["corruption"] = "clean"
        bound["control_materialization"] = "deterministic_jpeg95"
    elif source_kind == "corrupted":
        if bound["source_corruption"] not in corruptions or bound["severity"] == 0:
            return None
        bound["control_materialization"] = "corrupted"
    else:
        raise ValueError(f"unknown TIDE source kind: {source_kind}")
    return bound


def _validate_upstream(
    *,
    dataset: dict[str, Any],
    fields: dict[str, Any],
    annotations: Path,
    prediction: Path,
    run_path: Path,
    input_path: Path,
    metric_path: Path,
) -> dict[str, Any]:
    run = _read_json(run_path, "run record")
    inputs = _read_json(input_path, "input record")
    metric = _read_json(metric_path, "metric record")
    if not all(isinstance(value, dict) for value in (run, inputs, metric)):
        raise ValueError(f"upstream records must be JSON objects: {prediction.name}")
    for key in TREATMENT_FIELDS:
        if not isinstance(run.get(key), str) or not run[key]:
            raise ValueError(f"missing upstream treatment {key}: {prediction.name}")
    prediction_sha = sha256_file(prediction)
    annotation_sha = sha256_file(annotations)
    run_sha = sha256_file(run_path)
    expected_metadata = {
        "condition_id": prediction.stem,
        "model": fields["model"],
        "corruption": fields["source_corruption"],
        "severity": fields["severity"],
    }
    for key, expected in expected_metadata.items():
        if run.get(key) != expected or metric.get(key) != expected:
            raise ValueError(f"upstream {key} mismatch for {prediction.name}")
    if (
        run.get("dataset") != dataset["name"]
        or run.get("split") != dataset["split"]
        or normalize_precision(str(run.get("precision", ""))) != fields["precision"]
        or normalize_precision(str(metric.get("precision", ""))) != fields["precision"]
        or metric.get("dataset") not in (None, dataset["name"])
        or metric.get("split") not in (None, dataset["split"])
    ):
        raise ValueError(f"upstream dataset/precision metadata mismatch for {prediction.name}")
    image_ids = inputs.get("image_ids")
    if (
        inputs.get("condition_id") != prediction.stem
        or not isinstance(image_ids, list)
        or not image_ids
        or len(set(image_ids)) != len(image_ids)
        or run.get("n_images") != len(image_ids)
        or metric.get("n_images") != len(image_ids)
        or run.get("input_manifest_sha256") != inputs.get("input_manifest_sha256")
        or metric.get("input_manifest_sha256") != inputs.get("input_manifest_sha256")
        or run.get("input_image_ids_sha256") != inputs.get("image_ids_sha256")
        or metric.get("input_image_ids_sha256") != inputs.get("image_ids_sha256")
    ):
        raise ValueError(f"upstream image-universe binding mismatch for {prediction.name}")
    if (
        run.get("prediction_sha256") != prediction_sha
        or metric.get("prediction_sha256") != prediction_sha
        or run.get("annotation_sha256") != annotation_sha
        or metric.get("run_record_sha256") != run_sha
    ):
        raise ValueError(f"upstream SHA-256 binding mismatch for {prediction.name}")
    try:
        coco_ap50 = float(metric["stats"]["AP50"]) * 100.0
        n_predictions = int(run["n_detections"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"missing numeric upstream metrics for {prediction.name}") from exc
    if not math.isfinite(coco_ap50) or not 0.0 <= coco_ap50 <= 100.0 or n_predictions < 0:
        raise ValueError(f"invalid upstream AP50/detection count for {prediction.name}")
    return {
        **{key: run[key] for key in TREATMENT_FIELDS},
        "run_record": str(run_path),
        "run_record_sha256": run_sha,
        "input_record": str(input_path),
        "input_record_sha256": sha256_file(input_path),
        "metric_record": str(metric_path),
        "metric_record_sha256": sha256_file(metric_path),
        "prediction_sha256": prediction_sha,
        "annotation_sha256": annotation_sha,
        "input_manifest_sha256": inputs["input_manifest_sha256"],
        "input_image_ids_sha256": inputs["image_ids_sha256"],
        "image_ids": [int(value) for value in image_ids],
        "n_images": len(image_ids),
        "n_predictions": n_predictions,
        "cocoeval_ap50": coco_ap50,
    }


def discover_jobs(config: dict[str, Any], root: Path) -> list[dict[str, Any]]:
    requested = set(config["tide"]["precisions"])
    jobs: list[dict[str, Any]] = []
    for dataset in config["tide"]["datasets"]:
        annotations = _resolve_dataset_path(root, dataset["annotations"])
        if not annotations.is_file():
            raise FileNotFoundError(f"annotation file is missing: {annotations}")
        sources = (
            ("matched_clean", dataset["clean_source"], int(dataset["expected_clean_files"])),
            ("corrupted", dataset["corruption_source"], int(dataset["expected_corrupted_files"])),
        )
        for source_kind, source, expected_source_files in sources:
            directory = _resolve_dataset_path(root, source["prediction_dir"])
            run_directory = _resolve_dataset_path(root, source["run_dir"])
            input_directory = _resolve_dataset_path(root, source["input_dir"])
            metric_directory = _resolve_dataset_path(root, source["metric_dir"])
            if not directory.is_dir():
                raise FileNotFoundError(f"prediction directory is missing: {directory}")
            selected = 0
            for path in sorted(directory.glob("*.json")):
                fields = parse_prediction_name(path)
                if fields["precision"] not in requested or fields["image_universe"] != dataset["filename_prefix"]:
                    continue
                fields = bind_source_condition(
                    fields,
                    source_kind=source_kind,
                    corruptions=set(config["tide"].get("corruptions", DEFAULT_CORRUPTIONS)),
                )
                if fields is None:
                    continue
                binding = _validate_upstream(
                    dataset=dataset,
                    fields=fields,
                    annotations=annotations,
                    prediction=path,
                    run_path=run_directory / path.name,
                    input_path=input_directory / path.name,
                    metric_path=metric_directory / path.name,
                )
                jobs.append(
                    {
                        **fields,
                        "dataset": dataset["name"],
                        "split": dataset["split"],
                        "annotations": str(annotations),
                        "predictions": str(path.resolve()),
                        **binding,
                    }
                )
                selected += 1
            if selected != expected_source_files:
                raise RuntimeError(
                    f"{dataset['name']} {source_kind} has {selected} selected prediction files; "
                    f"expected {expected_source_files}"
                )
    expected_total = int(config["tide"]["expected_records"])
    if len(jobs) != expected_total:
        raise RuntimeError(f"discovered {len(jobs)} jobs; expected {expected_total}")
    identities = {
        (job["dataset"], job["model"], job["precision"], job["corruption"], job["severity"])
        for job in jobs
    }
    if len(identities) != len(jobs):
        raise RuntimeError("duplicate semantic TIDE conditions were discovered")
    expected_identities = expected_primary_identities(config)
    if identities != expected_identities:
        missing = sorted(expected_identities - identities)[:5]
        extra = sorted(identities - expected_identities)[:5]
        raise RuntimeError(f"primary TIDE Cartesian grid mismatch: missing={missing}, extra={extra}")
    validate_treatment_pairing(jobs)
    return jobs


_GT_CACHE: dict[str, Any] = {}
_GT_METADATA: dict[str, tuple[int, frozenset[int]]] = {}


def load_box_ground_truth(path: str):
    """Load COCO JSON into TIDE without requiring segmentation fields."""
    if path in _GT_CACHE:
        positives, image_ids = _GT_METADATA[path]
        return _GT_CACHE[path], positives, image_ids
    from tidecv.data import Data

    document = _read_json(Path(path), "COCO annotation")
    if not isinstance(document, dict):
        raise ValueError(f"COCO annotation must be an object: {path}")
    images = document.get("images")
    annotations = document.get("annotations")
    categories = document.get("categories")
    if not isinstance(images, list) or not images or not isinstance(annotations, list) or not isinstance(categories, list):
        raise ValueError(f"invalid COCO annotation arrays: {path}")
    image_ids = [int(image["id"]) for image in images]
    category_ids = [int(category["id"]) for category in categories]
    if len(set(image_ids)) != len(image_ids) or len(set(category_ids)) != len(category_ids):
        raise ValueError(f"duplicate image/category IDs in annotations: {path}")
    image_universe = frozenset(image_ids)
    category_universe = frozenset(category_ids)
    data = Data(Path(path).stem, max_dets=MAX_DETECTIONS)
    for image, image_id in zip(images, image_ids):
        data.add_image(image_id, image.get("file_name", str(image_id)))
    for category, category_id in zip(categories, category_ids):
        data.add_class(category_id, category.get("name", str(category_id)))
    positives = 0
    for annotation in annotations:
        image_id = int(annotation["image_id"])
        category_id = int(annotation["category_id"])
        bbox = [float(value) for value in annotation["bbox"]]
        if (
            image_id not in image_universe
            or category_id not in category_universe
            or len(bbox) != 4
            or not all(math.isfinite(value) for value in bbox)
            or bbox[2] <= 0.0
            or bbox[3] <= 0.0
        ):
            raise ValueError(f"invalid bbox/reference in annotations: {path}")
        if annotation.get("iscrowd", 0):
            data.add_ignore_region(image_id, category_id, bbox, None)
        else:
            data.add_ground_truth(image_id, category_id, bbox, None)
            positives += 1
    if positives < 1:
        raise ValueError(f"annotations contain no non-crowd positives: {path}")
    _GT_CACHE[path] = data
    _GT_METADATA[path] = (positives, image_universe)
    return data, positives, image_universe


def analyze_one(job: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any]:
    from tidecv import TIDE, datasets

    annotations = Path(job["annotations"])
    predictions = Path(job["predictions"])
    if not annotations.is_file() or not predictions.is_file():
        raise FileNotFoundError(f"missing TIDE input for {predictions.name}")
    if (
        sha256_file(annotations) != job["annotation_sha256"]
        or sha256_file(predictions) != job["prediction_sha256"]
    ):
        raise RuntimeError(f"TIDE source bytes changed after preflight: {predictions.name}")
    gt, n_gt, gt_image_ids = load_box_ground_truth(str(annotations))
    if gt_image_ids != frozenset(job["image_ids"]):
        raise RuntimeError(f"TIDE GT/input image universe mismatch: {predictions.name}")
    pred_data = datasets.COCOResult(str(predictions), name=predictions.stem)
    if len(pred_data.annotations) != job["n_predictions"]:
        raise RuntimeError(f"TIDE prediction count disagrees with run record: {predictions.name}")
    tide = TIDE(
        pos_threshold=float(parameters["positive_iou"]),
        background_threshold=float(parameters["background_iou"]),
        mode=TIDE.BOX,
    )
    run = tide.evaluate(gt, pred_data, name=predictions.stem)
    main_dap = {
        error.short_name: float(value) for error, value in run.fix_main_errors().items()
    }
    main_count = {
        error.short_name: int(value) for error, value in run.count_errors().items()
    }
    special_dap = {
        error.short_name: float(value) for error, value in run.fix_special_errors().items()
    }
    missing_main = set(MAIN_ERRORS).difference(main_dap)
    missing_special = set(SPECIAL_ERRORS).difference(special_dap)
    if missing_main or missing_special:
        raise RuntimeError(
            f"TIDE returned incomplete components: main={sorted(missing_main)}, "
            f"special={sorted(missing_special)}"
        )
    numeric_values = [float(run.ap), *main_dap.values(), *special_dap.values()]
    if (
        not all(math.isfinite(value) for value in numeric_values)
        or any(value < 0.0 for value in main_dap.values())
        or any(value < 0 for value in main_count.values())
    ):
        raise RuntimeError(f"TIDE returned invalid numeric diagnostics: {predictions.name}")
    if sha256_file(predictions) != job["prediction_sha256"]:
        raise RuntimeError(f"TIDE prediction bytes changed during analysis: {predictions.name}")
    result: dict[str, Any] = {
        "schema_version": 3,
        **{key: job[key] for key in TREATMENT_FIELDS},
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "condition_id": predictions.stem,
        "dataset": job["dataset"],
        "split": job["split"],
        "image_universe": job["image_universe"],
        "model": job["model"],
        "precision": job["precision"],
        "precision_token": job["precision_token"],
        "corruption": job["corruption"],
        "filename_corruption": job["filename_corruption"],
        "source_corruption": job["source_corruption"],
        "control_materialization": job["control_materialization"],
        "severity": job["severity"],
        "positive_iou": float(parameters["positive_iou"]),
        "background_iou": float(parameters["background_iou"]),
        "max_detections_per_image": MAX_DETECTIONS,
        "tide_ap50": float(run.ap),
        "cocoeval_ap50": float(job["cocoeval_ap50"]),
        "tide_minus_cocoeval_ap50": float(run.ap) - float(job["cocoeval_ap50"]),
        "main_error_dap50": main_dap,
        "main_error_count": main_count,
        "main_error_count_per_100_gt": {
            key: 100.0 * value / n_gt for key, value in main_count.items()
        },
        "special_error_dap50": special_dap,
        "n_ground_truth": n_gt,
        "n_images": job["n_images"],
        "n_predictions": job["n_predictions"],
        "annotations": str(annotations),
        "annotations_sha256": job["annotation_sha256"],
        "predictions": str(predictions),
        "predictions_sha256": job["prediction_sha256"],
        "run_record": job["run_record"],
        "run_record_sha256": job["run_record_sha256"],
        "input_record": job["input_record"],
        "input_record_sha256": job["input_record_sha256"],
        "metric_record": job["metric_record"],
        "metric_record_sha256": job["metric_record_sha256"],
        "input_manifest_sha256": job["input_manifest_sha256"],
        "input_image_ids_sha256": job["input_image_ids_sha256"],
        "implementation_sha256": parameters["implementation_sha256"],
        "config_sha256": parameters["config_sha256"],
        "tidecv_version": parameters["tidecv_version"],
        "diagnostic_scope": (
            "TIDE isolated error fixes at IoU=0.50; components are non-additive and "
            "do not decompose COCO AP@[0.50:0.95]."
        ),
    }
    result["record_sha256"] = canonical_hash(result, "record_sha256")
    return result


def record_is_valid(path: Path, job: dict[str, Any], parameters: dict[str, Any]) -> bool:
    if not path.is_file():
        return False
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        return _record_matches_job(record, job, parameters)
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def _record_matches_job(record: dict[str, Any], job: dict[str, Any], parameters: dict[str, Any]) -> bool:
    try:
        return (
            record.get("schema_version") == 3
            and all(record.get(key) == job[key] for key in TREATMENT_FIELDS)
            and record.get("record_sha256") == canonical_hash(record, "record_sha256")
            and record.get("condition_id") == Path(job["predictions"]).stem
            and all(
                record.get(key) == job[key]
                for key in (
                    "dataset",
                    "split",
                    "model",
                    "precision",
                    "corruption",
                    "filename_corruption",
                    "source_corruption",
                    "control_materialization",
                    "severity",
                )
            )
            and record.get("predictions_sha256") == job["prediction_sha256"]
            and record.get("annotations_sha256") == job["annotation_sha256"]
            and record.get("run_record_sha256") == job["run_record_sha256"]
            and record.get("input_record_sha256") == job["input_record_sha256"]
            and record.get("metric_record_sha256") == job["metric_record_sha256"]
            and record.get("input_manifest_sha256") == job["input_manifest_sha256"]
            and record.get("input_image_ids_sha256") == job["input_image_ids_sha256"]
            and record.get("n_images") == job["n_images"]
            and record.get("n_predictions") == job["n_predictions"]
            and record.get("cocoeval_ap50") == job["cocoeval_ap50"]
            and record.get("positive_iou") == float(parameters["positive_iou"])
            and record.get("background_iou") == float(parameters["background_iou"])
            and record.get("max_detections_per_image") == MAX_DETECTIONS
            and record.get("implementation_sha256") == parameters["implementation_sha256"]
            and record.get("config_sha256") == parameters["config_sha256"]
            and record.get("tidecv_version") == parameters["tidecv_version"]
        )
    except (OSError, ValueError, KeyError, json.JSONDecodeError):
        return False


def revalidate_cached_record(path: Path, job: dict[str, Any], parameters: dict[str, Any]) -> dict[str, Any] | None:
    """Reuse only the audited v2 evaluator, with identical verified upstream bytes.

    The old implementation's GT adapter and analyze_one numerical operations are
    unchanged. This migration adds treatment metadata; it never recomputes or
    silently relabels a diagnostic from different predictions. Both original
    implementation/config digests and the original record digest are retained.
    """
    audited_implementation = "f2b6a5a8abd9c385f26978fd918b06b9a7d6885305aa1fe99b3b922a0381f703"
    try:
        record = json.loads(path.read_text(encoding="utf-8"))
        if (record.get("schema_version") != 2
            or record.get("implementation_sha256") != audited_implementation
            or record.get("record_sha256") != canonical_hash(record, "record_sha256")):
            return None
        candidate = dict(record)
        candidate.update({
            "schema_version": 3,
            **{key: job[key] for key in TREATMENT_FIELDS},
            "cache_source_record_sha256": record["record_sha256"],
            "cache_source_file_sha256": sha256_file(path),
            "cache_source_implementation_sha256": record["implementation_sha256"],
            "cache_source_config_sha256": record["config_sha256"],
            "revalidated_at_utc": datetime.now(timezone.utc).isoformat(),
            "implementation_sha256": parameters["implementation_sha256"],
            "config_sha256": parameters["config_sha256"],
        })
        candidate["record_sha256"] = canonical_hash(candidate, "record_sha256")
        return candidate if _record_matches_job(candidate, job, parameters) else None
    except (OSError, ValueError, KeyError, TypeError):
        return None


def flatten_record(record: dict[str, Any]) -> dict[str, Any]:
    row = {
        key: record[key]
        for key in (
            "condition_id",
            "dataset",
            "split",
            "model",
            "precision",
            "corruption",
            "filename_corruption",
            "source_corruption",
            "control_materialization",
            "severity",
            "tide_ap50",
            "cocoeval_ap50",
            "tide_minus_cocoeval_ap50",
            "n_ground_truth",
            "n_images",
            "n_predictions",
            "predictions_sha256",
            *TREATMENT_FIELDS,
            "input_manifest_sha256",
            "input_image_ids_sha256",
        )
    }
    for key, value in record["main_error_dap50"].items():
        row[f"dap50_{key}"] = value
    for key, value in record["main_error_count"].items():
        row[f"count_{key}"] = value
    for key, value in record["main_error_count_per_100_gt"].items():
        row[f"count_per_100_gt_{key}"] = value
    for key, value in record["special_error_dap50"].items():
        row[f"dap50_{key}"] = value
    return row


def write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"refusing to write empty table: {path}")
    fields = list(rows[0])
    if any(set(row) != set(fields) for row in rows):
        raise RuntimeError(f"non-rectangular table: {path}")
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary output exists: {temporary}")
    with temporary.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader()
        writer.writerows(rows)
    os.replace(temporary, path)


def write_json_atomic(path: Path, document: dict[str, Any]) -> None:
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    if temporary.exists():
        raise RuntimeError(f"stale temporary output exists: {temporary}")
    temporary.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    os.replace(temporary, path)


TREATMENT_FIELDS = (
    "engine_sha256", "runner_sha256", "preprocess_sha256", "decoder_sha256",
    "class_map_sha256",
)


def validate_treatment_pairing(records: list[dict[str, Any]]) -> None:
    """Fail before computation if clean/corrupt change the executable treatment."""
    treatments: dict[tuple[str, str, str], tuple[str, ...]] = {}
    inputs: dict[tuple[str, str, str, int], str] = {}
    for record in records:
        key = (record["dataset"], record["model"], record["precision"])
        # The complete CLI wrapper evolved historically; preserve and report its
        # hash separately. Engine bytes and numerical preprocessing/decoding/class
        # mapping must match. Do not claim full wrapper-source equivalence.
        identity = tuple(record.get(field) for field in TREATMENT_FIELDS if field != "runner_sha256")
        if any(not isinstance(value, str) or not value for value in identity):
            raise RuntimeError(f"missing TIDE treatment identity: {key}")
        if treatments.setdefault(key, identity) != identity:
            raise RuntimeError(f"clean-corrupt TIDE treatment mismatch: {key}")
        input_key = (record["dataset"], record["model"], record["corruption"], record["severity"])
        digest = record.get("input_manifest_sha256")
        if not digest or inputs.setdefault(input_key, digest) != digest:
            raise RuntimeError(f"TIDE encoded-input mismatch between formats: {input_key}")


def paired_interactions(records: list[dict[str, Any]]) -> list[dict[str, Any]]:
    keys = [
        (
            row["dataset"],
            row["model"],
            row["precision"],
            row["corruption"],
            row["severity"],
        )
        for row in records
    ]
    if len(set(keys)) != len(keys):
        raise RuntimeError("duplicate semantic records supplied to paired TIDE analysis")
    validate_treatment_pairing(records)
    by_key = {key: row for key, row in zip(keys, records)}
    output: list[dict[str, Any]] = []
    direct_keys = sorted(
        {
            (dataset, model, corruption, severity)
            for dataset, model, _, corruption, severity in by_key
            if corruption != "clean"
        }
    )
    for dataset, model, corruption, severity in direct_keys:
        keys = {
            "fp8_clean": (dataset, model, "fp8", "clean", 0),
            "int8_clean": (dataset, model, "int8", "clean", 0),
            "fp8_corrupt": (dataset, model, "fp8", corruption, severity),
            "int8_corrupt": (dataset, model, "int8", corruption, severity),
        }
        if not all(key in by_key for key in keys.values()):
            raise RuntimeError(f"incomplete four-cell TIDE contrast: {(dataset, model, corruption, severity)}")
        arm = {name: by_key[key] for name, key in keys.items()}
        if (
            len({value["input_image_ids_sha256"] for value in arm.values()}) != 1
            or len({value["annotations_sha256"] for value in arm.values()}) != 1
            or len({value["n_images"] for value in arm.values()}) != 1
            or len({value["n_ground_truth"] for value in arm.values()}) != 1
        ):
            raise RuntimeError(
                f"four-cell TIDE image/annotation universe mismatch: "
                f"{(dataset, model, corruption, severity)}"
            )
        row: dict[str, Any] = {
            "dataset": dataset,
            "model": model,
            "corruption": corruption,
            "severity": severity,
            "wrapper_source_identity_matched": all(
                arm[f"{precision}_clean"]["runner_sha256"] == arm[f"{precision}_corrupt"]["runner_sha256"]
                for precision in ("int8", "fp8")
            ),
        }
        tide_clean_gap = arm["fp8_clean"]["tide_ap50"] - arm["int8_clean"]["tide_ap50"]
        tide_corrupt_gap = arm["fp8_corrupt"]["tide_ap50"] - arm["int8_corrupt"]["tide_ap50"]
        coco_clean_gap = arm["fp8_clean"]["cocoeval_ap50"] - arm["int8_clean"]["cocoeval_ap50"]
        coco_corrupt_gap = arm["fp8_corrupt"]["cocoeval_ap50"] - arm["int8_corrupt"]["cocoeval_ap50"]
        row["fp8_minus_int8_tide_ap50_clean"] = tide_clean_gap
        row["fp8_minus_int8_tide_ap50_corrupt"] = tide_corrupt_gap
        row["delta_e_tide_ap50"] = tide_corrupt_gap - tide_clean_gap
        row["fp8_minus_int8_cocoeval_ap50_clean"] = coco_clean_gap
        row["fp8_minus_int8_cocoeval_ap50_corrupt"] = coco_corrupt_gap
        row["delta_e_cocoeval_ap50"] = coco_corrupt_gap - coco_clean_gap
        for component in MAIN_ERRORS:
            clean_burden = (
                arm["int8_clean"]["main_error_dap50"][component]
                - arm["fp8_clean"]["main_error_dap50"][component]
            )
            corrupt_burden = (
                arm["int8_corrupt"]["main_error_dap50"][component]
                - arm["fp8_corrupt"]["main_error_dap50"][component]
            )
            row[f"int8_minus_fp8_error_burden_dap50_{component}_clean"] = clean_burden
            row[f"int8_minus_fp8_error_burden_dap50_{component}_corrupt"] = corrupt_burden
            row[f"delta_error_burden_dap50_{component}"] = corrupt_burden - clean_burden
            clean_count_burden = (
                arm["int8_clean"]["main_error_count_per_100_gt"][component]
                - arm["fp8_clean"]["main_error_count_per_100_gt"][component]
            )
            corrupt_count_burden = (
                arm["int8_corrupt"]["main_error_count_per_100_gt"][component]
                - arm["fp8_corrupt"]["main_error_count_per_100_gt"][component]
            )
            row[f"delta_error_burden_count_per_100_gt_{component}"] = (
                corrupt_count_burden - clean_count_burden
            )
        output.append(row)
    unique = {
        (row["dataset"], row["model"], row["corruption"], row["severity"]) for row in output
    }
    if len(unique) != len(output):
        raise RuntimeError("duplicate paired TIDE interactions")
    return output


def mean(values: list[float]) -> float:
    if not values:
        raise ValueError("mean of empty values")
    return sum(values) / len(values)


def summarize_interactions(rows: list[dict[str, Any]]) -> dict[str, Any]:
    metrics = ["delta_e_cocoeval_ap50", "delta_e_tide_ap50"] + [
        f"delta_error_burden_dap50_{name}" for name in MAIN_ERRORS
    ]
    groups: dict[str, dict[str, dict[str, float]]] = {}
    groupings = {
        "overall": lambda row: "all",
        "dataset": lambda row: str(row["dataset"]),
        "model": lambda row: str(row["model"]),
        "corruption": lambda row: str(row["corruption"]),
        "severity": lambda row: str(row["severity"]),
    }
    for group_name, selector in groupings.items():
        members: dict[str, list[dict[str, Any]]] = {}
        for row in rows:
            members.setdefault(selector(row), []).append(row)
        groups[group_name] = {
            key: {metric: mean([float(row[metric]) for row in subset]) for metric in metrics}
            | {"n": len(subset)}
            for key, subset in sorted(members.items())
        }
    return {
        "schema_version": 1,
        "interpretation": (
            "TIDE dAP50 components are counterfactual diagnostics and are not an additive "
            "decomposition of COCO AP. The direct AP50 interaction is anchored to the "
            "existing COCOeval metric records. Positive component interactions mean that the "
            "INT8-minus-FP8 correctable error burden increased under corruption relative to clean."
        ),
        "n_direct_cells": len(rows),
        "groups": groups,
    }


def validate_direct_identities(
    rows: list[dict[str, Any]], config: dict[str, Any]
) -> None:
    tide = config["tide"]
    expected = {
        (dataset["name"], model, corruption, int(severity))
        for dataset in tide["datasets"]
        for model in tide.get("models", DEFAULT_MODELS)
        for corruption in tide.get("corruptions", DEFAULT_CORRUPTIONS)
        for severity in tide.get("severities", DEFAULT_SEVERITIES)
    }
    observed = {
        (row["dataset"], row["model"], row["corruption"], int(row["severity"]))
        for row in rows
    }
    if len(rows) != len(observed) or observed != expected:
        missing = sorted(expected - observed)[:5]
        extra = sorted(observed - expected)[:5]
        raise RuntimeError(f"direct TIDE Cartesian grid mismatch: missing={missing}, extra={extra}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--root", default=".")
    parser.add_argument("--workers", type=int)
    parser.add_argument("--reuse-records-from", type=Path,
                        help="revalidate unchanged diagnostics from the audited legacy evaluator")
    parser.add_argument("--limit", type=int, help="development-only limit after deterministic discovery")
    args = parser.parse_args()
    root = Path(args.root).resolve()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    parameters = dict(config["tide"]["parameters"])
    positive_iou = float(parameters.get("positive_iou", float("nan")))
    background_iou = float(parameters.get("background_iou", float("nan")))
    if (
        not math.isfinite(positive_iou)
        or not math.isfinite(background_iou)
        or not 0.0 <= background_iou < positive_iou <= 1.0
    ):
        raise ValueError("TIDE IoU thresholds must satisfy 0 <= background < positive <= 1")
    try:
        tidecv_version = importlib.metadata.version("tidecv")
    except importlib.metadata.PackageNotFoundError as exc:
        raise RuntimeError("tidecv is required; install the pinned analysis dependency") from exc
    required_tidecv = str(config["tide"].get("required_tidecv_version", ""))
    if not required_tidecv or tidecv_version != required_tidecv:
        raise RuntimeError(
            f"tidecv version mismatch: required {required_tidecv!r}, found {tidecv_version!r}"
        )
    parameters.update(
        {
            "implementation_sha256": sha256_file(__file__),
            "config_sha256": sha256_file(config_path),
            "tidecv_version": tidecv_version,
        }
    )
    jobs = discover_jobs(config, root)
    if args.limit is not None:
        if args.limit < 1:
            raise SystemExit("--limit must be positive")
        jobs = jobs[: args.limit]
    attempt = config["attempt"]
    output_root = root / "outputs" / "analysis" / attempt / "tide"
    records_dir = output_root / "records"
    records_dir.mkdir(parents=True, exist_ok=True)
    workers = args.workers or int(config["tide"]["workers"])
    if workers < 1:
        raise ValueError("--workers must be positive")

    pending: list[tuple[dict[str, Any], Path]] = []
    for job in jobs:
        output = records_dir / f"{Path(job['predictions']).stem}.json"
        if record_is_valid(output, job, parameters):
            print(f"TIDE SKIP {output.name}", flush=True)
        else:
            if output.exists():
                raise RuntimeError(f"invalid existing TIDE record; refusing overwrite: {output}")
            if args.reuse_records_from is not None:
                cached = revalidate_cached_record(args.reuse_records_from / output.name, job, parameters)
                if cached is not None:
                    write_json_atomic(output, cached)
                    print(f"TIDE REVALIDATED CACHE {output.name}", flush=True)
                    continue
            pending.append((job, output))
    print(f"TIDE PLAN total={len(jobs)} pending={len(pending)} workers={workers}", flush=True)
    with ProcessPoolExecutor(max_workers=workers) as executor:
        futures = {executor.submit(analyze_one, job, parameters): (job, path) for job, path in pending}
        completed = 0
        for future in as_completed(futures):
            _, output = futures[future]
            record = future.result()
            write_json_atomic(output, record)
            completed += 1
            print(f"TIDE COMPLETE {completed}/{len(pending)} {output.name}", flush=True)

    if args.limit is not None:
        print("TIDE DEVELOPMENT LIMIT COMPLETE; aggregate outputs intentionally omitted", flush=True)
        return
    expected_paths = {
        records_dir / f"{Path(job['predictions']).stem}.json" for job in jobs
    }
    record_paths = sorted(records_dir.glob("*.json"))
    expected = int(config["tide"]["expected_records"])
    if len(record_paths) != expected or set(record_paths) != expected_paths:
        raise RuntimeError(f"found {len(record_paths)} final records; expected {expected}")
    jobs_by_name = {Path(job["predictions"]).stem: job for job in jobs}
    if any(
        not record_is_valid(path, jobs_by_name[path.stem], parameters)
        for path in record_paths
    ):
        raise RuntimeError("one or more final TIDE records failed provenance revalidation")
    records = [json.loads(path.read_text(encoding="utf-8")) for path in record_paths]
    interactions = paired_interactions(records)
    validate_direct_identities(interactions, config)
    expected_interactions = int(config["tide"]["expected_direct_cells"])
    if len(interactions) != expected_interactions:
        raise RuntimeError(
            f"formed {len(interactions)} paired interactions; expected {expected_interactions}"
        )
    records_csv = output_root / "condition_records.csv"
    interactions_csv = output_root / "paired_error_interactions.csv"
    summary_json = output_root / "summary.json"
    write_csv(records_csv, [flatten_record(record) for record in records])
    write_csv(interactions_csv, interactions)
    summary = summarize_interactions(interactions)
    summary["created_at_utc"] = datetime.now(timezone.utc).isoformat()
    summary["config"] = str(config_path)
    summary["config_sha256"] = sha256_file(config_path)
    summary["condition_records_sha256"] = sha256_file(records_csv)
    summary["paired_error_interactions_sha256"] = sha256_file(interactions_csv)
    summary["summary_sha256"] = canonical_hash(summary, "summary_sha256")
    write_json_atomic(summary_json, summary)
    complete = {
        "schema_version": 1,
        "attempt": attempt,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "records": len(records),
        "direct_cells": len(interactions),
        "summary_sha256": sha256_file(summary_json),
        "config_sha256": parameters["config_sha256"],
        "implementation_sha256": parameters["implementation_sha256"],
        "tidecv_version": parameters["tidecv_version"],
        "record_hashes": {path.name: sha256_file(path) for path in record_paths},
        "hostname": os.uname().nodename,
    }
    complete["complete_sha256"] = canonical_hash(complete, "complete_sha256")
    write_json_atomic(output_root / "complete.json", complete)
    print(f"TIDE AUDIT COMPLETE records={len(records)} direct_cells={len(interactions)}", flush=True)


if __name__ == "__main__":
    main()
