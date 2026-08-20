#!/usr/bin/env python3
"""Run the four-backend clean parity gate for confirmatory VOC/KITTI models."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path

from run_confirmatory_ladder import (
    DATASETS,
    MODELS,
    PRECISIONS,
    artifact_paths,
    canonical_hash,
    complete,
    load_execution_config,
)
from topic_c.manifest import read_manifest, sha256_file


BACKENDS = ("pytorch", "onnxruntime", "trt-fp32", "trt-fp16")
SAFE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class ParityPaths:
    prediction: Path
    input_record: Path
    run_record: Path
    metric: Path


def parity_paths(root: Path, attempt: str, dataset: str, model: str, label: str) -> ParityPaths:
    if not SAFE.fullmatch(attempt) or dataset not in DATASETS or model not in MODELS or label not in BACKENDS:
        raise ValueError("invalid confirmatory parity identity")
    base = root / "outputs" / "reference_parity" / attempt / dataset / model
    return ParityPaths(
        prediction=base / f"{label}.predictions.json",
        input_record=base / f"{label}.inputs.json",
        run_record=base / f"{label}.run.json",
        metric=base / f"{label}.metric.json",
    )


def resolve_inside(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise SystemExit(f"CONFIRMATORY PARITY REFUSED: missing {label}")
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise SystemExit(f"CONFIRMATORY PARITY REFUSED: {label} escapes project root")
    return path


def load_parity_config(root: Path, path: Path) -> dict:
    root, path = root.resolve(), path.resolve()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("CONFIRMATORY PARITY REFUSED: unreadable parity config") from exc
    if config.get("config_sha256") != canonical_hash(config, "config_sha256"):
        raise SystemExit("CONFIRMATORY PARITY REFUSED: parity config hash mismatch")
    if not isinstance(config.get("attempt"), str) or not SAFE.fullmatch(config["attempt"]):
        raise SystemExit("CONFIRMATORY PARITY REFUSED: unsafe attempt")
    if tuple(config.get("backends", ())) != BACKENDS:
        raise SystemExit("CONFIRMATORY PARITY REFUSED: exact four-backend order is required")
    source_tolerance = config.get("source_ap_tolerance_native")
    fp16_tolerance = config.get("fp16_ap_tolerance_native")
    if (
        not isinstance(source_tolerance, (int, float))
        or not 0 <= source_tolerance <= 0.01
        or not isinstance(fp16_tolerance, (int, float))
        or not 0 <= fp16_tolerance <= 0.02
    ):
        raise SystemExit("CONFIRMATORY PARITY REFUSED: invalid AP tolerance")
    execution = resolve_inside(root, config.get("execution_config"), "execution config")
    if not execution.is_file() or sha256_file(execution) != config.get("execution_config_sha256"):
        raise SystemExit("CONFIRMATORY PARITY REFUSED: execution config hash mismatch")
    manifest = config.get("source_manifest")
    if not isinstance(manifest, list) or not manifest:
        raise SystemExit("CONFIRMATORY PARITY REFUSED: empty source manifest")
    seen: set[str] = set()
    for record in manifest:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"} or record["path"] in seen:
            raise SystemExit("CONFIRMATORY PARITY REFUSED: malformed source manifest")
        seen.add(record["path"])
        source = resolve_inside(root, record["path"], "source manifest path")
        if not source.is_file() or sha256_file(source) != record["sha256"]:
            raise SystemExit(f"CONFIRMATORY PARITY REFUSED: source manifest hash mismatch: {record['path']}")
    return config


def validate_ladder_report(path: Path, *, attempt: str, execution_config: Path) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("CONFIRMATORY PARITY REFUSED: unreadable ladder report") from exc
    if report.get("ladder_report_sha256") != canonical_hash(report, "ladder_report_sha256"):
        raise SystemExit("CONFIRMATORY PARITY REFUSED: ladder report hash mismatch")
    if report.get("attempt") != attempt or report.get("config_sha256") != sha256_file(execution_config):
        raise SystemExit("CONFIRMATORY PARITY REFUSED: ladder execution binding mismatch")
    artifacts = report.get("artifacts")
    keys = {
        (row.get("dataset"), row.get("model"), row.get("precision"))
        for row in artifacts or []
        if isinstance(row, dict)
    }
    expected = {
        (dataset, model, precision)
        for dataset in DATASETS
        for model in MODELS
        for precision in PRECISIONS
    }
    if not isinstance(artifacts, list) or len(artifacts) != 24 or keys != expected:
        raise SystemExit("CONFIRMATORY PARITY REFUSED: ladder lacks exact 24-artifact grid")
    registries = report.get("dataset_registries")
    if (
        not isinstance(registries, list)
        or len(registries) != 2
        or {row.get("dataset") for row in registries if isinstance(row, dict)} != set(DATASETS)
    ):
        raise SystemExit("CONFIRMATORY PARITY REFUSED: ladder dataset registries are incomplete")
    return report


def validate_backend_output(
    paths: ParityPaths,
    *,
    label: str,
    dataset: str,
    split: str,
    model: str,
    expected_images: int,
    manifest_sha256: str,
) -> tuple[dict, dict]:
    if not all(path.is_file() for path in paths.__dict__.values()):
        raise SystemExit(f"CONFIRMATORY PARITY REFUSED: incomplete backend output: {dataset}/{model}/{label}")
    prediction_sha = sha256_file(paths.prediction)
    inputs = json.loads(paths.input_record.read_text(encoding="utf-8"))
    run = json.loads(paths.run_record.read_text(encoding="utf-8"))
    metric = json.loads(paths.metric.read_text(encoding="utf-8"))
    ids = inputs.get("image_ids")
    calculated_ids_sha = (
        hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
        if isinstance(ids, list)
        else None
    )
    expected_backend = label if label in {"pytorch", "onnxruntime"} else None
    expected_precision = {
        "pytorch": "fp32-reference",
        "onnxruntime": "fp32-reference",
        "trt-fp32": "fp32",
        "trt-fp16": "fp16",
    }[label]
    if run.get("prediction_sha256") != prediction_sha:
        raise SystemExit(f"CONFIRMATORY PARITY REFUSED: prediction hash mismatch: {dataset}/{model}/{label}")
    if (
        (expected_backend is not None and run.get("backend") != expected_backend)
        or run.get("precision") != expected_precision
        or run.get("dataset") != dataset
        or run.get("split") != split
        or run.get("model") != model
        or run.get("corruption") != "clean"
        or run.get("severity") != 0
        or run.get("n_images") != expected_images
        or run.get("input_manifest_sha256") != manifest_sha256
        or run.get("input_image_ids_sha256") != calculated_ids_sha
        or inputs.get("input_manifest_sha256") != manifest_sha256
        or inputs.get("image_ids_sha256") != calculated_ids_sha
        or len(ids or []) != expected_images
    ):
        raise SystemExit(f"CONFIRMATORY PARITY REFUSED: run/input provenance mismatch: {dataset}/{model}/{label}")
    for field in (
        "condition_id",
        "dataset",
        "split",
        "model",
        "corruption",
        "severity",
        "n_images",
        "prediction_sha256",
        "input_manifest_sha256",
        "input_image_ids_sha256",
    ):
        if metric.get(field) != run.get(field):
            raise SystemExit(f"CONFIRMATORY PARITY REFUSED: metric {field} mismatch: {dataset}/{model}/{label}")
    if metric.get("run_record_sha256") != sha256_file(paths.run_record):
        raise SystemExit(f"CONFIRMATORY PARITY REFUSED: metric run-record hash mismatch: {dataset}/{model}/{label}")
    return metric, run


def load_dataset_registry(root: Path, row: dict, attempt: str, dataset: str) -> dict:
    path = Path(row.get("path", "")).resolve()
    if not complete(path) or sha256_file(path) != row.get("sha256"):
        raise SystemExit(f"CONFIRMATORY PARITY REFUSED: dataset registry hash mismatch: {dataset}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if (
        document.get("registry_sha256") != canonical_hash(document, "registry_sha256")
        or document.get("attempt") != attempt
        or document.get("dataset") != dataset
        or tuple(document.get("models", ())) != MODELS
        or tuple(document.get("precisions", ())) != PRECISIONS
    ):
        raise SystemExit(f"CONFIRMATORY PARITY REFUSED: invalid dataset registry: {dataset}")
    for model in MODELS:
        for precision in PRECISIONS:
            entry = document.get("engines", {}).get(model, {}).get(precision, {})
            engine, registry = Path(entry.get("path", "")).resolve(), Path(entry.get("engine_registry", "")).resolve()
            if (
                not engine.is_file()
                or sha256_file(engine) != entry.get("sha256")
                or not complete(registry)
                or sha256_file(registry) != entry.get("engine_registry_sha256")
            ):
                raise SystemExit(f"CONFIRMATORY PARITY REFUSED: engine binding mismatch: {dataset}/{model}/{precision}")
            record = json.loads(registry.read_text(encoding="utf-8"))
            if record.get("tf32_enabled") is not False:
                raise SystemExit(f"CONFIRMATORY PARITY REFUSED: TF32 not disabled: {dataset}/{model}/{precision}")
    return document


def run(command: list[str]) -> None:
    print("CONFIRMATORY PARITY COMMAND " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def evaluate(root: Path, annotation: Path, paths: ParityPaths) -> None:
    run(
        [
            sys.executable,
            str(root / "src" / "coco_eval.py"),
            "--annotations",
            str(annotation),
            "--predictions",
            str(paths.prediction),
            "--input-record",
            str(paths.input_record),
            "--run-record",
            str(paths.run_record),
            "--out",
            str(paths.metric),
        ]
    )


def trt_backend_command(
    *,
    python: str,
    runner: Path,
    engine: Path,
    annotation: Path,
    manifest: Path,
    clean_root: Path,
    prediction: Path,
    input_record: Path,
    run_record: Path,
    class_map: Path,
    condition: str,
    dataset: str,
    split: str,
    model: str,
    imgsz: int,
    precision: str,
) -> list[str]:
    return [
        python,
        str(runner),
        "--engine",
        str(engine),
        "--annotations",
        str(annotation),
        "--image-manifest",
        str(manifest),
        "--manifest-cache-root",
        str(clean_root),
        "--out",
        str(prediction),
        "--input-record",
        str(input_record),
        "--run-record",
        str(run_record),
        "--condition-id",
        condition,
        "--dataset",
        dataset,
        "--split",
        split,
        "--imgsz",
        str(imgsz),
        "--model",
        model,
        "--precision",
        precision,
        "--calibrator",
        "none",
        "--corruption",
        "clean",
        "--severity",
        "0",
        "--class-map",
        str(class_map),
    ]


def execute_backend(
    *,
    root: Path,
    execution: dict,
    dataset_registry: dict,
    dataset: str,
    model: str,
    label: str,
    manifest_path: Path,
    annotation: Path,
    class_map: Path,
    clean_root: Path,
) -> ParityPaths:
    data = execution["datasets"][dataset]
    paths = parity_paths(root, execution["attempt"], dataset, model, label)
    existing = [path.exists() for path in paths.__dict__.values()]
    if any(existing) and not all(existing):
        raise SystemExit(f"CONFIRMATORY PARITY REFUSED: partial output quadruple: {dataset}/{model}/{label}")
    if all(existing):
        return paths
    condition = f"{dataset}_final__{model}__{label}__clean-s0"
    if label in {"pytorch", "onnxruntime"}:
        export_registry = artifact_paths(
            root, execution["attempt"], dataset, model, "fp32"
        ).onnx_registry
        run(
            [
                sys.executable,
                str(root / "src" / "yolo_reference_infer.py"),
                "--backend",
                label,
                "--onnx-registry",
                str(export_registry),
                "--annotations",
                str(annotation),
                "--image-manifest",
                str(manifest_path),
                "--manifest-cache-root",
                str(clean_root),
                "--class-map",
                str(class_map),
                "--out",
                str(paths.prediction),
                "--input-record",
                str(paths.input_record),
                "--run-record",
                str(paths.run_record),
                "--condition-id",
                condition,
                "--dataset",
                dataset,
                "--split",
                data["split"],
                "--model",
                model,
                "--imgsz",
                str(data["imgsz"]),
            ]
        )
    else:
        precision = "fp32" if label == "trt-fp32" else "fp16"
        engine = dataset_registry["engines"][model][precision]
        run(
            trt_backend_command(
                python=sys.executable,
                runner=root / "src" / "coco_infer_trt.py",
                engine=Path(engine["path"]),
                annotation=annotation,
                manifest=manifest_path,
                clean_root=clean_root,
                prediction=paths.prediction,
                input_record=paths.input_record,
                run_record=paths.run_record,
                class_map=class_map,
                condition=condition,
                dataset=dataset,
                split=data["split"],
                model=model,
                imgsz=data["imgsz"],
                precision=precision,
            )
        )
    evaluate(root, annotation, paths)
    return paths


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--wait-for-ladder-report", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    parity_config_path = Path(args.config).resolve()
    ladder_path = Path(args.wait_for_ladder_report).resolve()
    report_out = Path(args.report_out).resolve()
    if report_out.exists() or report_out.with_suffix(report_out.suffix + ".complete").exists():
        raise SystemExit(f"CONFIRMATORY PARITY REFUSED: completion report exists: {report_out}")
    parity_config = load_parity_config(root, parity_config_path)
    execution_path = resolve_inside(root, parity_config["execution_config"], "execution config")
    execution, _ = load_execution_config(root, execution_path)
    if execution["attempt"] != parity_config["attempt"]:
        raise SystemExit("CONFIRMATORY PARITY REFUSED: parity/execution attempt mismatch")
    while not ladder_path.is_file():
        print(f"CONFIRMATORY PARITY waiting for ladder report: {ladder_path}", flush=True)
        time.sleep(args.poll_seconds)
    ladder = validate_ladder_report(
        ladder_path, attempt=execution["attempt"], execution_config=execution_path
    )
    registry_rows = {row["dataset"]: row for row in ladder["dataset_registries"]}
    reports: list[dict] = []
    for dataset in DATASETS:
        data = execution["datasets"][dataset]
        dataset_registry = load_dataset_registry(
            root, registry_rows[dataset], execution["attempt"], dataset
        )
        annotation = resolve_inside(root, data["annotation"], f"{dataset} annotation")
        manifest_path = resolve_inside(root, data["clean_manifest"], f"{dataset} clean manifest")
        class_map = resolve_inside(root, data["class_map"], f"{dataset} class map")
        clean_root = resolve_inside(root, data["clean_root"], f"{dataset} clean root")
        manifest = read_manifest(manifest_path)
        marker = manifest_path.with_suffix(manifest_path.suffix + ".complete")
        if (
            not marker.is_file()
            or marker.read_text(encoding="utf-8").strip() != manifest.get("manifest_sha256")
            or len(manifest.get("records", [])) != data["expected_images"]
        ):
            raise SystemExit(f"CONFIRMATORY PARITY REFUSED: clean manifest incomplete: {dataset}")
        for model in MODELS:
            paths_by_label: dict[str, ParityPaths] = {}
            for label in BACKENDS:
                paths = execute_backend(
                    root=root,
                    execution=execution,
                    dataset_registry=dataset_registry,
                    dataset=dataset,
                    model=model,
                    label=label,
                    manifest_path=manifest_path,
                    annotation=annotation,
                    class_map=class_map,
                    clean_root=clean_root,
                )
                validate_backend_output(
                    paths,
                    label=label,
                    dataset=dataset,
                    split=data["split"],
                    model=model,
                    expected_images=data["expected_images"],
                    manifest_sha256=manifest["manifest_sha256"],
                )
                paths_by_label[label] = paths
            parity_report = (
                root
                / "outputs"
                / "reports"
                / "reference_parity"
                / execution["attempt"]
                / f"{dataset}_{model}.json"
            )
            if not complete(parity_report):
                if parity_report.exists() or parity_report.with_suffix(parity_report.suffix + ".complete").exists():
                    raise SystemExit(f"CONFIRMATORY PARITY REFUSED: partial parity report: {parity_report}")
                command = [
                    sys.executable,
                    str(root / "src" / "verify_reference_parity.py"),
                    "--source-tolerance",
                    str(parity_config["source_ap_tolerance_native"]),
                    "--fp16-tolerance",
                    str(parity_config["fp16_ap_tolerance_native"]),
                    "--out",
                    str(parity_report),
                ]
                for label in BACKENDS:
                    command.extend(
                        [
                            f"--{label}-metric",
                            str(paths_by_label[label].metric),
                            f"--{label}-run",
                            str(paths_by_label[label].run_record),
                        ]
                    )
                run(command)
            result = json.loads(parity_report.read_text(encoding="utf-8"))
            if not result.get("pass"):
                raise SystemExit(f"CONFIRMATORY PARITY REFUSED: parity failed: {dataset}/{model}")
            reports.append(
                {
                    "dataset": dataset,
                    "model": model,
                    "report": str(parity_report),
                    "report_sha256": sha256_file(parity_report),
                }
            )
    if len(reports) != 6:
        raise SystemExit("CONFIRMATORY PARITY REFUSED: exact six parity reports absent")
    document = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": execution["attempt"],
        "parity_config": str(parity_config_path),
        "parity_config_sha256": sha256_file(parity_config_path),
        "execution_config": str(execution_path),
        "execution_config_sha256": sha256_file(execution_path),
        "ladder_report": str(ladder_path),
        "ladder_report_sha256": sha256_file(ladder_path),
        "reports": reports,
    }
    document["parity_completion_sha256"] = canonical_hash(
        document, "parity_completion_sha256"
    )
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    report_out.with_suffix(report_out.suffix + ".complete").write_text(
        sha256_file(report_out) + "\n", encoding="utf-8"
    )
    print(f"CONFIRMATORY PARITY COMPLETE reports=6 output={report_out}")


if __name__ == "__main__":
    main()
