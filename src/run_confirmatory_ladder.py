#!/usr/bin/env python3
"""Build a namespaced, hash-bound VOC/KITTI confirmatory engine ladder.

The exploratory ladder uses stable legacy filenames.  This runner deliberately
keeps the prospectively split confirmatory checkpoints, ONNX graphs, engines,
and registries under an attempt namespace so the two evidence chains cannot be
silently mixed.
"""
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

from topic_c.manifest import sha256_file


MODELS = ("yolo11n", "yolo11m", "yolo11x")
PRECISIONS = ("fp32", "fp16", "int8-entropy", "fp8")
DATASETS = ("voc", "kitti")
SAFE_ATTEMPT = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


@dataclass(frozen=True)
class ArtifactPaths:
    onnx: Path
    onnx_registry: Path
    engine: Path
    engine_registry: Path
    build_log: Path


def canonical_hash(document: dict, excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def complete(path: Path) -> bool:
    marker = path.with_suffix(path.suffix + ".complete")
    return (
        path.is_file()
        and marker.is_file()
        and marker.read_text(encoding="utf-8").strip() == sha256_file(path)
    )


def resolve_inside(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise SystemExit(f"CONFIRMATORY LADDER REFUSED: missing {label}")
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise SystemExit(f"CONFIRMATORY LADDER REFUSED: {label} escapes project root")
    return path


def load_execution_config(
    root: Path, path: Path, *, validate_dataset_assets: bool = True
) -> tuple[dict, dict[tuple[str, str], str]]:
    root, path = root.resolve(), path.resolve()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("CONFIRMATORY LADDER REFUSED: unreadable execution config") from exc
    if config.get("config_sha256") != canonical_hash(config, "config_sha256"):
        raise SystemExit("CONFIRMATORY LADDER REFUSED: execution config hash mismatch")
    attempt = config.get("attempt")
    if not isinstance(attempt, str) or not SAFE_ATTEMPT.fullmatch(attempt):
        raise SystemExit("CONFIRMATORY LADDER REFUSED: unsafe attempt namespace")
    if tuple(config.get("models", ())) != MODELS or tuple(config.get("precisions", ())) != PRECISIONS:
        raise SystemExit("CONFIRMATORY LADDER REFUSED: unexpected model/precision grid")
    dataset_config = config.get("datasets")
    if not isinstance(dataset_config, dict) or tuple(dataset_config) != DATASETS:
        raise SystemExit("CONFIRMATORY LADDER REFUSED: exact VOC/KITTI dataset order is required")
    for dataset, item in dataset_config.items():
        if (
            not isinstance(item, dict)
            or item.get("imgsz") != 640
            or not isinstance(item.get("calibration"), str)
            or item.get("partition_role") != "final"
            or not isinstance(item.get("split"), str)
            or not isinstance(item.get("expected_images"), int)
            or item["expected_images"] <= 0
        ):
            raise SystemExit(f"CONFIRMATORY LADDER REFUSED: invalid dataset config: {dataset}")
        if validate_dataset_assets:
            annotation = resolve_inside(root, item.get("annotation"), f"{dataset} annotation")
            clean_manifest = resolve_inside(root, item.get("clean_manifest"), f"{dataset} clean manifest")
            class_map = resolve_inside(root, item.get("class_map"), f"{dataset} class map")
            if not annotation.is_file() or not clean_manifest.is_file() or not class_map.is_file():
                raise SystemExit(f"CONFIRMATORY LADDER REFUSED: frozen dataset asset missing: {dataset}")
            manifest_document = json.loads(clean_manifest.read_text(encoding="utf-8"))
            class_document = json.loads(class_map.read_text(encoding="utf-8"))
            if (
                manifest_document.get("dataset") != dataset
                or class_document.get("dataset") != dataset
                or manifest_document.get("split") != item["split"]
                or class_document.get("split") != item["split"]
            ):
                raise SystemExit(
                    f"CONFIRMATORY LADDER REFUSED: machine split differs from frozen assets: {dataset}"
                )

    protocol_path = resolve_inside(root, config.get("protocol_config"), "protocol config")
    if not protocol_path.is_file() or sha256_file(protocol_path) != config.get("protocol_config_sha256"):
        raise SystemExit("CONFIRMATORY LADDER REFUSED: protocol config hash mismatch")
    protocol = json.loads(protocol_path.read_text(encoding="utf-8"))
    if (
        protocol.get("config_sha256") != canonical_hash(protocol, "config_sha256")
        or protocol.get("attempt") != attempt
        or protocol.get("protocol") != "prospectively_locked_confirmatory_resplit"
        or tuple(protocol.get("models", ())) != MODELS
        or tuple(protocol.get("formats", ())) != PRECISIONS
        or tuple(protocol.get("datasets", {})) != DATASETS
    ):
        raise SystemExit("CONFIRMATORY LADDER REFUSED: protocol and execution grids differ")

    queue_path = resolve_inside(root, config.get("training_queue"), "training queue")
    if not queue_path.is_file() or sha256_file(queue_path) != config.get("training_queue_sha256"):
        raise SystemExit("CONFIRMATORY LADDER REFUSED: training queue hash mismatch")
    queue = json.loads(queue_path.read_text(encoding="utf-8"))
    jobs: dict[tuple[str, str], str] = {}
    for row in queue.get("jobs", []):
        if not isinstance(row, dict):
            raise SystemExit("CONFIRMATORY LADDER REFUSED: malformed training job")
        key, run_id = (row.get("dataset"), row.get("model")), row.get("run_id")
        if key in jobs or key[0] not in DATASETS or key[1] not in MODELS:
            raise SystemExit("CONFIRMATORY LADDER REFUSED: duplicate or out-of-grid training job")
        if not isinstance(run_id, str) or not SAFE_ATTEMPT.fullmatch(run_id):
            raise SystemExit("CONFIRMATORY LADDER REFUSED: unsafe training run ID")
        jobs[key] = run_id
    expected = {(dataset, model) for dataset in DATASETS for model in MODELS}
    if set(jobs) != expected or len(set(jobs.values())) != len(expected):
        raise SystemExit("CONFIRMATORY LADDER REFUSED: training queue is not the exact six-job grid")
    return config, jobs


def validate_training_completion(
    root: Path, report_path: Path, jobs: dict[tuple[str, str], str]
) -> dict[tuple[str, str], Path]:
    try:
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("CONFIRMATORY LADDER REFUSED: unreadable training completion report") from exc
    if report.get("queue_report_sha256") != canonical_hash(report, "queue_report_sha256"):
        raise SystemExit("CONFIRMATORY LADDER REFUSED: training completion hash mismatch")
    rows = report.get("jobs")
    if not isinstance(rows, list) or len(rows) != len(jobs):
        raise SystemExit("CONFIRMATORY LADDER REFUSED: training completion grid mismatch")
    by_run: dict[str, dict] = {}
    for row in rows:
        if not isinstance(row, dict) or not isinstance(row.get("run_id"), str) or row["run_id"] in by_run:
            raise SystemExit("CONFIRMATORY LADDER REFUSED: duplicate training completion row")
        by_run[row["run_id"]] = row
    if set(by_run) != set(jobs.values()):
        raise SystemExit("CONFIRMATORY LADDER REFUSED: completion run IDs differ from frozen queue")

    registries: dict[tuple[str, str], Path] = {}
    for key, run_id in jobs.items():
        path = root / "manifests" / "training" / f"{run_id}.json"
        row = by_run[run_id]
        if not complete(path) or row.get("registry_sha256") != sha256_file(path):
            raise SystemExit(f"CONFIRMATORY LADDER REFUSED: training registry hash mismatch: {run_id}")
        document = json.loads(path.read_text(encoding="utf-8"))
        if document.get("dataset") != key[0] or document.get("model") != key[1]:
            raise SystemExit(f"CONFIRMATORY LADDER REFUSED: training registry identity mismatch: {run_id}")
        compaction = root / "manifests" / "training" / f"{run_id}_best_only_v1.json"
        if not complete(compaction) or row.get("compaction_report_sha256") != sha256_file(compaction):
            raise SystemExit(f"CONFIRMATORY LADDER REFUSED: compaction proof missing or invalid: {run_id}")
        compacted = json.loads(compaction.read_text(encoding="utf-8"))
        best = Path(compacted.get("retained_best_weights", "")).resolve()
        last = Path(compacted.get("deleted_last_weights", "")).resolve()
        if (
            compacted.get("run_id") != run_id
            or compacted.get("training_registry_sha256") != sha256_file(path)
            or compacted.get("retained_best_weights") != document.get("best_weights")
            or not best.is_file()
            or sha256_file(best) != document.get("best_weights_sha256")
            or sha256_file(best) != compacted.get("retained_best_weights_sha256")
            or last.exists()
        ):
            raise SystemExit(f"CONFIRMATORY LADDER REFUSED: compaction proof mismatch: {run_id}")
        registries[key] = path
    return registries


def artifact_paths(root: Path, attempt: str, dataset: str, model: str, precision: str) -> ArtifactPaths:
    if not SAFE_ATTEMPT.fullmatch(attempt) or dataset not in DATASETS or model not in MODELS or precision not in PRECISIONS:
        raise ValueError("invalid confirmatory artifact identity")
    stem = f"{dataset}_{model}_{precision}"
    graph_root = root / "engines" / attempt / dataset / model
    return ArtifactPaths(
        onnx=graph_root / f"{precision}.onnx",
        onnx_registry=root / "manifests" / "onnx" / attempt / f"{stem}.json",
        engine=graph_root / f"{precision}.plan",
        engine_registry=root / "manifests" / "engines" / attempt / f"{stem}.json",
        build_log=root / "outputs" / "logs" / "engines" / attempt / f"{stem}.log",
    )


def run(command: list[str]) -> None:
    print("CONFIRMATORY LADDER COMMAND " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def stage(
    *,
    root: Path,
    config: dict,
    training_registry: Path,
    dataset: str,
    model: str,
    precision: str,
    calibration: Path,
    imgsz: int,
) -> tuple[Path, Path]:
    attempt = config["attempt"]
    fp32 = artifact_paths(root, attempt, dataset, model, "fp32")
    if not complete(fp32.onnx_registry):
        if fp32.onnx_registry.exists() or fp32.onnx.exists() or fp32.onnx.with_suffix(".pt").exists():
            raise SystemExit(f"CONFIRMATORY LADDER REFUSED: partial FP32 export: {dataset}/{model}")
        run(
            [
                sys.executable,
                str(root / "src" / "export_yolo_onnx.py"),
                "--training-registry",
                str(training_registry),
                "--imgsz",
                str(imgsz),
                "--out",
                str(fp32.onnx),
                "--registry-out",
                str(fp32.onnx_registry),
            ]
        )
    if not complete(fp32.onnx_registry):
        raise SystemExit(f"CONFIRMATORY LADDER REFUSED: invalid FP32 registry: {dataset}/{model}")

    paths = artifact_paths(root, attempt, dataset, model, precision)
    source_registry = fp32.onnx_registry
    if precision != "fp32":
        source_registry = paths.onnx_registry
        if not complete(source_registry):
            if source_registry.exists() or paths.onnx.exists():
                raise SystemExit(f"CONFIRMATORY LADDER REFUSED: partial quantization: {dataset}/{model}/{precision}")
            command = [
                sys.executable,
                str(root / "src" / "quantize_yolo_onnx.py"),
                "--onnx-registry",
                str(fp32.onnx_registry),
                "--mode",
                precision,
                "--imgsz",
                str(imgsz),
                "--out",
                str(paths.onnx),
                "--registry-out",
                str(source_registry),
            ]
            if precision != "fp16":
                command.extend(["--calibration-list", str(calibration)])
            run(command)
    if not complete(source_registry):
        raise SystemExit(f"CONFIRMATORY LADDER REFUSED: invalid ONNX registry: {dataset}/{model}/{precision}")

    if not complete(paths.engine_registry):
        if paths.engine_registry.exists() or paths.engine.exists() or paths.build_log.exists():
            raise SystemExit(f"CONFIRMATORY LADDER REFUSED: partial engine build: {dataset}/{model}/{precision}")
        run(
            [
                sys.executable,
                str(root / "src" / "build_yolo_trt_engine.py"),
                "--onnx-registry",
                str(source_registry),
                "--precision",
                precision,
                "--trt-root",
                config["trt_root"],
                "--engine",
                str(paths.engine),
                "--build-log",
                str(paths.build_log),
                "--registry-out",
                str(paths.engine_registry),
                "--workspace",
                config["workspace"],
            ]
        )
    if not complete(paths.engine_registry):
        raise SystemExit(f"CONFIRMATORY LADDER REFUSED: invalid engine registry: {dataset}/{model}/{precision}")
    return source_registry, paths.engine_registry


def freeze_dataset_registry(
    *, dataset: str, attempt: str, records: dict[tuple[str, str], Path], output: Path
) -> None:
    expected = {(model, precision) for model in MODELS for precision in PRECISIONS}
    if set(records) != expected:
        raise SystemExit("CONFIRMATORY LADDER REFUSED: dataset registry requires exact engine grid")
    if output.exists() or output.with_suffix(output.suffix + ".complete").exists():
        raise SystemExit(f"CONFIRMATORY LADDER REFUSED: dataset registry already exists: {output}")
    engines: dict[str, dict[str, dict]] = {}
    for model in MODELS:
        engines[model] = {}
        for precision in PRECISIONS:
            path = records[(model, precision)]
            if not complete(path):
                raise SystemExit(f"CONFIRMATORY LADDER REFUSED: incomplete engine record: {path}")
            record = json.loads(path.read_text(encoding="utf-8"))
            if (
                record.get("dataset") != dataset
                or record.get("model") != model
                or record.get("precision") != precision
            ):
                raise SystemExit(f"CONFIRMATORY LADDER REFUSED: engine identity mismatch: {path}")
            engine = Path(record.get("engine", "")).resolve()
            if not engine.is_file() or sha256_file(engine) != record.get("engine_sha256"):
                raise SystemExit(f"CONFIRMATORY LADDER REFUSED: engine hash mismatch: {path}")
            engines[model][precision] = {
                "path": str(engine),
                "sha256": record["engine_sha256"],
                "engine_registry": str(path.resolve()),
                "engine_registry_sha256": sha256_file(path),
                "calibration_sha256": record.get("calibration_sha256"),
                "imgsz": record.get("imgsz"),
                "tf32_enabled": record.get("tf32_enabled"),
            }
    document = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": attempt,
        "dataset": dataset,
        "models": list(MODELS),
        "precisions": list(PRECISIONS),
        "engines": engines,
    }
    document["registry_sha256"] = canonical_hash(document, "registry_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    output.with_suffix(output.suffix + ".complete").write_text(
        sha256_file(output) + "\n", encoding="utf-8"
    )


def valid_calibration(path: Path, dataset: str) -> bool:
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.is_file() or not marker.is_file():
        return False
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    digest = document.get("calibration_sha256")
    return (
        document.get("dataset") == dataset
        and document.get("split") == "train"
        and document.get("n_images") == 512
        and isinstance(document.get("records"), list)
        and len(document["records"]) == 512
        and digest == canonical_hash(document, "calibration_sha256")
        and marker.read_text(encoding="utf-8").strip() == digest
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--wait-for-training-report", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = Path(args.config).resolve()
    training_report = Path(args.wait_for_training_report).resolve()
    report_out = Path(args.report_out).resolve()
    if report_out.exists() or report_out.with_suffix(report_out.suffix + ".complete").exists():
        raise SystemExit(f"CONFIRMATORY LADDER REFUSED: completion report already exists: {report_out}")
    config, jobs = load_execution_config(root, config_path)
    while not training_report.is_file():
        print(f"CONFIRMATORY LADDER waiting for training report: {training_report}", flush=True)
        time.sleep(args.poll_seconds)
    training = validate_training_completion(root, training_report, jobs)

    artifacts: list[dict] = []
    dataset_registries: list[dict] = []
    for dataset in DATASETS:
        data = config["datasets"][dataset]
        calibration = resolve_inside(root, data["calibration"], f"{dataset} calibration")
        if not valid_calibration(calibration, dataset):
            raise SystemExit(f"CONFIRMATORY LADDER REFUSED: invalid calibration list: {calibration}")
        engine_records: dict[tuple[str, str], Path] = {}
        for model in MODELS:
            for precision in PRECISIONS:
                onnx_registry, engine_registry = stage(
                    root=root,
                    config=config,
                    training_registry=training[(dataset, model)],
                    dataset=dataset,
                    model=model,
                    precision=precision,
                    calibration=calibration,
                    imgsz=int(data["imgsz"]),
                )
                engine_records[(model, precision)] = engine_registry
                artifacts.append(
                    {
                        "dataset": dataset,
                        "model": model,
                        "precision": precision,
                        "training_registry_sha256": sha256_file(training[(dataset, model)]),
                        "onnx_registry": str(onnx_registry),
                        "onnx_registry_sha256": sha256_file(onnx_registry),
                        "engine_registry": str(engine_registry),
                        "engine_registry_sha256": sha256_file(engine_registry),
                    }
                )
        dataset_registry = (
            root
            / "manifests"
            / "engines"
            / config["attempt"]
            / f"{dataset}_yolo11_nmx_ladder.json"
        )
        freeze_dataset_registry(
            dataset=dataset,
            attempt=config["attempt"],
            records=engine_records,
            output=dataset_registry,
        )
        dataset_registries.append(
            {
                "dataset": dataset,
                "path": str(dataset_registry),
                "sha256": sha256_file(dataset_registry),
            }
        )
    if len(artifacts) != 24:
        raise SystemExit("CONFIRMATORY LADDER REFUSED: exact 24-engine completion grid absent")
    document = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": config["attempt"],
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "training_report": str(training_report),
        "training_report_sha256": sha256_file(training_report),
        "artifacts": artifacts,
        "dataset_registries": dataset_registries,
    }
    document["ladder_report_sha256"] = canonical_hash(document, "ladder_report_sha256")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    report_out.with_suffix(report_out.suffix + ".complete").write_text(
        sha256_file(report_out) + "\n", encoding="utf-8"
    )
    print(f"CONFIRMATORY LADDER COMPLETE engines=24 report={report_out}")


if __name__ == "__main__":
    main()
