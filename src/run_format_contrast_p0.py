#!/usr/bin/env python3
"""Orchestrate the immutable P0 INT8--FP8 contrast evidence grid."""
from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from format_contrast_scheduler import CellJob, run_bounded_cells

from bootstrap_format_contrast import (
    ARM_NAMES,
    canonical_hash,
    component_draw_cache_identity,
    dataset_bootstrap_seed,
    create_clean_arm_cache,
    load_clean_arm_cache,
    load_draw_cache,
    materialize_bootstrap_schedule,
    resolve_relative_evidence_path,
    schedule_binding,
    validate_annotation_binding,
    validate_component_seed,
    validate_dataset_image_universes,
    validate_expected_image_count,
    validate_schedule_evidence,
    validate_linked_inputs,
    write_clean_contrast_from_cache,
)
from topic_c.manifest import sha256_file


# The read-only frozen COCO B=8 multicell probe demonstrated this exact cap
# (56.315 s; 11.673 GB aggregate RSS on a 67.116 GB host).  A higher cap has
# no resource measurement and must not enter an immutable evidence attempt.
MAX_CELL_WORKERS = 4
ATTEMPT = "ivc_format_contrast_v1"
CONFIG_RELATIVE_PATH = "configs/ivc_format_contrast_v1.json"
EXECUTION_PACKAGE_FILES = (
    CONFIG_RELATIVE_PATH,
    "src/run_format_contrast_p0.py",
    "src/format_contrast_scheduler.py",
    "src/bootstrap_format_contrast.py",
    "src/bootstrap_format_contrast_macro.py",
    "src/validate_format_contrast_evidence.py",
    "src/paired_bootstrap.py",
    "src/topic_c/manifest.py",
    "src/topic_c/tt100k_height.py",
)


def _resolved(root: Path, value: str) -> Path:
    path = Path(value)
    return path if path.is_absolute() else root / path


def _root_checked_path(root: Path, path: Path, label: str) -> Path:
    """Reject absolute/traversal/symlink escapes before any execution write."""
    root = root.resolve()
    candidate = path if path.is_absolute() else root / path
    resolved = candidate.resolve()
    if resolved != root and root not in resolved.parents:
        raise ValueError(f"{label} path escapes project root: {candidate}")
    return candidate


def _attempt_path(root: Path, config: dict[str, Any], *parts: str, label: str) -> Path:
    return _root_checked_path(root, root / "outputs" / "work" / config["attempt"] / Path(*parts), label)


def _report_path(root: Path, name: str) -> Path:
    return _root_checked_path(root, root / "outputs" / "reports" / name, "report")


def _execution_package_path(root: Path) -> Path:
    return _report_path(root, f"{ATTEMPT}_execution_package.json")


def _macro_path(root: Path, config: dict[str, Any]) -> Path:
    return _root_checked_path(
        root, root / "outputs" / "bootstrap" / config["attempt"] / f"{ATTEMPT}_joint_macro.json",
        "joint macro",
    )


def _scheduler_ledger_path(root: Path, phase: str) -> Path:
    if phase not in {"clean", "corrupt"}:
        raise ValueError("scheduler phase is invalid")
    return _report_path(root, f"{ATTEMPT}_{phase}_scheduler.json")


def validate_execution_paths(root: Path, config: dict[str, Any], tasks: dict[str, Sequence[dict[str, Any]]]) -> None:
    """Root-check every planned execution artifact before the first directory/file write."""
    root = root.resolve()
    _execution_package_path(root)
    _macro_path(root, config)
    _report_path(root, f"{ATTEMPT}_complete.json")
    _scheduler_ledger_path(root, "clean")
    _scheduler_ledger_path(root, "corrupt")
    for dataset in config.get("datasets", []):
        if not isinstance(dataset, dict) or not isinstance(dataset.get("dataset"), str):
            raise ValueError("dataset path plan is invalid")
        _attempt_path(root, config, "schedules", f"{dataset['dataset']}.npz", label="bootstrap schedule")
    for task in tasks.get("clean", []):
        _root_checked_path(root, Path(task["output"]), "contrast artifact")
        _clean_arm_cache(root, config, task)
        _cell_log(root, config, task, "clean")
    for task in tasks.get("corrupted", []):
        _root_checked_path(root, Path(task["output"]), "contrast artifact")
        _draw_cache(root, config, task)
        _clean_arm_cache(root, config, task)
        _cell_log(root, config, task, "corrupt")


def _execution_package_binding(root: Path, manifest: Path, document: dict[str, Any]) -> dict[str, str]:
    return {
        "path": str(manifest.resolve().relative_to(root.resolve())),
        "sha256": sha256_file(manifest),
        "manifest_sha256": document["manifest_sha256"],
    }


def validate_execution_package(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    """Fail closed unless the local source/config bytes match a precommitted package."""
    root = root.resolve()
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256", "manifest_sha256"}
            or any(not isinstance(binding.get(name), str) or len(binding[name]) != 64
                   for name in ("sha256", "manifest_sha256"))):
        raise ValueError("execution package binding is invalid")
    relative = Path(binding["path"])
    if relative.is_absolute() or ".." in relative.parts:
        raise ValueError("execution package path must be safe and relative")
    manifest = _root_checked_path(root, root / relative, "execution package")
    if not manifest.is_file() or sha256_file(manifest) != binding["sha256"]:
        raise ValueError("execution package file SHA-256 mismatch")
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("execution package manifest is invalid JSON") from exc
    if (not isinstance(document, dict) or document.get("schema_version") != 1 or document.get("attempt") != ATTEMPT
            or document.get("execution_root") != str(root)
            or document.get("manifest_sha256") != canonical_hash(document, "manifest_sha256")
            or document.get("manifest_sha256") != binding["manifest_sha256"]):
        raise ValueError("execution package manifest hash mismatch")
    files = document.get("files_sha256")
    if not isinstance(files, dict) or set(files) != set(EXECUTION_PACKAGE_FILES):
        raise ValueError("execution package file set is invalid")
    for value, expected in files.items():
        path = Path(value)
        if path.is_absolute() or ".." in path.parts or not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("execution package source binding is invalid")
        source = _root_checked_path(root, root / path, "execution package source")
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"execution package source SHA-256 mismatch: {value}")
    return document


def materialize_execution_package(root: Path, config_path: Path) -> dict[str, str]:
    """Precommit every production source/config byte before schedules or children exist."""
    root = root.resolve()
    expected_config = _root_checked_path(root, root / CONFIG_RELATIVE_PATH, "config")
    if config_path.resolve() != expected_config.resolve():
        raise ValueError("config path must be the reviewed project-relative production config")
    manifest = _execution_package_path(root)
    if manifest.exists():
        raise ValueError(f"execution package already exists: {manifest}")
    files: dict[str, str] = {}
    for relative in EXECUTION_PACKAGE_FILES:
        source = _root_checked_path(root, root / relative, "execution package source")
        if not source.is_file():
            raise ValueError(f"missing execution package source: {relative}")
        files[relative] = sha256_file(source)
    document: dict[str, Any] = {
        "schema_version": 1,
        "attempt": ATTEMPT,
        "execution_root": str(root),
        "files_sha256": files,
    }
    document["manifest_sha256"] = canonical_hash(document, "manifest_sha256")
    manifest.parent.mkdir(parents=True, exist_ok=True)
    with manifest.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(document, indent=2) + "\n")
    return _execution_package_binding(root, manifest, document)


def precommitted_execution_package(root: Path) -> dict[str, str]:
    """Load the parent-created package for an independently scheduled clean cell."""
    root = root.resolve()
    manifest = _execution_package_path(root)
    if not manifest.is_file():
        raise ValueError("missing precommitted execution package")
    try:
        document = json.loads(manifest.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("execution package manifest is invalid JSON") from exc
    binding = _execution_package_binding(root, manifest, document)
    validate_execution_package(root, binding)
    return binding


def scheduler_ledger_binding(
    root: Path, ledger_path: Path, execution_package: dict[str, str], phase: str, *, expected_records: int,
) -> dict[str, str]:
    """Return a completion-safe binding only for the just-verified scheduler phase."""
    if phase not in {"clean", "corrupt"} or type(expected_records) is not int or expected_records <= 0:
        raise ValueError("scheduler ledger phase/cardinality is invalid")
    root = root.resolve()
    ledger_path = _root_checked_path(root, ledger_path, "scheduler ledger")
    if not ledger_path.is_file():
        raise ValueError(f"missing scheduler ledger: {ledger_path}")
    try:
        document = json.loads(ledger_path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError("scheduler ledger is invalid JSON") from exc
    if (not isinstance(document, dict) or document.get("ledger_sha256") != canonical_hash(document, "ledger_sha256")
            or document.get("execution_package") != execution_package or document.get("status") != "complete"
            or document.get("failure") is not None or not isinstance(document.get("records"), list)
            or len(document["records"]) != expected_records):
        raise ValueError("scheduler ledger completion binding is invalid")
    return {
        "path": str(ledger_path.resolve().relative_to(root)),
        "sha256": sha256_file(ledger_path),
        "ledger_sha256": document["ledger_sha256"],
    }


def _index_records(root: Path, attempt: str, dataset: str) -> dict[tuple[str, str, str, int], tuple[dict[str, Any], Path]]:
    directory = root / "manifests" / "runs" / attempt
    if not directory.is_dir():
        raise ValueError(f"missing run-record attempt: {attempt}")
    records: dict[tuple[str, str, str, int], tuple[dict[str, Any], Path]] = {}
    for path in sorted(directory.glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        if record.get("dataset") != dataset:
            continue
        key = (str(record.get("model")), str(record.get("corruption")), int(record.get("severity", -1)))
        precision_key = (key[0], str(record.get("precision")), key[1], key[2])
        if precision_key in records:
            raise ValueError(f"duplicate run record: {attempt}/{dataset}/{precision_key}")
        records[precision_key] = (record, path)
    return records


def _record(index: dict, model: str, precision: str, corruption: str, severity: int, context: str) -> tuple[dict[str, Any], Path]:
    key = (model, precision, corruption, severity)
    if key not in index:
        raise ValueError(f"missing {precision} {context} run record: {model}/{corruption}-s{severity}")
    return index[key]


def _task(root: Path, attempt: str, dataset: dict[str, Any], model: str, corruption: str, severity: int, clean: dict, corrupt: dict | None) -> dict[str, Any]:
    int8_clean, int8_clean_path = _record(clean, model, "int8-entropy", dataset.get("clean_corruption", "clean"), 0, "clean")
    fp8_clean, fp8_clean_path = _record(clean, model, "fp8", dataset.get("clean_corruption", "clean"), 0, "clean")
    if corrupt is None:
        records = {
            "int8_clean": (int8_clean, int8_clean_path), "fp8_clean": (fp8_clean, fp8_clean_path),
            "int8_corrupt": (int8_clean, int8_clean_path), "fp8_corrupt": (fp8_clean, fp8_clean_path),
        }
    else:
        int8_corrupt, int8_corrupt_path = _record(corrupt, model, "int8-entropy", corruption, severity, "corruption")
        fp8_corrupt, fp8_corrupt_path = _record(corrupt, model, "fp8", corruption, severity, "corruption")
        records = {
            "int8_clean": (int8_clean, int8_clean_path), "fp8_clean": (fp8_clean, fp8_clean_path),
            "int8_corrupt": (int8_corrupt, int8_corrupt_path), "fp8_corrupt": (fp8_corrupt, fp8_corrupt_path),
        }
    expected_images = int(dataset["expected_images"])
    if any(record.get("n_images") != expected_images for record, _ in records.values()):
        raise ValueError(f"unexpected image count in linked run records: {dataset['dataset']}/{model}/{corruption}-s{severity}")
    input_id_hashes = {record.get("input_image_ids_sha256") for record, _ in records.values()}
    if len(input_id_hashes) != 1 or None in input_id_hashes:
        raise ValueError(f"linked run records do not share one image-ID SHA-256: {dataset['dataset']}/{model}/{corruption}-s{severity}")
    for arm_names in (("int8_clean", "fp8_clean"), ("int8_corrupt", "fp8_corrupt")):
        manifest_hashes = {records[name][0].get("input_manifest_sha256") for name in arm_names}
        if len(manifest_hashes) != 1 or None in manifest_hashes:
            raise ValueError(f"linked {arm_names[0]} / {arm_names[1]} input-manifest SHA-256 mismatch")
    annotation_hashes = {record.get("annotation_sha256") for record, _ in records.values()}
    if len(annotation_hashes) != 1 or None in annotation_hashes:
        raise ValueError(f"linked run records do not share one annotation SHA-256: {dataset['dataset']}/{model}/{corruption}-s{severity}")
    suffix = "clean-s0" if corrupt is None else f"{corruption}-s{severity}"
    return {
        "dataset": dataset["dataset"], "model": model, "corruption": corruption, "severity": severity,
        "endpoint": "tt100k-height" if dataset["dataset"] == "tt100k" else "area",
        "annotations": _resolved(root, dataset["annotations"]), "annotation_sha256": annotation_hashes.pop(),
        "expected_images": expected_images,
        "records": records,
        "output": _root_checked_path(
            root, root / "outputs" / "bootstrap" / attempt / f"{dataset['dataset']}__{model}__{suffix}.json",
            "contrast artifact",
        ),
    }


def build_tasks(root: Path, config: dict[str, Any]) -> dict[str, list[dict[str, Any]]]:
    """Derive and strictly complete the 12 clean + 144 corrupted task grid."""
    required = ("attempt", "datasets", "models", "precisions", "corruptions", "severities")
    if any(name not in config for name in required):
        raise ValueError("format-contrast config is incomplete")
    if config["precisions"] != ["int8-entropy", "fp8"]:
        raise ValueError("format-contrast config must specify INT8 and FP8 only")
    clean_tasks, corrupted_tasks = [], []
    for dataset in config["datasets"]:
        name = dataset["dataset"]
        clean = _index_records(root, dataset["clean_source_attempt"], name)
        corrupt = _index_records(root, dataset["corruption_source_attempt"], name)
        for model in config["models"]:
            clean_tasks.append(_task(root, config["attempt"], dataset, model, "clean", 0, clean, None))
            for corruption in config["corruptions"]:
                for severity in config["severities"]:
                    corrupted_tasks.append(_task(root, config["attempt"], dataset, model, corruption, int(severity), clean, corrupt))
    if len(clean_tasks) != 12 or len(corrupted_tasks) != 144:
        raise ValueError(f"expected 12 clean and 144 corrupted contrast tasks, found {len(clean_tasks)} and {len(corrupted_tasks)}")
    return {"clean": clean_tasks, "corrupted": corrupted_tasks}


def _paths(root: Path, attempt: str, record: dict[str, Any]) -> tuple[Path, Path]:
    condition = record["condition_id"]
    return (
        root / "outputs" / "predictions" / attempt / f"{condition}.json",
        root / "outputs" / "inputs" / attempt / f"{condition}.json",
    )


def _seed(namespace: str, task: dict[str, Any]) -> int:
    return dataset_bootstrap_seed(namespace, str(task["dataset"]))


def _source_paths(root: Path, task: dict[str, Any]) -> tuple[list[Path], list[Path], list[Path]]:
    predictions, inputs, runs = [], [], []
    for name in ARM_NAMES:
        record, run_path = task["records"][name]
        attempt = run_path.parent.name
        prediction, input_record = _paths(root, attempt, record)
        predictions.append(prediction); inputs.append(input_record); runs.append(run_path)
    return predictions, inputs, runs


def _verify_artifact(
    task: dict[str, Any], path: Path, n_boot: int, schedule: dict[str, Any] | None = None,
    expected_seed: int | None = None,
) -> None:
    if not path.is_file():
        raise ValueError(f"missing contrast artifact: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    validate_annotation_binding(task["annotations"], task["annotation_sha256"])
    if document.get("artifact_sha256") != canonical_hash(document, "artifact_sha256"):
        raise ValueError(f"artifact SHA-256 mismatch: {path}")
    if document.get("schema_version") != 1 or document.get("n_boot") != n_boot or document.get("n_images") != task["expected_images"]:
        raise ValueError(f"malformed contrast artifact: {path}")
    if document.get("endpoint_type") != task["endpoint"] or set(document.get("input_hashes", {})) != set(ARM_NAMES):
        raise ValueError(f"invalid contrast artifact endpoint/bindings: {path}")
    annotation = document.get("annotation", {})
    if annotation.get("sha256") != task["annotation_sha256"]:
        raise ValueError(f"artifact annotation SHA-256 mismatch: {path}")
    if schedule is not None:
        binding = document.get("bootstrap_schedule", {})
        root = path.parents[3]
        if binding != schedule_binding(root, schedule):
            raise ValueError(f"artifact bootstrap schedule binding mismatch: {path}")
        validate_schedule_evidence(root, binding)
        validate_component_seed(document, expected_seed=expected_seed if expected_seed is not None else schedule["seed"])
    for name, (record, run_path) in task["records"].items():
        prediction, input_record = _paths(path.parents[3], run_path.parent.name, record)
        binding = document["input_hashes"][name]
        if (binding.get("prediction_sha256") != sha256_file(prediction)
                or binding.get("input_record_sha256") != sha256_file(input_record)
                or binding.get("run_record_sha256") != sha256_file(run_path)
                or binding.get("input_manifest_sha256") != record.get("input_manifest_sha256")):
            raise ValueError(f"artifact linked input mismatch: {path}/{name}")
    clean_reference = document.get("clean_arm_cache")
    if not isinstance(clean_reference, dict):
        raise ValueError(f"artifact clean-arm cache binding is missing: {path}")
    if schedule is None:
        raise ValueError(f"artifact clean-arm cache requires a schedule: {path}")
    root = path.parents[3]
    clean_path = resolve_relative_evidence_path(root, clean_reference.get("path"), "clean-arm cache")
    load_clean_arm_cache(
        clean_path, endpoint=task["endpoint"], expected_images=task["expected_images"],
        annotation_sha256=task["annotation_sha256"], schedule=schedule,
        expected_sha256=clean_reference.get("sha256"), expected_identity_sha256=clean_reference.get("identity_sha256"),
        expected_dataset=task["dataset"], expected_model=task["model"],
        expected_input_hashes={name: document["input_hashes"][name] for name in ("int8_clean", "fp8_clean")},
    )


def _draw_cache(root: Path, config: dict[str, Any], task: dict[str, Any]) -> Path:
    return _attempt_path(root, config, "format_contrast_draws", f"{task['output'].stem}.npz", label="draw cache")


def _clean_arm_cache(root: Path, config: dict[str, Any], task: dict[str, Any]) -> Path:
    return _attempt_path(root, config, "clean_arm_aps", f"{task['dataset']}__{task['model']}.npz", label="clean-arm cache")


def _cell_log(root: Path, config: dict[str, Any], task: dict[str, Any], phase: str) -> Path:
    return _root_checked_path(
        root, root / "outputs" / "logs" / config["attempt"] / "format_contrast_cells" / f"{phase}__{task['output'].stem}.log",
        "task log",
    )


def build_clean_cell_jobs(
    root: Path, config_path: Path, config: dict[str, Any], tasks: Sequence[dict[str, Any]],
) -> list[CellJob]:
    """Create deterministic subprocess jobs for the 12 clean-cache cells."""
    return [
        CellJob(
            key=f"clean:{task['output'].stem}",
            command=(
                sys.executable, str(root / "src" / "run_format_contrast_p0.py"), "--project-root", str(root),
                "--config", str(config_path), "--execute-clean-cell", "--dataset", task["dataset"], "--model", task["model"],
            ),
            output_paths=(task["output"], _clean_arm_cache(root, config, task)),
            log_path=_cell_log(root, config, task, "clean"),
        )
        for task in sorted(tasks, key=lambda item: item["output"].stem)
    ]


def build_corrupt_cell_jobs(
    root: Path, config: dict[str, Any], tasks: Sequence[dict[str, Any]], schedules: dict[str, dict[str, Any]],
    clean_caches: dict[tuple[str, str], dict[str, Any]],
    source_paths: Callable[[dict[str, Any]], tuple[list[Path], list[Path], list[Path]]],
) -> list[CellJob]:
    """Create deterministic one-bootstrap-worker jobs for corrupted cells."""
    jobs = []
    for task in sorted(tasks, key=lambda item: item["output"].stem):
        reference = clean_caches[(task["dataset"], task["model"])]
        draw_cache = _draw_cache(root, config, task)
        command = contrast_command(
            root, config, task, source_paths(task), draw_cache,
            Path(schedules[task["dataset"]]["path"]), reference,
        )
        jobs.append(CellJob(
            key=f"corrupt:{task['output'].stem}", command=tuple(command),
            output_paths=(task["output"], draw_cache), log_path=_cell_log(root, config, task, "corrupt"),
        ))
    return jobs


def _dataset_image_universes(root: Path, tasks: Sequence[dict[str, Any]]) -> dict[str, list[int]]:
    """Read and validate every planned input record before evaluator construction."""
    by_dataset: dict[str, list[list[int]]] = {}
    seen_inputs: set[Path] = set()
    for task in tasks:
        for record, run_path in task["records"].values():
            _, input_path = _paths(root, run_path.parent.name, record)
            if input_path in seen_inputs:
                continue
            seen_inputs.add(input_path)
            document = json.loads(input_path.read_text(encoding="utf-8"))
            image_ids = validate_linked_inputs([document])
            validate_expected_image_count(image_ids, task["expected_images"])
            by_dataset.setdefault(task["dataset"], []).append(image_ids)
    return validate_dataset_image_universes(by_dataset)


def materialize_dataset_schedules(
    root: Path, config: dict[str, Any], universes: dict[str, list[int]],
) -> dict[str, dict[str, Any]]:
    """Create one immutable schedule per validated dataset; refuse stale work files."""
    schedules: dict[str, dict[str, Any]] = {}
    for dataset, image_ids in sorted(universes.items()):
        path = _attempt_path(root, config, "schedules", f"{dataset}.npz", label="bootstrap schedule")
        schedules[dataset] = materialize_bootstrap_schedule(
            path, image_ids, n_boot=int(config["n_boot"]), seed=dataset_bootstrap_seed(config["seed_namespace"], dataset),
        )
    return schedules


def _verify_draw_cache(
    path: Path, reference: dict[str, str], n_boot: int, schedule_sha256: str, identity_sha256: str,
) -> None:
    """Verify the bytes precommitted by a component before macro consumption."""
    if (not isinstance(reference, dict) or reference.get("identity_sha256") != identity_sha256
            or not isinstance(reference.get("sha256"), str)):
        raise ValueError(f"draw-cache component binding mismatch: {path}")
    load_draw_cache(
        path, expected_sha256=reference["sha256"], expected_identity_sha256=identity_sha256,
        n_boot=n_boot, schedule_sha256=schedule_sha256,
    )


def contrast_command(
    root: Path, config: dict[str, Any], task: dict[str, Any],
    source_paths: tuple[list[Path], list[Path], list[Path]], draw_cache: Path | None = None, schedule_path: Path | None = None,
    clean_arm_cache: dict[str, Any] | None = None,
) -> list[str]:
    """Build the exact subprocess command for one immutable contrast cell."""
    predictions, inputs, runs = source_paths
    command = [
        sys.executable, str(root / "src" / "bootstrap_format_contrast.py"), "--endpoint", task["endpoint"],
        "--annotations", str(task["annotations"]), "--expected-images", str(task["expected_images"]),
        "--annotation-sha256", task["annotation_sha256"],
    ]
    for name, prediction, input_record, run in zip(ARM_NAMES, predictions, inputs, runs):
        argument = "--" + name.replace("_", "-")
        command += [argument, str(prediction), argument + "-input", str(input_record), argument + "-run", str(run)]
    command += [
        "--n-boot", str(config["n_boot"]), "--seed", str(_seed(config["seed_namespace"], task)),
        "--workers", str(config["bootstrap_workers"]), "--out", str(task["output"]), "--evidence-root", str(root),
    ]
    if draw_cache is not None:
        command += ["--draw-cache", str(draw_cache)]
    if schedule_path is not None:
        command += ["--schedule", str(schedule_path)]
    if clean_arm_cache is not None:
        if (not isinstance(clean_arm_cache.get("path"), Path)
                or not isinstance(clean_arm_cache.get("sha256"), str)
                or not isinstance(clean_arm_cache.get("identity_sha256"), str)):
            raise ValueError("clean-arm cache reference is invalid")
        command += [
            "--clean-arm-cache", str(clean_arm_cache["path"]),
            "--clean-arm-cache-sha256", clean_arm_cache["sha256"],
            "--clean-arm-cache-identity-sha256", clean_arm_cache["identity_sha256"],
            "--dataset", task["dataset"], "--model", task["model"],
        ]
    return command


def run_task(
    root: Path, config: dict[str, Any], task: dict[str, Any], draw_cache: Path | None = None,
    schedule: dict[str, Any] | None = None, clean_arm_cache: dict[str, Any] | None = None,
) -> Path:
    output = task["output"]
    if output.exists():
        raise ValueError(f"output already exists; use a new output attempt: {output}")
    predictions, inputs, runs = _source_paths(root, task)
    validate_annotation_binding(task["annotations"], task["annotation_sha256"])
    if any(not path.is_file() or path.stat().st_size == 0 for path in predictions):
        raise ValueError(f"missing/empty prediction input for {output.name}")
    if any(not path.is_file() for path in inputs + runs):
        raise ValueError(f"missing linked input/run record for {output.name}")
    command = contrast_command(
        root, config, task, (predictions, inputs, runs), draw_cache,
        Path(schedule["path"]) if schedule is not None else None, clean_arm_cache,
    )
    subprocess.run(command, check=True)
    _verify_artifact(
        task, output, int(config["n_boot"]), schedule,
        expected_seed=_seed(config["seed_namespace"], task),
    )
    if draw_cache is not None:
        if schedule is None:
            raise ValueError("draw cache requires a bound bootstrap schedule")
        document = json.loads(output.read_text(encoding="utf-8"))
        reference = document.get("temporary_draw_cache")
        if not isinstance(reference, dict):
            raise ValueError(f"component did not precommit its draw cache: {output}")
        if resolve_relative_evidence_path(root, reference.get("path"), "draw cache") != draw_cache.resolve():
            raise ValueError(f"component draw-cache path binding mismatch: {output}")
        identity = component_draw_cache_identity(document)
        _verify_draw_cache(draw_cache, reference, int(config["n_boot"]), schedule["sha256"], identity)
    return output


def materialize_one_clean_arm_cache(root: Path, config: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Create one immutable clean cache/cell; safe to invoke in an independent process."""
    cache_path = _clean_arm_cache(root, config, task)
    if cache_path.exists():
        raise ValueError(f"fresh output attempt has an existing clean-arm cache: {cache_path}")
    schedule_path = _attempt_path(root, config, "schedules", f"{task['dataset']}.npz", label="bootstrap schedule")
    predictions, inputs, runs = _source_paths(root, task)
    cache = create_clean_arm_cache(
        endpoint=task["endpoint"], annotations=task["annotations"], prediction_paths=predictions[:2], input_paths=inputs[:2], run_paths=runs[:2],
        dataset=task["dataset"], model=task["model"],
        expected_images=task["expected_images"], annotation_sha256=task["annotation_sha256"], schedule_path=schedule_path,
        n_boot=int(config["n_boot"]), seed=_seed(config["seed_namespace"], task), workers=int(config["bootstrap_workers"]), output=cache_path,
    )
    write_clean_contrast_from_cache(cache, task["output"], annotations=task["annotations"], evidence_root=root)
    _verify_artifact(
        task, task["output"], int(config["n_boot"]), cache["bootstrap_schedule"],
        expected_seed=_seed(config["seed_namespace"], task),
    )
    clean_document = json.loads(task["output"].read_text(encoding="utf-8"))
    reference = clean_document["clean_arm_cache"]
    return {
        "path": cache_path, "sha256": reference["sha256"], "identity_sha256": reference["identity_sha256"],
    }


def _schedule_from_component(root: Path, document: dict[str, Any]) -> dict[str, Any]:
    binding = document.get("bootstrap_schedule")
    if not isinstance(binding, dict):
        raise ValueError("component bootstrap schedule binding is missing")
    return {**binding, "path": str(resolve_relative_evidence_path(root, binding.get("path"), "bootstrap schedule"))}


def verified_clean_cache_reference(root: Path, config: dict[str, Any], task: dict[str, Any]) -> dict[str, Any]:
    """Reconstruct a clean-cache reference only after all current bindings verify."""
    document = json.loads(task["output"].read_text(encoding="utf-8"))
    schedule = _schedule_from_component(root, document)
    _verify_artifact(
        task, task["output"], int(config["n_boot"]), schedule,
        expected_seed=_seed(config["seed_namespace"], task),
    )
    reference = document["clean_arm_cache"]
    return {
        "path": resolve_relative_evidence_path(root, reference["path"], "clean-arm cache"),
        "sha256": reference["sha256"], "identity_sha256": reference["identity_sha256"],
    }


def verify_corrupt_cell(root: Path, config: dict[str, Any], task: dict[str, Any], schedule: dict[str, Any]) -> dict[str, str]:
    """Verify one completed corrupt cell/cache and return ledger-safe file hashes."""
    output, draw_cache = task["output"], _draw_cache(root, config, task)
    _verify_artifact(
        task, output, int(config["n_boot"]), schedule,
        expected_seed=_seed(config["seed_namespace"], task),
    )
    document = json.loads(output.read_text(encoding="utf-8"))
    reference = document.get("temporary_draw_cache")
    if not isinstance(reference, dict):
        raise ValueError(f"component did not precommit its draw cache: {output}")
    if resolve_relative_evidence_path(root, reference.get("path"), "draw cache") != draw_cache.resolve():
        raise ValueError(f"component draw-cache path binding mismatch: {output}")
    _verify_draw_cache(
        draw_cache, reference, int(config["n_boot"]), schedule["sha256"], component_draw_cache_identity(document),
    )
    return {"artifact_sha256": sha256_file(output), "draw_cache_sha256": sha256_file(draw_cache)}


def validate_runtime_config(config: dict[str, Any]) -> None:
    """Reject unmeasured fork fan-out before an immutable remote grid begins."""
    if not isinstance(config, dict):
        raise ValueError("invalid config")
    if config.get("attempt") != ATTEMPT:
        raise ValueError(f"attempt must be exactly {ATTEMPT}")
    if (type(config.get("schema_version")) is not int or config["schema_version"] != 1
            or type(config.get("n_boot")) is not int or config["n_boot"] != 2000
            or not isinstance(config.get("seed_namespace"), str) or not config["seed_namespace"]
            or type(config.get("bootstrap_workers")) is not int or config["bootstrap_workers"] <= 0):
        raise ValueError("invalid config")
    if config["bootstrap_workers"] > 4:
        raise ValueError("bootstrap worker cap exceeds the measured memory limit of four")
    cell_workers = config.get("cell_workers")
    if type(cell_workers) is not int or cell_workers <= 0:
        raise ValueError("invalid config: cell_workers must be a positive integer")
    if cell_workers > MAX_CELL_WORKERS:
        raise ValueError("cell worker cap exceeds the measured limit of four")
    if cell_workers > 1 and config["bootstrap_workers"] != 1:
        raise ValueError("nested cell_workers and bootstrap workers are prohibited")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--execute-clean-cell", action="store_true")
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--resume-verified", action="store_true")
    args = parser.parse_args()
    root, config_path = Path(args.project_root).resolve(), Path(args.config).resolve()
    config = json.loads(config_path.read_text(encoding="utf-8"))
    try:
        validate_runtime_config(config)
        expected_config = _root_checked_path(root, root / CONFIG_RELATIVE_PATH, "config")
        if config_path != expected_config.resolve():
            raise ValueError("config path must be the reviewed project-relative production config")
    except ValueError as exc:
        raise SystemExit(f"FORMAT CONTRAST REFUSED: {exc}") from exc
    # A provenance-complete resumable scheduler has not been reviewed.  Refuse
    # before even inspecting planned tasks so stale partial evidence cannot be
    # mistaken for a candidate to repair, skip, or overwrite.
    if args.resume_verified:
        raise SystemExit("FORMAT CONTRAST REFUSED: --resume-verified is not implemented; stale partial evidence is not repaired")
    tasks = build_tasks(root, config)
    try:
        validate_execution_paths(root, config, tasks)
    except ValueError as exc:
        raise SystemExit(f"FORMAT CONTRAST REFUSED: {exc}") from exc
    if args.execute_clean_cell:
        if args.execute or not args.dataset or not args.model:
            raise SystemExit("FORMAT CONTRAST REFUSED: clean-cell mode requires exactly --dataset and --model")
        matches = [
            task for task in tasks["clean"]
            if task["dataset"] == args.dataset and task["model"] == args.model
        ]
        if len(matches) != 1:
            raise SystemExit("FORMAT CONTRAST REFUSED: clean-cell selector is not one planned task")
        try:
            precommitted_execution_package(root)
            reference = materialize_one_clean_arm_cache(root, config, matches[0])
        except ValueError as exc:
            raise SystemExit(f"FORMAT CONTRAST REFUSED: {exc}") from exc
        print(json.dumps({"clean_cell": matches[0]["output"].name, "clean_arm_cache_sha256": reference["sha256"]}))
        return
    all_tasks = tasks["clean"] + tasks["corrupted"]
    universes = _dataset_image_universes(root, all_tasks)
    if not args.execute:
        print(json.dumps({
            "dry_run": True, "clean_tasks": len(tasks["clean"]), "corrupted_tasks": len(tasks["corrupted"]),
            "validated_dataset_image_universes": {dataset: len(image_ids) for dataset, image_ids in universes.items()},
        }))
        return
    try:
        execution_package = materialize_execution_package(root, config_path)
    except ValueError as exc:
        raise SystemExit(f"FORMAT CONTRAST REFUSED: {exc}") from exc
    schedules = materialize_dataset_schedules(root, config, universes)
    clean_jobs = build_clean_cell_jobs(root, config_path, config, tasks["clean"])
    clean_by_key = {job.key: task for job, task in zip(clean_jobs, sorted(tasks["clean"], key=lambda item: item["output"].stem))}
    clean_ledger = _scheduler_ledger_path(root, "clean")
    try:
        run_bounded_cells(
            clean_jobs, cell_workers=int(config["cell_workers"]), ledger_path=clean_ledger, root=root,
            verify=lambda job: {
                "artifact_sha256": sha256_file(clean_by_key[job.key]["output"]),
                "clean_cache_sha256": verified_clean_cache_reference(root, config, clean_by_key[job.key])["sha256"],
            },
            execution_package=execution_package,
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"FORMAT CONTRAST REFUSED: clean scheduler failed: {exc}") from exc
    try:
        clean_ledger_binding = scheduler_ledger_binding(
            root, clean_ledger, execution_package, "clean", expected_records=len(tasks["clean"]),
        )
    except ValueError as exc:
        raise SystemExit(f"FORMAT CONTRAST REFUSED: clean scheduler ledger is invalid: {exc}") from exc
    clean_caches = {
        (task["dataset"], task["model"]): verified_clean_cache_reference(root, config, task)
        for task in tasks["clean"]
    }
    if len(clean_caches) != 12:
        raise SystemExit("FORMAT CONTRAST REFUSED: verified clean-cache count is incomplete")
    draw_caches = {task["output"]: _draw_cache(root, config, task) for task in tasks["corrupted"]}
    if any(path.exists() for path in draw_caches.values()):
        raise SystemExit("FORMAT CONTRAST REFUSED: a fresh output attempt has existing temporary draw cache(s)")
    corrupt_jobs = build_corrupt_cell_jobs(root, config, tasks["corrupted"], schedules, clean_caches, lambda task: _source_paths(root, task))
    corrupt_by_key = {job.key: task for job, task in zip(corrupt_jobs, sorted(tasks["corrupted"], key=lambda item: item["output"].stem))}
    corrupt_ledger = _scheduler_ledger_path(root, "corrupt")
    try:
        run_bounded_cells(
            corrupt_jobs, cell_workers=int(config["cell_workers"]), ledger_path=corrupt_ledger, root=root,
            verify=lambda job: verify_corrupt_cell(root, config, corrupt_by_key[job.key], schedules[corrupt_by_key[job.key]["dataset"]]),
            execution_package=execution_package,
        )
    except (ValueError, RuntimeError) as exc:
        raise SystemExit(f"FORMAT CONTRAST REFUSED: corrupt scheduler failed: {exc}") from exc
    try:
        corrupt_ledger_binding = scheduler_ledger_binding(
            root, corrupt_ledger, execution_package, "corrupt", expected_records=len(tasks["corrupted"]),
        )
    except ValueError as exc:
        raise SystemExit(f"FORMAT CONTRAST REFUSED: corrupt scheduler ledger is invalid: {exc}") from exc
    from bootstrap_format_contrast_macro import run_joint_macro

    try:
        validate_execution_package(root, execution_package)
    except ValueError as exc:
        raise SystemExit(f"FORMAT CONTRAST REFUSED: {exc}") from exc
    macro_path, macro_document = run_joint_macro(root, config, tasks["corrupted"], draw_caches)
    if (macro_document.get("component_cells") != 144 or len(macro_document.get("component_artifacts", {})) != 144
            or macro_document.get("n_boot") != 2000
            or set(macro_document.get("point", {})) != {"four_dataset_macro_delta_e", "area_macro_delta_psi", "tt100k_height_macro_delta_psi"}
            or {name: values.get("planned_component_count") for name, values in macro_document.get("coverage", {}).items()}
            != {"four_dataset_macro_delta_e": 144, "area_macro_delta_psi": 108, "tt100k_height_macro_delta_psi": 36}):
        raise SystemExit("FORMAT CONTRAST REFUSED: invalid joint macro artifact")
    report = _report_path(root, f"{ATTEMPT}_complete.json")
    if report.exists():
        raise SystemExit(f"FORMAT CONTRAST REFUSED: completion report already exists: {report}")
    component_hashes = {str(task["output"].relative_to(root)): sha256_file(task["output"]) for task in all_tasks}
    if len(component_hashes) != 156:
        raise SystemExit("FORMAT CONTRAST REFUSED: expected artifact hashes are incomplete")
    artifact_hashes = {**component_hashes, str(macro_path.relative_to(root)): sha256_file(macro_path)}
    component_documents = {
        relative: json.loads((root / relative).read_text(encoding="utf-8"))
        for relative in component_hashes
    }
    schedule_evidence = {dataset: schedule_binding(root, schedule) for dataset, schedule in schedules.items()}
    clean_cache_evidence = {
        document["clean_arm_cache"]["path"]: {
            "sha256": document["clean_arm_cache"]["sha256"],
            "identity_sha256": document["clean_arm_cache"]["identity_sha256"],
        }
        for document in component_documents.values()
    }
    draw_cache_evidence = {
        relative: document["temporary_draw_cache"]
        for relative, document in component_documents.items()
        if "temporary_draw_cache" in document
    }
    if len(schedule_evidence) != 4 or len(clean_cache_evidence) != 12 or len(draw_cache_evidence) != 144:
        raise SystemExit("FORMAT CONTRAST REFUSED: supporting-evidence counts are incomplete")
    supporting_hashes = {
        **{binding["path"]: binding["sha256"] for binding in schedule_evidence.values()},
        **{relative: binding["sha256"] for relative, binding in clean_cache_evidence.items()},
        **{binding["path"]: binding["sha256"] for binding in draw_cache_evidence.values()},
    }
    if len(supporting_hashes) != 160:
        raise SystemExit("FORMAT CONTRAST REFUSED: supporting-evidence paths are not unique")
    evidence_hashes = {**artifact_hashes, **supporting_hashes}
    if len(evidence_hashes) != 317:
        raise SystemExit("FORMAT CONTRAST REFUSED: complete evidence hash chain is incomplete")
    document = {
        "schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": config["attempt"], "clean_contrasts": len(tasks["clean"]), "corrupted_contrasts": len(tasks["corrupted"]),
        "n_boot": config["n_boot"], "config": str(config_path), "config_sha256": sha256_file(config_path),
        "execution_package": execution_package,
        "scheduler_ledgers": {"clean": clean_ledger_binding, "corrupt": corrupt_ledger_binding},
        "component_artifacts_sha256": component_hashes,
        "joint_macro_artifact_sha256": artifact_hashes[str(macro_path.relative_to(root))],
        "artifacts_sha256": artifact_hashes,
        "bootstrap_schedules": schedule_evidence,
        "clean_arm_caches": clean_cache_evidence,
        "draw_caches": draw_cache_evidence,
        "supporting_evidence_sha256": supporting_hashes,
        "evidence_files_sha256": evidence_hashes,
        "annotations_sha256": {task["dataset"]: task["annotation_sha256"] for task in all_tasks},
    }
    document["report_sha256"] = canonical_hash(document, "report_sha256")
    report.parent.mkdir(parents=True, exist_ok=True)
    with report.open("x", encoding="utf-8") as handle:
        handle.write(json.dumps(document, indent=2) + "\n")
    print(f"FORMAT CONTRAST COMPLETE clean=12 corrupted=144 report={report}")


if __name__ == "__main__":
    main()
