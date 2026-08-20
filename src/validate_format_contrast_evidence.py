#!/usr/bin/env python3
"""Fail-closed local validation for synchronized format-contrast evidence."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import numpy as np

from bootstrap_format_contrast import (
    ARM_NAMES,
    canonical_hash,
    component_draw_cache_identity,
    dataset_bootstrap_seed,
    load_clean_arm_cache,
    load_draw_cache,
    percentile,
    resolve_relative_evidence_path,
    validate_component_seed,
    validate_schedule_evidence,
)
from bootstrap_format_contrast_macro import macro_draws_from_components, macro_endpoint_plan
from run_format_contrast_p0 import CONFIG_RELATIVE_PATH, EXECUTION_PACKAGE_FILES, validate_runtime_config
from topic_c.manifest import sha256_file


ATTEMPT = "ivc_format_contrast_v1"
COMPONENT_COUNT = 156
ARTIFACT_COUNT = 157
SCHEDULE_COUNT = 4
CLEAN_CACHE_COUNT = 12
DRAW_CACHE_COUNT = 144
SUPPORTING_EVIDENCE_COUNT = SCHEDULE_COUNT + CLEAN_CACHE_COUNT + DRAW_CACHE_COUNT
EVIDENCE_FILE_COUNT = ARTIFACT_COUNT + SUPPORTING_EVIDENCE_COUNT
MACRO_NAME = f"outputs/bootstrap/{ATTEMPT}/{ATTEMPT}_joint_macro.json"
DATASETS = ("coco", "voc", "kitti", "tt100k")
MODELS = ("yolo11n", "yolo11m", "yolo11x")
CORRUPTIONS = ("gaussian_noise", "motion_blur", "fog", "jpeg")
SEVERITIES = (1, 3, 5)


def _component_plan() -> tuple[dict[str, dict[str, str]], list[str], list[str]]:
    """Derive the fixed complete component grid independent of mutable report declarations."""
    all_components: dict[str, dict[str, str]] = {}
    clean, corrupted = [], []
    for dataset in DATASETS:
        endpoint = "tt100k-height" if dataset == "tt100k" else "area"
        for model in MODELS:
            clean_relative = f"outputs/bootstrap/{ATTEMPT}/{dataset}__{model}__clean-s0.json"
            all_components[clean_relative] = {"dataset": dataset, "model": model, "endpoint": endpoint}
            clean.append(clean_relative)
            for corruption in CORRUPTIONS:
                for severity in SEVERITIES:
                    relative = f"outputs/bootstrap/{ATTEMPT}/{dataset}__{model}__{corruption}-s{severity}.json"
                    all_components[relative] = {"dataset": dataset, "model": model, "endpoint": endpoint}
                    corrupted.append(relative)
    return all_components, clean, corrupted


def _read_object(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing/empty {label}: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON {label}: {path}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"{label} must be a JSON object: {path}")
    return document


def _relative_evidence_path(root: Path, relative: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise ValueError("artifact path is missing")
    path = Path(relative)
    if path.is_absolute() or ".." in path.parts:
        raise ValueError(f"artifact path must be a safe relative path: {relative}")
    resolved_root, resolved_path = root.resolve(), (root / path).resolve()
    if resolved_root not in resolved_path.parents:
        raise ValueError(f"artifact path escapes project root: {relative}")
    return resolved_path


def _component_dataset_model(relative: str) -> tuple[str, str]:
    parts = Path(relative).stem.split("__")
    if len(parts) < 3 or not parts[0] or not parts[1]:
        raise ValueError(f"component filename does not bind dataset/model: {relative}")
    return parts[0], parts[1]


def _validate_execution_package(root: Path, binding: Any) -> dict[str, Any]:
    """Validate the source/config package independently of the 317 scientific files."""
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256", "manifest_sha256"}
            or any(not isinstance(binding.get(name), str) or len(binding[name]) != 64
                   for name in ("sha256", "manifest_sha256"))):
        raise ValueError("completion report execution package binding is invalid")
    path = _relative_evidence_path(root, binding["path"])
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError("execution package file SHA-256 mismatch")
    document = _read_object(path, "execution package")
    execution_root = document.get("execution_root")
    remote_root = Path(execution_root) if isinstance(execution_root, str) else None
    if (document.get("schema_version") != 1 or document.get("attempt") != ATTEMPT
            or remote_root is None or not remote_root.is_absolute() or ".." in remote_root.parts
            or len(remote_root.parts) < 2
            or document.get("manifest_sha256") != canonical_hash(document, "manifest_sha256")
            or document.get("manifest_sha256") != binding["manifest_sha256"]):
        raise ValueError("execution package manifest hash mismatch")
    files = document.get("files_sha256")
    if not isinstance(files, dict) or set(files) != set(EXECUTION_PACKAGE_FILES):
        raise ValueError("execution package file set is invalid")
    for relative, expected in files.items():
        if not isinstance(expected, str) or len(expected) != 64:
            raise ValueError("execution package source SHA-256 binding is invalid")
        source = _relative_evidence_path(root, relative)
        if not source.is_file() or sha256_file(source) != expected:
            raise ValueError(f"execution package source SHA-256 mismatch: {relative}")
    return document


def _scheduler_log_path(phase: str, relative: str) -> str:
    return f"outputs/logs/{ATTEMPT}/format_contrast_cells/{phase}__{Path(relative).stem}.log"


def _remote_evidence_path(execution_root: str, relative: str, label: str) -> str:
    """Reconstruct a remote command path from a safe evidence-relative binding."""
    path = Path(relative)
    if not isinstance(relative, str) or not relative or path.is_absolute() or ".." in path.parts:
        raise ValueError(f"{label} path is not a safe relative binding")
    return str(Path(execution_root) / path)


def _is_remote_path(value: Any, execution_root: str) -> bool:
    """Accept an absolute production argument only when it remains below the committed root."""
    if not isinstance(value, str):
        return False
    candidate, root = Path(value), Path(execution_root)
    if not candidate.is_absolute() or ".." in candidate.parts:
        return False
    try:
        candidate.relative_to(root)
    except ValueError:
        return False
    return True


def _normalized_absolute_path(value: Any, label: str) -> str:
    """Accept only an already-normalized absolute path from hash-bound metadata."""
    if not isinstance(value, str) or not value:
        raise ValueError(f"{label} path is missing")
    path = Path(value)
    if not path.is_absolute() or ".." in path.parts or str(path) != value:
        raise ValueError(f"{label} path must be normalized and absolute")
    return value


def _valid_python_executable(value: Any) -> bool:
    path = Path(value) if isinstance(value, str) else None
    return bool(path is not None and path.is_absolute() and ".." not in path.parts)


def _validate_clean_command(command: list[str], execution_root: str, relative: str) -> None:
    dataset, model = _component_dataset_model(relative)
    expected_tail = [
        _remote_evidence_path(execution_root, "src/run_format_contrast_p0.py", "clean runner"),
        "--project-root", execution_root,
        "--config", _remote_evidence_path(execution_root, CONFIG_RELATIVE_PATH, "config"),
        "--execute-clean-cell", "--dataset", dataset, "--model", model,
    ]
    if not isinstance(command, list) or not command or not _valid_python_executable(command[0]) or command[1:] != expected_tail:
        raise ValueError(f"scheduler clean command is invalid: {relative}")


def _consume_command_option(command: list[str], index: int, option: str, expected: str | None = None) -> tuple[int, str]:
    if index + 1 >= len(command) or command[index] != option:
        raise ValueError(f"expected option {option}")
    value = command[index + 1]
    if expected is not None and value != expected:
        raise ValueError(f"option {option} has an unexpected value")
    return index + 2, value


def _validate_corrupt_command(command: list[str], execution_root: str, relative: str, component: dict[str, Any]) -> None:
    if (not isinstance(command, list) or len(command) < 2 or not _valid_python_executable(command[0])
            or command[1] != _remote_evidence_path(execution_root, "src/bootstrap_format_contrast.py", "contrast runner")):
        raise ValueError(f"scheduler corrupt command is invalid: {relative}")
    try:
        index, _ = _consume_command_option(command, 2, "--endpoint", component["endpoint_type"])
        annotation = component["annotation"]
        expected_annotation_path = _normalized_absolute_path(annotation["path"], "component annotation")
        index, _ = _consume_command_option(command, index, "--annotations", expected_annotation_path)
        index, _ = _consume_command_option(command, index, "--expected-images", str(component["n_images"]))
        index, _ = _consume_command_option(command, index, "--annotation-sha256", annotation["sha256"])
        for arm in ARM_NAMES:
            option = "--" + arm.replace("_", "-")
            for arm_option in (option, option + "-input", option + "-run"):
                index, source = _consume_command_option(command, index, arm_option)
                if not _is_remote_path(source, execution_root):
                    raise ValueError(f"{arm_option} must be rooted in the execution package")
        index, _ = _consume_command_option(command, index, "--n-boot", "2000")
        index, _ = _consume_command_option(command, index, "--seed", str(component["seed"]))
        index, _ = _consume_command_option(command, index, "--workers", "1")
        index, _ = _consume_command_option(
            command, index, "--out", _remote_evidence_path(execution_root, relative, "contrast artifact"),
        )
        index, _ = _consume_command_option(command, index, "--evidence-root", execution_root)
        temporary = component["temporary_draw_cache"]
        index, _ = _consume_command_option(
            command, index, "--draw-cache", _remote_evidence_path(execution_root, temporary["path"], "draw cache"),
        )
        schedule = component["bootstrap_schedule"]
        index, _ = _consume_command_option(
            command, index, "--schedule", _remote_evidence_path(execution_root, schedule["path"], "bootstrap schedule"),
        )
        clean = component["clean_arm_cache"]
        index, _ = _consume_command_option(
            command, index, "--clean-arm-cache", _remote_evidence_path(execution_root, clean["path"], "clean-arm cache"),
        )
        index, _ = _consume_command_option(command, index, "--clean-arm-cache-sha256", clean["sha256"])
        index, _ = _consume_command_option(command, index, "--clean-arm-cache-identity-sha256", clean["identity_sha256"])
        dataset, model = _component_dataset_model(relative)
        index, _ = _consume_command_option(command, index, "--dataset", dataset)
        index, _ = _consume_command_option(command, index, "--model", model)
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"scheduler corrupt command is invalid: {relative}") from exc
    if index != len(command):
        raise ValueError(f"scheduler corrupt command is invalid: {relative}")


def _validate_scheduler_ledger(
    root: Path, binding: Any, package: dict[str, Any], package_document: dict[str, Any], config: dict[str, Any],
    phase: str, relatives: list[str], component_documents: dict[str, dict[str, Any]],
) -> None:
    """Verify every scheduler record, preserved output, and task log in one phase."""
    if (not isinstance(binding, dict) or set(binding) != {"path", "sha256", "ledger_sha256"}
            or any(not isinstance(binding.get(name), str) or len(binding[name]) != 64
                   for name in ("sha256", "ledger_sha256"))):
        raise ValueError(f"scheduler {phase} ledger binding is invalid")
    path = _relative_evidence_path(root, binding["path"])
    if not path.is_file() or sha256_file(path) != binding["sha256"]:
        raise ValueError("scheduler ledger file SHA-256 mismatch")
    ledger = _read_object(path, f"scheduler {phase} ledger")
    if (ledger.get("ledger_sha256") != canonical_hash(ledger, "ledger_sha256")
            or ledger.get("ledger_sha256") != binding["ledger_sha256"]
            or ledger.get("execution_package") != package
            or ledger.get("schema_version") != 1
            or ledger.get("scheduler") != "bounded independent immutable format-contrast cells"
            or ledger.get("status") != "complete" or ledger.get("failure") is not None):
        raise ValueError(f"scheduler {phase} ledger integrity is invalid")
    if (type(ledger.get("cell_workers")) is not int or ledger["cell_workers"] != config["cell_workers"]
            or config["cell_workers"] != 4):
        raise ValueError(f"scheduler worker cap is invalid: {phase}")
    if (type(ledger.get("peak_active")) is not int or not 1 <= ledger["peak_active"] <= ledger["cell_workers"]):
        raise ValueError(f"scheduler peak is invalid: {phase}")
    records = ledger.get("records")
    if not isinstance(records, list) or len(records) != len(relatives):
        raise ValueError(f"scheduler {phase} ledger cardinality is invalid")
    expected: dict[str, tuple[str, str, list[str]]] = {}
    for relative in relatives:
        component = component_documents[relative]
        outputs = [relative]
        if phase == "clean":
            outputs.append(component["clean_arm_cache"]["path"])
        else:
            outputs.append(component["temporary_draw_cache"]["path"])
        expected[f"{phase}:{Path(relative).stem}"] = (relative, _scheduler_log_path(phase, relative), outputs)
    by_key = {record.get("key"): record for record in records if isinstance(record, dict)}
    if len(by_key) != len(records) or set(by_key) != set(expected):
        raise ValueError(f"scheduler {phase} ledger task membership is invalid")
    execution_root = package_document["execution_root"]
    for key, (relative, log_relative, output_relatives) in expected.items():
        record = by_key[key]
        if (record.get("log_path") != log_relative or record.get("output_paths") != output_relatives
                or record.get("missing_output_paths") != [] or record.get("exit_status") != 0
                or not isinstance(record.get("pid"), int) or record["pid"] <= 0
                or not isinstance(record.get("command"), list) or not all(isinstance(value, str) for value in record["command"])
                or not isinstance(record.get("started_at_utc"), str) or not isinstance(record.get("ended_at_utc"), str)):
            raise ValueError(f"scheduler {phase} ledger record is invalid: {key}")
        log = _relative_evidence_path(root, log_relative)
        if not log.is_file() or sha256_file(log) != record.get("log_sha256"):
            raise ValueError(f"scheduler task log SHA-256 mismatch: {key}")
        expected_outputs: dict[str, str] = {}
        for output_relative in output_relatives:
            output = _relative_evidence_path(root, output_relative)
            if not output.is_file():
                raise ValueError(f"scheduler declared output is missing: {key}/{output_relative}")
            expected_outputs[output_relative] = sha256_file(output)
        if record.get("output_sha256") != expected_outputs:
            raise ValueError(f"scheduler output SHA-256 mismatch: {key}")
        if phase == "clean":
            _validate_clean_command(record["command"], execution_root, relative)
        else:
            _validate_corrupt_command(record["command"], execution_root, relative, component_documents[relative])


def _validate_component(root: Path, document: dict[str, Any], relative: str, *, expected_seed: int) -> None:
    if document.get("artifact_sha256") != canonical_hash(document, "artifact_sha256"):
        raise ValueError(f"component artifact self-hash mismatch: {relative}")
    if document.get("schema_version") != 1 or document.get("n_boot") != 2000:
        raise ValueError(f"invalid component artifact schema/bootstrap count: {relative}")
    if document.get("endpoint_type") not in {"area", "tt100k-height"}:
        raise ValueError(f"invalid component endpoint: {relative}")
    annotation = document.get("annotation")
    if (not isinstance(annotation, dict) or set(annotation) != {"path", "sha256"}
            or not isinstance(annotation.get("sha256"), str) or len(annotation["sha256"]) != 64):
        raise ValueError(f"component annotation binding is invalid: {relative}")
    _normalized_absolute_path(annotation["path"], "component annotation")
    if set(document.get("input_hashes", {})) != set(ARM_NAMES):
        raise ValueError(f"component input-hash binding is incomplete: {relative}")
    schedule = document.get("bootstrap_schedule")
    if not isinstance(schedule, dict) or schedule.get("n_boot") != 2000:
        raise ValueError(f"component bootstrap schedule binding is invalid: {relative}")
    validate_schedule_evidence(root, schedule)
    validate_component_seed(document, expected_seed=expected_seed)
    if {binding.get("image_ids_sha256") for binding in document["input_hashes"].values()} != {schedule["image_ids_sha256"]}:
        raise ValueError(f"component input image universe does not match schedule: {relative}")
    dataset, model = _component_dataset_model(relative)
    clean_reference = document.get("clean_arm_cache")
    if (not isinstance(clean_reference, dict) or set(clean_reference) != {"path", "sha256", "identity_sha256"}):
        raise ValueError(f"component clean-arm cache binding is invalid: {relative}")
    clean_path = resolve_relative_evidence_path(root, clean_reference["path"], "clean-arm cache")
    load_clean_arm_cache(
        clean_path, endpoint=document["endpoint_type"], expected_images=document["n_images"],
        annotation_sha256=annotation["sha256"], schedule=schedule,
        expected_sha256=clean_reference["sha256"], expected_identity_sha256=clean_reference["identity_sha256"],
        expected_dataset=dataset, expected_model=model,
        expected_input_hashes={name: document["input_hashes"][name] for name in ("int8_clean", "fp8_clean")},
    )
    is_clean = Path(relative).name.endswith("clean-s0.json")
    temporary = document.get("temporary_draw_cache")
    if is_clean:
        if temporary is not None:
            raise ValueError(f"clean component must not carry a macro draw cache: {relative}")
    else:
        if not isinstance(temporary, dict) or set(temporary) != {"path", "sha256", "identity_sha256"}:
            raise ValueError(f"corrupted component draw-cache binding is invalid: {relative}")
        draw_path = resolve_relative_evidence_path(root, temporary["path"], "draw cache")
        identity = component_draw_cache_identity(document)
        if temporary["identity_sha256"] != identity:
            raise ValueError(f"component draw-cache identity mismatch: {relative}")
        load_draw_cache(
            draw_path, expected_sha256=temporary["sha256"], expected_identity_sha256=identity,
            n_boot=2000, schedule_sha256=schedule["sha256"],
        )


def _same_statistic(actual: Any, expected: Any) -> bool:
    """Compare JSON numeric statistics exactly, including the deliberate all-NaN case."""
    try:
        return bool(np.array_equal(np.asarray(actual, dtype=float), np.asarray(expected, dtype=float), equal_nan=True))
    except (TypeError, ValueError):
        return False


def _validate_macro(
    root: Path, document: dict[str, Any], component_hashes: dict[str, str], component_documents: dict[str, dict[str, Any]],
    corrupted_relatives: list[str], component_plan: dict[str, dict[str, str]],
) -> None:
    if document.get("n_boot") != 2000 or document.get("component_cells") != 144:
        raise ValueError("invalid joint macro bootstrap/component count")
    components = document.get("component_artifacts")
    if not isinstance(components, dict) or len(components) != 144:
        raise ValueError("joint macro must bind exactly 144 corrupted component artifacts")
    if set(components) != set(corrupted_relatives):
        raise ValueError("joint macro exact corrupted component membership is invalid")
    caches: list[dict[str, np.ndarray]] = []
    tasks: list[dict[str, Any]] = []
    for relative in corrupted_relatives:
        binding = components[relative]
        if not isinstance(binding, dict) or binding.get("artifact_sha256") != component_hashes[relative]:
            raise ValueError(f"joint macro component-hash binding mismatch: {relative}")
        component = component_documents[relative]
        if (binding.get("bootstrap_schedule") != component.get("bootstrap_schedule")
                or binding.get("clean_arm_cache") != component.get("clean_arm_cache")
                or binding.get("temporary_draw_cache") != component.get("temporary_draw_cache")):
            raise ValueError(f"joint macro cache/schedule binding mismatch: {relative}")
        expected = component_plan[relative]
        if component.get("endpoint_type") != expected["endpoint"]:
            raise ValueError(f"joint macro component endpoint membership is invalid: {relative}")
        temporary = component["temporary_draw_cache"]
        draw_path = resolve_relative_evidence_path(root, temporary["path"], "draw cache")
        identity = component_draw_cache_identity(component)
        caches.append(load_draw_cache(
            draw_path, expected_sha256=temporary["sha256"], expected_identity_sha256=identity,
            n_boot=2000, schedule_sha256=component["bootstrap_schedule"]["sha256"],
        ))
        tasks.append({"dataset": expected["dataset"], "endpoint": expected["endpoint"]})
    coverage = document.get("coverage", {})
    expected_coverage = {
        "four_dataset_macro_delta_e": 144,
        "area_macro_delta_psi": 108,
        "tt100k_height_macro_delta_psi": 36,
    }
    if {name: values.get("planned_component_count") for name, values in coverage.items()} != expected_coverage:
        raise ValueError("joint macro planned coverage is invalid")
    plan = macro_endpoint_plan(tasks)
    metric_draws = macro_draws_from_components(tasks, caches)
    point_by_task = {id(task): component_documents[relative]["point"] for task, relative in zip(tasks, corrupted_relatives)}
    expected_point = {
        "four_dataset_macro_delta_e": float(np.mean([
            point_by_task[id(task)]["delta_e"]["all"] for task in plan["four_dataset_macro_delta_e"]
        ])),
        "area_macro_delta_psi": float(np.mean([
            point_by_task[id(task)]["delta_psi"] for task in plan["area_macro_delta_psi"]
        ])),
        "tt100k_height_macro_delta_psi": float(np.mean([
            point_by_task[id(task)]["delta_psi"] for task in plan["tt100k_height_macro_delta_psi"]
        ])),
    }
    if set(document.get("point", {})) != set(expected_point) or any(
        not _same_statistic(document["point"][name], value) for name, value in expected_point.items()
    ):
        raise ValueError("joint macro point recomputation mismatch")
    expected_intervals = {name: percentile(draws) for name, draws in metric_draws.items()}
    if set(document.get("percentile_intervals", {})) != set(expected_intervals) or any(
        not _same_statistic(document["percentile_intervals"][name], value)
        for name, value in expected_intervals.items()
    ):
        raise ValueError("joint macro percentile recomputation mismatch")
    expected_finite = {name: int(np.count_nonzero(np.isfinite(draws))) for name, draws in metric_draws.items()}
    if any(
        coverage[name].get("finite_complete_replicates") != expected_finite[name]
        or coverage[name].get("total_replicates") != 2000
        for name in expected_finite
    ):
        raise ValueError("joint macro finite coverage recomputation mismatch")
    if document.get("artifact_sha256") != canonical_hash(document, "artifact_sha256"):
        raise ValueError("joint macro artifact self-hash mismatch")


def validate_complete_report(root: Path, config_path: Path) -> dict[str, Any]:
    """Validate all synchronized summaries against the immutable completion chain."""
    root, config_path = root.resolve(), config_path.resolve()
    report_path = root / "outputs" / "reports" / f"{ATTEMPT}_complete.json"
    report = _read_object(report_path, "completion report")
    if report.get("report_sha256") != canonical_hash(report, "report_sha256"):
        raise ValueError("completion report self-hash mismatch")
    if (report.get("attempt") != ATTEMPT or report.get("clean_contrasts") != 12
            or report.get("corrupted_contrasts") != 144 or report.get("n_boot") != 2000):
        raise ValueError("completion report grid/bootstrap contract is invalid")
    if not config_path.is_file() or sha256_file(config_path) != report.get("config_sha256"):
        raise ValueError("local configuration does not match the remote completion report")
    package_binding = report.get("execution_package")
    package_document = _validate_execution_package(root, package_binding)
    scheduler_ledgers = report.get("scheduler_ledgers")
    if not isinstance(scheduler_ledgers, dict) or set(scheduler_ledgers) != {"clean", "corrupt"}:
        raise ValueError("completion report scheduler ledger bindings are invalid")
    component_hashes = report.get("component_artifacts_sha256")
    artifact_hashes = report.get("artifacts_sha256")
    schedules_report = report.get("bootstrap_schedules")
    clean_caches_report = report.get("clean_arm_caches")
    draw_caches_report = report.get("draw_caches")
    supporting_hashes = report.get("supporting_evidence_sha256")
    evidence_hashes = report.get("evidence_files_sha256")
    component_plan, clean_relatives, corrupted_relatives = _component_plan()
    if not isinstance(component_hashes, dict) or len(component_hashes) != COMPONENT_COUNT:
        raise ValueError("completion report must bind exactly 156 component artifacts")
    if set(component_hashes) != set(component_plan):
        raise ValueError("completion report exact component grid membership is invalid")
    if not isinstance(artifact_hashes, dict) or len(artifact_hashes) != ARTIFACT_COUNT:
        raise ValueError("completion report must bind exactly 157 artifacts including joint macro")
    if set(component_hashes) - set(artifact_hashes):
        raise ValueError("component artifacts are missing from full completion binding")
    if set(artifact_hashes) - set(component_hashes) != {MACRO_NAME}:
        raise ValueError("completion report must contain exactly one joint macro artifact")
    if (artifact_hashes.get(MACRO_NAME) != report.get("joint_macro_artifact_sha256")):
        raise ValueError("completion report joint macro hash binding is invalid")
    if (not isinstance(schedules_report, dict) or len(schedules_report) != SCHEDULE_COUNT
            or not isinstance(clean_caches_report, dict) or len(clean_caches_report) != CLEAN_CACHE_COUNT
            or not isinstance(draw_caches_report, dict) or len(draw_caches_report) != DRAW_CACHE_COUNT
            or not isinstance(supporting_hashes, dict) or len(supporting_hashes) != SUPPORTING_EVIDENCE_COUNT
            or not isinstance(evidence_hashes, dict) or len(evidence_hashes) != EVIDENCE_FILE_COUNT
            or evidence_hashes != {**artifact_hashes, **supporting_hashes}):
        raise ValueError("completion report supporting evidence chain is invalid")

    config = _read_object(config_path, "format-contrast config")
    try:
        validate_runtime_config(config)
    except ValueError as exc:
        raise ValueError("format-contrast config runtime contract is invalid") from exc
    documents: dict[str, dict[str, Any]] = {}
    for relative, expected_sha256 in evidence_hashes.items():
        artifact = _relative_evidence_path(root, relative)
        if not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
            raise ValueError(f"evidence SHA-256 binding is invalid: {relative}")
        if not artifact.is_file() or sha256_file(artifact) != expected_sha256:
            raise ValueError(f"evidence SHA-256 mismatch: {relative}")
        if relative in artifact_hashes:
            documents[relative] = _read_object(artifact, "contrast artifact")
    for relative in component_hashes:
        dataset, _ = _component_dataset_model(relative)
        if (dataset != component_plan[relative]["dataset"]
                or documents[relative].get("endpoint_type") != component_plan[relative]["endpoint"]):
            raise ValueError(f"component endpoint/dataset plan is invalid: {relative}")
        _validate_component(
            root, documents[relative], relative,
            expected_seed=dataset_bootstrap_seed(config["seed_namespace"], dataset),
        )
    _validate_scheduler_ledger(
        root, scheduler_ledgers["clean"], package_binding, package_document, config,
        "clean", clean_relatives, documents,
    )
    _validate_scheduler_ledger(
        root, scheduler_ledgers["corrupt"], package_binding, package_document, config,
        "corrupt", corrupted_relatives, documents,
    )
    _validate_macro(root, documents[MACRO_NAME], component_hashes, documents, corrupted_relatives, component_plan)

    schedule_bindings: dict[str, dict[str, Any]] = {}
    for relative in component_hashes:
        dataset = Path(relative).name.split("__", 1)[0]
        schedule = documents[relative]["bootstrap_schedule"]
        if schedule.get("seed") != dataset_bootstrap_seed(config["seed_namespace"], dataset):
            raise ValueError(f"component bootstrap schedule seed is invalid: {relative}")
        if dataset in schedule_bindings and schedule_bindings[dataset] != schedule:
            raise ValueError(f"components do not share one bootstrap schedule: {dataset}")
        schedule_bindings[dataset] = schedule
    if schedules_report != schedule_bindings:
        raise ValueError("completion report schedule bindings do not match components")
    if documents[MACRO_NAME].get("bootstrap_schedules") != schedule_bindings:
        raise ValueError("joint macro bootstrap-schedule bindings do not match components")
    expected_clean_caches = {
        document["clean_arm_cache"]["path"]: {
            "sha256": document["clean_arm_cache"]["sha256"],
            "identity_sha256": document["clean_arm_cache"]["identity_sha256"],
        }
        for document in (documents[relative] for relative in component_hashes)
    }
    if clean_caches_report != expected_clean_caches:
        raise ValueError("completion report clean-arm cache bindings do not match components")
    expected_draw_caches = {
        relative: documents[relative]["temporary_draw_cache"]
        for relative in component_hashes
        if "temporary_draw_cache" in documents[relative]
    }
    if draw_caches_report != expected_draw_caches:
        raise ValueError("completion report draw-cache bindings do not match components")
    expected_supporting_hashes = {
        **{binding["path"]: binding["sha256"] for binding in schedule_bindings.values()},
        **{relative: binding["sha256"] for relative, binding in expected_clean_caches.items()},
        **{binding["path"]: binding["sha256"] for binding in expected_draw_caches.values()},
    }
    if supporting_hashes != expected_supporting_hashes:
        raise ValueError("completion report supporting hashes do not match component bindings")

    expected_run_hashes = {
        binding.get("run_record_sha256")
        for relative in component_hashes
        for binding in documents[relative].get("input_hashes", {}).values()
    }
    if None in expected_run_hashes or not expected_run_hashes:
        raise ValueError("component artifact run-record bindings are incomplete")
    local_run_hashes = {
        sha256_file(path)
        for path in (root / "manifests" / "runs").glob("**/*.json")
        if path.is_file()
    }
    missing_run_hashes = expected_run_hashes - local_run_hashes
    if missing_run_hashes:
        raise ValueError(f"synchronized run manifests are missing {len(missing_run_hashes)} bound records")
    return {
        "status": "valid",
        "attempt": ATTEMPT,
        "clean_contrasts": 12,
        "corrupted_contrasts": 144,
        "n_boot": 2000,
        "component_artifacts": COMPONENT_COUNT,
        "artifacts": ARTIFACT_COUNT,
        "supporting_evidence_files": SUPPORTING_EVIDENCE_COUNT,
        "evidence_files": EVIDENCE_FILE_COUNT,
        "completion_report_sha256": report["report_sha256"],
        "completion_report_file_sha256": sha256_file(report_path),
        "config_sha256": report["config_sha256"],
        "execution_package": package_binding,
        "scheduler_ledgers": scheduler_ledgers,
        "bootstrap_schedules": schedule_bindings,
        "artifacts_sha256": artifact_hashes,
        "supporting_evidence_sha256": supporting_hashes,
        "validated_at_utc": datetime.now(timezone.utc).isoformat(),
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--remote-completion-file-sha256")
    args = parser.parse_args()
    root, config_path, output = Path(args.project_root), Path(args.config), Path(args.out)
    document = validate_complete_report(root, config_path)
    if args.remote_completion_file_sha256:
        if document["completion_report_file_sha256"] != args.remote_completion_file_sha256:
            raise SystemExit("FORMAT CONTRAST REFUSED: local completion report differs from remote source")
        document["remote_completion_report_file_sha256"] = args.remote_completion_file_sha256
    document["validation_sha256"] = canonical_hash(document, "validation_sha256")
    if output.exists():
        raise SystemExit(f"FORMAT CONTRAST REFUSED: refusing to overwrite validation report: {output}")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"FORMAT CONTRAST LOCAL VALIDATION status=valid artifacts={ARTIFACT_COUNT} out={output}")


if __name__ == "__main__":
    main()
