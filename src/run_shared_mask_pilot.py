#!/usr/bin/env python3
"""Resume-safe single-GPU driver for the three-block shared-Q/DQ-mask pilot.

The immutable phase order is:

1. bind the immutable V1 scientific-gate failure;
2. freeze FP8-derived source-node and input-edge contracts;
3. quantize and topology-normalize INT8 against the frozen FP8 contract;
4. prove exact INT8/frozen-FP8 compute attachment topology;
5. build TensorRT engines with TF32 disabled, one process at a time;
6. bind exact historical FP8 evidence and infer/evaluate aligned INT8;
7. mark shared/concurrent timing fields scientifically inadmissible; and
8. run the paired AP/Omega analysis.

Completed artifacts are reused only after their content and provenance bindings
are checked.  Interrupted partial outputs are moved to a recoverable quarantine
directory and never admitted as evidence. V2 retains its original serial,
exclusive-GPU behavior. V3 uses bounded CPU/inference concurrency and a fresh
attempt namespace while holding FP8 evidence byte-identical by reuse.
"""
from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ThreadPoolExecutor, wait
from contextlib import contextmanager
from datetime import datetime, timezone
import fcntl
import hashlib
import importlib
import json
import os
from pathlib import Path
import shutil
import subprocess
import sys
import threading
from typing import Any, Iterable

from topic_c.manifest import sha256_file
from topic_c.shared_mask_pilot import (
    ATTEMPT,
    CLEAN_GATE_STATS,
    CLEAN_GATE_MAX_DETECTION_COUNT_RELATIVE_DIFFERENCE,
    CLEAN_GATE_TOLERANCE_AP_POINTS,
    CORRUPTIONS,
    FORMATS,
    MASK_POLICY,
    PARENT_ATTEMPT,
    PARENT_EXECUTION_ARCHIVE_SHA256,
    PARENT_FAILURE_MANIFEST_CANONICAL_SHA256,
    PARENT_FAILURE_MANIFEST_FILE_SHA256,
    SEVERITIES,
    TRTEXEC_SHA256,
    V2_CONFIG_CANONICAL_SHA256,
    V2_CONFIG_FILE_SHA256,
    V2_PREFLIGHT_PACKAGE_CANONICAL_SHA256,
    V2_PREFLIGHT_PACKAGE_FILE_SHA256,
    EvidenceBundle,
    PilotError,
    canonical_hash,
    locate_default_bundle,
    read_complete_json,
    resolve_under,
    same_input_identity,
    shared_bundle,
    shared_condition_id,
    validate_bundle,
    validate_config,
    validate_image_manifest,
    write_complete_json,
)
from topic_c.shared_quantization_mask import (
    compute_attachment_contract,
    normalized_attachment_topology,
    read_mask,
    registry_onnx,
    selected_compute_sites,
)


EXECUTION_FILES = (
    f"configs/{ATTEMPT}.json",
    (
        "docs/shared_mask_pilot_v3_execution.md"
        if ATTEMPT == "shared_mask_pilot_v3"
        else "docs/shared_mask_pilot_execution.md"
    ),
    "src/run_shared_mask_pilot.py",
    "src/analyze_shared_mask_pilot.py",
    "src/supervise_shared_mask_pilot.sh",
    "src/quantize_yolo_onnx.py",
    "src/freeze_shared_quantization_mask.py",
    "src/verify_shared_quantization_mask.py",
    "src/build_yolo_trt_engine.py",
    "src/coco_infer_trt.py",
    "src/cross_family_infer_trt.py",
    "src/coco_eval.py",
    "src/paired_bootstrap.py",
    "src/pilot_registry.py",
    "src/topic_c/manifest.py",
    "src/topic_c/coco_data.py",
    "src/topic_c/cross_family.py",
    "src/topic_c/yolo_decode.py",
    "src/topic_c/shared_quantization_mask.py",
    "src/topic_c/shared_mask_pilot.py",
) + (
    (
        "src/run_shared_mask_pilot_v3.py",
        "src/preserve_shared_mask_v2_preflight.py",
    )
    if ATTEMPT == "shared_mask_pilot_v3"
    else ()
)

_PROGRESS_LOCK = threading.Lock()
_GPU_ADMISSION_LOCK = threading.Lock()
_GPU_RESERVED_MIB = 0


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def emit(event: str, **values: Any) -> None:
    print(json.dumps({"time_utc": utc_now(), "event": event, **values}, sort_keys=True), flush=True)


def progress_path(root: Path) -> Path:
    return root / "outputs" / "reports" / ATTEMPT / "progress.json"


def update_progress(root: Path, stage: str, **values: Any) -> None:
    with _PROGRESS_LOCK:
        path = progress_path(root)
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary = path.with_name(
            f".{path.name}.{os.getpid()}.{threading.get_ident()}.tmp"
        )
        document = {"schema_version": 1, "attempt": ATTEMPT, "updated_at_utc": utc_now(), "stage": stage, **values}
        temporary.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
        os.replace(temporary, path)
    emit("progress", stage=stage, **values)


def run_checked(
    root: Path,
    command: list[str],
    *,
    stage: str,
    label: str,
    gpu: bool = False,
    gpu_policy: dict[str, Any] | None = None,
    gpu_reservation_mib: int = 0,
    environment_overrides: dict[str, str] | None = None,
) -> None:
    emit("command_start", stage=stage, label=label, command=command)
    environment = dict(os.environ)
    environment["PYTHONPATH"] = str(root / "src")
    environment.update(environment_overrides or {})
    with gpu_admission(gpu_policy, gpu_reservation_mib) if gpu else _null_admission():
        process = subprocess.run(command, cwd=root, env=environment)
    if process.returncode != 0:
        raise PilotError(f"{stage}/{label} failed with exit code {process.returncode}")
    emit("command_complete", stage=stage, label=label)


def assert_gpu_idle() -> None:
    """Refuse to start a GPU child while another compute process is present."""
    command = [
        "nvidia-smi",
        "--query-compute-apps=pid,process_name",
        "--format=csv,noheader,nounits",
    ]
    try:
        process = subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE, check=True)
    except (OSError, subprocess.CalledProcessError) as exc:
        raise PilotError("GPU idle gate could not query nvidia-smi") from exc
    rows = [row.strip() for row in process.stdout.splitlines() if row.strip()]
    active, ignored = [], []
    for row in rows:
        fields = [value.strip() for value in row.split(",", 1)]
        process_name = Path(fields[1]).name if len(fields) == 2 else ""
        # Sunshine owns a small persistent display/encode context on this host
        # while reporting 0% compute.  It is not a pilot worker.  No Python,
        # trtexec, or other process name is allow-listed.
        (ignored if process_name == "sunshine" else active).append(row)
    if ignored:
        emit("gpu_idle_ignored_display_context", processes=ignored)
    if active:
        raise PilotError("GPU idle gate found another compute process: " + "; ".join(active))


@contextmanager
def _null_admission():
    yield


def gpu_free_memory_mib() -> int:
    try:
        process = subprocess.run(
            [
                "nvidia-smi",
                "--query-gpu=memory.free",
                "--format=csv,noheader,nounits",
            ],
            text=True,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
            check=True,
        )
        rows = [int(value.strip()) for value in process.stdout.splitlines() if value.strip()]
    except (OSError, ValueError, subprocess.CalledProcessError) as exc:
        raise PilotError("GPU capacity gate could not query free memory") from exc
    if len(rows) != 1:
        raise PilotError("GPU capacity gate requires exactly one visible GPU")
    return rows[0]


def effective_gpu_workers(
    policy: dict[str, Any], *, reservation_mib: int, requested: int
) -> int:
    free_mib = gpu_free_memory_mib()
    safety_mib = int(policy["safety_margin_mib"])
    capacity = max(0, (free_mib - safety_mib) // reservation_mib)
    workers = min(requested, int(policy["maximum_concurrent_pilot_processes"]), capacity)
    if workers < 1:
        raise PilotError(
            f"GPU capacity gate found {free_mib} MiB free; requires at least "
            f"{safety_mib + reservation_mib} MiB"
        )
    emit(
        "gpu_worker_capacity",
        free_mib=free_mib,
        safety_margin_mib=safety_mib,
        reservation_mib=reservation_mib,
        requested=requested,
        admitted=workers,
    )
    return workers


@contextmanager
def gpu_admission(policy: dict[str, Any] | None, reservation_mib: int):
    """Reserve conservative VRAM before launching one independent GPU child."""
    global _GPU_RESERVED_MIB
    if policy is None:
        assert_gpu_idle()
        yield
        return
    if reservation_mib <= 0:
        raise PilotError("shared-GPU launch lacks a positive VRAM reservation")
    with _GPU_ADMISSION_LOCK:
        free_mib = gpu_free_memory_mib()
        safety_mib = int(policy["safety_margin_mib"])
        required = safety_mib + reservation_mib + _GPU_RESERVED_MIB
        if free_mib < required:
            raise PilotError(
                f"GPU capacity gate found {free_mib} MiB free with "
                f"{_GPU_RESERVED_MIB} MiB already reserved; requires {required} MiB"
            )
        _GPU_RESERVED_MIB += reservation_mib
        emit(
            "gpu_lease_acquired",
            free_mib=free_mib,
            reservation_mib=reservation_mib,
            total_reserved_mib=_GPU_RESERVED_MIB,
            unrelated_compute_allowed=True,
        )
    try:
        yield
    finally:
        with _GPU_ADMISSION_LOCK:
            _GPU_RESERVED_MIB -= reservation_mib
            emit(
                "gpu_lease_released",
                reservation_mib=reservation_mib,
                total_reserved_mib=_GPU_RESERVED_MIB,
            )


def run_bounded_jobs(
    jobs: list[Any],
    *,
    workers: int,
    work,
    stage: str,
    on_complete=None,
) -> None:
    """Run bounded independent jobs and stop dispatching after the first failure."""
    if workers < 1:
        raise PilotError(f"{stage} worker count must be positive")
    if workers == 1:
        for ordinal, job in enumerate(jobs, 1):
            work(job)
            if on_complete is not None:
                on_complete(ordinal, len(jobs), job)
        return
    pending_jobs = iter(jobs)
    completed = 0
    first_error: BaseException | None = None
    with ThreadPoolExecutor(max_workers=workers, thread_name_prefix=f"shared-mask-{stage}") as executor:
        active = {}
        for _ in range(min(workers, len(jobs))):
            job = next(pending_jobs)
            active[executor.submit(work, job)] = job
        while active:
            done, _ = wait(tuple(active), return_when=FIRST_COMPLETED)
            successful_slots = 0
            for future in done:
                job = active.pop(future)
                try:
                    future.result()
                except BaseException as exc:  # drain launched siblings before preserving the first failure
                    if first_error is None:
                        first_error = exc
                else:
                    completed += 1
                    successful_slots += 1
                    if on_complete is not None:
                        on_complete(completed, len(jobs), job)
            if first_error is None:
                for _ in range(successful_slots):
                    try:
                        next_job = next(pending_jobs)
                    except StopIteration:
                        break
                    else:
                        active[executor.submit(work, next_job)] = next_job
        if first_error is not None:
            raise first_error


def assert_disk_margin(root: Path, minimum_gib: int = 20) -> None:
    free = shutil.disk_usage(root).free
    if free < minimum_gib * 1024**3:
        raise PilotError(f"free-disk gate requires {minimum_gib} GiB; found {free / 1024**3:.2f} GiB")


def quarantine_partial(root: Path, label: str, paths: Iterable[Path], reason: str) -> None:
    existing = [path.resolve() for path in paths if path.exists()]
    if not existing:
        return
    timestamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    destination = root / "outputs" / "quarantine" / ATTEMPT / f"{timestamp}__{label}"
    destination.mkdir(parents=True, exist_ok=False)
    moved = []
    for ordinal, source in enumerate(existing):
        target = destination / f"{ordinal:02d}__{source.name}"
        shutil.move(str(source), str(target))
        moved.append({"source": str(source), "quarantined": str(target), "sha256": sha256_file(target)})
    report = {"schema_version": 1, "created_at_utc": utc_now(), "label": label, "reason": reason, "files": moved}
    write_complete_json(destination / "quarantine.json", report, self_hash_field="report_sha256")
    emit("partial_quarantined", label=label, reason=reason, destination=str(destination), files=len(moved))


def execution_package_path(root: Path) -> Path:
    return root / "outputs" / "reports" / ATTEMPT / "execution_package.json"


def validate_parent_failure(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Prove that V2 is a fresh, explicitly superseding scientific attempt."""
    binding = config["parent_failure"]
    manifest_path = resolve_under(root, binding["failure_manifest"], "parent V1 failure manifest")
    archive_path = resolve_under(
        root, binding["execution_sources_archive"], "parent V1 execution-source archive"
    )
    manifest = read_complete_json(
        manifest_path, self_hash_field="failure_manifest_sha256"
    )
    archive_record = manifest.get("execution_source_archive", {})
    if not isinstance(archive_record, dict):
        raise PilotError("parent V1 failure manifest has no execution-source archive binding")
    recorded_archive = resolve_under(
        root, archive_record.get("path", ""), "recorded parent V1 execution-source archive"
    )
    if (
        sha256_file(manifest_path) != PARENT_FAILURE_MANIFEST_FILE_SHA256
        or manifest.get("failure_manifest_sha256")
        != PARENT_FAILURE_MANIFEST_CANONICAL_SHA256
        or manifest.get("attempt") != PARENT_ATTEMPT
        or manifest.get("status") != "terminal_scientific_gate_failure"
        or manifest.get("admissible_as_scientific_result") is not False
        or recorded_archive != archive_path
        or archive_record.get("sha256") != PARENT_EXECUTION_ARCHIVE_SHA256
        or archive_record.get("all_member_hashes_match_execution_package") is not True
        or not archive_path.is_file()
        or sha256_file(archive_path) != PARENT_EXECUTION_ARCHIVE_SHA256
    ):
        raise PilotError("immutable parent V1 failure binding is invalid or has changed")
    return {
        "attempt": PARENT_ATTEMPT,
        "status": "terminal_scientific_gate_failure",
        "failure_manifest": str(manifest_path),
        "failure_manifest_file_sha256": PARENT_FAILURE_MANIFEST_FILE_SHA256,
        "failure_manifest_canonical_sha256": PARENT_FAILURE_MANIFEST_CANONICAL_SHA256,
        "execution_sources_archive": str(archive_path),
        "execution_sources_archive_sha256": PARENT_EXECUTION_ARCHIVE_SHA256,
    }


V2_PREFLIGHT_ABSENT_PATHS = (
    "outputs/reports/shared_mask_pilot_v2/progress.json",
    "outputs/reports/shared_mask_pilot_v2/complete.json",
    "manifests/quantization_masks/shared_mask_pilot_v2",
    "outputs/onnx/shared_mask_pilot_v2",
    "manifests/onnx/shared_mask_pilot_v2",
    "outputs/engines/shared_mask_pilot_v2",
    "manifests/engines/shared_mask_pilot_v2",
    "outputs/predictions/shared_mask_pilot_v2",
    "outputs/inputs/shared_mask_pilot_v2",
    "manifests/runs/shared_mask_pilot_v2",
    "outputs/metrics/shared_mask_pilot_v2",
    "outputs/bootstrap/shared_mask_pilot_v2",
    "outputs/analysis/shared_mask_pilot_v2",
)


def _pid_identity_alive(pid_file: Path, required_text: str) -> bool:
    try:
        pid = int(pid_file.read_text(encoding="utf-8").strip())
        command = (Path("/proc") / str(pid) / "cmdline").read_bytes().replace(b"\0", b" ").decode()
    except (OSError, ValueError, UnicodeDecodeError):
        return False
    return required_text in command


def validate_or_write_v2_preflight_abandonment(
    root: Path, config: dict[str, Any]
) -> dict[str, Any]:
    """Immutably prove that V2 stopped after packaging and before science."""
    if ATTEMPT != "shared_mask_pilot_v3":
        return {}
    binding = config["superseded_preflight"]
    package_path = resolve_under(root, binding["execution_package"], "V2 execution package")
    v2_config_path = resolve_under(root, binding["config"], "V2 config")
    archive_path = resolve_under(
        root, binding["execution_sources_archive"], "V2 execution-source archive"
    )
    preservation_path = resolve_under(
        root, binding["preservation_report"], "V2 preservation report"
    )
    package = read_complete_json(package_path, self_hash_field="package_sha256")
    preservation = read_complete_json(preservation_path, self_hash_field="report_sha256")
    v2_config = json.loads(v2_config_path.read_text(encoding="utf-8"))
    if (
        sha256_file(package_path) != V2_PREFLIGHT_PACKAGE_FILE_SHA256
        or package.get("package_sha256") != V2_PREFLIGHT_PACKAGE_CANONICAL_SHA256
        or package.get("attempt") != "shared_mask_pilot_v2"
        or package.get("config_file_sha256") != V2_CONFIG_FILE_SHA256
        or sha256_file(v2_config_path) != V2_CONFIG_FILE_SHA256
        or v2_config.get("config_sha256") != V2_CONFIG_CANONICAL_SHA256
        or not archive_path.is_file()
        or sha256_file(archive_path) != binding["execution_sources_archive_sha256"]
        or sha256_file(preservation_path) != binding["preservation_report_file_sha256"]
        or preservation.get("report_sha256")
        != binding["preservation_report_canonical_sha256"]
        or preservation.get("archive_sha256")
        != binding["execution_sources_archive_sha256"]
        or preservation.get("member_count") != binding["execution_sources_member_count"]
        or preservation.get("members_sha256") != package.get("files_sha256")
        or preservation.get("progress_artifact_absent") is not True
        or preservation.get("scientific_artifacts_absent") is not True
    ):
        raise PilotError("V2 preflight package/config binding changed")
    for name in ("driver", "supervisor"):
        pid_file = root / "outputs" / "logs" / "shared_mask_pilot_v2" / f"{name}.pid"
        if _pid_identity_alive(pid_file, "shared_mask_pilot"):
            raise PilotError(f"V2 {name} is still active; stop it before superseding preflight")
    present = [relative for relative in V2_PREFLIGHT_ABSENT_PATHS if (root / relative).exists()]
    if present:
        raise PilotError(
            "V2 is not preflight-only; scientific/progress artifacts exist: " + ", ".join(present)
        )
    expected = {
        "schema_version": 1,
        "attempt": "shared_mask_pilot_v2",
        "status": "abandoned_after_preflight",
        "admissible_as_scientific_result": False,
        "reason": binding["reason"],
        "successor_attempt": ATTEMPT,
        "execution_package": {
            "path": binding["execution_package"],
            "file_sha256": V2_PREFLIGHT_PACKAGE_FILE_SHA256,
            "canonical_sha256": V2_PREFLIGHT_PACKAGE_CANONICAL_SHA256,
        },
        "config": {
            "path": binding["config"],
            "file_sha256": V2_CONFIG_FILE_SHA256,
            "canonical_sha256": V2_CONFIG_CANONICAL_SHA256,
        },
        "preservation": {
            "report": binding["preservation_report"],
            "report_file_sha256": binding["preservation_report_file_sha256"],
            "report_canonical_sha256": binding["preservation_report_canonical_sha256"],
            "execution_sources_archive": binding["execution_sources_archive"],
            "execution_sources_archive_sha256": binding["execution_sources_archive_sha256"],
            "execution_sources_member_count": binding["execution_sources_member_count"],
        },
        "preflight_only": True,
        "progress_artifact_absent": True,
        "scientific_artifacts_absent": True,
        "checked_absent_paths": list(V2_PREFLIGHT_ABSENT_PATHS),
    }
    expected["abandonment_manifest_sha256"] = canonical_hash(
        expected, "abandonment_manifest_sha256"
    )
    encoded = (json.dumps(expected, indent=2, sort_keys=True) + "\n").encode("utf-8")
    expected_file_sha256 = hashlib.sha256(encoded).hexdigest()
    if (
        binding["abandonment_manifest_canonical_sha256"]
        != expected["abandonment_manifest_sha256"]
        or binding["abandonment_manifest_file_sha256"] != expected_file_sha256
    ):
        raise PilotError("V2 abandonment-manifest precommit hash mismatch")
    output = resolve_under(root, binding["abandonment_manifest"], "V2 abandonment manifest")
    if output.exists() or output.with_suffix(output.suffix + ".complete").exists():
        observed = read_complete_json(output, self_hash_field="abandonment_manifest_sha256")
        if observed != expected:
            raise PilotError("V2 abandonment manifest differs from the precommit")
        return observed
    return write_complete_json(
        output, expected, self_hash_field="abandonment_manifest_sha256"
    )


def ensure_execution_package(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    parent_failure: dict[str, Any],
    superseded_preflight: dict[str, Any] | None = None,
) -> dict[str, Any]:
    path = execution_package_path(root)
    files = {}
    for relative in EXECUTION_FILES:
        source = resolve_under(root, relative, "execution-package source")
        if not source.is_file():
            raise PilotError(f"execution-package source is absent: {source}")
        files[relative] = sha256_file(source)
    expected = {
        "schema_version": 1,
        "attempt": ATTEMPT,
        "project_root": str(root),
        "config": str(config_path),
        "config_file_sha256": sha256_file(config_path),
        "config_sha256": config["config_sha256"],
        "mask_policy": MASK_POLICY,
        "supersedes_failed_attempt": parent_failure,
        "files_sha256": files,
        "clean_gate": (
            {
                "type": "exact_existing_artifact_identity",
                "bundles": 39,
                "fp8_onnx_rebuilt": False,
                "fp8_engine_rebuilt": False,
                "fp8_inference_rerun": False,
            }
            if ATTEMPT == "shared_mask_pilot_v3"
            else {
                "statistics": list(CLEAN_GATE_STATS),
                "maximum_absolute_difference_ap_points": CLEAN_GATE_TOLERANCE_AP_POINTS,
                "maximum_detection_count_relative_difference": CLEAN_GATE_MAX_DETECTION_COUNT_RELATIVE_DIFFERENCE,
                "interpretation": "FP8 replay-equivalence guardrail; not a universal practical-equivalence margin",
            }
        ),
        "trtexec_sha256": TRTEXEC_SHA256,
    }
    if superseded_preflight is not None:
        expected["supersedes_preflight_attempt"] = superseded_preflight
        expected["execution_policy"] = config["execution"]
    if path.exists() or path.with_suffix(path.suffix + ".complete").exists():
        observed = read_complete_json(path, self_hash_field="package_sha256")
        comparable = {key: observed.get(key) for key in expected}
        if comparable != expected:
            raise PilotError("execution package differs from current reviewed source/config bytes")
        return observed
    return write_complete_json(path, expected, self_hash_field="package_sha256")


def mask_path(root: Path, block: dict[str, Any]) -> Path:
    return root / "manifests" / "quantization_masks" / ATTEMPT / f"{block['id']}.json"


def validate_mask_for_block(root: Path, block: dict[str, Any]) -> dict[str, Any]:
    path = mask_path(root, block)
    mask = read_mask(path)
    if (
        mask.get("attempt") != ATTEMPT
        or mask.get("block_id") != block["id"]
        or mask.get("dataset") != block["dataset"]
        or mask.get("model") != block["model"]
        or mask.get("imgsz") != int(block["imgsz"])
    ):
        raise PilotError(f"frozen mask scientific identity mismatch: {path}")
    registry_fields = (
        ("source", "source_onnx_registry"),
        ("baseline_int8", "baseline_int8_registry"),
        ("baseline_fp8", "baseline_fp8_registry"),
    )
    for mask_key, config_key in registry_fields:
        expected = resolve_under(root, block[config_key], f"{block['id']} {config_key}")
        if (
            Path(mask.get(mask_key, {}).get("registry", "")).resolve() != expected
            or mask.get(mask_key, {}).get("registry_file_sha256") != sha256_file(expected)
        ):
            raise PilotError(f"frozen mask registry binding mismatch: {path}/{mask_key}")
    calibration = resolve_under(root, block["calibration_list"], f"{block['id']} calibration")
    calibration_marker = calibration.with_suffix(calibration.suffix + ".complete")
    if (
        Path(mask.get("calibration", {}).get("list", "")).resolve() != calibration
        or not calibration_marker.is_file()
        or calibration_marker.read_text(encoding="utf-8").strip()
        != mask.get("calibration", {}).get("calibration_sha256")
    ):
        raise PilotError(f"frozen mask calibration binding mismatch: {path}")
    return mask


def ensure_masks(root: Path, config: dict[str, Any]) -> None:
    freeze_module = importlib.import_module("freeze_shared_quantization_mask")
    for ordinal, block in enumerate(config["blocks"], 1):
        path = mask_path(root, block)
        marker = path.with_suffix(path.suffix + ".complete")
        if path.exists() and marker.exists():
            validate_mask_for_block(root, block)
            emit("artifact_reused", stage="freeze_masks", block=block["id"], path=str(path))
        else:
            quarantine_partial(root, f"mask__{block['id']}", (path, marker), "incomplete mask pair")
            update_progress(root, "freeze_masks", completed=ordinal - 1, total=3, current=block["id"])
            try:
                freeze_module.freeze_block(root, ATTEMPT, block, path)
            except ValueError as exc:
                raise PilotError(f"mask freeze failed for {block['id']}: {exc}") from exc
            validate_mask_for_block(root, block)
        update_progress(root, "freeze_masks", completed=ordinal, total=3)


def onnx_paths(root: Path, block: dict[str, Any], precision: str) -> tuple[Path, Path]:
    stem = f"{block['id']}__{precision}"
    return (
        root / "outputs" / "onnx" / ATTEMPT / f"{stem}.onnx",
        root / "manifests" / "onnx" / ATTEMPT / f"{stem}.json",
    )


def validate_quantized(root: Path, block: dict[str, Any], precision: str) -> dict[str, Any]:
    output, registry_path = onnx_paths(root, block, precision)
    registry, resolved_output, _ = registry_onnx(registry_path)
    mask = validate_mask_for_block(root, block)
    if resolved_output != output.resolve() or registry.get("precision") != precision:
        raise PilotError(f"quantized ONNX identity mismatch: {registry_path}")
    baseline_attachment_sha = mask["baseline_fp8_normalized_topology"][
        "compute_attachment_sha256"
    ]
    if (
        registry.get("dataset") != block["dataset"]
        or registry.get("model") != block["model"]
        or registry.get("node_mask_sha256") != mask["mask_sha256"]
        or registry.get("node_mask_file_sha256") != sha256_file(mask_path(root, block))
        or registry.get("nodes_to_quantize_count") != len(mask["nodes"])
        or registry.get("no_quantize_inputs_count")
        != len(mask["edge_policy"]["no_quantize_inputs"])
        or registry.get("edge_policy_sha256")
        != mask["edge_policy"]["edge_policy_sha256"]
        or registry.get("compute_input_policy_sha256")
        != mask["compute_input_policy"]["policy_sha256"]
        or registry.get("baseline_fp8_onnx_sha256")
        != mask["baseline_fp8"]["onnx_sha256"]
        or registry.get("post_enforcement_compute_attachment_sha256")
        != baseline_attachment_sha
    ):
        raise PilotError(f"quantized ONNX provenance mismatch: {registry_path}")
    if precision == "fp8":
        if (
            registry.get("fp8_baseline_byte_replay") is not True
            or registry.get("shared_mask_role") != "frozen_baseline_fp8_control"
            or registry.get("output_onnx_sha256") != mask["baseline_fp8"]["onnx_sha256"]
            or registry.get("bypassed_compute_inputs_count") != 0
        ):
            raise PilotError(f"FP8 arm is not the exact frozen baseline byte control: {registry_path}")
    elif (
        registry.get("fp8_baseline_byte_replay") is not False
        or registry.get("shared_mask_role")
        != "int8_aligned_to_frozen_fp8_compute_inputs"
        or not isinstance(registry.get("bypassed_compute_inputs"), list)
        or registry.get("bypassed_compute_inputs_count")
        != len(registry["bypassed_compute_inputs"])
    ):
        raise PilotError(f"INT8 arm lacks auditable frozen-FP8 contract enforcement: {registry_path}")
    return registry


def ensure_quantized(root: Path, config: dict[str, Any]) -> None:
    formats = ("int8-entropy",) if ATTEMPT == "shared_mask_pilot_v3" else FORMATS
    jobs = [(block, precision) for block in config["blocks"] for precision in formats]
    thread_cap = str(config["execution"].get("cpu_threads_per_quantizer", 1))

    def work(job: tuple[dict[str, Any], str]) -> None:
        block, precision = job
        output, registry = onnx_paths(root, block, precision)
        marker = registry.with_suffix(registry.suffix + ".complete")
        if output.exists() and registry.exists() and marker.exists():
            validate_quantized(root, block, precision)
            emit("artifact_reused", stage="quantize", block=block["id"], precision=precision)
            return
        quarantine_partial(
            root,
            f"quantize__{block['id']}__{precision}",
            (output, registry, marker),
            "incomplete quantization artifact set",
        )
        stage = "copy_frozen_fp8" if precision == "fp8" else "quantize_and_align_int8"
        command = [
            sys.executable,
            str(root / "src" / "quantize_yolo_onnx.py"),
            "--onnx-registry", str(resolve_under(root, block["source_onnx_registry"], "source registry")),
            "--mode", precision,
            "--imgsz", str(block["imgsz"]),
            "--calibration-list", str(resolve_under(root, block["calibration_list"], "calibration list")),
            "--node-mask", str(mask_path(root, block)),
            "--out", str(output),
            "--registry-out", str(registry),
        ]
        run_checked(
            root,
            command,
            stage=stage,
            label=f"{block['id']}/{precision}",
            # Calibration is explicitly CPU-EP. Limit each independent
            # process so two quantizers do not oversubscribe the 14-core host.
            gpu=False if ATTEMPT == "shared_mask_pilot_v3" else precision != "fp8",
            environment_overrides={
                "OMP_NUM_THREADS": thread_cap,
                "MKL_NUM_THREADS": thread_cap,
                "OPENBLAS_NUM_THREADS": thread_cap,
                "NUMEXPR_NUM_THREADS": thread_cap,
                "ORT_NUM_THREADS": thread_cap,
            },
        )
        validate_quantized(root, block, precision)

    def completed(done: int, total: int, job: tuple[dict[str, Any], str]) -> None:
        block, precision = job
        update_progress(
            root,
            "quantize",
            completed=done,
            total=total,
            last=f"{block['id']}/{precision}",
            workers=config["execution"]["quantization_workers"],
        )

    run_bounded_jobs(
        jobs,
        workers=int(config["execution"]["quantization_workers"]),
        work=work,
        stage="quantize",
        on_complete=completed,
    )


def topology_report_path(root: Path, block: dict[str, Any]) -> Path:
    return root / "outputs" / "reports" / ATTEMPT / "topology" / f"{block['id']}.json"


def validate_topology_report(root: Path, block: dict[str, Any]) -> dict[str, Any]:
    report = read_complete_json(topology_report_path(root, block), self_hash_field="report_sha256")
    mask = validate_mask_for_block(root, block)
    replay_fp8 = (
        None
        if ATTEMPT == "shared_mask_pilot_v3"
        else validate_quantized(root, block, "fp8")
    )
    # The aligned policy is defined by the recorded baseline-FP8 placement.
    # V2 copies those FP8 bytes directly, so byte equality proves that arm did
    # not absorb recalibration or graph-serialization variation. Fail before
    # any TensorRT build if that frozen control or the INT8 alignment changed.
    if replay_fp8 is not None and replay_fp8.get("output_onnx_sha256") != mask["baseline_fp8"]["onnx_sha256"]:
        raise PilotError(
            f"FP8 replay ONNX is not byte-identical to frozen baseline FP8: {block['id']}"
        )
    if (
        report.get("schema_version")
        != (4 if ATTEMPT == "shared_mask_pilot_v3" else 3)
        or report.get("status") != "pass"
        or report.get("block_id") != block["id"]
        or report.get("mask", {}).get("file_sha256") != sha256_file(mask_path(root, block))
        or report.get("mask", {}).get("mask_sha256") != mask["mask_sha256"]
        or report.get("baseline_compute_topology_replayed_exactly") is not True
        or report.get("baseline_fp8_bytes_replayed_exactly") is not True
        or report.get("compute_attachment_sha256")
        != report.get("baseline_fp8_compute_attachment_sha256")
        or report.get("compute_input_policy_sha256")
        != mask["compute_input_policy"]["policy_sha256"]
    ):
        raise PilotError(f"topology report did not pass: {topology_report_path(root, block)}")
    registry_pairs = [(onnx_paths(root, block, "int8-entropy")[1], "int8")]
    if ATTEMPT == "shared_mask_pilot_v3":
        registry_pairs.append(
            (resolve_under(root, block["baseline_fp8_registry"], "baseline FP8 registry"), "fp8")
        )
    else:
        registry_pairs.append((onnx_paths(root, block, "fp8")[1], "fp8"))
    for registry, key in registry_pairs:
        if report.get(key, {}).get("registry_file_sha256") != sha256_file(registry):
            raise PilotError(f"topology report registry binding changed: {report}")
    if ATTEMPT == "shared_mask_pilot_v3" and report.get("fp8_evidence_reused_without_materialization") is not True:
        raise PilotError(f"V3 topology report did not reuse frozen FP8 evidence: {report}")
    return report


def write_v3_topology_report(root: Path, block: dict[str, Any], output: Path) -> None:
    mask = validate_mask_for_block(root, block)
    int8_registry_path = onnx_paths(root, block, "int8-entropy")[1]
    int8_registry, int8_onnx, int8_registry_sha = registry_onnx(int8_registry_path)
    baseline_registry_path = resolve_under(
        root, block["baseline_fp8_registry"], "baseline FP8 registry"
    )
    baseline_registry, fp8_onnx, baseline_registry_sha = registry_onnx(
        baseline_registry_path
    )
    validate_quantized(root, block, "int8-entropy")
    if (
        baseline_registry.get("precision") != "fp8"
        or baseline_registry.get("dataset") != block["dataset"]
        or baseline_registry.get("model") != block["model"]
        or sha256_file(fp8_onnx) != mask["baseline_fp8"]["onnx_sha256"]
    ):
        raise PilotError(f"baseline FP8 registry identity changed: {baseline_registry_path}")
    source_onnx = Path(mask["source"]["onnx"]).resolve()
    required_nodes = {item["name"] for item in mask["nodes"]}
    int8_topology = normalized_attachment_topology(
        source_onnx, int8_onnx, required_node_names=required_nodes
    )
    fp8_topology = normalized_attachment_topology(
        source_onnx, fp8_onnx, required_node_names=required_nodes
    )
    baseline_topology = mask["baseline_fp8_normalized_topology"]
    if (
        compute_attachment_contract(int8_topology)
        != compute_attachment_contract(fp8_topology)
        or compute_attachment_contract(fp8_topology)
        != compute_attachment_contract(baseline_topology)
    ):
        raise PilotError(f"V3 INT8 does not match frozen FP8 attachment contract: {block['id']}")
    expected_nodes = [item["name"] for item in mask["nodes"]]
    if [item["name"] for item in selected_compute_sites(source_onnx, int8_topology)] != expected_nodes:
        raise PilotError(f"V3 INT8 selected compute sites changed: {block['id']}")
    report = {
        "schema_version": 4,
        "created_at_utc": utc_now(),
        "status": "pass",
        "scope": "aligned INT8 versus immutable baseline-FP8 ONNX topology; not TensorRT kernel precision",
        "block_id": block["id"],
        "dataset": block["dataset"],
        "model": block["model"],
        "mask": {
            "path": str(mask_path(root, block)),
            "file_sha256": sha256_file(mask_path(root, block)),
            "mask_sha256": mask["mask_sha256"],
            "nodes": len(expected_nodes),
        },
        "compute_attachment_sha256": int8_topology["compute_attachment_sha256"],
        "int8_full_topology_sha256": int8_topology["topology_sha256"],
        "fp8_full_topology_sha256": fp8_topology["topology_sha256"],
        "full_graph_diagnostics_equal": int8_topology == fp8_topology,
        "baseline_fp8_compute_attachment_sha256": baseline_topology["compute_attachment_sha256"],
        "baseline_compute_topology_replayed_exactly": True,
        "baseline_fp8_bytes_replayed_exactly": True,
        "fp8_evidence_reused_without_materialization": True,
        "compute_input_policy_sha256": mask["compute_input_policy"]["policy_sha256"],
        "no_quantize_inputs": len(mask["edge_policy"]["no_quantize_inputs"]),
        "raw_unquantized_inputs": len(mask["edge_policy"]["raw_inputs"]),
        "int8": {
            "registry": str(int8_registry_path.resolve()),
            "registry_file_sha256": int8_registry_sha,
            "onnx": str(int8_onnx),
            "onnx_sha256": sha256_file(int8_onnx),
        },
        "fp8": {
            "registry": str(baseline_registry_path),
            "registry_file_sha256": baseline_registry_sha,
            "onnx": str(fp8_onnx),
            "onnx_sha256": sha256_file(fp8_onnx),
        },
    }
    write_complete_json(output, report, self_hash_field="report_sha256")


def ensure_topology(root: Path, config: dict[str, Any]) -> None:
    for ordinal, block in enumerate(config["blocks"], 1):
        output = topology_report_path(root, block)
        marker = output.with_suffix(output.suffix + ".complete")
        if output.exists() and marker.exists():
            validate_topology_report(root, block)
            emit("artifact_reused", stage="verify_qdq_topology", block=block["id"])
        else:
            quarantine_partial(root, f"topology__{block['id']}", (output, marker), "incomplete topology report")
            if ATTEMPT == "shared_mask_pilot_v3":
                write_v3_topology_report(root, block, output)
            else:
                _, int8_registry = onnx_paths(root, block, "int8-entropy")
                _, fp8_registry = onnx_paths(root, block, "fp8")
                command = [
                    sys.executable,
                    str(root / "src" / "verify_shared_quantization_mask.py"),
                    "--mask", str(mask_path(root, block)),
                    "--int8-registry", str(int8_registry),
                    "--fp8-registry", str(fp8_registry),
                    "--out", str(output),
                ]
                run_checked(root, command, stage="verify_qdq_topology", label=block["id"])
            validate_topology_report(root, block)
        update_progress(root, "verify_qdq_topology", completed=ordinal, total=3)


def engine_paths(root: Path, block: dict[str, Any], precision: str) -> tuple[Path, Path, Path]:
    stem = f"{block['id']}__{precision}"
    return (
        root / "outputs" / "engines" / ATTEMPT / f"{stem}.engine",
        root / "outputs" / "logs" / ATTEMPT / "build" / f"{stem}.log",
        root / "manifests" / "engines" / ATTEMPT / f"{stem}.json",
    )


def validate_engine(root: Path, block: dict[str, Any], precision: str) -> dict[str, Any]:
    engine, build_log, registry_path = engine_paths(root, block, precision)
    registry = read_complete_json(registry_path)
    _, onnx_registry = onnx_paths(root, block, precision)
    if (
        registry.get("dataset") != block["dataset"]
        or registry.get("model") != block["model"]
        or registry.get("precision") != precision
        or Path(registry.get("engine", "")).resolve() != engine.resolve()
        or registry.get("engine_sha256") != sha256_file(engine)
        or registry.get("build_log_sha256") != sha256_file(build_log)
        or registry.get("source_onnx_registry_sha256") != sha256_file(onnx_registry)
        or registry.get("tf32_enabled") is not False
        or "--noTF32" not in registry.get("command", [])
        or registry.get("trtexec_sha256") != TRTEXEC_SHA256
    ):
        raise PilotError(f"TensorRT engine provenance mismatch: {registry_path}")
    return registry


def ensure_engines(root: Path, config: dict[str, Any], trt_root: Path) -> None:
    trtexec = trt_root / "bin" / "trtexec"
    if not trtexec.is_file() or sha256_file(trtexec) != TRTEXEC_SHA256:
        raise PilotError("TensorRT trtexec identity differs from the frozen pilot runtime")
    formats = ("int8-entropy",) if ATTEMPT == "shared_mask_pilot_v3" else FORMATS
    total, completed = 3 * len(formats), 0
    gpu_policy = config["execution"].get("gpu_admission")
    reservation_mib = int(gpu_policy["engine_build_reservation_mib"]) if gpu_policy else 0
    for block in config["blocks"]:
        for precision in formats:
            engine, build_log, registry = engine_paths(root, block, precision)
            marker = registry.with_suffix(registry.suffix + ".complete")
            if all(path.exists() for path in (engine, build_log, registry, marker)):
                validate_engine(root, block, precision)
                emit("artifact_reused", stage="build_engines", block=block["id"], precision=precision)
            else:
                quarantine_partial(
                    root,
                    f"engine__{block['id']}__{precision}",
                    (engine, build_log, registry, marker),
                    "incomplete TensorRT artifact set",
                )
                _, onnx_registry = onnx_paths(root, block, precision)
                command = [
                    sys.executable,
                    str(root / "src" / "build_yolo_trt_engine.py"),
                    "--onnx-registry", str(onnx_registry),
                    "--precision", precision,
                    "--trt-root", str(trt_root),
                    "--engine", str(engine),
                    "--build-log", str(build_log),
                    "--registry-out", str(registry),
                ]
                update_progress(root, "build_engines", completed=completed, total=total, current=f"{block['id']}/{precision}")
                run_checked(
                    root,
                    command,
                    stage="build_engines",
                    label=f"{block['id']}/{precision}",
                    gpu=True,
                    gpu_policy=gpu_policy,
                    gpu_reservation_mib=reservation_mib,
                )
                validate_engine(root, block, precision)
            completed += 1
            update_progress(root, "build_engines", completed=completed, total=total)


def _validate_inference_trio(
    root: Path,
    bundle: EvidenceBundle,
    *,
    block: dict[str, Any],
    precision: str,
    corruption: str,
    severity: int,
    manifest_sha256: str,
    engine_sha256: str,
) -> None:
    try:
        predictions = json.loads(bundle.prediction.read_text(encoding="utf-8"))
        input_record = json.loads(bundle.input_record.read_text(encoding="utf-8"))
        run = json.loads(bundle.run_record.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"invalid inference trio: {bundle.prediction}") from exc
    if not isinstance(predictions, list) or run.get("prediction_sha256") != sha256_file(bundle.prediction):
        raise PilotError(f"prediction/run binding mismatch: {bundle.prediction}")
    if run.get("engine_sha256") != engine_sha256 or run.get("annotation_sha256") != sha256_file(resolve_under(root, block["annotations"], "annotations")):
        raise PilotError(f"engine/annotation run binding mismatch: {bundle.run_record}")
    if (
        input_record.get("input_manifest_sha256") != manifest_sha256
        or run.get("input_manifest_sha256") != manifest_sha256
        or run.get("input_image_ids_sha256") != input_record.get("image_ids_sha256")
    ):
        raise PilotError(f"inference input binding mismatch: {bundle.run_record}")
    expected_corruption = "clean" if corruption == "clean" and block["family"] == "cross_family" else (
        "codec_control" if corruption == "clean" else corruption
    )
    if (
        run.get("dataset") != block["dataset"]
        or run.get("split") != block["split"]
        or run.get("model") != block["model"]
        or run.get("precision") != precision
        or run.get("corruption") != expected_corruption
        or run.get("severity") != severity
    ):
        raise PilotError(f"inference scientific identity mismatch: {bundle.run_record}")


def _inference_command(
    root: Path,
    block: dict[str, Any],
    precision: str,
    corruption: str,
    severity: int,
    manifest: Path,
    bundle: EvidenceBundle,
) -> list[str]:
    annotations = resolve_under(root, block["annotations"], "annotations")
    cache = resolve_under(
        root,
        block["matched_clean_cache_root"] if corruption == "clean" else block["corruption_cache_root"],
        "image cache root",
    )
    engine, _, engine_registry = engine_paths(root, block, precision)
    condition = shared_condition_id(block, precision, corruption, severity)
    if block["family"] == "yolo":
        command = [
            sys.executable,
            str(root / "src" / "coco_infer_trt.py"),
            "--engine", str(engine),
            "--annotations", str(annotations),
            "--image-manifest", str(manifest),
            "--manifest-cache-root", str(cache),
            "--out", str(bundle.prediction),
            "--input-record", str(bundle.input_record),
            "--run-record", str(bundle.run_record),
            "--condition-id", condition,
            "--dataset", block["dataset"],
            "--split", block["split"],
            "--model", block["model"],
            "--precision", precision,
            "--calibrator", "entropy",
            "--calibration-list", str(resolve_under(root, block["calibration_list"], "calibration list")),
            "--calibration-method", "entropy",
            "--calibration-provenance", "verified",
            "--corruption", "codec_control" if corruption == "clean" else corruption,
            "--severity", str(severity),
            "--class-map", str(resolve_under(root, block["class_map"], "class map")),
            "--imgsz", str(block["imgsz"]),
        ]
    else:
        command = [
            sys.executable,
            str(root / "src" / "cross_family_infer_trt.py"),
            "--engine-registry", str(engine_registry),
            "--annotations", str(annotations),
            "--image-manifest", str(manifest),
            "--manifest-cache-root", str(cache),
            "--out", str(bundle.prediction),
            "--input-record", str(bundle.input_record),
            "--run-record", str(bundle.run_record),
            "--condition-id", condition,
            "--dataset", block["dataset"],
            "--split", block["split"],
            "--corruption", "clean" if corruption == "clean" else corruption,
            "--severity", str(severity),
        ]
    return command


def ensure_condition(
    root: Path,
    block: dict[str, Any],
    precision: str,
    corruption: str,
    severity: int,
    *,
    gpu_policy: dict[str, Any] | None = None,
) -> dict[str, Any]:
    condition = shared_condition_id(block, precision, corruption, severity)
    bundle = shared_bundle(root, condition)
    manifest, manifest_sha = validate_image_manifest(root, block, corruption=corruption, severity=severity)
    engine_record = validate_engine(root, block, precision)
    engine_sha = engine_record["engine_sha256"]
    exists = [path.exists() for path in bundle.paths]
    if all(exists):
        try:
            return validate_bundle(
                bundle,
                root=root,
                block=block,
                precision=precision,
                corruption=corruption,
                severity=severity,
                expected_manifest_sha256=manifest_sha,
                expected_engine_sha256=engine_sha,
            )
        except PilotError as bundle_error:
            # Evaluation writes the metric last.  If interruption leaves a
            # truncated/invalid metric beside an intact inference trio, retain
            # the expensive predictions and recompute only that metric.
            try:
                _validate_inference_trio(
                    root,
                    bundle,
                    block=block,
                    precision=precision,
                    corruption=corruption,
                    severity=severity,
                    manifest_sha256=manifest_sha,
                    engine_sha256=engine_sha,
                )
            except PilotError as trio_error:
                quarantine_partial(
                    root,
                    f"inference__{condition}",
                    bundle.paths,
                    f"invalid complete bundle: {bundle_error}; invalid trio: {trio_error}",
                )
            else:
                quarantine_partial(
                    root,
                    f"metric__{condition}",
                    (bundle.metric,),
                    f"invalid interrupted metric: {bundle_error}",
                )
                exists = [True, True, True, False]
    inference_ready = False
    if exists[:3] == [True, True, True] and not exists[3]:
        try:
            _validate_inference_trio(
                root,
                bundle,
                block=block,
                precision=precision,
                corruption=corruption,
                severity=severity,
                manifest_sha256=manifest_sha,
                engine_sha256=engine_sha,
            )
            inference_ready = True
            emit("artifact_reused", stage="inference", condition=condition)
        except PilotError as exc:
            quarantine_partial(
                root,
                f"inference__{condition}",
                bundle.paths,
                f"invalid interrupted inference trio: {exc}",
            )
    elif any(exists):
        quarantine_partial(root, f"inference__{condition}", bundle.paths, "incomplete inference/evaluation bundle")
    if not inference_ready:
        run_checked(
            root,
            _inference_command(root, block, precision, corruption, severity, manifest, bundle),
            stage="inference",
            label=condition,
            gpu=True,
            gpu_policy=gpu_policy,
            gpu_reservation_mib=(
                int(gpu_policy["inference_reservation_mib"]) if gpu_policy else 0
            ),
        )
        _validate_inference_trio(
            root,
            bundle,
            block=block,
            precision=precision,
            corruption=corruption,
            severity=severity,
            manifest_sha256=manifest_sha,
            engine_sha256=engine_sha,
        )
    if not bundle.metric.exists():
        run_checked(
            root,
            [
                sys.executable,
                str(root / "src" / "coco_eval.py"),
                "--annotations", str(resolve_under(root, block["annotations"], "annotations")),
                "--predictions", str(bundle.prediction),
                "--input-record", str(bundle.input_record),
                "--run-record", str(bundle.run_record),
                "--out", str(bundle.metric),
            ],
            stage="evaluation",
            label=condition,
        )
    return validate_bundle(
        bundle,
        root=root,
        block=block,
        precision=precision,
        corruption=corruption,
        severity=severity,
        expected_manifest_sha256=manifest_sha,
        expected_engine_sha256=engine_sha,
    )


def ensure_clean(root: Path, config: dict[str, Any]) -> None:
    formats = ("int8-entropy",) if ATTEMPT == "shared_mask_pilot_v3" else FORMATS
    jobs = [(block, precision) for block in config["blocks"] for precision in formats]
    gpu_policy = config["execution"].get("gpu_admission")
    workers = int(config["execution"]["inference_workers"])
    if gpu_policy:
        workers = effective_gpu_workers(
            gpu_policy,
            reservation_mib=int(gpu_policy["inference_reservation_mib"]),
            requested=workers,
        )

    def work(job: tuple[dict[str, Any], str]) -> None:
        block, precision = job
        ensure_condition(
            root, block, precision, "clean", 0, gpu_policy=gpu_policy
        )

    def completed(done: int, total: int, job: tuple[dict[str, Any], str]) -> None:
        update_progress(
            root,
            "matched_clean",
            completed=done,
            total=total,
            workers=workers,
            last=f"{job[0]['id']}/{job[1]}",
        )

    run_bounded_jobs(
        jobs,
        workers=workers,
        work=work,
        stage="matched_clean",
        on_complete=completed,
    )


def clean_gate_path(root: Path) -> Path:
    name = (
        "fp8_default_identity_gate.json"
        if ATTEMPT == "shared_mask_pilot_v3"
        else "clean_fp8_replay_gate.json"
    )
    return root / "outputs" / "reports" / ATTEMPT / name


def _validate_bound_files(bindings: list[dict[str, str]]) -> None:
    for binding in bindings:
        path = Path(binding["path"])
        if not path.is_file() or sha256_file(path) != binding["sha256"]:
            raise PilotError(f"clean gate source evidence changed: {path}")


def validate_default_fp8_engine_binding(
    root: Path, block: dict[str, Any]
) -> dict[str, Any]:
    """Bind one historical FP8 engine to its frozen baseline-FP8 ONNX bytes."""
    onnx_registry_path = resolve_under(
        root, block["baseline_fp8_registry"], "baseline FP8 ONNX registry"
    )
    _, onnx_path, onnx_registry_sha = registry_onnx(onnx_registry_path)
    engine_registry_path = resolve_under(
        root,
        block["baseline_fp8_engine_registry"],
        "baseline FP8 engine registry",
    )
    engine_registry = read_complete_json(engine_registry_path)
    try:
        engine_path = resolve_under(
            root, engine_registry["engine"], "baseline FP8 engine"
        )
    except (KeyError, TypeError) as exc:
        raise PilotError(
            f"baseline FP8 engine registry lacks an engine path: {engine_registry_path}"
        ) from exc
    if not engine_path.is_file():
        raise PilotError(f"baseline FP8 engine is absent: {engine_path}")
    engine_sha = sha256_file(engine_path)
    onnx_sha = sha256_file(onnx_path)
    if (
        engine_registry.get("dataset") != block["dataset"]
        or engine_registry.get("model") != block["model"]
        or engine_registry.get("precision") != "fp8"
        or engine_registry.get("source_onnx_registry_sha256")
        != onnx_registry_sha
        or Path(str(engine_registry.get("source_onnx", ""))).resolve()
        != onnx_path.resolve()
        or engine_registry.get("source_onnx_sha256") != onnx_sha
        or engine_registry.get("engine_sha256") != engine_sha
        or (
            engine_registry.get("engine_bytes") is not None
            and engine_registry.get("engine_bytes") != engine_path.stat().st_size
        )
    ):
        raise PilotError(
            f"baseline FP8 engine/ONNX provenance mismatch: {engine_registry_path}"
        )
    return {
        "onnx_registry_path": onnx_registry_path,
        "onnx_registry_sha256": onnx_registry_sha,
        "onnx_path": onnx_path,
        "onnx_sha256": onnx_sha,
        "engine_registry_path": engine_registry_path,
        "engine_registry_sha256": sha256_file(engine_registry_path),
        "engine_path": engine_path,
        "engine_sha256": engine_sha,
    }


def validate_default_fp8_run_engine_binding(
    block: dict[str, Any], run: dict[str, Any], binding: dict[str, Any]
) -> None:
    """Validate the family-specific run-record pointer to the bound engine."""
    if run.get("engine_sha256") != binding["engine_sha256"]:
        raise PilotError("default FP8 run uses a different engine SHA-256")
    if block["family"] == "yolo":
        if Path(run.get("engine_path", "")).resolve() != binding[
            "engine_path"
        ].resolve():
            raise PilotError("default YOLO FP8 run uses a different engine path")
    elif (
        Path(run.get("engine_registry", "")).resolve()
        != binding["engine_registry_path"].resolve()
        or run.get("engine_registry_sha256")
        != binding["engine_registry_sha256"]
    ):
        raise PilotError(
            "default cross-family FP8 run uses a different engine registry"
        )


def ensure_fp8_identity_gate(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Bind all 39 frozen default-FP8 bundles; never rebuild or rerun FP8."""
    if ATTEMPT != "shared_mask_pilot_v3":
        raise PilotError("exact default-FP8 identity gate is V3-only")
    path = clean_gate_path(root)
    if path.exists() or path.with_suffix(path.suffix + ".complete").exists():
        report = read_complete_json(path, self_hash_field="report_sha256")
        _validate_bound_files(report.get("source_artifacts", []))
        if (
            report.get("schema_version") != 2
            or report.get("status") != "pass"
            or report.get("config_sha256") != config["config_sha256"]
            or report.get("fp8_inference_rerun") is not False
            or report.get("bundles") != 39
            or len(report.get("engine_bindings", [])) != 3
        ):
            raise PilotError(f"pre-existing FP8 identity gate failed: {path}")
        emit("artifact_reused", stage="bind_default_fp8_artifacts", path=str(path))
        return report
    rows: list[dict[str, Any]] = []
    engine_bindings: list[dict[str, Any]] = []
    sources: dict[str, str] = {}
    for block in config["blocks"]:
        mask = validate_mask_for_block(root, block)
        baseline_registry = resolve_under(
            root, block["baseline_fp8_registry"], "baseline FP8 registry"
        )
        _, baseline_onnx, baseline_registry_sha = registry_onnx(baseline_registry)
        if sha256_file(baseline_onnx) != mask["baseline_fp8"]["onnx_sha256"]:
            raise PilotError(f"baseline FP8 ONNX changed: {block['id']}")
        engine_binding = validate_default_fp8_engine_binding(root, block)
        if engine_binding["onnx_registry_sha256"] != baseline_registry_sha:
            raise PilotError(
                f"default FP8 engine is not bound to the frozen ONNX registry: {block['id']}"
            )
        sources[str(baseline_registry)] = baseline_registry_sha
        sources[str(baseline_onnx)] = sha256_file(baseline_onnx)
        sources[str(engine_binding["engine_registry_path"])] = engine_binding[
            "engine_registry_sha256"
        ]
        sources[str(engine_binding["engine_path"])] = engine_binding["engine_sha256"]
        engine_bindings.append(
            {
                "block_id": block["id"],
                "onnx_registry": str(engine_binding["onnx_registry_path"]),
                "onnx_registry_sha256": engine_binding["onnx_registry_sha256"],
                "onnx": str(engine_binding["onnx_path"]),
                "onnx_sha256": engine_binding["onnx_sha256"],
                "engine_registry": str(engine_binding["engine_registry_path"]),
                "engine_registry_sha256": engine_binding[
                    "engine_registry_sha256"
                ],
                "engine": str(engine_binding["engine_path"]),
                "engine_sha256": engine_binding["engine_sha256"],
            }
        )
        clean_ids_sha = None
        for corruption, severity in [("clean", 0)] + [
            (name, level) for name in CORRUPTIONS for level in SEVERITIES
        ]:
            _, manifest_sha = validate_image_manifest(
                root, block, corruption=corruption, severity=severity
            )
            bundle = locate_default_bundle(
                root,
                block,
                precision="fp8",
                corruption=corruption,
                severity=severity,
            )
            validated = validate_bundle(
                bundle,
                root=root,
                block=block,
                precision="fp8",
                corruption=corruption,
                severity=severity,
                expected_manifest_sha256=manifest_sha,
                expected_engine_sha256=engine_binding["engine_sha256"],
            )
            run_document = json.loads(bundle.run_record.read_text(encoding="utf-8"))
            validate_default_fp8_run_engine_binding(
                block, run_document, engine_binding
            )
            if corruption == "clean":
                clean_ids_sha = validated["input_image_ids_sha256"]
            elif validated["input_image_ids_sha256"] != clean_ids_sha:
                raise PilotError(
                    f"default FP8 clean/corrupt image IDs differ: {block['id']}/{corruption}-s{severity}"
                )
            for artifact in bundle.paths:
                sources[str(artifact)] = sha256_file(artifact)
            rows.append(
                {
                    "block_id": block["id"],
                    "corruption": corruption,
                    "severity": severity,
                    "engine_sha256": run_document["engine_sha256"],
                    "prediction_sha256": validated["prediction_sha256"],
                    "metric_sha256": validated["metric_sha256"],
                    "input_manifest_sha256": validated["input_manifest_sha256"],
                    "input_image_ids_sha256": validated["input_image_ids_sha256"],
                }
            )
    if len(rows) != 39:
        raise PilotError(f"default FP8 identity gate expected 39 bundles; found {len(rows)}")
    report = {
        "schema_version": 2,
        "created_at_utc": utc_now(),
        "attempt": ATTEMPT,
        "status": "pass",
        "config_sha256": config["config_sha256"],
        "definition": "shared-policy FP8 is the exact frozen default FP8 evidence",
        "fp8_onnx_rebuilt": False,
        "fp8_engine_rebuilt": False,
        "fp8_inference_rerun": False,
        "bundles": len(rows),
        "engine_bindings": engine_bindings,
        "rows": rows,
        "source_artifacts": [
            {"path": name, "sha256": value}
            for name, value in sorted(sources.items())
        ],
    }
    report = write_complete_json(path, report, self_hash_field="report_sha256")
    update_progress(
        root, "bind_default_fp8_artifacts", completed=39, total=39, status="pass"
    )
    return report


def ensure_clean_gate(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = clean_gate_path(root)
    if path.exists() or path.with_suffix(path.suffix + ".complete").exists():
        report = read_complete_json(path, self_hash_field="report_sha256")
        _validate_bound_files(report.get("source_artifacts", []))
        if (
            report.get("status") != "pass"
            or report.get("config_sha256") != config["config_sha256"]
            or report.get("statistics") != list(CLEAN_GATE_STATS)
        or report.get("maximum_absolute_difference_ap_points")
            != CLEAN_GATE_TOLERANCE_AP_POINTS
            or report.get("maximum_detection_count_relative_difference")
            != CLEAN_GATE_MAX_DETECTION_COUNT_RELATIVE_DIFFERENCE
        ):
            raise PilotError(f"pre-existing clean FP8 replay gate failed: {path}")
        emit("artifact_reused", stage="clean_gate", path=str(path))
        return report
    rows, source_artifacts, passed = [], [], True
    for block in config["blocks"]:
        manifest, manifest_sha = validate_image_manifest(root, block, corruption="clean", severity=0)
        shared = validate_bundle(
            shared_bundle(root, shared_condition_id(block, "fp8", "clean", 0)),
            root=root,
            block=block,
            precision="fp8",
            corruption="clean",
            severity=0,
            expected_manifest_sha256=manifest_sha,
            expected_engine_sha256=validate_engine(root, block, "fp8")["engine_sha256"],
        )
        default_bundle = locate_default_bundle(root, block, precision="fp8", corruption="clean", severity=0)
        default = validate_bundle(
            default_bundle,
            root=root,
            block=block,
            precision="fp8",
            corruption="clean",
            severity=0,
            expected_manifest_sha256=manifest_sha,
        )
        same_input_identity((shared, default), f"clean FP8 replay {block['id']}")
        differences = {
            name: (float(shared["stats"][name]) - float(default["stats"][name])) * 100.0
            for name in CLEAN_GATE_STATS
        }
        detection_count_relative_difference = abs(
            shared["n_detections"] - default["n_detections"]
        ) / max(default["n_detections"], 1)
        row_pass = (
            all(abs(value) <= CLEAN_GATE_TOLERANCE_AP_POINTS + 1e-12 for value in differences.values())
            and detection_count_relative_difference
            <= CLEAN_GATE_MAX_DETECTION_COUNT_RELATIVE_DIFFERENCE + 1e-12
        )
        mask = validate_mask_for_block(root, block)
        shared_fp8_registry = validate_quantized(root, block, "fp8")
        passed = passed and row_pass
        rows.append(
            {
                "block_id": block["id"],
                "dataset": block["dataset"],
                "model": block["model"],
                "baseline": {name: float(default["stats"][name]) * 100.0 for name in CLEAN_GATE_STATS},
                "replay": {name: float(shared["stats"][name]) * 100.0 for name in CLEAN_GATE_STATS},
                "replay_minus_baseline_ap_points": differences,
                "baseline_detections": default["n_detections"],
                "replay_detections": shared["n_detections"],
                "detection_count_relative_difference": detection_count_relative_difference,
                "prediction_bytes_identical": shared["prediction_sha256"] == default["prediction_sha256"],
                "onnx_bytes_identical": (
                    shared_fp8_registry["output_onnx_sha256"]
                    == mask["baseline_fp8"]["onnx_sha256"]
                ),
                "pass": row_pass,
            }
        )
        for record in (shared, default):
            for key, value in record["paths"].items():
                source_artifacts.append({"role": key, "path": value, "sha256": sha256_file(Path(value))})
        source_artifacts.append({"role": "matched_clean_manifest", "path": str(manifest), "sha256": sha256_file(manifest)})
    report = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "attempt": ATTEMPT,
        "status": "pass" if passed else "fail",
        "config_sha256": config["config_sha256"],
        "statistics": list(CLEAN_GATE_STATS),
        "maximum_absolute_difference_ap_points": CLEAN_GATE_TOLERANCE_AP_POINTS,
        "maximum_detection_count_relative_difference": CLEAN_GATE_MAX_DETECTION_COUNT_RELATIVE_DIFFERENCE,
        "rationale": (
            "Frozen before replay: 0.10 AP point is 1/14 of the recorded 1.40-point primary clean gap "
            "and is intentionally tighter than the historical 1-point gross FP16 parity gate. "
            "The additional 0.1% detection-count margin checks prediction-level stability. "
            "These are treatment-replay guardrails, not universal equivalence margins."
        ),
        "rows": rows,
        "source_artifacts": source_artifacts,
    }
    report = write_complete_json(path, report, self_hash_field="report_sha256")
    if not passed:
        raise PilotError(f"clean FP8 replay gate failed; corrupted inference is blocked: {path}")
    update_progress(root, "clean_gate", completed=3, total=3, status="pass")
    return report


def ensure_corrupted(root: Path, config: dict[str, Any]) -> None:
    formats = ("int8-entropy",) if ATTEMPT == "shared_mask_pilot_v3" else FORMATS
    jobs = [
        (block, precision, corruption, severity)
        for block in config["blocks"]
        for precision in formats
        for corruption in CORRUPTIONS
        for severity in SEVERITIES
    ]
    gpu_policy = config["execution"].get("gpu_admission")
    workers = int(config["execution"]["inference_workers"])
    if gpu_policy:
        workers = effective_gpu_workers(
            gpu_policy,
            reservation_mib=int(gpu_policy["inference_reservation_mib"]),
            requested=workers,
        )

    def work(job: tuple[dict[str, Any], str, str, int]) -> None:
        block, precision, corruption, severity = job
        ensure_condition(
            root,
            block,
            precision,
            corruption,
            severity,
            gpu_policy=gpu_policy,
        )

    def completed(done: int, total: int, job: tuple[dict[str, Any], str, str, int]) -> None:
        block, precision, corruption, severity = job
        update_progress(
            root,
            "corrupted",
            completed=done,
            total=total,
            workers=workers,
            last=f"{block['id']}/{precision}/{corruption}-s{severity}",
        )

    run_bounded_jobs(
        jobs,
        workers=workers,
        work=work,
        stage="corrupted",
        on_complete=completed,
    )


def timing_validity_path(root: Path) -> Path:
    return root / "outputs" / "reports" / ATTEMPT / "timing_evidence_validity.json"


def ensure_timing_inadmissibility(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    """Bind every new run record while explicitly excluding its timing field."""
    if ATTEMPT != "shared_mask_pilot_v3":
        return {}
    path = timing_validity_path(root)
    if path.exists() or path.with_suffix(path.suffix + ".complete").exists():
        report = read_complete_json(path, self_hash_field="report_sha256")
        _validate_bound_files(report.get("run_records", []))
        if (
            report.get("status") != "timing_inadmissible"
            or report.get("run_record_count") != 39
            or report.get("config_sha256") != config["config_sha256"]
        ):
            raise PilotError(f"timing-validity report changed: {path}")
        return report
    bindings: list[dict[str, str]] = []
    for block in config["blocks"]:
        for corruption, severity in [("clean", 0)] + [
            (name, level) for name in CORRUPTIONS for level in SEVERITIES
        ]:
            bundle = shared_bundle(
                root,
                shared_condition_id(
                    block, "int8-entropy", corruption, severity
                ),
            )
            try:
                run = json.loads(bundle.run_record.read_text(encoding="utf-8"))
            except (OSError, json.JSONDecodeError) as exc:
                raise PilotError(f"invalid V3 run record: {bundle.run_record}") from exc
            if not isinstance(run.get("runtime_seconds"), (int, float)):
                raise PilotError(f"V3 run record lacks recorded runtime field: {bundle.run_record}")
            bindings.append(
                {"path": str(bundle.run_record), "sha256": sha256_file(bundle.run_record)}
            )
    if len(bindings) != 39:
        raise PilotError(f"timing report expected 39 aligned-INT8 runs; found {len(bindings)}")
    timing_policy = config["execution"]["timing_evidence"]
    report = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "attempt": ATTEMPT,
        "status": "timing_inadmissible",
        "config_sha256": config["config_sha256"],
        "accuracy_outputs_eligible_if_completion_passes": True,
        "runtime_seconds_eligible_for_scientific_comparison": False,
        "reason": timing_policy["reason"],
        "required_remeasurement": timing_policy["required_remeasurement"],
        "shared_gpu_allowed": True,
        "maximum_concurrent_accuracy_workers": config["execution"]["inference_workers"],
        "run_record_count": len(bindings),
        "run_records": bindings,
    }
    return write_complete_json(path, report, self_hash_field="report_sha256")


def ensure_analysis(root: Path, config_path: Path, config: dict[str, Any], workers: int) -> None:
    command = [
        sys.executable,
        str(root / "src" / "analyze_shared_mask_pilot.py"),
        "--project-root", str(root),
        "--config", str(config_path),
        "--workers", str(workers),
    ]
    run_checked(root, command, stage="paired_bootstrap", label="AP/Omega")
    summary = root / "outputs" / "analysis" / ATTEMPT / "bootstrap_summary.json"
    report = read_complete_json(summary, self_hash_field="report_sha256")
    if report.get("status") != "complete" or report.get("cells") != 36:
        raise PilotError("shared-mask bootstrap analysis did not complete its exact grid")
    update_progress(root, "paired_bootstrap", completed=36, total=36)


def completion_path(root: Path) -> Path:
    return root / "outputs" / "reports" / ATTEMPT / "complete.json"


def iter_evidence_paths(root: Path, config: dict[str, Any]) -> Iterable[Path]:
    yield resolve_under(
        root, config["parent_failure"]["failure_manifest"], "parent V1 failure manifest"
    )
    yield resolve_under(
        root,
        config["parent_failure"]["execution_sources_archive"],
        "parent V1 execution-source archive",
    )
    if ATTEMPT == "shared_mask_pilot_v3":
        superseded = config["superseded_preflight"]
        for key in (
            "execution_package",
            "config",
            "execution_sources_archive",
            "preservation_report",
            "abandonment_manifest",
        ):
            yield resolve_under(root, superseded[key], f"V2 {key}")
    yield execution_package_path(root)
    yield clean_gate_path(root)
    if ATTEMPT == "shared_mask_pilot_v3":
        yield timing_validity_path(root)
    for block in config["blocks"]:
        yield mask_path(root, block)
        yield topology_report_path(root, block)
        formats = ("int8-entropy",) if ATTEMPT == "shared_mask_pilot_v3" else FORMATS
        for precision in formats:
            yield from onnx_paths(root, block, precision)
            yield from engine_paths(root, block, precision)
            for corruption, severity in [("clean", 0)] + [
                (name, level) for name in CORRUPTIONS for level in SEVERITIES
            ]:
                yield from shared_bundle(root, shared_condition_id(block, precision, corruption, severity)).paths
    yield root / "outputs" / "analysis" / ATTEMPT / "point_summary.json"
    yield root / "outputs" / "analysis" / ATTEMPT / "bootstrap_summary.json"
    for block in config["blocks"]:
        for corruption in CORRUPTIONS:
            for severity in SEVERITIES:
                stem = f"{block['id']}__{corruption}-s{severity}"
                yield root / "outputs" / "bootstrap" / ATTEMPT / f"{stem}.npz"
                yield root / "outputs" / "bootstrap" / ATTEMPT / f"{stem}.json"


def ensure_completion(root: Path, config: dict[str, Any]) -> dict[str, Any]:
    path = completion_path(root)
    identity: dict[str, Any] = {}
    if ATTEMPT == "shared_mask_pilot_v3":
        identity = read_complete_json(clean_gate_path(root), self_hash_field="report_sha256")
        timing = read_complete_json(timing_validity_path(root), self_hash_field="report_sha256")
        _validate_bound_files(identity.get("source_artifacts", []))
        _validate_bound_files(timing.get("run_records", []))
    artifacts: dict[str, str] = {}

    def add_artifact(source: Path, expected_sha: str | None = None) -> None:
        source = resolve_under(root, source, "completion evidence")
        if not source.is_file():
            raise PilotError(f"completion artifact is absent: {source}")
        observed_sha = sha256_file(source)
        if expected_sha is not None and observed_sha != expected_sha:
            raise PilotError(f"completion artifact hash changed: {source}")
        relative = str(source.relative_to(root))
        previous = artifacts.get(relative)
        if previous is not None and previous != observed_sha:
            raise PilotError(f"conflicting completion hashes for: {source}")
        artifacts[relative] = observed_sha

    for source in iter_evidence_paths(root, config):
        add_artifact(source)
    if ATTEMPT == "shared_mask_pilot_v3":
        for binding in identity.get("source_artifacts", []):
            if not isinstance(binding, dict):
                raise PilotError("FP8 identity gate contains an invalid source binding")
            add_artifact(Path(binding.get("path", "")), binding.get("sha256"))
    expected = {
        "schema_version": 1,
        "attempt": ATTEMPT,
        "status": "complete",
        "config_sha256": config["config_sha256"],
        "mask_policy": MASK_POLICY,
        "supersedes_failed_attempt": {
            "attempt": PARENT_ATTEMPT,
            "failure_manifest_canonical_sha256": PARENT_FAILURE_MANIFEST_CANONICAL_SHA256,
            "execution_sources_archive_sha256": PARENT_EXECUTION_ARCHIVE_SHA256,
        },
        "shared_clean_arms": 3 if ATTEMPT == "shared_mask_pilot_v3" else 6,
        "shared_corrupted_arms": 36 if ATTEMPT == "shared_mask_pilot_v3" else 72,
        "reused_default_fp8_arms": 39 if ATTEMPT == "shared_mask_pilot_v3" else 0,
        "direct_cells": 36,
        "artifacts_sha256": artifacts,
    }
    if ATTEMPT == "shared_mask_pilot_v3":
        expected["runtime_evidence_status"] = "inadmissible"
        expected["fp8_rebuilt_or_rerun"] = False
        expected["supersedes_preflight_attempt"] = {
            "attempt": "shared_mask_pilot_v2",
            "abandonment_manifest_canonical_sha256": config["superseded_preflight"][
                "abandonment_manifest_canonical_sha256"
            ],
            "execution_sources_archive_sha256": config["superseded_preflight"][
                "execution_sources_archive_sha256"
            ],
        }
        expected["execution_policy"] = config["execution"]
    if path.exists() or path.with_suffix(path.suffix + ".complete").exists():
        report = read_complete_json(path, self_hash_field="report_sha256")
        if {key: report.get(key) for key in expected} != expected:
            raise PilotError("completion report no longer matches the immutable artifact set")
        return report
    expected["completed_at_utc"] = utc_now()
    return write_complete_json(path, expected, self_hash_field="report_sha256")


def acquire_driver_lock(root: Path):
    path = root / "outputs" / "logs" / ATTEMPT / "driver.lock"
    path.parent.mkdir(parents=True, exist_ok=True)
    handle = path.open("a+")
    try:
        fcntl.flock(handle.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        handle.close()
        raise PilotError(f"another {ATTEMPT} driver holds {path}") from exc
    handle.seek(0)
    handle.truncate()
    handle.write(f"pid={os.getpid()} started_at_utc={utc_now()}\n")
    handle.flush()
    return handle


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=f"configs/{ATTEMPT}.json")
    parser.add_argument("--trt-root", default="/home/thuan/traffic/third_party/TensorRT-11.1.0.106")
    parser.add_argument("--bootstrap-workers", type=int, default=3)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = resolve_under(root, args.config, "pilot config")
    trt_root = Path(args.trt_root).resolve()
    if not 1 <= args.bootstrap_workers <= 4:
        raise SystemExit("SHARED-MASK PILOT REFUSED: --bootstrap-workers must be 1..4")
    lock = None
    try:
        expected_config_path = (root / "configs" / f"{ATTEMPT}.json").resolve()
        if config_path != expected_config_path:
            raise PilotError(f"pilot requires the reviewed configs/{ATTEMPT}.json")
        config = validate_config(root, config_path)
        assert_disk_margin(root)
        parent_failure = validate_parent_failure(root, config)
        lock = acquire_driver_lock(root)
        superseded_preflight = validate_or_write_v2_preflight_abandonment(root, config)
        ensure_execution_package(
            root,
            config_path,
            config,
            parent_failure,
            superseded_preflight or None,
        )
        # Validate all immutable input manifests before any ModelOpt/TensorRT work.
        for block in config["blocks"]:
            validate_image_manifest(root, block, corruption="clean", severity=0)
            for corruption in CORRUPTIONS:
                for severity in SEVERITIES:
                    validate_image_manifest(root, block, corruption=corruption, severity=severity)
        if args.preflight_only:
            emit(
                "preflight_complete",
                blocks=3,
                new_shared_arms=39 if ATTEMPT == "shared_mask_pilot_v3" else 78,
                reused_default_fp8_arms=39 if ATTEMPT == "shared_mask_pilot_v3" else 0,
                direct_cells=36,
            )
            return
        update_progress(root, "starting", pid=os.getpid())
        ensure_masks(root, config)
        ensure_quantized(root, config)
        ensure_topology(root, config)
        ensure_engines(root, config, trt_root)
        if ATTEMPT == "shared_mask_pilot_v3":
            ensure_fp8_identity_gate(root, config)
        ensure_clean(root, config)
        if ATTEMPT != "shared_mask_pilot_v3":
            ensure_clean_gate(root, config)
        ensure_corrupted(root, config)
        ensure_timing_inadmissibility(root, config)
        ensure_analysis(root, config_path, config, args.bootstrap_workers)
        report = ensure_completion(root, config)
        update_progress(root, "complete", report=str(completion_path(root)), report_sha256=report["report_sha256"])
        emit("shared_mask_pilot_complete", report=str(completion_path(root)), report_sha256=report["report_sha256"])
    except (PilotError, ValueError) as exc:
        if root.is_dir():
            update_progress(root, "blocked", error=str(exc))
        raise SystemExit(f"SHARED-MASK PILOT REFUSED: {exc}") from exc
    finally:
        if lock is not None:
            lock.close()


if __name__ == "__main__":
    main()
