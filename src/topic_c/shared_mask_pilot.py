"""Fail-closed paths and evidence checks for the shared-Q/DQ-mask pilot.

This module deliberately contains no TensorRT or ModelOpt imports.  The driver
uses it to decide whether an artifact is complete before starting the next GPU
process; the analysis uses the same checks before reading any AP value.
"""
from __future__ import annotations

from dataclasses import dataclass
import hashlib
import json
import os
from pathlib import Path
from typing import Any, Iterable

from topic_c.manifest import read_manifest, sha256_file


DEFAULT_ATTEMPT = "shared_mask_pilot_v2"
ATTEMPT = os.environ.get("SHARED_MASK_PILOT_ATTEMPT", DEFAULT_ATTEMPT)
CONFIG_SHA256_BY_ATTEMPT = {
    "shared_mask_pilot_v2": "b6c9e114a2fffbb2e220b0798c264d3aa834f105accdf0e042ad932450ac6427",
    "shared_mask_pilot_v3": "904fb5aef600dfbd15302b16a2e3189743f0b142a9d3020f258bd3dae4e64efe",
}
if ATTEMPT not in CONFIG_SHA256_BY_ATTEMPT:
    raise RuntimeError(f"unsupported SHARED_MASK_PILOT_ATTEMPT: {ATTEMPT}")
CONFIG_SHA256 = CONFIG_SHA256_BY_ATTEMPT[ATTEMPT]
MASK_POLICY = "frozen_baseline_fp8_compute_attachment_contract_v3"
PARENT_ATTEMPT = "shared_mask_pilot_v1"
PARENT_FAILURE_MANIFEST_CANONICAL_SHA256 = (
    "de9f8f60bd7ebbe2f3a4995127ee10fa0ec6e2492b42d6a2baf3bad4f891a2d2"
)
PARENT_FAILURE_MANIFEST_FILE_SHA256 = (
    "759380731f11f426e034f0987ef3571698b5c0fec1c50f1703635e86374b53c2"
)
PARENT_EXECUTION_ARCHIVE_SHA256 = (
    "badf87386e66885b2059a2f85b0e5beebb565f60a2c0516419b98684c34dce6f"
)
FORMATS = ("int8-entropy", "fp8")
CORRUPTIONS = ("fog", "gaussian_noise", "jpeg", "motion_blur")
SEVERITIES = (1, 3, 5)
CLEAN_GATE_STATS = ("AP", "AP50", "AP75")
# Frozen before observing replay results.  A 0.10-point bound is one fourteenth
# of the recorded 1.40-point primary clean gap and much tighter than the old
# 1-point FP16 gross-parity gate.  It is a replay-equivalence guardrail, not a
# claim that 0.10 AP is universally negligible.
CLEAN_GATE_TOLERANCE_AP_POINTS = 0.10
CLEAN_GATE_MAX_DETECTION_COUNT_RELATIVE_DIFFERENCE = 0.001
TRTEXEC_SHA256 = "68b061d276601b7c6d8aafa4d8d75319f32382c85673360407a6d7ee6411aa4d"
V2_PREFLIGHT_PACKAGE_FILE_SHA256 = (
    "cceeb271bc4755a5ce15008a3e2ae08187ab4ecad506eed908a8d32d86398b62"
)
V2_PREFLIGHT_PACKAGE_CANONICAL_SHA256 = (
    "db8d1312b1d80fea9610c1ce1a1511b3c103c366859cea7f4690bdce45ff556f"
)
V2_CONFIG_FILE_SHA256 = (
    "7fc84968392615292932a59bb246bfc256f1ef74465047f1106c3902c080ade9"
)
V2_CONFIG_CANONICAL_SHA256 = (
    "b6c9e114a2fffbb2e220b0798c264d3aa834f105accdf0e042ad932450ac6427"
)


class PilotError(RuntimeError):
    """An evidence or execution contract was not satisfied."""


def canonical_hash(document: dict[str, Any], field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def resolve_under(root: Path, value: str | Path, label: str) -> Path:
    root = root.resolve()
    path = Path(value)
    result = path.resolve() if path.is_absolute() else (root / path).resolve()
    if result != root and root not in result.parents:
        raise PilotError(f"{label} escapes project root: {value}")
    return result


def write_complete_json(
    path: Path,
    document: dict[str, Any],
    *,
    self_hash_field: str | None = None,
) -> dict[str, Any]:
    """Atomically publish an immutable JSON document and completion marker."""
    path = path.resolve()
    marker = path.with_suffix(path.suffix + ".complete")
    if path.exists() or marker.exists():
        raise PilotError(f"refusing to overwrite immutable artifact: {path}")
    value = dict(document)
    if self_hash_field is not None:
        value[self_hash_field] = canonical_hash(value, self_hash_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp")
    temporary.write_text(json.dumps(value, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    os.replace(temporary, path)
    marker.write_text(
        (value[self_hash_field] if self_hash_field is not None else sha256_file(path)) + "\n",
        encoding="utf-8",
    )
    return value


def read_complete_json(
    path: Path,
    *,
    self_hash_field: str | None = None,
) -> dict[str, Any]:
    path = path.resolve()
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.is_file() or not marker.is_file():
        raise PilotError(f"incomplete JSON artifact: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"invalid JSON artifact: {path}") from exc
    expected = sha256_file(path)
    if self_hash_field is not None:
        declared = document.get(self_hash_field)
        if not isinstance(declared, str) or declared != canonical_hash(document, self_hash_field):
            raise PilotError(f"invalid {self_hash_field}: {path}")
        expected = declared
    if marker.read_text(encoding="utf-8").strip() != expected:
        raise PilotError(f"completion marker mismatch: {path}")
    return document


def validate_config(
    root: Path,
    path: Path,
    *,
    expected_attempt: str | None = None,
) -> dict[str, Any]:
    root, path = root.resolve(), resolve_under(root, path, "pilot config")
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"invalid pilot config: {path}") from exc
    expected_attempt = ATTEMPT if expected_attempt is None else expected_attempt
    expected_config_sha256 = CONFIG_SHA256_BY_ATTEMPT.get(expected_attempt)
    if expected_config_sha256 is None:
        raise PilotError(f"unsupported shared-mask attempt: {expected_attempt}")
    if config.get("schema_version") != 1 or config.get("attempt") != expected_attempt:
        raise PilotError("unsupported shared-mask pilot config")
    if (
        config.get("config_sha256") != expected_config_sha256
        or config.get("config_sha256") != canonical_hash(config, "config_sha256")
    ):
        raise PilotError("pilot config canonical SHA-256 mismatch")
    if config.get("mask_policy") != MASK_POLICY:
        raise PilotError("pilot mask policy changed")
    parent = config.get("parent_failure")
    expected_parent = {
        "attempt": PARENT_ATTEMPT,
        "status": "terminal_scientific_gate_failure",
        "failure_manifest": "outputs/quarantine/shared_mask_pilot_v1_failed/failure_manifest.json",
        "failure_manifest_file_sha256": PARENT_FAILURE_MANIFEST_FILE_SHA256,
        "failure_manifest_canonical_sha256": PARENT_FAILURE_MANIFEST_CANONICAL_SHA256,
        "execution_sources_archive": (
            "outputs/quarantine/shared_mask_pilot_v1_failed/execution_sources.tar.gz"
        ),
        "execution_sources_archive_sha256": PARENT_EXECUTION_ARCHIVE_SHA256,
    }
    if parent != expected_parent:
        raise PilotError("pilot parent-failure binding changed")
    resolve_under(root, parent["failure_manifest"], "parent V1 failure manifest")
    resolve_under(root, parent["execution_sources_archive"], "parent V1 source archive")
    if expected_attempt == "shared_mask_pilot_v3":
        superseded = config.get("superseded_preflight")
        expected_superseded = {
            "attempt": "shared_mask_pilot_v2",
            "status": "abandoned_after_preflight",
            "reason": "execution_policy_changed_to_bounded_shared_gpu_parallelism_by_user_request",
            "execution_package": "outputs/reports/shared_mask_pilot_v2/execution_package.json",
            "execution_package_file_sha256": V2_PREFLIGHT_PACKAGE_FILE_SHA256,
            "execution_package_canonical_sha256": V2_PREFLIGHT_PACKAGE_CANONICAL_SHA256,
            "config": "configs/shared_mask_pilot_v2.json",
            "config_file_sha256": V2_CONFIG_FILE_SHA256,
            "config_canonical_sha256": V2_CONFIG_CANONICAL_SHA256,
            "execution_sources_archive": "outputs/quarantine/shared_mask_pilot_v2_preflight_abandoned/execution_sources.tar.gz",
            "execution_sources_archive_sha256": "ab877ace598b54ef21f1eec751a26b005a80c91d4ad403001a34904e88339db1",
            "execution_sources_member_count": 20,
            "preservation_report": "outputs/quarantine/shared_mask_pilot_v2_preflight_abandoned/preservation_report.json",
            "preservation_report_file_sha256": "cddd46e784afcf131faaf324e04b3703881ca7f93709a5288e5bc94235d49a98",
            "preservation_report_canonical_sha256": "3ae722ae6dcc85385814589c9b757be0750f3372ab6ebd8b027d851ddbbf9a2e",
            "abandonment_manifest": "outputs/quarantine/shared_mask_pilot_v2_preflight_abandoned/abandonment_manifest.json",
            "abandonment_manifest_file_sha256": "35b98c775481a3dd4a569871194f7b22cefd5c5f0e719407e84e30d3bb0b8ee8",
            "abandonment_manifest_canonical_sha256": "91e5e79f4cfa0bdf7bc20fefc985b32dc53651617479565a621a90e5a84ba156",
        }
        if superseded != expected_superseded:
            raise PilotError("V3 superseded-preflight binding changed")
        for field in (
            "execution_package",
            "config",
            "execution_sources_archive",
            "preservation_report",
            "abandonment_manifest",
        ):
            resolve_under(root, superseded[field], f"V2 preflight {field}")
    if tuple(config.get("formats", ())) != FORMATS:
        raise PilotError("pilot formats/order must be exactly INT8-entropy then FP8")
    if tuple(config.get("corruptions", ())) != CORRUPTIONS:
        raise PilotError("pilot corruption grid/order changed")
    if tuple(config.get("severities", ())) != SEVERITIES:
        raise PilotError("pilot severity grid/order changed")
    execution = config.get("execution", {})
    expected_workers = {
        "quantization_workers": 1 if expected_attempt == "shared_mask_pilot_v2" else 2,
        "engine_build_workers": 1,
        "inference_workers": 1 if expected_attempt == "shared_mask_pilot_v2" else 3,
        "evaluation_workers": 1 if expected_attempt == "shared_mask_pilot_v2" else 3,
    }
    for field, expected in expected_workers.items():
        # V2 predates the explicit engine/evaluation worker fields; its serial
        # defaults remain part of the frozen historical contract.
        observed = execution.get(field, 1 if expected_attempt == "shared_mask_pilot_v2" else None)
        if observed != expected:
            raise PilotError(f"pilot {field} must be exactly {expected}")
    if expected_attempt == "shared_mask_pilot_v3":
        if execution.get("cpu_threads_per_quantizer") != 4:
            raise PilotError("V3 CPU thread cap per quantizer changed")
        gpu = execution.get("gpu_admission", {})
        expected_gpu = {
            "allow_unrelated_compute_processes": True,
            "maximum_concurrent_pilot_processes": 3,
            "safety_margin_mib": 4096,
            "quantization_reservation_mib": 8192,
            "engine_build_reservation_mib": 8192,
            "inference_reservation_mib": 4096,
            "poll_seconds": 30,
        }
        if gpu != expected_gpu:
            raise PilotError("V3 GPU-admission policy changed")
        timing = execution.get("timing_evidence", {})
        if timing != {
            "status": "inadmissible",
            "reason": "shared_gpu_and_concurrent_accuracy_inference",
            "required_remeasurement": "isolated_gpu_separate_attempt",
        }:
            raise PilotError("V3 timing-evidence policy changed")
    expected_phases = (
        (
            "bind_parent_failure",
            "freeze_masks",
            "materialize_int8_and_copy_frozen_fp8",
            "verify_qdq_topology",
            "build_engines",
            "matched_clean",
            "corrupted",
            "paired_bootstrap",
        )
        if expected_attempt == "shared_mask_pilot_v2"
        else (
            "bind_parent_failure",
            "bind_v2_preflight_abandonment",
            "freeze_masks",
            "materialize_aligned_int8",
            "verify_int8_against_frozen_fp8_topology",
            "build_int8_engines",
            "bind_default_fp8_artifacts",
            "matched_clean_int8",
            "corrupted_int8",
            "paired_bootstrap",
        )
    )
    if tuple(execution.get("phase_order", ())) != expected_phases:
        raise PilotError("pilot phase order changed")
    if config.get("bootstrap_replicates") != 2000:
        raise PilotError("pilot bootstrap replicate count changed")
    blocks = config.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 3:
        raise PilotError("pilot requires exactly three declared blocks")
    expected_ids = (
        "voc_yolo11m",
        "voc_rtdetr_l",
        "kitti_retinanet_r50_fpn_v2",
    )
    if tuple(block.get("id") for block in blocks if isinstance(block, dict)) != expected_ids:
        raise PilotError("pilot block identity/order changed")
    for block in blocks:
        for field in (
            "id", "dataset", "split", "model", "family", "imgsz",
            "source_onnx_registry", "baseline_int8_registry", "baseline_fp8_registry",
            "calibration_list", "annotations", "matched_clean_manifest",
            "matched_clean_cache_root", "corruption_manifest_template", "corruption_cache_root",
        ):
            if field not in block:
                raise PilotError(f"pilot block lacks {field}: {block.get('id')}")
        for field in (
            "source_onnx_registry", "baseline_int8_registry", "baseline_fp8_registry",
            "calibration_list", "annotations", "matched_clean_manifest",
            "matched_clean_cache_root", "corruption_cache_root",
        ):
            resolve_under(root, block[field], f"{block['id']} {field}")
        if expected_attempt == "shared_mask_pilot_v3":
            if "baseline_fp8_engine_registry" not in block:
                raise PilotError(
                    f"V3 block lacks baseline_fp8_engine_registry: {block.get('id')}"
                )
            resolve_under(
                root,
                block["baseline_fp8_engine_registry"],
                f"{block['id']} baseline FP8 engine registry",
            )
        if block["family"] == "yolo":
            if "class_map" not in block:
                raise PilotError("YOLO block lacks immutable class map")
            resolve_under(root, block["class_map"], f"{block['id']} class map")
        elif block["family"] != "cross_family":
            raise PilotError(f"unsupported family: {block['family']}")
    return config


def validate_image_manifest(
    root: Path,
    block: dict[str, Any],
    *,
    corruption: str,
    severity: int,
) -> tuple[Path, str]:
    if corruption == "clean":
        value = block["matched_clean_manifest"]
        expected_corruption, expected_severity = "codec_control", 0
    else:
        value = block["corruption_manifest_template"].format(
            corruption=corruption, severity=severity
        )
        expected_corruption, expected_severity = corruption, severity
    path = resolve_under(root, value, "image manifest")
    try:
        manifest = read_manifest(path)
    except Exception as exc:  # read_manifest exposes format errors as ValueError/SystemExit across revisions.
        raise PilotError(f"invalid image manifest: {path}") from exc
    marker = path.with_suffix(path.suffix + ".complete")
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != manifest.get("manifest_sha256"):
        raise PilotError(f"image manifest completion marker mismatch: {path}")
    if manifest.get("dataset") != block["dataset"] or manifest.get("split") != block["split"]:
        raise PilotError(f"image manifest dataset/split mismatch: {path}")
    records = manifest.get("records")
    if not isinstance(records, list) or not records:
        raise PilotError(f"image manifest has no records: {path}")
    if any(
        row.get("corruption") != expected_corruption or row.get("severity") != expected_severity
        for row in records
    ):
        raise PilotError(f"image manifest condition mismatch: {path}")
    return path, str(manifest["manifest_sha256"])


@dataclass(frozen=True)
class EvidenceBundle:
    prediction: Path
    input_record: Path
    run_record: Path
    metric: Path

    @property
    def paths(self) -> tuple[Path, Path, Path, Path]:
        return self.prediction, self.input_record, self.run_record, self.metric


def shared_bundle(root: Path, condition_id: str) -> EvidenceBundle:
    return EvidenceBundle(
        root / "outputs" / "predictions" / ATTEMPT / f"{condition_id}.json",
        root / "outputs" / "inputs" / ATTEMPT / f"{condition_id}.json",
        root / "manifests" / "runs" / ATTEMPT / f"{condition_id}.json",
        root / "outputs" / "metrics" / ATTEMPT / f"{condition_id}.json",
    )


def shared_condition_id(block: dict[str, Any], precision: str, corruption: str, severity: int) -> str:
    suffix = "q95-clean-s0" if corruption == "clean" else f"{corruption}-s{severity}"
    return f"{block['dataset']}_{block['split']}__{block.get('model_slug', block['model'])}__{precision}__shared-mask__{suffix}"


def _load_json(path: Path, label: str) -> dict[str, Any]:
    if not path.is_file():
        raise PilotError(f"missing {label}: {path}")
    try:
        value = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"invalid {label}: {path}") from exc
    if not isinstance(value, dict):
        raise PilotError(f"{label} must be a JSON object: {path}")
    return value


def validate_bundle(
    bundle: EvidenceBundle,
    *,
    root: Path,
    block: dict[str, Any],
    precision: str,
    corruption: str,
    severity: int,
    expected_manifest_sha256: str | None = None,
    expected_engine_sha256: str | None = None,
) -> dict[str, Any]:
    """Validate prediction -> run/input -> metric bindings and scientific identity."""
    prediction_path, input_path, run_path, metric_path = bundle.paths
    if not prediction_path.is_file():
        raise PilotError(f"missing prediction: {prediction_path}")
    # Parse predictions now, not merely when COCOeval eventually consumes them.
    try:
        predictions = json.loads(prediction_path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise PilotError(f"invalid prediction JSON: {prediction_path}") from exc
    if not isinstance(predictions, list):
        raise PilotError(f"prediction payload is not a list: {prediction_path}")
    input_record = _load_json(input_path, "input record")
    run = _load_json(run_path, "run record")
    metric = _load_json(metric_path, "metric")
    prediction_sha = sha256_file(prediction_path)
    if run.get("prediction_sha256") != prediction_sha or metric.get("prediction_sha256") != prediction_sha:
        raise PilotError(f"prediction hash binding mismatch: {prediction_path}")
    if run.get("n_detections") != len(predictions):
        raise PilotError(f"prediction cardinality/run binding mismatch: {prediction_path}")
    if metric.get("run_record_sha256") != sha256_file(run_path):
        raise PilotError(f"metric/run binding mismatch: {metric_path}")
    input_manifest_sha = input_record.get("input_manifest_sha256")
    ids_sha = input_record.get("image_ids_sha256")
    if not isinstance(input_record.get("image_ids"), list) or len(input_record["image_ids"]) != metric.get("n_images"):
        raise PilotError(f"input image cardinality mismatch: {input_path}")
    if len(input_record["image_ids"]) != len(set(input_record["image_ids"])):
        raise PilotError(f"pre-bootstrap image IDs are duplicated: {input_path}")
    computed_ids_sha = hashlib.sha256(
        json.dumps(input_record["image_ids"], separators=(",", ":")).encode("utf-8")
    ).hexdigest()
    if ids_sha != computed_ids_sha:
        raise PilotError(f"input image-ID digest mismatch: {input_path}")
    if any(
        document.get("input_manifest_sha256") != input_manifest_sha
        or document.get("input_image_ids_sha256") != ids_sha
        for document in (run, metric)
    ):
        raise PilotError(f"run/metric input binding mismatch: {metric_path}")
    expected_corruption = "clean" if corruption == "clean" else corruption
    allowed_clean_labels = {"clean", "codec_control"}
    for document in (run, metric):
        if (
            document.get("dataset") != block["dataset"]
            or document.get("split") != block["split"]
            or document.get("model") != block["model"]
            or document.get("precision") != precision
            or int(document.get("severity", -1)) != severity
        ):
            raise PilotError(f"scientific identity mismatch: {metric_path}")
        observed_corruption = str(document.get("corruption"))
        if corruption == "clean":
            if observed_corruption not in allowed_clean_labels:
                raise PilotError(f"matched-clean label mismatch: {metric_path}")
        elif observed_corruption != expected_corruption:
            raise PilotError(f"corruption label mismatch: {metric_path}")
    if (
        not isinstance(run.get("condition_id"), str)
        or metric.get("condition_id") != run.get("condition_id")
        or run.get("n_images") != metric.get("n_images")
    ):
        raise PilotError(f"run/metric condition or cardinality mismatch: {metric_path}")
    if run.get("annotation_sha256") != sha256_file(resolve_under(root, block["annotations"], "annotations")):
        raise PilotError(f"run annotation binding mismatch: {run_path}")
    if expected_manifest_sha256 is not None and input_manifest_sha != expected_manifest_sha256:
        raise PilotError(f"input manifest differs from the frozen condition: {input_path}")
    if expected_engine_sha256 is not None and run.get("engine_sha256") != expected_engine_sha256:
        raise PilotError(f"run engine binding mismatch: {run_path}")
    stats = metric.get("stats")
    if not isinstance(stats, dict) or any(name not in stats for name in CLEAN_GATE_STATS):
        raise PilotError(f"metric lacks required AP statistics: {metric_path}")
    return {
        "prediction_sha256": prediction_sha,
        "input_record_sha256": sha256_file(input_path),
        "run_record_sha256": sha256_file(run_path),
        "metric_sha256": sha256_file(metric_path),
        "input_manifest_sha256": input_manifest_sha,
        "input_image_ids_sha256": ids_sha,
        "image_ids": input_record["image_ids"],
        "stats": stats,
        "condition_id": metric.get("condition_id"),
        "n_detections": len(predictions),
        "paths": {
            "prediction": str(prediction_path),
            "input_record": str(input_path),
            "run_record": str(run_path),
            "metric": str(metric_path),
        },
    }


def default_attempt(block: dict[str, Any], *, clean: bool) -> str:
    if block["family"] == "yolo":
        return "codec_control_p0_v1" if clean else f"{block['dataset']}_pilot_117_v1"
    return "cross_family_q95_clean_v1" if clean else "cross_family_v1"


def locate_default_bundle(
    root: Path,
    block: dict[str, Any],
    *,
    precision: str,
    corruption: str,
    severity: int,
) -> EvidenceBundle:
    """Find exactly one historical treatment cell by parsed scientific identity."""
    clean = corruption == "clean"
    attempt = default_attempt(block, clean=clean)
    metric_dir = root / "outputs" / "metrics" / attempt
    matches: list[Path] = []
    for path in sorted(metric_dir.glob("*.json")):
        try:
            metric = json.loads(path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            continue
        observed_corruption = metric.get("corruption")
        corruption_matches = (
            observed_corruption in {"clean", "codec_control"}
            if clean
            else observed_corruption == corruption
        )
        if (
            metric.get("dataset") == block["dataset"]
            and metric.get("split") == block["split"]
            and metric.get("model") == block["model"]
            and metric.get("precision") == precision
            and corruption_matches
            and metric.get("severity") == severity
        ):
            matches.append(path)
    if len(matches) != 1:
        raise PilotError(
            f"expected exactly one default cell for {block['id']}/{precision}/{corruption}-s{severity}; "
            f"found {len(matches)} in {metric_dir}"
        )
    name = matches[0].name
    return EvidenceBundle(
        root / "outputs" / "predictions" / attempt / name,
        root / "outputs" / "inputs" / attempt / name,
        root / "manifests" / "runs" / attempt / name,
        matches[0],
    )


def same_input_identity(records: Iterable[dict[str, Any]], label: str) -> tuple[str, str]:
    values = {
        (record["input_manifest_sha256"], record["input_image_ids_sha256"])
        for record in records
    }
    if len(values) != 1:
        raise PilotError(f"{label} arms do not share identical encoded bytes and ordered image IDs")
    return next(iter(values))
