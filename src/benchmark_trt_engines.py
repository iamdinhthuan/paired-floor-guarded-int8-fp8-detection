#!/usr/bin/env python3
"""Run immutable, hash-bound TensorRT deployment benchmarks.

The data-loading and validation helpers are deliberately usable without a GPU.
Only ``run_benchmarks`` invokes ``nvidia-smi`` or ``trtexec``.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import os
import platform
import re
import shlex
import socket
import subprocess
import sys
import uuid
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Iterable, Mapping, Sequence


EXPECTED_DATASETS = ("coco", "voc", "kitti", "tt100k")
EXPECTED_MODELS = ("yolo11n", "yolo11m", "yolo11x")
EXPECTED_PRECISIONS = ("fp32", "int8-entropy", "fp8")
EXPECTED_CONDITIONS = 36
EXPECTED_REPETITIONS = 3
EXPECTED_RECORDS = EXPECTED_CONDITIONS * EXPECTED_REPETITIONS
GPU_IDLE_COMMAND = ("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader")
DEFAULT_CONFIG = Path("configs/ivc_deployment_benchmark_v1.json")
BENCHMARK_LOG_PREFIX = "IVC_TRTEXEC_LOG_V1 "
BENCHMARK_LOG_FORMAT = "ivc_trtexec_log_v1"
_NUMBER = r"[+-]?(?:nan|inf(?:inity)?|(?:\d+(?:\.\d*)?|\.\d+)(?:[eE][+-]?\d+)?)"


def sha256_file(path: Path | str) -> str:
    path = Path(path)
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(document: Mapping[str, Any], excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    raw = json.dumps(payload, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
    return hashlib.sha256(raw).hexdigest()


def encode_benchmark_log(metadata: Mapping[str, Any], raw_output: str) -> str:
    """Wrap immutable raw trtexec output in a hash-bound, machine-readable header."""
    if not isinstance(raw_output, str):
        raise TypeError("raw trtexec output must be text")
    raw_bytes = raw_output.encode("utf-8")
    header = dict(metadata)
    header["raw_output_bytes"] = len(raw_bytes)
    header["raw_output_sha256"] = hashlib.sha256(raw_bytes).hexdigest()
    encoded = json.dumps(header, sort_keys=True, separators=(",", ":"), allow_nan=False)
    return f"{BENCHMARK_LOG_PREFIX}{encoded}\n{raw_output}"


def parse_benchmark_log(document: str) -> tuple[dict[str, Any], str]:
    """Verify and split a runner-owned benchmark log envelope."""
    first_line, separator, raw_output = document.partition("\n")
    if not separator or not first_line.startswith(BENCHMARK_LOG_PREFIX):
        raise ValueError("invalid benchmark log envelope")
    try:
        metadata = json.loads(first_line[len(BENCHMARK_LOG_PREFIX) :])
    except json.JSONDecodeError as exc:
        raise ValueError("invalid benchmark log envelope metadata") from exc
    if not isinstance(metadata, dict):
        raise ValueError("invalid benchmark log envelope metadata")
    if metadata.get("schema_version") != 1 or metadata.get("log_format") != BENCHMARK_LOG_FORMAT:
        raise ValueError("unsupported benchmark log envelope format")
    raw_bytes = raw_output.encode("utf-8")
    if metadata.get("raw_output_bytes") != len(raw_bytes):
        raise ValueError("benchmark log raw output byte-count mismatch")
    if metadata.get("raw_output_sha256") != hashlib.sha256(raw_bytes).hexdigest():
        raise ValueError("benchmark log raw output SHA-256 mismatch")
    return metadata, raw_output


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _load_json(path: Path) -> dict[str, Any]:
    if not path.is_file():
        raise ValueError(f"missing JSON record: {path}")
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid JSON record: {path}: {exc}") from exc
    if not isinstance(document, dict):
        raise ValueError(f"JSON record must be an object: {path}")
    return document


def _load_complete_json(path: Path) -> tuple[dict[str, Any], str]:
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.is_file() or not marker.is_file():
        raise ValueError(f"incomplete registry: {path}")
    actual = sha256_file(path)
    if marker.read_text(encoding="utf-8").strip() != actual:
        raise ValueError(f"completion marker SHA-256 mismatch: {path}")
    document = _load_json(path)
    if "registry_sha256" in document and document["registry_sha256"] != canonical_hash(
        document, "registry_sha256"
    ):
        raise ValueError(f"canonical registry SHA-256 mismatch: {path}")
    return document, actual


def _resolve(root: Path, value: str | Path) -> Path:
    candidate = Path(value)
    return candidate.resolve() if candidate.is_absolute() else (root / candidate).resolve()


def _inside(root: Path, path: Path, label: str) -> Path:
    try:
        path.relative_to(root)
    except ValueError as exc:
        raise ValueError(f"{label} must remain inside project root: {path}") from exc
    return path


def _positive_int(value: Any, label: str) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise ValueError(f"{label} must be a positive integer")
    return value


def _positive_metric(value: Any, label: str) -> float:
    try:
        parsed = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{label} must be positive finite") from exc
    if not math.isfinite(parsed) or parsed <= 0:
        raise ValueError(f"{label} must be positive finite")
    return parsed


def parse_trtexec_latency(output: str) -> dict[str, float]:
    """Parse one standard TensorRT performance summary ending at PASSED."""
    error = (
        "trtexec metrics must contain positive finite values in exactly one complete final "
        "performance summary"
    )
    log_prefix = r"^[ \t]*(?:\[[^\]\r\n]+\][ \t]*)*"
    starts = list(
        re.finditer(
            log_prefix + r"===\s*Performance summary\s*===\s*$",
            output,
            flags=re.IGNORECASE | re.MULTILINE,
        )
    )
    passed = list(
        re.finditer(
            r"^&&&& PASSED TensorRT\.trtexec(?:\s+\[[^\]\r\n]*\])*\s+#\s+.+?\s*$",
            output,
            flags=re.MULTILINE,
        )
    )
    if len(starts) != 1 or len(passed) != 1 or passed[0].start() <= starts[0].end():
        raise ValueError(error)
    block = output[starts[0].end() : passed[0].start()]
    throughput_pattern = log_prefix + rf"Throughput:[ \t]*({_NUMBER})[ \t]*qps[ \t]*$"
    latency_pattern = (
        log_prefix
        + rf"Latency:[ \t]*min[ \t]*=[ \t]*({_NUMBER})[ \t]*ms,[ \t]*"
        rf"max[ \t]*=[ \t]*({_NUMBER})[ \t]*ms,[ \t]*"
        rf"mean[ \t]*=[ \t]*({_NUMBER})[ \t]*ms,[ \t]*"
        rf"median[ \t]*=[ \t]*({_NUMBER})[ \t]*ms,[ \t]*"
        rf"percentile\(90%\)[ \t]*=[ \t]*({_NUMBER})[ \t]*ms,[ \t]*"
        rf"percentile\(95%\)[ \t]*=[ \t]*({_NUMBER})[ \t]*ms,[ \t]*"
        rf"percentile\(99%\)[ \t]*=[ \t]*({_NUMBER})[ \t]*ms[ \t]*$"
    )
    flags = re.IGNORECASE | re.MULTILINE
    throughput_matches = re.findall(throughput_pattern, block, flags=flags)
    latency_matches = re.findall(latency_pattern, block, flags=flags)
    if (
        len(throughput_matches) != 1
        or len(latency_matches) != 1
        or len(re.findall(throughput_pattern, output, flags=flags)) != 1
        or len(re.findall(latency_pattern, output, flags=flags)) != 1
    ):
        raise ValueError(error)
    minimum, maximum, mean, median, p90, p95, p99 = (
        _positive_metric(value, label)
        for value, label in zip(
            latency_matches[0],
            (
                "minimum latency",
                "maximum latency",
                "mean latency",
                "median latency",
                "p90 latency",
                "p95 latency",
                "p99 latency",
            ),
        )
    )
    if not (
        minimum <= median <= p90 <= p95 <= p99 <= maximum
        and minimum <= mean <= maximum
    ):
        raise ValueError("primary latency tuple must describe an ordered latency distribution")
    parsed = {
        "throughput_qps": _positive_metric(throughput_matches[0], "throughput"),
        "latency_mean_ms": mean,
        "latency_median_ms": median,
        "latency_p99_ms": p99,
    }
    return parsed


def validate_idle_gpu(
    query_output: str, *, own_pid: int | None = None, allowed_pids: Iterable[int] = ()
) -> None:
    """Reject a compute-app query containing any PID outside an explicit allow-list."""
    allowed = {os.getpid() if own_pid is None else int(own_pid), *(int(pid) for pid in allowed_pids)}
    foreign: list[int] = []
    for line in query_output.splitlines():
        value = line.strip()
        if not value or value.lower() in {"n/a", "[n/a]", "no running processes found"}:
            continue
        if not value.isdigit():
            raise RuntimeError(f"malformed nvidia-smi compute PID output: {value!r}")
        pid = int(value)
        if pid not in allowed:
            foreign.append(pid)
    if foreign:
        raise RuntimeError(f"benchmark refused: foreign compute PID(s) present: {foreign}")


def _config_document(root: Path, config: Path | str | Mapping[str, Any] | None) -> tuple[dict[str, Any], str]:
    if isinstance(config, Mapping):
        document = dict(config)
        raw = json.dumps(document, sort_keys=True, separators=(",", ":"), allow_nan=False).encode("utf-8")
        return document, hashlib.sha256(raw).hexdigest()
    config_path = _resolve(root, config or DEFAULT_CONFIG)
    if not config_path.is_file():
        raise ValueError(
            f"benchmark requires exactly 36 unique engine conditions; missing config: {config_path}"
        )
    return _load_json(config_path), sha256_file(config_path)


def _registry_locations(root: Path, entry: Mapping[str, Any]) -> tuple[Path, Path, Path]:
    primary_ladder = _resolve(root, str(entry.get("ladder_registry", "")))
    use_mirror = not primary_ladder.is_file() and bool(entry.get("local_mirror_ladder_registry"))
    if use_mirror:
        return (
            _resolve(root, str(entry["local_mirror_ladder_registry"])),
            _resolve(root, str(entry["local_mirror_engine_registry_dir"])),
            _resolve(root, str(entry["local_mirror_onnx_registry_dir"])),
        )
    return (
        primary_ladder,
        _resolve(root, str(entry.get("engine_registry_dir", ""))),
        _resolve(root, str(entry.get("onnx_registry_dir", ""))),
    )


def _grid_error(detail: str) -> ValueError:
    return ValueError(f"benchmark requires exactly 36 unique engine conditions: {detail}")


def _onnx_asset(record: Mapping[str, Any], registry_path: Path) -> tuple[Path, str]:
    path_value = record.get("output_onnx") or record.get("onnx")
    digest = record.get("output_onnx_sha256") or record.get("onnx_sha256")
    if not isinstance(path_value, str) or not path_value:
        raise ValueError(f"ONNX asset path is absent: {registry_path}")
    if not isinstance(digest, str) or not re.fullmatch(r"[0-9a-f]{64}", digest):
        raise ValueError(f"ONNX asset SHA-256 is absent: {registry_path}")
    path = Path(path_value)
    if not path.is_file():
        raise ValueError(f"hash-bound ONNX asset is missing: {path}")
    if sha256_file(path) != digest:
        raise ValueError(f"hash-bound ONNX asset SHA-256 mismatch: {path}")
    return path, digest


def _inspect_onnx_input(path: Path) -> tuple[str, list[int | None], bool]:
    try:
        import onnx

        graph = onnx.load(path, load_external_data=False).graph
    except Exception as exc:
        raise ValueError(f"unable to inspect hash-bound ONNX input: {path}: {exc}") from exc
    if len(graph.input) != 1:
        raise ValueError(f"hash-bound ONNX must have exactly one input: {path}")
    tensor = graph.input[0]
    dimensions: list[int | None] = []
    for dimension in tensor.type.tensor_type.shape.dim:
        dimensions.append(int(dimension.dim_value) if dimension.HasField("dim_value") else None)
    if len(dimensions) != 4:
        raise ValueError(f"hash-bound ONNX input must be rank 4: {path}")
    return tensor.name, dimensions, any(value is None or value <= 0 for value in dimensions)


def build_benchmark_tasks(
    project_root: Path | str,
    config: Path | str | Mapping[str, Any] | None = None,
) -> list[dict[str, Any]]:
    """Load the validated scientific ladder and expand 36 conditions to 108 runs."""
    root = Path(project_root).resolve()
    document, config_sha256 = _config_document(root, config)
    if tuple(document.get("models", ())) != EXPECTED_MODELS:
        raise _grid_error("model grid differs from yolo11n/yolo11m/yolo11x")
    if tuple(document.get("precisions", ())) != EXPECTED_PRECISIONS:
        raise _grid_error("precision grid must be fp32/int8-entropy/fp8")
    if document.get("repetitions") != EXPECTED_REPETITIONS:
        raise _grid_error("exactly three repetitions are required")
    if document.get("warmup_ms") != 5000 or document.get("duration_s") != 30:
        raise ValueError("benchmark timing must be warmup_ms=5000 and duration_s=30")
    required_gpu_name = document.get("required_gpu_name")
    required_hostname = document.get("required_hostname")
    required_project_root = document.get("required_project_root")
    required_engine_root = document.get("required_engine_root")
    required_trtexec = document.get("required_trtexec")
    for label, value in (
        ("required_gpu_name", required_gpu_name),
        ("required_hostname", required_hostname),
        ("required_project_root", required_project_root),
        ("required_engine_root", required_engine_root),
        ("required_trtexec", required_trtexec),
    ):
        if not isinstance(value, str) or not value:
            raise ValueError(f"benchmark config {label} is required")
    engine_root = Path(required_engine_root)
    trtexec_binding = Path(required_trtexec)
    if not engine_root.is_absolute() or not trtexec_binding.is_absolute():
        raise ValueError("required engine root and trtexec binding must be absolute")
    output_dir = _inside(root, _resolve(root, str(document.get("output_dir", ""))), "output_dir")
    registry_sets = document.get("registry_sets")
    if not isinstance(registry_sets, list):
        raise _grid_error("registry_sets is absent")

    conditions: dict[tuple[str, str, str], dict[str, Any]] = {}
    for entry in registry_sets:
        if not isinstance(entry, dict):
            raise _grid_error("registry set is not an object")
        dataset = entry.get("dataset")
        if dataset not in EXPECTED_DATASETS:
            raise _grid_error(f"unexpected dataset: {dataset!r}")
        ladder_path, engine_dir, onnx_dir = _registry_locations(root, entry)
        ladder, ladder_sha256 = _load_complete_json(ladder_path)
        if ladder.get("dataset") != dataset:
            raise ValueError(f"ladder dataset mismatch: {ladder_path}")
        ladder_engines = ladder.get("engines")
        if not isinstance(ladder_engines, dict):
            raise ValueError(f"ladder engines are absent: {ladder_path}")
        for model in EXPECTED_MODELS:
            input_registry_path = onnx_dir / f"{dataset}_{model}_fp32_v1.json"
            input_record, input_registry_sha256 = _load_complete_json(input_registry_path)
            if (input_record.get("dataset"), input_record.get("model")) != (dataset, model):
                raise ValueError(f"source ONNX registry provenance mismatch: {input_registry_path}")
            source_input_name = input_record.get("input_name")
            if not isinstance(source_input_name, str) or not source_input_name:
                raise ValueError(f"source ONNX input_name is absent: {input_registry_path}")
            source_dynamic = input_record.get("dynamic")
            if not isinstance(source_dynamic, bool):
                raise ValueError(f"source ONNX dynamic flag is absent: {input_registry_path}")
            source_imgsz = _positive_int(input_record.get("imgsz"), "source ONNX imgsz")
            input_onnx_path, source_onnx_sha256 = _onnx_asset(input_record, input_registry_path)
            inspected_name, inspected_dimensions, inspected_dynamic = _inspect_onnx_input(input_onnx_path)
            if inspected_name != source_input_name or inspected_dynamic != source_dynamic:
                raise ValueError(f"source ONNX input metadata disagrees with graph: {input_registry_path}")
            if source_dynamic:
                requested_shape = input_record.get("benchmark_input_shape")
                if not isinstance(requested_shape, list) or len(requested_shape) != 4:
                    raise ValueError(
                        f"dynamic source ONNX requires benchmark_input_shape: {input_registry_path}"
                    )
                source_input_shape = [
                    _positive_int(value, "source ONNX benchmark input shape") for value in requested_shape
                ]
                for graph_value, requested in zip(inspected_dimensions, source_input_shape):
                    if graph_value is not None and graph_value > 0 and graph_value != requested:
                        raise ValueError(
                            f"dynamic source ONNX benchmark shape disagrees with graph: {input_registry_path}"
                        )
            else:
                if any(value is None or value <= 0 for value in inspected_dimensions):
                    raise ValueError(f"static source ONNX has unresolved dimensions: {input_registry_path}")
                source_input_shape = [int(value) for value in inspected_dimensions if value is not None]
                if source_input_shape != [1, 3, source_imgsz, source_imgsz]:
                    raise ValueError(f"source ONNX shape/imgsz disagreement: {input_registry_path}")
            for precision in EXPECTED_PRECISIONS:
                key = (dataset, model, precision)
                if key in conditions:
                    raise _grid_error(f"duplicate registry condition: {'/'.join(key)}")
                ladder_item = ladder_engines.get(model, {}).get(precision)
                if not isinstance(ladder_item, dict):
                    raise _grid_error(f"missing ladder item: {'/'.join(key)}")
                engine_registry_path = engine_dir / f"{dataset}_{model}_{precision}_v1.json"
                engine_record, engine_registry_sha256 = _load_complete_json(engine_registry_path)
                expected_identity = (engine_record.get("dataset"), engine_record.get("model"), engine_record.get("precision"))
                if expected_identity != key:
                    raise ValueError(f"engine registry provenance mismatch: {engine_registry_path}")
                if ladder_item.get("engine_registry_sha256") != engine_registry_sha256:
                    raise ValueError(f"ladder/engine registry SHA-256 mismatch: {engine_registry_path}")
                if ladder_item.get("sha256") != engine_record.get("engine_sha256"):
                    raise ValueError(f"ladder/engine SHA-256 mismatch: {engine_registry_path}")
                if ladder_item.get("path") != engine_record.get("engine"):
                    raise ValueError(f"ladder/engine path mismatch: {engine_registry_path}")

                onnx_registry_path = onnx_dir / f"{dataset}_{model}_{precision}_v1.json"
                onnx_record, onnx_registry_sha256 = _load_complete_json(onnx_registry_path)
                if engine_record.get("source_onnx_registry_sha256") != onnx_registry_sha256:
                    raise ValueError(f"engine/ONNX registry SHA-256 mismatch: {engine_registry_path}")
                onnx_identity = (onnx_record.get("dataset"), onnx_record.get("model"))
                if onnx_identity != (dataset, model):
                    raise ValueError(f"ONNX registry provenance mismatch: {onnx_registry_path}")
                recorded_precision = onnx_record.get("precision")
                if recorded_precision not in (None, precision):
                    raise ValueError(f"ONNX registry precision mismatch: {onnx_registry_path}")
                if precision == "fp32":
                    if onnx_registry_path != input_registry_path or onnx_registry_sha256 != input_registry_sha256:
                        raise ValueError(f"FP32 ONNX/input registry mismatch: {onnx_registry_path}")
                else:
                    if onnx_record.get("source_onnx_registry_sha256") != input_registry_sha256:
                        raise ValueError(f"quantized/source ONNX registry SHA-256 mismatch: {onnx_registry_path}")
                    if onnx_record.get("source_onnx_sha256") != source_onnx_sha256:
                        raise ValueError(f"quantized/source ONNX SHA-256 mismatch: {onnx_registry_path}")
                    if onnx_record.get("imgsz") != source_imgsz:
                        raise ValueError(f"quantized/source ONNX imgsz mismatch: {onnx_registry_path}")
                condition_onnx_path, condition_onnx_sha256 = _onnx_asset(
                    onnx_record, onnx_registry_path
                )
                if engine_record.get("source_onnx_sha256") != condition_onnx_sha256:
                    raise ValueError(f"engine/source ONNX SHA-256 mismatch: {engine_registry_path}")
                condition_name, condition_dimensions, condition_dynamic = _inspect_onnx_input(
                    condition_onnx_path
                )
                if condition_name != source_input_name or condition_dynamic != source_dynamic:
                    raise ValueError(f"condition/source ONNX input mismatch: {onnx_registry_path}")
                for axis, (graph_value, selected) in enumerate(
                    zip(condition_dimensions, source_input_shape)
                ):
                    if graph_value is not None and graph_value > 0 and graph_value != selected:
                        raise ValueError(
                            "condition ONNX fixed axis disagrees with benchmark shape "
                            f"at axis {axis}: {onnx_registry_path}"
                        )

                imgsz = _positive_int(engine_record.get("imgsz"), f"{key} imgsz")
                if imgsz != source_imgsz or ladder_item.get("imgsz") != source_imgsz:
                    raise ValueError(f"engine/source ONNX imgsz mismatch: {engine_registry_path}")
                engine_bytes = _positive_int(engine_record.get("engine_bytes"), f"{key} engine_bytes")
                for label in ("engine_sha256", "trtexec_sha256"):
                    value = engine_record.get(label)
                    if not isinstance(value, str) or not re.fullmatch(r"[0-9a-f]{64}", value):
                        raise ValueError(f"invalid {label}: {engine_registry_path}")
                if not engine_record.get("engine") or not engine_record.get("trtexec"):
                    raise ValueError(f"engine and trtexec paths are required: {engine_registry_path}")
                engine_path = Path(engine_record["engine"])
                try:
                    engine_path.relative_to(engine_root)
                except ValueError as exc:
                    raise ValueError(
                        f"engine path is outside required engine root: {engine_registry_path}"
                    ) from exc
                if Path(engine_record["trtexec"]) != trtexec_binding:
                    raise ValueError(f"trtexec path differs from required binding: {engine_registry_path}")
                condition_id = "__".join(key)
                conditions[key] = {
                    "condition_id": condition_id,
                    "dataset": dataset,
                    "model": model,
                    "precision": precision,
                    "engine": str(Path(engine_record["engine"])),
                    "engine_sha256": engine_record["engine_sha256"],
                    "engine_bytes": engine_bytes,
                    "trtexec": str(Path(engine_record["trtexec"])),
                    "trtexec_sha256": engine_record["trtexec_sha256"],
                    "trt_lib": str(engine_record.get("trt_lib", "")),
                    "tensorrt_python_version": engine_record.get("tensorrt_python_version", "unrecorded"),
                    "input_name": source_input_name,
                    "input_shape": source_input_shape,
                    "input_dynamic": source_dynamic,
                    "shape_flag_required": source_dynamic,
                    "input_binding_registry": str(input_registry_path),
                    "input_binding_registry_sha256": input_registry_sha256,
                    "input_onnx": str(condition_onnx_path),
                    "input_onnx_sha256": condition_onnx_sha256,
                    "warmup_ms": 5000,
                    "duration_s": 30,
                    "config_sha256": config_sha256,
                    "attempt": document.get("attempt"),
                    "required_gpu_name": required_gpu_name,
                    "required_hostname": required_hostname,
                    "required_project_root": required_project_root,
                    "ladder_registry": str(ladder_path),
                    "ladder_registry_sha256": ladder_sha256,
                    "engine_registry": str(engine_registry_path),
                    "engine_registry_sha256": engine_registry_sha256,
                    "onnx_registry": str(onnx_registry_path),
                    "onnx_registry_sha256": onnx_registry_sha256,
                }

    expected_keys = {
        (dataset, model, precision)
        for dataset in EXPECTED_DATASETS
        for model in EXPECTED_MODELS
        for precision in EXPECTED_PRECISIONS
    }
    if set(conditions) != expected_keys or len(conditions) != EXPECTED_CONDITIONS:
        missing = sorted(expected_keys - set(conditions))
        extra = sorted(set(conditions) - expected_keys)
        raise _grid_error(f"missing={missing}, extra={extra}")

    tasks: list[dict[str, Any]] = []
    for key in sorted(conditions):
        condition = conditions[key]
        for repetition in range(1, EXPECTED_REPETITIONS + 1):
            stem = f"{condition['condition_id']}__rep-{repetition:02d}"
            tasks.append(
                {
                    **condition,
                    "repetition": repetition,
                    "record_path": str(output_dir / f"{stem}.json"),
                    "log_path": str(output_dir / "logs" / f"{stem}.log"),
                }
            )
    if len(tasks) != EXPECTED_RECORDS:
        raise _grid_error(f"expanded task count is {len(tasks)}, expected {EXPECTED_RECORDS}")
    return tasks


def build_trtexec_command(task: Mapping[str, Any]) -> list[str]:
    command = [
        str(task["trtexec"]),
        f"--loadEngine={task['engine']}",
        f"--warmUp={task['warmup_ms']}",
        f"--duration={task['duration_s']}",
    ]
    if task.get("shape_flag_required"):
        shape = "x".join(str(value) for value in task["input_shape"])
        command.append(f"--shapes={task['input_name']}:{shape}")
    return command


def validate_prewrite(task: Mapping[str, Any]) -> None:
    """Validate all immutable inputs and output vacancies before a repetition."""
    for output_key in ("record_path", "log_path"):
        output = Path(str(task[output_key]))
        if output.exists():
            raise FileExistsError(f"refusing to overwrite benchmark artifact: {output}")
    engine = Path(str(task["engine"]))
    if not engine.is_file():
        raise ValueError(f"engine path is missing: {engine}")
    if engine.stat().st_size != task.get("engine_bytes"):
        raise ValueError(f"engine byte-size mismatch: {engine}")
    if sha256_file(engine) != task.get("engine_sha256"):
        raise ValueError(f"engine SHA-256 mismatch: {engine}")
    trtexec = Path(str(task["trtexec"]))
    if not trtexec.is_file() or not os.access(trtexec, os.X_OK):
        raise ValueError(f"trtexec binding is missing or not executable: {trtexec}")
    if sha256_file(trtexec) != task.get("trtexec_sha256"):
        raise ValueError(f"trtexec SHA-256 mismatch: {trtexec}")
    for prefix in (
        "ladder_registry",
        "engine_registry",
        "onnx_registry",
        "input_binding_registry",
    ):
        registry = Path(str(task[prefix]))
        if not registry.is_file() or sha256_file(registry) != task.get(f"{prefix}_sha256"):
            raise ValueError(f"{prefix.replace('_', ' ')} SHA-256 mismatch: {registry}")
    input_onnx = Path(str(task["input_onnx"]))
    if not input_onnx.is_file() or sha256_file(input_onnx) != task.get("input_onnx_sha256"):
        raise ValueError(f"input ONNX SHA-256 mismatch: {input_onnx}")


def _recorded(record: Mapping[str, Any], field: str) -> Any:
    value = record.get(field, "unrecorded")
    return "unrecorded" if value is None else value


def _record_key(record: Mapping[str, Any]) -> tuple[Any, Any, Any]:
    return record.get("dataset"), record.get("model"), record.get("precision")


def extract_engine_treatments(
    onnx_records: Sequence[Mapping[str, Any]], engine_records: Sequence[Mapping[str, Any]]
) -> list[dict[str, Any]]:
    """Join only recorded ONNX/engine provenance; never infer FP8 details."""
    onnx_by_key: dict[tuple[Any, Any, Any], Mapping[str, Any]] = {}
    for onnx in onnx_records:
        key = _record_key(onnx)
        if key in onnx_by_key:
            raise ValueError(f"duplicate ONNX treatment record: {key}")
        onnx_by_key[key] = onnx
    treatments: list[dict[str, Any]] = []
    seen: set[tuple[Any, Any, Any]] = set()
    for engine in engine_records:
        key = _record_key(engine)
        if key in seen:
            raise ValueError(f"duplicate engine treatment record: {key}")
        seen.add(key)
        onnx = onnx_by_key.get(key)
        if onnx is None and key[2] == "fp32":
            # The frozen FP32 export registries predate the explicit precision
            # field; the engine registry supplies the recorded treatment label.
            onnx = onnx_by_key.get((key[0], key[1], None))
        if onnx is None:
            raise ValueError(f"missing ONNX treatment record: {key}")
        onnx_sha = onnx.get("output_onnx_sha256") or onnx.get("onnx_sha256")
        source_sha = engine.get("source_onnx_sha256")
        if not isinstance(onnx_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", onnx_sha):
            raise ValueError(f"treatment requires recorded ONNX SHA-256 on both sides: {key}")
        if not isinstance(source_sha, str) or not re.fullmatch(r"[0-9a-f]{64}", source_sha):
            raise ValueError(f"treatment requires recorded ONNX SHA-256 on both sides: {key}")
        if source_sha != onnx_sha:
            raise ValueError(f"ONNX SHA-256 mismatch for treatment: {key}")
        treatments.append(
            {
                "dataset": key[0],
                "model": key[1],
                "precision": key[2],
                "onnx_sha256": onnx_sha or "unrecorded",
                "engine_sha256": _recorded(engine, "engine_sha256"),
                "engine_bytes": _recorded(engine, "engine_bytes"),
                "quantize_mode": _recorded(onnx, "quantize_mode"),
                "calibration_list": _recorded(onnx, "calibration_list"),
                "calibration_sha256": _recorded(onnx, "calibration_sha256"),
                "calibration_method": _recorded(onnx, "calibration_method"),
                "calibration_eps": _recorded(onnx, "calibration_eps"),
                "op_types_to_exclude": _recorded(onnx, "op_types_to_exclude"),
                "modelopt_version": _recorded(onnx, "modelopt_version"),
                "onnx_version": _recorded(onnx, "onnx_version"),
                "onnx_command": _recorded(onnx, "command"),
                "tensorrt_python_version": _recorded(engine, "tensorrt_python_version"),
                "trtexec_sha256": _recorded(engine, "trtexec_sha256"),
                "engine_command": _recorded(engine, "command"),
                "fp8_encoding": _recorded(onnx, "fp8_encoding"),
                "fp8_granularity": _recorded(onnx, "fp8_granularity"),
            }
        )
    return sorted(treatments, key=lambda row: (str(row["dataset"]), str(row["model"]), str(row["precision"])))


def treatments_from_tasks(tasks: Sequence[Mapping[str, Any]]) -> list[dict[str, Any]]:
    unique: dict[str, Mapping[str, Any]] = {}
    for task in tasks:
        unique.setdefault(str(task["condition_id"]), task)
    onnx_records: list[dict[str, Any]] = []
    engine_records: list[dict[str, Any]] = []
    for task in unique.values():
        onnx = _load_json(Path(str(task["onnx_registry"])))
        engine = _load_json(Path(str(task["engine_registry"])))
        if onnx.get("precision") is None:
            onnx = {**onnx, "precision": task["precision"]}
        onnx_records.append(onnx)
        engine_records.append(engine)
    treatments = extract_engine_treatments(onnx_records, engine_records)
    if len(treatments) != EXPECTED_CONDITIONS:
        raise ValueError(f"engine treatment table requires {EXPECTED_CONDITIONS} records")
    return treatments


def _validate_successful_trtexec_output(
    raw_output: str, expected_command: Sequence[str]
) -> None:
    running = re.findall(
        r"^&&&& RUNNING TensorRT\.trtexec(?:\s+\[[^\]\r\n]*\])*\s+#\s+(.+?)\s*$",
        raw_output,
        flags=re.MULTILINE,
    )
    if len(running) != 1:
        raise ValueError("raw trtexec command banner must occur exactly once")
    try:
        banner_command = shlex.split(running[0])
    except ValueError as exc:
        raise ValueError("raw trtexec command banner is malformed") from exc
    if banner_command != list(expected_command):
        raise ValueError("raw trtexec command banner disagrees with the recorded command")
    passed = re.findall(
        r"^&&&& PASSED TensorRT\.trtexec(?:\s+\[[^\]\r\n]*\])*\s+#\s+(.+?)\s*$",
        raw_output,
        flags=re.MULTILINE,
    )
    if len(passed) != 1:
        raise ValueError("raw trtexec successful envelope requires one PASSED and no FAILED marker")
    try:
        passed_command = shlex.split(passed[0])
    except ValueError as exc:
        raise ValueError("raw trtexec PASSED command is malformed") from exc
    if passed_command != list(expected_command):
        raise ValueError("raw trtexec PASSED command disagrees with the recorded command")
    failed = re.findall(
        r"^&&&& FAILED TensorRT\.trtexec\b[^\r\n]*$",
        raw_output,
        flags=re.MULTILINE,
    )
    if failed:
        raise ValueError("raw trtexec successful envelope requires one PASSED and no FAILED marker")


_RUNTIME_IDENTITY_FIELDS = (
    "hostname",
    "project_root",
    "platform",
    "python_version",
    "gpu_name",
    "gpu_uuid",
    "driver_version",
    "cuda_version",
    "tensorrt_python_version",
    "trtexec_version_output",
    "trtexec_executable",
    "trtexec_sha256",
)


def _validate_runtime_identity(
    identity: Any, task: Mapping[str, Any], location: Path | str
) -> dict[str, Any]:
    if not isinstance(identity, dict) or not identity:
        raise ValueError(f"GPU/runtime identity is absent: {location}")
    for field in _RUNTIME_IDENTITY_FIELDS:
        if not isinstance(identity.get(field), str) or not identity[field].strip():
            raise ValueError(f"GPU/runtime identity {field} is absent: {location}")
    if set(identity) != set(_RUNTIME_IDENTITY_FIELDS):
        raise ValueError(f"GPU/runtime identity has an unexpected field set: {location}")
    identity_patterns = {
        "gpu_uuid": r"GPU-[0-9a-fA-F]{8}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{4}-[0-9a-fA-F]{12}",
        "driver_version": r"[0-9]+(?:\.[0-9]+){1,3}",
        "cuda_version": r"[0-9]+(?:\.[0-9]+){1,2}",
        "python_version": r"[0-9]+\.[0-9]+\.[0-9]+(?:[-+._a-zA-Z0-9]*)?",
    }
    for field, pattern in identity_patterns.items():
        if re.fullmatch(pattern, identity[field]) is None:
            raise ValueError(f"GPU/runtime identity {field} is invalid: {location}")
    if identity["platform"].strip().lower() in {"unknown", "unrecorded", "tbd"}:
        raise ValueError(f"GPU/runtime identity platform is invalid: {location}")
    pinned_trt = str(task.get("tensorrt_python_version", ""))
    direct_version = re.search(
        rf"(?<![0-9.]){re.escape(pinned_trt)}(?![0-9.])",
        identity["trtexec_version_output"],
    )
    pinned_parts = re.fullmatch(r"(\d+)\.(\d+)\.(\d+)\.(\d+)", pinned_trt)
    numeric_banner = None
    if pinned_parts is not None:
        major, minor, patch, build = (int(value) for value in pinned_parts.groups())
        numeric_version = f"{major}{minor:02d}{patch:02d}"
        numeric_banner = re.search(
            rf"\[TensorRT v{re.escape(numeric_version)}\]\s*\[b{build}\]",
            identity["trtexec_version_output"],
        )
    if direct_version is None and numeric_banner is None:
        raise ValueError(f"GPU/runtime identity does not report the pinned TensorRT version: {location}")
    runtime_expected = {
        "hostname": task.get("required_hostname"),
        "project_root": task.get("required_project_root"),
        "gpu_name": task.get("required_gpu_name"),
        "tensorrt_python_version": task.get("tensorrt_python_version"),
        "trtexec_executable": task.get("trtexec"),
        "trtexec_sha256": task.get("trtexec_sha256"),
    }
    for field, expected in runtime_expected.items():
        if expected is None or identity.get(field) != expected:
            raise ValueError(f"GPU/runtime identity {field} mismatch: {location}")
    return identity


def build_completion_report(
    tasks: Sequence[Mapping[str, Any]], *, config_sha256: str
) -> dict[str, Any]:
    """Validate the complete 36 x 3 evidence chain and return its report."""
    if len(tasks) != EXPECTED_RECORDS:
        raise ValueError("completion report requires exactly 36 engines x 3 repetitions")
    expected_conditions = {
        (dataset, model, precision)
        for dataset in EXPECTED_DATASETS
        for model in EXPECTED_MODELS
        for precision in EXPECTED_PRECISIONS
    }
    repetitions_by_condition: dict[tuple[Any, Any, Any], set[Any]] = {}
    pair_identities: set[tuple[tuple[Any, Any, Any], Any]] = set()
    for task in tasks:
        condition = (task.get("dataset"), task.get("model"), task.get("precision"))
        if task.get("condition_id") != "__".join(str(value) for value in condition):
            raise ValueError("completion report requires the exact 36-condition Cartesian grid")
        repetitions_by_condition.setdefault(condition, set()).add(task.get("repetition"))
        pair_identities.add((condition, task.get("repetition")))
    if set(repetitions_by_condition) != expected_conditions:
        raise ValueError("completion report requires the exact 36-condition Cartesian grid")
    expected_repetitions = {1, 2, 3}
    if any(values != expected_repetitions for values in repetitions_by_condition.values()):
        raise ValueError("every benchmark condition requires repetitions {1, 2, 3}")
    if len(pair_identities) != EXPECTED_RECORDS:
        raise ValueError("completion report requires 108 unique condition/repetition identities")
    record_paths = [Path(str(task.get("record_path", ""))).resolve() for task in tasks]
    log_paths = [Path(str(task.get("log_path", ""))).resolve() for task in tasks]
    if len(set(record_paths)) != EXPECTED_RECORDS:
        raise ValueError("completion report requires 108 unique record paths")
    if len(set(log_paths)) != EXPECTED_RECORDS:
        raise ValueError("completion report requires 108 unique log paths")
    record_hashes: dict[str, str] = {}
    log_hashes: dict[str, str] = {}
    run_ids: set[str] = set()
    raw_output_hashes: set[str] = set()
    full_log_hashes: set[str] = set()
    common_runtime_identity: dict[str, Any] | None = None
    for task in tasks:
        record_path = Path(str(task["record_path"]))
        log_path = Path(str(task["log_path"]))
        artifact_stem = f"{task['condition_id']}__rep-{task['repetition']:02d}"
        if record_path.name != f"{artifact_stem}.json" or log_path.name != f"{artifact_stem}.log":
            raise ValueError(f"benchmark artifact path is noncanonical: {record_path}")
        if not record_path.is_file() or not log_path.is_file():
            raise ValueError(f"missing benchmark record/log: {record_path}")
        record = _load_json(record_path)
        if record.get("schema_version") != 1:
            raise ValueError(f"benchmark record schema_version mismatch: {record_path}")
        expected_fields = (
            "attempt",
            "condition_id",
            "dataset",
            "model",
            "precision",
            "repetition",
            "engine",
            "engine_sha256",
            "engine_bytes",
            "ladder_registry",
            "ladder_registry_sha256",
            "engine_registry",
            "engine_registry_sha256",
            "onnx_registry",
            "onnx_registry_sha256",
            "input_binding_registry",
            "input_binding_registry_sha256",
            "input_onnx",
            "input_onnx_sha256",
            "input_name",
            "input_shape",
            "input_dynamic",
            "shape_flag_required",
            "trtexec",
            "trtexec_sha256",
        )
        for field in expected_fields:
            if record.get(field) != task.get(field):
                raise ValueError(f"benchmark record {field} mismatch: {record_path}")
        if record.get("config_sha256") != config_sha256 or task.get("config_sha256") not in (
            None,
            config_sha256,
        ):
            raise ValueError(f"benchmark record config_sha256 mismatch: {record_path}")
        for field in ("started_at_utc", "ended_at_utc"):
            value = record.get(field)
            if not isinstance(value, str):
                raise ValueError(f"benchmark record {field} is absent: {record_path}")
            try:
                parsed_time = datetime.fromisoformat(value)
            except ValueError as exc:
                raise ValueError(f"benchmark record {field} is invalid: {record_path}") from exc
            if parsed_time.tzinfo is None:
                raise ValueError(f"benchmark record {field} must include a timezone: {record_path}")
        if datetime.fromisoformat(record["ended_at_utc"]) < datetime.fromisoformat(record["started_at_utc"]):
            raise ValueError(f"benchmark record timestamps are reversed: {record_path}")
        if record.get("trtexec_command") != build_trtexec_command(task):
            raise ValueError(f"benchmark command mismatch: {record_path}")
        identity = _validate_runtime_identity(record.get("gpu_runtime_identity"), task, record_path)
        if common_runtime_identity is None:
            common_runtime_identity = dict(identity)
        elif identity != common_runtime_identity:
            raise ValueError("completion report requires identical full GPU/runtime identity across all 108 records")
        if record.get("gpu_idle_query_command") != list(GPU_IDLE_COMMAND):
            raise ValueError(f"benchmark record gpu_idle_query_command mismatch: {record_path}")
        idle_output = record.get("gpu_idle_query_output")
        if not isinstance(idle_output, str):
            raise ValueError(f"benchmark record gpu_idle_query_output is absent: {record_path}")
        try:
            validate_idle_gpu(idle_output, own_pid=-1)
        except RuntimeError as exc:
            raise ValueError(f"benchmark record gpu_idle_query_output is not idle: {record_path}") from exc
        log_sha = sha256_file(log_path)
        if record.get("log_sha256") != log_sha:
            raise ValueError(f"log SHA-256 mismatch: {log_path}")
        metadata, raw_output = parse_benchmark_log(log_path.read_text(encoding="utf-8"))
        expected_command = build_trtexec_command(task)
        log_expected = {
            "attempt": task.get("attempt"),
            "config_sha256": config_sha256,
            "condition_id": task.get("condition_id"),
            "dataset": task.get("dataset"),
            "model": task.get("model"),
            "precision": task.get("precision"),
            "repetition": task.get("repetition"),
            "started_at_utc": record.get("started_at_utc"),
            "ended_at_utc": record.get("ended_at_utc"),
            "return_code": 0,
            "engine_sha256": task.get("engine_sha256"),
            "trtexec_command": expected_command,
        }
        for field, expected in log_expected.items():
            if metadata.get(field) != expected:
                label = "trtexec return code" if field == "return_code" else f"benchmark log {field}"
                raise ValueError(f"{label} mismatch: {log_path}")
        if record.get("log_format") != BENCHMARK_LOG_FORMAT:
            raise ValueError(f"benchmark record log_format mismatch: {record_path}")
        if record.get("trtexec_return_code") != 0:
            raise ValueError(
                f"benchmark record trtexec_return_code mismatch (trtexec return code): {record_path}"
            )
        run_id = record.get("run_id")
        if (
            not isinstance(run_id, str)
            or re.fullmatch(r"[0-9a-f]{32}", run_id) is None
            or metadata.get("run_id") != run_id
        ):
            raise ValueError(f"benchmark log run_id mismatch: {log_path}")
        raw_output_sha = hashlib.sha256(raw_output.encode("utf-8")).hexdigest()
        if (
            record.get("raw_output_sha256") != raw_output_sha
            or metadata.get("raw_output_sha256") != raw_output_sha
        ):
            raise ValueError(f"benchmark log raw_output_sha256 mismatch: {log_path}")
        if run_id in run_ids:
            raise ValueError("completion report requires 108 distinct benchmark run IDs")
        if raw_output_sha in raw_output_hashes:
            raise ValueError("completion report requires 108 distinct raw trtexec outputs")
        if log_sha in full_log_hashes:
            raise ValueError("completion report requires 108 distinct immutable log digests")
        run_ids.add(run_id)
        raw_output_hashes.add(raw_output_sha)
        full_log_hashes.add(log_sha)
        _validate_successful_trtexec_output(raw_output, expected_command)
        parsed_metrics = parse_trtexec_latency(raw_output)
        for field, parsed in parsed_metrics.items():
            recorded = _positive_metric(record.get(field), field)
            if recorded != parsed:
                raise ValueError(f"raw log metric mismatch for {field}: {record_path}")
        record_hashes[f"{artifact_stem}.json"] = sha256_file(record_path)
        log_hashes[f"{artifact_stem}.log"] = log_sha
    if len(record_hashes) != EXPECTED_RECORDS or len(log_hashes) != EXPECTED_RECORDS:
        raise ValueError("completion report requires 108 unique record/log digest entries")
    if len(run_ids) != EXPECTED_RECORDS or len(raw_output_hashes) != EXPECTED_RECORDS:
        raise ValueError("completion report requires 108 distinct successful trtexec runs")
    report: dict[str, Any] = {
        "schema_version": 1,
        "attempt": "ivc_deployment_benchmark_v1",
        "created_at_utc": _utc_now(),
        "config_sha256": config_sha256,
        "engine_conditions": len(repetitions_by_condition),
        "repetitions_per_engine": EXPECTED_REPETITIONS,
        "repetition_records": len(record_hashes),
        "raw_logs": len(log_hashes),
        "records_sha256": dict(sorted(record_hashes.items())),
        "logs_sha256": dict(sorted(log_hashes.items())),
        "limitations": ["No memory claim.", "No power or energy claim."],
    }
    report["report_sha256"] = canonical_hash(report, "report_sha256")
    return report


def _run(command: Sequence[str], *, env: Mapping[str, str] | None = None) -> subprocess.CompletedProcess[str]:
    return subprocess.run(command, text=True, stdout=subprocess.PIPE, stderr=subprocess.STDOUT, env=env, check=False)


def _capture_runtime_identity(task: Mapping[str, Any], root: Path) -> dict[str, Any]:
    gpu_query = _run(("nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"))
    if gpu_query.returncode != 0:
        raise RuntimeError(f"nvidia-smi GPU identity query failed:\n{gpu_query.stdout}")
    rows = [line.strip() for line in gpu_query.stdout.splitlines() if line.strip()]
    if len(rows) != 1:
        raise RuntimeError(f"benchmark requires exactly one visible GPU, observed {len(rows)}")
    parts = [part.strip() for part in rows[0].split(",")]
    if len(parts) != 3:
        raise RuntimeError(f"unexpected nvidia-smi GPU identity output: {rows[0]!r}")
    smi = _run(("nvidia-smi",))
    cuda_match = (
        re.search(r"CUDA (?:UMD )?Version:\s*([0-9.]+)", smi.stdout)
        if smi.returncode == 0
        else None
    )
    if cuda_match is None:
        raise RuntimeError("unable to record CUDA version from nvidia-smi")
    trtexec_env = os.environ.copy()
    if task.get("trt_lib"):
        current = trtexec_env.get("LD_LIBRARY_PATH", "")
        trtexec_env["LD_LIBRARY_PATH"] = str(task["trt_lib"]) + (
            os.pathsep + current if current else ""
        )
    trtexec_version = _run((str(task["trtexec"]), "--help"), env=trtexec_env)
    if trtexec_version.returncode != 0:
        raise RuntimeError(f"trtexec version query failed:\n{trtexec_version.stdout}")
    return {
        "hostname": socket.gethostname(),
        "project_root": str(root.resolve()),
        "platform": platform.platform(),
        "python_version": platform.python_version(),
        "gpu_name": parts[0],
        "gpu_uuid": parts[1],
        "driver_version": parts[2],
        "cuda_version": cuda_match.group(1),
        "tensorrt_python_version": task.get("tensorrt_python_version", "unrecorded"),
        "trtexec_version_output": trtexec_version.stdout.strip(),
        "trtexec_executable": str(task["trtexec"]),
        "trtexec_sha256": task["trtexec_sha256"],
    }


def _write_json_exclusive(path: Path, document: Mapping[str, Any]) -> None:
    with path.open("x", encoding="utf-8") as handle:
        json.dump(document, handle, indent=2, allow_nan=False)
        handle.write("\n")


def _validate_execution_contract(root: Path, config: Mapping[str, Any]) -> None:
    expected_root = config.get("required_project_root")
    if not isinstance(expected_root, str) or root.resolve() != Path(expected_root).resolve():
        raise ValueError(
            f"benchmark project root must equal required_project_root: {expected_root!r}"
        )
    hostname = config.get("required_hostname")
    if not isinstance(hostname, str) or not hostname.strip() or hostname.strip().lower() in {
        "unrecorded",
        "unknown",
        "tbd",
    }:
        raise ValueError("benchmark required_hostname must be concretely recorded before execution")


def run_benchmarks(
    root: Path,
    config_path: Path,
    *,
    preflight_only: bool = False,
    command_runner=_run,
    runtime_identity_provider=_capture_runtime_identity,
) -> dict[str, Any] | None:
    root = Path(root).resolve()
    config_path = Path(config_path).resolve()
    config = _load_json(config_path)
    _validate_execution_contract(root, config)
    tasks = build_benchmark_tasks(root, config_path)
    config_sha256 = sha256_file(config_path)
    report_path = _inside(root, _resolve(root, str(config["report"])), "report")
    treatments_path = _inside(root, _resolve(root, str(config["engine_treatments"])), "engine_treatments")
    for output in (report_path, treatments_path):
        if output.exists():
            raise FileExistsError(f"refusing to overwrite benchmark artifact: {output}")
    for task in tasks:
        validate_prewrite(task)
    treatments = treatments_from_tasks(tasks)
    if preflight_only:
        return None

    runtime_identity = runtime_identity_provider(tasks[0], root)
    required_gpu = config.get("required_gpu_name")
    if required_gpu and runtime_identity["gpu_name"] != required_gpu:
        raise RuntimeError(
            f"benchmark requires GPU {required_gpu!r}, observed {runtime_identity['gpu_name']!r}"
        )
    if runtime_identity.get("hostname") != config.get("required_hostname"):
        raise RuntimeError(
            f"benchmark requires hostname {config.get('required_hostname')!r}, "
            f"observed {runtime_identity.get('hostname')!r}"
        )
    if Path(str(runtime_identity.get("project_root", ""))).resolve() != root:
        raise RuntimeError("runtime project root identity mismatch")
    try:
        _validate_runtime_identity(runtime_identity, tasks[0], "pre-execution runtime gate")
    except ValueError as exc:
        raise RuntimeError(str(exc)) from exc
    initial_idle = command_runner(GPU_IDLE_COMMAND)
    if initial_idle.returncode != 0:
        raise RuntimeError(f"nvidia-smi compute PID query failed:\n{initial_idle.stdout}")
    validate_idle_gpu(initial_idle.stdout)
    Path(str(tasks[0]["record_path"])).parent.mkdir(parents=True, exist_ok=False)
    Path(str(tasks[0]["log_path"])).parent.mkdir(parents=True, exist_ok=False)
    report_path.parent.mkdir(parents=True, exist_ok=True)
    for task in tasks:
        # Repeat validation immediately before every run, including the idle-GPU gate.
        validate_prewrite(task)
        idle = command_runner(GPU_IDLE_COMMAND)
        if idle.returncode != 0:
            raise RuntimeError(f"nvidia-smi compute PID query failed:\n{idle.stdout}")
        validate_idle_gpu(idle.stdout)
        command = build_trtexec_command(task)
        env = os.environ.copy()
        if task.get("trt_lib"):
            current = env.get("LD_LIBRARY_PATH", "")
            env["LD_LIBRARY_PATH"] = str(task["trt_lib"]) + (os.pathsep + current if current else "")
        started = _utc_now()
        result = command_runner(command, env=env)
        ended = _utc_now()
        run_id = uuid.uuid4().hex
        log_metadata = {
            "schema_version": 1,
            "log_format": BENCHMARK_LOG_FORMAT,
            "attempt": config["attempt"],
            "config_sha256": config_sha256,
            "condition_id": task["condition_id"],
            "dataset": task["dataset"],
            "model": task["model"],
            "precision": task["precision"],
            "repetition": task["repetition"],
            "run_id": run_id,
            "started_at_utc": started,
            "ended_at_utc": ended,
            "return_code": result.returncode,
            "engine_sha256": task["engine_sha256"],
            "trtexec_command": command,
        }
        log_document = encode_benchmark_log(log_metadata, result.stdout)
        log_path = Path(str(task["log_path"]))
        with log_path.open("x", encoding="utf-8") as handle:
            handle.write(log_document)
        if result.returncode != 0:
            raise RuntimeError(f"trtexec failed for {task['condition_id']} repetition {task['repetition']}")
        _validate_successful_trtexec_output(result.stdout, command)
        metrics = parse_trtexec_latency(result.stdout)
        record = {
            "schema_version": 1,
            "attempt": config["attempt"],
            "config_sha256": config_sha256,
            "condition_id": task["condition_id"],
            "dataset": task["dataset"],
            "model": task["model"],
            "precision": task["precision"],
            "repetition": task["repetition"],
            "run_id": run_id,
            "log_format": BENCHMARK_LOG_FORMAT,
            "trtexec_return_code": result.returncode,
            "raw_output_sha256": hashlib.sha256(result.stdout.encode("utf-8")).hexdigest(),
            "started_at_utc": started,
            "ended_at_utc": ended,
            "engine": task["engine"],
            "engine_sha256": task["engine_sha256"],
            "engine_bytes": task["engine_bytes"],
            "ladder_registry": task["ladder_registry"],
            "ladder_registry_sha256": task["ladder_registry_sha256"],
            "engine_registry": task["engine_registry"],
            "engine_registry_sha256": task["engine_registry_sha256"],
            "onnx_registry": task["onnx_registry"],
            "onnx_registry_sha256": task["onnx_registry_sha256"],
            "input_binding_registry": task["input_binding_registry"],
            "input_binding_registry_sha256": task["input_binding_registry_sha256"],
            "input_onnx": task["input_onnx"],
            "input_onnx_sha256": task["input_onnx_sha256"],
            "input_name": task["input_name"],
            "input_shape": task["input_shape"],
            "input_dynamic": task["input_dynamic"],
            "shape_flag_required": task["shape_flag_required"],
            "trtexec": task["trtexec"],
            "trtexec_sha256": task["trtexec_sha256"],
            "gpu_runtime_identity": runtime_identity,
            "gpu_idle_query_command": list(GPU_IDLE_COMMAND),
            "gpu_idle_query_output": idle.stdout,
            "trtexec_command": command,
            "log_sha256": sha256_file(log_path),
            **metrics,
        }
        _write_json_exclusive(Path(str(task["record_path"])), record)

    treatment_document: dict[str, Any] = {
        "schema_version": 1,
        "attempt": config["attempt"],
        "records": treatments,
        "engine_conditions": len(treatments),
    }
    treatment_document["treatments_sha256"] = canonical_hash(
        treatment_document, "treatments_sha256"
    )
    treatment_bytes = (json.dumps(treatment_document, indent=2, allow_nan=False) + "\n").encode("utf-8")
    report = build_completion_report(tasks, config_sha256=config_sha256)
    report["engine_treatments_sha256"] = hashlib.sha256(treatment_bytes).hexdigest()
    report["report_sha256"] = canonical_hash(report, "report_sha256")
    with treatments_path.open("xb") as handle:
        handle.write(treatment_bytes)
    _write_json_exclusive(report_path, report)
    return report


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = _resolve(root, args.config)
    report = run_benchmarks(root, config_path, preflight_only=args.preflight_only)
    if report is None:
        print(json.dumps({"BENCHMARK PREFLIGHT VALID": True, "tasks": EXPECTED_RECORDS}, indent=2))
    else:
        print(json.dumps({"BENCHMARK COMPLETE": str(_resolve(root, _load_json(config_path)["report"])),
                          "report_sha256": report["report_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
