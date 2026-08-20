from __future__ import annotations

import hashlib
import json
import os
import shlex
import subprocess
from pathlib import Path

import pytest
import onnx
from onnx import TensorProto, helper

import benchmark_trt_engines as trt

from benchmark_trt_engines import (
    _validate_successful_trtexec_output,
    build_benchmark_tasks,
    build_completion_report,
    build_trtexec_command,
    encode_benchmark_log,
    extract_engine_treatments,
    parse_benchmark_log,
    parse_trtexec_latency,
    run_benchmarks,
    validate_idle_gpu,
    validate_prewrite,
)


DATASETS = ("coco", "voc", "kitti", "tt100k")
MODELS = ("yolo11n", "yolo11m", "yolo11x")
PRECISIONS = ("fp32", "int8-entropy", "fp8")
METRIC_SUMMARY = (
    "[08/12/2026-12:34:56] [I] === Performance summary ===\n"
    "[08/12/2026-12:34:56] [I] Throughput: 100.00 qps\n"
    "[08/12/2026-12:34:56] [I] Latency: min = 0.80 ms, max = 1.80 ms, mean = 1.20 ms, "
    "median = 1.10 ms, percentile(90%) = 1.50 ms, "
    "percentile(95%) = 1.65 ms, percentile(99%) = 1.80 ms\n"
    "[08/12/2026-12:34:56] [I] End-to-End Host Latency: min = 0.90 ms, max = 1.90 ms, "
    "mean = 1.30 ms, median = 1.20 ms, percentile(90%) = 1.60 ms, "
    "percentile(95%) = 1.75 ms, percentile(99%) = 1.90 ms\n"
    "[08/12/2026-12:34:56] [I] Enqueue Time: min = 0.01 ms, max = 0.05 ms, "
    "mean = 0.02 ms, median = 0.02 ms, percentile(90%) = 0.03 ms, "
    "percentile(95%) = 0.04 ms, percentile(99%) = 0.05 ms\n"
    "[08/12/2026-12:34:56] [I] H2D Latency: min = 0.01 ms, max = 0.04 ms, "
    "mean = 0.02 ms, median = 0.02 ms, percentile(90%) = 0.03 ms, "
    "percentile(95%) = 0.03 ms, percentile(99%) = 0.04 ms\n"
    "[08/12/2026-12:34:56] [I] GPU Compute Time: min = 0.70 ms, max = 1.70 ms, "
    "mean = 1.10 ms, median = 1.00 ms, percentile(90%) = 1.40 ms, "
    "percentile(95%) = 1.55 ms, percentile(99%) = 1.70 ms\n"
    "[08/12/2026-12:34:56] [I] D2H Latency: min = 0.01 ms, max = 0.04 ms, "
    "mean = 0.02 ms, median = 0.02 ms, percentile(90%) = 0.03 ms, "
    "percentile(95%) = 0.03 ms, percentile(99%) = 0.04 ms\n"
    "[08/12/2026-12:34:56] [I] Total Host Walltime: 30.001 s\n"
    "[08/12/2026-12:34:56] [I] Total GPU Compute Time: 29.991 s\n"
    "[08/12/2026-12:34:56] [I] Explanations of the performance metrics are printed "
    "in the verbose logs.\n"
)


def trtexec_terminal(status: str, command: list[str]) -> str:
    return (
        f"&&&& {status} TensorRT.trtexec [TensorRT v11.1.0.106] [b48] "
        f"# {shlex.join(command)}\n"
    )


def raw_trtexec_output(command: list[str], unique: str, *, passed: bool = True) -> str:
    banner = shlex.join(command)
    status = "PASSED" if passed else "FAILED"
    return (
        f"&&&& RUNNING TensorRT.trtexec [TensorRT v11.1.0.106] [b48] # {banner}\n"
        f"IVC synthetic run token: {unique}\n"
        f"{METRIC_SUMMARY}"
        f"{trtexec_terminal(status, command)}"
    )


def sha256(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def rewrite_evidence_log(task: dict, *, metadata_updates: dict | None = None, raw_output: str | None = None) -> None:
    log_path = Path(task["log_path"])
    metadata, original_raw = parse_benchmark_log(log_path.read_text(encoding="utf-8"))
    metadata.update(metadata_updates or {})
    rewritten_raw = original_raw if raw_output is None else raw_output
    log_path.write_text(encode_benchmark_log(metadata, rewritten_raw), encoding="utf-8")
    record_path = Path(task["record_path"])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["log_sha256"] = sha256(log_path)
    record["raw_output_sha256"] = hashlib.sha256(rewritten_raw.encode()).hexdigest()
    if metadata_updates and "return_code" in metadata_updates:
        record["trtexec_return_code"] = metadata_updates["return_code"]
    record_path.write_text(json.dumps(record) + "\n", encoding="utf-8")


def write_json(path: Path, document: dict, *, complete: bool = True) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    if complete:
        path.with_suffix(path.suffix + ".complete").write_text(sha256(path) + "\n", encoding="utf-8")
    return path


def write_onnx(path: Path, *, imgsz: int, dynamic: bool, channels: int = 3) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    dims = ["batch", channels, "height", "width"] if dynamic else [1, channels, imgsz, imgsz]
    graph = helper.make_graph(
        [helper.make_node("Identity", ["images"], ["output0"])],
        "benchmark-fixture",
        [helper.make_tensor_value_info("images", TensorProto.FLOAT, dims)],
        [helper.make_tensor_value_info("output0", TensorProto.FLOAT, dims)],
    )
    onnx.save(helper.make_model(graph), path)
    return path


def benchmark_fixture(
    root: Path,
    *,
    missing_input_name: bool = False,
    dynamic_input: bool = False,
    bad_dynamic_condition_axis: bool = False,
) -> Path:
    trtexec = root / "bin" / "trtexec"
    trtexec.parent.mkdir(parents=True)
    trtexec.write_bytes(b"frozen-trtexec")
    trtexec.chmod(0o755)
    registry_sets = []
    for dataset in DATASETS:
        registry_dir = root / "registries" / dataset / "engines"
        onnx_dir = root / "registries" / dataset / "onnx"
        ladder_engines = {}
        for model in MODELS:
            ladder_engines[model] = {}
            imgsz = 1280 if dataset == "tt100k" else 640
            fp32_onnx = write_onnx(
                root / "onnx" / dataset / model / "fp32.onnx",
                imgsz=imgsz,
                dynamic=dynamic_input,
            )
            fp32_onnx_sha = sha256(fp32_onnx)
            fp32_onnx_record = {
                "schema_version": 1,
                "dataset": dataset,
                "model": model,
                "onnx": str(fp32_onnx),
                "onnx_sha256": fp32_onnx_sha,
                "imgsz": imgsz,
                "dynamic": dynamic_input,
                "input_name": None if missing_input_name else "images",
            }
            if dynamic_input:
                fp32_onnx_record["benchmark_input_shape"] = [1, 3, imgsz, imgsz]
            fp32_onnx_path = write_json(
                onnx_dir / f"{dataset}_{model}_fp32_v1.json", fp32_onnx_record
            )
            for precision in PRECISIONS:
                engine = root / "engines" / dataset / model / f"{precision}.plan"
                engine.parent.mkdir(parents=True, exist_ok=True)
                engine.write_bytes(f"{dataset}/{model}/{precision}".encode())
                if precision == "fp32":
                    onnx_record = fp32_onnx_record
                    onnx_path = fp32_onnx_path
                    output_onnx_sha = fp32_onnx_sha
                else:
                    output_onnx = write_onnx(
                        root / "onnx" / dataset / model / f"{precision}.onnx",
                        imgsz=imgsz,
                        dynamic=dynamic_input,
                        channels=(
                            4
                            if bad_dynamic_condition_axis
                            and dataset == "coco"
                            and model == "yolo11n"
                            and precision == "fp8"
                            else 3
                        ),
                    )
                    output_onnx_sha = sha256(output_onnx)
                    onnx_record = {
                        "schema_version": 1,
                        "dataset": dataset,
                        "model": model,
                        "precision": precision,
                        "source_onnx_registry_sha256": sha256(fp32_onnx_path),
                        "source_onnx_sha256": fp32_onnx_sha,
                        "output_onnx": str(output_onnx),
                        "output_onnx_sha256": output_onnx_sha,
                        "imgsz": imgsz,
                        "quantize_mode": "fp8" if precision == "fp8" else "int8",
                        "calibration_list": str(root / "calibration" / f"{dataset}.json"),
                        "calibration_sha256": "d" * 64,
                        "calibration_method": "entropy",
                        "calibration_eps": ["cpu"],
                        "op_types_to_exclude": ["Sigmoid"],
                        "modelopt_version": "0.45.0",
                        "onnx_version": "1.21.0",
                        "command": ["quantize_yolo_onnx.py", "--mode", precision],
                    }
                    onnx_path = write_json(
                        onnx_dir / f"{dataset}_{model}_{precision}_v1.json", onnx_record
                    )
                engine_record = {
                    "schema_version": 1,
                    "dataset": dataset,
                    "model": model,
                    "precision": precision,
                    "source_onnx_registry_sha256": sha256(onnx_path),
                    "source_onnx_sha256": output_onnx_sha,
                    "engine": str(engine),
                    "engine_sha256": sha256(engine),
                    "engine_bytes": engine.stat().st_size,
                    "trtexec": str(trtexec),
                    "trtexec_sha256": sha256(trtexec),
                    "trt_lib": str(root / "lib"),
                    "tensorrt_python_version": "11.1.0.106",
                    "imgsz": imgsz,
                    "calibration_method": onnx_record.get("calibration_method", "not_applicable"),
                }
                engine_path = write_json(
                    registry_dir / f"{dataset}_{model}_{precision}_v1.json", engine_record
                )
                ladder_engines[model][precision] = {
                    "path": str(engine),
                    "sha256": sha256(engine),
                    "engine_registry_sha256": sha256(engine_path),
                    "imgsz": engine_record["imgsz"],
                }
        ladder = write_json(
            registry_dir / f"{dataset}_yolo11_nmx_ladder_v1.json",
            {
                "schema_version": 1,
                "dataset": dataset,
                "models": list(MODELS),
                "precisions": ["fp32", "fp16", "int8-entropy", "fp8"],
                "engines": ladder_engines,
            },
        )
        registry_sets.append(
            {
                "dataset": dataset,
                "ladder_registry": str(ladder.relative_to(root)),
                "engine_registry_dir": str(registry_dir.relative_to(root)),
                "onnx_registry_dir": str(onnx_dir.relative_to(root)),
            }
        )
    config = {
        "schema_version": 1,
        "attempt": "ivc_deployment_benchmark_v1",
        "models": list(MODELS),
        "precisions": list(PRECISIONS),
        "repetitions": 3,
        "warmup_ms": 5000,
        "duration_s": 30,
        "required_gpu_name": "NVIDIA GeForce RTX 5090",
        "required_hostname": "rtx5090-test-host",
        "required_project_root": str(root.resolve()),
        "required_engine_root": str((root / "engines").resolve()),
        "required_trtexec": str(trtexec.resolve()),
        "output_dir": "outputs/benchmarks/ivc_deployment_benchmark_v1",
        "report": "outputs/reports/ivc_deployment_benchmark_v1_complete.json",
        "engine_treatments": "outputs/benchmarks/ivc_deployment_benchmark_v1/engine_treatments.json",
        "registry_sets": registry_sets,
    }
    return write_json(root / "config.json", config, complete=False)


def add_execution_contract(tasks: list[dict], root: Path) -> None:
    for task in tasks:
        task.update(
            {
                "attempt": "ivc_deployment_benchmark_v1",
                "required_gpu_name": "NVIDIA GeForce RTX 5090",
                "required_hostname": "rtx5090-test-host",
                "required_project_root": str(root.resolve()),
                "config_sha256": "c" * 64,
            }
        )


def write_valid_evidence(tasks: list[dict], *, config_sha256: str = "c" * 64) -> None:
    for task in tasks:
        log = Path(task["log_path"])
        log.parent.mkdir(parents=True, exist_ok=True)
        command = build_trtexec_command(task)
        run_id = hashlib.sha256(
            f"{task['condition_id']}/{task['repetition']}".encode()
        ).hexdigest()[:32]
        raw_output = raw_trtexec_output(command, run_id)
        log_metadata = {
            "schema_version": 1,
            "log_format": "ivc_trtexec_log_v1",
            "attempt": task["attempt"],
            "config_sha256": config_sha256,
            "condition_id": task["condition_id"],
            "dataset": task["dataset"],
            "model": task["model"],
            "precision": task["precision"],
            "repetition": task["repetition"],
            "run_id": run_id,
            "started_at_utc": "2026-08-12T00:00:00+00:00",
            "ended_at_utc": "2026-08-12T00:00:31+00:00",
            "return_code": 0,
            "engine_sha256": task["engine_sha256"],
            "trtexec_command": command,
        }
        log.write_text(encode_benchmark_log(log_metadata, raw_output), encoding="utf-8")
        record = {
            "schema_version": 1,
            "attempt": task["attempt"],
            "config_sha256": config_sha256,
            "condition_id": task["condition_id"],
            "dataset": task["dataset"],
            "model": task["model"],
            "precision": task["precision"],
            "repetition": task["repetition"],
            "run_id": run_id,
            "log_format": "ivc_trtexec_log_v1",
            "trtexec_return_code": 0,
            "raw_output_sha256": hashlib.sha256(raw_output.encode()).hexdigest(),
            "started_at_utc": "2026-08-12T00:00:00+00:00",
            "ended_at_utc": "2026-08-12T00:00:31+00:00",
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
            "gpu_runtime_identity": {
                "hostname": task["required_hostname"],
                "project_root": task["required_project_root"],
                "platform": "Linux-test",
                "python_version": "3.10.0",
                "gpu_name": task["required_gpu_name"],
                "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
                "driver_version": "999.0",
                "cuda_version": "13.0",
                "tensorrt_python_version": task["tensorrt_python_version"],
                "trtexec_version_output": "TensorRT.trtexec 11.1.0.106",
                "trtexec_executable": task["trtexec"],
                "trtexec_sha256": task["trtexec_sha256"],
            },
            "gpu_idle_query_command": [
                "nvidia-smi",
                "--query-compute-apps=pid",
                "--format=csv,noheader",
            ],
            "gpu_idle_query_output": "\n",
            "trtexec_command": command,
            "log_sha256": sha256(log),
            "throughput_qps": 100.0,
            "latency_mean_ms": 1.2,
            "latency_median_ms": 1.1,
            "latency_p99_ms": 1.8,
        }
        Path(task["record_path"]).write_text(json.dumps(record) + "\n", encoding="utf-8")


def fake_runtime_identity(task: dict, root: Path) -> dict:
    return {
        "hostname": task["required_hostname"],
        "project_root": str(root.resolve()),
        "platform": "Linux-test",
        "python_version": "3.10.0",
        "gpu_name": task["required_gpu_name"],
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "driver_version": "999.0",
        "cuda_version": "13.0",
        "tensorrt_python_version": task["tensorrt_python_version"],
        "trtexec_version_output": "TensorRT.trtexec 11.1.0.106",
        "trtexec_executable": task["trtexec"],
        "trtexec_sha256": task["trtexec_sha256"],
    }


class FakeCommandRunner:
    def __init__(self, *, trtexec_returncode: int = 0, idle_output: str = "\n") -> None:
        self.calls: list[list[str]] = []
        self.trtexec_returncode = trtexec_returncode
        self.idle_output = idle_output
        self.benchmark_calls = 0

    def __call__(self, command, *, env=None):
        command = list(command)
        self.calls.append(command)
        if command[0] == "nvidia-smi":
            return subprocess.CompletedProcess(command, 0, self.idle_output)
        self.benchmark_calls += 1
        output = raw_trtexec_output(
            command,
            f"fake-invocation-{self.benchmark_calls}",
            passed=self.trtexec_returncode == 0,
        )
        return subprocess.CompletedProcess(command, self.trtexec_returncode, output)


def test_parse_trtexec_latency_reads_milliseconds_and_throughput() -> None:
    command = ["/opt/tensorrt/bin/trtexec", "--loadEngine=model.plan"]

    record = parse_trtexec_latency(raw_trtexec_output(command, "authentic-checked-fixture"))

    assert record == {
        "throughput_qps": 100.0,
        "latency_mean_ms": 1.2,
        "latency_median_ms": 1.1,
        "latency_p99_ms": 1.8,
    }


def test_validate_successful_trtexec_output_accepts_checked_canonical_terminal_argv() -> None:
    command = ["/opt/tensorrt/bin/trtexec", "--loadEngine=model with spaces.plan"]

    _validate_successful_trtexec_output(
        raw_trtexec_output(command, "canonical-terminal"), command
    )


def test_validate_successful_trtexec_output_rejects_canonical_failed_with_passed() -> None:
    command = ["/opt/tensorrt/bin/trtexec", "--loadEngine=model.plan"]
    output = raw_trtexec_output(command, "failed-and-passed").replace(
        trtexec_terminal("PASSED", command),
        trtexec_terminal("FAILED", command) + trtexec_terminal("PASSED", command),
    )

    with pytest.raises(ValueError, match="no FAILED"):
        _validate_successful_trtexec_output(output, command)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda text: text.replace("[08/12/2026-12:34:56] [I] Latency:", "[I] Missing Latency:", 1),
        lambda text: text.replace("mean = 1.20 ms", "mean = 0 ms", 1),
        lambda text: text.replace("Throughput: 100.00 qps", "Throughput: nan qps", 1),
    ],
)
def test_parse_trtexec_latency_rejects_missing_zero_or_nonfinite_values(mutation) -> None:
    command = ["/opt/tensorrt/bin/trtexec", "--loadEngine=model.plan"]
    text = mutation(raw_trtexec_output(command, "invalid-metric"))
    with pytest.raises(ValueError, match="positive finite"):
        parse_trtexec_latency(text)


@pytest.mark.parametrize(
    ("field", "needle", "replacement"),
    [
        ("minimum_zero", "min = 0.80 ms", "min = 0 ms"),
        ("minimum_nonfinite", "min = 0.80 ms", "min = nan ms"),
        ("maximum_zero", "max = 1.80 ms", "max = 0 ms"),
        ("maximum_nonfinite", "max = 1.80 ms", "max = nan ms"),
        ("p90_zero", "percentile(90%) = 1.50 ms", "percentile(90%) = 0 ms"),
        ("p90_nonfinite", "percentile(90%) = 1.50 ms", "percentile(90%) = nan ms"),
        ("p95_zero", "percentile(95%) = 1.65 ms", "percentile(95%) = 0 ms"),
        ("p95_nonfinite", "percentile(95%) = 1.65 ms", "percentile(95%) = nan ms"),
    ],
)
def test_parse_trtexec_latency_rejects_unreported_nonpositive_or_nonfinite_tuple_values(
    field: str, needle: str, replacement: str
) -> None:
    command = ["/opt/tensorrt/bin/trtexec", "--loadEngine=model.plan"]
    text = raw_trtexec_output(command, field).replace(needle, replacement, 1)

    with pytest.raises(ValueError, match="positive finite"):
        parse_trtexec_latency(text)


@pytest.mark.parametrize(
    ("ordering", "needle", "replacement"),
    [
        ("minimum_above_median", "min = 0.80 ms", "min = 1.15 ms"),
        ("median_above_p90", "median = 1.10 ms", "median = 1.55 ms"),
        ("p90_above_p95", "percentile(90%) = 1.50 ms", "percentile(90%) = 1.70 ms"),
        ("p95_above_p99", "percentile(95%) = 1.65 ms", "percentile(95%) = 1.85 ms"),
        ("p99_above_maximum", "percentile(99%) = 1.80 ms", "percentile(99%) = 1.81 ms"),
        ("mean_below_minimum", "mean = 1.20 ms", "mean = 0.70 ms"),
        ("mean_above_maximum", "mean = 1.20 ms", "mean = 1.81 ms"),
    ],
)
def test_parse_trtexec_latency_rejects_impossible_tuple_ordering(
    ordering: str, needle: str, replacement: str
) -> None:
    command = ["/opt/tensorrt/bin/trtexec", "--loadEngine=model.plan"]
    text = raw_trtexec_output(command, ordering).replace(needle, replacement, 1)

    with pytest.raises(ValueError, match="ordered latency distribution"):
        parse_trtexec_latency(text)


@pytest.mark.parametrize(
    "mutation",
    [
        lambda text, footer: text.replace(footer, "[I] Throughput: 999 qps\n" + footer),
        lambda text, footer: text.replace(footer, METRIC_SUMMARY + footer),
        lambda text, footer: text.replace(
            footer, "[I] === Performance summary ===\n[I] Throughput: 999 qps\n" + footer
        ),
    ],
)
def test_parse_trtexec_latency_rejects_spliced_or_ambiguous_summary(mutation) -> None:
    command = ["/opt/tensorrt/bin/trtexec", "--loadEngine=model.plan"]
    footer = trtexec_terminal("PASSED", command)
    text = mutation(raw_trtexec_output(command, "ambiguous-summary"), footer)
    with pytest.raises(ValueError, match="exactly one complete final performance summary"):
        parse_trtexec_latency(text)


def test_benchmark_log_envelope_round_trips_and_rejects_raw_mutation() -> None:
    metadata = {"schema_version": 1, "log_format": "ivc_trtexec_log_v1", "run_id": "a" * 32}
    encoded = encode_benchmark_log(metadata, "raw trtexec bytes\n")

    parsed, raw = parse_benchmark_log(encoded)

    assert parsed["run_id"] == "a" * 32
    assert raw == "raw trtexec bytes\n"
    with pytest.raises(ValueError, match="SHA-256|byte-count"):
        parse_benchmark_log(encoded + "mutation")


def test_build_benchmark_tasks_requires_36_unique_engines(tmp_path: Path) -> None:
    with pytest.raises(ValueError, match="36 unique engine conditions"):
        build_benchmark_tasks(tmp_path)


def test_build_benchmark_tasks_expands_exact_grid_to_three_repetitions(tmp_path: Path) -> None:
    config = benchmark_fixture(tmp_path)

    tasks = build_benchmark_tasks(tmp_path, config)

    assert len(tasks) == 108
    assert len({task["condition_id"] for task in tasks}) == 36
    assert {task["repetition"] for task in tasks} == {1, 2, 3}
    tt100k = next(task for task in tasks if task["dataset"] == "tt100k")
    assert tt100k["input_name"] == "images"
    assert tt100k["input_shape"] == [1, 3, 1280, 1280]
    assert tt100k["input_dynamic"] is False
    assert tt100k["shape_flag_required"] is False
    assert tt100k["input_binding_registry"].endswith("tt100k_yolo11m_fp32_v1.json")
    assert tt100k["input_binding_registry_sha256"] == sha256(
        Path(tt100k["input_binding_registry"])
    )


def test_build_benchmark_tasks_rejects_missing_hash_bound_input_name(tmp_path: Path) -> None:
    config = benchmark_fixture(tmp_path, missing_input_name=True)

    with pytest.raises(ValueError, match="source ONNX input_name"):
        build_benchmark_tasks(tmp_path, config)


def test_build_benchmark_tasks_marks_hash_bound_dynamic_input_for_shape_flag(tmp_path: Path) -> None:
    config = benchmark_fixture(tmp_path, dynamic_input=True)

    task = build_benchmark_tasks(tmp_path, config)[0]

    assert task["input_dynamic"] is True
    assert task["shape_flag_required"] is True
    assert build_trtexec_command(task)[-1] == "--shapes=images:1x3x640x640"


def test_build_benchmark_tasks_rejects_dynamic_condition_fixed_axis_mismatch(tmp_path: Path) -> None:
    config = benchmark_fixture(tmp_path, dynamic_input=True, bad_dynamic_condition_axis=True)

    with pytest.raises(ValueError, match="condition ONNX fixed axis disagrees with benchmark shape"):
        build_benchmark_tasks(tmp_path, config)


def test_build_benchmark_tasks_rejects_engine_outside_required_remote_layout(tmp_path: Path) -> None:
    config = benchmark_fixture(tmp_path)
    document = json.loads(config.read_text(encoding="utf-8"))
    document["required_engine_root"] = str(tmp_path / "different-engine-root")
    config.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="required engine root"):
        build_benchmark_tasks(tmp_path, config)


def test_preflight_refuses_unrecorded_required_hostname_without_subprocess(tmp_path: Path) -> None:
    config = benchmark_fixture(tmp_path)
    document = json.loads(config.read_text(encoding="utf-8"))
    document["required_hostname"] = "unrecorded"
    config.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="required_hostname"):
        run_benchmarks(tmp_path, config, preflight_only=True)


def test_runner_injected_lifecycle_writes_exact_108_immutable_records(tmp_path: Path) -> None:
    config = benchmark_fixture(tmp_path)
    runner = FakeCommandRunner()

    report = run_benchmarks(
        tmp_path,
        config,
        command_runner=runner,
        runtime_identity_provider=fake_runtime_identity,
    )

    assert report is not None
    assert report["repetition_records"] == 108
    assert report["raw_logs"] == 108
    idle_calls = [call for call in runner.calls if call[0] == "nvidia-smi"]
    benchmark_calls = [call for call in runner.calls if call[0] != "nvidia-smi"]
    assert len(idle_calls) == 109  # one pre-write gate plus one per repetition
    assert len(benchmark_calls) == 108
    output_dir = tmp_path / "outputs" / "benchmarks" / "ivc_deployment_benchmark_v1"
    assert len(list(output_dir.glob("*.json"))) == 109  # 108 records plus treatments
    assert len(list((output_dir / "logs").glob("*.log"))) == 108
    assert (tmp_path / "outputs" / "reports" / "ivc_deployment_benchmark_v1_complete.json").is_file()

    calls_before = len(runner.calls)
    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        run_benchmarks(
            tmp_path,
            config,
            command_runner=runner,
            runtime_identity_provider=fake_runtime_identity,
        )
    assert len(runner.calls) == calls_before


def test_runner_failure_keeps_raw_log_but_writes_no_record_or_completion(tmp_path: Path) -> None:
    config = benchmark_fixture(tmp_path)
    runner = FakeCommandRunner(trtexec_returncode=1)

    with pytest.raises(RuntimeError, match="trtexec failed"):
        run_benchmarks(
            tmp_path,
            config,
            command_runner=runner,
            runtime_identity_provider=fake_runtime_identity,
        )

    output_dir = tmp_path / "outputs" / "benchmarks" / "ivc_deployment_benchmark_v1"
    assert len(list((output_dir / "logs").glob("*.log"))) == 1
    assert list(output_dir.glob("*.json")) == []
    assert not (tmp_path / "outputs" / "reports" / "ivc_deployment_benchmark_v1_complete.json").exists()


def test_runner_rejects_command_free_success_after_one_immutable_log(tmp_path: Path) -> None:
    config = benchmark_fixture(tmp_path)

    class CommandFreeRunner(FakeCommandRunner):
        def __call__(self, command, *, env=None):
            result = super().__call__(command, env=env)
            if list(command)[0] != "nvidia-smi":
                return subprocess.CompletedProcess(
                    command,
                    0,
                    METRIC_SUMMARY + trtexec_terminal("PASSED", list(command)),
                )
            return result

    runner = CommandFreeRunner()
    with pytest.raises(ValueError, match="trtexec command banner"):
        run_benchmarks(
            tmp_path,
            config,
            command_runner=runner,
            runtime_identity_provider=fake_runtime_identity,
        )

    output_dir = tmp_path / "outputs" / "benchmarks" / "ivc_deployment_benchmark_v1"
    logs = list((output_dir / "logs").glob("*.log"))
    assert len(logs) == 1
    metadata, _ = parse_benchmark_log(logs[0].read_text(encoding="utf-8"))
    assert metadata["return_code"] == 0
    assert list(output_dir.glob("*.json")) == []


def test_runner_initial_idle_gate_blocks_before_any_output_parent_or_artifact_creation(
    tmp_path: Path,
) -> None:
    config = benchmark_fixture(tmp_path)
    runner = FakeCommandRunner(idle_output="456\n")

    with pytest.raises(RuntimeError, match="foreign compute PID"):
        run_benchmarks(
            tmp_path,
            config,
            command_runner=runner,
            runtime_identity_provider=fake_runtime_identity,
        )

    assert runner.calls == [
        ["nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader"]
    ]
    assert not (tmp_path / "outputs").exists()


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("hostname", "wrong-host", "requires hostname"),
        ("project_root", "/wrong/project", "project root identity mismatch"),
    ],
)
def test_runner_rejects_runtime_host_layout_before_commands(
    tmp_path: Path, field: str, bad_value: str, message: str
) -> None:
    config = benchmark_fixture(tmp_path)
    runner = FakeCommandRunner()

    def bad_identity(task, root):
        identity = fake_runtime_identity(task, root)
        identity[field] = bad_value
        return identity

    with pytest.raises(RuntimeError, match=message):
        run_benchmarks(
            tmp_path,
            config,
            command_runner=runner,
            runtime_identity_provider=bad_identity,
        )
    assert runner.calls == []


def test_runner_rejects_unpinned_trtexec_identity_before_commands(tmp_path: Path) -> None:
    config = benchmark_fixture(tmp_path)
    runner = FakeCommandRunner()

    def bad_identity(task, root):
        identity = fake_runtime_identity(task, root)
        identity["trtexec_version_output"] = "TensorRT.trtexec unknown"
        return identity

    with pytest.raises(RuntimeError, match="pinned TensorRT version"):
        run_benchmarks(
            tmp_path,
            config,
            command_runner=runner,
            runtime_identity_provider=bad_identity,
        )
    assert runner.calls == []
    assert not list(tmp_path.glob("outputs/benchmarks/**/*.log"))


def test_build_benchmark_tasks_rejects_duplicate_registry_condition(tmp_path: Path) -> None:
    config = benchmark_fixture(tmp_path)
    document = json.loads(config.read_text(encoding="utf-8"))
    document["registry_sets"][1] = dict(document["registry_sets"][0])
    config.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="36 unique engine conditions"):
        build_benchmark_tasks(tmp_path, config)


def test_validate_idle_gpu_accepts_empty_output_and_rejects_foreign_pid() -> None:
    validate_idle_gpu("\n", own_pid=123)
    validate_idle_gpu("123\n", own_pid=123)

    with pytest.raises(RuntimeError, match="foreign compute PID"):
        validate_idle_gpu("456\n", own_pid=123)


@pytest.mark.parametrize("cuda_banner", ("CUDA Version: 13.3", "CUDA UMD Version: 13.3"))
def test_capture_runtime_identity_accepts_nvidia_smi_cuda_version_banners(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path, cuda_banner: str
) -> None:
    """Records CUDA provenance from legacy and current NVIDIA SMI banners."""
    task = {
        "trtexec": "/opt/tensorrt/bin/trtexec",
        "trtexec_sha256": "a" * 64,
        "tensorrt_python_version": "11.1.0",
    }

    def fake_run(command, *, env=None):
        if tuple(command) == ("nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"):
            return subprocess.CompletedProcess(
                command,
                0,
                "NVIDIA GeForce RTX 5090, GPU-73cd7254-dd5d-7323-f317-b05645d2bdcb, 610.43.02\n",
            )
        if tuple(command) == ("nvidia-smi",):
            return subprocess.CompletedProcess(
                command,
                0,
                f"NVIDIA-SMI 610.43.02  {cuda_banner}\n",
            )
        if tuple(command) == ("/opt/tensorrt/bin/trtexec", "--help"):
            return subprocess.CompletedProcess(command, 0, "TensorRT.trtexec 11.1.0.106\n")
        raise AssertionError(command)

    monkeypatch.setattr(trt, "_run", fake_run)

    identity = trt._capture_runtime_identity(task, tmp_path)

    assert identity["cuda_version"] == "13.3"


def test_capture_runtime_identity_sets_tensor_rt_library_path_for_trtexec_version(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """The provenance version probe must use the same TensorRT library path as runs."""
    task = {
        "trtexec": "/opt/tensorrt/bin/trtexec",
        "trtexec_sha256": "a" * 64,
        "tensorrt_python_version": "11.1.0",
        "trt_lib": "/opt/tensorrt/lib",
    }
    monkeypatch.setenv("LD_LIBRARY_PATH", "/system/lib")

    def fake_run(command, *, env=None):
        if tuple(command) == ("nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"):
            return subprocess.CompletedProcess(
                command,
                0,
                "NVIDIA GeForce RTX 5090, GPU-73cd7254-dd5d-7323-f317-b05645d2bdcb, 610.43.02\n",
            )
        if tuple(command) == ("nvidia-smi",):
            return subprocess.CompletedProcess(command, 0, "CUDA UMD Version: 13.3\n")
        if tuple(command) == ("/opt/tensorrt/bin/trtexec", "--help"):
            assert env is not None
            assert env["LD_LIBRARY_PATH"] == f"/opt/tensorrt/lib{os.pathsep}/system/lib"
            return subprocess.CompletedProcess(command, 0, "TensorRT.trtexec 11.1.0.106\n")
        raise AssertionError(command)

    monkeypatch.setattr(trt, "_run", fake_run)

    identity = trt._capture_runtime_identity(task, tmp_path)

    assert identity["trtexec_version_output"] == "TensorRT.trtexec 11.1.0.106"


def test_capture_runtime_identity_queries_trtexec_help_for_supported_version_banner(
    monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> None:
    """TensorRT 11.1 exposes its successful version banner through --help, not --version."""
    task = {
        "trtexec": "/opt/tensorrt/bin/trtexec",
        "trtexec_sha256": "a" * 64,
        "tensorrt_python_version": "11.1.0.106",
        "trt_lib": "/opt/tensorrt/lib",
    }

    def fake_run(command, *, env=None):
        if tuple(command) == ("nvidia-smi", "--query-gpu=name,uuid,driver_version", "--format=csv,noheader"):
            return subprocess.CompletedProcess(
                command,
                0,
                "NVIDIA GeForce RTX 5090, GPU-73cd7254-dd5d-7323-f317-b05645d2bdcb, 610.43.02\n",
            )
        if tuple(command) == ("nvidia-smi",):
            return subprocess.CompletedProcess(command, 0, "CUDA UMD Version: 13.3\n")
        if tuple(command) == ("/opt/tensorrt/bin/trtexec", "--help"):
            assert env is not None
            return subprocess.CompletedProcess(
                command,
                0,
                "&&&& RUNNING TensorRT.trtexec [TensorRT v110100] [b106] # /opt/tensorrt/bin/trtexec --help\n",
            )
        raise AssertionError(command)

    monkeypatch.setattr(trt, "_run", fake_run)

    identity = trt._capture_runtime_identity(task, tmp_path)

    assert "TensorRT v110100" in identity["trtexec_version_output"]


def test_runtime_identity_accepts_tensor_rt_numeric_help_banner(tmp_path: Path) -> None:
    """Validates TensorRT's real vMMmmpp/bBUILD help-banner encoding."""
    task = {
        "tensorrt_python_version": "11.1.0.106",
        "required_hostname": "rtx5090-test-host",
        "required_project_root": str(tmp_path.resolve()),
        "required_gpu_name": "NVIDIA GeForce RTX 5090",
        "trtexec": "/opt/tensorrt/bin/trtexec",
        "trtexec_sha256": "a" * 64,
    }
    identity = {
        "hostname": "rtx5090-test-host",
        "project_root": str(tmp_path.resolve()),
        "platform": "Linux-test",
        "python_version": "3.10.0",
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "driver_version": "610.43.02",
        "cuda_version": "13.3",
        "tensorrt_python_version": "11.1.0.106",
        "trtexec_version_output": "&&&& RUNNING TensorRT.trtexec [TensorRT v110100] [b106] # trtexec --help",
        "trtexec_executable": "/opt/tensorrt/bin/trtexec",
        "trtexec_sha256": "a" * 64,
    }

    assert trt._validate_runtime_identity(identity, task, "test") == identity


@pytest.mark.parametrize("banner", ("v110100] [b105", "v110101] [b106"))
def test_runtime_identity_rejects_near_miss_tensor_rt_numeric_help_banner(
    tmp_path: Path, banner: str
) -> None:
    """Prevents a superficially similar TensorRT help banner from satisfying the pin."""
    task = {
        "tensorrt_python_version": "11.1.0.106",
        "required_hostname": "rtx5090-test-host",
        "required_project_root": str(tmp_path.resolve()),
        "required_gpu_name": "NVIDIA GeForce RTX 5090",
        "trtexec": "/opt/tensorrt/bin/trtexec",
        "trtexec_sha256": "a" * 64,
    }
    identity = {
        "hostname": "rtx5090-test-host",
        "project_root": str(tmp_path.resolve()),
        "platform": "Linux-test",
        "python_version": "3.10.0",
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "gpu_uuid": "GPU-12345678-1234-1234-1234-123456789abc",
        "driver_version": "610.43.02",
        "cuda_version": "13.3",
        "tensorrt_python_version": "11.1.0.106",
        "trtexec_version_output": f"&&&& RUNNING TensorRT.trtexec [TensorRT {banner}] # trtexec --help",
        "trtexec_executable": "/opt/tensorrt/bin/trtexec",
        "trtexec_sha256": "a" * 64,
    }

    with pytest.raises(ValueError, match="pinned TensorRT version"):
        trt._validate_runtime_identity(identity, task, "test")


def test_command_uses_registry_binding_and_static_engine_needs_no_shape_flag(tmp_path: Path) -> None:
    task = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))[0]

    command = build_trtexec_command(task)

    assert command == [
        task["trtexec"],
        f"--loadEngine={task['engine']}",
        "--warmUp=5000",
        "--duration=30",
    ]


def test_command_adds_shape_only_when_registry_marks_dynamic_input(tmp_path: Path) -> None:
    task = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))[0]
    task["shape_flag_required"] = True

    assert build_trtexec_command(task)[-1] == "--shapes=images:1x3x640x640"


def test_prewrite_validates_hashes_and_refuses_existing_artifact(tmp_path: Path) -> None:
    task = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))[0]
    validate_prewrite(task)
    Path(task["record_path"]).parent.mkdir(parents=True, exist_ok=True)
    Path(task["record_path"]).write_text("occupied", encoding="utf-8")

    with pytest.raises(FileExistsError, match="refusing to overwrite"):
        validate_prewrite(task)

    Path(task["record_path"]).unlink()
    engine = Path(task["engine"])
    engine.write_bytes(b"x" * engine.stat().st_size)
    with pytest.raises(ValueError, match="engine SHA-256 mismatch"):
        validate_prewrite(task)


def test_prewrite_rechecks_hash_bound_input_onnx(tmp_path: Path) -> None:
    task = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))[0]
    onnx_path = Path(task["input_onnx"])
    payload = bytearray(onnx_path.read_bytes())
    payload[-1] ^= 1
    onnx_path.write_bytes(payload)

    with pytest.raises(ValueError, match="input ONNX SHA-256 mismatch"):
        validate_prewrite(task)


def test_completion_report_requires_all_108_hash_valid_records(tmp_path: Path) -> None:
    tasks = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))
    add_execution_contract(tasks, tmp_path)
    write_valid_evidence(tasks)

    report = build_completion_report(tasks, config_sha256="c" * 64)

    assert report["engine_conditions"] == 36
    assert report["repetition_records"] == 108
    assert report["raw_logs"] == 108
    assert len(report["records_sha256"]) == 108
    assert len(report["logs_sha256"]) == 108
    assert all("/" not in key for key in report["records_sha256"])
    assert all(key.endswith(".json") for key in report["records_sha256"])
    assert all("/" not in key for key in report["logs_sha256"])
    assert all(key.endswith(".log") for key in report["logs_sha256"])

    Path(tasks[-1]["log_path"]).write_text("mutated", encoding="utf-8")
    with pytest.raises(ValueError, match="log SHA-256 mismatch"):
        build_completion_report(tasks, config_sha256="c" * 64)


def test_completion_report_rejects_unrelated_raw_log_even_when_digest_matches(tmp_path: Path) -> None:
    tasks = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))
    add_execution_contract(tasks, tmp_path)
    write_valid_evidence(tasks)
    log = Path(tasks[0]["log_path"])
    log.write_text("unrelated text with no TensorRT summary\n", encoding="utf-8")
    record_path = Path(tasks[0]["record_path"])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["log_sha256"] = sha256(log)
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="benchmark log envelope"):
        build_completion_report(tasks, config_sha256="c" * 64)


@pytest.mark.parametrize(
    "attack",
    [
        "command_free",
        "wrong_engine",
        "wrong_passed_command",
        "missing_passed",
        "failed",
        "failed_plus_passed",
        "nonzero",
    ],
)
def test_completion_report_rejects_non_authentic_or_failed_trtexec_output(
    tmp_path: Path, attack: str
) -> None:
    tasks = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))
    add_execution_contract(tasks, tmp_path)
    write_valid_evidence(tasks)
    task = tasks[0]
    command = build_trtexec_command(task)
    metadata_updates = None
    if attack == "command_free":
        raw = METRIC_SUMMARY + trtexec_terminal("PASSED", command)
    elif attack == "wrong_engine":
        wrong = list(command)
        wrong[1] = "--loadEngine=/copied/wrong.plan"
        raw = raw_trtexec_output(wrong, "wrong-engine")
    elif attack == "wrong_passed_command":
        wrong = list(command)
        wrong[1] = "--loadEngine=/copied/wrong.plan"
        raw = raw_trtexec_output(command, "wrong-passed-command").replace(
            trtexec_terminal("PASSED", command), trtexec_terminal("PASSED", wrong)
        )
    elif attack == "missing_passed":
        raw = raw_trtexec_output(command, "missing-passed").replace(
            trtexec_terminal("PASSED", command), ""
        )
    elif attack == "failed":
        raw = raw_trtexec_output(command, "failed", passed=False)
    elif attack == "failed_plus_passed":
        raw = raw_trtexec_output(command, "failed-plus-passed").replace(
            trtexec_terminal("PASSED", command),
            trtexec_terminal("FAILED", command) + trtexec_terminal("PASSED", command),
        )
    else:
        raw = raw_trtexec_output(command, "nonzero")
        metadata_updates = {"return_code": 1}
    rewrite_evidence_log(task, metadata_updates=metadata_updates, raw_output=raw)

    with pytest.raises(
        ValueError,
        match="trtexec (command banner|PASSED command|successful envelope|return code)",
    ):
        build_completion_report(tasks, config_sha256="c" * 64)


def test_completion_report_rejects_duplicate_raw_output_across_repetitions(tmp_path: Path) -> None:
    tasks = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))
    add_execution_contract(tasks, tmp_path)
    write_valid_evidence(tasks)
    first = tasks[0]
    second = tasks[1]
    _, first_raw = parse_benchmark_log(Path(first["log_path"]).read_text(encoding="utf-8"))
    rewrite_evidence_log(second, raw_output=first_raw)

    with pytest.raises(ValueError, match="108 distinct raw trtexec outputs"):
        build_completion_report(tasks, config_sha256="c" * 64)


def test_completion_report_rejects_metric_that_disagrees_with_raw_log(tmp_path: Path) -> None:
    tasks = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))
    add_execution_contract(tasks, tmp_path)
    write_valid_evidence(tasks)
    record_path = Path(tasks[0]["record_path"])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["latency_median_ms"] = 9.9
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="raw log metric mismatch"):
        build_completion_report(tasks, config_sha256="c" * 64)


def test_completion_report_reparse_rejects_invalid_unreported_latency_tuple_value(
    tmp_path: Path,
) -> None:
    tasks = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))
    add_execution_contract(tasks, tmp_path)
    write_valid_evidence(tasks)
    task = tasks[0]
    _, raw_output = parse_benchmark_log(Path(task["log_path"]).read_text(encoding="utf-8"))
    rewrite_evidence_log(
        task,
        raw_output=raw_output.replace("min = 0.80 ms", "min = nan ms", 1),
    )

    with pytest.raises(ValueError, match="positive finite"):
        build_completion_report(tasks, config_sha256="c" * 64)


@pytest.mark.parametrize(
    "missing",
    [
        "attempt",
        "config_sha256",
        "dataset",
        "started_at_utc",
        "ladder_registry_sha256",
        "engine_registry",
        "onnx_registry_sha256",
        "input_binding_registry_sha256",
        "input_onnx_sha256",
        "gpu_idle_query_command",
        "gpu_idle_query_output",
        "run_id",
        "log_format",
        "trtexec_return_code",
        "raw_output_sha256",
    ],
)
def test_completion_report_rejects_missing_record_provenance(tmp_path: Path, missing: str) -> None:
    tasks = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))
    add_execution_contract(tasks, tmp_path)
    write_valid_evidence(tasks)
    record_path = Path(tasks[0]["record_path"])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record.pop(missing)
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match=missing):
        build_completion_report(tasks, config_sha256="c" * 64)


def test_completion_report_rejects_incomplete_runtime_identity(tmp_path: Path) -> None:
    tasks = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))
    add_execution_contract(tasks, tmp_path)
    write_valid_evidence(tasks)
    record_path = Path(tasks[0]["record_path"])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["gpu_runtime_identity"].pop("gpu_uuid")
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="gpu_uuid"):
        build_completion_report(tasks, config_sha256="c" * 64)


def test_completion_report_requires_identical_full_runtime_identity(tmp_path: Path) -> None:
    tasks = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))
    add_execution_contract(tasks, tmp_path)
    write_valid_evidence(tasks)
    record_path = Path(tasks[1]["record_path"])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["gpu_runtime_identity"]["gpu_uuid"] = "GPU-aaaaaaaa-bbbb-cccc-dddd-eeeeeeeeeeee"
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="identical full GPU/runtime identity"):
        build_completion_report(tasks, config_sha256="c" * 64)


@pytest.mark.parametrize(
    ("field", "bad_value", "message"),
    [
        ("gpu_uuid", "GPU-unrecorded", "gpu_uuid is invalid"),
        ("driver_version", "fabricated-driver", "driver_version is invalid"),
        ("cuda_version", "CUDA-ish", "cuda_version is invalid"),
        ("python_version", "python3", "python_version is invalid"),
        ("trtexec_version_output", "TensorRT.trtexec unknown", "pinned TensorRT version"),
        ("trtexec_executable", "/wrong/trtexec", "trtexec_executable mismatch"),
    ],
)
def test_completion_report_rejects_unverifiable_runtime_identity(
    tmp_path: Path, field: str, bad_value: str, message: str
) -> None:
    tasks = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))
    add_execution_contract(tasks, tmp_path)
    write_valid_evidence(tasks)
    record_path = Path(tasks[0]["record_path"])
    record = json.loads(record_path.read_text(encoding="utf-8"))
    record["gpu_runtime_identity"][field] = bad_value
    record_path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        build_completion_report(tasks, config_sha256="c" * 64)


def test_completion_report_rejects_uneven_repetition_allocation(tmp_path: Path) -> None:
    tasks = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))
    add_execution_contract(tasks, tmp_path)
    tasks[2].update(
        {
            "condition_id": tasks[3]["condition_id"],
            "dataset": tasks[3]["dataset"],
            "model": tasks[3]["model"],
            "precision": tasks[3]["precision"],
            "repetition": 4,
        }
    )

    with pytest.raises(ValueError, match=r"repetitions \{1, 2, 3\}"):
        build_completion_report(tasks, config_sha256="c" * 64)


def test_completion_report_rejects_out_of_range_repetition(tmp_path: Path) -> None:
    tasks = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))
    add_execution_contract(tasks, tmp_path)
    tasks[2]["repetition"] = 4

    with pytest.raises(ValueError, match=r"repetitions \{1, 2, 3\}"):
        build_completion_report(tasks, config_sha256="c" * 64)


def test_completion_report_rejects_noncartesian_scientific_identity(tmp_path: Path) -> None:
    tasks = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))
    add_execution_contract(tasks, tmp_path)
    tasks[0]["dataset"] = "unexpected"

    with pytest.raises(ValueError, match="exact 36-condition Cartesian grid"):
        build_completion_report(tasks, config_sha256="c" * 64)


@pytest.mark.parametrize("path_field", ["record_path", "log_path"])
def test_completion_report_rejects_duplicate_resolved_artifact_path(
    tmp_path: Path, path_field: str
) -> None:
    tasks = build_benchmark_tasks(tmp_path, benchmark_fixture(tmp_path))
    add_execution_contract(tasks, tmp_path)
    tasks[-1][path_field] = tasks[0][path_field]

    with pytest.raises(ValueError, match=f"108 unique {path_field.replace('_', ' ')}s"):
        build_completion_report(tasks, config_sha256="c" * 64)


def test_engine_treatments_join_recorded_fields_without_inventing_fp8_details() -> None:
    onnx_records = [
        {
            "dataset": "coco",
            "model": "yolo11n",
            "precision": "fp8",
            "output_onnx_sha256": "a" * 64,
            "quantize_mode": "fp8",
            "calibration_list": "/frozen/calibration.json",
            "calibration_sha256": "d" * 64,
            "calibration_method": "entropy",
            "calibration_eps": ["cpu"],
            "op_types_to_exclude": ["Sigmoid"],
            "modelopt_version": "0.45.0",
            "onnx_version": "1.21.0",
            "command": ["quantize_yolo_onnx.py", "--mode", "fp8"],
        }
    ]
    engine_records = [
        {
            "dataset": "coco",
            "model": "yolo11n",
            "precision": "fp8",
            "source_onnx_sha256": "a" * 64,
            "engine_sha256": "b" * 64,
            "engine_bytes": 123,
            "tensorrt_python_version": "11.1.0.106",
            "trtexec_sha256": "c" * 64,
            "command": ["trtexec", "--onnx=x", "--saveEngine=y"],
        }
    ]

    treatments = extract_engine_treatments(onnx_records, engine_records)

    assert treatments[0]["quantize_mode"] == "fp8"
    assert treatments[0]["modelopt_version"] == "0.45.0"
    assert treatments[0]["calibration_list"] == "/frozen/calibration.json"
    assert treatments[0]["calibration_sha256"] == "d" * 64
    assert treatments[0]["fp8_encoding"] == "unrecorded"
    assert treatments[0]["fp8_granularity"] == "unrecorded"


def test_engine_treatments_reject_mismatched_onnx_hash() -> None:
    onnx = [{"dataset": "voc", "model": "yolo11n", "precision": "fp8", "output_onnx_sha256": "a" * 64}]
    engine = [{"dataset": "voc", "model": "yolo11n", "precision": "fp8", "source_onnx_sha256": "b" * 64}]

    with pytest.raises(ValueError, match="ONNX SHA-256 mismatch"):
        extract_engine_treatments(onnx, engine)


@pytest.mark.parametrize("missing_side", ["onnx", "engine"])
def test_engine_treatments_reject_missing_onnx_join_hash(missing_side: str) -> None:
    onnx = [
        {
            "dataset": "voc",
            "model": "yolo11n",
            "precision": "fp8",
            "output_onnx_sha256": "a" * 64,
        }
    ]
    engine = [
        {
            "dataset": "voc",
            "model": "yolo11n",
            "precision": "fp8",
            "source_onnx_sha256": "a" * 64,
        }
    ]
    if missing_side == "onnx":
        onnx[0].pop("output_onnx_sha256")
    else:
        engine[0].pop("source_onnx_sha256")

    with pytest.raises(ValueError, match="requires recorded ONNX SHA-256"):
        extract_engine_treatments(onnx, engine)


def test_engine_treatments_join_fp32_export_record_with_unrecorded_precision() -> None:
    onnx = [{"dataset": "coco", "model": "yolo11n", "onnx_sha256": "a" * 64}]
    engine = [
        {
            "dataset": "coco",
            "model": "yolo11n",
            "precision": "fp32",
            "source_onnx_sha256": "a" * 64,
            "engine_sha256": "b" * 64,
        }
    ]

    treatments = extract_engine_treatments(onnx, engine)

    assert treatments[0]["precision"] == "fp32"
    assert treatments[0]["quantize_mode"] == "unrecorded"
