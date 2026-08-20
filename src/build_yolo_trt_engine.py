#!/usr/bin/env python3
"""Build one TensorRT engine from a hash-verified ONNX precision artifact."""
from __future__ import annotations

import argparse
import json
import os
import subprocess
from datetime import datetime, timezone
from pathlib import Path

import tensorrt as trt

from topic_c.manifest import sha256_file


def read_complete(path: Path) -> dict:
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.is_file() or not marker.is_file() or marker.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise SystemExit(f"TRT BUILD REFUSED: incomplete input registry: {path}")
    return json.loads(path.read_text(encoding="utf-8"))


def onnx_from_registry(document: dict) -> tuple[Path, str, str]:
    if "output_onnx" in document:
        path, expected = Path(document["output_onnx"]).resolve(), document.get("output_onnx_sha256")
        precision = document.get("precision")
    else:
        path, expected = Path(document.get("onnx", "")).resolve(), document.get("onnx_sha256")
        precision = "fp32"
    if not path.is_file() or sha256_file(path) != expected:
        raise SystemExit("TRT BUILD REFUSED: ONNX hash mismatch")
    if precision not in {"fp32", "fp16", "int8-entropy", "fp8"}:
        raise SystemExit(f"TRT BUILD REFUSED: unsupported ONNX precision: {precision}")
    return path, expected, precision


def trtexec_build_command(*, executable: Path, onnx_path: Path, engine: Path, workspace: str) -> list[str]:
    """Build every confirmatory treatment with TF32 explicitly disabled."""
    return [
        str(executable),
        f"--onnx={onnx_path}",
        f"--saveEngine={engine}",
        "--skipInference",
        f"--memPoolSize=workspace:{workspace}",
        "--builderOptimizationLevel=3",
        "--noTF32",
    ]


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-registry", required=True)
    parser.add_argument("--precision", choices=("fp32", "fp16", "int8-entropy", "fp8"), required=True)
    parser.add_argument("--trt-root", required=True, help="read-only TensorRT installation whose bin/lib are provenance inputs")
    parser.add_argument("--engine", required=True)
    parser.add_argument("--build-log", required=True)
    parser.add_argument("--registry-out", required=True)
    parser.add_argument("--workspace", default="4096M")
    args = parser.parse_args()
    source_path, engine, build_log, record_out = (Path(args.onnx_registry).resolve(), Path(args.engine).resolve(),
                                                   Path(args.build_log).resolve(), Path(args.registry_out).resolve())
    if any(path.exists() for path in (engine, build_log, record_out)):
        raise SystemExit("TRT BUILD REFUSED: engine, log, or registry already exists")
    source = read_complete(source_path)
    onnx_path, onnx_sha, source_precision = onnx_from_registry(source)
    if source_precision != args.precision:
        raise SystemExit(f"TRT BUILD REFUSED: requested {args.precision} but source ONNX is {source_precision}")
    trt_root = Path(args.trt_root).resolve()
    executable, library = trt_root / "bin" / "trtexec", trt_root / "lib"
    if not executable.is_file() or not library.is_dir():
        raise SystemExit("TRT BUILD REFUSED: TensorRT bin/lib layout unavailable")
    engine.parent.mkdir(parents=True, exist_ok=True)
    build_log.parent.mkdir(parents=True, exist_ok=True)
    command = trtexec_build_command(
        executable=executable, onnx_path=onnx_path, engine=engine, workspace=args.workspace
    )
    environment = dict(os.environ)
    environment["LD_LIBRARY_PATH"] = str(library) + (":" + environment["LD_LIBRARY_PATH"] if environment.get("LD_LIBRARY_PATH") else "")
    with build_log.open("x", encoding="utf-8") as log:
        process = subprocess.run(command, stdout=log, stderr=subprocess.STDOUT, text=True, env=environment)
    if process.returncode != 0 or not engine.is_file() or engine.stat().st_size == 0:
        raise SystemExit(f"TRT BUILD REFUSED: trtexec failed; inspect immutable log: {build_log}")
    logger = trt.Logger(trt.Logger.WARNING)
    with engine.open("rb") as handle, trt.Runtime(logger) as runtime:
        parsed = runtime.deserialize_cuda_engine(handle.read())
    expected_outputs = len(source.get("output_names", ["output0"]))
    if parsed is None or parsed.num_io_tensors != 1 + expected_outputs:
        raise SystemExit("TRT BUILD REFUSED: resulting engine IO count does not match its ONNX registry")
    modes = [parsed.get_tensor_mode(parsed.get_tensor_name(index)) for index in range(parsed.num_io_tensors)]
    if modes.count(trt.TensorIOMode.INPUT) != 1 or modes.count(trt.TensorIOMode.OUTPUT) != expected_outputs:
        raise SystemExit("TRT BUILD REFUSED: resulting engine IO modes are invalid")
    record = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": source["dataset"],
        "model": source["model"], "precision": args.precision, "source_onnx_registry_sha256": sha256_file(source_path),
        "source_onnx": str(onnx_path), "source_onnx_sha256": onnx_sha, "engine": str(engine),
        "engine_sha256": sha256_file(engine), "engine_bytes": engine.stat().st_size,
        "trtexec": str(executable), "trtexec_sha256": sha256_file(executable), "trt_lib": str(library),
        "tensorrt_python_version": trt.__version__, "workspace": args.workspace, "command": command,
        "tf32_enabled": False,
        "build_log": str(build_log), "build_log_sha256": sha256_file(build_log),
        "imgsz": source.get("imgsz"),
        "calibration_list": source.get("calibration_list"),
        "calibration_sha256": source.get("calibration_sha256"),
        "calibration_method": source.get("calibration_method", "not_applicable"),
        "input_names": source.get("input_names"), "output_names": source.get("output_names"),
        "decoder": source.get("decoder"),
    }
    record_out.parent.mkdir(parents=True, exist_ok=True)
    record_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record_out.with_suffix(record_out.suffix + ".complete").write_text(sha256_file(record_out) + "\n", encoding="utf-8")
    print(json.dumps({"TRT ENGINE COMPLETE": str(engine), "engine_sha256": record["engine_sha256"],
                      "registry": str(record_out)}, indent=2))


if __name__ == "__main__":
    main()
