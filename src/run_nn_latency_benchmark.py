#!/usr/bin/env python3
"""NN-scoped TensorRT latency benchmark via the Python runtime.

Per-repetition: deserialize the hash-bound engine, create a fresh execution
context, warm up, then time N iterations of execute_async_v3 with CUDA events.
The GPU is idle-gated before every repetition (allow-listing sunshine's
display-streaming PIDs and this process). Emits per-iteration samples and
hash-bound JSON records mirroring the ivc benchmark envelope conventions.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import statistics
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path

IDLE_POLL_S = 30
IDLE_TIMEOUT_S = 7200
GPU_IDLE_COMMAND = ("nvidia-smi", "--query-compute-apps=pid", "--format=csv,noheader")


def _utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def sha256_file(path) -> str:
    digest = hashlib.sha256()
    with open(path, "rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _run(cmd):
    return subprocess.run(cmd, text=True, stdout=subprocess.PIPE,
                          stderr=subprocess.STDOUT, check=False)


def sunshine_pids() -> set[int]:
    out = set()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit():
            continue
        try:
            cmd = (entry / "cmdline").read_bytes().split(b"\x00")[0].decode()
        except (OSError, UnicodeDecodeError):
            continue
        if cmd.endswith("sunshine"):
            out.add(int(entry.name))
    return out


def foreign_compute_pids(query_output: str, allowed: set[int]) -> list[int]:
    foreign = []
    for line in query_output.splitlines():
        v = line.strip()
        if not v or v.lower() in {"n/a", "[n/a]", "no running processes found"}:
            continue
        if v.isdigit() and int(v) not in allowed:
            foreign.append(int(v))
    return foreign


def wait_idle(allowed: set[int]) -> str:
    deadline = time.monotonic() + IDLE_TIMEOUT_S
    while True:
        q = _run(GPU_IDLE_COMMAND)
        if q.returncode != 0:
            raise RuntimeError(f"nvidia-smi idle query failed:\n{q.stdout}")
        foreign = foreign_compute_pids(q.stdout, allowed)
        if not foreign:
            return q.stdout
        if time.monotonic() > deadline:
            raise RuntimeError(f"GPU never became idle; foreign PIDs: {foreign}")
        print(f"  waiting for GPU idle (foreign: {foreign})", flush=True)
        time.sleep(IDLE_POLL_S)


def measure_engine(engine_path: str, imgsz: int, warmup_iters: int,
                   measure_iters: int, trt_lib: str | None):
    """Returns list of per-iteration latencies (ms)."""
    if trt_lib:
        os.environ["LD_LIBRARY_PATH"] = trt_lib + os.pathsep + os.environ.get("LD_LIBRARY_PATH", "")
    import numpy as np
    import tensorrt as trt
    try:
        import cuda.bindings.runtime as cudart
    except ImportError:
        import cuda.cudart as cudart

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as fh, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(fh.read())
    if engine is None:
        raise RuntimeError("deserialize failed")
    context = engine.create_execution_context()
    names = [engine.get_tensor_name(i) for i in range(engine.num_io_tensors)]
    inputs = [n for n in names if engine.get_tensor_mode(n) == trt.TensorIOMode.INPUT]
    if len(inputs) != 1:
        raise RuntimeError(f"expected 1 input, got {inputs}")
    input_name = inputs[0]
    context.set_input_shape(input_name, (1, 3, imgsz, imgsz))
    device, host = {}, {}
    for name in names:
        shape = tuple(context.get_tensor_shape(name))
        dtype = trt.nptype(engine.get_tensor_dtype(name))
        host[name] = np.zeros(shape, dtype=dtype)
        err, device[name] = cudart.cudaMalloc(host[name].nbytes)
        if err != cudart.cudaError_t.cudaSuccess:
            raise RuntimeError(f"cudaMalloc failed: {name}")
        context.set_tensor_address(name, device[name])
    err, stream = cudart.cudaStreamCreate()
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError("cudaStreamCreate failed")
    err, = cudart.cudaMemcpyAsync(device[input_name], host[input_name].ctypes.data,
                                   host[input_name].nbytes,
                                   cudart.cudaMemcpyKind.cudaMemcpyHostToDevice, stream)
    if err != cudart.cudaError_t.cudaSuccess:
        raise RuntimeError("H2D failed")
    err, ev_start = cudart.cudaEventCreate()
    err2, ev_end = cudart.cudaEventCreate()
    lat = []
    try:
        for _ in range(warmup_iters):
            if not context.execute_async_v3(stream_handle=stream):
                raise RuntimeError("execute_async_v3 false")
        cudart.cudaStreamSynchronize(stream)
        for _ in range(measure_iters):
            cudart.cudaEventRecord(ev_start, stream)
            if not context.execute_async_v3(stream_handle=stream):
                raise RuntimeError("execute_async_v3 false")
            cudart.cudaEventRecord(ev_end, stream)
            cudart.cudaStreamSynchronize(stream)
            err, ms = cudart.cudaEventElapsedTime(ev_start, ev_end)
            if err != cudart.cudaError_t.cudaSuccess:
                raise RuntimeError("cudaEventElapsedTime failed")
            lat.append(float(ms))
    finally:
        cudart.cudaEventDestroy(ev_start)
        cudart.cudaEventDestroy(ev_end)
        cudart.cudaStreamDestroy(stream)
        for name in list(device):
            cudart.cudaFree(device[name])
    return lat


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    config = json.loads((root / args.config).read_text())
    out_dir = root / config["output_dir"]
    reps = int(config.get("repetitions", 3))
    warmup = int(config.get("warmup_iters", 200))
    iters = int(config.get("measure_iters", 500))
    trt_lib = config.get("trt_lib")

    for eng in config["engines"]:
        manifest_path = root / eng["manifest"]
        manifest = json.loads(manifest_path.read_text())
        for rep in range(1, reps + 1):
            stem = f"{eng['name']}__rep-{rep:02d}"
            rec_path = out_dir / f"{stem}.json"
            if rec_path.exists():
                print(f"SKIP {stem}", flush=True)
                continue
            eng_path = Path(manifest["engine"])
            assert eng_path.is_file(), eng_path
            assert sha256_file(eng_path) == manifest["engine_sha256"], "engine sha"
            allowed = sunshine_pids() | {os.getpid()}
            idle_out = wait_idle(allowed)
            started = _utc_now()
            samples = measure_engine(str(eng_path), int(manifest["imgsz"]),
                                     warmup, iters, trt_lib)
            ended = _utc_now()
            record = {
                "schema_version": 1, "attempt": config["attempt"],
                "condition_id": eng["name"], "repetition": rep,
                "run_id": uuid.uuid4().hex,
                "model": manifest.get("model"), "dataset": manifest.get("dataset"),
                "precision": manifest.get("precision"), "imgsz": manifest["imgsz"],
                "engine": str(eng_path), "engine_sha256": manifest["engine_sha256"],
                "engine_bytes": eng_path.stat().st_size,
                "engine_manifest": str(manifest_path),
                "engine_manifest_sha256": sha256_file(manifest_path),
                "build_policy": manifest.get("build_policy", {}),
                "timer": "cudaEventElapsedTime around execute_async_v3, batch=1",
                "warmup_iters": warmup, "measure_iters": iters,
                "gpu_idle_allow_pids": sorted(allowed),
                "gpu_idle_query_output": idle_out,
                "started_at_utc": started, "ended_at_utc": ended,
                "latency_median_ms": statistics.median(samples),
                "latency_mean_ms": statistics.fmean(samples),
                "latency_iqr_ms": (statistics.quantiles(samples, n=4)[2]
                                   - statistics.quantiles(samples, n=4)[0]),
                "latency_p95_ms": statistics.quantiles(samples, n=20)[18],
                "n_samples": len(samples),
                "samples_ms_sha256": hashlib.sha256(
                    json.dumps(samples).encode()).hexdigest(),
                "samples_ms": samples,
            }
            rec_path.parent.mkdir(parents=True, exist_ok=True)
            rec_path.write_text(json.dumps(record, indent=2) + "\n")
            print(f"DONE {stem} median={record['latency_median_ms']:.3f}ms "
                  f"iqr={record['latency_iqr_ms']:.3f}ms", flush=True)

    # summary
    by = {}
    for rec_path in sorted(out_dir.glob("*__rep-*.json")):
        r = json.loads(rec_path.read_text())
        by.setdefault(r["condition_id"], []).append(r["latency_median_ms"])
    summary = {c: {"reps": len(v), "median_ms": statistics.median(v),
                   "min_ms": min(v), "max_ms": max(v)}
               for c, v in by.items()}
    report = {"schema_version": 1, "attempt": config["attempt"],
              "created_at_utc": _utc_now(), "config_sha256": sha256_file(root / args.config),
              "summary": summary}
    rpt = root / config["report"]
    rpt.parent.mkdir(parents=True, exist_ok=True)
    rpt.write_text(json.dumps(report, indent=2) + "\n")
    print(f"wrote {rpt}", flush=True)


if __name__ == "__main__":
    main()
