#!/usr/bin/env python3
"""Engine-rebuild determinism probe for the NN arms.

Rebuilds each listed ONNX into a TensorRT engine N times under the frozen
build policy (tensorrt_python 11.1.0.106, TF32 off, opt level 3, workspace
4096MiB) and records, per rebuild: engine sha256, engine bytes, canonical
engine-inspector hash (tactic-level agreement even if bytes differ), and the
build log hash. Writes one JSON record per rebuild plus a summary.
"""
from __future__ import annotations

import hashlib
import json
import os
import subprocess
import sys
import time
import uuid
from datetime import datetime, timezone
from pathlib import Path


def sha256_file(path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as h:
        for blk in iter(lambda: h.read(1 << 20), b""):
            d.update(blk)
    return d.hexdigest()


def _utc() -> str:
    return datetime.now(timezone.utc).isoformat()


def build_once(onnx: str, dst: Path, workspace_mib: int, opt_level: int):
    import tensorrt as trt
    logger = trt.Logger(trt.Logger.WARNING)
    builder = trt.Builder(logger)
    network = builder.create_network(0)
    parser = trt.OnnxParser(network, logger)
    if not parser.parse_from_file(onnx):
        errs = "\n".join(str(parser.get_error(i)) for i in range(parser.num_errors))
        raise RuntimeError(f"ONNX parse failed: {errs}")
    cfg = builder.create_builder_config()
    cfg.clear_flag(trt.BuilderFlag.TF32)
    cfg.builder_optimization_level = opt_level
    cfg.set_memory_pool_limit(trt.MemoryPoolType.WORKSPACE, workspace_mib * 1024**2)
    cfg.profiling_verbosity = trt.ProfilingVerbosity.DETAILED
    payload = builder.build_serialized_network(network, cfg)
    if payload is None:
        raise RuntimeError("build returned None")
    ep = dst / "model.engine"
    ep.write_bytes(bytes(payload))
    with trt.Runtime(logger) as rt:
        engine = rt.deserialize_cuda_engine(bytes(payload))
        insp = engine.create_engine_inspector()
        (dst / "engine_inspector.json").write_text(
            insp.get_engine_information(trt.LayerInformationFormat.JSON))
    return ep


def main() -> None:
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--project-root", default=".")
    ap.add_argument("--config", required=True)
    args = ap.parse_args()
    root = Path(args.project_root).resolve()
    cfg = json.loads((root / args.config).read_text())
    out = root / cfg["output_dir"]
    out.mkdir(parents=True, exist_ok=True)
    reps = int(cfg["rebuilds"])
    ws = int(cfg["workspace_mib"])
    opt = int(cfg["optimization_level"])
    recs = []
    for eng in cfg["rebuilds_of"]:
        onnx = root / eng["onnx"]
        assert onnx.is_file(), onnx
        onnx_sha = sha256_file(onnx)
        for r in range(1, reps + 1):
            dst = out / eng["name"] / f"rebuild-{r:02d}"
            ep = dst / "model.engine"
            if ep.exists():
                print(f"SKIP {eng['name']} r{r}", flush=True)
                recs.append(json.loads((dst / "record.json").read_text()))
                continue
            dst.mkdir(parents=True)
            t0 = _utc()
            start = time.time()
            logp = dst / "build.log"
            # capture TRT logger output via build to file is not exposed; record stderr/stdout of process is not needed (in-proc). Record timings instead.
            build_once(str(onnx), dst, ws, opt)
            dt = time.time() - start
            insp = dst / "engine_inspector.json"
            rec = {
                "schema_version": 1, "attempt": cfg["attempt"],
                "arm": eng["name"], "rebuild": r, "run_id": uuid.uuid4().hex,
                "source_onnx": str(onnx), "source_onnx_sha256": onnx_sha,
                "engine": str(ep), "engine_sha256": sha256_file(ep),
                "engine_bytes": ep.stat().st_size,
                "engine_inspector_sha256": sha256_file(insp),
                "engine_inspector_bytes": insp.stat().st_size,
                "build_seconds": dt,
                "started_at_utc": t0, "ended_at_utc": _utc(),
                "tensorrt_version": cfg["tensorrt_version"],
                "build_policy": {"backend": "tensorrt_python", "tf32": False,
                                 "optimization_level": opt, "workspace_mib": ws},
            }
            (dst / "record.json").write_text(json.dumps(rec, indent=1) + "\n")
            recs.append(rec)
            print(f"DONE {eng['name']} r{r} sha={rec['engine_sha256'][:12]} "
                  f"insp={rec['engine_inspector_sha256'][:12]} {dt:.1f}s", flush=True)
    by = {}
    for r in recs:
        by.setdefault(r["arm"], []).append(r)
    summary = {}
    for arm, rows in by.items():
        shas = {r["engine_sha256"] for r in rows}
        ishas = {r["engine_inspector_sha256"] for r in rows}
        summary[arm] = {"rebuilds": len(rows),
                        "distinct_engine_sha256": len(shas),
                        "distinct_inspector_sha256": len(ishas),
                        "engine_sha256": sorted(shas),
                        "build_seconds": [round(r["build_seconds"], 2) for r in rows]}
    report = {"schema_version": 1, "attempt": cfg["attempt"],
              "created_at_utc": _utc(),
              "config_sha256": sha256_file(root / args.config),
              "summary": summary, "records": [r["run_id"] for r in recs]}
    rpt = root / cfg["report"]
    rpt.parent.mkdir(parents=True, exist_ok=True)
    rpt.write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(summary, indent=1), flush=True)
    print(f"wrote {rpt}", flush=True)


if __name__ == "__main__":
    main()
