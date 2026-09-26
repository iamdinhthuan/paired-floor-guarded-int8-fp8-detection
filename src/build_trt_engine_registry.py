#!/usr/bin/env python3
"""CLI wrapper for the frozen TensorRT engine builder.

Builds one serialized engine from a completed ONNX registry into an
isolated destination directory and writes the hash-bound engine.json
registry plus its .complete marker.  Reuses run_nn_preprocessing_pilot's
audited builder so build policy stays identical across attempts.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from run_nn_preprocessing_pilot import build_engine

FROZEN_BUILD_POLICY = {
    "backend": "tensorrt_python",
    "required_version": "11.1.0.106",
    "workspace_mib": 4096,
    "optimization_level": 3,
    "tf32": False,
}


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--onnx-registry", type=Path, required=True)
    parser.add_argument("--precision", required=True)
    parser.add_argument("--destination", type=Path, required=True)
    parser.add_argument("--policy", type=Path, default=None,
                        help="optional JSON override; default is the frozen Phase-B policy")
    args = parser.parse_args()
    policy = FROZEN_BUILD_POLICY
    if args.policy is not None:
        policy = json.loads(args.policy.read_text())
        if policy.get("tf32") or policy.get("required_version") != "11.1.0.106":
            raise SystemExit("ENGINE BUILD REFUSED: policy must keep TF32 off on TensorRT 11.1.0.106")
    destination = args.destination.resolve()
    destination.mkdir(parents=True, exist_ok=False)
    registry = build_engine(args.onnx_registry.resolve(), args.precision, destination, policy)
    print(json.dumps({"engine_registry": str(registry)}, indent=2))


if __name__ == "__main__":
    main()
