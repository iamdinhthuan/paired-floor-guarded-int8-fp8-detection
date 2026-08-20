#!/usr/bin/env python3
"""Fail closed if a proposed pilot plan is not the predeclared 117-run matrix."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pilot_registry import canonical_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    args = parser.parse_args()
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != canonical_hash(plan, "plan_sha256"):
        raise SystemExit("PILOT PLAN INVALID: hash mismatch")
    runs = plan.get("runs", [])
    required = {"yolo11n", "yolo11m", "yolo11x"}
    if len(runs) != 117 or {run.get("model") for run in runs} != required:
        raise SystemExit("PILOT PLAN INVALID: expected 117 runs for YOLO11 n/m/x")
    if {run.get("precision") for run in runs} != {"fp32", "int8-entropy", "fp8"}:
        raise SystemExit("PILOT PLAN INVALID: precision ladder differs")
    if any("{manifest8}" not in run.get("condition_id_template", "") for run in runs):
        raise SystemExit("PILOT PLAN INVALID: condition ID does not await a frozen manifest hash")
    for run in runs:
        identifier = run["condition_id_template"]
        expected_precision = "int8" if run["precision"] == "int8-entropy" else run["precision"]
        expected_segment = f"__{expected_precision}-{run['calibrator']}__"
        if expected_segment not in identifier or "entropy-entropy" in identifier:
            raise SystemExit(f"PILOT PLAN INVALID: non-canonical condition ID: {identifier}")
    print(f"PILOT PLAN VALID runs={len(runs)} sha256={plan['plan_sha256']}")


if __name__ == "__main__":
    main()
