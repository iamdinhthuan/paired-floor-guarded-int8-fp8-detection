#!/usr/bin/env python3
"""Create a non-executable, hash-bound 117-condition COCO pilot plan."""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pilot_registry import canonical_hash


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--matrix", required=True)
    parser.add_argument("--engine-registry", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit(f"refusing to overwrite pilot plan: {output}")
    matrix = json.loads(Path(args.matrix).read_text(encoding="utf-8"))
    registry = json.loads(Path(args.engine_registry).read_text(encoding="utf-8"))
    if registry.get("registry_sha256") != canonical_hash(registry, "registry_sha256"):
        raise SystemExit("engine registry hash mismatch")
    runs = []
    for model in matrix["models"]:
        for precision in matrix["precisions"]:
            engine = registry["engines"][model][precision]
            calibrator = "entropy" if precision == "int8-entropy" else "none"
            # ``int8-entropy`` is the internal engine-ladder label.  The
            # immutable condition-ID contract stores it as precision=int8 and
            # calibrator=entropy, i.e. ``int8-entropy`` exactly once.
            condition_precision = "int8" if precision == "int8-entropy" else precision
            for condition in matrix["conditions"]:
                corruption, severity = condition["corruption"], condition["severity"]
                manifest = ("manifests/images/coco_val2017_full_clean.json" if corruption == "clean"
                            else f"manifests/images/coco_val2017_full_{corruption}_s{severity}.json")
                runs.append({
                    "dataset": matrix["dataset"], "split": matrix["split"], "model": model,
                    "precision": precision, "calibrator": calibrator,
                    "corruption": corruption, "severity": severity,
                    "engine_path": engine["path"], "engine_sha256": engine["sha256"],
                    "calibration_sha256": engine["calibration_sha256"],
                    "input_manifest_path": manifest,
                    "condition_id_template": f"coco_val2017__{model}__{condition_precision}-{calibrator}__{corruption}-s{severity}__{engine['sha256'][:8]}__{{manifest8}}",
                })
    if len(runs) != matrix["expected_runs"] or len({run["condition_id_template"] for run in runs}) != len(runs):
        raise SystemExit("pilot matrix does not contain the required unique 117 runs")
    document = {"schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(),
                "execution_policy": matrix["execution_policy"], "engine_registry_sha256": registry["registry_sha256"],
                "matrix_sha256": canonical_hash(matrix, "_none_"), "runs": runs}
    document["plan_sha256"] = canonical_hash(document, "plan_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"PILOT PLAN VALID runs={len(runs)} sha256={document['plan_sha256']}")


if __name__ == "__main__":
    main()
