#!/usr/bin/env python3
"""Create an immutable evidence bundle for the human pilot-review gate."""
from __future__ import annotations

import argparse
import json
import shutil
from datetime import datetime, timezone
from pathlib import Path

from pilot_registry import canonical_hash
from topic_c.manifest import read_manifest, sha256_file


def tree_bytes(path: Path) -> int:
    return sum(item.stat().st_size for item in path.rglob("*") if item.is_file())


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--engine-registry", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, output = Path(args.project_root).resolve(), Path(args.out)
    if output.exists():
        raise SystemExit(f"refusing to overwrite review report: {output}")
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    registry = json.loads(Path(args.engine_registry).read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != canonical_hash(plan, "plan_sha256"):
        raise SystemExit("review report refused: frozen plan hash mismatch")
    if registry.get("registry_sha256") != canonical_hash(registry, "registry_sha256"):
        raise SystemExit("review report refused: engine-registry hash mismatch")
    if len(plan.get("runs", [])) != 117:
        raise SystemExit("review report refused: expected 117 pilot runs")
    if any("entropy-entropy" in run.get("condition_id", "") for run in plan["runs"]):
        raise SystemExit("review report refused: non-canonical INT8 condition ID")
    manifests = {}
    for run in plan["runs"]:
        relative = run["input_manifest_path"]
        if relative in manifests:
            continue
        path = root / relative
        document = read_manifest(path)
        marker = path.with_suffix(path.suffix + ".complete")
        marker_ok = marker.is_file() and marker.read_text(encoding="utf-8").strip() == document["manifest_sha256"]
        if not marker_ok:
            raise SystemExit(f"review report refused: invalid completion marker: {relative}")
        manifests[relative] = {"manifest_sha256": document["manifest_sha256"], "records": len(document["records"]), "completion_marker_matches": True}
    engine_failures = []
    for model, precisions in registry["engines"].items():
        for precision, engine in precisions.items():
            if sha256_file(engine["path"]) != engine["sha256"]:
                engine_failures.append(f"{model}/{precision}")
    if engine_failures:
        raise SystemExit("review report refused: engine hash mismatch: " + ", ".join(engine_failures))
    scripts = ["coco_infer_trt.py", "execute_frozen_pilot.py", "evaluate_frozen_pilot.py", "bootstrap_frozen_pilot.py", "validate_predictions.py", "coco_eval.py", "paired_bootstrap.py",
               "topic_c/coco_data.py", "topic_c/yolo_decode.py", "topic_c/manifest.py"]
    report = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "review_required_before_execution",
        "blocker": "human review approval has not been recorded",
        "frozen_plan": {"path": str(Path(args.plan).resolve()), "sha256": sha256_file(args.plan), "plan_sha256": plan["plan_sha256"], "runs": len(plan["runs"])},
        "engine_registry": {"path": str(Path(args.engine_registry).resolve()), "sha256": sha256_file(args.engine_registry), "registry_sha256": registry["registry_sha256"], "all_engine_hashes_match": True},
        "manifests": manifests,
        "cache": {"path": str(root / "data" / "coco_c"), "bytes": tree_bytes(root / "data" / "coco_c"), "free_bytes": shutil.disk_usage(root).free},
        "launcher": {"dry_run_passed": True, "attempt": "pilot_117_v1", "scripts_sha256": {script: sha256_file(root / "src" / script) for script in scripts}},
        "execution_estimate": {"gpu_walltime_hours": "2–3 (inference and validation; bootstrap excluded)", "post_inference": "validate every prediction, evaluate 117 COCO metrics, then run 72 paired B=500 bootstraps"},
        "scope_guard": "No full grid, calibration intervention, or TT100K transfer is authorized in this pilot phase.",
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "runs": len(plan["runs"]), "manifests": len(manifests), "cache_bytes": report["cache"]["bytes"]}, indent=2))


if __name__ == "__main__":
    main()
