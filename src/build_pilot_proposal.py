#!/usr/bin/env python3
"""Write the review-only pilot proposal, including measured smoke scaling."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from pilot_registry import canonical_hash
from topic_c.manifest import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--engine-registry", required=True)
    parser.add_argument("--smoke-report", required=True)
    parser.add_argument("--smoke-cache", required=True)
    parser.add_argument("--smoke-predictions", required=True)
    parser.add_argument("--parity-report", action="append", default=[])
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit(f"refusing to overwrite pilot proposal: {output}")
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    registry = json.loads(Path(args.engine_registry).read_text(encoding="utf-8"))
    smoke = json.loads(Path(args.smoke_report).read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != canonical_hash(plan, "plan_sha256"):
        raise SystemExit("pilot plan hash mismatch")
    if registry.get("registry_sha256") != canonical_hash(registry, "registry_sha256"):
        raise SystemExit("engine registry hash mismatch")
    parity = []
    for raw_path in args.parity_report:
        path = Path(raw_path)
        item = json.loads(path.read_text(encoding="utf-8"))
        if not item.get("pass"):
            raise SystemExit(f"FP16 parity report is not a pass: {path}")
        parity.append({"path": str(path.resolve()), "sha256": sha256_file(path),
                       "absolute_ap_gap": item["absolute_ap_gap"], "tolerance": item["tolerance"]})
    cache_bytes = sum(path.stat().st_size for path in Path(args.smoke_cache).rglob("*") if path.is_file())
    prediction_bytes = sum(path.stat().st_size for path in Path(args.smoke_predictions).rglob("*.json") if path.is_file())
    # Smoke cache is 6 corrupt conditions × 300 images; pilot needs 12 × 5000.
    cache_scale = (12 * 5000) / (6 * 300)
    proposal = {
        "schema_version": 1,
        "status": "proposal_only; do_not_execute_without_human_review",
        "plan": {"path": str(Path(args.plan).resolve()), "sha256": sha256_file(args.plan), "plan_sha256": plan["plan_sha256"], "runs": len(plan["runs"])},
        "engine_registry": {"path": str(Path(args.engine_registry).resolve()), "sha256": sha256_file(args.engine_registry), "registry_sha256": registry["registry_sha256"]},
        "smoke_report": {"path": str(Path(args.smoke_report).resolve()), "sha256": sha256_file(args.smoke_report)},
        "fp16_parity_gates": parity,
        "estimates": {
            "inference_seconds_naive_smoke_scaling": smoke["runtime"]["mean_inference_seconds_per_300_image_run"] * (5000 / 300) * len(plan["runs"]),
            "recommended_gpu_walltime_hours": "2–3 (includes slower x-rung, validation, and operational overhead; excludes B=500 CPU bootstrap)",
            "pilot_corruption_cache_bytes": int(cache_bytes * cache_scale),
            "pilot_prediction_bytes": int(prediction_bytes * (5000 * len(plan["runs"])) / (300 * 28)),
            "recommended_free_disk_bytes": 30 * 1024 ** 3,
            "free_disk_bytes_at_proposal": shutil.disk_usage(output.resolve().anchor).free,
        },
        "required_before_execution": [
            "Generate and validate the 12 full 5,000-image corruption manifests with completion markers.",
            "Review the frozen engine registry and 117-run plan hashes.",
            "Use only the generated manifest hashes to instantiate final condition IDs.",
            "Do not begin calibration interventions or the full grid in this phase."
        ]
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(proposal, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"runs": proposal["plan"]["runs"], "status": proposal["status"], "recommended_free_disk_bytes": proposal["estimates"]["recommended_free_disk_bytes"]}, indent=2))


if __name__ == "__main__":
    main()
