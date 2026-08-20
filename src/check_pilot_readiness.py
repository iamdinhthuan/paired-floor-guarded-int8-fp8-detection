#!/usr/bin/env python3
"""Read-only fail-closed check before any proposed pilot execution."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pilot_registry import canonical_hash
from topic_c.manifest import read_manifest, sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--project-root", required=True)
    args = parser.parse_args()
    root = Path(args.project_root)
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != canonical_hash(plan, "plan_sha256"):
        raise SystemExit("PILOT NOT READY: plan hash mismatch")
    missing, mismatched, manifest_mismatches, condition_id_mismatches = [], [], [], []
    for run in plan["runs"]:
        engine = Path(run["engine_path"])
        if not engine.is_file() or sha256_file(engine) != run["engine_sha256"]:
            mismatched.append(f"engine: {run['model']} {run['precision']}")
        manifest = root / run["input_manifest_path"]
        marker = manifest.with_suffix(manifest.suffix + ".complete")
        if not manifest.is_file() or not marker.is_file():
            missing.append(run["input_manifest_path"])
            continue
        try:
            document = read_manifest(manifest)
        except ValueError as exc:
            manifest_mismatches.append(f"invalid: {run['input_manifest_path']}: {exc}")
            continue
        if marker.read_text(encoding="utf-8").strip() != document["manifest_sha256"]:
            manifest_mismatches.append(f"marker: {run['input_manifest_path']}")
        if (expected := run.get("input_manifest_sha256")) and expected != document["manifest_sha256"]:
            manifest_mismatches.append(f"frozen hash: {run['input_manifest_path']}")
        if expected and run.get("condition_id_template"):
            expected_id = run["condition_id_template"].replace("{manifest8}", document["manifest_sha256"][:8])
            if run.get("condition_id") != expected_id:
                condition_id_mismatches.append(run["input_manifest_path"])
    missing = sorted(set(missing))
    manifest_mismatches = sorted(set(manifest_mismatches))
    condition_id_mismatches = sorted(set(condition_id_mismatches))
    status = {"ready": not missing and not mismatched and not manifest_mismatches and not condition_id_mismatches, "runs": len(plan["runs"]),
              "missing_input_manifests": missing, "engine_hash_mismatches": sorted(set(mismatched)),
              "manifest_hash_mismatches": manifest_mismatches,
              "condition_id_mismatches": condition_id_mismatches}
    print(json.dumps(status, indent=2))
    if not status["ready"]:
        raise SystemExit(2)


if __name__ == "__main__":
    main()
