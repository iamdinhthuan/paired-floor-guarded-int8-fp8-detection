#!/usr/bin/env python3
"""Create the immutable post-pilot review report from all frozen artifacts."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from bootstrap_frozen_pilot import bootstrap_path, expected_cells
from execute_frozen_pilot import paths
from pilot_registry import canonical_hash
from topic_c.manifest import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--attempt", default="pilot_117_v1")
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, out = Path(args.project_root).resolve(), Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite pilot report: {out}")
    plan = json.loads(Path(args.plan).read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != canonical_hash(plan, "plan_sha256") or len(plan.get("runs", [])) != 117:
        raise SystemExit("PILOT REPORT REFUSED: frozen plan is invalid")
    evaluation = root / "outputs" / "reports" / f"{args.attempt}_evaluation_complete.json"
    bootstrap_report = root / "outputs" / "reports" / f"{args.attempt}_bootstrap_complete.json"
    if not evaluation.is_file() or not bootstrap_report.is_file():
        raise SystemExit("PILOT REPORT REFUSED: evaluation and bootstrap completion reports are required")
    evaluation_doc = json.loads(evaluation.read_text(encoding="utf-8"))
    bootstrap_doc = json.loads(bootstrap_report.read_text(encoding="utf-8"))
    if evaluation_doc.get("plan_sha256") != plan["plan_sha256"] or bootstrap_doc.get("plan_sha256") != plan["plan_sha256"]:
        raise SystemExit("PILOT REPORT REFUSED: completion report plan hash mismatch")
    if bootstrap_doc.get("cells") != 72 or bootstrap_doc.get("n_boot") != 500 or not bootstrap_doc.get("all_bootstrap_input_hashes_validated"):
        raise SystemExit("PILOT REPORT REFUSED: bootstrap completion report is incomplete")
    rows = []
    runtime_by_model: dict[str, float] = {}
    for run in plan["runs"]:
        prediction, _, run_path = paths(root, args.attempt, run["condition_id"])
        metric_path = root / "outputs" / "metrics" / args.attempt / f"{run['condition_id']}.json"
        if not all(path.is_file() for path in (prediction, run_path, metric_path)):
            raise SystemExit(f"PILOT REPORT REFUSED: missing run artifact: {run['condition_id']}")
        record = json.loads(run_path.read_text(encoding="utf-8"))
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        if (record.get("condition_id") != run["condition_id"] or record.get("prediction_sha256") != sha256_file(prediction)
                or metric.get("condition_id") != run["condition_id"] or metric.get("prediction_sha256") != record["prediction_sha256"]
                or metric.get("input_manifest_sha256") != run["input_manifest_sha256"]):
            raise SystemExit(f"PILOT REPORT REFUSED: provenance mismatch: {run['condition_id']}")
        runtime_by_model[run["model"]] = runtime_by_model.get(run["model"], 0.0) + float(record["runtime_seconds"])
        rows.append({"condition_id": run["condition_id"], "model": run["model"], "precision": run["precision"],
                     "calibrator": run["calibrator"], "corruption": run["corruption"], "severity": run["severity"],
                     "runtime_seconds": record["runtime_seconds"], "prediction_sha256": record["prediction_sha256"],
                     "metric_sha256": sha256_file(metric_path), "input_manifest_sha256": run["input_manifest_sha256"],
                     "AP": metric["stats"]["AP"], "AP_small": metric["stats"]["AP_small"], "AP_medium": metric["stats"]["AP_medium"], "AP_large": metric["stats"]["AP_large"]})
    bootstrap_rows = []
    for cell in expected_cells(plan):
        path = bootstrap_path(root, args.attempt, cell["model"], cell["precision"], cell["corruption"], cell["severity"])
        if not path.is_file():
            raise SystemExit(f"PILOT REPORT REFUSED: missing bootstrap: {path.name}")
        result = json.loads(path.read_text(encoding="utf-8"))
        if result.get("n_boot") != 500 or result.get("quant_label") != cell["precision"]:
            raise SystemExit(f"PILOT REPORT REFUSED: malformed bootstrap: {path.name}")
        bootstrap_rows.append({"model": cell["model"], "precision": cell["precision"], "corruption": cell["corruption"],
                               "severity": cell["severity"], "sha256": sha256_file(path),
                               "psi_small_minus_large": result["point"]["psi_small_minus_large"],
                               "psi_ci95": result["ci95_psi_small_minus_large"], "excess": result["point"]["excess"]})
    usage = shutil.disk_usage(root)
    report = {"schema_version": 1, "phase": "COCO pilot 117", "status": "complete; reduced transfer extension is separately authorized by the project owner",
              "attempt": args.attempt, "frozen_plan_sha256": plan["plan_sha256"],
              "evaluation_report_sha256": sha256_file(evaluation), "bootstrap_report_sha256": sha256_file(bootstrap_report),
              "checks": {"runs": len(rows), "metrics": len(rows), "bootstrap_cells": len(bootstrap_rows),
                         "bootstrap_replicates": 500, "all_prediction_metric_manifest_provenance_matches": True,
                         "all_bootstrap_input_hashes_validated": True},
              "runtime": {"total_inference_seconds": sum(float(row["runtime_seconds"]) for row in rows),
                          "inference_seconds_by_model": runtime_by_model},
              "disk": {"free_bytes": usage.free, "total_bytes": usage.total},
              "runs": rows, "bootstrap": bootstrap_rows,
              "scope_guard": "Do not begin the full grid or calibration intervention without further human review. The separately frozen VOC/KITTI/TT100K reduced pilot is covered by explicit project-owner authorization."}
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["checks"], indent=2))


if __name__ == "__main__":
    main()
