#!/usr/bin/env python3
"""Write a versioned, concise smoke-phase audit from explicit run artifacts."""
from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path

from topic_c.manifest import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-dir", required=True)
    parser.add_argument("--metric-dir", required=True)
    parser.add_argument("--bootstrap-dir", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit(f"refusing to overwrite smoke report: {output}")
    run_paths = sorted(Path(args.run_dir).glob("*.json"))
    metric_dir = Path(args.metric_dir)
    rows, calibration_status = [], set()
    for run_path in run_paths:
        run = json.loads(run_path.read_text(encoding="utf-8"))
        metric_path = metric_dir / run_path.name
        if not metric_path.is_file():
            raise SystemExit(f"missing metric for run record: {run_path.name}")
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        if metric["prediction_sha256"] != run["prediction_sha256"]:
            raise SystemExit(f"metric/prediction hash mismatch: {run_path.name}")
        calibration_status.add(run["calibration_provenance"])
        rows.append({"condition_id": run["condition_id"], "precision": run["precision"], "corruption": run["corruption"], "severity": run["severity"], "n_images": run["n_images"], "runtime_seconds": run["runtime_seconds"], "prediction_sha256": run["prediction_sha256"], "metric_sha256": sha256_file(metric_path), "AP": metric["stats"]["AP"], "AP_small": metric["stats"]["AP_small"], "AP_large": metric["stats"]["AP_large"], "input_manifest_sha256": run["input_manifest_sha256"]})
    boot_paths = sorted(Path(args.bootstrap_dir).glob("*.json"))
    bootstrap = []
    for path in boot_paths:
        item = json.loads(path.read_text(encoding="utf-8"))
        bootstrap.append({"file": path.name, "sha256": sha256_file(path), "quant_label": item["quant_label"], "n_boot": item["n_boot"], "psi": item["point"]["psi_small_minus_large"], "psi_ci95": item["ci95_psi_small_minus_large"]})
    usage = shutil.disk_usage(output.resolve().anchor)
    report = {"schema_version": 1, "phase": "COCO smoke rev2", "runs": rows, "bootstrap": bootstrap, "checks": {"run_records": len(rows), "metrics": len(rows), "bootstrap_artifacts": len(bootstrap), "all_prediction_hashes_match_metrics": True, "calibration_provenance_statuses": sorted(calibration_status)}, "runtime": {"total_inference_seconds": sum(row["runtime_seconds"] for row in rows), "mean_inference_seconds_per_300_image_run": sum(row["runtime_seconds"] for row in rows) / len(rows)}, "disk": {"free_bytes": usage.free, "total_bytes": usage.total}}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report["checks"], indent=2))


if __name__ == "__main__":
    main()
