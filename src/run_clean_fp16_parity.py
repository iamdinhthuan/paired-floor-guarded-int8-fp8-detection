#!/usr/bin/env python3
"""Run immutable clean FP32/FP16 inference, evaluation, and parity gates for one dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from topic_c.manifest import read_manifest, sha256_file, validate_manifest


def canonical_hash(document: dict, excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def complete(path: Path) -> bool:
    marker = path.with_suffix(path.suffix + ".complete")
    return path.is_file() and marker.is_file() and marker.read_text(encoding="utf-8").strip() == sha256_file(path)


def paths(root: Path, attempt: str, condition: str) -> tuple[Path, Path, Path, Path]:
    return (root / "outputs" / "predictions" / attempt / f"{condition}.json", root / "outputs" / "inputs" / attempt / f"{condition}.json",
            root / "manifests" / "runs" / attempt / f"{condition}.json", root / "outputs" / "metrics" / attempt / f"{condition}.json")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True); parser.add_argument("--config", required=True)
    parser.add_argument("--dataset", choices=("coco", "voc", "kitti", "tt100k"), required=True); parser.add_argument("--resume-verified", action="store_true")
    args = parser.parse_args()
    root, config = Path(args.project_root).resolve(), json.loads(Path(args.config).read_text(encoding="utf-8"))
    data = next((item for item in config.get("datasets", []) if item.get("dataset") == args.dataset), None)
    if not data or config.get("models") != ["yolo11n", "yolo11m", "yolo11x"]:
        raise SystemExit("FP16 PARITY REFUSED: frozen config is invalid")
    registry_path = root / "manifests" / "engines" / f"{args.dataset}_yolo11_nmx_ladder_v1.json"
    if not complete(registry_path):
        raise SystemExit("FP16 PARITY REFUSED: completed engine ladder registry is required")
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    if registry.get("registry_sha256") != canonical_hash(registry, "registry_sha256") or registry.get("dataset") != args.dataset:
        raise SystemExit("FP16 PARITY REFUSED: invalid engine ladder registry")
    annotations, clean_root = root / data["annotations"], Path(data["clean_root"])
    if not clean_root.is_absolute():
        clean_root = root / clean_root
    class_map = root / data["class_map"] if data.get("class_map") else None
    manifest_path = root / data["clean_manifest"]
    manifest = read_manifest(manifest_path)
    marker = manifest_path.with_suffix(manifest_path.suffix + ".complete")
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != manifest.get("manifest_sha256") or len(manifest["records"]) != data["expected_images"]:
        raise SystemExit("FP16 PARITY REFUSED: clean manifest is not complete")
    failures = validate_manifest(manifest, annotations, clean_root, clean_root, require_pixels_changed=False)
    if failures:
        raise SystemExit("FP16 PARITY REFUSED: clean manifest source-byte validation failed:\n" + "\n".join(failures[:20]))
    attempt, reports = f"parity_{args.dataset}_v1", []
    evaluator = root / "src" / ("tt100k_eval.py" if args.dataset == "tt100k" else "coco_eval.py")
    for model in config["models"]:
        metrics = {}
        for precision in ("fp32", "fp16"):
            engine = registry["engines"][model][precision]
            engine_path = Path(engine["path"])
            if not engine_path.is_file() or sha256_file(engine_path) != engine["sha256"]:
                raise SystemExit(f"FP16 PARITY REFUSED: engine hash mismatch: {model}/{precision}")
            condition = f"{args.dataset}_{data['split']}__{model}__{precision}-none__clean-s0__{engine['sha256'][:8]}__{manifest['manifest_sha256'][:8]}"
            prediction, input_record, run_record, metric = paths(root, attempt, condition)
            existing = [path.exists() for path in (prediction, input_record, run_record)]
            if any(existing) and not all(existing):
                raise SystemExit(f"FP16 PARITY REFUSED: partial inference output: {condition}")
            if not all(existing):
                command = [sys.executable, str(root / "src" / "coco_infer_trt.py"), "--engine", str(engine_path), "--annotations", str(annotations),
                           "--image-manifest", str(manifest_path), "--manifest-cache-root", str(clean_root), "--out", str(prediction),
                           "--input-record", str(input_record), "--run-record", str(run_record), "--condition-id", condition, "--dataset", args.dataset,
                           "--split", data["split"], "--imgsz", str(data["imgsz"]), "--model", model,
                           "--precision", precision, "--calibrator", "none", "--corruption", "clean", "--severity", "0"]
                if class_map:
                    command += ["--class-map", str(class_map)]
                subprocess.run(command, check=True)
                subprocess.run([sys.executable, str(root / "src" / "validate_predictions.py"), "--predictions", str(prediction),
                                "--input-record", str(input_record), "--annotations", str(annotations)], check=True)
            record = json.loads(run_record.read_text(encoding="utf-8"))
            inputs = json.loads(input_record.read_text(encoding="utf-8"))
            if (record.get("condition_id") != condition or record.get("prediction_sha256") != sha256_file(prediction)
                    or record.get("n_images") != data["expected_images"] or len(inputs.get("image_ids", [])) != data["expected_images"]):
                raise SystemExit(f"FP16 PARITY REFUSED: invalid inference provenance: {condition}")
            if metric.exists():
                if not args.resume_verified:
                    raise SystemExit(f"FP16 PARITY REFUSED: metric already exists: {metric}")
            else:
                subprocess.run([sys.executable, str(evaluator), "--annotations", str(annotations), "--predictions", str(prediction),
                                "--input-record", str(input_record), "--run-record", str(run_record), "--out", str(metric)], check=True)
            metric_data = json.loads(metric.read_text(encoding="utf-8"))
            if (metric_data.get("dataset") != args.dataset or metric_data.get("split") != data["split"]
                    or metric_data.get("condition_id") != condition or metric_data.get("prediction_sha256") != record["prediction_sha256"]
                    or metric_data.get("run_record_sha256") != sha256_file(run_record) or metric_data.get("n_images") != data["expected_images"]):
                raise SystemExit(f"FP16 PARITY REFUSED: metric provenance mismatch: {metric}")
            metrics[precision] = metric
        report = root / "outputs" / "reports" / f"{args.dataset}_{model}_fp16_parity_v1.json"
        if report.exists():
            if not args.resume_verified:
                raise SystemExit(f"FP16 PARITY REFUSED: report already exists: {report}")
            parity = json.loads(report.read_text(encoding="utf-8"))
            if not parity.get("pass"):
                raise SystemExit(f"FP16 PARITY REFUSED: pre-existing failed gate: {report}")
        else:
            subprocess.run([sys.executable, str(root / "src" / "verify_fp16_parity.py"), "--fp32", str(metrics["fp32"]), "--fp16", str(metrics["fp16"]),
                            "--out", str(report), "--tolerance", str(config["tolerance_ap"])], check=True)
        reports.append({"model": model, "report": str(report), "sha256": sha256_file(report)})
    output = root / "outputs" / "reports" / f"{args.dataset}_clean_fp16_parity_complete_v1.json"
    if output.exists():
        raise SystemExit(f"FP16 PARITY REFUSED: completion report already exists: {output}")
    document = {"schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": args.dataset,
                "attempt": attempt, "engine_registry_sha256": sha256_file(registry_path), "clean_manifest_sha256": manifest["manifest_sha256"], "reports": reports}
    document["parity_completion_sha256"] = canonical_hash(document, "parity_completion_sha256")
    output.parent.mkdir(parents=True, exist_ok=True); output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"FP16 PARITY COMPLETE dataset={args.dataset} models=3")


if __name__ == "__main__":
    main()
