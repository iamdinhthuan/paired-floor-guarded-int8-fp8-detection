#!/usr/bin/env python3
"""Build a provenance-complete COCO YOLO11 TensorRT ladder with train-only calibration."""

from __future__ import annotations

import argparse
import hashlib
import json
import random
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from defer_yolo_ladder_queue import stage
from pilot_registry import canonical_hash
from topic_c.manifest import sha256_file


def whole_file_complete(path: Path) -> bool:
    marker = path.with_suffix(path.suffix + ".complete")
    return path.is_file() and marker.is_file() and marker.read_text().strip() == sha256_file(path)


def create_calibration(root: Path, config: dict) -> Path:
    data = config["coco"]
    output = root / data["calibration_list"]
    marker = output.with_suffix(output.suffix + ".complete")
    if output.exists():
        document = json.loads(output.read_text())
        if (
            document.get("calibration_sha256")
            != canonical_hash(document, "calibration_sha256")
            or marker.read_text().strip() != document["calibration_sha256"]
            or len(document.get("records", [])) != config["calibration_images"]
        ):
            raise SystemExit("COCO P0 REFUSED: existing calibration registry is invalid")
        return output
    dataset_root = Path(data["calibration_dataset_root"]).resolve()
    train_root = dataset_root / data["calibration_train_dir"]
    images = sorted(
        path
        for path in train_root.iterdir()
        if path.is_file() and path.suffix.lower() in {".jpg", ".jpeg", ".png"}
    )
    n_images = int(config["calibration_images"])
    if len(images) < n_images:
        raise SystemExit("COCO P0 REFUSED: insufficient train calibration images")
    chosen = random.Random(int(config["seed"])).sample(images, n_images)
    chosen.sort(key=lambda path: path.relative_to(dataset_root).as_posix())
    document = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "coco",
        "split": "train",
        "selection": "uniform_without_replacement_from_images_only",
        "n_images": n_images,
        "seed": int(config["seed"]),
        "dataset_root": str(dataset_root),
        "records": [
            {
                "source_relpath": path.relative_to(dataset_root).as_posix(),
                "sha256": sha256_file(path),
            }
            for path in chosen
        ],
    }
    document["calibration_sha256"] = canonical_hash(document, "calibration_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n")
    marker.write_text(document["calibration_sha256"] + "\n")
    print(f"COCO P0 CALIBRATION COMPLETE images={n_images} sha256={document['calibration_sha256']}")
    return output


def create_pretrained_registry(root: Path, model: str) -> Path:
    output = root / "manifests" / "training" / f"coco_{model}_train_v1.json"
    if whole_file_complete(output):
        return output
    if output.exists() or output.with_suffix(output.suffix + ".complete").exists():
        raise SystemExit(f"COCO P0 REFUSED: partial pretrained registry: {output}")
    checkpoint = root / f"{model}.pt"
    if not checkpoint.is_file():
        raise SystemExit(f"COCO P0 REFUSED: pretrained checkpoint missing: {checkpoint}")
    document = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "dataset": "coco",
        "model": model,
        "source": "official Ultralytics pretrained checkpoint; no COCO retraining",
        "best_weights": str(checkpoint.resolve()),
        "best_weights_sha256": sha256_file(checkpoint),
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n")
    output.with_suffix(output.suffix + ".complete").write_text(sha256_file(output) + "\n")
    return output


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    if config.get("models") != ["yolo11n", "yolo11m", "yolo11x"]:
        raise SystemExit("COCO P0 REFUSED: unexpected model ladder")
    calibration = create_calibration(root, config)
    records = []
    stage_config = {"trt_root": config["trt_root"], "workspace": config["workspace"]}
    for model in config["models"]:
        create_pretrained_registry(root, model)
        for precision in config["precisions"]:
            onnx_registry, engine_registry = stage(
                root, stage_config, "coco", model, precision, calibration, int(config["coco"]["imgsz"])
            )
            records.append(
                {
                    "model": model,
                    "precision": precision,
                    "onnx_registry": str(onnx_registry),
                    "onnx_registry_sha256": sha256_file(onnx_registry),
                    "engine_registry": str(engine_registry),
                    "engine_registry_sha256": sha256_file(engine_registry),
                }
            )
    combined = root / "manifests" / "engines" / "coco_yolo11_nmx_ladder_v1.json"
    if not whole_file_complete(combined):
        if combined.exists():
            raise SystemExit("COCO P0 REFUSED: partial combined engine registry")
        subprocess.run(
            [
                sys.executable,
                str(root / "src" / "freeze_yolo_engine_registry.py"),
                "--project-root",
                str(root),
                "--dataset",
                "coco",
                "--out",
                str(combined),
            ],
            check=True,
        )
    report = root / "outputs" / "reports" / "coco_uniform_ladder_p0_v1.json"
    if report.exists():
        old = json.loads(report.read_text())
        if old.get("report_sha256") != canonical_hash(old, "report_sha256"):
            raise SystemExit("COCO P0 REFUSED: invalid existing ladder report")
        print(f"COCO P0 LADDER RESUME COMPLETE artifacts={len(records)}")
        return
    document = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "calibration_list": str(calibration),
        "calibration_sha256": json.loads(calibration.read_text())["calibration_sha256"],
        "combined_engine_registry": str(combined),
        "combined_engine_registry_sha256": sha256_file(combined),
        "artifacts": records,
    }
    document["report_sha256"] = canonical_hash(document, "report_sha256")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(document, indent=2) + "\n")
    print(f"COCO P0 LADDER COMPLETE artifacts={len(records)} report={report}")


if __name__ == "__main__":
    main()
