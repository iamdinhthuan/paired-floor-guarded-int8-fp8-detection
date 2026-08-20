#!/usr/bin/env python3
"""Fail-closed AP parity gate for four clean YOLO reference backends."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

from topic_c.manifest import sha256_file


LABELS = ("pytorch", "onnxruntime", "trt-fp32", "trt-fp16")


def verify_reference_metrics(
    pairs: dict[str, tuple[Path, Path]], *, source_tolerance: float, fp16_tolerance: float
) -> dict:
    errors: list[str] = []
    if set(pairs) != set(LABELS):
        return {
            "pass": False,
            "errors": ["exact four-backend parity grid is required"],
            "source_tolerance": source_tolerance,
            "fp16_tolerance": fp16_tolerance,
        }
    if not 0 <= source_tolerance <= 0.01 or not 0 <= fp16_tolerance <= 0.02:
        raise ValueError("parity tolerances are outside the guarded range")
    metrics: dict[str, dict] = {}
    runs: dict[str, dict] = {}
    artifacts: dict[str, dict] = {}
    for label in LABELS:
        metric_path, run_path = (Path(value).resolve() for value in pairs[label])
        try:
            metric = json.loads(metric_path.read_text(encoding="utf-8"))
            run = json.loads(run_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError):
            errors.append(f"{label}: metric/run record is unreadable")
            continue
        if metric.get("run_record_sha256") != sha256_file(run_path):
            errors.append(f"{label}: metric does not bind run-record bytes")
        for field in (
            "condition_id",
            "dataset",
            "split",
            "model",
            "corruption",
            "severity",
            "n_images",
            "input_manifest_sha256",
            "input_image_ids_sha256",
            "prediction_sha256",
        ):
            if metric.get(field) != run.get(field):
                errors.append(f"{label}: metric/run {field} mismatch")
        expected_backend = label if label in {"pytorch", "onnxruntime"} else None
        expected_precision = {
            "pytorch": "fp32-reference",
            "onnxruntime": "fp32-reference",
            "trt-fp32": "fp32",
            "trt-fp16": "fp16",
        }[label]
        if (expected_backend is not None and run.get("backend") != expected_backend) or run.get(
            "precision"
        ) != expected_precision:
            errors.append(f"{label}: backend identity mismatch")
        ap = metric.get("stats", {}).get("AP")
        if not isinstance(ap, (int, float)) or not math.isfinite(ap) or not 0 <= ap <= 1:
            errors.append(f"{label}: AP is not a finite native fraction")
        metrics[label], runs[label] = metric, run
        artifacts[label] = {
            "metric": str(metric_path),
            "metric_sha256": sha256_file(metric_path),
            "run_record": str(run_path),
            "run_record_sha256": sha256_file(run_path),
        }
    if len(metrics) != len(LABELS):
        return {
            "pass": False,
            "errors": errors,
            "source_tolerance": source_tolerance,
            "fp16_tolerance": fp16_tolerance,
            "artifacts": artifacts,
        }

    for field in (
        "dataset",
        "split",
        "model",
        "corruption",
        "severity",
        "n_images",
        "input_manifest_sha256",
        "input_image_ids_sha256",
    ):
        values = {metric.get(field) for metric in metrics.values()}
        if len(values) != 1:
            errors.append(f"four-backend {field} mismatch")
    annotations = {run.get("annotation_sha256") for run in runs.values()}
    if len(annotations) != 1:
        errors.append("four-backend annotation_sha256 mismatch")

    ap = {label: float(metrics[label].get("stats", {}).get("AP", math.nan)) for label in LABELS}
    gaps = {
        "pytorch_to_onnxruntime": abs(ap["pytorch"] - ap["onnxruntime"]),
        "pytorch_to_trt_fp32": abs(ap["pytorch"] - ap["trt-fp32"]),
        "trt_fp32_to_trt_fp16": abs(ap["trt-fp32"] - ap["trt-fp16"]),
    }
    if gaps["pytorch_to_onnxruntime"] > source_tolerance:
        errors.append(
            "PyTorch--ONNX Runtime AP gap "
            f"{gaps['pytorch_to_onnxruntime']:.8f} exceeds {source_tolerance:.8f}"
        )
    if gaps["pytorch_to_trt_fp32"] > source_tolerance:
        errors.append(
            "PyTorch--TensorRT FP32 AP gap "
            f"{gaps['pytorch_to_trt_fp32']:.8f} exceeds {source_tolerance:.8f}"
        )
    if gaps["trt_fp32_to_trt_fp16"] > fp16_tolerance:
        errors.append(
            "TensorRT FP32--FP16 AP gap "
            f"{gaps['trt_fp32_to_trt_fp16']:.8f} exceeds {fp16_tolerance:.8f}"
        )
    first = metrics["pytorch"]
    return {
        "schema_version": 1,
        "dataset": first.get("dataset"),
        "split": first.get("split"),
        "model": first.get("model"),
        "n_images": first.get("n_images"),
        "input_manifest_sha256": first.get("input_manifest_sha256"),
        "input_image_ids_sha256": first.get("input_image_ids_sha256"),
        "source_tolerance": source_tolerance,
        "fp16_tolerance": fp16_tolerance,
        "ap": ap,
        "absolute_ap_gaps": gaps,
        "artifacts": artifacts,
        "pass": not errors,
        "errors": errors,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    for label in LABELS:
        option = label.replace("-", "_")
        parser.add_argument(f"--{label}-metric", dest=f"{option}_metric", required=True)
        parser.add_argument(f"--{label}-run", dest=f"{option}_run", required=True)
    parser.add_argument("--source-tolerance", type=float, required=True)
    parser.add_argument("--fp16-tolerance", type=float, required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    output = Path(args.out).resolve()
    marker = output.with_suffix(output.suffix + ".complete")
    if output.exists() or marker.exists():
        raise SystemExit(f"REFERENCE PARITY REFUSED: output already exists: {output}")
    pairs = {
        label: (
            Path(getattr(args, f"{label.replace('-', '_')}_metric")),
            Path(getattr(args, f"{label.replace('-', '_')}_run")),
        )
        for label in LABELS
    }
    report = verify_reference_metrics(
        pairs,
        source_tolerance=args.source_tolerance,
        fp16_tolerance=args.fp16_tolerance,
    )
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    marker.write_text(sha256_file(output) + "\n", encoding="utf-8")
    print("REFERENCE PARITY PASS" if report["pass"] else "REFERENCE PARITY FAIL")
    if not report["pass"]:
        print("\n".join(f"- {error}" for error in report["errors"]))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
