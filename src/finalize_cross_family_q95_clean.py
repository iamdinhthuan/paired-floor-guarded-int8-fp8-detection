#!/usr/bin/env python3
"""Fail-closed completion report for the cross-family JPEG-95 controls."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
ATTEMPT = "cross_family_q95_clean_v1"
DATASETS = {"voc": "val", "kitti": "val", "tt100k": "test"}
MODELS = {"rtdetr_l": "rtdetr-l", "retinanet_r50_fpn_v2": "retinanet-r50-fpn-v2"}
PRECISIONS = ("fp32", "int8-entropy", "fp8")


def sha(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(document: dict, field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def load(path: Path) -> dict:
    if not path.is_file():
        raise RuntimeError(f"missing artifact: {path}")
    return json.loads(path.read_text())


def main() -> None:
    artifact_hashes: dict[str, str] = {}
    local_hashes: dict[str, str] = {}
    input_identity: dict[str, tuple[str, str]] = {}
    conditions: list[str] = []
    for dataset, split in DATASETS.items():
        for model_file, model in MODELS.items():
            for precision in PRECISIONS:
                condition = f"{dataset}_{split}__{model_file}__{precision}__q95-clean-s0"
                paths = {
                    "metric": ROOT / "outputs/metrics" / ATTEMPT / f"{condition}.json",
                    "run": ROOT / "manifests/runs" / ATTEMPT / f"{condition}.json",
                    "prediction": ROOT / "outputs/predictions" / ATTEMPT / f"{condition}.json",
                    "input": ROOT / "outputs/inputs" / ATTEMPT / f"{condition}.json",
                }
                documents = {kind: load(path) for kind, path in paths.items()}
                metric, run, input_record = documents["metric"], documents["run"], documents["input"]
                identity = (dataset, split, model, precision, "clean", 0)
                for document in (metric, run):
                    observed = tuple(document[key] for key in ("dataset", "split", "model", "precision", "corruption", "severity"))
                    if observed != identity or document["condition_id"] != condition:
                        raise RuntimeError(f"scientific identity mismatch: {condition}")
                if metric["run_record_sha256"] != sha(paths["run"]):
                    raise RuntimeError(f"run binding mismatch: {condition}")
                if metric["prediction_sha256"] != sha(paths["prediction"]) or run["prediction_sha256"] != sha(paths["prediction"]):
                    raise RuntimeError(f"prediction binding mismatch: {condition}")
                if input_record["input_manifest_sha256"] != metric["input_manifest_sha256"] or input_record["image_ids_sha256"] != metric["input_image_ids_sha256"]:
                    raise RuntimeError(f"input binding mismatch: {condition}")
                if run["input_manifest_sha256"] != metric["input_manifest_sha256"] or run["input_image_ids_sha256"] != metric["input_image_ids_sha256"]:
                    raise RuntimeError(f"run/input binding mismatch: {condition}")
                current_identity = (metric["input_manifest_sha256"], metric["input_image_ids_sha256"])
                previous = input_identity.setdefault(dataset, current_identity)
                if previous != current_identity:
                    raise RuntimeError(f"input universe differs within dataset: {dataset}")
                for kind, path in paths.items():
                    artifact_hashes[f"{kind}/{path.name}"] = sha(path)
                local_hashes[f"outputs/q95_metrics/{paths['metric'].name}"] = sha(paths["metric"])
                local_hashes[f"manifests/q95_runs/{paths['run'].name}"] = sha(paths["run"])
                conditions.append(condition)
    if len(conditions) != 18 or len(set(conditions)) != 18 or len(artifact_hashes) != 72:
        raise RuntimeError("JPEG-95 completion grid is not exactly 18 conditions / 72 artifacts")
    report = {
        "schema_version": 1,
        "attempt": ATTEMPT,
        "conditions": len(conditions),
        "linked_artifacts": len(artifact_hashes),
        "condition_ids": conditions,
        "input_identity": {key: {"input_manifest_sha256": value[0], "input_image_ids_sha256": value[1]} for key, value in input_identity.items()},
        "artifact_sha256": artifact_hashes,
        "local_artifact_sha256": local_hashes,
    }
    report["report_sha256"] = canonical_sha(report, "report_sha256")
    output = ROOT / "outputs/reports/cross_family_q95_clean_v1_complete.json"
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n")
    print(json.dumps({"status": "valid", "report": str(output), "report_sha256": report["report_sha256"]}))


if __name__ == "__main__":
    main()
