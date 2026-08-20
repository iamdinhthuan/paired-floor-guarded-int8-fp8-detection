#!/usr/bin/env python3
"""Run one immutable YOLO11 training job for a newly acquired Topic C dataset."""
from __future__ import annotations

import argparse
import hashlib
import json
import os
from datetime import datetime, timezone
from pathlib import Path

import torch
import ultralytics
from ultralytics import RTDETR, YOLO
from ultralytics.utils import YAML

from topic_c.manifest import sha256_file


def canonical_hash(document: dict) -> str:
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def detector_class_name(model: str) -> str:
    if model.startswith("yolo"):
        return "YOLO"
    if model.startswith("rtdetr-"):
        return "RTDETR"
    raise SystemExit(f"TRAINING REFUSED: unsupported detector family: {model}")


def load_detector(weights: str | Path, model_name: str):
    detector_class = YOLO if detector_class_name(model_name) == "YOLO" else RTDETR
    return detector_class(str(weights))


def resolve_profile_batch(profile: dict, dataset: str) -> int:
    """Resolve an explicit per-dataset batch while preserving legacy profiles."""
    if "batch_by_dataset" in profile:
        mapping = profile["batch_by_dataset"]
        value = mapping.get(dataset) if isinstance(mapping, dict) else None
        if not isinstance(value, int) or value <= 0:
            raise SystemExit(f"TRAINING REFUSED: frozen batch is invalid for {dataset}")
        return value
    value = profile.get("batch")
    if not isinstance(value, int) or value == 0 or value < -1:
        raise SystemExit("TRAINING REFUSED: scalar batch is invalid")
    return value


def resolve_profile_imgsz(profile: dict, dataset: str) -> int:
    """Resolve dataset geometry without changing legacy scalar profiles."""
    mapping = profile.get("imgsz_by_dataset")
    value = mapping.get(dataset) if isinstance(mapping, dict) else profile.get("imgsz")
    if not isinstance(value, int) or value <= 0:
        raise SystemExit(f"TRAINING REFUSED: frozen image size is invalid for {dataset}")
    return value


def validate_training_data_provenance(
    *, dataset: str, data_yaml: Path, acquisition_registry: Path, split_registry: Path | None = None
) -> dict:
    """Validate either the historical acquisition YAML or a derived locked split."""
    data_yaml = data_yaml.resolve()
    acquisition_registry = acquisition_registry.resolve()
    completion = acquisition_registry.with_suffix(acquisition_registry.suffix + ".complete")
    if (
        not acquisition_registry.is_file()
        or not completion.is_file()
        or completion.read_text(encoding="utf-8").strip() != sha256_file(acquisition_registry)
    ):
        raise SystemExit("TRAINING REFUSED: acquisition registry is incomplete")
    acquired = json.loads(acquisition_registry.read_text(encoding="utf-8"))
    if acquired.get("dataset") != dataset:
        raise SystemExit("TRAINING REFUSED: acquisition registry dataset mismatch")
    if split_registry is None:
        if Path(acquired.get("vendored_yaml", "")).resolve() != data_yaml:
            raise SystemExit("TRAINING REFUSED: dataset YAML does not match acquisition registry")
        if acquired.get("vendored_yaml_sha256") != sha256_file(data_yaml):
            raise SystemExit("TRAINING REFUSED: dataset YAML hash mismatch")
        return {"mode": "historical_acquisition_yaml", "split_content_sha256": None}

    from freeze_confirmatory_splits import validate_split_bundle

    split_registry = split_registry.resolve()
    split = validate_split_bundle(split_registry)
    if split.get("dataset") != dataset or Path(split.get("yaml_path", "")).resolve() != data_yaml:
        raise SystemExit("TRAINING REFUSED: confirmatory split dataset/YAML mismatch")
    if split.get("yaml_sha256") != sha256_file(data_yaml):
        raise SystemExit("TRAINING REFUSED: confirmatory split YAML hash mismatch")
    if (
        Path(split.get("acquisition_registry", "")).resolve() != acquisition_registry
        or split.get("acquisition_registry_sha256") != sha256_file(acquisition_registry)
    ):
        raise SystemExit("TRAINING REFUSED: confirmatory split acquisition binding mismatch")
    return {
        "mode": split["protocol"],
        "split_registry": str(split_registry),
        "split_registry_sha256": sha256_file(split_registry),
        "split_content_sha256": split["content_sha256"],
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--profile", required=True)
    parser.add_argument("--dataset", choices=("voc", "kitti", "tt100k"), required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--data-yaml", required=True)
    parser.add_argument("--acquisition-registry", required=True)
    parser.add_argument("--split-registry", help="hash-complete prospectively locked split registry")
    parser.add_argument("--run-id", required=True)
    parser.add_argument("--registry-out", required=True)
    parser.add_argument("--resume-from", help="resume only from this run's verified weights/last.pt")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    profile_path = Path(args.profile).resolve()
    registry_path = Path(args.acquisition_registry).resolve()
    split_registry_path = Path(args.split_registry).resolve() if args.split_registry else None
    data_yaml = Path(args.data_yaml).resolve()
    output = root / "outputs" / "training" / args.dataset / args.run_id
    record_out = Path(args.registry_out).resolve()
    resume_path = Path(args.resume_from).resolve() if args.resume_from else None
    if record_out.exists():
        raise SystemExit("TRAINING REFUSED: training registry already exists")
    if output.exists():
        expected_resume = output / "weights" / "last.pt"
        if resume_path is None:
            raise SystemExit("TRAINING REFUSED: partial output requires an explicit resume checkpoint")
        if resume_path != expected_resume.resolve() or not resume_path.is_file():
            raise SystemExit("TRAINING REFUSED: resume checkpoint must be this run's weights/last.pt")
        prior_args = output / "args.yaml"
        if not prior_args.is_file():
            raise SystemExit("TRAINING REFUSED: partial output lacks args.yaml")
    elif resume_path is not None:
        raise SystemExit("TRAINING REFUSED: resume checkpoint requires an existing partial output")
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    if args.model not in profile.get("models", []):
        raise SystemExit("TRAINING REFUSED: model is not in frozen profile")
    if args.dataset not in {"voc", "kitti", "tt100k"}:
        raise SystemExit("TRAINING REFUSED: unsupported dataset")
    resolved_batch = resolve_profile_batch(profile, args.dataset)
    resolved_imgsz = resolve_profile_imgsz(profile, args.dataset)
    if resume_path is not None:
        previous = YAML.load(output / "args.yaml")
        expected = {
            "model": f"{args.model}.pt", "data": str(data_yaml), "epochs": profile["epochs"],
            "imgsz": resolved_imgsz, "workers": profile["workers"], "seed": profile["seed"],
            "deterministic": profile["deterministic"], "optimizer": profile["optimizer"],
            "pretrained": profile["pretrained"],
            "patience": profile["patience"], "close_mosaic": profile["close_mosaic"],
            "cache": profile["cache"], "amp": profile["amp"], "save_period": profile["save_period"],
            "batch": resolved_batch,
            "project": str(output.parent), "name": output.name,
        }
        mismatches = [key for key, value in expected.items() if str(previous.get(key)) != str(value)]
        if mismatches:
            raise SystemExit(f"TRAINING REFUSED: partial output args mismatch frozen profile: {','.join(mismatches)}")
    data_provenance = validate_training_data_provenance(
        dataset=args.dataset,
        data_yaml=data_yaml,
        acquisition_registry=registry_path,
        split_registry=split_registry_path,
    )
    data = YAML.load(data_yaml)
    for required in ("train", "val", "names", "path"):
        if required not in data:
            raise SystemExit(f"TRAINING REFUSED: required data YAML key absent: {required}")
    os.chdir(root)
    pretrained = f"{args.model}.pt"
    model = load_detector(pretrained, args.model)
    pretrained_path = Path(model.ckpt_path or pretrained).resolve()
    if not pretrained_path.is_file():
        raise SystemExit(f"TRAINING REFUSED: pretrained weights unavailable: {pretrained_path}")
    resume_sha256 = sha256_file(resume_path) if resume_path is not None else None
    if resume_path is not None:
        del model
        model = load_detector(resume_path, args.model)
        results = model.train(resume=str(resume_path))
    else:
        results = model.train(
            data=str(data_yaml), epochs=profile["epochs"], imgsz=resolved_imgsz, batch=resolved_batch,
            device=profile["device"], workers=profile["workers"], seed=profile["seed"],
            deterministic=profile["deterministic"], pretrained=profile["pretrained"], optimizer=profile["optimizer"],
            patience=profile["patience"], close_mosaic=profile["close_mosaic"], cache=profile["cache"], amp=profile["amp"],
            save_period=profile["save_period"], project=str(output.parent), name=output.name, exist_ok=False,
        )
    del results
    best, last, train_args = output / "weights" / "best.pt", output / "weights" / "last.pt", output / "args.yaml"
    if not all(path.is_file() for path in (best, last, train_args)):
        raise SystemExit("TRAINING REFUSED: Ultralytics job ended without best.pt, last.pt, and args.yaml")
    record = {
        "schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "run_id": args.run_id,
        "dataset": args.dataset, "model": args.model, "profile": str(profile_path), "profile_sha256": sha256_file(profile_path),
        "resolved_batch": resolved_batch, "resolved_imgsz": resolved_imgsz,
        "profile_canonical_sha256": canonical_hash(profile), "data_yaml": str(data_yaml), "data_yaml_sha256": sha256_file(data_yaml),
        "acquisition_registry": str(registry_path), "acquisition_registry_sha256": sha256_file(registry_path),
        "data_provenance": data_provenance,
        "pretrained_weights": str(pretrained_path), "pretrained_weights_sha256": sha256_file(pretrained_path),
        "resumed_from": str(resume_path) if resume_path is not None else None,
        "resumed_from_sha256": resume_sha256,
        "best_weights": str(best), "best_weights_sha256": sha256_file(best), "last_weights": str(last),
        "last_weights_sha256": sha256_file(last), "ultralytics_args": str(train_args), "ultralytics_args_sha256": sha256_file(train_args),
        "ultralytics_version": ultralytics.__version__, "torch_version": torch.__version__, "cuda_version": torch.version.cuda,
    }
    record_out.parent.mkdir(parents=True, exist_ok=True)
    record_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record_out.with_suffix(record_out.suffix + ".complete").write_text(sha256_file(record_out) + "\n", encoding="utf-8")
    print(json.dumps({"TRAINING COMPLETE": args.run_id, "best_sha256": record["best_weights_sha256"],
                      "registry": str(record_out), "registry_sha256": sha256_file(record_out)}, indent=2))


if __name__ == "__main__":
    main()
