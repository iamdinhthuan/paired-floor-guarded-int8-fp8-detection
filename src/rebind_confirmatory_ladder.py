#!/usr/bin/env python3
"""Rebind a completed ladder after a metadata-only split-key correction.

No ONNX graph or engine is rebuilt.  The amendment is accepted only when all
24 original registry bytes still match and the new execution config differs
solely in the machine split key plus an explicit scientific partition role.
"""
from __future__ import annotations

import argparse
import json
from copy import deepcopy
from datetime import datetime, timezone
from pathlib import Path

from run_confirmatory_ladder import (
    DATASETS,
    MODELS,
    PRECISIONS,
    canonical_hash,
    complete,
    load_execution_config,
)
from topic_c.manifest import sha256_file


def build_input_signature(document: dict) -> dict:
    value = deepcopy(document)
    for key in (
        "schema_version",
        "config_sha256",
        "supersedes_execution_config",
        "supersedes_execution_config_sha256",
        "correction",
    ):
        value.pop(key, None)
    for dataset in DATASETS:
        item = value.get("datasets", {}).get(dataset, {})
        item.pop("split", None)
        item.pop("partition_role", None)
    return value


def validate_metadata_only_correction(old: dict, new: dict) -> None:
    if old.get("attempt") != new.get("attempt"):
        raise SystemExit("LADDER REBIND REFUSED: attempt changed")
    for dataset in DATASETS:
        old_item = old.get("datasets", {}).get(dataset, {})
        new_item = new.get("datasets", {}).get(dataset, {})
        if (
            old_item.get("split") != "final"
            or new_item.get("split") != "test"
            or new_item.get("partition_role") != "final"
        ):
            raise SystemExit("LADDER REBIND REFUSED: split correction is not the approved final-to-test mapping")
    if build_input_signature(old) != build_input_signature(new):
        raise SystemExit("LADDER REBIND REFUSED: build-affecting configuration changed")


def validate_source_artifacts(report: dict) -> None:
    artifacts = report.get("artifacts")
    expected = {
        (dataset, model, precision)
        for dataset in DATASETS
        for model in MODELS
        for precision in PRECISIONS
    }
    keys = {
        (row.get("dataset"), row.get("model"), row.get("precision"))
        for row in artifacts or []
        if isinstance(row, dict)
    }
    if not isinstance(artifacts, list) or len(artifacts) != 24 or keys != expected:
        raise SystemExit("LADDER REBIND REFUSED: source ladder lacks exact 24-artifact grid")
    for row in artifacts:
        for path_field, hash_field in (
            ("onnx_registry", "onnx_registry_sha256"),
            ("engine_registry", "engine_registry_sha256"),
        ):
            path = Path(row.get(path_field, "")).resolve()
            if not path.is_file() or sha256_file(path) != row.get(hash_field):
                raise SystemExit(
                    f"LADDER REBIND REFUSED: artifact hash mismatch: {row.get('dataset')}/"
                    f"{row.get('model')}/{row.get('precision')}/{path_field}"
                )
    registries = report.get("dataset_registries")
    if (
        not isinstance(registries, list)
        or len(registries) != 2
        or {row.get("dataset") for row in registries if isinstance(row, dict)} != set(DATASETS)
    ):
        raise SystemExit("LADDER REBIND REFUSED: dataset registries are incomplete")
    for row in registries:
        path = Path(row.get("path", "")).resolve()
        if not complete(path) or sha256_file(path) != row.get("sha256"):
            raise SystemExit(f"LADDER REBIND REFUSED: dataset registry hash mismatch: {row.get('dataset')}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--source-ladder-report", required=True)
    parser.add_argument("--old-config", required=True)
    parser.add_argument("--new-config", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    source_path = Path(args.source_ladder_report).resolve()
    old_path = Path(args.old_config).resolve()
    new_path = Path(args.new_config).resolve()
    output = Path(args.out).resolve()
    marker = output.with_suffix(output.suffix + ".complete")
    if output.exists() or marker.exists():
        raise SystemExit(f"LADDER REBIND REFUSED: output exists: {output}")
    old = json.loads(old_path.read_text(encoding="utf-8"))
    new = json.loads(new_path.read_text(encoding="utf-8"))
    if old.get("config_sha256") != canonical_hash(old, "config_sha256"):
        raise SystemExit("LADDER REBIND REFUSED: old config canonical hash mismatch")
    if new.get("config_sha256") != canonical_hash(new, "config_sha256"):
        raise SystemExit("LADDER REBIND REFUSED: new config canonical hash mismatch")
    if (
        Path(new.get("supersedes_execution_config", "")).name != old_path.name
        or new.get("supersedes_execution_config_sha256") != sha256_file(old_path)
    ):
        raise SystemExit("LADDER REBIND REFUSED: new config does not bind old config bytes")
    validate_metadata_only_correction(old, new)
    load_execution_config(root, new_path)
    source = json.loads(source_path.read_text(encoding="utf-8"))
    if (
        source.get("ladder_report_sha256") != canonical_hash(source, "ladder_report_sha256")
        or source.get("attempt") != new["attempt"]
        or source.get("config_sha256") != sha256_file(old_path)
    ):
        raise SystemExit("LADDER REBIND REFUSED: source ladder/config binding mismatch")
    validate_source_artifacts(source)
    document = {
        "schema_version": 2,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": new["attempt"],
        "config": str(new_path),
        "config_sha256": sha256_file(new_path),
        "source_ladder_report": str(source_path),
        "source_ladder_report_sha256": sha256_file(source_path),
        "source_config": str(old_path),
        "source_config_sha256": sha256_file(old_path),
        "correction": {
            "scientific_partition_role": "final",
            "old_machine_split": "final",
            "corrected_machine_split": "test",
            "engine_build_inputs_changed": False,
        },
        "training_report": source.get("training_report"),
        "training_report_sha256": source.get("training_report_sha256"),
        "artifacts": source["artifacts"],
        "dataset_registries": source["dataset_registries"],
    }
    document["ladder_report_sha256"] = canonical_hash(document, "ladder_report_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    marker.write_text(sha256_file(output) + "\n", encoding="utf-8")
    print(f"LADDER REBIND COMPLETE artifacts=24 output={output}")


if __name__ == "__main__":
    main()
