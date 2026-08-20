from __future__ import annotations

import hashlib
import json
from pathlib import Path

import pytest
import numpy as np

import run_confirmatory_bootstrap as runner


def digest(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical(document: dict, field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()


def test_dataset_seed_is_shared_within_dataset_and_distinct_across_datasets() -> None:
    assert runner.dataset_seed("confirmatory-20260818", "voc") == runner.dataset_seed(
        "confirmatory-20260818", "voc"
    )
    assert runner.dataset_seed("confirmatory-20260818", "voc") != runner.dataset_seed(
        "confirmatory-20260818", "kitti"
    )


def test_inference_completion_requires_exact_two_hash_bound_datasets(tmp_path: Path) -> None:
    rows = []
    for dataset in ("voc", "kitti"):
        plan = tmp_path / f"{dataset}.plan.json"
        plan.write_text(json.dumps({"plan_sha256": dataset * 16}) + "\n", encoding="utf-8")
        evaluation = tmp_path / f"{dataset}.evaluation.json"
        evaluation.write_text(json.dumps({"runs": 117}) + "\n", encoding="utf-8")
        rows.append(
            {
                "dataset": dataset,
                "partition_role": "final",
                "machine_split": "test",
                "attempt": f"{dataset}_confirmatory_final_117_v1",
                "plan": str(plan),
                "plan_sha256": dataset * 16,
                "evaluation_report": str(evaluation),
                "evaluation_report_sha256": digest(evaluation),
            }
        )
    report = {"attempt": "voc_kitti_confirmatory_v1", "datasets": rows}
    report["inference_report_sha256"] = canonical(report, "inference_report_sha256")
    path = tmp_path / "inference.json"
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    validated = runner.validate_inference_completion(
        path, attempt="voc_kitti_confirmatory_v1"
    )
    assert set(validated) == {"voc", "kitti"}

    rows[0]["partition_role"] = "selection"
    report["inference_report_sha256"] = canonical(report, "inference_report_sha256")
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="final-partition binding"):
        runner.validate_inference_completion(path, attempt="voc_kitti_confirmatory_v1")


def test_bootstrap_completion_rehashes_all_72_artifacts(tmp_path: Path) -> None:
    artifacts = {}
    draw_caches = {}
    for index in range(72):
        artifact = tmp_path / "outputs" / "bootstrap" / f"cell-{index}.json"
        artifact.parent.mkdir(parents=True, exist_ok=True)
        artifact.write_text(json.dumps({"n_boot": 2000, "seed": 17}) + "\n", encoding="utf-8")
        artifacts[str(artifact.relative_to(tmp_path))] = digest(artifact)
        cache = artifact.with_suffix(".draws.npz")
        np.savez_compressed(cache, excess=np.zeros((2000, 4)), psi=np.zeros(2000))
        draw_caches[str(cache.relative_to(tmp_path))] = digest(cache)
    report = {
        "attempt": "voc_confirmatory_final_117_v1",
        "plan_sha256": "p" * 64,
        "cells": 72,
        "n_boot": 2000,
        "shared_seed": 17,
        "joint_dataset_draw_sequence": True,
        "all_bootstrap_input_hashes_validated": True,
        "artifacts_sha256": artifacts,
        "draw_caches_sha256": draw_caches,
    }
    path = tmp_path / "bootstrap.json"
    path.write_text(json.dumps(report) + "\n", encoding="utf-8")

    runner.validate_bootstrap_completion(
        tmp_path,
        path,
        attempt="voc_confirmatory_final_117_v1",
        plan_sha256="p" * 64,
        n_boot=2000,
        shared_seed=17,
    )

    artifact = tmp_path / next(iter(artifacts))
    artifact.write_text("mutated\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="artifact hash mismatch"):
        runner.validate_bootstrap_completion(
            tmp_path,
            path,
            attempt="voc_confirmatory_final_117_v1",
            plan_sha256="p" * 64,
            n_boot=2000,
            shared_seed=17,
        )
