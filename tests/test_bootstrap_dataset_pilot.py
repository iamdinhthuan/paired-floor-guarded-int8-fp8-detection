from __future__ import annotations

from pathlib import Path

import bootstrap_dataset_pilot as bootstrap


def cell(model: str, corruption: str) -> dict:
    def run(label: str) -> dict:
        return {
            "condition_id": f"{model}__{label}__{corruption}",
            "input_manifest_sha256": label[0] * 64,
        }

    return {
        "model": model,
        "precision": "fp8",
        "corruption": corruption,
        "severity": 5,
        "fp32_clean": run("fp32_clean"),
        "quant_clean": run("quant_clean"),
        "fp32_corrupt": run("fp32_corrupt"),
        "quant_corrupt": run("quant_corrupt"),
    }


def test_confirmatory_commands_share_dataset_draw_seed_and_use_requested_b(tmp_path: Path) -> None:
    first = bootstrap.command(
        tmp_path,
        tmp_path / "annotations.json",
        "attempt",
        cell("yolo11n", "fog"),
        tmp_path / "paired_bootstrap.py",
        n_boot=2000,
        shared_seed=20260818,
    )
    second = bootstrap.command(
        tmp_path,
        tmp_path / "annotations.json",
        "attempt",
        cell("yolo11x", "jpeg"),
        tmp_path / "paired_bootstrap.py",
        n_boot=2000,
        shared_seed=20260818,
    )

    assert first[first.index("--n-boot") + 1] == "2000"
    assert second[second.index("--n-boot") + 1] == "2000"
    assert first[first.index("--seed") + 1] == "20260818"
    assert second[second.index("--seed") + 1] == "20260818"
    assert "--draw-cache" in first
    assert first[first.index("--draw-cache") + 1].endswith(".draws.npz")


def test_legacy_commands_keep_condition_specific_seed(tmp_path: Path) -> None:
    first_cell = cell("yolo11n", "fog")
    second_cell = cell("yolo11x", "jpeg")
    first = bootstrap.command(
        tmp_path,
        tmp_path / "annotations.json",
        "attempt",
        first_cell,
        tmp_path / "paired_bootstrap.py",
        n_boot=500,
        shared_seed=None,
    )
    second = bootstrap.command(
        tmp_path,
        tmp_path / "annotations.json",
        "attempt",
        second_cell,
        tmp_path / "paired_bootstrap.py",
        n_boot=500,
        shared_seed=None,
    )

    assert first[first.index("--seed") + 1] != second[second.index("--seed") + 1]
