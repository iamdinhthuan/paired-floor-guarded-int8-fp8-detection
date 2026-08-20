from __future__ import annotations

import json
from pathlib import Path

import pytest

import run_corruption_realization_bootstrap as runner
from pilot_registry import canonical_hash
from topic_c.manifest import sha256_file


def test_planned_cells_are_exact_three_by_three_by_four_by_three_grid() -> None:
    cells = runner.planned_cells((101, 202, 303))

    assert len(cells) == 108
    assert len({runner.cell_identity(cell) for cell in cells}) == 108
    assert {
        (cell["dataset"], cell["realization_seed"], cell["corruption"], cell["severity"])
        for cell in cells
    } == {
        (dataset, seed, corruption, severity)
        for dataset in runner.DATASETS
        for seed in (101, 202, 303)
        for corruption in runner.CORRUPTIONS
        for severity in runner.SEVERITIES
    }


def test_planned_cells_refuse_nonexact_realization_seed_design() -> None:
    with pytest.raises(ValueError, match="three unique"):
        runner.planned_cells((101, 101, 303))


def test_dataset_seed_is_stable_and_dataset_specific() -> None:
    first = runner.dataset_seed("realization-bootstrap-v1", "voc")

    assert first == runner.dataset_seed("realization-bootstrap-v1", "voc")
    assert first != runner.dataset_seed("realization-bootstrap-v1", "kitti")
    assert 0 <= first < 2**32


def test_bootstrap_config_rejects_changed_source_bytes(tmp_path: Path) -> None:
    source = tmp_path / "source.py"
    source.write_text("original\n", encoding="utf-8")
    config = {
        "schema_version": 1,
        "attempt": "corruption_realization_bootstrap_v1",
        "n_boot": 2000,
        "workers": 2,
        "seed_namespace": "test",
        "source_manifest": [{"path": "source.py", "sha256": sha256_file(source)}],
    }
    config["config_sha256"] = canonical_hash(config, "config_sha256")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(config), encoding="utf-8")
    runner._load_config(tmp_path, path)

    source.write_text("changed\n", encoding="utf-8")
    with pytest.raises(SystemExit, match="source hash mismatch"):
        runner._load_config(tmp_path, path)
