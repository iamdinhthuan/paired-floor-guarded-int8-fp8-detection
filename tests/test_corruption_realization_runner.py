from __future__ import annotations

from pathlib import Path

import run_corruption_realization_sensitivity as runner


def test_exact_prespecified_realization_grid() -> None:
    datasets = ("voc", "kitti", "tt100k")
    seeds = (202608181, 202608182, 202608183)

    manifests = runner.manifest_tasks(datasets=datasets, realization_seeds=seeds)
    inference = runner.inference_tasks(datasets=datasets, realization_seeds=seeds)

    assert len(manifests) == 108
    assert len(inference) == 222  # 6 clean arms + 216 corrupted arms.
    assert len({runner.task_identity(task) for task in manifests}) == 108
    assert len({runner.task_identity(task) for task in inference}) == 222
    assert {task["precision"] for task in inference} == {"int8-entropy", "fp8"}


def test_jpeg_cache_is_shared_because_realization_seed_cannot_change_codec_bytes(tmp_path: Path) -> None:
    first = runner.cache_root(tmp_path, "voc", 101, "jpeg")
    second = runner.cache_root(tmp_path, "voc", 202, "jpeg")
    noise_first = runner.cache_root(tmp_path, "voc", 101, "gaussian_noise")
    noise_second = runner.cache_root(tmp_path, "voc", 202, "gaussian_noise")

    assert first == second
    assert noise_first != noise_second
    assert "jpeg_deterministic" in str(first)
