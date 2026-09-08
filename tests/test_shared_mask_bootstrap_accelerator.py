from __future__ import annotations

from concurrent.futures import ThreadPoolExecutor
import math
from pathlib import Path
from types import SimpleNamespace
import threading
import time

import numpy as np
import pytest

import accelerate_shared_mask_bootstrap_v3 as accelerator
import analyze_shared_mask_pilot as legacy_analysis
from paired_bootstrap import accumulate_ap
from topic_c.manifest import sha256_file
from topic_c.shared_mask_pilot import (
    PilotError,
    read_complete_json,
    write_complete_json,
)


def _entry(
    scores: list[float],
    matches: list[list[int]],
    ignored: list[list[bool]],
    gt_ignored: list[int],
) -> dict:
    return {
        "dtScores": np.asarray(scores, dtype=np.float64),
        "dtMatches": np.asarray(matches, dtype=np.float64),
        "dtIgnore": np.asarray(ignored, dtype=bool),
        "gtIgnore": np.asarray(gt_ignored, dtype=np.int64),
    }


def _fake_evaluation(*, area_zero_none: bool = False) -> SimpleNamespace:
    params = SimpleNamespace(
        iouThrs=np.asarray([0.5, 0.75]),
        recThrs=np.asarray([0.0, 0.5, 1.0]),
        catIds=[1, 2],
        areaRng=[[0, 1e10], [0, 1024], [1024, 9216], [9216, 1e10]],
    )
    n_images = 3
    entries = []
    for category in range(2):
        for area in range(4):
            for image in range(n_images):
                if area == 0 and area_zero_none:
                    entries.append(None)
                    continue
                if category == 1 and area == 0 and image == 1:
                    entries.append(None)
                    continue
                # Scores deliberately tie across images.  Matches, ignored
                # detections and out-of-order/repeated image draws exercise the
                # exact stable-sort and concatenation contract.
                entries.append(
                    _entry(
                        [0.9, 0.5, 0.5],
                        [
                            [1 if (category + image + area) % 2 == 0 else 0, 0, 1],
                            [1, 0, 1 if image % 2 else 0],
                        ],
                        [
                            [False, image == 2, False],
                            [False, False, category == 1],
                        ],
                        [0, 1 if area > 0 else 0],
                    )
                )
    return SimpleNamespace(
        params=params,
        _paramsEval=SimpleNamespace(imgIds=[10, 20, 30]),
        evalImgs=entries,
    )


@pytest.mark.parametrize(
    "positions",
    ([0, 1, 2], [2, 0, 2, 1], [1, 1, 0], [2, 2, 2]),
)
def test_overall_only_is_bit_identical_to_legacy(positions: list[int]) -> None:
    evaluation = _fake_evaluation()
    expected = float(accumulate_ap(evaluation, positions)[0])
    observed = accelerator.accumulate_ap_overall(evaluation, positions)

    assert expected.hex() == observed.hex()


def test_overall_only_preserves_all_invalid_nan() -> None:
    evaluation = _fake_evaluation(area_zero_none=True)

    assert math.isnan(float(accumulate_ap(evaluation, [0, 0, 2])[0]))
    assert math.isnan(accelerator.accumulate_ap_overall(evaluation, [0, 0, 2]))


def test_full_2000_draw_vector_is_bit_identical() -> None:
    evaluation = _fake_evaluation()
    samples = accelerator.make_samples(seed=941, n_boot=2000, n_images=3)
    legacy = np.asarray(
        [float(accumulate_ap(evaluation, row.tolist())[0]) for row in samples]
    )
    accelerated = np.asarray(
        [accelerator.accumulate_ap_overall(evaluation, row.tolist()) for row in samples]
    )

    assert np.array_equal(legacy, accelerated, equal_nan=True)


def _fp8_records(tmp_path: Path) -> dict[str, dict]:
    values = {}
    for field in ("prediction", "input_record", "run_record", "metric"):
        path = tmp_path / f"{field}.json"
        path.write_text(f'{{"field":"{field}"}}\n', encoding="utf-8")
        values[field] = str(path)
        values[f"{field}_sha256"] = sha256_file(path)
    values.update(
        {
            "input_manifest_sha256": "m" * 64,
            "input_image_ids_sha256": "i" * 64,
            "image_ids": [3, 1, 2],
            "ap": 0.25,
        }
    )
    return {"default_fp8": dict(values), "shared_fp8": dict(values)}


def test_fp8_identity_gate_requires_fresh_path_and_sha_identity(tmp_path: Path) -> None:
    records = _fp8_records(tmp_path)
    proof = accelerator.fp8_identity_gate(records, "cell")

    assert proof["prediction_sha256"] == sha256_file(Path(proof["prediction"]))

    forged = {name: dict(value) for name, value in records.items()}
    forged["shared_fp8"]["prediction_sha256"] = "0" * 64
    with pytest.raises(accelerator.AcceleratorError, match="prediction bytes"):
        accelerator.fp8_identity_gate(forged, "cell")


def test_fp8_identity_gate_rejects_different_paths_with_identical_bytes(
    tmp_path: Path,
) -> None:
    records = _fp8_records(tmp_path)
    alias = tmp_path / "prediction-copy.json"
    alias.write_bytes(Path(records["default_fp8"]["prediction"]).read_bytes())
    records["shared_fp8"]["prediction"] = str(alias)
    records["shared_fp8"]["prediction_sha256"] = sha256_file(alias)

    with pytest.raises(accelerator.AcceleratorError, match="paths differ"):
        accelerator.fp8_identity_gate(records, "cell")


def test_fp8_identity_gate_rejects_different_ap(tmp_path: Path) -> None:
    records = _fp8_records(tmp_path)
    records["shared_fp8"]["ap"] = 0.25000000000000006

    with pytest.raises(accelerator.AcceleratorError, match="recorded AP differs"):
        accelerator.fp8_identity_gate(records, "cell")


def _contract() -> SimpleNamespace:
    return SimpleNamespace(
        sha256_file=sha256_file,
        write_complete_json=write_complete_json,
        read_complete_json=read_complete_json,
    )


def test_clean_cache_round_trip_is_atomic_and_source_bound(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("immutable\n", encoding="utf-8")
    npz = tmp_path / "clean.npz"
    record = tmp_path / "clean.json"
    arrays = {
        "default_int8": np.asarray([0.1, 0.2, 0.3]),
        "fp8": np.asarray([0.4, 0.5, 0.6]),
        "shared_int8": np.asarray([0.7, 0.8, 0.9]),
    }
    samples = accelerator.make_samples(7, 3, 2)
    accelerator.write_clean_cache(
        npz,
        record,
        block_id="voc_rtdetr_l",
        dataset="voc",
        model="rtdetr-l",
        config_sha256="c" * 64,
        seed=7,
        n_images=2,
        input_image_ids_sha256="i" * 64,
        annotation_sha256="a" * 64,
        samples_digest=accelerator.samples_sha256(samples),
        arrays=arrays,
        point={name: float(value[0]) for name, value in arrays.items()},
        source_artifacts_sha256={str(source): sha256_file(source)},
        acceleration_provenance={"accelerator_sha256": "x" * 64},
        contract=_contract(),
    )

    observed_record, observed = accelerator.validate_clean_cache(
        npz,
        record,
        block_id="voc_rtdetr_l",
        dataset="voc",
        model="rtdetr-l",
        config_sha256="c" * 64,
        seed=7,
        n_boot=3,
        n_images=2,
        input_image_ids_sha256="i" * 64,
        annotation_sha256="a" * 64,
        samples_digest=accelerator.samples_sha256(samples),
        contract=_contract(),
    )
    assert observed_record["record_sha256"]
    assert all(np.array_equal(observed[name], arrays[name]) for name in arrays)
    with pytest.raises(accelerator.AcceleratorError, match="overwrite immutable"):
        accelerator.write_clean_cache(
            npz,
            record,
            block_id="voc_rtdetr_l",
            dataset="voc",
            model="rtdetr-l",
            config_sha256="c" * 64,
            seed=7,
            n_images=2,
            input_image_ids_sha256="i" * 64,
            annotation_sha256="a" * 64,
            samples_digest=accelerator.samples_sha256(samples),
            arrays=arrays,
            point={name: float(value[0]) for name, value in arrays.items()},
            source_artifacts_sha256={str(source): sha256_file(source)},
            acceleration_provenance={},
            contract=_contract(),
        )

    source.write_text("changed\n", encoding="utf-8")
    with pytest.raises(accelerator.AcceleratorError, match="source changed"):
        accelerator.validate_clean_cache(
            npz,
            record,
            block_id="voc_rtdetr_l",
            dataset="voc",
            model="rtdetr-l",
            config_sha256="c" * 64,
            seed=7,
            n_boot=3,
            n_images=2,
            input_image_ids_sha256="i" * 64,
            annotation_sha256="a" * 64,
            samples_digest=accelerator.samples_sha256(samples),
            contract=_contract(),
        )


def test_npz_without_record_marker_is_not_a_valid_clean_cache(tmp_path: Path) -> None:
    npz = tmp_path / "interrupted.npz"
    accelerator._atomic_npz(
        npz,
        schema_version=np.asarray(1),
        n_boot=np.asarray(1),
        seed=np.asarray(1),
        n_images=np.asarray(1),
        default_int8=np.asarray([0.1]),
        fp8=np.asarray([0.2]),
        shared_int8=np.asarray([0.3]),
    )

    with pytest.raises(PilotError, match="incomplete JSON artifact"):
        accelerator.validate_clean_cache(
            npz,
            tmp_path / "interrupted.json",
            block_id="voc_rtdetr_l",
            dataset="voc",
            model="rtdetr-l",
            config_sha256="c" * 64,
            seed=1,
            n_boot=1,
            n_images=1,
            input_image_ids_sha256="i" * 64,
            annotation_sha256="a" * 64,
            samples_digest="s" * 64,
            contract=_contract(),
        )


def test_accelerated_cell_cache_is_accepted_by_frozen_validator(tmp_path: Path) -> None:
    source = tmp_path / "source.txt"
    source.write_text("bound\n", encoding="utf-8")
    npz, record = tmp_path / "cell.npz", tmp_path / "cell.json"
    payload = {
        "block_id": "voc_rtdetr_l",
        "dataset": "voc",
        "model": "rtdetr-l",
        "n_boot": 3,
        "seed": 11,
        "config_sha256": "c" * 64,
        "clean": {"default_int8": {"image_ids": [1, 2]}},
    }
    condition = {"corruption": "fog", "severity": 1}
    arrays = {
        "delta_e_default": np.asarray([1.0, 2.0, 3.0]),
        "delta_e_shared": np.asarray([0.5, 1.5, 2.5]),
        "omega": np.asarray([0.5, 0.5, 0.5]),
    }
    accelerator.write_standard_cell_cache(
        npz,
        record,
        payload=payload,
        condition=condition,
        arrays=arrays,
        point={
            "clean_gap_default": 1.0,
            "corrupt_gap_default": 2.0,
            "delta_e_default": 1.0,
            "clean_gap_shared": 0.5,
            "corrupt_gap_shared": 1.0,
            "delta_e_shared": 0.5,
            "omega_default_minus_shared": 0.5,
        },
        source_artifacts_sha256={str(source): sha256_file(source)},
        acceleration_provenance={"accelerator_sha256": "x" * 64},
        contract=_contract(),
    )

    observed_record, observed_arrays = legacy_analysis.validate_cell_cache(
        npz,
        record,
        block_id="voc_rtdetr_l",
        corruption="fog",
        severity=1,
        n_boot=3,
        config_sha256="c" * 64,
    )
    assert observed_record["acceleration_provenance"]["accelerator_sha256"] == "x" * 64
    assert np.array_equal(observed_arrays["omega"], arrays["omega"])
    with np.load(npz, allow_pickle=False) as data:
        assert set(data.files) == accelerator.CELL_ARRAY_FIELDS


def test_worker_limit_is_hard_refusal() -> None:
    assert accelerator.validate_worker_count(3) == 3
    with pytest.raises(accelerator.AcceleratorError, match="1..3"):
        accelerator.validate_worker_count(4)


def test_bounded_dispatch_never_starts_a_fourth_job_after_failure() -> None:
    started = []
    lock = threading.Lock()

    def work(job: dict) -> dict:
        with lock:
            started.append(job["id"])
        if job["id"] == 0:
            raise RuntimeError("scientific gate")
        time.sleep(0.05)
        return {"id": job["id"]}

    with pytest.raises(RuntimeError, match="scientific gate"):
        accelerator.run_bounded_cells(
            [{"id": value} for value in range(8)],
            workers=3,
            work=work,
            executor_factory=ThreadPoolExecutor,
        )

    assert set(started).issubset({0, 1, 2})


def test_sample_schedule_matches_frozen_generator_call() -> None:
    expected = np.random.default_rng(1234).integers(
        0, 7, size=(2000, 7), dtype=np.int32
    )

    assert np.array_equal(accelerator.make_samples(1234, 2000, 7), expected)
