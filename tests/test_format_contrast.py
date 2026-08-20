from __future__ import annotations

import json
import hashlib
from pathlib import Path

import pytest
import numpy as np

from bootstrap_format_contrast import (
    canonical_hash,
    bootstrap_arm_ap_draws,
    clean_cache_identity,
    compute_contrast_from_aps,
    draw_cache_identity,
    format_draws_from_arm_aps,
    load_bootstrap_schedule,
    load_clean_arm_cache,
    materialize_bootstrap_schedule,
    paired_format_bootstrap_draws,
    percentile,
    tt100k_endpoint_spec,
    validate_annotation_binding,
    validate_component_seed,
    validate_dataset_image_universes,
    validate_expected_image_count,
    validate_linked_inputs,
    write_clean_arm_cache,
    write_draw_cache,
)
from bootstrap_format_contrast_macro import (
    _macro_output_path,
    joint_macro_draws,
    macro_draws_from_components,
    macro_endpoint_plan,
    validate_component_artifact,
)
from bootstrap_format_contrast_macro import _seed as macro_seed
from run_format_contrast_p0 import _seed as contrast_seed
from run_format_contrast_p0 import _seed as contrast_seed
from run_format_contrast_p0 import _verify_artifact, build_tasks, contrast_command


def test_delta_e_cancels_fp32_and_has_positive_int8_worsening_sign() -> None:
    """Catches a reversed format contrast or failure to cancel FP32."""
    values = compute_contrast_from_aps(
        int8_clean=[0.50, 0.20, 0.45], fp8_clean=[0.54, 0.22, 0.49],
        int8_corrupt=[0.30, 0.07, 0.31], fp8_corrupt=[0.37, 0.12, 0.38],
    )

    assert values["delta_q"]["all"] == pytest.approx(0.04)
    assert values["delta_e"]["all"] == pytest.approx(0.03)
    assert values["delta_psi"] == pytest.approx(
        values["delta_e"]["small"] - values["delta_e"]["large"]
    )


def test_linked_inputs_reject_different_image_order(tmp_path: Path) -> None:
    """Catches a paired bootstrap whose arms would resample different images."""
    with pytest.raises(ValueError, match="identical ordered image IDs"):
        validate_linked_inputs([{"image_ids": [1, 2]}, {"image_ids": [2, 1]}])


def test_canonical_hash_excludes_its_named_hash_field() -> None:
    """Catches a completion hash that accidentally includes itself."""
    assert canonical_hash({"schema_version": 1, "report_sha256": "ignored"}, "report_sha256") == canonical_hash(
        {"schema_version": 1}, "report_sha256"
    )


def test_production_shaped_coco_corrupt_command_requires_exact_external_annotation_binding() -> None:
    """Catches a legitimate external COCO annotation path or a forged path in a self-hashed component command."""
    from validate_format_contrast_evidence import _validate_corrupt_command

    root = Path("/home/thuan/topic_c_ivc")
    relative = "outputs/bootstrap/ivc_format_contrast_v1/coco__yolo11n__gaussian_noise-s1.json"
    annotation_path = "/home/thuan/coco_journal/data/coco_src/annotations/instances_val2017.json"
    task = {
        "dataset": "coco", "model": "yolo11n", "endpoint": "area", "annotations": Path(annotation_path),
        "expected_images": 5000, "annotation_sha256": "a" * 64, "output": root / relative,
    }
    config = {"n_boot": 2000, "bootstrap_workers": 1, "seed_namespace": "ivc-format-contrast-v1-20260812"}
    seed = contrast_seed(config["seed_namespace"], task)
    draw_relative = "outputs/work/ivc_format_contrast_v1/format_contrast_draws/coco__yolo11n__gaussian_noise-s1.npz"
    schedule_relative = "outputs/work/ivc_format_contrast_v1/schedules/coco.npz"
    clean_relative = "outputs/work/ivc_format_contrast_v1/clean_arm_aps/coco__yolo11n.npz"
    clean_cache = {"path": root / clean_relative, "sha256": "b" * 64, "identity_sha256": "c" * 64}
    source_paths = tuple(
        [root / "frozen" / f"{arm}.json" for arm in ("int8_clean", "fp8_clean", "int8_corrupt", "fp8_corrupt")]
        for _ in range(3)
    )
    command = contrast_command(
        root, config, task, source_paths, root / draw_relative, root / schedule_relative, clean_cache,
    )
    component = {
        "endpoint_type": "area", "n_images": 5000, "seed": seed,
        "annotation": {"path": annotation_path, "sha256": "a" * 64},
        "temporary_draw_cache": {"path": draw_relative},
        "bootstrap_schedule": {"path": schedule_relative},
        "clean_arm_cache": {"path": clean_relative, "sha256": "b" * 64, "identity_sha256": "c" * 64},
    }

    _validate_corrupt_command(command, str(root), relative, component)
    forged = list(command)
    forged[forged.index("--annotations") + 1] = "/tmp/forged-instances_val2017.json"
    with pytest.raises(ValueError, match="scheduler corrupt command"):
        _validate_corrupt_command(forged, str(root), relative, component)


def test_joint_macro_output_path_rejects_a_symlink_escape_before_write(tmp_path: Path) -> None:
    """Catches a macro artifact path that would traverse a symlinked output directory."""
    outside = tmp_path.parent / f"{tmp_path.name}-macro-outside"
    outside.mkdir()
    (tmp_path / "outputs").symlink_to(outside, target_is_directory=True)

    with pytest.raises(ValueError, match="joint macro path escapes project root"):
        _macro_output_path(tmp_path, {"attempt": "ivc_format_contrast_v1"})


def test_self_hashed_component_with_false_seed_is_rejected() -> None:
    """Catches a component whose declared seed diverges from its shared dataset schedule."""
    component = {
        "seed": 17,
        "bootstrap_schedule": {"seed": 19, "sha256": "s" * 64, "n_boot": 2000, "n_images": 2, "image_ids_sha256": "i" * 64},
    }
    component["artifact_sha256"] = canonical_hash(component, "artifact_sha256")

    with pytest.raises(ValueError, match="component seed"):
        validate_component_seed(component, expected_seed=19)


def test_bootstrap_requires_exactly_2000_replicates() -> None:
    """Catches a contrast artifact with a non-prespecified bootstrap count."""
    from bootstrap_format_contrast import run_contrast

    with pytest.raises(ValueError, match="exactly 2000"):
        run_contrast(
            endpoint="area", annotations=Path("missing.json"), prediction_paths=[], input_paths=[], run_paths=[],
            n_boot=1999, seed=1, expected_images=1, annotation_sha256="a" * 64, output=Path("out.json"),
        )


def test_nonfinite_bootstrap_draw_is_retained_then_excluded_from_percentile() -> None:
    """Catches NaN strata being rejected before percentile filtering."""
    values = compute_contrast_from_aps(
        int8_clean=[0.50, float("nan"), 0.45], fp8_clean=[0.54, 0.22, 0.49],
        int8_corrupt=[0.30, 0.07, 0.31], fp8_corrupt=[0.37, 0.12, 0.38], allow_nonfinite=True,
    )

    assert np.isnan(values["delta_e"]["small"])
    assert percentile(np.asarray([float("nan"), 0.20])) == pytest.approx([0.20, 0.20, 0.20])
    with pytest.raises(ValueError, match="non-finite"):
        compute_contrast_from_aps(
            int8_clean=[0.50, float("nan"), 0.45], fp8_clean=[0.54, 0.22, 0.49],
            int8_corrupt=[0.30, 0.07, 0.31], fp8_corrupt=[0.37, 0.12, 0.38],
        )


def test_annotation_and_expected_image_bindings_fail_closed(tmp_path: Path) -> None:
    """Catches evaluating a different annotation file or image universe than planned."""
    annotations = tmp_path / "annotations.json"
    annotations.write_text("{}", encoding="utf-8")

    with pytest.raises(ValueError, match="annotation SHA-256"):
        validate_annotation_binding(annotations, "0" * 64)
    with pytest.raises(ValueError, match="expected image count"):
        validate_expected_image_count([1, 2], 3)


def test_runner_checks_annotation_hash_before_evaluator_construction(tmp_path: Path) -> None:
    """Catches an evaluator being built from annotations outside the run-record hash chain."""
    from bootstrap_format_contrast import run_contrast

    annotations = tmp_path / "annotations.json"
    annotations.write_text("{}", encoding="utf-8")
    with pytest.raises(ValueError, match="annotation SHA-256 mismatch"):
        run_contrast(
            endpoint="area", annotations=annotations, prediction_paths=[], input_paths=[], run_paths=[],
            n_boot=2000, seed=1, expected_images=1, annotation_sha256="0" * 64, output=tmp_path / "out.json",
        )


def test_tt100k_endpoint_spec_includes_all_and_existing_height_groups() -> None:
    """Catches TT100K output omitting all-AP or its two planned height strata."""
    labels, bins = tt100k_endpoint_spec()

    assert labels == ("all", "small_like", "large_like")
    assert bins["small_like"] == (0.0, 24.0)
    assert bins["large_like"] == (48.0, float("inf"))


def test_joint_macro_draws_preserves_replicate_alignment_across_cells() -> None:
    """Catches macro intervals assembled from independently permuted cell draws."""
    aligned = joint_macro_draws([np.asarray([0.0, 2.0]), np.asarray([0.0, 2.0])])
    independently_permuted = joint_macro_draws([np.asarray([0.0, 2.0]), np.asarray([2.0, 0.0])])

    assert aligned.tolist() == [0.0, 2.0]
    assert independently_permuted.tolist() == [1.0, 1.0]


def test_joint_macro_draws_retains_planned_weights_when_a_component_is_nan() -> None:
    """Catches a macro draw that silently reweights around a missing cell value."""
    values = joint_macro_draws([np.asarray([0.0, 2.0]), np.asarray([float("nan"), 4.0])])

    assert np.isnan(values[0])
    assert values[1] == pytest.approx(3.0)


def test_format_contrast_uses_one_seed_sequence_per_dataset_across_cells_and_macro() -> None:
    """Catches cell CIs and the joint macro using different within-dataset resamples."""
    namespace = "format-contrast-seed-test"
    coco_noise = {"dataset": "coco", "model": "yolo11n", "corruption": "gaussian_noise", "severity": 1}
    coco_jpeg = {"dataset": "coco", "model": "yolo11x", "corruption": "jpeg", "severity": 5}
    voc_noise = {"dataset": "voc", "model": "yolo11n", "corruption": "gaussian_noise", "severity": 1}

    assert contrast_seed(namespace, coco_noise) == contrast_seed(namespace, coco_jpeg)
    assert contrast_seed(namespace, coco_noise) != contrast_seed(namespace, voc_noise)
    assert contrast_seed(namespace, coco_noise) == macro_seed(namespace, "coco")


def test_paired_format_bootstrap_draws_reuses_each_replicate_sample_for_all_four_arms() -> None:
    """Catches an accelerated bootstrap that resamples formats independently."""
    def accumulator(scale: float, sample: list[int]) -> np.ndarray:
        repeated_sum = float(sum(sample))
        return scale * np.asarray([repeated_sum, 2.0 * repeated_sum, 3.0 * repeated_sum])

    evaluations = [1.0, 2.0, 3.0, 5.0]
    draws = paired_format_bootstrap_draws(
        evaluations, ("all", "small", "large"), n_images=3, n_boot=4, seed=17,
        accumulator=accumulator,
    )
    rng = np.random.default_rng(17)
    expected = [float(rng.choice(3, size=3, replace=True).sum()) for _ in range(4)]

    assert draws["delta_q"]["all"].tolist() == pytest.approx(expected)
    assert draws["delta_e"]["all"].tolist() == pytest.approx(expected)
    assert draws["delta_psi"].tolist() == pytest.approx([-value for value in expected])


def test_macro_draws_from_component_caches_matches_fixed_endpoint_membership() -> None:
    """Catches a fast macro aggregation that mixes area and TT100K draw caches."""
    tasks = [
        {"dataset": dataset, "endpoint": "tt100k-height" if dataset == "tt100k" else "area", "ordinal": ordinal}
        for dataset in ("coco", "voc", "kitti", "tt100k")
        for ordinal in range(36)
    ]
    components = [
        {
            "delta_e_all": np.concatenate([
                np.full(1000, float(index)), np.full(1000, float(index + 1)),
            ]),
            "delta_psi": np.full(2000, 3.0 if task["dataset"] == "tt100k" else 2.0),
        }
        for index, task in enumerate(tasks)
    ]

    draws = macro_draws_from_components(tasks, components)

    assert draws["four_dataset_macro_delta_e"][0] == pytest.approx(71.5)
    assert draws["four_dataset_macro_delta_e"][-1] == pytest.approx(72.5)
    assert np.all(draws["area_macro_delta_psi"] == 2.0)
    assert np.all(draws["tt100k_height_macro_delta_psi"] == 3.0)


def test_contrast_command_carries_parallel_worker_and_temporary_macro_cache_arguments(tmp_path: Path) -> None:
    """Catches an orchestrator that silently runs serially or cannot feed the macro cache."""
    from run_format_contrast_p0 import contrast_command

    prediction = tmp_path / "prediction.json"
    input_record = tmp_path / "input.json"
    run_record = tmp_path / "run.json"
    task = {
        "endpoint": "area", "annotations": tmp_path / "annotations.json", "expected_images": 2,
        "annotation_sha256": "a" * 64, "dataset": "coco", "model": "yolo11n", "corruption": "fog", "severity": 1,
        "output": tmp_path / "contrast.json",
        "records": {
            arm: ({"condition_id": arm}, run_record)
            for arm in ("int8_clean", "fp8_clean", "int8_corrupt", "fp8_corrupt")
        },
    }
    config = {"n_boot": 2000, "seed_namespace": "seed", "bootstrap_workers": 4}
    command = contrast_command(
        tmp_path, config, task,
        ([prediction] * 4, [input_record] * 4, [run_record] * 4), tmp_path / "draws.npz", tmp_path / "schedule.npz",
        {
            "path": tmp_path / "clean.npz", "sha256": "c" * 64,
            "identity_sha256": "d" * 64,
        },
    )

    assert command[command.index("--workers") + 1] == "4"
    assert command[command.index("--draw-cache") + 1] == str(tmp_path / "draws.npz")
    assert command[command.index("--schedule") + 1] == str(tmp_path / "schedule.npz")
    assert command[command.index("--evidence-root") + 1] == str(tmp_path)
    assert command[command.index("--clean-arm-cache") + 1] == str(tmp_path / "clean.npz")
    assert command[command.index("--clean-arm-cache-sha256") + 1] == "c" * 64
    assert command[command.index("--clean-arm-cache-identity-sha256") + 1] == "d" * 64
    assert command[command.index("--dataset") + 1] == "coco"
    assert command[command.index("--model") + 1] == "yolo11n"
    assert command[command.index("--seed") + 1] == str(contrast_seed("seed", task))


def test_draw_cache_bytes_are_precommitted_before_macro_consumption(tmp_path: Path) -> None:
    """Catches changed Delta draws whose mutable NPZ headers remain untouched."""
    from run_format_contrast_p0 import _verify_draw_cache

    cache = tmp_path / "draws.npz"
    identity = draw_cache_identity({
        "endpoint_type": "area", "n_images": 2, "n_boot": 2, "seed": 7,
        "annotation_sha256": "a" * 64, "input_hashes": {"int8_clean": {"prediction_sha256": "b" * 64}},
        "schedule_sha256": "s" * 64, "path": "outputs/work/attempt/format_contrast_draws/coco__cell.npz",
    })
    reference = write_draw_cache(
        cache, identity_sha256=identity, n_boot=2, schedule_sha256="s" * 64,
        delta_e_all=np.asarray([0.1, 0.2]), delta_psi=np.asarray([0.3, 0.4]),
    )

    _verify_draw_cache(cache, reference, 2, "s" * 64, identity)
    with np.load(cache, allow_pickle=False) as original:
        np.savez_compressed(
            cache,
            schema_version=original["schema_version"],
            cache_identity_sha256=original["cache_identity_sha256"],
            n_boot=original["n_boot"],
            schedule_sha256=original["schedule_sha256"],
            delta_e_all=np.asarray([0.9, 0.2]),
            delta_psi=original["delta_psi"],
        )
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        _verify_draw_cache(cache, reference, 2, "s" * 64, identity)


def test_clean_arm_cache_rejects_file_and_source_provenance_mutation(tmp_path: Path) -> None:
    """Catches clean draws reused after their bytes or source-arm binding changed."""
    cache_path = tmp_path / "coco__yolo11n.npz"
    schedule = {
        "path": "outputs/work/attempt/schedules/coco.npz", "sha256": "s" * 64,
        "n_boot": 2, "seed": 7, "n_images": 2, "image_ids_sha256": "i" * 64,
    }
    inputs = {
        "int8_clean": {"prediction_sha256": "1" * 64, "input_record_sha256": "2" * 64},
        "fp8_clean": {"prediction_sha256": "3" * 64, "input_record_sha256": "4" * 64},
    }
    metadata = {
        "dataset": "coco", "model": "yolo11n", "endpoint_type": "area", "n_images": 2, "n_boot": 2,
        "annotation_sha256": "a" * 64, "labels": ["all", "small", "large"],
        "input_hashes": inputs, "bootstrap_schedule": schedule,
    }
    identity = clean_cache_identity(metadata)
    reference = write_clean_arm_cache(
        cache_path, metadata=metadata, identity_sha256=identity,
        draws={"int8_clean": np.ones((2, 3)), "fp8_clean": np.full((2, 3), 2.0)},
        point={"int8_clean": [0.1, 0.2, 0.3], "fp8_clean": [0.2, 0.3, 0.4]},
    )

    loaded = load_clean_arm_cache(
        cache_path, endpoint="area", expected_images=2, annotation_sha256="a" * 64, schedule=schedule,
        expected_sha256=reference["sha256"], expected_identity_sha256=identity,
        expected_dataset="coco", expected_model="yolo11n", expected_input_hashes=inputs,
    )
    assert loaded["draws"]["int8_clean"].shape == (2, 3)
    with pytest.raises(ValueError, match="input provenance"):
        load_clean_arm_cache(
            cache_path, endpoint="area", expected_images=2, annotation_sha256="a" * 64, schedule=schedule,
            expected_sha256=reference["sha256"], expected_identity_sha256=identity,
            expected_dataset="coco", expected_model="yolo11n",
            expected_input_hashes={**inputs, "fp8_clean": {"prediction_sha256": "z" * 64}},
        )
    with np.load(cache_path, allow_pickle=False) as original:
        np.savez_compressed(
            cache_path,
            metadata=original["metadata"],
            int8_clean=np.full((2, 3), 9.0), fp8_clean=original["fp8_clean"],
            int8_clean_point=original["int8_clean_point"], fp8_clean_point=original["fp8_clean_point"],
        )
    with pytest.raises(ValueError, match="file SHA-256 mismatch"):
        load_clean_arm_cache(
            cache_path, endpoint="area", expected_images=2, annotation_sha256="a" * 64, schedule=schedule,
            expected_sha256=reference["sha256"], expected_identity_sha256=identity,
            expected_dataset="coco", expected_model="yolo11n", expected_input_hashes=inputs,
        )


def test_materialized_dataset_schedule_matches_legacy_seed_sequence_and_rejects_image_hash_mismatch(tmp_path: Path) -> None:
    """Catches cells using ad-hoc samples instead of one hash-bound dataset schedule."""
    image_ids = [11, 22, 33]
    schedule_path = tmp_path / "coco.npz"
    record = materialize_bootstrap_schedule(schedule_path, image_ids, n_boot=4, seed=91)
    loaded = load_bootstrap_schedule(schedule_path, image_ids, n_boot=4, seed=91)
    rng = np.random.default_rng(91)
    legacy = np.asarray([rng.choice(3, size=3, replace=True) for _ in range(4)], dtype=np.int32)

    assert record["sha256"] == hashlib.sha256(schedule_path.read_bytes()).hexdigest()
    assert np.array_equal(loaded["samples"], legacy)
    with pytest.raises(ValueError, match="image-ID SHA-256"):
        load_bootstrap_schedule(schedule_path, [11, 33, 22], n_boot=4, seed=91)


def test_precomputed_clean_arm_draws_match_legacy_four_arm_algebra() -> None:
    """Catches clean-arm reuse changing the paired Delta Q, Delta E, or Delta Psi calculation."""
    arm_aps = {
        "int8_clean": np.asarray([[0.30, 0.10, 0.20], [0.40, 0.20, 0.30]]),
        "fp8_clean": np.asarray([[0.35, 0.12, 0.24], [0.45, 0.22, 0.34]]),
        "int8_corrupt": np.asarray([[0.15, 0.04, 0.11], [0.20, 0.08, 0.16]]),
        "fp8_corrupt": np.asarray([[0.23, 0.08, 0.17], [0.29, 0.12, 0.22]]),
    }

    draws = format_draws_from_arm_aps(arm_aps, ("all", "small", "large"))

    assert draws["delta_q"]["all"].tolist() == pytest.approx([0.05, 0.05])
    assert draws["delta_e"]["all"].tolist() == pytest.approx([0.03, 0.04])
    assert draws["delta_psi"].tolist() == pytest.approx([0.0, 0.0])


def test_clean_reuse_matches_legacy_four_arm_draws_points_percentiles_and_macro() -> None:
    """Catches a cache path that changes any paired draw, point, interval, or macro value."""
    def accumulator(scale: float, sample: list[int]) -> np.ndarray:
        total = float(sum(sample))
        return scale * np.asarray([total, 2.0 * total, 3.0 * total])

    schedule = np.tile(np.asarray([[0, 1, 2], [1, 1, 1], [2, 0, 0], [2, 2, 1]], dtype=np.int32), (500, 1))
    evaluations = [1.0, 2.0, 3.0, 5.0]
    legacy = paired_format_bootstrap_draws(
        evaluations, ("all", "small", "large"), n_images=3, n_boot=2000, seed=101,
        schedule=schedule, accumulator=accumulator,
    )
    reusable_arms = bootstrap_arm_ap_draws(evaluations, schedule, accumulator=accumulator)
    cached = format_draws_from_arm_aps(
        {name: reusable_arms[:, index, :] for index, name in enumerate(("int8_clean", "fp8_clean", "int8_corrupt", "fp8_corrupt"))},
        ("all", "small", "large"),
    )
    full = [0, 1, 2]
    point_legacy = compute_contrast_from_aps(
        int8_clean=accumulator(1.0, full), fp8_clean=accumulator(2.0, full),
        int8_corrupt=accumulator(3.0, full), fp8_corrupt=accumulator(5.0, full),
    )
    reusable_points = {scale: accumulator(scale, full) for scale in evaluations}
    point_cached = compute_contrast_from_aps(
        int8_clean=reusable_points[1.0], fp8_clean=reusable_points[2.0],
        int8_corrupt=reusable_points[3.0], fp8_corrupt=reusable_points[5.0],
    )

    for metric in ("delta_q", "delta_e"):
        for label in legacy[metric]:
            assert np.array_equal(legacy[metric][label], cached[metric][label])
            assert percentile(legacy[metric][label]) == percentile(cached[metric][label])
    assert np.array_equal(legacy["delta_psi"], cached["delta_psi"])
    assert percentile(legacy["delta_psi"]) == percentile(cached["delta_psi"])
    assert point_cached == point_legacy
    tasks = [
        {"dataset": dataset, "endpoint": "tt100k-height" if dataset == "tt100k" else "area", "ordinal": ordinal}
        for dataset in ("coco", "voc", "kitti", "tt100k") for ordinal in range(36)
    ]
    macro = macro_draws_from_components(tasks, [
        {"delta_e_all": cached["delta_e"]["all"], "delta_psi": cached["delta_psi"]} for _ in tasks
    ])
    assert np.array_equal(macro["four_dataset_macro_delta_e"], legacy["delta_e"]["all"])
    assert np.array_equal(macro["area_macro_delta_psi"], legacy["delta_psi"])
    assert np.array_equal(macro["tt100k_height_macro_delta_psi"], legacy["delta_psi"])


def test_fork_workers_preserve_exact_scheduled_draw_order() -> None:
    """Catches a forked cache producer that reorders or changes fixed schedule rows."""
    def accumulator(scale: float, sample: list[int]) -> np.ndarray:
        return scale * np.asarray([float(sum(sample)), float(len(sample))])

    schedule = np.asarray([[0, 1, 1], [2, 0, 2], [1, 2, 0], [0, 0, 0]], dtype=np.int32)
    serial = bootstrap_arm_ap_draws([1.0, 5.0], schedule, workers=1, accumulator=accumulator)
    forked = bootstrap_arm_ap_draws([1.0, 5.0], schedule, workers=2, accumulator=accumulator)

    assert np.array_equal(forked, serial)


def test_bootstrap_arm_ap_draws_preserves_schedule_row_order_for_reusable_clean_arms() -> None:
    """Catches clean-arm reuse evaluating a different bootstrap schedule than corrupt arms."""
    def accumulator(offset: float, sample: list[int]) -> np.ndarray:
        return np.asarray([offset + sum(sample), offset + 2.0 * sum(sample)])

    schedule = np.asarray([[0, 1, 1], [2, 0, 2]], dtype=np.int32)
    draws = bootstrap_arm_ap_draws([0.0, 5.0], schedule, accumulator=accumulator)

    assert np.allclose(draws, np.asarray([
        [[2.0, 4.0], [7.0, 9.0]],
        [[4.0, 8.0], [9.0, 13.0]],
    ]))


def test_dataset_image_universe_validation_rejects_one_cross_cell_order_mismatch() -> None:
    """Catches a dataset schedule shared by cells that do not name one ordered image universe."""
    universes = validate_dataset_image_universes({
        "coco": [[1, 2, 3], [1, 2, 3]],
        "voc": [[4, 5]],
    })

    assert universes == {"coco": [1, 2, 3], "voc": [4, 5]}
    with pytest.raises(ValueError, match="cross-cell image universe mismatch"):
        validate_dataset_image_universes({"coco": [[1, 2, 3], [1, 3, 2]]})


def test_macro_endpoint_plan_has_balanced_overall_delta_e_and_disjoint_psi_membership() -> None:
    """Catches a macro whose endpoint membership or equal planned weights drift."""
    tasks = [
        {"dataset": dataset, "endpoint": "tt100k-height" if dataset == "tt100k" else "area", "ordinal": ordinal}
        for dataset in ("coco", "voc", "kitti", "tt100k")
        for ordinal in range(36)
    ]

    plan = macro_endpoint_plan(tasks)

    assert set(plan) == {"four_dataset_macro_delta_e", "area_macro_delta_psi", "tt100k_height_macro_delta_psi"}
    assert len(plan["four_dataset_macro_delta_e"]) == 144
    assert len(plan["area_macro_delta_psi"]) == 108
    assert len(plan["tt100k_height_macro_delta_psi"]) == 36
    assert {task["dataset"] for task in plan["four_dataset_macro_delta_e"]} == {"coco", "voc", "kitti", "tt100k"}


def test_macro_endpoint_plan_rejects_dataset_endpoint_swap_even_when_counts_match() -> None:
    """Catches a 108/36 macro plan whose area and TT100K memberships are swapped."""
    tasks = [
        {"dataset": dataset, "endpoint": "tt100k-height" if dataset == "tt100k" else "area", "ordinal": ordinal}
        for dataset in ("coco", "voc", "kitti", "tt100k")
        for ordinal in range(36)
    ]
    tasks[0]["endpoint"] = "tt100k-height"
    tasks[-1]["endpoint"] = "area"

    with pytest.raises(ValueError, match="semantic endpoint membership"):
        macro_endpoint_plan(tasks)


def test_mutated_component_artifact_is_rejected_before_joint_macro_output() -> None:
    """Catches a changed cell artifact entering macro inference under a stale hash."""
    artifact = {"schema_version": 1, "n_boot": 2000, "point": {"delta_e": {"all": 0.1}}}
    artifact["artifact_sha256"] = canonical_hash(artifact, "artifact_sha256")
    validate_component_artifact(artifact)
    artifact["point"]["delta_e"]["all"] = 0.2

    with pytest.raises(ValueError, match="component artifact SHA-256"):
        validate_component_artifact(artifact)


def _record(dataset: str, model: str, precision: str, corruption: str, severity: int) -> dict:
    return {
        "condition_id": f"{dataset}__{model}__{precision}__{corruption}-s{severity}",
        "dataset": dataset,
        "model": model,
        "precision": precision,
        "corruption": corruption,
        "severity": severity,
        "input_manifest_sha256": "a" * 64,
        "input_image_ids_sha256": "b" * 64,
        "annotation_sha256": "c" * 64,
        "n_images": 2,
    }


def _write_record(root: Path, attempt: str, record: dict) -> None:
    path = root / "manifests" / "runs" / attempt / f"{record['condition_id']}.json"
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(record), encoding="utf-8")


def _config() -> dict:
    datasets = ["coco", "voc", "kitti", "tt100k"]
    return {
        "attempt": "ivc_format_contrast_v1",
        "models": ["yolo11n", "yolo11m", "yolo11x"],
        "precisions": ["int8-entropy", "fp8"],
        "corruptions": ["gaussian_noise", "motion_blur", "fog", "jpeg"],
        "severities": [1, 3, 5],
        "datasets": [
            {
                "dataset": dataset,
                "annotations": f"annotations/{dataset}.json",
                "expected_images": 2,
                "clean_source_attempt": "clean",
                "corruption_source_attempt": "corrupt",
            }
            for dataset in datasets
        ],
    }


def _write_complete_grid(root: Path, *, omit_fp8_corruption: bool = False) -> dict:
    config = _config()
    for dataset in config["datasets"]:
        for model in config["models"]:
            for precision in config["precisions"]:
                _write_record(root, "clean", _record(dataset["dataset"], model, precision, "clean", 0))
                for corruption in config["corruptions"]:
                    for severity in config["severities"]:
                        if omit_fp8_corruption and (dataset["dataset"], model, precision, corruption, severity) == (
                            "coco", "yolo11n", "fp8", "fog", 1,
                        ):
                            continue
                        _write_record(root, "corrupt", _record(dataset["dataset"], model, precision, corruption, severity))
    return config


def test_build_tasks_refuses_missing_fp8_corruption_cell(tmp_path: Path) -> None:
    """Catches silently incomplete format-comparison grids."""
    config = _write_complete_grid(tmp_path, omit_fp8_corruption=True)

    with pytest.raises(ValueError, match="missing.*fp8"):
        build_tasks(tmp_path, config)


def test_build_tasks_returns_exact_clean_and_corrupted_grid(tmp_path: Path) -> None:
    """Catches an orchestrator that drops planned format-contrast conditions."""
    config = _write_complete_grid(tmp_path)

    tasks = build_tasks(tmp_path, config)

    assert len(tasks["clean"]) == 12
    assert len(tasks["corrupted"]) == 144


def test_runtime_config_rejects_worker_caps_above_measured_memory_limit() -> None:
    """Catches a grid launch that exceeds the four-worker real-RSS safety cap."""
    from run_format_contrast_p0 import validate_runtime_config

    valid = {
        "schema_version": 1, "attempt": "ivc_format_contrast_v1", "n_boot": 2000, "seed_namespace": "seed", "bootstrap_workers": 4, "cell_workers": 1,
    }
    validate_runtime_config(valid)
    with pytest.raises(ValueError, match="worker cap"):
        validate_runtime_config({**valid, "bootstrap_workers": 5})


def test_execution_package_manifest_commits_every_production_source_and_rejects_source_mutation(tmp_path: Path) -> None:
    """Catches a scheduler ledger that names a source package without binding its current bytes."""
    from run_format_contrast_p0 import (
        EXECUTION_PACKAGE_FILES,
        materialize_execution_package,
        validate_execution_package,
    )

    required_package_files = {
        "configs/ivc_format_contrast_v1.json",
        "src/run_format_contrast_p0.py",
        "src/format_contrast_scheduler.py",
        "src/bootstrap_format_contrast.py",
        "src/bootstrap_format_contrast_macro.py",
        "src/validate_format_contrast_evidence.py",
        "src/paired_bootstrap.py",
        "src/topic_c/manifest.py",
        "src/topic_c/tt100k_height.py",
    }
    assert set(EXECUTION_PACKAGE_FILES) == required_package_files
    config = tmp_path / "configs" / "ivc_format_contrast_v1.json"
    for relative in required_package_files:
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(f"frozen {relative}\n", encoding="utf-8")
    config.write_text(json.dumps({"attempt": "ivc_format_contrast_v1"}), encoding="utf-8")

    binding = materialize_execution_package(tmp_path, config)
    manifest = json.loads((tmp_path / binding["path"]).read_text(encoding="utf-8"))
    assert set(manifest["files_sha256"]) == required_package_files
    assert manifest["execution_root"] == str(tmp_path.resolve())
    assert validate_execution_package(tmp_path, binding)["manifest_sha256"] == binding["manifest_sha256"]

    changed = tmp_path / "src" / "topic_c" / "tt100k_height.py"
    changed.write_text("changed after precommit\n", encoding="utf-8")
    with pytest.raises(ValueError, match="execution package source SHA-256 mismatch"):
        validate_execution_package(tmp_path, binding)


def test_schedule_materialization_refuses_a_symlinked_output_escape_before_write(tmp_path: Path) -> None:
    """Catches an attempt directory whose schedule path resolves outside the reviewed root."""
    from run_format_contrast_p0 import materialize_dataset_schedules

    outside = tmp_path.parent / f"{tmp_path.name}-outside"
    outside.mkdir()
    (tmp_path / "outputs").symlink_to(outside, target_is_directory=True)
    config = {"attempt": "ivc_format_contrast_v1", "n_boot": 2000, "seed_namespace": "schedule-test"}

    with pytest.raises(ValueError, match="path escapes project root"):
        materialize_dataset_schedules(tmp_path, config, {"coco": [1, 2]})

    assert not (outside / "work").exists()


def test_execution_path_preflight_checks_report_macro_ledger_and_task_outputs_before_write(tmp_path: Path) -> None:
    """Catches a path escape that would otherwise occur after schedule materialization."""
    from run_format_contrast_p0 import validate_execution_paths

    outside = tmp_path.parent / f"{tmp_path.name}-outside-execution"
    outside.mkdir()
    (tmp_path / "outputs").symlink_to(outside, target_is_directory=True)
    config = {
        "attempt": "ivc_format_contrast_v1",
        "datasets": [{"dataset": name} for name in ("coco", "voc", "kitti", "tt100k")],
    }
    task = {
        "dataset": "coco", "model": "yolo11n",
        "output": tmp_path / "outputs" / "bootstrap" / "ivc_format_contrast_v1" / "coco__yolo11n__clean-s0.json",
    }

    with pytest.raises(ValueError, match="path escapes project root"):
        validate_execution_paths(tmp_path, config, {"clean": [task], "corrupted": []})

    assert not (outside / "reports").exists()


def test_build_tasks_rejects_annotation_hash_chain_mismatch(tmp_path: Path) -> None:
    """Catches source arms that bind one contrast to different annotations."""
    config = _write_complete_grid(tmp_path)
    path = next((tmp_path / "manifests" / "runs" / "corrupt").glob("coco__yolo11n__fp8__fog-s1.json"))
    record = json.loads(path.read_text(encoding="utf-8"))
    record["annotation_sha256"] = "d" * 64
    path.write_text(json.dumps(record), encoding="utf-8")

    with pytest.raises(ValueError, match="annotation SHA-256"):
        build_tasks(tmp_path, config)


def test_verify_artifact_rejects_mutated_clean_artifact_self_hash(tmp_path: Path) -> None:
    """Catches a clean artifact mutation before the completion path accepts it."""
    config = _write_complete_grid(tmp_path)
    clean_task = build_tasks(tmp_path, config)["clean"][0]
    annotations = tmp_path / "annotations.json"
    annotations.write_text("{}", encoding="utf-8")
    clean_task["annotations"] = annotations
    clean_task["annotation_sha256"] = __import__("hashlib").sha256(annotations.read_bytes()).hexdigest()
    artifact = {
        "schema_version": 1, "n_boot": 2000, "n_images": 2, "endpoint_type": "area",
        "annotation": {"sha256": clean_task["annotation_sha256"]}, "input_hashes": {},
    }
    artifact["artifact_sha256"] = canonical_hash(artifact, "artifact_sha256")
    artifact["n_images"] = 3
    path = tmp_path / "mutated-clean.json"
    path.write_text(json.dumps(artifact), encoding="utf-8")

    with pytest.raises(ValueError, match="artifact SHA-256"):
        _verify_artifact(clean_task, path, 2000)


def test_complete_remote_contrast_report_has_exact_grid_and_hashes() -> None:
    """Catches a locally presented remote result with missing or altered evidence."""
    project_root = Path(__file__).resolve().parents[1]
    report_path = project_root / "outputs" / "reports" / "ivc_format_contrast_v1_complete.json"

    assert report_path.is_file(), "remote completion report has not been synchronized"
    report = json.loads(report_path.read_text(encoding="utf-8"))
    assert report["clean_contrasts"] == 12
    assert report["corrupted_contrasts"] == 144
    assert report["n_boot"] == 2000
    assert len(report["component_artifacts_sha256"]) == 156
    assert len(report["artifacts_sha256"]) == 157
    assert report["report_sha256"] == canonical_hash(report, "report_sha256")
    assert report["joint_macro_artifact_sha256"] == report["artifacts_sha256"][
        "outputs/bootstrap/ivc_format_contrast_v1/ivc_format_contrast_v1_joint_macro.json"
    ]
    for relative_path, expected_sha256 in report["artifacts_sha256"].items():
        artifact = project_root / relative_path
        assert artifact.is_file(), f"missing synchronized contrast artifact: {relative_path}"
        assert hashlib.sha256(artifact.read_bytes()).hexdigest() == expected_sha256


def _write_validator_complete_grid(tmp_path: Path) -> dict[str, Path]:
    """Write one hash-bound synthetic 12/144 grid with zero-valued exact macro statistics."""
    from validate_format_contrast_evidence import validate_complete_report

    config = tmp_path / "configs" / "ivc_format_contrast_v1.json"
    config.parent.mkdir(parents=True)
    config_document = {
        "schema_version": 1,
        "attempt": "ivc_format_contrast_v1",
        "n_boot": 2000,
        "seed_namespace": "validator-chain",
        "bootstrap_workers": 1,
        "cell_workers": 4,
    }
    config.write_text(json.dumps(config_document), encoding="utf-8")
    run_record = tmp_path / "manifests" / "runs" / "witness.json"
    run_record.parent.mkdir(parents=True)
    run_record.write_text("{}\n", encoding="utf-8")
    run_hash = hashlib.sha256(run_record.read_bytes()).hexdigest()
    models = ("yolo11n", "yolo11m", "yolo11x")
    datasets = ("coco", "voc", "kitti", "tt100k")
    schedules, clean_references = {}, {}
    for dataset in datasets:
        ids = [1, 2]
        schedule_path = tmp_path / "outputs" / "work" / "ivc_format_contrast_v1" / "schedules" / f"{dataset}.npz"
        record = materialize_bootstrap_schedule(
            schedule_path, ids, n_boot=2000, seed=contrast_seed("validator-chain", {"dataset": dataset}),
        )
        schedules[dataset] = {
            "path": str(schedule_path.relative_to(tmp_path)), "sha256": record["sha256"], "n_boot": 2000,
            "seed": record["seed"], "n_images": 2, "image_ids_sha256": record["image_ids_sha256"],
        }
        for model in models:
            inputs = {
                "int8_clean": {
                    "prediction_sha256": "1" * 64, "input_record_sha256": "2" * 64,
                    "input_manifest_sha256": "3" * 64, "image_ids_sha256": record["image_ids_sha256"],
                    "run_record_sha256": run_hash,
                },
                "fp8_clean": {
                    "prediction_sha256": "4" * 64, "input_record_sha256": "5" * 64,
                    "input_manifest_sha256": "6" * 64, "image_ids_sha256": record["image_ids_sha256"],
                    "run_record_sha256": run_hash,
                },
            }
            metadata = {
                "dataset": dataset, "model": model,
                "endpoint_type": "tt100k-height" if dataset == "tt100k" else "area", "n_images": 2, "n_boot": 2000,
                "annotation_sha256": "a" * 64,
                "labels": ["all", "small_like", "large_like"] if dataset == "tt100k" else ["all", "small", "large"],
                "input_hashes": inputs, "bootstrap_schedule": schedules[dataset],
            }
            cache_path = tmp_path / "outputs" / "work" / "ivc_format_contrast_v1" / "clean_arm_aps" / f"{dataset}__{model}.npz"
            reference = write_clean_arm_cache(
                cache_path, metadata=metadata, identity_sha256=clean_cache_identity(metadata),
                draws={"int8_clean": np.zeros((2000, 3)), "fp8_clean": np.zeros((2000, 3))},
                point={"int8_clean": [0.1, 0.1, 0.1], "fp8_clean": [0.1, 0.1, 0.1]},
            )
            clean_references[(dataset, model)] = {
                "path": str(cache_path.relative_to(tmp_path)), **reference,
            }

    component_hashes, component_documents, draw_references = {}, {}, {}
    corrupted_relatives = []
    for dataset in datasets:
        for model in models:
            # The cache metadata is not a raw prediction payload; retain the exact two source bindings in every cell.
            with np.load(tmp_path / clean_references[(dataset, model)]["path"], allow_pickle=False) as cache:
                clean_inputs = json.loads(str(cache["metadata"].item()))["input_hashes"]
            entries = [("clean-s0", True)] + [
                (f"{corruption}-s{severity}", False)
                for corruption in ("gaussian_noise", "motion_blur", "fog", "jpeg")
                for severity in (1, 3, 5)
            ]
            for suffix, is_clean in entries:
                relative_path = f"outputs/bootstrap/ivc_format_contrast_v1/{dataset}__{model}__{suffix}.json"
                inputs = {
                    **clean_inputs,
                    "int8_corrupt": clean_inputs["int8_clean"] if is_clean else {
                        **clean_inputs["int8_clean"], "prediction_sha256": "7" * 64,
                    },
                    "fp8_corrupt": clean_inputs["fp8_clean"] if is_clean else {
                        **clean_inputs["fp8_clean"], "prediction_sha256": "8" * 64,
                    },
                }
                document = {
                    "schema_version": 1, "method": "test", "endpoint_type": "tt100k-height" if dataset == "tt100k" else "area",
                    "annotation": {
                        "path": f"/remote/topic_c_ivc/annotations/{dataset}.json",
                        "sha256": "a" * 64,
                    }, "n_images": 2, "n_boot": 2000,
                    "seed": schedules[dataset]["seed"], "point": {"delta_e": {"all": 0.0}, "delta_psi": 0.0},
                    "input_hashes": inputs, "bootstrap_schedule": schedules[dataset],
                    "clean_arm_cache": clean_references[(dataset, model)],
                }
                if not is_clean:
                    draw_path = tmp_path / "outputs" / "work" / "ivc_format_contrast_v1" / "format_contrast_draws" / f"{Path(relative_path).stem}.npz"
                    temporary = {"path": str(draw_path.relative_to(tmp_path))}
                    provisional = {**document, "temporary_draw_cache": temporary}
                    identity = draw_cache_identity({
                        "endpoint_type": document["endpoint_type"], "n_images": 2, "n_boot": 2000, "seed": document["seed"],
                        "annotation_sha256": "a" * 64, "input_hashes": inputs,
                        "schedule_sha256": schedules[dataset]["sha256"], "path": temporary["path"],
                    })
                    reference = write_draw_cache(
                        draw_path, identity_sha256=identity, n_boot=2000, schedule_sha256=schedules[dataset]["sha256"],
                        delta_e_all=np.zeros(2000), delta_psi=np.zeros(2000),
                    )
                    document["temporary_draw_cache"] = {**temporary, **reference}
                    draw_references[relative_path] = document["temporary_draw_cache"]
                    corrupted_relatives.append(relative_path)
                document["artifact_sha256"] = canonical_hash(document, "artifact_sha256")
                artifact = tmp_path / relative_path
                artifact.parent.mkdir(parents=True, exist_ok=True)
                artifact.write_text(json.dumps(document), encoding="utf-8")
                component_hashes[relative_path] = hashlib.sha256(artifact.read_bytes()).hexdigest()
                component_documents[relative_path] = document
    macro_relative_path = "outputs/bootstrap/ivc_format_contrast_v1/ivc_format_contrast_v1_joint_macro.json"
    macro = tmp_path / macro_relative_path
    macro_document = {
        "schema_version": 1, "n_boot": 2000, "component_cells": 144,
        "bootstrap_schedules": schedules,
        "component_artifacts": {
            relative: {
                "artifact_sha256": component_hashes[relative], "bootstrap_schedule": component_documents[relative]["bootstrap_schedule"],
                "clean_arm_cache": component_documents[relative]["clean_arm_cache"],
                "temporary_draw_cache": component_documents[relative]["temporary_draw_cache"],
            }
            for relative in corrupted_relatives
        },
        "coverage": {
            "four_dataset_macro_delta_e": {"planned_component_count": 144, "finite_complete_replicates": 2000, "total_replicates": 2000},
            "area_macro_delta_psi": {"planned_component_count": 108, "finite_complete_replicates": 2000, "total_replicates": 2000},
            "tt100k_height_macro_delta_psi": {"planned_component_count": 36, "finite_complete_replicates": 2000, "total_replicates": 2000},
        },
        "point": {
            "four_dataset_macro_delta_e": 0.0,
            "area_macro_delta_psi": 0.0,
            "tt100k_height_macro_delta_psi": 0.0,
        },
        "percentile_intervals": {
            "four_dataset_macro_delta_e": [0.0, 0.0, 0.0],
            "area_macro_delta_psi": [0.0, 0.0, 0.0],
            "tt100k_height_macro_delta_psi": [0.0, 0.0, 0.0],
        },
    }
    macro_document["artifact_sha256"] = canonical_hash(macro_document, "artifact_sha256")
    macro.write_text(json.dumps(macro_document), encoding="utf-8")
    artifact_hashes = {
        **component_hashes,
        macro_relative_path: hashlib.sha256(macro.read_bytes()).hexdigest(),
    }
    clean_cache_report = {
        reference["path"]: {"sha256": reference["sha256"], "identity_sha256": reference["identity_sha256"]}
        for reference in clean_references.values()
    }
    supporting_hashes = {
        **{reference["path"]: reference["sha256"] for reference in schedules.values()},
        **{relative: reference["sha256"] for relative, reference in clean_cache_report.items()},
        **{reference["path"]: reference["sha256"] for reference in draw_references.values()},
    }
    from run_format_contrast_p0 import EXECUTION_PACKAGE_FILES

    for relative in EXECUTION_PACKAGE_FILES:
        source = tmp_path / relative
        source.parent.mkdir(parents=True, exist_ok=True)
        if source != config:
            source.write_text(f"synthetic reviewed source {relative}\n", encoding="utf-8")
    package_path = tmp_path / "outputs" / "reports" / "ivc_format_contrast_v1_execution_package.json"
    package_document = {
        "schema_version": 1,
        "attempt": "ivc_format_contrast_v1",
        "execution_root": "/remote/topic_c_ivc",
        "files_sha256": {relative: hashlib.sha256((tmp_path / relative).read_bytes()).hexdigest() for relative in EXECUTION_PACKAGE_FILES},
    }
    package_document["manifest_sha256"] = canonical_hash(package_document, "manifest_sha256")
    package_path.parent.mkdir(parents=True, exist_ok=True)
    package_path.write_text(json.dumps(package_document), encoding="utf-8")
    package = {
        "path": str(package_path.relative_to(tmp_path)),
        "sha256": hashlib.sha256(package_path.read_bytes()).hexdigest(),
        "manifest_sha256": package_document["manifest_sha256"],
    }

    execution_root = package_document["execution_root"]

    def remote_path(relative: str) -> str:
        return str(Path(execution_root) / relative)

    def command_for(phase: str, relative: str, component: dict) -> list[str]:
        dataset, model = Path(relative).stem.split("__")[:2]
        if phase == "clean":
            return [
                "/remote/qtsd/bin/python", remote_path("src/run_format_contrast_p0.py"),
                "--project-root", execution_root,
                "--config", remote_path("configs/ivc_format_contrast_v1.json"),
                "--execute-clean-cell", "--dataset", dataset, "--model", model,
            ]
        command = [
            "/remote/qtsd/bin/python", remote_path("src/bootstrap_format_contrast.py"),
            "--endpoint", component["endpoint_type"],
            "--annotations", component["annotation"]["path"],
            "--expected-images", str(component["n_images"]),
            "--annotation-sha256", component["annotation"]["sha256"],
        ]
        for arm in ("int8_clean", "fp8_clean", "int8_corrupt", "fp8_corrupt"):
            option = "--" + arm.replace("_", "-")
            command += [
                option, remote_path(f"inputs/{Path(relative).stem}/{arm}.json"),
                option + "-input", remote_path(f"inputs/{Path(relative).stem}/{arm}-input.json"),
                option + "-run", remote_path(f"manifests/runs/{Path(relative).stem}/{arm}.json"),
            ]
        clean = component["clean_arm_cache"]
        return command + [
            "--n-boot", "2000", "--seed", str(component["seed"]), "--workers", "1",
            "--out", remote_path(relative), "--evidence-root", execution_root,
            "--draw-cache", remote_path(component["temporary_draw_cache"]["path"]),
            "--schedule", remote_path(component["bootstrap_schedule"]["path"]),
            "--clean-arm-cache", remote_path(clean["path"]),
            "--clean-arm-cache-sha256", clean["sha256"],
            "--clean-arm-cache-identity-sha256", clean["identity_sha256"],
            "--dataset", dataset, "--model", model,
        ]

    def write_ledger(phase: str, relatives: list[str]) -> tuple[Path, dict]:
        records = []
        for relative in relatives:
            component = component_documents[relative]
            artifact = tmp_path / relative
            stem = artifact.stem
            log_relative = f"outputs/logs/ivc_format_contrast_v1/format_contrast_cells/{phase}__{stem}.log"
            log = tmp_path / log_relative
            log.parent.mkdir(parents=True, exist_ok=True)
            log.write_text(f"synthetic {phase} {stem}\n", encoding="utf-8")
            output_paths = [relative]
            if phase == "clean":
                output_paths.append(component["clean_arm_cache"]["path"])
            else:
                output_paths.append(component["temporary_draw_cache"]["path"])
            records.append({
                "key": f"{phase}:{stem}", "command": command_for(phase, relative, component), "pid": 1,
                "log_path": log_relative, "log_sha256": hashlib.sha256(log.read_bytes()).hexdigest(),
                "output_paths": output_paths,
                "output_sha256": {path: hashlib.sha256((tmp_path / path).read_bytes()).hexdigest() for path in output_paths},
                "missing_output_paths": [], "started_at_utc": "2026-08-12T00:00:00Z",
                "ended_at_utc": "2026-08-12T00:00:01Z", "exit_status": 0, "verification": {},
            })
        ledger = {
            "schema_version": 1, "scheduler": "bounded independent immutable format-contrast cells",
            "execution_package": package, "cell_workers": 4, "peak_active": 4,
            "status": "complete", "failure": None, "records": records,
        }
        ledger["ledger_sha256"] = canonical_hash(ledger, "ledger_sha256")
        path = tmp_path / "outputs" / "reports" / f"ivc_format_contrast_v1_{phase}_scheduler.json"
        path.write_text(json.dumps(ledger), encoding="utf-8")
        return path, {
            "path": str(path.relative_to(tmp_path)), "sha256": hashlib.sha256(path.read_bytes()).hexdigest(),
            "ledger_sha256": ledger["ledger_sha256"],
        }

    clean_relatives = [relative for relative in component_hashes if relative.endswith("clean-s0.json")]
    clean_ledger_path, clean_ledger = write_ledger("clean", clean_relatives)
    corrupt_ledger_path, corrupt_ledger = write_ledger("corrupt", corrupted_relatives)
    report = {
        "attempt": "ivc_format_contrast_v1",
        "clean_contrasts": 12,
        "corrupted_contrasts": 144,
        "n_boot": 2000,
        "config_sha256": hashlib.sha256(config.read_bytes()).hexdigest(),
        "component_artifacts_sha256": component_hashes,
        "joint_macro_artifact_sha256": artifact_hashes[macro_relative_path],
        "artifacts_sha256": artifact_hashes,
        "bootstrap_schedules": schedules,
        "clean_arm_caches": clean_cache_report,
        "draw_caches": draw_references,
        "supporting_evidence_sha256": supporting_hashes,
        "evidence_files_sha256": {**artifact_hashes, **supporting_hashes},
        "execution_package": package,
        "scheduler_ledgers": {"clean": clean_ledger, "corrupt": corrupt_ledger},
    }
    report["report_sha256"] = canonical_hash(report, "report_sha256")
    report_path = tmp_path / "outputs" / "reports" / "ivc_format_contrast_v1_complete.json"
    report_path.parent.mkdir(parents=True, exist_ok=True)
    report_path.write_text(json.dumps(report), encoding="utf-8")
    assert validate_complete_report(tmp_path, config)["status"] == "valid"
    return {
        "config": config,
        "report": report_path,
        "macro": macro,
        "draw_cache": tmp_path / next(iter(draw_references.values()))["path"],
        "package_source": tmp_path / "src" / "topic_c" / "tt100k_height.py",
        "clean_log": tmp_path / json.loads(clean_ledger_path.read_text(encoding="utf-8"))["records"][0]["log_path"],
        "clean_ledger": clean_ledger_path,
        "corrupt_ledger": corrupt_ledger_path,
    }


def _rebind_scheduler_ledger_and_report(
    tmp_path: Path, bundle: dict[str, Path], phase: str, mutate: callable,
) -> None:
    """Apply one adversarial ledger edit while preserving every surrounding byte hash."""
    report = json.loads(bundle["report"].read_text(encoding="utf-8"))
    ledger_path = tmp_path / report["scheduler_ledgers"][phase]["path"]
    ledger = json.loads(ledger_path.read_text(encoding="utf-8"))
    mutate(ledger)
    ledger["ledger_sha256"] = canonical_hash(ledger, "ledger_sha256")
    ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    report["scheduler_ledgers"][phase] = {
        "path": report["scheduler_ledgers"][phase]["path"],
        "sha256": hashlib.sha256(ledger_path.read_bytes()).hexdigest(),
        "ledger_sha256": ledger["ledger_sha256"],
    }
    report["report_sha256"] = canonical_hash(report, "report_sha256")
    bundle["report"].write_text(json.dumps(report), encoding="utf-8")


@pytest.mark.parametrize(
    ("phase", "mutate", "message"),
    [
        ("clean", lambda ledger: ledger.__setitem__("cell_workers", 99), "scheduler worker cap"),
        ("corrupt", lambda ledger: ledger.__setitem__("peak_active", 99), "scheduler peak"),
        (
            "clean",
            lambda ledger: ledger["records"][0]["command"].__setitem__(
                ledger["records"][0]["command"].index("--execute-clean-cell"), "--execute"
            ),
            "scheduler clean command",
        ),
        (
            "corrupt",
            lambda ledger: ledger["records"][0]["command"].__setitem__(
                ledger["records"][0]["command"].index("--workers") + 1, "2"
            ),
            "scheduler corrupt command",
        ),
        (
            "corrupt",
            lambda ledger: ledger["records"][0]["command"].__setitem__(
                ledger["records"][0]["command"].index("--out") + 1,
                "/remote/topic_c_ivc/outputs/bootstrap/ivc_format_contrast_v1/forged.json",
            ),
            "scheduler corrupt command",
        ),
        (
            "corrupt",
            lambda ledger: ledger["records"][0]["command"].__setitem__(
                ledger["records"][0]["command"].index("--draw-cache") + 1,
                "/remote/topic_c_ivc/outputs/work/ivc_format_contrast_v1/forged-draw-cache.npz",
            ),
            "scheduler corrupt command",
        ),
        (
            "corrupt",
            lambda ledger: ledger["records"][0]["command"].__setitem__(
                ledger["records"][0]["command"].index("--schedule") + 1,
                "/remote/topic_c_ivc/outputs/work/ivc_format_contrast_v1/schedules/forged.npz",
            ),
            "scheduler corrupt command",
        ),
        (
            "corrupt",
            lambda ledger: ledger["records"][0]["command"].__setitem__(
                ledger["records"][0]["command"].index("--clean-arm-cache") + 1,
                "/remote/topic_c_ivc/outputs/work/ivc_format_contrast_v1/clean_arm_aps/forged.npz",
            ),
            "scheduler corrupt command",
        ),
        (
            "corrupt",
            lambda ledger: ledger["records"][0]["command"].__setitem__(
                ledger["records"][0]["command"].index("--dataset") + 1, "forged-dataset"
            ),
            "scheduler corrupt command",
        ),
        (
            "corrupt",
            lambda ledger: ledger["records"][0]["command"].__setitem__(
                ledger["records"][0]["command"].index("--model") + 1, "forged-model"
            ),
            "scheduler corrupt command",
        ),
    ],
)
def test_local_validator_rejects_rebound_scheduler_cap_peak_and_phase_command_contract(
    tmp_path: Path, phase: str, mutate: callable, message: str,
) -> None:
    """Catches self-hashed ledgers that forge resource limits or production entry-point semantics."""
    from validate_format_contrast_evidence import validate_complete_report

    bundle = _write_validator_complete_grid(tmp_path)
    _rebind_scheduler_ledger_and_report(tmp_path, bundle, phase, mutate)

    with pytest.raises(ValueError, match=message):
        validate_complete_report(tmp_path, bundle["config"])


def test_local_validation_rejects_changed_draw_cache_bytes_with_headers_unchanged(tmp_path: Path) -> None:
    """Catches an end-to-end synchronized cache mutation before any macro claim is accepted."""
    from validate_format_contrast_evidence import validate_complete_report

    bundle = _write_validator_complete_grid(tmp_path)
    changed_path = bundle["draw_cache"]
    with np.load(changed_path, allow_pickle=False) as cache:
        np.savez_compressed(
            changed_path,
            schema_version=cache["schema_version"], cache_identity_sha256=cache["cache_identity_sha256"],
            n_boot=cache["n_boot"], schedule_sha256=cache["schedule_sha256"],
            delta_e_all=np.ones(2000), delta_psi=cache["delta_psi"],
        )

    with pytest.raises(ValueError, match="evidence SHA-256 mismatch"):
        validate_complete_report(tmp_path, bundle["config"])


def test_local_validator_rejects_a_clean_component_substituted_into_the_macro_grid(tmp_path: Path) -> None:
    """Catches a self-consistent 144-entry macro that omits a corrupt cell for a clean cell."""
    from validate_format_contrast_evidence import validate_complete_report

    bundle = _write_validator_complete_grid(tmp_path)
    report = json.loads(bundle["report"].read_text(encoding="utf-8"))
    macro = json.loads(bundle["macro"].read_text(encoding="utf-8"))
    corrupt_relative = sorted(macro["component_artifacts"])[0]
    clean_relative = next(relative for relative in report["component_artifacts_sha256"] if relative.endswith("clean-s0.json"))
    clean = json.loads((tmp_path / clean_relative).read_text(encoding="utf-8"))
    macro["component_artifacts"].pop(corrupt_relative)
    macro["component_artifacts"][clean_relative] = {
        "artifact_sha256": report["component_artifacts_sha256"][clean_relative],
        "bootstrap_schedule": clean["bootstrap_schedule"],
        "clean_arm_cache": clean["clean_arm_cache"],
        "temporary_draw_cache": None,
    }
    macro["artifact_sha256"] = canonical_hash(macro, "artifact_sha256")
    bundle["macro"].write_text(json.dumps(macro), encoding="utf-8")
    macro_relative = str(bundle["macro"].relative_to(tmp_path))
    report["artifacts_sha256"][macro_relative] = hashlib.sha256(bundle["macro"].read_bytes()).hexdigest()
    report["joint_macro_artifact_sha256"] = report["artifacts_sha256"][macro_relative]
    report["evidence_files_sha256"][macro_relative] = report["artifacts_sha256"][macro_relative]
    report["report_sha256"] = canonical_hash(report, "report_sha256")
    bundle["report"].write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="exact corrupted component membership"):
        validate_complete_report(tmp_path, bundle["config"])


@pytest.mark.parametrize(
    ("field", "message"),
    [
        ("point", "joint macro point recomputation mismatch"),
        ("percentile_intervals", "joint macro percentile recomputation mismatch"),
        ("coverage", "joint macro finite coverage recomputation mismatch"),
    ],
)
def test_local_validator_recomputes_macro_statistics_from_bound_draw_caches(
    tmp_path: Path, field: str, message: str,
) -> None:
    """Catches a self-hashed macro whose reported statistics diverge from its 144 cached draws."""
    from validate_format_contrast_evidence import validate_complete_report

    bundle = _write_validator_complete_grid(tmp_path)
    macro = json.loads(bundle["macro"].read_text(encoding="utf-8"))
    if field == "point":
        macro[field]["four_dataset_macro_delta_e"] = 0.5
    elif field == "percentile_intervals":
        macro[field]["area_macro_delta_psi"] = [0.0, 0.5, 0.0]
    else:
        macro[field]["tt100k_height_macro_delta_psi"]["finite_complete_replicates"] = 1999
    macro["artifact_sha256"] = canonical_hash(macro, "artifact_sha256")
    bundle["macro"].write_text(json.dumps(macro), encoding="utf-8")
    report = json.loads(bundle["report"].read_text(encoding="utf-8"))
    macro_relative = str(bundle["macro"].relative_to(tmp_path))
    report["artifacts_sha256"][macro_relative] = hashlib.sha256(bundle["macro"].read_bytes()).hexdigest()
    report["joint_macro_artifact_sha256"] = report["artifacts_sha256"][macro_relative]
    report["evidence_files_sha256"][macro_relative] = report["artifacts_sha256"][macro_relative]
    report["report_sha256"] = canonical_hash(report, "report_sha256")
    bundle["report"].write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match=message):
        validate_complete_report(tmp_path, bundle["config"])


def test_local_validator_rejects_an_endpoint_swap_even_when_hashes_and_ledgers_are_rebound(tmp_path: Path) -> None:
    """Catches an endpoint declaration that conflicts with the fixed TT100K/area component plan."""
    from validate_format_contrast_evidence import validate_complete_report

    bundle = _write_validator_complete_grid(tmp_path)
    report = json.loads(bundle["report"].read_text(encoding="utf-8"))
    relative = next(path for path in report["component_artifacts_sha256"] if path.startswith("outputs/bootstrap/ivc_format_contrast_v1/tt100k") and not path.endswith("clean-s0.json"))
    component_path = tmp_path / relative
    component = json.loads(component_path.read_text(encoding="utf-8"))
    component["endpoint_type"] = "area"
    component["artifact_sha256"] = canonical_hash(component, "artifact_sha256")
    component_path.write_text(json.dumps(component), encoding="utf-8")
    component_hash = hashlib.sha256(component_path.read_bytes()).hexdigest()
    report["component_artifacts_sha256"][relative] = component_hash
    report["artifacts_sha256"][relative] = component_hash
    report["evidence_files_sha256"][relative] = component_hash

    macro = json.loads(bundle["macro"].read_text(encoding="utf-8"))
    macro["component_artifacts"][relative]["artifact_sha256"] = component_hash
    macro["artifact_sha256"] = canonical_hash(macro, "artifact_sha256")
    bundle["macro"].write_text(json.dumps(macro), encoding="utf-8")
    macro_relative = str(bundle["macro"].relative_to(tmp_path))
    macro_hash = hashlib.sha256(bundle["macro"].read_bytes()).hexdigest()
    report["artifacts_sha256"][macro_relative] = macro_hash
    report["evidence_files_sha256"][macro_relative] = macro_hash
    report["joint_macro_artifact_sha256"] = macro_hash

    corrupt_ledger_path = tmp_path / report["scheduler_ledgers"]["corrupt"]["path"]
    ledger = json.loads(corrupt_ledger_path.read_text(encoding="utf-8"))
    record = next(item for item in ledger["records"] if item["key"] == f"corrupt:{component_path.stem}")
    record["output_sha256"][relative] = component_hash
    ledger["ledger_sha256"] = canonical_hash(ledger, "ledger_sha256")
    corrupt_ledger_path.write_text(json.dumps(ledger), encoding="utf-8")
    report["scheduler_ledgers"]["corrupt"] = {
        "path": report["scheduler_ledgers"]["corrupt"]["path"],
        "sha256": hashlib.sha256(corrupt_ledger_path.read_bytes()).hexdigest(),
        "ledger_sha256": ledger["ledger_sha256"],
    }
    report["report_sha256"] = canonical_hash(report, "report_sha256")
    bundle["report"].write_text(json.dumps(report), encoding="utf-8")

    with pytest.raises(ValueError, match="component endpoint/dataset plan"):
        validate_complete_report(tmp_path, bundle["config"])


@pytest.mark.parametrize(
    ("target", "message"),
    [
        ("package_source", "execution package source SHA-256 mismatch"),
        ("clean_log", "scheduler task log SHA-256 mismatch"),
        ("clean_ledger", "scheduler ledger file SHA-256 mismatch"),
    ],
)
def test_local_validator_rejects_mutated_execution_envelope_bytes(
    tmp_path: Path, target: str, message: str,
) -> None:
    """Catches source, task-log, or scheduler-ledger mutation outside the 317 scientific files."""
    from validate_format_contrast_evidence import validate_complete_report

    bundle = _write_validator_complete_grid(tmp_path)
    with bundle[target].open("ab") as handle:
        handle.write(b"mutated execution envelope bytes\n")

    with pytest.raises(ValueError, match=message):
        validate_complete_report(tmp_path, bundle["config"])
