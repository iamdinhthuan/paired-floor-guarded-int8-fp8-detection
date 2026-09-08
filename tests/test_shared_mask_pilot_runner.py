from __future__ import annotations

from pathlib import Path
import subprocess
import threading

import numpy as np
import pytest

import analyze_shared_mask_pilot as analysis
import run_shared_mask_pilot as runner
from topic_c.manifest import sha256_file
from topic_c.shared_mask_pilot import (
    ATTEMPT,
    CLEAN_GATE_MAX_DETECTION_COUNT_RELATIVE_DIFFERENCE,
    CLEAN_GATE_TOLERANCE_AP_POINTS,
    MASK_POLICY,
    PARENT_ATTEMPT,
    PARENT_EXECUTION_ARCHIVE_SHA256,
    PARENT_FAILURE_MANIFEST_CANONICAL_SHA256,
    canonical_hash,
    read_complete_json,
    shared_condition_id,
    validate_config,
    write_complete_json,
)


ROOT = Path(__file__).resolve().parents[1]


def test_real_shared_mask_config_is_frozen_and_exact() -> None:
    config = validate_config(
        ROOT,
        ROOT / "configs/shared_mask_pilot_v2.json",
        expected_attempt="shared_mask_pilot_v2",
    )

    assert config["config_sha256"] == canonical_hash(config, "config_sha256")
    assert config["attempt"] == "shared_mask_pilot_v2"
    assert config["mask_policy"] == MASK_POLICY
    assert config["parent_failure"]["attempt"] == PARENT_ATTEMPT
    assert (
        config["parent_failure"]["failure_manifest_canonical_sha256"]
        == PARENT_FAILURE_MANIFEST_CANONICAL_SHA256
    )
    assert (
        config["parent_failure"]["execution_sources_archive_sha256"]
        == PARENT_EXECUTION_ARCHIVE_SHA256
    )
    assert config["bootstrap_replicates"] == 2000
    assert config["execution"]["inference_workers"] == 1
    assert CLEAN_GATE_TOLERANCE_AP_POINTS == 0.10
    assert CLEAN_GATE_MAX_DETECTION_COUNT_RELATIVE_DIFFERENCE == 0.001


def test_v3_config_freezes_bounded_parallelism_and_inadmissible_timing() -> None:
    config = validate_config(
        ROOT,
        ROOT / "configs/shared_mask_pilot_v3.json",
        expected_attempt="shared_mask_pilot_v3",
    )

    assert config["config_sha256"] == canonical_hash(config, "config_sha256")
    execution = config["execution"]
    assert execution["quantization_workers"] == 2
    assert execution["cpu_threads_per_quantizer"] == 4
    assert execution["engine_build_workers"] == 1
    assert execution["inference_workers"] == execution["evaluation_workers"] == 3
    assert execution["gpu_admission"]["maximum_concurrent_pilot_processes"] == 3
    assert execution["timing_evidence"] == {
        "status": "inadmissible",
        "reason": "shared_gpu_and_concurrent_accuracy_inference",
        "required_remeasurement": "isolated_gpu_separate_attempt",
    }
    assert config["superseded_preflight"]["execution_sources_member_count"] == 20
    assert [block["baseline_fp8_engine_registry"] for block in config["blocks"]] == [
        "manifests/engines/voc_yolo11m_fp8_v1.json",
        "manifests/engines/cross_family_v1/voc_rtdetr_l_fp8.json",
        "manifests/engines/cross_family_v1/kitti_retinanet_r50_fpn_v2_fp8.json",
    ]


def test_historical_v1_config_is_not_admitted_as_v2() -> None:
    with pytest.raises(runner.PilotError, match="unsupported shared-mask pilot config"):
        validate_config(ROOT, ROOT / "configs/shared_mask_pilot_v1.json")


def test_parent_failure_must_be_immutable_and_scientifically_inadmissible(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    archive = tmp_path / "outputs/quarantine/shared_mask_pilot_v1_failed/execution_sources.tar.gz"
    archive.parent.mkdir(parents=True)
    archive.write_bytes(b"frozen-v1-sources")
    archive_sha = sha256_file(archive)
    manifest_path = archive.parent / "failure_manifest.json"
    manifest = write_complete_json(
        manifest_path,
        {
            "schema_version": 1,
            "attempt": "shared_mask_pilot_v1",
            "status": "terminal_scientific_gate_failure",
            "admissible_as_scientific_result": False,
            "execution_source_archive": {
                "path": str(archive.relative_to(tmp_path)),
                "sha256": archive_sha,
                "all_member_hashes_match_execution_package": True,
            },
        },
        self_hash_field="failure_manifest_sha256",
    )
    monkeypatch.setattr(
        runner,
        "PARENT_FAILURE_MANIFEST_CANONICAL_SHA256",
        manifest["failure_manifest_sha256"],
    )
    monkeypatch.setattr(
        runner, "PARENT_FAILURE_MANIFEST_FILE_SHA256", sha256_file(manifest_path)
    )
    monkeypatch.setattr(runner, "PARENT_EXECUTION_ARCHIVE_SHA256", archive_sha)
    config = {
        "parent_failure": {
            "failure_manifest": str(manifest_path.relative_to(tmp_path)),
            "execution_sources_archive": str(archive.relative_to(tmp_path)),
        }
    }

    observed = runner.validate_parent_failure(tmp_path, config)

    assert observed["attempt"] == "shared_mask_pilot_v1"
    assert observed["failure_manifest_canonical_sha256"] == manifest["failure_manifest_sha256"]
    assert observed["execution_sources_archive_sha256"] == archive_sha


@pytest.mark.parametrize(
    ("precision", "role", "byte_replay", "bypasses"),
    [
        ("fp8", "frozen_baseline_fp8_control", True, []),
        ("int8-entropy", "int8_aligned_to_frozen_fp8_compute_inputs", False, [{"x": 1}]),
    ],
)
def test_quantized_registry_requires_v2_contract_fields(
    tmp_path: Path,
    monkeypatch: pytest.MonkeyPatch,
    precision: str,
    role: str,
    byte_replay: bool,
    bypasses: list[dict[str, int]],
) -> None:
    output = tmp_path / f"{precision}.onnx"
    registry_path = tmp_path / f"{precision}.json"
    mask_file = tmp_path / "mask.json"
    output.write_bytes(b"onnx")
    registry_path.write_text("{}\n", encoding="utf-8")
    mask_file.write_text("{}\n", encoding="utf-8")
    mask = {
        "mask_sha256": "m" * 64,
        "nodes": [{"name": "node"}],
        "edge_policy": {"no_quantize_inputs": [], "edge_policy_sha256": "e" * 64},
        "compute_input_policy": {"policy_sha256": "p" * 64},
        "baseline_fp8": {"onnx_sha256": "b" * 64},
        "baseline_fp8_normalized_topology": {"compute_attachment_sha256": "t" * 64},
    }
    registry = {
        "precision": precision,
        "dataset": "voc",
        "model": "model",
        "node_mask_sha256": mask["mask_sha256"],
        "node_mask_file_sha256": "file-hash",
        "nodes_to_quantize_count": 1,
        "no_quantize_inputs_count": 0,
        "edge_policy_sha256": "e" * 64,
        "compute_input_policy_sha256": "p" * 64,
        "baseline_fp8_onnx_sha256": "b" * 64,
        "post_enforcement_compute_attachment_sha256": "t" * 64,
        "fp8_baseline_byte_replay": byte_replay,
        "shared_mask_role": role,
        "output_onnx_sha256": "b" * 64 if precision == "fp8" else "i" * 64,
        "bypassed_compute_inputs": bypasses,
        "bypassed_compute_inputs_count": len(bypasses),
    }
    monkeypatch.setattr(runner, "onnx_paths", lambda *_: (output, registry_path))
    monkeypatch.setattr(runner, "registry_onnx", lambda *_: (registry, output.resolve(), "r" * 64))
    monkeypatch.setattr(runner, "validate_mask_for_block", lambda *_: mask)
    monkeypatch.setattr(runner, "mask_path", lambda *_: mask_file)
    monkeypatch.setattr(runner, "sha256_file", lambda *_: "file-hash")

    assert runner.validate_quantized(
        tmp_path, {"dataset": "voc", "model": "model"}, precision
    ) == registry


def test_condition_ids_distinguish_matched_clean_and_corruption() -> None:
    block = {
        "dataset": "voc",
        "split": "val",
        "model": "rtdetr-l",
        "model_slug": "rtdetr_l",
    }

    assert shared_condition_id(block, "fp8", "clean", 0) == (
        "voc_val__rtdetr_l__fp8__shared-mask__q95-clean-s0"
    )
    assert shared_condition_id(block, "int8-entropy", "fog", 3) == (
        "voc_val__rtdetr_l__int8-entropy__shared-mask__fog-s3"
    )


def test_omega_is_difference_between_default_and_shared_interactions() -> None:
    result = analysis.contrast(
        default_int8_clean=60.0,
        default_fp8_clean=62.0,
        default_int8_corrupt=40.0,
        default_fp8_corrupt=43.0,
        shared_int8_clean=61.5,
        shared_fp8_clean=62.0,
        shared_int8_corrupt=42.4,
        shared_fp8_corrupt=43.0,
    )

    assert result == {
        "clean_gap_default": 2.0,
        "corrupt_gap_default": 3.0,
        "delta_e_default": 1.0,
        "clean_gap_shared": 0.5,
        "corrupt_gap_shared": pytest.approx(0.6),
        "delta_e_shared": pytest.approx(0.1),
        "omega_default_minus_shared": pytest.approx(0.9),
    }


def test_gpu_idle_gate_ignores_only_sunshine(monkeypatch: pytest.MonkeyPatch) -> None:
    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "3635790, /usr/bin/sunshine\n", ""
        ),
    )
    runner.assert_gpu_idle()

    monkeypatch.setattr(
        runner.subprocess,
        "run",
        lambda *args, **kwargs: subprocess.CompletedProcess(
            args[0], 0, "3635790, /usr/bin/sunshine\n42, python\n", ""
        ),
    )
    with pytest.raises(runner.PilotError, match="another compute process"):
        runner.assert_gpu_idle()


def test_v3_gpu_capacity_admits_three_inference_workers(
    monkeypatch: pytest.MonkeyPatch,
) -> None:
    monkeypatch.setattr(runner, "gpu_free_memory_mib", lambda: 20_000)
    policy = {
        "safety_margin_mib": 4096,
        "maximum_concurrent_pilot_processes": 3,
    }

    assert runner.effective_gpu_workers(
        policy, reservation_mib=4096, requested=3
    ) == 3

    monkeypatch.setattr(runner, "gpu_free_memory_mib", lambda: 7000)
    with pytest.raises(runner.PilotError, match="GPU capacity gate"):
        runner.effective_gpu_workers(policy, reservation_mib=4096, requested=3)


def test_bounded_scheduler_stops_dispatch_after_first_failure() -> None:
    started: list[int] = []
    barrier = threading.Barrier(2)

    def work(job: int) -> None:
        started.append(job)
        barrier.wait(timeout=2)
        if job == 0:
            raise runner.PilotError("scientific gate")

    with pytest.raises(runner.PilotError, match="scientific gate"):
        runner.run_bounded_jobs(
            list(range(6)), workers=2, work=work, stage="test"
        )

    # A sibling may finish and release one slot before the failing future is
    # observed; no dispatch is allowed after that failure becomes visible.
    assert {0, 1}.issubset(started)
    assert set(started).issubset({0, 1, 2})


def test_v3_shared_fp8_arm_aliases_exact_default_bundle(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    expected = runner.EvidenceBundle(
        tmp_path / "prediction.json",
        tmp_path / "input.json",
        tmp_path / "run.json",
        tmp_path / "metric.json",
    )
    monkeypatch.setattr(analysis, "ATTEMPT", "shared_mask_pilot_v3")
    monkeypatch.setattr(analysis, "locate_default_bundle", lambda *args, **kwargs: expected)
    monkeypatch.setattr(
        analysis,
        "shared_bundle",
        lambda *args, **kwargs: pytest.fail("V3 must not materialize an FP8 shared bundle"),
    )
    monkeypatch.setattr(analysis, "validate_bundle", lambda bundle, **kwargs: {"ok": True})

    bundle, validated = analysis._validated_arm(
        tmp_path,
        {"dataset": "voc", "split": "val", "model": "yolo11m"},
        policy="shared",
        precision="fp8",
        corruption="clean",
        severity=0,
        manifest_sha="m" * 64,
    )

    assert bundle == expected
    assert validated == {"ok": True}


@pytest.mark.parametrize("family", ["yolo", "cross_family"])
def test_default_fp8_engine_binding_reaches_exact_baseline_onnx_and_run(
    tmp_path: Path, family: str
) -> None:
    onnx = tmp_path / "baseline.fp8.onnx"
    onnx.write_bytes(b"frozen-fp8-onnx")
    onnx_registry = tmp_path / "baseline-fp8.json"
    write_complete_json(
        onnx_registry,
        {
            "dataset": "voc",
            "model": "model",
            "precision": "fp8",
            "output_onnx": str(onnx),
            "output_onnx_sha256": sha256_file(onnx),
        },
    )
    engine = tmp_path / "baseline.fp8.engine"
    engine.write_bytes(b"frozen-fp8-engine")
    engine_registry = tmp_path / "baseline-fp8-engine.json"
    write_complete_json(
        engine_registry,
        {
            "dataset": "voc",
            "model": "model",
            "precision": "fp8",
            "source_onnx_registry_sha256": sha256_file(onnx_registry),
            "source_onnx": str(onnx),
            "source_onnx_sha256": sha256_file(onnx),
            "engine": str(engine),
            "engine_sha256": sha256_file(engine),
            "engine_bytes": engine.stat().st_size,
        },
    )
    block = {
        "id": "block",
        "dataset": "voc",
        "model": "model",
        "family": family,
        "baseline_fp8_registry": str(onnx_registry),
        "baseline_fp8_engine_registry": str(engine_registry),
    }

    binding = runner.validate_default_fp8_engine_binding(tmp_path, block)
    run = {"engine_sha256": sha256_file(engine)}
    if family == "yolo":
        run["engine_path"] = str(engine)
    else:
        run["engine_registry"] = str(engine_registry)
        run["engine_registry_sha256"] = sha256_file(engine_registry)

    runner.validate_default_fp8_run_engine_binding(block, run, binding)
    assert binding["engine_sha256"] == sha256_file(engine)
    assert binding["onnx_sha256"] == sha256_file(onnx)

    run["engine_sha256"] = "0" * 64
    with pytest.raises(runner.PilotError, match="different engine SHA"):
        runner.validate_default_fp8_run_engine_binding(block, run, binding)


def test_all_present_invalid_metric_is_quarantined_and_recomputed(
    tmp_path: Path, monkeypatch: pytest.MonkeyPatch
) -> None:
    paths = [tmp_path / name for name in ("prediction.json", "input.json", "run.json", "metric.json")]
    for path in paths:
        path.write_text("{}\n", encoding="utf-8")
    bundle = runner.EvidenceBundle(*paths)
    block = {
        "id": "voc_yolo11m",
        "dataset": "voc",
        "split": "val",
        "model": "yolo11m",
        "family": "yolo",
        "annotations": "annotations.json",
    }
    calls: list[str] = []
    monkeypatch.setattr(runner, "shared_bundle", lambda *_: bundle)
    monkeypatch.setattr(runner, "validate_image_manifest", lambda *_args, **_kwargs: (tmp_path / "m.json", "m" * 64))
    monkeypatch.setattr(runner, "validate_engine", lambda *_: {"engine_sha256": "e" * 64})
    monkeypatch.setattr(runner, "_validate_inference_trio", lambda *_args, **_kwargs: None)
    results = iter([runner.PilotError("invalid metric"), {"status": "valid"}])

    def validate(*_args, **_kwargs):
        value = next(results)
        if isinstance(value, BaseException):
            raise value
        return value

    monkeypatch.setattr(runner, "validate_bundle", validate)

    def quarantine(_root, label, selected, reason):
        calls.append(f"quarantine:{label}:{reason}")
        for path in selected:
            if path.exists():
                path.unlink()

    monkeypatch.setattr(runner, "quarantine_partial", quarantine)

    def run_checked(_root, _command, *, stage, **_kwargs):
        calls.append(stage)
        assert stage == "evaluation"
        bundle.metric.write_text("{}\n", encoding="utf-8")

    monkeypatch.setattr(runner, "run_checked", run_checked)

    result = runner.ensure_condition(
        tmp_path, block, "int8-entropy", "clean", 0, gpu_policy={}
    )

    assert result == {"status": "valid"}
    assert calls[0].startswith("quarantine:metric__")
    assert calls[-1] == "evaluation"
    assert "inference" not in calls


def test_supervisor_completion_validator_rehashes_artifact_ledger() -> None:
    source = (ROOT / "src/supervise_shared_mask_pilot.sh").read_text(encoding="utf-8")
    assert 'artifacts = report.get("artifacts_sha256")' in source
    assert 'file_hash(resolved) != expected' in source
    assert 'report.get("reused_default_fp8_arms") == 39' in source
    assert "pilot_child_is_alive" in source
    assert "max_driver_retries=8" in source


def test_complete_json_is_immutable_and_self_hash_checked(tmp_path: Path) -> None:
    path = tmp_path / "record.json"
    value = write_complete_json(path, {"schema_version": 1, "attempt": ATTEMPT}, self_hash_field="record_sha256")

    assert read_complete_json(path, self_hash_field="record_sha256") == value
    with pytest.raises(runner.PilotError, match="overwrite immutable"):
        write_complete_json(path, {"schema_version": 1})

    path.write_text("{}\n", encoding="utf-8")
    with pytest.raises(runner.PilotError, match="invalid record_sha256"):
        read_complete_json(path, self_hash_field="record_sha256")


def test_bootstrap_cache_requires_drawwise_omega_identity(tmp_path: Path) -> None:
    npz = tmp_path / "cell.npz"
    np.savez_compressed(
        npz,
        schema_version=np.asarray(1),
        n_boot=np.asarray(3),
        seed=np.asarray(7),
        delta_e_default=np.asarray([1.0, 2.0, 3.0]),
        delta_e_shared=np.asarray([0.5, 1.5, 2.5]),
        omega=np.asarray([0.5, 0.5, 0.5]),
    )
    record = tmp_path / "cell.json"
    write_complete_json(
        record,
        {
            "block_id": "voc_yolo11m",
            "corruption": "fog",
            "severity": 1,
            "n_boot": 3,
            "config_sha256": "c" * 64,
            "draw_cache_sha256": sha256_file(npz),
            "source_artifacts_sha256": {},
        },
        self_hash_field="record_sha256",
    )

    _, arrays = analysis.validate_cell_cache(
        npz,
        record,
        block_id="voc_yolo11m",
        corruption="fog",
        severity=1,
        n_boot=3,
        config_sha256="c" * 64,
    )
    assert arrays["omega"].tolist() == [0.5, 0.5, 0.5]

    bad_npz = tmp_path / "bad.npz"
    np.savez_compressed(
        bad_npz,
        schema_version=np.asarray(1),
        n_boot=np.asarray(3),
        seed=np.asarray(7),
        delta_e_default=np.asarray([1.0, 2.0, 3.0]),
        delta_e_shared=np.asarray([0.5, 1.5, 2.5]),
        omega=np.asarray([0.0, 0.0, 0.0]),
    )
    bad_record = tmp_path / "bad.json"
    write_complete_json(
        bad_record,
        {
            "block_id": "voc_yolo11m",
            "corruption": "fog",
            "severity": 1,
            "n_boot": 3,
            "config_sha256": "c" * 64,
            "draw_cache_sha256": sha256_file(bad_npz),
            "source_artifacts_sha256": {},
        },
        self_hash_field="record_sha256",
    )
    with pytest.raises(runner.PilotError, match="Omega identity"):
        analysis.validate_cell_cache(
            bad_npz,
            bad_record,
            block_id="voc_yolo11m",
            corruption="fog",
            severity=1,
            n_boot=3,
            config_sha256="c" * 64,
        )
