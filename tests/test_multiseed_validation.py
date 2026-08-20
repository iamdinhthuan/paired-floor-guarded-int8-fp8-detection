from __future__ import annotations

import importlib.util
import hashlib
import json
import sys
import threading
import time
from copy import deepcopy
from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]
RUNNER_PATH = PROJECT_ROOT / "src" / "run_multiseed_validation.py"
CONFIG_PATH = PROJECT_ROOT / "configs" / "ivc_multiseed_yolo11m_s5_v1.json"


def write_json(path: Path, document: object) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def make_completed_condition(tmp_path: Path):
    runner = load_runner()
    config = deepcopy(runner.load_config(CONFIG_PATH))
    dataset = config["datasets"][0]
    dataset.update(
        {
            "expected_images": 2,
            "annotations": "annotations.json",
            "class_map": "classes.json",
            "clean_manifest": "input-manifest.json",
        }
    )
    annotations = tmp_path / dataset["annotations"]
    class_map = tmp_path / dataset["class_map"]
    runner_source = tmp_path / "src" / "coco_infer_trt.py"
    preprocess_source = tmp_path / "src" / "topic_c" / "coco_data.py"
    decoder_source = tmp_path / "src" / "topic_c" / "yolo_decode.py"
    for path, payload in (
        (runner_source, b"runner-source\n"),
        (preprocess_source, b"preprocess-source\n"),
        (decoder_source, b"decoder-source\n"),
    ):
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)
    write_json(
        annotations,
        {
            "images": [
                {"id": 1, "file_name": "one.jpg", "width": 10, "height": 10},
                {"id": 2, "file_name": "two.jpg", "width": 10, "height": 10},
            ],
            "annotations": [],
            "categories": [{"id": 1, "name": "object"}],
        },
    )
    write_json(class_map, {"class_to_category_id": {"0": 1}})

    manifest = tmp_path / "input-manifest.json"
    manifest_document = {
        "schema_version": 1,
        "dataset": "voc",
        "split": "val",
        "expected_image_ids": [1, 2],
        "records": [
            {"image_id": 1, "output_relpath": "one.jpg"},
            {"image_id": 2, "output_relpath": "two.jpg"},
        ],
    }
    manifest_document["manifest_sha256"] = runner.canonical_hash(
        manifest_document, "manifest_sha256"
    )
    write_json(manifest, manifest_document)
    manifest.with_suffix(".json.complete").write_text(
        manifest_document["manifest_sha256"] + "\n", encoding="utf-8"
    )

    calibration = tmp_path / "calibration.json"
    calibration_document = {"schema_version": 1, "dataset": "voc", "seed": 20260813}
    calibration_document["calibration_sha256"] = runner.canonical_hash(
        calibration_document, "calibration_sha256"
    )
    write_json(calibration, calibration_document)
    calibration.with_suffix(".json.complete").write_text(
        calibration_document["calibration_sha256"] + "\n", encoding="utf-8"
    )

    engine_registry = (
        tmp_path
        / "manifests"
        / "engines"
        / config["attempt"]
        / "voc__yolo11m__ts20260813__cs20260813__int8-entropy.json"
    )
    job = {
        "job_id": "inference__condition",
        "condition_id": "condition",
        "dataset": "voc",
        "training_seed": 20260813,
        "calibration_seed": 20260813,
        "precision": "int8-entropy",
        "calibration_manifest": str(calibration),
        "corruption": "codec_control",
        "severity": 0,
        "input_manifest": str(manifest),
        "engine_registry": str(engine_registry),
    }
    engine, build_log, derived_registry = runner._engine_paths(
        {"registry": str(engine_registry), "precision": "int8-entropy"},
        config,
        tmp_path,
    )
    assert derived_registry == engine_registry
    engine.parent.mkdir(parents=True, exist_ok=True)
    engine.write_bytes(b"engine")
    build_log.parent.mkdir(parents=True, exist_ok=True)
    build_log.write_text("PASSED\n", encoding="utf-8")
    engine_document = {
            "schema_version": 1,
            "dataset": "voc",
            "model": "yolo11m",
            "precision": "int8-entropy",
            "engine": str(engine),
            "engine_sha256": sha256_file(engine),
            "calibration_sha256": calibration_document["calibration_sha256"],
            "calibration_method": "entropy",
        }
    write_json(engine_registry, engine_document)
    engine_registry.with_suffix(".json.complete").write_text(
        sha256_file(engine_registry) + "\n", encoding="utf-8"
    )

    prediction, inputs, run_record, metric = runner._condition_paths(
        tmp_path, config["attempt"], job["condition_id"]
    )
    write_json(prediction, [])
    image_ids_hash = hashlib.sha256(b"[1,2]").hexdigest()
    write_json(
        inputs,
        {
            "schema_version": 1,
            "condition_id": "condition",
            "image_ids": [1, 2],
            "image_ids_sha256": image_ids_hash,
            "input_manifest_sha256": manifest_document["manifest_sha256"],
        },
    )
    write_json(
        run_record,
        {
            "schema_version": 2,
            "condition_id": "condition",
            "dataset": "voc",
            "split": "val",
            "model": "yolo11m",
            "precision": "int8-entropy",
            "calibrator": "entropy",
            "calibration_list_path": str(calibration.resolve()),
            "calibration_sha256": calibration_document["calibration_sha256"],
            "calibration_method": "entropy",
            "calibration_provenance": "verified",
            "corruption": "codec_control",
            "severity": 0,
            "engine_path": str(engine.resolve()),
            "engine_sha256": sha256_file(engine),
            "class_map_path": str(class_map.resolve()),
            "class_map_sha256": sha256_file(class_map),
            "annotation_sha256": sha256_file(annotations),
            "runner_sha256": sha256_file(runner_source),
            "preprocess_sha256": sha256_file(preprocess_source),
            "decoder_sha256": sha256_file(decoder_source),
            "input_manifest_sha256": manifest_document["manifest_sha256"],
            "input_image_ids_sha256": image_ids_hash,
            "prediction_sha256": sha256_file(prediction),
            "n_images": 2,
            "n_detections": 0,
        },
    )
    write_json(
        metric,
        {
            "schema_version": 1,
            "condition_id": "condition",
            "dataset": "voc",
            "split": "val",
            "model": "yolo11m",
            "precision": "int8-entropy",
            "corruption": "codec_control",
            "severity": 0,
            "n_images": 2,
            "stats": {"AP": 0.25},
            "prediction_sha256": sha256_file(prediction),
            "input_manifest_sha256": manifest_document["manifest_sha256"],
            "input_image_ids_sha256": image_ids_hash,
            "run_record_sha256": sha256_file(run_record),
        },
    )
    return (
        runner,
        config,
        job,
        (prediction, inputs, run_record, metric),
        manifest,
        runner_source,
    )


def load_runner():
    assert RUNNER_PATH.is_file(), "multi-seed runner has not been implemented"
    spec = importlib.util.spec_from_file_location("run_multiseed_validation", RUNNER_PATH)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_training_module():
    path = PROJECT_ROOT / "src" / "train_yolo_dataset.py"
    spec = importlib.util.spec_from_file_location("train_yolo_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_exact_frozen_factorial_grid() -> None:
    runner = load_runner()
    config = runner.load_config(CONFIG_PATH)
    jobs = runner.derive_jobs(config, PROJECT_ROOT)

    assert config["attempt"] == "ivc_multiseed_yolo11m_s5_v1"
    assert config["training_seeds"] == [20260807, 20260813, 20260814]
    assert config["calibration_seeds"] == [20260807, 20260813, 20260814]
    assert config["trt_root"] == "/home/thuan/traffic/third_party/TensorRT-11.1.0.106"
    assert {item["dataset"]: item["batch"] for item in config["datasets"]} == {
        "voc": 16,
        "kitti": 16,
        "tt100k": 4,
    }
    assert {stage: len(items) for stage, items in jobs.items()} == {
        "training": 6,
        "calibration": 9,
        "export": 9,
        "quantization": 54,
        "engine": 54,
        "inference": 270,
        "metric": 270,
        "direct_cell": 108,
    }
    assert sum(
        job["corruption"] == "codec_control" for job in jobs["inference"]
    ) == 54
    assert sum(
        job["corruption"] != "codec_control" for job in jobs["inference"]
    ) == 216
    assert len({job["job_id"] for items in jobs.values() for job in items}) == sum(
        len(items) for items in jobs.values()
    )


def test_config_rejects_an_extra_training_seed(tmp_path: Path) -> None:
    runner = load_runner()
    document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    document["training_seeds"].append(7)
    document["config_sha256"] = runner.canonical_hash(document, "config_sha256")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="training seeds"):
        runner.load_config(path)


def test_config_rejects_auto_batch_even_with_a_valid_self_hash(tmp_path: Path) -> None:
    runner = load_runner()
    document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    document["datasets"][0]["batch"] = -1
    document["config_sha256"] = runner.canonical_hash(document, "config_sha256")
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="batch"):
        runner.load_config(path)


def test_config_rejects_a_false_canonical_hash(tmp_path: Path) -> None:
    runner = load_runner()
    document = json.loads(CONFIG_PATH.read_text(encoding="utf-8"))
    document["config_sha256"] = "0" * 64
    path = tmp_path / "config.json"
    path.write_text(json.dumps(document), encoding="utf-8")

    with pytest.raises(ValueError, match="SHA-256"):
        runner.load_config(path)


def test_training_profile_resolves_frozen_dataset_specific_batch() -> None:
    training = load_training_module()
    assert hasattr(training, "resolve_profile_batch")

    profile = {"batch_by_dataset": {"voc": 16, "kitti": 16, "tt100k": 4}}
    assert training.resolve_profile_batch(profile, "voc") == 16
    assert training.resolve_profile_batch(profile, "tt100k") == 4
    with pytest.raises(SystemExit, match="batch"):
        training.resolve_profile_batch({"batch_by_dataset": {"voc": -1}}, "voc")


def test_training_profile_keeps_scalar_batch_compatibility() -> None:
    training = load_training_module()
    assert hasattr(training, "resolve_profile_batch")
    assert training.resolve_profile_batch({"batch": -1}, "voc") == -1


def test_training_profile_resolves_dataset_specific_image_size() -> None:
    training = load_training_module()
    assert hasattr(training, "resolve_profile_imgsz")
    profile = {"imgsz_by_dataset": {"voc": 640, "kitti": 640, "tt100k": 1280}}
    assert training.resolve_profile_imgsz(profile, "voc") == 640
    assert training.resolve_profile_imgsz(profile, "tt100k") == 1280


def test_ultralytics_training_selects_rtdetr_without_treating_it_as_yolo() -> None:
    training = load_training_module()

    assert training.detector_class_name("yolo11m") == "YOLO"
    assert training.detector_class_name("rtdetr-l") == "RTDETR"
    with pytest.raises(SystemExit, match="unsupported detector family"):
        training.detector_class_name("retinanet-r50-fpn-v2")


def test_resource_gate_rejects_foreign_gpu_processes_and_low_disk() -> None:
    runner = load_runner()
    good = {
        "gpu_name": "NVIDIA GeForce RTX 5090",
        "disk_free_bytes": 25 * 1024**3,
        "compute_processes": [],
    }
    runner.validate_resource_snapshot(good, minimum_free_gib=20)

    with pytest.raises(RuntimeError, match="compute process"):
        runner.validate_resource_snapshot(
            {**good, "compute_processes": [{"pid": 77, "name": "worker"}]},
            minimum_free_gib=20,
        )
    with pytest.raises(RuntimeError, match="free disk"):
        runner.validate_resource_snapshot(
            {**good, "disk_free_bytes": 19 * 1024**3}, minimum_free_gib=20
        )


def test_trt_runtime_gate_requires_frozen_layout_and_executable_hash(tmp_path: Path) -> None:
    runner = load_runner()
    trt_root = tmp_path / "TensorRT-11.1.0.106"

    with pytest.raises(RuntimeError, match="bin/lib layout"):
        runner.validate_trt_runtime(trt_root)

    executable = trt_root / "bin" / "trtexec"
    executable.parent.mkdir(parents=True)
    executable.write_bytes(b"wrong trtexec\n")
    (trt_root / "lib").mkdir()
    with pytest.raises(RuntimeError, match="SHA-256"):
        runner.validate_trt_runtime(trt_root)

    assert runner.validate_trt_runtime(
        trt_root, expected_sha256=sha256_file(executable)
    ) == {
        "trtexec": str(executable.resolve()),
        "trtexec_sha256": sha256_file(executable),
        "trt_lib": str((trt_root / "lib").resolve()),
    }


def test_training_command_uses_seed_profile_and_fixed_dataset_geometry() -> None:
    runner = load_runner()
    config = runner.load_config(CONFIG_PATH)
    job = next(
        item
        for item in runner.derive_jobs(config, PROJECT_ROOT)["training"]
        if item["dataset"] == "tt100k" and item["training_seed"] == 20260813
    )
    command = runner.training_command(job, config, PROJECT_ROOT)

    assert command[0].endswith("python") or "python" in Path(command[0]).name
    assert command[command.index("--dataset") + 1] == "tt100k"
    assert command[command.index("--model") + 1] == "yolo11m"
    assert command[command.index("--run-id") + 1] == (
        "ivc_multiseed_yolo11m_s5_v1__tt100k__yolo11m__ts20260813"
    )
    profile = Path(command[command.index("--profile") + 1])
    assert profile.name == "ivc_multiseed_yolo11m_s20260813_v1.json"


def test_leaf_commands_bind_crossed_seeds_and_frozen_inputs() -> None:
    runner = load_runner()
    config = runner.load_config(CONFIG_PATH)
    jobs = runner.derive_jobs(config, PROJECT_ROOT)

    calibration = next(
        item for item in jobs["calibration"]
        if item["dataset"] == "voc" and item["calibration_seed"] == 20260813
    )
    calibration_command = runner.calibration_command(calibration, config, PROJECT_ROOT)
    assert calibration_command[calibration_command.index("--n-images") + 1] == "512"
    assert calibration_command[calibration_command.index("--seed") + 1] == "20260813"

    export = next(
        item for item in jobs["export"]
        if item["dataset"] == "tt100k" and item["training_seed"] == 20260814
    )
    export_command = runner.export_command(export, config, PROJECT_ROOT)
    assert export_command[export_command.index("--imgsz") + 1] == "1280"
    assert export_command[export_command.index("--training-registry") + 1].endswith(
        "ivc_multiseed_yolo11m_s5_v1__tt100k__yolo11m__ts20260814.json"
    )

    quantization = next(
        item for item in jobs["quantization"]
        if item["dataset"] == "kitti"
        and item["training_seed"] == 20260813
        and item["calibration_seed"] == 20260814
        and item["precision"] == "fp8"
    )
    quantization_command = runner.quantization_command(
        quantization, config, PROJECT_ROOT
    )
    assert quantization_command[quantization_command.index("--mode") + 1] == "fp8"
    assert quantization_command[quantization_command.index("--imgsz") + 1] == "640"
    assert "s20260814" in quantization_command[
        quantization_command.index("--calibration-list") + 1
    ]

    engine = next(
        item for item in jobs["engine"]
        if item["dataset"] == "kitti"
        and item["training_seed"] == 20260813
        and item["calibration_seed"] == 20260814
        and item["precision"] == "fp8"
    )
    engine_command = runner.engine_command(engine, config, PROJECT_ROOT)
    assert engine_command[engine_command.index("--precision") + 1] == "fp8"
    assert engine_command[engine_command.index("--workspace") + 1] == "4096M"
    assert "cs20260814__fp8.json" in engine_command[
        engine_command.index("--onnx-registry") + 1
    ]


def test_inference_and_metric_commands_select_codec_and_tt100k_height_evaluator() -> None:
    runner = load_runner()
    config = runner.load_config(CONFIG_PATH)
    jobs = runner.derive_jobs(config, PROJECT_ROOT)
    clean = next(
        item for item in jobs["inference"]
        if item["dataset"] == "voc"
        and item["corruption"] == "codec_control"
        and item["training_seed"] == 20260807
        and item["calibration_seed"] == 20260807
        and item["precision"] == "int8-entropy"
    )
    clean_command = runner.inference_command(clean, config, PROJECT_ROOT)
    assert clean_command[clean_command.index("--manifest-cache-root") + 1].endswith(
        "data/codec_control/voc"
    )
    assert clean_command[clean_command.index("--severity") + 1] == "0"

    corrupt = next(
        item for item in jobs["inference"]
        if item["dataset"] == "tt100k"
        and item["corruption"] == "motion_blur"
        and item["training_seed"] == 20260814
        and item["calibration_seed"] == 20260813
        and item["precision"] == "fp8"
    )
    corrupt_command = runner.inference_command(corrupt, config, PROJECT_ROOT)
    assert corrupt_command[corrupt_command.index("--manifest-cache-root") + 1].endswith(
        "data/tt100k_c"
    )
    assert corrupt_command[corrupt_command.index("--severity") + 1] == "5"
    metric_command = runner.metric_command(corrupt, config, PROJECT_ROOT)
    assert Path(metric_command[1]).name == "tt100k_eval.py"


def test_calibration_resume_validation_rejects_wrong_seed(tmp_path: Path) -> None:
    runner = load_runner()
    config = runner.load_config(CONFIG_PATH)
    job = {
        "job_id": "calibration__voc__cs20260813",
        "dataset": "voc",
        "calibration_seed": 20260813,
        "manifest": str(tmp_path / "calibration.json"),
        "existing": False,
    }
    document = {
        "schema_version": 1,
        "dataset": "voc",
        "split": "train",
        "n_images": 512,
        "seed": 20260814,
        "records": [
            {"source_relpath": f"image-{index}.jpg", "sha256": "a" * 64}
            for index in range(512)
        ],
    }
    document["calibration_sha256"] = runner.canonical_hash(
        document, "calibration_sha256"
    )
    path = Path(job["manifest"])
    path.write_text(json.dumps(document), encoding="utf-8")
    path.with_suffix(".json.complete").write_text(
        document["calibration_sha256"] + "\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="seed"):
        runner.validate_calibration_job(job, config, tmp_path, verify_images=False)


def test_stage_completion_report_rejects_changed_registry_bytes(tmp_path: Path) -> None:
    runner = load_runner()
    registry = tmp_path / "registry.json"
    registry.write_text('{"value": 1}\n', encoding="utf-8")
    report = tmp_path / "complete.json"
    runner.write_stage_report(
        report,
        attempt="attempt",
        stage="export",
        config_sha256="c" * 64,
        artifacts=[registry],
        root=tmp_path,
    )
    runner.validate_stage_report(
        report,
        attempt="attempt",
        stage="export",
        config_sha256="c" * 64,
        artifacts=[registry],
        root=tmp_path,
    )

    registry.write_text('{"value": 2}\n', encoding="utf-8")
    with pytest.raises(RuntimeError, match="artifact hash"):
        runner.validate_stage_report(
            report,
            attempt="attempt",
            stage="export",
            config_sha256="c" * 64,
            artifacts=[registry],
            root=tmp_path,
        )


def test_inference_resume_accepts_only_a_fully_bound_output_triple(tmp_path: Path) -> None:
    runner, config, job, paths, _, _ = make_completed_condition(tmp_path)

    assert runner.validate_inference_job(job, config, tmp_path) is True

    paths[0].write_text("[]\n ", encoding="utf-8")
    with pytest.raises(RuntimeError, match="prediction hash"):
        runner.validate_inference_job(job, config, tmp_path)


def test_inference_resume_rejects_a_changed_input_manifest(tmp_path: Path) -> None:
    runner, config, job, _, manifest, _ = make_completed_condition(tmp_path)
    document = json.loads(manifest.read_text(encoding="utf-8"))
    document["records"].reverse()
    document["expected_image_ids"] = [2, 1]
    document["manifest_sha256"] = runner.canonical_hash(document, "manifest_sha256")
    write_json(manifest, document)
    manifest.with_suffix(".json.complete").write_text(
        document["manifest_sha256"] + "\n", encoding="utf-8"
    )

    with pytest.raises(RuntimeError, match="input"):
        runner.validate_inference_job(job, config, tmp_path)


def test_pending_inference_validates_parent_manifest_before_launch(tmp_path: Path) -> None:
    runner, config, job, paths, manifest, _ = make_completed_condition(tmp_path)
    for path in paths[:3]:
        path.unlink()
    manifest.with_suffix(".json.complete").write_text("0" * 64 + "\n", encoding="utf-8")

    with pytest.raises(RuntimeError, match="input manifest"):
        runner.validate_inference_job(job, config, tmp_path)


def test_metric_resume_accepts_only_exact_inference_provenance(tmp_path: Path) -> None:
    runner, config, job, paths, _, _ = make_completed_condition(tmp_path)

    assert runner.validate_metric_job(job, config, tmp_path) is True

    metric = json.loads(paths[3].read_text(encoding="utf-8"))
    metric["run_record_sha256"] = "0" * 64
    write_json(paths[3], metric)
    with pytest.raises(RuntimeError, match="metric provenance"):
        runner.validate_metric_job(job, config, tmp_path)


def test_inference_resume_rejects_a_partial_output_triple(tmp_path: Path) -> None:
    runner, config, job, paths, _, _ = make_completed_condition(tmp_path)
    paths[2].unlink()

    with pytest.raises(RuntimeError, match="partial output triple"):
        runner.validate_inference_job(job, config, tmp_path)


def test_inference_resume_rejects_changed_runner_source(tmp_path: Path) -> None:
    runner, config, job, _, _, runner_source = make_completed_condition(tmp_path)
    assert runner.validate_inference_job(job, config, tmp_path) is True

    runner_source.write_bytes(b"changed-runner-source\n")
    with pytest.raises(RuntimeError, match="source provenance"):
        runner.validate_inference_job(job, config, tmp_path)


def test_failed_evidence_child_stops_and_preserves_completed_bytes(tmp_path: Path) -> None:
    runner = load_runner()
    completed = tmp_path / "completed.json"
    completed.write_bytes(b"validated-before-failure\n")
    forbidden = tmp_path / "must-not-run"
    commands = [
        [sys.executable, "-c", "raise SystemExit(7)"],
        [sys.executable, "-c", f"from pathlib import Path; Path({str(forbidden)!r}).touch()"],
    ]

    with pytest.raises(RuntimeError, match="failed"):
        for index, command in enumerate(commands):
            runner.run_evidence_child(
                command,
                phase="inference",
                condition_id=f"condition-{index}",
                root=tmp_path,
            )

    assert completed.read_bytes() == b"validated-before-failure\n"
    assert not forbidden.exists()
    ledger = tmp_path / "outputs" / "logs" / "ivc_multiseed_yolo11m_s5_v1.evidence.failures.jsonl"
    rows = [json.loads(line) for line in ledger.read_text(encoding="utf-8").splitlines()]
    assert len(rows) == 1
    assert rows[0]["condition_id"] == "condition-0"
    assert rows[0]["returncode"] == 7


def test_evidence_completion_artifact_grid_is_exact_and_parent_bound() -> None:
    runner = load_runner()
    config = runner.load_config(CONFIG_PATH)

    artifacts = runner.evidence_artifact_paths(config, PROJECT_ROOT)

    assert len(artifacts["inference_outputs"]) == 810
    assert len(set(artifacts["inference_outputs"])) == 810
    assert len(artifacts["input_manifests"]) == 15
    assert len(set(artifacts["input_manifests"])) == 15
    assert len(artifacts["metrics"]) == 270
    assert len(set(artifacts["metrics"])) == 270
    assert artifacts["engine_report"].name.endswith("_engine_complete.json")
    assert artifacts["inference_report"].name.endswith("_inference_complete.json")


def test_evidence_execution_pairs_all_270_inference_and_metric_jobs() -> None:
    runner = load_runner()
    config = runner.load_config(CONFIG_PATH)
    jobs = runner.derive_jobs(config, PROJECT_ROOT)

    pairs = runner.evidence_execution_order(jobs)

    assert len(pairs) == 270
    assert all(
        inference["condition_id"] == metric["condition_id"]
        for inference, metric in pairs
    )
    broken = {**jobs, "metric": list(jobs["metric"])}
    broken["metric"][0], broken["metric"][1] = broken["metric"][1], broken["metric"][0]
    with pytest.raises(RuntimeError, match="pairing"):
        runner.evidence_execution_order(broken)


def test_bounded_evidence_pool_runs_each_condition_once_and_respects_cap() -> None:
    runner = load_runner()
    active = 0
    peak = 0
    seen: list[str] = []
    lock = threading.Lock()

    def work(state: tuple[dict, dict, bool, bool]) -> str:
        nonlocal active, peak
        condition_id = state[0]["condition_id"]
        with lock:
            active += 1
            peak = max(peak, active)
        time.sleep(0.02)
        with lock:
            seen.append(condition_id)
            active -= 1
        return condition_id

    states = [
        ({"condition_id": f"condition-{index}"}, {"condition_id": f"condition-{index}"}, False, False)
        for index in range(7)
    ]
    completed = runner.run_bounded_evidence_states(states, workers=4, work=work)

    assert sorted(completed) == [f"condition-{index}" for index in range(7)]
    assert sorted(seen) == sorted(completed)
    assert peak == 4


@pytest.mark.parametrize("workers", [0, 9, True, "8"])
def test_bounded_evidence_pool_rejects_invalid_worker_count(workers) -> None:
    runner = load_runner()
    with pytest.raises(ValueError, match="evidence workers"):
        runner.run_bounded_evidence_states([], workers=workers, work=lambda state: state)
