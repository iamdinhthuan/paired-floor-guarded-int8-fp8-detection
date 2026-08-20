from __future__ import annotations

import importlib.util
import json
import os
from pathlib import Path
import subprocess
import sys

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "src" / "freeze_confirmatory_splits.py"
    assert path.is_file(), "confirmatory split freezer has not been implemented"
    spec = importlib.util.spec_from_file_location("freeze_confirmatory_splits", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_training_module():
    path = ROOT / "src" / "train_yolo_dataset.py"
    spec = importlib.util.spec_from_file_location("train_yolo_dataset_confirmatory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_queue_module():
    path = ROOT / "src" / "run_training_queue.py"
    spec = importlib.util.spec_from_file_location("run_training_queue_confirmatory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def load_script(name: str):
    path = ROOT / "src" / f"{name}.py"
    spec = importlib.util.spec_from_file_location(f"{name}_confirmatory", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def add_example(root: Path, split: str, stem: str, classes: tuple[int, ...]) -> Path:
    image = root / "images" / split / f"{stem}.jpg"
    label = root / "labels" / split / f"{stem}.txt"
    image.parent.mkdir(parents=True, exist_ok=True)
    label.parent.mkdir(parents=True, exist_ok=True)
    image.write_bytes(f"image:{split}:{stem}".encode())
    label.write_text(
        "".join(f"{class_id} 0.5 0.5 0.2 0.2\n" for class_id in classes),
        encoding="utf-8",
    )
    return image


def acquisition_registry(tmp_path: Path, dataset: str, dataset_root: Path) -> Path:
    path = tmp_path / f"{dataset}_acquisition.json"
    path.write_text(
        json.dumps({"schema_version": 1, "dataset": dataset, "dataset_root": str(dataset_root.resolve())}) + "\n",
        encoding="utf-8",
    )
    digest = __import__("hashlib").sha256(path.read_bytes()).hexdigest()
    path.with_suffix(path.suffix + ".complete").write_text(digest + "\n", encoding="utf-8")
    return path


def test_voc_uses_prospectively_locked_official_partitions_and_is_reproducible(tmp_path: Path) -> None:
    module = load_module()
    dataset_root = tmp_path / "VOC"
    add_example(dataset_root, "train2007", "a", (0,))
    add_example(dataset_root, "train2012", "b", (1,))
    add_example(dataset_root, "val2007", "c", (0, 1))
    add_example(dataset_root, "val2012", "d", (0,))
    add_example(dataset_root, "val2012", "e", (1,))
    registry = acquisition_registry(tmp_path, "voc", dataset_root)

    first = module.freeze_confirmatory_split(
        dataset="voc",
        dataset_root=dataset_root,
        names={0: "zero", 1: "one"},
        acquisition_registry=registry,
        output_root=tmp_path / "first",
        seed=20260818,
    )
    second = module.freeze_confirmatory_split(
        dataset="voc",
        dataset_root=dataset_root,
        names={0: "zero", 1: "one"},
        acquisition_registry=registry,
        output_root=tmp_path / "second",
        seed=20260818,
    )

    assert first["partition_sources"] == {
        "train": ["images/train2007", "images/train2012"],
        "selection": ["images/val2007"],
        "final": ["images/val2012"],
    }
    assert {name: split["images"] for name, split in first["splits"].items()} == {
        "train": 2,
        "selection": 1,
        "final": 2,
    }
    assert first["content_sha256"] == second["content_sha256"]
    assert module.validate_split_bundle(Path(first["report_path"]))["content_sha256"] == first["content_sha256"]


def test_kitti_hash_partition_covers_every_class_in_train_selection_and_final(tmp_path: Path) -> None:
    module = load_module()
    dataset_root = tmp_path / "kitti"
    train_classes = ((0,), (0,), (1,), (1,), (2,), (2,), (0, 1), (1, 2), (0, 2), (0, 1, 2))
    for index, classes in enumerate(train_classes):
        add_example(dataset_root, "train", f"train-{index:02d}", classes)
    for index, classes in enumerate(((0,), (1,), (2,), (0, 1, 2))):
        add_example(dataset_root, "val", f"val-{index:02d}", classes)
    registry = acquisition_registry(tmp_path, "kitti", dataset_root)

    report = module.freeze_confirmatory_split(
        dataset="kitti",
        dataset_root=dataset_root,
        names={0: "zero", 1: "one", 2: "two"},
        acquisition_registry=registry,
        output_root=tmp_path / "bundle",
        seed=20260818,
        final_fraction=0.3,
    )

    assert report["splits"]["final"]["images"] == 3
    assert set(report["splits"]["train"]["class_image_counts"]) == {"0", "1", "2"}
    assert set(report["splits"]["selection"]["class_image_counts"]) == {"0", "1", "2"}
    assert set(report["splits"]["final"]["class_image_counts"]) == {"0", "1", "2"}
    train = set(Path(report["splits"]["train"]["list_path"]).read_text().splitlines())
    selection = set(Path(report["splits"]["selection"]["list_path"]).read_text().splitlines())
    final = set(Path(report["splits"]["final"]["list_path"]).read_text().splitlines())
    assert not (train & selection or train & final or selection & final)


def test_split_validation_rejects_resolved_path_overlap(tmp_path: Path) -> None:
    module = load_module()
    dataset_root = tmp_path / "dataset"
    image = add_example(dataset_root, "train", "a", (0,))
    final = add_example(dataset_root, "final", "b", (0,))

    with pytest.raises(SystemExit, match="partition overlap"):
        module.validate_partitions({"train": [image], "selection": [image], "final": [final]})


def test_split_bundle_rejects_list_mutation(tmp_path: Path) -> None:
    module = load_module()
    dataset_root = tmp_path / "VOC"
    for split, stem in (("train2007", "a"), ("train2012", "b"), ("val2007", "c"), ("val2012", "d")):
        add_example(dataset_root, split, stem, (0,))
    registry = acquisition_registry(tmp_path, "voc", dataset_root)
    report = module.freeze_confirmatory_split(
        dataset="voc",
        dataset_root=dataset_root,
        names={0: "zero"},
        acquisition_registry=registry,
        output_root=tmp_path / "bundle",
        seed=20260818,
    )
    train_list = Path(report["splits"]["train"]["list_path"])
    train_list.write_text(train_list.read_text() + "/forged/image.jpg\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="list hash mismatch"):
        module.validate_split_bundle(Path(report["report_path"]))


def test_training_provenance_accepts_only_the_hash_valid_confirmatory_bundle(tmp_path: Path) -> None:
    freezer = load_module()
    training = load_training_module()
    dataset_root = tmp_path / "VOC"
    for split, stem in (("train2007", "a"), ("train2012", "b"), ("val2007", "c"), ("val2012", "d")):
        add_example(dataset_root, split, stem, (0,))
    registry = acquisition_registry(tmp_path, "voc", dataset_root)
    report = freezer.freeze_confirmatory_split(
        dataset="voc",
        dataset_root=dataset_root,
        names={0: "zero"},
        acquisition_registry=registry,
        output_root=tmp_path / "bundle",
        seed=20260818,
    )

    provenance = training.validate_training_data_provenance(
        dataset="voc",
        data_yaml=Path(report["yaml_path"]),
        acquisition_registry=registry,
        split_registry=Path(report["report_path"]),
    )

    assert provenance["mode"] == "prospectively_locked_confirmatory_resplit"
    assert provenance["split_content_sha256"] == report["content_sha256"]


def test_training_provenance_rejects_mutated_confirmatory_list(tmp_path: Path) -> None:
    freezer = load_module()
    training = load_training_module()
    dataset_root = tmp_path / "VOC"
    for split, stem in (("train2007", "a"), ("train2012", "b"), ("val2007", "c"), ("val2012", "d")):
        add_example(dataset_root, split, stem, (0,))
    registry = acquisition_registry(tmp_path, "voc", dataset_root)
    report = freezer.freeze_confirmatory_split(
        dataset="voc",
        dataset_root=dataset_root,
        names={0: "zero"},
        acquisition_registry=registry,
        output_root=tmp_path / "bundle",
        seed=20260818,
    )
    Path(report["splits"]["final"]["list_path"]).write_text("/forged.jpg\n", encoding="utf-8")

    with pytest.raises(SystemExit, match="list hash mismatch"):
        training.validate_training_data_provenance(
            dataset="voc",
            data_yaml=Path(report["yaml_path"]),
            acquisition_registry=registry,
            split_registry=Path(report["report_path"]),
        )


def test_training_queue_resolves_prespecified_confirmatory_inputs(tmp_path: Path) -> None:
    queue_module = load_queue_module()
    for relative in (
        "manifests/splits/voc/voc.yaml",
        "manifests/splits/voc/voc.json",
        "manifests/datasets/voc.json",
    ):
        path = tmp_path / relative
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text("{}\n", encoding="utf-8")
    queue = {
        "datasets": {
            "voc": {
                "data_yaml": "manifests/splits/voc/voc.yaml",
                "split_registry": "manifests/splits/voc/voc.json",
                "acquisition_registry": "manifests/datasets/voc.json",
            }
        }
    }

    resolved = queue_module.resolve_job_inputs(tmp_path, queue, "voc")

    assert resolved == {
        "data_yaml": (tmp_path / "manifests/splits/voc/voc.yaml").resolve(),
        "split_registry": (tmp_path / "manifests/splits/voc/voc.json").resolve(),
        "acquisition_registry": (tmp_path / "manifests/datasets/voc.json").resolve(),
    }


def test_confirmatory_training_profile_and_queue_are_single_process_and_frozen() -> None:
    profile_path = ROOT / "configs" / "training" / "voc_kitti_confirmatory_yolo11_nmx_v1.json"
    queue_path = ROOT / "configs" / "training" / "voc_kitti_confirmatory_yolo11_nmx_queue_v1.json"
    assert profile_path.is_file(), "confirmatory training profile is absent"
    assert queue_path.is_file(), "confirmatory training queue is absent"
    profile = json.loads(profile_path.read_text(encoding="utf-8"))
    queue = json.loads(queue_path.read_text(encoding="utf-8"))

    assert profile["models"] == ["yolo11n", "yolo11m", "yolo11x"]
    assert profile["workers"] == 4
    assert profile["batch_by_dataset"] == {"voc": 16, "kitti": 16}
    assert profile["seed"] == 20260818
    assert queue["profile"] == "configs/training/voc_kitti_confirmatory_yolo11_nmx_v1.json"
    assert len(queue["jobs"]) == 6
    assert len({job["run_id"] for job in queue["jobs"]}) == 6
    assert set(queue["datasets"]) == {"voc", "kitti"}
    assert all("split_registry" in item for item in queue["datasets"].values())
    assert queue["retain_only_best"] is True


def test_validate_only_cli_does_not_require_generation_arguments(tmp_path: Path) -> None:
    freezer = load_module()
    dataset_root = tmp_path / "VOC"
    for split, stem in (("train2007", "a"), ("train2012", "b"), ("val2007", "c"), ("val2012", "d")):
        add_example(dataset_root, split, stem, (0,))
    registry = acquisition_registry(tmp_path, "voc", dataset_root)
    report = freezer.freeze_confirmatory_split(
        dataset="voc",
        dataset_root=dataset_root,
        names={0: "zero"},
        acquisition_registry=registry,
        output_root=tmp_path / "bundle",
        seed=20260818,
    )

    completed = subprocess.run(
        [sys.executable, str(ROOT / "src" / "freeze_confirmatory_splits.py"), "--validate-only", report["report_path"]],
        env={**os.environ, "PYTHONPATH": str(ROOT / "src")},
        text=True,
        capture_output=True,
    )

    assert completed.returncode == 0, completed.stderr
    assert json.loads(completed.stdout)["content_sha256"] == report["content_sha256"]


@pytest.mark.parametrize("script_name", ["build_train_calibration_list", "build_yolo_dataset_eval_assets"])
def test_confirmatory_consumers_resolve_a_frozen_image_list(script_name: str, tmp_path: Path) -> None:
    module = load_script(script_name)
    dataset_root = tmp_path / "dataset"
    first = add_example(dataset_root, "train", "a", (0,))
    second = add_example(dataset_root, "train", "b", (0,))
    image_list = tmp_path / "train.txt"
    image_list.write_text(f"{second.resolve()}\n{first.resolve()}\n", encoding="utf-8")

    images = module.split_images(dataset_root, str(image_list))

    assert images == sorted([first.resolve(), second.resolve()], key=lambda path: str(path))


def test_calibration_and_eval_provenance_accept_the_locked_split_bundle(tmp_path: Path) -> None:
    freezer = load_module()
    calibration = load_script("build_train_calibration_list")
    evaluation = load_script("build_yolo_dataset_eval_assets")
    dataset_root = tmp_path / "VOC"
    for split, stem in (("train2007", "a"), ("train2012", "b"), ("val2007", "c"), ("val2012", "d")):
        add_example(dataset_root, split, stem, (0,))
    registry = acquisition_registry(tmp_path, "voc", dataset_root)
    report = freezer.freeze_confirmatory_split(
        dataset="voc", dataset_root=dataset_root, names={0: "zero"},
        acquisition_registry=registry, output_root=tmp_path / "bundle", seed=20260818,
    )

    for module in (calibration, evaluation):
        provenance = module.validate_dataset_provenance(
            dataset="voc",
            data_yaml=Path(report["yaml_path"]),
            acquisition_registry=registry,
            split_registry=Path(report["report_path"]),
        )
        assert provenance["split_content_sha256"] == report["content_sha256"]


def test_confirmatory_protocol_config_is_self_hashed_and_exact() -> None:
    path = ROOT / "configs" / "confirmatory_voc_kitti_v1.json"
    assert path.is_file(), "confirmatory protocol config is absent"
    document = json.loads(path.read_text(encoding="utf-8"))
    payload = {key: value for key, value in document.items() if key != "config_sha256"}
    expected = __import__("hashlib").sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    ).hexdigest()

    assert document["config_sha256"] == expected
    assert document["models"] == ["yolo11n", "yolo11m", "yolo11x"]
    assert document["corruptions"] == ["gaussian_noise", "motion_blur", "fog", "jpeg"]
    assert document["severities"] == [1, 3, 5]
    assert document["reference"]["tf32_enabled"] is False
    assert document["smoke"] == {"dataset": "voc", "model": "yolo11n", "corruption": "gaussian_noise", "severity": 3}
    assert {name: item["final_images"] for name, item in document["datasets"].items()} == {
        "voc": 5823,
        "kitti": 1197,
    }
