from __future__ import annotations

import importlib.util
from pathlib import Path

import pytest


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "src" / "train_retinanet_dataset.py"
    assert path.is_file(), "RetinaNet trainer has not been implemented"
    spec = importlib.util.spec_from_file_location("train_retinanet_dataset", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_yolo_box_conversion_clips_and_offsets_labels() -> None:
    module = load_module()
    boxes, labels = module.parse_yolo_rows(
        ["0 0.5 0.5 0.4 0.2", "2 0.0 0.0 0.2 0.2"], width=100, height=50
    )

    assert boxes.tolist() == [
        pytest.approx([30.0, 20.0, 70.0, 30.0]),
        pytest.approx([0.0, 0.0, 10.0, 5.0]),
    ]
    assert labels.tolist() == [1, 3]


def test_dataset_roots_preserve_all_voc_training_partitions(tmp_path: Path) -> None:
    module = load_module()
    roots = module.resolve_training_roots(
        tmp_path,
        ["images/train2012", "images/train2007", "images/val2012", "images/val2007"],
    )

    assert [item.name for item in roots] == ["train2012", "train2007", "val2012", "val2007"]
