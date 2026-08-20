from __future__ import annotations

import pytest

pytest.importorskip("pycocotools")

from topic_c.tt100k_height import HEIGHT_GROUPS, accumulated_height_stats, height_evaluation


def fixture() -> tuple[dict, list[dict], list[int]]:
    images = [{"id": index, "file_name": f"{index}.jpg", "width": 200, "height": 200} for index in (1, 2, 3, 4)]
    annotations = []
    annotation_id = 1
    for image_id in (1, 2):
        annotations.append({"id": annotation_id, "image_id": image_id, "category_id": 1,
                            "bbox": [10, 10, 10, 10], "area": 100, "iscrowd": 0})
        annotation_id += 1
    for image_id in (3, 4):
        annotations.append({"id": annotation_id, "image_id": image_id, "category_id": 1,
                            "bbox": [10, 10, 60, 60], "area": 3600, "iscrowd": 0})
        annotation_id += 1
    ground_truth = {"images": images, "annotations": annotations,
                    "categories": [{"id": 1, "name": "sign"}]}
    predictions = [
        {"image_id": 3, "category_id": 1, "bbox": [10, 10, 60, 60], "score": 0.99},
        {"image_id": 4, "category_id": 1, "bbox": [10, 10, 60, 60], "score": 0.98},
    ]
    return ground_truth, predictions, [1, 2, 3, 4]


def test_height_ranges_separate_small_and_large_objects() -> None:
    ground_truth, predictions, image_ids = fixture()
    bins = [(name, lower, upper) for name, (lower, upper) in HEIGHT_GROUPS.items()]
    stats = accumulated_height_stats(height_evaluation(ground_truth, predictions, image_ids, bins))

    assert stats["small_like"]["AP"] == pytest.approx(0.0)
    assert stats["large_like"]["AP"] == pytest.approx(1.0)
