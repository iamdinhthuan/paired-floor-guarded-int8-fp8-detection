from __future__ import annotations

import importlib

import numpy as np
import pytest


def module():
    import importlib.util
    assert importlib.util.find_spec("aggregate_fixed_universe") is not None, "macro aggregator is missing"
    return importlib.import_module("aggregate_fixed_universe")


def test_joint_aggregation_preserves_covariance_instead_of_averaging_intervals():
    # Anti-correlated cells have an exactly constant macro despite wide cell intervals.
    m = module()
    first = np.linspace(-0.1, 0.1, 10000)
    cells = [first if index % 2 == 0 else 0.04 - first for index in range(36)]
    result = m.summarize_macro(cells, [0.02] * 36)
    assert result["point_native_ap"] == pytest.approx(0.02)
    assert result["percentile_interval"] == pytest.approx([0.02, 0.02, 0.02])
    assert result["max_checkpoint_percentile_shift_native_ap"] == pytest.approx(0)


@pytest.mark.parametrize("problem", ["missing", "short", "nonfinite", "bad_point"])
def test_incomplete_or_invalid_macro_is_rejected(problem):
    m = module()
    cells = [np.zeros(10000) for _ in range(36)]
    points = [0.0] * 36
    if problem == "missing":
        cells.pop()
    elif problem == "short":
        cells[0] = np.zeros(2000)
    elif problem == "nonfinite":
        cells[0][0] = np.nan
    else:
        points[0] = float("inf")
    with pytest.raises(ValueError):
        m.summarize_macro(cells, points)


def test_schedule_or_image_order_mismatch_is_rejected():
    m = module()
    reference = {"schedule": {"schedule_identity_sha256": "one", "seed": 1},
                 "annotation": {"sha256": "annotation"},
                 "input_hashes": {arm: {"image_ids_sha256": "ordered"} for arm in m.ARM_NAMES}}
    import copy
    for field in ("schedule", "images", "annotation"):
        other = copy.deepcopy(reference)
        if field == "schedule":
            other["schedule"]["schedule_identity_sha256"] = "another"
        elif field == "images":
            other["input_hashes"]["fp8_corrupt"]["image_ids_sha256"] = "reordered"
        else:
            other["annotation"]["sha256"] = "other"
        with pytest.raises(ValueError):
            m.validate_joint_identity([reference, other])


def test_existing_destination_is_not_overwritten(tmp_path):
    m = module()
    destination = tmp_path / "result.json"
    destination.write_text("user evidence")
    with pytest.raises(FileExistsError):
        m.aggregate(tmp_path, tmp_path / "absent", destination)
    assert destination.read_text() == "user evidence"
