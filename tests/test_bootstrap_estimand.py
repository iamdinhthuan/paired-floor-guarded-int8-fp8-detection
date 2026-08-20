"""Contracts for ordinary and fixed-class-universe detection bootstraps."""

import sys
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

_module_names = ("pycocotools", "pycocotools.coco", "pycocotools.cocoeval")
_previous_modules = {name: sys.modules.get(name) for name in _module_names}
pycocotools = ModuleType("pycocotools")
pycocotools_coco = ModuleType("pycocotools.coco")
pycocotools_cocoeval = ModuleType("pycocotools.cocoeval")
pycocotools_coco.COCO = object
pycocotools_cocoeval.COCOeval = object
sys.modules["pycocotools"] = pycocotools
sys.modules["pycocotools.coco"] = pycocotools_coco
sys.modules["pycocotools.cocoeval"] = pycocotools_cocoeval
try:
    from paired_bootstrap import accumulate_ap
finally:
    for _name, _previous in _previous_modules.items():
        if _previous is None:
            sys.modules.pop(_name, None)
        else:
            sys.modules[_name] = _previous
from fixed_universe_bootstrap import (
    accumulate_prepared_ap,
    accumulate_ap_weighted,
    fixed_category_area_mask,
    prepare_weighted_ap,
)


def _entry(scores: list[float], matches: list[int], positives: int) -> dict:
    return {
        "dtScores": np.asarray(scores, dtype=float),
        "dtMatches": np.asarray([matches], dtype=float),
        "dtIgnore": np.zeros((1, len(scores)), dtype=bool),
        "gtIgnore": np.zeros(positives, dtype=bool),
    }


def _evaluation(entries: list[dict], *, categories: int, images: int):
    params = SimpleNamespace(
        iouThrs=np.asarray([0.5]),
        recThrs=np.asarray([0.0, 0.5, 1.0]),
        catIds=list(range(categories)),
        areaRng=[[0.0, 1e10]],
    )
    return SimpleNamespace(
        params=params,
        _paramsEval=SimpleNamespace(imgIds=list(range(images))),
        evalImgs=entries,
    )


def test_ordinary_bootstrap_preserves_duplicate_image_multiplicity() -> None:
    """Catches collapsing a resampled image index to a unique COCO image ID."""
    evaluation = _evaluation(
        [
            _entry([0.9], [1], 1),
            _entry([0.8, 0.7], [0, 1], 1),
        ],
        categories=1,
        images=2,
    )

    ordinary = accumulate_ap(evaluation, [0, 1])[0]
    duplicated = accumulate_ap(evaluation, [0, 0, 1])[0]

    assert ordinary == pytest.approx(8 / 9)
    assert duplicated == pytest.approx(11 / 12)
    assert duplicated != ordinary


def test_ordinary_bootstrap_can_change_the_macro_category_universe() -> None:
    """Catches treating TT100K rare-class drift as Monte Carlo noise only."""
    evaluation = _evaluation(
        [
            _entry([0.9], [1], 1),
            _entry([], [], 0),
            _entry([], [], 0),
            _entry([0.8], [0], 1),
        ],
        categories=2,
        images=2,
    )

    full = accumulate_ap(evaluation, [0, 1])[0]
    category_one_absent = accumulate_ap(evaluation, [0, 0])[0]

    assert full == pytest.approx(0.5)
    assert category_one_absent == pytest.approx(1.0)


def test_fixed_universe_weighted_ap_matches_uniform_point_and_retains_rare_class() -> None:
    """Catches dropping a rare category when its image receives a small positive weight."""
    evaluation = _evaluation(
        [
            _entry([0.9], [1], 1),
            _entry([], [], 0),
            _entry([], [], 0),
            _entry([0.8], [0], 1),
        ],
        categories=2,
        images=2,
    )
    mask = fixed_category_area_mask(evaluation)

    assert mask.tolist() == [[True], [True]]
    assert accumulate_ap_weighted(evaluation, np.ones(2), mask)[0] == pytest.approx(0.5)
    assert accumulate_ap_weighted(evaluation, np.asarray([1.999, 0.001]), mask)[0] == pytest.approx(0.5)


def test_prepared_weighted_accumulator_is_exactly_equal_to_reference() -> None:
    evaluation = _evaluation(
        [
            _entry([0.9, 0.7], [1, 0], 1),
            _entry([0.9, 0.8], [0, 1], 1),
        ],
        categories=1,
        images=2,
    )
    weights = np.asarray([1.6, 0.4])
    mask = fixed_category_area_mask(evaluation)

    reference = accumulate_ap_weighted(evaluation, weights, mask)
    prepared = prepare_weighted_ap(evaluation, mask)
    accelerated = accumulate_prepared_ap(prepared, weights)

    np.testing.assert_array_equal(accelerated, reference)


@pytest.mark.parametrize("weights", [np.asarray([1.0, 0.0]), np.asarray([1.0, -0.1])])
def test_fixed_universe_weighted_ap_requires_strictly_positive_weights(weights: np.ndarray) -> None:
    evaluation = _evaluation(
        [_entry([0.9], [1], 1), _entry([0.8], [1], 1)],
        categories=1,
        images=2,
    )

    with pytest.raises(ValueError, match="strictly positive"):
        accumulate_ap_weighted(evaluation, weights, fixed_category_area_mask(evaluation))
