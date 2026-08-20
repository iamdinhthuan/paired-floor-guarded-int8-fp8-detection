from __future__ import annotations

import freeze_corruption_sensitivity_subset as subset


def test_greedy_subset_preserves_every_observed_category_then_fills_deterministically() -> None:
    image_ids = [1, 2, 3, 4, 5, 6]
    categories = {
        1: {1},
        2: {2},
        3: {3},
        4: {1, 2},
        5: {2, 3},
        6: set(),
    }

    first = subset.select_images(
        dataset="tt100k",
        ordered_image_ids=image_ids,
        categories_by_image=categories,
        target=3,
        seed=20260818,
    )
    second = subset.select_images(
        dataset="tt100k",
        ordered_image_ids=image_ids,
        categories_by_image=categories,
        target=3,
        seed=20260818,
    )

    assert first == second
    assert len(first) == 3
    assert set().union(*(categories[image_id] for image_id in first)) == {1, 2, 3}
    assert first == sorted(first, key=image_ids.index)


def test_subset_refuses_target_too_small_for_category_coverage() -> None:
    try:
        subset.select_images(
            dataset="voc",
            ordered_image_ids=[1, 2, 3],
            categories_by_image={1: {1}, 2: {2}, 3: {3}},
            target=2,
            seed=7,
        )
    except ValueError as exc:
        assert "category coverage" in str(exc)
    else:
        raise AssertionError("coverage-impossible subset was accepted")
