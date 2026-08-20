from __future__ import annotations

import numpy as np
import pytest

import analyze_corruption_realization as analysis


def _cell(dataset: str, corruption: str, severity: int, realization_seed: int, point: float) -> dict:
    return {
        "dataset": dataset,
        "corruption": corruption,
        "severity": severity,
        "realization_seed": realization_seed,
        "delta_e": point,
        "delta_e_draws": np.asarray([point - 0.01, point + 0.01], dtype=np.float64),
    }


def test_variance_decomposition_separates_image_and_realization_components() -> None:
    cells = [
        _cell("voc", "fog", 5, 11, 0.10),
        _cell("voc", "fog", 5, 22, 0.20),
        _cell("voc", "fog", 5, 33, 0.30),
    ]

    result = analysis.decompose_condition(cells, expected_seeds=(11, 22, 33))

    assert result["mean_delta_e"] == pytest.approx(0.20)
    assert result["between_realization_variance"] == pytest.approx(0.01)
    assert result["within_image_variance"] == pytest.approx(0.0002)
    assert result["descriptive_total_variance"] == pytest.approx(0.0102)
    assert result["between_realization_share"] == pytest.approx(0.01 / 0.0102)


def test_nested_draws_resample_realizations_and_preserve_image_draw_index() -> None:
    cells = [
        {**_cell("voc", "fog", 5, 11, 0.10), "delta_e_draws": np.array([1.0, 10.0])},
        {**_cell("voc", "fog", 5, 22, 0.20), "delta_e_draws": np.array([2.0, 20.0])},
        {**_cell("voc", "fog", 5, 33, 0.30), "delta_e_draws": np.array([3.0, 30.0])},
    ]
    realization_indices = np.array([[0, 1, 2], [2, 2, 1]], dtype=np.int64)

    draws = analysis.nested_condition_draws(
        cells,
        expected_seeds=(11, 22, 33),
        realization_indices=realization_indices,
    )

    assert np.array_equal(draws, np.array([2.0, 80.0 / 3.0]))


def test_validate_exact_grid_rejects_one_missing_realization_cell() -> None:
    cells = [
        _cell(dataset, corruption, severity, seed, 0.0)
        for dataset in analysis.DATASETS
        for corruption in analysis.CORRUPTIONS
        for severity in analysis.SEVERITIES
        for seed in (11, 22, 33)
    ]
    analysis.validate_exact_grid(cells, expected_seeds=(11, 22, 33), n_boot=2)

    with pytest.raises(ValueError, match="exact 108-cell"):
        analysis.validate_exact_grid(cells[:-1], expected_seeds=(11, 22, 33), n_boot=2)


def test_validate_exact_grid_rejects_misaligned_bootstrap_lengths() -> None:
    cells = [
        _cell(dataset, corruption, severity, seed, 0.0)
        for dataset in analysis.DATASETS
        for corruption in analysis.CORRUPTIONS
        for severity in analysis.SEVERITIES
        for seed in (11, 22, 33)
    ]
    cells[0]["delta_e_draws"] = np.array([0.0])

    with pytest.raises(ValueError, match="B=2"):
        analysis.validate_exact_grid(cells, expected_seeds=(11, 22, 33), n_boot=2)


def test_summarize_grid_equal_weights_36_base_conditions_and_uses_nested_draws() -> None:
    cells = []
    for dataset in analysis.DATASETS:
        for corruption in analysis.CORRUPTIONS:
            for severity in analysis.SEVERITIES:
                for seed, value in zip((11, 22, 33), (1.0, 2.0, 3.0)):
                    cells.append(
                        {
                            **_cell(dataset, corruption, severity, seed, value),
                            "delta_e_draws": np.array([value, value * 10.0]),
                        }
                    )
    realization_indices = np.array([[0, 1, 2], [2, 2, 1]], dtype=np.int64)

    result = analysis.summarize_grid(
        cells,
        expected_seeds=(11, 22, 33),
        realization_indices=realization_indices,
    )

    assert result["n_realization_cells"] == 108
    assert result["n_base_conditions"] == 36
    assert result["mean_delta_e"] == pytest.approx(2.0)
    assert np.allclose(result["nested_macro_draws"], np.array([2.0, 80.0 / 3.0]))
    assert result["mean_between_realization_variance"] == pytest.approx(1.0)


def test_realization_schedule_is_deterministic_and_samples_only_three_indices() -> None:
    first = analysis.realization_schedule(n_boot=2000, n_realizations=3, seed=9001)
    second = analysis.realization_schedule(n_boot=2000, n_realizations=3, seed=9001)

    assert first.shape == (2000, 3)
    assert np.array_equal(first, second)
    assert set(np.unique(first)) <= {0, 1, 2}


def test_variance_decomposition_supports_size_interaction_endpoint() -> None:
    cells = [
        {**_cell("voc", "fog", 5, 11, 0.0), "delta_psi": -0.2, "delta_psi_draws": np.array([-0.3, -0.1])},
        {**_cell("voc", "fog", 5, 22, 0.0), "delta_psi": 0.0, "delta_psi_draws": np.array([-0.1, 0.1])},
        {**_cell("voc", "fog", 5, 33, 0.0), "delta_psi": 0.2, "delta_psi_draws": np.array([0.1, 0.3])},
    ]

    result = analysis.decompose_condition(
        cells,
        expected_seeds=(11, 22, 33),
        point_key="delta_psi",
        draws_key="delta_psi_draws",
    )

    assert result["mean_delta_psi"] == pytest.approx(0.0)
    assert result["between_realization_variance"] == pytest.approx(0.04)
