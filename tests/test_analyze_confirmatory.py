from __future__ import annotations

import numpy as np
import pytest

import analyze_confirmatory as analysis


def test_direct_contrast_is_int8_excess_minus_fp8_excess_with_shared_draws() -> None:
    int8_point = np.array([0.04, 0.06, 0.03, 0.02])
    fp8_point = np.array([0.01, 0.02, 0.01, 0.00])
    int8_draws = np.array([[0.03, 0.05, 0.02, 0.01], [0.05, 0.07, 0.04, 0.03]])
    fp8_draws = np.array([[0.01, 0.02, 0.01, 0.00], [0.02, 0.03, 0.01, 0.01]])

    result = analysis.direct_contrast(
        int8_point=int8_point,
        fp8_point=fp8_point,
        int8_draws=int8_draws,
        fp8_draws=fp8_draws,
    )

    assert np.allclose(result["point"], np.array([0.03, 0.04, 0.02, 0.02]))
    assert np.allclose(
        result["draws"], np.array([[0.02, 0.03, 0.01, 0.01], [0.03, 0.04, 0.03, 0.02]])
    )
    assert result["psi_point"] == pytest.approx(0.02)
    assert np.allclose(result["psi_draws"], np.array([0.02, 0.02]))


def test_balanced_macro_averages_cells_not_images_and_keeps_joint_draw_index() -> None:
    cells = [
        {"delta_e": 0.01, "delta_psi": 0.03, "delta_e_draws": np.array([0.00, 0.02]), "delta_psi_draws": np.array([0.02, 0.04])},
        {"delta_e": 0.03, "delta_psi": -0.01, "delta_e_draws": np.array([0.02, 0.04]), "delta_psi_draws": np.array([-0.02, 0.00])},
    ]

    summary = analysis.summarize_cells(cells)

    assert summary["n_cells"] == 2
    assert summary["delta_e_point"] == 0.02
    assert summary["delta_psi_point"] == pytest.approx(0.01)
    assert np.array_equal(summary["delta_e_draws"], np.array([0.01, 0.03]))
    assert np.array_equal(summary["delta_psi_draws"], np.array([0.00, 0.02]))
    assert summary["delta_e_percentile95"] == pytest.approx([0.0105, 0.02, 0.0295])
