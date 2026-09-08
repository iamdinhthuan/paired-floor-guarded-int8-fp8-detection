import numpy as np
import pytest
from pathlib import Path

import audit_completed_followups as audit


def test_macro_preserves_draw_pairing_instead_of_averaging_interval_endpoints():
    # Opposite cell draws cancel: averaging their individual intervals would not.
    assert audit.mean_percentiles([[0., 2., 4.], [4., 2., 0.]]) == [2., 2., 2.]


@pytest.mark.parametrize('values', [[], [[1., 2.], [1.]], [[float('nan'), 1.]]])
def test_macro_rejects_empty_ragged_or_nonfinite_draws(values):
    with pytest.raises(ValueError):
        audit.mean_percentiles(values)


def test_recipe_contrast_subtracts_clean_gain_and_cancels_shared_fp8_draws():
    result = audit.recipe_draws(
        np.array([2., 2., 2.]), np.array([5., 5., 5.]), np.array([10., 10., 10.]),
        np.array([1., 2., 3.]), np.array([3., 4., 5.]), np.array([7., 8., 9.]))
    np.testing.assert_array_equal(result['delta_e_default'], [-2., -2., -2.])
    np.testing.assert_array_equal(result['delta_e_aligned'], [-1., -1., -1.])
    np.testing.assert_array_equal(result['omega'], [-1., -1., -1.])


def test_checked_file_rejects_tampered_evidence(tmp_path):
    path = tmp_path / 'record.json'
    path.write_text('{}')
    expected = audit.digest(path)
    assert audit.checked_file(tmp_path, 'record.json', expected) == path
    path.write_text('{"changed":true}')
    with pytest.raises(ValueError, match='hash'):
        audit.checked_file(tmp_path, 'record.json', expected)


def test_retained_followups_recompute_joint_macros_with_distinct_cache_schemas():
    root = Path(__file__).resolve().parents[1]
    if not (root / audit.TT / 'joint_macro.json').exists():
        pytest.skip('retained follow-up evidence is not installed')
    result = audit.audit(root)
    assert result['tt100k']['fixed_universe']['delta_psi_height']['point'] == pytest.approx(-1.37775360379)
    assert result['controlled']['omega']['point'] == pytest.approx(-3.05769922151)
