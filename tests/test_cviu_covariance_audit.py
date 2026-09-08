import numpy as np
import pytest

from analyze_pairing_covariance import covariance_diagnostic


def test_common_image_variation_cancels_only_when_pairing_is_kept():
    result = covariance_diagnostic(np.array([1., 2., 3.]), np.array([0.5, 0.5, 0.5]))
    assert result["paired_variance"] == 0.0
    assert result["zero_covariance_variance"] == 2.0
    assert result["twice_covariance"] == 2.0
    assert result["sd_ratio"] is None


def test_negative_covariance_can_make_unpaired_uncertainty_smaller():
    # Clean gap [1,2,3], corrupted gap [3,2,1], interaction [2,0,-2].
    result = covariance_diagnostic(np.array([1., 2., 3.]), np.array([2., 0., -2.]))
    assert result["paired_variance"] == 4.0
    assert result["zero_covariance_variance"] == 2.0
    assert result["twice_covariance"] == -2.0
    assert result["sd_ratio"] == pytest.approx(2 ** -0.5)


@pytest.mark.parametrize("clean,direct", [([1, 2], [1]), ([1, float('nan')], [1, 2]), ([1], [2])])
def test_covariance_audit_rejects_invalid_or_unpaired_vectors(clean, direct):
    with pytest.raises(ValueError):
        covariance_diagnostic(np.asarray(clean), np.asarray(direct))
