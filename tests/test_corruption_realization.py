from __future__ import annotations

import numpy as np
from PIL import Image

import generate_corruption as corruption
import generate_corruption_realization as realization


def test_realization_seed_is_nested_across_severity_and_distinct_across_replicates() -> None:
    s1 = realization.stable_realization_seed("voc", 17, "gaussian_noise", 101)
    s5 = realization.stable_realization_seed("voc", 17, "gaussian_noise", 101)
    other = realization.stable_realization_seed("voc", 17, "gaussian_noise", 202)

    assert s1 == s5
    assert s1 != other
    assert corruption.stable_seed("voc", 17, "gaussian_noise", 1) != corruption.stable_seed(
        "voc", 17, "gaussian_noise", 5
    )


def test_same_realization_seed_reuses_gaussian_base_field_across_dose() -> None:
    image = Image.fromarray(np.full((16, 16, 3), 128, dtype=np.uint8))
    seed = realization.stable_realization_seed("voc", 17, "gaussian_noise", 101)

    low = np.asarray(
        corruption.transform(image, "gaussian_noise", {"sigma_255": 2.0}, seed),
        dtype=np.int16,
    )
    high = np.asarray(
        corruption.transform(image, "gaussian_noise", {"sigma_255": 8.0}, seed),
        dtype=np.int16,
    )

    # Quantization to uint8 permits a small rounding error, but signs and the
    # underlying spatial realization must remain aligned.
    low_delta = low - 128
    high_delta = high - 128
    informative = np.abs(low_delta) >= 1
    assert np.mean(np.sign(low_delta[informative]) == np.sign(high_delta[informative])) > 0.99
    assert np.corrcoef(low_delta.ravel(), high_delta.ravel())[0, 1] > 0.98
