from __future__ import annotations

import pytest

from build_dataset_pilot_plan import calibrator_for_precision


@pytest.mark.parametrize(
    ("precision", "calibrator"),
    [
        ("fp32", "none"),
        ("int8-entropy", "entropy"),
        ("fp8", "entropy"),
    ],
)
def test_calibrator_label_matches_actual_quantization_treatment(
    precision: str, calibrator: str
) -> None:
    assert calibrator_for_precision(precision) == calibrator


def test_calibrator_label_rejects_unplanned_precision() -> None:
    with pytest.raises(ValueError, match="unsupported precision"):
        calibrator_for_precision("int4")
