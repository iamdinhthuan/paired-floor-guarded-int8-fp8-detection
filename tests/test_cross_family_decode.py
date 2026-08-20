from __future__ import annotations

import numpy as np

from topic_c.cross_family import decode_rtdetr


def test_rtdetr_normalized_cxcywh_is_unletterboxed() -> None:
    output = np.array([[[0.5, 0.5, 0.5, 0.25, 0.9, 3.0]]], dtype=np.float32)
    rows = decode_rtdetr(output, 0.1, gain=2.0, padx=0, pady=50, width=100, height=50)
    x1, y1, x2, y2, score, label = rows[0]
    assert (x1, y1, x2, y2) == (25.0, 12.5, 75.0, 37.5)
    assert score == pytest.approx(0.9)
    assert label == 3


import pytest
