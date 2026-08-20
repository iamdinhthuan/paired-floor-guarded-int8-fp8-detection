from __future__ import annotations

import io

import numpy as np
from PIL import Image

from materialize_codec_control import encode_jpeg


def test_codec_control_encoding_is_deterministic_and_dimension_preserving() -> None:
    pixels = np.arange(19 * 23 * 3, dtype=np.uint8).reshape(19, 23, 3)
    image = Image.fromarray(pixels, mode="RGB")
    first = encode_jpeg(image, quality=95, subsampling=0)
    second = encode_jpeg(image, quality=95, subsampling=0)
    assert first == second
    assert first[:2] == b"\xff\xd8"
    with Image.open(io.BytesIO(first)) as decoded:
        assert decoded.size == image.size
        assert decoded.mode == "RGB"
