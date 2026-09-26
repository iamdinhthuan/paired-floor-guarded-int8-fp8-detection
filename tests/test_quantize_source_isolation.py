from __future__ import annotations

import importlib.util
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "src" / "quantize_yolo_onnx.py"
    spec = importlib.util.spec_from_file_location("quantize_yolo_onnx", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_modelopt_receives_an_isolated_copy_and_cannot_mutate_source(tmp_path: Path) -> None:
    module = load_module()
    source = tmp_path / "frozen.onnx"
    source.write_bytes(b"immutable-source")

    with module.isolated_onnx_source(source) as working:
        assert working != source
        assert working.read_bytes() == source.read_bytes()
        working.write_bytes(b"modelopt-mutated-copy")

    assert source.read_bytes() == b"immutable-source"
    assert not working.exists()


def test_retinanet_calibration_matches_inference_preprocessing(tmp_path: Path) -> None:
    import numpy as np
    from PIL import Image
    from topic_c.cross_family import preprocess_retinanet
    from topic_c.manifest import sha256_file

    module = load_module()
    image = tmp_path / "sample.png"
    pixels = np.zeros((12, 24, 3), dtype=np.uint8)
    pixels[:, 12:] = 255
    Image.fromarray(pixels).save(image)
    document = {"dataset_root": str(tmp_path), "n_images": 1,
                "records": [{"source_relpath": image.name, "sha256": sha256_file(image)}]}
    actual = module.calibration_tensor(document, 32, decoder="torchvision_retinanet_raw_v1")
    expected = preprocess_retinanet(image, 32)[0]
    np.testing.assert_array_equal(actual, expected)
    assert actual.min() < 0 and actual.max() > 1


def test_yolo_and_rtdetr_calibration_preserve_legacy_preprocessing(tmp_path: Path) -> None:
    import numpy as np
    from PIL import Image
    from topic_c.coco_data import preprocess
    from topic_c.manifest import sha256_file

    module = load_module()
    image = tmp_path / "sample.png"
    Image.new("RGB", (24, 12), (30, 80, 150)).save(image)
    document = {"dataset_root": str(tmp_path), "n_images": 1,
                "records": [{"source_relpath": image.name, "sha256": sha256_file(image)}]}
    for decoder in (None, "ultralytics_rtdetr_raw_v1"):
        actual = module.calibration_tensor(document, 32, decoder=decoder)
        np.testing.assert_array_equal(actual, preprocess(str(image), 32)[0])


def test_calibration_rejects_unknown_decoder_and_unapproved_mismatch() -> None:
    import pytest

    module = load_module()
    with pytest.raises(ValueError, match="decoder"):
        module.calibration_preprocessing("unknown_decoder")
    with pytest.raises(ValueError, match="mismatch"):
        module.calibration_preprocessing("torchvision_retinanet_raw_v1", "yolo_letterbox")
    assert module.calibration_preprocessing(
        "torchvision_retinanet_raw_v1", "yolo_letterbox", allow_mismatch=True
    ) == "yolo_letterbox"


def test_calibration_method_mapping_for_supported_modes() -> None:
    import pytest

    module = load_module()
    assert module.calibration_method("int8-entropy") == "entropy"
    assert module.calibration_method("int8-max") == "max"
    assert module.calibration_method("fp8") == "entropy"
    with pytest.raises(ValueError, match="mode"):
        module.calibration_method("fp16")


def test_calibration_still_rejects_changed_image_bytes(tmp_path: Path) -> None:
    import pytest

    module = load_module()
    image = tmp_path / "sample.png"
    image.write_bytes(b"not-the-recorded-image")
    document = {"dataset_root": str(tmp_path), "n_images": 1,
                "records": [{"source_relpath": image.name, "sha256": "0" * 64}]}
    with pytest.raises(SystemExit, match="hash mismatch"):
        module.calibration_tensor(document, 32, decoder="torchvision_retinanet_raw_v1")
