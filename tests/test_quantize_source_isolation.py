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
