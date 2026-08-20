from __future__ import annotations

import importlib.util
from pathlib import Path
import sys
from types import ModuleType


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    sys.modules.setdefault("tensorrt", ModuleType("tensorrt"))
    path = ROOT / "src" / "build_yolo_trt_engine.py"
    spec = importlib.util.spec_from_file_location("build_yolo_trt_engine_strict", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def test_confirmatory_engine_build_disables_tf32_for_every_treatment(tmp_path: Path) -> None:
    module = load_module()

    command = module.trtexec_build_command(
        executable=tmp_path / "trtexec",
        onnx_path=tmp_path / "model.onnx",
        engine=tmp_path / "model.plan",
        workspace="4096M",
    )

    assert command.count("--noTF32") == 1
    assert "--skipInference" in command
    assert "--builderOptimizationLevel=3" in command
