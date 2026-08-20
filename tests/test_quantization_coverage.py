from __future__ import annotations

import importlib.util
from pathlib import Path

import onnx
from onnx import TensorProto, helper


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "src" / "analyze_quantization_coverage.py"
    assert path.is_file(), "quantization coverage analyzer has not been implemented"
    spec = importlib.util.spec_from_file_location("analyze_quantization_coverage", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def synthetic_qdq_model(path: Path) -> None:
    initializers = [
        helper.make_tensor("weight", TensorProto.FLOAT, [2, 1, 1, 1], [1.0, 2.0]),
        helper.make_tensor("w_scale", TensorProto.FLOAT, [2], [0.1, 0.2]),
        helper.make_tensor("w_zero", TensorProto.INT8, [2], [0, 0]),
        helper.make_tensor("a_scale", TensorProto.FLOAT, [], [0.1]),
        helper.make_tensor("a_zero", TensorProto.INT8, [], [0]),
    ]
    nodes = [
        helper.make_node("QuantizeLinear", ["weight", "w_scale", "w_zero"], ["weight_q"], axis=0),
        helper.make_node("DequantizeLinear", ["weight_q", "w_scale", "w_zero"], ["weight_dq"], axis=0),
        helper.make_node("QuantizeLinear", ["input", "a_scale", "a_zero"], ["input_q"]),
        helper.make_node("DequantizeLinear", ["input_q", "a_scale", "a_zero"], ["input_dq"]),
        helper.make_node("Conv", ["input_dq", "weight_dq"], ["conv"]),
        helper.make_node("Sigmoid", ["conv"], ["output"]),
    ]
    graph = helper.make_graph(
        nodes,
        "qdq",
        [helper.make_tensor_value_info("input", TensorProto.FLOAT, [1, 1, 4, 4])],
        [helper.make_tensor_value_info("output", TensorProto.FLOAT, [1, 2, 4, 4])],
        initializer=initializers,
    )
    onnx.save(helper.make_model(graph), path)


def test_graph_analyzer_distinguishes_weight_activation_and_fallback_ops(tmp_path: Path) -> None:
    module = load_module()
    model_path = tmp_path / "model.onnx"
    synthetic_qdq_model(model_path)

    report = module.analyze_onnx_graph(model_path)

    assert report["quantize_linear_nodes"] == 2
    assert report["dequantize_linear_nodes"] == 2
    assert report["weight_quantizers"] == 1
    assert report["activation_quantizers"] == 1
    assert report["quantizer_granularity"] == {"per_channel": 1, "per_tensor": 1}
    assert report["quantizer_zero_point_types"] == {"INT8": 2}
    assert report["operators_with_dequantized_input"] == {"Conv": 1}
    assert report["unquantized_operator_counts"]["Sigmoid"] == 1


def test_graph_analyzer_binds_the_exact_onnx_bytes(tmp_path: Path) -> None:
    module = load_module()
    model_path = tmp_path / "model.onnx"
    synthetic_qdq_model(model_path)

    report = module.analyze_onnx_graph(model_path)

    assert report["onnx_sha256"] == module.sha256_file(model_path)
    assert report["graph_identity_sha256"] == module.canonical_hash(report, "graph_identity_sha256")

