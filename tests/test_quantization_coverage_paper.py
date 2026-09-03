from __future__ import annotations

import importlib.util
import json
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]


def load_module():
    path = ROOT / "analysis" / "build_quantization_coverage_table.py"
    assert path.is_file(), "quantization coverage paper builder has not been implemented"
    spec = importlib.util.spec_from_file_location("build_quantization_coverage_table", path)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module


def graph(path: str, *, q: int, weight: int, activation: int, direct: int, operators: int) -> dict:
    return {
        "onnx_path": path,
        "quantize_linear_nodes": q,
        "dequantize_linear_nodes": q,
        "weight_quantizers": weight,
        "activation_quantizers": activation,
        "operator_nodes_with_dequantized_input": direct,
        "operator_nodes_excluding_qdq": operators,
        "quantizer_zero_point_types": {"INT8" if "int8" in path else "FLOAT8E4M3FN": q},
    }


def test_summary_keeps_graph_coverage_separate_by_family_model_and_format() -> None:
    module = load_module()
    reports = [
        graph("artifacts/x/engines/voc/yolo11n/int8-entropy.onnx", q=10, weight=4, activation=6, direct=7, operators=20),
        graph("artifacts/x/engines/kitti/yolo11n/int8-entropy.onnx", q=12, weight=5, activation=7, direct=8, operators=20),
        graph("outputs/onnx/cross_family_v1/voc_rtdetr_l_fp8.onnx", q=30, weight=12, activation=18, direct=15, operators=50),
    ]

    rows = module.summarize_graphs(reports)

    yolo = next(row for row in rows if row["family"] == "YOLO11")
    assert yolo["model"] == "YOLO11n" and yolo["format"] == "INT8"
    assert yolo["graphs"] == 2
    assert yolo["quantizers"] == "10--12"
    rtdetr = next(row for row in rows if row["family"] == "RT-DETR")
    assert rtdetr["zero_point_type"] == "FLOAT8E4M3FN"


def test_builder_writes_hash_bound_table_with_non_kernel_scope(tmp_path: Path) -> None:
    module = load_module()
    transfer = tmp_path / "transfer.json"
    cross = tmp_path / "cross.json"
    transfer.write_text(json.dumps({"graphs": [graph("artifacts/x/engines/voc/yolo11n/int8-entropy.onnx", q=10, weight=4, activation=6, direct=7, operators=20)]}), encoding="utf-8")
    cross.write_text(json.dumps({"graphs": [graph("outputs/onnx/cross_family_v1/voc_rtdetr_l_fp8.onnx", q=30, weight=12, activation=18, direct=15, operators=50)]}), encoding="utf-8")

    audit = module.build(transfer, cross, tmp_path / "generated")

    tex = (tmp_path / "generated" / "quantization_graph_coverage.tex").read_text(encoding="utf-8")
    assert "graph-level" in tex
    assert "not TensorRT kernel" in tex
    assert audit["rows"] == 2
    assert len(audit["input_reports_sha256"]) == 2


def test_supplement_labels_qdq_counts_as_graph_not_kernel_coverage() -> None:
    supplement = (ROOT / "paper" / "supplement.tex").read_text(encoding="utf-8")
    normalized = " ".join(supplement.split())

    assert r"\input{generated/quantization_graph_coverage.tex}" in supplement
    assert "not a claim that every TensorRT kernel executes at the nominal precision" in normalized
    assert "failure of recipe portability under these recorded treatments" in normalized
