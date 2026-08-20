#!/usr/bin/env python3
"""Report graph-level Q/DQ coverage without inferring TensorRT kernel types."""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import hashlib
import json
from pathlib import Path
from typing import Any

import onnx
from onnx import TensorProto, numpy_helper


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(document: dict[str, Any], excluded: str | None = None) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def analyze_onnx_graph(path: Path) -> dict[str, Any]:
    path = path.resolve()
    model = onnx.load(path, load_external_data=False)
    graph = model.graph
    initializers = {value.name: value for value in graph.initializer}
    consumers: dict[str, set[int]] = defaultdict(set)
    for node_index, node in enumerate(graph.node):
        for name in node.input:
            consumers[name].add(node_index)

    q_nodes = [node for node in graph.node if node.op_type == "QuantizeLinear"]
    dq_nodes = [node for node in graph.node if node.op_type == "DequantizeLinear"]
    weight_quantizers = 0
    activation_quantizers = 0
    granularities: Counter[str] = Counter()
    zero_types: Counter[str] = Counter()
    scale_types: Counter[str] = Counter()
    axes: Counter[str] = Counter()
    for node in q_nodes:
        if node.input[0] in initializers:
            weight_quantizers += 1
        else:
            activation_quantizers += 1
        scale = initializers.get(node.input[1]) if len(node.input) > 1 else None
        zero = initializers.get(node.input[2]) if len(node.input) > 2 else None
        if scale is None:
            granularities["dynamic_or_unresolved"] += 1
            scale_types["unresolved"] += 1
        else:
            values = numpy_helper.to_array(scale)
            granularities["per_tensor" if values.size == 1 else "per_channel"] += 1
            scale_types[TensorProto.DataType.Name(scale.data_type)] += 1
        zero_types[TensorProto.DataType.Name(zero.data_type) if zero is not None else "implicit_UINT8"] += 1
        axis = next((attribute.i for attribute in node.attribute if attribute.name == "axis"), None)
        axes["default" if axis is None else str(axis)] += 1

    low_precision_consumers: set[int] = set()
    for node in dq_nodes:
        for output in node.output:
            low_precision_consumers.update(consumers.get(output, set()))
    excluded_types = {"QuantizeLinear", "DequantizeLinear"}
    operators_with_dq = Counter(
        graph.node[index].op_type for index in low_precision_consumers
        if graph.node[index].op_type not in excluded_types
    )
    unquantized = Counter(
        node.op_type for index, node in enumerate(graph.node)
        if node.op_type not in excluded_types and index not in low_precision_consumers
    )
    operator_nodes = sum(1 for node in graph.node if node.op_type not in excluded_types)
    report = {
        "schema_version": 1,
        "scope": "ONNX graph Q/DQ coverage only; does not assert TensorRT kernel datatype or runtime fallback",
        "onnx_path": str(path),
        "onnx_sha256": sha256_file(path),
        "ir_version": int(model.ir_version),
        "opset_imports": {item.domain or "ai.onnx": int(item.version) for item in model.opset_import},
        "nodes": len(graph.node),
        "operator_nodes_excluding_qdq": operator_nodes,
        "quantize_linear_nodes": len(q_nodes),
        "dequantize_linear_nodes": len(dq_nodes),
        "weight_quantizers": weight_quantizers,
        "activation_quantizers": activation_quantizers,
        "quantizer_granularity": dict(sorted(granularities.items())),
        "quantizer_scale_types": dict(sorted(scale_types.items())),
        "quantizer_zero_point_types": dict(sorted(zero_types.items())),
        "quantizer_axes": dict(sorted(axes.items())),
        "operators_with_dequantized_input": dict(sorted(operators_with_dq.items())),
        "operator_nodes_with_dequantized_input": sum(operators_with_dq.values()),
        "unquantized_operator_counts": dict(sorted(unquantized.items())),
    }
    report["graph_identity_sha256"] = canonical_hash(report, "graph_identity_sha256")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--onnx", nargs="+", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root, output = Path(args.project_root).resolve(), Path(args.out).resolve()
    if output.exists():
        raise SystemExit(f"QDQ COVERAGE REFUSED: refusing to overwrite: {output}")
    reports = []
    for value in args.onnx:
        path = Path(value).resolve()
        if root not in path.parents or not path.is_file():
            raise SystemExit(f"QDQ COVERAGE REFUSED: unsafe or missing ONNX: {path}")
        report = analyze_onnx_graph(path)
        report["onnx_path"] = str(path.relative_to(root))
        report["graph_identity_sha256"] = canonical_hash(report, "graph_identity_sha256")
        reports.append(report)
    document = {
        "schema_version": 1,
        "scope": "graph-level Q/DQ treatment coverage; TensorRT layer/kernel precision requires separate inspector evidence",
        "graphs": reports,
    }
    document["artifact_sha256"] = canonical_hash(document, "artifact_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "graphs": len(reports), "artifact_sha256": document["artifact_sha256"]}, indent=2))


if __name__ == "__main__":
    main()
