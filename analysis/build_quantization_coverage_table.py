#!/usr/bin/env python3
"""Build a compact, hash-bound ONNX Q/DQ coverage table for Supplement S1."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import defaultdict
from pathlib import Path
from typing import Any


def sha256_file(path: Path) -> str:
    return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_hash(document: dict[str, Any], excluded: str | None = None) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def identity(path_text: str) -> tuple[str, str, str]:
    path = Path(path_text)
    if "cross_family_v1" in path.parts:
        stem = path.stem
        dataset, remainder = stem.split("_", 1)
        del dataset
        if remainder.endswith("_int8-entropy"):
            model, precision = remainder[: -len("_int8-entropy")], "INT8"
        elif remainder.endswith("_fp8"):
            model, precision = remainder[:-4], "FP8"
        else:
            raise ValueError(f"unrecognized cross-family precision: {path_text}")
        if model == "rtdetr_l":
            return "RT-DETR", "RT-DETR-L", precision
        if model == "retinanet_r50_fpn_v2":
            return "RetinaNet", "R50-FPN-v2", precision
        raise ValueError(f"unrecognized cross-family model: {path_text}")
    model = path.parent.name
    precision = {"int8-entropy": "INT8", "fp8": "FP8"}.get(path.stem)
    if model not in {"yolo11n", "yolo11m", "yolo11x"} or precision is None:
        raise ValueError(f"unrecognized YOLO graph identity: {path_text}")
    return "YOLO11", model.replace("yolo", "YOLO"), precision


def numeric_range(values: list[int]) -> str:
    low, high = min(values), max(values)
    return str(low) if low == high else f"{low}--{high}"


def summarize_graphs(graphs: list[dict[str, Any]]) -> list[dict[str, Any]]:
    grouped: dict[tuple[str, str, str], list[dict[str, Any]]] = defaultdict(list)
    for graph in graphs:
        grouped[identity(graph["onnx_path"])].append(graph)
    order = {"YOLO11": 0, "RT-DETR": 1, "RetinaNet": 2, "INT8": 0, "FP8": 1}
    rows = []
    for (family, model, precision), items in sorted(
        grouped.items(), key=lambda item: (order[item[0][0]], item[0][1], order[item[0][2]])
    ):
        zero_types = {key for item in items for key in item["quantizer_zero_point_types"]}
        rows.append(
            {
                "family": family,
                "model": model,
                "format": precision,
                "graphs": len(items),
                "quantizers": numeric_range([item["quantize_linear_nodes"] for item in items]),
                "weight_quantizers": numeric_range([item["weight_quantizers"] for item in items]),
                "activation_quantizers": numeric_range([item["activation_quantizers"] for item in items]),
                "direct_qdq_consumer_nodes": numeric_range(
                    [item["operator_nodes_with_dequantized_input"] for item in items]
                ),
                "non_qdq_operator_nodes": numeric_range(
                    [item["operator_nodes_excluding_qdq"] for item in items]
                ),
                "zero_point_type": "/".join(sorted(zero_types)),
            }
        )
    return rows


def build(transfer_report: Path, cross_report: Path, output_dir: Path) -> dict[str, Any]:
    transfer_report, cross_report = transfer_report.resolve(), cross_report.resolve()
    output_dir = output_dir.resolve()
    reports = [json.loads(path.read_text(encoding="utf-8")) for path in (transfer_report, cross_report)]
    graphs = [graph for report in reports for graph in report["graphs"]]
    rows = summarize_graphs(graphs)
    output_dir.mkdir(parents=True, exist_ok=True)
    csv_path = output_dir / "quantization_graph_coverage.csv"
    tex_path = output_dir / "quantization_graph_coverage.tex"
    audit_path = output_dir / "quantization_graph_coverage_audit.json"
    if any(path.exists() for path in (csv_path, tex_path, audit_path)):
        raise SystemExit("QDQ PAPER TABLE REFUSED: refusing to overwrite generated artifacts")
    with csv_path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    lines = [
        "% Generated from hash-bound ONNX graphs; graph-level coverage, not TensorRT kernel precision.",
        r"\begin{tabular}{lllrrrrrl}",
        r"\toprule",
        r"Family & Model & Format & Graphs & Q & Weight Q & Act. Q & Direct Q/DQ nodes & Zero point \\",
        r"\midrule",
    ]
    for row in rows:
        lines.append(
            f"{row['family']} & {row['model']} & {row['format']} & {row['graphs']} & "
            f"{row['quantizers']} & {row['weight_quantizers']} & {row['activation_quantizers']} & "
            f"{row['direct_qdq_consumer_nodes']}/{row['non_qdq_operator_nodes']} & "
            f"{row['zero_point_type']} \\\\" 
        )
    lines.extend([r"\bottomrule", r"\end{tabular}"])
    tex_path.write_text("\n".join(lines) + "\n", encoding="utf-8")
    audit = {
        "schema_version": 1,
        "scope": "ONNX graph-level Q/DQ coverage; not TensorRT kernel precision or fallback evidence",
        "rows": len(rows),
        "graphs": len(graphs),
        "input_reports_sha256": {
            str(transfer_report): sha256_file(transfer_report),
            str(cross_report): sha256_file(cross_report),
        },
        "generated_artifacts_sha256": {
            csv_path.name: sha256_file(csv_path),
            tex_path.name: sha256_file(tex_path),
        },
    }
    audit["audit_sha256"] = canonical_hash(audit, "audit_sha256")
    audit_path.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    return audit


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--transfer-report", required=True)
    parser.add_argument("--cross-report", required=True)
    parser.add_argument("--output-dir", required=True)
    args = parser.parse_args()
    audit = build(Path(args.transfer_report), Path(args.cross_report), Path(args.output_dir))
    print(json.dumps(audit, indent=2))


if __name__ == "__main__":
    main()
