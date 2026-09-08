#!/usr/bin/env python3
"""Freeze exact FP8-derived compute masks for the three-block novelty pilot."""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import json
from pathlib import Path
import re
from typing import Any

from pilot_registry import calibration_sha256
from topic_c.shared_quantization_mask import (
    canonical_hash,
    compute_input_policy_sha256,
    edge_policy_sha256,
    frozen_compute_input_policy,
    frozen_no_quantize_input_edges,
    normalized_attachment_topology,
    read_complete_json,
    registry_onnx,
    selected_compute_sites,
    sha256_file,
    write_complete_json,
)


def _resolve(root: Path, value: str, label: str) -> Path:
    path = Path(value)
    path = path.resolve() if path.is_absolute() else (root / path).resolve()
    if root not in path.parents and path != root:
        raise ValueError(f"{label} escapes project root: {value}")
    return path


def _validate_config(config: dict[str, Any]) -> None:
    attempt = config.get("attempt")
    if (
        config.get("schema_version") != 1
        or not isinstance(attempt, str)
        or re.fullmatch(r"shared_mask_pilot_v[1-9][0-9]*", attempt) is None
    ):
        raise ValueError("unsupported shared-mask pilot config")
    if config.get("config_sha256") != canonical_hash(config, "config_sha256"):
        raise ValueError("shared-mask config canonical SHA-256 mismatch")
    blocks = config.get("blocks")
    if not isinstance(blocks, list) or len(blocks) != 3:
        raise ValueError("shared-mask pilot requires exactly three blocks")
    identifiers = [item.get("id") for item in blocks if isinstance(item, dict)]
    if len(identifiers) != 3 or len(set(identifiers)) != 3:
        raise ValueError("shared-mask block identifiers must be unique")


def freeze_block(root: Path, attempt: str, block: dict[str, Any], output: Path) -> dict[str, Any]:
    source_registry_path = _resolve(root, block["source_onnx_registry"], "source registry")
    int8_registry_path = _resolve(root, block["baseline_int8_registry"], "baseline INT8 registry")
    fp8_registry_path = _resolve(root, block["baseline_fp8_registry"], "baseline FP8 registry")
    calibration_path = _resolve(root, block["calibration_list"], "calibration list")
    source_registry, source_onnx, source_registry_file_sha = registry_onnx(source_registry_path)
    int8_registry, int8_onnx, int8_registry_file_sha = registry_onnx(int8_registry_path)
    fp8_registry, fp8_onnx, fp8_registry_file_sha = registry_onnx(fp8_registry_path)

    dataset, model, imgsz = block["dataset"], block["model"], int(block["imgsz"])
    for label, registry in (("source", source_registry), ("INT8", int8_registry), ("FP8", fp8_registry)):
        if registry.get("dataset") != dataset or registry.get("model") != model:
            raise ValueError(f"{label} registry dataset/model mismatch for {block['id']}")
        if int(registry.get("imgsz", imgsz)) != imgsz:
            raise ValueError(f"{label} registry image size mismatch for {block['id']}")
    if int8_registry.get("precision") != "int8-entropy" or fp8_registry.get("precision") != "fp8":
        raise ValueError(f"baseline precision mismatch for {block['id']}")
    source_sha = source_registry.get("onnx_sha256")
    if source_sha != sha256_file(source_onnx):
        raise ValueError(f"source ONNX registry mismatch for {block['id']}")
    if any(registry.get("source_onnx_sha256") != source_sha for registry in (int8_registry, fp8_registry)):
        raise ValueError(f"baseline quantizers do not share the frozen source for {block['id']}")

    calibration_sha = calibration_sha256(calibration_path)
    calibration_marker = calibration_path.with_suffix(calibration_path.suffix + ".complete")
    if not calibration_marker.is_file() or calibration_marker.read_text(encoding="utf-8").strip() != calibration_sha:
        raise ValueError(f"calibration completion marker mismatch for {block['id']}")
    if any(registry.get("calibration_sha256") != calibration_sha for registry in (int8_registry, fp8_registry)):
        raise ValueError(f"baseline quantizers do not share calibration bytes for {block['id']}")
    if any(registry.get("calibration_method") != "entropy" for registry in (int8_registry, fp8_registry)):
        raise ValueError(f"baseline calibration method mismatch for {block['id']}")

    fp8_topology = normalized_attachment_topology(source_onnx, fp8_onnx)
    int8_topology = normalized_attachment_topology(source_onnx, int8_onnx)
    nodes = selected_compute_sites(source_onnx, fp8_topology)
    compute_inputs = frozen_compute_input_policy(source_onnx, fp8_topology)
    no_quantize_inputs, raw_inputs = frozen_no_quantize_input_edges(
        source_onnx, fp8_topology, nodes
    )
    int8_nodes = selected_compute_sites(source_onnx, int8_topology)
    node_names = {item["name"] for item in nodes}
    int8_node_names = {item["name"] for item in int8_nodes}
    if not node_names <= int8_node_names:
        missing = sorted(node_names - int8_node_names)
        raise ValueError(f"FP8 compute sites are not a subset of INT8 sites for {block['id']}: {missing[:3]}")

    document: dict[str, Any] = {
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": attempt,
        "policy": "frozen_baseline_fp8_compute_attachment_contract_v3",
        "scope": (
            "post-hoc mechanistic pilot; INT8 is aligned to the frozen baseline-FP8 "
            "source-compute Q/DQ input contract while the FP8 control reuses the exact frozen "
            "baseline ONNX bytes; not TensorRT kernel precision"
        ),
        "block_id": block["id"],
        "dataset": dataset,
        "model": model,
        "imgsz": imgsz,
        "source": {
            "registry": str(source_registry_path),
            "registry_file_sha256": source_registry_file_sha,
            "onnx": str(source_onnx),
            "onnx_sha256": source_sha,
        },
        "calibration": {
            "list": str(calibration_path),
            "calibration_sha256": calibration_sha,
            "method": "entropy",
        },
        "baseline_int8": {
            "registry": str(int8_registry_path),
            "registry_file_sha256": int8_registry_file_sha,
            "onnx": str(int8_onnx),
            "onnx_sha256": sha256_file(int8_onnx),
            "selected_compute_sites": len(int8_nodes),
            "extra_compute_sites": [
                {"name": item["name"], "op_type": item["op_type"]}
                for item in int8_nodes if item["name"] not in node_names
            ],
        },
        "baseline_fp8": {
            "registry": str(fp8_registry_path),
            "registry_file_sha256": fp8_registry_file_sha,
            "onnx": str(fp8_onnx),
            "onnx_sha256": sha256_file(fp8_onnx),
            "selected_compute_sites": len(nodes),
        },
        "op_types_to_quantize": sorted({item["op_type"] for item in nodes}),
        "node_type_counts": dict(sorted(Counter(item["op_type"] for item in nodes).items())),
        "nodes": nodes,
        "compute_input_policy": {
            "derivation": (
                "every named Conv/Gemm/MatMul/Add input in the frozen FP32 source is labelled "
                "attached or raw from the frozen baseline-FP8 normalized Q/DQ topology before "
                "the aligned INT8 artifact is generated"
            ),
            "inputs": compute_inputs,
            "attached_inputs": sum(item["requires_qdq"] for item in compute_inputs),
            "raw_inputs": sum(not item["requires_qdq"] for item in compute_inputs),
            "policy_sha256": compute_input_policy_sha256(compute_inputs),
        },
        "edge_policy": {
            "no_quantize_inputs": no_quantize_inputs,
            "raw_inputs": raw_inputs,
            "edge_policy_sha256": edge_policy_sha256(no_quantize_inputs, raw_inputs),
        },
        "baseline_fp8_normalized_topology": fp8_topology,
        "generator": {
            "path": str(Path(__file__).resolve()),
            "sha256": sha256_file(Path(__file__).resolve()),
        },
    }
    write_complete_json(output, document, self_hash_field="mask_sha256")
    return json.loads(output.read_text(encoding="utf-8"))


def freeze(root: Path, config_path: Path, output_dir: Path) -> dict[str, Any]:
    root, config_path, output_dir = root.resolve(), config_path.resolve(), output_dir.resolve()
    if root not in config_path.parents or root not in output_dir.parents:
        raise ValueError("config and output directory must remain under the project root")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    _validate_config(config)
    report_path = output_dir / f"{config['attempt']}_complete.json"
    if report_path.exists() or report_path.with_suffix(report_path.suffix + ".complete").exists():
        raise ValueError(f"shared-mask completion report already exists: {report_path}")
    masks = []
    for block in config["blocks"]:
        output = output_dir / f"{block['id']}.json"
        document = freeze_block(root, config["attempt"], block, output)
        masks.append(
            {
                "block_id": block["id"],
                "path": str(output),
                "file_sha256": sha256_file(output),
                "mask_sha256": document["mask_sha256"],
                "nodes": len(document["nodes"]),
            }
        )
    report: dict[str, Any] = {
        "schema_version": 1,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": config["attempt"],
        "config": str(config_path),
        "config_file_sha256": sha256_file(config_path),
        "config_sha256": config["config_sha256"],
        "masks": masks,
    }
    write_complete_json(report_path, report, self_hash_field="report_sha256")
    return json.loads(report_path.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", required=True)
    parser.add_argument("--out-dir", required=True)
    args = parser.parse_args()
    report = freeze(Path(args.project_root), Path(args.config), Path(args.out_dir))
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
