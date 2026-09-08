#!/usr/bin/env python3
"""Verify exact normalized Q/DQ topology for a shared INT8/FP8 mask pair."""
from __future__ import annotations

import argparse
from datetime import datetime, timezone
import json
from pathlib import Path
from typing import Any

from topic_c.shared_quantization_mask import (
    compute_attachment_contract,
    normalized_attachment_topology,
    read_mask,
    registry_onnx,
    selected_compute_sites,
    sha256_file,
    write_complete_json,
)


def _validate_registry(
    registry: dict[str, Any],
    *,
    expected_precision: str,
    mask: dict[str, Any],
    label: str,
) -> None:
    if registry.get("precision") != expected_precision:
        raise ValueError(f"{label} precision mismatch")
    if registry.get("dataset") != mask.get("dataset") or registry.get("model") != mask.get("model"):
        raise ValueError(f"{label} dataset/model mismatch")
    if int(registry.get("imgsz", -1)) != int(mask.get("imgsz", -2)):
        raise ValueError(f"{label} image-size mismatch")
    if registry.get("source_onnx_sha256") != mask.get("source", {}).get("onnx_sha256"):
        raise ValueError(f"{label} source ONNX mismatch")
    if registry.get("calibration_sha256") != mask.get("calibration", {}).get("calibration_sha256"):
        raise ValueError(f"{label} calibration mismatch")
    if registry.get("node_mask_sha256") != mask.get("mask_sha256"):
        raise ValueError(f"{label} is not bound to the requested shared mask")
    registry_mask_value = registry.get("node_mask")
    if not isinstance(registry_mask_value, str) or not registry_mask_value:
        raise ValueError(f"{label} shared-mask path is absent")
    registry_mask_path = Path(registry_mask_value).resolve()
    if not registry_mask_path.is_file() or registry.get("node_mask_file_sha256") != sha256_file(registry_mask_path):
        raise ValueError(f"{label} shared-mask file binding mismatch")
    if read_mask(registry_mask_path).get("mask_sha256") != mask.get("mask_sha256"):
        raise ValueError(f"{label} shared-mask content binding mismatch")
    if registry.get("nodes_to_quantize_count") != len(mask["nodes"]):
        raise ValueError(f"{label} shared-mask node-count mismatch")
    edge_policy = mask["edge_policy"]
    if registry.get("no_quantize_inputs_count") != len(edge_policy["no_quantize_inputs"]):
        raise ValueError(f"{label} partial-input edge-count mismatch")
    if registry.get("edge_policy_sha256") != edge_policy["edge_policy_sha256"]:
        raise ValueError(f"{label} partial-input edge-policy binding mismatch")
    compute_policy = mask["compute_input_policy"]
    if registry.get("compute_input_policy_sha256") != compute_policy["policy_sha256"]:
        raise ValueError(f"{label} compute-input policy binding mismatch")
    if registry.get("baseline_fp8_onnx_sha256") != mask["baseline_fp8"]["onnx_sha256"]:
        raise ValueError(f"{label} frozen baseline FP8 binding mismatch")
    baseline_attachment_sha = mask["baseline_fp8_normalized_topology"][
        "compute_attachment_sha256"
    ]
    if registry.get("post_enforcement_compute_attachment_sha256") != baseline_attachment_sha:
        raise ValueError(f"{label} post-enforcement topology binding mismatch")
    if expected_precision == "fp8":
        if (
            registry.get("fp8_baseline_byte_replay") is not True
            or registry.get("shared_mask_role") != "frozen_baseline_fp8_control"
            or registry.get("bypassed_compute_inputs_count") != 0
        ):
            raise ValueError("FP8 registry is not an unchanged frozen-baseline control")
    elif (
        registry.get("fp8_baseline_byte_replay") is not False
        or registry.get("shared_mask_role") != "int8_aligned_to_frozen_fp8_compute_inputs"
    ):
        raise ValueError("INT8 registry is not a contract-aligned treatment")


def verify(mask_path: Path, int8_registry_path: Path, fp8_registry_path: Path, output: Path) -> dict[str, Any]:
    mask_path = mask_path.resolve()
    mask = read_mask(mask_path)
    int8_registry, int8_onnx, int8_registry_sha = registry_onnx(int8_registry_path)
    fp8_registry, fp8_onnx, fp8_registry_sha = registry_onnx(fp8_registry_path)
    _validate_registry(int8_registry, expected_precision="int8-entropy", mask=mask, label="INT8")
    _validate_registry(fp8_registry, expected_precision="fp8", mask=mask, label="FP8")
    source_onnx = Path(mask["source"]["onnx"]).resolve()
    if not source_onnx.is_file() or sha256_file(source_onnx) != mask["source"]["onnx_sha256"]:
        raise ValueError("shared-mask source ONNX bytes are unavailable or changed")
    if sha256_file(fp8_onnx) != mask["baseline_fp8"]["onnx_sha256"]:
        raise ValueError("FP8 control is not byte-identical to the frozen baseline FP8 ONNX")

    required_nodes = {item["name"] for item in mask["nodes"]}
    int8_topology = normalized_attachment_topology(
        source_onnx, int8_onnx, required_node_names=required_nodes
    )
    fp8_topology = normalized_attachment_topology(
        source_onnx, fp8_onnx, required_node_names=required_nodes
    )
    baseline_topology = mask["baseline_fp8_normalized_topology"]
    if compute_attachment_contract(int8_topology) != compute_attachment_contract(fp8_topology):
        raise ValueError("INT8 and FP8 shared-mask attachment topologies differ")
    if compute_attachment_contract(fp8_topology) != compute_attachment_contract(baseline_topology):
        raise ValueError("shared-mask outputs do not exactly replay the frozen baseline FP8 compute topology")

    expected_nodes = [item["name"] for item in mask["nodes"]]
    int8_nodes = [item["name"] for item in selected_compute_sites(source_onnx, int8_topology)]
    fp8_nodes = [item["name"] for item in selected_compute_sites(source_onnx, fp8_topology)]
    if int8_nodes != expected_nodes or fp8_nodes != expected_nodes:
        raise ValueError("realized INT8/FP8 compute sites differ from the frozen mask")

    report: dict[str, Any] = {
        "schema_version": 3,
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "pass",
        "scope": "normalized ONNX Q/DQ attachment topology; not TensorRT kernel precision",
        "block_id": mask["block_id"],
        "dataset": mask["dataset"],
        "model": mask["model"],
        "mask": {
            "path": str(mask_path),
            "file_sha256": sha256_file(mask_path),
            "mask_sha256": mask["mask_sha256"],
            "nodes": len(expected_nodes),
        },
        "compute_attachment_sha256": int8_topology["compute_attachment_sha256"],
        "int8_full_topology_sha256": int8_topology["topology_sha256"],
        "fp8_full_topology_sha256": fp8_topology["topology_sha256"],
        "full_graph_diagnostics_equal": int8_topology == fp8_topology,
        "baseline_fp8_compute_attachment_sha256": baseline_topology["compute_attachment_sha256"],
        "baseline_compute_topology_replayed_exactly": True,
        "baseline_fp8_bytes_replayed_exactly": True,
        "compute_input_policy_sha256": mask["compute_input_policy"]["policy_sha256"],
        "no_quantize_inputs": len(mask["edge_policy"]["no_quantize_inputs"]),
        "raw_unquantized_inputs": len(mask["edge_policy"]["raw_inputs"]),
        "int8": {
            "registry": str(int8_registry_path.resolve()),
            "registry_file_sha256": int8_registry_sha,
            "onnx": str(int8_onnx),
            "onnx_sha256": sha256_file(int8_onnx),
        },
        "fp8": {
            "registry": str(fp8_registry_path.resolve()),
            "registry_file_sha256": fp8_registry_sha,
            "onnx": str(fp8_onnx),
            "onnx_sha256": sha256_file(fp8_onnx),
        },
    }
    write_complete_json(output, report, self_hash_field="report_sha256")
    return json.loads(output.read_text(encoding="utf-8"))


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--mask", required=True)
    parser.add_argument("--int8-registry", required=True)
    parser.add_argument("--fp8-registry", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    try:
        report = verify(Path(args.mask), Path(args.int8_registry), Path(args.fp8_registry), Path(args.out))
    except ValueError as exc:
        raise SystemExit(f"SHARED-MASK VERIFICATION REFUSED: {exc}") from exc
    print(json.dumps(report, indent=2))


if __name__ == "__main__":
    main()
