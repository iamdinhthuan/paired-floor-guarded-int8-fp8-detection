#!/usr/bin/env python3
"""Create FP16, INT8-entropy, or FP8 ModelOpt ONNX from frozen inputs."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata
import json
import os
import re
import shutil
import tempfile
from contextlib import contextmanager
from datetime import datetime, timezone
from pathlib import Path

import numpy as np
import onnx

from topic_c.coco_data import preprocess
from topic_c.manifest import sha256_file
from topic_c.shared_quantization_mask import (
    compute_attachment_contract,
    compute_input_policy_sha256,
    enforce_frozen_compute_input_policy,
    frozen_compute_input_policy,
    normalized_attachment_topology,
    read_mask,
)


@contextmanager
def isolated_onnx_source(source: Path):
    """Shield a frozen source graph from in-place ModelOpt normalization."""
    source = Path(source).resolve()
    before = sha256_file(source)
    with tempfile.TemporaryDirectory(prefix="modelopt-source-") as directory:
        working = Path(directory) / source.name
        shutil.copy2(source, working)
        if sha256_file(working) != before:
            raise RuntimeError("isolated ONNX copy hash mismatch")
        yield working
    if sha256_file(source) != before:
        raise RuntimeError("ModelOpt mutated the frozen source ONNX")


def canonical_hash(document: dict, field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def load_complete(path: Path, marker_field: str | None = None) -> dict:
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.is_file() or not marker.is_file():
        raise SystemExit(f"ONNX QUANTIZATION REFUSED: incomplete input: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    expected = document.get(marker_field) if marker_field else sha256_file(path)
    if marker_field and (
        not isinstance(expected, str) or expected != canonical_hash(document, marker_field)
    ):
        raise SystemExit(f"ONNX QUANTIZATION REFUSED: canonical {marker_field} mismatch: {path}")
    if marker.read_text(encoding="utf-8").strip() != expected:
        raise SystemExit(f"ONNX QUANTIZATION REFUSED: completion marker mismatch: {path}")
    return document


def calibration_tensor(document: dict, imgsz: int) -> np.ndarray:
    root = Path(document["dataset_root"]).resolve()
    tensors = []
    for record in document["records"]:
        path = root / record["source_relpath"]
        if not path.is_file() or sha256_file(path) != record["sha256"]:
            raise SystemExit(f"ONNX QUANTIZATION REFUSED: calibration image hash mismatch: {path}")
        tensors.append(preprocess(str(path), imgsz)[0][0])
    values = np.stack(tensors).astype(np.float32)
    if values.shape != (document["n_images"], 3, imgsz, imgsz):
        raise SystemExit(f"ONNX QUANTIZATION REFUSED: unexpected calibration tensor shape: {values.shape}")
    return values


def exact_node_patterns(names: list[str]) -> list[str]:
    """Convert literal ONNX node names to exact ModelOpt regex patterns."""
    return [f"^{re.escape(name)}$" for name in names]


def validate_edge_policy(mask: dict, graph: onnx.GraphProto) -> list[dict]:
    """Bind every serialized partial-input edge to the frozen source graph."""
    named_nodes = [node for node in graph.node if node.name]
    catalog = {node.name: node for node in named_nodes}
    if len(catalog) != len(named_nodes):
        raise ValueError("source graph contains duplicate named nodes")
    initializers = {value.name for value in graph.initializer}
    graph_inputs = {value.name for value in graph.input}
    edges = mask["edge_policy"]["no_quantize_inputs"]
    for edge in edges:
        source_node = catalog.get(edge["source_node"])
        target_node = catalog.get(edge["target_node"])
        source_output_index = edge["source_output_index"]
        target_input_index = edge["target_input_index"]
        if source_node is None or source_node.op_type != edge["source_op"]:
            raise ValueError(f"partial-input source node is absent or changed: {edge['source_node']}")
        if target_node is None or target_node.op_type != edge["target_op"]:
            raise ValueError(f"partial-input target node is absent or changed: {edge['target_node']}")
        if (
            source_output_index >= len(source_node.output)
            or target_input_index >= len(target_node.input)
            or source_node.output[source_output_index] != edge["input_name"]
            or target_node.input[target_input_index] != edge["input_name"]
        ):
            raise ValueError(
                f"partial-input source edge changed: {edge['source_node']} -> {edge['target_node']}"
            )
    for raw in mask["edge_policy"]["raw_inputs"]:
        target_node = catalog.get(raw["target_node"])
        target_input_index = raw["target_input_index"]
        expected_pool = initializers if raw["input_kind"] == "initializer" else graph_inputs
        if (
            target_node is None
            or target_node.op_type != raw["target_op"]
            or target_input_index >= len(target_node.input)
            or target_node.input[target_input_index] != raw["input_name"]
            or raw["input_name"] not in expected_pool
        ):
            raise ValueError(
                f"raw partial-input anchor changed: {raw['target_node']}[{target_input_index}]"
            )
    return edges


def materialize_no_quantize_inputs(mask: dict, graph: onnx.GraphProto) -> list[tuple]:
    """Reconstruct ModelOpt's GraphSurgeon tuple proxies from frozen names."""
    edges = validate_edge_policy(mask, graph)
    if not edges:
        return []
    import onnx_graphsurgeon as gs

    return [
        (
            gs.Node(op=edge["source_op"], name=edge["source_node"]),
            gs.Node(op=edge["target_op"], name=edge["target_node"]),
            edge["input_name"],
        )
        for edge in edges
    ]


def resolve_node_mask(
    path: Path,
    *,
    source: dict,
    source_registry_file_sha256: str,
    source_onnx_sha256: str,
    graph: onnx.GraphProto,
    mode: str,
    imgsz: int,
    calibration_sha256: str | None,
    requested_op_types: list[str] | None,
) -> tuple[dict, list[str], list[str]]:
    """Validate a frozen shared mask and return safe ModelOpt options."""
    if mode not in {"int8-entropy", "fp8"}:
        raise ValueError("a shared node mask is valid only for INT8 or FP8")
    mask = read_mask(path)
    if (
        mask.get("dataset") != source.get("dataset")
        or mask.get("model") != source.get("model")
        or int(mask.get("imgsz", -1)) != imgsz
        or mask.get("source", {}).get("registry_file_sha256") != source_registry_file_sha256
        or mask.get("source", {}).get("onnx_sha256") != source_onnx_sha256
    ):
        raise ValueError("shared node mask does not match source dataset/model/image-size/ONNX")
    if mask.get("calibration", {}).get("calibration_sha256") != calibration_sha256:
        raise ValueError("shared node mask does not match calibration bytes")
    mask_op_types = list(mask["op_types_to_quantize"])
    relevant_nodes = [node for node in graph.node if node.op_type in mask_op_types]
    node_catalog = {node.name: node.op_type for node in relevant_nodes}
    if len(node_catalog) != len(relevant_nodes) or "" in node_catalog:
        raise ValueError("quantizable source-node names must be nonempty and unique for a shared mask")
    for item in mask["nodes"]:
        if node_catalog.get(item["name"]) != item["op_type"]:
            raise ValueError(f"shared-mask node is absent or has changed type: {item['name']}")
    if requested_op_types is not None and sorted(set(requested_op_types)) != mask_op_types:
        raise ValueError("--quantize-op-types disagrees with the frozen shared mask")
    validate_edge_policy(mask, graph)
    expected_compute_inputs = frozen_compute_input_policy(
        Path(mask["source"]["onnx"]), mask["baseline_fp8_normalized_topology"]
    )
    compute_policy = mask["compute_input_policy"]
    if (
        compute_policy.get("inputs") != expected_compute_inputs
        or compute_policy.get("policy_sha256")
        != compute_input_policy_sha256(expected_compute_inputs)
    ):
        raise ValueError("shared-mask compute-input policy does not replay its frozen baseline")
    names = [item["name"] for item in mask["nodes"]]
    return mask, mask_op_types, exact_node_patterns(names)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--onnx-registry", required=True)
    parser.add_argument("--mode", choices=("fp16", "int8-entropy", "fp8"), required=True)
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--calibration-list")
    parser.add_argument("--quantize-op-types", help="comma-separated audited operator allowlist")
    parser.add_argument("--node-mask", help="completed shared-mask manifest with literal source-node identities")
    parser.add_argument("--out", required=True)
    parser.add_argument("--registry-out", required=True)
    args = parser.parse_args()
    onnx_registry, output, record_out = Path(args.onnx_registry).resolve(), Path(args.out).resolve(), Path(args.registry_out).resolve()
    if output.exists() or record_out.exists():
        raise SystemExit("ONNX QUANTIZATION REFUSED: output or registry already exists")
    source = load_complete(onnx_registry)
    onnx_path = Path(source.get("onnx", "")).resolve()
    if not onnx_path.is_file() or sha256_file(onnx_path) != source.get("onnx_sha256"):
        raise SystemExit("ONNX QUANTIZATION REFUSED: source FP32 ONNX hash mismatch")
    if args.mode == "fp16" and args.calibration_list:
        raise SystemExit("ONNX QUANTIZATION REFUSED: FP16 conversion has no calibration list")
    calibration = None
    calibration_sha = None
    node_mask = None
    node_patterns = None
    quantize_op_types = [value.strip() for value in args.quantize_op_types.split(",") if value.strip()] if args.quantize_op_types else None
    calibration_path = Path(args.calibration_list).resolve() if args.calibration_list else None
    if args.mode != "fp16":
        if calibration_path is None:
            raise SystemExit("ONNX QUANTIZATION REFUSED: INT8/FP8 requires a train-only calibration list")
        calibration = load_complete(calibration_path, "calibration_sha256")
        if calibration.get("dataset") != source.get("dataset") or calibration.get("split") != "train":
            raise SystemExit("ONNX QUANTIZATION REFUSED: calibration list dataset/split mismatch")
    graph = onnx.load(onnx_path, load_external_data=False).graph
    if len(graph.input) != 1:
        raise SystemExit("ONNX QUANTIZATION REFUSED: expected exactly one model input")
    if args.node_mask:
        try:
            node_mask, quantize_op_types, node_patterns = resolve_node_mask(
                Path(args.node_mask).resolve(),
                source=source,
                source_registry_file_sha256=sha256_file(onnx_registry),
                source_onnx_sha256=sha256_file(onnx_path),
                graph=graph,
                mode=args.mode,
                imgsz=args.imgsz,
                calibration_sha256=calibration.get("calibration_sha256") if calibration else None,
                requested_op_types=quantize_op_types,
            )
        except ValueError as exc:
            raise SystemExit(f"ONNX QUANTIZATION REFUSED: {exc}") from exc
    output.parent.mkdir(parents=True, exist_ok=True)
    enforcement = None
    fp8_baseline_byte_replay = False

    def run_conversion(target: Path) -> None:
        nonlocal calibration_sha
        with isolated_onnx_source(onnx_path) as modelopt_input:
            if args.mode == "fp16":
                import modelopt.onnx.autocast as autocast
                random_input = np.random.default_rng(20260807).normal(
                    size=(1, 3, args.imgsz, args.imgsz)
                ).astype(np.float32)
                onnx.save(
                    autocast.convert_to_mixed_precision(
                        modelopt_input,
                        low_precision_type="fp16",
                        keep_io_types=True,
                        calibration_data={graph.input[0].name: random_input},
                    ),
                    target,
                )
                calibration_sha = None
                return
            from modelopt.onnx.quantization import quantize

            # ModelOpt may normalize its input graph in place.  It receives a
            # disposable byte-identical copy so one treatment cannot mutate the
            # frozen source used by another treatment.
            options = {
                "quantize_mode": "fp8" if args.mode == "fp8" else "int8",
                "calibration_data": {
                    graph.input[0].name: calibration_tensor(calibration, args.imgsz)
                },
                "calibration_method": "entropy",
                "calibration_eps": ["cpu"],
                "op_types_to_exclude": ["Sigmoid"],
                "output_path": str(target),
            }
            if quantize_op_types is not None:
                options["op_types_to_quantize"] = quantize_op_types
            if node_patterns is not None:
                options["nodes_to_quantize"] = node_patterns
                # Explicit nodes bypass ModelOpt's default placement discovery,
                # including its partial-input policy.  The complete prospective
                # baseline contract is enforced target-by-target below instead
                # of calling ModelOpt's private first-consumer rewiring helper.
            quantize(str(modelopt_input), **options)
            calibration_sha = calibration["calibration_sha256"]

    if node_mask is None:
        run_conversion(output)
    else:
        with tempfile.TemporaryDirectory(prefix="shared-mask-output-", dir=output.parent) as directory:
            directory_path = Path(directory)
            candidate = directory_path / "candidate.onnx"
            finalized = directory_path / "finalized.onnx"
            if args.mode == "fp8":
                baseline = Path(node_mask["baseline_fp8"]["onnx"]).resolve()
                baseline_sha = node_mask["baseline_fp8"]["onnx_sha256"]
                if not baseline.is_file() or sha256_file(baseline) != baseline_sha:
                    raise SystemExit(
                        "ONNX QUANTIZATION REFUSED: frozen baseline FP8 ONNX is unavailable or changed"
                    )
                shutil.copyfile(baseline, finalized)
                if sha256_file(finalized) != baseline_sha:
                    raise SystemExit("ONNX QUANTIZATION REFUSED: baseline FP8 byte replay failed")
                replay_topology = normalized_attachment_topology(onnx_path, finalized)
                baseline_topology = node_mask["baseline_fp8_normalized_topology"]
                if compute_attachment_contract(replay_topology) != compute_attachment_contract(
                    baseline_topology
                ):
                    raise SystemExit(
                        "ONNX QUANTIZATION REFUSED: frozen baseline FP8 topology binding failed"
                    )
                enforcement = {
                    "pre_compute_attachment_sha256": replay_topology[
                        "compute_attachment_sha256"
                    ],
                    "post_compute_attachment_sha256": replay_topology[
                        "compute_attachment_sha256"
                    ],
                    "bypassed_inputs": [],
                    "bypassed_inputs_count": 0,
                    "removed_orphan_qdq_nodes": 0,
                }
                calibration_sha = calibration["calibration_sha256"]
                fp8_baseline_byte_replay = True
            else:
                run_conversion(candidate)
                try:
                    enforcement = enforce_frozen_compute_input_policy(
                        onnx_path, candidate, node_mask, finalized
                    )
                except ValueError as exc:
                    raise SystemExit(
                        f"ONNX QUANTIZATION REFUSED: compute-input contract enforcement failed: {exc}"
                    ) from exc
            os.replace(finalized, output)
    if not output.is_file():
        raise SystemExit("ONNX QUANTIZATION REFUSED: ModelOpt produced no ONNX output")
    record = {
        "schema_version": 1, "created_at_utc": datetime.now(timezone.utc).isoformat(), "dataset": source["dataset"],
        "model": source["model"], "precision": args.mode, "source_onnx_registry_sha256": sha256_file(onnx_registry),
        "source_onnx_sha256": sha256_file(onnx_path), "output_onnx": str(output), "output_onnx_sha256": sha256_file(output),
        "calibration_list": str(calibration_path) if calibration_path else None, "calibration_sha256": calibration_sha,
        "calibration_method": "entropy" if args.mode != "fp16" else "not_applicable", "imgsz": args.imgsz,
        "quantize_mode": "fp8" if args.mode == "fp8" else ("int8" if args.mode == "int8-entropy" else "fp16_autocast"),
        "calibration_eps": ["cpu"] if args.mode != "fp16" else [],
        "op_types_to_exclude": ["Sigmoid"] if args.mode != "fp16" else [],
        "op_types_to_quantize": quantize_op_types,
        "node_mask": str(Path(args.node_mask).resolve()) if args.node_mask else None,
        "node_mask_file_sha256": sha256_file(Path(args.node_mask).resolve()) if args.node_mask else None,
        "node_mask_sha256": node_mask.get("mask_sha256") if node_mask else None,
        "nodes_to_quantize_count": len(node_mask["nodes"]) if node_mask else None,
        "no_quantize_inputs_count": len(node_mask["edge_policy"]["no_quantize_inputs"]) if node_mask else None,
        "edge_policy_sha256": node_mask["edge_policy"]["edge_policy_sha256"] if node_mask else None,
        "compute_input_policy_sha256": node_mask["compute_input_policy"]["policy_sha256"] if node_mask else None,
        "baseline_fp8_onnx_sha256": node_mask["baseline_fp8"]["onnx_sha256"] if node_mask else None,
        "fp8_baseline_byte_replay": fp8_baseline_byte_replay if node_mask else None,
        "shared_mask_role": (
            "frozen_baseline_fp8_control"
            if fp8_baseline_byte_replay
            else "int8_aligned_to_frozen_fp8_compute_inputs"
            if node_mask
            else None
        ),
        "pre_enforcement_compute_attachment_sha256": (
            enforcement["pre_compute_attachment_sha256"] if enforcement else None
        ),
        "post_enforcement_compute_attachment_sha256": (
            enforcement["post_compute_attachment_sha256"] if enforcement else None
        ),
        "bypassed_compute_inputs_count": enforcement["bypassed_inputs_count"] if enforcement else None,
        "bypassed_compute_inputs": enforcement["bypassed_inputs"] if enforcement else None,
        "removed_orphan_qdq_nodes": enforcement["removed_orphan_qdq_nodes"] if enforcement else None,
        "input_names": source.get("input_names", [value.name for value in graph.input]),
        "output_names": source.get("output_names", [value.name for value in graph.output]),
        "decoder": source.get("decoder"),
        "num_classes_excluding_background": source.get("num_classes_excluding_background"),
        "modelopt_version": importlib.metadata.version("nvidia-modelopt"),
        "onnx_version": onnx.__version__,
        "command": list(__import__("sys").argv),
    }
    record["registry_sha256"] = canonical_hash(record, "registry_sha256")
    record_out.parent.mkdir(parents=True, exist_ok=True)
    record_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record_out.with_suffix(record_out.suffix + ".complete").write_text(sha256_file(record_out) + "\n", encoding="utf-8")
    print(json.dumps({"ONNX QUANTIZATION COMPLETE": args.mode, "onnx_sha256": record["output_onnx_sha256"],
                      "registry": str(record_out)}, indent=2))


if __name__ == "__main__":
    main()
