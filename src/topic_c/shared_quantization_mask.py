"""Fail-closed helpers for a format-matched ONNX Q/DQ placement pilot.

The shared mask is anchored to operator names and input/output positions in a
frozen FP32 source graph.  Quantizer node names, scale values, zero-point
types, and Q/DQ node order are deliberately excluded from the normalized
topology: those details necessarily differ between INT8 and FP8 even when the
same source edges are quantized.
"""
from __future__ import annotations

from collections import Counter
import copy
import hashlib
import json
from pathlib import Path
from typing import Any

import onnx


SHARED_COMPUTE_OPS = frozenset({"Conv", "Gemm", "MatMul", "Add"})


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def canonical_hash(document: dict[str, Any], excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def read_complete_json(path: Path, *, self_hash_field: str | None = None) -> dict[str, Any]:
    """Read JSON only when its completion marker and optional self hash agree."""
    path = Path(path).resolve()
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.is_file() or not marker.is_file():
        raise ValueError(f"incomplete JSON artifact: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    expected_marker = sha256_file(path)
    if self_hash_field is not None:
        declared = document.get(self_hash_field)
        if not isinstance(declared, str) or declared != canonical_hash(document, self_hash_field):
            raise ValueError(f"invalid {self_hash_field}: {path}")
        expected_marker = declared
    if marker.read_text(encoding="utf-8").strip() != expected_marker:
        raise ValueError(f"completion marker mismatch: {path}")
    return document


def write_complete_json(path: Path, document: dict[str, Any], *, self_hash_field: str | None = None) -> None:
    path = Path(path).resolve()
    marker = path.with_suffix(path.suffix + ".complete")
    if path.exists() or marker.exists():
        raise ValueError(f"refusing to overwrite immutable artifact: {path}")
    if self_hash_field is not None:
        document[self_hash_field] = canonical_hash(document, self_hash_field)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(document, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    marker.write_text(
        (document[self_hash_field] if self_hash_field is not None else sha256_file(path)) + "\n",
        encoding="utf-8",
    )


def registry_onnx(registry_path: Path) -> tuple[dict[str, Any], Path, str]:
    """Resolve and hash-check an FP32 or quantized ONNX registry."""
    registry_path = Path(registry_path).resolve()
    registry = read_complete_json(registry_path)
    declared_registry_sha = registry.get("registry_sha256")
    if declared_registry_sha is not None and (
        not isinstance(declared_registry_sha, str)
        or declared_registry_sha != canonical_hash(registry, "registry_sha256")
    ):
        raise ValueError(f"registry canonical SHA-256 mismatch: {registry_path}")
    if "output_onnx" in registry:
        onnx_path = Path(registry.get("output_onnx", "")).resolve()
        expected = registry.get("output_onnx_sha256")
    else:
        onnx_path = Path(registry.get("onnx", "")).resolve()
        expected = registry.get("onnx_sha256")
    if not onnx_path.is_file() or not isinstance(expected, str) or sha256_file(onnx_path) != expected:
        raise ValueError(f"registry ONNX hash mismatch: {registry_path}")
    return registry, onnx_path, sha256_file(registry_path)


def _node_catalog(
    model: onnx.ModelProto, *, compute_only: bool = False
) -> dict[str, onnx.NodeProto]:
    result: dict[str, onnx.NodeProto] = {}
    for node in model.graph.node:
        if compute_only and node.op_type not in SHARED_COMPUTE_OPS:
            continue
        if not node.name:
            raise ValueError("source/quantized graph contains an unnamed quantizable compute operator")
        if node.name in result:
            raise ValueError(f"graph contains duplicate node name: {node.name}")
        result[node.name] = node
    return result


def normalized_attachment_topology(
    source_path: Path,
    quantized_path: Path,
    *,
    required_node_names: set[str] | None = None,
) -> dict[str, Any]:
    """Describe Q/DQ attachments using only frozen-source graph anchors.

    An input attachment records that a source operator input is fed directly by
    a DequantizeLinear.  An output attachment records that a source operator
    output feeds a QuantizeLinear.  This is strict enough to detect placement
    changes while remaining comparable across INT8 and FP8 tensor datatypes.
    """
    source = onnx.load(Path(source_path), load_external_data=False)
    quantized = onnx.load(Path(quantized_path), load_external_data=False)
    source_nodes = _node_catalog(source, compute_only=True)
    quantized_nodes = _node_catalog(quantized, compute_only=True)
    required_node_names = required_node_names or set()
    unknown_required = required_node_names - set(source_nodes)
    if unknown_required:
        raise ValueError(f"required compute node is absent from source graph: {sorted(unknown_required)[0]}")
    for name in sorted(required_node_names):
        candidate = quantized_nodes.get(name)
        if candidate is None or candidate.op_type != source_nodes[name].op_type:
            raise ValueError(f"required compute node is absent or changed in quantized graph: {name}")

    # ModelOpt may normalize, fold, or remove non-compute operators (for
    # example an exporter Cast).  Such nodes are intentionally outside the
    # shared placement estimand.  Only source compute anchors that survive
    # with the same identity and type participate in the normalized topology.
    anchored_source_nodes = {
        name: node
        for name, node in source_nodes.items()
        if name in quantized_nodes and quantized_nodes[name].op_type == node.op_type
    }

    producers = {output: node for node in quantized.graph.node for output in node.output}
    initializers = {value.name for value in quantized.graph.initializer}
    graph_inputs = {value.name for value in quantized.graph.input}
    quantizers = [node for node in quantized.graph.node if node.op_type == "QuantizeLinear"]
    dequantizers = [node for node in quantized.graph.node if node.op_type == "DequantizeLinear"]

    input_attachments: list[dict[str, Any]] = []
    for name, source_node in sorted(anchored_source_nodes.items()):
        node = quantized_nodes[name]
        for index, tensor_name in enumerate(node.input):
            producer = producers.get(tensor_name)
            if producer is None or producer.op_type != "DequantizeLinear":
                continue
            upstream = producers.get(producer.input[0]) if producer.input else None
            if upstream is not None and upstream.op_type == "QuantizeLinear":
                original = upstream.input[0] if upstream.input else ""
                origin = "weight_initializer" if original in initializers else "activation"
            elif producer.input and producer.input[0] in initializers:
                origin = "dequantized_initializer"
            else:
                origin = "unresolved"
            input_attachments.append(
                {
                    "node_name": name,
                    "op_type": source_node.op_type,
                    "input_index": index,
                    "origin": origin,
                }
            )

    q_inputs = Counter(node.input[0] for node in quantizers if node.input)
    output_attachments: list[dict[str, Any]] = []
    anchored_q_inputs: set[str] = set()
    for name, source_node in sorted(anchored_source_nodes.items()):
        quantized_node = quantized_nodes[name]
        if len(quantized_node.output) != len(source_node.output):
            raise ValueError(f"source operator output arity changed in quantized graph: {name}")
        for index, tensor_name in enumerate(quantized_node.output):
            if q_inputs[tensor_name]:
                anchored_q_inputs.add(tensor_name)
                output_attachments.append(
                    {
                        "node_name": name,
                        "op_type": source_node.op_type,
                        "output_index": index,
                        "quantizer_count": int(q_inputs[tensor_name]),
                    }
                )
    graph_input_attachments = sorted(
        {name for name in graph_inputs if q_inputs[name]}
    )
    initializer_attachments = sorted(
        {name for name in initializers if q_inputs[name]}
    )
    anchored_q_inputs.update(graph_input_attachments)
    anchored_q_inputs.update(initializer_attachments)
    unanchored_q_inputs = sorted(
        name for name in q_inputs if name not in anchored_q_inputs
    )
    topology = {
        "quantize_linear_nodes": len(quantizers),
        "dequantize_linear_nodes": len(dequantizers),
        "input_attachments": input_attachments,
        "output_attachments": output_attachments,
        "graph_input_attachments": graph_input_attachments,
        "initializer_attachments": initializer_attachments,
        "unanchored_q_inputs": unanchored_q_inputs,
    }
    topology["compute_attachment_sha256"] = compute_attachment_sha256(topology)
    topology["topology_sha256"] = canonical_hash(topology, "topology_sha256")
    return topology


def compute_attachment_contract(topology: dict[str, Any]) -> dict[str, Any]:
    """Return only source-compute anchors relevant to a shared node mask.

    Whole-graph Q/DQ counts and unanchored non-compute tensors remain useful
    diagnostics, but are not part of the cross-format identity contract because
    ModelOpt may normalize or fold those operators differently by format.
    """
    return {
        "input_attachments": topology.get("input_attachments", []),
        "output_attachments": topology.get("output_attachments", []),
    }


def compute_attachment_sha256(topology: dict[str, Any]) -> str:
    payload = compute_attachment_contract(topology)
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def selected_compute_sites(
    source_path: Path,
    topology: dict[str, Any],
    *,
    allowed_op_types: frozenset[str] = SHARED_COMPUTE_OPS,
) -> list[dict[str, Any]]:
    """Recover every compute site with a source-anchored Q/DQ input.

    In particular, an ``Add`` with only one attached Q/DQ input is part of the
    executable FP8 placement contract.  Dropping such partial sites would turn
    an alleged baseline replay into a different, fully-quantized-input policy.
    """
    source = onnx.load(Path(source_path), load_external_data=False)
    nodes = _node_catalog(source, compute_only=True)
    attachments: dict[str, list[dict[str, Any]]] = {}
    for item in topology.get("input_attachments", []):
        attachments.setdefault(item["node_name"], []).append(item)
    selected = []
    for name, node in sorted(nodes.items()):
        if node.op_type not in allowed_op_types:
            continue
        linked = sorted(attachments.get(name, []), key=lambda value: value["input_index"])
        if linked:
            selected.append(
                {
                    "name": name,
                    "op_type": node.op_type,
                    "input_attachments": linked,
                }
            )
    if not selected:
        raise ValueError("no shared quantized compute sites were recovered")
    return selected


def frozen_no_quantize_input_edges(
    source_path: Path,
    topology: dict[str, Any],
    selected_nodes: list[dict[str, Any]],
) -> tuple[list[dict[str, Any]], list[dict[str, Any]]]:
    """Freeze every baseline-FP8 raw input of a source compute operator.

    ModelOpt's default placement algorithm returns ``no_quantize_inputs`` in
    addition to its selected nodes.  Supplying ``nodes_to_quantize`` directly
    bypasses that discovery step.  More subtly, ORT may attach one Q/DQ pair to
    *all* consumers of a tensor even when only one consumer was selected.  The
    resulting policy therefore cannot be recovered by looking only at selected
    nodes.  It is the complement of the frozen baseline attachments over every
    source compute input.  This prospective source/baseline rule covers partial
    residual inputs, unselected sibling consumers, and raw constants without
    detector-name heuristics.

    The returned source-produced edges are retained as an auditable rendering
    of ModelOpt's default raw-edge policy.  Quantization enforces the complete
    input contract target-by-target after ORT rather than relying on ModelOpt's
    private, first-consumer traversal helper.
    """
    source = onnx.load(Path(source_path), load_external_data=False)
    _node_catalog(source, compute_only=True)
    selected_catalog = {item["name"]: item["op_type"] for item in selected_nodes}
    if selected_catalog != {
        item["name"]: item["op_type"] for item in selected_compute_sites(source_path, topology)
    }:
        raise ValueError("selected-node inventory disagrees with the frozen baseline topology")

    policy = frozen_compute_input_policy(source_path, topology)
    edges: list[dict[str, Any]] = []
    raw_inputs: list[dict[str, Any]] = []
    for item in policy:
        if item["requires_qdq"]:
            continue
        if item["anchor_kind"] == "node_output":
            edges.append(
                {
                    "source_node": item["source_node"],
                    "source_op": item["source_op"],
                    "source_output_index": item["source_output_index"],
                    "target_node": item["node_name"],
                    "target_op": item["op_type"],
                    "target_input_index": item["input_index"],
                    "input_name": item["input_name"],
                }
            )
        else:
            raw_inputs.append(
                {
                    "target_node": item["node_name"],
                    "target_op": item["op_type"],
                    "target_input_index": item["input_index"],
                    "input_name": item["input_name"],
                    "input_kind": item["anchor_kind"],
                }
            )
    edges.sort(key=lambda item: (item["target_node"], item["target_input_index"], item["source_node"]))
    raw_inputs.sort(key=lambda item: (item["target_node"], item["target_input_index"], item["input_name"]))
    return edges, raw_inputs


def edge_policy_sha256(edges: list[dict[str, Any]], raw_inputs: list[dict[str, Any]]) -> str:
    payload = {"no_quantize_inputs": edges, "raw_inputs": raw_inputs}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def compute_input_policy_sha256(inputs: list[dict[str, Any]]) -> str:
    encoded = json.dumps(inputs, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode()
    return hashlib.sha256(encoded).hexdigest()


def frozen_compute_input_policy(
    source_path: Path,
    baseline_topology: dict[str, Any],
) -> list[dict[str, Any]]:
    """Describe the required Q/DQ status of every frozen-source compute input."""
    source = onnx.load(Path(source_path), load_external_data=False)
    nodes = _node_catalog(source, compute_only=True)
    initializers = {value.name for value in source.graph.initializer}
    graph_inputs = {value.name for value in source.graph.input}
    producers: dict[str, tuple[onnx.NodeProto, int]] = {}
    for producer in source.graph.node:
        if not producer.name:
            raise ValueError("source graph contains an unnamed producer node")
        for output_index, tensor_name in enumerate(producer.output):
            if not tensor_name:
                continue
            if tensor_name in producers:
                raise ValueError(f"source tensor has multiple producers: {tensor_name}")
            producers[tensor_name] = (producer, output_index)

    attachments: dict[tuple[str, int], dict[str, Any]] = {}
    for attachment in baseline_topology.get("input_attachments", []):
        key = (attachment.get("node_name"), attachment.get("input_index"))
        if key in attachments:
            raise ValueError(f"duplicate baseline compute-input attachment: {key}")
        node = nodes.get(key[0])
        if (
            node is None
            or node.op_type != attachment.get("op_type")
            or not isinstance(key[1], int)
            or key[1] < 0
            or key[1] >= len(node.input)
        ):
            raise ValueError(f"baseline attachment lacks a frozen-source input anchor: {key}")
        attachments[key] = attachment

    policy: list[dict[str, Any]] = []
    for node_name, node in sorted(nodes.items()):
        for input_index, input_name in enumerate(node.input):
            if not input_name:
                continue
            producer_info = producers.get(input_name)
            if producer_info is not None:
                producer, source_output_index = producer_info
                anchor_kind = "node_output"
                source_node, source_op = producer.name, producer.op_type
            elif input_name in initializers:
                anchor_kind = "initializer"
                source_node = source_op = None
                source_output_index = None
            elif input_name in graph_inputs:
                anchor_kind = "graph_input"
                source_node = source_op = None
                source_output_index = None
            else:
                raise ValueError(
                    f"compute input has no frozen source anchor: {node_name}[{input_index}]={input_name}"
                )
            attachment = attachments.get((node_name, input_index))
            policy.append(
                {
                    "node_name": node_name,
                    "op_type": node.op_type,
                    "input_index": input_index,
                    "input_name": input_name,
                    "anchor_kind": anchor_kind,
                    "source_node": source_node,
                    "source_op": source_op,
                    "source_output_index": source_output_index,
                    "requires_qdq": attachment is not None,
                    "baseline_origin": attachment.get("origin") if attachment is not None else None,
                }
            )
    return policy


def _remove_orphan_qdq_nodes(model: onnx.ModelProto) -> int:
    """Remove only Q/DQ nodes made dead by target-specific input bypasses."""
    removed = 0
    graph_outputs = {value.name for value in model.graph.output}
    while True:
        consumers = Counter(input_name for node in model.graph.node for input_name in node.input)
        doomed = {
            index
            for index, node in enumerate(model.graph.node)
            if node.op_type in {"QuantizeLinear", "DequantizeLinear"}
            and all(consumers[output] == 0 and output not in graph_outputs for output in node.output)
        }
        if not doomed:
            break
        kept = [copy.deepcopy(node) for index, node in enumerate(model.graph.node) if index not in doomed]
        removed += len(doomed)
        model.graph.ClearField("node")
        model.graph.node.extend(kept)
    return removed


def enforce_frozen_compute_input_policy(
    source_path: Path,
    candidate_path: Path,
    mask: dict[str, Any],
    output_path: Path,
) -> dict[str, Any]:
    """Make an explicit-node result obey the prospective baseline FP8 contract.

    Only surplus candidate Q/DQ attachments are bypassed.  Any missing required
    attachment, changed source edge, output-placement difference, or unresolved
    Q/DQ chain is a hard failure.  No AP value or detector-specific node name is
    consulted.
    """
    source_path = Path(source_path).resolve()
    candidate_path = Path(candidate_path).resolve()
    output_path = Path(output_path).resolve()
    if output_path.exists():
        raise ValueError(f"refusing to overwrite enforced ONNX artifact: {output_path}")
    policy_document = mask.get("compute_input_policy", {})
    inputs = policy_document.get("inputs")
    if not isinstance(inputs, list) or policy_document.get("policy_sha256") != compute_input_policy_sha256(inputs):
        raise ValueError("shared mask compute-input policy is invalid")
    expected_inputs = frozen_compute_input_policy(
        source_path, mask["baseline_fp8_normalized_topology"]
    )
    if inputs != expected_inputs:
        raise ValueError("shared mask compute-input policy disagrees with source/baseline topology")

    baseline_topology = mask["baseline_fp8_normalized_topology"]
    before = normalized_attachment_topology(source_path, candidate_path)
    required_keys = {
        (item["node_name"], item["input_index"])
        for item in inputs
        if item["requires_qdq"]
    }
    candidate_keys = {
        (item["node_name"], item["input_index"])
        for item in before["input_attachments"]
    }
    missing = sorted(required_keys - candidate_keys)
    if missing:
        raise ValueError(f"candidate is missing a required baseline FP8 Q/DQ input: {missing[0]}")
    policy_by_key = {(item["node_name"], item["input_index"]): item for item in inputs}
    extras = sorted(candidate_keys - required_keys)
    unknown = [key for key in extras if key not in policy_by_key]
    if unknown:
        raise ValueError(f"candidate attachment lacks a frozen-source policy anchor: {unknown[0]}")

    model = onnx.load(candidate_path, load_external_data=False)
    node_catalog = {node.name: node for node in model.graph.node if node.name}
    if len(node_catalog) != sum(1 for node in model.graph.node if node.name):
        raise ValueError("candidate graph contains duplicate named nodes")
    producers = {output: node for node in model.graph.node for output in node.output if output}
    bypassed: list[dict[str, Any]] = []
    for key in extras:
        item = policy_by_key[key]
        target = node_catalog.get(item["node_name"])
        if (
            target is None
            or target.op_type != item["op_type"]
            or item["input_index"] >= len(target.input)
        ):
            raise ValueError(f"candidate target changed before input-policy enforcement: {key}")
        dq = producers.get(target.input[item["input_index"]])
        if dq is None or dq.op_type != "DequantizeLinear" or not dq.input:
            raise ValueError(f"surplus candidate attachment is not a direct DequantizeLinear: {key}")
        upstream = producers.get(dq.input[0])
        if upstream is not None and upstream.op_type == "QuantizeLinear":
            if not upstream.input:
                raise ValueError(f"surplus QuantizeLinear has no source input: {key}")
            raw_name = upstream.input[0]
        else:
            raw_name = dq.input[0]
        if raw_name != item["input_name"]:
            raise ValueError(
                f"surplus Q/DQ does not resolve to the frozen source input: {key}: "
                f"{raw_name} != {item['input_name']}"
            )
        target.input[item["input_index"]] = raw_name
        bypassed.append(
            {
                "node_name": item["node_name"],
                "op_type": item["op_type"],
                "input_index": item["input_index"],
                "input_name": raw_name,
                "anchor_kind": item["anchor_kind"],
            }
        )

    removed_qdq_nodes = _remove_orphan_qdq_nodes(model)
    onnx.checker.check_model(model)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, output_path)
    after = normalized_attachment_topology(source_path, output_path)
    if compute_attachment_contract(after) != compute_attachment_contract(baseline_topology):
        output_path.unlink(missing_ok=True)
        raise ValueError("enforced candidate does not exactly match the frozen baseline FP8 compute topology")
    return {
        "pre_compute_attachment_sha256": before["compute_attachment_sha256"],
        "post_compute_attachment_sha256": after["compute_attachment_sha256"],
        "bypassed_inputs": bypassed,
        "bypassed_inputs_count": len(bypassed),
        "removed_orphan_qdq_nodes": removed_qdq_nodes,
    }


def validate_mask_document(document: dict[str, Any]) -> None:
    if (
        document.get("schema_version") != 3
        or document.get("policy") != "frozen_baseline_fp8_compute_attachment_contract_v3"
    ):
        raise ValueError("unsupported shared-mask schema or policy")
    declared = document.get("mask_sha256")
    if not isinstance(declared, str) or declared != canonical_hash(document, "mask_sha256"):
        raise ValueError("shared-mask canonical SHA-256 mismatch")
    nodes = document.get("nodes")
    if not isinstance(nodes, list) or not nodes:
        raise ValueError("shared mask has no nodes")
    names = [item.get("name") for item in nodes if isinstance(item, dict)]
    if len(names) != len(nodes) or any(not isinstance(name, str) or not name for name in names):
        raise ValueError("shared mask contains an invalid node entry")
    if len(set(names)) != len(names):
        raise ValueError("shared mask contains duplicate node names")
    op_types = sorted({item.get("op_type") for item in nodes})
    if any(not isinstance(value, str) or value not in SHARED_COMPUTE_OPS for value in op_types):
        raise ValueError("shared mask contains an unsupported operator type")
    if document.get("op_types_to_quantize") != op_types:
        raise ValueError("shared-mask operator allowlist does not match its nodes")
    compute_policy = document.get("compute_input_policy")
    if not isinstance(compute_policy, dict) or not isinstance(compute_policy.get("inputs"), list):
        raise ValueError("shared mask has no frozen compute-input policy")
    compute_inputs = compute_policy["inputs"]
    if compute_policy.get("policy_sha256") != compute_input_policy_sha256(compute_inputs):
        raise ValueError("shared-mask compute-input policy SHA-256 mismatch")
    if (
        compute_policy.get("attached_inputs")
        != sum(item.get("requires_qdq") is True for item in compute_inputs if isinstance(item, dict))
        or compute_policy.get("raw_inputs")
        != sum(item.get("requires_qdq") is False for item in compute_inputs if isinstance(item, dict))
    ):
        raise ValueError("shared-mask compute-input policy counts disagree")
    compute_node_catalog: dict[str, str] = {}
    compute_input_keys: set[tuple[str, int]] = set()
    required_compute_keys = {
        "node_name", "op_type", "input_index", "input_name", "anchor_kind",
        "source_node", "source_op", "source_output_index", "requires_qdq",
        "baseline_origin",
    }
    for item in compute_inputs:
        if not isinstance(item, dict) or set(item) != required_compute_keys:
            raise ValueError("shared mask contains a malformed compute-input policy record")
        key = (item["node_name"], item["input_index"])
        if (
            not isinstance(item["node_name"], str)
            or not item["node_name"]
            or item["op_type"] not in SHARED_COMPUTE_OPS
            or not isinstance(item["input_index"], int)
            or item["input_index"] < 0
            or not isinstance(item["input_name"], str)
            or not item["input_name"]
            or item["anchor_kind"] not in {"node_output", "initializer", "graph_input"}
            or not isinstance(item["requires_qdq"], bool)
            or key in compute_input_keys
        ):
            raise ValueError("shared mask contains an invalid compute-input policy record")
        if item["anchor_kind"] == "node_output":
            if (
                not isinstance(item["source_node"], str)
                or not item["source_node"]
                or not isinstance(item["source_op"], str)
                or not item["source_op"]
                or not isinstance(item["source_output_index"], int)
                or item["source_output_index"] < 0
            ):
                raise ValueError("shared mask contains an invalid source-produced input anchor")
        elif any(item[field] is not None for field in ("source_node", "source_op", "source_output_index")):
            raise ValueError("raw compute input unexpectedly declares a producer anchor")
        if item["requires_qdq"] and not isinstance(item["baseline_origin"], str):
            raise ValueError("attached compute input has no baseline origin")
        if not item["requires_qdq"] and item["baseline_origin"] is not None:
            raise ValueError("raw compute input unexpectedly has a baseline origin")
        existing_type = compute_node_catalog.setdefault(item["node_name"], item["op_type"])
        if existing_type != item["op_type"]:
            raise ValueError("compute-input policy changes a node operator type")
        compute_input_keys.add(key)
    edge_policy = document.get("edge_policy")
    if not isinstance(edge_policy, dict):
        raise ValueError("shared mask has no partial-input edge policy")
    edges = edge_policy.get("no_quantize_inputs")
    raw_inputs = edge_policy.get("raw_inputs")
    if not isinstance(edges, list) or not isinstance(raw_inputs, list):
        raise ValueError("shared-mask partial-input edge policy is malformed")
    if edge_policy.get("edge_policy_sha256") != edge_policy_sha256(edges, raw_inputs):
        raise ValueError("shared-mask partial-input edge-policy SHA-256 mismatch")
    node_catalog = compute_node_catalog
    seen_targets: set[tuple[str, int]] = set()
    for edge in edges:
        required = {
            "source_node", "source_op", "source_output_index", "target_node",
            "target_op", "target_input_index", "input_name",
        }
        if not isinstance(edge, dict) or set(edge) != required:
            raise ValueError("shared mask contains a malformed no-quantize-input edge")
        target_key = (edge["target_node"], edge["target_input_index"])
        if (
            node_catalog.get(edge["target_node"]) != edge["target_op"]
            or not isinstance(edge["source_node"], str)
            or not edge["source_node"]
            or not isinstance(edge["source_op"], str)
            or not edge["source_op"]
            or not isinstance(edge["source_output_index"], int)
            or edge["source_output_index"] < 0
            or not isinstance(edge["target_input_index"], int)
            or edge["target_input_index"] < 0
            or not isinstance(edge["input_name"], str)
            or not edge["input_name"]
            or target_key in seen_targets
        ):
            raise ValueError("shared mask contains an invalid no-quantize-input edge")
        seen_targets.add(target_key)
    for raw in raw_inputs:
        required = {"target_node", "target_op", "target_input_index", "input_name", "input_kind"}
        if not isinstance(raw, dict) or set(raw) != required:
            raise ValueError("shared mask contains a malformed raw-input record")
        target_key = (raw["target_node"], raw["target_input_index"])
        if (
            node_catalog.get(raw["target_node"]) != raw["target_op"]
            or raw.get("input_kind") not in {"initializer", "graph_input"}
            or not isinstance(raw.get("input_name"), str)
            or not raw["input_name"]
            or not isinstance(raw.get("target_input_index"), int)
            or raw["target_input_index"] < 0
            or target_key in seen_targets
        ):
            raise ValueError("shared mask contains an invalid raw-input record")
        seen_targets.add(target_key)
    topology = document.get("baseline_fp8_normalized_topology")
    if (
        not isinstance(topology, dict)
        or topology.get("topology_sha256") != canonical_hash(topology, "topology_sha256")
        or topology.get("compute_attachment_sha256") != compute_attachment_sha256(topology)
    ):
        raise ValueError("shared-mask baseline FP8 topology is invalid")


def read_mask(path: Path) -> dict[str, Any]:
    document = read_complete_json(path, self_hash_field="mask_sha256")
    validate_mask_document(document)
    return document
