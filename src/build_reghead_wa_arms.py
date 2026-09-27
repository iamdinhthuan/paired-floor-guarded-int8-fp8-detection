#!/usr/bin/env python3
"""Derive weight-only / activation-only INT8 regression-head arms by QDQ rewiring.

Starting from a completed int8-matched QDQ ONNX registry, produces two
mechanism-ablation variants restricted to a target subgraph (default
^/head/regression_head/):

* ``wonly`` — keep the weight-side QuantizeLinear/DequantizeLinear chains;
  bypass activation QDQ pairs on every input consumed by a target node.
  Semantics: weights INT8, activations floating point inside the head.
* ``aonly`` — keep the activation QDQ pairs; bypass initializer-fed
  (weight/bias) QDQ chains so target nodes consume the pre-quantization
  floating-point initializer.
  Semantics: weights floating point, activations INT8 inside the head.

Rewiring is consumer-local: a shared boundary DequantizeLinear keeps
serving non-target consumers unchanged.  Dead Q/DQ nodes and orphaned
initializers are pruned afterwards; every rewired edge is recorded in the
derived registry for audit.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
from datetime import datetime, timezone
from pathlib import Path

import onnx


def sha256_file(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def canonical_hash(document: dict, field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def load_complete(path: Path) -> dict:
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.is_file() or not marker.is_file():
        raise SystemExit(f"SURGERY REFUSED: incomplete source registry: {path}")
    document = json.loads(path.read_text(encoding="utf-8"))
    if marker.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise SystemExit("SURGERY REFUSED: source registry completion marker mismatch")
    if document.get("registry_sha256") != canonical_hash(document, "registry_sha256"):
        raise SystemExit("SURGERY REFUSED: source registry canonical hash mismatch")
    return document


def producer_map(graph) -> dict:
    return {o: n for n in graph.node for o in n.output}


def qdq_producer(edge: str, producers: dict, initializers: set):
    """Return (dq_node, q_node) if edge is produced by a Q->DQ chain."""
    dq = producers.get(edge)
    if dq is None or dq.op_type != "DequantizeLinear":
        return None
    q = producers.get(dq.input[0])
    if q is None or q.op_type != "QuantizeLinear":
        return None
    return dq, q


def rewire(model: onnx.ModelProto, variant: str, target_re: re.Pattern) -> dict:
    graph = model.graph
    producers = producer_map(graph)
    initializers = {init.name for init in graph.initializer}
    rewired = []
    for node in graph.node:
        if not target_re.match(node.name or ""):
            continue
        for index, edge in enumerate(node.input):
            pair = qdq_producer(edge, producers, initializers)
            if pair is None:
                continue
            dq, q = pair
            from_initializer = q.input[0] in initializers
            if (variant == "wonly" and from_initializer) or (
                variant == "aonly" and not from_initializer
            ):
                continue
            rewired.append(
                {
                    "consumer": node.name,
                    "input_index": index,
                    "was": edge,
                    "now": q.input[0],
                    "kind": "weight" if from_initializer else "activation",
                }
            )
            node.input[index] = q.input[0]
    return {"rewired": rewired}


def prune_dead(graph) -> list[str]:
    """Iteratively drop nodes whose outputs are all unconsumed non-outputs."""
    graph_outputs = {o.name for o in graph.output}
    removed: list[str] = []
    while True:
        consumed = set(graph_outputs)
        consumed.update(e for n in graph.node for e in n.input)
        alive = []
        dead_names = []
        for node in graph.node:
            if node.output and all(o not in consumed for o in node.output):
                dead_names.append(node.name)
            else:
                alive.append(node)
        if not dead_names:
            break
        removed.extend(dead_names)
        del graph.node[:]
        graph.node.extend(alive)
    used = {e for n in graph.node for e in n.input}
    kept = [i for i in graph.initializer if i.name in used]
    del graph.initializer[:]
    graph.initializer.extend(kept)
    return removed


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--source-onnx-registry", required=True)
    parser.add_argument("--variant", choices=("wonly", "aonly"), required=True)
    parser.add_argument("--target-regex", default="^/head/regression_head/")
    parser.add_argument("--reference-onnx",
                        help="optional sibling ONNX (e.g. the full-exclusion arm) "
                             "whose target-node inputs must equal the rewired values")
    parser.add_argument("--out", required=True)
    parser.add_argument("--registry-out", required=True)
    args = parser.parse_args()

    registry_path = Path(args.source_onnx_registry).resolve()
    output = Path(args.out).resolve()
    record_out = Path(args.registry_out).resolve()
    if output.exists() or record_out.exists():
        raise SystemExit("SURGERY REFUSED: output or registry already exists")
    source_record = load_complete(registry_path)
    onnx_path = Path(source_record["output_onnx"]).resolve()
    if sha256_file(onnx_path) != source_record["output_onnx_sha256"]:
        raise SystemExit("SURGERY REFUSED: source ONNX hash mismatch")

    model = onnx.load(str(onnx_path), load_external_data=False)
    target_re = re.compile(args.target_regex)
    surgery = rewire(model, args.variant, target_re)
    if not surgery["rewired"]:
        raise SystemExit("SURGERY REFUSED: no edges matched the variant rule")
    removed_nodes = prune_dead(model.graph)
    onnx.checker.check_model(model)

    reference_mismatches = []
    if args.reference_onnx:
        ref = onnx.load(str(Path(args.reference_onnx).resolve()), load_external_data=False)
        ref_nodes = {n.name: n for n in ref.graph.node}
        for edge in surgery["rewired"]:
            ref_node = ref_nodes.get(edge["consumer"])
            if ref_node is None or ref_node.input[edge["input_index"]] != edge["now"]:
                reference_mismatches.append(edge)
        if reference_mismatches:
            raise SystemExit(
                f"SURGERY REFUSED: {len(reference_mismatches)} rewired edges disagree "
                f"with the reference arm's inputs"
            )

    output.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(output))
    record = dict(source_record)
    record.pop("registry_sha256", None)
    record.update(
        created_at_utc=datetime.now(timezone.utc).isoformat(),
        output_onnx=str(output),
        output_onnx_sha256=sha256_file(output),
        derived_from_registry_sha256=source_record["registry_sha256"],
        derived_from_onnx_sha256=source_record["output_onnx_sha256"],
        surgery={
            "tool": Path(__file__).name,
            "variant": args.variant,
            "target_regex": args.target_regex,
            "semantics": (
                "target-subgraph weights INT8, activations floating point"
                if args.variant == "wonly"
                else "target-subgraph weights floating point, activations INT8"
            ),
            "rewired_edges": surgery["rewired"],
            "rewired_edge_count": len(surgery["rewired"]),
            "removed_dead_nodes": removed_nodes,
        },
    )
    record["registry_sha256"] = canonical_hash(record, "registry_sha256")
    record_out.parent.mkdir(parents=True, exist_ok=True)
    record_out.write_text(json.dumps(record, indent=2) + "\n", encoding="utf-8")
    record_out.with_suffix(record_out.suffix + ".complete").write_text(
        sha256_file(record_out) + "\n", encoding="utf-8"
    )
    print(json.dumps({
        "SURGERY COMPLETE": args.variant,
        "rewired": len(surgery["rewired"]),
        "removed_nodes": len(removed_nodes),
        "onnx_sha256": record["output_onnx_sha256"],
    }, indent=2))


if __name__ == "__main__":
    main()
