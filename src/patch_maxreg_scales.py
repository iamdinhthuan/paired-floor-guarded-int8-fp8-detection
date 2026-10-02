#!/usr/bin/env python3
"""Head-restricted max-calibration counterfactual (phase K, ``int8-maxreg512``).

Starting from a frozen entropy-calibrated INT8-matched Q/DQ graph, replace the
scale of every *activation* quantizer whose dequantized output is consumed by a
``/head/regression_head/`` node with ``fp16(amax_max / 127)``, where
``amax_max`` is the per-tensor maximum of per-channel |x|max captured on the
same frozen 512-image clean-calibration manifest (``nn_actstats`` captures).
Every other initializer and every node are left untouched.

The selected set includes the five FPN-boundary tensors that feed both heads,
so the classification head's input scales change too; the arm is restricted to
regression-head-consumed quantizers, not strictly regression-head-local.

This script reconstructs the step that produced
``outputs/nn_maxreg_v1_20261001/engines/<ds>_retinanet_int8-maxreg512/model.onnx``;
``--verify-against`` checks initializer-level identity with that graph.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import numpy as np
import onnx
from onnx import numpy_helper

HEAD_PREFIX = "/head/regression_head/"


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def regression_consumed_quantizers(model: onnx.ModelProto) -> dict[str, list[str]]:
    """Map activation tensor -> scale initializer names of its Q and DQ nodes."""
    inits = {init.name for init in model.graph.initializer}
    consumers: dict[str, list[onnx.NodeProto]] = {}
    for node in model.graph.node:
        for name in node.input:
            consumers.setdefault(name, []).append(node)
    selected: dict[str, list[str]] = {}
    for q in model.graph.node:
        if q.op_type != "QuantizeLinear" or q.input[0] in inits:
            continue  # weight quantizers sit on initializers
        dqs = [n for n in consumers.get(q.output[0], []) if n.op_type == "DequantizeLinear"]
        used_by_reg = any(c.name.startswith(HEAD_PREFIX)
                          for dq in dqs for c in consumers.get(dq.output[0], []))
        if used_by_reg:
            selected.setdefault(q.input[0], [])
            selected[q.input[0]] += [q.input[1]] + [dq.input[1] for dq in dqs]
    return selected


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--matched-onnx", type=Path, required=True)
    parser.add_argument("--actstats", type=Path, required=True,
                        help="nn_actstats capture JSON (per_tensor.<name>.per_channel.amax_max)")
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--verify-against", type=Path, default=None)
    args = parser.parse_args()
    if args.out.exists():
        raise SystemExit(f"refusing to overwrite {args.out}")
    model = onnx.load(str(args.matched_onnx))
    stats = json.loads(args.actstats.read_text(encoding="utf-8"))["per_tensor"]
    selected = regression_consumed_quantizers(model)
    by_name = {init.name: init for init in model.graph.initializer}
    changed = {}
    for tensor, scale_names in sorted(selected.items()):
        amax = float(stats[tensor]["per_channel"]["amax_max"])
        for name in sorted(set(scale_names)):
            old = numpy_helper.to_array(by_name[name])
            new = np.asarray(amax / 127.0, dtype=old.dtype).reshape(old.shape)
            by_name[name].CopyFrom(numpy_helper.from_array(new, name))
            changed[name] = {"tensor": tensor, "old": float(old), "new": float(new)}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    onnx.save(model, str(args.out))
    report = {"matched_onnx_sha256": sha256_file(args.matched_onnx),
              "actstats_sha256": sha256_file(args.actstats),
              "out_sha256": sha256_file(args.out),
              "n_quantizers": len(selected), "n_scale_initializers": len(changed),
              "shared_boundary_tensors": sorted(t for t in selected if not t.startswith(HEAD_PREFIX))}
    if args.verify_against is not None:
        ref = {i.name: numpy_helper.to_array(i) for i in onnx.load(str(args.verify_against)).graph.initializer}
        ours = {i.name: numpy_helper.to_array(i) for i in model.graph.initializer}
        mismatched = [k for k in ref if k not in ours or ref[k].dtype != ours[k].dtype
                      or not np.array_equal(ref[k], ours[k])]
        report["verify_against_sha256"] = sha256_file(args.verify_against)
        report["initializers_identical"] = not mismatched and set(ref) == set(ours)
        report["n_mismatched"] = len(mismatched)
    print(json.dumps(report, indent=2))
    if args.verify_against is not None and not report["initializers_identical"]:
        raise SystemExit("MAXREG PATCH VERIFY FAILED")


if __name__ == "__main__":
    main()
