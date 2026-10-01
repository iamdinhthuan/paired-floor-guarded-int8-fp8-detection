"""Quantizer-structure report for a matched INT8 QDQ graph (regression head).

For each frozen int8-matched512 ONNX (hash-bound by the onnx.json manifest
under outputs/nn_paired_protocol_v1_20260924/engines/) this records, per
activation quantizer on a tensor consumed by /head/regression_head/ nodes:

  - the tensor name, its FPN level (P3-P7, mapped as in nn_actstats_report),
  - whether it is a shared head boundary tensor (consumed by both heads),
  - the calibrated scale and whether the quantizer is per-tensor
    (scalar scale) or per-axis (vector scale + axis attribute),

plus a graph-level summary: number of regression-head activation quantizers,
their level coverage, and the weight-quantizer granularity inside the head.
The ONNX file is not redistributed; the record is bound by
source_onnx_sha256 from the frozen onnx.json manifest.

Usage:
    python3 analysis/nn_qdq_structure_report.py \
        --onnx /path/to/model.onnx \
        --registry outputs/.../kitti_retinanet_int8-matched512/onnx.json \
        --out submission_support_20260911/nn_actstats/qdq_structure_kitti.json
"""
import argparse
import json
import re
import sys
from pathlib import Path



import onnx  # noqa: E402
from onnx import numpy_helper  # noqa: E402

# duplicated from nn_actstats_report.py on purpose: that module executes its
# report at import time from sys.argv, so importing it would clobber files.
_BACKBONE_LVL = {"layer_blocks.0": "P3", "layer_blocks.1": "P4",
                 "layer_blocks.2": "P5", "extra_blocks/p6": "P6",
                 "extra_blocks/p7": "P7"}
_UNROLL_IDX = {None: "P3", "1": "P4", "2": "P5", "3": "P6", "4": "P7"}


def tensor_level(name: str):
    for k, v in _BACKBONE_LVL.items():
        if k in name:
            return v
    mm = re.search(r"conv\.(\d)\.(\d)(?:_(\d))?/", name)
    return _UNROLL_IDX.get(mm.group(3)) if mm else None


def sha256_file(path: Path) -> str:
    import hashlib
    h = hashlib.sha256()
    with open(path, "rb") as fh:
        for chunk in iter(lambda: fh.read(1 << 20), b""):
            h.update(chunk)
    return h.hexdigest()


def main() -> None:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--onnx", type=Path, required=True)
    ap.add_argument("--registry", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()

    reg = json.loads(args.registry.read_text())
    onnx_sha = sha256_file(args.onnx)
    if onnx_sha != reg.get("output_onnx_sha256"):
        raise SystemExit(f"ONNX sha mismatch: {onnx_sha} != manifest")

    m = onnx.load(str(args.onnx))
    g = m.graph
    inits = {i.name: numpy_helper.to_array(i) for i in g.initializer}
    prod = {}
    consumers = {}
    for n in g.node:
        for o in n.output:
            prod[o] = n
        for i in n.input:
            consumers.setdefault(i, []).append(n)

    reg_nodes = [n for n in g.node if n.name.startswith("/head/regression_head/")]
    consumed = set()
    for n in reg_nodes:
        consumed.update(n.input)

    rows = []
    for t in sorted(consumed):
        dq = prod.get(t)
        if dq is None or dq.op_type != "DequantizeLinear":
            continue
        q = prod.get(dq.input[0])
        if q is None or q.op_type != "QuantizeLinear" or q.input[0] in inits:
            continue
        x = q.input[0]
        scale = inits.get(q.input[1])
        shared = any(c.name.startswith("/head/classification_head/")
                     for c in consumers.get(t, []))
        rows.append({
            "tensor": x,
            "fpn_level": tensor_level(x),
            "shared_with_classification_head": bool(shared),
            "scale": float(scale.ravel()[0]) if scale is not None else None,
            "scale_elements": int(scale.size) if scale is not None else None,
            "granularity": "per-tensor" if scale is not None and scale.size == 1
                           else "per-axis",
        })

    w_granularity = {}
    for n in g.node:
        if n.op_type != "DequantizeLinear":
            continue
        src = n.input[0]
        if "regression_head" not in src and "regression_head" not in n.name:
            continue
        wq = prod.get(src)
        if wq is None or wq.op_type != "QuantizeLinear":
            continue  # activation path, already covered above
        arr = inits.get(wq.input[1])
        axis = next((a.i for a in wq.attribute if a.name == "axis"),
                    next((a.i for a in n.attribute if a.name == "axis"), None))
        w_granularity[src] = {
            "scale_elements": int(arr.size) if arr is not None else None,
            "axis": axis}

    out = {
        "schema_version": 1,
        "dataset": reg.get("dataset"),
        "arm": "int8-matched512",
        "source_onnx": str(args.onnx),
        "source_onnx_sha256": onnx_sha,
        "registry": str(args.registry),
        "registry_sha256": sha256_file(args.registry),
        "n_regression_head_nodes": len(reg_nodes),
        "n_regression_head_activation_quantizers": len(rows),
        "activation_quantizers": rows,
        "weight_quantizers": w_granularity,
    }
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(out, indent=2))
    print(json.dumps({"written": str(args.out),
                      "n_act_quantizers": len(rows),
                      "levels": sorted({r["fpn_level"] for r in rows}),
                      "all_per_tensor": all(r["granularity"] == "per-tensor"
                                            for r in rows)}, indent=1))


if __name__ == "__main__":
    main()
