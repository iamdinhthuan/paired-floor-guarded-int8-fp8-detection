#!/usr/bin/env python3
"""Extract per-tensor activation/weight scales from a QDQ ONNX, split by head.

For every QuantizeLinear node we record (region, tensor-role, scale).
Region: regression_head / classification_head / other, from the QL node's
position (its name prefix under /head/.../). Role: activation if the QL input
is produced by a non-initializer tensor, weight if the QL sits on a
quantized initializer path. Reports amax (=scale*127) summaries per region.
"""
import json
import sys
import numpy as np
import onnx
from onnx import numpy_helper

def main(path):
    m = onnx.load(path)
    init = {i.name: numpy_helper.to_array(i) for i in m.graph.initializer}
    scale_of = {}   # ql_node_output -> scale array
    recs = []
    for n in m.graph.node:
        if n.op_type != "QuantizeLinear":
            continue
        region = ("regression_head" if "/head/regression_head/" in n.name
                  else "classification_head" if "/head/classification_head/" in n.name
                  else "other")
        x, s = n.input[0], n.input[1] if len(n.input) > 1 else None
        if s not in init:
            continue
        scale = np.asarray(init[s]).ravel()
        # weight QL: input is an initializer (constant-folded conv weight)
        role = "weight" if x in init else "activation"
        for sv in scale:
            recs.append((region, role, float(sv)))
    agg = {}
    for region in ("regression_head", "classification_head", "other"):
        for role in ("weight", "activation"):
            v = np.array([s for r, ro, s in recs if r == region and ro == role])
            if v.size == 0:
                continue
            amax = v * 127.0
            agg[f"{region}/{role}"] = {
                "n_scales": int(v.size),
                "amax_min": float(amax.min()), "amax_median": float(np.median(amax)),
                "amax_p95": float(np.percentile(amax, 95)),
                "amax_max": float(amax.max()),
                "amax_max_over_median": float(amax.max() / np.median(amax)),
            }
    print(json.dumps(agg, indent=1))
    out = {"onnx": path, "records": recs, "summary": agg}
    with open(path + ".head_scales.json", "w") as f:
        json.dump({"onnx": path, "summary": agg,
                   "records": [{"region": r, "role": ro, "scale": s} for r, ro, s in recs]}, f, indent=1)
    print("wrote", path + ".head_scales.json")

if __name__ == "__main__":
    main(sys.argv[1])
