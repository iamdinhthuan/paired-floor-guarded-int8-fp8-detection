"""Aggregate NN TensorRT latency benchmarks (B4).

Reads per-repetition JSON records from
outputs/benchmarks/nn_latency_v1_20260930/<condition>__rep-NN.json
(custom cudaEventElapsedTime harness, batch=1, warmup=200, iters=500,
idle-gated) and emits a numeric ledger JSON:

- per-condition: median-of-medians latency (ms), IQR across 3 reps
- ratio table per model/dataset pair for precision arms
- keeps provenance: engine sha256, gpu-idle allow list, timestamps

The generator turns this into \\nnLat* macros / supplement table rows.
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np

BENCH = Path(sys.argv[1] if len(sys.argv) > 1 else "outputs/benchmarks/nn_latency_v1_20260930")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "submission_support_20260911/nn_latency_summary.json")

reps = defaultdict(list)
for p in sorted(BENCH.glob("*__rep-*.json")):
    d = json.loads(p.read_text())
    reps[d["condition_id"]].append(d)

summary = {"benchmark_root": str(BENCH), "n_reps_expected": 3,
           "timer": "cudaEventElapsedTime around execute_async_v3, batch=1",
           "conditions": {}}
for cid, ds in sorted(reps.items()):
    meds = [float(d["latency_median_ms"]) for d in ds]
    iqrs = [float(d["latency_iqr_ms"]) for d in ds]
    shas = {d["engine_sha256"] for d in ds}
    allows = {d.get("gpu_idle_query_output", "") for d in ds}
    entry = {
        "model": ds[0]["model"], "dataset": ds[0]["dataset"],
        "precision": ds[0]["precision"],
        "engine_sha256": sorted(shas),
        "n_reps": len(ds), "n_samples_each": [int(d["n_samples"]) for d in ds],
        "rep_medians_ms": meds,
        "median_ms": float(np.median(meds)),
        "spread_ms": float(max(meds) - min(meds)),
        "iqr_ms_typ": float(np.median(iqrs)),
        "idle_allow_consistent": len(allows) == 1,
        "gpu_idle_query_output": sorted(allows),
    }
    summary["conditions"][cid] = entry

# per-family ratios vs fp32 within same model+dataset stem; arm label taken
# from the condition id (precision field collapses int8 variants to one label)
groups = defaultdict(dict)
for cid, e in sorted(summary["conditions"].items()):
    key = (e["dataset"], e["model"])
    # strip dataset+model prefix: condition is <dataset>_<modelish>_<arm>
    arm = cid[len(e["dataset"]) + 1:]
    for prefix in ("retinanet_pretrained_", "retinanet_", "fcos_pretrained_", "fcos_"):
        if arm.startswith(prefix):
            arm = arm[len(prefix):]
            break
    groups[key][arm] = e
ratios = {}
for (dataset, model), arms in sorted(groups.items()):
    base = arms.get("fp32")
    if base is None:
        continue
    ratios[f"{dataset}__{model}"] = {
        arm: round(e["median_ms"] / base["median_ms"], 4)
        for arm, e in sorted(arms.items())
    }
summary["latency_ratio_vs_fp32"] = ratios

OUT.parent.mkdir(parents=True, exist_ok=True)
OUT.write_text(json.dumps(summary, indent=2))
for cid, e in summary["conditions"].items():
    print(f"{cid:50s} {e['median_ms']:7.3f} ms  spread {e['spread_ms']:.3f}  iqr {e['iqr_ms_typ']:.3f}  reps {e['n_reps']}")
print("\nratios vs fp32:")
for k, v in ratios.items():
    print(f"  {k}: {v}")
