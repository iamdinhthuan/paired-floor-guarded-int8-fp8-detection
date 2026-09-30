"""Aggregate TensorRT rebuild-variance evidence (B3).

Reads outputs under submission_support_20260911/nn_rebuild_variance_v1_20260930/:
  <arm>/rebuild-NN/record.json          -- engine + inspector sha256, build policy
  <arm>/rebuild-NN/engine_inspector.json
  ap_eval/metrics/<arm>__rN__<cond>.json -- AP per rebuild (subset of arms)

Emits a JSON ledger + console table:
- distinct engine_sha256 / inspector_sha256 counts per arm (expect 5/5 each)
- AP spread (max-min) across rebuilds where ap_eval exists
"""
import json
import sys
from collections import defaultdict
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "submission_support_20260911/nn_rebuild_variance_v1_20260930")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "submission_support_20260911/nn_rebuild_variance_summary.json")

arms = defaultdict(list)
for rec in sorted(ROOT.glob("*/rebuild-*/record.json")):
    d = json.loads(rec.read_text())
    ins = rec.with_name("engine_inspector.json")
    if ins.is_file():
        d["n_layers"] = len(json.loads(ins.read_text()).get("Layers", []))
    arms[d["arm"]].append(d)

ap = defaultdict(dict)
for m in sorted(ROOT.glob("ap_eval/metrics/*.json")):
    # <arm>__rN__<condition>.json
    stem = m.stem
    arm, rest = stem.rsplit("__r", 1)
    r, cond = rest.split("__", 1)
    d = json.loads(m.read_text())
    ap[(arm, cond)][int(r)] = d.get("stats", {}).get("AP")

summary = {"root": str(ROOT), "arms": {}, "ap_spread": {}}
all_distinct = True
for arm, recs in sorted(arms.items()):
    eng = {r["engine_sha256"] for r in recs}
    insp = {r["engine_inspector_sha256"] for r in recs}
    src = {r["source_onnx_sha256"] for r in recs}
    distinct = len(eng) == len(recs) and len(insp) == len(recs)
    all_distinct &= distinct
    summary["arms"][arm] = {
        "n_rebuilds": len(recs),
        "distinct_engine_sha256": len(eng),
        "distinct_inspector_sha256": len(insp),
        "source_onnx_sha256": sorted(src),
        "tensorrt_version": recs[0].get("tensorrt_version"),
        "build_seconds": [round(r.get("build_seconds", 0), 1) for r in recs],
        "engine_bytes": [r.get("engine_bytes") for r in recs],
        "n_layers": [r.get("n_layers") for r in recs],
        "distinct_layer_counts": len({r.get("n_layers") for r in recs}),
        "engine_sha256": sorted(eng),
        "inspector_sha256": sorted(insp),
    }

for (arm, cond), d in sorted(ap.items()):
    vals = {r: v for r, v in d.items() if v is not None}
    if vals:
        summary["ap_spread"][f"{arm}__{cond}"] = {
            "per_rebuild_ap": vals,
            "n": len(vals),
            "min": min(vals.values()), "max": max(vals.values()),
            "spread": round(max(vals.values()) - min(vals.values()), 4),
        }

summary["all_rebuilds_distinct"] = all_distinct
summary["total_rebuilds"] = sum(a["n_rebuilds"] for a in summary["arms"].values())
OUT.write_text(json.dumps(summary, indent=2))

for arm, a in summary["arms"].items():
    print(f"{arm:45s} n={a['n_rebuilds']} eng {a['distinct_engine_sha256']}/{a['n_rebuilds']} insp {a['distinct_inspector_sha256']}/{a['n_rebuilds']}")
print(f"\nALL DISTINCT: {all_distinct}  total={summary['total_rebuilds']}")
print("\nAP spread across rebuilds:")
for k, v in summary["ap_spread"].items():
    print(f"  {k:60s} spread={v['spread']:.4f}  {v['per_rebuild_ap']}")
