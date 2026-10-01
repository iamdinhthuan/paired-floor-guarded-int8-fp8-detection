"""Aggregate head-activation statistics (B5).

Reads capture reports under submission_support_20260911/nn_actstats/*.json
(each produced by src/capture_head_activations.py over the 512-image
clean-calibration manifest of the dataset) and emits a compact ledger:

  per graph: {region: {n_tensors, amax_med_of_channel_medians, max_amax,
                       kurt_med_of_channel_medians, max_kurt}}

plus the static quantizer-scale summary already inspected
(analysis/inspect_head_scales.py output, if present).
"""
import json
import re
import sys
from pathlib import Path

ROOT = Path(sys.argv[1] if len(sys.argv) > 1 else "submission_support_20260911/nn_actstats")
OUT = Path(sys.argv[2] if len(sys.argv) > 2 else "submission_support_20260911/nn_actstats_summary.json")

# FPN level mapping on the torchvision graph: layer_blocks.0/1/2 are the
# P3/P4/P5 lateral blocks, extra_blocks/p6 /p7 the coarser levels; head-internal
# conv tensors carry a per-level unroll suffix (conv.N.M_K -> level K, with the
# unsuffixed name being P3, verified against the boundary DQ edges of the
# matched INT8 QDQ graph).
_BACKBONE_LVL = {"layer_blocks.0": "P3", "layer_blocks.1": "P4",
                 "layer_blocks.2": "P5", "extra_blocks/p6": "P6",
                 "extra_blocks/p7": "P7"}
_UNROLL_IDX = {None: "P3", "1": "P4", "2": "P5", "3": "P6", "4": "P7"}


def tensor_level(name: str):
    for k, v in _BACKBONE_LVL.items():
        if k in name:
            return v
    # RetinaNet export: conv.<tower>.<stage>[_<level>]/
    mm = re.search(r"conv\.(\d)\.(\d)(?:_(\d))?/", name)
    if mm:
        return _UNROLL_IDX.get(mm.group(3))
    # FCOS export: conv.<stage>[_<level>]/ (head module unrolled per level)
    mm = re.search(r"conv\.(\d+)(?:_(\d+))?/", name)
    return _UNROLL_IDX.get(mm.group(2)) if mm else None


def per_level(d: dict) -> dict:
    import numpy as np
    rows = {}
    for name, rec in (d.get("per_tensor") or {}).items():
        lvl = tensor_level(name)
        if lvl is None:
            continue
        pc = rec.get("per_channel") or {}
        rows.setdefault(rec.get("region"), {}).setdefault(lvl, []).append(pc)
    out = {}
    for region, lvls in rows.items():
        out[region] = {}
        for lvl, pcs in lvls.items():
            am = np.array([x["amax_med"] for x in pcs])
            ku = np.array([x["kurt_med"] for x in pcs])
            out[region][lvl] = {"n_tensors": len(pcs),
                                "amax_med": float(np.median(am)),
                                "kurt_med": float(np.median(ku))}
    return out

def main() -> None:
    reports = {}
    for p in sorted(ROOT.glob("*.json")):
        if p.name == "nn_actstats_summary.json" or p.name.startswith("qdq_"):
            continue
        d = json.loads(p.read_text())
        # additionally aggregate per-tensor p95-of-channel statistics so the
        # tail claim does not rest on a single max
        import numpy as np
        tail = {}
        for t, s in (d.get("per_tensor") or {}).items():
            pc = s.get("per_channel") or {}
            tail.setdefault(s.get("region"), []).append(pc)
        p95 = {}
        for region, lst in tail.items():
            kp = np.array([x["kurt_p95"] for x in lst if "kurt_p95" in x])
            ap95 = np.array([x["amax_p95"] for x in lst if "amax_p95" in x])
            if kp.size:
                p95[region] = {"kurt_p95_med": float(np.median(kp)),
                               "amax_p95_med": float(np.median(ap95))}
        reports[p.stem] = {
            "onnx_sha256": d.get("onnx_sha256"),
            "calib_sha256": d.get("calib_sha256"),
            "n_images": d.get("n_images"),
            "n_captured_tensors": d.get("n_captured_tensors"),
            "summary": d.get("summary", {}),
            "p95_of_channel": p95,
            "per_level": per_level(d),
        }

    # static quantizer scales for the INT8-matched siblings (inspect_head_scales)
    # key map: capture graph -> matched quantized ONNX head-scales record
    static_dir = ROOT / "static_scales"
    static_map = {
        "kitti_retinanet": "kitti_retinanet_int8-matched512",
        "voc_retinanet": "voc_retinanet_int8-matched512",
        "coco_retinanet_pretrained": "coco_retinanet_int8-matched512",
    }
    for cap, st in static_map.items():
        p = static_dir / f"{st}.json"
        if cap in reports and p.is_file():
            s = json.loads(p.read_text()).get("summary", {})
            reports[cap]["static_scales"] = {
                k: {"n_scales": v["n_scales"], "amax_median": v["amax_median"],
                    "amax_max": v["amax_max"],
                    "amax_max_over_median": v["amax_max_over_median"]}
                for k, v in s.items() if "/" in k
            }

    OUT.parent.mkdir(parents=True, exist_ok=True)
    OUT.write_text(json.dumps({"root": str(ROOT), "graphs": reports}, indent=2))
    for name, r in reports.items():
        print(f"== {name} (n={r['n_images']} imgs, {r['n_captured_tensors']} tensors)")
        for region, s in sorted(r["summary"].items()):
            print(f"   {region:22s} nt={s['n_tensors']:3d} "
                  f"amax_med={s['amax_med_of_channel_medians']:8.3f} "
                  f"amax_max={s['max_amax']:9.2f} "
                  f"kurt_med={s['kurt_med_of_channel_medians']:8.2f} "
                  f"kurt_max={s['max_kurt']:10.1f}")


if __name__ == "__main__":
    main()
