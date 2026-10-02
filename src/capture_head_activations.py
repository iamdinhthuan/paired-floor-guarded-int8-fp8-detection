#!/usr/bin/env python3
"""Per-channel activation statistics inside detector heads on calibration images.

Loads the frozen FP32 source ONNX, marks every non-initializer tensor consumed
by nodes under /head/regression_head/ and /head/classification_head/ as a graph
output, runs the hash-bound calibration images through onnxruntime, and reports
per-channel amax / kurtosis summaries per head region (boundary inputs vs
internal conv activations vs output-conv inputs).
"""
from __future__ import annotations

import hashlib
import json
import sys
import time
from pathlib import Path

import numpy as np
import onnx
from onnx import helper, TensorProto

ROOT = Path("/home/thuan/topic_c_ivc")
sys.path.insert(0, str(ROOT / "src"))


def sha256_file(path) -> str:
    d = hashlib.sha256()
    with open(path, "rb") as h:
        for blk in iter(lambda: h.read(1 << 20), b""):
            d.update(blk)
    return d.hexdigest()


def head_consumed_tensors(model):
    """Map head-region -> tensors consumed by that region's nodes (non-initializer)."""
    inits = {i.name for i in model.graph.initializer}
    regions = {"regression_head": set(), "classification_head": set(), "head_shared": set()}
    for n in model.graph.node:
        name = n.name or ""
        if "regression_head" in name:
            key = "regression_head"
        elif "classification_head" in name:
            key = "classification_head"
        elif "/head/" in name:
            key = "head_shared"
        else:
            continue
        for t in n.input:
            if t and t not in inits:
                regions[key].add(t)
    return {k: sorted(v) for k, v in regions.items()}


def channel_stats(acc: dict) -> dict:
    """Per-channel max|x| and excess kurtosis from streaming raw moments."""
    out = {}
    for name, a in acc.items():
        n = a["n"]
        if n == 0:
            continue
        m1 = a["s1"] / n
        m2 = np.maximum(a["s2"] / n - m1**2, 1e-12)
        m3 = a["s3"] / n - 3 * m1 * (a["s2"] / n) + 2 * m1**3
        m4 = (a["s4"] / n - 4 * m1 * (a["s3"] / n)
              + 6 * m1**2 * (a["s2"] / n) - 3 * m1**4)
        kurt = m4 / m2**2 - 3.0
        amax = a["amax"]
        out[name] = {"tensor": name, "channels": int(amax.size),
                     "pixels_per_channel": int(n),
                     "amax_min": float(amax.min()), "amax_med": float(np.median(amax)),
                     "amax_max": float(amax.max()),
                     "amax_p95": float(np.percentile(amax, 95)),
                     "kurt_med": float(np.median(kurt)), "kurt_max": float(kurt.max()),
                     "kurt_p95": float(np.percentile(kurt, 95)),
                     "sd_med": float(np.median(np.sqrt(m2)))}
    return out


def _accumulate(e, arr: np.ndarray) -> None:
    flat = np.moveaxis(arr, 1, 0).reshape(arr.shape[1], -1).astype(np.float64)
    e["amax"] = np.maximum(e["amax"], np.abs(flat).max(axis=1))
    e["s1"] += flat.sum(axis=1)
    e["s2"] += (flat**2).sum(axis=1)
    e["s3"] += (flat**3).sum(axis=1)
    e["s4"] += (flat**4).sum(axis=1)
    e["n"] += flat.shape[1]


def main():
    import argparse
    ap = argparse.ArgumentParser()
    ap.add_argument("--onnx", required=True)
    ap.add_argument("--calib-manifest", required=True)
    ap.add_argument("--imgsz", type=int, default=640)
    ap.add_argument("--limit", type=int, default=0)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()

    model = onnx.load(args.onnx)
    regions = head_consumed_tensors(model)
    capture = sorted(set().union(*regions.values()))
    init_names = {i.name for i in model.graph.initializer}
    capture = [t for t in capture if t not in init_names]
    # mark capture tensors as outputs
    existing = {o.name for o in model.graph.output}
    vi = {v.name: v for v in list(model.graph.value_info) + list(model.graph.output) + list(model.graph.input)}
    for t in capture:
        if t in existing:
            continue
        v = vi.get(t)
        if v is None:
            v = helper.make_empty_tensor_value_info(t)
        else:
            v = helper.make_empty_tensor_value_info(t)  # dtype/shape not needed
        model.graph.output.append(v)
    tmp = Path(args.out).with_suffix(".capture.onnx")
    onnx.save(model, tmp)

    manifest = json.loads(Path(args.calib_manifest).read_text())
    marker = Path(str(args.calib_manifest) + ".complete")
    expected = manifest.get("manifest_sha256") or manifest.get("calibration_sha256")
    if not marker.is_file() or marker.read_text().strip() != expected:
        raise SystemExit("calibration manifest lacks .complete marker")
    manifest_hash = expected
    from topic_c.cross_family import preprocess_retinanet
    root = Path(manifest["dataset_root"]).resolve()
    imgs = manifest["records"][: args.limit or len(manifest["records"])]

    import onnxruntime as ort
    sess = ort.InferenceSession(str(tmp), providers=["CUDAExecutionProvider", "CPUExecutionProvider"])
    out_names = [o.name for o in sess.get_outputs()]
    want = [t for t in capture if t in out_names]
    acc = {}
    t0 = time.time()
    for i, rec in enumerate(imgs, 1):
        p = root / rec["source_relpath"]
        if sha256_file(p) != rec["sha256"]:
            raise SystemExit(f"calib image hash mismatch: {p}")
        x = preprocess_retinanet(str(p), args.imgsz)[0]
        outs = sess.run(want, {"images": x})
        for t, o in zip(want, outs):
            a = np.asarray(o)
            if a.dtype != np.float32 or a.ndim < 3:
                continue
            c = a.shape[1]
            e = acc.setdefault(t, {"amax": np.zeros(c, dtype=np.float64),
                                   "s1": np.zeros(c), "s2": np.zeros(c),
                                   "s3": np.zeros(c), "s4": np.zeros(c), "n": 0})
            _accumulate(e, a)
        if i % 64 == 0:
            print(f"{i}/{len(imgs)} {time.time()-t0:.0f}s", flush=True)

    def region_of(t):
        # tensor consumed by which region's nodes; multi-head tensors = shared
        hits = [r for r, names in regions.items() if t in names]
        if "regression_head" in hits and "classification_head" in hits:
            return "shared_by_heads"
        if "regression_head" in hits:
            return "regression_head"
        if "classification_head" in hits:
            return "classification_head"
        return "+".join(hits) if hits else "other"
    per_tensor = channel_stats(acc)
    stats = {t: {"region": region_of(t), "per_channel": pc} for t, pc in per_tensor.items()}

    # aggregate per region
    agg = {}
    for t, s in stats.items():
        agg.setdefault(s["region"], []).append(s["per_channel"])
    summary = {}
    for region, lst in agg.items():
        amax_med = np.array([x["amax_med"] for x in lst])
        amax_max = np.array([x["amax_max"] for x in lst])
        kurt_max = np.array([x["kurt_max"] for x in lst])
        kurt_med = np.array([x["kurt_med"] for x in lst])
        summary[region] = {
            "n_tensors": len(lst),
            "amax_med_of_channel_medians": float(np.median(amax_med)),
            "max_amax": float(amax_max.max()),
            "kurt_med_of_channel_medians": float(np.median(kurt_med)),
            "max_kurt": float(kurt_max.max()),
        }
    report = {
        "schema_version": 1, "onnx": args.onnx, "onnx_sha256": sha256_file(args.onnx),
        "calib_manifest": args.calib_manifest, "calib_sha256": manifest_hash,
        "n_images": len(imgs), "imgsz": args.imgsz,
        "n_captured_tensors": len(want), "summary": summary, "per_tensor": stats,
        "created_at_utc": __import__("datetime").datetime.now(__import__("datetime").timezone.utc).isoformat(),
    }
    Path(args.out).write_text(json.dumps(report, indent=1) + "\n")
    print(json.dumps(summary, indent=1))
    print("wrote", args.out)


if __name__ == "__main__":
    main()
