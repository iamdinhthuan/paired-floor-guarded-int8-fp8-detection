#!/usr/bin/env python3
"""Phase-B paired image bootstrap: per-block arm x cell AP draw caches.

For each block (dataset x model) every arm/condition cell is evaluated
once with pycocotools, then B shared image resamples are accumulated so
all arm contrasts are paired by a common schedule. Outputs one npz draw
cache plus a JSON summary (point estimates + percentile intervals for
clean AP, corrupted-mean AP, retention, and G0/Gc/deltaE contrasts).
"""
from __future__ import annotations

import argparse
import hashlib
import json
import multiprocessing as mp
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parent))
from topic_c.manifest import sha256_file  # noqa: E402

CORRUPTIONS = ("fog", "gaussian_noise", "jpeg", "motion_blur")
SEVERITIES = (1, 3, 5)
ATTEMPT_DIRS = [
    "nn_fcos_replication_v1_20260927",
    "nn_corruptcalib_fold2_v1_20260925",
    "nn_q95calib_v1_20260924",
    "nn_paired_protocol_v1_20260924",
    "nn_corruptcalib_v1_20260924",
    "kitti_pilot_117_v1",
    "voc_pilot_117_v1",
    "coco_uniform_p0_v1",
    "codec_control_p0_v1",
    "cross_family_v1",
]
ANNOTATIONS = {
    "kitti": "manifests/annotations/kitti_val_ultralytics_v1_coco.json",
    "voc": "manifests/annotations/voc_val_ultralytics_v1_coco.json",
    "coco": "data/datasets/coco_pilot_v1_20260923/annotations/instances_val2017.json",
}
COCO_SUBSET = "manifests/subsets/nn_paired_protocol_v1_20260924/coco_val2017_paired_2000_s20260924.json"

_GT = None
_SAMPLES = None
_SUBSET_IDS = None


def block_seed(namespace: str, dataset: str, model: str,
               shared_dataset_schedule: bool = False) -> int:
    key = (f"{namespace}|{dataset}" if shared_dataset_schedule
           else f"{namespace}|{dataset}|{model}")
    digest = hashlib.sha256(key.encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def locate_cell(root: Path, dataset: str, model: str, arm: str, corruption: str,
                severity: int, coco_subset: bool, attempts=None, prefix: str | None = None) -> dict | None:
    """Resolve prediction + input record paths for one arm/condition cell."""
    if attempts is None:
        attempts = ("coco_uniform_p0_v1", "codec_control_p0_v1", "nn_coco_pretrained_v1_20260930") if coco_subset else tuple(ATTEMPT_DIRS)
    prefix = prefix or f"{dataset}_*"
    for attempt in attempts:
        base = root / "outputs" / "predictions" / attempt
        if not base.is_dir():
            continue
        for pred in sorted(base.glob(f"{prefix}__{model}__*__{corruption}-s{severity}*.json")):
            stem = pred.stem
            run = root / "manifests" / "runs" / attempt / f"{stem}.json"
            inp = root / "outputs" / "inputs" / attempt / f"{stem}.json"
            if not run.is_file() or not inp.is_file():
                continue
            record = json.loads(run.read_text(encoding="utf-8"))
            precision = record.get("precision", "")
            parts = stem.split("__")
            arm_ok = len(parts) > 2 and parts[2] == arm
            if model.startswith("retinanet"):
                arm_ok = arm_ok or (attempt == "cross_family_v1" and precision == arm)
            else:
                arm_ok = arm_ok or precision in (arm, f"{arm}-entropy", f"{arm}-none")
            if not arm_ok:
                continue
            if record.get("prediction_sha256") and sha256_file(pred) != record["prediction_sha256"]:
                continue
            return {"attempt": attempt, "predictions": str(pred), "input_record": str(inp),
                    "run_record": str(run), "stem": stem}
    return None


def discover_block(root: Path, dataset: str, model: str, arms: dict[str, str],
                   attempts=None, prefix: str | None = None, annotations: str | None = None) -> dict:
    coco_subset = dataset == "coco"
    cells = []
    for label, arm in arms.items():
        for corruption in ("clean", "codec-control", *CORRUPTIONS):
            severities = (0,) if corruption in ("clean", "codec-control") else SEVERITIES
            for severity in severities:
                found = locate_cell(root, dataset, model, arm, corruption, severity, coco_subset,
                                    attempts, prefix)
                if found:
                    found.update(arm=label, corruption=corruption, severity=severity)
                    cells.append(found)
    return {"dataset": dataset, "model": model, "annotations": annotations or ANNOTATIONS[dataset],
            "cells": cells}


def cluster_schedule(seed: int, n_boot: int, position_clusters: list[str]) -> list[np.ndarray]:
    """Cluster (e.g. KITTI drive) bootstrap: resample whole clusters with
    replacement and keep every image of each drawn cluster; shared by all arms."""
    labels = sorted(set(position_clusters))
    members = {label: [] for label in labels}
    for position, label in enumerate(position_clusters):
        members[label].append(position)
    members = [np.asarray(members[label], dtype=np.int64) for label in labels]
    rng = np.random.default_rng(seed)
    draws = rng.choice(len(labels), size=(n_boot, len(labels)), replace=True)
    return [np.concatenate([members[k] for k in row]) for row in draws]


def _init_worker(annotations: str, seed: int, n_boot: int, n_images: int,
                 subset_ids, src_dir: str, position_clusters=None):
    """Each worker builds its own GT handle and regenerates the shared schedule."""
    global _GT, _SAMPLES, _SUBSET_IDS
    sys.path.insert(0, src_dir)
    from pycocotools.coco import COCO
    _GT = COCO(annotations)
    if position_clusters is None:
        rng = np.random.default_rng(seed)
        _SAMPLES = rng.choice(n_images, size=(n_boot, n_images), replace=True)
    else:
        _SAMPLES = cluster_schedule(seed, n_boot, position_clusters)
    _SUBSET_IDS = subset_ids


def _accumulate_cell(task):
    """One task per cell: build the eval, then accumulate all shared draws."""
    from paired_bootstrap import accumulate_ap, build_eval
    cell_index, cell = task
    ids = _SUBSET_IDS if _SUBSET_IDS is not None else \
        json.loads(Path(cell["input_record"]).read_text(encoding="utf-8"))["image_ids"]
    predictions = json.loads(Path(cell["predictions"]).read_text(encoding="utf-8"))
    evaluation = build_eval(_GT, predictions, ids)
    point = np.asarray(accumulate_ap(evaluation, list(range(len(ids)))))
    draws = np.stack([np.asarray(accumulate_ap(evaluation, sample)) for sample in _SAMPLES])
    return cell_index, point, draws


def run_block(root: Path, spec: dict, n_boot: int, seed_namespace: str,
              jobs: int, out_dir: Path, shared_schedule: bool = False,
              cluster_manifest: Path | None = None) -> dict:
    from paired_bootstrap import percentile

    dataset, model = spec["dataset"], spec["model"]
    cells = spec["cells"]
    if not cells:
        raise SystemExit(f"BOOTSTRAP REFUSED: no cells resolved for {dataset}/{model}")
    annotations = (root / spec["annotations"]).resolve()
    if root.resolve() not in annotations.parents:
        raise SystemExit("BOOTSTRAP REFUSED: annotations escape root")
    if not annotations.is_file():
        raise SystemExit(f"BOOTSTRAP REFUSED: missing annotations {annotations}")

    coco_subset = dataset == "coco"
    subset_ids = None
    if coco_subset:
        subset = json.loads((root / COCO_SUBSET).read_text(encoding="utf-8"))
        subset_ids = subset["image_ids"]

    meta = []
    for cell in cells:
        input_record = json.loads(Path(cell["input_record"]).read_text(encoding="utf-8"))
        ids = subset_ids if coco_subset else input_record["image_ids"]
        run_record = json.loads(Path(cell["run_record"]).read_text(encoding="utf-8"))
        if sha256_file(cell["predictions"]) != run_record["prediction_sha256"]:
            raise SystemExit(f"BOOTSTRAP REFUSED: prediction hash mismatch {cell['stem']}")
        meta.append({"arm": cell["arm"], "condition": f"{cell['corruption']}-s{cell['severity']}",
                     "ids": ids, "stem": cell["stem"]})

    reference_ids = meta[0]["ids"]
    if any(entry["ids"] != reference_ids for entry in meta):
        raise SystemExit(f"BOOTSTRAP REFUSED: paired image ids diverge in {dataset}/{model}")
    if len(reference_ids) != len(set(reference_ids)):
        raise SystemExit("BOOTSTRAP REFUSED: image ids not unique before resampling")
    for entry in meta:
        del entry["ids"]

    n_images = len(reference_ids)
    seed = block_seed(seed_namespace, dataset, model,
                      shared_dataset_schedule=shared_schedule)
    position_clusters = None
    resampling = "image"
    if cluster_manifest is not None:
        clusters = json.loads(Path(cluster_manifest).read_text(encoding="utf-8"))["clusters"]
        missing = [i for i in reference_ids if str(i) not in clusters]
        if missing:
            raise SystemExit(f"BOOTSTRAP REFUSED: {len(missing)} image ids lack a cluster label")
        position_clusters = [clusters[str(i)] for i in reference_ids]
        resampling = f"cluster:{sha256_file(cluster_manifest)}"

    point = np.full((len(cells), 4), np.nan, dtype=np.float64)
    draws = np.full((n_boot, len(cells), 4), np.nan, dtype=np.float64)
    src_dir = str(Path(__file__).resolve().parent)
    initargs = (str(annotations), seed, n_boot, n_images, subset_ids, src_dir, position_clusters)
    tasks = list(enumerate(cells))
    done = 0
    if jobs > 1:
        with mp.Pool(jobs, initializer=_init_worker, initargs=initargs) as pool:
            for cell_index, cell_point, cell_draws in pool.imap_unordered(_accumulate_cell, tasks):
                point[cell_index] = cell_point
                draws[:, cell_index, :] = cell_draws
                done += 1
                print(f"cells {done}/{len(cells)} ({meta[cell_index]['stem']})", flush=True)
    else:
        _init_worker(*initargs)
        for task in tasks:
            cell_index, cell_point, cell_draws = _accumulate_cell(task)
            point[cell_index] = cell_point
            draws[:, cell_index, :] = cell_draws
            done += 1
            print(f"cells {done}/{len(cells)} ({meta[cell_index]['stem']})", flush=True)

    arms = []
    for entry in meta:
        if entry["arm"] not in arms:
            arms.append(entry["arm"])
    arm_cond = {arm: {entry["condition"]: i for i, entry in enumerate(meta) if entry["arm"] == arm}
                for arm in arms}
    arm_corrupt = {arm: [i for c, i in conds.items() if c not in ("clean-s0", "codec-control-s0")]
                   for arm, conds in arm_cond.items()}

    npz = {"samples_seed": np.asarray(seed, dtype=np.int64), "n_boot": np.asarray(n_boot, dtype=np.int64)}
    for arm in arms:
        conds = arm_cond[arm]
        npz[f"{arm}__clean"] = draws[:, conds["clean-s0"], 0] if "clean-s0" in conds else np.full(n_boot, np.nan)
        npz[f"{arm}__q95"] = draws[:, conds["codec-control-s0"], 0] if "codec-control-s0" in conds else np.full(n_boot, np.nan)
        npz[f"{arm}__corr_mean"] = draws[:, arm_corrupt[arm], 0].mean(axis=1)
        for condition, i in conds.items():
            npz[f"{arm}__cell__{condition}"] = draws[:, i, 0]
    out_dir.mkdir(parents=True, exist_ok=True)
    cache = out_dir / f"{dataset}__{model}__draws.npz"
    if cache.exists():
        raise SystemExit(f"refusing to overwrite draw cache: {cache}")
    np.savez_compressed(cache, **npz)

    summary = {"dataset": dataset, "model": model, "n_images": n_images, "n_boot": n_boot,
               "seed": seed, "resampling": resampling,
               "n_clusters": len(set(position_clusters)) if position_clusters else None, "draw_cache_sha256": sha256_file(cache),
               "cells": meta, "arms": {}, "contrasts": {}}
    for arm in arms:
        conds = arm_cond[arm]
        corr = arm_corrupt[arm]
        summary["arms"][arm] = {
            "n_cells": len(conds),
            "clean_orig_ap": float(point[conds["clean-s0"], 0]) if "clean-s0" in conds else None,
            "clean_q95_ap": float(point[conds["codec-control-s0"], 0]) if "codec-control-s0" in conds else None,
            "corrupted_mean_ap": float(np.nanmean(point[corr, 0])) if corr else None,
            "ci95_clean": percentile(npz[f"{arm}__clean"]),
            "ci95_corr_mean": percentile(npz[f"{arm}__corr_mean"]),
        }

    def contrast(a: str, b: str) -> dict | None:
        qa, qb = npz[f"{a}__q95"], npz[f"{b}__q95"]
        if np.isnan(qa).all() or np.isnan(qb).all():
            clean_a, clean_b, basis = npz[f"{a}__clean"], npz[f"{b}__clean"], "orig"
        else:
            clean_a, clean_b, basis = qa, qb, "q95"
        g0 = clean_a - clean_b
        gc = npz[f"{a}__corr_mean"] - npz[f"{b}__corr_mean"]
        return {"basis": basis, "G0": percentile(g0), "Gc": percentile(gc),
                "deltaE": percentile(gc - g0),
                "point_deltaE": float(np.nanmean(gc - g0))}

    for a, b in spec.get("contrasts", []):
        result = contrast(a, b)
        if result:
            summary["contrasts"][f"{a}_minus_{b}"] = result

    out_json = out_dir / f"{dataset}__{model}__bootstrap.json"
    if out_json.exists():
        raise SystemExit(f"refusing to overwrite bootstrap summary: {out_json}")
    out_json.write_text(json.dumps(summary, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(out_json), "n_cells": len(cells), "n_boot": n_boot}, indent=2))
    return summary


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("/home/thuan/topic_c_ivc"))
    parser.add_argument("--dataset", required=True, choices=("kitti", "voc", "coco"))
    parser.add_argument("--model", required=True)
    parser.add_argument("--arms", required=True,
                        help="JSON mapping of arm label to run-precision name")
    parser.add_argument("--contrasts", default="[]",
                        help="JSON list of [ref, arm] contrast pairs")
    parser.add_argument("--n-boot", type=int, default=2000)
    parser.add_argument("--jobs", type=int, default=4)
    parser.add_argument("--seed-namespace", default="nn_paired_bootstrap_v1_20260924")
    parser.add_argument("--discover-only", action="store_true")
    parser.add_argument("--cluster-manifest", type=Path, default=None,
                        help="JSON {'clusters': {image_id: label}}; resample whole clusters")
    parser.add_argument("--annotations", default=None,
                        help="override the dataset annotation file (relative to --root)")
    parser.add_argument("--attempts", default=None,
                        help="comma-separated prediction attempt dirs to search (default: built-in list)")
    parser.add_argument("--prefix", default=None,
                        help="prediction stem prefix glob (default: '<dataset>_*')")
    parser.add_argument("--shared-dataset-schedule", action="store_true",
                        help="Seed the resample schedule by (namespace|dataset) only, "
                        "so model blocks on the same dataset share resamples")
    parser.add_argument("--out-dir", type=Path, default=None)
    args = parser.parse_args()
    root = args.root.resolve()
    arms = json.loads(args.arms)
    attempts = tuple(args.attempts.split(",")) if args.attempts else None
    spec = discover_block(root, args.dataset, args.model, arms, attempts, args.prefix, args.annotations)
    spec["contrasts"] = json.loads(args.contrasts)
    print(f"resolved {len(spec['cells'])} cells for {args.dataset}/{args.model}", flush=True)
    if args.discover_only:
        print(json.dumps(spec, indent=2))
        return
    out_dir = args.out_dir or root / "outputs" / "analysis" / "nn_paired_protocol_v1_20260924" / "bootstrap"
    run_block(root, spec, args.n_boot, args.seed_namespace, args.jobs, out_dir,
              shared_schedule=args.shared_dataset_schedule, cluster_manifest=args.cluster_manifest)


if __name__ == "__main__":
    main()
