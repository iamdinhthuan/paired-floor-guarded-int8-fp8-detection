#!/usr/bin/env python3
"""Continuous, fail-closed analysis of the completed controlled RetinaNet queue.

Reuses the tested overall-AP accumulator; never changes frozen experiments or
the manuscript. Completed AP caches are reusable only after identity checks.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import csv
import fcntl
import io
import json
from pathlib import Path

import numpy as np

from accelerate_shared_mask_bootstrap_v3 import accumulate_ap_overall
from paired_bootstrap import build_eval
from run_fixed_universe_sensitivity import canonical_hash
from topic_c.manifest import sha256_file
from topic_c.shared_mask_pilot import EvidenceBundle, validate_bundle

ATTEMPT = "controlled_retinanet_tf32off_v1_20260906"
CONDITIONS = [(c, s) for c in ("fog", "gaussian_noise", "jpeg", "motion_blur") for s in (1, 3, 5)]
ARMS = ("default", "aligned", "fp8")
METRICS = ("delta_e_default", "delta_e_aligned", "omega")


def exact_text(path: Path, text: str):
    if path.exists():
        if path.read_bytes() != text.encode("utf-8"):
            raise ValueError(f"immutable evidence differs: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("xb") as handle:
            handle.write(text.encode("utf-8"))


def write_json(path, value):
    exact_text(path, json.dumps(value, indent=2, sort_keys=True, allow_nan=False) + "\n")


def contrasts(clean, corrupt):
    default = (corrupt["fp8"] - corrupt["default"]) - (clean["fp8"] - clean["default"])
    aligned = (corrupt["fp8"] - corrupt["aligned"]) - (clean["fp8"] - clean["aligned"])
    omega = default - aligned
    direct = (corrupt["aligned"] - corrupt["default"]) - (clean["aligned"] - clean["default"])
    if any(not np.isfinite(value).all() for value in (default, aligned, omega)):
        raise ValueError("nonfinite contrast")
    np.testing.assert_allclose(omega, direct, atol=1e-12, rtol=0)
    return dict(zip(METRICS, (default, aligned, omega)))


def joint_interval(rows):
    values = np.asarray(rows, dtype=float)
    if values.shape != (12, 2000) or not np.isfinite(values).all():
        raise ValueError("exact 12 finite cells by 2000 aligned draws required")
    return np.percentile(values.mean(axis=0), [2.5, 50, 97.5]).tolist()


def validate_build_policy(records):
    policies = []
    for r in records:
        if r.get("tf32_enabled") is not False or "--noTF32" not in r.get("command", []):
            raise ValueError("TF32-off required")
        command = [v for v in r["command"] if not v.startswith(("--onnx=", "--saveEngine="))]
        policies.append((r["workspace"], r["trtexec_sha256"], r["tensorrt_python_version"], command))
    if not policies or any(p != policies[0] for p in policies):
        raise ValueError("build policy differs between compared engines")


def bootstrap_task(job):
    """One overall-AP vector; duplicated bootstrap positions remain duplicated."""
    for path, digest in ((job["annotations"], job["annotation_sha256"]),
                         (job["prediction"], job["prediction_sha256"]),
                         (job["schedule"], job["schedule_sha256"])):
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"bootstrap source changed: {path}")
    identity = canonical_hash(job)
    out = Path(job["out"])
    marker = out.with_suffix(".json")
    if out.exists() or marker.exists():
        if not out.is_file() or not marker.is_file():
            raise ValueError(f"partial cache preserved; review required: {out}")
        metadata = json.loads(marker.read_text())
        if metadata != {"identity": identity, "npz_sha256": sha256_file(out)}:
            raise ValueError(f"cache identity/hash mismatch: {out}")
        with np.load(out, allow_pickle=False) as cache:
            if set(cache.files) != {"ap"}:
                raise ValueError("cache fields differ")
            values = cache["ap"].copy()
        if values.shape != (job["n_boot"],) or not np.isfinite(values).all():
            raise ValueError("invalid cached AP vector")
        print(f"REUSED {job['name']}", flush=True)
        return values
    from pycocotools.coco import COCO
    with np.load(job["schedule"], allow_pickle=False) as cache:
        samples = cache["samples"]
    ids = job["image_ids"]
    if (samples.shape != (job["n_boot"], len(ids)) or samples.dtype.kind not in "iu"
            or samples.min() < 0 or samples.max() >= len(ids)):
        raise ValueError("invalid schedule positions")
    evaluation = build_eval(COCO(job["annotations"]), json.loads(Path(job["prediction"]).read_text()), ids)
    if list(evaluation._paramsEval.imgIds) != ids:
        raise ValueError("evaluator changed schedule image order")
    uniform = accumulate_ap_overall(evaluation, list(range(len(ids))))
    if not np.isclose(uniform, job["point"], rtol=0, atol=1e-10):
        raise ValueError(f"uniform AP does not reconstruct metric: {job['name']}")
    values = np.empty(job["n_boot"])
    for i, draw in enumerate(samples):
        values[i] = accumulate_ap_overall(evaluation, draw.tolist())
        if (i + 1) % 100 == 0:
            print(f"BOOTSTRAP {job['name']} {i+1}/{job['n_boot']}", flush=True)
    if not np.isfinite(values).all():
        raise ValueError(f"nonfinite draws: {job['name']}")
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("xb") as handle:
        np.savez_compressed(handle, ap=values)
    write_json(marker, {"identity": identity, "npz_sha256": sha256_file(out)})
    print(f"AP_COMPLETE {job['name']}", flush=True)
    return values


def prepare(root, out):
    config_path = root / "configs" / f"{ATTEMPT}.json"
    config = json.loads(config_path.read_text())
    if config["attempt"] != ATTEMPT:
        raise ValueError("wrong controlled attempt")
    bindings = dict(config["files"])
    bindings[str(config_path)] = sha256_file(config_path)
    for path, digest in bindings.items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"frozen execution input changed: {path}")
    for name in ("run_controlled_postprocess.py", "accelerate_shared_mask_bootstrap_v3.py",
                 "paired_bootstrap.py", "run_fixed_universe_sensitivity.py"):
        p = Path(__file__).with_name(name)
        bindings[str(p)] = sha256_file(p)
    base = root / "outputs" / "controlled_build" / ATTEMPT
    engines = {}
    for arm in ("fp32", "default", "aligned"):
        p = base / f"{arm}.engine.json"
        marker = p.with_suffix(".json.complete")
        if marker.read_text().strip() != sha256_file(p):
            raise ValueError("uncommitted engine registry")
        r = json.loads(p.read_text())
        for file, digest in ((r["engine"], r["engine_sha256"]),
                             (r["build_log"], r["build_log_sha256"]),
                             (r["source_onnx"], r["source_onnx_sha256"]),
                             (r["trtexec"], r["trtexec_sha256"])):
            if sha256_file(Path(file)) != digest:
                raise ValueError(f"engine input/output hash mismatch: {file}")
            bindings[file] = digest
        bindings[str(p)] = sha256_file(p)
        bindings[str(marker)] = sha256_file(marker)
        engines[arm] = r
    validate_build_policy(list(engines.values()))
    block = config["block"]
    records = {}
    all_ids = None
    for c, s in [("clean", 0)] + CONDITIONS:
        condition = f"{c}-s{s}"
        records[condition] = {}
        for arm in ARMS + (("fp32",) if c == "clean" else ()):
            if arm == "fp8":
                d = config["fp8_bundles"][condition]["paths"]
                bundle = EvidenceBundle(*[Path(d[k]) for k in ("prediction", "input_record", "run_record", "metric")])
                engine = json.loads((root / block["baseline_fp8_engine_registry"]).read_text())
            else:
                prefix = base / f"{arm}__{condition}"
                bundle = EvidenceBundle(*[Path(str(prefix) + suffix) for suffix in
                    (".predictions.json", ".inputs.json", ".run.json", ".metric.json")])
                engine = engines[arm]
            precision = arm if arm in ("fp8", "fp32") else "int8-entropy"
            r = validate_bundle(bundle, root=root, block=block, precision=precision,
                corruption=c, severity=s,
                expected_manifest_sha256=config["fp8_bundles"][condition]["input_manifest_sha256"],
                expected_engine_sha256=engine["engine_sha256"])
            if r["input_image_ids_sha256"] != config["fp8_bundles"][condition]["input_image_ids_sha256"]:
                raise ValueError("frozen FP8 image identity differs")
            ids = sorted(r["image_ids"])
            if all_ids is not None and ids != all_ids:
                raise ValueError("clean/corrupt image universe differs")
            all_ids = ids
            records[condition][arm] = {"prediction": str(bundle.prediction),
                "prediction_sha256": r["prediction_sha256"], "point": r["stats"]["AP"]}
            for p in bundle.paths:
                bindings[str(p)] = sha256_file(p)
    if not all_ids:
        raise ValueError("empty image universe")
    manifest = {"attempt": ATTEMPT, "n_boot": 2000, "seed": 20260906,
        "image_ids": all_ids, "schedule_order": "sorted evaluator image IDs",
        "records": records, "source_sha256": dict(sorted(bindings.items())),
        "scope": "post-hoc KITTI RetinaNet; fixed engines/labels; no population, causal or runtime inference"}
    manifest["manifest_sha256"] = canonical_hash(manifest)
    write_json(out / "manifest.json", manifest)
    schedule = out / "schedule.npz"
    samples = np.random.default_rng(20260906).integers(0, len(all_ids), (2000, len(all_ids)), dtype=np.int32)
    if schedule.exists():
        with np.load(schedule, allow_pickle=False) as stored:
            if set(stored.files) != {"samples"} or not np.array_equal(stored["samples"], samples):
                raise ValueError("schedule mismatch")
    else:
        with schedule.open("xb") as handle:
            np.savez_compressed(handle, samples=samples)
    jobs = []
    annotation = root / block["annotations"]
    for condition, arms in records.items():
        for arm in ARMS:
            name = f"{arm}__{condition}"
            jobs.append({"name": name, **arms[arm], "annotations": str(annotation),
                "annotation_sha256": sha256_file(annotation), "image_ids": all_ids,
                "schedule": str(schedule), "schedule_sha256": sha256_file(schedule),
                "n_boot": 2000, "manifest_sha256": manifest["manifest_sha256"],
                "out": str(out / "draws" / f"{name}.npz")})
    return manifest, jobs


def finish(out, manifest, jobs, draws):
    rows, joint = [], {key: [] for key in METRICS}
    clean_points = {arm: manifest["records"]["clean-s0"][arm]["point"] for arm in ARMS}
    clean_draws = {arm: draws[f"{arm}__clean-s0"] for arm in ARMS}
    for c, s in CONDITIONS:
        condition = f"{c}-s{s}"
        cp = {arm: manifest["records"][condition][arm]["point"] for arm in ARMS}
        cd = {arm: draws[f"{arm}__{condition}"] for arm in ARMS}
        point = contrasts(clean_points, cp)
        vectors = contrasts(clean_draws, cd)
        row = {"corruption": c, "severity": s}
        for arm in ARMS:
            row[f"{arm}_clean_ap"] = 100 * clean_points[arm]
            row[f"{arm}_corrupt_ap"] = 100 * cp[arm]
            row[f"{arm}_signed_loss"] = 100 * (clean_points[arm] - cp[arm])
        for key in METRICS:
            row[key] = 100 * float(point[key])
            interval = np.percentile(100 * vectors[key], [2.5, 50, 97.5]).tolist()
            row[key + "_interval"] = interval
            joint[key].append(100 * vectors[key])
        rows.append(row)
    summary = {"scope": manifest["scope"], "units": "AP points",
        "n_boot": 2000, "cells": rows,
        "clean_fp32_diagnostic_ap": 100 * manifest["records"]["clean-s0"]["fp32"]["point"],
        "macro": {key: {"point": float(np.mean([r[key] for r in rows])),
                        "percentile_interval": joint_interval(joint[key])} for key in METRICS},
        "manifest_sha256": manifest["manifest_sha256"],
        "limitations": ["One post-hoc block and one build per policy; no pure causal localization.",
            "Ordinary image bootstrap is conditional on frozen treatments and image universe.",
            "Strict TRT FP32 diagnostic alone does not establish source/ORT parity.",
            "No deployment speed claim; manuscript integration requires scientific review."],
        "draw_caches": {j["out"]: sha256_file(Path(j["out"])) for j in jobs}}
    for path, digest in manifest["source_sha256"].items():
        if sha256_file(Path(path)) != digest:
            raise ValueError(f"source changed during bootstrap: {path}")
    write_json(out / "summary.json", summary)
    stream = io.StringIO()
    writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
    exact_text(out / "cells.csv", stream.getvalue())
    table = [r"\begin{tabular}{lrrr}", r"\hline", r"Endpoint & Point & Lower & Upper \\", r"\hline"]
    for key, label in zip(METRICS, (r"$\Delta E_{\mathrm{default}}$", r"$\Delta E_{\mathrm{aligned}}$", r"$\Omega$")):
        value = summary["macro"][key]
        lo, _, hi = value["percentile_interval"]
        table.append(f"{label} & {value['point']:.3f} & {lo:.3f} & {hi:.3f}" + r" \\")
    table += [r"\hline", r"\end{tabular}"]
    exact_text(out / "macro_table.tex", "\n".join(table) + "\n")
    report = ["# Controlled RetinaNet analysis", "", summary["scope"], "",
              "Units: AP points; percentile bounds: 2.5% / 97.5%.", ""]
    for key, value in summary["macro"].items():
        lo, _, hi = value["percentile_interval"]
        report.append(f"- {key}: {value['point']:.4f} [{lo:.4f}, {hi:.4f}].")
    report += ["", "## Boundaries", ""] + ["- " + s for s in summary["limitations"]]
    exact_text(out / "report.md", "\n".join(report) + "\n")
    completed = {name: sha256_file(out / name) for name in
                 ("manifest.json", "schedule.npz", "summary.json", "cells.csv", "macro_table.tex", "report.md")}
    write_json(out / "complete.json", completed)
    print("PIPELINE_COMPLETE " + json.dumps(summary["macro"]), flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=4)
    args = parser.parse_args()
    if not 1 <= args.workers <= 4:
        raise ValueError("workers must be 1..4")
    root = args.project_root.resolve()
    out = root / "outputs" / "controlled_build" / ATTEMPT / "analysis_v1"
    out.mkdir(parents=True, exist_ok=True)
    with (out / "driver.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        manifest, jobs = prepare(root, out)
        print(f"PREFLIGHT_VALIDATED {len(jobs)} AP tasks; 12 cells; shared 2000-draw schedule", flush=True)
        results = {}
        executor = ProcessPoolExecutor(max_workers=args.workers)
        try:
            futures = {executor.submit(bootstrap_task, j): j for j in jobs}
            for future in as_completed(futures):
                job = futures[future]
                results[job["name"]] = future.result()
                print(f"TASK_DONE {len(results)}/{len(jobs)} {job['name']}", flush=True)
        except BaseException:
            executor.shutdown(wait=True, cancel_futures=True)
            raise
        else:
            executor.shutdown(wait=True)
        finish(out, manifest, jobs, results)


if __name__ == "__main__":
    try:
        main()
    except BaseException as error:
        print("PIPELINE_FAILED " + repr(error), flush=True)
        raise
