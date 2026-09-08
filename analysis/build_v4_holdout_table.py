#!/usr/bin/env python3
"""Regenerate original-source holdout synthesis from retained metrics/draws only.

No inference or bootstrap is run. --ssh fetches small historical bootstrap
artifacts read-only; component AP draws absent from those artifacts stay missing.
"""
from __future__ import annotations

import argparse
from concurrent.futures import ProcessPoolExecutor, as_completed
import copy
import csv
import fcntl
import hashlib
import io
import json
import os
from pathlib import Path
import re
import shlex
import subprocess
import sys
import tarfile
import time

if __name__ == "__main__":
    for _variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS", "NUMEXPR_NUM_THREADS", "VECLIB_MAXIMUM_THREADS"):
        os.environ[_variable] = "1"

import numpy as np

DATASETS = {"voc": 5823, "kitti": 1197}
MODELS = ("yolo11n", "yolo11m", "yolo11x")
CONDITIONS = [(c, s) for c in ("gaussian_noise", "motion_blur", "fog", "jpeg") for s in (1, 3, 5)]
REMOTE_ROOT = "/home/thuan/topic_c_ivc"


def require(value, message):
    if not value:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical(document, field):
    return hashlib.sha256(json.dumps({k: v for k, v in document.items() if k != field},
                                    sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def draw_matrix(values, rows):
    result = np.asarray(values, dtype=np.float64)
    require(result.shape == (rows, 2000) and np.isfinite(result).all(), "aligned finite 2000-draw matrix required")
    return result


def interval(values):
    return np.percentile(np.asarray(values) * 100, [2.5, 50, 97.5]).tolist()


def summarize_block(cells, delta_e_draws, corrupted_gap_draws=None):
    require(len(cells) == 12 and {(c["corruption"], c["severity"]) for c in cells} == set(CONDITIONS),
            "complete unique twelve-condition block required")
    require(len({(c["dataset"], c["model"]) for c in cells}) == 1, "mixed block identity")
    clean = [(c["int8_clean_ap"], c["fp8_clean_ap"]) for c in cells]
    require(all(pair == clean[0] for pair in clean), "clean AP changes across cells")
    for cell in cells:
        aps = [cell[k] for k in ("int8_clean_ap", "fp8_clean_ap", "int8_corrupted_ap", "fp8_corrupted_ap")]
        require(all(np.isfinite(v) and 0 <= v <= 1 for v in aps), "invalid AP")
        effect = (aps[3] - aps[2]) - (aps[1] - aps[0])
        require(np.isclose(effect, cell["delta_e"], atol=1e-12, rtol=0), "four-arm identity mismatch")
    i0, f0 = clean[0]
    ic, fc = (float(np.mean([c[key] for c in cells])) for key in ("int8_corrupted_ap", "fp8_corrupted_ap"))
    draws = draw_matrix(delta_e_draws, 12).mean(axis=0)
    gap_interval = None if corrupted_gap_draws is None else interval(draw_matrix(corrupted_gap_draws, 12).mean(axis=0))
    return {"dataset": cells[0]["dataset"], "model": cells[0]["model"], "conditions": 12,
            "clean_control": "original_source", "int8_clean_ap_points": 100 * i0,
            "fp8_clean_ap_points": 100 * f0, "int8_corrupted_mean_ap_points": 100 * ic,
            "fp8_corrupted_mean_ap_points": 100 * fc, "clean_gap_ap_points": 100 * (f0 - i0),
            "corrupted_gap_ap_points": 100 * (fc - ic), "delta_e_ap_points": 100 * ((fc - ic) - (f0 - i0)),
            "delta_e_percentile95_ap_points": interval(draws),
            "delta_e_interval_status": "verified_paired_within_draw_mean",
            "corrupted_gap_percentile95_ap_points": gap_interval,
            "corrupted_gap_interval_status": "missing_component_ap_draws" if gap_interval is None else "verified_paired_within_draw_mean"}


def summarize_macro(blocks, draws):
    expected = {(d, m) for d in DATASETS for m in MODELS}
    require(len(blocks) == 6 and {(b["dataset"], b["model"]) for b in blocks} == expected, "exact six unique primary blocks required")
    return {"blocks": 6, "weighting": "equal block; twelve equally weighted corruption cells per block",
            "delta_e_ap_points": float(np.mean([b["delta_e_ap_points"] for b in blocks])),
            "delta_e_percentile95_ap_points": interval(draw_matrix(draws, 6).mean(axis=0)),
            "corrupted_gap_ap_points": float(np.mean([b["corrupted_gap_ap_points"] for b in blocks])),
            "corrupted_gap_percentile95_ap_points": None,
            "corrupted_gap_interval_status": "missing_component_ap_draws"}


def validate_pairing(documents, seed, n_images):
    identities = set()
    require(bool(documents), "no bootstrap records")
    for document in documents:
        require(document.get("seed") == seed and document.get("n_boot") == 2000 and document.get("n_images") == n_images,
                "bootstrap schedule metadata mismatch")
        hashes = document.get("input_hashes", {})
        require(bool(hashes), "missing paired input identities")
        for value in hashes.values():
            identity = value.get("image_ids_sha256")
            require(isinstance(identity, str) and re.fullmatch(r"[0-9a-f]{64}", identity), "invalid image identity hash")
            identities.add(identity)
    require(len(identities) == 1, "paired ordered image identities differ")


def historical_schedule(ids, seed, n_boot=2000):
    require(bool(ids) and ids == sorted(set(ids)), "historical ordered IDs must be unique")
    rng = np.random.default_rng(seed)
    # Exact loop/sampler in the hash-verified historical paired_bootstrap.py.
    return np.asarray([rng.choice(len(ids), size=len(ids), replace=True) for _ in range(n_boot)], dtype=np.int64)


def recover_intervals(result, delta_e_draws, clean_ap_draws):
    expected = {d + "_" + m for d in DATASETS for m in MODELS}
    require(len(result["blocks"]) == 6 and set(clean_ap_draws) == expected, "six complete clean block vectors required")
    delta = draw_matrix(delta_e_draws, 6)
    recovered = copy.deepcopy(result)
    gaps = []
    for index, block in enumerate(recovered["blocks"]):
        key = block["dataset"] + "_" + block["model"]
        records = clean_ap_draws[key]
        require(set(records) == {"int8-entropy", "fp8"}, "both clean precision vectors required")
        vectors = [np.asarray(records[p], dtype=float) for p in ("int8-entropy", "fp8")]
        require(all(v.shape == (2000,) and np.isfinite(v).all() and (v >= 0).all() and (v <= 1).all() for v in vectors),
                "invalid clean AP vector")
        clean_gap = vectors[1] - vectors[0]
        corrupted_gap = delta[index] + clean_gap
        gaps.append(corrupted_gap)
        block["clean_gap_percentile95_ap_points"] = interval(clean_gap)
        block["corrupted_gap_percentile95_ap_points"] = interval(corrupted_gap)
        block["corrupted_gap_interval_status"] = "recovered_paired_historical_schedule"
    recovered["primary_macro"]["corrupted_gap_percentile95_ap_points"] = interval(np.mean(gaps, axis=0))
    recovered["primary_macro"]["corrupted_gap_interval_status"] = "recovered_paired_historical_schedule"
    recovered["corrupted_gap_interval_status"] = "recovered_paired_historical_schedule"
    recovered["scope"] = "six final-holdout blocks with original-source clean; only twelve original-clean AP vectors are recomputed on the historical schedule; no new inference or corrupted AP bootstrap"
    recovered["limits"] = [line for line in recovered["limits"] if "no new resampling is performed" not in line]
    recovered["limits"].append("Corrupted-gap intervals recovered from retained DeltaE draws plus newly recomputed original-clean gap draws under the recorded historical seed/sampler/order; no new corrupted AP bootstrap or inference.")
    recovered["limits"].append("Historical NumPy and evaluator versions were not recorded. Recovery pins current evaluator files/versions, reproduces each historical clean AP point, and reconstructs the recorded seed/order with the hash-verified historical sampler; this is not historical runtime attestation.")
    return recovered, np.asarray(gaps)


def completion_ready(marker):
    marker = Path(marker)
    if not marker.is_file():
        return False
    hashes = json.loads(marker.read_text())
    require(bool(hashes), "empty upstream completion marker")
    for relative, expected in hashes.items():
        path = marker.parent / relative
        require(not Path(relative).is_absolute() and ".." not in Path(relative).parts,
                "unsafe upstream completion path")
        require(path.is_file() and sha256_file(path) == expected, "upstream completion hash mismatch")
    return True


def validate_clean_binding(documents, expected):
    require(len(documents) == 12, "twelve linked corruption records required")
    require(all(document.get("input_hashes", {}).get("quant_clean") == expected for document in documents),
            "historical clean prediction/input binding changed across corruption cells")


def retain(path, payload):
    path = Path(path)
    if path.exists():
        require(path.read_bytes() == payload, f"refusing to overwrite differing retained evidence: {path}")
    else:
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_bytes(payload)


def fetch_sources(host, remote_root, destination):
    selected = [f"outputs/{folder}/{dataset}_confirmatory_final_117_v1{suffix}"
                for dataset in DATASETS for folder, suffix in (("bootstrap", ""), ("reports", "_bootstrap_complete.json"))]
    command = "tar -czf - -C " + shlex.quote(remote_root) + " " + " ".join(map(shlex.quote, selected))
    payload = subprocess.run(["ssh", "-o", "BatchMode=yes", host, command], stdout=subprocess.PIPE, check=True).stdout
    with tarfile.open(fileobj=io.BytesIO(payload), mode="r:gz") as archive:
        for member in archive.getmembers():
            if member.isdir():
                continue
            path = Path(member.name)
            require(member.isfile() and not path.is_absolute() and ".." not in path.parts
                    and any(member.name == prefix or member.name.startswith(prefix + "/") for prefix in selected),
                    "unsafe or unexpected archive member")
            with archive.extractfile(member) as stream:
                retain(destination / path, stream.read())


def build(root, source_root, ledger_path=None):
    ledger_path = Path(ledger_path) if ledger_path is not None else root / "paper/confirmatory_evidence/untouched_holdout_analysis.json"
    ledger = json.loads(ledger_path.read_text())
    require(ledger.get("analysis_sha256") == canonical(ledger, "analysis_sha256"), "ledger canonical hash mismatch")
    expected = {(d, m, c, s) for d in DATASETS for m in MODELS for c, s in CONDITIONS}
    cells = ledger["cells"]
    require(len(cells) == 72 and {(c["dataset"], c["model"], c["corruption"], c["severity"]) for c in cells} == expected,
            "complete unique 72-cell ledger required")
    sources = [{"path": str(ledger_path), "sha256": sha256_file(ledger_path), "hash_kind": "file_bytes"}]
    cell_index = {(c["dataset"], c["model"], c["corruption"], c["severity"]): c for c in cells}
    report_index = {e["dataset"]: e for e in ledger["evidence"]}
    blocks, block_draws, draw_labels, archive_fields = [], [], [], set()

    def read_bound(relative, expected_hash):
        path = source_root / relative
        require(path.is_file() and sha256_file(path) == expected_hash, f"retained source hash mismatch or missing: {relative}")
        sources.append({"path": str(path), "historical_path": REMOTE_ROOT + "/" + str(relative),
                        "sha256": expected_hash, "hash_kind": "file_bytes"})
        return path

    for dataset, n_images in DATASETS.items():
        relative = Path("outputs/reports") / f"{dataset}_confirmatory_final_117_v1_bootstrap_complete.json"
        report = json.loads(read_bound(relative, report_index[dataset]["sha256"]).read_text())
        require(report.get("joint_dataset_draw_sequence") is True and report.get("n_boot") == 2000,
                "historical shared-draw schedule not verified")
        seed = report["shared_seed"]
        dataset_documents = []
        for model in MODELS:
            block_cells, direct_draws = [], []
            for corruption, severity in CONDITIONS:
                block_cells.append(cell_index[(dataset, model, corruption, severity)])
                effects, documents = {}, {}
                for precision in ("int8-entropy", "fp8"):
                    name = f"outputs/bootstrap/{dataset}_confirmatory_final_117_v1/{model}__{precision}__{corruption}-s{severity}"
                    path = read_bound(name + ".json", report["artifacts_sha256"][name + ".json"])
                    document = json.loads(path.read_text())
                    documents[precision] = document
                    require(document.get("quant_label") == precision, "bootstrap precision mismatch")
                    cached = read_bound(name + ".draws.npz", report["draw_caches_sha256"][name + ".draws.npz"])
                    require(document["draw_cache"]["sha256"] == sha256_file(cached), "document-cache hash mismatch")
                    require(Path(document["draw_cache"]["path"]).name == cached.name, "cache path mismatch")
                    with np.load(cached, allow_pickle=False) as cache:
                        archive_fields.update(cache.files)
                        require(int(cache["n_boot"]) == 2000 and int(cache["seed"]) == seed,
                                "draw cache schedule mismatch")
                        values = np.asarray(cache["excess"], dtype=float)
                        require(values.shape == (2000, 4) and np.isfinite(values[:, 0]).all(), "invalid excess draw array")
                        effects[precision] = values[:, 0]
                    dataset_documents.append(document)
                validate_pairing(list(documents.values()), seed, n_images)
                point = documents["int8-entropy"]["point"]["excess"]["all"] - documents["fp8"]["point"]["excess"]["all"]
                require(np.isclose(point, block_cells[-1]["delta_e"], atol=1e-12, rtol=0), "ledger-bootstrap point mismatch")
                # Both precision records must bind the same FP32 references.
                for key in ("fp32_clean", "fp32_corrupt"):
                    require(documents["int8-entropy"]["input_hashes"][key] == documents["fp8"]["input_hashes"][key],
                            "FP32 reference cancellation is not bound")
                direct_draws.append(effects["int8-entropy"] - effects["fp8"])
            blocks.append(summarize_block(block_cells, np.asarray(direct_draws)))
            block_draws.append(np.mean(direct_draws, axis=0))
            draw_labels.append(dataset + "_" + model)
        validate_pairing(dataset_documents, seed, n_images)
    macro = summarize_macro(blocks, np.asarray(block_draws))
    published = ledger["overall_balanced_equal_cell"]
    require(np.isclose(macro["delta_e_ap_points"], 100 * published["delta_e_point"], atol=1e-10, rtol=0),
            "unrounded historical macro mismatch")
    require(np.allclose(macro["delta_e_percentile95_ap_points"], np.asarray(published["delta_e_percentile95"]) * 100,
                        atol=1e-10, rtol=0), "historical macro interval mismatch")
    result = {"schema_version": 1, "scope": "six final-holdout blocks; original-source clean; no new AP/bootstrap execution",
              "units": "AP points", "n_boot": 2000, "blocks": blocks, "primary_macro": macro,
              "historical_unrounded_macro_verified": True, "historical_display_macro": f"{macro['delta_e_ap_points']:.2f}",
              "retained_draw_cache_fields": sorted(archive_fields),
              "corrupted_gap_interval_status": "missing_component_ap_draws",
              "limits": ["The historical clean control is original-source, not terminal JPEG-95.",
                         "Retained bootstrap caches contain excess/psi only, not component clean/corrupt AP draws.",
                         "A corrupted-gap CI cannot be reconstructed by adding the clean point gap or interval endpoints to DeltaE.",
                         "Historical joint sampling is bound by dataset seed, ordered image identities and completion records; no new resampling is performed.",
                         "Intervals are descriptive paired percentile intervals conditional on this finite corruption grid and retained models."],
              "implementation_sha256": sha256_file(__file__), "sources": sources}
    return result, np.asarray(block_draws), draw_labels


def write_outputs(out, result, draws, labels):
    out.mkdir(parents=True, exist_ok=True)
    (out / "holdout_synthesis.json").write_text(json.dumps(result, indent=2, allow_nan=False) + "\n")
    rows = []
    for block in result["blocks"]:
        row = {k: v for k, v in block.items() if not isinstance(v, list)}
        for kind in ("delta_e", "corrupted_gap"):
            values = block[kind + "_percentile95_ap_points"]
            row.pop(kind + "_percentile95_ap_points", None)
            for position, label in enumerate(("low95", "median", "high95")):
                row[kind + "_" + label] = None if values is None else values[position]
        rows.append(row)
    with (out / "holdout_synthesis.csv").open("w", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    np.savez_compressed(out / "paired_delta_e_draws.npz", block_delta_e_native=draws,
                        macro_delta_e_native=draws.mean(axis=0), block_labels=labels)
    lines = ["% Generated from unrounded original-source holdout metrics; units AP points.",
             r"\begin{tabular}{llrrrrrrrrl}", r"\toprule",
             r"Dataset & Model & $I_0$ & $F_0$ & $\bar I_c$ & $\bar F_c$ & $G_0$ & $\bar G_c$ & $\Delta E$ & $G_c$ CI & $\Delta E$ 95\% CI\\",
             r"\midrule"]
    for block in result["blocks"]:
        numbers = [block[k] for k in ("int8_clean_ap_points", "fp8_clean_ap_points", "int8_corrupted_mean_ap_points",
                   "fp8_corrupted_mean_ap_points", "clean_gap_ap_points", "corrupted_gap_ap_points", "delta_e_ap_points")]
        ci = block["delta_e_percentile95_ap_points"]
        lines.append(block["dataset"].upper() + " & " + block["model"] + " & " + " & ".join(f"{v:.2f}" for v in numbers)
                     + " & " + ("---" if block["corrupted_gap_percentile95_ap_points"] is None else
                                  f"[{block['corrupted_gap_percentile95_ap_points'][0]:.2f}, {block['corrupted_gap_percentile95_ap_points'][2]:.2f}]")
                     + rf" & [{ci[0]:.2f}, {ci[2]:.2f}]\\")
    macro = result["primary_macro"]
    ci = macro["delta_e_percentile95_ap_points"]
    macro_gap = macro.get("corrupted_gap_percentile95_ap_points")
    macro_gap_text = "---" if macro_gap is None else f"[{macro_gap[0]:.2f}, {macro_gap[2]:.2f}]"
    lines += [r"\midrule", rf"\multicolumn{{8}}{{l}}{{Equal-block macro}} & {macro['delta_e_ap_points']:.2f} & {macro_gap_text} & [{ci[0]:.2f}, {ci[2]:.2f}]\\",
              r"\bottomrule", r"\end{tabular}",
              "% ---: corrupted-gap CI unavailable because historical component AP draws were not retained."]
    (out / "holdout_synthesis.tex").write_text("\n".join(lines) + "\n")
    (out / "README.md").write_text("# Six-block original-source holdout synthesis\n\n"
        + f"Unrounded equal-block DeltaE: {macro['delta_e_ap_points']:.15g} AP; 95% CI [{ci[0]:.15g}, {ci[2]:.15g}].\n\n"
        + "\n".join("- " + line for line in result["limits"]) + "\n")
    files = ["holdout_synthesis.json", "holdout_synthesis.csv", "holdout_synthesis.tex", "paired_delta_e_draws.npz", "README.md"]
    completion = {"files_sha256": {name: sha256_file(out / name) for name in files},
                  "source_file_count": len(result["sources"]), "blocks": 6, "corruption_cells": 72,
                  "delta_e_intervals_verified": 6,
                  "corrupted_gap_intervals_missing": sum(b["corrupted_gap_percentile95_ap_points"] is None for b in result["blocks"]),
                  "historical_display_macro": result["historical_display_macro"], "implementation_sha256": sha256_file(__file__)}
    (out / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    return completion


def prepare_recovery(root, source_root, out, result, upstream_marker):
    """Freeze twelve clean-only jobs after verifying the historical sampler."""
    sys.path.insert(0, str(root / "src"))
    from importlib.metadata import version
    import pycocotools.coco
    import pycocotools.cocoeval
    import pycocotools._mask

    source_files = {}

    def bind(path, expected=None):
        path = Path(path)
        observed = sha256_file(path)
        require(expected is None or observed == expected, f"recovery source mismatch: {path}")
        source_files[str(path)] = observed
        return observed

    config_path = root / "configs/confirmatory_bootstrap_v1.json"
    config = json.loads(config_path.read_text())
    require(config.get("config_sha256") == canonical(config, "config_sha256"), "historical bootstrap config hash mismatch")
    bind(config_path)
    frozen_sources = {item["path"]: item["sha256"] for item in config["source_manifest"]}
    require(frozen_sources.get("src/paired_bootstrap.py") == "30d96c5e78aeb8054498c65f22c00828a5cb4fefc13fb81ca19b69a0c32653ef",
            "unrecognized historical sampler implementation; refuse schedule substitution")
    for name in ("src/paired_bootstrap.py", "src/run_confirmatory_bootstrap.py"):
        bind(root / name, frozen_sources[name])
    for path in (root / "src/accelerate_shared_mask_bootstrap_v3.py", root / "src/run_controlled_postprocess.py",
                 root / "src/run_fixed_universe_sensitivity.py", Path(__file__),
                 Path(pycocotools.coco.__file__), Path(pycocotools.cocoeval.__file__), Path(pycocotools._mask.__file__), upstream_marker):
        bind(path)
    context = {"sources_sha256": dict(source_files), "numpy": np.__version__, "python": sys.version,
               "pycocotools": version("pycocotools"), "sampler": "one numpy.default_rng(seed).choice(n,size=n,replace=True) per draw",
               "historical_numpy_version": "not_recorded", "historical_evaluator_version": "not_recorded",
               "scope": "recompute only original clean AP draws, require historical point reconstruction; combine with retained paired DeltaE draws"}
    context_hash = hashlib.sha256(json.dumps(context, sort_keys=True, separators=(",", ":")).encode()).hexdigest()
    jobs, schedules = [], {}
    annotations = {item["dataset"]: item for item in config["datasets"]}
    for block in result["blocks"]:
        dataset, model = block["dataset"], block["model"]
        report = json.loads((source_root / f"outputs/reports/{dataset}_confirmatory_final_117_v1_bootstrap_complete.json").read_text())
        seed = int.from_bytes(hashlib.sha256(f"{config['seed_namespace']}|{dataset}".encode()).digest()[:8], "big") % (2 ** 32)
        require(seed == report["shared_seed"], "historical namespace/recorded seed mismatch")
        annotation_path = root / annotations[dataset]["annotations"]
        annotation_hash = bind(annotation_path, annotations[dataset]["annotations_sha256"])
        for precision in ("int8-entropy", "fp8"):
            prefix = "int8-entropy" if precision == "int8-entropy" else "fp8-entropy"
            namespace = f"{dataset}_confirmatory_final_117_v1"
            paths = list((root / "manifests/runs" / namespace).glob(f"{dataset}_test__{model}__{prefix}__clean-s0__*.json"))
            require(len(paths) == 1, "exact one historical original-clean run required")
            run_path = paths[0]
            run = json.loads(run_path.read_text())
            cid = run_path.stem
            prediction = root / "outputs/predictions" / namespace / (cid + ".json")
            input_path = root / "outputs/inputs" / namespace / (cid + ".json")
            metric_path = root / "outputs/metrics" / namespace / (cid + ".json")
            input_record = json.loads(input_path.read_text())
            metric = json.loads(metric_path.read_text())
            metric_report_path = root / "outputs/reports" / (namespace + "_evaluation_complete.json")
            metric_report = json.loads(metric_report_path.read_text())
            bind(metric_report_path)
            bind(metric_path, metric_report["metric_sha256"][cid])
            bind(run_path, metric["run_record_sha256"])
            prediction_hash = bind(prediction, run["prediction_sha256"])
            require(metric["prediction_sha256"] == prediction_hash, "clean metric prediction binding mismatch")
            require((run["dataset"], run["model"], run["precision"], run["corruption"], run["severity"]) ==
                    (dataset, model, precision, "clean", 0), "clean recovery treatment mismatch")
            ids = input_record["image_ids"]
            ids_hash = hashlib.sha256(json.dumps(ids, separators=(",", ":")).encode()).hexdigest()
            require(len(ids) == DATASETS[dataset] and ids_hash == input_record["image_ids_sha256"] == run["input_image_ids_sha256"],
                    "historical clean input identity mismatch")
            binding = {"prediction_sha256": prediction_hash, "input_record_sha256": bind(input_path),
                       "input_manifest_sha256": input_record["input_manifest_sha256"], "image_ids_sha256": ids_hash}
            docs = [json.loads((source_root / f"outputs/bootstrap/{namespace}/{model}__{precision}__{corruption}-s{severity}.json").read_text())
                    for corruption, severity in CONDITIONS]
            validate_pairing(docs, seed, DATASETS[dataset])
            validate_clean_binding(docs, binding)
            point = block[("int8" if precision == "int8-entropy" else "fp8") + "_clean_ap_points"] / 100
            require(np.isclose(point, metric["stats"]["AP"], atol=1e-12, rtol=0), "historical clean point mismatch")
            schedule_path = out / "schedules" / (dataset + "_historical.npz")
            if dataset not in schedules:
                samples = historical_schedule(ids, seed)
                schedule_path.parent.mkdir(parents=True, exist_ok=True)
                if schedule_path.exists():
                    with np.load(schedule_path, allow_pickle=False) as stored:
                        require(np.array_equal(stored["samples"], samples) and np.array_equal(stored["image_ids"], ids), "historical schedule cache mismatch")
                else:
                    with schedule_path.open("xb") as stream:
                        np.savez_compressed(stream, samples=samples, image_ids=ids, seed=seed)
                schedules[dataset] = {"seed": seed, "image_ids_sha256": ids_hash, "schedule_sha256": bind(schedule_path),
                                      "schedule_path": str(schedule_path), "status": "reconstructed_from_hash_verified_historical_sampler_seed_and_order"}
            require(schedules[dataset]["image_ids_sha256"] == ids_hash, "dataset schedule order changes between models/precisions")
            name = dataset + "_" + model + "__" + precision
            jobs.append({"name": name, "annotations": str(annotation_path), "annotation_sha256": annotation_hash,
                         "prediction": str(prediction), "prediction_sha256": prediction_hash,
                         "schedule": str(schedule_path), "schedule_sha256": schedules[dataset]["schedule_sha256"],
                         "image_ids": ids, "n_boot": 2000, "point": point, "out": str(out / "clean_ap" / (name + ".npz")),
                         "recovery_context_sha256": context_hash})
    registry = {"schema_version": 1, "context": context, "source_files_sha256": source_files,
                "schedules": schedules, "jobs": jobs, "expected_jobs": 12,
                "base_synthesis_sources": result["sources"]}
    retain(out / "recovery_registry.json", (json.dumps(registry, indent=2, sort_keys=True) + "\n").encode())
    return registry


def execute_recovery(root, source_root, out, result, draws, labels, workers, upstream_marker):
    require(1 <= workers <= 8, "recovery workers must be 1..8")
    for variable in ("OMP_NUM_THREADS", "OPENBLAS_NUM_THREADS", "MKL_NUM_THREADS"):
        require(os.environ.get(variable) == "1", "recovery requires single-threaded BLAS/OpenMP before NumPy import")
    while not completion_ready(upstream_marker):
        print(f"WAIT clean-control completion before historical clean-only bootstrap: {upstream_marker}", flush=True)
        time.sleep(30)
    out.mkdir(parents=True, exist_ok=True)
    with (out / "recovery.lock").open("a") as lock:
        fcntl.flock(lock, fcntl.LOCK_EX | fcntl.LOCK_NB)
        registry = prepare_recovery(root, source_root, out, result, upstream_marker)
        sys.path.insert(0, str(root / "src"))
        from run_controlled_postprocess import bootstrap_task
        print("RUN historical original-clean bootstrap 12 arms; no corrupted bootstrap or inference", flush=True)
        clean_draws = {}
        with ProcessPoolExecutor(max_workers=workers) as pool:
            futures = {pool.submit(bootstrap_task, job): job for job in registry["jobs"]}
            for future in as_completed(futures):
                job = futures[future]
                block, precision = job["name"].split("__")
                clean_draws.setdefault(block, {})[precision] = future.result()
                print(f"DONE {job['name']}", flush=True)
        recovered, gap_draws = recover_intervals(result, draws, clean_draws)
        recovered["recovery_registry_sha256"] = sha256_file(out / "recovery_registry.json")
        write_outputs(out, recovered, draws, labels)
        np.savez_compressed(out / "paired_corrupted_gap_draws.npz", block_corrupted_gap_native=gap_draws,
                            macro_corrupted_gap_native=gap_draws.mean(axis=0), block_labels=labels)
        files = [out / "recovery_registry.json", out / "completion.json", out / "paired_corrupted_gap_draws.npz"]
        files += sorted((out / "clean_ap").glob("*")) + sorted((out / "schedules").glob("*.npz"))
        marker = {"schema_version": 1, "complete": True, "clean_bootstrap_arms": 12,
                  "corrupted_gap_intervals_recovered": 6, "macro_interval_recovered": True,
                  "files_sha256": {str(path.relative_to(out)): sha256_file(path) for path in files}}
        retain(out / "recovery.complete.json", (json.dumps(marker, indent=2, sort_keys=True) + "\n").encode())
        print("COMPLETE historical-schedule corrupted-gap intervals: 6 blocks + macro", flush=True)


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--out", type=Path, default=Path("outputs/analysis/cviu_v4/holdout_synthesis"))
    parser.add_argument("--source-root", type=Path)
    parser.add_argument("--ledger", type=Path, help="explicit immutable ledger when remote paper tree is absent")
    parser.add_argument("--ssh")
    parser.add_argument("--remote-root", default=REMOTE_ROOT)
    parser.add_argument("--recover-gap-intervals", action="store_true")
    parser.add_argument("--workers", type=int, default=8)
    parser.add_argument("--wait-for-clean-control-completion", type=Path)
    args = parser.parse_args()
    source_root = args.source_root or args.out / "sources"
    if args.ssh:
        fetch_sources(args.ssh, args.remote_root, source_root)
    result, draws, labels = build(args.root.resolve(), source_root.resolve(), args.ledger)
    if not args.recover_gap_intervals:
        print(json.dumps(write_outputs(args.out, result, draws, labels), indent=2))
    if args.recover_gap_intervals:
        marker = args.wait_for_clean_control_completion or args.root / "outputs/cviu_v4_clean_control_v1/complete.json"
        execute_recovery(args.root.resolve(), source_root.resolve(), args.out.resolve() / "recovered", result, draws, labels,
                         args.workers, marker.resolve())


if __name__ == "__main__":
    main()
