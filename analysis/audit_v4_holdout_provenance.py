"""Read-only retained holdout lineage audit; never repairs historical records.

Run locally with --ssh thuan@100.111.139.103 to execute this same stdlib-only
source over stdin remotely (no remote files written). Only local reports are
created. Verified means recorded artifact consistency, not execution attestation.
"""
from __future__ import annotations

import argparse
from collections import Counter
from datetime import datetime, timezone
import hashlib
import json
from pathlib import Path
import re
import shlex
import subprocess

ATTEMPT = "voc_kitti_confirmatory_v1"
DATASETS = {"voc": 5823, "kitti": 1197}
MODELS = ("yolo11n", "yolo11m", "yolo11x")
PRECISIONS = ("int8-entropy", "fp8")
CONDITIONS = [("clean", 0)] + [(c, s) for c in ("gaussian_noise", "motion_blur", "fog", "jpeg") for s in (1, 3, 5)]


def classify_lineage(links):
    statuses = list(links.values())
    if not statuses or all(s == "missing" for s in statuses):
        return "missing"
    return "verified" if all(s == "verified" for s in statuses) else "partial"


def sha256_file(path):
    value = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            value.update(chunk)
    return value.hexdigest()


def json_digest(document):
    return hashlib.sha256(json.dumps(document, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def same(actual, expected):
    def display(value):
        # Full ordered universes are retained once under dataset_inputs. Keep
        # repeated link checks compact without dropping order sensitivity.
        if isinstance(value, list) and len(value) > 16:
            return {"length": len(value), "ordered_canonical_sha256": json_digest(value)}
        return value
    return {"status": "verified" if actual is not None and expected is not None and actual == expected else "partial",
            "actual": display(actual), "expected": display(expected)}


def clean_manifest_semantics(document):
    records = document.get("records", [])
    original = sum(bool(record.get("source_relpath")) and bool(record.get("source_sha256"))
                   and record.get("source_relpath") == record.get("output_relpath")
                   and record.get("source_sha256") == record.get("sha256")
                   and record.get("generator") == "clean_reference"
                   and record.get("corruption") == "clean" and record.get("severity") == 0 for record in records)
    return {"records": len(records), "original_source_identity_records": original,
            "control": "missing" if not records else "original_source" if original == len(records) else "not_verified_as_original_source"}


class Auditor:
    def __init__(self, root):
        self.root = Path(root).resolve()
        self.sources = {}
        self.documents = {}

    def path(self, path):
        path = Path(path)
        return path if path.is_absolute() else self.root / path

    def source(self, path):
        path = self.path(path)
        key = str(path)
        if key not in self.sources:
            self.sources[key] = {"path": key, "sha256": sha256_file(path) if path.is_file() else None,
                                 "bytes": path.stat().st_size if path.is_file() else None, "hash_kind": "file_bytes"}
        return self.sources[key]

    def read(self, path):
        path = self.path(path)
        if str(path) not in self.documents:
            self.source(path)
            self.documents[str(path)] = json.loads(path.read_text()) if path.is_file() else {}
        return self.documents[str(path)]

    def payload(self, path, expected):
        if not path:
            return {"path": path, "expected_sha256": expected, "actual_sha256": None, "status": "missing", "reason": "path_unrecorded"}
        record = self.source(path)
        actual = record["sha256"]
        full = isinstance(expected, str) and re.fullmatch(r"[0-9a-f]{64}", expected) is not None
        return {**record, "actual_sha256": actual, "expected_sha256": expected,
                "status": "missing" if actual is None else "verified" if full and actual == expected else "partial",
                "reason": "payload_absent" if actual is None else "full_recorded_hash_missing" if not full else "hash_match" if actual == expected else "hash_mismatch"}

    def canonical(self, path, field):
        document = self.read(path)
        actual = json_digest({k: v for k, v in document.items() if k != field}) if document else None
        expected = document.get(field)
        return {"path": str(self.path(path)), "status": "missing" if not document else "verified" if actual == expected else "partial",
                "expected_sha256": expected, "actual_sha256": actual,
                "hash_kind": "canonical_json_excluding_" + field}


def verify_run_binding(auditor, run_path, run, prediction_path, metric, engine_sha256):
    if not run:
        return {"status": "missing", "checks": {}}
    checks = {"engine": same(run.get("engine_sha256"), engine_sha256),
              "prediction_bytes": auditor.payload(prediction_path, run.get("prediction_sha256")),
              "metric_run_bytes": auditor.payload(run_path, metric.get("run_record_sha256"))}
    for key in ("condition_id", "prediction_sha256", "input_manifest_sha256", "input_image_ids_sha256"):
        checks["metric_" + key] = same(metric.get(key), run.get(key))
    return {"status": classify_lineage({k: v["status"] for k, v in checks.items()}), "checks": checks}


def command_option(command, flag):
    words = shlex.split(command or "")
    return words[words.index(flag) + 1] if flag in words and words.index(flag) + 1 < len(words) else None


def audit_run(auditor, dataset, model, precision, condition, report, engine, config):
    corruption, severity = condition
    attempt = f"{dataset}_confirmatory_final_117_v1"
    matches = [cid for cid in report.get("metric_sha256", {}) if len(cid.split("__")) == 6
               and cid.split("__")[1] == model and cid.split("__")[2] in (("fp8", "fp8-none", "fp8-entropy") if precision == "fp8" else (precision,))
               and cid.split("__")[3] == f"{corruption}-s{severity}"]
    if len(matches) != 1:
        return {"corruption": corruption, "severity": severity, "status": "missing" if not matches else "partial",
                "reason": "missing_or_ambiguous_report_condition", "candidate_condition_ids": matches}
    cid = matches[0]
    paths = {kind: str(auditor.path(f"{folder}/{attempt}/{cid}.json")) for kind, folder in
             (("run", "manifests/runs"), ("prediction", "outputs/predictions"), ("input", "outputs/inputs"), ("metric", "outputs/metrics"))}
    run, metric, inputs = (auditor.read(paths[k]) for k in ("run", "metric", "input"))
    checks = {"run_prediction_metric": verify_run_binding(auditor, paths["run"], run, paths["prediction"], metric, engine.get("engine_sha256")),
              "metric_report": auditor.payload(paths["metric"], report["metric_sha256"][cid]),
              "engine_path": same(run.get("engine_path"), engine.get("engine")),
              "calibration": same(run.get("calibration_sha256"), engine.get("calibration_sha256")),
              "run_identity": same([run.get(k) for k in ("dataset", "model", "precision", "corruption", "severity")], [dataset, model, precision, corruption, severity]),
              "metric_identity": same([metric.get(k) for k in ("dataset", "model", "precision", "corruption", "severity")], [dataset, model, precision, corruption, severity]),
              "count": same(run.get("n_images"), DATASETS[dataset]),
              "metric_count": same(metric.get("n_images"), DATASETS[dataset]),
              "input_condition": same(inputs.get("condition_id"), cid)}
    for field, path in (("runner_sha256", "src/coco_infer_trt.py"), ("preprocess_sha256", "src/topic_c/coco_data.py"), ("decoder_sha256", "src/topic_c/yolo_decode.py")):
        checks[field] = auditor.payload(path, run.get(field))
    for field, key in (("annotation_sha256", "annotation"), ("class_map_sha256", "class_map")):
        checks[field] = auditor.payload(config.get(key), run.get(field))
    manifest_path = command_option(run.get("command"), "--image-manifest")
    if manifest_path:
        manifest = auditor.read(manifest_path)
        checks["manifest_canonical"] = auditor.canonical(manifest_path, "manifest_sha256")
        checks["manifest_run"] = same(manifest.get("manifest_sha256"), run.get("input_manifest_sha256"))
        checks["manifest_ids"] = same([r.get("image_id") for r in manifest.get("records", [])], inputs.get("image_ids"))
        checks["manifest_expected_ids"] = same(manifest.get("expected_image_ids"), inputs.get("image_ids"))
    else:
        checks["manifest_canonical"] = {"status": "missing"}
    ids = inputs.get("image_ids")
    annotations = auditor.read(config["annotation"]) if config.get("annotation") else {}
    annotation_ids = sorted(image["id"] for image in annotations.get("images", []))
    checks["annotation_image_ids"] = same(annotation_ids, ids)
    clean = auditor.read(config["clean_manifest"]) if config.get("clean_manifest") else {}
    checks["clean_image_ids"] = same(clean.get("expected_image_ids"), ids)
    checks["input_ids_canonical"] = same(json_digest(ids) if ids is not None else None, inputs.get("image_ids_sha256"))
    checks["input_ids_run"] = same(inputs.get("image_ids_sha256"), run.get("input_image_ids_sha256"))
    checks["input_manifest_run"] = same(inputs.get("input_manifest_sha256"), run.get("input_manifest_sha256"))
    checks["input_count_unique"] = same(len(set(ids)) if isinstance(ids, list) else None, DATASETS[dataset])
    # Prefix checks supplement full historical links; they can never replace them.
    checks["condition_engine_prefix"] = same(cid.split("__")[4], (run.get("engine_sha256") or "")[:8])
    checks["condition_manifest_prefix"] = same(cid.split("__")[5], (run.get("input_manifest_sha256") or "")[:8])
    return {"condition_id": cid, "corruption": corruption, "severity": severity,
            "status": classify_lineage({k: v["status"] for k, v in checks.items()}),
            "paths": paths, "input_manifest": manifest_path, "checks": checks, "run_record": run,
            "metric_record": metric, "historical_evaluator_source_binding": "missing_not_recorded_in_metric_schema"}


def audit(root):
    auditor = Auditor(root)
    execution_path = "configs/confirmatory_execution_v2.json"
    execution = auditor.read(execution_path)
    rows, dataset_inputs = [], {}
    for dataset, count in DATASETS.items():
        config = execution.get("datasets", {}).get(dataset, {})
        report_path = f"outputs/reports/{dataset}_confirmatory_final_117_v1_evaluation_complete.json"
        report = auditor.read(report_path)
        clean_path = config.get("clean_manifest", f"manifests/images/{dataset}_confirmatory_final_clean_v1.json")
        clean = auditor.read(clean_path)
        dataset_inputs[dataset] = {"expected_images": count, "original_manifest": auditor.source(clean_path),
                                  "original_control_semantics": clean_manifest_semantics(clean),
                                  "original_manifest_canonical": auditor.canonical(clean_path, "manifest_sha256"),
                                  "original_root": str(auditor.path(config.get("clean_root", f"data/datasets/{dataset}"))),
                                  "annotation": auditor.source(config["annotation"]) if config.get("annotation") else None,
                                  "class_map": auditor.source(config["class_map"]) if config.get("class_map") else None,
                                  "ordered_image_ids": clean.get("expected_image_ids"),
                                  "j95_holdout_manifest_status": "not_in_frozen_execution_config_requires_separate_generation",
                                  "image_payload_status": "not_rehashed_in_this_artifact_lineage_audit"}
        for model in MODELS:
            for precision in PRECISIONS:
                stem = f"{dataset}_{model}_{precision}.json"
                engine_path = f"manifests/engines/{ATTEMPT}/{stem}"
                graph_path = f"manifests/onnx/{ATTEMPT}/{stem}"
                fp32_path = f"manifests/onnx/{ATTEMPT}/{dataset}_{model}_fp32.json"
                training_path = f"manifests/training/{dataset}_{model}_confirmatory_s20260818_v1.json"
                engine, graph, fp32, training = (auditor.read(p) for p in (engine_path, graph_path, fp32_path, training_path))
                if not engine and not graph and not report:
                    rows.append({"dataset": dataset, "model": model, "precision": precision, "status": "missing", "checks": {}, "runs": []})
                    continue
                calibration_path = engine.get("calibration_list") or config.get("calibration")
                calibration = auditor.read(calibration_path) if calibration_path else {}
                checks = {"engine_bytes": auditor.payload(engine.get("engine"), engine.get("engine_sha256")),
                          "quantized_graph_bytes": auditor.payload(graph.get("output_onnx"), graph.get("output_onnx_sha256")),
                          "engine_graph_registry": auditor.payload(graph_path, engine.get("source_onnx_registry_sha256")),
                          "engine_graph_hash": same(engine.get("source_onnx_sha256"), graph.get("output_onnx_sha256")),
                          "engine_graph_path": same(engine.get("source_onnx"), graph.get("output_onnx")),
                          "graph_fp32_registry": auditor.payload(fp32_path, graph.get("source_onnx_registry_sha256")),
                          "graph_fp32_hash": same(graph.get("source_onnx_sha256"), fp32.get("onnx_sha256")),
                          "fp32_graph_bytes": auditor.payload(fp32.get("onnx"), fp32.get("onnx_sha256")),
                          "checkpoint_bytes": auditor.payload(fp32.get("source_checkpoint"), fp32.get("source_checkpoint_sha256")),
                          "training_registry": auditor.payload(training_path, fp32.get("training_registry_sha256")),
                          "training_checkpoint": same(training.get("best_weights_sha256"), fp32.get("source_checkpoint_sha256")),
                          "training_best_bytes": auditor.payload(training.get("best_weights"), training.get("best_weights_sha256")),
                          "build_log": auditor.payload(engine.get("build_log"), engine.get("build_log_sha256")),
                          "trtexec_binary": auditor.payload(engine.get("trtexec"), engine.get("trtexec_sha256")),
                          "calibration_canonical": auditor.canonical(calibration_path, "calibration_sha256") if calibration_path else {"status": "missing"},
                          "calibration_engine": same(calibration.get("calibration_sha256"), engine.get("calibration_sha256")),
                          "calibration_graph": same(calibration.get("calibration_sha256"), graph.get("calibration_sha256")),
                          "calibration_count": same(calibration.get("n_images"), 512),
                          "calibration_dataset_split": same([calibration.get("dataset"), calibration.get("split")], [dataset, "train"])}
                for name, document in (("engine", engine), ("graph", graph)):
                    checks[name + "_identity"] = same([document.get(k) for k in ("dataset", "model", "precision")], [dataset, model, precision])
                runs = [audit_run(auditor, dataset, model, precision, c, report, engine, config) for c in CONDITIONS]
                checks["historical_runs"] = {"status": classify_lineage({str(i): r["status"] for i, r in enumerate(runs)})}
                rows.append({"dataset": dataset, "model": model, "precision": precision,
                             "status": classify_lineage({k: v["status"] for k, v in checks.items()}),
                             "checks": checks, "runs": runs,
                             "registry_paths": {"engine": str(auditor.path(engine_path)), "quantized_onnx": str(auditor.path(graph_path)),
                                                "fp32_onnx": str(auditor.path(fp32_path)), "training": str(auditor.path(training_path))},
                             "engine_registry": engine, "quantized_onnx_registry": graph,
                             "fp32_onnx_registry": fp32, "training_registry": training,
                             "calibration_image_payload_status": "not_rehashed"})
    counts = Counter(row["status"] for row in rows)
    return {"schema_version": 1, "audited_at_utc": datetime.now(timezone.utc).isoformat(),
            "root": str(auditor.root), "execution_config": auditor.source(execution_path),
            "status_semantics": {"verified": "retained artifact bytes and explicit historical record links are consistent",
                                 "partial": "some retained evidence exists but one or more links are missing or inconsistent",
                                 "missing": "no retained evidence for the scoped treatment"},
            "limits": ["Newly computed engine hashes alone never verify historical predictions.",
                       "Consistency is not independent execution attestation or proof of an unmodified historical runtime.",
                       "Original/calibration/corruption image bytes are recorded but not rehashed here.",
                       "Historical metric schemas do not record an evaluator source/environment hash; AP is not recomputed by this audit."],
            "counts": {"treatments": len(rows), **{key: counts[key] for key in ("verified", "partial", "missing")},
                       "expected_runs": 156, "retained_runs": sum("condition_id" in run for row in rows for run in row["runs"])},
            "dataset_inputs": dataset_inputs, "treatments": rows,
            "fixed_example": {"dataset": "kitti", "model": "yolo11m", "corruption": "fog", "severity": 1,
                              "runs": [run for row in rows if row["dataset"] == "kitti" and row["model"] == "yolo11m"
                                       for run in row["runs"] if run.get("corruption") in ("clean", "fog") and run.get("severity") in (0, 1)]},
            "sources": sorted(auditor.sources.values(), key=lambda r: r["path"])}


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", default=".")
    parser.add_argument("--ssh", help="read-only SSH host; source is executed through stdin")
    parser.add_argument("--remote-root", default="/home/thuan/topic_c_ivc")
    parser.add_argument("--stdout", action="store_true")
    parser.add_argument("--out", default="outputs/analysis/cviu_v4/holdout_provenance")
    args = parser.parse_args()
    if args.ssh:
        command = "python3 - --stdout --root " + shlex.quote(args.remote_root)
        result = subprocess.run(["ssh", "-o", "BatchMode=yes", args.ssh, command],
                                input=Path(__file__).read_bytes(), stdout=subprocess.PIPE, check=True)
        report = json.loads(result.stdout)
    else:
        report = audit(args.root)
    if args.stdout:
        print(json.dumps(report, indent=2))
        return
    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    report["audit_code_sha256"] = sha256_file(__file__)
    target = out / "holdout_provenance.json"
    target.write_text(json.dumps(report, indent=2) + "\n")
    summary = ["# V4 holdout provenance audit", "", json.dumps(report["counts"], sort_keys=True), "",
               "Verified denotes retained artifact consistency, not execution attestation.", "",
               "| Dataset | Model | Precision | Lineage | Runs verified |", "|---|---|---|---|---|"]
    for row in report["treatments"]:
        summary.append(f"| {row['dataset']} | {row['model']} | {row['precision']} | {row['status']} | {sum(r['status'] == 'verified' for r in row['runs'])}/13 |")
    summary.extend(["", "## Limits", ""] + ["- " + s for s in report["limits"]])
    summary.extend(["", "Final-holdout JPEG-95 manifests are not part of the historical frozen execution config; generate them as new versioned inputs.", "",
                    "Report SHA-256 (file bytes): `" + sha256_file(target) + "`", ""])
    (out / "README.md").write_text("\n".join(summary))
    completion = {"counts": report["counts"], "report_sha256": sha256_file(target),
                  "readme_sha256": sha256_file(out / "README.md"), "audit_code_sha256": sha256_file(__file__),
                  "source_file_count": len(report["sources"]), "source_bytes": sum(s["bytes"] or 0 for s in report["sources"])}
    (out / "completion.json").write_text(json.dumps(completion, indent=2) + "\n")
    print(json.dumps(completion, indent=2))


if __name__ == "__main__":
    main()
