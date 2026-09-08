"""Audit retained treatment provenance without inference or invented metadata.

Run from the repository root: python analysis/build_treatment_verification.py
Missing binary payloads are reported, whereas contradictory retained evidence
raises ValueError. Hashes establish consistency, not historical execution proof.
"""
from __future__ import annotations

import argparse
from collections import Counter
import hashlib
import json
from pathlib import Path
import re

MODELS = ("yolo11n", "yolo11m", "yolo11x")
PRECISIONS = ("int8-entropy", "fp8")
DATASETS = ("coco", "voc", "kitti", "tt100k")
HOST_PREFIX = "/home/thuan/topic_c_ivc/"
ARCHIVE = Path("artifacts/four_dataset_pilot_v1")
FULL_HASH = re.compile(r"[0-9a-f]{64}\Z")


def require(condition, message):
    if not condition:
        raise ValueError(message)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_digest(document, self_field):
    content = {k: v for k, v in document.items() if k != self_field}
    return hashlib.sha256(json.dumps(content, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def check_hash(value, name):
    require(isinstance(value, str) and FULL_HASH.fullmatch(value), f"invalid full SHA-256: {name}")


def verify_byte_link(path, expected):
    check_hash(expected, str(path))
    require(sha256_file(path) == expected, f"file-byte hash mismatch: {path}")


def verify_calibration(path, dataset):
    path = Path(path)
    doc = json.loads(path.read_text())
    require(doc.get("dataset") == dataset, f"calibration dataset mismatch: {path}")
    require(doc.get("split") == "train", f"calibration split mismatch: {path}")
    records = doc.get("records", [])
    require(doc.get("n_images") == len(records) == 512, f"calibration count mismatch: {path}")
    require(len({r["source_relpath"] for r in records}) == len(records), f"duplicate calibration images: {path}")
    for record in records:
        check_hash(record.get("sha256"), "calibration image recorded hash")
    actual = canonical_digest(doc, "calibration_sha256")
    require(doc.get("calibration_sha256") == actual, f"calibration canonical hash mismatch: {path}")
    marker = Path(str(path) + ".complete")
    require(marker.is_file() and marker.read_text().strip() == actual, f"calibration marker mismatch/missing: {path}")
    return {"status": "canonical_identity_verified", "canonical_sha256": actual,
            "hash_kind": "canonical_json_excluding_calibration_sha256",
            "n_images": len(records), "split": "train", "image_payload_status": "not_assessed"}


def verify_payload(path, expected, expected_bytes=None):
    path = Path(path)
    check_hash(expected, str(path))
    if not path.is_file():
        return {"status": "recorded_only_payload_absent", "expected_sha256": expected,
                "hash_kind": "file_bytes", "actual_sha256": None}
    actual = sha256_file(path)
    require(actual == expected, f"payload hash mismatch: {path}")
    if expected_bytes is not None:
        require(path.stat().st_size == expected_bytes, f"payload size mismatch: {path}")
    return {"status": "artifact_bytes_verified", "expected_sha256": expected,
            "actual_sha256": actual, "hash_kind": "file_bytes", "bytes": path.stat().st_size}


def map_historical_path(root, dataset, recorded):
    require(isinstance(recorded, str) and recorded.startswith(HOST_PREFIX), f"unrecognized historical path: {recorded}")
    relative = Path(recorded[len(HOST_PREFIX):])
    require(".." not in relative.parts and not relative.is_absolute(), f"unsafe historical path: {recorded}")
    return Path(root) / (Path() if dataset == "coco" else ARCHIVE) / relative


def normalize_precision(value):
    return "fp8" if value in {"fp8", "fp8-none", "fp8-entropy"} else value


def verify_universe(rows, expected):
    keys = [(r["layer"], r["dataset"], r["model"], r["precision"]) for r in rows]
    duplicates = [k for k, n in Counter(keys).items() if n != 1]
    require(not duplicates, f"duplicate treatment identities: {duplicates}")
    require(set(keys) == expected, f"treatment universe missing={expected - set(keys)}, extra={set(keys) - expected}")


def verify_runs(records, identity, engine_sha256, calibration_sha256, expected_conditions, expected_images):
    conditions = [(r["corruption"], r["severity"]) for r in records]
    require(len(set(conditions)) == len(conditions), f"duplicate run conditions: {identity}")
    require(set(conditions) == expected_conditions,
            f"run scope missing={expected_conditions - set(conditions)}, extra={set(conditions) - expected_conditions}: {identity}")
    check_hash(engine_sha256, "engine")
    present_calibration = 0
    for run in records:
        require((run["dataset"], run["model"], normalize_precision(run["precision"])) == identity,
                f"run treatment identity mismatch: {identity}")
        require(run.get("engine_sha256") == engine_sha256, f"run engine identity mismatch: {identity}")
        parts = run["condition_id"].split("__")
        require(len(parts) == 6 and parts[4] == engine_sha256[:8], f"condition engine identity mismatch: {identity}")
        # The matched-clean wrapper shortened its ID token; the structured
        # precision above must still explicitly bind this run to entropy INT8.
        token_precision = "int8-entropy" if parts[2] == "int8" and run["corruption"] == "codec_control" else normalize_precision(parts[2])
        require(parts[0].startswith(identity[0] + "_") and parts[1] == identity[1]
                and token_precision == identity[2], f"condition treatment mismatch: {identity}")
        condition_token = f"{run['corruption']}-s{run['severity']}"
        require(parts[3].replace("codec-control", "codec_control") == condition_token,
                f"condition corruption/severity mismatch: {identity}")
        require(run.get("n_images") == expected_images, f"run image count mismatch: {identity}")
        for field in ("preprocess_sha256", "decoder_sha256", "annotation_sha256",
                      "input_image_ids_sha256", "input_manifest_sha256", "prediction_sha256", "runner_sha256"):
            check_hash(run.get(field), field)
        require(parts[5] == run["input_manifest_sha256"][:8], f"condition input identity mismatch: {identity}")
        if run.get("calibration_sha256") is not None:
            require(run["calibration_sha256"] == calibration_sha256, f"run calibration binding mismatch: {identity}")
            present_calibration += 1
    stable = {}
    for field in ("preprocess_sha256", "decoder_sha256", "class_map_sha256", "annotation_sha256", "input_image_ids_sha256"):
        values = {r.get(field) for r in records}
        require(None not in values and len(values) == 1, f"run numerical contract differs: {identity} {field}")
        stable[field] = next(iter(values))
    environments = sorted({json.dumps(r.get("runtime_environment", {}), sort_keys=True) for r in records})
    return {"engine_binding_status": "record_binding_verified", "count": len(records),
            "conditions": [list(c) for c in sorted(conditions)], "stable_recorded_contract": stable,
            "calibration_hash_present_count": present_calibration,
            "calibration_hash_unrecorded_count": len(records) - present_calibration,
            "runner_sha256_values": sorted({r["runner_sha256"] for r in records}),
            "source_precision_tokens": sorted({r["condition_id"].split("__")[2] for r in records}),
            "runtime_environment_records": [json.loads(e) for e in environments],
            "runtime_environment_status": "partial_recorded_not_reconstructed",
            "prediction_payload_status": "not_assessed",
            "execution_authenticity_status": "not_independently_established"}


def holdout_identity(report, model, precision, expected_conditions):
    selected = []
    for condition_id, metric_hash in report["metric_sha256"].items():
        parts = condition_id.split("__")
        require(len(parts) == 6, f"malformed holdout condition: {condition_id}")
        if parts[1] != model or normalize_precision(parts[2]) != precision:
            continue
        require(parts[0] == f"{report['dataset']}_{report['split']}", f"holdout dataset/split mismatch: {condition_id}")
        corruption, severity = parts[3].rsplit("-s", 1)
        require(re.fullmatch(r"[0-9a-f]{8}", parts[4]) is not None, f"invalid holdout engine prefix: {condition_id}")
        check_hash(metric_hash, "holdout metric (not engine)")
        selected.append({"condition_id": condition_id, "metric_sha256": metric_hash,
                         "condition": (corruption, int(severity)), "prefix": parts[4], "token": parts[2]})
    conditions = [item["condition"] for item in selected]
    require(len(set(conditions)) == len(conditions), "duplicate holdout conditions")
    require(set(conditions) == expected_conditions, "holdout conditions missing or extra")
    prefixes = {item["prefix"] for item in selected}
    require(len(prefixes) == 1, "inconsistent holdout engine prefix")
    return {"status": "digest_prefix_only", "engine_sha256": None,
            "engine_digest_prefix": next(iter(prefixes)), "hash_kind": "condition_id_8_hex_prefix",
            "source_precision_tokens": sorted({item["token"] for item in selected}),
            "metric_record_count": len(selected), "metric_payload_status": "not_assessed",
            "metric_records": [{k: item[k] for k in ("condition_id", "metric_sha256")} for item in selected]}


def build(root):
    root = Path(root).resolve()
    sources = {}

    def source(path):
        path = Path(path)
        if not path.is_absolute():
            path = root / path
        relative = path.relative_to(root).as_posix()
        sources[relative] = {"path": relative, "sha256": sha256_file(path), "hash_kind": "file_bytes"}
        return relative

    def read(path):
        return json.loads((root / source(path)).read_text())

    def payload(dataset, recorded, digest, size=None):
        local = map_historical_path(root, dataset, recorded)
        result = verify_payload(local, digest, size)
        result.update(recorded_path=recorded, mapped_local_path=local.relative_to(root).as_posix())
        return result

    config_path = "configs/ivc_format_contrast_v1.json"
    config = read(config_path)
    require(set(config["models"]) == set(MODELS) and set(config["precisions"]) == set(PRECISIONS), "primary configuration universe changed")
    require([d["dataset"] for d in config["datasets"]] == list(DATASETS), "primary dataset universe changed")
    graph_path = "outputs/reports/ivc_quantization_graph_coverage_transfer_v1.json"
    graph_report = read(graph_path)
    graph_entries = graph_report["graphs"]
    graph_by_hash = {g["onnx_sha256"]: g for g in graph_entries}
    require(len(graph_entries) == len(graph_by_hash) == 18, "duplicate/missing transfer graph reports")
    used_graphs = set()
    rows = []
    for dataset_config in config["datasets"]:
        dataset = dataset_config["dataset"]
        base = Path() if dataset == "coco" else ARCHIVE
        calibration_name = f"{dataset}_train_clean_512_s20260807{'_p0' if dataset == 'coco' else ''}_v1.json"
        calibration_path = base / "manifests/calibration" / calibration_name
        calibration = verify_calibration(root / calibration_path, dataset)
        calibration["source"] = source(calibration_path)
        calibration["marker_source"] = source(str(calibration_path) + ".complete")
        conditions = {(dataset_config["clean_corruption"], 0)} | {
            (c, s) for c in config["corruptions"] for s in config["severities"]}
        candidates = []
        for attempt, clean in [(dataset_config["clean_source_attempt"], True), (dataset_config["corruption_source_attempt"], False)]:
            for path in sorted((root / "manifests/runs" / attempt).glob("*.json")):
                run = json.loads(path.read_text())
                if run.get("dataset") != dataset or run.get("model") not in MODELS or normalize_precision(run.get("precision")) not in PRECISIONS:
                    continue
                if clean and (run.get("corruption"), run.get("severity")) != (dataset_config["clean_corruption"], 0):
                    continue
                if not clean and run.get("corruption") == "clean":
                    continue
                candidates.append((source(path), run))
        for model in MODELS:
            for precision in PRECISIONS:
                key = (dataset, model, precision)
                stem = f"{dataset}_{model}_{precision}_v1.json"
                engine_path = base / "manifests/engines" / stem
                onnx_path = base / "manifests/onnx" / stem
                engine, onnx = read(engine_path), read(onnx_path)
                for record in (engine, onnx):
                    require((record["dataset"], record["model"], normalize_precision(record["precision"])) == key, f"registry identity mismatch: {key}")
                    require(record["calibration_sha256"] == calibration["canonical_sha256"], f"registry calibration mismatch: {key}")
                    require(map_historical_path(root, dataset, record["calibration_list"]) == root / calibration_path, f"registry calibration path mismatch: {key}")
                verify_byte_link(root / onnx_path, engine["source_onnx_registry_sha256"])
                require(engine["source_onnx_sha256"] == onnx["output_onnx_sha256"], f"engine/ONNX digest mismatch: {key}")
                require(engine["source_onnx"] == onnx["output_onnx"], f"engine/ONNX path mismatch: {key}")
                selected = [(p, r) for p, r in candidates if r["model"] == model and normalize_precision(r["precision"]) == precision]
                require(all(r.get("engine_path") == engine["engine"] for _, r in selected), f"run engine path mismatch: {key}")
                run_scope = verify_runs([r for _, r in selected], key, engine["engine_sha256"], calibration["canonical_sha256"], conditions, dataset_config["expected_images"])
                run_scope["sources"] = [p for p, _ in selected]
                engine_payload = payload(dataset, engine["engine"], engine["engine_sha256"], engine["engine_bytes"])
                onnx_payload = payload(dataset, onnx["output_onnx"], onnx["output_onnx_sha256"])
                log_payload = payload(dataset, engine["build_log"], engine["build_log_sha256"])
                if log_payload["status"] == "artifact_bytes_verified":
                    source(log_payload["mapped_local_path"])
                graph = graph_by_hash.get(onnx["output_onnx_sha256"])
                if graph is None:
                    require(dataset == "coco", f"missing transfer graph report: {key}")
                    graph_status = {"status": "report_not_retained", "scope": "Q/DQ graph coverage", "source": None}
                else:
                    require(dataset != "coco", "unexpected COCO binding to transfer graph report")
                    require(graph["onnx_path"] == onnx_payload["mapped_local_path"], f"graph report path mismatch: {key}")
                    used_graphs.add(graph["onnx_sha256"])
                    graph_status = {"status": "report_bound_to_registered_onnx", "source": graph_path,
                                    "graph_payload_status": onnx_payload["status"], "retained_report": graph,
                                    "audit_recomputed": False}
                quantizer_fields = ("quantize_mode", "calibration_eps", "op_types_to_exclude", "modelopt_version", "onnx_version", "command")
                command = engine.get("command", [])
                rows.append({"layer": "primary", "dataset": dataset, "model": model, "precision": precision,
                             "identity": {"status": "full_digest_record_bound", "engine_sha256": engine["engine_sha256"], "hash_kind": "file_bytes"},
                             "engine_registry_source": str(engine_path), "onnx_registry_source": str(onnx_path),
                             "onnx_registry_byte_link_status": "record_binding_verified", "run_scope": run_scope,
                             "engine_payload": engine_payload, "onnx_payload": onnx_payload, "build_log_payload": log_payload,
                             "calibration": {**calibration, "per_engine_binding_status": "registry_digest_and_path_verified"},
                             "graph": graph_status, "kernel_precision_status": "not_assessed",
                             "build_recorded": {k: engine.get(k) for k in ("command", "trtexec", "trtexec_sha256", "tensorrt_python_version", "workspace", "imgsz", "calibration_method")},
                             "tf32_command_status": "explicitly_disabled" if "--noTF32" in command else "not_explicitly_disabled",
                             "quantizer_recorded": {k: onnx[k] for k in quantizer_fields if k in onnx},
                             "quantizer_unrecorded_fields": [k for k in quantizer_fields if k not in onnx],
                             "full_environment_status": "not_reconstructed"})
    require(used_graphs == set(graph_by_hash), "unbound transfer graph report")
    holdout_config_path = "configs/confirmatory_execution_v2.json"
    holdout_config = read(holdout_config_path)
    holdout_conditions = {("clean", 0)} | {(c, s) for c in holdout_config["corruptions"] for s in holdout_config["severities"]}
    for dataset in ("voc", "kitti"):
        report_path = f"outputs/reports/{dataset}_confirmatory_final_117_v1_evaluation_complete.json"
        report = read(report_path)
        require(report["dataset"] == dataset and report["split"] == "test", "holdout completion identity mismatch")
        require(report["runs"] == len(report["metric_sha256"]) == 117, "holdout completion count mismatch")
        calibration_path = holdout_config["datasets"][dataset]["calibration"]
        calibration = verify_calibration(root / calibration_path, dataset)
        calibration["source"] = source(calibration_path)
        calibration["marker_source"] = source(calibration_path + ".complete")
        for model in MODELS:
            for precision in PRECISIONS:
                rows.append({"layer": "holdout", "dataset": dataset, "model": model, "precision": precision,
                             "identity": holdout_identity(report, model, precision, holdout_conditions),
                             "completion_report_source": report_path, "configuration_source": holdout_config_path,
                             "calibration": {**calibration, "per_engine_binding_status": "declared_in_config_not_per_engine_verified"},
                             "engine_registry_source": None, "onnx_registry_source": None,
                             "engine_payload": {"status": "unrecorded_full_identity_and_payload"},
                             "onnx_payload": {"status": "unrecorded_full_identity_and_payload"},
                             "graph": {"status": "report_not_retained"}, "kernel_precision_status": "not_assessed",
                             "full_environment_status": "not_reconstructed", "tf32_command_status": "unrecorded_for_quantized_engine",
                             "reference_config_not_quantized_engine_policy": holdout_config["reference"],
                             "run_scope": {"engine_binding_status": "full_run_records_not_retained", "count": 0}})
    expected = {("primary", d, m, p) for d in DATASETS for m in MODELS for p in PRECISIONS} | {
        ("holdout", d, m, p) for d in ("voc", "kitti") for m in MODELS for p in PRECISIONS}
    verify_universe(rows, expected)
    primary = [r for r in rows if r["layer"] == "primary"]
    paired = 0
    for dataset in DATASETS:
        for model in MODELS:
            pair = [r for r in primary if (r["dataset"], r["model"]) == (dataset, model)]
            require(len({r["calibration"]["canonical_sha256"] for r in pair}) == 1, "paired calibration list differs")
            paired += 1
    summary = {"schema_version": 1, "primary_treatments": len(primary), "holdout_treatments": len(rows) - len(primary),
               "primary_run_records_bound": sum(r["run_scope"]["count"] for r in primary),
               "primary_engine_payloads_verified": sum(r["engine_payload"]["status"] == "artifact_bytes_verified" for r in primary),
               "primary_engine_payloads_absent": sum(r["engine_payload"]["status"] == "recorded_only_payload_absent" for r in primary),
               "primary_onnx_payloads_verified": sum(r["onnx_payload"]["status"] == "artifact_bytes_verified" for r in primary),
               "primary_build_logs_verified": sum(r["build_log_payload"]["status"] == "artifact_bytes_verified" for r in primary),
               "primary_graph_reports_bound": sum(r["graph"]["status"] == "report_bound_to_registered_onnx" for r in primary),
               "primary_graph_reports_absent": sum(r["graph"]["status"] == "report_not_retained" for r in primary),
               "primary_paired_calibration_blocks_verified": paired,
               "primary_run_calibration_hash_present": sum(r["run_scope"]["calibration_hash_present_count"] for r in primary),
               "primary_run_calibration_hash_unrecorded": sum(r["run_scope"]["calibration_hash_unrecorded_count"] for r in primary),
               "holdout_full_engine_hashes_available": 0, "holdout_prefix_only_identities": 12,
               "calibration_documents_verified": 6, "kernel_precision_verified": 0,
               "full_environments_reconstructed": 0, "calibration_image_payloads_checked": 0}
    summary["groups"] = []
    for label, group in [("Primary COCO", [r for r in primary if r["dataset"] == "coco"]),
                         ("Primary transfer", [r for r in primary if r["dataset"] != "coco"]),
                         ("Holdout (partial)", [r for r in rows if r["layer"] == "holdout"])]:
        summary["groups"].append({"label": label, "treatments": len(group),
                                  "full_run_binding": sum(r["run_scope"]["engine_binding_status"] == "record_binding_verified" for r in group),
                                  "engine_bytes": sum(r["engine_payload"]["status"] == "artifact_bytes_verified" for r in group),
                                  "graph_reports": sum(r["graph"]["status"] == "report_bound_to_registered_onnx" for r in group)})
    source("analysis/build_treatment_verification.py")
    source("tests/test_treatment_verification.py")
    manifest = {"schema_version": 1, "scope": "Retained-record consistency and available local payload hashes; not independent historical execution or kernel-precision verification.",
                "path_mapping": {"recorded_prefix": HOST_PREFIX, "primary_coco_base": ".", "primary_transfer_base": str(ARCHIVE), "fallback_search": False},
                "graph_scope": "18 primary transfer graphs; six primary COCO reports absent; non-YOLO extensions outside this 36-row inventory.",
                "hash_semantics": {"source_and_payload": "SHA-256 of file bytes", "calibration": "SHA-256 of sorted compact JSON excluding calibration_sha256", "holdout_engine": "eight-hex condition-ID prefix only; metric hashes are not engine hashes"},
                "treatments": rows, "summary": summary,
                "compact_evidence_sources": [sources[p] for p in sorted(sources)],
                "compact_evidence_exclusions": ["TensorRT binary engines", "ONNX binary graphs", "calibration image payloads", "bulk predictions"],
                "portability_note": "Rerunning with metadata only preserves record and canonical-calibration checks but reports absent payloads; local byte-verification statuses are not independently reproducible without those payloads."}
    return manifest, summary


def render_table(summary):
    return "\n".join([
        "% Generated by analysis/build_treatment_verification.py; do not edit.",
        r"\begin{tabular}{lrrrr}", r"\toprule",
        r"Evidence layer & Treatments & Full run binding & Engine bytes & Q/DQ reports \\", r"\midrule",
        *[f"{g['label']} & {g['treatments']} & {g['full_run_binding']} & {g['engine_bytes']} & {g['graph_reports']} " + r"\\"
          for g in summary["groups"]], r"\bottomrule", r"\end{tabular}",
        r"\par\smallskip\noindent\footnotesize "
        + f"The {summary['primary_run_records_bound']} primary condition records bind to 24 full recorded engine hashes. "
        + "Zero denotes unavailable verification evidence, not zero quantized operators. "
        + "All 12 primary format pairs share canonically verified calibration-list identities; image bytes were not rechecked. "
        + "Holdout identities are eight-hex prefixes only. Graph reports are retained audits, not newly recomputed audits or kernel-precision measurements. "
        + "No full execution environment was reconstructed.", ""])


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    parser.add_argument("--output", type=Path, default=Path("outputs/analysis/treatment_verification_v1"))
    parser.add_argument("--table", type=Path, default=Path("Thuan_paper_3_CVIU_revised/generated/treatment_verification_summary.tex"))
    args = parser.parse_args()
    manifest, summary = build(args.root)
    output, table = args.root / args.output, args.root / args.table
    output.mkdir(parents=True, exist_ok=True)
    table.parent.mkdir(parents=True, exist_ok=True)
    for name, document in (("manifest.json", manifest), ("summary.json", summary)):
        (output / name).write_text(json.dumps(document, indent=2, sort_keys=True) + "\n")
    table.write_text(render_table(summary))
    print(json.dumps(summary, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
