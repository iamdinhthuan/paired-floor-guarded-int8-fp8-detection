#!/usr/bin/env python3
"""Validate and summarize the RT-DETR/RetinaNet cross-family extension."""
from __future__ import annotations

import csv
import hashlib
import json
import statistics
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
EVIDENCE = ROOT / "artifacts/cross_family_v1"


def sha(path: Path) -> str: return hashlib.sha256(path.read_bytes()).hexdigest()


def canonical_sha(document: dict, field: str) -> str:
    payload = {key: value for key, value in document.items() if key != field}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def main() -> None:
    metrics_dir = EVIDENCE / "outputs/metrics"; runs_dir = EVIDENCE / "manifests/runs"
    metric_paths, run_paths = sorted(metrics_dir.glob("*.json")), sorted(runs_dir.glob("*.json"))
    if len(metric_paths) != 234 or len(run_paths) != 234: raise RuntimeError("cross-family grid must contain 234 metrics and runs")
    metrics = [json.loads(path.read_text()) for path in metric_paths]
    grid = {(r["dataset"], r["model"], r["precision"], r["corruption"], r["severity"]): r for r in metrics}
    if len(grid) != 234: raise RuntimeError("duplicate cross-family metric cells")
    expected = {(d,m,p,c,s) for d in ("voc","kitti","tt100k") for m in ("rtdetr-l","retinanet-r50-fpn-v2")
                for p in ("fp32","int8-entropy","fp8") for c,s in [("clean",0)] +
                [(c,s) for c in ("fog","gaussian_noise","jpeg","motion_blur") for s in (1,3,5)]}
    if set(grid) != expected: raise RuntimeError("cross-family metric Cartesian grid mismatch")
    for metric_path in metric_paths:
        metric = json.loads(metric_path.read_text()); run_path = runs_dir / metric_path.name
        run = json.loads(run_path.read_text())
        if metric["run_record_sha256"] != sha(run_path) or metric["prediction_sha256"] != run["prediction_sha256"]:
            raise RuntimeError(f"metric provenance mismatch: {metric_path}")
    q95_metrics_dir = EVIDENCE / "outputs/q95_metrics"
    q95_runs_dir = EVIDENCE / "manifests/q95_runs"
    q95_metric_paths = sorted(q95_metrics_dir.glob("*.json"))
    q95_run_paths = sorted(q95_runs_dir.glob("*.json"))
    if len(q95_metric_paths) != 18 or len(q95_run_paths) != 18:
        raise RuntimeError("matched JPEG-95 grid must contain exactly 18 metrics and runs")
    q95_metrics = [json.loads(path.read_text()) for path in q95_metric_paths]
    q95_grid = {(r["dataset"], r["model"], r["precision"]): r for r in q95_metrics}
    q95_expected = {
        (dataset, model, precision)
        for dataset in ("voc", "kitti", "tt100k")
        for model in ("rtdetr-l", "retinanet-r50-fpn-v2")
        for precision in ("fp32", "int8-entropy", "fp8")
    }
    if set(q95_grid) != q95_expected:
        raise RuntimeError("matched JPEG-95 Cartesian grid mismatch")
    input_identity: dict[str, tuple[str, str]] = {}
    for metric_path in q95_metric_paths:
        metric = json.loads(metric_path.read_text())
        run_path = q95_runs_dir / metric_path.name
        run = json.loads(run_path.read_text())
        if metric["corruption"] != "clean" or metric["severity"] != 0:
            raise RuntimeError(f"invalid matched-clean condition: {metric_path}")
        if metric["run_record_sha256"] != sha(run_path) or metric["prediction_sha256"] != run["prediction_sha256"]:
            raise RuntimeError(f"matched-clean provenance mismatch: {metric_path}")
        identity = (metric["input_manifest_sha256"], metric["input_image_ids_sha256"])
        previous = input_identity.setdefault(metric["dataset"], identity)
        if previous != identity:
            raise RuntimeError(f"matched-clean input identity differs within {metric['dataset']}")

    q95_report_path = EVIDENCE / "outputs/reports/cross_family_q95_clean_v1_complete.json"
    q95_report = json.loads(q95_report_path.read_text())
    if q95_report.get("report_sha256") != canonical_sha(q95_report, "report_sha256"):
        raise RuntimeError("matched-clean completion report canonical hash mismatch")
    if q95_report.get("conditions") != 18 or q95_report.get("linked_artifacts") != 72:
        raise RuntimeError("matched-clean completion report grid mismatch")
    for relative, digest in q95_report.get("local_artifact_sha256", {}).items():
        path = EVIDENCE / relative
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"matched-clean completion artifact mismatch: {path}")

    accuracy_report_path = EVIDENCE / "outputs/reports/cross_family_accuracy_v1_complete.json"
    accuracy_report = json.loads(accuracy_report_path.read_text())
    if accuracy_report.get("report_sha256") != canonical_sha(accuracy_report, "report_sha256"):
        raise RuntimeError("cross-family accuracy completion report canonical hash mismatch")
    if accuracy_report.get("conditions") != 234 or accuracy_report.get("linked_artifacts") != 936:
        raise RuntimeError("cross-family accuracy completion report grid mismatch")
    for relative, digest in accuracy_report.get("local_artifact_sha256", {}).items():
        path = EVIDENCE / relative
        if not path.is_file() or sha(path) != digest:
            raise RuntimeError(f"cross-family accuracy completion artifact mismatch: {path}")

    deployment_report = EVIDENCE / "outputs/reports/cross_family_deployment_v1_complete.json"
    report = json.loads(deployment_report.read_text())
    if report["conditions"] != 18 or report["records"] != 54 or len(report["artifact_sha256"]) != 108:
        raise RuntimeError("deployment completion grid mismatch")
    for relative, digest in report["artifact_sha256"].items():
        path = EVIDENCE / "outputs/deployment" / relative
        if not path.is_file() or sha(path) != digest: raise RuntimeError(f"deployment artifact mismatch: {path}")
    clean = {
        (dataset, model, precision): grid[dataset, model, precision, "clean", 0]["stats"]["AP"]
        for dataset, model, precision in q95_expected
    }
    matched_clean = {key: value["stats"]["AP"] for key, value in q95_grid.items()}
    summary=[]
    for model in ("rtdetr-l","retinanet-r50-fpn-v2"):
        for precision in ("int8-entropy","fp8"):
            values=[]; corrupted=[]
            for dataset in ("voc","kitti","tt100k"):
                for corruption in ("fog","gaussian_noise","jpeg","motion_blur"):
                    for severity in (1,3,5):
                        q=grid[dataset,model,precision,corruption,severity]["stats"]["AP"]
                        f=grid[dataset,model,"fp32",corruption,severity]["stats"]["AP"]
                        values.append(((q-f)-(matched_clean[dataset,model,precision]-matched_clean[dataset,model,"fp32"]))*100)
                        corrupted.append(q*100)
            summary.append({"model":model,"precision":precision,"mean_interaction_ap_points":statistics.mean(values),
                            "min_interaction_ap_points":min(values),"max_interaction_ap_points":max(values),
                            "mean_corrupted_ap_points":statistics.mean(corrupted),
                            "corrupted_cells_below_5_ap":sum(value < 5 for value in corrupted),"cells":36})
    records=[json.loads(path.read_text()) for path in (EVIDENCE/"outputs/deployment/records").glob("*.json")]
    deployment=[]
    for model in ("rtdetr-l","retinanet-r50-fpn-v2"):
        for precision in ("fp32","int8-entropy","fp8"):
            values=[r["latency_median_ms"] for r in records if r["model"]==model and r["precision"]==precision]
            deployment.append({"model":model,"precision":precision,"median_latency_ms":statistics.median(values),
                               "min_latency_ms":min(values),"max_latency_ms":max(values),"n":len(values)})
    generated=ROOT/"paper/generated"; generated.mkdir(exist_ok=True)
    with (generated/"cross_family_cells.csv").open("w",newline="") as stream:
        writer=csv.DictWriter(stream,fieldnames=["dataset","model","precision","corruption","severity","ap"]); writer.writeheader()
        for key,value in sorted(grid.items()): writer.writerow(dict(zip(writer.fieldnames,key+(value["stats"]["AP"],))))
    with (generated/"cross_family_q95_clean.csv").open("w", newline="") as stream:
        fields = ["dataset", "model", "precision", "ap", "input_manifest_sha256", "input_image_ids_sha256"]
        writer = csv.DictWriter(stream, fieldnames=fields); writer.writeheader()
        for key, value in sorted(q95_grid.items()):
            writer.writerow(dict(zip(fields, key + (value["stats"]["AP"], value["input_manifest_sha256"], value["input_image_ids_sha256"]))))
    clean_rows=[]
    for dataset in ("voc","kitti","tt100k"):
        for model in ("rtdetr-l","retinanet-r50-fpn-v2"):
            clean_rows.append((dataset,model,*[clean[dataset,model,p]*100 for p in ("fp32","int8-entropy","fp8")]))
    labels={"voc":"VOC","kitti":"KITTI","tt100k":"TT100K","rtdetr-l":"RT-DETR-L","retinanet-r50-fpn-v2":"RetinaNet-R50-FPN-v2"}
    lines=[r"\begin{tabular}{llrrr}",r"\toprule",r"Dataset & Family & FP32 & INT8 & FP8 \\",r"\midrule"]
    lines += [f"{labels[d]} & {labels[m]} & {a:.2f} & {b:.2f} & {c:.2f} \\\\" for d,m,a,b,c in clean_rows]
    lines += [r"\bottomrule",r"\end{tabular}"]
    (generated/"cross_family_clean.tex").write_text("\n".join(lines)+"\n")
    lines=[r"\begin{tabular}{llrrrr}",r"\toprule",r"Family & Format & Mean $E$ & $E$ range & Mean corrupted AP & AP$<5$ \\",r"\midrule"]
    for row in summary:
        lines.append(f"{labels[row['model']]} & {row['precision'].replace('int8-entropy','INT8').upper()} & "
                     f"{row['mean_interaction_ap_points']:+.2f} & {row['min_interaction_ap_points']:+.2f}--{row['max_interaction_ap_points']:+.2f} & {row['mean_corrupted_ap_points']:.2f} & "
                     f"{row['corrupted_cells_below_5_ap']}/36 \\\\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (generated/"cross_family_interaction.tex").write_text("\n".join(lines)+"\n")
    lines=[r"\begin{tabular}{llrr}",r"\toprule",r"Family & Format & Median latency (ms) & Range (ms) \\",r"\midrule"]
    for row in deployment:
        lines.append(f"{labels[row['model']]} & {row['precision'].replace('int8-entropy','INT8').upper()} & "
                     f"{row['median_latency_ms']:.3f} & {row['min_latency_ms']:.3f}--{row['max_latency_ms']:.3f} \\\\")
    lines += [r"\bottomrule",r"\end{tabular}"]
    (generated/"cross_family_deployment.tex").write_text("\n".join(lines)+"\n")
    source_artifacts = {
        str(path.relative_to(ROOT)): sha(path)
        for path in metric_paths + run_paths + q95_metric_paths + q95_run_paths
    }
    source_artifacts[str(deployment_report.relative_to(ROOT))] = sha(deployment_report)
    source_artifacts[str(q95_report_path.relative_to(ROOT))] = sha(q95_report_path)
    source_artifacts[str(accuracy_report_path.relative_to(ROOT))] = sha(accuracy_report_path)
    generated_outputs = [
        generated / name
        for name in (
            "cross_family_cells.csv",
            "cross_family_q95_clean.csv",
            "cross_family_clean.tex",
            "cross_family_interaction.tex",
            "cross_family_deployment.tex",
        )
    ]
    generated_artifacts = {str(path.relative_to(ROOT)): sha(path) for path in generated_outputs}
    analysis={"schema_version":2,"metrics":234,"runs":234,"linked_accuracy_artifacts":936,
              "matched_clean_metrics":18,"matched_clean_runs":18,"matched_clean_linked_artifacts":72,
              "matched_clean_completion_report_sha256":sha(q95_report_path),
              "deployment_records":54,"deployment_artifacts":108,"summary":summary,"deployment":deployment,
              "source_artifacts_sha256":source_artifacts,
              "generated_artifacts_sha256":generated_artifacts,
              "limitations":["one training seed","one calibration sample","descriptive cells","RT-DETR INT8 PTQ accuracy collapse",
                             "RetinaNet TT100K absolute accuracy floor"]}
    analysis["analysis_sha256"]=hashlib.sha256(json.dumps(analysis,sort_keys=True,separators=(",", ":")).encode()).hexdigest()
    analysis_path = EVIDENCE / "analysis.json"
    analysis_path.write_text(json.dumps(analysis,indent=2)+"\n")
    audit = {
        "schema_version": 1,
        "status": "valid",
        "analysis_file_sha256": sha(analysis_path),
        "analysis_sha256": analysis["analysis_sha256"],
        "q95_completion_file_sha256": sha(q95_report_path),
        "accuracy_completion_file_sha256": sha(accuracy_report_path),
        "deployment_completion_file_sha256": sha(deployment_report),
        "generated_artifacts_sha256": generated_artifacts,
    }
    audit["audit_sha256"] = canonical_sha(audit, "audit_sha256")
    (generated / "cross_family_evidence_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    print(json.dumps({"status":"valid","metrics":234,"deployment_records":54,"analysis_sha256":analysis["analysis_sha256"]}))


if __name__ == "__main__": main()
