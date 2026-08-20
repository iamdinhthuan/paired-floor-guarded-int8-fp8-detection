#!/usr/bin/env python3
"""Execute and evaluate the two locked confirmatory 117-condition plans."""
from __future__ import annotations

import argparse
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_confirmatory_ladder import DATASETS, MODELS, canonical_hash, load_execution_config
from run_confirmatory_prep import load_prep_config
from topic_c.manifest import sha256_file


SAFE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def dataset_attempt(attempt: str, dataset: str) -> str:
    if not SAFE.fullmatch(attempt) or dataset not in DATASETS:
        raise ValueError("invalid dataset or attempt")
    return f"{dataset}_confirmatory_final_117_v1"


def resolve_inside(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise SystemExit(f"CONFIRMATORY INFERENCE REFUSED: missing {label}")
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise SystemExit(f"CONFIRMATORY INFERENCE REFUSED: {label} escapes project root")
    return path


def load_inference_config(root: Path, path: Path) -> dict:
    root, path = root.resolve(), path.resolve()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("CONFIRMATORY INFERENCE REFUSED: unreadable config") from exc
    if config.get("config_sha256") != canonical_hash(config, "config_sha256"):
        raise SystemExit("CONFIRMATORY INFERENCE REFUSED: config hash mismatch")
    if not isinstance(config.get("attempt"), str) or not SAFE.fullmatch(config["attempt"]):
        raise SystemExit("CONFIRMATORY INFERENCE REFUSED: unsafe attempt")
    if config.get("reviewed_by") != "project-owner-explicit-confirmatory-approval-2026-08-18":
        raise SystemExit("CONFIRMATORY INFERENCE REFUSED: explicit approval receipt label missing")
    for key, label in (("execution_config", "execution config"), ("prep_config", "prep config")):
        bound = resolve_inside(root, config.get(key), label)
        if not bound.is_file() or sha256_file(bound) != config.get(f"{key}_sha256"):
            raise SystemExit(f"CONFIRMATORY INFERENCE REFUSED: {label} hash mismatch")
    manifest = config.get("source_manifest")
    if not isinstance(manifest, list) or not manifest:
        raise SystemExit("CONFIRMATORY INFERENCE REFUSED: empty source manifest")
    seen: set[str] = set()
    for record in manifest:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"} or record["path"] in seen:
            raise SystemExit("CONFIRMATORY INFERENCE REFUSED: malformed source manifest")
        seen.add(record["path"])
        source = resolve_inside(root, record["path"], "source manifest path")
        if not source.is_file() or sha256_file(source) != record["sha256"]:
            raise SystemExit(f"CONFIRMATORY INFERENCE REFUSED: source manifest hash mismatch: {record['path']}")
    return config


def validate_prep_completion(
    path: Path,
    *,
    attempt: str,
    prep_config_sha256: str,
    parity_completion_sha256: str,
) -> dict[str, Path]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("CONFIRMATORY INFERENCE REFUSED: unreadable prep completion") from exc
    if report.get("prep_report_sha256") != canonical_hash(report, "prep_report_sha256"):
        raise SystemExit("CONFIRMATORY INFERENCE REFUSED: prep completion hash mismatch")
    if (
        report.get("attempt") != attempt
        or report.get("config_sha256") != prep_config_sha256
        or report.get("parity_completion_sha256") != parity_completion_sha256
    ):
        raise SystemExit("CONFIRMATORY INFERENCE REFUSED: prep completion binding mismatch")
    rows = report.get("datasets")
    if (
        not isinstance(rows, list)
        or len(rows) != 2
        or {row.get("dataset") for row in rows if isinstance(row, dict)} != set(DATASETS)
    ):
        raise SystemExit("CONFIRMATORY INFERENCE REFUSED: prep dataset grid mismatch")
    plans: dict[str, Path] = {}
    for row in rows:
        plan_path = Path(row.get("frozen_plan", "")).resolve()
        if not plan_path.is_file() or sha256_file(plan_path) != row.get("frozen_plan_sha256"):
            raise SystemExit(f"CONFIRMATORY INFERENCE REFUSED: frozen plan hash mismatch: {row.get('dataset')}")
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        if (
            plan.get("plan_sha256") != canonical_hash(plan, "plan_sha256")
            or plan.get("plan_sha256") != row.get("plan_sha256")
            or len(plan.get("runs", [])) != 117
        ):
            raise SystemExit(f"CONFIRMATORY INFERENCE REFUSED: invalid frozen plan: {row.get('dataset')}")
        plans[row["dataset"]] = plan_path
    return plans


def parity_paths_for_dataset(rows: list[dict], dataset: str) -> list[Path]:
    selected = [row for row in rows if row.get("dataset") == dataset]
    if len(selected) != 3 or {row.get("model") for row in selected} != set(MODELS):
        raise SystemExit(f"CONFIRMATORY INFERENCE REFUSED: exactly three parity reports required: {dataset}")
    by_model = {row["model"]: row for row in selected}
    result = []
    for model in MODELS:
        row = by_model[model]
        path = Path(row.get("report", "")).resolve()
        marker = path.with_suffix(path.suffix + ".complete")
        if (
            not path.is_file()
            or not marker.is_file()
            or sha256_file(path) != row.get("report_sha256")
            or marker.read_text(encoding="utf-8").strip() != sha256_file(path)
        ):
            raise SystemExit(f"CONFIRMATORY INFERENCE REFUSED: parity report hash mismatch: {dataset}/{model}")
        report = json.loads(path.read_text(encoding="utf-8"))
        if not report.get("pass") or report.get("dataset") != dataset or report.get("model") != model:
            raise SystemExit(f"CONFIRMATORY INFERENCE REFUSED: parity report failed: {dataset}/{model}")
        result.append(path)
    return result


def valid_evaluation(path: Path, plan_sha: str, dataset: str, split: str) -> bool:
    if not path.is_file():
        return False
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        return False
    metrics = report.get("metric_sha256")
    return (
        report.get("dataset") == dataset
        and report.get("split") == split
        and report.get("runs") == 117
        and report.get("plan_sha256") == plan_sha
        and isinstance(metrics, dict)
        and len(metrics) == 117
    )


def run(command: list[str]) -> None:
    print("CONFIRMATORY INFERENCE COMMAND " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--wait-for-prep-report", required=True)
    parser.add_argument("--wait-for-parity-report", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = Path(args.config).resolve()
    prep_report_path = Path(args.wait_for_prep_report).resolve()
    parity_completion_path = Path(args.wait_for_parity_report).resolve()
    report_out = Path(args.report_out).resolve()
    if report_out.exists() or report_out.with_suffix(report_out.suffix + ".complete").exists():
        raise SystemExit(f"CONFIRMATORY INFERENCE REFUSED: completion report exists: {report_out}")
    config = load_inference_config(root, config_path)
    execution_path = resolve_inside(root, config["execution_config"], "execution config")
    prep_config_path = resolve_inside(root, config["prep_config"], "prep config")
    execution, _ = load_execution_config(root, execution_path)
    prep_config = load_prep_config(root, prep_config_path)
    if execution["attempt"] != config["attempt"] or prep_config["attempt"] != config["attempt"]:
        raise SystemExit("CONFIRMATORY INFERENCE REFUSED: attempt mismatch across configs")
    while not prep_report_path.is_file() or not parity_completion_path.is_file():
        print("CONFIRMATORY INFERENCE waiting for prep and parity completion reports", flush=True)
        time.sleep(args.poll_seconds)
    parity_completion = json.loads(parity_completion_path.read_text(encoding="utf-8"))
    if parity_completion.get("parity_completion_sha256") != canonical_hash(
        parity_completion, "parity_completion_sha256"
    ):
        raise SystemExit("CONFIRMATORY INFERENCE REFUSED: parity completion hash mismatch")
    plans = validate_prep_completion(
        prep_report_path,
        attempt=config["attempt"],
        prep_config_sha256=sha256_file(prep_config_path),
        parity_completion_sha256=sha256_file(parity_completion_path),
    )
    completed: list[dict] = []
    for dataset in DATASETS:
        data = execution["datasets"][dataset]
        plan_path = plans[dataset]
        plan = json.loads(plan_path.read_text(encoding="utf-8"))
        attempt = dataset_attempt(config["attempt"], dataset)
        evaluation = root / "outputs" / "reports" / f"{attempt}_evaluation_complete.json"
        if not valid_evaluation(evaluation, plan["plan_sha256"], dataset, data["split"]):
            command = [
                sys.executable,
                str(root / "src" / "execute_dataset_pilot.py"),
                "--plan",
                str(plan_path),
                "--project-root",
                str(root),
                "--dataset",
                dataset,
                "--split",
                data["split"],
                "--annotations",
                str(resolve_inside(root, data["annotation"], f"{dataset} annotation")),
                "--class-map",
                str(resolve_inside(root, data["class_map"], f"{dataset} class map")),
                "--clean-root",
                str(resolve_inside(root, data["clean_root"], f"{dataset} clean root")),
                "--corruption-root",
                str(resolve_inside(root, data["corruption_root"], f"{dataset} corruption root")),
                "--calibration-list",
                str(resolve_inside(root, data["calibration"], f"{dataset} calibration")),
                "--imgsz",
                str(data["imgsz"]),
                "--expected-images",
                str(data["expected_images"]),
                "--attempt",
                attempt,
                "--execute",
                "--resume-verified",
                "--reviewed-by",
                config["reviewed_by"],
            ]
            for parity_path in parity_paths_for_dataset(parity_completion["reports"], dataset):
                command.extend(["--fp16-parity", str(parity_path)])
            run(command)
            run(
                [
                    sys.executable,
                    str(root / "src" / "evaluate_dataset_pilot.py"),
                    "--plan",
                    str(plan_path),
                    "--project-root",
                    str(root),
                    "--dataset",
                    dataset,
                    "--split",
                    data["split"],
                    "--annotations",
                    str(resolve_inside(root, data["annotation"], f"{dataset} annotation")),
                    "--attempt",
                    attempt,
                    "--expected-images",
                    str(data["expected_images"]),
                    "--execute",
                    "--resume-verified",
                ]
            )
        if not valid_evaluation(evaluation, plan["plan_sha256"], dataset, data["split"]):
            raise SystemExit(f"CONFIRMATORY INFERENCE REFUSED: evaluation incomplete: {dataset}")
        completed.append(
            {
                "dataset": dataset,
                "partition_role": data["partition_role"],
                "machine_split": data["split"],
                "attempt": attempt,
                "plan": str(plan_path),
                "plan_sha256": plan["plan_sha256"],
                "evaluation_report": str(evaluation),
                "evaluation_report_sha256": sha256_file(evaluation),
            }
        )
    document = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": config["attempt"],
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "prep_report": str(prep_report_path),
        "prep_report_sha256": sha256_file(prep_report_path),
        "parity_completion": str(parity_completion_path),
        "parity_completion_sha256": sha256_file(parity_completion_path),
        "datasets": completed,
    }
    document["inference_report_sha256"] = canonical_hash(document, "inference_report_sha256")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    report_out.with_suffix(report_out.suffix + ".complete").write_text(
        sha256_file(report_out) + "\n", encoding="utf-8"
    )
    print(f"CONFIRMATORY INFERENCE COMPLETE datasets=2 output={report_out}")


if __name__ == "__main__":
    main()
