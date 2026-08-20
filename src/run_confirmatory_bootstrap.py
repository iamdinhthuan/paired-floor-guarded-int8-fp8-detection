#!/usr/bin/env python3
"""Run joint-schedule B=2,000 bootstraps after confirmatory inference."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from pilot_registry import canonical_hash
from topic_c.manifest import sha256_file


DATASETS = ("voc", "kitti")
SAFE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def dataset_seed(namespace: str, dataset: str) -> int:
    if not namespace or dataset not in DATASETS:
        raise ValueError("invalid bootstrap seed namespace or dataset")
    digest = hashlib.sha256(f"{namespace}|{dataset}".encode()).digest()
    return int.from_bytes(digest[:8], "big") % (2**32)


def resolve_inside(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SystemExit(f"CONFIRMATORY BOOTSTRAP REFUSED: missing {label}")
    root = root.resolve()
    path = (root / value).resolve()
    if path != root and root not in path.parents:
        raise SystemExit(f"CONFIRMATORY BOOTSTRAP REFUSED: {label} escapes project root")
    return path


def load_config(root: Path, path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("CONFIRMATORY BOOTSTRAP REFUSED: unreadable config") from exc
    if config.get("config_sha256") != canonical_hash(config, "config_sha256"):
        raise SystemExit("CONFIRMATORY BOOTSTRAP REFUSED: config hash mismatch")
    if not isinstance(config.get("attempt"), str) or not SAFE.fullmatch(config["attempt"]):
        raise SystemExit("CONFIRMATORY BOOTSTRAP REFUSED: unsafe attempt")
    if (
        not isinstance(config.get("n_boot"), int)
        or config["n_boot"] < 1
        or not isinstance(config.get("jobs"), int)
        or not 1 <= config["jobs"] <= 4
        or not isinstance(config.get("seed_namespace"), str)
        or not config["seed_namespace"]
    ):
        raise SystemExit("CONFIRMATORY BOOTSTRAP REFUSED: invalid execution envelope")
    rows = config.get("datasets")
    if (
        not isinstance(rows, list)
        or len(rows) != 2
        or [row.get("dataset") for row in rows if isinstance(row, dict)] != list(DATASETS)
    ):
        raise SystemExit("CONFIRMATORY BOOTSTRAP REFUSED: dataset sequence mismatch")
    for row in rows:
        annotation = resolve_inside(root, row.get("annotations"), "annotations")
        if not annotation.is_file() or sha256_file(annotation) != row.get("annotations_sha256"):
            raise SystemExit(
                f"CONFIRMATORY BOOTSTRAP REFUSED: annotation hash mismatch: {row.get('dataset')}"
            )
        if not isinstance(row.get("expected_images"), int) or row["expected_images"] < 1:
            raise SystemExit("CONFIRMATORY BOOTSTRAP REFUSED: invalid expected image count")
    manifest = config.get("source_manifest")
    if not isinstance(manifest, list) or not manifest:
        raise SystemExit("CONFIRMATORY BOOTSTRAP REFUSED: empty source manifest")
    seen: set[str] = set()
    for record in manifest:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"}:
            raise SystemExit("CONFIRMATORY BOOTSTRAP REFUSED: malformed source manifest")
        relative = record["path"]
        if relative in seen:
            raise SystemExit("CONFIRMATORY BOOTSTRAP REFUSED: duplicate source manifest path")
        seen.add(relative)
        source = resolve_inside(root, relative, "source manifest path")
        if not source.is_file() or sha256_file(source) != record["sha256"]:
            raise SystemExit(
                f"CONFIRMATORY BOOTSTRAP REFUSED: source manifest hash mismatch: {relative}"
            )
    return config


def validate_inference_completion(path: Path, *, attempt: str) -> dict[str, dict]:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("CONFIRMATORY BOOTSTRAP REFUSED: unreadable inference completion") from exc
    if report.get("inference_report_sha256") != canonical_hash(
        report, "inference_report_sha256"
    ) or report.get("attempt") != attempt:
        raise SystemExit("CONFIRMATORY BOOTSTRAP REFUSED: invalid inference completion")
    rows = report.get("datasets")
    if (
        not isinstance(rows, list)
        or len(rows) != 2
        or {row.get("dataset") for row in rows if isinstance(row, dict)} != set(DATASETS)
    ):
        raise SystemExit("CONFIRMATORY BOOTSTRAP REFUSED: inference dataset grid mismatch")
    validated: dict[str, dict] = {}
    for row in rows:
        dataset = row["dataset"]
        if (
            row.get("partition_role") != "final"
            or row.get("machine_split") != "test"
            or row.get("attempt") != f"{dataset}_confirmatory_final_117_v1"
        ):
            raise SystemExit(
                f"CONFIRMATORY BOOTSTRAP REFUSED: final-partition binding mismatch: {dataset}"
            )
        plan = Path(row.get("plan", "")).resolve()
        evaluation = Path(row.get("evaluation_report", "")).resolve()
        if not plan.is_file() or not evaluation.is_file():
            raise SystemExit(f"CONFIRMATORY BOOTSTRAP REFUSED: missing input report: {dataset}")
        plan_doc = json.loads(plan.read_text(encoding="utf-8"))
        if plan_doc.get("plan_sha256") != row.get("plan_sha256"):
            raise SystemExit(f"CONFIRMATORY BOOTSTRAP REFUSED: plan hash mismatch: {dataset}")
        if sha256_file(evaluation) != row.get("evaluation_report_sha256"):
            raise SystemExit(f"CONFIRMATORY BOOTSTRAP REFUSED: evaluation hash mismatch: {dataset}")
        validated[dataset] = row
    return validated


def validate_bootstrap_completion(
    root: Path,
    path: Path,
    *,
    attempt: str,
    plan_sha256: str,
    n_boot: int,
    shared_seed: int,
) -> dict:
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("CONFIRMATORY BOOTSTRAP REFUSED: unreadable bootstrap completion") from exc
    artifacts = report.get("artifacts_sha256")
    draw_caches = report.get("draw_caches_sha256")
    if (
        report.get("attempt") != attempt
        or report.get("plan_sha256") != plan_sha256
        or report.get("cells") != 72
        or report.get("n_boot") != n_boot
        or report.get("shared_seed") != shared_seed
        or report.get("joint_dataset_draw_sequence") is not True
        or report.get("all_bootstrap_input_hashes_validated") is not True
        or not isinstance(artifacts, dict)
        or len(artifacts) != 72
        or not isinstance(draw_caches, dict)
        or len(draw_caches) != 72
    ):
        raise SystemExit("CONFIRMATORY BOOTSTRAP REFUSED: incomplete bootstrap report")
    for relative, expected in artifacts.items():
        artifact = resolve_inside(root, relative, "bootstrap artifact")
        if not artifact.is_file() or sha256_file(artifact) != expected:
            raise SystemExit(
                f"CONFIRMATORY BOOTSTRAP REFUSED: artifact hash mismatch: {relative}"
            )
        document = json.loads(artifact.read_text(encoding="utf-8"))
        if document.get("n_boot") != n_boot or document.get("seed") != shared_seed:
            raise SystemExit(
                f"CONFIRMATORY BOOTSTRAP REFUSED: artifact schedule mismatch: {relative}"
            )
    for relative, expected in draw_caches.items():
        cache = resolve_inside(root, relative, "bootstrap draw cache")
        if not cache.is_file() or sha256_file(cache) != expected:
            raise SystemExit(
                f"CONFIRMATORY BOOTSTRAP REFUSED: draw-cache hash mismatch: {relative}"
            )
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--wait-for-inference-report", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = Path(args.config).resolve()
    prerequisite = Path(args.wait_for_inference_report).resolve()
    report_out = Path(args.report_out).resolve()
    if report_out.exists() or report_out.with_suffix(report_out.suffix + ".complete").exists():
        raise SystemExit(f"CONFIRMATORY BOOTSTRAP REFUSED: completion report exists: {report_out}")
    config = load_config(root, config_path)
    while not prerequisite.is_file():
        print("CONFIRMATORY BOOTSTRAP waiting for inference completion", flush=True)
        time.sleep(args.poll_seconds)
    inference = validate_inference_completion(prerequisite, attempt=config["attempt"])
    config_rows = {row["dataset"]: row for row in config["datasets"]}
    completed = []
    for dataset in DATASETS:
        row = inference[dataset]
        data = config_rows[dataset]
        shared_seed = dataset_seed(config["seed_namespace"], dataset)
        bootstrap_report = (
            root / "outputs" / "reports" / f"{row['attempt']}_bootstrap_complete.json"
        )
        if not bootstrap_report.is_file():
            command = [
                sys.executable,
                str(root / "src" / "bootstrap_dataset_pilot.py"),
                "--plan",
                row["plan"],
                "--project-root",
                str(root),
                "--annotations",
                str(resolve_inside(root, data["annotations"], "annotations")),
                "--attempt",
                row["attempt"],
                "--expected-images",
                str(data["expected_images"]),
                "--n-boot",
                str(config["n_boot"]),
                "--shared-seed",
                str(shared_seed),
                "--jobs",
                str(config["jobs"]),
                "--execute",
                "--resume-verified",
            ]
            print("CONFIRMATORY BOOTSTRAP COMMAND " + " ".join(command), flush=True)
            subprocess.run(command, check=True)
        validate_bootstrap_completion(
            root,
            bootstrap_report,
            attempt=row["attempt"],
            plan_sha256=row["plan_sha256"],
            n_boot=config["n_boot"],
            shared_seed=shared_seed,
        )
        completed.append(
            {
                "dataset": dataset,
                "attempt": row["attempt"],
                "n_boot": config["n_boot"],
                "shared_seed": shared_seed,
                "bootstrap_report": str(bootstrap_report),
                "bootstrap_report_sha256": sha256_file(bootstrap_report),
            }
        )
    document = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": config["attempt"],
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "inference_report": str(prerequisite),
        "inference_report_sha256": sha256_file(prerequisite),
        "datasets": completed,
    }
    document["bootstrap_report_sha256"] = canonical_hash(
        document, "bootstrap_report_sha256"
    )
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    report_out.with_suffix(report_out.suffix + ".complete").write_text(
        sha256_file(report_out) + "\n", encoding="utf-8"
    )
    print(f"CONFIRMATORY BOOTSTRAP COMPLETE datasets=2 output={report_out}")


if __name__ == "__main__":
    main()
