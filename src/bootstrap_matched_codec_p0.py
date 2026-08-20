#!/usr/bin/env python3
"""Recompute all paired cells against the JPEG-95 matched clean baseline."""

from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from pilot_registry import canonical_hash
from topic_c.manifest import sha256_file


CORRUPTIONS = ("gaussian_noise", "motion_blur", "fog", "jpeg")
SEVERITIES = (1, 3, 5)


def resolved(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def index_records(root: Path, attempt: str, dataset: str) -> dict[tuple, tuple[dict, Path]]:
    result = {}
    for path in sorted((root / "manifests" / "runs" / attempt).glob("*.json")):
        record = json.loads(path.read_text())
        if record.get("dataset") != dataset:
            continue
        key = (
            record.get("model"),
            record.get("precision"),
            record.get("corruption"),
            int(record.get("severity", -1)),
        )
        if key in result:
            raise SystemExit(f"MATCHED BOOTSTRAP REFUSED: duplicate run key: {dataset}/{key}")
        result[key] = (record, path)
    return result


def prediction_and_input(root: Path, attempt: str, record: dict) -> tuple[Path, Path]:
    condition = record["condition_id"]
    return (
        root / "outputs" / "predictions" / attempt / f"{condition}.json",
        root / "outputs" / "inputs" / attempt / f"{condition}.json",
    )


def seed(dataset: str, model: str, precision: str, corruption: str, severity: int) -> int:
    text = f"matched-codec|{dataset}|{model}|{precision}|{corruption}|{severity}"
    return int.from_bytes(hashlib.sha256(text.encode()).digest()[:8], "big") % (2**32)


def verify(path: Path, n_boot: int, expected_images: int, precision: str) -> None:
    data = json.loads(path.read_text())
    if (
        data.get("n_boot") != n_boot
        or data.get("n_images") != expected_images
        or data.get("quant_label") != precision
        or set(data.get("input_hashes", {}))
        != {"fp32_clean", "quant_clean", "fp32_corrupt", "quant_corrupt"}
    ):
        raise SystemExit(f"MATCHED BOOTSTRAP REFUSED: malformed output: {path}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--jobs", type=int)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    codec = config["codec_control"]
    attempt = codec["bootstrap_attempt"]
    clean_attempt = codec["attempt"]
    n_boot = int(codec["bootstrap_replicates"])
    jobs = args.jobs or int(codec["bootstrap_jobs"])
    if jobs < 1:
        raise SystemExit("MATCHED BOOTSTRAP REFUSED: jobs must be positive")

    tasks = []
    for data in codec["datasets"]:
        dataset = data["dataset"]
        clean = index_records(root, clean_attempt, dataset)
        corrupt = index_records(root, data["corruption_attempt"], dataset)
        annotations = resolved(root, data["annotations"])
        runner = root / "src" / ("paired_bootstrap_tt100k.py" if dataset == "tt100k" else "paired_bootstrap.py")
        for model in config["models"]:
            for precision in ("int8-entropy", "fp8"):
                clean_keys = [
                    (model, "fp32", "codec_control", 0),
                    (model, precision, "codec_control", 0),
                ]
                if any(key not in clean for key in clean_keys):
                    raise SystemExit(f"MATCHED BOOTSTRAP REFUSED: missing codec baseline: {dataset}/{model}/{precision}")
                for corruption in CORRUPTIONS:
                    for severity in SEVERITIES:
                        corrupt_keys = [
                            (model, "fp32", corruption, severity),
                            (model, precision, corruption, severity),
                        ]
                        if any(key not in corrupt for key in corrupt_keys):
                            raise SystemExit(f"MATCHED BOOTSTRAP REFUSED: missing corrupt cell: {dataset}/{model}/{precision}/{corruption}/{severity}")
                        output = (
                            root
                            / "outputs"
                            / "bootstrap"
                            / attempt
                            / f"{dataset}__{model}__{precision}__{corruption}-s{severity}.json"
                        )
                        tasks.append(
                            {
                                "dataset": dataset,
                                "model": model,
                                "precision": precision,
                                "corruption": corruption,
                                "severity": severity,
                                "annotations": annotations,
                                "runner": runner,
                                "clean_attempt": clean_attempt,
                                "corrupt_attempt": data["corruption_attempt"],
                                "records": [
                                    clean[clean_keys[0]][0],
                                    clean[clean_keys[1]][0],
                                    corrupt[corrupt_keys[0]][0],
                                    corrupt[corrupt_keys[1]][0],
                                ],
                                "expected_images": int(data["expected_images"]),
                                "output": output,
                            }
                        )
    if len(tasks) != 288:
        raise SystemExit(f"MATCHED BOOTSTRAP REFUSED: expected 288 tasks, found {len(tasks)}")

    def run_task(ordinal: int, task: dict) -> tuple[int, Path]:
        output = task["output"]
        if output.exists():
            verify(output, n_boot, task["expected_images"], task["precision"])
            print(f"MATCHED BOOTSTRAP RESUME {ordinal}/288 {output.name}", flush=True)
            return ordinal, output
        attempts = [task["clean_attempt"], task["clean_attempt"], task["corrupt_attempt"], task["corrupt_attempt"]]
        pairs = [prediction_and_input(root, attempt_name, record) for attempt_name, record in zip(attempts, task["records"])]
        if any(not prediction.is_file() or not inputs.is_file() for prediction, inputs in pairs):
            raise SystemExit(f"MATCHED BOOTSTRAP REFUSED: missing input files for {output}")
        output.parent.mkdir(parents=True, exist_ok=True)
        command = [
            sys.executable,
            str(task["runner"]),
            "--annotations",
            str(task["annotations"]),
            "--fp32-clean",
            str(pairs[0][0]),
            "--quant-clean",
            str(pairs[1][0]),
            "--fp32-corrupt",
            str(pairs[2][0]),
            "--quant-corrupt",
            str(pairs[3][0]),
            "--fp32-clean-input",
            str(pairs[0][1]),
            "--quant-clean-input",
            str(pairs[1][1]),
            "--fp32-corrupt-input",
            str(pairs[2][1]),
            "--quant-corrupt-input",
            str(pairs[3][1]),
            "--quant-label",
            task["precision"],
            "--n-boot",
            str(n_boot),
            "--seed",
            str(seed(task["dataset"], task["model"], task["precision"], task["corruption"], task["severity"])),
            "--out",
            str(output),
        ]
        print(f"MATCHED BOOTSTRAP RUN {ordinal}/288 {output.name}", flush=True)
        subprocess.run(command, check=True)
        verify(output, n_boot, task["expected_images"], task["precision"])
        return ordinal, output

    completed = []
    with ThreadPoolExecutor(max_workers=jobs) as executor:
        futures = [executor.submit(run_task, ordinal, task) for ordinal, task in enumerate(tasks, 1)]
        for future in as_completed(futures):
            completed.append(future.result())

    report = root / "outputs" / "reports" / f"{attempt}_complete.json"
    if report.exists():
        old = json.loads(report.read_text())
        if old.get("report_sha256") != canonical_hash(old, "report_sha256"):
            raise SystemExit("MATCHED BOOTSTRAP REFUSED: invalid existing completion report")
        print("MATCHED BOOTSTRAP RESUME COMPLETE cells=288")
        return
    hashes = {
        str(task["output"].relative_to(root)): sha256_file(task["output"])
        for task in tasks
    }
    document = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": attempt,
        "clean_baseline": "deterministic JPEG quality 95, subsampling 0",
        "cells": 288,
        "n_boot": n_boot,
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "artifacts_sha256": hashes,
    }
    document["report_sha256"] = canonical_hash(document, "report_sha256")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps(document, indent=2) + "\n")
    print(f"MATCHED BOOTSTRAP COMPLETE cells=288 report={report}")


if __name__ == "__main__":
    main()
