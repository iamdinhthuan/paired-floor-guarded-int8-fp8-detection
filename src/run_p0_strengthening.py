#!/usr/bin/env python3
"""One-command, resumable P0 strengthening queue for the RTX 5090 host."""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from pilot_registry import canonical_hash
from topic_c.manifest import sha256_file


def resolved(root: Path, value: str) -> Path:
    path = Path(value)
    return path.resolve() if path.is_absolute() else (root / path).resolve()


def run(command: list[str]) -> None:
    print("P0 COMMAND " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def canonical_valid(path: Path, field: str) -> bool:
    if not path.is_file():
        return False
    try:
        document = json.loads(path.read_text())
    except json.JSONDecodeError:
        return False
    return document.get(field) == canonical_hash(document, field)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--bootstrap-jobs", type=int)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = Path(args.config).resolve()
    config = json.loads(config_path.read_text())
    coco = config["coco"]
    python = sys.executable

    clean_manifest = resolved(root, coco["clean_manifest"])
    if not clean_manifest.is_file():
        run(
            [
                python,
                str(root / "src" / "generate_coco_manifest.py"),
                "--annotations",
                str(resolved(root, coco["annotations"])),
                "--image-root",
                str(resolved(root, coco["clean_root"])),
                "--out",
                str(clean_manifest),
            ]
        )

    run(
        [
            python,
            str(root / "src" / "prepare_coco_uniform_p0.py"),
            "--project-root",
            str(root),
            "--config",
            str(config_path),
        ]
    )

    parity_complete = root / "outputs" / "reports" / "coco_clean_fp16_parity_complete_v1.json"
    if not canonical_valid(parity_complete, "parity_completion_sha256"):
        run(
            [
                python,
                str(root / "src" / "run_clean_fp16_parity.py"),
                "--project-root",
                str(root),
                "--config",
                str(config_path),
                "--dataset",
                "coco",
                "--resume-verified",
            ]
        )
    if not canonical_valid(parity_complete, "parity_completion_sha256"):
        raise SystemExit("P0 REFUSED: COCO consistency gate did not close")

    proposal = root / "manifests" / "plans" / "coco_uniform_p0_proposal_v1.json"
    frozen = root / "manifests" / "plans" / "coco_uniform_p0_frozen_v1.json"
    engine_registry = root / "manifests" / "engines" / "coco_yolo11_nmx_ladder_v1.json"
    calibration = resolved(root, coco["calibration_list"])
    if not proposal.exists():
        run(
            [
                python,
                str(root / "src" / "build_dataset_pilot_plan.py"),
                "--matrix",
                str(root / config["matrix"]),
                "--engine-registry",
                str(engine_registry),
                "--dataset",
                "coco",
                "--split",
                coco["split"],
                "--clean-manifest",
                coco["clean_manifest"],
                "--corruption-manifest-template",
                coco["corruption_manifest_template"],
                "--calibration-list",
                str(calibration),
                "--out",
                str(proposal),
            ]
        )
    if not frozen.exists():
        run(
            [
                python,
                str(root / "src" / "freeze_pilot_plan.py"),
                "--proposal",
                str(proposal),
                "--project-root",
                str(root),
                "--out",
                str(frozen),
            ]
        )
    plan = json.loads(frozen.read_text())
    if plan.get("plan_sha256") != canonical_hash(plan, "plan_sha256"):
        raise SystemExit("P0 REFUSED: frozen COCO plan hash mismatch")

    parity_args = []
    for model in config["models"]:
        parity_args += [
            "--fp16-parity",
            str(root / "outputs" / "reports" / f"coco_{model}_fp16_parity_v1.json"),
        ]
    attempt = coco["attempt"]
    execute = [
        python,
        str(root / "src" / "execute_dataset_pilot.py"),
        "--plan",
        str(frozen),
        "--project-root",
        str(root),
        "--dataset",
        "coco",
        "--split",
        coco["split"],
        "--annotations",
        str(resolved(root, coco["annotations"])),
        "--clean-root",
        str(resolved(root, coco["clean_root"])),
        "--corruption-root",
        str(resolved(root, coco["corruption_root"])),
        "--calibration-list",
        str(calibration),
        "--imgsz",
        str(coco["imgsz"]),
        "--expected-images",
        str(coco["expected_images"]),
        "--attempt",
        attempt,
        "--execute",
        "--resume-verified",
        "--reviewed-by",
        "project-owner-authorization-2026-08-10",
        *parity_args,
    ]
    run(execute)
    evaluation_report = root / "outputs" / "reports" / f"{attempt}_evaluation_complete.json"
    evaluation_valid = False
    if evaluation_report.is_file():
        evaluation_valid = json.loads(evaluation_report.read_text()).get("plan_sha256") == plan["plan_sha256"]
    if not evaluation_valid:
        run(
            [
                python,
                str(root / "src" / "evaluate_dataset_pilot.py"),
                "--plan",
                str(frozen),
                "--project-root",
                str(root),
                "--dataset",
                "coco",
                "--split",
                coco["split"],
                "--annotations",
                str(resolved(root, coco["annotations"])),
                "--attempt",
                attempt,
                "--expected-images",
                str(coco["expected_images"]),
                "--execute",
                "--resume-verified",
            ]
        )

    run(
        [
            python,
            str(root / "src" / "execute_codec_control_p0.py"),
            "--project-root",
            str(root),
            "--config",
            str(config_path),
        ]
    )
    bootstrap_command = [
        python,
        str(root / "src" / "bootstrap_matched_codec_p0.py"),
        "--project-root",
        str(root),
        "--config",
        str(config_path),
    ]
    if args.bootstrap_jobs:
        bootstrap_command += ["--jobs", str(args.bootstrap_jobs)]
    run(bootstrap_command)

    report = root / "outputs" / "reports" / "p0_strengthening_complete_v1.json"
    dependencies = [
        root / "outputs" / "reports" / "coco_uniform_ladder_p0_v1.json",
        parity_complete,
        evaluation_report,
        root / "outputs" / "reports" / f"{config['codec_control']['attempt']}_complete.json",
        root / "outputs" / "reports" / f"{config['codec_control']['bootstrap_attempt']}_complete.json",
    ]
    if not all(path.is_file() for path in dependencies):
        raise SystemExit("P0 REFUSED: one or more completion reports are missing")
    document = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "status": "complete",
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "reports": {str(path.relative_to(root)): sha256_file(path) for path in dependencies},
    }
    document["report_sha256"] = canonical_hash(document, "report_sha256")
    if report.exists():
        old = json.loads(report.read_text())
        if old.get("report_sha256") != canonical_hash(old, "report_sha256"):
            raise SystemExit("P0 REFUSED: existing final report is invalid")
    else:
        report.write_text(json.dumps(document, indent=2) + "\n")
    print(f"P0 STRENGTHENING COMPLETE report={report}")


if __name__ == "__main__":
    main()
