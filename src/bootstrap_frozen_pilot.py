#!/usr/bin/env python3
"""Run the prespecified 72 paired B=500 bootstrap cells for a frozen pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from execute_frozen_pilot import paths
from pilot_registry import canonical_hash
from topic_c.manifest import sha256_file


def load_plan(path: Path) -> dict:
    plan = json.loads(path.read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != canonical_hash(plan, "plan_sha256") or len(plan.get("runs", [])) != 117:
        raise SystemExit("PILOT BOOTSTRAP REFUSED: frozen plan is invalid")
    return plan


def cell_seed(model: str, corruption: str, severity: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{model}|{corruption}|{severity}".encode("utf-8")).digest()[:8], "big") % (2 ** 32)


def bootstrap_path(root: Path, attempt: str, model: str, precision: str, corruption: str, severity: int) -> Path:
    return root / "outputs" / "bootstrap" / attempt / f"{model}__{precision}__{corruption}-s{severity}.json"


def expected_cells(plan: dict) -> list[dict]:
    index = {(run["model"], run["precision"], run["corruption"], run["severity"]): run for run in plan["runs"]}
    cells = []
    for model in ("yolo11n", "yolo11m", "yolo11x"):
        for precision in ("int8-entropy", "fp8"):
            for corruption in ("gaussian_noise", "motion_blur", "fog", "jpeg"):
                for severity in (1, 3, 5):
                    keys = [(model, "fp32", "clean", 0), (model, precision, "clean", 0),
                            (model, "fp32", corruption, severity), (model, precision, corruption, severity)]
                    if any(key not in index for key in keys):
                        raise SystemExit(f"PILOT BOOTSTRAP REFUSED: incomplete frozen matrix for {model}/{precision}/{corruption}/s{severity}")
                    cells.append({"model": model, "precision": precision, "corruption": corruption, "severity": severity,
                                  "fp32_clean": index[keys[0]], "quant_clean": index[keys[1]],
                                  "fp32_corrupt": index[keys[2]], "quant_corrupt": index[keys[3]]})
    if len(cells) != 72:
        raise SystemExit("PILOT BOOTSTRAP REFUSED: expected 72 paired cells")
    return cells


def complete_evaluation(root: Path, attempt: str, plan_sha256: str) -> bool:
    report = root / "outputs" / "reports" / f"{attempt}_evaluation_complete.json"
    if not report.is_file():
        return False
    return json.loads(report.read_text(encoding="utf-8")).get("plan_sha256") == plan_sha256


def runner_command(root: Path, annotations: Path, attempt: str, cell: dict) -> list[str]:
    records = [cell[name] for name in ("fp32_clean", "quant_clean", "fp32_corrupt", "quant_corrupt")]
    files = [paths(root, attempt, record["condition_id"]) for record in records]
    output = bootstrap_path(root, attempt, cell["model"], cell["precision"], cell["corruption"], cell["severity"])
    return [
        sys.executable, str(root / "src" / "paired_bootstrap.py"), "--annotations", str(annotations),
        "--fp32-clean", str(files[0][0]), "--quant-clean", str(files[1][0]),
        "--fp32-corrupt", str(files[2][0]), "--quant-corrupt", str(files[3][0]),
        "--fp32-clean-input", str(files[0][1]), "--quant-clean-input", str(files[1][1]),
        "--fp32-corrupt-input", str(files[2][1]), "--quant-corrupt-input", str(files[3][1]),
        "--quant-label", cell["precision"], "--n-boot", "500", "--seed", str(cell_seed(cell["model"], cell["corruption"], cell["severity"])),
        "--out", str(output),
    ]


def validate_output(root: Path, attempt: str, cell: dict) -> None:
    output = bootstrap_path(root, attempt, cell["model"], cell["precision"], cell["corruption"], cell["severity"])
    result = json.loads(output.read_text(encoding="utf-8"))
    if result.get("n_boot") != 500 or result.get("n_images") != 5000 or result.get("quant_label") != cell["precision"]:
        raise SystemExit(f"PILOT BOOTSTRAP REFUSED: malformed bootstrap result: {output}")
    names = ("fp32_clean", "quant_clean", "fp32_corrupt", "quant_corrupt")
    for name in names:
        run = cell[name]
        prediction, input_record, _ = paths(root, attempt, run["condition_id"])
        actual = result.get("input_hashes", {}).get(name, {})
        if actual.get("prediction_sha256") != sha256_file(prediction) or actual.get("input_record_sha256") != sha256_file(input_record):
            raise SystemExit(f"PILOT BOOTSTRAP REFUSED: input hash mismatch in {output}: {name}")
        if actual.get("input_manifest_sha256") != run["input_manifest_sha256"]:
            raise SystemExit(f"PILOT BOOTSTRAP REFUSED: manifest hash mismatch in {output}: {name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--attempt", default="pilot_117_v1")
    parser.add_argument("--execute", action="store_true")
    parser.add_argument("--resume-verified", action="store_true", help="skip only existing bootstrap cells with exact linked-input hashes")
    parser.add_argument("--jobs", type=int, default=1,
                        help="bounded number of independent bootstrap cells to run concurrently (default: 1)")
    args = parser.parse_args()
    root, annotations = Path(args.project_root).resolve(), Path(args.annotations).resolve()
    if args.jobs < 1:
        raise SystemExit("PILOT BOOTSTRAP REFUSED: --jobs must be positive")
    plan = load_plan(Path(args.plan))
    cells = expected_cells(plan)
    outputs = [bootstrap_path(root, args.attempt, cell["model"], cell["precision"], cell["corruption"], cell["severity"]) for cell in cells]
    all_sources = [path for cell in cells for run in (cell["fp32_clean"], cell["quant_clean"], cell["fp32_corrupt"], cell["quant_corrupt"])
                   for path in paths(root, args.attempt, run["condition_id"])[:2]]
    status = {"dry_run": not args.execute, "attempt": args.attempt, "cells": len(cells),
              "evaluation_complete": complete_evaluation(root, args.attempt, plan["plan_sha256"]),
              "missing_prediction_or_input_files": sum(not path.is_file() for path in all_sources),
              "existing_bootstrap_outputs": sum(path.exists() for path in outputs),
              "first_command": runner_command(root, annotations, args.attempt, cells[0])}
    if not args.execute:
        print(json.dumps(status, indent=2))
        return
    if not status["evaluation_complete"]:
        raise SystemExit("PILOT BOOTSTRAP REFUSED: complete, hash-matched evaluation report is required")
    if status["missing_prediction_or_input_files"]:
        raise SystemExit("PILOT BOOTSTRAP REFUSED: linked prediction/input files are missing")
    if status["existing_bootstrap_outputs"] and not args.resume_verified:
        raise SystemExit("PILOT BOOTSTRAP REFUSED: bootstrap output already exists; use a new reviewed attempt")
    completed, pending = [], []
    for cell in cells:
        output = bootstrap_path(root, args.attempt, cell["model"], cell["precision"], cell["corruption"], cell["severity"])
        if output.exists():
            validate_output(root, args.attempt, cell)
            completed.append(cell)
        else:
            pending.append(cell)
    def run_cell(index: int, cell: dict) -> tuple[int, dict]:
        print(f"BOOTSTRAP RESUME {index}/{len(pending)} (completed={len(completed)}) {cell['model']} {cell['precision']} {cell['corruption']}-s{cell['severity']}", flush=True)
        subprocess.run(runner_command(root, annotations, args.attempt, cell), check=True)
        validate_output(root, args.attempt, cell)
        print(f"BOOTSTRAP VALID {index}/{len(pending)} {cell['model']} {cell['precision']} {cell['corruption']}-s{cell['severity']}", flush=True)
        return index, cell

    if args.jobs == 1:
        for index, cell in enumerate(pending, start=1):
            run_cell(index, cell)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(run_cell, index, cell) for index, cell in enumerate(pending, start=1)]
            for future in as_completed(futures):
                future.result()
    report = root / "outputs" / "reports" / f"{args.attempt}_bootstrap_complete.json"
    if report.exists():
        raise SystemExit(f"PILOT BOOTSTRAP REFUSED: completion report already exists: {report}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
                                  "attempt": args.attempt, "plan_sha256": plan["plan_sha256"], "cells": len(cells),
                                  "n_boot": 500, "all_bootstrap_input_hashes_validated": True,
                                  "artifacts_sha256": {str(path.relative_to(root)): sha256_file(path) for path in outputs}}, indent=2) + "\n", encoding="utf-8")
    print(f"PILOT BOOTSTRAP COMPLETE cells={len(cells)} attempt={args.attempt}")


if __name__ == "__main__":
    main()
