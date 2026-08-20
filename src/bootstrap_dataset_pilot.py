#!/usr/bin/env python3
"""Run the 72 paired COCO-area bootstrap cells for a frozen dataset pilot."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from datetime import datetime, timezone
from pathlib import Path

from execute_dataset_pilot import output_paths
from pilot_registry import canonical_hash
from topic_c.manifest import sha256_file


def seed(model: str, precision: str, corruption: str, severity: int) -> int:
    return int.from_bytes(hashlib.sha256(f"{model}|{precision}|{corruption}|{severity}".encode()).digest()[:8], "big") % (2 ** 32)


def cells(plan: dict) -> list[dict]:
    index = {(run["model"], run["precision"], run["corruption"], run["severity"]): run for run in plan["runs"]}
    result = []
    for model in ("yolo11n", "yolo11m", "yolo11x"):
        for precision in ("int8-entropy", "fp8"):
            for corruption in ("gaussian_noise", "motion_blur", "fog", "jpeg"):
                for severity in (1, 3, 5):
                    keys = [(model, "fp32", "clean", 0), (model, precision, "clean", 0), (model, "fp32", corruption, severity), (model, precision, corruption, severity)]
                    if any(key not in index for key in keys):
                        raise SystemExit(f"DATASET BOOTSTRAP REFUSED: incomplete linked cell: {keys}")
                    result.append({"model": model, "precision": precision, "corruption": corruption, "severity": severity,
                                   "fp32_clean": index[keys[0]], "quant_clean": index[keys[1]], "fp32_corrupt": index[keys[2]], "quant_corrupt": index[keys[3]]})
    if len(result) != 72:
        raise SystemExit("DATASET BOOTSTRAP REFUSED: expected exactly 72 cells")
    return result


def output(root: Path, attempt: str, cell: dict) -> Path:
    return root / "outputs" / "bootstrap" / attempt / f"{cell['model']}__{cell['precision']}__{cell['corruption']}-s{cell['severity']}.json"


def draw_output(root: Path, attempt: str, cell: dict) -> Path:
    return output(root, attempt, cell).with_suffix(".draws.npz")


def command(
    root: Path,
    annotations: Path,
    attempt: str,
    cell: dict,
    runner: Path,
    *,
    n_boot: int = 500,
    shared_seed: int | None = None,
) -> list[str]:
    records = [cell[name] for name in ("fp32_clean", "quant_clean", "fp32_corrupt", "quant_corrupt")]
    files = [output_paths(root, attempt, record["condition_id"]) for record in records]
    draw_seed = (
        shared_seed
        if shared_seed is not None
        else seed(cell["model"], cell["precision"], cell["corruption"], cell["severity"])
    )
    result = [sys.executable, str(runner), "--annotations", str(annotations),
            "--fp32-clean", str(files[0][0]), "--quant-clean", str(files[1][0]), "--fp32-corrupt", str(files[2][0]), "--quant-corrupt", str(files[3][0]),
            "--fp32-clean-input", str(files[0][1]), "--quant-clean-input", str(files[1][1]), "--fp32-corrupt-input", str(files[2][1]), "--quant-corrupt-input", str(files[3][1]),
            "--quant-label", cell["precision"], "--n-boot", str(n_boot), "--seed", str(draw_seed), "--out", str(output(root, attempt, cell))]
    if shared_seed is not None:
        result.extend(["--draw-cache", str(draw_output(root, attempt, cell))])
    return result


def verify(
    root: Path,
    attempt: str,
    cell: dict,
    expected_images: int,
    *,
    n_boot: int = 500,
    shared_seed: int | None = None,
) -> None:
    path = output(root, attempt, cell)
    data = json.loads(path.read_text(encoding="utf-8"))
    expected_seed = (
        shared_seed
        if shared_seed is not None
        else seed(cell["model"], cell["precision"], cell["corruption"], cell["severity"])
    )
    if (
        data.get("n_boot") != n_boot
        or data.get("seed") != expected_seed
        or data.get("n_images") != expected_images
        or data.get("quant_label") != cell["precision"]
    ):
        raise SystemExit(f"DATASET BOOTSTRAP REFUSED: malformed output: {path}")
    if shared_seed is not None:
        cache = draw_output(root, attempt, cell)
        linked_cache = data.get("draw_cache", {})
        if (
            not cache.is_file()
            or linked_cache.get("path") != str(cache.resolve())
            or linked_cache.get("sha256") != sha256_file(cache)
            or linked_cache.get("excess_shape") != [n_boot, 4]
            or linked_cache.get("psi_shape") != [n_boot]
        ):
            raise SystemExit(f"DATASET BOOTSTRAP REFUSED: draw-cache mismatch: {path}")
    for name in ("fp32_clean", "quant_clean", "fp32_corrupt", "quant_corrupt"):
        prediction, inputs, _ = output_paths(root, attempt, cell[name]["condition_id"])
        linked = data.get("input_hashes", {}).get(name, {})
        if (linked.get("prediction_sha256") != sha256_file(prediction) or linked.get("input_record_sha256") != sha256_file(inputs)
                or linked.get("input_manifest_sha256") != cell[name]["input_manifest_sha256"]):
            raise SystemExit(f"DATASET BOOTSTRAP REFUSED: linked input mismatch: {path}/{name}")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--plan", required=True); parser.add_argument("--project-root", required=True); parser.add_argument("--annotations", required=True)
    parser.add_argument("--attempt", required=True); parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--bootstrap-runner", help="defaults to paired_bootstrap.py; TT100K uses paired_bootstrap_tt100k.py")
    parser.add_argument("--execute", action="store_true"); parser.add_argument("--resume-verified", action="store_true")
    parser.add_argument("--jobs", type=int, default=1, help="bounded number of independent bootstrap cells to run concurrently (default: 1)")
    parser.add_argument("--n-boot", type=int, default=500)
    parser.add_argument(
        "--shared-seed",
        type=int,
        help="reuse one dataset-level draw sequence across every cell",
    )
    args = parser.parse_args()
    if args.jobs < 1 or args.n_boot < 1 or args.shared_seed is not None and args.shared_seed < 0:
        raise SystemExit("DATASET BOOTSTRAP REFUSED: jobs/n-boot must be positive and seed nonnegative")
    root, plan = Path(args.project_root).resolve(), json.loads(Path(args.plan).read_text(encoding="utf-8"))
    runner = Path(args.bootstrap_runner).resolve() if args.bootstrap_runner else root / "src" / "paired_bootstrap.py"
    if not runner.is_file():
        raise SystemExit(f"DATASET BOOTSTRAP REFUSED: bootstrap runner missing: {runner}")
    if plan.get("plan_sha256") != canonical_hash(plan, "plan_sha256") or len(plan.get("runs", [])) != 117:
        raise SystemExit("DATASET BOOTSTRAP REFUSED: invalid frozen plan")
    evaluation = root / "outputs" / "reports" / f"{args.attempt}_evaluation_complete.json"
    if not evaluation.is_file() or json.loads(evaluation.read_text(encoding="utf-8")).get("plan_sha256") != plan["plan_sha256"]:
        raise SystemExit("DATASET BOOTSTRAP REFUSED: complete evaluation is required")
    matrix = cells(plan); done, pending = [], []
    for cell in matrix:
        path = output(root, args.attempt, cell)
        if path.exists():
            if not args.resume_verified:
                raise SystemExit(f"DATASET BOOTSTRAP REFUSED: output already exists: {path}")
            verify(
                root,
                args.attempt,
                cell,
                args.expected_images,
                n_boot=args.n_boot,
                shared_seed=args.shared_seed,
            ); done.append(cell)
        else:
            pending.append(cell)
    if not args.execute:
        print(json.dumps({"dry_run": True, "cells": 72, "completed": len(done), "pending": len(pending)}, indent=2)); return
    def run_cell(ordinal: int, cell: dict) -> tuple[int, dict]:
        print(f"DATASET BOOTSTRAP {ordinal}/{len(pending)} {cell['model']} {cell['precision']} {cell['corruption']}-s{cell['severity']}", flush=True)
        subprocess.run(
            command(
                root,
                Path(args.annotations),
                args.attempt,
                cell,
                runner,
                n_boot=args.n_boot,
                shared_seed=args.shared_seed,
            ),
            check=True,
        )
        verify(
            root,
            args.attempt,
            cell,
            args.expected_images,
            n_boot=args.n_boot,
            shared_seed=args.shared_seed,
        )
        print(f"DATASET BOOTSTRAP VALID {ordinal}/{len(pending)} {cell['model']} {cell['precision']} {cell['corruption']}-s{cell['severity']}", flush=True)
        return ordinal, cell

    if args.jobs == 1:
        for ordinal, cell in enumerate(pending, start=1):
            run_cell(ordinal, cell)
    else:
        with ThreadPoolExecutor(max_workers=args.jobs) as executor:
            futures = [executor.submit(run_cell, ordinal, cell) for ordinal, cell in enumerate(pending, start=1)]
            for future in as_completed(futures):
                future.result()
    report = root / "outputs" / "reports" / f"{args.attempt}_bootstrap_complete.json"
    if report.exists():
        raise SystemExit(f"DATASET BOOTSTRAP REFUSED: report already exists: {report}")
    report.parent.mkdir(parents=True, exist_ok=True)
    report.write_text(json.dumps({"schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "attempt": args.attempt,
                                  "plan_sha256": plan["plan_sha256"], "cells": 72, "n_boot": args.n_boot,
                                  "shared_seed": args.shared_seed, "joint_dataset_draw_sequence": args.shared_seed is not None,
                                  "all_bootstrap_input_hashes_validated": True,
                                  "artifacts_sha256": {
                                      str(output(root, args.attempt, cell).relative_to(root)):
                                          sha256_file(output(root, args.attempt, cell))
                                      for cell in matrix
                                  },
                                  "draw_caches_sha256": {
                                      str(draw_output(root, args.attempt, cell).relative_to(root)):
                                          sha256_file(draw_output(root, args.attempt, cell))
                                      for cell in matrix
                                  } if args.shared_seed is not None else {}}, indent=2) + "\n", encoding="utf-8")
    print(f"DATASET BOOTSTRAP COMPLETE cells=72 attempt={args.attempt}")


if __name__ == "__main__":
    main()
