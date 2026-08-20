#!/usr/bin/env python3
"""Create immutable transfer corruption caches and freeze their 117-run plans after parity."""
from __future__ import annotations

import argparse
import hashlib
import json
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from topic_c.manifest import read_manifest, sha256_file, validate_manifest


def canonical_hash(document: dict, excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def valid(path: Path, field: str) -> bool:
    if not path.is_file():
        return False
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return document.get(field) == canonical_hash(document, field)


def complete_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    manifest = read_manifest(path)
    marker = path.with_suffix(path.suffix + ".complete")
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != manifest["manifest_sha256"]:
        return None
    return manifest


def source_bytes(manifest: dict, clean_root: Path) -> int:
    return sum((clean_root / record["source_relpath"]).stat().st_size for record in manifest["records"])


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True); parser.add_argument("--wait-for-parity-report", required=True)
    parser.add_argument("--config", required=True); parser.add_argument("--report-out", required=True); parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root, prerequisite, config_path, report = (Path(args.project_root).resolve(), Path(args.wait_for_parity_report).resolve(),
                                                Path(args.config).resolve(), Path(args.report_out).resolve())
    if report.exists():
        raise SystemExit(f"TRANSFER PREP REFUSED: report already exists: {report}")
    config = json.loads(config_path.read_text(encoding="utf-8"))
    matrix_path, corruption_config = root / config["matrix"], root / config["corruption_config"]
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    if matrix.get("expected_runs_per_dataset") != 117 or len(matrix.get("conditions", [])) != 13:
        raise SystemExit("TRANSFER PREP REFUSED: unexpected pilot matrix")
    while not valid(prerequisite, "parity_queue_report_sha256"):
        print(f"TRANSFER PREP waiting for hash-valid parity report: {prerequisite}", flush=True)
        time.sleep(args.poll_seconds)
    source_total = 0
    for data in config["datasets"]:
        manifest = complete_manifest(root / data["clean_manifest"])
        if not manifest or len(manifest["records"]) != data["expected_images"]:
            raise SystemExit(f"TRANSFER PREP REFUSED: invalid clean manifest: {data['dataset']}")
        failures = validate_manifest(manifest, root / data["annotations"], root / data["clean_root"], root / data["clean_root"], require_pixels_changed=False)
        if failures:
            raise SystemExit(f"TRANSFER PREP REFUSED: clean manifest source-byte validation failed: {data['dataset']}\n" + "\n".join(failures[:20]))
        source_total += source_bytes(manifest, root / data["clean_root"])
    estimate = source_total * 13  # 12 JPEG cache conditions plus one 8% encoding/headroom margin.
    free = shutil.disk_usage(root).free
    reserve = int(config["minimum_free_gib_after_estimate"]) * 1024 ** 3
    if free - estimate < reserve:
        raise SystemExit(f"TRANSFER PREP REFUSED: free space {free} - conservative cache estimate {estimate} < reserve {reserve}")
    results = []
    for data in config["datasets"]:
        dataset, split = data["dataset"], data["split"]
        annotations, clean_root, cache_root = root / data["annotations"], root / data["clean_root"], root / data["corruption_root"]
        manifests = []
        for condition in matrix["conditions"]:
            corruption, severity = condition["corruption"], int(condition["severity"])
            if corruption == "clean":
                manifests.append({"condition": condition, "path": data["clean_manifest"], "sha256": complete_manifest(root / data["clean_manifest"])["manifest_sha256"]})
                continue
            relative = f"manifests/images/{dataset}_{split}_full_{corruption}_s{severity}.json"
            path = root / relative
            manifest = complete_manifest(path)
            if manifest is None:
                command = [sys.executable, str(root / "src" / "generate_corruption.py"), "--dataset", dataset, "--split", split,
                           "--annotations", str(annotations), "--clean-root", str(clean_root), "--cache-root", str(cache_root),
                           "--config", str(corruption_config), "--corruption", corruption, "--severity", str(severity),
                           "--manifest-out", str(path), "--resume-validated"]
                print("TRANSFER CACHE " + " ".join(command), flush=True)
                subprocess.run(command, check=True)
                manifest = complete_manifest(path)
            if manifest is None or len(manifest["records"]) != data["expected_images"]:
                raise SystemExit(f"TRANSFER PREP REFUSED: corrupt manifest is incomplete: {path}")
            failures = validate_manifest(manifest, annotations, cache_root, clean_root, require_pixels_changed=True)
            if failures:
                raise SystemExit(f"TRANSFER PREP REFUSED: corrupt manifest byte/pixel validation failed: {path}\n" + "\n".join(failures[:20]))
            manifests.append({"condition": condition, "path": relative, "sha256": manifest["manifest_sha256"]})
        proposal = root / "manifests" / "plans" / f"{dataset}_yolo11_nmx_117_proposal_v1.json"
        frozen = root / "manifests" / "plans" / f"{dataset}_yolo11_nmx_117_frozen_v1.json"
        engine_registry = root / "manifests" / "engines" / f"{dataset}_yolo11_nmx_ladder_v1.json"
        if not proposal.exists():
            template = f"manifests/images/{dataset}_{split}_full_{{corruption}}_s{{severity}}.json"
            subprocess.run([sys.executable, str(root / "src" / "build_dataset_pilot_plan.py"), "--matrix", str(matrix_path),
                            "--engine-registry", str(engine_registry), "--dataset", dataset, "--split", split,
                            "--clean-manifest", data["clean_manifest"], "--corruption-manifest-template", template,
                            "--calibration-list", str(root / data["calibration_list"]), "--out", str(proposal)], check=True)
        if not frozen.exists():
            subprocess.run([sys.executable, str(root / "src" / "freeze_pilot_plan.py"), "--proposal", str(proposal),
                            "--project-root", str(root), "--out", str(frozen)], check=True)
        plan = json.loads(frozen.read_text(encoding="utf-8"))
        if plan.get("plan_sha256") != canonical_hash(plan, "plan_sha256") or len(plan.get("runs", [])) != 117:
            raise SystemExit(f"TRANSFER PREP REFUSED: frozen plan invalid: {frozen}")
        results.append({"dataset": dataset, "cache_manifests": manifests, "proposal_sha256": sha256_file(proposal),
                        "frozen_plan": str(frozen), "frozen_plan_sha256": plan["plan_sha256"]})
    document = {"schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(), "config": str(config_path),
                "config_sha256": sha256_file(config_path), "parity_report": str(prerequisite), "parity_report_sha256": sha256_file(prerequisite),
                "source_bytes": source_total, "conservative_cache_estimate_bytes": estimate, "free_bytes_at_preflight": free,
                "reserve_bytes": reserve, "datasets": results}
    document["transfer_prep_report_sha256"] = canonical_hash(document, "transfer_prep_report_sha256")
    report.parent.mkdir(parents=True, exist_ok=True); report.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"TRANSFER PREP COMPLETE datasets={len(results)} report={report}")


if __name__ == "__main__":
    main()
