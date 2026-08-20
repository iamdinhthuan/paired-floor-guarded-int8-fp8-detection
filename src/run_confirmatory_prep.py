#!/usr/bin/env python3
"""Materialize confirmatory corruptions and freeze namespaced 117-run plans."""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import shutil
import subprocess
import sys
import time
from datetime import datetime, timezone
from pathlib import Path

from run_confirmatory_ladder import DATASETS, MODELS, canonical_hash, load_execution_config
from run_confirmatory_parity import load_parity_config
from topic_c.manifest import read_manifest, sha256_file, validate_manifest


SAFE = re.compile(r"^[a-z0-9][a-z0-9_-]*$")


def resolve_inside(root: Path, relative: object, label: str) -> Path:
    if not isinstance(relative, str) or not relative:
        raise SystemExit(f"CONFIRMATORY PREP REFUSED: missing {label}")
    root = root.resolve()
    path = (root / relative).resolve()
    if path != root and root not in path.parents:
        raise SystemExit(f"CONFIRMATORY PREP REFUSED: {label} escapes project root")
    return path


def corruption_manifest_path(
    root: Path, attempt: str, dataset: str, corruption: str, severity: int
) -> Path:
    if (
        not SAFE.fullmatch(attempt)
        or dataset not in DATASETS
        or corruption not in {"gaussian_noise", "motion_blur", "fog", "jpeg"}
        or severity not in {1, 3, 5}
    ):
        raise ValueError("invalid confirmatory corruption identity")
    return (
        root
        / "manifests"
        / "images"
        / attempt
        / f"{dataset}_final_{corruption}_s{severity}.json"
    )


def load_prep_config(root: Path, path: Path) -> dict:
    root, path = root.resolve(), path.resolve()
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("CONFIRMATORY PREP REFUSED: unreadable prep config") from exc
    if config.get("config_sha256") != canonical_hash(config, "config_sha256"):
        raise SystemExit("CONFIRMATORY PREP REFUSED: prep config hash mismatch")
    if not isinstance(config.get("attempt"), str) or not SAFE.fullmatch(config["attempt"]):
        raise SystemExit("CONFIRMATORY PREP REFUSED: unsafe attempt")
    for key, label in (
        ("execution_config", "execution config"),
        ("parity_config", "parity config"),
        ("matrix", "matrix"),
    ):
        bound = resolve_inside(root, config.get(key), label)
        expected = config.get(f"{key}_sha256")
        if not bound.is_file() or sha256_file(bound) != expected:
            raise SystemExit(f"CONFIRMATORY PREP REFUSED: {label} hash mismatch")
    reserve = config.get("minimum_free_gib_after_estimate")
    if not isinstance(reserve, int) or reserve < 10:
        raise SystemExit("CONFIRMATORY PREP REFUSED: invalid free-space reserve")
    manifest = config.get("source_manifest")
    if not isinstance(manifest, list) or not manifest:
        raise SystemExit("CONFIRMATORY PREP REFUSED: empty source manifest")
    seen: set[str] = set()
    for record in manifest:
        if not isinstance(record, dict) or set(record) != {"path", "sha256"} or record["path"] in seen:
            raise SystemExit("CONFIRMATORY PREP REFUSED: malformed source manifest")
        seen.add(record["path"])
        source = resolve_inside(root, record["path"], "source manifest path")
        if not source.is_file() or sha256_file(source) != record["sha256"]:
            raise SystemExit(f"CONFIRMATORY PREP REFUSED: source manifest hash mismatch: {record['path']}")
    return config


def validate_parity_completion(
    path: Path,
    *,
    attempt: str,
    parity_config_sha256: str,
    ladder_report_sha256: str,
) -> dict:
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("CONFIRMATORY PREP REFUSED: unreadable parity completion") from exc
    if document.get("parity_completion_sha256") != canonical_hash(
        document, "parity_completion_sha256"
    ):
        raise SystemExit("CONFIRMATORY PREP REFUSED: parity completion hash mismatch")
    if (
        document.get("attempt") != attempt
        or document.get("parity_config_sha256") != parity_config_sha256
        or document.get("ladder_report_sha256") != ladder_report_sha256
    ):
        raise SystemExit("CONFIRMATORY PREP REFUSED: parity completion binding mismatch")
    reports = document.get("reports")
    keys = {
        (row.get("dataset"), row.get("model"))
        for row in reports or []
        if isinstance(row, dict)
    }
    expected = {(dataset, model) for dataset in DATASETS for model in MODELS}
    if not isinstance(reports, list) or len(reports) != 6 or keys != expected:
        raise SystemExit("CONFIRMATORY PREP REFUSED: exact six parity reports required")
    for row in reports:
        nested = Path(row.get("report", "")).resolve()
        marker = nested.with_suffix(nested.suffix + ".complete")
        if (
            not nested.is_file()
            or not marker.is_file()
            or sha256_file(nested) != row.get("report_sha256")
            or marker.read_text(encoding="utf-8").strip() != sha256_file(nested)
        ):
            raise SystemExit("CONFIRMATORY PREP REFUSED: nested parity report hash mismatch")
        result = json.loads(nested.read_text(encoding="utf-8"))
        if not result.get("pass") or result.get("dataset") != row["dataset"] or result.get("model") != row["model"]:
            raise SystemExit("CONFIRMATORY PREP REFUSED: nested parity report failed or mismatched")
    return document


def validate_matrix(matrix: dict, execution: dict) -> None:
    if (
        matrix.get("expected_runs_per_dataset") != 117
        or tuple(matrix.get("models", ())) != MODELS
        or tuple(matrix.get("precisions", ())) != ("fp32", "int8-entropy", "fp8")
        or len(matrix.get("conditions", [])) != 13
        or tuple(matrix.get("dataset_protocols", {})) != DATASETS
    ):
        raise SystemExit("CONFIRMATORY PREP REFUSED: unexpected confirmatory matrix")
    for dataset in DATASETS:
        protocol = matrix["dataset_protocols"][dataset]
        data = execution.get("datasets", {}).get(dataset, {})
        if protocol.get("split") != data.get("split"):
            raise SystemExit(f"CONFIRMATORY PREP REFUSED: matrix machine split mismatch: {dataset}")
        if protocol.get("partition_role") != "final" or data.get("partition_role") != "final":
            raise SystemExit(f"CONFIRMATORY PREP REFUSED: final partition role missing: {dataset}")


def complete_manifest(path: Path) -> dict | None:
    if not path.is_file():
        return None
    try:
        document = read_manifest(path)
    except Exception:
        return None
    marker = path.with_suffix(path.suffix + ".complete")
    if not marker.is_file() or marker.read_text(encoding="utf-8").strip() != document.get("manifest_sha256"):
        return None
    return document


def run(command: list[str]) -> None:
    print("CONFIRMATORY PREP COMMAND " + " ".join(command), flush=True)
    subprocess.run(command, check=True)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--config", required=True)
    parser.add_argument("--wait-for-parity-report", required=True)
    parser.add_argument("--wait-for-ladder-report", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = Path(args.config).resolve()
    parity_completion_path = Path(args.wait_for_parity_report).resolve()
    ladder_path = Path(args.wait_for_ladder_report).resolve()
    report_out = Path(args.report_out).resolve()
    if report_out.exists() or report_out.with_suffix(report_out.suffix + ".complete").exists():
        raise SystemExit(f"CONFIRMATORY PREP REFUSED: completion report exists: {report_out}")
    config = load_prep_config(root, config_path)
    execution_path = resolve_inside(root, config["execution_config"], "execution config")
    parity_config_path = resolve_inside(root, config["parity_config"], "parity config")
    matrix_path = resolve_inside(root, config["matrix"], "matrix")
    execution, _ = load_execution_config(root, execution_path)
    parity_config = load_parity_config(root, parity_config_path)
    if execution["attempt"] != config["attempt"] or parity_config["attempt"] != config["attempt"]:
        raise SystemExit("CONFIRMATORY PREP REFUSED: attempt mismatch across frozen configs")
    matrix = json.loads(matrix_path.read_text(encoding="utf-8"))
    validate_matrix(matrix, execution)
    while not parity_completion_path.is_file() or not ladder_path.is_file():
        print("CONFIRMATORY PREP waiting for parity and ladder completion reports", flush=True)
        time.sleep(args.poll_seconds)
    parity_completion = validate_parity_completion(
        parity_completion_path,
        attempt=config["attempt"],
        parity_config_sha256=sha256_file(parity_config_path),
        ladder_report_sha256=sha256_file(ladder_path),
    )
    ladder = json.loads(ladder_path.read_text(encoding="utf-8"))
    if ladder.get("ladder_report_sha256") != canonical_hash(ladder, "ladder_report_sha256"):
        raise SystemExit("CONFIRMATORY PREP REFUSED: invalid ladder report")
    registry_by_dataset = {row["dataset"]: row for row in ladder["dataset_registries"]}

    source_bytes = 0
    clean_manifests: dict[str, dict] = {}
    for dataset in DATASETS:
        data = execution["datasets"][dataset]
        annotation = resolve_inside(root, data["annotation"], f"{dataset} annotation")
        clean_root = resolve_inside(root, data["clean_root"], f"{dataset} clean root")
        clean_path = resolve_inside(root, data["clean_manifest"], f"{dataset} clean manifest")
        manifest = complete_manifest(clean_path)
        if manifest is None or len(manifest.get("records", [])) != data["expected_images"]:
            raise SystemExit(f"CONFIRMATORY PREP REFUSED: invalid clean manifest: {dataset}")
        failures = validate_manifest(
            manifest, annotation, clean_root, clean_root, require_pixels_changed=False
        )
        if failures:
            raise SystemExit(
                f"CONFIRMATORY PREP REFUSED: clean source-byte validation failed: {dataset}\n"
                + "\n".join(failures[:20])
            )
        source_bytes += sum(
            (clean_root / record["source_relpath"]).stat().st_size
            for record in manifest["records"]
        )
        clean_manifests[dataset] = manifest
    estimate = source_bytes * 13
    free = shutil.disk_usage(root).free
    reserve = config["minimum_free_gib_after_estimate"] * 1024**3
    if free - estimate < reserve:
        raise SystemExit(
            f"CONFIRMATORY PREP REFUSED: free {free} - estimate {estimate} < reserve {reserve}"
        )

    results: list[dict] = []
    corruption_config = resolve_inside(
        root, execution["corruption_config"], "corruption config"
    )
    for dataset in DATASETS:
        data = execution["datasets"][dataset]
        annotation = resolve_inside(root, data["annotation"], f"{dataset} annotation")
        clean_root = resolve_inside(root, data["clean_root"], f"{dataset} clean root")
        cache_root = resolve_inside(root, data["corruption_root"], f"{dataset} corruption root")
        manifest_records = [
            {
                "corruption": "clean",
                "severity": 0,
                "path": data["clean_manifest"],
                "manifest_sha256": clean_manifests[dataset]["manifest_sha256"],
            }
        ]
        for condition in matrix["conditions"]:
            corruption, severity = condition["corruption"], int(condition["severity"])
            if corruption == "clean":
                continue
            manifest_path = corruption_manifest_path(
                root, config["attempt"], dataset, corruption, severity
            )
            manifest = complete_manifest(manifest_path)
            if manifest is None:
                if manifest_path.exists() or manifest_path.with_suffix(manifest_path.suffix + ".complete").exists():
                    raise SystemExit(f"CONFIRMATORY PREP REFUSED: partial corruption manifest: {manifest_path}")
                run(
                    [
                        sys.executable,
                        str(root / "src" / "generate_corruption.py"),
                        "--dataset",
                        dataset,
                        "--split",
                        data["split"],
                        "--annotations",
                        str(annotation),
                        "--clean-root",
                        str(clean_root),
                        "--cache-root",
                        str(cache_root),
                        "--config",
                        str(corruption_config),
                        "--corruption",
                        corruption,
                        "--severity",
                        str(severity),
                        "--manifest-out",
                        str(manifest_path),
                        "--resume-validated",
                    ]
                )
                manifest = complete_manifest(manifest_path)
            if manifest is None or len(manifest.get("records", [])) != data["expected_images"]:
                raise SystemExit(f"CONFIRMATORY PREP REFUSED: corruption manifest incomplete: {manifest_path}")
            failures = validate_manifest(
                manifest, annotation, cache_root, clean_root, require_pixels_changed=True
            )
            if failures:
                raise SystemExit(
                    f"CONFIRMATORY PREP REFUSED: corruption validation failed: {manifest_path}\n"
                    + "\n".join(failures[:20])
                )
            manifest_records.append(
                {
                    "corruption": corruption,
                    "severity": severity,
                    "path": str(manifest_path.relative_to(root)),
                    "manifest_sha256": manifest["manifest_sha256"],
                }
            )
        plan_root = root / "manifests" / "plans" / config["attempt"]
        proposal = plan_root / f"{dataset}_proposal.json"
        frozen = plan_root / f"{dataset}_frozen.json"
        engine_registry = Path(registry_by_dataset[dataset]["path"]).resolve()
        template = (
            Path("manifests")
            / "images"
            / config["attempt"]
            / f"{dataset}_final_{{corruption}}_s{{severity}}.json"
        ).as_posix()
        if not proposal.exists():
            run(
                [
                    sys.executable,
                    str(root / "src" / "build_dataset_pilot_plan.py"),
                    "--matrix",
                    str(matrix_path),
                    "--engine-registry",
                    str(engine_registry),
                    "--dataset",
                    dataset,
                    "--split",
                    data["split"],
                    "--clean-manifest",
                    data["clean_manifest"],
                    "--corruption-manifest-template",
                    template,
                    "--calibration-list",
                    str(resolve_inside(root, data["calibration"], f"{dataset} calibration")),
                    "--out",
                    str(proposal),
                ]
            )
        proposal_doc = json.loads(proposal.read_text(encoding="utf-8"))
        if proposal_doc.get("plan_sha256") != canonical_hash(proposal_doc, "plan_sha256"):
            raise SystemExit(f"CONFIRMATORY PREP REFUSED: invalid proposal: {proposal}")
        if not frozen.exists():
            run(
                [
                    sys.executable,
                    str(root / "src" / "freeze_pilot_plan.py"),
                    "--proposal",
                    str(proposal),
                    "--project-root",
                    str(root),
                    "--out",
                    str(frozen),
                ]
            )
        frozen_doc = json.loads(frozen.read_text(encoding="utf-8"))
        if (
            frozen_doc.get("plan_sha256") != canonical_hash(frozen_doc, "plan_sha256")
            or len(frozen_doc.get("runs", [])) != 117
        ):
            raise SystemExit(f"CONFIRMATORY PREP REFUSED: invalid frozen plan: {frozen}")
        results.append(
            {
                "dataset": dataset,
                "manifests": manifest_records,
                "proposal": str(proposal),
                "proposal_sha256": sha256_file(proposal),
                "frozen_plan": str(frozen),
                "frozen_plan_sha256": sha256_file(frozen),
                "plan_sha256": frozen_doc["plan_sha256"],
            }
        )
    document = {
        "schema_version": 1,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": config["attempt"],
        "config": str(config_path),
        "config_sha256": sha256_file(config_path),
        "execution_config_sha256": sha256_file(execution_path),
        "parity_completion": str(parity_completion_path),
        "parity_completion_sha256": sha256_file(parity_completion_path),
        "ladder_report": str(ladder_path),
        "ladder_report_sha256": sha256_file(ladder_path),
        "source_bytes": source_bytes,
        "conservative_cache_estimate_bytes": estimate,
        "free_bytes_at_preflight": free,
        "reserve_bytes": reserve,
        "datasets": results,
    }
    document["prep_report_sha256"] = canonical_hash(document, "prep_report_sha256")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    report_out.with_suffix(report_out.suffix + ".complete").write_text(
        sha256_file(report_out) + "\n", encoding="utf-8"
    )
    print(f"CONFIRMATORY PREP COMPLETE datasets=2 output={report_out}")


if __name__ == "__main__":
    main()
