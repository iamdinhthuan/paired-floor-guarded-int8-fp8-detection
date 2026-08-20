#!/usr/bin/env python3
"""Run a resumable B=2,000 paired bootstrap for corruption realizations."""
from __future__ import annotations

import argparse
import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

from bootstrap_format_contrast import (
    component_draw_cache_identity,
    create_clean_arm_cache,
    load_bootstrap_schedule,
    load_clean_arm_cache,
    load_draw_cache,
    materialize_bootstrap_schedule,
    run_contrast,
    validate_linked_arm_bindings,
)
from pilot_registry import canonical_hash
from run_corruption_realization_sensitivity import (
    CORRUPTIONS,
    DATASETS,
    PRECISIONS,
    SEVERITIES,
    condition_paths,
    load_config as load_realization_config,
    load_subset,
)
from topic_c.manifest import sha256_file


def dataset_seed(namespace: str, dataset: str) -> int:
    if not namespace or dataset not in DATASETS:
        raise ValueError("invalid bootstrap seed namespace or dataset")
    return int.from_bytes(hashlib.sha256(f"{namespace}|{dataset}".encode()).digest()[:8], "big") % (2**32)


def planned_cells(realization_seeds: tuple[int, ...]) -> list[dict]:
    if len(realization_seeds) != 3 or len(set(realization_seeds)) != 3:
        raise ValueError("realization design requires three unique seeds")
    return [
        {
            "dataset": dataset,
            "realization_seed": seed,
            "corruption": corruption,
            "severity": severity,
        }
        for dataset in DATASETS
        for seed in realization_seeds
        for corruption in CORRUPTIONS
        for severity in SEVERITIES
    ]


def cell_identity(cell: dict) -> str:
    return (
        f"{cell['dataset']}__r{cell['realization_seed']}__yolo11m__"
        f"{cell['corruption']}-s{cell['severity']}"
    )


def _condition_task(cell: dict, precision: str, *, clean: bool) -> dict:
    return {
        **cell,
        "realization_seed": None if clean else cell["realization_seed"],
        "corruption": "clean" if clean else cell["corruption"],
        "severity": 0 if clean else cell["severity"],
        "precision": precision,
    }


def _source_paths(root: Path, cell: dict) -> tuple[list[Path], list[Path], list[Path]]:
    tasks = [
        _condition_task(cell, "int8-entropy", clean=True),
        _condition_task(cell, "fp8", clean=True),
        _condition_task(cell, "int8-entropy", clean=False),
        _condition_task(cell, "fp8", clean=False),
    ]
    paths = [condition_paths(root, task)[:3] for task in tasks]
    return ([row[0] for row in paths], [row[1] for row in paths], [row[2] for row in paths])


def _validate_upstream(root: Path, path: Path, config: dict) -> dict:
    marker = path.with_suffix(path.suffix + ".complete")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("REALIZATION BOOTSTRAP REFUSED: unreadable upstream completion") from exc
    if (
        not marker.is_file()
        or marker.read_text(encoding="utf-8").strip() != sha256_file(path)
        or report.get("completion_sha256") != canonical_hash(report, "completion_sha256")
        or report.get("attempt") != config["attempt"]
        or report.get("realization_seeds") != config["realization_seeds"]
        or len(report.get("conditions", [])) != 222
    ):
        raise SystemExit("REALIZATION BOOTSTRAP REFUSED: invalid upstream completion")
    records = {row.get("condition_id"): row for row in report["conditions"]}
    if len(records) != 222:
        raise SystemExit("REALIZATION BOOTSTRAP REFUSED: duplicate upstream condition")
    return report


def _verify_condition_files(root: Path, upstream: dict, cells: list[dict]) -> None:
    records = {row["condition_id"]: row for row in upstream["conditions"]}
    for cell in cells:
        for precision in PRECISIONS:
            for clean in (True, False):
                task = _condition_task(cell, precision, clean=clean)
                from run_corruption_realization_sensitivity import task_identity

                identity = task_identity(task)
                prediction, inputs, run, metric = condition_paths(root, task)
                record = records.get(identity)
                if not isinstance(record, dict) or any(
                    not path.is_file() for path in (prediction, inputs, run, metric)
                ):
                    raise SystemExit(f"REALIZATION BOOTSTRAP REFUSED: missing condition {identity}")
                expected = {
                    prediction: record.get("prediction_sha256"),
                    inputs: record.get("input_record_sha256"),
                    run: record.get("run_record_sha256"),
                    metric: record.get("metric_sha256"),
                }
                if any(sha256_file(path) != digest for path, digest in expected.items()):
                    raise SystemExit(f"REALIZATION BOOTSTRAP REFUSED: condition hash mismatch {identity}")


def _load_config(root: Path, path: Path) -> dict:
    try:
        config = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError) as exc:
        raise SystemExit("REALIZATION BOOTSTRAP REFUSED: unreadable bootstrap config") from exc
    if (
        config.get("config_sha256") != canonical_hash(config, "config_sha256")
        or config.get("attempt") != "corruption_realization_bootstrap_v1"
        or config.get("n_boot") != 2000
        or type(config.get("workers")) is not int
        or not 1 <= config["workers"] <= 4
        or not isinstance(config.get("seed_namespace"), str)
        or not config["seed_namespace"]
    ):
        raise SystemExit("REALIZATION BOOTSTRAP REFUSED: invalid bootstrap config")
    manifest = config.get("source_manifest")
    if not isinstance(manifest, list) or not manifest:
        raise SystemExit("REALIZATION BOOTSTRAP REFUSED: empty source manifest")
    seen = set()
    for record in manifest:
        relative = record.get("path") if isinstance(record, dict) else None
        try:
            source = (root / relative).resolve()
            source.relative_to(root.resolve())
        except (TypeError, ValueError) as exc:
            raise SystemExit("REALIZATION BOOTSTRAP REFUSED: invalid source path") from exc
        if relative in seen or not source.is_file() or sha256_file(source) != record.get("sha256"):
            raise SystemExit(f"REALIZATION BOOTSTRAP REFUSED: source hash mismatch: {relative}")
        seen.add(relative)
    return config


def _paths(root: Path, dataset: str, stem: str, suffix: str) -> Path:
    return root / "outputs" / "bootstrap" / "corruption_realization_v1" / suffix / f"{stem}.{suffix if suffix != 'components' else 'json'}"


def _load_or_create_schedule(root: Path, dataset: str, seed: int, n_boot: int, cell: dict) -> dict:
    clean_input = condition_paths(root, _condition_task(cell, "int8-entropy", clean=True))[1]
    image_ids = json.loads(clean_input.read_text(encoding="utf-8"))["image_ids"]
    path = root / "outputs" / "bootstrap" / "corruption_realization_v1" / "schedules" / f"{dataset}.npz"
    if not path.exists():
        return materialize_bootstrap_schedule(path, image_ids, n_boot=n_boot, seed=seed)
    return load_bootstrap_schedule(path, image_ids, n_boot=n_boot, seed=seed)


def _clean_cache(
    root: Path, dataset: str, row: dict, subset: dict, schedule: dict, cell: dict, *, n_boot: int, seed: int, workers: int
) -> dict:
    path = root / "outputs" / "bootstrap" / "corruption_realization_v1" / "clean_caches" / f"{dataset}.npz"
    predictions, inputs, runs = _source_paths(root, cell)
    annotation_sha = sha256_file(subset["annotation"])
    if not path.exists():
        return create_clean_arm_cache(
            endpoint="tt100k-height" if dataset == "tt100k" else "area",
            annotations=subset["annotation"], prediction_paths=predictions[:2], input_paths=inputs[:2], run_paths=runs[:2],
            dataset=dataset, model="yolo11m", expected_images=512, annotation_sha256=annotation_sha,
            schedule_path=Path(schedule["path"]), n_boot=n_boot, seed=seed, workers=workers, output=path,
        )
    _, bindings = validate_linked_arm_bindings(
        prediction_paths=predictions[:2], input_paths=inputs[:2], run_paths=runs[:2], expected_images=512,
        arm_names=("int8_clean", "fp8_clean"),
    )
    return load_clean_arm_cache(
        path, endpoint="tt100k-height" if dataset == "tt100k" else "area", expected_images=512,
        annotation_sha256=annotation_sha, schedule=schedule, expected_dataset=dataset, expected_model="yolo11m",
        expected_input_hashes=bindings,
    )


def _verify_component(root: Path, output: Path, draw_cache: Path, *, n_boot: int, schedule: dict) -> dict:
    document = json.loads(output.read_text(encoding="utf-8"))
    reference = document.get("temporary_draw_cache", {})
    if (
        document.get("artifact_sha256") != canonical_hash(document, "artifact_sha256")
        or document.get("n_boot") != n_boot
        or document.get("bootstrap_schedule", {}).get("sha256") != schedule["sha256"]
        or reference.get("sha256") != sha256_file(draw_cache)
    ):
        raise ValueError(f"invalid realization bootstrap component: {output}")
    load_draw_cache(
        draw_cache, expected_sha256=reference["sha256"],
        expected_identity_sha256=component_draw_cache_identity(document), n_boot=n_boot,
        schedule_sha256=schedule["sha256"],
    )
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--realization-config", required=True)
    parser.add_argument("--bootstrap-config", required=True)
    parser.add_argument("--wait-for-realization-report", required=True)
    parser.add_argument("--report-out", required=True)
    parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    realization_config_path = Path(args.realization_config).resolve()
    bootstrap_config_path = Path(args.bootstrap_config).resolve()
    prerequisite = Path(args.wait_for_realization_report).resolve()
    report_out = Path(args.report_out).resolve()
    if report_out.exists() or report_out.with_suffix(report_out.suffix + ".complete").exists():
        raise SystemExit("REALIZATION BOOTSTRAP REFUSED: completion report exists")
    realization_config = load_realization_config(root, realization_config_path)
    bootstrap_config = _load_config(root, bootstrap_config_path)
    while not prerequisite.is_file():
        print("REALIZATION BOOTSTRAP waiting for realization inference", flush=True)
        time.sleep(args.poll_seconds)
    upstream = _validate_upstream(root, prerequisite, realization_config)
    seeds = tuple(realization_config["realization_seeds"])
    cells = planned_cells(seeds)
    _verify_condition_files(root, upstream, cells)
    rows = {row["dataset"]: row for row in realization_config["datasets"]}
    subsets = {dataset: load_subset(root, rows[dataset]) for dataset in DATASETS}
    schedules = {}
    clean_caches = {}
    for dataset in DATASETS:
        example = next(cell for cell in cells if cell["dataset"] == dataset)
        seed = dataset_seed(bootstrap_config["seed_namespace"], dataset)
        schedules[dataset] = _load_or_create_schedule(root, dataset, seed, bootstrap_config["n_boot"], example)
        clean_caches[dataset] = _clean_cache(
            root, dataset, rows[dataset], subsets[dataset], schedules[dataset], example,
            n_boot=bootstrap_config["n_boot"], seed=seed, workers=bootstrap_config["workers"],
        )
    artifacts = []
    for index, cell in enumerate(cells, start=1):
        dataset = cell["dataset"]
        stem = cell_identity(cell)
        base = root / "outputs" / "bootstrap" / "corruption_realization_v1"
        output = base / "components" / f"{stem}.json"
        draw_cache = base / "draw_caches" / f"{stem}.npz"
        schedule = schedules[dataset]
        if not output.exists() and not draw_cache.exists():
            predictions, inputs, runs = _source_paths(root, cell)
            clean = clean_caches[dataset]
            print(f"REALIZATION BOOTSTRAP CELL {index}/108 {stem}", flush=True)
            run_contrast(
                endpoint="tt100k-height" if dataset == "tt100k" else "area",
                annotations=subsets[dataset]["annotation"], prediction_paths=predictions, input_paths=inputs, run_paths=runs,
                n_boot=bootstrap_config["n_boot"], seed=dataset_seed(bootstrap_config["seed_namespace"], dataset),
                expected_images=512, annotation_sha256=sha256_file(subsets[dataset]["annotation"]), output=output,
                workers=bootstrap_config["workers"], draw_cache=draw_cache, schedule_path=Path(schedule["path"]),
                clean_arm_cache_path=Path(clean["path"]), clean_arm_cache_sha256=clean["sha256"],
                clean_arm_cache_identity_sha256=clean["cache_identity_sha256"], dataset=dataset, model="yolo11m",
                evidence_root=root,
            )
        elif not output.exists() or not draw_cache.exists():
            raise SystemExit(f"REALIZATION BOOTSTRAP REFUSED: partial cell {stem}")
        _verify_component(root, output, draw_cache, n_boot=bootstrap_config["n_boot"], schedule=schedule)
        artifacts.append({
            **cell, "component": str(output), "component_sha256": sha256_file(output),
            "draw_cache": str(draw_cache), "draw_cache_sha256": sha256_file(draw_cache),
        })
    report = {
        "schema_version": 1, "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "attempt": bootstrap_config["attempt"], "n_boot": bootstrap_config["n_boot"],
        "seed_namespace": bootstrap_config["seed_namespace"], "workers": bootstrap_config["workers"],
        "realization_config": str(realization_config_path), "realization_config_sha256": sha256_file(realization_config_path),
        "bootstrap_config": str(bootstrap_config_path), "bootstrap_config_sha256": sha256_file(bootstrap_config_path),
        "upstream_report": str(prerequisite), "upstream_report_sha256": sha256_file(prerequisite),
        "realization_seeds": list(seeds),
        "schedules": [{"dataset": d, **{k: v for k, v in schedules[d].items() if k != "samples"}} for d in DATASETS],
        "clean_caches": [{"dataset": d, "path": clean_caches[d]["path"], "sha256": clean_caches[d]["sha256"], "identity_sha256": clean_caches[d]["cache_identity_sha256"]} for d in DATASETS],
        "artifacts": artifacts,
    }
    report["completion_sha256"] = canonical_hash(report, "completion_sha256")
    report_out.parent.mkdir(parents=True, exist_ok=True)
    report_out.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    report_out.with_suffix(report_out.suffix + ".complete").write_text(sha256_file(report_out) + "\n", encoding="utf-8")
    print(f"REALIZATION BOOTSTRAP COMPLETE cells=108 output={report_out}", flush=True)


if __name__ == "__main__":
    main()
