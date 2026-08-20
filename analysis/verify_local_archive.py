#!/usr/bin/env python3
"""Verify that canonical scientific artifacts were mirrored into the project."""

from __future__ import annotations

import argparse
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path


DEFAULT_ROOT = Path("artifacts/four_dataset_pilot_v1")
RUN_DIRS = (
    "pilot_117_v1",
    "voc_pilot_117_v1",
    "kitti_pilot_117_v1",
    "tt100k_pilot_117_v1",
)
REMOTE_TOPIC_ENGINES = Path("/home/thuan/topic_c_ivc/engines")
REMOTE_TOPIC_ROOT = Path("/home/thuan/topic_c_ivc")
REMOTE_COCO_ROOT = Path("/home/thuan/coco_journal")


def sha256(path: Path, cache: dict[Path, str]) -> str:
    if path not in cache:
        digest = hashlib.sha256()
        with path.open("rb") as handle:
            for chunk in iter(lambda: handle.read(8 * 1024 * 1024), b""):
                digest.update(chunk)
        cache[path] = digest.hexdigest()
    return cache[path]


def canonical_manifest_sha256(document: dict[str, object]) -> str:
    payload = {key: value for key, value in document.items() if key != "manifest_sha256"}
    encoded = json.dumps(
        payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True
    ).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def local_engine_path(root: Path, remote: str) -> Path:
    path = Path(remote)
    try:
        return root / "engines" / path.relative_to(REMOTE_TOPIC_ENGINES)
    except ValueError:
        pass
    try:
        return root / "legacy_coco" / path.relative_to(REMOTE_COCO_ROOT)
    except ValueError as exc:
        raise ValueError(f"unmapped engine path: {remote}") from exc


def verify_file(
    path: Path,
    expected_sha256: str,
    kind: str,
    failures: list[dict[str, str]],
    cache: dict[Path, str],
) -> None:
    if not path.is_file():
        failures.append({"kind": kind, "path": str(path), "error": "missing"})
        return
    actual = sha256(path, cache)
    if actual != expected_sha256:
        failures.append(
            {
                "kind": kind,
                "path": str(path),
                "error": "sha256_mismatch",
                "expected": expected_sha256,
                "actual": actual,
            }
        )


def build_verification(root: Path) -> dict[str, object]:
    failures: list[dict[str, str]] = []
    cache: dict[Path, str] = {}
    run_records = 0
    engines: set[tuple[Path, str]] = set()
    referenced_manifest_hashes: set[str] = set()
    image_manifests: dict[str, Path] = {}
    for path in sorted((root / "manifests" / "images").glob("*.json")):
        record = json.loads(path.read_text(encoding="utf-8"))
        stored = record.get("manifest_sha256")
        if stored:
            actual = canonical_manifest_sha256(record)
            if actual != stored:
                failures.append(
                    {
                        "kind": "source_image_manifest",
                        "path": str(path),
                        "error": "canonical_sha256_mismatch",
                        "expected": str(stored),
                        "actual": actual,
                    }
                )
            image_manifests[str(stored)] = path

    for run_dir in RUN_DIRS:
        manifest_dir = root / "manifests" / "runs" / run_dir
        manifests = sorted(manifest_dir.glob("*.json"))
        if len(manifests) != 117:
            failures.append(
                {
                    "kind": "run_directory",
                    "path": str(manifest_dir),
                    "error": f"expected_117_found_{len(manifests)}",
                }
            )
        for manifest_path in manifests:
            record = json.loads(manifest_path.read_text(encoding="utf-8"))
            run_records += 1
            referenced_manifest_hashes.add(record["input_manifest_sha256"])
            condition_id = record["condition_id"]
            prediction = root / "outputs" / "predictions" / run_dir / f"{condition_id}.json"
            input_manifest = root / "outputs" / "inputs" / run_dir / f"{condition_id}.json"
            verify_file(
                prediction,
                record["prediction_sha256"],
                "prediction",
                failures,
                cache,
            )
            if not input_manifest.is_file():
                failures.append(
                    {"kind": "input_sidecar", "path": str(input_manifest), "error": "missing"}
                )
            else:
                sidecar = json.loads(input_manifest.read_text(encoding="utf-8"))
                actual_ids_sha256 = hashlib.sha256(
                    json.dumps(sidecar.get("image_ids", []), separators=(",", ":")).encode()
                ).hexdigest()
                expected_fields = {
                    "condition_id": condition_id,
                    "input_manifest_sha256": record["input_manifest_sha256"],
                    "image_ids_sha256": record["input_image_ids_sha256"],
                }
                for key, expected in expected_fields.items():
                    if sidecar.get(key) != expected:
                        failures.append(
                            {
                                "kind": "input_sidecar",
                                "path": str(input_manifest),
                                "error": f"{key}_mismatch",
                                "expected": str(expected),
                                "actual": str(sidecar.get(key)),
                            }
                        )
                if actual_ids_sha256 != record["input_image_ids_sha256"]:
                    failures.append(
                        {
                            "kind": "input_sidecar",
                            "path": str(input_manifest),
                            "error": "ordered_image_ids_sha256_mismatch",
                            "expected": record["input_image_ids_sha256"],
                            "actual": actual_ids_sha256,
                        }
                    )
                if record["input_manifest_sha256"] not in image_manifests:
                    failures.append(
                        {
                            "kind": "source_image_manifest",
                            "path": str(input_manifest),
                            "error": "referenced_manifest_not_found",
                            "expected": record["input_manifest_sha256"],
                        }
                    )
            try:
                engine = local_engine_path(root, record["engine_path"])
            except ValueError as exc:
                failures.append(
                    {
                        "kind": "engine",
                        "path": record["engine_path"],
                        "error": str(exc),
                    }
                )
            else:
                engines.add((engine, record["engine_sha256"]))

    for engine, expected in sorted(engines, key=lambda item: str(item[0])):
        verify_file(engine, expected, "engine", failures, cache)

    registry_path = root / "manifests" / "engines" / "coco_yolo11_nmx_pilot.json"
    registry = json.loads(registry_path.read_text(encoding="utf-8"))
    legacy_records = [registry["calibration"]["list"], *registry["evidence"]]
    for record in legacy_records:
        remote = Path(record["path"])
        try:
            local = root / "legacy_coco" / remote.relative_to(REMOTE_COCO_ROOT)
        except ValueError:
            failures.append(
                {"kind": "legacy_coco", "path": str(remote), "error": "unmapped_path"}
            )
            continue
        verify_file(local, record["sha256"], "legacy_coco", failures, cache)

    pretrained: set[tuple[Path, str]] = set()
    selected_checkpoints: set[tuple[Path, str]] = set()
    for manifest_path in sorted((root / "manifests" / "training").glob("*_train_v1.json")):
        record = json.loads(manifest_path.read_text(encoding="utf-8"))
        remote = Path(record["pretrained_weights"])
        pretrained.add((root / "pretrained" / remote.name, record["pretrained_weights_sha256"]))
        for path_key, hash_key in (
            ("best_weights", "best_weights_sha256"),
            ("last_weights", "last_weights_sha256"),
        ):
            remote_checkpoint = Path(record[path_key])
            try:
                local_checkpoint = root / remote_checkpoint.relative_to(REMOTE_TOPIC_ROOT)
            except ValueError:
                failures.append(
                    {
                        "kind": "selected_checkpoint",
                        "path": str(remote_checkpoint),
                        "error": "unmapped_path",
                    }
                )
            else:
                selected_checkpoints.add((local_checkpoint, record[hash_key]))
    for checkpoint, expected in sorted(pretrained, key=lambda item: str(item[0])):
        verify_file(checkpoint, expected, "pretrained_checkpoint", failures, cache)
    for checkpoint, expected in sorted(selected_checkpoints, key=lambda item: str(item[0])):
        verify_file(checkpoint, expected, "selected_checkpoint", failures, cache)

    return {
        "schema": "topic-c-local-archive-verification-v1",
        "generated_at_utc": datetime.now(timezone.utc).isoformat(),
        "root": str(root.resolve()),
        "canonical_run_records": run_records,
        "canonical_predictions_verified": run_records,
        "canonical_input_sidecars_validated": run_records,
        "unique_source_image_manifests_verified_available": len(referenced_manifest_hashes),
        "unique_scientific_engines_verified": len(engines),
        "legacy_coco_provenance_files_verified": len(legacy_records),
        "pretrained_checkpoints_verified": len(pretrained),
        "selected_training_checkpoints_verified": len(selected_checkpoints),
        "hashed_bytes": sum(path.stat().st_size for path in cache),
        "status": "pass" if not failures else "fail",
        "failures": failures,
    }


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=DEFAULT_ROOT)
    parser.add_argument("--out", type=Path)
    args = parser.parse_args()
    output = args.out or args.root / "local_archive_verification.json"
    report = build_verification(args.root)
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps(report, indent=2))
    if report["status"] != "pass":
        raise SystemExit(1)


if __name__ == "__main__":
    main()
