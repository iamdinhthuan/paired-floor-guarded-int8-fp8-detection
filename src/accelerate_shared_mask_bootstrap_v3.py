#!/usr/bin/env python3
"""Exact, resumable accelerator for the V3 RT-DETR bootstrap bottleneck.

This is an analysis-only addendum.  It deliberately does not replace or edit
the frozen V3 analyzer.  It writes the same per-cell NPZ/JSON cache schema that
``analyze_shared_mask_pilot.py`` validates, after applying two algebraically
and numerically exact reductions:

* only overall COCO AP is accumulated because the V3 analyzer immediately
  selects area index zero; and
* the default/shared FP8 evaluation is computed once, but only after exact
  resolved-path and SHA-256 identity checks.

Clean bootstrap draws are published atomically once.  The twelve RT-DETR
corruption cells are then independent and may be evaluated by at most three
workers.  Existing complete cell caches remain immutable and are only reused
after validation by the original V3 cache validator.
"""

from __future__ import annotations

import argparse
from concurrent.futures import FIRST_COMPLETED, ProcessPoolExecutor, wait
from datetime import datetime, timezone
import fcntl
import gc
import hashlib
import json
import math
import os
from pathlib import Path
import shutil
from typing import Any, Iterable

import numpy as np


EXPECTED_ATTEMPT = "shared_mask_pilot_v3"
TARGET_BLOCK = "voc_rtdetr_l"
MAX_WORKERS = 3
ACCELERATOR_VERSION = 1
ARM_ORDER_UNIQUE = ("default_int8", "fp8", "shared_int8")
CELL_ARRAY_FIELDS = {
    "schema_version",
    "n_boot",
    "seed",
    "delta_e_default",
    "delta_e_shared",
    "omega",
}
CLEAN_ARRAY_FIELDS = {
    "schema_version",
    "n_boot",
    "seed",
    "n_images",
    "default_int8",
    "fp8",
    "shared_int8",
}


class AcceleratorError(RuntimeError):
    """The accelerator cannot preserve the frozen scientific contract."""


def utc_now() -> str:
    return datetime.now(timezone.utc).isoformat()


def _project_modules() -> tuple[Any, Any, Any]:
    """Load attempt-sensitive project modules only in an executing process."""
    observed = os.environ.get("SHARED_MASK_PILOT_ATTEMPT")
    if observed not in (None, EXPECTED_ATTEMPT):
        raise AcceleratorError(
            f"conflicting SHARED_MASK_PILOT_ATTEMPT={observed}; expected {EXPECTED_ATTEMPT}"
        )
    os.environ["SHARED_MASK_PILOT_ATTEMPT"] = EXPECTED_ATTEMPT
    import analyze_shared_mask_pilot as legacy
    import paired_bootstrap as paired
    from topic_c import shared_mask_pilot as contract

    if legacy.ATTEMPT != EXPECTED_ATTEMPT or contract.ATTEMPT != EXPECTED_ATTEMPT:
        raise AcceleratorError(
            "attempt-sensitive modules were imported before V3 was selected; use a fresh process"
        )
    return legacy, paired, contract


def validate_worker_count(workers: int) -> int:
    if not 1 <= int(workers) <= MAX_WORKERS:
        raise AcceleratorError(f"workers must be in 1..{MAX_WORKERS}")
    return int(workers)


def accumulate_ap_overall(evaluation: Any, image_positions: list[int]) -> float:
    """Return area-index-zero AP with the legacy operation order unchanged.

    This is the area-zero branch of ``paired_bootstrap.accumulate_ap`` copied
    line for line.  Category, detection, IoU-threshold and recall-threshold
    ordering are unchanged.  The omitted area branches cannot affect area zero.
    """
    params = evaluation.params
    thresholds, recalls = len(params.iouThrs), len(params.recThrs)
    categories = len(params.catIds)
    areas = len(params.areaRng)
    n_images = len(evaluation._paramsEval.imgIds)
    precision = -np.ones((thresholds, recalls, categories))
    area = 0
    for category in range(categories):
        base_category = category * areas * n_images
        base_area = base_category + area * n_images
        entries = [
            evaluation.evalImgs[base_area + position] for position in image_positions
        ]
        entries = [entry for entry in entries if entry is not None]
        if not entries:
            continue
        scores = np.concatenate([entry["dtScores"][:100] for entry in entries])
        order = np.argsort(-scores, kind="mergesort")
        matches = np.concatenate(
            [entry["dtMatches"][:, :100] for entry in entries], axis=1
        )[:, order]
        ignored_detections = np.concatenate(
            [entry["dtIgnore"][:, :100] for entry in entries], axis=1
        )[:, order]
        ignored_gt = np.concatenate([entry["gtIgnore"] for entry in entries])
        positives = np.count_nonzero(ignored_gt == 0)
        if positives == 0:
            continue
        true_positive = np.cumsum(
            np.logical_and(matches, np.logical_not(ignored_detections)), axis=1
        )
        false_positive = np.cumsum(
            np.logical_and(np.logical_not(matches), np.logical_not(ignored_detections)),
            axis=1,
        )
        for threshold in range(thresholds):
            tp, fp = true_positive[threshold], false_positive[threshold]
            recall = tp / positives
            curve = tp / (tp + fp + np.spacing(1))
            curve = np.maximum.accumulate(curve[::-1])[::-1]
            sampled = np.zeros(recalls)
            locations = np.searchsorted(recall, params.recThrs, side="left")
            valid = locations < len(curve)
            sampled[valid] = curve[locations[valid]]
            precision[threshold, :, category] = sampled
    values = precision[precision > -1]
    return float(values.mean()) if len(values) else float("nan")


def contrast_values(
    *,
    default_int8_clean: float,
    fp8_clean: float,
    shared_int8_clean: float,
    default_int8_corrupt: float,
    fp8_corrupt: float,
    shared_int8_corrupt: float,
) -> dict[str, float]:
    """Mirror the legacy arithmetic, including AP-point scaling order."""
    values = (
        default_int8_clean,
        fp8_clean,
        shared_int8_clean,
        default_int8_corrupt,
        fp8_corrupt,
        shared_int8_corrupt,
    )
    if not all(math.isfinite(float(value)) for value in values):
        raise AcceleratorError("contrast inputs must be finite")
    default_clean_gap = fp8_clean - default_int8_clean
    default_corrupt_gap = fp8_corrupt - default_int8_corrupt
    shared_clean_gap = fp8_clean - shared_int8_clean
    shared_corrupt_gap = fp8_corrupt - shared_int8_corrupt
    delta_default = default_corrupt_gap - default_clean_gap
    delta_shared = shared_corrupt_gap - shared_clean_gap
    return {
        "clean_gap_default": float(default_clean_gap),
        "corrupt_gap_default": float(default_corrupt_gap),
        "delta_e_default": float(delta_default),
        "clean_gap_shared": float(shared_clean_gap),
        "corrupt_gap_shared": float(shared_corrupt_gap),
        "delta_e_shared": float(delta_shared),
        "omega_default_minus_shared": float(delta_default - delta_shared),
    }


def fp8_identity_gate(records: dict[str, dict[str, Any]], label: str) -> dict[str, Any]:
    """Admit a single FP8 evaluation only after exact path and digest identity."""
    try:
        default = records["default_fp8"]
        shared = records["shared_fp8"]
    except KeyError as exc:
        raise AcceleratorError(f"{label}: missing FP8 arm") from exc
    path_fields = ("prediction", "input_record", "run_record", "metric")
    hash_fields = (
        "prediction_sha256",
        "input_record_sha256",
        "run_record_sha256",
        "metric_sha256",
        "input_manifest_sha256",
        "input_image_ids_sha256",
    )
    for field in path_fields:
        default_path = Path(default.get(field, "")).resolve()
        shared_path = Path(shared.get(field, "")).resolve()
        if default_path != shared_path:
            raise AcceleratorError(
                f"{label}: default/shared FP8 {field} paths differ: "
                f"{default_path} != {shared_path}"
            )
        if not default_path.is_file():
            raise AcceleratorError(f"{label}: FP8 {field} is absent: {default_path}")
        hash_field = f"{field}_sha256"
        observed_sha = sha256_file(default_path)
        expected_sha = (
            default["input_record_sha256"]
            if field == "input_record"
            else default.get(hash_field)
        )
        shared_sha = (
            shared["input_record_sha256"]
            if field == "input_record"
            else shared.get(hash_field)
        )
        if observed_sha != expected_sha or observed_sha != shared_sha:
            raise AcceleratorError(
                f"{label}: FP8 {field} bytes do not match both declared SHA-256 values"
            )
    for field in hash_fields:
        left, right = default.get(field), shared.get(field)
        if not isinstance(left, str) or left != right:
            raise AcceleratorError(f"{label}: default/shared FP8 {field} differs")
    if default.get("image_ids") != shared.get("image_ids"):
        raise AcceleratorError(f"{label}: default/shared FP8 ordered image IDs differ")
    default_ap = float(default.get("ap", float("nan")))
    shared_ap = float(shared.get("ap", float("nan")))
    if (
        not math.isfinite(default_ap)
        or not math.isfinite(shared_ap)
        or default_ap.hex() != shared_ap.hex()
    ):
        raise AcceleratorError(f"{label}: default/shared FP8 recorded AP differs")
    return {
        "prediction": str(Path(default["prediction"]).resolve()),
        "prediction_sha256": default["prediction_sha256"],
        "input_record": str(Path(default["input_record"]).resolve()),
        "input_record_sha256": default["input_record_sha256"],
        "run_record": str(Path(default["run_record"]).resolve()),
        "run_record_sha256": default["run_record_sha256"],
        "metric": str(Path(default["metric"]).resolve()),
        "metric_sha256": default["metric_sha256"],
    }


def unique_arms(
    records: dict[str, dict[str, Any]],
    label: str,
    *,
    identity_checked: bool = False,
) -> dict[str, dict[str, Any]]:
    if not identity_checked:
        fp8_identity_gate(records, label)
    return {
        "default_int8": records["default_int8"],
        "fp8": records["default_fp8"],
        "shared_int8": records["shared_int8"],
    }


def make_samples(seed: int, n_boot: int, n_images: int) -> np.ndarray:
    rng = np.random.default_rng(int(seed))
    return rng.integers(
        0, int(n_images), size=(int(n_boot), int(n_images)), dtype=np.int32
    )


def samples_sha256(samples: np.ndarray) -> str:
    value = np.ascontiguousarray(samples, dtype=np.int32)
    return hashlib.sha256(value.tobytes(order="C")).hexdigest()


def percentile(values: np.ndarray) -> list[float]:
    finite = np.asarray(values, dtype=float)
    finite = finite[np.isfinite(finite)]
    if not len(finite):
        return [float("nan")] * 3
    return [float(value) for value in np.percentile(finite, [2.5, 50.0, 97.5])]


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def _atomic_npz(path: Path, **arrays: np.ndarray) -> None:
    if path.exists():
        raise AcceleratorError(f"refusing to overwrite immutable array cache: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    temporary = path.with_name(f".{path.name}.{os.getpid()}.tmp.npz")
    try:
        np.savez_compressed(temporary, **arrays)
        os.replace(temporary, path)
    finally:
        if temporary.exists():
            temporary.unlink()


def _quarantine_partial(root: Path, paths: Iterable[Path], label: str) -> None:
    existing = [path for path in paths if path.exists()]
    if not existing:
        return
    stamp = datetime.now(timezone.utc).strftime("%Y%m%dT%H%M%S%fZ")
    directory = (
        root
        / "outputs"
        / "quarantine"
        / EXPECTED_ATTEMPT
        / f"{stamp}__bootstrap_accelerator__{label}"
    )
    directory.mkdir(parents=True, exist_ok=False)
    for ordinal, source in enumerate(existing):
        shutil.move(str(source), str(directory / f"{ordinal:02d}__{source.name}"))


def _partial_paths(npz_path: Path, record_path: Path) -> list[Path]:
    marker = record_path.with_suffix(record_path.suffix + ".complete")
    paths = [npz_path, record_path, marker]
    paths.extend(npz_path.parent.glob(f".{npz_path.name}.*.tmp.npz"))
    paths.extend(record_path.parent.glob(f".{record_path.name}.*.tmp"))
    return paths


def clean_cache_paths(root: Path, block_id: str) -> tuple[Path, Path]:
    directory = (
        root / "outputs" / "reports" / EXPECTED_ATTEMPT / "bootstrap_accelerator_v1"
    )
    return (
        directory / f"{block_id}__clean_draws.npz",
        directory / f"{block_id}__clean_draws.json",
    )


def write_clean_cache(
    npz_path: Path,
    record_path: Path,
    *,
    block_id: str,
    dataset: str,
    model: str,
    config_sha256: str,
    seed: int,
    n_images: int,
    input_image_ids_sha256: str,
    annotation_sha256: str,
    samples_digest: str,
    arrays: dict[str, np.ndarray],
    point: dict[str, float],
    source_artifacts_sha256: dict[str, str],
    acceleration_provenance: dict[str, Any],
    contract: Any,
) -> dict[str, Any]:
    n_boot = len(arrays["default_int8"])
    if set(arrays) != set(ARM_ORDER_UNIQUE):
        raise AcceleratorError("clean draw fields differ from the three-arm contract")
    if any(
        np.asarray(values).shape != (n_boot,) or not np.all(np.isfinite(values))
        for values in arrays.values()
    ):
        raise AcceleratorError("clean draw arrays are incomplete or non-finite")
    _atomic_npz(
        npz_path,
        schema_version=np.asarray(ACCELERATOR_VERSION),
        n_boot=np.asarray(n_boot),
        seed=np.asarray(seed),
        n_images=np.asarray(n_images),
        default_int8=np.asarray(arrays["default_int8"], dtype=np.float64),
        fp8=np.asarray(arrays["fp8"], dtype=np.float64),
        shared_int8=np.asarray(arrays["shared_int8"], dtype=np.float64),
    )
    record = {
        "schema_version": ACCELERATOR_VERSION,
        "created_at_utc": utc_now(),
        "attempt": EXPECTED_ATTEMPT,
        "status": "complete",
        "role": "reusable_clean_overall_ap_draws",
        "block_id": block_id,
        "dataset": dataset,
        "model": model,
        "config_sha256": config_sha256,
        "n_boot": n_boot,
        "seed": int(seed),
        "n_images": int(n_images),
        "input_image_ids_sha256": input_image_ids_sha256,
        "annotation_sha256": annotation_sha256,
        "samples_sha256": samples_digest,
        "point": {name: float(point[name]) for name in ARM_ORDER_UNIQUE},
        "draw_cache": str(npz_path),
        "draw_cache_sha256": contract.sha256_file(npz_path),
        "source_artifacts_sha256": dict(sorted(source_artifacts_sha256.items())),
        "acceleration_provenance": acceleration_provenance,
    }
    return contract.write_complete_json(
        record_path, record, self_hash_field="record_sha256"
    )


def validate_clean_cache(
    npz_path: Path,
    record_path: Path,
    *,
    block_id: str,
    dataset: str,
    model: str,
    config_sha256: str,
    seed: int,
    n_boot: int,
    n_images: int,
    input_image_ids_sha256: str,
    annotation_sha256: str,
    samples_digest: str,
    contract: Any,
) -> tuple[dict[str, Any], dict[str, np.ndarray]]:
    record = contract.read_complete_json(record_path, self_hash_field="record_sha256")
    if (
        record.get("attempt") != EXPECTED_ATTEMPT
        or record.get("status") != "complete"
        or record.get("role") != "reusable_clean_overall_ap_draws"
        or record.get("block_id") != block_id
        or record.get("dataset") != dataset
        or record.get("model") != model
        or record.get("config_sha256") != config_sha256
        or record.get("seed") != seed
        or record.get("n_boot") != n_boot
        or record.get("n_images") != n_images
        or record.get("input_image_ids_sha256") != input_image_ids_sha256
        or record.get("annotation_sha256") != annotation_sha256
        or record.get("samples_sha256") != samples_digest
        or record.get("draw_cache_sha256") != contract.sha256_file(npz_path)
    ):
        raise AcceleratorError(f"clean cache identity mismatch: {record_path}")
    for path, expected in record.get("source_artifacts_sha256", {}).items():
        source = Path(path)
        if not source.is_file() or contract.sha256_file(source) != expected:
            raise AcceleratorError(f"clean cache source changed: {source}")
    try:
        with np.load(npz_path, allow_pickle=False) as data:
            if set(data.files) != CLEAN_ARRAY_FIELDS:
                raise AcceleratorError(f"clean cache fields mismatch: {npz_path}")
            if (
                int(data["schema_version"].item()) != ACCELERATOR_VERSION
                or int(data["n_boot"].item()) != n_boot
                or int(data["seed"].item()) != seed
                or int(data["n_images"].item()) != n_images
            ):
                raise AcceleratorError(f"clean cache dimensions mismatch: {npz_path}")
            arrays = {
                name: np.asarray(data[name], dtype=np.float64)
                for name in ARM_ORDER_UNIQUE
            }
    except OSError as exc:
        raise AcceleratorError(f"invalid clean cache: {npz_path}") from exc
    if any(
        values.shape != (n_boot,) or not np.all(np.isfinite(values))
        for values in arrays.values()
    ):
        raise AcceleratorError(f"clean cache replicate vector mismatch: {npz_path}")
    return record, arrays


def write_standard_cell_cache(
    npz_path: Path,
    record_path: Path,
    *,
    payload: dict[str, Any],
    condition: dict[str, Any],
    arrays: dict[str, np.ndarray],
    point: dict[str, float],
    source_artifacts_sha256: dict[str, str],
    acceleration_provenance: dict[str, Any],
    contract: Any,
) -> dict[str, Any]:
    n_boot = int(payload["n_boot"])
    if set(arrays) != {"delta_e_default", "delta_e_shared", "omega"}:
        raise AcceleratorError("cell arrays differ from the standard V3 schema")
    if any(
        np.asarray(values).shape != (n_boot,) or not np.all(np.isfinite(values))
        for values in arrays.values()
    ):
        raise AcceleratorError("cell draw arrays are incomplete or non-finite")
    if not np.allclose(
        arrays["omega"],
        arrays["delta_e_default"] - arrays["delta_e_shared"],
        atol=1e-12,
        rtol=0,
    ):
        raise AcceleratorError("cell Omega draw identity failed")
    _atomic_npz(
        npz_path,
        schema_version=np.asarray(1),
        n_boot=np.asarray(n_boot),
        seed=np.asarray(int(payload["seed"])),
        delta_e_default=np.asarray(arrays["delta_e_default"], dtype=np.float64),
        delta_e_shared=np.asarray(arrays["delta_e_shared"], dtype=np.float64),
        omega=np.asarray(arrays["omega"], dtype=np.float64),
    )
    record = {
        "schema_version": 1,
        "created_at_utc": utc_now(),
        "attempt": EXPECTED_ATTEMPT,
        "block_id": payload["block_id"],
        "dataset": payload["dataset"],
        "model": payload["model"],
        "corruption": condition["corruption"],
        "severity": int(condition["severity"]),
        "metric": "COCO-style AP@[0.50:0.95], AP points",
        "n_images": len(payload["clean"]["default_int8"]["image_ids"]),
        "n_boot": n_boot,
        "seed": int(payload["seed"]),
        "common_draw_contract": (
            "same image-position resample across 8 default/shared x "
            "INT8/FP8 x clean/corrupt arms"
        ),
        "config_sha256": payload["config_sha256"],
        "point": point,
        "percentile_2.5_50_97.5": {
            name: percentile(values) for name, values in arrays.items()
        },
        "draw_cache": str(npz_path),
        "draw_cache_sha256": contract.sha256_file(npz_path),
        "source_artifacts_sha256": dict(sorted(source_artifacts_sha256.items())),
        "acceleration_provenance": acceleration_provenance,
    }
    return contract.write_complete_json(
        record_path, record, self_hash_field="record_sha256"
    )


def _source_hashes(
    clean: dict[str, dict[str, Any]],
    corrupt: dict[str, dict[str, Any]],
    annotation: Path,
    *,
    extra: dict[Path, str],
    contract: Any,
) -> dict[str, str]:
    result = {str(annotation): contract.sha256_file(annotation)}
    for record in list(clean.values()) + list(corrupt.values()):
        for key in ("prediction", "input_record", "run_record", "metric"):
            source = Path(record[key])
            result[str(source)] = contract.sha256_file(source)
    for source, expected in extra.items():
        if contract.sha256_file(source) != expected:
            raise AcceleratorError(f"acceleration provenance changed: {source}")
        result[str(source.resolve())] = expected
    return dict(sorted(result.items()))


def _load_ids(record: dict[str, Any], contract: Any) -> list[int]:
    for key in ("prediction", "input_record", "run_record", "metric"):
        source = Path(record[key])
        expected = (
            record[f"{key}_sha256"]
            if key != "input_record"
            else record["input_record_sha256"]
        )
        if not source.is_file() or contract.sha256_file(source) != expected:
            raise AcceleratorError(f"source changed before bootstrap: {source}")
    document = json.loads(Path(record["input_record"]).read_text(encoding="utf-8"))
    if document.get("image_ids") != record.get("image_ids"):
        raise AcceleratorError(f"ordered image IDs changed: {record['input_record']}")
    return document["image_ids"]


def _load_predictions(record: dict[str, Any], contract: Any) -> list[dict[str, Any]]:
    source = Path(record["prediction"])
    if contract.sha256_file(source) != record["prediction_sha256"]:
        raise AcceleratorError(f"prediction changed before bootstrap: {source}")
    predictions = json.loads(source.read_text(encoding="utf-8"))
    if not isinstance(predictions, list):
        raise AcceleratorError(f"prediction payload is not a list: {source}")
    return predictions


def _build_unique_evaluations(
    gt: Any,
    records: dict[str, dict[str, Any]],
    ids: list[int],
    *,
    paired: Any,
    contract: Any,
) -> dict[str, Any]:
    result = {}
    for name in ARM_ORDER_UNIQUE:
        result[name] = paired.build_eval(
            gt, _load_predictions(records[name], contract), ids
        )
    return result


def _overall_values(
    evaluations: dict[str, Any], positions: list[int]
) -> dict[str, float]:
    return {
        name: accumulate_ap_overall(evaluations[name], positions)
        for name in ARM_ORDER_UNIQUE
    }


def _ensure_clean_draws(
    root: Path,
    payload: dict[str, Any],
    *,
    plan_path: Path,
    plan_file_sha256: str,
    accelerator_path: Path,
    accelerator_sha256: str,
) -> tuple[dict[str, Any], dict[str, np.ndarray], np.ndarray]:
    legacy, paired, contract = _project_modules()
    del legacy
    fp8_proof = fp8_identity_gate(payload["clean"], f"{payload['block_id']}/clean")
    records = unique_arms(
        payload["clean"], f"{payload['block_id']}/clean", identity_checked=True
    )
    ids = _load_ids(records["default_int8"], contract)
    if any(_load_ids(records[name], contract) != ids for name in ARM_ORDER_UNIQUE[1:]):
        raise AcceleratorError(f"clean ordered image IDs differ: {payload['block_id']}")
    n_boot, seed = int(payload["n_boot"]), int(payload["seed"])
    samples = make_samples(seed, n_boot, len(ids))
    sample_digest = samples_sha256(samples)
    npz_path, record_path = clean_cache_paths(root, payload["block_id"])
    marker = record_path.with_suffix(record_path.suffix + ".complete")
    if npz_path.exists() and record_path.exists() and marker.exists():
        record, arrays = validate_clean_cache(
            npz_path,
            record_path,
            block_id=payload["block_id"],
            dataset=payload["dataset"],
            model=payload["model"],
            config_sha256=payload["config_sha256"],
            seed=seed,
            n_boot=n_boot,
            n_images=len(ids),
            input_image_ids_sha256=payload["clean"]["default_int8"][
                "input_image_ids_sha256"
            ],
            annotation_sha256=payload["annotation_sha256"],
            samples_digest=sample_digest,
            contract=contract,
        )
        return record, arrays, samples
    partial = _partial_paths(npz_path, record_path)
    _quarantine_partial(root, partial, f"{payload['block_id']}__clean")

    from pycocotools.coco import COCO

    annotation = Path(payload["annotations"])
    if contract.sha256_file(annotation) != payload["annotation_sha256"]:
        raise AcceleratorError(
            f"annotation changed before clean bootstrap: {annotation}"
        )
    gt = COCO(str(annotation))
    evaluations = _build_unique_evaluations(
        gt, records, ids, paired=paired, contract=contract
    )
    full = list(range(len(ids)))
    point = _overall_values(evaluations, full)
    for name in ARM_ORDER_UNIQUE:
        expected_name = "default_fp8" if name == "fp8" else name
        if abs(point[name] - payload["clean"][expected_name]["ap"]) > 5e-8:
            raise AcceleratorError(
                f"overall accumulator disagrees with clean metric: {payload['block_id']}/{name}"
            )
    arrays = {name: np.empty(n_boot, dtype=np.float64) for name in ARM_ORDER_UNIQUE}
    for index, sample in enumerate(samples):
        values = _overall_values(evaluations, sample.tolist())
        for name in ARM_ORDER_UNIQUE:
            arrays[name][index] = values[name]
        if (index + 1) % 50 == 0 or index + 1 == n_boot:
            print(
                json.dumps(
                    {
                        "ACCELERATOR_CLEAN_PROGRESS": payload["block_id"],
                        "completed": index + 1,
                        "total": n_boot,
                    }
                ),
                flush=True,
            )
    provenance = {
        "accelerator_version": ACCELERATOR_VERSION,
        "accelerator_path": str(accelerator_path),
        "accelerator_sha256": accelerator_sha256,
        "plan_path": str(plan_path),
        "plan_file_sha256": plan_file_sha256,
        "overall_only_operation_order": "paired_bootstrap.accumulate_ap area index 0",
        "fp8_evaluations": 1,
        "fp8_identity_gate": fp8_proof,
    }
    sources = _source_hashes(
        payload["clean"],
        {},
        annotation,
        extra={
            accelerator_path: accelerator_sha256,
            plan_path: plan_file_sha256,
        },
        contract=contract,
    )
    record = write_clean_cache(
        npz_path,
        record_path,
        block_id=payload["block_id"],
        dataset=payload["dataset"],
        model=payload["model"],
        config_sha256=payload["config_sha256"],
        seed=seed,
        n_images=len(ids),
        input_image_ids_sha256=payload["clean"]["default_int8"][
            "input_image_ids_sha256"
        ],
        annotation_sha256=payload["annotation_sha256"],
        samples_digest=sample_digest,
        arrays=arrays,
        point=point,
        source_artifacts_sha256=sources,
        acceleration_provenance=provenance,
        contract=contract,
    )
    del evaluations, gt
    gc.collect()
    return record, arrays, samples


def _accelerate_cell(job: dict[str, Any]) -> dict[str, Any]:
    legacy, paired, contract = _project_modules()
    root = Path(job["project_root"])
    payload = job["payload"]
    condition = job["condition"]
    corruption, severity = condition["corruption"], int(condition["severity"])
    npz_path, record_path = legacy.cell_paths(
        root, payload["block_id"], corruption, severity
    )
    marker = record_path.with_suffix(record_path.suffix + ".complete")
    if npz_path.exists() and record_path.exists() and marker.exists():
        legacy.validate_cell_cache(
            npz_path,
            record_path,
            block_id=payload["block_id"],
            corruption=corruption,
            severity=severity,
            n_boot=int(payload["n_boot"]),
            config_sha256=payload["config_sha256"],
        )
        return {"status": "reused", "record": str(record_path)}
    _quarantine_partial(
        root,
        _partial_paths(npz_path, record_path),
        f"{payload['block_id']}__{corruption}-s{severity}",
    )

    clean_npz = Path(job["clean_npz"])
    clean_record_path = Path(job["clean_record"])
    ids = _load_ids(payload["clean"]["default_int8"], contract)
    samples = make_samples(int(payload["seed"]), int(payload["n_boot"]), len(ids))
    clean_record, clean_draws = validate_clean_cache(
        clean_npz,
        clean_record_path,
        block_id=payload["block_id"],
        dataset=payload["dataset"],
        model=payload["model"],
        config_sha256=payload["config_sha256"],
        seed=int(payload["seed"]),
        n_boot=int(payload["n_boot"]),
        n_images=len(ids),
        input_image_ids_sha256=payload["clean"]["default_int8"][
            "input_image_ids_sha256"
        ],
        annotation_sha256=payload["annotation_sha256"],
        samples_digest=samples_sha256(samples),
        contract=contract,
    )
    fp8_proof = fp8_identity_gate(
        condition["arms"], f"{payload['block_id']}/{corruption}-s{severity}"
    )
    records = unique_arms(
        condition["arms"],
        f"{payload['block_id']}/{corruption}-s{severity}",
        identity_checked=True,
    )
    if any(_load_ids(records[name], contract) != ids for name in ARM_ORDER_UNIQUE):
        raise AcceleratorError(
            f"corrupt ordered image IDs differ: {payload['block_id']}/{corruption}-s{severity}"
        )

    from pycocotools.coco import COCO

    annotation = Path(payload["annotations"])
    if contract.sha256_file(annotation) != payload["annotation_sha256"]:
        raise AcceleratorError(
            f"annotation changed before cell bootstrap: {annotation}"
        )
    gt = COCO(str(annotation))
    evaluations = _build_unique_evaluations(
        gt, records, ids, paired=paired, contract=contract
    )
    full = list(range(len(ids)))
    corrupt_point = _overall_values(evaluations, full)
    for name in ARM_ORDER_UNIQUE:
        expected_name = "default_fp8" if name == "fp8" else name
        if abs(corrupt_point[name] - condition["arms"][expected_name]["ap"]) > 5e-8:
            raise AcceleratorError(
                f"overall accumulator disagrees with metric: "
                f"{payload['block_id']}/{corruption}-s{severity}/{name}"
            )
    clean_point = clean_record["point"]
    point = contrast_values(
        default_int8_clean=clean_point["default_int8"] * 100.0,
        fp8_clean=clean_point["fp8"] * 100.0,
        shared_int8_clean=clean_point["shared_int8"] * 100.0,
        default_int8_corrupt=corrupt_point["default_int8"] * 100.0,
        fp8_corrupt=corrupt_point["fp8"] * 100.0,
        shared_int8_corrupt=corrupt_point["shared_int8"] * 100.0,
    )
    if any(abs(point[key] - condition["point"][key]) > 5e-6 for key in point):
        raise AcceleratorError(
            f"point metric/bootstrap identity mismatch: "
            f"{payload['block_id']}/{corruption}-s{severity}"
        )

    n_boot = int(payload["n_boot"])
    arrays = {
        "delta_e_default": np.empty(n_boot, dtype=np.float64),
        "delta_e_shared": np.empty(n_boot, dtype=np.float64),
        "omega": np.empty(n_boot, dtype=np.float64),
    }
    for index, sample in enumerate(samples):
        corrupt_values = _overall_values(evaluations, sample.tolist())
        value = contrast_values(
            default_int8_clean=clean_draws["default_int8"][index] * 100.0,
            fp8_clean=clean_draws["fp8"][index] * 100.0,
            shared_int8_clean=clean_draws["shared_int8"][index] * 100.0,
            default_int8_corrupt=corrupt_values["default_int8"] * 100.0,
            fp8_corrupt=corrupt_values["fp8"] * 100.0,
            shared_int8_corrupt=corrupt_values["shared_int8"] * 100.0,
        )
        arrays["delta_e_default"][index] = value["delta_e_default"]
        arrays["delta_e_shared"][index] = value["delta_e_shared"]
        arrays["omega"][index] = value["omega_default_minus_shared"]
        if (index + 1) % 50 == 0 or index + 1 == n_boot:
            print(
                json.dumps(
                    {
                        "ACCELERATOR_CELL_PROGRESS": (
                            f"{payload['block_id']}/{corruption}-s{severity}"
                        ),
                        "completed": index + 1,
                        "total": n_boot,
                    }
                ),
                flush=True,
            )
    accelerator_path = Path(job["accelerator_path"])
    plan_path = Path(job["plan_path"])
    provenance = {
        "accelerator_version": ACCELERATOR_VERSION,
        "accelerator_path": str(accelerator_path),
        "accelerator_sha256": job["accelerator_sha256"],
        "plan_path": str(plan_path),
        "plan_file_sha256": job["plan_file_sha256"],
        "clean_cache": str(clean_npz),
        "clean_cache_sha256": contract.sha256_file(clean_npz),
        "samples_sha256": clean_record["samples_sha256"],
        "overall_only_operation_order": "paired_bootstrap.accumulate_ap area index 0",
        "fp8_evaluations": 1,
        "fp8_identity_gate": fp8_proof,
    }
    sources = _source_hashes(
        payload["clean"],
        condition["arms"],
        annotation,
        extra={
            accelerator_path: job["accelerator_sha256"],
            plan_path: job["plan_file_sha256"],
            clean_npz: contract.sha256_file(clean_npz),
            clean_record_path: contract.sha256_file(clean_record_path),
        },
        contract=contract,
    )
    write_standard_cell_cache(
        npz_path,
        record_path,
        payload=payload,
        condition=condition,
        arrays=arrays,
        point=point,
        source_artifacts_sha256=sources,
        acceleration_provenance=provenance,
        contract=contract,
    )
    legacy.validate_cell_cache(
        npz_path,
        record_path,
        block_id=payload["block_id"],
        corruption=corruption,
        severity=severity,
        n_boot=n_boot,
        config_sha256=payload["config_sha256"],
    )
    return {"status": "created", "record": str(record_path)}


def _ancestor_pids() -> set[int]:
    result = {os.getpid()}
    current = os.getpid()
    while current > 1:
        try:
            fields = (Path("/proc") / str(current) / "stat").read_text().split()
            parent = int(fields[3])
        except (OSError, ValueError, IndexError):
            break
        if parent in result or parent <= 0:
            break
        result.add(parent)
        current = parent
    return result


def _active_v3_processes() -> list[tuple[int, str]]:
    selected = []
    excluded = _ancestor_pids()
    for entry in Path("/proc").iterdir():
        if not entry.name.isdigit() or int(entry.name) in excluded:
            continue
        try:
            command = (entry / "cmdline").read_bytes().replace(b"\0", b" ").decode()
        except (OSError, UnicodeDecodeError):
            continue
        if "shared_mask_pilot_v3" not in command:
            continue
        if any(
            name in command
            for name in (
                "analyze_shared_mask_pilot.py",
                "run_shared_mask_pilot_v3.py",
                "supervise_shared_mask_pilot.sh",
                Path(__file__).name,
            )
        ):
            selected.append((int(entry.name), command.strip()))
    return selected


def assert_no_active_v3_process() -> None:
    active = _active_v3_processes()
    if active:
        details = "; ".join(f"pid={pid} {command}" for pid, command in active)
        raise AcceleratorError(
            "active V3 process would race bootstrap cache publication: " + details
        )


def _plan(
    root: Path,
    config_path: Path,
    config: dict[str, Any],
    *,
    accelerator_path: Path,
    contract: Any,
) -> tuple[Path, dict[str, Any]]:
    directory = (
        root / "outputs" / "reports" / EXPECTED_ATTEMPT / "bootstrap_accelerator_v1"
    )
    path = directory / "plan.json"
    frozen_sources = {
        str(candidate.relative_to(root)): contract.sha256_file(candidate)
        for candidate in (
            root / "src" / "analyze_shared_mask_pilot.py",
            root / "src" / "paired_bootstrap.py",
            root / "src" / "topic_c" / "shared_mask_pilot.py",
        )
    }
    expected = {
        "schema_version": ACCELERATOR_VERSION,
        "attempt": EXPECTED_ATTEMPT,
        "status": "frozen_before_accelerated_cache_generation",
        "role": "analysis_only_execution_addendum",
        "target_block": TARGET_BLOCK,
        "config": str(config_path),
        "config_file_sha256": contract.sha256_file(config_path),
        "config_sha256": config["config_sha256"],
        "n_boot": int(config["bootstrap_replicates"]),
        "maximum_cell_workers": MAX_WORKERS,
        "accelerator": str(accelerator_path),
        "accelerator_sha256": contract.sha256_file(accelerator_path),
        "frozen_sources_sha256": frozen_sources,
        "equivalence_contract": {
            "draw_schedule": "unchanged NumPy Generator.integers schedule and dataset seed",
            "ap_endpoint": "area index zero only; operation order copied from frozen accumulator",
            "fp8_reuse": "one evaluation only after exact path and SHA-256 equality",
            "output_schema": "frozen validate_cell_cache-compatible NPZ/JSON",
            "completed_cache_policy": "validate and reuse; never overwrite",
        },
    }
    if path.exists() or path.with_suffix(path.suffix + ".complete").exists():
        observed = contract.read_complete_json(path, self_hash_field="plan_sha256")
        if {key: observed.get(key) for key in expected} != expected:
            raise AcceleratorError(f"existing accelerator plan differs: {path}")
        return path, observed
    return path, contract.write_complete_json(
        path, expected, self_hash_field="plan_sha256"
    )


def _write_completion(
    root: Path,
    payload: dict[str, Any],
    *,
    plan_path: Path,
    plan: dict[str, Any],
    accelerator_path: Path,
    contract: Any,
    legacy: Any,
) -> dict[str, Any]:
    records = {}
    for condition in payload["conditions"]:
        corruption, severity = condition["corruption"], int(condition["severity"])
        npz_path, record_path = legacy.cell_paths(
            root, payload["block_id"], corruption, severity
        )
        legacy.validate_cell_cache(
            npz_path,
            record_path,
            block_id=payload["block_id"],
            corruption=corruption,
            severity=severity,
            n_boot=int(payload["n_boot"]),
            config_sha256=payload["config_sha256"],
        )
        records[str(npz_path.relative_to(root))] = contract.sha256_file(npz_path)
        records[str(record_path.relative_to(root))] = contract.sha256_file(record_path)
    clean_npz, clean_record = clean_cache_paths(root, payload["block_id"])
    evidence = {
        str(plan_path.relative_to(root)): contract.sha256_file(plan_path),
        str(accelerator_path.relative_to(root)): contract.sha256_file(accelerator_path),
        str(clean_npz.relative_to(root)): contract.sha256_file(clean_npz),
        str(clean_record.relative_to(root)): contract.sha256_file(clean_record),
        **records,
    }
    expected = {
        "schema_version": ACCELERATOR_VERSION,
        "attempt": EXPECTED_ATTEMPT,
        "status": "complete",
        "role": "analysis_only_execution_addendum",
        "target_block": payload["block_id"],
        "config_sha256": payload["config_sha256"],
        "n_boot": int(payload["n_boot"]),
        "cells": len(payload["conditions"]),
        "maximum_cell_workers": MAX_WORKERS,
        "plan_sha256": plan["plan_sha256"],
        "artifacts_sha256": dict(sorted(evidence.items())),
        "next_step": "run the unchanged V3 analyzer to validate all 36 caches and publish the summary",
    }
    path = plan_path.parent / "complete.json"
    if path.exists() or path.with_suffix(path.suffix + ".complete").exists():
        observed = contract.read_complete_json(path, self_hash_field="report_sha256")
        if {key: observed.get(key) for key in expected} != expected:
            raise AcceleratorError(f"existing accelerator completion differs: {path}")
        return observed
    expected["completed_at_utc"] = utc_now()
    return contract.write_complete_json(path, expected, self_hash_field="report_sha256")


def run_bounded_cells(
    jobs: list[dict[str, Any]],
    *,
    workers: int,
    work: Any = _accelerate_cell,
    executor_factory: Any = ProcessPoolExecutor,
) -> list[dict[str, Any]]:
    """Run at most ``workers`` cells and stop dispatch after an observed error."""
    workers = validate_worker_count(workers)
    remaining = iter(jobs)
    results: list[dict[str, Any]] = []
    with executor_factory(max_workers=workers) as executor:
        active: dict[Any, dict[str, Any]] = {}
        for _ in range(min(workers, len(jobs))):
            try:
                job = next(remaining)
            except StopIteration:
                break
            active[executor.submit(work, job)] = job
        while active:
            completed, _ = wait(active, return_when=FIRST_COMPLETED)
            errors = [future.exception() for future in completed]
            errors = [error for error in errors if error is not None]
            if errors:
                for future in active:
                    if future not in completed:
                        future.cancel()
                raise errors[0]
            batch = []
            for future in completed:
                active.pop(future)
                result = future.result()
                results.append(result)
                batch.append(result)
            for result in batch:
                print(json.dumps({"ACCELERATOR_CELL_COMPLETE": result}), flush=True)
                try:
                    job = next(remaining)
                except StopIteration:
                    continue
                active[executor.submit(work, job)] = job
    return results


def run(
    root: Path, config_path: Path, *, workers: int, preflight_only: bool = False
) -> dict[str, Any]:
    workers = validate_worker_count(workers)
    root, config_path = root.resolve(), config_path.resolve()
    assert_no_active_v3_process()
    legacy, _paired, contract = _project_modules()
    config = contract.validate_config(
        root, config_path, expected_attempt=EXPECTED_ATTEMPT
    )
    if config_path != (root / "configs" / f"{EXPECTED_ATTEMPT}.json").resolve():
        raise AcceleratorError("accelerator requires the frozen V3 config path")
    accelerator_path = Path(__file__).resolve()
    plan_path, plan = _plan(
        root,
        config_path,
        config,
        accelerator_path=accelerator_path,
        contract=contract,
    )
    blocks, _point = legacy.prepare_blocks(root, config)
    matches = [block for block in blocks if block["block_id"] == TARGET_BLOCK]
    if len(matches) != 1:
        raise AcceleratorError(f"expected exactly one target block: {TARGET_BLOCK}")
    payload = matches[0]
    if len(payload["conditions"]) != 12 or int(payload["n_boot"]) != 2000:
        raise AcceleratorError("RT-DETR grid or bootstrap replicate count changed")
    fp8_identity_gate(payload["clean"], f"{TARGET_BLOCK}/clean")
    for condition in payload["conditions"]:
        fp8_identity_gate(
            condition["arms"],
            f"{TARGET_BLOCK}/{condition['corruption']}-s{condition['severity']}",
        )
    if preflight_only:
        return {
            "status": "preflight_complete",
            "target_block": TARGET_BLOCK,
            "cells": 12,
            "n_boot": 2000,
            "workers": workers,
            "plan": str(plan_path),
        }
    accelerator_sha = contract.sha256_file(accelerator_path)
    plan_file_sha = contract.sha256_file(plan_path)
    _ensure_clean_draws(
        root,
        payload,
        plan_path=plan_path,
        plan_file_sha256=plan_file_sha,
        accelerator_path=accelerator_path,
        accelerator_sha256=accelerator_sha,
    )
    clean_npz, clean_record = clean_cache_paths(root, payload["block_id"])
    jobs = [
        {
            "project_root": str(root),
            "payload": payload,
            "condition": condition,
            "clean_npz": str(clean_npz),
            "clean_record": str(clean_record),
            "accelerator_path": str(accelerator_path),
            "accelerator_sha256": accelerator_sha,
            "plan_path": str(plan_path),
            "plan_file_sha256": plan_file_sha,
        }
        for condition in payload["conditions"]
    ]
    run_bounded_cells(jobs, workers=workers)
    return _write_completion(
        root,
        payload,
        plan_path=plan_path,
        plan=plan,
        accelerator_path=accelerator_path,
        contract=contract,
        legacy=legacy,
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", default=".")
    parser.add_argument("--config", default=f"configs/{EXPECTED_ATTEMPT}.json")
    parser.add_argument("--workers", type=int, default=MAX_WORKERS)
    parser.add_argument("--preflight-only", action="store_true")
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    config_path = Path(args.config)
    if not config_path.is_absolute():
        config_path = root / config_path
    lock_path = (
        root / "outputs" / "logs" / EXPECTED_ATTEMPT / "bootstrap_accelerator.lock"
    )
    lock_path.parent.mkdir(parents=True, exist_ok=True)
    driver_lock_path = lock_path.parent / "driver.lock"
    with lock_path.open("a+") as lock, driver_lock_path.open("a+") as driver_lock:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("bootstrap accelerator is already active") from exc
        try:
            # Share the frozen driver's lock rather than inventing a race-prone
            # convention.  The accelerator never truncates its contents.
            fcntl.flock(driver_lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise SystemExit("frozen V3 driver is still active") from exc
        try:
            result = run(
                root,
                config_path,
                workers=args.workers,
                preflight_only=args.preflight_only,
            )
        except (RuntimeError, ValueError) as exc:
            raise SystemExit(f"V3 BOOTSTRAP ACCELERATOR REFUSED: {exc}") from exc
    print(json.dumps({"V3_BOOTSTRAP_ACCELERATOR": result}, indent=2), flush=True)


if __name__ == "__main__":
    main()
