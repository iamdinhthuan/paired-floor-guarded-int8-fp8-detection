#!/usr/bin/env python3
"""Fail-closed paired INT8--FP8 bootstrap contrast calculation."""
from __future__ import annotations

import argparse
import hashlib
import json
import math
import multiprocessing as mp
from pathlib import Path
from typing import Any, Callable, Sequence

import numpy as np

from topic_c.manifest import sha256_file


HEIGHT_GROUPS = {
    "small_like": (0.0, 24.0),
    "large_like": (48.0, float("inf")),
}


ARM_NAMES = ("int8_clean", "fp8_clean", "int8_corrupt", "fp8_corrupt")

DRAW_CACHE_FIELDS = {
    "schema_version", "cache_identity_sha256", "n_boot", "schedule_sha256", "delta_e_all", "delta_psi",
}

CLEAN_CACHE_FIELDS = {"metadata", "int8_clean", "fp8_clean", "int8_clean_point", "fp8_clean_point"}
CLEAN_CACHE_METADATA_FIELDS = {
    "schema_version", "dataset", "model", "endpoint_type", "n_images", "n_boot", "annotation_sha256",
    "labels", "input_hashes", "bootstrap_schedule", "cache_identity_sha256",
}

_FORK_EVALUATIONS: Sequence[Any] | None = None
_FORK_ACCUMULATOR: Callable[[Any, list[int]], Sequence[float]] | None = None


def canonical_hash(document: dict[str, Any], hash_key: str) -> str:
    """Hash a JSON document without its self-referential hash field."""
    payload = {key: value for key, value in document.items() if key != hash_key}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":"), ensure_ascii=True).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def draw_cache_identity(binding: dict[str, Any]) -> str:
    """Return the cache identity fixed before a component artifact is written."""
    if not isinstance(binding, dict) or not binding or "cache_identity_sha256" in binding:
        raise ValueError("draw-cache identity binding is invalid")
    return canonical_hash(
        {"schema_version": 1, "cache_kind": "format_contrast_draws", **binding},
        "cache_identity_sha256",
    )


def write_draw_cache(
    path: Path, *, identity_sha256: str, n_boot: int, schedule_sha256: str,
    delta_e_all: np.ndarray, delta_psi: np.ndarray,
) -> dict[str, str]:
    """Write cache bytes before the component commits their file digest."""
    if path.exists():
        raise ValueError(f"refusing to overwrite draw cache: {path}")
    if (not isinstance(identity_sha256, str) or len(identity_sha256) != 64
            or not isinstance(schedule_sha256, str) or len(schedule_sha256) != 64 or n_boot <= 0):
        raise ValueError("draw-cache binding metadata is invalid")
    delta_e_all, delta_psi = np.asarray(delta_e_all, dtype=float), np.asarray(delta_psi, dtype=float)
    if delta_e_all.shape != (n_boot,) or delta_psi.shape != (n_boot,):
        raise ValueError("draw-cache replicate cardinality is invalid")
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(1),
        cache_identity_sha256=np.asarray(identity_sha256),
        n_boot=np.asarray(n_boot),
        schedule_sha256=np.asarray(schedule_sha256),
        delta_e_all=delta_e_all,
        delta_psi=delta_psi,
    )
    return {"sha256": sha256_file(path), "identity_sha256": identity_sha256}


def load_draw_cache(
    path: Path, *, expected_sha256: str, expected_identity_sha256: str, n_boot: int, schedule_sha256: str,
) -> dict[str, np.ndarray]:
    """Read a precommitted draw cache only when bytes and internal bindings agree."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing/empty draw cache: {path}")
    if sha256_file(path) != expected_sha256:
        raise ValueError(f"draw-cache file SHA-256 mismatch: {path}")
    try:
        with np.load(path, allow_pickle=False) as cache:
            if set(cache.files) != DRAW_CACHE_FIELDS:
                raise ValueError(f"invalid draw-cache fields: {path}")
            if (int(cache["schema_version"].item()) != 1
                    or str(cache["cache_identity_sha256"].item()) != expected_identity_sha256
                    or int(cache["n_boot"].item()) != n_boot
                    or str(cache["schedule_sha256"].item()) != schedule_sha256):
                raise ValueError(f"draw-cache internal binding mismatch: {path}")
            result = {
                "delta_e_all": np.asarray(cache["delta_e_all"], dtype=float),
                "delta_psi": np.asarray(cache["delta_psi"], dtype=float),
            }
    except OSError as exc:
        raise ValueError(f"invalid draw cache: {path}") from exc
    if any(values.shape != (n_boot,) for values in result.values()):
        raise ValueError(f"invalid draw-cache replicate cardinality: {path}")
    return result


def safe_relative_evidence_path(root: Path, path: Path, label: str) -> str:
    """Return a fail-closed, project-relative evidence path."""
    if not isinstance(label, str) or not label:
        raise ValueError("evidence path label is required")
    try:
        return str(path.resolve().relative_to(root.resolve()))
    except ValueError as exc:
        raise ValueError(f"{label} path escapes project root: {path}") from exc


def resolve_relative_evidence_path(root: Path, relative: str, label: str) -> Path:
    """Resolve an evidence path only when its spelling and target remain under root."""
    if not isinstance(relative, str) or not relative:
        raise ValueError(f"{label} path is missing")
    candidate = Path(relative)
    if candidate.is_absolute() or ".." in candidate.parts:
        raise ValueError(f"{label} path must be a safe relative path: {relative}")
    resolved = (root.resolve() / candidate).resolve()
    if root.resolve() not in resolved.parents:
        raise ValueError(f"{label} path escapes project root: {relative}")
    return resolved


def schedule_binding(root: Path, schedule: dict[str, Any]) -> dict[str, Any]:
    """Convert a loaded schedule record to the immutable public evidence binding."""
    required = {"path", "sha256", "n_boot", "seed", "n_images", "image_ids_sha256"}
    if not isinstance(schedule, dict) or not required.issubset(schedule):
        raise ValueError("bootstrap schedule record is incomplete")
    return {
        "path": safe_relative_evidence_path(root, Path(schedule["path"]), "bootstrap schedule"),
        **{key: schedule[key] for key in required - {"path"}},
    }


def validate_schedule_evidence(root: Path, binding: dict[str, Any]) -> dict[str, Any]:
    """Validate a materialized schedule's exact bytes and self-described dimensions."""
    required = {"path", "sha256", "n_boot", "seed", "n_images", "image_ids_sha256"}
    if not isinstance(binding, dict) or set(binding) != required:
        raise ValueError("bootstrap schedule evidence binding is invalid")
    path = resolve_relative_evidence_path(root, binding["path"], "bootstrap schedule")
    if not path.is_file() or path.stat().st_size == 0 or sha256_file(path) != binding["sha256"]:
        raise ValueError(f"bootstrap schedule file SHA-256 mismatch: {path}")
    try:
        with np.load(path, allow_pickle=False) as schedule:
            if set(schedule.files) != {"schema_version", "n_boot", "seed", "n_images", "image_ids_sha256", "samples"}:
                raise ValueError("bootstrap schedule fields are invalid")
            if (int(schedule["schema_version"].item()) != 1
                    or any(schedule[key].item() != binding[key] for key in ("n_boot", "seed", "n_images", "image_ids_sha256"))):
                raise ValueError("bootstrap schedule metadata does not match evidence binding")
            samples = np.asarray(schedule["samples"], dtype=np.int32)
    except OSError as exc:
        raise ValueError(f"invalid bootstrap schedule evidence: {path}") from exc
    if (samples.shape != (binding["n_boot"], binding["n_images"])
            or np.any(samples < 0) or np.any(samples >= binding["n_images"])):
        raise ValueError("bootstrap schedule samples are invalid")
    return {**binding, "resolved_path": str(path)}


def component_draw_cache_identity(document: dict[str, Any]) -> str:
    """Derive the identity a component must share with its precommitted draw cache."""
    temporary = document.get("temporary_draw_cache")
    schedule = document.get("bootstrap_schedule")
    annotation = document.get("annotation")
    if (not isinstance(temporary, dict) or not isinstance(schedule, dict) or not isinstance(annotation, dict)
            or not isinstance(document.get("input_hashes"), dict)):
        raise ValueError("component draw-cache identity inputs are missing")
    return draw_cache_identity({
        "endpoint_type": document.get("endpoint_type"), "n_images": document.get("n_images"),
        "n_boot": document.get("n_boot"), "seed": document.get("seed"),
        "annotation_sha256": annotation.get("sha256"), "input_hashes": document["input_hashes"],
        "schedule_sha256": schedule.get("sha256"), "path": temporary.get("path"),
    })


def validate_component_seed(document: dict[str, Any], *, expected_seed: int | None = None) -> int:
    """Require a component's declared seed to be its schedule and dataset seed."""
    schedule = document.get("bootstrap_schedule") if isinstance(document, dict) else None
    component_seed = document.get("seed") if isinstance(document, dict) else None
    schedule_seed = schedule.get("seed") if isinstance(schedule, dict) else None
    if (not isinstance(component_seed, int) or isinstance(component_seed, bool)
            or not isinstance(schedule_seed, int) or isinstance(schedule_seed, bool)
            or component_seed != schedule_seed):
        raise ValueError("component seed does not match bootstrap schedule seed")
    if expected_seed is not None and component_seed != expected_seed:
        raise ValueError("component seed does not match deterministic dataset seed")
    return component_seed


def clean_cache_identity(metadata: dict[str, Any]) -> str:
    """Return the clean-arm cache identity fixed before cache bytes are written."""
    if not isinstance(metadata, dict) or not metadata or "cache_identity_sha256" in metadata:
        raise ValueError("clean-arm cache identity metadata is invalid")
    return canonical_hash(
        {"schema_version": 1, "cache_kind": "format_contrast_clean_arms", **metadata},
        "cache_identity_sha256",
    )


def write_clean_arm_cache(
    path: Path, *, metadata: dict[str, Any], identity_sha256: str,
    draws: dict[str, np.ndarray], point: dict[str, Sequence[float]],
) -> dict[str, str]:
    """Write a two-clean-arm cache before its component commits the file SHA-256."""
    if path.exists():
        raise ValueError(f"refusing to overwrite clean-arm cache: {path}")
    required_metadata = CLEAN_CACHE_METADATA_FIELDS - {"schema_version", "cache_identity_sha256"}
    if (set(metadata) != required_metadata or clean_cache_identity(metadata) != identity_sha256
            or not isinstance(identity_sha256, str) or len(identity_sha256) != 64):
        raise ValueError("clean-arm cache metadata is invalid")
    labels = metadata.get("labels")
    if not isinstance(labels, list) or not labels or any(not isinstance(label, str) or not label for label in labels):
        raise ValueError("clean-arm cache labels are invalid")
    if set(draws) != {"int8_clean", "fp8_clean"} or set(point) != set(draws):
        raise ValueError("clean-arm cache arm data are invalid")
    n_boot, width = int(metadata["n_boot"]), len(labels)
    arrays = {name: np.asarray(values, dtype=float) for name, values in draws.items()}
    points = {name: np.asarray(values, dtype=float) for name, values in point.items()}
    if (n_boot <= 0 or any(values.shape != (n_boot, width) for values in arrays.values())
            or any(values.shape != (width,) for values in points.values())):
        raise ValueError("clean-arm cache draw shape is invalid")
    complete_metadata = {"schema_version": 1, **metadata, "cache_identity_sha256": identity_sha256}
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path, metadata=np.asarray(json.dumps(complete_metadata, sort_keys=True, separators=(",", ":"))),
        int8_clean=arrays["int8_clean"], fp8_clean=arrays["fp8_clean"],
        int8_clean_point=points["int8_clean"], fp8_clean_point=points["fp8_clean"],
    )
    return {"sha256": sha256_file(path), "identity_sha256": identity_sha256}


def dataset_bootstrap_seed(namespace: str, dataset: str) -> int:
    """Derive the one planned bootstrap sequence shared by a dataset's cells."""
    if not isinstance(namespace, str) or not namespace or not isinstance(dataset, str) or not dataset:
        raise ValueError("bootstrap seed namespace and dataset are required")
    value = f"{namespace}|paired-format-dataset|{dataset}"
    return int.from_bytes(hashlib.sha256(value.encode("utf-8")).digest()[:8], "big") % (2 ** 32)


def _labels(count: int) -> tuple[str, ...]:
    if count == 4:
        return ("all", "small", "medium", "large")
    if count == 3:
        # The compact public algebra API intentionally accepts all/small/large.
        return ("all", "small", "large")
    if count == 2:
        return ("small_like", "large_like")
    raise ValueError("AP vectors must contain all/small/large or all/small/medium/large values")


def _named(
    values: Sequence[float], labels: Sequence[str] | None = None, *, allow_nonfinite: bool = False,
) -> dict[str, float]:
    labels = tuple(labels) if labels is not None else _labels(len(values))
    if len(values) != len(labels):
        raise ValueError("AP vectors do not match endpoint labels")
    result = {label: float(value) for label, value in zip(labels, values)}
    if not allow_nonfinite and not all(math.isfinite(value) for value in result.values()):
        raise ValueError("non-finite AP point estimate")
    return result


def compute_contrast_from_aps(
    *,
    int8_clean: Sequence[float],
    fp8_clean: Sequence[float],
    int8_corrupt: Sequence[float],
    fp8_corrupt: Sequence[float],
    labels: Sequence[str] | None = None,
    allow_nonfinite: bool = False,
) -> dict[str, Any]:
    """Return positive-is-worse-for-INT8 format contrasts from matched APs."""
    vectors = [_named(values, labels, allow_nonfinite=allow_nonfinite) for values in (int8_clean, fp8_clean, int8_corrupt, fp8_corrupt)]
    if len({tuple(value) for value in vectors}) != 1:
        raise ValueError("all AP vectors must use identical endpoint labels")
    int8_clean_values, fp8_clean_values, int8_corrupt_values, fp8_corrupt_values = vectors
    delta_q = {name: fp8_clean_values[name] - int8_clean_values[name] for name in int8_clean_values}
    delta_e = {
        name: (int8_clean_values[name] - int8_corrupt_values[name])
        - (fp8_clean_values[name] - fp8_corrupt_values[name])
        for name in int8_clean_values
    }
    if "small" in delta_e:
        delta_psi = delta_e["small"] - delta_e["large"]
    else:
        delta_psi = delta_e["small_like"] - delta_e["large_like"]
    if (not allow_nonfinite
            and (not all(math.isfinite(value) for values in (delta_q, delta_e) for value in values.values())
                 or not math.isfinite(delta_psi))):
        raise ValueError("non-finite contrast point estimate")
    return {"delta_q": delta_q, "delta_e": delta_e, "delta_psi": delta_psi}


def format_draws_from_arm_aps(arm_aps: dict[str, np.ndarray], labels: Sequence[str]) -> dict[str, Any]:
    """Derive paired-format draws from cached AP vectors without re-evaluation."""
    labels = tuple(labels)
    if set(arm_aps) != set(ARM_NAMES):
        raise ValueError("format draws require exactly four named arm AP arrays")
    arrays = {name: np.asarray(values, dtype=float) for name, values in arm_aps.items()}
    shape = arrays[ARM_NAMES[0]].shape
    if len(shape) != 2 or shape[0] <= 0 or shape[1] != len(labels) or any(values.shape != shape for values in arrays.values()):
        raise ValueError("arm AP draw arrays must have one shared non-empty (replicate, endpoint) shape")
    delta_q = arrays["fp8_clean"] - arrays["int8_clean"]
    delta_e = ((arrays["int8_clean"] - arrays["int8_corrupt"])
               - (arrays["fp8_clean"] - arrays["fp8_corrupt"]))
    named_e = {name: delta_e[:, index] for index, name in enumerate(labels)}
    if "small" in named_e:
        delta_psi = named_e["small"] - named_e["large"]
    elif "small_like" in named_e and "large_like" in named_e:
        delta_psi = named_e["small_like"] - named_e["large_like"]
    else:
        raise ValueError("format draw labels must define small/large or small_like/large_like")
    return {
        "delta_q": {name: delta_q[:, index] for index, name in enumerate(labels)},
        "delta_e": named_e,
        "delta_psi": delta_psi,
    }


def _image_ids_sha256(image_ids: list[int]) -> str:
    return hashlib.sha256(json.dumps(image_ids, separators=(",", ":")).encode("utf-8")).hexdigest()


def materialize_bootstrap_schedule(path: Path, image_ids: list[int], *, n_boot: int, seed: int) -> dict[str, Any]:
    """Write one immutable, dataset-level paired bootstrap schedule."""
    if path.exists():
        raise ValueError(f"refusing to overwrite bootstrap schedule: {path}")
    validate_expected_image_count(image_ids, len(image_ids))
    if n_boot <= 0:
        raise ValueError("bootstrap schedule replicate count must be positive")
    rng = np.random.default_rng(seed)
    samples = np.asarray([rng.choice(len(image_ids), size=len(image_ids), replace=True) for _ in range(n_boot)], dtype=np.int32)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(1),
        n_boot=np.asarray(n_boot),
        seed=np.asarray(seed),
        n_images=np.asarray(len(image_ids)),
        image_ids_sha256=np.asarray(_image_ids_sha256(image_ids)),
        samples=samples,
    )
    return {
        "path": str(path), "sha256": sha256_file(path), "n_boot": n_boot, "seed": seed,
        "n_images": len(image_ids), "image_ids_sha256": _image_ids_sha256(image_ids),
    }


def load_bootstrap_schedule(path: Path, image_ids: list[int], *, n_boot: int, seed: int) -> dict[str, Any]:
    """Load and validate an immutable schedule against one image universe."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing/empty bootstrap schedule: {path}")
    expected_hash = _image_ids_sha256(image_ids)
    try:
        with np.load(path, allow_pickle=False) as schedule:
            if set(schedule.files) != {"schema_version", "n_boot", "seed", "n_images", "image_ids_sha256", "samples"}:
                raise ValueError("bootstrap schedule fields are invalid")
            if (int(schedule["schema_version"].item()) != 1 or int(schedule["n_boot"].item()) != n_boot
                    or int(schedule["seed"].item()) != seed or int(schedule["n_images"].item()) != len(image_ids)):
                raise ValueError("bootstrap schedule metadata does not match the planned dataset")
            if str(schedule["image_ids_sha256"].item()) != expected_hash:
                raise ValueError("bootstrap schedule image-ID SHA-256 mismatch")
            samples = np.asarray(schedule["samples"], dtype=np.int32)
    except OSError as exc:
        raise ValueError(f"invalid bootstrap schedule: {path}") from exc
    if samples.shape != (n_boot, len(image_ids)) or np.any(samples < 0) or np.any(samples >= len(image_ids)):
        raise ValueError("bootstrap schedule samples are invalid")
    return {
        "path": str(path), "sha256": sha256_file(path), "n_boot": n_boot, "seed": seed,
        "n_images": len(image_ids), "image_ids_sha256": expected_hash, "samples": samples,
    }


def validate_annotation_binding(annotations: Path, expected_sha256: str) -> str:
    """Refuse evaluation unless the exact source annotation bytes are bound."""
    if not annotations.is_file() or not isinstance(expected_sha256, str) or len(expected_sha256) != 64:
        raise ValueError("annotation SHA-256 binding is missing")
    actual = sha256_file(annotations)
    if actual != expected_sha256:
        raise ValueError("annotation SHA-256 mismatch")
    return actual


def validate_expected_image_count(image_ids: Sequence[int], expected_images: int) -> None:
    if expected_images <= 0 or len(image_ids) != expected_images:
        raise ValueError(f"expected image count mismatch: expected {expected_images}, found {len(image_ids)}")


def tt100k_endpoint_spec() -> tuple[tuple[str, ...], dict[str, tuple[float, float]]]:
    """Return all-AP plus the immutable TT100K height strata."""
    return ("all", "small_like", "large_like"), {"all": (0.0, float("inf")), **HEIGHT_GROUPS}


def validate_linked_inputs(records: Sequence[dict[str, Any]]) -> list[int]:
    """Require every bootstrap arm to name one ordered, unique image universe."""
    if not records:
        raise ValueError("linked inputs are required")
    image_ids = records[0].get("image_ids")
    if not isinstance(image_ids, list) or not image_ids:
        raise ValueError("linked inputs must contain non-empty image IDs")
    if any(not isinstance(image_id, int) for image_id in image_ids) or len(image_ids) != len(set(image_ids)):
        raise ValueError("linked inputs must contain unique image IDs")
    if any(record.get("image_ids") != image_ids for record in records[1:]):
        raise ValueError("linked inputs must contain identical ordered image IDs")
    for record in records:
        supplied_hash = record.get("image_ids_sha256")
        if supplied_hash is not None and supplied_hash != _image_ids_sha256(image_ids):
            raise ValueError("linked input image-ID SHA-256 mismatch")
    return image_ids


def validate_dataset_image_universes(universes: dict[str, Sequence[list[int]]]) -> dict[str, list[int]]:
    """Require every planned cell of a dataset to use one ordered image universe."""
    result: dict[str, list[int]] = {}
    for dataset, image_lists in universes.items():
        if not isinstance(dataset, str) or not dataset or not image_lists:
            raise ValueError("dataset image universes must be non-empty")
        reference = validate_linked_inputs([{"image_ids": list(image_lists[0])}])
        for image_ids in image_lists[1:]:
            candidate = validate_linked_inputs([{"image_ids": list(image_ids)}])
            if candidate != reference:
                raise ValueError(f"cross-cell image universe mismatch: {dataset}")
        result[dataset] = reference
    return result


def percentile(values: np.ndarray) -> list[float]:
    finite = values[np.isfinite(values)]
    if not len(finite):
        return [float("nan")] * 3
    return [float(value) for value in np.percentile(finite, [2.5, 50.0, 97.5])]


def _fork_arm_ap_draw(sample: list[int]) -> list[list[float]]:
    """Evaluate one schedule row in a Linux-forked worker."""
    if _FORK_EVALUATIONS is None:
        raise RuntimeError("fork bootstrap worker was not initialized")
    accumulator = _FORK_ACCUMULATOR
    if accumulator is None:
        from paired_bootstrap import accumulate_ap

        accumulator = accumulate_ap
    return [[float(value) for value in accumulator(evaluation, sample)] for evaluation in _FORK_EVALUATIONS]


def bootstrap_arm_ap_draws(
    evaluations: Sequence[Any], schedule: np.ndarray, *, workers: int = 1,
    accumulator: Callable[[Any, list[int]], Sequence[float]] | None = None,
) -> np.ndarray:
    """Return AP draws in immutable schedule-row order for one or more arms."""
    samples = np.asarray(schedule, dtype=np.int32)
    if samples.ndim != 2 or not samples.shape[0] or not samples.shape[1] or np.any(samples < 0) or workers <= 0:
        raise ValueError("bootstrap arm schedule and worker count must be positive")
    if not evaluations:
        raise ValueError("bootstrap arm AP draws require at least one evaluation")
    sample_rows = (row.tolist() for row in samples)
    if workers == 1:
        if accumulator is None:
            from paired_bootstrap import accumulate_ap

            accumulator = accumulate_ap
        draws = [
            [[float(value) for value in accumulator(evaluation, sample)] for evaluation in evaluations]
            for sample in sample_rows
        ]
    else:
        if "fork" not in mp.get_all_start_methods():
            raise ValueError("parallel paired bootstrap requires a fork-capable platform")
        global _FORK_EVALUATIONS, _FORK_ACCUMULATOR
        _FORK_EVALUATIONS = tuple(evaluations)
        _FORK_ACCUMULATOR = accumulator
        try:
            context = mp.get_context("fork")
            with context.Pool(processes=workers) as pool:
                draws = list(pool.imap(_fork_arm_ap_draw, sample_rows, chunksize=1))
        finally:
            _FORK_EVALUATIONS = None
            _FORK_ACCUMULATOR = None
    values = np.asarray(draws, dtype=float)
    if values.ndim != 3 or values.shape[:2] != (samples.shape[0], len(evaluations)):
        raise RuntimeError("bootstrap arm AP draw shape is invalid")
    return values


def paired_format_bootstrap_draws(
    evaluations: Sequence[Any], labels: Sequence[str], *, n_images: int, n_boot: int, seed: int,
    workers: int = 1, accumulator: Callable[[Any, list[int]], Sequence[float]] | None = None,
    schedule: np.ndarray | None = None,
) -> dict[str, Any]:
    """Compute paired-format draws with one sample shared across all four arms.

    The parallel path uses ``fork`` so workers inherit the already evaluated,
    read-only COCO structures without serializing or rebuilding them.  Results
    are collected in replicate order; the random sample generator remains in
    the parent, making a multi-worker execution algebraically identical to the
    serial bootstrap for a fixed seed.
    """
    if n_images <= 0 or n_boot <= 0 or workers <= 0:
        raise ValueError("bootstrap dimensions and worker count must be positive")
    labels = tuple(labels)
    if len(evaluations) != len(ARM_NAMES):
        raise ValueError("paired format bootstrap requires exactly four evaluations")
    if schedule is None:
        rng = np.random.default_rng(seed)
        schedule = np.asarray([
            rng.choice(n_images, size=n_images, replace=True) for _ in range(n_boot)
        ], dtype=np.int32)
    else:
        schedule = np.asarray(schedule, dtype=np.int32)
        if schedule.shape != (n_boot, n_images):
            raise ValueError("provided paired bootstrap schedule does not match n_boot and n_images")
    arm_draws = bootstrap_arm_ap_draws(evaluations, schedule, workers=workers, accumulator=accumulator)
    return format_draws_from_arm_aps(
        {name: arm_draws[:, index, :] for index, name in enumerate(ARM_NAMES)}, labels,
    )


def _read_json(path: Path, label: str) -> Any:
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing/empty {label}: {path}")
    try:
        return json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"invalid JSON {label}: {path}") from exc


def _read_predictions(path: Path) -> list[dict[str, Any]]:
    data = _read_json(path, "prediction file")
    if not isinstance(data, list) or not data:
        raise ValueError(f"missing/empty prediction file: {path}")
    return data


def _validate_binding(
    name: str, run: dict[str, Any], inputs: dict[str, Any], prediction: Path, input_path: Path, run_path: Path,
) -> dict[str, str]:
    if inputs.get("condition_id") != run.get("condition_id"):
        raise ValueError(f"{name}: input condition does not match run record")
    for field in ("input_manifest_sha256", "input_image_ids_sha256"):
        if not isinstance(run.get(field), str) or not run[field]:
            raise ValueError(f"{name}: missing {field}")
    if inputs.get("input_manifest_sha256") != run.get("input_manifest_sha256"):
        raise ValueError(f"{name}: mismatched input-manifest SHA-256")
    if inputs.get("image_ids_sha256") != run.get("input_image_ids_sha256"):
        raise ValueError(f"{name}: mismatched image-ID SHA-256")
    if run.get("prediction_sha256") != sha256_file(prediction):
        raise ValueError(f"{name}: mismatched prediction SHA-256")
    return {
        "prediction_sha256": sha256_file(prediction),
        "input_record_sha256": sha256_file(input_path),
        "input_manifest_sha256": run["input_manifest_sha256"],
        "image_ids_sha256": inputs["image_ids_sha256"],
        "run_record_sha256": sha256_file(run_path),
    }


def _area_evaluations(annotations: Path, predictions: Sequence[list[dict[str, Any]]], image_ids: list[int]):
    from pycocotools.coco import COCO
    from paired_bootstrap import BINS, build_eval

    gt = COCO(str(annotations))
    return [build_eval(gt, prediction, image_ids) for prediction in predictions], [name for name, _, _ in BINS]


def _height_evaluations(annotations: Path, predictions: Sequence[list[dict[str, Any]]], image_ids: list[int]):
    from topic_c.tt100k_height import HEIGHT_GROUPS as runtime_height_groups, height_evaluation

    ground_truth = _read_json(annotations, "annotation file")
    groups = {"all": (0.0, float("inf")), **runtime_height_groups}
    bins = [(name, lower, upper) for name, (lower, upper) in groups.items()]
    return [height_evaluation(ground_truth, prediction, image_ids, bins) for prediction in predictions], list(groups)


def _as_lists(values: Sequence[np.ndarray]) -> list[list[float]]:
    return [[float(item) for item in value] for value in values]


def validate_linked_arm_bindings(
    *, prediction_paths: Sequence[Path], input_paths: Sequence[Path], run_paths: Sequence[Path],
    expected_images: int, arm_names: Sequence[str],
) -> tuple[list[int], dict[str, dict[str, str]]]:
    """Revalidate linked arm bytes without constructing evaluators or parsing predictions."""
    arm_names = tuple(arm_names)
    if (not arm_names or len(set(arm_names)) != len(arm_names)
            or not (len(prediction_paths) == len(input_paths) == len(run_paths) == len(arm_names))):
        raise ValueError("linked evaluator paths must match one non-empty unique arm sequence")
    inputs = [_read_json(path, "input record") for path in input_paths]
    runs = [_read_json(path, "run record") for path in run_paths]
    if any(not isinstance(value, dict) for value in inputs + runs):
        raise ValueError("input and run records must be JSON objects")
    image_ids = validate_linked_inputs(inputs)
    validate_expected_image_count(image_ids, expected_images)
    if any(not path.is_file() or path.stat().st_size == 0 for path in prediction_paths):
        raise ValueError("missing/empty prediction file")
    bindings = {
        name: _validate_binding(name, run, inputs_value, prediction, input_path, run_path)
        for name, run, inputs_value, prediction, input_path, run_path in zip(arm_names, runs, inputs, prediction_paths, input_paths, run_paths)
    }
    return image_ids, bindings


def build_linked_evaluations(
    *, endpoint: str, annotations: Path, prediction_paths: Sequence[Path], input_paths: Sequence[Path], run_paths: Sequence[Path],
    expected_images: int, annotation_sha256: str, arm_names: Sequence[str],
) -> tuple[list[int], Sequence[Any], list[str], dict[str, dict[str, str]]]:
    """Validate linked arms and build each evaluator exactly once."""
    if endpoint not in {"area", "tt100k-height"}:
        raise ValueError(f"unsupported endpoint: {endpoint}")
    validate_annotation_binding(annotations, annotation_sha256)
    image_ids, bindings = validate_linked_arm_bindings(
        prediction_paths=prediction_paths, input_paths=input_paths, run_paths=run_paths,
        expected_images=expected_images, arm_names=arm_names,
    )
    predictions = [_read_predictions(path) for path in prediction_paths]
    if endpoint == "area":
        evaluations, endpoint_labels = _area_evaluations(annotations, predictions, image_ids)
    else:
        evaluations, endpoint_labels = _height_evaluations(annotations, predictions, image_ids)
    return image_ids, evaluations, endpoint_labels, bindings


def build_contrast_evaluations(
    *, endpoint: str, annotations: Path, prediction_paths: Sequence[Path], input_paths: Sequence[Path], run_paths: Sequence[Path],
    expected_images: int, annotation_sha256: str,
) -> tuple[list[int], Sequence[Any], list[str], dict[str, dict[str, str]]]:
    """Validate one four-arm cell and build each evaluator exactly once."""
    return build_linked_evaluations(
        endpoint=endpoint, annotations=annotations, prediction_paths=prediction_paths, input_paths=input_paths, run_paths=run_paths,
        expected_images=expected_images, annotation_sha256=annotation_sha256, arm_names=ARM_NAMES,
    )


def create_clean_arm_cache(
    *, endpoint: str, annotations: Path, prediction_paths: Sequence[Path], input_paths: Sequence[Path], run_paths: Sequence[Path],
    dataset: str, model: str, expected_images: int, annotation_sha256: str, schedule_path: Path, n_boot: int, seed: int,
    workers: int, output: Path,
) -> dict[str, Any]:
    """Evaluate each model's two clean arms once for reuse by its 12 corruptions."""
    if output.exists():
        raise ValueError(f"refusing to overwrite clean-arm cache: {output}")
    clean_names = ("int8_clean", "fp8_clean")
    image_ids, evaluations, labels, bindings = build_linked_evaluations(
        endpoint=endpoint, annotations=annotations, prediction_paths=prediction_paths, input_paths=input_paths, run_paths=run_paths,
        expected_images=expected_images, annotation_sha256=annotation_sha256, arm_names=clean_names,
    )
    schedule = load_bootstrap_schedule(schedule_path, image_ids, n_boot=n_boot, seed=seed)
    draws = bootstrap_arm_ap_draws(evaluations, schedule["samples"], workers=workers)
    from paired_bootstrap import accumulate_ap

    point = {name: _as_lists([accumulate_ap(evaluation, list(range(len(image_ids))) )])[0] for name, evaluation in zip(clean_names, evaluations)}
    metadata = {
        "dataset": dataset, "model": model, "endpoint_type": endpoint, "n_images": len(image_ids), "n_boot": n_boot,
        "annotation_sha256": annotation_sha256, "labels": labels, "input_hashes": bindings,
        "bootstrap_schedule": {key: value for key, value in schedule.items() if key != "samples"},
    }
    identity = clean_cache_identity(metadata)
    reference = write_clean_arm_cache(
        output, metadata=metadata, identity_sha256=identity,
        draws={"int8_clean": draws[:, 0, :], "fp8_clean": draws[:, 1, :]}, point=point,
    )
    return {"schema_version": 1, **metadata, "cache_identity_sha256": identity, "path": str(output), **reference, "draws": {
        "int8_clean": draws[:, 0, :], "fp8_clean": draws[:, 1, :],
    }, "point": point}


def load_clean_arm_cache(
    path: Path, *, endpoint: str, expected_images: int, annotation_sha256: str, schedule: dict[str, Any],
    expected_sha256: str | None = None, expected_identity_sha256: str | None = None,
    expected_dataset: str | None = None, expected_model: str | None = None,
    expected_input_hashes: dict[str, dict[str, str]] | None = None,
) -> dict[str, Any]:
    """Load a clean cache only when its input, schedule, and endpoint bindings match."""
    if not path.is_file() or path.stat().st_size == 0:
        raise ValueError(f"missing/empty clean-arm cache: {path}")
    if expected_sha256 is not None and sha256_file(path) != expected_sha256:
        raise ValueError(f"clean-arm cache file SHA-256 mismatch: {path}")
    try:
        with np.load(path, allow_pickle=False) as cache:
            if set(cache.files) != CLEAN_CACHE_FIELDS:
                raise ValueError("clean-arm cache fields are invalid")
            metadata = json.loads(str(cache["metadata"].item()))
            draws = {name: np.asarray(cache[name], dtype=float) for name in ("int8_clean", "fp8_clean")}
            point = {name: [float(value) for value in cache[name + "_point"]] for name in ("int8_clean", "fp8_clean")}
    except (OSError, json.JSONDecodeError) as exc:
        raise ValueError(f"invalid clean-arm cache: {path}") from exc
    if (not isinstance(metadata, dict) or set(metadata) != CLEAN_CACHE_METADATA_FIELDS
            or metadata.get("schema_version") != 1 or metadata.get("endpoint_type") != endpoint
            or metadata.get("n_images") != expected_images or metadata.get("annotation_sha256") != annotation_sha256
            or metadata.get("n_boot") != schedule["n_boot"]
            or any(metadata.get("bootstrap_schedule", {}).get(key) != schedule.get(key)
                   for key in ("sha256", "n_boot", "seed", "n_images", "image_ids_sha256"))):
        raise ValueError(f"clean-arm cache binding mismatch: {path}")
    identity_metadata = {key: value for key, value in metadata.items() if key not in {"schema_version", "cache_identity_sha256"}}
    identity = metadata.get("cache_identity_sha256")
    if (not isinstance(identity, str) or clean_cache_identity(identity_metadata) != identity
            or (expected_identity_sha256 is not None and identity != expected_identity_sha256)):
        raise ValueError(f"clean-arm cache identity mismatch: {path}")
    if (expected_dataset is not None and metadata.get("dataset") != expected_dataset) or (
            expected_model is not None and metadata.get("model") != expected_model):
        raise ValueError(f"clean-arm cache task binding mismatch: {path}")
    if expected_input_hashes is not None and metadata.get("input_hashes") != expected_input_hashes:
        raise ValueError(f"clean-arm cache input provenance mismatch: {path}")
    labels = metadata.get("labels")
    if (not isinstance(labels, list) or not labels or any(not isinstance(label, str) or not label for label in labels)
            or any(value.shape != (schedule["n_boot"], len(labels)) for value in draws.values())
            or any(len(value) != len(labels) for value in point.values())):
        raise ValueError(f"clean-arm cache draw shape mismatch: {path}")
    return {
        **metadata, "path": str(path), "sha256": sha256_file(path), "draws": draws, "point": point,
    }


def write_clean_contrast_from_cache(
    clean_cache: dict[str, Any], output: Path, *, annotations: Path, evidence_root: Path,
) -> dict[str, Any]:
    """Write the planned clean cell using already computed two-arm AP draws."""
    if output.exists():
        raise ValueError(f"refusing to overwrite contrast artifact: {output}")
    labels = tuple(clean_cache["labels"])
    arm_aps = {
        "int8_clean": clean_cache["draws"]["int8_clean"],
        "fp8_clean": clean_cache["draws"]["fp8_clean"],
        "int8_corrupt": clean_cache["draws"]["int8_clean"],
        "fp8_corrupt": clean_cache["draws"]["fp8_clean"],
    }
    draws = format_draws_from_arm_aps(arm_aps, labels)
    point = compute_contrast_from_aps(
        int8_clean=clean_cache["point"]["int8_clean"], fp8_clean=clean_cache["point"]["fp8_clean"],
        int8_corrupt=clean_cache["point"]["int8_clean"], fp8_corrupt=clean_cache["point"]["fp8_clean"], labels=labels,
    )
    bindings = {
        "int8_clean": clean_cache["input_hashes"]["int8_clean"],
        "fp8_clean": clean_cache["input_hashes"]["fp8_clean"],
        "int8_corrupt": clean_cache["input_hashes"]["int8_clean"],
        "fp8_corrupt": clean_cache["input_hashes"]["fp8_clean"],
    }
    document = {
        "schema_version": 1,
        "method": "paired image bootstrap; one shared resample across INT8/FP8 × clean/corrupt",
        "endpoint_type": clean_cache["endpoint_type"],
        "annotation": {"path": str(annotations.resolve()), "sha256": clean_cache["annotation_sha256"]},
        "n_images": clean_cache["n_images"], "n_boot": clean_cache["n_boot"],
        "seed": clean_cache["bootstrap_schedule"]["seed"], "point": point,
        "percentile_intervals": {
            "percentiles": [2.5, 50.0, 97.5],
            "delta_q": {name: percentile(values) for name, values in draws["delta_q"].items()},
            "delta_e": {name: percentile(values) for name, values in draws["delta_e"].items()},
            "delta_psi": percentile(draws["delta_psi"]),
        },
        "input_hashes": bindings,
        "bootstrap_schedule": schedule_binding(evidence_root, clean_cache["bootstrap_schedule"]),
        "clean_arm_cache": {
            "path": safe_relative_evidence_path(evidence_root, Path(clean_cache["path"]), "clean-arm cache"),
            "sha256": clean_cache["sha256"], "identity_sha256": clean_cache["cache_identity_sha256"],
        },
        "sign_convention": {
            "delta_q": "AP(fp8, clean) - AP(int8, clean); positive means INT8 has lower clean AP",
            "delta_e": "[AP(int8, clean)-AP(int8, corrupt)] - [AP(fp8, clean)-AP(fp8, corrupt)]; positive means INT8 amplifies corruption more",
            "delta_psi": "delta_e(small) - delta_e(large), or small_like - large_like for TT100K height",
        },
    }
    if clean_cache["endpoint_type"] == "tt100k-height":
        document["height_bins_px"] = {name: {"min": lower, "max": None if math.isinf(upper) else upper} for name, (lower, upper) in HEIGHT_GROUPS.items()}
    document["artifact_sha256"] = canonical_hash(document, "artifact_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def run_contrast(
    *, endpoint: str, annotations: Path, prediction_paths: Sequence[Path], input_paths: Sequence[Path], run_paths: Sequence[Path],
    n_boot: int, seed: int, expected_images: int, annotation_sha256: str, output: Path, workers: int = 1,
    draw_cache: Path | None = None, schedule_path: Path | None = None, clean_arm_cache_path: Path | None = None,
    clean_arm_cache_sha256: str | None = None, clean_arm_cache_identity_sha256: str | None = None,
    dataset: str | None = None, model: str | None = None,
    evidence_root: Path | None = None,
) -> dict[str, Any]:
    if output.exists():
        raise ValueError(f"refusing to overwrite contrast artifact: {output}")
    if n_boot != 2000:
        raise ValueError("n_boot must be exactly 2000")
    if workers <= 0:
        raise ValueError("bootstrap workers must be positive")
    if draw_cache is not None and draw_cache.exists():
        raise ValueError(f"refusing to overwrite draw cache: {draw_cache}")
    if (draw_cache is not None or schedule_path is not None or clean_arm_cache_path is not None) and evidence_root is None:
        raise ValueError("cache-backed contrast requires an evidence project root")
    if ((clean_arm_cache_path is None) != (clean_arm_cache_sha256 is None)
            or (clean_arm_cache_path is None) != (clean_arm_cache_identity_sha256 is None)):
        raise ValueError("clean-arm cache path and precommitted bindings must be supplied together")
    if clean_arm_cache_path is not None and (not isinstance(dataset, str) or not dataset or not isinstance(model, str) or not model):
        raise ValueError("clean-arm cache reuse requires the expected dataset and model")
    clean_cache = None
    if clean_arm_cache_path is None:
        image_ids, evaluations, endpoint_labels, bindings = build_contrast_evaluations(
            endpoint=endpoint, annotations=annotations, prediction_paths=prediction_paths, input_paths=input_paths,
            run_paths=run_paths, expected_images=expected_images, annotation_sha256=annotation_sha256,
        )
        from paired_bootstrap import accumulate_ap

        schedule = (load_bootstrap_schedule(schedule_path, image_ids, n_boot=n_boot, seed=seed)
                    if schedule_path is not None else None)
        full = list(range(len(image_ids)))
        point = compute_contrast_from_aps(
            **dict(zip(ARM_NAMES, _as_lists([accumulate_ap(value, full) for value in evaluations]))), labels=endpoint_labels,
        )
        draws = paired_format_bootstrap_draws(
            evaluations, endpoint_labels, n_images=len(image_ids), n_boot=n_boot, seed=seed, workers=workers,
            schedule=None if schedule is None else schedule["samples"],
        )
    else:
        if schedule_path is None:
            raise ValueError("clean-arm reuse requires a bootstrap schedule")
        corrupt_names = ("int8_corrupt", "fp8_corrupt")
        image_ids, evaluations, endpoint_labels, corrupt_bindings = build_linked_evaluations(
            endpoint=endpoint, annotations=annotations, prediction_paths=prediction_paths[2:], input_paths=input_paths[2:], run_paths=run_paths[2:],
            expected_images=expected_images, annotation_sha256=annotation_sha256, arm_names=corrupt_names,
        )
        from paired_bootstrap import accumulate_ap

        schedule = load_bootstrap_schedule(schedule_path, image_ids, n_boot=n_boot, seed=seed)
        clean_image_ids, clean_bindings = validate_linked_arm_bindings(
            prediction_paths=prediction_paths[:2], input_paths=input_paths[:2], run_paths=run_paths[:2],
            expected_images=expected_images, arm_names=("int8_clean", "fp8_clean"),
        )
        if clean_image_ids != image_ids:
            raise ValueError("clean-arm cache source image universe does not match corrupt evaluation")
        clean_cache = load_clean_arm_cache(
            clean_arm_cache_path, endpoint=endpoint, expected_images=expected_images, annotation_sha256=annotation_sha256,
            schedule=schedule, expected_sha256=clean_arm_cache_sha256,
            expected_identity_sha256=clean_arm_cache_identity_sha256,
            expected_dataset=dataset, expected_model=model, expected_input_hashes=clean_bindings,
        )
        if clean_cache["labels"] != endpoint_labels:
            raise ValueError("clean-arm cache endpoint labels do not match corrupt evaluation")
        corrupt_draws = bootstrap_arm_ap_draws(evaluations, schedule["samples"], workers=workers)
        arm_aps = {
            "int8_clean": clean_cache["draws"]["int8_clean"], "fp8_clean": clean_cache["draws"]["fp8_clean"],
            "int8_corrupt": corrupt_draws[:, 0, :], "fp8_corrupt": corrupt_draws[:, 1, :],
        }
        draws = format_draws_from_arm_aps(arm_aps, endpoint_labels)
        full = list(range(len(image_ids)))
        corrupt_points = _as_lists([accumulate_ap(value, full) for value in evaluations])
        point = compute_contrast_from_aps(
            int8_clean=clean_cache["point"]["int8_clean"], fp8_clean=clean_cache["point"]["fp8_clean"],
            int8_corrupt=corrupt_points[0], fp8_corrupt=corrupt_points[1], labels=endpoint_labels,
        )
        bindings = {**clean_cache["input_hashes"], **corrupt_bindings}
    document = {
        "schema_version": 1,
        "method": "paired image bootstrap; one shared resample across INT8/FP8 × clean/corrupt",
        "endpoint_type": endpoint,
        "annotation": {"path": str(annotations.resolve()), "sha256": annotation_sha256},
        "n_images": len(image_ids),
        "n_boot": n_boot,
        "seed": seed,
        "point": point,
        "percentile_intervals": {
            "percentiles": [2.5, 50.0, 97.5],
            "delta_q": {name: percentile(values) for name, values in draws["delta_q"].items()},
            "delta_e": {name: percentile(values) for name, values in draws["delta_e"].items()},
            "delta_psi": percentile(draws["delta_psi"]),
        },
        "input_hashes": bindings,
        "sign_convention": {
            "delta_q": "AP(fp8, clean) - AP(int8, clean); positive means INT8 has lower clean AP",
            "delta_e": "[AP(int8, clean)-AP(int8, corrupt)] - [AP(fp8, clean)-AP(fp8, corrupt)]; positive means INT8 amplifies corruption more",
            "delta_psi": "delta_e(small) - delta_e(large), or small_like - large_like for TT100K height",
        },
    }
    if schedule is not None:
        document["bootstrap_schedule"] = schedule_binding(evidence_root, schedule)
    if clean_cache is not None:
        document["clean_arm_cache"] = {
            "path": safe_relative_evidence_path(evidence_root, Path(clean_cache["path"]), "clean-arm cache"),
            "sha256": clean_cache["sha256"], "identity_sha256": clean_cache["cache_identity_sha256"],
        }
    if endpoint == "tt100k-height":
        document["height_bins_px"] = {name: {"min": lower, "max": None if math.isinf(upper) else upper} for name, (lower, upper) in HEIGHT_GROUPS.items()}
    if draw_cache is not None:
        if schedule is None:
            raise ValueError("draw cache requires a bootstrap schedule")
        temporary = {
            "path": safe_relative_evidence_path(evidence_root, draw_cache, "draw cache"),
        }
        provisional = {**document, "temporary_draw_cache": temporary}
        identity = component_draw_cache_identity(provisional)
        reference = write_draw_cache(
            draw_cache, identity_sha256=identity, n_boot=n_boot, schedule_sha256=document["bootstrap_schedule"]["sha256"],
            delta_e_all=np.asarray(draws["delta_e"]["all"], dtype=float), delta_psi=np.asarray(draws["delta_psi"], dtype=float),
        )
        document["temporary_draw_cache"] = {**temporary, **reference}
    document["artifact_sha256"] = canonical_hash(document, "artifact_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    return document


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--endpoint", choices=("area", "tt100k-height"), required=True)
    parser.add_argument("--annotations", required=True)
    for arm in ARM_NAMES:
        parser.add_argument("--" + arm.replace("_", "-"), required=True)
        parser.add_argument("--" + arm.replace("_", "-") + "-input", required=True)
        parser.add_argument("--" + arm.replace("_", "-") + "-run", required=True)
    parser.add_argument("--n-boot", type=int, required=True)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--expected-images", type=int, required=True)
    parser.add_argument("--annotation-sha256", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--workers", type=int, default=1)
    parser.add_argument("--draw-cache")
    parser.add_argument("--schedule")
    parser.add_argument("--clean-arm-cache")
    parser.add_argument("--clean-arm-cache-sha256")
    parser.add_argument("--clean-arm-cache-identity-sha256")
    parser.add_argument("--dataset")
    parser.add_argument("--model")
    parser.add_argument("--evidence-root")
    args = parser.parse_args()
    try:
        document = run_contrast(
            endpoint=args.endpoint,
            annotations=Path(args.annotations),
            prediction_paths=[Path(getattr(args, arm)) for arm in ARM_NAMES],
            input_paths=[Path(getattr(args, arm + "_input")) for arm in ARM_NAMES],
            run_paths=[Path(getattr(args, arm + "_run")) for arm in ARM_NAMES],
            n_boot=args.n_boot,
            seed=args.seed,
            expected_images=args.expected_images,
            annotation_sha256=args.annotation_sha256,
            output=Path(args.out),
            workers=args.workers,
            draw_cache=Path(args.draw_cache) if args.draw_cache else None,
            schedule_path=Path(args.schedule) if args.schedule else None,
            clean_arm_cache_path=Path(args.clean_arm_cache) if args.clean_arm_cache else None,
            clean_arm_cache_sha256=args.clean_arm_cache_sha256,
            clean_arm_cache_identity_sha256=args.clean_arm_cache_identity_sha256,
            dataset=args.dataset,
            model=args.model,
            evidence_root=Path(args.evidence_root) if args.evidence_root else None,
        )
    except ValueError as exc:
        raise SystemExit(f"FORMAT CONTRAST REFUSED: {exc}") from exc
    print(json.dumps({"out": args.out, "point": document["point"]}, indent=2))


if __name__ == "__main__":
    main()
