#!/usr/bin/env python3
"""Run a fixed-category-universe Bayesian image-bootstrap sensitivity cell."""
from __future__ import annotations

import argparse
import hashlib
import json
import shlex
import time
from pathlib import Path
from typing import Any

import numpy as np

from fixed_universe_bootstrap import (
    accumulate_prepared_ap,
    bayesian_image_weights,
    fixed_category_area_mask,
    prepare_weighted_ap,
)
from topic_c.manifest import sha256_file


ARM_NAMES = ("int8_clean", "fp8_clean", "int8_corrupt", "fp8_corrupt")


def canonical_hash(document: dict[str, Any], excluded: str | None = None) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()).hexdigest()


def paired_contrast(arms: list[np.ndarray]) -> dict[str, Any]:
    if len(arms) != 4 or len({np.asarray(arm).shape for arm in arms}) != 1:
        raise ValueError("paired contrast requires four shape-matched arms")
    int8_clean, fp8_clean, int8_corrupt, fp8_corrupt = [np.asarray(arm, dtype=float) for arm in arms]
    delta_e = (fp8_corrupt - int8_corrupt) - (fp8_clean - int8_clean)
    return {
        "delta_e": delta_e,
        "delta_psi": float(delta_e[0] - delta_e[-1]) if len(delta_e) > 1 else float("nan"),
    }


def summarize_draws(values: np.ndarray, *, checkpoint: int) -> dict[str, Any]:
    values = np.asarray(values, dtype=float)
    if values.ndim != 1 or not np.isfinite(values).all() or not 0 < checkpoint <= len(values):
        raise ValueError("draw summary inputs are invalid")
    percentiles = [2.5, 50.0, 97.5]
    checkpoint_interval = np.percentile(values[:checkpoint], percentiles)
    full_interval = np.percentile(values, percentiles)
    return {
        "n_boot": len(values),
        "checkpoint_n_boot": checkpoint,
        "percentiles": percentiles,
        "percentile_interval": [float(value) for value in full_interval],
        "checkpoint_percentile_interval": [float(value) for value in checkpoint_interval],
        "max_checkpoint_percentile_shift_native_ap": float(np.max(np.abs(full_interval - checkpoint_interval))),
    }


def evaluate_uniform_point(
    standard_prepared: list[dict[str, Any]],
    height_prepared: list[dict[str, Any]],
    *,
    n_images: int,
) -> tuple[dict[str, Any], dict[str, Any]]:
    uniform = np.ones(n_images, dtype=float)
    overall = paired_contrast(
        [accumulate_prepared_ap(value, uniform) for value in standard_prepared]
    )
    height = paired_contrast(
        [accumulate_prepared_ap(value, uniform) for value in height_prepared]
    )
    return overall, height


def weights_digest(weights: np.ndarray) -> str:
    return hashlib.sha256(np.ascontiguousarray(weights).tobytes()).hexdigest()


def write_weight_schedule(path: Path, *, n_images: int, n_boot: int, seed: int) -> dict[str, Any]:
    path = path.resolve()
    if path.exists():
        raise SystemExit(f"FIXED-UNIVERSE REFUSED: refusing to overwrite schedule: {path}")
    weights = bayesian_image_weights(n_images, n_boot=n_boot, seed=seed).astype(np.float32)
    digest = weights_digest(weights)
    identity = canonical_hash(
        {"schema_version": 1, "method": "bayesian_image_weights_exponential_mean_one", "n_images": n_images,
         "n_boot": n_boot, "seed": seed, "dtype": "float32", "weights_sha256": digest}
    )
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        path,
        schema_version=np.asarray(1),
        method=np.asarray("bayesian_image_weights_exponential_mean_one"),
        n_images=np.asarray(n_images),
        n_boot=np.asarray(n_boot),
        seed=np.asarray(seed),
        dtype=np.asarray("float32"),
        weights_sha256=np.asarray(digest),
        schedule_identity_sha256=np.asarray(identity),
        weights=weights,
    )
    return {
        "path": str(path), "sha256": sha256_file(path), "weights_sha256": digest,
        "schedule_identity_sha256": identity, "n_images": n_images, "n_boot": n_boot, "seed": seed,
    }


def load_weight_schedule(path: Path, *, n_images: int, n_boot: int, seed: int) -> tuple[dict[str, Any], np.ndarray]:
    path = path.resolve()
    with np.load(path, allow_pickle=False) as schedule:
        required = {
            "schema_version", "method", "n_images", "n_boot", "seed", "dtype", "weights_sha256",
            "schedule_identity_sha256", "weights",
        }
        if set(schedule.files) != required:
            raise SystemExit("FIXED-UNIVERSE REFUSED: schedule fields mismatch")
        weights = np.asarray(schedule["weights"], dtype=np.float32)
        metadata = {
            "schema_version": int(schedule["schema_version"].item()),
            "method": str(schedule["method"].item()),
            "n_images": int(schedule["n_images"].item()),
            "n_boot": int(schedule["n_boot"].item()),
            "seed": int(schedule["seed"].item()),
            "dtype": str(schedule["dtype"].item()),
            "weights_sha256": str(schedule["weights_sha256"].item()),
        }
        identity = str(schedule["schedule_identity_sha256"].item())
    if (
        metadata != {"schema_version": 1, "method": "bayesian_image_weights_exponential_mean_one",
                     "n_images": n_images, "n_boot": n_boot, "seed": seed, "dtype": "float32",
                     "weights_sha256": weights_digest(weights)}
        or weights.shape != (n_boot, n_images)
        or np.any(weights <= 0)
        or not np.isfinite(weights).all()
        or identity != canonical_hash(metadata)
    ):
        raise SystemExit("FIXED-UNIVERSE REFUSED: schedule binding mismatch")
    return {
        "path": str(path), "sha256": sha256_file(path), "weights_sha256": metadata["weights_sha256"],
        "schedule_identity_sha256": identity, "n_images": n_images, "n_boot": n_boot, "seed": seed,
    }, weights


def safe_path(root: Path, value: str, label: str) -> Path:
    path = Path(value).resolve()
    if root.resolve() not in path.parents or not path.is_file():
        raise SystemExit(f"FIXED-UNIVERSE REFUSED: unsafe or missing {label}: {path}")
    return path


def command_argument(command: str, option: str) -> str:
    tokens = shlex.split(command)
    if tokens.count(option) != 1:
        raise SystemExit(f"FIXED-UNIVERSE REFUSED: run command lacks unique {option}")
    index = tokens.index(option)
    if index + 1 >= len(tokens):
        raise SystemExit(f"FIXED-UNIVERSE REFUSED: run command lacks value for {option}")
    return tokens[index + 1]


def index_run_records(root: Path) -> dict[str, Path]:
    result: dict[str, Path] = {}
    for path in sorted((root / "manifests" / "runs").glob("**/*.json")):
        digest = sha256_file(path)
        if digest in result:
            raise SystemExit("FIXED-UNIVERSE REFUSED: duplicate run-record bytes")
        result[digest] = path.resolve()
    return result


def resolve_arms(root: Path, component: dict[str, Any]) -> tuple[list[Path], list[Path], list[int], dict[str, Any]]:
    run_index = index_run_records(root)
    prediction_paths: list[Path] = []
    input_paths: list[Path] = []
    input_records: list[dict[str, Any]] = []
    bindings: dict[str, Any] = {}
    for arm in ARM_NAMES:
        source = component["input_hashes"][arm]
        run_path = run_index.get(source["run_record_sha256"])
        if run_path is None:
            raise SystemExit(f"FIXED-UNIVERSE REFUSED: run record unresolved: {arm}")
        run = json.loads(run_path.read_text(encoding="utf-8"))
        prediction = safe_path(root, command_argument(run["command"], "--out"), f"{arm} prediction")
        input_path = safe_path(root, command_argument(run["command"], "--input-record"), f"{arm} input record")
        input_record = json.loads(input_path.read_text(encoding="utf-8"))
        checks = {
            "prediction_sha256": sha256_file(prediction),
            "input_record_sha256": sha256_file(input_path),
            "input_manifest_sha256": input_record.get("input_manifest_sha256"),
            "image_ids_sha256": input_record.get("image_ids_sha256"),
            "run_record_sha256": sha256_file(run_path),
        }
        if checks != source:
            raise SystemExit(f"FIXED-UNIVERSE REFUSED: component/run binding mismatch: {arm}")
        prediction_paths.append(prediction)
        input_paths.append(input_path)
        input_records.append(input_record)
        bindings[arm] = {
            **checks,
            "prediction_path": str(prediction.relative_to(root)),
            "input_record_path": str(input_path.relative_to(root)),
            "run_record_path": str(run_path.relative_to(root)),
        }
    image_ids = input_records[0].get("image_ids")
    if (
        not isinstance(image_ids, list)
        or len(image_ids) != len(set(image_ids))
        or any(record.get("image_ids") != image_ids for record in input_records[1:])
    ):
        raise SystemExit("FIXED-UNIVERSE REFUSED: four arms lack one ordered unique image universe")
    return prediction_paths, input_paths, image_ids, bindings


def _relative_file(root: Path, value: object, label: str) -> Path:
    if not isinstance(value, str) or not value:
        raise SystemExit(f"FIXED-UNIVERSE REFUSED: missing {label}")
    path = (root.resolve() / value).resolve()
    try:
        path.relative_to(root.resolve())
    except ValueError as exc:
        raise SystemExit(f"FIXED-UNIVERSE REFUSED: {label} escapes project root") from exc
    if not path.is_file():
        raise SystemExit(f"FIXED-UNIVERSE REFUSED: missing {label}: {path}")
    return path


def validate_fixed_universe_artifact(
    root: Path,
    report_path: Path,
    *,
    expected_n_boot: int = 10_000,
    expected_images: int = 3_067,
) -> dict[str, Any]:
    """Recompute and hash-check a completed fixed-category sensitivity artifact."""
    root, report_path = root.resolve(), report_path.resolve()
    try:
        report_path.relative_to(root)
        report = json.loads(report_path.read_text(encoding="utf-8"))
    except (ValueError, OSError, json.JSONDecodeError) as exc:
        raise SystemExit("FIXED-UNIVERSE REFUSED: unreadable report under project root") from exc
    if (
        report.get("artifact_sha256") != canonical_hash(report, "artifact_sha256")
        or report.get("schema_version") != 1
        or report.get("method")
        != "paired fixed-category-universe Bayesian image bootstrap with positive image weights"
        or report.get("n_boot") != expected_n_boot
        or report.get("n_images") != expected_images
        or type(report.get("seed")) is not int
    ):
        raise SystemExit("FIXED-UNIVERSE REFUSED: report identity or exact design mismatch")
    component = _relative_file(root, report.get("component"), "component")
    annotation_record = report.get("annotation", {})
    annotation = _relative_file(root, annotation_record.get("path"), "annotation")
    if (
        sha256_file(component) != report.get("component_sha256")
        or sha256_file(annotation) != annotation_record.get("sha256")
    ):
        raise SystemExit("FIXED-UNIVERSE REFUSED: component or annotation hash mismatch")
    component_document = json.loads(component.read_text(encoding="utf-8"))
    point = report.get("point", {})
    ordinary = report.get("ordinary_reference", {})
    if (
        not np.isclose(point.get("delta_e_all"), component_document["point"]["delta_e"]["all"], atol=1e-12, rtol=0)
        or not np.isclose(point.get("delta_psi_height"), component_document["point"]["delta_psi"], atol=1e-12, rtol=0)
        or ordinary.get("n_boot") != component_document.get("n_boot")
        or ordinary.get("delta_e_all") != component_document["percentile_intervals"]["delta_e"]["all"]
        or ordinary.get("delta_psi_height") != component_document["percentile_intervals"]["delta_psi"]
    ):
        raise SystemExit("FIXED-UNIVERSE REFUSED: component estimand reconstruction mismatch")
    schedule_record = report.get("schedule", {})
    schedule_path = _relative_file(root, schedule_record.get("path"), "weight schedule")
    if sha256_file(schedule_path) != schedule_record.get("sha256"):
        raise SystemExit("FIXED-UNIVERSE REFUSED: schedule hash mismatch")
    schedule, _ = load_weight_schedule(
        schedule_path,
        n_images=expected_images,
        n_boot=expected_n_boot,
        seed=report["seed"],
    )
    for key in ("weights_sha256", "schedule_identity_sha256", "n_images", "n_boot", "seed"):
        if schedule.get(key) != schedule_record.get(key):
            raise SystemExit("FIXED-UNIVERSE REFUSED: schedule report binding mismatch")
    image_hashes = set()
    bindings = report.get("input_hashes", {})
    if set(bindings) != set(ARM_NAMES):
        raise SystemExit("FIXED-UNIVERSE REFUSED: four-arm binding grid mismatch")
    for arm in ARM_NAMES:
        binding = bindings[arm]
        prediction = _relative_file(root, binding.get("prediction_path"), f"{arm} prediction")
        inputs = _relative_file(root, binding.get("input_record_path"), f"{arm} input record")
        run = _relative_file(root, binding.get("run_record_path"), f"{arm} run record")
        input_document = json.loads(inputs.read_text(encoding="utf-8"))
        if (
            sha256_file(prediction) != binding.get("prediction_sha256")
            or sha256_file(inputs) != binding.get("input_record_sha256")
            or sha256_file(run) != binding.get("run_record_sha256")
            or input_document.get("input_manifest_sha256") != binding.get("input_manifest_sha256")
            or input_document.get("image_ids_sha256") != binding.get("image_ids_sha256")
        ):
            raise SystemExit(f"FIXED-UNIVERSE REFUSED: four-arm hash mismatch: {arm}")
        image_hashes.add(binding.get("image_ids_sha256"))
    if len(image_hashes) != 1 or None in image_hashes:
        raise SystemExit("FIXED-UNIVERSE REFUSED: four arms do not share one image universe")
    draws_record = report.get("draws", {})
    draws_path = _relative_file(root, draws_record.get("path"), "draw cache")
    if sha256_file(draws_path) != draws_record.get("sha256"):
        raise SystemExit("FIXED-UNIVERSE REFUSED: draw cache hash mismatch")
    try:
        with np.load(draws_path, allow_pickle=False) as cache:
            if set(cache.files) != {
                "schema_version", "schedule_identity_sha256", "delta_e_all", "delta_psi_height"
            }:
                raise ValueError("fields")
            schema_version = int(cache["schema_version"].item())
            schedule_identity = str(cache["schedule_identity_sha256"].item())
            delta_e = np.asarray(cache["delta_e_all"], dtype=float)
            delta_psi = np.asarray(cache["delta_psi_height"], dtype=float)
    except (OSError, ValueError) as exc:
        raise SystemExit("FIXED-UNIVERSE REFUSED: invalid draw cache") from exc
    if (
        schema_version != 1
        or schedule_identity != schedule["schedule_identity_sha256"]
        or delta_e.shape != (expected_n_boot,)
        or delta_psi.shape != (expected_n_boot,)
        or not np.isfinite(delta_e).all()
        or not np.isfinite(delta_psi).all()
    ):
        raise SystemExit("FIXED-UNIVERSE REFUSED: draw cache binding mismatch")
    fixed = report.get("fixed_universe", {})
    for key, values in (("delta_e_all", delta_e), ("delta_psi_height", delta_psi)):
        stored = fixed.get(key, {})
        checkpoint = stored.get("checkpoint_n_boot")
        if type(checkpoint) is not int or summarize_draws(values, checkpoint=checkpoint) != stored:
            raise SystemExit(f"FIXED-UNIVERSE REFUSED: recomputed summary mismatch: {key}")
    return report


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--component", required=True)
    parser.add_argument("--n-boot", type=int, required=True)
    parser.add_argument("--checkpoint", type=int, default=2000)
    parser.add_argument("--seed", type=int, required=True)
    parser.add_argument("--schedule-out", required=True)
    parser.add_argument("--draws-out", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()
    root = Path(args.project_root).resolve()
    component_path = safe_path(root, args.component, "component")
    output, draws_out = Path(args.out).resolve(), Path(args.draws_out).resolve()
    if output.exists() or draws_out.exists():
        raise SystemExit("FIXED-UNIVERSE REFUSED: refusing to overwrite output")
    component = json.loads(component_path.read_text(encoding="utf-8"))
    if component.get("endpoint_type") != "tt100k-height" or component.get("n_images") != 3067:
        raise SystemExit("FIXED-UNIVERSE REFUSED: expected one validated TT100K direct component")
    if not 0 < args.checkpoint <= args.n_boot:
        raise SystemExit("FIXED-UNIVERSE REFUSED: invalid Monte Carlo checkpoint")
    annotation = safe_path(root, component["annotation"]["path"], "annotation")
    if sha256_file(annotation) != component["annotation"]["sha256"]:
        raise SystemExit("FIXED-UNIVERSE REFUSED: annotation hash mismatch")
    prediction_paths, _, image_ids, bindings = resolve_arms(root, component)

    schedule_path = Path(args.schedule_out)
    if schedule_path.exists():
        schedule, weights = load_weight_schedule(
            schedule_path, n_images=len(image_ids), n_boot=args.n_boot, seed=args.seed
        )
    else:
        schedule = write_weight_schedule(
            schedule_path, n_images=len(image_ids), n_boot=args.n_boot, seed=args.seed
        )
        _, weights = load_weight_schedule(
            schedule_path, n_images=len(image_ids), n_boot=args.n_boot, seed=args.seed
        )

    from paired_bootstrap import build_eval
    from pycocotools.coco import COCO
    from topic_c.tt100k_height import HEIGHT_GROUPS, height_evaluation

    prediction_documents = [json.loads(path.read_text(encoding="utf-8")) for path in prediction_paths]
    gt = COCO(str(annotation))
    standard = [build_eval(gt, predictions, image_ids) for predictions in prediction_documents]
    annotation_document = json.loads(annotation.read_text(encoding="utf-8"))
    height_bins = [(name, low, high) for name, (low, high) in HEIGHT_GROUPS.items()]
    height = [height_evaluation(annotation_document, predictions, image_ids, height_bins) for predictions in prediction_documents]
    standard_prepared = [prepare_weighted_ap(value, fixed_category_area_mask(value)) for value in standard]
    height_prepared = [prepare_weighted_ap(value, fixed_category_area_mask(value)) for value in height]
    overall_point, height_point = evaluate_uniform_point(
        standard_prepared, height_prepared, n_images=len(image_ids)
    )
    if (
        not np.isclose(overall_point["delta_e"][0], component["point"]["delta_e"]["all"], atol=1e-12, rtol=0)
        or not np.isclose(height_point["delta_psi"], component["point"]["delta_psi"], atol=1e-12, rtol=0)
    ):
        raise SystemExit("FIXED-UNIVERSE REFUSED: uniform point does not reconstruct direct component")

    started = time.monotonic()
    delta_e = np.empty(args.n_boot, dtype=float)
    delta_psi = np.empty(args.n_boot, dtype=float)
    for index, draw_weights in enumerate(weights):
        delta_e[index] = paired_contrast(
            [accumulate_prepared_ap(value, draw_weights) for value in standard_prepared]
        )["delta_e"][0]
        delta_psi[index] = paired_contrast(
            [accumulate_prepared_ap(value, draw_weights) for value in height_prepared]
        )["delta_psi"]
        if (index + 1) % 25 == 0 or index + 1 == args.n_boot:
            print(f"FIXED-UNIVERSE BOOTSTRAP {index + 1}/{args.n_boot}", flush=True)
    runtime = time.monotonic() - started
    draws_out.parent.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(
        draws_out,
        schema_version=np.asarray(1),
        schedule_identity_sha256=np.asarray(schedule["schedule_identity_sha256"]),
        delta_e_all=delta_e,
        delta_psi_height=delta_psi,
    )
    report = {
        "schema_version": 1,
        "method": "paired fixed-category-universe Bayesian image bootstrap with positive image weights",
        "scope": "TT100K sensitivity estimand; not a replacement for the ordinary multinomial image bootstrap",
        "component": str(component_path.relative_to(root)),
        "component_sha256": sha256_file(component_path),
        "annotation": {"path": str(annotation.relative_to(root)), "sha256": sha256_file(annotation)},
        "n_images": len(image_ids),
        "n_boot": args.n_boot,
        "seed": args.seed,
        "schedule": {**schedule, "path": str(Path(schedule["path"]).relative_to(root))},
        "input_hashes": bindings,
        "point": {"delta_e_all": float(overall_point["delta_e"][0]), "delta_psi_height": height_point["delta_psi"]},
        "fixed_universe": {
            "delta_e_all": summarize_draws(delta_e, checkpoint=args.checkpoint),
            "delta_psi_height": summarize_draws(delta_psi, checkpoint=args.checkpoint),
        },
        "ordinary_reference": {
            "n_boot": component["n_boot"],
            "delta_e_all": component["percentile_intervals"]["delta_e"]["all"],
            "delta_psi_height": component["percentile_intervals"]["delta_psi"],
        },
        "draws": {"path": str(draws_out.relative_to(root)), "sha256": sha256_file(draws_out)},
        "runtime_seconds_draws_only": runtime,
    }
    report["artifact_sha256"] = canonical_hash(report, "artifact_sha256")
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(report, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "n_boot": args.n_boot, "runtime_seconds": runtime,
                      "delta_e": report["fixed_universe"]["delta_e_all"],
                      "delta_psi": report["fixed_universe"]["delta_psi_height"]}, indent=2))


if __name__ == "__main__":
    main()
