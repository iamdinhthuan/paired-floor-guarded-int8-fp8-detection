#!/usr/bin/env python3
"""Audit the unrounded clean-control substitution component of Delta E.

This is a descriptive reuse of fixed AP ledgers.  It creates no interval and
does not attribute a shift to the codec when executable-component identity is
unknown or differs between the original-source and JPEG-95 clean records.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from pathlib import Path
from typing import Any


DATASETS = ("coco", "voc", "kitti", "tt100k")
MODELS = ("yolo11n", "yolo11m", "yolo11x")
PRECISIONS = ("fp32", "int8-entropy", "fp8")
CORRUPTIONS = ("gaussian_noise", "motion_blur", "fog", "jpeg")
SEVERITIES = (1, 3, 5)
QUANTIZED = ("int8-entropy", "fp8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as handle:
        for chunk in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def canonical_hash(document: dict[str, Any], excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    encoded = json.dumps(payload, sort_keys=True, separators=(",", ":")).encode()
    return hashlib.sha256(encoded).hexdigest()


def identity_status(left: str | None, right: str | None) -> str:
    if left is None or right is None:
        return "unknown"
    return "verified_same" if left == right else "verified_mismatch"


def _float(row: dict[str, Any], field: str, *, native_ap: bool = False) -> float:
    try:
        value = float(row[field])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"finite {field} required") from exc
    if not math.isfinite(value):
        raise ValueError(f"finite {field} required")
    if native_ap and not -1 <= value <= 1:
        raise ValueError(f"{field} is outside the native AP scale")
    return value


def _validate_unique(rows: list[dict[str, Any]], fields: tuple[str, ...], expected: set[tuple], label: str) -> None:
    try:
        keys = [tuple(row[field] if field != "severity" else int(row[field]) for field in fields)
                for row in rows]
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError(f"complete unique {label} grid required") from exc
    if len(keys) != len(expected) or len(set(keys)) != len(keys) or set(keys) != expected:
        raise ValueError(f"complete unique {label} grid required")


def _sign(value: float) -> str:
    if math.isclose(value, 0.0, rel_tol=0, abs_tol=1e-12):
        return "zero"
    return "positive" if value > 0 else "negative" if value < 0 else "zero"


def analyze(codec_rows: list[dict[str, Any]], direct_rows: list[dict[str, Any]]):
    expected_codec = set((d, m, p) for d in DATASETS for m in MODELS for p in PRECISIONS)
    expected_direct = set((d, m, c, s) for d in DATASETS for m in MODELS
                          for c in CORRUPTIONS for s in SEVERITIES)
    _validate_unique(codec_rows, ("dataset", "model", "precision"), expected_codec, "codec")
    _validate_unique(direct_rows, ("dataset", "model", "corruption", "severity"),
                     expected_direct, "direct-cell")

    codec: dict[tuple[str, str, str], dict[str, float]] = {}
    for row in codec_rows:
        original = _float(row, "original_clean_ap", native_ap=True)
        j95 = _float(row, "codec_clean_ap", native_ap=True)
        if original < 0 or j95 < 0:
            raise ValueError("clean AP must be on the native AP scale [0,1]")
        reported = _float(row, "codec_minus_original")
        calculated = 100 * (j95 - original)
        if not math.isclose(reported, calculated, rel_tol=0, abs_tol=1e-10):
            raise ValueError("codec-minus-original identity mismatch")
        codec[(row["dataset"], row["model"], row["precision"])] = {
            "original": original, "j95": j95, "delta_ap": calculated,
        }

    direct_by_block: dict[tuple[str, str], list[dict[str, Any]]] = {
        (d, m): [] for d in DATASETS for m in MODELS
    }
    for row in direct_rows:
        delta_e = _float(row, "delta_e_all", native_ap=True)
        delta_q = _float(row, "delta_q_all", native_ap=True)
        direct_by_block[(row["dataset"], row["model"])].append({**row,
            "severity": int(row["severity"]), "delta_e_all": delta_e,
            "delta_q_all": delta_q})

    blocks, cells = [], []
    for dataset in DATASETS:
        for model in MODELS:
            int8 = codec[(dataset, model, "int8-entropy")]
            fp8 = codec[(dataset, model, "fp8")]
            original_gap = fp8["original"] - int8["original"]
            j95_gap = fp8["j95"] - int8["j95"]
            shift_ap = int8["delta_ap"] - fp8["delta_ap"]
            block_cells = sorted(direct_by_block[(dataset, model)],
                                 key=lambda row: (CORRUPTIONS.index(row["corruption"]), row["severity"]))
            for row in block_cells:
                if not math.isclose(row["delta_q_all"], j95_gap, rel_tol=0, abs_tol=1e-10):
                    raise ValueError(f"J95 clean-gap identity mismatch: {dataset}/{model}")
                j95_delta = 100 * row["delta_e_all"]
                corrupted_gap = j95_delta + 100 * j95_gap
                original_delta = corrupted_gap - 100 * original_gap
                identity_error = j95_delta - original_delta - shift_ap
                if not math.isclose(identity_error, 0, rel_tol=0, abs_tol=1e-10):
                    raise ValueError(f"four-arm arithmetic mismatch: {dataset}/{model}")
                sign_j95, sign_original = _sign(j95_delta), _sign(original_delta)
                cells.append({
                    "dataset": dataset, "model": model,
                    "corruption": row["corruption"], "severity": row["severity"],
                    "delta_e_j95_ap": j95_delta,
                    "delta_e_original_ap": original_delta,
                    "delta_e_j95_minus_original_ap": shift_ap,
                    "original_clean_gap_ap": 100 * original_gap,
                    "j95_clean_gap_ap": 100 * j95_gap,
                    "corrupted_gap_reconstructed_ap": corrupted_gap,
                    "sign_j95": sign_j95, "sign_original": sign_original,
                    "strict_sign_disagreement": sign_j95 != "zero" and sign_original != "zero"
                                                and sign_j95 != sign_original,
                    "four_arm_identity_error_ap": identity_error,
                })
            blocks.append({
                "dataset": dataset, "model": model, "cells": len(block_cells),
                "int8_original_clean_ap": 100 * int8["original"],
                "int8_j95_clean_ap": 100 * int8["j95"],
                "int8_j95_minus_original_ap": int8["delta_ap"],
                "fp8_original_clean_ap": 100 * fp8["original"],
                "fp8_j95_clean_ap": 100 * fp8["j95"],
                "fp8_j95_minus_original_ap": fp8["delta_ap"],
                "original_clean_gap_ap": 100 * original_gap,
                "j95_clean_gap_ap": 100 * j95_gap,
                "delta_e_j95_mean_ap": sum(100 * r["delta_e_all"] for r in block_cells) / len(block_cells),
                "delta_e_original_mean_ap": sum(c["delta_e_original_ap"] for c in cells[-len(block_cells):]) / len(block_cells),
                "delta_e_j95_minus_original_ap": shift_ap,
                "abs_shift_ap": abs(shift_ap),
            })

    inventory = {
        "cells": len(cells),
        "j95_positive": sum(row["sign_j95"] == "positive" for row in cells),
        "j95_negative": sum(row["sign_j95"] == "negative" for row in cells),
        "j95_zero": sum(row["sign_j95"] == "zero" for row in cells),
        "original_positive": sum(row["sign_original"] == "positive" for row in cells),
        "original_negative": sum(row["sign_original"] == "negative" for row in cells),
        "original_zero": sum(row["sign_original"] == "zero" for row in cells),
        "strict_nonzero_both": sum(row["sign_j95"] != "zero" and row["sign_original"] != "zero"
                                   for row in cells),
        "strict_sign_disagreements": sum(row["strict_sign_disagreement"] for row in cells),
        "j95_positive_to_original_negative": sum(row["sign_j95"] == "positive"
                                                   and row["sign_original"] == "negative" for row in cells),
        "j95_negative_to_original_positive": sum(row["sign_j95"] == "negative"
                                                   and row["sign_original"] == "positive" for row in cells),
    }
    shifts = [row["delta_e_j95_minus_original_ap"] for row in blocks]
    summary = {
        "schema_version": 1,
        "scope": "Post hoc unrounded clean-control substitution audit; no new interval or inference.",
        "formula": "DeltaE_J95 - DeltaE_original = delta_INT8 - delta_FP8",
        "macro": {
            "delta_e_j95_mean_ap": sum(row["delta_e_j95_ap"] for row in cells) / len(cells),
            "delta_e_original_mean_ap": sum(row["delta_e_original_ap"] for row in cells) / len(cells),
            "j95_minus_original_mean_ap": sum(shifts) / len(shifts),
            "mean_absolute_block_shift_ap": sum(abs(value) for value in shifts) / len(shifts),
            "min_block_shift_ap": min(shifts), "max_block_shift_ap": max(shifts),
        },
        "sign_inventory": inventory,
        "dependency_note": "Each block-level clean-control shift is reused by 12 cells; these are not 12 independent codec observations.",
        "uncertainty": "None: fixed-ledger point audit only.",
    }
    return summary, blocks, cells


def _read_csv(path: Path) -> list[dict[str, Any]]:
    with path.open(newline="", encoding="utf-8") as handle:
        return list(csv.DictReader(handle))


def _metric_index(directory: Path, dataset: str, corruption: str) -> dict[tuple[str, str], tuple[Path, dict]]:
    result = {}
    for path in directory.glob("*.json"):
        document = json.loads(path.read_text())
        if (document.get("dataset") != dataset or document.get("corruption") != corruption
                or int(document.get("severity", -1)) != 0):
            continue
        precision = str(document["precision"])
        if precision.startswith("int8"):
            precision = "int8-entropy"
        elif precision.startswith("fp8"):
            precision = "fp8"
        elif precision.startswith("fp32"):
            precision = "fp32"
        key = (document["model"], precision)
        if key in result:
            raise ValueError(f"duplicate metric binding: {directory}/{key}")
        result[key] = (path, document)
    return result


def _aggregate_status(values: list[str]) -> str:
    if "verified_mismatch" in values:
        return "verified_mismatch"
    if "unknown" in values:
        return "unknown"
    return "verified_same"


def audit_component_bindings(root: Path, codec_rows: list[dict[str, Any]]) -> dict[tuple[str, str], dict[str, Any]]:
    legacy = root / "artifacts/four_dataset_pilot_v1"
    sources = {
        "coco": (root / "outputs/metrics/coco_uniform_p0_v1", root / "manifests/runs/coco_uniform_p0_v1"),
        "voc": (legacy / "outputs/metrics/voc_pilot_117_v1", legacy / "manifests/runs/voc_pilot_117_v1"),
        "kitti": (legacy / "outputs/metrics/kitti_pilot_117_v1", legacy / "manifests/runs/kitti_pilot_117_v1"),
        "tt100k": (legacy / "outputs/metrics/tt100k_pilot_117_v1", legacy / "manifests/runs/tt100k_pilot_117_v1"),
    }
    codec_metric_dir = root / "outputs/metrics/codec_control_p0_v1"
    codec_run_dir = root / "manifests/runs/codec_control_p0_v1"
    csv_index = {(r["dataset"], r["model"], r["precision"]): r for r in codec_rows}
    result = {}
    components = ("engine_sha256", "runner_sha256", "preprocess_sha256", "decoder_sha256",
                  "class_map_sha256", "annotation_sha256")
    for dataset in DATASETS:
        original_metric_dir, original_run_dir = sources[dataset]
        original = _metric_index(original_metric_dir, dataset, "clean")
        j95 = _metric_index(codec_metric_dir, dataset, "codec_control")
        for model in MODELS:
            pair_data: dict[str, Any] = {}
            per_component = {field: [] for field in components}
            for precision in QUANTIZED:
                try:
                    original_path, original_metric = original[(model, precision)]
                    j95_path, j95_metric = j95[(model, precision)]
                except KeyError as exc:
                    raise ValueError(f"missing metric binding: {dataset}/{model}/{precision}") from exc
                row = csv_index[(dataset, model, precision)]
                if (not math.isclose(float(original_metric["stats"]["AP"]), float(row["original_clean_ap"]), abs_tol=1e-14)
                        or not math.isclose(float(j95_metric["stats"]["AP"]), float(row["codec_clean_ap"]), abs_tol=1e-14)):
                    raise ValueError(f"clean AP source binding mismatch: {dataset}/{model}/{precision}")
                original_run_path = original_run_dir / original_path.name
                j95_run_path = codec_run_dir / j95_path.name
                original_run = json.loads(original_run_path.read_text())
                j95_run = json.loads(j95_run_path.read_text())
                if (sha256_file(original_run_path) != original_metric["run_record_sha256"]
                        or sha256_file(j95_run_path) != j95_metric["run_record_sha256"]):
                    raise ValueError(f"run-record source binding mismatch: {dataset}/{model}/{precision}")
                prefix = "int8" if precision.startswith("int8") else "fp8"
                pair_data.update({
                    f"original_{prefix}_metric_sha256": sha256_file(original_path),
                    f"j95_{prefix}_metric_sha256": sha256_file(j95_path),
                    f"original_{prefix}_run_sha256": sha256_file(original_run_path),
                    f"j95_{prefix}_run_sha256": sha256_file(j95_run_path),
                })
                for field in components:
                    per_component[field].append(identity_status(original_run.get(field), j95_run.get(field)))
            for field, values in per_component.items():
                pair_data[field.removesuffix("_sha256") + "_identity"] = _aggregate_status(values)
            statuses = list(pair_data[key] for key in pair_data if key.endswith("_identity"))
            pair_data["complete_treatment_identity"] = _aggregate_status(statuses)
            result[(dataset, model)] = pair_data
    return result


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _write_tex(table: Path, values: Path, blocks: list[dict[str, Any]], summary: dict[str, Any]) -> None:
    table.parent.mkdir(parents=True, exist_ok=True)
    lines = [r"\begin{tabular}{llrrrrl}", r"\toprule",
             r"Dataset & Model & $\delta_{\rm INT8}$ & $\delta_{\rm FP8}$ & $\Delta E_{J95}$ & $\Delta E_{orig}$ & Binding\\",
             r"\midrule"]
    for row in blocks:
        binding = "same components" if row["complete_treatment_identity"] == "verified_same" else "runner differs"
        lines.append(f"{row['dataset'].upper()} & {row['model']} & {row['int8_j95_minus_original_ap']:+.3f} & "
                     f"{row['fp8_j95_minus_original_ap']:+.3f} & {row['delta_e_j95_mean_ap']:+.3f} & "
                     f"{row['delta_e_original_mean_ap']:+.3f} & {binding}\\\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    table.write_text("\n".join(lines) + "\n")
    macro = summary["macro"]
    signs = summary["sign_inventory"]
    value_lines = ["% Generated from the unrounded clean-control substitution audit.",
        rf"\newcommand{{\CodecDeltaEJMean}}{{{macro['delta_e_j95_mean_ap']:+.3f}}}",
        rf"\newcommand{{\CodecDeltaEOriginalMean}}{{{macro['delta_e_original_mean_ap']:+.3f}}}",
        rf"\newcommand{{\CodecInteractionShift}}{{{macro['j95_minus_original_mean_ap']:+.3f}}}",
        rf"\newcommand{{\CodecInteractionAbsMean}}{{{macro['mean_absolute_block_shift_ap']:.3f}}}",
        rf"\newcommand{{\CodecInteractionMin}}{{{macro['min_block_shift_ap']:+.3f}}}",
        rf"\newcommand{{\CodecInteractionMax}}{{{macro['max_block_shift_ap']:+.3f}}}",
        rf"\newcommand{{\CodecSignDisagreements}}{{{signs['strict_sign_disagreements']}}}",
        rf"\newcommand{{\CodecSignDenominator}}{{{signs['strict_nonzero_both']}}}"]
    values.write_text("\n".join(value_lines) + "\n")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--output", type=Path, required=True)
    parser.add_argument("--table", type=Path, required=True)
    parser.add_argument("--values", type=Path, required=True)
    args = parser.parse_args()
    root = args.root.resolve()
    codec_path = root / "paper/generated/codec_sensitivity.csv"
    direct_path = root / "paper/generated/direct_format_contrast_cells.csv"
    audit_path = root / "paper/generated/direct_evidence_audit.json"
    source_audit = json.loads(audit_path.read_text())
    for path in (codec_path, direct_path):
        expected = source_audit["generated_artifacts_sha256"].get(str(path.relative_to(root)))
        if expected != sha256_file(path):
            raise ValueError(f"source evidence hash mismatch: {path}")
    codec_rows, direct_rows = _read_csv(codec_path), _read_csv(direct_path)
    summary, blocks, cells = analyze(codec_rows, direct_rows)
    bindings = audit_component_bindings(root, codec_rows)
    for block in blocks:
        block.update(bindings[(block["dataset"], block["model"])])
    status_counts = {}
    for block in blocks:
        status = block["complete_treatment_identity"]
        status_counts[status] = status_counts.get(status, 0) + 1
    summary["component_identity"] = {
        "block_status_counts": status_counts,
        "engine_verified_same_blocks": sum(b["engine_identity"] == "verified_same" for b in blocks),
        "runner_verified_same_blocks": sum(b["runner_identity"] == "verified_same" for b in blocks),
        "runner_verified_mismatch_blocks": sum(b["runner_identity"] == "verified_mismatch" for b in blocks),
        "interpretation": "Engine identity is verified in all blocks. Runner identity differs in nine transfer blocks, so those shifts are empirical clean-control substitutions, not codec-isolated effects.",
    }
    args.output.mkdir(parents=True, exist_ok=True)
    blocks_path, cells_path = args.output / "blocks.csv", args.output / "cells.csv"
    _write_csv(blocks_path, blocks)
    _write_csv(cells_path, cells)
    _write_tex(args.table, args.values, blocks, summary)
    summary.update({
        "source_sha256": {str(p.relative_to(root)): sha256_file(p)
                          for p in (codec_path, direct_path, audit_path)},
        "implementation_sha256": sha256_file(Path(__file__)),
        "blocks_sha256": sha256_file(blocks_path), "cells_sha256": sha256_file(cells_path),
        "table_sha256": sha256_file(args.table), "values_sha256": sha256_file(args.values),
    })
    summary["summary_sha256"] = canonical_hash(summary, "summary_sha256")
    (args.output / "summary.json").write_text(json.dumps(summary, indent=2) + "\n")
    print(json.dumps(summary, indent=2))


if __name__ == "__main__":
    main()
