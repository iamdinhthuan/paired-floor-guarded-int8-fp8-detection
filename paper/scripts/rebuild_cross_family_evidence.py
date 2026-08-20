#!/usr/bin/env python3
"""Canonicalize the cross-family interaction sign from distributed ledgers.

Sign convention used throughout the manuscript:
  Q = AP_FP32 - AP_quantized
  E = Q_corrupt - Q_clean
  DeltaE = E_INT8 - E_FP8
         = (FP8-INT8)_corrupt - (FP8-INT8)_clean
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

MODEL_ORDER = ["rtdetr-l", "retinanet-r50-fpn-v2"]
FORMAT_ORDER = ["int8-entropy", "fp8"]
MODEL_LABEL = {"rtdetr-l": "RT-DETR-L", "retinanet-r50-fpn-v2": "RetinaNet-R50-FPN-v2"}
FORMAT_LABEL = {"int8-entropy": "INT8", "fp8": "FP8"}


def sha256(path: Path) -> str:
    h = hashlib.sha256()
    with path.open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            h.update(block)
    return h.hexdigest()


def signed(value: float) -> str:
    return f"{value:+.2f}"


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path(__file__).resolve().parents[1])
    args = parser.parse_args()
    root = args.root.resolve()

    cells_path = root / "generated/cross_family_cells.csv"
    q95_path = root / "generated/cross_family_q95_clean.csv"
    cells = pd.read_csv(cells_path)
    q95 = pd.read_csv(q95_path)

    expected_cells = {"dataset", "model", "precision", "corruption", "severity", "ap"}
    expected_clean = {"dataset", "model", "precision", "ap"}
    if not expected_cells.issubset(cells.columns) or not expected_clean.issubset(q95.columns):
        raise ValueError("Cross-family input schema mismatch")

    corrupt = cells[cells["corruption"] != "clean"].copy()
    if len(corrupt) != 216:
        raise ValueError(f"Expected 216 corrupted format arms, found {len(corrupt)}")

    clean_pivot = q95.pivot(index=["dataset", "model"], columns="precision", values="ap")
    corrupt_pivot = corrupt.pivot(
        index=["dataset", "model", "corruption", "severity"], columns="precision", values="ap"
    ).reset_index()
    required_precisions = {"fp32", "int8-entropy", "fp8"}
    if not required_precisions.issubset(corrupt_pivot.columns):
        raise ValueError("Cross-family corrupted cells lack a required precision")

    rows = []
    for _, row in corrupt_pivot.iterrows():
        ds, model = row["dataset"], row["model"]
        clean = clean_pivot.loc[(ds, model)]
        for precision in FORMAT_ORDER:
            q95_gap = float(clean["fp32"] - clean[precision])
            qc_gap = float(row["fp32"] - row[precision])
            e = qc_gap - q95_gap
            rows.append(
                {
                    "dataset": ds,
                    "model": model,
                    "precision": precision,
                    "corruption": row["corruption"],
                    "severity": int(row["severity"]),
                    "ap_fp32_clean": float(clean["fp32"]),
                    "ap_q_clean": float(clean[precision]),
                    "ap_fp32_corrupt": float(row["fp32"]),
                    "ap_q_corrupt": float(row[precision]),
                    "q95": q95_gap,
                    "qc": qc_gap,
                    "e": e,
                }
            )
    interactions = pd.DataFrame(rows)
    if len(interactions) != 144:
        raise ValueError(f"Expected 144 quantized interaction rows, found {len(interactions)}")

    direct = interactions.pivot(
        index=["dataset", "model", "corruption", "severity"], columns="precision", values="e"
    ).reset_index()
    direct["delta_e"] = direct["int8-entropy"] - direct["fp8"]
    direct = direct.rename(columns={"int8-entropy": "e_int8", "fp8": "e_fp8"})

    summary = []
    for model in MODEL_ORDER:
        for precision in FORMAT_ORDER:
            block = interactions[(interactions["model"] == model) & (interactions["precision"] == precision)]
            summary.append(
                {
                    "model": model,
                    "precision": precision,
                    "cells": int(len(block)),
                    "mean_q95_ap_points": float(100 * block["q95"].mean()),
                    "mean_e_ap_points": float(100 * block["e"].mean()),
                    "min_e_ap_points": float(100 * block["e"].min()),
                    "max_e_ap_points": float(100 * block["e"].max()),
                    "mean_corrupted_ap_points": float(100 * block["ap_q_corrupt"].mean()),
                    "corrupted_cells_below_5_ap": int((block["ap_q_corrupt"] < 0.05).sum()),
                }
            )

    direct_summary = []
    for model in MODEL_ORDER:
        block = direct[direct["model"] == model]
        direct_summary.append(
            {
                "model": model,
                "cells": int(len(block)),
                "mean_delta_e_ap_points": float(100 * block["delta_e"].mean()),
                "min_delta_e_ap_points": float(100 * block["delta_e"].min()),
                "max_delta_e_ap_points": float(100 * block["delta_e"].max()),
            }
        )

    expected = {
        ("rtdetr-l", "int8-entropy"): -6.811576541336124,
        ("rtdetr-l", "fp8"): -0.017908668770093106,
        ("retinanet-r50-fpn-v2", "int8-entropy"): -6.214841431638553,
        ("retinanet-r50-fpn-v2", "fp8"): 0.012380115006584319,
    }
    for row in summary:
        observed = row["mean_e_ap_points"]
        target = expected[(row["model"], row["precision"])]
        if abs(observed - target) > 1e-9:
            raise ValueError(f"Sign/value mismatch for {(row['model'], row['precision'])}: {observed}")

    interaction_csv = root / "generated/cross_family_interaction_cells.csv"
    direct_csv = root / "generated/cross_family_direct_cells.csv"
    table_tex = root / "generated/cross_family_interaction.tex"
    direct_tex = root / "generated/cross_family_direct_summary.tex"
    analysis_json = root / "cross_family_evidence/analysis.json"
    audit_json = root / "generated/cross_family_evidence_audit.json"

    interactions.sort_values(["model", "dataset", "corruption", "severity", "precision"]).to_csv(
        interaction_csv, index=False, float_format="%.10f"
    )
    direct.sort_values(["model", "dataset", "corruption", "severity"]).to_csv(
        direct_csv, index=False, float_format="%.10f"
    )

    table_lines = [
        "% Generated by scripts/rebuild_cross_family_evidence.py; do not edit manually.",
        r"\begin{tabular}{llrrrrr}",
        r"\toprule",
        r"Family & Format & Mean $Q_{95}$ & Mean $E$ & $E$ range & Mean corrupted AP & AP$<5$ \\",
        r"\midrule",
    ]
    for model in MODEL_ORDER:
        for precision in FORMAT_ORDER:
            row = next(r for r in summary if r["model"] == model and r["precision"] == precision)
            table_lines.append(
                f"{MODEL_LABEL[model]} & {FORMAT_LABEL[precision]} & "
                f"{row['mean_q95_ap_points']:.2f} & {signed(row['mean_e_ap_points'])} & "
                f"{signed(row['min_e_ap_points'])}--{signed(row['max_e_ap_points'])} & "
                f"{row['mean_corrupted_ap_points']:.2f} & {row['corrupted_cells_below_5_ap']}/36 \\\\"
            )
    table_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    table_tex.write_text("\n".join(table_lines), encoding="utf-8")

    direct_lines = [
        "% Generated by scripts/rebuild_cross_family_evidence.py; do not edit manually.",
        r"\begin{tabular}{lrrrr}",
        r"\toprule",
        r"Family & Cells & Mean $\Delta E$ & Minimum & Maximum \\",
        r"\midrule",
    ]
    for row in direct_summary:
        direct_lines.append(
            f"{MODEL_LABEL[row['model']]} & {row['cells']} & {signed(row['mean_delta_e_ap_points'])} & "
            f"{signed(row['min_delta_e_ap_points'])} & {signed(row['max_delta_e_ap_points'])} \\\\"
        )
    direct_lines.extend([r"\bottomrule", r"\end{tabular}", ""])
    direct_tex.write_text("\n".join(direct_lines), encoding="utf-8")

    analysis = {
        "schema_version": 3,
        "status": "valid",
        "scope": "cross-family descriptive portability stress cases",
        "sign_convention": {
            "Q": "AP_FP32-AP_quantized",
            "E": "Q_corrupt-Q_clean",
            "DeltaE": "E_INT8-E_FP8=(FP8-INT8)_corrupt-(FP8-INT8)_clean",
        },
        "rows": {"quantized_interactions": len(interactions), "direct_interactions": len(direct)},
        "summary": summary,
        "direct_summary": direct_summary,
        "limitations": [
            "one recorded training seed",
            "one recorded calibration sample",
            "descriptive cells without bootstrap intervals",
            "RT-DETR INT8 clean-fidelity gate failure",
            "RetinaNet TT100K absolute-accuracy floor",
        ],
        "source_sha256": {
            str(cells_path.relative_to(root)): sha256(cells_path),
            str(q95_path.relative_to(root)): sha256(q95_path),
        },
    }
    analysis_json.write_text(json.dumps(analysis, indent=2, sort_keys=True) + "\n", encoding="utf-8")

    audit = {
        "schema_version": 2,
        "status": "valid",
        "formula": analysis["sign_convention"],
        "source_sha256": analysis["source_sha256"],
        "generated_sha256": {
            str(interaction_csv.relative_to(root)): sha256(interaction_csv),
            str(direct_csv.relative_to(root)): sha256(direct_csv),
            str(table_tex.relative_to(root)): sha256(table_tex),
            str(direct_tex.relative_to(root)): sha256(direct_tex),
            str(analysis_json.relative_to(root)): sha256(analysis_json),
        },
        "expected_mean_e_ap_points": {f"{m}/{p}": v for (m, p), v in expected.items()},
    }
    audit_json.write_text(json.dumps(audit, indent=2, sort_keys=True) + "\n", encoding="utf-8")
    print(json.dumps({"summary": summary, "direct_summary": direct_summary}, indent=2))


if __name__ == "__main__":
    main()
