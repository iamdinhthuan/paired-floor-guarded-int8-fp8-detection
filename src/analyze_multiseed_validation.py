#!/usr/bin/env python3
"""Descriptive analysis for the frozen YOLO11m 3x3 seed validation."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from datetime import datetime, timezone
from pathlib import Path
from typing import Any


TRAINING_SEEDS = [20260807, 20260813, 20260814]
CALIBRATION_SEEDS = [20260807, 20260813, 20260814]
DATASETS = ["voc", "kitti", "tt100k"]
FORMATS = ["int8-entropy", "fp8"]
CORRUPTIONS = ["gaussian_noise", "motion_blur", "fog", "jpeg"]


def canonical_hash(document: dict[str, Any], excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    return hashlib.sha256(
        json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
    ).hexdigest()


def sha256_file(path: Path) -> str:
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def direct_cell(
    *,
    dataset: str,
    training_seed: int,
    calibration_seed: int,
    corruption: str,
    int8_clean_ap: float,
    int8_corrupt_ap: float,
    fp8_clean_ap: float,
    fp8_corrupt_ap: float,
) -> dict[str, Any]:
    d_int8 = int8_clean_ap - int8_corrupt_ap
    d_fp8 = fp8_clean_ap - fp8_corrupt_ap
    delta_e = d_int8 - d_fp8
    return {
        "dataset": dataset,
        "training_seed": training_seed,
        "calibration_seed": calibration_seed,
        "corruption": corruption,
        "int8_clean_ap": int8_clean_ap,
        "int8_corrupt_ap": int8_corrupt_ap,
        "fp8_clean_ap": fp8_clean_ap,
        "fp8_corrupt_ap": fp8_corrupt_ap,
        "d_int8_native_ap": d_int8,
        "d_fp8_native_ap": d_fp8,
        "delta_e_native_ap": delta_e,
        "delta_e_ap_points": 100.0 * delta_e,
    }


def build_direct_cells(metric_rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    if len(metric_rows) != 270:
        raise RuntimeError("multi-seed analysis requires exactly 270 metric records")
    index: dict[tuple[Any, ...], dict[str, Any]] = {}
    for row in metric_rows:
        key = (
            row.get("dataset"),
            row.get("training_seed"),
            row.get("calibration_seed"),
            row.get("precision"),
            row.get("corruption"),
            row.get("severity"),
        )
        if key in index:
            raise RuntimeError(f"multi-seed metric grid contains a duplicate cell: {key}")
        index[key] = row
    expected = {
        (dataset, training_seed, calibration_seed, precision, corruption, severity)
        for dataset in DATASETS
        for training_seed in TRAINING_SEEDS
        for calibration_seed in CALIBRATION_SEEDS
        for precision in FORMATS
        for corruption, severity in [("codec_control", 0), *[(name, 5) for name in CORRUPTIONS]]
    }
    if set(index) != expected:
        raise RuntimeError("multi-seed metric grid is not the exact frozen 270-condition set")

    cells: list[dict[str, Any]] = []
    for dataset in DATASETS:
        for training_seed in TRAINING_SEEDS:
            for calibration_seed in CALIBRATION_SEEDS:
                clean = {
                    precision: index[
                        (
                            dataset,
                            training_seed,
                            calibration_seed,
                            precision,
                            "codec_control",
                            0,
                        )
                    ]
                    for precision in FORMATS
                }
                for corruption in CORRUPTIONS:
                    corrupt = {
                        precision: index[
                            (
                                dataset,
                                training_seed,
                                calibration_seed,
                                precision,
                                corruption,
                                5,
                            )
                        ]
                        for precision in FORMATS
                    }
                    row = direct_cell(
                        dataset=dataset,
                        training_seed=training_seed,
                        calibration_seed=calibration_seed,
                        corruption=corruption,
                        int8_clean_ap=float(clean["int8-entropy"]["AP"]),
                        int8_corrupt_ap=float(corrupt["int8-entropy"]["AP"]),
                        fp8_clean_ap=float(clean["fp8"]["AP"]),
                        fp8_corrupt_ap=float(corrupt["fp8"]["AP"]),
                    )
                    row.update(
                        {
                            "int8_clean_metric_sha256": clean["int8-entropy"].get("metric_sha256"),
                            "int8_corrupt_metric_sha256": corrupt["int8-entropy"].get("metric_sha256"),
                            "fp8_clean_metric_sha256": clean["fp8"].get("metric_sha256"),
                            "fp8_corrupt_metric_sha256": corrupt["fp8"].get("metric_sha256"),
                        }
                    )
                    cells.append(row)
    return cells


def two_way_seed_decomposition(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 9:
        raise RuntimeError("multi-seed decomposition requires an exact 3x3 seed grid")
    keys = [(row.get("training_seed"), row.get("calibration_seed")) for row in rows]
    if len(set(keys)) != len(keys):
        raise RuntimeError("multi-seed decomposition contains a duplicate seed cell")
    expected = {
        (training_seed, calibration_seed)
        for training_seed in TRAINING_SEEDS
        for calibration_seed in CALIBRATION_SEEDS
    }
    if set(keys) != expected:
        raise RuntimeError("multi-seed decomposition requires an exact 3x3 seed grid")
    values = {
        key: float(row["delta_e_ap_points"])
        for key, row in zip(keys, rows, strict=True)
    }
    grand = sum(values.values()) / 9.0
    train_means = {
        seed: sum(values[(seed, calibration)] for calibration in CALIBRATION_SEEDS) / 3.0
        for seed in TRAINING_SEEDS
    }
    calibration_means = {
        seed: sum(values[(training, seed)] for training in TRAINING_SEEDS) / 3.0
        for seed in CALIBRATION_SEEDS
    }
    ss_training = 3.0 * sum((value - grand) ** 2 for value in train_means.values())
    ss_calibration = 3.0 * sum(
        (value - grand) ** 2 for value in calibration_means.values()
    )
    ss_interaction = sum(
        (
            value
            - train_means[training_seed]
            - calibration_means[calibration_seed]
            + grand
        )
        ** 2
        for (training_seed, calibration_seed), value in values.items()
    )
    ss_total = sum((value - grand) ** 2 for value in values.values())
    share = lambda value: value / ss_total if ss_total > 0.0 else 0.0
    return {
        "dataset": rows[0].get("dataset"),
        "corruption": rows[0].get("corruption"),
        "n": 9,
        "grand_mean_ap_points": grand,
        "ss_training": ss_training,
        "ss_calibration": ss_calibration,
        "ss_interaction": ss_interaction,
        "ss_total": ss_total,
        "share_training": share(ss_training),
        "share_calibration": share(ss_calibration),
        "share_interaction": share(ss_interaction),
    }


def summarize_cells(rows: list[dict[str, Any]]) -> dict[str, Any]:
    if len(rows) != 108:
        raise RuntimeError("multi-seed summary requires exactly 108 direct cells")
    keys = [
        (
            row.get("dataset"),
            row.get("training_seed"),
            row.get("calibration_seed"),
            row.get("corruption"),
        )
        for row in rows
    ]
    if len(set(keys)) != 108:
        raise RuntimeError("multi-seed summary contains duplicate direct cells")

    def describe(selected: list[dict[str, Any]]) -> dict[str, Any]:
        values = [float(row["delta_e_ap_points"]) for row in selected]
        return {
            "n": len(values),
            "mean_delta_e_ap_points": sum(values) / len(values),
            "min_delta_e_ap_points": min(values),
            "max_delta_e_ap_points": max(values),
        }

    marginal_rows: list[dict[str, Any]] = [
        {"factor": "overall", "level": "all", **describe(rows)}
    ]
    levels: list[tuple[str, list[Any]]] = [
        ("training_seed", TRAINING_SEEDS),
        ("calibration_seed", CALIBRATION_SEEDS),
        ("dataset", DATASETS),
        ("corruption", CORRUPTIONS),
    ]
    for factor, factor_levels in levels:
        for level in factor_levels:
            selected = [row for row in rows if row[factor] == level]
            marginal_rows.append({"factor": factor, "level": level, **describe(selected)})

    decompositions = [
        two_way_seed_decomposition(
            [
                row
                for row in rows
                if row["dataset"] == dataset and row["corruption"] == corruption
            ]
        )
        for dataset in DATASETS
        for corruption in CORRUPTIONS
    ]
    values = [float(row["delta_e_ap_points"]) for row in rows]
    return {
        "overall": describe(rows),
        "sign_inventory": {
            "positive": sum(value > 0.0 for value in values),
            "negative": sum(value < 0.0 for value in values),
            "zero": sum(value == 0.0 for value in values),
        },
        "marginals": marginal_rows,
        "variance_decomposition": decompositions,
    }


def _write_csv(path: Path, rows: list[dict[str, Any]]) -> None:
    if not rows:
        raise RuntimeError(f"multi-seed analysis refuses an empty CSV: {path}")
    with path.open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def _signed(value: float, digits: int = 3) -> str:
    return f"{value:+.{digits}f}"


def render_main_tex(summary: dict[str, Any]) -> str:
    marginal_index = {
        (row["factor"], row["level"]): row for row in summary["marginals"]
    }
    training = [marginal_index[("training_seed", seed)] for seed in TRAINING_SEEDS]
    calibration = [
        marginal_index[("calibration_seed", seed)] for seed in CALIBRATION_SEEDS
    ]
    train_means = [float(row["mean_delta_e_ap_points"]) for row in training]
    calibration_means = [
        float(row["mean_delta_e_ap_points"]) for row in calibration
    ]
    overall = summary["overall"]
    signs = summary["sign_inventory"]
    table_rows = []
    for label, rows in (("Training seed", training), ("Calibration seed", calibration)):
        for row in rows:
            table_rows.append(
                f"{label} & {row['level']} & {row['n']} & "
                f"{_signed(float(row['mean_delta_e_ap_points']))} & "
                f"[{_signed(float(row['min_delta_e_ap_points']))}, "
                f"{_signed(float(row['max_delta_e_ap_points']))}] \\\\"
            )
    return (
        "\\subsection{Targeted training--calibration seed sensitivity}\n"
        "\\label{sec:multiseed-sensitivity}\n\n"
        f"Across 108 direct cells, the equally weighted descriptive mean "
        f"$\\Delta E$ was {_signed(float(overall['mean_delta_e_ap_points']), 2)} "
        "AP points.  Training-seed marginal means ranged from "
        f"{_signed(min(train_means), 2)} to {_signed(max(train_means), 2)} AP points, "
        "and calibration-seed marginal means ranged from "
        f"{_signed(min(calibration_means), 2)} to "
        f"{_signed(max(calibration_means), 2)} AP points.  The cell-wise sign "
        f"inventory was {signs['positive']} positive, {signs['negative']} negative, "
        f"and {signs['zero']} exactly zero.  This targeted sensitivity analysis "
        "covers YOLO11m on VOC, KITTI, and TT100K at severity~5; it does not "
        "extend the seed result to COCO, other capacity rungs, or lower severities.\n\n"
        "\\begin{table}[t]\n"
        "\\centering\n"
        "\\caption{Targeted $3\\times3$ training--calibration seed sensitivity. "
        "Means and ranges equally weight the relevant dataset--corruption cells "
        "and are descriptive AP-point summaries, not confidence intervals.}\n"
        "\\label{tab:multiseed-sensitivity}\n"
        "\\small\n"
        "\\resizebox{\\columnwidth}{!}{%\n"
        "\\begin{tabular}{llrrr}\n"
        "\\toprule\n"
        "Factor & Level & Cells & Mean $\\Delta E$ & Cell range \\\\\n"
        "\\midrule\n"
        + "\n".join(table_rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}}\n"
        "\\end{table}\n"
    )


def render_supplement_tex(summary: dict[str, Any]) -> str:
    display_dataset = {"voc": "VOC", "kitti": "KITTI", "tt100k": "TT100K"}
    display_corruption = {
        "gaussian_noise": "Gaussian noise",
        "motion_blur": "Motion blur",
        "fog": "Fog",
        "jpeg": "JPEG",
    }
    rows = []
    for item in summary["variance_decomposition"]:
        rows.append(
            f"{display_dataset[item['dataset']]} & "
            f"{display_corruption[item['corruption']]} & "
            f"{_signed(float(item['grand_mean_ap_points']))} & "
            f"{100.0 * float(item['share_training']):.1f} & "
            f"{100.0 * float(item['share_calibration']):.1f} & "
            f"{100.0 * float(item['share_interaction']):.1f} \\\\"
        )
    return (
        "\\section{Targeted seed-sensitivity decomposition}\n"
        "\\label{app:multiseed-sensitivity}\n\n"
        "The 12 dataset--corruption blocks below decompose the balanced "
        "$3\\times3$ training--calibration grid descriptively.  Shares are sums "
        "of squares within each block and are not population variance estimates "
        "or hypothesis tests.\n\n"
        "\\begin{table}[htbp]\n"
        "\\centering\n"
        "\\caption{Descriptive two-way seed decomposition within each targeted "
        "YOLO11m severity-5 block.  The final three columns are percentages of "
        "the within-block total sum of squares; the last is the "
        "training-by-calibration interaction (non-additivity).}\n"
        "\\label{tab:multiseed-decomposition}\n"
        "\\small\n"
        "\\begin{tabular}{llrrrr}\n"
        "\\toprule\n"
        "Dataset & Corruption & Mean & Train (\\%) & Cal. (\\%) & "
        "Train$\\times$cal. interaction (\\%) \\\\\n"
        "\\midrule\n"
        + "\n".join(rows)
        + "\n\\bottomrule\n"
        "\\end{tabular}\n"
        "\\end{table}\n"
    )


def write_analysis_outputs(
    cells: list[dict[str, Any]],
    *,
    output_dir: Path,
    attempt: str,
    config_sha256: str,
    metric_completion_report_sha256: str,
) -> dict[str, Path]:
    summary = summarize_cells(cells)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    paths = {
        "cells_csv": output_dir / f"{attempt}_direct_cells.csv",
        "marginals_csv": output_dir / f"{attempt}_marginals.csv",
        "decomposition_csv": output_dir / f"{attempt}_variance_decomposition.csv",
        "main_tex": output_dir / f"{attempt}_main.tex",
        "supplement_tex": output_dir / f"{attempt}_supplement.tex",
        "report_json": output_dir / f"{attempt}_analysis.json",
    }
    existing = [path for path in paths.values() if path.exists()]
    if existing:
        raise RuntimeError(f"multi-seed analysis refuses to overwrite: {existing[0]}")
    _write_csv(paths["cells_csv"], cells)
    _write_csv(paths["marginals_csv"], summary["marginals"])
    _write_csv(paths["decomposition_csv"], summary["variance_decomposition"])
    paths["main_tex"].write_text(render_main_tex(summary), encoding="utf-8")
    paths["supplement_tex"].write_text(
        render_supplement_tex(summary), encoding="utf-8"
    )
    report = {
        "schema_version": 1,
        "status": "complete",
        "attempt": attempt,
        "completed_at_utc": datetime.now(timezone.utc).isoformat(),
        "config_sha256": config_sha256,
        "analyzer_sha256": sha256_file(Path(__file__).resolve()),
        "metric_completion_report_sha256": metric_completion_report_sha256,
        "counts": {
            "direct_cells": len(cells),
            "marginals": len(summary["marginals"]),
            "variance_blocks": len(summary["variance_decomposition"]),
        },
        "summary": {
            "overall": summary["overall"],
            "sign_inventory": summary["sign_inventory"],
        },
        "artifacts_sha256": {
            name: sha256_file(path)
            for name, path in paths.items()
            if name != "report_json"
        },
    }
    report["report_sha256"] = canonical_hash(report, "report_sha256")
    paths["report_json"].write_text(
        json.dumps(report, indent=2, sort_keys=True) + "\n", encoding="utf-8"
    )
    return paths


def validate_analysis_outputs(paths: dict[str, Path]) -> dict[str, Any]:
    expected_keys = {
        "cells_csv",
        "marginals_csv",
        "decomposition_csv",
        "main_tex",
        "supplement_tex",
        "report_json",
    }
    if set(paths) != expected_keys or any(not Path(path).is_file() for path in paths.values()):
        raise RuntimeError("multi-seed analysis output grid is incomplete")
    try:
        report = json.loads(Path(paths["report_json"]).read_text(encoding="utf-8"))
    except json.JSONDecodeError as error:
        raise RuntimeError("multi-seed analysis report JSON is invalid") from error
    if (
        not isinstance(report, dict)
        or report.get("status") != "complete"
        or report.get("report_sha256") != canonical_hash(report, "report_sha256")
        or report.get("analyzer_sha256") != sha256_file(Path(__file__).resolve())
        or report.get("counts")
        != {"direct_cells": 108, "marginals": 14, "variance_blocks": 12}
    ):
        raise RuntimeError("multi-seed analysis report identity/hash mismatch")
    bindings = report.get("artifacts_sha256")
    expected_names = {
        "cells_csv",
        "marginals_csv",
        "decomposition_csv",
        "main_tex",
        "supplement_tex",
    }
    if not isinstance(bindings, dict) or set(bindings) != expected_names:
        raise RuntimeError("multi-seed analysis artifact grid mismatch")
    for name in expected_names:
        if sha256_file(Path(paths[name])) != bindings[name]:
            raise RuntimeError(f"multi-seed analysis artifact hash mismatch: {name}")
    expected_rows = {"cells_csv": 108, "marginals_csv": 14, "decomposition_csv": 12}
    for name, count in expected_rows.items():
        with Path(paths[name]).open(encoding="utf-8", newline="") as handle:
            if len(list(csv.DictReader(handle))) != count:
                raise RuntimeError(f"multi-seed analysis CSV row count mismatch: {name}")
    return report


def analysis_output_paths(config: dict[str, Any], root: Path) -> dict[str, Path]:
    output_dir = Path(root).resolve() / "outputs" / "analysis" / config["attempt"]
    return {
        "cells_csv": output_dir / f"{config['attempt']}_direct_cells.csv",
        "marginals_csv": output_dir / f"{config['attempt']}_marginals.csv",
        "decomposition_csv": output_dir
        / f"{config['attempt']}_variance_decomposition.csv",
        "main_tex": output_dir / f"{config['attempt']}_main.tex",
        "supplement_tex": output_dir / f"{config['attempt']}_supplement.tex",
        "report_json": output_dir / f"{config['attempt']}_analysis.json",
    }


def load_validated_metric_rows(
    config: dict[str, Any], root: Path
) -> tuple[list[dict[str, Any]], str, Path]:
    import run_multiseed_validation as runner

    root = Path(root).resolve()
    jobs = runner.derive_jobs(config, root)
    evidence_paths = runner.evidence_artifact_paths(config, root)
    metric_report = (
        root / "outputs" / "reports" / f"{config['attempt']}_metric_complete.json"
    )
    runner.validate_stage_report(
        metric_report,
        attempt=config["attempt"],
        stage="metric",
        config_sha256=config["config_sha256"],
        artifacts=[evidence_paths["inference_report"], *evidence_paths["metrics"]],
        root=root,
    )
    rows: list[dict[str, Any]] = []
    for job in jobs["metric"]:
        if not runner.validate_metric_job(job, config, root):
            raise RuntimeError(f"multi-seed metric is incomplete: {job['condition_id']}")
        metric_path = runner._condition_paths(
            root, config["attempt"], job["condition_id"]
        )[3]
        metric = json.loads(metric_path.read_text(encoding="utf-8"))
        rows.append(
            {
                "dataset": job["dataset"],
                "training_seed": job["training_seed"],
                "calibration_seed": job["calibration_seed"],
                "precision": job["precision"],
                "corruption": job["corruption"],
                "severity": job["severity"],
                "AP": float(metric["stats"]["AP"]),
                "metric_sha256": sha256_file(metric_path),
                "metric_path": str(metric_path.relative_to(root)),
            }
        )
    if len(rows) != 270:
        raise RuntimeError("multi-seed validated metric grid is not exactly 270 records")
    return rows, sha256_file(metric_report), metric_report


def execute_analysis(
    config: dict[str, Any], root: Path, *, resume_verified: bool
) -> dict[str, Any]:
    import run_multiseed_validation as runner

    root = Path(root).resolve()
    metric_rows, metric_report_sha, metric_report = load_validated_metric_rows(
        config, root
    )
    paths = analysis_output_paths(config, root)
    completion = (
        root / "outputs" / "reports" / f"{config['attempt']}_analysis_complete.json"
    )
    analysis_artifacts = [metric_report, *paths.values()]
    if completion.exists():
        if not resume_verified:
            raise RuntimeError("multi-seed analysis completion report already exists")
        report = validate_analysis_outputs(paths)
        stage = runner.validate_stage_report(
            completion,
            attempt=config["attempt"],
            stage="analysis",
            config_sha256=config["config_sha256"],
            artifacts=analysis_artifacts,
            root=root,
        )
        return {"status": "complete", "analysis": report, "completion": stage}
    partial = [path for path in paths.values() if path.exists()]
    if partial:
        raise RuntimeError(f"multi-seed analysis has partial output: {partial[0]}")

    cells = build_direct_cells(metric_rows)
    written = write_analysis_outputs(
        cells,
        output_dir=next(iter(paths.values())).parent,
        attempt=config["attempt"],
        config_sha256=config["config_sha256"],
        metric_completion_report_sha256=metric_report_sha,
    )
    report = validate_analysis_outputs(written)
    stage = runner._finish_stage(
        config,
        root,
        "analysis",
        [metric_report, *written.values()],
        resume_verified=resume_verified,
    )
    return {"status": "complete", "analysis": report, "completion": stage}


def main() -> None:
    import run_multiseed_validation as runner

    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", type=Path, required=True)
    parser.add_argument("--config", type=Path, required=True)
    parser.add_argument("--execute", action="store_true", required=True)
    parser.add_argument("--resume-verified", action="store_true")
    args = parser.parse_args()
    config = runner.load_config(args.config)
    result = execute_analysis(
        config, args.project_root, resume_verified=args.resume_verified
    )
    print(json.dumps(result, indent=2, sort_keys=True))


if __name__ == "__main__":
    main()
