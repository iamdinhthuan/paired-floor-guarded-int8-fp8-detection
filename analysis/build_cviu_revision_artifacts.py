#!/usr/bin/env python3
"""Build audit tables added for the CVIU manuscript revision.

The script consumes only retained, already validated ledgers.  It performs no
model inference and refuses to write publication tables unless the expected
factorial grids and parity records are complete.
"""

from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import statistics
from collections import defaultdict
from pathlib import Path
from typing import Any, Iterable


PRIMARY_KEYS = ("dataset", "model", "corruption", "severity")
DATASET_ORDER = ("coco", "voc", "kitti", "tt100k")
DISPLAY = {
    "coco": "COCO",
    "voc": "VOC",
    "kitti": "KITTI",
    "tt100k": "TT100K",
    "yolo11n": "YOLO11n",
    "yolo11m": "YOLO11m",
    "yolo11x": "YOLO11x",
    "gaussian_noise": "Gaussian noise",
    "motion_blur": "Motion blur",
    "fog": "Fog",
    "jpeg": "JPEG",
    "rtdetr-l": "RT-DETR-L",
    "retinanet-r50-fpn-v2": "RetinaNet-R50-FPN-v2",
}


def read_json(path: Path) -> dict[str, Any]:
    return json.loads(path.read_text(encoding="utf-8"))


def read_csv(path: Path) -> list[dict[str, str]]:
    with path.open(newline="", encoding="utf-8") as stream:
        return list(csv.DictReader(stream))


def write_text(path: Path, text: str) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(text.rstrip() + "\n", encoding="utf-8")


def sha256_file(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open("rb") as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(chunk)
    return digest.hexdigest()


def file_set_digest(root: Path, paths: Iterable[Path]) -> str:
    """Hash a sorted path/content-hash ledger without depending on root location."""
    records = []
    for path in sorted(set(paths)):
        records.append(f"{path.relative_to(root).as_posix()}\0{sha256_file(path)}\n")
    if not records:
        raise RuntimeError("cannot compute provenance digest for an empty file set")
    return hashlib.sha256("".join(records).encode("utf-8")).hexdigest()


def tex_table(headers: list[str], rows: Iterable[Iterable[str]], spec: str) -> str:
    body = [f"\\begin{{tabular}}{{{spec}}}", "\\toprule", " & ".join(headers) + r" \\", "\\midrule"]
    body.extend(" & ".join(row) + r" \\" for row in rows)
    body.extend(["\\bottomrule", "\\end{tabular}"])
    return "\n".join(body)


def sign(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def fmt(value: float, digits: int = 2, signed: bool = True) -> str:
    pattern = f"{{:{'+' if signed else ''}.{digits}f}}"
    return pattern.format(value)


def _exploratory_training_log_realizations(root: Path) -> tuple[dict[str, tuple[int, str, float, float, float]], list[Path]]:
    """Recover the auto-batch/optimizer realization associated with each run."""
    log_paths = sorted((root / "artifacts/four_dataset_pilot_v1/outputs/logs/training").glob("*.log"))
    if not log_paths:
        raise RuntimeError("no retained exploratory training logs")
    ansi_escape = re.compile(r"\x1b\[[0-?]*[ -/]*[@-~]")
    realized: dict[str, tuple[int, str, float, float, float]] = {}
    for path in log_paths:
        lines = ansi_escape.sub("", path.read_text(encoding="utf-8", errors="replace")).splitlines()
        starts = [index for index, line in enumerate(lines) if "engine/trainer:" in line]
        for start, stop in zip(starts, starts[1:] + [len(lines)]):
            header = lines[start]
            # Resume invocations report a fixed batch inherited from the original
            # run.  The original auto-batch block is the auditable realization.
            if "batch=-1" not in header:
                continue
            name_match = re.search(r"(?:^|, )name=([^,]+)", header)
            if name_match is None:
                continue
            run_id = name_match.group(1)
            if not re.fullmatch(r"(?:voc|kitti|tt100k)_yolo11[nmx]_train_v1", run_id):
                continue
            block = "\n".join(lines[start:stop])
            batch_match = re.search(r"Using batch-size (\d+)", block)
            optimizer_match = re.search(
                r"(MuSGD|AdamW)\(lr=([^,]+), momentum=([^\)]+)\).*", block
            )
            if batch_match is None or optimizer_match is None:
                raise RuntimeError(f"incomplete optimizer realization for {run_id} in {path}")
            decays = [
                float(value)
                for value in re.findall(r"weight\(decay=([0-9.eE+-]+)\)", optimizer_match.group(0))
                if float(value) > 0.0
            ]
            if len(decays) != 1:
                raise RuntimeError(f"expected one positive effective weight decay for {run_id}")
            value = (
                int(batch_match.group(1)),
                optimizer_match.group(1),
                float(optimizer_match.group(2)),
                float(optimizer_match.group(3)),
                decays[0],
            )
            if run_id in realized and realized[run_id] != value:
                raise RuntimeError(f"conflicting retained training realizations for {run_id}")
            realized[run_id] = value
    return realized, log_paths


def training_realization_table(root: Path, out: Path) -> None:
    """Report the realized exploratory runs and compactly bind held-out audits."""
    pretrained_sha = {
        "yolo11n": "0ebbc80d4a7680d14987a577cd21342b65ecfd94632bd9a8da63ae6417644ee1",
        "yolo11m": "d5ffc1a674953a08e11a8d21e022781b1b23a19b730afc309290bd9fb5305b95",
        "yolo11x": "7bc158aa95c0ebfdd87f70f01653c1131b93e92522dbe15c228bcd742e773a24",
    }
    # Expected values make the generator fail closed if a retained log or
    # results ledger is silently replaced.  AP values below are native scale.
    exploratory_expected = {
        ("voc", "yolo11n"): (52, "MuSGD", 0.01, 0.9, 0.00040625, 99, 0.63653, False),
        ("voc", "yolo11m"): (14, "MuSGD", 0.01, 0.9, 0.000546875, 65, 0.69593, False),
        ("voc", "yolo11x"): (6, "MuSGD", 0.01, 0.9, 0.000515625, 67, 0.71228, True),
        ("kitti", "yolo11n"): (55, "AdamW", 0.000833, 0.9, 0.0004296875, 100, 0.65203, False),
        ("kitti", "yolo11m"): (14, "AdamW", 0.000833, 0.9, 0.000546875, 91, 0.73850, False),
        ("kitti", "yolo11x"): (6, "AdamW", 0.000833, 0.9, 0.000515625, 100, 0.73012, True),
        ("tt100k", "yolo11n"): (11, "AdamW", 0.000044, 0.9, 0.000515625, 81, 0.07197, False),
        ("tt100k", "yolo11m"): (3, "AdamW", 0.000044, 0.9, 0.0004921875, 74, 0.26746, False),
        ("tt100k", "yolo11x"): (1, "AdamW", 0.000044, 0.9, 0.0005, 78, 0.26117, False),
    }
    # The six held-out result ledgers were audited on the execution host.  The
    # compact workspace retains their frozen manifests (including args/best.pt
    # hashes), but not the original results.csv files.  These constants are an
    # explicit audit transcription, rather than a value inferred from AP tests.
    heldout_audit = {
        ("voc", "yolo11n"): (16, "MuSGD", 0.01, 0.9, 0.0005, 82, 0.57925),
        ("voc", "yolo11m"): (16, "MuSGD", 0.01, 0.9, 0.0005, 64, 0.63428),
        ("voc", "yolo11x"): (16, "MuSGD", 0.01, 0.9, 0.0005, 73, 0.63662),
        ("kitti", "yolo11n"): (16, "AdamW", 0.000833, 0.9, 0.0005, 100, 0.62026),
        ("kitti", "yolo11m"): (16, "AdamW", 0.000833, 0.9, 0.0005, 99, 0.71211),
        ("kitti", "yolo11x"): (16, "AdamW", 0.000833, 0.9, 0.0005, 98, 0.71811),
    }
    realized, log_paths = _exploratory_training_log_realizations(root)
    expected_run_ids = {f"{dataset}_{model}_train_v1" for dataset, model in exploratory_expected}
    if set(realized) != expected_run_ids:
        missing = sorted(expected_run_ids - set(realized))
        extra = sorted(set(realized) - expected_run_ids)
        raise RuntimeError(f"training-log grid mismatch; missing={missing}, extra={extra}")

    profile_by_dataset = {
        "voc": root / "configs/training/voc_kitti_yolo11_nmx_v1.json",
        "kitti": root / "configs/training/voc_kitti_yolo11_nmx_v1.json",
        "tt100k": root / "configs/training/tt100k_yolo11_nmx_v1.json",
    }
    manifest_paths: list[Path] = []
    result_paths: list[Path] = []
    rows: list[list[str]] = []
    for dataset in ("voc", "kitti", "tt100k"):
        for model in ("yolo11n", "yolo11m", "yolo11x"):
            run_id = f"{dataset}_{model}_train_v1"
            manifest_path = root / f"artifacts/four_dataset_pilot_v1/manifests/training/{run_id}.json"
            result_path = root / f"artifacts/four_dataset_pilot_v1/outputs/training/{dataset}/{run_id}/results.csv"
            manifest = read_json(manifest_path)
            if (manifest.get("dataset"), manifest.get("model"), manifest.get("run_id")) != (
                dataset,
                model,
                run_id,
            ):
                raise RuntimeError(f"training manifest identity mismatch: {manifest_path}")
            if manifest.get("pretrained_weights_sha256") != pretrained_sha[model]:
                raise RuntimeError(f"unexpected pretrained initialization for {run_id}")
            profile = profile_by_dataset[dataset]
            if manifest.get("profile_sha256") != sha256_file(profile):
                raise RuntimeError(f"training profile hash mismatch for {run_id}")
            expected = exploratory_expected[(dataset, model)]
            logged = realized[run_id]
            for observed, wanted in zip(logged, expected[:5]):
                if isinstance(wanted, float):
                    if not math.isclose(float(observed), wanted, rel_tol=0.0, abs_tol=1e-12):
                        raise RuntimeError(f"training realization changed for {run_id}: {logged}")
                elif observed != wanted:
                    raise RuntimeError(f"training realization changed for {run_id}: {logged}")
            resumed = bool(manifest.get("resumed_from"))
            if resumed != expected[7] or resumed != bool(manifest.get("resumed_from_sha256")):
                raise RuntimeError(f"resume provenance mismatch for {run_id}")
            results = read_csv(result_path)
            epochs = [int(float(row["epoch"])) for row in results]
            if epochs != list(range(1, 101)):
                raise RuntimeError(f"expected a complete 100-epoch ledger for {run_id}")
            best = max(results, key=lambda row: float(row["metrics/mAP50-95(B)"]))
            best_epoch = int(float(best["epoch"]))
            best_ap = float(best["metrics/mAP50-95(B)"])
            if best_epoch != expected[5] or not math.isclose(best_ap, expected[6], abs_tol=5e-8):
                raise RuntimeError(f"checkpoint-selection result changed for {run_id}")
            batch, optimizer, learning_rate, momentum, weight_decay = logged
            rows.append(
                [
                    "Expl.",
                    DISPLAY[dataset],
                    DISPLAY[model],
                    str(batch),
                    optimizer,
                    f"{learning_rate:.6g}",
                    f"{momentum:.1f}",
                    f"{weight_decay:.9g}",
                    str(best_epoch),
                    f"{100.0 * best_ap:.3f}",
                ]
            )
            manifest_paths.append(manifest_path)
            result_paths.append(result_path)

    heldout_manifest_paths: list[Path] = []
    heldout_best_only_paths: list[Path] = []
    heldout_profile = root / "configs/training/voc_kitti_confirmatory_yolo11_nmx_v1.json"
    heldout_profile_sha = sha256_file(heldout_profile)
    for dataset in ("voc", "kitti"):
        for model in ("yolo11n", "yolo11m", "yolo11x"):
            run_id = f"{dataset}_{model}_confirmatory_s20260818_v1"
            manifest_path = root / f"manifests/training/{run_id}.json"
            best_only_path = root / f"manifests/training/{run_id}_best_only_v1.json"
            manifest = read_json(manifest_path)
            best_only = read_json(best_only_path)
            if (manifest.get("dataset"), manifest.get("model"), manifest.get("run_id")) != (
                dataset,
                model,
                run_id,
            ):
                raise RuntimeError(f"held-out training manifest identity mismatch: {manifest_path}")
            if manifest.get("profile_sha256") != heldout_profile_sha:
                raise RuntimeError(f"held-out training profile hash mismatch for {run_id}")
            if manifest.get("pretrained_weights_sha256") != pretrained_sha[model]:
                raise RuntimeError(f"unexpected held-out initialization for {run_id}")
            if manifest.get("resolved_batch") != 16 or manifest.get("resumed_from") is not None:
                raise RuntimeError(f"unexpected held-out batch/resume state for {run_id}")
            if best_only.get("training_registry_sha256") != sha256_file(manifest_path):
                raise RuntimeError(f"best-only registry does not bind {run_id}")
            if best_only.get("retained_best_weights_sha256") != manifest.get("best_weights_sha256"):
                raise RuntimeError(f"best-only weight identity mismatch for {run_id}")
            batch, optimizer, learning_rate, momentum, weight_decay, best_epoch, best_ap = heldout_audit[
                (dataset, model)
            ]
            rows.append(
                [
                    "Holdout",
                    DISPLAY[dataset],
                    DISPLAY[model],
                    str(batch),
                    optimizer,
                    f"{learning_rate:.6g}",
                    f"{momentum:.1f}",
                    f"{weight_decay:.9g}",
                    str(best_epoch),
                    f"{100.0 * best_ap:.3f}",
                ]
            )
            heldout_manifest_paths.append(manifest_path)
            heldout_best_only_paths.append(best_only_path)

    comments = (
        "% Generated by analysis/build_cviu_revision_artifacts.py.\n"
        f"% Exploratory manifests set SHA-256: {file_set_digest(root, manifest_paths)}\n"
        f"% Exploratory results.csv set SHA-256: {file_set_digest(root, result_paths)}\n"
        f"% Exploratory training-log set SHA-256: {file_set_digest(root, log_paths)}\n"
        f"% Held-out training-manifest set SHA-256: {file_set_digest(root, heldout_manifest_paths)}\n"
        f"% Held-out best-only-manifest set SHA-256: {file_set_digest(root, heldout_best_only_paths)}\n"
        "% Held-out best epochs/AP are a fixed transcription of execution-host results.csv files audited on 2026-09-03;\n"
        "% the compact workspace binds args.yaml/best.pt hashes but does not retain those six results.csv files.\n"
    )
    write_text(
        out / "training_realization_summary.tex",
        comments
        + tex_table(
            [
                "Branch",
                "Dataset",
                "Model",
                "Batch",
                "Optimizer",
                "Base LR",
                r"Mom./$\beta_1$",
                "Effective WD",
                "Best epoch",
                r"Selection AP$_{50:95}$",
            ],
            rows,
            "lllrlrrrrr",
        ),
    )


def tt100k_combined_endpoint_guardrail_table(root: Path, out: Path) -> None:
    """Emit the audited combined-height guardrail used by TT100K Delta Psi."""
    bootstrap_dir = root / "outputs/bootstrap/ivc_format_contrast_v1"
    contrast_paths = sorted(
        path for path in bootstrap_dir.glob("tt100k__yolo11*__*-s*.json") if "clean-s0" not in path.name
    )
    if len(contrast_paths) != 36:
        raise RuntimeError(f"expected 36 non-clean TT100K contrast records, found {len(contrast_paths)}")
    expected_conditions = {
        (model, corruption, severity)
        for model in ("yolo11n", "yolo11m", "yolo11x")
        for corruption in ("gaussian_noise", "motion_blur", "fog", "jpeg")
        for severity in (1, 3, 5)
    }
    conditions: set[tuple[str, str, int]] = set()
    delta_small: list[float] = []
    delta_large: list[float] = []
    delta_psi: list[float] = []
    corrupt_prediction_hashes: set[str] = set()
    clean_cache_hashes: set[str] = set()
    annotation_hashes: set[str] = set()
    for path in contrast_paths:
        match = re.fullmatch(r"tt100k__(yolo11[nmx])__(.+)-s([135])\.json", path.name)
        if match is None:
            raise RuntimeError(f"unexpected TT100K contrast filename: {path.name}")
        conditions.add((match.group(1), match.group(2), int(match.group(3))))
        report = read_json(path)
        if report.get("endpoint_type") != "tt100k-height" or report.get("n_images") != 3067:
            raise RuntimeError(f"unexpected TT100K endpoint record: {path}")
        bins = report.get("height_bins_px", {})
        if bins.get("small_like") != {"min": 0.0, "max": 24.0}:
            raise RuntimeError(f"small-like height range changed: {path}")
        if bins.get("large_like") != {"min": 48.0, "max": None}:
            raise RuntimeError(f"large-like height range changed: {path}")
        point = report["point"]
        small = float(point["delta_e"]["small_like"])
        large = float(point["delta_e"]["large_like"])
        psi = float(point["delta_psi"])
        if not math.isclose(psi, small - large, abs_tol=1e-12):
            raise RuntimeError(f"TT100K delta-Psi identity failed: {path}")
        delta_small.append(100.0 * small)
        delta_large.append(100.0 * large)
        delta_psi.append(100.0 * psi)
        for precision in ("int8_corrupt", "fp8_corrupt"):
            corrupt_prediction_hashes.add(report["input_hashes"][precision]["prediction_sha256"])
        clean_cache_hashes.add(report["clean_arm_cache"]["sha256"])
        annotation_hashes.add(report["annotation"]["sha256"])
    if conditions != expected_conditions:
        raise RuntimeError("TT100K combined-endpoint factorial grid is incomplete")
    if len(corrupt_prediction_hashes) != 72 or len(clean_cache_hashes) != 3 or len(annotation_hashes) != 1:
        raise RuntimeError("TT100K input-provenance cardinality changed")

    # Values were recomputed from the 72 hash-bound corrupted prediction
    # payloads plus the three retained JPEG-95 clean-arm caches by applying the
    # exact combined original-height endpoints [0,24) and [48,infinity).  They
    # must not be obtained by averaging the separately evaluated XS/S or L/XL AP.
    guardrail = {
        ("int8-entropy", "small-like [0,24)"): (37.7415, 25.2526, 12.4889, 8, 9),
        ("int8-entropy", r"large-like [48,$\infty$)"): (55.6112, 41.5734, 14.0378, 1, 2),
        ("fp8", "small-like [0,24)"): (39.9199, 26.9140, 13.0058, 8, 9),
        ("fp8", r"large-like [48,$\infty$)"): (56.2881, 43.1112, 13.1769, 0, 2),
    }
    for clean_ap, corrupt_ap, loss, _, _ in guardrail.values():
        # Independently rounded four-decimal summaries can differ by one unit
        # in the last printed decimal even when the unrounded identity holds.
        if not math.isclose(clean_ap - corrupt_ap, loss, abs_tol=1.5e-4):
            raise RuntimeError("TT100K combined-endpoint loss identity failed")
    small_from_guardrail = guardrail[("int8-entropy", "small-like [0,24)")][2] - guardrail[
        ("fp8", "small-like [0,24)")
    ][2]
    large_from_guardrail = guardrail[("int8-entropy", r"large-like [48,$\infty$)")][2] - guardrail[
        ("fp8", r"large-like [48,$\infty$)")
    ][2]
    if not math.isclose(small_from_guardrail, statistics.fmean(delta_small), abs_tol=5e-5):
        raise RuntimeError("small-like guardrail does not reproduce retained mean Delta E")
    if not math.isclose(large_from_guardrail, statistics.fmean(delta_large), abs_tol=5e-5):
        raise RuntimeError("large-like guardrail does not reproduce retained mean Delta E")
    if not math.isclose(small_from_guardrail - large_from_guardrail, statistics.fmean(delta_psi), abs_tol=5e-5):
        raise RuntimeError("combined guardrail does not reproduce retained mean Delta Psi")

    row_order = (
        ("int8-entropy", "small-like [0,24)"),
        ("int8-entropy", r"large-like [48,$\infty$)"),
        ("fp8", "small-like [0,24)"),
        ("fp8", r"large-like [48,$\infty$)"),
    )
    rows = []
    for precision, endpoint in row_order:
        clean_ap, corrupt_ap, loss, under_five, under_ten = guardrail[(precision, endpoint)]
        rows.append(
            [
                "INT8" if precision == "int8-entropy" else "FP8",
                endpoint,
                "36",
                f"{clean_ap:.2f}",
                f"{corrupt_ap:.2f}",
                f"{loss:+.2f}",
                str(under_five),
                str(under_ten),
            ]
        )
    cache_paths = sorted((root / "outputs/work/ivc_format_contrast_v1/clean_arm_aps").glob("tt100k__yolo11*.npz"))
    if len(cache_paths) != 3 or {sha256_file(path) for path in cache_paths} != clean_cache_hashes:
        raise RuntimeError("retained TT100K clean-arm caches do not match the contrast records")
    comments = (
        "% Generated by analysis/build_cviu_revision_artifacts.py.\n"
        f"% 36-record TT100K paired-contrast set SHA-256: {file_set_digest(root, contrast_paths)}\n"
        f"% Three JPEG-95 clean-arm-cache set SHA-256: {file_set_digest(root, cache_paths)}\n"
        f"% Annotation SHA-256: {next(iter(annotation_hashes))}\n"
        "% Means were audited with the same maxDets=100 evaluator used for Delta Psi; medium [24,48) is excluded.\n"
        "% Combined endpoints were evaluated directly and are not arithmetic averages of XS/S or L/XL AP.\n"
    )
    write_text(
        out / "tt100k_combined_endpoint_guardrail.tex",
        comments
        + tex_table(
            [
                "Treatment",
                "Original-height endpoint",
                "Cells",
                "Clean AP",
                "Corrupted AP",
                r"Mean $D$ (AP)",
                r"AP$<5$",
                r"AP$<10$",
            ],
            rows,
            "llrrrrrr",
        ),
    )


def relative_retention_tables(root: Path, out: Path) -> None:
    arms = read_csv(root / "paper/generated/direct_absolute_guardrail.csv")
    if len(arms) != 288:
        raise RuntimeError(f"expected 288 primary arms, found {len(arms)}")

    grouped: dict[tuple[str, ...], dict[str, dict[str, str]]] = defaultdict(dict)
    for row in arms:
        key = tuple(row[item] for item in PRIMARY_KEYS)
        grouped[key][row["precision"]] = row
    if len(grouped) != 144 or any(set(v) != {"int8-entropy", "fp8"} for v in grouped.values()):
        raise RuntimeError("primary relative-retention pivot is incomplete")

    cells: list[dict[str, Any]] = []
    for key, formats in grouped.items():
        int8, fp8 = formats["int8-entropy"], formats["fp8"]
        ji, ci = float(int8["clean_ap_native"]), float(int8["corrupt_ap_native"])
        jf, cf = float(fp8["clean_ap_native"]), float(fp8["corrupt_ap_native"])
        if min(ji, jf) <= 0:
            raise RuntimeError(f"nonpositive clean AP in primary cell {key}")
        delta_e = 100.0 * ((cf - ci) - (jf - ji))
        delta_r = 100.0 * (cf / jf - ci / ji)
        cells.append(
            {
                **dict(zip(PRIMARY_KEYS, key)),
                "delta_e_ap": delta_e,
                "delta_r_pp": delta_r,
                "sign_disagreement": sign(delta_e) != sign(delta_r),
            }
        )

    holdout = read_json(root / "paper/confirmatory_evidence/untouched_holdout_analysis.json")
    holdout_cells = holdout.get("cells", [])
    if len(holdout_cells) != 72:
        raise RuntimeError(f"expected 72 holdout cells, found {len(holdout_cells)}")
    hcells: list[dict[str, Any]] = []
    for row in holdout_cells:
        ji, ci = float(row["int8_clean_ap"]), float(row["int8_corrupted_ap"])
        jf, cf = float(row["fp8_clean_ap"]), float(row["fp8_corrupted_ap"])
        delta_e = 100.0 * float(row["delta_e"])
        recomputed = 100.0 * ((cf - ci) - (jf - ji))
        if not math.isclose(delta_e, recomputed, abs_tol=1e-8):
            raise RuntimeError("holdout delta E does not match its four AP components")
        delta_r = 100.0 * (cf / jf - ci / ji)
        hcells.append(
            {
                "dataset": row["dataset"],
                "delta_e_ap": delta_e,
                "delta_r_pp": delta_r,
                "sign_disagreement": sign(delta_e) != sign(delta_r),
            }
        )

    def summarize(label: str, selected: list[dict[str, Any]]) -> list[str]:
        return [
            label,
            str(len(selected)),
            fmt(statistics.fmean(item["delta_e_ap"] for item in selected)),
            fmt(statistics.fmean(item["delta_r_pp"] for item in selected)),
            str(sum(bool(item["sign_disagreement"]) for item in selected)),
        ]

    rows = [summarize("Exploratory grid", cells)]
    rows.extend(
        summarize(f"Exploratory: {DISPLAY[dataset]}", [row for row in cells if row["dataset"] == dataset])
        for dataset in DATASET_ORDER
    )
    rows.append(summarize("Final holdouts", hcells))
    rows.extend(
        summarize(f"Final: {DISPLAY[dataset]}", [row for row in hcells if row["dataset"] == dataset])
        for dataset in ("voc", "kitti")
    )
    write_text(
        out / "relative_retention_sensitivity.tex",
        "% Generated by analysis/build_cviu_revision_artifacts.py.\n"
        + tex_table(
            ["Scope", "Cells", r"Mean $\Delta E$ (AP)", r"Mean $\Delta R$ (pp)", "Sign differences"],
            rows,
            "lrrrr",
        ),
    )

    disagreements = [row for row in cells if row["sign_disagreement"]]
    if len(disagreements) != 18:
        raise RuntimeError(f"expected 18 primary scale-sign differences, found {len(disagreements)}")
    disagreements.sort(
        key=lambda row: (
            DATASET_ORDER.index(row["dataset"]),
            row["model"],
            row["corruption"],
            int(row["severity"]),
        )
    )
    disagreement_rows = [
        [
            DISPLAY[row["dataset"]],
            row["model"].replace("yolo11", "YOLO11"),
            DISPLAY[row["corruption"]],
            str(row["severity"]),
            fmt(row["delta_e_ap"]),
            fmt(row["delta_r_pp"]),
        ]
        for row in disagreements
    ]
    write_text(
        out / "relative_retention_sign_disagreements.tex",
        "% Generated by analysis/build_cviu_revision_artifacts.py.\n"
        + tex_table(
            ["Dataset", "Model", "Corruption", "Severity", r"$\Delta E$ (AP)", r"$\Delta R$ (pp)"],
            disagreement_rows,
            "lllrrr",
        ),
    )


def reversal_table(root: Path, out: Path) -> None:
    cells = read_csv(root / "paper/generated/decision_impact_cells.csv")
    if len(cells) != 144:
        raise RuntimeError(f"expected 144 decision-impact cells, found {len(cells)}")
    reversals = [
        row
        for row in cells
        if float(row["raw_corrupt_gap_ap"]) > 0 and float(row["adjusted_interaction_ap"]) < 0
    ]
    if len(reversals) != 56:
        raise RuntimeError(f"expected 56 point-sign reversals, found {len(reversals)}")
    thresholds = (0.0, 0.25, 0.50, 1.00)
    rows = []
    for threshold in thresholds:
        retained = [row for row in reversals if abs(float(row["adjusted_interaction_ap"])) >= threshold]
        interval_negative = [row for row in retained if 100.0 * float(row["delta_e_ci_high"]) < 0]
        rows.append(
            [
                f"$|\\Delta E|\\geq {threshold:.2f}$",
                str(len(retained)),
                str(len(interval_negative)),
                str(len(retained) - len(interval_negative)),
            ]
        )
    if rows[0][2:] != ["17", "39"]:
        raise RuntimeError("interval-qualified reversal inventory changed")
    write_text(
        out / "decision_reversal_sensitivity.tex",
        "% Generated by analysis/build_cviu_revision_artifacts.py.\n"
        + tex_table(
            ["Point-effect filter (AP)", "Reversals", "Interval below zero", "Interval crosses zero"],
            rows,
            "lrrr",
        ),
    )


def holdout_tables(root: Path, out: Path) -> None:
    split_rows: list[list[str]] = []
    for dataset in ("voc", "kitti"):
        registry = read_json(root / f"manifests/splits/{dataset}_confirmatory_v1/{dataset}_confirmatory_v1.json")
        for partition in ("train", "selection", "final"):
            record = registry["splits"][partition]
            split_rows.append(
                [
                    DISPLAY[dataset],
                    partition.capitalize(),
                    f"{int(record['images']):,}",
                    f"{int(record['objects']):,}",
                    f"{len(record['class_object_counts'])}/{len(registry['names'])}",
                ]
            )
    write_text(
        out / "holdout_split_summary.tex",
        "% Generated by analysis/build_cviu_revision_artifacts.py.\n"
        + tex_table(["Dataset", "Partition", "Images", "Objects", "Observed/declared classes"], split_rows, "llrrr"),
    )

    parity_rows: list[list[str]] = []
    reports = sorted((root / "outputs/reports/reference_parity/voc_kitti_confirmatory_v1").glob("*.json"))
    if len(reports) != 6:
        raise RuntimeError(f"expected six holdout parity reports, found {len(reports)}")
    for path in reports:
        report = read_json(path)
        if report.get("pass") is not True or report.get("errors"):
            raise RuntimeError(f"failed parity report: {path}")
        gaps = report["absolute_ap_gaps"]
        parity_rows.append(
            [
                DISPLAY[report["dataset"]],
                report["model"].replace("yolo11", "YOLO11"),
                fmt(100.0 * float(gaps["pytorch_to_onnxruntime"]), 4, False),
                fmt(100.0 * float(gaps["pytorch_to_trt_fp32"]), 4, False),
                fmt(100.0 * float(gaps["trt_fp32_to_trt_fp16"]), 4, False),
                "Pass",
            ]
        )
    parity_rows.sort(key=lambda row: (("VOC", "KITTI").index(row[0]), row[1]))
    write_text(
        out / "holdout_reference_parity.tex",
        "% Generated by analysis/build_cviu_revision_artifacts.py; absolute AP-point gaps.\n"
        + tex_table(
            ["Dataset", "Model", "Source--ORT", "Source--TRT32", "TRT32--TRT16", "Gate"],
            parity_rows,
            "llrrrr",
        ),
    )


def realization_tables(root: Path, out: Path) -> None:
    report = read_json(root / "outputs/reports/corruption_realization_analysis_v1.json")
    cells = report.get("realization_cells", [])
    conditions = report.get("conditions", [])
    if len(cells) != 108 or len(conditions) != 36:
        raise RuntimeError("corruption-realization grid is incomplete")

    by_seed: dict[int, list[dict[str, Any]]] = defaultdict(list)
    for row in cells:
        by_seed[int(row["realization_seed"])].append(row)
    seed_rows = []
    delta_e_seed_means = []
    delta_psi_seed_means = []
    for seed in sorted(by_seed):
        rows = by_seed[seed]
        de = 100.0 * statistics.fmean(float(row["delta_e"]) for row in rows)
        dp = 100.0 * statistics.fmean(float(row["delta_psi"]) for row in rows)
        delta_e_seed_means.append(de)
        delta_psi_seed_means.append(dp)
        seed_rows.append([str(seed), str(len(rows)), fmt(de), fmt(dp)])
    seed_rows.append(
        [
            "Across-seed summary",
            "108",
            f"{fmt(statistics.fmean(delta_e_seed_means))}; SD {statistics.stdev(delta_e_seed_means):.2f}",
            f"{fmt(statistics.fmean(delta_psi_seed_means))}; SD {statistics.stdev(delta_psi_seed_means):.2f}",
        ]
    )
    write_text(
        out / "corruption_realization_seed_summary.tex",
        "% Generated by analysis/build_cviu_revision_artifacts.py.\n"
        + tex_table(["Realization seed", "Cells", r"Mean $\Delta E$ (AP)", r"Mean $\Delta\Psi$ (AP)"], seed_rows, "lrrr"),
    )

    by_corruption: dict[str, list[dict[str, Any]]] = defaultdict(list)
    for row in conditions:
        by_corruption[row["corruption"]].append(row)
    corruption_rows: list[list[str]] = []
    stochastic_between: list[float] = []
    for corruption in ("gaussian_noise", "motion_blur", "fog", "jpeg"):
        rows = by_corruption[corruption]
        if len(rows) != 9:
            raise RuntimeError(f"expected nine base conditions for {corruption}")
        mean_de = 100.0 * statistics.fmean(float(row["delta_e"]["mean_delta_e"]) for row in rows)
        between = 10000.0 * statistics.fmean(float(row["delta_e"]["between_realization_variance"]) for row in rows)
        within = 10000.0 * statistics.fmean(float(row["delta_e"]["within_image_variance"]) for row in rows)
        if corruption == "jpeg":
            if not math.isclose(between, 0.0, abs_tol=1e-12):
                raise RuntimeError("deterministic JPEG unexpectedly has realization variance")
            shown_between = "n/a (deterministic)"
        else:
            stochastic_between.append(between)
            shown_between = f"{between:.4f}"
        corruption_rows.append([DISPLAY[corruption], fmt(mean_de), shown_between, f"{within:.4f}"])
    corruption_rows.append(
        ["Stochastic-only mean", "--", f"{statistics.fmean(stochastic_between):.4f}", "--"]
    )
    write_text(
        out / "corruption_realization_variance.tex",
        "% Generated by analysis/build_cviu_revision_artifacts.py; variances use AP-points squared.\n"
        + tex_table(
            ["Corruption", r"Mean $\Delta E$ (AP)", r"Between-realization variance", r"Within-image variance"],
            corruption_rows,
            "lrrr",
        ),
    )


def conditionality_scope_table(root: Path, out: Path) -> None:
    primary_rows = read_csv(root / "paper/generated/direct_format_contrast_macro.csv")
    primary = next(row for row in primary_rows if row["endpoint"] == "four_dataset_macro_delta_e")
    holdout = read_json(root / "paper/confirmatory_evidence/untouched_holdout_analysis.json")
    realization = read_json(root / "outputs/reports/corruption_realization_analysis_v1.json")["overall_delta_e"]
    seeds = read_json(root / "paper/multiseed_evidence/analysis.json")["summary"]["overall"]

    rows = [
        [
            "Exploratory grid",
            "4; n/m/x; S1/3/5",
            "144",
            fmt(100.0 * float(primary["point"])),
            f"[{100.0 * float(primary['ci_low']):+.2f}, {100.0 * float(primary['ci_high']):+.2f}]",
            "Paired image bootstrap",
        ],
        [
            "Final holdouts",
            "VOC/KITTI; n/m/x; S1/3/5",
            "72",
            fmt(100.0 * float(holdout["overall_balanced_equal_cell"]["delta_e_point"])),
            f"[{100.0 * float(holdout['overall_balanced_equal_cell']['delta_e_percentile95'][0]):+.2f}, {100.0 * float(holdout['overall_balanced_equal_cell']['delta_e_percentile95'][2]):+.2f}]",
            "Paired image bootstrap",
        ],
        [
            "Realization subset",
            "3 datasets; m; S1/3/5",
            "36 base / 108",
            fmt(100.0 * float(realization["mean_delta_e"])),
            f"[{100.0 * float(realization['nested_percentile95'][0]):+.2f}, {100.0 * float(realization['nested_percentile95'][2]):+.2f}]",
            "3-label nested sensitivity",
        ],
        [
            "Training/calibration seeds",
            "3 datasets; m; S5",
            str(int(seeds["n"])),
            fmt(float(seeds["mean_delta_e_ap_points"])),
            "not estimated",
            "Descriptive $3\\times3$ grid",
        ],
    ]
    write_text(
        out / "conditionality_scope_summary.tex",
        "% Generated by analysis/build_cviu_revision_artifacts.py. Intervals have different conditional scopes.\n"
        + tex_table(
            ["Analysis", "Dataset/model/severity scope", "Cells", r"Mean $\Delta E$", "Interval (AP)", "Uncertainty target"],
            rows,
            "llrrll",
        ),
    )


def runtime_tables(root: Path, out: Path) -> None:
    records = read_csv(root / "paper/generated/direct_deployment_conditions.csv")
    if len(records) != 36:
        raise RuntimeError(f"expected 36 YOLO runtime records, found {len(records)}")
    grouped: dict[tuple[str, str], dict[str, float]] = defaultdict(dict)
    for row in records:
        grouped[(row["dataset"], row["model"])][row["precision"]] = float(row["latency_median_ms"])
    if len(grouped) != 12 or any(set(row) != {"fp32", "int8-entropy", "fp8"} for row in grouped.values()):
        raise RuntimeError("YOLO runtime pivot is incomplete")
    rows = []
    for (dataset, model), latency in sorted(
        grouped.items(), key=lambda item: (DATASET_ORDER.index(item[0][0]), item[0][1])
    ):
        rows.append(
            [
                DISPLAY[dataset],
                model.replace("yolo11", "YOLO11"),
                f"{latency['fp32']:.3f}",
                f"{latency['int8-entropy']:.3f}",
                f"{latency['fp8']:.3f}",
                f"{latency['fp32'] / latency['int8-entropy']:.2f}",
                f"{latency['fp32'] / latency['fp8']:.2f}",
                f"{latency['int8-entropy'] / latency['fp8']:.2f}",
            ]
        )
    write_text(
        out / "direct_runtime_conditions.tex",
        "% Generated by analysis/build_cviu_revision_artifacts.py.\n"
        + tex_table(
            ["Dataset", "Model", "FP32 ms", "INT8 ms", "FP8 ms", "32/INT8", "32/FP8", "INT8/FP8"],
            rows,
            "llrrrrrr",
        ),
    )

    paths = sorted((root / "artifacts/cross_family_v1/outputs/deployment/records").glob("*.json"))
    if len(paths) != 54:
        raise RuntimeError(f"expected 54 cross-family timing records, found {len(paths)}")
    raw: dict[tuple[str, str, str], list[float]] = defaultdict(list)
    for path in paths:
        row = read_json(path)
        raw[(row["model"], row["dataset"], row["precision"])].append(float(row["latency_median_ms"]))
    medians = {key: statistics.median(values) for key, values in raw.items()}
    cross_rows = []
    for model in ("rtdetr-l", "retinanet-r50-fpn-v2"):
        for dataset in ("voc", "kitti", "tt100k"):
            fp32 = medians[(model, dataset, "fp32")]
            int8 = medians[(model, dataset, "int8-entropy")]
            fp8 = medians[(model, dataset, "fp8")]
            cross_rows.append(
                [
                    DISPLAY[model],
                    DISPLAY[dataset],
                    f"{fp32:.3f}",
                    f"{int8:.3f}",
                    f"{fp8:.3f}",
                    f"{fp32 / int8:.2f}",
                    f"{fp32 / fp8:.2f}",
                ]
            )
    write_text(
        out / "cross_family_runtime_by_dataset.tex",
        "% Generated by analysis/build_cviu_revision_artifacts.py.\n"
        + tex_table(
            ["Family", "Dataset", "FP32 ms", "INT8 ms", "FP8 ms", "32/INT8", "32/FP8"],
            cross_rows,
            "llrrrrr",
        ),
    )


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument(
        "--output",
        type=Path,
        default=Path("paper/generated"),
    )
    args = parser.parse_args()
    root = args.root.resolve()
    out = args.output if args.output.is_absolute() else root / args.output

    training_realization_table(root, out)
    tt100k_combined_endpoint_guardrail_table(root, out)
    relative_retention_tables(root, out)
    reversal_table(root, out)
    holdout_tables(root, out)
    realization_tables(root, out)
    conditionality_scope_table(root, out)
    runtime_tables(root, out)
    print(f"wrote CVIU revision artifacts to {out}")


if __name__ == "__main__":
    main()
