#!/usr/bin/env python3
"""Validate final strengthening reports and generate manuscript fragments."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from pilot_registry import canonical_hash
from topic_c.manifest import sha256_file


def validate_report(path: Path, *, hash_key: str) -> dict:
    marker = path.with_suffix(path.suffix + ".complete")
    if not path.is_file() or not marker.is_file():
        raise ValueError(f"report or completion marker is missing: {path}")
    if marker.read_text(encoding="utf-8").strip() != sha256_file(path):
        raise ValueError(f"completion marker does not match report bytes: {path}")
    try:
        report = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError as exc:
        raise ValueError(f"report is not valid JSON: {path}") from exc
    if report.get(hash_key) != canonical_hash(report, hash_key):
        raise ValueError(f"report self-hash is invalid: {path}")
    return report


def validate_master_bindings(
    master: dict,
    confirmatory: Path,
    realization: Path,
    fixed_universe: Path | None = None,
) -> None:
    components = master.get("validated_components", {})
    expected = {
        "confirmatory_analysis": confirmatory.resolve(),
        "realization_analysis": realization.resolve(),
    }
    if fixed_universe is not None:
        expected["fixed_universe_sensitivity"] = fixed_universe.resolve()
    if master.get("status") != "complete":
        raise ValueError("master completion is not complete")
    for name, path in expected.items():
        binding = components.get(name, {})
        # The immutable master records the original 5090 path.  Evidence is
        # deliberately relocatable; identity is established by its byte hash.
        if not path.is_file() or binding.get("sha256") != sha256_file(path):
            raise ValueError(f"master completion binding mismatch: {name}")

def _points(value: float) -> float:
    return 100.0 * float(value)


def _signed(value: float) -> str:
    converted = _points(value)
    return f"{converted:+.2f}" if converted >= 0 else f"{converted:.2f}"


def _interval(values: list[float]) -> str:
    if len(values) != 3:
        raise ValueError("percentile interval must contain lower, median, and upper")
    return "$[" + ",\\,".join(_signed(value) for value in values) + "]$"


def _bounds(values: list[float]) -> str:
    if len(values) != 3:
        raise ValueError("percentile interval must contain lower, median, and upper")
    return "$[" + _signed(values[0]) + ",\\," + _signed(values[2]) + "]$"


def render_combined_abstract(confirmatory: dict, realization: dict) -> str:
    overall = confirmatory["overall_balanced_equal_cell"]
    sensitivity = realization["overall_delta_e"]
    return (
        "% Generated from validated direct and confirmatory evidence; do not edit manually.\n"
        "Post-training quantization is commonly evaluated on clean images, leaving unclear whether "
        "corruption changes the performance gap between deployment formats. We present a paired, "
        "floor-aware evaluation of executable INT8 and FP8 YOLO11 TensorRT engines. For each "
        "condition, a deterministic JPEG-95 clean control and a corrupted counterpart use identical "
        "encoded images, and the format interaction is measured as "
        "$\\Delta E=(\\mathrm{FP8}-\\mathrm{INT8})_{\\mathrm{corrupt}}-"
        "(\\mathrm{FP8}-\\mathrm{INT8})_{\\mathrm{clean}}$ with common image-bootstrap draws. "
        "The primary exploratory grid covers 144 corruption cells over four datasets, three "
        "capacities, four corruptions, and three severities. FP8 exceeded INT8 matched-clean AP by "
        "1.40 points on average, whereas the balanced four-dataset $\\Delta E$ was -0.02 AP points "
        "(95\\% percentile interval, -0.24 to +0.15). An untouched-holdout validation on "
        f"VOC and KITTI yielded ${_signed(overall['delta_e_point'])}$ AP points "
        f"(95\\% percentile interval {_bounds(overall['delta_e_percentile95'])}). A "
        f"three-realization sensitivity yielded ${_signed(sensitivity['mean_delta_e'])}$ AP points "
        f"({_bounds(sensitivity['nested_percentile95'])}) across 36 base conditions; unlike the exploratory grid, "
        "this subset sensitivity nests each stochastic field or blur angle across severity. These analyses "
        "address distinct conditional scopes rather than repeated estimates of one parameter. Absolute "
        "corrupted accuracy frequently deteriorated, so gap contraction near a common AP floor is "
        "not robustness evidence. Matched-clean fidelity, absolute corrupted accuracy, and the "
        "format--corruption interaction must therefore be reported separately.\n"
    )


def render_combined_highlights(confirmatory: dict, realization: dict) -> str:
    overall = confirmatory["overall_balanced_equal_cell"]
    if realization["overall_delta_e"].get("n_realization_cells") != 108:
        raise ValueError("highlight generation requires the exact realization grid")
    interval = overall["delta_e_percentile95"]
    lines = [
        "Paired JPEG-95 controls isolate corruption-induced INT8-FP8 gap changes.",
        "FP8 gains 1.40 matched-clean AP points on average over INT8.",
        "The 144-cell primary macro is -0.02 AP points (-0.24 to +0.15).",
        (
            f"Untouched VOC/KITTI delta-E is {_signed(overall['delta_e_point'])} AP points "
            f"({_signed(interval[0])} to {_signed(interval[2])})."
        ),
        "Absolute AP plus realization and fixed-class checks expose conditionality.",
    ]
    if any(len(line) > 85 for line in lines):
        raise ValueError("generated highlight exceeds 85 characters")
    return "".join(f"- {line}\n" for line in lines)


def render_confirmatory_results(report: dict) -> str:
    scope = report["scope"]
    overall = report["overall_balanced_equal_cell"]
    clean = report["matched_clean"]
    guardrail = report["absolute_corrupted_ap_guardrail"]
    by_dataset = report.get("by_dataset", {})
    if scope.get("direct_cells") != 72 or scope.get("partition_role") != "untouched final holdout":
        raise ValueError("confirmatory report does not bind the exact untouched 72-cell design")
    if set(by_dataset) != {"voc", "kitti"} or any(
        by_dataset[name].get("n_cells") != 36 for name in ("voc", "kitti")
    ):
        raise ValueError("confirmatory report does not bind two exact 36-cell dataset margins")
    dataset_rows = []
    for name, label, selection_images, final_images in (
        ("voc", "VOC", 2_510, 5_823),
        ("kitti", "KITTI", 1_496, 1_197),
    ):
        row = by_dataset[name]
        dataset_rows.append(
            f"{label} & {selection_images:,} & {final_images:,} & {row['n_cells']} & "
            f"{_signed(row['delta_e_point'])} & {_bounds(row['delta_e_percentile95'])} \\\\\n"
        )
    return (
        "\\subsection{Untouched-holdout validation on VOC and KITTI}\n"
        "The prespecified rerun uses untouched final partitions for VOC and KITTI "
        f"and retains {scope['direct_cells']} equal-weight dataset--model--corruption--severity cells. "
        f"Its balanced direct interaction was ${_signed(overall['delta_e_point'])}$ AP points "
        f"(2.5th/50th/97.5th percentiles {_interval(overall['delta_e_percentile95'])}). "
        f"The corresponding size interaction was ${_signed(overall['delta_psi_point'])}$ AP points "
        f"({_interval(overall['delta_psi_percentile95'])}). "
        f"FP8 retained {abs(_points(clean['fp8_minus_int8_mean'])):.2f} AP points more matched-clean "
        f"accuracy on average and the signed clean advantage was positive in {clean['positive_blocks']}/{clean['blocks']} blocks. "
        f"As an absolute floor guardrail, {guardrail['below_5_ap_points']} of {guardrail['format_arms']} corrupted "
        f"format arms were below 5 AP points and {guardrail['below_10_ap_points']} were below 10 AP points. "
        "This rerun removes evaluation-after-selection reuse for these two datasets; it does not retroactively "
        "turn the broader four-dataset exploratory grid into a confirmatory study.\n"
        "\\begin{table}[t]\n\\centering\n"
        "\\caption{Untouched-holdout dataset margins. Image counts come from the frozen split registries; "
        "effects and intervals are AP points from the shared paired schedule.}\n"
        "\\label{tab:holdout-by-dataset}\n\\small\n"
        "\\resizebox{\\columnwidth}{!}{%\n\\begin{tabular}{lrrrrr}\n\\toprule\n"
        "Dataset & Selection & Final & Cells & Mean $\\Delta E$ & 95\\% interval \\\\\n"
        "\\midrule\n" + "".join(dataset_rows) + "\\bottomrule\n\\end{tabular}%\n}\n\\end{table}\n"
    )


def render_confirmatory_methods(report: dict) -> str:
    scope = report["scope"]
    if scope.get("direct_cells") != 72 or scope.get("partition_role") != "untouched final holdout":
        raise ValueError("confirmatory report does not bind the exact untouched 72-cell design")
    return (
        "\\subsection{Untouched-holdout validation protocol}\n"
        "The prospectively locked checkpoint-selection/final split registry uses seed 20260818. VOC assigns train2007+train2012 "
        "to training (8,218 images), val2007 to checkpoint selection (2,510 images), and val2012 to the "
        "untouched final partition (5,823 final images). KITTI retains 4,788 training images, uses the "
        "1,496-image validation partition for checkpoint selection. Of the remaining 5,985 images, a deterministic "
        "class-covered 20\\% subset forms the 1,197-image final set and the other 4,788 images form training. All "
        "observed classes occur in every frozen final partition. Checkpoint selection uses the trainer's "
        "validation metric; the confidence threshold, NMS, decoder, and calibration protocol are fixed "
        "before final evaluation. Calibration lists contain 512 images drawn only from the corresponding "
        "training partition. The reference chain uses TF32 disabled and gates PyTorch, FP32 ONNX Runtime, "
        "FP32 TensorRT, and FP16 TensorRT before the quantized comparison. The final factorial rerun contains three YOLO11 "
        "capacities, INT8 and FP8, four corruption families, and severities 1, 3, and 5. "
        "Within each dataset, all direct cells reuse the same dataset-level draw schedule for 2,000 "
        "paired image-bootstrap replicates. The rerun therefore quantifies finite-final-partition "
        "uncertainty conditional on the six retrained checkpoints and frozen engines; it does not "
        "propagate population uncertainty over training seeds or datasets.\n"
    )


def render_realization_results(report: dict) -> str:
    delta_e = report["overall_delta_e"]
    delta_psi = report["overall_delta_psi"]
    if (
        report.get("n_boot") != 2000
        or len(report.get("realization_seeds", [])) != 3
        or delta_e.get("n_realization_cells") != 108
        or delta_e.get("n_base_conditions") != 36
    ):
        raise ValueError("corruption-realization report does not bind the exact sensitivity design")
    between = 10000.0 * float(delta_e["mean_between_realization_variance"])
    within = 10000.0 * float(delta_e["mean_within_image_variance"])
    return (
        "\\subsection{Corruption-realization sensitivity}\n"
        f"A nested sensitivity over {delta_e['n_realization_cells']} realization cells and "
        f"{delta_e['n_base_conditions']} base conditions used three fixed realization seeds on "
        "class-covering 512-image subsets of VOC, KITTI, and TT100K. "
        f"The equal-cell mean $\\Delta E$ was ${_signed(delta_e['mean_delta_e'])}$ AP points with a nested "
        f"realization/image percentile summary of {_interval(delta_e['nested_percentile95'])}. "
        f"Across base conditions, the mean between-realization variance was {between:.4f} AP-points$^2$ "
        f"and the mean within-image-bootstrap variance was {within:.4f} AP-points$^2$. "
        f"For $\\Delta\\Psi$, the mean was ${_signed(delta_psi['mean_delta_psi'])}$ AP points "
        f"with nested percentiles {_interval(delta_psi['nested_percentile95'])}. "
        "Because only three prespecified corruption realizations were used, this is a conditional sensitivity "
        "analysis rather than population-level uncertainty over all possible corruptions.\n"
    )


def render_confirmatory_conclusion(confirmatory: dict, realization: dict) -> str:
    overall = confirmatory["overall_balanced_equal_cell"]
    sensitivity = realization["overall_delta_e"]
    return (
        "The untouched VOC/KITTI final partitions yielded a balanced $\\Delta E$ of "
        f"${_signed(overall['delta_e_point'])}$ AP points with percentiles "
        f"{_interval(overall['delta_e_percentile95'])}. A separate three-realization sensitivity "
        f"yielded ${_signed(sensitivity['mean_delta_e'])}$ AP points across its 36 base conditions. "
        "Together these checks provide bounded checks against evaluation-selection and "
        "single-corruption-draw concerns. "
        "Using three fixed realizations does not establish a population-level corruption-randomness claim.\n"
    )


def render_confirmatory_supplement(confirmatory: dict, realization: dict) -> str:
    if confirmatory["scope"].get("direct_cells") != 72:
        raise ValueError("supplement requires the exact confirmatory grid")
    delta_e = realization["overall_delta_e"]
    return (
        "\\section{Untouched-holdout and corruption-realization sensitivity}\n"
        "The untouched-holdout validation branch froze disjoint checkpoint-selection and final image lists "
        "before training. Only the selection partition entered checkpoint choice; calibration lists "
        "were drawn from training data, and each final partition was used for 117 conditions per dataset (234 total). "
        "The direct analysis pairs INT8 and FP8 and reuses one B=2,000 image-index "
        "schedule within each dataset across every capacity, corruption, severity, and endpoint.\n\n"
        "The corruption-realization sensitivity uses 3 datasets $\\times$ 3 realization seeds "
        "$\\times$ 4 corruptions $\\times$ 3 severities, giving "
        f"{delta_e['n_realization_cells']} realization cells over {delta_e['n_base_conditions']} base conditions. "
        "Each dataset uses one deterministic fixed class-covering subset of 512 images. Gaussian-noise "
        "fields, fog fields, and motion-blur angles are nested across severity within an image and "
        "realization seed; deterministic JPEG materialization is shared across the nominal realization "
        "labels. The same clean INT8/FP8 arms and dataset-level image-bootstrap schedule are reused for "
        "all corresponding corrupted arms.\n\n"
        "For each base condition, between-realization variance is the sample variance of the three point "
        "estimates. Within-image variance is the mean of the three image-bootstrap variances, and their "
        "sum is reported as a descriptive two-source decomposition. The nested interval resamples three "
        "realization labels with replacement and uses the corresponding paired image draw at the same "
        "replicate index. This construction preserves within-replicate pairing but, with only three fixed "
        "realizations, is a sensitivity analysis rather than an asymptotic population interval.\n"
    )


def render_fixed_universe_results(report: dict) -> str:
    if report.get("n_boot") != 10_000 or report.get("n_images") != 3_067:
        raise ValueError("fixed-universe report does not bind the exact TT100K sensitivity")
    fixed_e = report["fixed_universe"]["delta_e_all"]
    fixed_psi = report["fixed_universe"]["delta_psi_height"]
    ordinary = report["ordinary_reference"]
    if ordinary.get("n_boot") != 2_000:
        raise ValueError("ordinary TT100K reference does not use B=2,000")
    e_shift = _points(fixed_e["max_checkpoint_percentile_shift_native_ap"])
    psi_shift = _points(fixed_psi["max_checkpoint_percentile_shift_native_ap"])
    return (
        "\\subsection{TT100K rare-class bootstrap sensitivity}\n"
        "For the prespecified YOLO11m/motion-blur/severity-5 stress cell, a positive-weight "
        "Bayesian image bootstrap retained the fixed category universe across all 3,067 TT100K "
        "images. With 10,000 draws, the $\\Delta E$ percentiles were "
        f"{_interval(fixed_e['percentile_interval'])} AP points, compared with "
        f"{_interval(ordinary['delta_e_all'])} under the ordinary B=2,000 image bootstrap. "
        "The height-based $\\Delta\\Psi$ percentiles were "
        f"{_interval(fixed_psi['percentile_interval'])} versus "
        f"{_interval(ordinary['delta_psi_height'])} AP points. The maximum percentile-endpoint "
        f"change from the first 2,000 to all 10,000 fixed-universe draws was {e_shift:.3f} AP "
        f"points for $\\Delta E$ and {psi_shift:.3f} for $\\Delta\\Psi$. The median sign and zero-crossing status "
        "of $\\Delta E$, and the negative-interval status of $\\Delta\\Psi$, were unchanged in this targeted cell, but one cell does "
        "not establish class-composition invariance for the entire TT100K grid.\n"
    )


def _write_new(path: Path, content: str) -> None:
    if path.exists():
        raise ValueError(f"refusing to overwrite generated evidence: {path}")
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(content, encoding="utf-8")


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--confirmatory-analysis", required=True)
    parser.add_argument("--realization-analysis", required=True)
    parser.add_argument("--master-completion", required=True)
    parser.add_argument("--fixed-universe-sensitivity", required=True)
    parser.add_argument("--generated-dir", required=True)
    parser.add_argument("--audit-out", required=True)
    args = parser.parse_args()
    confirm_path = Path(args.confirmatory_analysis).resolve()
    realization_path = Path(args.realization_analysis).resolve()
    master_path = Path(args.master_completion).resolve()
    fixed_path = Path(args.fixed_universe_sensitivity).resolve()
    generated = Path(args.generated_dir).resolve()
    audit_path = Path(args.audit_out).resolve()
    confirm = validate_report(confirm_path, hash_key="analysis_sha256")
    realization = validate_report(realization_path, hash_key="analysis_sha256")
    master = validate_report(master_path, hash_key="completion_sha256")
    validate_master_bindings(master, confirm_path, realization_path, fixed_path)
    fixed = json.loads(fixed_path.read_text(encoding="utf-8"))
    if fixed.get("artifact_sha256") != canonical_hash(fixed, "artifact_sha256"):
        raise ValueError("fixed-universe artifact self-hash mismatch")
    outputs = {
        "confirmatory_methods.tex": render_confirmatory_methods(confirm),
        "confirmatory_results.tex": render_confirmatory_results(confirm),
        "corruption_realization_results.tex": render_realization_results(realization),
        "confirmatory_conclusion.tex": render_confirmatory_conclusion(confirm, realization),
        "confirmatory_supplement.tex": render_confirmatory_supplement(confirm, realization),
        "fixed_universe_results.tex": render_fixed_universe_results(fixed),
        "confirmatory_abstract.tex": render_combined_abstract(confirm, realization),
        "confirmatory_highlights.txt": render_combined_highlights(confirm, realization),
    }
    generated_paths = {}
    for name, content in outputs.items():
        path = generated / name
        _write_new(path, content)
        generated_paths[name] = {"path": str(path), "sha256": sha256_file(path)}
    audit = {
        "schema_version": 1,
        "confirmatory_analysis": {"path": str(confirm_path), "sha256": sha256_file(confirm_path)},
        "realization_analysis": {"path": str(realization_path), "sha256": sha256_file(realization_path)},
        "master_completion": {"path": str(master_path), "sha256": sha256_file(master_path)},
        "fixed_universe_sensitivity": {"path": str(fixed_path), "sha256": sha256_file(fixed_path)},
        "generated": generated_paths,
        "unit_conversion": "native AP fraction multiplied by 100 for AP points; variance multiplied by 10000 for AP-points squared",
    }
    audit["audit_sha256"] = canonical_hash(audit, "audit_sha256")
    _write_new(audit_path, json.dumps(audit, indent=2) + "\n")
    audit_path.with_suffix(audit_path.suffix + ".complete").write_text(
        sha256_file(audit_path) + "\n", encoding="utf-8"
    )
    print(f"CONFIRMATORY PAPER EVIDENCE COMPLETE generated={len(outputs)} audit={audit_path}")


if __name__ == "__main__":
    main()
