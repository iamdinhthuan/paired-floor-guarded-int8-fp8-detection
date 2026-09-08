#!/usr/bin/env python3
"""Generate publication values only after validating the corrected TIDE chain."""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "src"))
from run_tide_error_decomposition import (
    MAIN_ERRORS, TREATMENT_FIELDS, canonical_hash, paired_interactions,
    summarize_interactions, validate_direct_identities,
)
from topic_c.manifest import sha256_file


def build(root: Path, evidence: Path, output: Path) -> dict:
    complete = json.loads((evidence / "complete.json").read_text())
    config_path = root / "configs/cviu_novelty_v1.json"
    config = json.loads(config_path.read_text())
    if (complete["complete_sha256"] != canonical_hash(complete, "complete_sha256")
            or complete["config_sha256"] != sha256_file(config_path)
            or complete["summary_sha256"] != sha256_file(evidence / "summary.json")
            or complete["records"] != 312 or complete["direct_cells"] != 144):
        raise RuntimeError("TIDE completion/config binding failed")
    primary = json.loads((root / "configs/ivc_format_contrast_v1.json").read_text())
    sources = {d["dataset"]: d for d in primary["datasets"]}
    records = []
    for name, digest in complete["record_hashes"].items():
        path = evidence / "records" / name
        record = json.loads(path.read_text())
        if (sha256_file(path) != digest or record["record_sha256"] != canonical_hash(record, "record_sha256")
                or record["schema_version"] != 3):
            raise RuntimeError(f"invalid TIDE record: {path}")
        source_key = "clean_source_attempt" if record["corruption"] == "clean" else "corruption_source_attempt"
        attempt = sources[record["dataset"]][source_key]
        upstream = root / "manifests/runs" / attempt / name
        if not upstream.is_file() or sha256_file(upstream) != record["run_record_sha256"]:
            raise RuntimeError(f"TIDE is not bound to the primary source: {name}")
        run = json.loads(upstream.read_text())
        if any(run[field] != record[field] for field in TREATMENT_FIELDS):
            raise RuntimeError(f"TIDE treatment metadata differs from upstream: {name}")
        records.append(record)
    interactions = paired_interactions(records)
    validate_direct_identities(interactions, config)
    recalculated = summarize_interactions(interactions)
    reported = json.loads((evidence / "summary.json").read_text())
    if (reported["summary_sha256"] != canonical_hash(reported, "summary_sha256")
            or recalculated["groups"] != reported["groups"]
            or sha256_file(evidence / "condition_records.csv") != reported["condition_records_sha256"]
            or sha256_file(evidence / "paired_error_interactions.csv") != reported["paired_error_interactions_sha256"]):
        raise RuntimeError("TIDE summary or ledger disagrees with validated records")
    output.mkdir(parents=True, exist_ok=True)
    groups = recalculated["groups"]
    lines = [r"\begin{tabular}{lrrrrrr}", r"\toprule",
             r"Dataset & Cls & Loc & Both & Dupe & Bkg & Miss\\", r"\midrule"]
    for dataset, label in (("coco", "COCO"), ("voc", "VOC"), ("kitti", "KITTI"), ("tt100k", "TT100K"), ("all", "All")):
        group = groups["overall"]["all"] if dataset == "all" else groups["dataset"][dataset]
        values = [f"${group['delta_error_burden_dap50_' + key]:+.3f}$" for key in MAIN_ERRORS]
        lines.append(label + " & " + " & ".join(values) + r"\\")
    lines += [r"\bottomrule", r"\end{tabular}"]
    (output / "tide_burdens.tex").write_text("\n".join(lines) + "\n")
    overall = groups["overall"]["all"]
    values = {f"Tide{key}": overall["delta_error_burden_dap50_" + key] for key in MAIN_ERRORS}
    values.update(TideCocoEval=overall["delta_e_cocoeval_ap50"], TideOwnAP=overall["delta_e_tide_ap50"])
    for dataset, label in (("coco", "Coco"), ("voc", "Voc"), ("kitti", "Kitti"), ("tt100k", "TT")):
        values[f"Tide{label}Cls"] = groups["dataset"][dataset]["delta_error_burden_dap50_Cls"]
    macros = [f"\\newcommand{{\\{key}}}{{{value:+.3f}}}" for key, value in values.items()]
    (output / "tide_values.tex").write_text("% Generated from corrected, primary-bound TIDE records.\n" + "\n".join(macros) + "\n")
    audit = dict(records=len(records), direct_cells=len(interactions),
                 fixed_engine_preprocessing_decoder_class_map=True,
                 encoded_bytes_matched_between_formats=True,
                 wrapper_source_mismatch_cells=sum(not r["wrapper_source_identity_matched"] for r in interactions),
                 completion_sha256=sha256_file(evidence / "complete.json"),
                 generator_sha256=sha256_file(__file__))
    (evidence / "publication_audit.json").write_text(json.dumps(audit, indent=2) + "\n")
    return audit


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path("."))
    parser.add_argument("--evidence", type=Path, required=True)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    print(json.dumps(build(args.root.resolve(), args.evidence.resolve(), args.output.resolve()), indent=2))
