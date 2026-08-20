#!/usr/bin/env python3
"""Hard FP32/FP16 parity gate; compares identical input image order only."""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--fp32", required=True)
    parser.add_argument("--fp16", required=True)
    parser.add_argument("--out", required=True)
    parser.add_argument("--tolerance", type=float, default=0.01)
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit(f"refusing to overwrite parity report: {output}")
    fp32, fp16 = json.loads(Path(args.fp32).read_text()), json.loads(Path(args.fp16).read_text())
    errors = []
    for field in ("n_images", "input_image_ids_sha256", "input_manifest_sha256", "corruption", "severity"):
        if fp32.get(field) != fp16.get(field):
            errors.append(f"{field} differs")
    if fp32.get("precision") != "fp32" or fp16.get("precision") != "fp16":
        errors.append("metrics are not labelled fp32/fp16")
    ap_gap = abs(fp32.get("stats", {}).get("AP", float("inf")) - fp16.get("stats", {}).get("AP", float("inf")))
    if ap_gap > args.tolerance:
        errors.append(f"absolute AP gap {ap_gap:.6f} exceeds tolerance {args.tolerance:.6f}")
    result = {"schema_version": 1, "fp32_metric": str(Path(args.fp32).resolve()), "fp16_metric": str(Path(args.fp16).resolve()), "tolerance": args.tolerance, "absolute_ap_gap": ap_gap, "pass": not errors, "errors": errors}
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(result, indent=2) + "\n")
    print("FP16 PARITY PASS" if not errors else "FP16 PARITY FAIL")
    if errors:
        print("\n".join(f"- {error}" for error in errors))
        raise SystemExit(2)


if __name__ == "__main__":
    main()
