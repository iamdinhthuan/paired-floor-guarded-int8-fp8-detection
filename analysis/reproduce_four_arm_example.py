#!/usr/bin/env python3
"""Recompute a complete primary four-arm AP contrast from retained predictions."""
from __future__ import annotations

import argparse
import importlib.metadata
import json
import sys
from pathlib import Path

for source in (Path(__file__).resolve().parent / "src", Path(__file__).resolve().parents[1] / "src"):
    sys.path.insert(0, str(source))
import numpy as np
from bootstrap_format_contrast import ARM_NAMES, run_contrast
from topic_c.manifest import sha256_file


def validate_example_files(root: Path, manifest: dict) -> None:
    root = root.resolve()
    for relative, digest in manifest["files"].items():
        path = Path(relative)
        if path.is_absolute() or ".." in path.parts or root not in (root / path).resolve().parents:
            raise ValueError(f"example files require safe relative paths: {relative}")
        if sha256_file(root / path) != digest:
            raise ValueError(f"example input hash mismatch: {relative}")


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--example", type=Path, default=Path(__file__).resolve().parent)
    parser.add_argument("--out", type=Path, required=True)
    parser.add_argument("--workers", type=int, default=1)
    args = parser.parse_args()
    root = args.example.resolve()
    manifest = json.loads((root / "example.json").read_text())
    validate_example_files(root, manifest)
    for package, expected in manifest["versions"].items():
        if importlib.metadata.version(package) != expected:
            raise RuntimeError(f"install pinned {package}=={expected} before reproducing")
    arms = manifest["arms"]
    result = run_contrast(
        endpoint="area", annotations=root / manifest["annotations"],
        prediction_paths=[root / arms[name]["prediction"] for name in ARM_NAMES],
        input_paths=[root / arms[name]["input"] for name in ARM_NAMES],
        run_paths=[root / arms[name]["run"] for name in ARM_NAMES],
        n_boot=2000, seed=manifest["seed"], expected_images=manifest["n_images"],
        annotation_sha256=manifest["files"][manifest["annotations"]],
        output=args.out.resolve(), workers=args.workers,
    )
    expected = json.loads((root / "expected.json").read_text())
    for key in ("delta_q", "delta_e"):
        for endpoint in expected["point"][key]:
            if not np.isclose(result["point"][key][endpoint], expected["point"][key][endpoint], rtol=0, atol=1e-10):
                raise RuntimeError(f"point estimate failed reproduction: {key}/{endpoint}")
            if not np.allclose(result["percentile_intervals"][key][endpoint], expected["percentile_intervals"][key][endpoint],
                               rtol=0, atol=1e-10, equal_nan=True):
                raise RuntimeError(f"interval failed reproduction: {key}/{endpoint}")
    if not np.allclose(result["percentile_intervals"]["delta_psi"], expected["percentile_intervals"]["delta_psi"],
                       rtol=0, atol=1e-10, equal_nan=True):
        raise RuntimeError("size-interaction interval failed reproduction")
    verification = dict(status="PASS", dataset="kitti", model="yolo11m", corruption="fog", severity=1,
                        n_boot=2000, point_delta_e_ap_points=100 * result["point"]["delta_e"]["all"],
                        interval_ap_points=[100 * v for v in result["percentile_intervals"]["delta_e"]["all"]],
                        absolute_tolerance_ap_scale_0_1=1e-10,
                        versions=manifest["versions"], example_manifest_sha256=sha256_file(root / "example.json"))
    args.out.with_suffix(".verification.json").write_text(json.dumps(verification, indent=2) + "\n")
    print(json.dumps(verification, indent=2))


if __name__ == "__main__":
    main()
