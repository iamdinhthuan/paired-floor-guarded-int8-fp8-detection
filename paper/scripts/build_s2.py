#!/usr/bin/env python3
"""Build deterministic Supplementary Data S2 ZIP from compact evidence files."""

from __future__ import annotations

import hashlib
import json
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUTPUT = ROOT / "supplementary_data_s2.zip"
FIXED_TIME = (2026, 8, 20, 0, 0, 0)

FILES = [
    "generated/direct_format_contrast_cells.csv",
    "generated/direct_format_contrast_macro.csv",
    "generated/direct_absolute_guardrail.csv",
    "generated/direct_size_guardrail.csv",
    "generated/direct_heterogeneity_summary.csv",
    "generated/direct_heterogeneity_sensitivity.csv",
    "generated/direct_runtime_sensitivity.csv",
    "generated/codec_sensitivity.csv",
    "generated/direct_deployment_conditions.csv",
    "generated/decision_impact_cells.csv",
    "generated/decision_impact_audit.json",
    "generated/direct_data_dictionary.json",
    "generated/direct_evidence_audit.json",
    "generated/confirmatory_evidence_audit.json",
    "generated/quantization_graph_coverage_audit.json",
    "generated/cross_family_cells.csv",
    "generated/cross_family_q95_clean.csv",
    "generated/cross_family_interaction_cells.csv",
    "generated/cross_family_direct_cells.csv",
    "generated/cross_family_evidence_audit.json",
    "scripts/make_decision_impact.py",
    "scripts/rebuild_cross_family_evidence.py",
    "scripts/validate_package.py",
]
DIRS = [
    "confirmatory_evidence",
    "multiseed_evidence",
    "cross_family_evidence",
]

README = """# Supplementary Data S2

This archive contains compact manuscript-level ledgers, retained evidence
reports, SHA-256 audit records, and deterministic scripts supporting the article
“A Paired, Floor-Guarded Evaluation Protocol for TensorRT INT8 and FP8 Object
Detectors under Synthetic Image Corruptions.”

## Scope

The archive reproduces and validates manuscript-level summaries. It does not
rerun training, quantization, TensorRT engine construction, corruption
materialization, or detector inference. Raw datasets, checkpoints, engines, and
per-image prediction payloads are not included.

## Canonical signs

- `Q = AP_FP32 - AP_quantized`
- `E = Q_corrupt - Q_clean`
- `DeltaE = E_INT8 - E_FP8`
- equivalently, `DeltaE = (FP8-INT8)_corrupt - (FP8-INT8)_clean`

The architecture-portability mean INT8 `E` values are −6.81 AP for RT-DETR-L
and −6.21 AP for RetinaNet-R50-FPN-v2.

## Regeneration

From the parent manuscript package, run:

```bash
python3 scripts/make_decision_impact.py
python3 scripts/rebuild_cross_family_evidence.py
python3 scripts/validate_package.py
```

External filesystem paths inside retained JSON audits are provenance labels
from the original research environment; they are not required for the packaged
manuscript-level checks.
"""


def sha256_bytes(data: bytes) -> str:
    return hashlib.sha256(data).hexdigest()


def write_bytes(zf: zipfile.ZipFile, arcname: str, data: bytes) -> None:
    info = zipfile.ZipInfo(arcname, FIXED_TIME)
    info.compress_type = zipfile.ZIP_DEFLATED
    info.external_attr = 0o644 << 16
    zf.writestr(info, data)


def main() -> None:
    selected: list[tuple[str, Path]] = []
    for rel in FILES:
        path = ROOT / rel
        if not path.is_file():
            raise SystemExit(f"Required S2 file missing: {rel}")
        selected.append((rel, path))
    for directory in DIRS:
        base = ROOT / directory
        if not base.is_dir():
            raise SystemExit(f"Required S2 directory missing: {directory}")
        for path in sorted(base.rglob("*")):
            if path.is_file():
                selected.append((path.relative_to(ROOT).as_posix(), path))

    manifest: dict[str, str] = {"README_S2.md": sha256_bytes(README.encode("utf-8"))}
    for rel, path in selected:
        manifest[rel] = hashlib.sha256(path.read_bytes()).hexdigest()
    manifest_bytes = (json.dumps({"schema_version": 1, "files_sha256": manifest}, indent=2, sort_keys=True) + "\n").encode("utf-8")

    with zipfile.ZipFile(OUTPUT, "w") as zf:
        write_bytes(zf, "README_S2.md", README.encode("utf-8"))
        for rel, path in sorted(selected):
            write_bytes(zf, rel, path.read_bytes())
        write_bytes(zf, "S2_MANIFEST.json", manifest_bytes)
    print(f"Wrote {OUTPUT.name} with {len(selected) + 2} entries ({OUTPUT.stat().st_size} bytes)")


if __name__ == "__main__":
    main()
