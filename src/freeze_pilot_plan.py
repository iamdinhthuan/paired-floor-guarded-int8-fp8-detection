#!/usr/bin/env python3
"""Freeze a reviewed pilot proposal to exact, validated input-manifest hashes.

The proposal deliberately contains ``{manifest8}`` placeholders.  This tool is
the one-way admission step that resolves those placeholders only after every
full manifest has a matching validation-completion marker.  It creates a new
document and never alters the original proposal.
"""
from __future__ import annotations

import argparse
import json
from datetime import datetime, timezone
from pathlib import Path

from pilot_registry import canonical_hash
from topic_c.manifest import read_manifest


def completed_manifest(path: Path) -> dict:
    manifest = read_manifest(path)
    marker = path.with_suffix(path.suffix + ".complete")
    if not marker.is_file():
        raise SystemExit(f"PILOT FREEZE REFUSED: missing validation marker: {path}")
    if marker.read_text(encoding="utf-8").strip() != manifest["manifest_sha256"]:
        raise SystemExit(f"PILOT FREEZE REFUSED: marker hash differs: {path}")
    return manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--proposal", required=True)
    parser.add_argument("--project-root", required=True)
    parser.add_argument("--out", required=True)
    args = parser.parse_args()

    out = Path(args.out)
    if out.exists():
        raise SystemExit(f"refusing to overwrite frozen plan: {out}")
    proposal = json.loads(Path(args.proposal).read_text(encoding="utf-8"))
    if proposal.get("plan_sha256") != canonical_hash(proposal, "plan_sha256"):
        raise SystemExit("PILOT FREEZE REFUSED: proposal hash mismatch")
    root = Path(args.project_root).resolve()
    frozen_runs = []
    for run in proposal.get("runs", []):
        template = run.get("condition_id_template", "")
        if template.count("{manifest8}") != 1:
            raise SystemExit("PILOT FREEZE REFUSED: malformed condition-ID template")
        relative_manifest = run["input_manifest_path"]
        manifest = completed_manifest(root / relative_manifest)
        frozen = dict(run)
        frozen["input_manifest_sha256"] = manifest["manifest_sha256"]
        frozen["condition_id"] = template.replace("{manifest8}", manifest["manifest_sha256"][:8])
        frozen_runs.append(frozen)
    if len(frozen_runs) != 117 or len({run["condition_id"] for run in frozen_runs}) != len(frozen_runs):
        raise SystemExit("PILOT FREEZE REFUSED: expected 117 unique frozen conditions")
    document = {
        "schema_version": 1,
        "frozen_at_utc": datetime.now(timezone.utc).isoformat(),
        "source_proposal_sha256": proposal["plan_sha256"],
        "engine_registry_sha256": proposal["engine_registry_sha256"],
        "matrix_sha256": proposal["matrix_sha256"],
        "execution_policy": proposal["execution_policy"],
        "runs": frozen_runs,
    }
    document["plan_sha256"] = canonical_hash(document, "plan_sha256")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(f"PILOT FROZEN runs={len(frozen_runs)} sha256={document['plan_sha256']}")


if __name__ == "__main__":
    main()
