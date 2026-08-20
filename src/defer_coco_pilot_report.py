#!/usr/bin/env python3
"""Write the immutable COCO pilot report immediately after its bootstrap completion marker."""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
import time
from pathlib import Path


def canonical_hash(document: dict, excluded: str) -> str:
    payload = {key: value for key, value in document.items() if key != excluded}
    return hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()


def complete_bootstrap(path: Path, plan_sha: str) -> bool:
    if not path.is_file():
        return False
    try:
        document = json.loads(path.read_text(encoding="utf-8"))
    except json.JSONDecodeError:
        return False
    return (document.get("plan_sha256") == plan_sha and document.get("cells") == 72 and document.get("n_boot") == 500
            and document.get("all_bootstrap_input_hashes_validated") is True and len(document.get("artifacts_sha256", {})) == 72)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--project-root", required=True); parser.add_argument("--plan", required=True); parser.add_argument("--attempt", default="pilot_117_v1")
    parser.add_argument("--out", required=True); parser.add_argument("--poll-seconds", type=int, default=60)
    args = parser.parse_args()
    root, plan_path, output = Path(args.project_root).resolve(), Path(args.plan).resolve(), Path(args.out).resolve()
    if output.exists():
        raise SystemExit(f"COCO REPORT REFUSED: report already exists: {output}")
    plan = json.loads(plan_path.read_text(encoding="utf-8"))
    if plan.get("plan_sha256") != canonical_hash(plan, "plan_sha256"):
        raise SystemExit("COCO REPORT REFUSED: invalid frozen plan")
    bootstrap = root / "outputs" / "reports" / f"{args.attempt}_bootstrap_complete.json"
    while not complete_bootstrap(bootstrap, plan["plan_sha256"]):
        print(f"COCO REPORT waiting for complete bootstrap: {bootstrap}", flush=True)
        time.sleep(args.poll_seconds)
    subprocess.run([sys.executable, str(root / "src" / "build_pilot_report.py"), "--plan", str(plan_path), "--project-root", str(root),
                    "--attempt", args.attempt, "--out", str(output)], check=True)


if __name__ == "__main__":
    main()
