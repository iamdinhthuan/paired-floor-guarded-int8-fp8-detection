#!/usr/bin/env python3
"""Capture executable environment and immutable input hashes before a run."""
from __future__ import annotations

import argparse
import hashlib
import importlib.metadata as metadata
import json
import platform
import shutil
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path

from topic_c.manifest import sha256_file


def command_output(command: list[str]) -> dict[str, object]:
    try:
        result = subprocess.run(command, text=True, capture_output=True, check=False, timeout=30)
        return {"command": command, "returncode": result.returncode, "stdout": result.stdout.strip(), "stderr": result.stderr.strip()}
    except (OSError, subprocess.SubprocessError) as exc:
        return {"command": command, "error": str(exc)}


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--out", required=True)
    parser.add_argument("--asset", action="append", default=[], help="immutable engine, script, annotation, or calibration file")
    args = parser.parse_args()
    output = Path(args.out)
    if output.exists():
        raise SystemExit(f"refusing to overwrite provenance artifact: {output}")
    assets = []
    for raw_path in args.asset:
        path = Path(raw_path)
        if not path.is_file():
            raise SystemExit(f"asset is not a file: {path}")
        assets.append({"path": str(path.resolve()), "bytes": path.stat().st_size, "sha256": sha256_file(path)})
    packages = {}
    for name in ("tensorrt", "numpy", "opencv-python", "Pillow", "pycocotools", "ultralytics", "scipy", "cuda-python"):
        try:
            packages[name] = metadata.version(name)
        except metadata.PackageNotFoundError:
            packages[name] = None
    trtexec = shutil.which("trtexec")
    document = {
        "schema_version": 1,
        "captured_at_utc": datetime.now(timezone.utc).isoformat(),
        "python": {"executable": sys.executable, "version": sys.version},
        "platform": platform.platform(),
        "packages": packages,
        "nvidia_smi": command_output(["nvidia-smi", "--query-gpu=index,name,uuid,driver_version,memory.total", "--format=csv,noheader"]),
        "trtexec": {"path": trtexec, "version": command_output([trtexec, "--version"]) if trtexec else None},
        "assets": assets,
    }
    output.parent.mkdir(parents=True, exist_ok=True)
    output.write_text(json.dumps(document, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({"out": str(output), "assets": len(assets), "trtexec_found": bool(trtexec)}, indent=2))


if __name__ == "__main__":
    main()
