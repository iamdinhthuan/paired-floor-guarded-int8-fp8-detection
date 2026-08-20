#!/usr/bin/env python3
"""Add a missing annotation binding to an otherwise complete cross-family run."""
from __future__ import annotations

import argparse
import json
import os
from pathlib import Path

from topic_c.manifest import sha256_file


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-record", required=True); parser.add_argument("--predictions", required=True)
    parser.add_argument("--annotations", required=True)
    args = parser.parse_args()
    run_path, prediction, annotations = map(lambda value: Path(value).resolve(),
                                             (args.run_record, args.predictions, args.annotations))
    run = json.loads(run_path.read_text())
    if run.get("dataset") != "tt100k" or run.get("prediction_sha256") != sha256_file(prediction):
        raise SystemExit("repair refused: run/prediction provenance mismatch")
    if "annotation_sha256" in run:
        raise SystemExit("repair refused: annotation binding already exists")
    run["annotation_sha256"] = sha256_file(annotations)
    temporary = run_path.with_suffix(run_path.suffix + ".repair")
    temporary.write_text(json.dumps(run, indent=2) + "\n")
    os.replace(temporary, run_path)
    print(json.dumps({"run_record": str(run_path), "run_record_sha256": sha256_file(run_path)}))


if __name__ == "__main__": main()
