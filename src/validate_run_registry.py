#!/usr/bin/env python3
"""Reject a run registry if precision formats did not consume identical bytes."""
from __future__ import annotations

import argparse
import json
from pathlib import Path

from topic_c.manifest import validate_shared_input_bytes


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--run-record", action="append", required=True)
    args = parser.parse_args()
    records = [json.loads(Path(path).read_text(encoding="utf-8")) for path in args.run_record]
    failures = validate_shared_input_bytes(records)
    if failures:
        print("RUN REGISTRY INVALID\n" + "\n".join(f"- {failure}" for failure in failures))
        raise SystemExit(2)
    print(f"RUN REGISTRY VALID records={len(records)}")


if __name__ == "__main__":
    main()
