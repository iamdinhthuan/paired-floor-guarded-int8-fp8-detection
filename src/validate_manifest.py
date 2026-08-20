#!/usr/bin/env python3
"""Fail closed before a manifest can drive labelled corruption inference."""
from __future__ import annotations

import argparse

from topic_c.manifest import read_manifest, validate_manifest


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--manifest", required=True)
    parser.add_argument("--annotations", required=True)
    parser.add_argument("--cache-root", required=True)
    parser.add_argument("--clean-root", required=True)
    parser.add_argument("--allow-clean-pixels", action="store_true", help="only for a clean-reference manifest")
    args = parser.parse_args()
    document = read_manifest(args.manifest)
    failures = validate_manifest(document, args.annotations, args.cache_root, args.clean_root, not args.allow_clean_pixels)
    if failures:
        print("MANIFEST INVALID")
        print("\n".join(f"- {failure}" for failure in failures))
        raise SystemExit(2)
    print(f"MANIFEST VALID records={len(document['records'])} sha256={document['manifest_sha256']}")


if __name__ == "__main__":
    main()
