#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

./build.sh
python3 scripts/make_manifest.py
sha256sum -c SOURCE_MANIFEST.sha256 >/dev/null
echo "CVIU manuscript-source verification passed."
