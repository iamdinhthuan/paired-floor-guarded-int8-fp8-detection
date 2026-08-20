#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

python3 scripts/make_decision_impact.py
python3 scripts/rebuild_cross_family_evidence.py
./build.sh
python3 scripts/build_s2.py
python3 scripts/make_manifest.py
python3 scripts/validate_package.py --check-manifest
sha256sum -c SOURCE_MANIFEST.sha256 >/dev/null
python3 scripts/build_submission_zip.py

echo "Full package verification passed."
