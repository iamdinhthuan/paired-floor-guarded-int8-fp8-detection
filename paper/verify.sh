#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

TEXLIVE_2026_BIN="/data_nvme/texlive/2026/bin/x86_64-linux"
if [[ -x "$TEXLIVE_2026_BIN/pdflatex" ]]; then
  PATH="$TEXLIVE_2026_BIN:$PATH"
  export PATH
fi
if ! command -v kpsewhich >/dev/null 2>&1 || ! kpsewhich stix.sty >/dev/null 2>&1; then
  echo "STIX fonts are required for a text-extractable CAS PDF." >&2
  exit 1
fi

./build.sh
python3 scripts/make_manifest.py
sha256sum -c SOURCE_MANIFEST.sha256 >/dev/null
echo "Neural Networks manuscript-source verification passed."
