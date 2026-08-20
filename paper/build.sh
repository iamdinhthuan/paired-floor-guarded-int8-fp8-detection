#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
cd "$ROOT"

find_bibtex() {
  if command -v bibtex >/dev/null 2>&1; then
    command -v bibtex
  elif command -v bibtex.original >/dev/null 2>&1; then
    command -v bibtex.original
  elif [[ -x /usr/bin/bibtex.original ]]; then
    printf '%s\n' /usr/bin/bibtex.original
  else
    printf '%s\n' "ERROR: bibtex executable not found" >&2
    exit 127
  fi
}

BIBTEX_BIN="$(find_bibtex)"
LATEX_FLAGS=(-interaction=nonstopmode -halt-on-error -file-line-error)

rm -f main.{aux,bbl,blg,log,out,abs,fls,fdb_latexmk,pdf} \
      supplement.{aux,log,out,fls,fdb_latexmk,pdf}

pdflatex "${LATEX_FLAGS[@]}" main.tex
"$BIBTEX_BIN" main
pdflatex "${LATEX_FLAGS[@]}" main.tex
pdflatex "${LATEX_FLAGS[@]}" main.tex

pdflatex "${LATEX_FLAGS[@]}" supplement.tex
pdflatex "${LATEX_FLAGS[@]}" supplement.tex

mkdir -p preview
cp main.pdf preview/main.pdf
cp supplement.pdf preview/supplement.pdf

python3 scripts/validate_package.py

printf '\nBuilt:\n  %s\n  %s\n' "$ROOT/main.pdf" "$ROOT/supplement.pdf"
