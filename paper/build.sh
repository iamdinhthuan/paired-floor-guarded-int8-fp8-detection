#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main_nn.tex
  latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
else
  command -v pdflatex >/dev/null 2>&1 || { echo "pdflatex is required" >&2; exit 127; }
  command -v bibtex >/dev/null 2>&1 || { echo "bibtex is required" >&2; exit 127; }
  pdflatex -interaction=nonstopmode -halt-on-error main_nn.tex
  bibtex main_nn
  pdflatex -interaction=nonstopmode -halt-on-error main_nn.tex
  pdflatex -interaction=nonstopmode -halt-on-error main_nn.tex
  pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
  pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
fi

mkdir -p preview
cp main_nn.pdf preview/main_nn.pdf
cp supplement.pdf preview/supplement.pdf

# The CVIU validator applies to the archived v2.1.0 package (main.tex). Run it
# only when validating that frozen release; the Neural Networks manuscript is
# main_nn.tex and has its own consistency checks in analysis/build_nn_tables.py.
if [ "${VALIDATE_CVIU_PACKAGE:-0}" = "1" ]; then
  python3 "$REPOSITORY_ROOT/analysis/validate_cviu_paper_package.py" --paper-root "$ROOT"
fi

printf '\nBuilt and validated:\n  %s\n  %s\n' "$ROOT/main_nn.pdf" "$ROOT/supplement.pdf"
