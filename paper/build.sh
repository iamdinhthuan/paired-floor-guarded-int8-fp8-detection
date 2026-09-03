#!/usr/bin/env bash
set -euo pipefail

ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPOSITORY_ROOT="$(cd "$ROOT/.." && pwd)"
cd "$ROOT"

if command -v latexmk >/dev/null 2>&1; then
  latexmk -pdf -interaction=nonstopmode -halt-on-error main.tex
  latexmk -pdf -interaction=nonstopmode -halt-on-error supplement.tex
else
  command -v pdflatex >/dev/null 2>&1 || { echo "pdflatex is required" >&2; exit 127; }
  command -v bibtex >/dev/null 2>&1 || { echo "bibtex is required" >&2; exit 127; }
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  bibtex main
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error main.tex
  pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
  pdflatex -interaction=nonstopmode -halt-on-error supplement.tex
fi

mkdir -p preview
cp main.pdf preview/main.pdf
cp supplement.pdf preview/supplement.pdf
python3 "$REPOSITORY_ROOT/analysis/validate_cviu_paper_package.py" --paper-root "$ROOT"

printf '\nBuilt and validated:\n  %s\n  %s\n' "$ROOT/main.pdf" "$ROOT/supplement.pdf"
