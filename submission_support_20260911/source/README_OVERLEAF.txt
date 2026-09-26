NEUROCOMPUTING SOURCE — 21 September 2026
====================

Main document: main.tex
Supplement: supplement.tex (compile separately)
Compiler: pdfLaTeX; bibliography: BibTeX

Build the main article from this source directory:
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex

Build the supplement separately from this source directory (two passes):
  pdflatex supplement.tex
  pdflatex supplement.tex

In this editable submission-package layout, main.tex and supplement.tex are the
document entry points in this source folder; cover_letter_Neurocomputing.tex and
graphical_abstract_Neurocomputing.tex are one level above it.

The delivered 06_Overleaf_Source_Neurocomputing.zip and
07_Elsevier_Flat_Source_Neurocomputing.zip use a different layout: all four .tex
entry points are at the extracted archive's top level, and the graphical-
abstract figure path is normalized during packaging. From that top level, use:
  pdflatex main.tex
  bibtex main
  pdflatex main.tex
  pdflatex main.tex
  pdflatex supplement.tex
  pdflatex supplement.tex
  pdflatex cover_letter_Neurocomputing.tex
  pdflatex graphical_abstract_Neurocomputing.tex

Use the Neurocomputing-named ZIPs for this submission. The frozen v2.2.0 DOI
retains the earlier manuscript title; it does not identify this revised text.
