# CVIU manuscript and reproducibility-package metadata

## Manuscript

**Title:** *A Paired, Floor-Guarded Evaluation Protocol for INT8 and FP8 Object Detectors under Image Corruptions*

**Target journal:** *Computer Vision and Image Understanding* (CVIU)

This study presents a paired evaluation protocol for separating an INT8--FP8
difference already present on matched clean images from the change associated
with image corruption. The reported TensorRT engines are executable treatment
instances used to evaluate the protocol; they are not interpreted as a
universal ranking of numerical formats. Clean fidelity and absolute corrupted
accuracy are retained as guardrails against favorable relative comparisons near
a shared accuracy floor.

## Authors

1. Dinh Thuan Nguyen
2. Lam Phuong Nguyen
3. Vinh Huy Nguyen
4. Sy Vu Quang
5. Mohan Rajesh Elara
6. Anh Vu Le (corresponding author)

Author order and contribution roles must remain synchronized with `main.tex`,
`supplement.tex`, `.zenodo.json`, and `CITATION.cff`.

## Local package map

- `main.tex`, `main.pdf`: main manuscript source and current compiled preview.
- `supplement.tex`, `supplement.pdf`: Supplementary File S1 source and preview.
- `references.bib`: shared bibliography.
- `figures/`: publication figures used by the LaTeX sources.
- `generated/`: generated LaTeX tables used by the manuscript and supplement.
- `graphical_abstract.tif`: preferred graphical-abstract submission file.
- `graphical_abstract.png`: graphical-abstract preview.
- `Highlights.docx`: Elsevier Highlights upload; `highlights.txt` is its text source.
- `cover_letter.txt`: editable cover-letter source.
- `AUTHOR_CHECKLIST_CVIU.txt`: author-controlled checks before submission.
- `.zenodo.json`, `CITATION.cff`: release and citation metadata.

For Overleaf, upload the LaTeX sources together with `figures/`, `generated/`,
the bibliography, and the required Elsevier class/style files. If Editorial
Manager requests LaTeX source, create a separate flat archive because Elsevier
Editorial Manager does not process source subfolders.

## Build and validation

A TeX Live installation with `latexmk` is preferred; `pdflatex` plus `bibtex`
is supported as a fallback.

```bash
cd paper
./build.sh
./verify.sh
```

`build.sh` compiles the article and Supplementary File S1 independently, copies
the verified PDFs into `preview/`, and runs the CVIU package validator.
`verify.sh` additionally refreshes and verifies `SOURCE_MANIFEST.sha256`. It
does not retrain detectors or rerun TensorRT inference.

From the repository root, the clean journal-upload directory can be created
with:

```bash
python3 analysis/build_cviu_submission_docx.py --paper-root paper
python3 analysis/build_cviu_submission_package.py \
  --paper-root paper --output CVIU_SUBMISSION_READY
```

## Reproducibility boundary

The compact evidence release is intended to reproduce manuscript summaries
from frozen CSV/JSON ledgers and deterministic validation scripts. It does not
redistribute third-party datasets, trained checkpoints, TensorRT engines, raw
predictions, or machine-specific caches. Dataset licenses remain with their
respective owners. The final release must document the regeneration or audit
path for every omitted artifact on which a reported result depends.

The project is versioned through the Zenodo concept DOI:
[10.5281/zenodo.22031663](https://doi.org/10.5281/zenodo.22031663). Cite the
version-specific DOI for the exact release used in an analysis. The v2.1.0
release is the CVIU-aligned source and compact-evidence package.

## License

Original project software and documentation are released under the MIT License.
Datasets, model weights, third-party code, and other external artifacts retain
their original licenses and terms.

## AI transparency

AI tools are not authors, creators, or contributors to this package. Any actual
AI assistance in manuscript preparation or the research workflow must be
disclosed according to Elsevier policy and verified by the human authors. See
`AUTHOR_CHECKLIST_CVIU.txt` for the required author confirmation.
