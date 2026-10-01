# Neural Networks manuscript and reproducibility-package metadata

## Manuscript

**Title:** *Quantization fragility under image corruption is recipe-dependent: paired evidence from INT8 and FP8 object detectors*

**Target journal:** *Neural Networks* (Elsevier)

This study shows that whether an 8-bit quantized object detector remains
reliable under image corruption cannot be ordered by the format label: the
corruption-associated change in the FP8--INT8 AP gap reverses sign across
dataset and model-scale cells even though FP8 exceeds INT8 on absolute AP in
every matched block, and where the fragility lives is a property of the
recipe and the model region. A paired, codec-controlled protocol estimates
the interaction within common-image bootstrap samples. A matched-recipe decomposition of RetinaNet localizes a
severe INT8 collapse to regression-head quantization, and a two-fold
corruption-aware calibration intervention with held-out families finds its
own effect context-dependent across fold-by-dataset cells---a
corruption-specific benefit in one cell, a significant harm in another---so
corruption-aware calibration is not supported as a reliable repair. Selective
precision (regression head in floating point) is the validated
retraining-free fix.

## Authors

1. Dinh Thuan Nguyen
2. Lam Phuong Nguyen
3. Vinh Huy Nguyen
4. Sy Vu Quang
5. Mohan Rajesh Elara
6. Anh Vu Le (corresponding author)

Author order and contribution roles must remain synchronized with `main_nn.tex`,
`supplement.tex`, `.zenodo.json`, and `CITATION.cff`.

## Local package map

- `main_nn.tex`, `main_nn.pdf`: main manuscript source and current compiled preview.
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

`build.sh` compiles the article and Supplementary File S1 independently and
copies the verified PDFs into `preview/`. The legacy CVIU package validator
(which checks the archived `main.tex` release) runs only when invoked with
`VALIDATE_CVIU_PACKAGE=1`; it is off by default.
`verify.sh` additionally refreshes and verifies `SOURCE_MANIFEST.sha256`. It
does not retrain detectors or rerun TensorRT inference.

From the repository root, the clean journal-upload directory can be created
with:

```bash
python3 analysis/build_cviu_submission_docx.py --paper-root paper
python3 analysis/build_cviu_submission_package.py \
  --paper-root paper --output NN_SUBMISSION_READY
```

## Reproducibility boundary

The compact evidence release is intended to reproduce manuscript summaries
from frozen CSV/JSON ledgers and deterministic validation scripts. It does not
redistribute third-party datasets, trained checkpoints, TensorRT engines, raw
predictions, or machine-specific caches. Dataset licenses remain with their
respective owners. The final release must document the regeneration or audit
path for every omitted artifact on which a reported result depends.

The CVIU-aligned `v2.1.0` source and compact-evidence package is archived at
[10.5281/zenodo.22275640](https://doi.org/10.5281/zenodo.22275640). Its
all-versions concept DOI is
[10.5281/zenodo.22031663](https://doi.org/10.5281/zenodo.22031663), which the
manuscript and supplement cite; the Neural Networks release `v3.0.3` is
archived under the same concept DOI with version DOI
[10.5281/zenodo.23082850](https://doi.org/10.5281/zenodo.23082850).

## License

Original project software and documentation are released under the MIT License.
Datasets, model weights, third-party code, and other external artifacts retain
their original licenses and terms.

## AI transparency

AI tools are not authors, creators, or contributors to this package. Any actual
AI assistance in manuscript preparation or the research workflow must be
disclosed according to Elsevier policy and verified by the human authors. See
`AUTHOR_CHECKLIST_CVIU.txt` for the required author confirmation.
