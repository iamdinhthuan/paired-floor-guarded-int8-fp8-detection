# IVC submission source package

**Manuscript:** *A Paired, Floor-Guarded Evaluation Protocol for TensorRT INT8 and FP8 Object Detectors under Synthetic Image Corruptions*

This package contains the revised source prepared for submission to **Image and Vision Computing**, the compiled manuscript and supplement, a graphical abstract, highlights, manuscript-level evidence ledgers, and deterministic consistency checks.

## Core submission files

| File | Purpose |
|---|---|
| `main.tex` | Main article source using Elsevier's `cas-dc` class |
| `supplement.tex` | Standalone Supplementary Information source |
| `references.bib` | Bibliography database |
| `highlights.txt` | Five submission highlights, each no longer than 85 characters |
| `graphical_abstract.png` | Graphical abstract, 3000 × 1200 pixels |
| `preview/main.pdf` | Verified compiled manuscript |
| `preview/supplement.pdf` | Verified compiled Supplementary Information |
| `supplementary_data_s2.zip` | Compact evidence ledgers, audits, and regeneration scripts |
| `cover_letter.txt` | Suggested cover letter for the journal submission system |
| `AUTHOR_CHECKLIST.md` | Author-dependent checks that cannot be inferred from the source |
| `CHANGELOG.md` | Scientific, editorial, and packaging changes in this revision |

## Source layout

- `sections/` contains the six rewritten article sections.
- `figures/` contains the manuscript figures, including the new decision-impact diagnostic.
- `generated/` contains manuscript-level CSV/JSON ledgers and generated LaTeX tables.
- `confirmatory_evidence/`, `multiseed_evidence/`, and `cross_family_evidence/` contain compact retained evidence reports.
- `scripts/` contains deterministic regeneration, validation, packaging, and manifest utilities.
- `cas-dc.cls`, `cas-common.sty`, and `cas-model2-names.bst` are the local Elsevier CAS template components; see `CAS_TEMPLATE_NOTICE.txt`.

## Build

A TeX Live installation with `pdflatex` and `bibtex` is required.

```bash
./build.sh
```

The script builds `main.pdf` and `supplement.pdf`, copies verified previews to `preview/`, and runs package-level checks. On systems that expose the TeX binary as `bibtex.original`, the script uses it automatically.

For a full evidence regeneration and manifest verification:

```bash
./verify.sh
```

This command regenerates the decision-impact and canonical cross-family artifacts, rebuilds both PDFs, recreates Supplementary Data S2, writes `SOURCE_MANIFEST.sha256`, verifies every listed digest, and creates `../IVC_submission_revised_20260820.zip`.

## Reproducibility boundary

The package reproduces the **manuscript-level summaries** from frozen ledgers and checks their internal consistency. It does not rerun training, TensorRT engine construction, corruption materialization, or detector inference. Raw benchmark datasets, checkpoints, engines, and per-image prediction payloads are not redistributed because of licensing and size constraints. External artifact paths recorded in audit JSON files are provenance labels; the packaged validators operate only on files included here.

The canonical cross-family convention is

```text
Q = AP_FP32 - AP_quantized
E = Q_corrupt - Q_clean
DeltaE = E_INT8 - E_FP8
       = (FP8 - INT8)_corrupt - (FP8 - INT8)_clean
```

Under this convention, the mean INT8 `E` values in the architecture-portability stress cases are **−6.81 AP** for RT-DETR-L and **−6.21 AP** for RetinaNet-R50-FPN-v2.

## Submission note

No public archival DOI is asserted in this package. Complete the author-dependent items in `AUTHOR_CHECKLIST.md`, especially the funding statement and confirmation of the AI/graphical-abstract declarations, before uploading the final files.
