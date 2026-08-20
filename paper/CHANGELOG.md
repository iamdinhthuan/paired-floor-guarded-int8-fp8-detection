# Revision changelog — 20 August 2026

## Scientific framing

- Repositioned the article from a nominal “FP8 versus INT8 robustness” comparison to an **evaluation-methodology contribution**.
- Replaced broad datatype claims with treatment-level language covering quantizer placement, calibration, Q/DQ coverage, mixed-precision execution, engine building, preprocessing, and decoding.
- Defined a finite-grid target for the 144-cell macro and explicitly separated it from a deployment-population estimand.
- Established an evidence hierarchy: exploratory factorial landscape, untouched holdout, targeted sensitivities, and architecture-portability stress cases.
- Replaced “floor-aware” with the more defensible term **floor-guarded** and stated that the absolute-AP guardrail is not a formal floor correction.

## New decision-impact contribution

- Added a deterministic cell-level reconstruction of the raw corrupted FP8–INT8 gap:
  `raw gap = matched-clean gap + clean-adjusted DeltaE`.
- Added `fig_decision_impact` and its source ledger/audit.
- Demonstrated that the raw corrupted gap favors FP8 in 134/144 cells, while clean adjustment changes interpretation in 56 cells.
- Bound the interaction interpretation to absolute corrupted AP and explicit AP<10/AP<5 inventories.

## Methods and statistics

- Rewrote the atomic four-cell design, sign conventions, paired image bootstrap, size interaction, and absolute-loss guardrail.
- Clarified that severity is an ordinal factor conditional on recorded materializations in the exploratory grid.
- Distinguished cell-wise percentile inventories from multiplicity-adjusted discoveries.
- Clarified uncertainty boundaries: evaluation-set resampling does not propagate dataset, training, calibration, engine-build, or corruption-population uncertainty.
- Added explicit codec-control, weighting, omission, runtime, class-universe, realization, and seed-sensitivity interpretation boundaries.

## Results and discussion

- Elevated the untouched VOC/KITTI rerun as the strongest split-independent conditional evidence.
- Separated matched-clean fidelity, absolute corrupted accuracy, and corruption-specific interaction throughout Results, Discussion, and Conclusion.
- Interpreted RT-DETR/RetinaNet experiments as **recipe-portability failures**, not evidence of intrinsic INT8 inferiority or FP8 superiority.
- Added practical reporting recommendations and expanded limitations for synthetic corruptions, finite-grid weighting, split reuse, seeds, runtime scope, graph-level precision evidence, and cross-family transfer.

## Evidence and reproducibility repairs

- Rebuilt all cross-family interaction artifacts under one canonical sign convention:
  `E = Q_corrupt - Q_clean`, `DeltaE = E_INT8 - E_FP8`.
- Corrected the architecture-level mean INT8 interactions to −6.81 AP (RT-DETR-L) and −6.21 AP (RetinaNet-R50-FPN-v2).
- Removed stale contradictory cross-family files and regenerated JSON/CSV/LaTeX audits.
- Replaced claims that LaTeX itself “fails closed” with an exact boundary: the external pipeline validates upstream hashes; LaTeX consumes prevalidated artifacts.
- Added package-level validators for row counts, formulas, sign inventories, citations, abstract/highlight constraints, graphical-abstract dimensions, placeholders, log warnings, and manifest hashes.

## Submission package

- Restored the author-created graphical abstract PNG from its vector PDF source and prepared a minimal standalone Overleaf package.
- Added the authors' confirmed no-specific-funding declaration and confirmed the manuscript authorship metadata.
- Archived software release `v2.0.0` under DOI `10.5281/zenodo.22031664` and updated the citation and availability metadata.
- Rewrote the title, abstract, highlights, Introduction, Related Work, Methods, Results, Discussion, Conclusion, and Supplementary Information.
- Added recent detector-PTQ, flexible 8-bit, and quantization/robustness references.
- Updated the data/code availability and generative-AI declarations without inventing a repository DOI or funding statement.
- Fixed the Elsevier CAS title-block overfull warning and all manuscript/supplement overfull, underfull, undefined-reference, and undefined-citation warnings.
- Added clean build, full verification, Supplementary Data S2, source manifest, cover letter, and author checklist workflows.
