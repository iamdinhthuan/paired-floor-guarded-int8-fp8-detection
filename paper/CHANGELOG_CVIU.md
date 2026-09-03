# CVIU retargeting changelog

## Target-journal corrections

- Retargeted the manuscript and submission package from *Image and Vision Computing* to *Computer Vision and Image Understanding*.
- Replaced the old journal name, article-type wording, date, title, and fit argument in `cover_letter.txt`.
- Updated `README.txt` and local `zenodo.json` metadata.

## Manuscript framing

- New title: **A Paired, Floor-Guarded Evaluation Protocol for INT8 and FP8 Object Detectors under Image Corruptions**.
- Reframed the contribution as a reusable evaluation protocol for executable vision systems rather than a datatype leaderboard.
- Rewrote the abstract and keywords for robustness evaluation, image corruption, paired evaluation, PTQ and object detection.
- Added a system-level motivation in the Introduction and a dedicated “Positioning of the present study” subsection.
- Clarified throughout that the four-cell interaction is algebraic/descriptive and is not a causal difference-in-differences estimator.
- Replaced causal-sounding wording such as “induced/caused by corruption” where it could overstate identification.
- Strengthened the separation among clean fidelity, absolute corrupted AP, and corruption-associated interaction.

## CVIU literature added

- Wen et al. (2020), UA-DETRAC benchmark and protocol, DOI `10.1016/j.cviu.2020.102907`.
- Bonnaerens et al. (2022), Anchor pruning for object detection, DOI `10.1016/j.cviu.2022.103445`.
- Cheng et al. (2025), Adversarial intensity awareness for robust object detection, DOI `10.1016/j.cviu.2024.104252`.
- Liu et al. (2026), Lightweight and robust small-object detection in UAV imagery, DOI `10.1016/j.cviu.2026.104735`.

## Submission compliance

- Rewrote `highlights.txt` as five bullets, each <=85 characters.
- Added a Funding section.
- Activated and updated the declaration of generative AI and AI-assisted technologies.
- Corrected the Data and code availability statement to distinguish the compact journal package from the public reproducibility archive.
- Updated the supplement title and non-causal/selection-independent wording.
- Updated PDF metadata and short title.

## Review materials added

- `REVIEW_CVIU_VI.md`: detailed Vietnamese journal-fit and scientific review.
- `AUTHOR_CHECKLIST_CVIU.txt`: author-dependent verification and submission checklist.
- `CHANGELOG_CVIU.md`: this change summary.

## Items intentionally not fabricated or altered

- No new experimental results were invented.
- Existing numerical results, tables, figures and reported intervals were retained.
- The author-created graphical abstract was re-exported in journal-ready PNG, PDF, and TIFF formats; no generative-image system was used.
- Published GitHub release `v2.1.0` and archived it at version DOI
  `10.5281/zenodo.22275640` under concept DOI `10.5281/zenodo.22031663`.
