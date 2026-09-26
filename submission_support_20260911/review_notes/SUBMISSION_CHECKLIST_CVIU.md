# CVIU submission checklist — 9 September 2026

## Completed locally

- [x] Treat the author's revised `submission_package/source/` as the sole active manuscript.
- [x] Independently audit scientific signs, scopes and headline values against retained ledgers; correct the identified wording ambiguities.
- [x] Regenerate the two central V4 tables identically from their bound inputs.
- [x] Recompute the KITTI final/YOLO11m/fog-1 example from predictions and annotations, without AP caches: zero point error and maximum interval error 8.22e-15 AP.
- [x] Make the clean-control sign, bootstrap multiplicity, realization dependence, INT8-only build-policy match and condition-wise runtime ratios explicit.
- [x] Correct the graphical-abstract source path and distinguish exploratory JPEG-95 from original-clean holdout evidence.
- [x] Preserve the six human authors and the confirmed no-grant Funding statement.
- [x] Compile both source ZIPs from fresh extraction (main, supplement, cover letter and graphical abstract); compare every rendered page with the delivered PDFs.
- [x] Remove build/cache artifacts from the source hand-off. The prior generated `main.bbl` is recoverable from the temporary pre-edit snapshot.
- [x] Five highlights are within 85 characters; all 44 cited bibliography keys resolve. The bibliography retains the author's additional unused entries.
- [x] Verify the actual freshly extracted evidence ZIP with its included code: 1,902 hashes, 55 active source/PDF bindings and V4 numerical regeneration passed.
- [x] Final full repository regression run: 702 passed, 10 skipped. Independent scientific, visual and binding-code reviews passed.
- [x] Record SHA-256 checksums for the entire local package in `SHA256SUMS.txt` (excluding that checksum file itself).

## Before author submission/publication

- [x] Push the aligned code and manuscript to GitHub: commit `c9899eace4725f9f7fc41af4afda58698235ad07`, tag `v2.2.0`.
- [x] Reserve Zenodo DOI `10.5281/zenodo.22664869` in a new version of concept `10.5281/zenodo.22031663`, update the manuscript and rebuild the PDFs/source/evidence bindings.
- [ ] Complete and independently verify Zenodo publication. During this attempt Zenodo returned 504/timeouts, including for its homepage and the prior public record. A bounded background job retries the same draft, checks uploaded MD5 values and exact file membership, and publishes only after they match. Read `FINAL_VERIFICATION.json` for the last verified publication state; the reserved DOI must not be represented as registered until publication succeeds.
- [x] Check KITTI's published CC BY-NC-SA 3.0 terms and include attribution, transformation details and the license URL in the evidence README, separately from the original software's MIT license.
- [ ] Attach `CVIU_Reviewer_Evidence.zip` as research evidence or arrange permitted reviewer access. It is not the Overleaf/source ZIP.
- [ ] Obtain final all-author approval of this version, affiliations/ORCIDs/CRediT, declarations, cover letter and submission exclusivity.
- [ ] Retain the honest missing-historical-model-snapshot disclosure unless genuine historical records recover it. Do not substitute a current model ID.
- [ ] Confirm the journal portal's current file categories and inspect the PDF it generates before pressing Submit.

The main text uses Zenodo's actually reserved v2.2.0 DOI. Its v2.2.0 availability wording is prepared for this release; do not submit it as an available archive until the pending publication step has been verified.

## Actual upload names

- Manuscript: `01_CVIU_main_revised.pdf` (13 pages).
- Supplement: `02_CVIU_supplement_revised.pdf` (19 pages).
- Highlights: `03_highlights_CVIU.txt`.
- Graphical abstract: `04_graphical_abstract_CVIU.pdf` or its PNG alternative.
- Cover letter: `05_CVIU_cover_letter.pdf`.
- Overleaf: `06_Overleaf_Source_CVIU.zip` (one ZIP upload).
- Journal source alternative: `07_Elsevier_Flat_Source_CVIU.zip`.
- Separate evidence: `CVIU_Reviewer_Evidence.zip`; see its README for verification and limits.

Source ZIPs each contain 50 necessary/hand-off files, not datasets or checkpoints. Do not upload `review_notes/` as manuscript material. General Elsevier guidance defers to each journal's actual portal configuration: https://www.elsevier.support/publishing/answer/how-do-i-prepare-my-files-for-submission-in-editorial-manager

## Evidence and limits

`FINAL_VERIFICATION.json` records source-ZIP/PDF checks. `FINALIZATION_20260909.md` records testing and the review process. Eight BibTeX empty-pages warnings concern retained conference/preprint entries without page ranges; no pages were fabricated to silence them. Final LaTeX logs have no unresolved citations/references or overfull/underfull boxes.

No new experiments are required by the completed internal audit. Clean-competitive additional detector families, lossless common-path four-arm studies and natural-shift datasets remain optional scope expansions, not promised completed work. Internal verification does not guarantee editorial acceptance, bootstrap coverage or independent external replication.
