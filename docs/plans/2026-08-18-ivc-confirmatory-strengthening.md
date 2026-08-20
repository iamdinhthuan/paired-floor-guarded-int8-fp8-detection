# IVC Confirmatory Strengthening Implementation Plan

> Approved scope: staged confirmatory strengthening (review option 2).
> Canonical manuscript source: `paper/` only. Do not create a second editable
> paper tree. Remote execution root: `/home/thuan/topic_c_ivc`.

## Objective

Repair the current manuscript/source drift, narrow unsupported claims, and add
the highest-value confirmatory evidence requested by the external review:
locked VOC/KITTI evaluation partitions, rare-class bootstrap sensitivity,
strict reference parity, quantization coverage, corruption-realization
sensitivity, heterogeneity/weighting diagnostics, and stratified runtime.

## Safety and evidence rules

- Preserve all user edits unless a tested manuscript/evidence contract requires
  a change.
- Every generated number must come from a hash-bound machine-readable source.
- Freeze configs, partitions, manifests, and analysis schemas before remote
  execution.
- Never overwrite a completed evidence artifact; resume only after validation.
- Retain only `best.pt` checkpoints and compact evidence needed for audit.
- Keep citations in the author-requested 2023--2026 window; do not add the
  pre-2023 references suggested by the external review.
- Do not claim a public DOI, funding source, or author fact not supplied by the
  authors.

## Task 1 — Restore one internally consistent manuscript source

1. Add/update metadata regression tests for the approved title, 1--7 keyword
   rule, abstract/generated-artifact identity, bibliography closure, and exact
   CFF/Zenodo title linkage.
2. Run the focused tests and record the expected RED failures.
3. Make `paper/generated/direct_abstract.tex` the canonical abstract content
   and synchronize its literal CAS copy.
4. Adopt the narrow title: “A Paired, Floor-Aware Evaluation of INT8 and FP8
   Object Detectors under Image Corruptions”; synchronize short title,
   PDF metadata, CFF, and Zenodo metadata.
5. Remove the orphaned bibliography entry and restore the 2023--2026 citation
   window without fabricating recent replacements for historical provenance.
6. Reduce the contribution list to three scientific contributions; move
   historical digest/audit detail out of the active main narrative.
7. Replace universal “admissibility” wording with continuous clean-fidelity
   diagnostics; describe cross-family collapse as recorded
   recipe--architecture incompatibility.
8. Rebuild and run focused metadata/submission tests.

## Task 2 — Activate existing high-value sensitivity evidence

1. Add a failing builder/audit test requiring the active source package to
   expose the existing original-source versus JPEG-95 sensitivity evidence.
2. Move the compact codec-sensitivity result into the active Supplement and
   bind it to its existing validated evidence; avoid duplicating the legacy
   inactive appendix.
3. Extend the absolute-accuracy builder from existing metric records to emit
   matched-clean and corrupted small/large (or TT100K height) guardrails.
4. Add tests for units, endpoint separation, missing-positive reporting, and
   hash closure; then regenerate the compact tables.

## Task 3 — Add descriptive heterogeneity and runtime sensitivities

1. Add independent tests for cell SD/IQR, leave-one-dataset-out,
   leave-one-corruption-out, and image/object-weighted macro summaries.
2. Implement a compact analysis artifact from the validated 144-cell ledger;
   explicitly label all summaries descriptive.
3. Add runtime stratification by input geometry and YOLO capacity using the
   existing 36-condition, three-repetition ledger.
4. Add the compact results to Supplement and one main-text sentence/table only
   where they change interpretation.

## Task 4 — Fix the TT100K rare-class bootstrap estimand

1. Add characterization tests proving current duplicate-image multiplicity is
   preserved and proving the current ordinary bootstrap can change the set of
   categories contributing to macro AP.
2. Define a fixed-category-universe sensitivity estimator before implementation
   (weighted image/Bayesian bootstrap or another exact prespecified estimator),
   including behavior for rare categories and empty size strata.
3. Add failing synthetic tests with multi-label images, rare categories,
   duplicate draws, and absent-positive strata.
4. Implement the estimator and verify stock-equivalence on the unweighted point
   estimate plus deterministic schedule/hash behavior.
5. Benchmark 2,000 versus 10,000 draws on a bounded TT100K cell. Launch the
   10,000-draw sensitivity only if the measured memory/time envelope is safe.
6. Report ordinary versus fixed-universe results and Monte Carlo endpoint
   stability; do not present increased draw count as a cure for estimand drift.

## Task 5 — Freeze untouched confirmatory VOC/KITTI partitions

1. Audit existing train/validation/test manifests and prove which image IDs
   entered training, checkpoint selection, calibration, and prior evaluation.
2. Design deterministic train/selection/final partitions with class and object
   coverage diagnostics. The final partition must be excluded from training,
   checkpoint selection, decoder tuning, and calibration.
3. Add fail-closed tests for overlap, ordering, hashes, class coverage, and
   immutable split regeneration.
4. Freeze training/evaluation configs for YOLO11n/m/x on VOC and KITTI.
5. Run a one-checkpoint smoke test through training, export, reference,
   quantization, inference, metric, and completion validation.
6. If the smoke chain passes, launch the six-checkpoint sequential queue and
   retain only `best.pt` plus compact run/evidence records.
7. Evaluate the full primary corruption grid on the locked final partitions and
   generate a confirmatory analysis kept separate from the exploratory grid.

## Task 6 — Strict reference and quantization-treatment audit

1. Add tests/config guards requiring TF32-disabled reference builds and explicit
   PyTorch → FP32 ONNX Runtime → FP32 TensorRT parity records.
2. Run strict reference parity for the confirmatory VOC/KITTI checkpoints and a
   prespecified representative set of existing engines.
3. Build a graph-coverage analyzer that reports Q/DQ node counts, quantized
   weights/activations, scale axes, zero-point types, and operator coverage.
4. Where retained engines expose layer information, capture TensorRT inspector
   output and report low-precision/fallback layers without inferring unavailable
   kernel datatypes.
5. For RT-DETR/RetinaNet, run one prespecified diagnostic rescue experiment
   only after the baseline coverage report: alternative calibration or selective
   mixed precision. Treat it as diagnosis, not a new main comparison.

## Task 7 — Corruption-realization sensitivity

1. Freeze a targeted YOLO11m subset covering VOC, KITTI, and TT100K, four
   corruptions, severities 1/3/5, and at least three corruption seeds.
2. Use shared base noise fields scaled across severity and a fixed per-image blur
   angle across severity; preserve the current independently seeded grid as the
   primary historical treatment.
3. Add manifest/hash/seed/nesting tests before materialization.
4. Run paired INT8/FP8 evaluation and variance decomposition across image and
   corruption realization; label it sensitivity evidence.

## Task 8 — Final manuscript, verification, and clean Overleaf hand-off

1. Update Methods, Results, Discussion, Abstract, Conclusion, highlights, and
   Supplement from the completed validated artifacts only.
2. Keep the cross-family experiment as an architecture stress/cautionary case;
   do not restore “across detector families” to the title without family-level
   replication and coverage diagnosis.
3. Obtain author-provided funding, CRediT confirmation, archive URL, and DOI;
   fail submission audit while any placeholder remains.
4. Rebuild main PDF, Supplement PDF, graphical abstract, and source manifest.
5. Run the complete pytest suite, submission audit, evidence validators,
   LaTeX log audit, PDF visual inspection, and clean extracted-package compile.
6. Regenerate one source-only Overleaf upload folder/ZIP from canonical
   `paper/`; remove stale duplicate hand-offs only after verifying the new
   package.

## Execution checkpoints

- Checkpoint A: Tasks 1--3 complete; manuscript/source consistency restored.
- Checkpoint B: Task 4 complete; bootstrap sensitivity validated.
- Checkpoint C: Task 5 smoke chain passes; long retraining queue may launch.
- Checkpoint D: Tasks 5--7 evidence complete; manuscript integration begins.
- Checkpoint E: Task 8 passes all submission and extraction gates.

