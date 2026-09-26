# Neural Networks upgrade — consolidated gap map and evidence plan

Date: 2026-09-23. Scope: consolidation of seven parallel read-only review
tracks (prior art, venue fit, statistics, claim–evidence, mechanism,
adversarial editorial, reproducibility) plus the two clean-only RetinaNet
diagnostics executed on the RTX 5090. This document is a working review
note, not manuscript text.

## 1. Verdict shared by all seven tracks

- The manuscript is technically careful but currently reads as a
  measurement/evaluation protocol. That is a plausible reason for the
  Neurocomputing desk rejection and remains the main risk at Neural
  Networks.
- Venue scope is adequate (neural-network quantization, reliability);
  the most plausible section is Learning Systems. Scope fit does not fix
  novelty.
- The paired clean/corruption estimands are valid descriptive contrasts
  but are not themselves new statistical machinery.
- A submission-ready upgrade needs at least one of: a mechanism that
  predicts failures, a validated intervention, or a controlled
  recipe-level decomposition that travels across recipes/architectures.

## 2. New evidence since the review request (development-only)

Clean AP on KITTI RetinaNet-R50-FPN-v2, 1,496 evaluation images,
identical build policy (TensorRT 11.1.0.106, TF32 off):

| Treatment | Calibration | Clean AP |
|---|---|---:|
| FP32 | — | 50.4387 |
| INT8 entropy | legacy YOLO letterbox (mismatched) | 6.7193 |
| INT8 entropy | matched RetinaNet contract, 128 img | 27.9064 |
| INT8 entropy, Conv+Add only | matched, 128 img | 15.7458 |
| INT8 max | matched, 128 img | 23.9351 |
| INT8 entropy | matched, full 512-img pool | 28.1827 |
| INT8 entropy, frozen FP8 compute-site mask (128 nodes) | matched, 512 img | 28.7403 |
| INT8 heads only (`/head/`, backbone FP32) | matched, 512 img | 17.0907 |
| INT8 backbone+FPN only (`/backbone/`, heads FP32) | matched, 512 img | 49.1363 |
| INT8 classification head only | matched, 512 img | 50.5726 |
| INT8 regression head only | matched, 512 img | 16.4137 |
| INT8 all-except-regression-head | matched, 512 img | 49.4966 |
| FP8 | matched, 128 img | 48.5004 |

Interpretation, all diagnostic-level:

- The calibration input contract explains a large share (+21.19 AP) of
  the historical RetinaNet INT8 collapse but not all of it
  (−22.53 AP remaining vs FP32).
- The residual gap is not explained by quantizable-op coverage
  (Conv+Add restriction is worse), by the range estimator
  (max calibration is worse than entropy), or by calibration budget
  (512-image pool yields 28.18 AP vs 27.91 for the 128-image subset).
- INT8 restricted to exactly the FP8 compute sites (frozen shared mask,
  same nodes/same preprocessing/same calibration bytes/same build
  policy) still yields only 28.74 AP. The residual ~22 AP gap is
  therefore most consistent with INT8 numerics on this graph rather
  than node selection, coverage, estimator, or calibration budget.
- Consequence: the shared-mask arm turns the FP8-vs-INT8 contrast into
  a much cleaner comparison — identical quantized compute sites — and
  the ~20 AP cost on those sites is now attributable to INT8 numerics
  (modulo replayed Q/DQ attachment details), not recipe coverage.
- Region sweep (recipe_v4) localizes the fragility: quantizing only the
  detection heads collapses AP to 17.09 while quantizing only the
  backbone+FPN leaves 49.14 (−1.30 vs FP32). Head-region INT8 is
  necessary and near-sufficient for the residual gap; a heads-excluded
  selective-precision INT8 recipe is already clean-competitive
  (49.14 vs FP8 full-graph 48.50).
- Head refinement (recipe_v5) isolates the fragile component further:
  quantizing only the classification head is harmless (50.57 ≈ FP32)
  while quantizing only the box regression head alone collapses AP to
  16.41.
- Minimal selective-precision recipe (recipe_v6): INT8 everywhere
  except the regression head recovers to 49.50 AP — −0.94 vs FP32 and
  +1.00 above FP8 full-graph. The regression head is confirmed as the
  dominant fragile component at clean level, and a clean-competitive
  INT8 RetinaNet baseline now exists without retraining. This closes
  Phase A: the recipe decomposition has identified the fragile region
  and produced a viable non-YOLO comparator arm.
- Note the non-additivity: full INT8 (27.91) outperforms heads-only
  quantization (17.09), so region effects interact; do not interpret
  the arms additively.
- The legacy-vs-matched contrast is a recipe-level contrast, not a pure
  format or pure normalization claim.
- Fixing preprocessing is an engineering correction, not novelty.
- The historical RetinaNet INT8 evidence must be relabelled as
  "legacy calibration contract"; it must not be described as a general
  portability failure until decomposition finishes.

## 3. Gap map by review track

### 3.1 Prior art / novelty

Closest themes: quantization robustness under input degradation,
degradation-aware calibration (Karimov et al. already tried mixed
clean–corrupt calibration), effective robustness (Taori et al.),
detector PTQ under sensor noise (InlierQ), test-time quantized
adaptation with perturbation consistency (TTAQ), absolute vs
clean-normalized corruption reporting (Michaelis et al.).

Gaps that remain defensible:

- Executable-engine evidence: most prior work quantizes in PyTorch or
  reports ONNX-level metrics, not deployed TensorRT engines with
  end-to-end decode/NMS.
- Recipe-level confounding is rarely measured: calibration contract,
  Q/DQ attachment, operator coverage, and build policy are usually
  bundled into "INT8" vs "FP8" labels.
- No existing work quantifies how much of a published "format effect"
  survives a corrected pipeline contract — our +21.19 AP preprocessing
  recovery is exactly this kind of measurement.

### 3.2 Venue fit

- Learning Systems section is the target; Mathematical and
  Computational Analysis is not (no new statistical theory).
- Framing must be methods-first: "when does quantization change
  corruption robustness and why", not "a protocol for reporting".
- All stale CVIU/Neurocomputing wording must be removed before
  submission.

### 3.3 Statistics / estimands

- Interaction ΔE algebra is correct but not robustness-pure: at equal
  relative retention, a positive clean gap shrinks under corruption and
  yields a negative interaction. The mean decomposition on the six
  holdout blocks is −0.4890 AP (clean-gap contraction) and −0.0608 AP
  (retention difference), summing to −0.5498 AP. Descriptive algebra,
  not causal attribution.
- CI on the headline is conditional on fixed engines/corruptions;
  seed-level variation is at least as large as the headline.
- Required additions for an upgraded paper: absolute corrupted AP
  alongside every normalized/interaction number; retention reported
  per arm; a baseline response model evaluated out-of-sample, or
  clean-performance-matched comparisons; grouped uncertainty where
  scene dependence exists.

### 3.4 Claim–evidence map

- "Decision impact" is currently a reporting property, not a measured
  deployment decision; downgrade wording or add an actual
  utility/constraint analysis.
- "56/144 sign disagreements" must not headline: most cells are
  compatible with zero and the estimands answer different questions.
- Cross-family claims are unsupported while non-YOLO INT8 baselines
  collapse (RT-DETR ~1 AP; legacy RetinaNet 7.1 AP). The corrected
  RetinaNet INT8 (27.9 AP) is still not clean-competitive.
- The VOC–YOLO11x clean→corrupt ranking reversal (+0.2291 → −0.3762
  AP) is real but small and single-block; it is a hypothesis, not a
  result.

### 3.5 Mechanism / intervention design

Candidate chain: internal signal → predicts failure → intervene on
that factor → failure decreases on conditions not used for development.

- Most testable signal: per-layer/per-stage activation-range drift or
  clipping under corruption, measured against clean; correlate with
  corruption-associated AP change.
- Intervention candidates (freeze rule on calibration/development
  data, evaluate on held-out conditions): corruption-aware or mixed
  calibration vs clean-only calibration at matched budget; selective
  higher-precision fallback for the most drift-sensitive nodes.
- Required controls: random layer selection and clean-only sensitivity
  selection at the same budget; mixed calibration must be compared
  against Karimov-style baselines and reported with clean cost.

### 3.6 Adversarial editorial assessment

- Highest desk-reject risk remains "incremental benchmarking study".
- Strongest single upgrade: one clean-competitive non-YOLO comparator
  plus a mechanism/intervention result that survives controls.
- If RetinaNet INT8 cannot be made clean-competitive, options: fix the
  recipe (in progress), switch the non-YOLO comparator to RT-DETR after
  its own diagnosis, or drop cross-family claims and bound the paper to
  YOLO-family recipe effects — the weakest option for novelty.

### 3.7 Reproducibility

Completed: decoder-aware calibration preprocessing with refusal on
mismatch; hash-bound registries; source-ONNX isolation; pilot output
immutability; 42 local + 10 remote tests.

Open hardening (recommended, not blocking): torchvision
GeneralizedRCNNTransform parity check for preprocess_retinanet (current
code is internally consistent, not proven exactly equal); golden
decoder parity tests; COCO eval annotation/image-ID validation; engine
output-name checks; precision consistency check in build_engine;
preprocessing consistency inside shared-mask validation; full remote
environment manifest; contract version in registries; atomic JSON
writes.

## 4. Novelty candidates, ranked by feasibility under current constraints

Constraint: calibration + inference only on one shared RTX 5090; no
retraining.

1. Recipe-level determinant study (highest feasibility): factorial
   separation of calibration contract, Q/DQ attachment, operator
   coverage, calibration budget, and format, with paired intervals.
   The preprocessing finding is already evidence this matters.
2. Mechanistic activation-range study (medium): capture per-stage
   activation ranges clean vs corrupted; test whether drift/clipping
   predicts corruption-associated AP change; needs an instrumentation
   pass on the ONNX/FP32 path, no retraining.
3. Calibration intervention (medium): frozen corruption-aware
   calibration rule vs clean-only at matched budget; report clean cost
   and corrupted benefit; must beat mixed-calibration baselines from
   prior work.
4. Cross-family validation (blocked until a non-YOLO INT8 baseline is
   clean-competitive).

## 5. Upgrade roadmap

Phase A (complete, 2026-09-24): recipe decomposition for RetinaNet
INT8 finished — coverage, estimator, and budget do not explain the
residual gap; the regression head is the fragile region, and
INT8-except-regression-head is clean-competitive (49.50 AP).

Phase B: pick the viable comparator set; freeze a paired-protocol
experiment on COCO (reacquisition complete 2026-09-23: 5,000 val +
512 calibration images verified at
data/datasets/coco_pilot_v1_20260923 on the RTX 5090 host) plus
KITTI/VOC.

Phase C: activation-range instrumentation and mechanism test.

Phase D: frozen intervention evaluated on held-out corruption
conditions with the required controls.

Phase E: manuscript rewrite — methods-first framing, Learning Systems
positioning, stale-venue cleanup, relabelled legacy RetinaNet evidence,
absolute AP everywhere, no universal INT8/FP8 claims.

Kill criteria: if the effect of interest vanishes under corrected
recipes and clean-performance matching, do not build a method claim on
it; report the recipe-confound study as the contribution instead.

## 6. Evidence status

Established (diagnostic-level): FP32 reference reproducibility;
preprocessing-contract confound (+21.19 AP); coverage/estimator
negative results; FP8 near-FP32 fidelity under matched contract;
regression-head fragility (head-only 16.41 vs cls-head-only 50.57);
clean-competitive selective-precision INT8 (all-except-reg-head
49.50 vs FP32 50.44, FP8 48.50); COCO pilot data verified (5,000
val + 512 calibration, hash-checked).

Not established: statistical significance of any new arm (single
build, single calibration subset); causal format attribution; the
correct comparator for a robustness study; whether selective
precision preserves corruption robustness (Phase B/D); any
intervention benefit under held-out conditions.

## 7. Phase B outcome (2026-09-24, paired protocol complete)

Frozen protocol executed in full: 84/84 RetinaNet matched-arm runs +
126/126 COCO subset evals + B=2000 paired bootstrap over all 11
blocks. Artifacts: `outputs/analysis/nn_paired_protocol_v1_20260924/`
(remote) and `submission_support_20260911/phase_b_results/` (local).

Established by Phase B:

- YOLO11 fp8−int8 corruption interaction (ΔE) is not universal: sign
  and significance vary across dataset × scale (e.g. voc-n +0.33
  [+0.07,+0.59] vs voc-m −0.57 [−0.80,−0.34]; coco-m +0.60 vs
  coco-x −0.53). Universal "INT8 hurts robustness" claim is refuted
  on this protocol's own evidence.
- RetinaNet INT8-matched collapse magnitude is dataset-dependent:
  −21.2 AP (KITTI) vs −2.9 AP (VOC) vs FP8-matched.
- Selective INT8 (reg-head FP32) is clean-competitive on both
  datasets and preserves corruption robustness on KITTI (≈FP8,
  CIs null); on VOC a small residual gap vs FP8 remains under
  corruption (Gc −0.96 [+0.83,+1.09]; ΔE +0.54 [+0.23,+0.82]) — i.e.
  the regression head is the dominant but not the only contributor.
- Codec-control (JPEG-Q95) clean ≈ original clean on all arms —
  pairing basis validated.

Still open / next decision: whether to proceed to Phase C
(activation-range instrumentation) or reframe the paper as a
recipe-determinants study. The VOC residual gap suggests reg-head
exclusion is not the whole mechanism; a follow-up could localize
which reg-head sub-structure (classification vs box-regression
sensitivity) drives the residual.

## 8. Phase C/D outcome (2026-09-24, corruptcalib intervention complete)

Frozen intervention `nn_corruptcalib_v1_20260924` executed in full:
2 corruptcalib manifests (deterministic in-family assignment,
gaussian_noise+jpeg @ s1/s3/s5; fog+motion_blur held out), 4 engine
builds, 56/56 inference conditions, B=2000 paired bootstrap on 8-arm
RetinaNet blocks (identical resample schedule to Phase B). Artifacts:
`submission_support_20260911/phase_cd_results/`; spec + verdict in
`NN_PHASE_C_CORRUPTCALIB_FREEZE_20260924.md`.

Established by Phase C/D:

- **Calibration-distribution mismatch rejected** as the collapse
  mechanism: `int8-corruptcalib512 − int8-matched512` gains only
  +0.6–1.6 AP absolute with ΔE null on BOTH datasets; the ~19.5 AP
  KITTI residual to FP8 persists.
- Regression-head quantization error confirmed as the operative
  mechanism; selective precision remains the validated fix.
- Corruptcalib adds a small dataset-dependent residual on the
  selective arm (VOC +0.6 significant; KITTI −0.5): VOC residual to
  FP8 shrinks −0.96 → −0.35 but is not fully closed.
- `motion_blur-s5` is a shared severity wall (FP32 ≈ 5.5 AP) — not
  INT8-attributable.

This resolves the open Phase-C/D question: the paper's mechanism story
is head-localized fragility + a ruled-out range-mismatch alternative —
sufficient for methods-level contribution without activation-range
instrumentation. Next: Phase E manuscript rewrite on this evidence
base.
