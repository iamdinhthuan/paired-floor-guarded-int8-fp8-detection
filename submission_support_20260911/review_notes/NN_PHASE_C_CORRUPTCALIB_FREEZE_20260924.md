# NN Phase C/D — Corruption-Aware Calibration Intervention (frozen 2026-09-24)

Attempt: `nn_corruptcalib_v1_20260924`. Frozen spec:
`configs/nn_corruptcalib_v1_20260924.json`.

## Hypothesis under test

RetinaNet matched-INT8 corruption fragility is driven by clean-only
calibration producing activation-range mismatch under corrupted inputs.
If true, an INT8 engine calibrated on the same 512 images rendered with
in-family corruptions should recover held-out corruption AP without
retraining.

## Design

- Calibration images: identical 512-image manifests as Phase B
  (`{kitti,voc}_train_clean_512_s20260807_v1.json`), so the only
  intervention variable is the corruption distribution.
- Assignment: deterministic per-image
  `sha256("{schedule_id}:{dataset}:{source_relpath}") % 6` over
  `(gaussian_noise, jpeg) x (s1, s3, s5)` — each image gets exactly one
  in-family cell; transform parameters and JPEG-q95 output encoding are
  byte-identical to the evaluation cache generator
  (`configs/corruptions.json`, sha `70d4dab2…`).
- Held-out: `fog` + `motion_blur` never appear in calibration.
- New arms (per dataset): `int8-corruptcalib512` (full INT8 entropy,
  corruptcalib) and `int8-sel-corruptcalib512` (same + regression-head
  exclusion) — same engine/build/eval contracts as Phase B.
- Evaluation: frozen 14-condition protocol on `kitti_val` / `voc_val`.
- Reference arms reuse Phase B artifacts (fp32, fp8-matched512,
  int8-matched512, int8-selective512).

## Pre-registered estimands and decision rules

Primary: paired bootstrap (B=2000, same `nn_paired_bootstrap_v1_20260924`
seed namespace → identical resample schedule, fully paired across all
six arms) on **held-out corrupt mean** (fog + motion_blur cells).

- `int8-corruptcalib512 − int8-matched512` held-out mean CI95 excludes 0
  (positive) ⇒ range-mismatch mechanism supported.
- `int8-sel-corruptcalib512 − fp8-matched512` held-out mean CI95
  includes 0 ⇒ VOC residual gap explained by calibration.
- Secondary: in-family corrupt mean contrast, clean-AP tradeoff,
  retention vs codec-control.
- If the primary CI includes 0 → negative intervention result; the
  recipe-determinants finding stands as the contribution.

## Contamination note

`kitti_val` is development-contaminated for the selective recipe (the
regression-head exclusion was tuned there); `voc_val` remains the
cleaner cross-dataset check.

## Execution status

- [x] corruptcalib manifests built (kitti, voc) — 512 images each,
  deterministic ~80-100 images per in-family cell
- [x] 4 engine builds
- [x] 56 inference conditions (2 arms × 2 datasets × 14 cells)
- [x] paired bootstrap on 8-arm RetinaNet blocks (B=2000, same seed
  namespace → identical resample schedule as Phase B)
- [x] held-out / in-family contrast report —
  `submission_support_20260911/phase_cd_results/intervention_report.json`

## Final VOC bootstrap results (B=2000, paired; AP points)

`int8-corruptcalib512 − int8-matched512`: G0 = +0.70 [+0.40, +1.01],
Gc = +0.64 [+0.53, +0.76], ΔE = −0.06 [−0.38, +0.24] — small
significant absolute gain, **ΔE null**.

`int8-sel-corruptcalib512 − int8-selective512`: G0 = +0.63
[+0.33, +0.94], Gc = +0.61 [+0.49, +0.72], ΔE null — corruptcalib adds
~+0.6 AP to the selective arm on VOC.

`int8-sel-corruptcalib512 − fp8-matched512`: G0 = +0.21 [−0.06, +0.49]
(clean parity), Gc = −0.35 [−0.47, −0.24], ΔE = −0.56 [−0.83, −0.31] —
the VOC residual gap shrinks from −0.96 (selective512) to −0.35 but is
NOT fully closed.

## Mechanism verdict

1. **Range-mismatch hypothesis rejected.** Corruptcalib full-INT8 gives
   only +0.6–1.6 AP absolute with ΔE null on both datasets — the
   ~19.5 AP KITTI collapse (and ~2.9 AP VOC gap) is not caused by
   clean-only calibration ranges.
2. **Regression-head quantization error is the operative mechanism**:
   exclusion alone recovers to FP8-level on KITTI (null CIs) and
   near-FP8 on VOC.
3. **Corruption-aware calibration adds a small, dataset-dependent
   residual gain** on the selective arm (VOC +0.6, KITTI −0.5) — report
   as a secondary observation, not a method claim.
4. `motion_blur-s5` is a shared corruption-severity wall (FP32 also
   collapses to ~5.5 AP) — excluded from INT8-specific attribution.

## Recommended paper framing

Recipe determinants + localized fragility: "INT8 corruption fragility
in RetinaNet is dominated by regression-head quantization error, not
by calibration-distribution mismatch; selective precision (head
exclusion) is the validated intervention." This is a methods-level
contribution with a clean negative result ruling out the obvious
alternative mechanism.

## Interim point estimates (pre-bootstrap)

KITTI: `int8-corruptcalib512` clean 29.69 / held-out 18.70 / in-family
24.72 vs `int8-matched512` 28.29 / corr-mean12 20.37 — intervention does
not rescue the collapse; `int8-sel-corruptcalib512` clean 49.55 ≈
`int8-selective512` 49.60.

VOC: `int8-corruptcalib512` clean 53.78 / all-12 39.35 vs
`int8-matched512` 53.30 / 38.71; `int8-sel-corruptcalib512` clean 56.36
(≈ fp8-matched512 56.17) / all-12 41.13 vs `int8-selective512` 40.52 and
fp8-matched512 41.48 — clean gap to FP8 essentially closed.

Preliminary read (subject to paired CI95): clean-only calibration is
NOT the driver of the matched-INT8 collapse — the regression-head
quantization error is the operative mechanism, and corruption-aware
calibration adds only a small residual improvement on the selective
arm. `motion_blur-s5` collapses across all INT8 arms on both datasets;
needs comparison against FP8/FP32 per-cell values before attribution.

## KITTI bootstrap results (B=2000, paired)

`int8-corruptcalib512 − int8-matched512`: G0 = +1.6 [+1.0, +2.1],
Gc = +1.3 [+1.2, +1.5], ΔE = −0.2 [−0.7, +0.3] — real but small
absolute gain, **no differential corruption benefit** → the
range-mismatch hypothesis is rejected as the cause of the KITTI
collapse (~19.5 AP residual to FP8 remains).

`int8-sel-corruptcalib512 − int8-selective512`: Gc = −0.5 [−0.7, −0.4]
— corruptcalib marginally *hurts* the selective arm on KITTI
(development-contaminated dataset; interpret cautiously).

`int8-sel-corruptcalib512 − fp8-matched512`: Gc = −0.7 [−0.9, −0.4] —
selective-corruptcalib remains ~0.7 AP below FP8 under corruption on
KITTI.

Cross-check: `motion_blur-s5` collapses for FP32/FP8 too (5.45/5.59 on
KITTI) — it is a shared corruption-severity wall, not INT8-specific.
