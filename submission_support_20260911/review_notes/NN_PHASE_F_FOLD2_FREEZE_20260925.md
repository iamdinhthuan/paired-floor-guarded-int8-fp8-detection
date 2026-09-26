# Phase F freeze: complementary held-out fold (2026-09-25)

Frozen before any fold-2 engine build or inference run. Config:
`configs/nn_corruptcalib_fold2_v1_20260925.json`.

## Why

Fold 1 (`nn_corruptcalib_v1_20260924`) calibrated on gaussian_noise+jpeg and
held out fog+motion_blur. Its frozen criterion was met on KITTI (+1.07) and not
on VOC. An interpretive analysis (held-out Delta E = held-out gain minus J95
clean gain) then showed no corruption-specific transfer on either dataset.
Reviewers called the single direction the weakest link and flagged the
interpretive analysis as post hoc. Fold 2 swaps the roles and pre-specifies
both rules.

## Design

- Same 512 calibration images per dataset; schedule `nn-corruptcalib-fold2-v1`
  assigns each image to one of fog/motion_blur x s1/s3/s5 (KITTI 75-95 per
  cell, VOC 81-89 per cell); JPEG-95 terminal encoding as in fold 1.
- Arms: `int8-cc2calib512` (full INT8), `int8-sel-cc2calib512` (regression
  head excluded), KITTI and VOC, frozen 14-condition protocol.
- Bootstrap: seed namespace `nn_paired_bootstrap_v1_20260924`, B=2000, with
  `int8-matched512` as a pairing anchor so contrasts against arms in other
  caches are paired (verified by identical anchor draws).

## Decision rules (pre-specified)

1. Primary (replicates fold-1 frozen rule): `int8-cc2calib512 - int8-matched512`
   on the fold-2 held-out mean (gaussian_noise+jpeg); CI95 excluding 0 =>
   rule met.
2. Corruption-specific: Delta E held-out for `int8-cc2calib512` minus
   `int8-matched512` and minus `int8-q95calib512`; CI95 excluding 0 on the
   positive side => corruption-specific transfer supported.
3. Both rules reported for both datasets, whichever way they point.

## Outcome

(filled 2026-09-26 from `submission_support_20260911/nn_final_stats.json`,
all values AP points, plug-in point + 95% paired-bootstrap CI, B=2000)

KITTI (`int8-cc2calib512 - int8-matched512`): J95 clean -2.71 [-3.38,-1.93];
in-family (fog+motion_blur) -1.35 [-1.56,-1.04]; held-out
(gaussian_noise+jpeg) -2.02 [-2.31,-1.73]; Delta E held-out +0.69
[-0.10,+1.37] (vs q95calib anchor: held-out -2.00 [-2.32,-1.69],
Delta E +0.37 [-0.28,+0.90]). Rule 1: NOT met (held-out significantly
negative). Rule 2: NOT met (Delta E null).

VOC (`int8-cc2calib512 - int8-matched512`): J95 clean +1.04 [+0.72,+1.34];
in-family +0.86 [+0.72,+0.99]; held-out +1.66 [+1.52,+1.83]; Delta E
held-out +0.62 [+0.30,+0.96] (vs q95calib anchor: held-out +1.89
[+1.75,+2.06], Delta E +1.06 [+0.76,+1.38]). Rule 1: MET. Rule 2: MET
(positive side), also significant against the codec anchor.

Selective fold-2 (`int8-sel-cc2calib512 - int8-selective512`): KITTI
J95 +1.01 [+0.37,+1.72], held-out +0.66 [+0.33,+0.92], Delta E -0.35
[-1.09,+0.28]; VOC J95 +0.63 [+0.34,+0.94], held-out +1.50
[+1.35,+1.66], Delta E +0.86 [+0.55,+1.18]. Against fp8-matched512
(fold-1 criterion-2 analog): VOC held-out +0.08 [-0.06,+0.23] (includes
zero); KITTI held-out +1.04 [+0.64,+1.40].

Interpretation recorded in the manuscript: the intervention's held-out
effect is sign-unstable across the four fold-dataset cells; on KITTI,
where the collapse lives, neither fold produces a corruption-specific
held-out gain (fold 1's raw held-out gain +1.07 is met in level terms but
not corruption-specific under the post-hoc Delta-E decomposition) and
fold 2 degrades every level basis including its calibrated families.

