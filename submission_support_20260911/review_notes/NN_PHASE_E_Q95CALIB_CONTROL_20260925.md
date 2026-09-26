# Phase E control freeze — q95-calib codec-control arms (2026-09-25)

## Motivation (reviewer-identified confound)

The Phase C/D corrupt-calibration manifests re-encode every calibration image
at the frozen `output_encoding` (JPEG q95, subsampling 0), while the Phase-B
matched calibration manifests point at *original* bytes (KITTI PNG sources,
VOC original JPEGs). Any gain of `int8-corruptcalib512` over
`int8-matched512` is therefore confounded: it could reflect codec adaptation
to the evaluation stream (all eval inputs are q95-encoded) rather than
corruption coverage of the activation ranges.

## Frozen design

- Same frozen 512-image calibration lists as Phase B/C-D
  (`{kitti,voc}_train_clean_512_s20260807_v1.json`).
- New manifests `{kitti,voc}_train_q95calib_512_nn-q95calib-v1.json`:
  each clean calibration image re-encoded at JPEG q95/subsampling 0 with no
  corruption (`src/build_q95_calibration.py`).
- New arms: `int8-q95calib512` and `int8-sel-q95calib512` (selective adds
  the usual `--exclude-node-regex ^/head/regression_head/`).
- Identical quantization mode (int8-entropy), preprocessing
  (`retinanet_normalized`), TensorRT build policy, and 14-condition
  evaluation protocol as the corruptcalib arms.
- Attempt dir: `nn_q95calib_v1_20260924`; runs land under the same
  `{dataset}_val__retinanet_r50_fpn_v2__{arm}__{corruption}-s{sev}` naming.
- Bootstrap: 10-arm RetinaNet blocks, same seed namespace
  (`nn_paired_bootstrap_v1_20260924`) => identical image-resample schedule,
  fully paired with Phase B/C-D; out-dir
  `outputs/analysis/nn_q95calib_v1_20260924/bootstrap`.

## Results (complete; bootstrap CIs final 2026-09-26)

All 56 inference conditions and paired bootstrap (B=2000, anchor-verified)
complete for both datasets.

| arm | KITTI clean / corr12 | VOC clean / corr12 |
|---|---:|---:|
| int8-matched512 | 28.29 / 20.37 | 53.30 / 38.71 |
| int8-q95calib512 | 28.26 / 20.42 | 53.33 / 38.54 |
| int8-corruptcalib512 | 29.69 / 21.71 | 53.78 / 39.35 |
| int8-selective512 | 49.60 / 36.58 | 55.87 / 40.52 |
| int8-sel-q95calib512 | 50.28 / 36.94 | 55.81 / 40.40 |
| int8-sel-corruptcalib512 | 49.55 / 36.08 | 56.36 / 41.13 |

**Reading (corrected 2026-09-25 after audit; VOC CIs added 2026-09-26):** on
the full-INT8 tier, codec-only calibration leaves clean AP unchanged on both
datasets (J95: KITTI -0.34 [-0.93,+0.35], VOC +0.20 [-0.07,+0.48]) and does
not improve the corrupt mean (KITTI +0.04 [-0.10,+0.22]; VOC -0.17
[-0.26,-0.09], slightly negative). On the selective tier it is NOT uniformly
negligible: KITTI corrupt mean +0.36 [+0.23,+0.49] (a genuine small uniform
shift, Delta E held-out +0.05 [-0.48,+0.57]) while VOC corrupt mean is -0.12
[-0.21,-0.04]. Codec adaptation is therefore real but small and
sign-variable across dataset x tier cells; it explains neither the collapse
nor the intervention gains.

**Key interpretive result:** corrupt calibration also raises J95 clean AP
(KITTI +1.56, VOC +0.73), so the held-out Delta E is null-to-negative on both
datasets (KITTI -0.50 [-1.01,+0.03]; VOC -0.71 [-1.02,-0.39]). The KITTI
held-out gain that met the frozen criterion is a uniform level shift, not
corruption-specific transfer. All numbers: `submission_support_20260911/nn_final_stats.json`
(plug-in points from `cell_points.json`, intervals from paired draws).

## Decision usage

- `int8-corruptcalib512 - int8-q95calib512` isolates the
  corruption-coverage component of the intervention gain.
- `int8-q95calib512 - int8-matched512` measures pure codec adaptation.
- The primary negative verdict on the range-mismatch mechanism is
  *conservative* under this confound (codec advantage inflates
  corruptcalib gains); the control mainly disciplines the positive
  residual claims (e.g. the VOC selective-arm +0.6 AP gain).
