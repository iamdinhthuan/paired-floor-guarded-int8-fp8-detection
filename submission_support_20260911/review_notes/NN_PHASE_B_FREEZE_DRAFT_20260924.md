# Phase B freeze draft — paired-protocol experiment for the NN upgrade

Date: 2026-09-24. Status: FROZEN AND EXECUTING — scope decisions
approved by user 2026-09-24 (COCO = frozen paired subset of 2,000;
RetinaNet primary on KITTI+VOC; RT-DETR excluded from primary).
Config `configs/nn_paired_protocol_v1_20260924.json`; queue
`src/run_nn_paired_protocol_queue.sh` launched on the RTX 5090 host
(attempt `nn_paired_protocol_v1_20260924`).

Important correction found during implementation: the `*_val`
universes already carry complete paired coverage for YOLO11 n/m/x
(3 precisions x 13 conditions + codec-control) and legacy-calibration
RetinaNet/RT-DETR; the confirmatory_final universes have no
codec-control manifests. Phase B therefore uses the val universes
(kitti_val 1,496; voc_val 4,952; COCO subset 2,000-of-5,000) so all
families share codec-matched clean controls. The confirmatory_final
blocks remain the historical original-clean replication. Caution:
kitti_val is development-contaminated for the selective recipe
(Phase A selected the reg-head exclusion there); voc_val is the
cleaner cross-dataset check for that arm.

New compute actually required (much less than first estimate):
6 engine builds (kitti/voc x fp8-m512/int8-m512/int8-sel512) +
84 RetinaNet matched-arm runs (2 datasets x 3 arms x 14 conditions).
COCO subset evaluation is a deterministic restriction of existing
5,000-image predictions — zero new inference.
Parent: NN_UPGRADE_GAP_MAP_20260923.md (Phase A complete).

## 1. What Phase B must establish

Per the gap map, the upgraded manuscript needs a paired-protocol
experiment that (a) replicates the INT8/FP8 corruption-robustness
contrast under corrected recipes, (b) adds a clean-competitive
non-YOLO comparator, and (c) reports absolute corrupted AP, per-arm
retention, and grouped uncertainty — the properties reviewers asked
for. Phase C (mechanism) and Phase D (intervention) build on this
frozen protocol; they are out of scope here.

## 2. Comparator set decision (informed by Phase A)

| Candidate | Evidence | Decision |
|---|---|---|
| YOLO11n/m/x | Primary family; matched-contract calibration already correct (yolo_letterbox is native) | INCLUDE — primary |
| RetinaNet-R50-FPN-v2 | Phase A: full INT8 still fails clean (28.18); INT8-except-reg-head is clean-competitive (49.50 vs FP32 50.44) | INCLUDE — 4 arms |
| RT-DETR-L | INT8 collapses ~1 AP; recipe undiagnosed | EXCLUDE from primary; optional diagnostic track after its own decomposition |

RetinaNet arms (per dataset): FP32 reference, FP8, INT8-matched-full,
INT8-selective (`exclude_node_regex=^/head/regression_head/`). The
selective arm is the recipe-level intervention evidence; reporting it
alongside full INT8 is what converts the Phase A finding into a
cross-architecture claim.

## 3. Blocks and data status (remote RTX 5090 host)

| Block | Images | Corruption assets | Engine status |
|---|---|---|---|
| COCO × YOLO11n/m/x | frozen paired subset ~2,000 of 5,000 val2017 (verified 2026-09-23) | coco_c 4 corruptions × s1/s3/s5 + codec_control Q95 manifests exist | fp32/fp16/fp8/int8 engine manifests exist for n/m/x; inference coverage partial (yolo11m: clean+fog+gn only) |
| KITTI holdout × YOLO11n/m/x | 1,197 | full 12-cell suite + Q95 control | retained engines from confirmatory_v1 |
| KITTI × RetinaNet | 1,496 val (dev partition) | same suite | FP32/FP8/INT8-legacy engines exist; INT8-matched + selective need new builds (recipe_v6 artifacts reusable for KITTI) |
| VOC holdout × YOLO11n/m/x | 5,823 | full suite + Q95 | retained engines exist |
| VOC × RetinaNet | 5,823 | full suite + Q95 | fp32/fp8/int8-legacy engine manifests exist; matched + selective builds needed |
| TT100K × RetinaNet | 3,067 | full suite + Q95 | diagnostic only, not pooled |

Conditions per block: JPEG-Q95 codec clean control + 4 corruptions
(gaussian_noise, fog, motion_blur, jpeg) × severities 1/3/5 = 13
evaluation conditions per arm. Original-source clean reported where
historical lineage permits.

## 4. Arms summary and new compute required

- YOLO11 blocks: reuse retained INT8-entropy + FP8 engines and any
  hash-verified historical predictions; add FP32 reference arm where
  missing; run missing conditions (COCO: most of n/x, jpeg +
  motion_blur everywhere, codec-control clean).
- RetinaNet blocks: build INT8-matched-full and INT8-selective engines
  per dataset (quantized ONNX differs only by calibration manifest;
  exclusion regex identical). Then 4 arms × 13 conditions × images.
- Estimated scale: COCO 3 models × ~10 missing conditions × 5,000 img
  ≈ 150k inferences; RetinaNet 2 datasets × 4 arms × 13 cond ≈ 96k
  inferences (KITTI) — plus engine builds (~6–10 builds, 5–15 min
  each on the 5090).

## 5. Estimands (frozen before any new AP is inspected)

Per block, all in AP points, all paired on identical ordered image
IDs:
- Absolute clean AP and absolute per-condition corrupted AP per arm.
- Retention per arm: corrupted mean / clean.
- Clean gap G0 = A_FP8 − A_INT8 (and A_FP32 references); corrupted gap
  Gc; interaction ΔE = Gc − G0.
- Recipe contrast: INT8-matched-full vs INT8-selective on clean and
  corrupted — the cross-architecture recipe-travel test.
- Paired 2,000-draw image-bootstrap percentile intervals; block means
  formed within draws; grouped by dataset (scene dependence).
- Macro summaries only within a family (YOLO11 n/m/x) and only across
  same-protocol blocks; RetinaNet reported per-dataset, not pooled
  with YOLO.

Kill criteria (carried from gap map): if corrected recipes erase the
effect, report the recipe-confound study; do not build a method claim.

## 6. Controls required by the gap map

- Report clean cost of the selective-precision arm (49.50 vs 50.44 on
  KITTI; measure on VOC/COCO before any corrupted claim).
- Retention reported per arm, never pooled across arms.
- No latency/throughput claims (shared GPU; latency inadmissible).
- Single build + single calibration subset per arm → all results are
  single-realization; label accordingly.
- RT-DETR exclusion is a scope bound, stated explicitly.

## 7. Execution mechanics (unchanged project rules)

- Frozen config JSON per attempt under `configs/`; hash-bound inputs;
  write-once outputs; `attempt.lock`; serial GPU execution
  (`gpu_workers=1`, TF32 off, TensorRT 11.1.0.106).
- Runner: extend `src/run_nn_preprocessing_pilot.py` pattern or reuse
  `cross_family_infer_trt.py` + `coco_eval.py` per condition with a
  queue script; execution on `thuan@100.111.139.103`, conda `qtsd`.
- Outputs isolated under `outputs/nn_paired_protocol_v1_*`.

## 8. Scope decisions (user, 2026-09-24)

1. COCO: frozen paired subset of ~2,000 images drawn once from the
   verified 5,000 val2017 universe, seed recorded in the subset
   manifest; all COCO arms share the identical subset.
2. RetinaNet is primary on both KITTI and VOC (4 arms each).
3. RT-DETR is excluded from primary; its INT8 collapse remains a
   stated scope bound. A separate recipe decomposition for RT-DETR may
   be considered later, outside this freeze.

## 9. Execution status (COMPLETE, 2026-09-24)

Queue `nn_paired_protocol_v1_20260924` finished on remote RTX 5090:

- 6/6 engine builds; 84/84 RetinaNet runs (preds + inputs + run records
  + metrics); 126/126 COCO subset evals; zero REFUSED/errors in queue
  or subset logs.
- COCO subset: 2,000 image ids, seed 20260924, selection SHA-256
  `1a966e4c…`; evals reuse 5,000-image predictions (no new GPU
  inference).
- Paired bootstrap B=2000 complete for all 11 blocks
  (`outputs/analysis/nn_paired_protocol_v1_20260924/bootstrap/`), using
  shared image-resample schedules per block; point AP reproduces
  recorded metrics exactly. Per-cell worker parallelism used after a
  memory-safety rework (cell-parallel, not draw-parallel).
- Point-estimate + CI summary: `analysis/nn_paired_protocol_summary.py`
  → `outputs/analysis/nn_paired_protocol_v1_20260924/summary.{json,md}`;
  local copies under
  `submission_support_20260911/phase_b_results/`.

Key results (all values AP points; corrupt-mean = equal weight over 12
cells; CI95 = paired image bootstrap, B=2000):

RetinaNet recipe decomposition:

| block | arm | clean | corrupt-mean |
|---|---|---:|---:|
| kitti | fp32 / fp8-m / int8-m / int8-sel | 50.15 / 49.48 / 28.29 / 49.60 | 37.80 / 36.75 / 20.37 / 36.58 |
| voc | fp32 / fp8-m / int8-m / int8-sel | 57.23 / 56.17 / 53.30 / 55.87 | 42.32 / 41.48 / 38.71 / 40.52 |

- INT8-matched collapse is dataset-dependent: −21.2 AP (KITTI) vs
  −2.9 AP (VOC) vs fp8-matched.
- Selective INT8 ≈ FP8-matched on KITTI (Gc diff [−0.07,+0.40] null);
  on VOC a small residual gap remains (clean −0.42 [+0.13,+0.74];
  corrupt −0.96 [+0.83,+1.09]; ΔE +0.54 [+0.23,+0.82]).
- fp8−int8-matched ΔE: KITTI −4.69 [−5.49,−3.86] (gap shrinks under
  corruption — floor effect, since INT8 clean is already collapsed);
  VOC −0.26 [−0.58,+0.06] (no differential damage).

YOLO11 fp8−int8 ΔE (q95 basis): sign flips across dataset × scale —
kitti: n +0.04 null / m −0.68 [−1.32,−0.05] / x −0.69 [−1.13,−0.33];
voc: n +0.33 [+0.07,+0.59] / m −0.57 [−0.80,−0.34] / x +0.15 null;
coco-subset: n −0.32 borderline / m +0.60 [+0.27,+0.97] / x −0.53
[−0.93,−0.18]. No universal direction — supports the recipe-determinants
framing and rules out "INT8 is uniformly worse under corruption".

Interim KITTI RetinaNet point estimates (corrupt-mean = equal weight
over 12 cells; retention vs Q95 clean):

| arm | clean-orig | clean-Q95 | corrupt-mean | retention |
|---|---:|---:|---:|---:|
| fp32 | 50.1456 | — | 37.8041 | — |
| fp8-legacy | 49.1854 | — | 36.4790 | — |
| int8-legacy | 7.2143 | — | 7.4316 | — |
| fp8-matched512 | 49.4785 | 49.2521 | 36.7488 | 0.7461 |
| int8-matched512 | 28.2930 | 28.1827 | 20.3712 | 0.7228 |
| int8-selective512 | 49.6001 | 49.5304 | 36.5815 | 0.7386 |

Reading (diagnostic, KITTI is development-contaminated for the
selective recipe): matched INT8 keeps the Phase-A clean collapse
(28.29) but its corrupted mean (20.37) is proportionally less damaged;
selective INT8's corrupted mean (36.58) ≈ FP8 matched (36.75) and FP32
(37.80), i.e. corruption robustness is preserved when the regression
head stays FP32. fp8−int8 ΔE = −4.69 on this block.
