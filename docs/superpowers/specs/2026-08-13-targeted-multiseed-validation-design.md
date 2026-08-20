# Targeted Multi-Seed Validation Design

## Purpose

Add a prospective seed-sensitivity stage that directly addresses the paper's
single-training-seed and single-calibration-list limitation without repeating the
entire discovery grid. The stage estimates how much the high-severity direct
INT8--FP8 corruption contrast changes across independently varied training and
calibration samples.

## Frozen scope

- Datasets: VOC, KITTI, and TT100K transfer datasets.
- Capacity: YOLO11m as the predeclared representative capacity.
- Training seeds: 20260807 (existing), 20260813, and 20260814.
- Calibration seeds: 20260807, 20260813, and 20260814.
- Calibration: 512 train-only images sampled uniformly without replacement,
  entropy calibration, CPU calibration execution, Sigmoid exclusion.
- Formats: the same mixed-graph ModelOpt INT8 and FP8 treatments as the direct
  analysis.
- Inputs: deterministic JPEG-95 matched-clean plus Gaussian noise, motion blur,
  fog, and JPEG corruption, all at severity 5.
- Evaluation: the same annotations, ordered images, materialized bytes,
  preprocessing, decoder, confidence threshold, NMS, detection cap, and
  COCO-style evaluator as the direct analysis.

The resulting factorial grid has 3 datasets x 3 training seeds x 3 calibration
seeds x 2 formats x 5 inputs = 270 inference/metric records. It yields 216
format-specific corrupted arms and 108 direct \(\Delta E\) cells after the clean
arm is reused within each dataset/training/calibration/format block.

## Training comparability

New runs use the existing 100-epoch augmentation and optimization profile but
freeze the effective baseline batch sizes instead of rerunning automatic batch
selection: 16 for VOC, 16 for KITTI, and 4 for TT100K. These values are derived
from the archived baseline's train-image and batch-index counts. All other
training controls remain unchanged: pretrained initialization, deterministic
mode, automatic optimizer, AMP, eight workers, patience zero, mosaic closed for
the final ten epochs, and the recorded augmentation recipe.

The original seed's existing checkpoint is never rewritten. Each new run has an
immutable run ID, separate output directory, complete training registry, and
hash-bound best/last checkpoints and args file.

## Estimands and reporting

For every dataset, training seed, calibration seed, and corruption,
\(D_p=AP_{p,95}-AP_{p,c}\) and
\(\Delta E=D_{INT8}-D_{FP8}\). The primary descriptive endpoint is the equally
weighted mean over the 108 direct cells. The report also gives:

- training-seed marginal means and ranges;
- calibration-seed marginal means and ranges;
- dataset and corruption marginal means;
- the 108-cell sign inventory and native-AP distribution;
- a balanced two-way decomposition into training-seed, calibration-seed, and
  training-by-calibration residual components within each dataset/corruption
  block.

With only three levels per seed factor, variance components are descriptive and
do not justify universal superiority, equivalence, or architecture-wide claims.
The stage is a targeted high-severity validation, not a replacement for the full
four-dataset discovery analysis.

## Provenance and failure behavior

A frozen self-hashed configuration enumerates every training profile, calibration
manifest, checkpoint registry, ONNX registry, engine registry, input manifest,
inference record, metric record, and expected condition. Every stage is
append-only and resume-verified. Existing partial triples or records are rejected
unless all bound hashes, semantics, and counts validate.

Before GPU work, the runner requires at least 20 GiB free project-filesystem
space, the RTX 5090 identity, no unapproved compute process, the `qtsd`
environment, complete source/acquisition manifests, and absence of conflicting
attempt outputs. It logs the exact command, source hashes, GPU snapshot, disk
snapshot, stage transition, exit code, and output hashes. A child failure stops
new work and preserves completed evidence.

## Operational stages

1. Validate and freeze configuration/source package.
2. Run six new serial training jobs (two seeds x three datasets).
3. Freeze six new calibration manifests and bind the three existing manifests.
4. Export nine FP32 ONNX checkpoints.
5. Quantize and build 54 engines.
6. Run and evaluate 270 conditions serially.
7. Build the 108-cell seed-sensitivity report and completion manifest.
8. Only after local validation, add a compact supplementary/main-paper result.

## Exclusions

COCO is excluded because its official checkpoint has no study-controlled training
seed. YOLO11n/x, severities 1/3, additional corruption families, engine-build
replicates, and new data splits are excluded. No result from this stage may be
inserted into the manuscript before its exact completion report validates.
