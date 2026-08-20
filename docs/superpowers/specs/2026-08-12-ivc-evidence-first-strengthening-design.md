# IVC Evidence-First Strengthening Design

**Date:** 2026-08-12  
**Target:** *Image and Vision Computing* (Q1)  
**Approved direction:** targeted evidence strengthening without retraining

## 1. Objective

Strengthen the manuscript from a technically valid exploratory study into a reviewer-ready IVC submission by adding one direct paired INT8--FP8 analysis, a reproducible RTX 5090 deployment benchmark, exact engine-treatment disclosure, a recent and claim-matched literature review, and a cleaner submission presentation.

The revision must not claim universal format superiority, practical equivalence, causal effects of checkpoint capacity, or robustness improvement from a narrowing accuracy gap.

## 2. Fixed scope

- Reuse the frozen YOLO11n/m/x checkpoints, TensorRT engines, input manifests, and prediction payloads already produced for COCO, VOC, KITTI, and TT100K.
- Do not retrain or rebuild checkpoints.
- Do not change corruption bytes, calibration lists, evaluator definitions, dataset splits, size definitions, or the primary JPEG-quality-95 matched-clean control.
- Run only new analysis over existing prediction payloads and new latency/throughput measurements of the existing engines on `thuan@100.111.139.103` in conda environment `qtsd`.
- Treat every new inferential result as a targeted post-hoc extension, not a confirmatory trial.
- Retain the existing fail-closed SHA-256 evidence boundary.

## 3. Scientific strengthening

### 3.1 Direct paired format contrasts

For each dataset, checkpoint rung, corruption, and severity, resample the same image identifiers once and apply that sample to both INT8 and FP8 prediction arms. Define

\[
\Delta E = E_{\mathrm{INT8}}-E_{\mathrm{FP8}}
= (AP_{\mathrm{FP8},c}-AP_{\mathrm{INT8},c})
 -(AP_{\mathrm{FP8},95}-AP_{\mathrm{INT8},95}).
\]

Positive `Delta E` means corruption increases the INT8 penalty more than the FP8 penalty. The FP32 terms cancel algebraically, but all arm identities and input hashes must still be validated before calculation.

Also compute the clean-format contrast

\[
\Delta Q = Q_{\mathrm{INT8}}-Q_{\mathrm{FP8}}
=AP_{\mathrm{FP8},95}-AP_{\mathrm{INT8},95},
\]

and size-interaction contrasts `Delta Psi = Psi_INT8 - Psi_FP8`, keeping COCO/VOC/KITTI area endpoints separate from TT100K height endpoints.

Use 2,000 paired bootstrap replicates. Within a dataset and replicate, use one shared image resample across checkpoint rungs, conditions, and formats. This preserves the dependence induced by evaluating the same images. Compute:

- 144 cell-level `Delta E` intervals;
- 12 dataset--checkpoint clean `Delta Q` intervals;
- balanced macro `Delta E` over the fixed four-dataset grid;
- balanced area `Delta Psi` over COCO/VOC/KITTI;
- a separate TT100K height `Delta Psi`;
- sign inventories for exploratory cell intervals.

Do not declare equivalence from an interval containing zero. Acceptable wording is “the targeted paired contrast did/did not resolve a directional difference under this fixed grid.”

### 3.2 Deployment benchmark

Benchmark all 36 existing dataset--checkpoint--precision engines on the same RTX 5090 host. Use batch size 1 and the input geometry bound to each engine. For each engine:

- run three independent timed repetitions;
- use a five-second warm-up and a 30-second timed window per repetition;
- record TensorRT and CUDA versions, GPU model, input tensor and shape, precision arm, engine SHA-256, serialized engine bytes, median latency, latency percentiles exposed by TensorRT, and throughput;
- record command, exit status, start/end timestamps, and raw log SHA-256;
- run engines serially and reject measurements if another compute process is active.

Report latency, throughput, and engine size as secondary deployment measurements. Do not infer power or energy because those outcomes are not measured.

### 3.3 Engine-treatment disclosure

Add one compact table that identifies, for FP32-typed, INT8, and FP8 arms:

- TensorRT, CUDA, ModelOpt, ONNX, and Ultralytics versions;
- quantization mode and calibration method;
- calibration-list size and hash policy;
- quantized/excluded operator classes, including the recorded Sigmoid exclusion;
- Q/DQ or engine-build path;
- input geometry and preprocessing;
- decoding/NMS path;
- checkpoint, ONNX, and engine registry bindings.

If the engine registries do not expose a requested treatment field, report it as unrecorded rather than infer it.

## 4. Literature and positioning

### 4.1 Date constraint

Every bibliography entry and every cited work must have a verified publication or software-release year from 2023 through 2026. Pre-2023 entries will be removed; their years will never be altered.

Because no truthful 2023--2026 paper can replace the original COCO, VOC, KITTI, or TT100K provenance papers, the manuscript will describe exact evaluated splits from the frozen manifests without claiming recent papers introduced those datasets. The bootstrap algorithm will be specified self-containedly without attaching an unrelated recent citation.

### 4.2 Required IVC conversation

Use exactly five claim-matched *Image and Vision Computing* papers:

1. Kolf et al., SyPer, 2023, DOI `10.1016/j.imavis.2023.104692`--quantized lightweight recognition outside detection.
2. Luo et al., LaLM, 2024, DOI `10.1016/j.imavis.2024.105035`--adverse-weather detector degradation and mitigation.
3. Zheng et al., small-object review, 2024, DOI `10.1016/j.imavis.2024.105054`--small-object structural difficulty.
4. Su et al., YOLIC, 2024, DOI `10.1016/j.imavis.2024.105095`--edge-vision efficiency and the need for deployment measurements.
5. Zhang et al., Dynamic Dual Teaching, 2025, DOI `10.1016/j.imavis.2025.105740`--adaptive detection under domain shift.

Each citation must be attached only to the scoped claim above. None may be cited as evidence for FP8 detector robustness.

### 4.3 Novelty statement

State explicitly that Karimov et al. already observed degradation- and model-dependent quantized-detector behavior. The present novelty is the controlled measurement protocol: codec-matched difference-in-differences, executable FP8, four datasets, size-conditioned interaction, direct paired format contrasts, and hash-bound evidence.

## 5. Manuscript and visual redesign

- Start Results with scientific outcomes; move detailed artifact ledgers to the supplement.
- Retain one concise integrity statement in Methods and one validation sentence in Results.
- Regenerate every retained figure at its final physical width with text no smaller than 7 pt.
- Split dense distribution/forest panels where necessary and remove raw dataframe labels such as `psi`.
- Standardize notation to `E`, `Delta E`, `Psi`, and `Delta Psi`.
- Split area and TT100K-height summaries when a combined table impairs readability.
- Keep the main manuscript focused; place full generated tables and hash inventories in a separately compiled supplement.
- Rewrite the abstract and highlights in plain scientific language and replace unsupported “similar” or deployment-choice wording with the direct-contrast result.
- Restrict the quantified accuracy-floor statement to `E` unless a separately implemented `Psi` floor diagnostic is added.

## 6. Submission packaging

Create and verify:

- a main reviewer PDF;
- a separate supplement PDF;
- a flat Editorial Manager LaTeX archive with no subdirectories;
- a reproducibility archive containing code, tests, configs, manifests, compact metrics/bootstrap evidence, benchmark logs, generated tables, and checksums;
- `CITATION.cff` and archive metadata suitable for a later Zenodo/GitHub release.

The agent will not fabricate or publish a DOI. Public repository creation, author consent, ORCIDs, funding, acknowledgements, and final CRediT confirmation remain author-controlled submission gates.

## 7. Fail-closed verification

Before the revised PDF is accepted locally:

- a citation audit must show zero pre-2023 entries, exactly five cited IVC papers, no orphan keys, and verified DOI metadata;
- synthetic tests must reproduce the signs and algebra of `Delta Q`, `Delta E`, and `Delta Psi`;
- the remote analysis must bind every raw prediction/input artifact and contain the exact 12/144 endpoint grids at 2,000 replicates;
- the benchmark must contain 36 engines times three successful repetitions and matching engine/log hashes;
- all existing and new tests must pass;
- the artifact builder and submission audit must pass;
- both PDFs and the flat source archive must compile in clean extracted directories;
- LaTeX logs must have no fatal errors, undefined citations/references, or overfull boxes;
- every PDF page must be rendered and visually inspected.

## 8. Success criterion

The revision is ready for author-side submission preparation when no independent reviewer reports a Critical or Major scientific/presentation defect, all local gates above pass, and the only remaining blockers are external author confirmation and publication of the prepared archive.
