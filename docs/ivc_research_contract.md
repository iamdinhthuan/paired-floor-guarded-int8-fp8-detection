# Topic C IVC research contract

Status: working scientific contract for owner review. This document strengthens the experiment design; it is not a results report and does not authorize a full grid by itself.

## 1. Paper thesis

The paper should answer one narrow, defensible question:

> Under a controlled TensorRT inference path, does input corruption amplify the accuracy loss caused by post-training quantization disproportionately for small objects, and does FP8 preserve the FP32 corruption profile better than INT8?

The paper is not strongest as “many AP tables.” Its contribution is a controlled interaction analysis in which corrupted bytes, image identities, model checkpoint, geometry, decoder, post-processing, calibration provenance, and bootstrap resamples are held fixed wherever the scientific contrast requires them to be fixed.

The result can be publishable in either direction:

- Positive interaction: INT8 widens the FP32 gap under corruption, especially for small objects.
- Additive/equivalent behavior: after eliminating pipeline confounds, quantization and corruption do not materially amplify each other within a prespecified margin.
- Heterogeneous behavior: amplification is restricted to particular corruption families, severities, capacities, or domains.

The manuscript must never assume the first scenario before the evidence is complete.

## 2. Why this can fit Image and Vision Computing

IVC explicitly welcomes image interpretation and object-recognition research, including applied work on real scenes, and encourages quantitative comparison and performance evaluation. Topic C fits that scope if it produces a deeper, reproducible understanding of detector failure rather than a leaderboard.

Official references:

- [Image and Vision Computing — journal page](https://www.sciencedirect.com/journal/image-and-vision-computing)
- [Image and Vision Computing — Guide for Authors](https://www.sciencedirect.com/journal/image-and-vision-computing/publish/guide-for-authors)

The submission should therefore emphasize:

1. a new interaction-based evaluation methodology;
2. controlled, auditable comparisons;
3. generalization across object sizes, model capacities, and domains;
4. actionable implications for deployment and calibration;
5. released code, protocols, manifests, and derived research data.

## 3. Contribution contract

The abstract and introduction may claim the following only after the associated evidence gate passes.

### C1 — Interaction formulation

Define quantization damage relative to FP32, then difference it again between clean and corrupt inputs. This separates ordinary corruption damage from corruption-induced widening of the quantization gap.

For dataset (d), model (m), quantized format (q), size stratum (s), and corruption condition (c):

\[
Q_{d,m,q,s,c}=AP_{d,m,FP32,s,c}-AP_{d,m,q,s,c},
\]

\[
E_{d,m,q,s,c}=Q_{d,m,q,s,c}-Q_{d,m,q,s,0},
\]

\[
\Psi_{d,m,q,c}=E_{d,m,q,small,c}-E_{d,m,q,large,c}.
\]

Interpretation:

- (E>0): corruption widens the FP32–quantized gap.
- (E\approx0): damage is approximately additive within uncertainty.
- (E<0): the clean quantization gap narrows under corruption; check floor effects before interpreting this as robustness.
- \(\Psi>0\): the excess gap is larger for small than large objects.

### C2 — Matched-input protocol

Every format in a linked contrast must consume the exact same ordered image IDs and the same image bytes. The contribution is not valid if each model generates its own corruption or if a corruption-labelled run silently reads clean images.

Required evidence:

- immutable clean/corrupt manifests;
- source and output byte SHA-256;
- deterministic seed and generator version;
- image-count and uniqueness checks;
- proof that corrupt pixels differ from clean pixels;
- proof that all formats cite the same manifest hash.

### C3 — Capacity and format comparison

YOLO11n/m/x provide a controlled capacity axis inside one architecture family. FP32 is the reference, FP16 is a pipeline-parity gate, and INT8-entropy/FP8 are the primary quantized formats.

Allowed scope:

- “Across three capacities of the YOLO11 family…”
- “Within the evaluated TensorRT pipeline…”

Forbidden scope unless another family passes the same parity gates:

- “Object detectors generally…”
- “All CNN detectors…”
- “Quantization universally…”

### C4 — Domain replication

COCO is discovery/primary development evidence. VOC and KITTI provide confirmatory replication with COCO-style area bins. TT100K is a small-object-heavy transfer endpoint with original-pixel height bins.

Do not compare absolute AP values across datasets as if they were exchangeable: their class ontologies, splits, object distributions, image resolutions, and size-bin meanings differ. Cross-dataset evidence should emphasize effect direction, interval overlap, heterogeneity, and replication of the interaction pattern.

### C5 — Calibration intervention

Corruption-aware calibration becomes a genuine methodological contribution only when it is a frozen, train-only intervention with clean, balanced mixed, and severity-matched arms.

If this phase is not completed, remove RQ4 and every mitigation claim from the title, abstract, highlights, and conclusion. Do not leave a planned experiment presented as a contribution.

## 4. Study phases and evidence roles

| Phase | Role | May support scientific claims? | Required output |
| --- | --- | --- | --- |
| Environment/provenance | Reproducibility | No numerical claim | Environment and asset hashes |
| Clean FP16 parity | Pipeline admission | Only pipeline validity | Per-model parity report |
| Smoke test | Debugging | No | Integrity/runtime report |
| COCO 117-run pilot, B=500 | Discovery | Exploratory only | Predictions, metrics, 72 paired artifacts |
| VOC/KITTI transfer pilot | Confirmatory replication candidate | Yes, only under a protocol frozen before inspection | Joint B=2,000 BCa analysis |
| TT100K transfer | Domain transfer | Secondary/replication | Height-stratified analysis |
| Calibration intervention | Mitigation | Yes, with separate frozen plan | Three calibration arms and trade-off analysis |
| Additional architecture | Breadth validation | Yes, scoped to that architecture | Clean parity plus reduced corruption grid |

Important: increasing COCO bootstrap replicates from 500 to 2,000 on the same already-inspected predictions reduces Monte Carlo error but does not turn discovery data into independent confirmatory evidence.

## 5. Experimental scope

### 5.1 Current reduced grid

- Datasets: COCO val2017, VOC2007 test, KITTI validation, TT100K test.
- Models: YOLO11n, YOLO11m, YOLO11x.
- Scientific formats: FP32, INT8-entropy, FP8.
- Gate format: FP16.
- Corruptions: Gaussian noise, motion blur, fog, JPEG.
- Severities: 1, 3, 5.
- Per-dataset scientific runs: (3\times3\times(12+1)=117).
- Per-dataset paired cells: (3\times2\times12=72).

This is sufficient for a focused interaction paper if the confirmatory analysis and reporting are rigorous. It does not establish conclusions for ten corruption types, five quantization formats, or detector families that were not run.

### 5.2 Optional breadth arm

One non-YOLO architecture materially strengthens external validity. It should be a reduced arm, not an uncontrolled expansion:

- one representative capacity;
- FP32, FP16 gate, INT8-entropy, FP8;
- clean plus the same four corruptions at severities 1/3/5;
- identical manifest and statistical contract;
- transformer collapse handled as a diagnostic case study.

Do not start this arm until the four-dataset YOLO pilot is audited and the owner approves the additional compute.

### 5.3 Training-seed boundary

Each precision contrast is paired within the same trained checkpoint, so training randomness does not contaminate FP32-versus-quantized differences for that checkpoint. However, one checkpoint per capacity does not estimate variability over training seeds.

Manuscript rule:

- Primary claim: conditional on the frozen checkpoints.
- Stronger optional check: retrain YOLO11m with three seeds on one representative dataset and repeat clean plus a small corruption subset.
- Never imply that one-seed capacity differences are population-level architecture laws.

## 6. Confirmatory question and hierarchy

The machine-readable candidate is [confirmatory_protocol_candidate_v1.json](../configs/statistics/confirmatory_protocol_candidate_v1.json). It must be reviewed, approved, timestamped, and hashed before transfer metrics are inspected.

### H1 — Primary

Let \(\Theta_q\) be the equal-weight mean of \(\Psi\) over VOC and KITTI, the three capacities, four corruptions, and three severities.

\[
H_0:\Theta_{INT8}\le0,
\qquad
H_1:\Theta_{INT8}>0.
\]

Pass criterion: the one-sided 95% BCa lower confidence bound is above zero.

### H2 — Fixed-sequence format contrast

Only if H1 passes:

\[
\Delta=\Theta_{INT8}-\Theta_{FP8},
\]

\[
H_0:\Delta\le0,
\qquad
H_1:\Delta>0.
\]

This directly tests whether INT8 amplification exceeds FP8 amplification. It is stronger than comparing whether two separate confidence intervals overlap.

### FP8 equivalence

Equivalence is secondary and requires TOST with a frozen practical margin of \(\pm0.01\) AP on \(\Theta_{FP8}\). Failure to reject a difference is not evidence of equivalence.

Report sensitivity at narrower margins only as sensitivity analysis; do not move the primary margin after seeing results.

### TT100K transfer

TT100K uses:

- small-like = XS + S;
- large-like = L + XL;
- \(\Psi^{TT}=E_{small-like}-E_{large-like}\).

Because this endpoint uses height rather than COCO area bins, report it separately. Directional agreement strengthens transfer; numerical pooling with VOC/KITTI is prohibited.

## 7. Bootstrap contract

### 7.1 Sampling unit

The image is the resampling unit. One draw must be shared across FP32/quantized × clean/corrupt for every linked estimand.

### 7.2 Composite estimands

The current B=500 cell-wise pilot is appropriate for discovery. Confirmatory averages require a joint bootstrap:

1. Draw one vector of image indices for VOC.
2. Reuse it across all VOC models, formats, clean and corrupt conditions entering \(\Theta\).
3. Draw a separate vector for KITTI and reuse it across all KITTI cells.
4. Recompute AP, Q, E, \(\Psi\), \(\Theta\), and \(\Delta\) inside each replicate.
5. Build BCa intervals from exactly 2,000 valid replicates.

Running independent cell bootstraps and averaging their interval endpoints is invalid because it discards covariance among cells.

### 7.3 Interval method

- Pilot: percentile intervals, B=500, clearly labelled exploratory.
- Confirmatory: BCa intervals, B=2,000.
- Save point estimate, bootstrap vector hash, seed, image-ID hash, linked artifact hashes, acceleration/bias-correction values, and valid-replicate count.

### 7.4 Multiplicity

- H1 then H2: fixed-sequence testing at alpha 0.05.
- FP8 TOST: secondary, reported separately.
- Dataset-specific and family-level intervals: secondary.
- Individual model × corruption × severity cells: exploratory with BH-FDR q=0.05 within each dataset/precision family.
- Never promote an exploratory cell to the main claim because its p-value is small.

## 8. Hard evidence gates

### G0 — Provenance

Pass only if the record contains:

- Python executable and version;
- Torch, Ultralytics, TensorRT, CUDA-related versions;
- GPU model and driver;
- `trtexec` version/path;
- OS/platform;
- script, annotation, checkpoint/ONNX/engine, calibration-list hashes;
- full command and UTC timestamps.

### G1 — Dataset integrity

Pass only if:

- image count exactly matches the frozen split;
- image IDs and output paths are unique;
- annotation and class-map hashes match the frozen registry;
- no train image enters validation/test inference;
- no validation/test image enters calibration;
- missing/unreadable images = 0.

### G2 — Corruption integrity

Pass only if:

- every output hash is recorded;
- clean and corrupt bytes differ for every expected condition;
- the corruption configuration/version is frozen;
- severity parameters are monotone by construction;
- image dimensions and IDs are preserved;
- sampled pixel-distance diagnostics increase sensibly with severity;
- all formats cite the same condition manifest hash.

AP is not required to decrease monotonically for every cell; that would be a result-dependent validity rule. Parameter and image-distance monotonicity are the integrity checks.

### G3 — FP16 parity

For each dataset × model input path:

- FP32 and FP16 use identical IDs, preprocessing, decoder and post-processing;
- absolute clean AP difference <= 0.01;
- output count/confidence diagnostics show no decoding collapse;
- report and all linked hashes are immutable.

If a gate fails, stop that dataset/model branch and diagnose. Do not interpret INT8/FP8 from a path that fails FP16 parity.

### G4 — Prediction validity

- prediction IDs belong to the frozen split;
- class IDs match the dataset class map;
- scores and boxes are finite;
- boxes are clipped and nonnegative;
- detection cap, confidence threshold and NMS are fixed;
- raw prediction JSON is retained;
- prediction hash is embedded in the metric record.

### G5 — Statistical admission

- all four linked cells exist for each Q/E contrast;
- linked image IDs are identical and ordered;
- no result-dependent exclusions occurred;
- joint bootstrap implementation passes stock-evaluator checks;
- confirmatory protocol hash predates metric inspection;
- every reported number is traceable to immutable inputs.

## 9. Collapse and floor-effect policy

A quantized output must be moved to a diagnostic case study when any predeclared mechanical condition is met, for example:

- clean AP is near the evaluator floor;
- a large prespecified fraction of images has empty output;
- decoder/parity validation fails;
- prediction counts or confidence distribution indicate catastrophic collapse.

The exact numerical thresholds must be frozen before inspecting the candidate architecture. Report clean/corrupt AP, recall, detections per image, confidence, and empty-output fraction. Never claim that a collapsed model is robust because its clean-to-corrupt difference is small.

## 10. Calibration intervention contract

### Arms

1. Clean: 512 train-split clean images.
2. Balanced mixed: fixed 50% clean and 50% corrupt, balanced over corruption/severity.
3. Severity-matched: train-split images corrupted with the target corruption/severity.

### Independence

- Evaluation split pixels and labels never enter calibration.
- Calibration images never enter the evaluation split.
- At least three independently sampled, hash-registered lists per arm are recommended.
- Engine build logs and calibration cache hashes are retained.

### Endpoint

Primary trade-off candidate:

> reduction in heavy-corruption \(E_{small}\) relative to clean calibration, subject to clean AP loss <= 0.01.

Report the full Pareto trade-off, not only the best corrupt condition. If a calibration arm improves corrupt AP by sacrificing more clean AP than the guardrail, describe it as a trade-off, not a mitigation success.

## 11. System/deployment metrics

Quantization is motivated by deployment, so one controlled table should report:

- engine file size;
- build time;
- warm-up protocol;
- batch size and input size;
- median and p95 latency;
- throughput;
- peak GPU memory if measured reliably;
- exact clock/power policy or an explicit statement that clocks were uncontrolled.

Do not mix inference latency collected while unrelated GPU work is active with scientific performance claims. Accuracy runs may continue, but latency benchmarking requires an isolated GPU window.

## 12. Reviewer-risk register

| Likely reviewer objection | Required response/evidence |
| --- | --- |
| “This is only another robustness benchmark.” | Lead with Q/E/Psi methodology, joint paired inference, and actionable calibration intervention. |
| “Corruption-labelled outputs may actually use clean images.” | Manifest-aware runner plus byte/pixel/hash validation. |
| “INT8/FP8 paths are not comparable.” | Same checkpoint/geometry/decoder and per-model FP16 parity gates. |
| “The bootstrap ignores dependence.” | Joint image resample reused across every cell in each composite estimand. |
| “The hypothesis was selected after looking at COCO.” | Label COCO discovery; freeze transfer protocol before VOC/KITTI metrics. |
| “Four datasets were pooled incorrectly.” | Dataset-specific effects; equal-weight composite only for matching area-bin endpoints; TT100K separate. |
| “One seed is insufficient.” | Scope claims to frozen checkpoints and add a targeted multi-seed robustness arm if resources permit. |
| “FP8 equivalence is inferred from non-significance.” | Formal TOST with frozen margin. |
| “Small-object behavior is a floor artifact.” | Clean-gap subtraction, recall/count/empty-output diagnostics, and collapse policy. |
| “No practical deployment benefit is shown.” | Controlled latency/size table and calibration trade-off. |
| “Reproducibility is weak.” | Immutable hashes, raw predictions, configs, bootstrap inputs, code and data statement. |

## 13. Claim discipline

Use:

- “is associated with” for exploratory cells;
- “the prespecified interaction estimate was…”;
- “within the evaluated YOLO11/TensorRT pipeline…”;
- “replicated in direction on…”;
- “equivalent within ±1 AP point” only after TOST passes.

Avoid:

- “proves”;
- “universally”;
- “robust” based only on a small clean-to-corrupt drop;
- “significant” without naming the test, alpha, estimate and interval;
- “FP8 is better” when only separate confidence intervals overlap;
- “confirmed on COCO” after the COCO pilot was used to shape the hypothesis.

## 14. Submission-ready definition

The paper is ready for manuscript finalization only when:

- every planned dataset/model has a hash-valid training/engine registry;
- FP16 parity passes for every interpreted branch;
- all 117 runs per retained dataset are complete and validated;
- raw predictions and metric records are immutable;
- the COCO discovery report is separated from confirmatory analysis;
- the joint B=2,000 BCa implementation is validated;
- H1/H2/TOST decisions follow the frozen hierarchy;
- TT100K is reported with its native size definition;
- calibration RQ is either complete or removed;
- limitations explicitly cover single architecture family, one training seed, synthetic corruption, TensorRT/hardware specificity, and dataset ontology differences;
- figures/tables can be regenerated from versioned artifacts;
- code/data availability and artifact inventory are written before submission.

## 15. Immediate next actions

1. Let the already-approved four-dataset reduced pipeline finish; do not inspect transfer cell metrics individually before freezing the candidate protocol.
2. Review and freeze the machine-readable confirmatory protocol with owner name, UTC timestamp, and SHA-256.
3. Implement a joint B=2,000 BCa runner; do not reuse independent B=500 cell intervals for the primary average.
4. Implement H1/H2 fixed-sequence decisions, FP8 TOST, BH-FDR exploratory tables, and dataset heterogeneity output.
5. Decide whether RQ4 calibration will be completed or removed before drafting the abstract.
6. Decide whether one non-YOLO reduced arm is affordable after the four-dataset audit.
7. Build manuscript tables/figures only from the final analysis registry, never by manual copy/paste from logs.
