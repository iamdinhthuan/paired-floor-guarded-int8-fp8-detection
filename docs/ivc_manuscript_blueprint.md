# IVC manuscript blueprint for Topic C

This document is a writing and evidence map. It deliberately contains no numerical result claims.

## 1. Recommended positioning

### Working title

Primary candidate:

> When Quantization Meets Corruption: A Size-Stratified Interaction Study of TensorRT Object Detectors

More conservative candidate:

> Quantization–Corruption Interactions in Object Detection: Matched-Input Evidence Across Object Sizes and Domains

Use “robustness” in the title only if the paper defines it precisely and the confirmatory evidence supports it.

### One-sentence pitch

Existing studies usually measure clean quantization loss or corruption loss separately; this work isolates their interaction using matched image bytes and paired image inference, then tests whether the excess loss concentrates on small objects.

### Novelty statement

The novelty should be framed as the combination of:

1. a difference-in-differences estimand for detector quantization under corruption;
2. object-size-specific interaction analysis;
3. a fail-closed, manifest- and hash-driven TensorRT protocol;
4. joint paired inference and cross-domain replication;
5. an optional calibration intervention with an explicit clean/corrupt trade-off.

Do not present the number of runs or datasets as novelty by itself.

## 2. Abstract contract

Use five compact moves:

1. Problem: quantized detectors are deployed under degraded inputs, but clean accuracy does not reveal compound failure on small objects.
2. Method: define Q, E and Psi; use matched bytes, FP16 parity and paired image resampling.
3. Scope: state exact models, formats, datasets and corruptions without implying untested breadth.
4. Results: report the primary effect and uncertainty, then one replication/format contrast; no cherry-picked cell list.
5. Meaning: state the deployment implication and the architecture/hardware boundary.

Abstract result sentence template:

> Under the prespecified confirmatory protocol, the equal-weight INT8 size-interaction estimate was [estimate, interval], while the direct INT8–FP8 contrast was [estimate, interval]. The direction [did/did not] replicate on TT100K under its native height-based strata.

Do not write the final sentence until the immutable analysis report exists.

## 3. Introduction structure

### Paragraph 1 — Deployment problem

Object detectors are quantized for latency/memory efficiency, yet real inputs contain noise, blur, weather and compression artifacts. Small objects have fewer informative pixels and may be especially vulnerable.

### Paragraph 2 — Literature gap

Separate clean-quantization and corruption-robustness tables cannot answer whether the two factors interact. A larger corrupt drop for INT8 is also not sufficient when INT8 already has a clean gap.

### Paragraph 3 — Methodological gap

Many apparent interactions can be caused by mismatched images, preprocessing, decoders, thresholds, calibration data or resampling. Explain why a paired difference-in-differences design is needed.

### Paragraph 4 — Research questions

- Does corruption widen the quantization gap?
- Is widening larger for small objects?
- Is FP8 closer to the FP32 corruption profile than INT8?
- Does the pattern replicate across domains?
- If retained: can train-only corruption-aware calibration reduce the excess loss without unacceptable clean loss?

### Paragraph 5 — Contributions

List only contributions that are complete. Use parallel language and attach each item to a figure/table/artifact.

## 4. Related-work matrix

Organize the literature review by question, not chronology.

| Theme | What to extract from each paper | Gap Topic C addresses |
| --- | --- | --- |
| Detector quantization | Format, calibration, architecture, clean/corrupt evaluation | Usually clean-only or no interaction estimand |
| Corruption robustness | Corruption suite, severity, detector metrics, object size | Usually compares models, not precision-linked matched cells |
| Small-object detection | Definition of small, failure mechanisms, datasets | Rarely studies quantization × corruption |
| Robust calibration | Calibration distribution and leakage controls | Rarely quantifies clean/corrupt trade-off by size |
| Statistical evaluation | Sampling unit, paired design, CI method | Often reports point AP without paired uncertainty |

For each cited work, record dataset, model, precision, corruption, size analysis, statistical method, and artifact availability in a literature spreadsheet. Do not claim “first” until this matrix is complete.

## 5. Methods outline

### 5.1 Problem formulation

Define Q, E and Psi before describing implementation. Include a four-cell diagram:

```text
                 clean              corrupt
FP32             AP_F,0             AP_F,c
quantized        AP_q,0             AP_q,c

E = (AP_F,c - AP_q,c) - (AP_F,0 - AP_q,0)
```

### 5.2 Datasets and strata

For each dataset report:

- exact release and split;
- image/object/class counts;
- annotation conversion and hash;
- evaluation resolution;
- size-bin definition and object counts per bin;
- why the dataset contributes discovery, replication or transfer evidence.

Never hide that TT100K uses height bins while the other datasets use COCO-style area bins.

### 5.3 Models and training

Report:

- checkpoint initialization;
- epochs, optimizer, image size, batch, augmentation, early stopping;
- seed and determinism flags;
- best-checkpoint selection rule;
- whether one or multiple training seeds are used;
- parameter count and capacity label.

State that format comparisons share the exact checkpoint.

### 5.4 Quantization and TensorRT

Report:

- export opset and dynamic/static shapes;
- TensorRT and ModelOpt versions;
- precision flags;
- calibration method, sample count and split;
- engine and calibration hashes;
- input/output tensors and decoder;
- thresholds, NMS and maximum detections.

FP16 belongs here as a parity gate, not as a primary scientific condition unless explicitly analysed.

### 5.5 Corruption generation

Include the frozen parameter table from `configs/corruptions.json`, deterministic seed rule, library versions, color/range conversions, encoding settings, dimension preservation, and manifest validation.

### 5.6 Experimental design

Separate smoke, discovery, confirmatory transfer and calibration phases. State which decisions were made before each phase was inspected.

### 5.7 Statistical analysis

State:

- image as sampling unit;
- shared draws across linked cells;
- joint draws across conditions entering composite estimands;
- B=500 percentile discovery versus B=2,000 BCa confirmation;
- H1/H2 fixed sequence;
- TOST margin;
- BH-FDR family definitions;
- collapse policy;
- software and bootstrap seeds.

## 6. Results outline

### 6.1 Integrity and parity first

Before accuracy findings, report:

- expected/actual image counts;
- manifest and prediction validation pass rates;
- FP16 parity gaps;
- any excluded/collapsed branch and its predeclared reason.

### 6.2 Clean reference

One compact table: clean AP by dataset/model/format. Use it to establish baseline gaps, not as the paper’s main contribution.

### 6.3 Primary interaction

Lead with \(\Theta_{INT8}\), its BCa interval and H1 decision. Then report dataset-level heterogeneity. Do not lead with the single largest cell.

### 6.4 FP8 comparison

Report direct \(\Delta=\Theta_{INT8}-\Theta_{FP8}\), H2 decision and FP8 TOST. Separate superiority from equivalence language.

### 6.5 Corruption/severity/capacity detail

Use effect plots, not enormous raw-AP tables. Individual cells remain exploratory.

### 6.6 TT100K transfer

Report original-height strata, direction and uncertainty. Explain why it is not pooled numerically with area-bin endpoints.

### 6.7 Calibration intervention

If completed, show a clean-versus-heavy-corruption Pareto plot and calibration-list sensitivity. If incomplete, omit the section and RQ entirely.

### 6.8 System trade-off

One controlled table for engine size, latency, throughput and memory. Keep system performance separate from accuracy runs performed under shared-GPU load.

## 7. Figure plan

### Figure 1 — Controlled pipeline

Show one clean source image becoming one immutable corrupt artifact, then branching into FP32/FP16/INT8/FP8 with shared IDs and a linked four-cell bootstrap.

### Figure 2 — Primary effect forest plot

Forest plot of dataset/model-level \(\Psi\) for INT8 and FP8, with the prespecified aggregate at the bottom.

### Figure 3 — Severity response

Lines for E_small and E_large over severities, faceted by corruption and precision. Use identical y-scales within comparison families.

### Figure 4 — Capacity × corruption heatmap

Heatmap of exploratory \(\Psi\), with uncertainty/significance encoding distinct from effect magnitude.

### Figure 5 — Domain transfer

Dataset-specific standardized layout; TT100K panel explicitly labelled height-based.

### Figure 6 — Calibration trade-off

Only if RQ4 is complete: clean AP change on x-axis, heavy-corruption E_small improvement on y-axis, one point per calibration arm/seed.

## 8. Table plan

| Table | Content | Main text or supplement |
| --- | --- | --- |
| 1 | Datasets, splits, classes, images, objects, strata | Main |
| 2 | Models/checkpoints, formats, engine/calibration provenance | Main compact |
| 3 | FP16 parity and integrity gates | Main |
| 4 | H1/H2/TOST estimates, intervals and decisions | Main |
| 5 | Dataset-level replication and heterogeneity | Main |
| 6 | Latency/size/memory | Main or supplement |
| S1 | Full clean metrics | Supplement |
| S2 | Full corruption metrics | Supplement |
| S3 | All exploratory cell-wise E/Psi and BH-FDR | Supplement |
| S4 | Corruption parameters and pixel diagnostics | Supplement |
| S5 | Artifact hashes and environment | Supplement/repository |

## 9. Result-scenario playbook

### Scenario A — H1 and H2 pass

Main message: corruption amplifies INT8 small-object damage on average, and FP8 reduces that amplification. Still report exceptions and domain heterogeneity.

### Scenario B — H1 passes, H2 does not

Main message: small-object amplification exists, but the evaluated evidence does not establish that FP8 attenuates it relative to INT8.

### Scenario C — H1 fails, FP8 TOST passes

Main message: the controlled pipeline supports an approximately additive model within the frozen margin, rather than compound damage. This is a valid null/equivalence contribution.

### Scenario D — Strong heterogeneity

Main message: average effects conceal corruption- or domain-specific regimes. Emphasize prespecified family analyses and avoid post-hoc storytelling.

### Scenario E — Collapse/floor effect

Move the affected rung to diagnostics. Report counts/recall/confidence and explain why normal interaction metrics are uninformative near the floor.

## 10. Limitations that should appear explicitly

- Main model family is YOLO11 unless a second architecture is added.
- Format behavior is conditional on TensorRT/ModelOpt, GPU generation and engine configuration.
- Synthetic corruptions approximate but do not exhaust real acquisition failures.
- One training seed limits claims about training variability.
- Calibration-list variability may remain unless replicated.
- VOC/KITTI/COCO area strata and TT100K height strata answer related but non-identical questions.
- AP is nonlinear and aggregate results depend on the frozen weighting rule.
- Latency measured on a shared GPU is invalid unless an isolated measurement window is used.

These limitations increase credibility when paired with precise scope; they do not weaken the paper as much as overclaiming does.

## 11. Reproducibility package

The final release should include:

- source code and dependency lock;
- frozen experiment/statistical configs;
- environment/provenance manifest;
- dataset acquisition and annotation-conversion manifests;
- corruption parameter file and manifests;
- checkpoint/ONNX/engine/calibration registries;
- raw predictions when licensing/storage permits;
- metric records and paired bootstrap inputs/outputs;
- scripts to regenerate every table/figure;
- README with exact commands and expected hashes;
- data availability statement and reasons for any restricted artifact.

The repository must never contain server credentials, tokens, private keys, copied shell history, or legacy artifacts that cannot be redistributed.

## 12. Pre-submission red-team checklist

- Can a reviewer trace every main number to a prediction hash?
- Can the same figure be regenerated without manual spreadsheet edits?
- Does every quantized comparison share checkpoint, images, geometry and decoder?
- Is COCO consistently labelled discovery rather than confirmatory?
- Was the transfer protocol frozen before transfer metrics were inspected?
- Does the joint bootstrap preserve covariance across all cells in the aggregate?
- Is TOST used for equivalence rather than non-significance?
- Are dataset-specific size definitions visible in the main paper?
- Are exclusions mechanical and predeclared?
- Are calibration images train-only?
- Is the architecture/hardware boundary repeated in abstract and conclusion?
- Would the main conclusion remain honest if H1 fails?

If any answer is “no,” the manuscript is not ready to submit.
