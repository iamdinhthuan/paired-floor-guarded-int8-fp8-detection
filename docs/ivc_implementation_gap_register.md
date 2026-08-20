# Topic C implementation and evidence gap register

Snapshot basis: repository implementation inspected on 2026-08-08. Runtime status is intentionally excluded because it changes while the queue runs.

Legend:

- Implemented: code/config exists and directly covers the requirement.
- Partially implemented: useful pilot behavior exists, but it is insufficient for the final claim.
- Missing: no evidence-producing implementation was found in this project.
- Deferred: intentionally outside the approved reduced-pilot scope.

## 1. Pipeline coverage

| Requirement | Status | Current implementation/evidence | Gap before manuscript claim |
| --- | --- | --- | --- |
| Isolated project and legacy read-only boundary | Implemented | Project paths are rooted under `/home/thuan/topic_c_ivc`; README prohibits legacy writes | Final artifact audit must prove no output path points into legacy roots |
| Environment and asset provenance | Implemented | `src/capture_provenance.py` | Include the final provenance manifest and all engine/checkpoint/calibration hashes in the release inventory |
| Manifest-aware inference | Implemented | `src/coco_infer_trt.py`, `src/topic_c/coco_data.py` | Retain unit/integration evidence for alternate image roots across every dataset |
| Clean/corrupt manifest generation | Implemented | `src/generate_coco_manifest.py`, `src/generate_corruption.py` | Add a final consolidated manifest index and generator dependency versions |
| Corruption byte/pixel validation | Implemented | `src/validate_manifest.py`; transfer preparation fails closed on invalid manifests | Add severity-distance diagnostics and a table proving parameter monotonicity |
| Prediction validation | Implemented | `src/validate_predictions.py` | Consolidate pass/fail counts across all final runs |
| Immutable run registry | Implemented | Inference/training/engine scripts embed hashes and reject overwrite | Add one top-level artifact ledger linking every final claim to the registry chain |
| FP16 parity gate | Implemented | `src/verify_fp16_parity.py`, `src/run_clean_fp16_parity.py` | Main paper needs one compact all-branch parity table, including failed/excluded branches |
| COCO reduced pilot | Implemented | Frozen 117-run plan, evaluation, 72 B=500 cells and completion reports | Treat as discovery only |
| VOC/KITTI/TT100K reduced pilots | In progress/deferred queue | `src/defer_*`, dataset pilot executors and transfer matrix | Validate completion reports and freeze confirmatory protocol before looking at cell metrics |
| One-GPU serial orchestration | Implemented | Training and deferred queues | Record interruptions/resumes and checkpoint hashes in the final operational appendix |

## 2. Statistical coverage

| Requirement | Status | Current implementation/evidence | Required action |
| --- | --- | --- | --- |
| Four-cell paired image bootstrap | Implemented for pilot | `src/paired_bootstrap.py`; shared resample within FP32/quant × clean/corrupt | Keep for discovery and per-cell diagnostics |
| TT100K native height strata | Implemented for pilot | `src/paired_bootstrap_tt100k.py` | Verify exact half-open thresholds and object counts in final report |
| B=500 percentile intervals | Implemented | Pilot bootstrap scripts | Label percentile/discovery everywhere |
| Joint bootstrap for aggregate Theta/Delta | Missing | Current cell jobs are independent | Implement one dataset-level draw reused across all cells contributing to each composite estimand |
| B=2,000 confirmatory resamples | Missing | Existing scripts are frozen at B=500 for pilot | New runner/config/report; do not overwrite pilot artifacts |
| BCa intervals | Missing | Current `percentile()` output is not BCa | Implement bias correction, jackknife acceleration, edge-case checks and unit tests |
| Fixed-sequence H1/H2 | Missing | No decision engine found | Implement directly from the frozen JSON protocol |
| FP8 TOST | Missing in Topic C pipeline | Historical code patterns exist in legacy work only | Port into the isolated project; freeze ±0.01 AP margin before transfer inspection |
| BH-FDR exploratory correction | Missing | No final family-analysis runner found | Define families exactly as protocol and save adjusted p/q values |
| Dataset heterogeneity | Missing | Completion report only enumerates datasets | Report dataset-specific effects; optional descriptive heterogeneity statistic |
| Collapse/floor diagnostics | Partial | Prediction counts and confidence exist in run records | Freeze mechanical collapse thresholds before adding another architecture |

Critical warning: independent B=500 cell intervals cannot be averaged to form a valid interval for the paper’s primary aggregate. The joint runner must preserve covariance by reusing the same image-index vector across all linked conditions within a dataset.

## 3. Scientific breadth

| Dimension | Current scope | Strength | Boundary/action |
| --- | --- | --- | --- |
| Capacity | YOLO11 n/m/x | Strong controlled within-family axis | Claims remain conditional on one family |
| Architecture | YOLO11 only in the main new pipeline | Limited | Add one reduced non-YOLO arm after four-dataset audit if compute and parity permit |
| Precision | FP32, INT8-entropy, FP8; FP16 gate | Appropriate focused comparison | Do not imply conclusions for INT8-max or every FP8 implementation |
| Corruption | Gaussian noise, motion blur, fog, JPEG × 1/3/5 | Good representative reduced grid | Do not claim a ten-corruption benchmark; full grid needs separate approval |
| Domain | COCO, VOC, KITTI, TT100K | Strong breadth | Keep dataset roles and size definitions separate |
| Training randomness | One frozen checkpoint per dataset/capacity | Paired precision comparisons are clean | Add targeted three-seed YOLO11m check or disclose checkpoint-conditional inference |
| Calibration randomness | One main 512-image list per dataset | Adequate pilot provenance | Calibration intervention should use at least three independent lists per arm |
| Real-world degradation | Synthetic corruptions | Controlled repeatability | Explicit limitation; optional real-weather/compression subset if available |

## 4. Calibration intervention gap

RQ4 is not satisfied by the clean INT8-entropy engines used in the reduced pilot.

Before retaining RQ4, implement a separately frozen phase with:

1. clean train-only calibration;
2. 50/50 balanced clean/corrupt train-only calibration;
3. target corruption/severity-matched train-only calibration;
4. at least three independently sampled list hashes per arm;
5. identical checkpoint, quantizer, engine build and evaluation path;
6. a clean AP guardrail of 0.01;
7. a primary heavy-corruption `E_small` endpoint and full Pareto plot;
8. no selection of the “best” calibration arm on evaluation labels without nested validation.

If this phase is not completed, remove calibration from the title, abstract, contribution list, RQs and conclusion.

## 5. Reproducibility artifact ledger to build

The final release needs one generated JSON/CSV ledger with one row per scientific condition and these minimum fields:

```text
condition_id
dataset / split
model / checkpoint_sha256
precision / calibrator
onnx_sha256 / engine_sha256
calibration_list_sha256
corruption / severity
input_manifest_sha256
ordered_image_ids_sha256
prediction_sha256
run_record_sha256
metric_sha256
bootstrap_group_id
environment_manifest_sha256
status
```

The ledger should be generated from registries, never manually transcribed. It must fail when hashes disagree, expected conditions are missing, or one file is referenced by incompatible conditions.

## 6. Figure/table implementation backlog

| Output | Inputs | Validation before use |
| --- | --- | --- |
| Pipeline diagram | Frozen protocol and artifact graph | Every arrow maps to an actual registry/hash field |
| Integrity/parity table | Manifest validators and parity reports | Expected/actual branches match |
| Primary forest plot | Joint B=2,000 report | Plot data hash equals report hash |
| Severity curves | Cell E/Psi table | Same scales, all severities present, no cherry-picking |
| Capacity heatmap | Exploratory adjusted table | Effect and uncertainty encoded separately |
| Domain-transfer plot | Dataset-specific reports | TT100K bin label visibly differs |
| Calibration Pareto plot | Separate intervention report | Clean guardrail displayed |
| Deployment table | Isolated-GPU timing protocol | Warm-up, sample count and load documented |

Each figure-generation script should emit a sidecar JSON containing source hashes and plotting arguments.

## 7. Operational completion audit

Run this audit after all deferred jobs finish, before reading scientific conclusions:

- [ ] All training queues have hash-valid completion reports.
- [ ] Every best checkpoint is copied/registered immutably.
- [ ] Every ONNX/engine registry cites the checkpoint and build log hash.
- [ ] Every calibration list is train-only and hash-valid.
- [ ] Every dataset/model FP16 parity report passes or the branch is excluded.
- [ ] Each dataset has exactly 117 valid scientific conditions.
- [ ] Every condition has prediction, input record, run record and metric.
- [ ] Every condition’s hashes agree across those four artifacts.
- [ ] Each dataset has exactly 72 valid pilot paired cells.
- [ ] Completion reports use canonical/self hashes and expected counts.
- [ ] No Topic C artifact was written into either legacy root.
- [ ] Disk and runtime totals are recorded.
- [ ] Failures/interruption/resume events are documented without deleting logs.

## 8. Confirmatory implementation backlog

Priority order:

1. Owner-review and freeze `configs/statistics/confirmatory_protocol_candidate_v1.json` before transfer metric inspection.
2. Add a joint bootstrap runner producing per-replicate Theta_INT8, Theta_FP8 and Delta.
3. Add BCa and jackknife validation tests using small deterministic fixtures.
4. Add H1/H2 fixed-sequence and FP8 TOST decisions.
5. Add BH-FDR cell table and dataset-specific heterogeneity output.
6. Add the final artifact ledger and claim-to-evidence map.
7. Add deterministic figure/table scripts with sidecar hashes.
8. Decide calibration and non-YOLO arms only after the reduced-pilot audit.

## 9. Stop conditions

Stop a branch and diagnose when:

- FP16 parity exceeds 0.01 AP;
- manifest/image/prediction hashes disagree;
- clean and corruption paths use different ID order;
- corrupt output is byte-identical to clean unexpectedly;
- calibration overlaps evaluation data;
- decoder outputs invalid/nonfinite boxes or class IDs;
- an output is partial but lacks a valid resumable checkpoint/receipt;
- shared-GPU load invalidates a latency measurement window.

Do not stop merely because an effect is small, negative or statistically inconclusive. Those are scientific outcomes, not pipeline failures.

## 10. Claim-to-evidence map

| Candidate claim | Minimum evidence | Status before gap closure |
| --- | --- | --- |
| “Corruption amplifies INT8 damage on small objects” | H1 joint BCa lower bound > 0 under frozen transfer protocol | Not yet authorized |
| “FP8 attenuates amplification relative to INT8” | H2 direct Delta lower bound > 0 after H1 gate | Not yet authorized |
| “FP8 is practically equivalent to FP32’s corruption profile” | Both TOST components pass within frozen margin | Not yet authorized |
| “The effect transfers to small-sign domains” | TT100K native-bin effect and interval, with floor diagnostics | Not yet authorized |
| “Corruption-aware calibration mitigates the effect” | Frozen multi-arm intervention plus clean-loss guardrail | Not yet authorized |
| “The protocol is reproducible” | Complete artifact ledger, scripts, hashes and data statement | Partially supported; final audit pending |
| “The result generalizes beyond YOLO” | Non-YOLO parity-gated reduced arm | Not authorized without added arm |

This table should be revisited before every abstract/conclusion revision. A claim whose evidence row remains incomplete must not appear as a conclusion.
