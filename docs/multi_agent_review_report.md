# Multi-agent manuscript audit and strengthening report

Date: 2026-08-10

This is an author-facing triage record. Three independent reviews examined
methods/statistics, Image and Vision Computing positioning, and artifact/code
integrity. Findings were checked against the manuscript, generated ledgers,
engine registries, and archived source before revision.

## Executive assessment

The paper has a credible measurement contribution and an unusually complete
internal evidence chain, but the current archive does **not** justify a universal
INT8-versus-FP8 robustness claim. The strongest defensible contribution is the
paired excess-gap methodology implemented with executable TensorRT engines and
byte-identical precision arms.

Two issues require new inference or regenerated inputs and cannot be solved by
prose:

1. every corrupted image includes deterministic JPEG-quality-95 materialization
   while clean images retain their original encoding; and
2. COCO FP8 was built through a legacy path whose exact calibration command was
   not archived, whereas the three transfer datasets use an explicit matched
   entropy configuration.

The revised manuscript makes both boundaries explicit, uses the three uniformly
built transfer datasets for primary overall means, uses VOC and KITTI only for
the primary area interaction, and retains COCO and TT100K as distinct sensitivity
arms.

## High-priority findings fixed in the manuscript and analysis

| Finding | Risk | Implemented response |
|---|---|---|
| VOC split was called validation | Protocol misreporting | Corrected to `VOC2007 test` throughout |
| FP16 gate was called checkpoint parity | Overstates validation | Renamed gross internal TensorRT consistency gate; explicitly not checkpoint parity |
| FP32 label omitted possible TF32 kernels | Treatment ambiguity | Defined as FP32-typed TensorRT reference; TF32 was not explicitly disabled |
| COCO and transfer FP8 provenance differ | Nonexchangeable treatment | Primary aggregates exclude COCO; COCO is a legacy-pipeline sensitivity arm |
| Area and TT100K height endpoints were pooled in summaries | Endpoint mismatch | Primary `Psi` summaries and severity plots now use VOC/KITTI area only; TT100K is separate |
| Clean/corrupted encoding differs | Corruption/codec confounding | Claims now concern the complete transform-plus-JPEG-95 pipeline |
| Severity randomness is not nested | No strict dose response | Severity profiles are described as ordinal, independently realized settings |
| VOC/KITTI use COCO conversions | Could be mistaken for official scores | Defined as COCO-style measurements, not official VOC/KITTI leaderboard protocols |
| Selection and evaluation reuse VOC/KITTI splits | Optimistic conditional evaluation | Disclosed as post-selection rather than untouched holdout evidence |
| Capacity language was causal | Training factors co-vary | Reframed as differences across capacity-labelled checkpoint rungs |
| TT100K class space is sparse | Bootstrap class composition varies | Reported 221 declared/136 observed classes and bootstrap limitation |
| TT100K archive has 3067 rather than 3071 test images | Reproducibility ambiguity | Disclosed four unavailable upstream-referenced files |
| Audit counted noncanonical JSON | Evidence-count contradiction | Schema 2 hashes canonical 468 metrics, 288 bootstraps, and 9 reports separately from excluded JSON |
| Interval sign counts dominated the claim | Multiplicity risk | Demoted to a descriptive inventory; no pooled or adjusted inference claimed |
| Closest compression--corruption work was absent | Novelty vulnerability | Added an explicit comparison matrix and removed priority language |
| Public auditability was implied | No DOI/public repository | Changed to internal traceability and added a pre-submission archive requirement |

## Experiments needed for a materially stronger submission

These are ordered by internal-validity value, not runtime.

### P0: resolve treatment confounds

1. **Codec-only control or lossless regeneration.** Evaluate clean originals,
   clean images re-encoded at JPEG 95, and each corruption materialized losslessly
   or with the same codec applied to both clean and corrupted arms. This separates
   corruption from materialization.
2. **Rebuild COCO through the transfer engine pipeline.** Use the same frozen
   512-image calibration protocol, explicit entropy setting, quantizer code,
   TensorRT/ModelOpt versions, and command logging. Training is not required.
3. **Strict numerical reference check.** Rebuild with TF32 explicitly disabled
   and compare the source checkpoint, ONNX, FP32-typed TensorRT, and FP16 engine
   on matched predictions. Set tolerances before looking at results.

### P1: strengthen uncertainty and evaluation validity

4. Run a joint image bootstrap with 5,000 replicates reused across all cells in
   each prespecified composite; report intervals for primary mean `E`, mean
   `Psi`, and paired INT8--FP8 differences.
5. Add class-aware or stratified resampling for sparse TT100K categories.
6. Create untouched transfer holdouts and reproduce official difficult/ignore
   semantics where official benchmark comparisons are claimed.
7. Repeat training and calibration seeds to identify between-checkpoint and
   between-build variability.

### P2: improve generality and deployment relevance

8. Add an architecturally distinct detector family and at least one competitive
   PTQ baseline or ablation.
9. Add isolated latency, serialized-engine size, peak memory, and energy
   measurements only under a synchronized deployment protocol.
10. Add real camera or natural-shift data to complement the synthetic suite.

## Submission blockers outside the experiment

- Obtain a permanent public repository/DOI or write the journal-compliant data
  availability statement and reviewer-access procedure.
- Confirm author order, affiliations, corresponding author, CRediT roles,
  funding, acknowledgments, and conflicts with every coauthor.
- Port the source to the current Elsevier/IVC article template at packaging.
- Confirm that the AI-assistance declaration matches every author's approved
  workflow.
- Recheck the live Guide for Authors immediately before submission.

## Recommended paper positioning

Use this claim:

> We provide an exploratory, executable-engine measurement framework that
> separates clean quantization loss from the corruption-induced change in that
> loss, preserves image-level pairing, and exposes size-conditional
> heterogeneity across datasets.

Avoid claims that FP8 is universally more corruption-robust, that negative
interaction means improved robustness, that the capacity rung causes the
observed pattern, or that the existing archive is already independently
auditable.
