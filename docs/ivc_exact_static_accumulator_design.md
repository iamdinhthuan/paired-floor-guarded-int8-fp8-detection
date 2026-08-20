# Exact static/multiplicity accumulator feasibility design

## Decision from the representative tie audit

The immutable COCOeval tie audit was run on the frozen four-arm COCO `yolo11n` clean cell and `gaussian_noise-s1` cell.  The remote log is `outputs/logs/ivc_format_contrast_tie_audit_20260811T201140Z.log`.

| Cell | Streams | Tied streams | Unique-score streams | Tie excess | Non-finite scores |
|---|---:|---:|---:|---:|---:|
| clean | 1,280 | 1,272 | 8 | 7,792,120 / 9,666,392 detections | 0 |
| gaussian_noise-s1 | 1,280 | 1,272 | 8 | 7,809,356 / 9,668,868 detections | 0 |

Every stream passed descending-score and stable-mergesort checks.  A fast path that assumes all scores are unique is therefore not viable: it would cover only 0.625% of these representative streams.  There must be no approximate tie handling.

## Non-negotiable equivalence contract

For every endpoint, replicate, arm, and category/area stream, the accelerated result must be bitwise identical to the current `paired_bootstrap.accumulate_ap` result for the fixed schedule row.  In particular it must preserve:

- the schedule row's image occurrence order, including duplicate sampled images;
- the per-image top-100 detection order;
- `np.argsort(-scores, kind="mergesort")` ordering, including the original concatenation order within equal-score ties;
- `dtMatches`, `dtIgnore`, `gtIgnore`, the positive-GT denominator, all IoU thresholds, precision envelope, and COCO recall-grid sampling;
- non-finite output behavior and the current percentile/macro rules.

Point estimates continue to use the stock full-image calculation.  Static draw construction is permitted only after exact parity is demonstrated; it may not change checkpoint, prediction, annotation, engine, corruption, seed, or resampling inputs.

## Candidate exact algorithm

Preprocess an immutable category/area stream into one block per image position.  A block retains its ordered top-100 scores, match/ignore columns for every IoU threshold, and ignored-GT count.  It also indexes its detections by their exact IEEE score value.

For a schedule row, process distinct scores in descending order.  For a score value, concatenate that score's per-image detection subsequences in the schedule-row occurrence order and preserve their within-image order.  This produces precisely the stable-mergesort tie order of the legacy concatenated score vector.  Cumulative TP/FP arrays then use the existing formulas and recall-grid sampling unchanged.

This avoids a replicate-wide score sort but still visits every sampled detection.  It is an exact *multiplicity* path, not a score histogram approximation.  It is only eligible when a direct small-replicate parity test proves it against the legacy stream implementation.

### Stable-tie sensitivity audit

The follow-up read-only audit, `outputs/logs/ivc_format_contrast_stable_tie_sensitivity_20260811T202554Z.log`, classified every tied score group in the same two four-arm COCO cells for every IoU/category/area stream.  A group was `mixed_tp_fp` when it contained both non-ignored TPs and FPs at that IoU threshold; only such a group is internally order-sensitive.

| Cell | Mixed groups | Mixed tied detections | Order-insensitive groups |
|---|---:|---:|---:|
| clean | 3.464% | 6.227% | 96.536% |
| gaussian_noise-s1 | 3.495% | 6.454% | 96.505% |

The high proportion of order-insensitive groups means an exact compiled score-group replay may still be worthwhile.  It does **not** permit discarding or aggregating mixed groups: their schedule-occurrence and within-image sequence must be replayed exactly, and a stream must remain `legacy_tied` if that replay cannot be proved equivalent.

## Fail-closed mode selection

Each prepared stream must carry one explicit mode in its evidence/debug record:

- `unique_exact`: no duplicate finite score value; the simplified unique-score path may be used only after parity tests.
- `multiplicity_exact`: ties exist and the ordered score-bucket implementation has passed parity for that stream type.
- `legacy_tied`: ties exist but the multiplicity path is unavailable or has not passed parity; invoke the unmodified legacy stream calculation.

An unknown score condition, a non-finite score, a shape mismatch, or an unsupported tie layout is a refusal or `legacy_tied`, never a silent approximation.  The initial implementation must make the per-stream selected mode auditable and retain the legacy fallback for every tied stream.

## Test-first gates before implementation

1. A synthetic two-image stream with repeated scores across images and schedule `[1, 0, 1]` must reproduce the legacy stable-mergesort rank sequence for every IoU threshold.
2. The same synthetic stream must produce bitwise-equal AP vectors for `legacy_tied` and `multiplicity_exact`, including all-NaN/no-positive cases.
3. A synthetic unique-score stream must prove `unique_exact` equals legacy and must fail if an equal score is inserted without switching mode.
4. A fixed real COCO clean and corrupt four-arm probe (at least B=8) must compare every arm AP draw, format draw, point estimate, percentile, and macro input bitwise between legacy and the static path.
5. The test suite must reject an unrecorded fallback, a mode/score-tie mismatch, a reordered duplicate image, and a changed schedule row.
6. Only after those gates may a bounded remote timing/RSS probe decide whether the multiplicity path makes the 2,000-replicate grid feasible.

## Current status

No static accumulator has been used for a scientific artifact or a full grid.  The design is pending implementation and independent review.  The current cache/schedule path remains stopped because its measured B=8 four-worker rate projects to roughly 48 hours for 624,000 arm draws.
