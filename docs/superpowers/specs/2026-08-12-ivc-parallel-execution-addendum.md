# Exact parallel execution addendum for paired-format evidence

## Decision

The static unique-score accelerator is not eligible: a representative audit found score ties in 1,272 of 1,280 COCO category--area--arm streams. Although only about 3.5% of tied groups are TP/FP order-sensitive, an exact compiled stable-tie backend is a separate research-engineering task and has not passed parity gates. It will not be used for scientific evidence in this revision.

The evidence run will instead retain the stock, validated `accumulate_ap` per stream and reduce wall time only through bounded **cross-cell** parallelism. This does not change the evaluator, score ordering, bootstrap schedule, percentile rule, AP reduction, or direct-format algebra.

## Execution contract

- Each corrupted cell reuses its hash-bound dataset/model clean-arm cache and evaluates only its two corrupt arms.
- All cells in one dataset continue to read the same immutable 2,000-row schedule. Concurrent processes may read it but never write it.
- A cell process uses exactly one bootstrap worker. Parallelism is only at the independent-cell level; nested cell × replicate worker pools are prohibited.
- A preflight probe must run real frozen two-corrupt-arm COCO cells at several concurrent-cell caps, record wall time, aggregate parent-plus-child RSS, command/PIDs, and exact draw digests. The production cap is no larger than the largest tested cap that leaves a documented safety margin on the 62-GiB host.
- Each task has its own immutable output JSON, draw cache, and task log. The scheduler records PID, command, start/end, exit status, and file SHA-256.
- A failure must stop new launches, wait for already launched tasks to exit, preserve their immutable outputs, and emit a failure ledger. It must never delete or overwrite artifacts.
- A `--resume-verified` mode, if implemented, may skip only artifacts/caches/logs that pass all current schedule, clean-cache, source-hash, self-hash, and precommitted draw-cache validations. It must reject stale/partial/mismatched state rather than repair it.
- The macro and completion report are produced only after all 12 clean and 144 corrupted cells verify. The 317-file evidence chain remains mandatory.

## Required gates

1. Synthetic scheduler tests prove cardinality, cap enforcement, non-overlapping outputs, failure stop behavior, and verified-resume rejection on one altered artifact/cache.
2. A real B=8 multicell probe proves exact draw equality against serial for each process and records RAM/time for the selected cap.
3. The existing 2,000-replicate cache/schedule/seed/validator suite passes; the remote completion test remains the only temporarily deselected test.
4. An independent review approves both statistical invariance and resource safety before the persistent full grid launches.

## Reporting

The manuscript will not describe this as a runtime or deployment result. It is an execution detail recorded in the reproducibility material. The manuscript's direct-format results remain explicitly targeted exploratory bootstrap analyses.
