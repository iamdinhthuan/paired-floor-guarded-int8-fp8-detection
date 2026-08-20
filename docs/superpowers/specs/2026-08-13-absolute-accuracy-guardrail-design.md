# Absolute Accuracy Guardrail Design

## Purpose

Strengthen the direct INT8--FP8 corruption analysis with absolute corrupted AP and
the loss statistic \(D=AP_{clean}-AP_{corrupt}\). The addition must reuse the
already completed metric chain and must not rerun inference. Its purpose is to
prevent a contracting INT8--FP8 gap from being mistaken for robustness when both
formats have collapsed in absolute accuracy.

## Evidence boundary

The direct paired components remain the source of the exploratory format contrast
\(\Delta E=D_{INT8}-D_{FP8}\). An absolute metric record is admissible only when
all of the following equal the corresponding direct arm: prediction SHA-256,
input-manifest SHA-256, ordered-image SHA-256, run-record SHA-256, dataset, model,
format, corruption, and severity. Before indexing those records, the builder must
validate the existing P0 completion chain.

The exact grid is 4 datasets x 3 YOLO11 capacities x 2 quantized formats x 4
corruptions x 3 severities = 288 corrupted arms. These arms reuse 24 clean
format/model/dataset records, so the evidence set contains exactly 312 distinct
metric files. Each direct component must satisfy
\(\Delta E=D_{INT8}-D_{FP8}\) to an absolute tolerance of \(10^{-12}\).

## Outputs

The builder emits a machine-readable 288-row CSV, a compact dataset-by-format
LaTeX table, and a generated narrative. Values in the CSV use native AP fractions;
the data dictionary defines the conversion to AP points. The direct evidence audit
binds every one of the 312 metric-file hashes, both parent completion reports, and
all generated guardrail files.

The manuscript reports clean AP, mean corrupted AP, mean \(D\), severity-5 AP,
and severity-5 \(D\). It must keep the direct bootstrap and the absolute metric
chain conceptually separate: the absolute values are hash-matched guardrails, not
additional bootstrap endpoints or confirmatory superiority tests.

## Failure behavior

Generation fails before creating plausible output if any completion report is
invalid, any direct arm lacks a metric, any provenance or semantic field differs,
a prediction hash is ambiguous, the 288-arm/312-record grid is incomplete, or a
component's stored \(\Delta E\) cannot be reconstructed from absolute losses.

## Verification

Tests exercise the real archived evidence and independently check the exact grid,
record count, median loss, low-AP count, and algebraic reconstruction. The final
gate rebuilds the direct artifacts, main PDF, supplement, submission audit, and an
isolated Overleaf ZIP; the PDFs must be visually inspected for table overflow and
front-matter regression.

## Scope

No predictions, metrics, engines, checkpoints, calibration subsets, or bootstrap
draws are regenerated. Multi-seed validation is a separate prospective stage with
its own frozen design and completion chain.
