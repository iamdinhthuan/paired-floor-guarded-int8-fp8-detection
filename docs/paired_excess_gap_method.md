# Paired excess-gap method

This note specifies the analysis contract used by the manuscript, the direct
INT8--FP8 contrast chain, and the generated reporting artifacts.  It is a
measurement framework: it does not identify a causal mechanism inside a
detector or establish a hardware-general ranking.

## Question and atomic comparison

For one dataset, checkpoint, size endpoint, and named corruption setting, the
atomic unit consists of four AP measurements evaluated on the same ordered
image universe and the same encoded input bytes:

| Input condition | Reference arm | Quantized arm |
| --- | --- | --- |
| Matched JPEG-95 clean control | FP32-typed TensorRT | format `q` |
| Named corruption followed by the same JPEG-95 materialization | FP32-typed TensorRT | format `q` |

`q` is INT8 or FP8.  The clean control is not the source-image clean score:
it is the deterministic JPEG quality-95, subsampling-0 materialization used
after every named corruption.  This removes a direct source-codec versus
corrupted-codec mismatch from the primary comparison.

## Estimands and signs

Let `AP(p, c, b)` denote AP for precision arm `p`, condition `c`, and endpoint
`b`.  Let `95` denote the matched clean control.

\[
Q_q(c,b) = AP(\mathrm{FP32},c,b)-AP(q,c,b),
\]

\[
E_q(c,b) = Q_q(c,b)-Q_q(95,b).
\]

Positive `Q` means the quantized engine is below the FP32-typed reference.
Positive `E` means the named corruption widens that quantization gap relative
to the matched clean condition.  Negative `E` means the gap narrows; it is not
evidence that either detector is robust or that the quantized arm improved.

For area endpoints, the size interaction is

\[
\Psi_q(c) = E_q(c,\mathrm{small})-E_q(c,\mathrm{large}).
\]

TT100K uses its native height-based small-like and large-like endpoints.  Its
`Psi` is reported separately and is never numerically pooled with area-based
`Psi`.

## Direct INT8--FP8 format contrast

The planned format-level quantity is not a comparison between two separately
averaged format summaries.  It is computed inside each paired resample:

\[
\begin{aligned}
\Delta E(c,b)
 &= E_{\mathrm{INT8}}(c,b)-E_{\mathrm{FP8}}(c,b)\\
 &= [AP(\mathrm{FP8},c,b)-AP(\mathrm{INT8},c,b)]\\
 &\quad-[AP(\mathrm{FP8},95,b)-AP(\mathrm{INT8},95,b)].
\end{aligned}
\]

The FP32-typed reference therefore cancels algebraically.  Positive `Delta E`
means the INT8--FP8 discrepancy expands under the named corruption relative to
matched clean; negative `Delta E` means it contracts.  The paired size
contrast is `Delta Psi = Psi_INT8 - Psi_FP8`.

## Pairing and uncertainty contract

Every four-arm cell must bind the following before any point estimate or
bootstrap result is accepted:

- identical ordered, unique image IDs and exact evaluation-image count;
- identical encoded-image bytes within a clean or corrupted condition;
- matching annotations, class map, inference geometry, decoder, and evaluator;
- a hash-bound checkpoint, engine, input manifest, and run record for every arm;
- one deterministic dataset schedule reused for all direct-format components;
- one image-level bootstrap draw reused across all arms of that component.

The direct contrast pipeline uses 2,000 image-paired bootstrap replicates.
Intervals are exploratory percentile summaries: they quantify finite
evaluation-set uncertainty conditional on the frozen engines and artifacts.
They do not cover training seeds, calibration samples, engine builds, dataset
selection, multiple testing, or a prospective confirmatory hypothesis.

## Absolute-loss guardrail

Signed interaction is always accompanied by absolute corruption loss,

\[
D_p(c,b)=AP(p,95,b)-AP(p,c,b).
\]

A negative `E`, `Psi`, or direct contrast can occur when both arms approach an
AP floor and their separation becomes small.  Reports must therefore show the
associated absolute AP or `D` before describing a narrowed gap as operationally
favorable.

## Minimum reporting checklist

1. Report `Q_95`, `E`, and `D` with their sign conventions.
2. Report `Delta E` and, where defined, `Delta Psi` from shared bootstrap draws.
3. Keep area and height endpoint families separate.
4. Provide condition-level machine-readable summaries in addition to balanced
   descriptive aggregates.
5. State that bootstrap intervals are exploratory and identify every excluded
   source of uncertainty.
6. Publish or retain hash-bound manifests, run records, compact metric summaries,
   bootstrap artifacts, and validation instructions; raw predictions can remain
   access-controlled when their identities are bound by the retained records.
