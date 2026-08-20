# IVC submission-strengthening notes

This note records how the manuscript was strengthened from a short pilot report
into a defensible full study. It is a working author document, not manuscript
text and not a substitute for the journal checklist.

## Central paper claim

The paper does **not** claim that one eight-bit format is universally robust or
that small objects always suffer an amplified quantization--corruption effect.
Its central contribution is a controlled, executable, paired framework that
separates three quantities:

1. clean quantization cost, `Q0`;
2. absolute corruption loss, `D`;
3. the corruption-induced change in the quantization gap, `E`, followed by the
   small-minus-large contrast, `Psi`.

This decomposition is the conceptual novelty. The 4-dataset × 3-capacity ×
2-format × 4-corruption × 3-severity grid provides the empirical stress test.

## Evidence now exposed in the manuscript

- 468 scientific metric records and 288 paired interaction cells;
- 500 image-paired percentile-bootstrap replicates per interaction cell;
- complete clean-reference and FP16 internal-consistency tables;
- absolute AP trajectories for every dataset and corruption family;
- corruption-family and model-capacity decompositions;
- a full atlas containing every `Psi` point estimate;
- a forest plot for the largest non-crossing exploratory intervals;
- TT100K native-height profiles and explicit separation from COCO area bins;
- SHA-256 linkage from checkpoint and engine to input, prediction, metric, and
  bootstrap artifact;
- a local archive of engines, manifests, raw predictions, metrics, bootstraps,
  logs, reports, and the historical quarantine.

## Strongest defensible findings

- On the uniformly built transfer datasets, the descriptive clean AP loss is
  +2.57 points for INT8 and +0.83 for FP8.
- The corresponding mean excess overall gaps are -0.39 and -0.17 AP points.
- Across the 288 cell-wise `Psi` intervals, 3 are wholly positive, 35 wholly
  negative, and 250 cross zero. These are exploratory interval classifications,
  not multiplicity-adjusted discoveries.
- On the primary VOC/KITTI area endpoint, motion blur produces the most negative
  INT8 family mean (`Psi = -1.76`), but
  absolute trajectories show that severe blur often drives both formats toward
  an AP floor. A negative interaction must not be called a robustness gain.
- Capacity-labelled checkpoint rungs are not monotone: YOLO11x has the smallest
  transfer-set clean INT8 gap but the most negative VOC/KITTI area interaction.
- FP8 is the stronger clean-accuracy candidate on the exact TensorRT/RTX 5090
  stack, but the present design does not establish formal interaction
  superiority or equivalence.

## Reviewer-facing strengths

1. **Matched treatment:** precision arms consume the same ordered image IDs and
   encoded bytes.
2. **Executable treatment:** results come from built TensorRT engines, not only
   fake-quantized tensors.
3. **Paired estimand:** the clean numerical penalty is removed before corruption
   interaction is interpreted.
4. **Heterogeneity is visible:** the paper reports the full factorial atlas and
   absolute trajectories rather than one grand mean.
5. **Endpoint discipline:** TT100K height strata are implemented through the
   evaluator's native range semantics, unit-tested, and never silently pooled
   with area strata.
6. **Fail-closed provenance:** missing cells, wrong hashes, wrong replicate
   counts, stale quarantine artifacts, or a degenerate height endpoint stop the
   paper build.

These strengths establish internal traceability. They do not remove the shared
JPEG-95 materialization confound or resolve the legacy COCO FP8 build command.

## Fit to Image and Vision Computing

The journal's current public scope emphasizes high-quality theoretical or
applied computer-vision research, real-world image interpretation, and
quantitative comparison and performance evaluation. The paper should therefore
be positioned around its paired evaluation methodology and auditable deployment
evidence—not as a claim that YOLO11 itself is a new detector. Object recognition,
real-scene deployment, quantitative evaluation, and small-object behavior give
the submission a direct scope connection. Recheck the live Guide for Authors at
submission time because the public page and template requirements can change.

## Claims to avoid during revision or rebuttal

- “FP8 is statistically superior to INT8 under corruption.”
- “Quantization improves robustness” when `E` or `Psi` is negative.
- “Small objects are universally more affected.”
- “The results generalize to all detectors, GPUs, or deployment compilers.”
- Latency, energy, throughput, memory, or efficiency claims from these accuracy
  runs.
- A “first” claim without a documented systematic literature search.
- “FP16 parity proves that TensorRT reproduces the source checkpoint.”
- “FP32 is strict IEEE FP32” because TF32 was not explicitly disabled.

## High-value follow-up experiments

If another experimental round is authorized, prioritize the following order:

1. add a codec-only control or regenerate clean and corrupted arms with matched
   lossless/materialization treatment;
2. rebuild COCO with the explicit transfer FP8/INT8 pipeline and command log;
3. define a strict no-TF32 source/ONNX/TensorRT consistency protocol;
4. freeze motion blur as the primary corruption and define a joint resampling
   plan with at least 2,000--5,000 replicates;
5. add a second detector family, preferably one architecturally distinct from
   YOLO11;
6. repeat training and calibration seeds to expose between-build variance;
7. compare clean versus corruption-aware calibration as a prespecified
   intervention;
8. add an isolated, synchronized latency/energy protocol only if efficiency is
   to become a paper claim;
9. collect real camera degradation or temporal sequences to complement the
   synthetic corruption grid.

## Final submission checks

- Confirm author order, affiliations, corresponding author, CRediT roles,
  funding, acknowledgments, and conflicts with every author.
- Keep the abstract at no more than 250 words and submit the separate 3--5
  highlights; the five current bullets are each below 85 characters.
- Confirm that the generative-AI disclosure accurately describes the tools and
  uses approved by all authors; revise it before submission if their workflow
  differs from the recorded wording.
- Replace the working two-column article class with the current Elsevier source
  template only at packaging time; preserve the generated tables and figures.
- Re-run the artifact builder and inventory after any artifact movement.
- Require zero undefined citations/references and zero overfull boxes.
- Inspect every PDF page at readable resolution; verify that figures are not
  split, cropped, or placed after the section that interprets them.
- Archive `main.tex`, `references.bib`, `generated/`, `figures/`, the build log,
  and both audit JSON files with the submitted source.
