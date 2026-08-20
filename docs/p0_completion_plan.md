# P0 paper completion plan

## Objective

Replace the codec-confounded/legacy-COCO analysis with the completed P0 evidence, revise the IVC manuscript around the matched-codec estimand, and produce an audited final LaTeX PDF.

## Global constraints

- Preserve every legacy artifact; P0 data live only in their versioned directories.
- Treat `outputs/bootstrap/matched_codec_p0_v1` as the primary interaction source: exactly 288 cells, 500 paired image-bootstrap replicates per cell, split evenly across COCO, VOC, KITTI, and TT100K.
- Treat `outputs/metrics/coco_uniform_p0_v1` as the canonical COCO metric source: exactly 117 conditions from the rebuilt uniform TensorRT ladder.
- Treat `outputs/metrics/codec_control_p0_v1` as the JPEG-quality-95 clean control source: exactly 36 conditions.
- Use the deterministic JPEG-quality-95 clean controls in the primary excess-gap estimand. Retain original-clean metrics only for the explicit codec sensitivity analysis.
- Validate all locally available completion-report hashes before generating paper artifacts.
- Do not pool TT100K's height endpoints numerically with COCO/VOC/KITTI area endpoints.
- State that COCO uses official Ultralytics pretrained checkpoints; VOC, KITTI, and TT100K use their recorded `best.pt` checkpoints.
- Never claim strict IEEE FP32 because TF32 was not explicitly disabled.
- Paper values must be generated programmatically, not copied manually.
- Final deliverables are `paper/main.tex`, generated tables/figures, `paper/main.pdf`, and an updated audit document.

## Task 1: Integrate P0 evidence into the artifact builder

1. Add failing tests for P0 artifact loading and validation.
2. Update `analysis/build_paper_artifacts.py` to validate the P0 completion chain, load unified COCO metrics, load codec-control clean metrics, and load all matched-codec bootstraps.
3. Generate a codec-sensitivity table comparing original-clean and JPEG-95-clean AP, plus machine-readable CSV output.
4. Make all primary summaries, figures, tables, and numerical macros use matched-codec bootstraps across the intended datasets.
5. Run the targeted tests and the full test suite.

## Task 2: Regenerate and interpret the paper artifacts

1. Execute the builder against the synchronized artifact root.
2. Inspect generated values for sign, scale, missing data, and endpoint consistency.
3. Record the principal findings and limitations in a machine-readable audit.

## Task 3: Revise the manuscript

1. Use the title `When Quantization Meets Corruption: Paired Excess-Gap Evaluation of TensorRT INT8 and FP8 Object Detectors` unless the completed evidence supports a more precise title.
2. Remove the legacy-COCO and unmatched-codec caveats that P0 resolved.
3. Explain the matched JPEG-95 control, uniform COCO engine build, codec-only sensitivity, and updated evidence chain precisely.
4. Rewrite results, abstract, highlights, discussion, limitations, and conclusion to match only generated evidence.
5. Keep claims exploratory and suitable for Image and Vision Computing.

## Task 4: Verify and deliver

1. Run all tests.
2. Compile LaTeX to PDF and inspect the log for undefined citations/references, overfull boxes, and fatal warnings.
3. Visually inspect representative PDF pages.
4. Perform an independent methods/artifact/manuscript review and resolve load-bearing findings.
5. Update the review report and provide exact remote COCO checkpoint paths and hashes.
