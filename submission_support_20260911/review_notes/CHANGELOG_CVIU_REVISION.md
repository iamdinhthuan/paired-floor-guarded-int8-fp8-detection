# CVIU revision changelog

## Story and positioning
- Retitled the paper to **A Paired Evaluation of Clean Accuracy and Corruption Sensitivity in Quantized Object Detection**.
- Rewrote the abstract around the central empirical insight: absolute corrupted superiority and clean-adjusted corruption sensitivity are different claims.
- Removed defensive novelty language from the abstract/introduction (e.g., repeated emphasis on what is *not* new).
- Reframed novelty as an executable paired measurement protocol with explicit controls, shared-image uncertainty, and absolute-accuracy interpretation.
- Scoped the primary empirical claim to clean-meaningful YOLO11 TensorRT INT8/FP8 treatments.

## Main-text restructuring
- Reduced the main manuscript from **8,809 to 5,098 words** (TeXcount) and from **18 to 13 compiled pages**.
- Consolidated Results from 13 subsections to 5 main result blocks.
- Kept the strongest evidence in the main text: selection-disjoint holdouts, four AP components, control sensitivity, absolute accuracy, sign-disagreement analysis, metric/covariance sensitivity, runtime, and cross-family failure boundary.
- Moved the interpretive burden of detailed realization/seed/class-universe/recipe-transfer diagnostics to Supplementary Information.

## Methods and statistics
- Simplified the estimand presentation around direct FP8--INT8 clean/corrupted gaps and the paired interaction.
- Replaced the term “floor-guarded” in the scientific narrative with **absolute-accuracy guardrail**.
- Kept the low-AP limiting argument but made clear it is not a statistical floor detector or deployment threshold.
- Clarified what the paired bootstrap interval does and does not cover.

## Reproducibility and disclosure
- Added an AI-assisted research-workflow subsection because AI tools were used in coding/orchestration support.
- Updated the end-of-manuscript generative-AI declaration to emphasize author review, testing, editing, and responsibility.
- Rewrote Data and Code Availability to distinguish the public v2.1.0 release from post-v2.1.0 evidence.

## Submission assets
- Added a CVIU cover letter.
- Added 5 highlights, each under 85 characters.
- Added a non-generative graphical abstract built from the manuscript’s existing scientific figure and final holdout result.
