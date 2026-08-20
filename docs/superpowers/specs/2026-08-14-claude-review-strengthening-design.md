# Claude-Review Manuscript Strengthening Design

## Goal

Resolve the technically verified findings from the independent Claude Code
Reviewer-2 audit without inventing author metadata or treating the in-flight
multi-seed study as completed evidence.

## Evidence boundary

- Only artifacts already bound by `paper/generated/direct_evidence_audit.json`
  may enter the current direct-evidence manuscript.
- The targeted multi-seed results remain absent until their metric and analysis
  completion reports validate.
- Funding and archive DOI remain explicit author gates; no factual declaration
  or public identifier is inferred.
- Citations remain 2023--2026 and retain exactly five Image and Vision
  Computing papers, per the author's policy.

## Changes

1. Convert every direct paired cell to AP points before plotting Figure 2 and
   label both panels unambiguously.
2. Answer RQ3 with compact, existing-evidence summaries by checkpoint rung and
   corruption family, plus condition-level severity/extrema prose. Avoid adding
   another floating figure while the multi-seed table is still pending.
3. Add the verified FP16-gate outcome, TT100K class-composition caveat, COCO
   historical-config exception, and guardrail lineage to the active direct
   branch.
4. Report both median and mean absolute corruption loss because their
   separation describes a strongly skewed loss distribution.
5. Make highlights quantitative and retain the current five-item/85-character
   submission constraints.
6. Add fail-closed regression tests for units, disclosures, RQ3 outputs, and
   rendered float ordering; rebuild and visually inspect the PDFs.

## Success criteria

- Direct figure source and rendered axis both use AP points.
- Main PDF answers RQ3 across dataset, capacity, corruption, and severity.
- Main PDF contains the FP16 outcome, TT100K bootstrap caveat, COCO provenance
  exception, and earlier-P0 guardrail lineage.
- Submission audit and full tests pass; every Results float remains before
  Discussion.
- No unvalidated multi-seed number, funding claim, DOI, or citation is added.

## Self-review

The design contains no placeholder or unresolved scientific choice. Author-only
facts are explicitly out of scope, and all manuscript changes are tied to
locally validated evidence.
