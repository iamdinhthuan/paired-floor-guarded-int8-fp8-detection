# Documentation index

## Authoritative CVIU release documents

For version `v2.1.0`, use these sources in this order:

1. [Paper package guide](../paper/README.md) — canonical CVIU sources,
   reproducibility boundary, build, validation, and upload-package commands.
2. [Paired interaction method](paired_excess_gap_method.md) — estimand,
   sign convention, pairing, and interpretation guardrails.
3. [Main manuscript](../paper/main.tex) and
   [Supplementary File S1](../paper/supplement.tex) — authoritative scientific
   claims and evidence scope.
4. [Final author checklist](../paper/AUTHOR_CHECKLIST_CVIU.txt) — the remaining
   Zenodo DOI and Editorial Manager checks.

Frozen configurations, manifests, execution reports, and dated plans document
how the recorded experiments were produced. They do not override the active
CVIU manuscript or its release validator.

## Historical IVC material

Files whose names contain `ivc`, and dated documents under `plans/` or
`superpowers/`, are retained solely as provenance for the earlier
*Image and Vision Computing* development phase. Some include machine-specific
paths, host gates, provisional citation constraints, or obsolete submission
instructions. Do not use them to build or submit the CVIU revision.

The authoritative release gate is:

```bash
cd paper
./verify.sh
```

This command validates the compact paper package; it does not retrain models,
rebuild TensorRT engines, or rerun detector inference.
