# Topic C documentation index

Read in this order:

1. [IVC research contract](ivc_research_contract.md) — scientific scope, estimands, evidence roles, hard gates and submission-ready definition.
2. [Confirmatory analysis candidate](../configs/statistics/confirmatory_protocol_candidate_v1.json) — machine-readable hypothesis hierarchy, bootstrap and interpretation rules. Owner freeze is required before transfer metrics are inspected.
3. [Implementation gap register](ivc_implementation_gap_register.md) — what the current pipeline already proves, what it does not prove, and the exact work needed before each manuscript claim.
4. [IVC manuscript blueprint](ivc_manuscript_blueprint.md) — title/positioning, section structure, figure/table plan, result scenarios and red-team checklist.

Source design documents outside this project:

- `../A_Capacity_Stratified_Analysis_of_INT8_and_FP8_Quantization_on_object_detection/docs/topic_c_quantization_corruption_ivc_handoff.md`
- `../AETA2026_paper_1/topic_c_quantization_corruption_vibecode.md`

These files have different roles:

- The source design explains the original idea and broad roadmap.
- The research contract narrows that roadmap into defensible IVC claims.
- The JSON candidate makes key statistical decisions machine-readable.
- The gap register prevents implemented pilot checks from being mistaken for missing confirmatory methods.
- The manuscript blueprint controls writing after results exist.

No document in this folder authorizes overwriting legacy artifacts or launching the full grid/calibration intervention without owner review.
