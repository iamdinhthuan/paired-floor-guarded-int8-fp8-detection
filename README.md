# Topic C — Quantization × Corruption × Object Size

This is the isolated project for the IVC Topic C experiment. Topic C outputs belong under `/home/thuan/topic_c_ivc`; the legacy COCO and TT100K roots are immutable inputs and must never be overwritten.

Run only in the approved RTX 5090 environment:

```bash
conda activate qtsd
python src/capture_provenance.py --out manifests/provenance/initial.json --asset /home/thuan/coco_journal/exports/coco_pilot/yolo11m_fp32.plan --asset /home/thuan/coco_journal/exports/coco_pilot/yolo11m_fp16.plan --asset /home/thuan/coco_journal/data/coco/calib/calib_list.json --asset /home/thuan/coco_journal/data/coco_src/annotations/instances_val2017.json
```

## Authorized reduced scope

The owner has approved the reduced four-dataset pipeline only:

- COCO val2017, VOC validation, KITTI validation and TT100K test;
- YOLO11n/m/x;
- FP32, INT8-entropy and FP8 scientific comparisons;
- FP16 as a hard parity gate;
- clean plus Gaussian noise, motion blur, fog and JPEG at severities 1/3/5;
- 117 scientific inference conditions and 72 paired bootstrap cells per dataset.

The ten-corruption full grid, calibration intervention and additional architecture remain separate decisions. Do not launch them from this README.

Safety gates: (1) record environment and hashes, (2) require clean FP32/FP16 parity with absolute AP difference <= 0.01 for every interpreted model/dataset path, (3) validate every corruption manifest before inference, (4) preserve raw predictions and linked hashes, and (5) reject missing/partial/mismatched artifacts. Never store credentials in this project.

## IVC paper design

The implementation is governed by two research-facing documents:

- [IVC research contract](docs/ivc_research_contract.md): contribution boundary, discovery/confirmation split, estimands, hard gates, statistical hierarchy and submission-ready definition.
- [Implementation gap register](docs/ivc_implementation_gap_register.md): implemented pilot safeguards versus missing confirmatory/statistical evidence.
- [IVC manuscript blueprint](docs/ivc_manuscript_blueprint.md): paper positioning, section plan, figures/tables, result-scenario language and reviewer-risk checklist.
- [Submission-strengthening notes](docs/ivc_submission_strengthening_notes.md):
  final claim hierarchy, evidence now exposed in the full manuscript,
  reviewer-facing strengths, forbidden overclaims, and prioritized follow-up
  experiments.

The compiled manuscript is [paper/main.pdf](paper/main.pdf); its LaTeX source,
generated ledgers, ten vector figures, and interactive framework companion are
under `paper/`. The complete local numerical/archive bundle and its verification
reports are under `artifacts/four_dataset_pilot_v1/`.

The proposed machine-readable confirmatory decisions are in [confirmatory_protocol_candidate_v1.json](configs/statistics/confirmatory_protocol_candidate_v1.json). It remains a candidate follow-up protocol and must not be back-applied to the completed results. The current four-dataset, B=500 cell-wise bootstrap is exploratory evidence and must not be relabelled confirmatory.
