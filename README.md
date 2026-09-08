# A Paired Evaluation of Clean Accuracy and Corruption Sensitivity in Quantized Object Detection

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22664869.svg)](https://doi.org/10.5281/zenodo.22664869)

Research software and evidence accompanying the manuscript prepared for
*Computer Vision and Image Understanding* (CVIU). Version **2.2.0** aligns this
repository with the revised manuscript in `submission_package/`.

- [Main manuscript](submission_package/01_CVIU_main_revised.pdf)
- [Supplementary Information](submission_package/02_CVIU_supplement_revised.pdf)
- [Archived software, source ZIPs and evidence](https://doi.org/10.5281/zenodo.22664869)
- [Evidence verification instructions and licensing](submission_package/REVIEWER_EVIDENCE_README.md)

## Research question and scope

The study separates clean accuracy, remaining corrupted accuracy, and the
corruption-associated change in the gap between two recorded executable
ModelOpt/TensorRT treatments. It does not propose a new quantizer or identify a
universal causal effect of INT8 versus FP8.

```text
clean gap     = AP_FP8,clean - AP_INT8,clean
corrupted gap = AP_FP8,corrupt - AP_INT8,corrupt
DeltaE        = corrupted gap - clean gap
```

All four arms use the same image universe and common image-bootstrap draws.
A negative interaction denotes gap contraction, not necessarily usable
corrupted accuracy or greater robustness. Absolute AP and clean fidelity must
be inspected alongside the interaction.

The evidence layers are deliberately not pooled as replications of one
population parameter:

- Exploratory grid: four datasets (COCO, VOC, KITTI, TT100K), three YOLO11
  capacities, four corruptions, three severities, 144 direct cells, and JPEG-95
  matched-clean controls.
- Selection-disjoint VOC/KITTI holdouts: six retrained dataset–capacity blocks,
  72 direct cells, and **original-source clean** inputs.
- Controlled clean-input substitution: the same engines and runner compare
  original-source and JPEG-95 clean controls in six holdout blocks; two selected
  TT100K blocks remain separate diagnostics.
- Additional checks: common-draw covariance, aggregation and metric-scale
  sensitivity, training/calibration seeds, three corruption materializations,
  full-grid TT100K fixed-universe bootstrap, and recorded recipe-portability
  stress cases using RT-DETR-L and RetinaNet.

On the six holdout blocks, the mean FP8–INT8 clean gap is +1.60 AP points,
the corrupted gap is +1.05, and the interaction is −0.55. The controlled
clean-input shift averages +0.0574 AP, with an interval crossing zero; this
does not establish codec equivalence. The four-dataset exploratory interaction
is approximately −0.02 AP. These summaries concern different conditional scopes.

## One active manuscript source

| Path | Purpose |
| --- | --- |
| `submission_package/source/` | Current main/Supplement LaTeX, figures and generated tables |
| `submission_package/` | Current PDF previews, ancillary sources and upload instructions |
| `src/` | Training, export, inference, corruption and evaluator programs |
| `analysis/` | Scientific analysis, validation and evidence packaging |
| `configs/`, `manifests/` | Experimental settings and retained provenance |
| `tests/` | Contract and regression tests |
| `paper/` | Historical release sources and evidence; not the current editable manuscript |

The Overleaf and flat-source ZIPs are generated deliverables on Zenodo, not
parallel editable sources. Do not run the historical `paper/build.sh` workflow
to overwrite the revised manuscript.

## Recompute evidence without a GPU

Download `CVIU_Reviewer_Evidence.zip` from the versioned Zenodo record and
extract it into a new directory. With Python 3.11 and NumPy installed:

```bash
export PYTHONDONTWRITEBYTECODE=1
python analysis/build_reviewer_evidence.py --root . --verify --include-v4 --submission-package submission_package
python analysis/build_v4_publication_tables.py --output-dir /tmp/cviu-v4-table-check
```

Keep generated outputs outside the sealed extraction. Verification checks its
exact hash inventory, current-manuscript binding and retained numerical inputs.
Hashes establish internal integrity, not independent execution or timestamps.

For the full prediction-level example, enter
`outputs/analysis/cviu_v4/holdout_example/package/` inside that extraction:

```bash
python -m pip install -r requirements.txt
python reproduce_v4_four_arm.py --manifest manifest.json --out /tmp/cviu-example-report.json --workers 4
python package_v4_holdout_example.py --verify-report /tmp/cviu-example-report.json --expected expected.json
```

Use a fresh output filename. This example evaluates KITTI final/YOLO11m/fog-1
from predictions and annotations, with 1,197 images and 2,000 paired draws,
without AP caches, images, checkpoints, engines or inference. The all-object
interaction is approximately −0.280484 AP with percentile endpoints
[−1.269237, +0.784413]. It tests implementation, not 95% interval coverage.

## Compile the manuscript

Upload `06_Overleaf_Source_CVIU.zip` to Overleaf. Compile `main.tex` with
pdfLaTeX/BibTeX; compile `supplement.tex` separately. Alternatively, from
`submission_package/source/` with TeX Live:

```bash
latexmk -pdf main.tex supplement.tex
```

The flat source ZIP is a separate journal-upload alternative. The large
research-evidence ZIP is not an Overleaf project.

## Tests and experimental reproduction

```bash
PYTHONPATH=src python -m pytest -q
```

The complete development suite needs the recorded scientific dependencies.
Some integration tests require retained artifacts distributed separately.
A checkout alone is not equivalent to the evidence extraction.

Full training/inference reproduction additionally requires licensed datasets,
checkpoints or retraining, calibration inputs, framework-specific dependencies,
and appropriate NVIDIA software/hardware. The historical executions used an
RTX 5090 and TensorRT; rebuilding engines on another stack is not a guarantee
of byte-identical results. Read the frozen registries before running workload
scripts. The lightweight `requirements-remote.txt` is not a complete locked
environment for every experiment.

## Reproducibility and licensing boundaries

Version 2.2.0 provides summary/draw-level evidence and one complete
prediction-level example. It does not reproduce every historical training,
engine build, prediction run or kernel-precision choice. Bootstrap uncertainty
is conditional on the stated artifacts and sampling law.

Original software is [MIT licensed](LICENSE). This does **not** relicense
third-party material. KITTI-derived annotations in the separate evidence
archive remain **CC BY-NC-SA 3.0**, with attribution and transformation details
in [the evidence README](submission_package/REVIEWER_EVIDENCE_README.md).
Elsevier CAS files retain their original notices. No dataset images, trained
checkpoints, engines or credentials are published in this release.

Historical evidence files retain their original bytes and dated statements;
their former “local/unpublished” notes describe their creation-time status.
Current release availability is defined by this README and its version DOI,
not by those historical notes.

## Citation, authors and funding

Use [CITATION.cff](CITATION.cff) and version DOI
[10.5281/zenodo.22664869](https://doi.org/10.5281/zenodo.22664869).
The all-versions concept DOI remains
[10.5281/zenodo.22031663](https://doi.org/10.5281/zenodo.22031663).
Version [2.1.0](https://doi.org/10.5281/zenodo.22275640) is historical and does
not contain the subsequent V4 follow-ups.

The six human authors are Dinh Thuan Nguyen, Lam Phuong Nguyen, Vinh Huy
Nguyen, Sy Vu Quang, Mohan Rajesh Elara and Anh Vu Le. They retain scientific
responsibility. AI-assisted tools are not listed as authors, archive creators,
or Git commit co-authors; use in preparing the work is disclosed in the paper.

This research did not receive any specific grant from funding agencies in the
public, commercial, or not-for-profit sectors.
