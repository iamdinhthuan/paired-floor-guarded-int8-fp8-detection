# Quantization fragility under image corruption is recipe-dependent

[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.23097634.svg)](https://doi.org/10.5281/zenodo.23097634)

Research software and compact evidence accompanying the *Neural Networks*
manuscript. The active manuscript sources live in `paper/` (`main_nn.tex`,
`supplement.tex`); `submission_package/` is the versioned upload bundle, not
the editable source.

- [Active manuscript README and build](paper/README.md)
- [Current Neural Networks submission files](submission_package/)
- [Current v3.0.4 reproducibility release](https://doi.org/10.5281/zenodo.23097634)
- [Historical CVIU-era v2.2.0 archive](https://doi.org/10.5281/zenodo.22664869)
- [Current evidence inventory](submission_support_20260911/NN_EVIDENCE_README.md)

## Current Neural Networks study

The study evaluates executable INT8 and FP8 detectors under paired image
corruptions. Across nine YOLO11 dataset-by-scale blocks, the corruption
interaction changes sign even though FP8 has higher absolute AP in every
matched block mean. In the tested RetinaNet recipe, the severe INT8 level
deficit localizes to regression-head activation quantization; FCOS shows no
comparable head-specific deficit and substantially lighter fine-level
activation tails. The head localization and the FCOS null recur on
off-the-shelf COCO-pretrained checkpoints, and on a KITTI holdout scored with
a RetinaNet checkpoint retrained without those images (`phase_n_kitti_holdout_v2`).
The earlier re-evaluation of the 1,197 KITTI "final" images with the original
checkpoint (`phase_l_holdout_results`) is NOT a holdout, because those images
were in its training partition. KITTI intervals are also checked with a
drive-clustered bootstrap (`phase_m_cluster_bootstrap`). Corruption-aware calibration transfers inconsistently across
folds and is not a reliable repair.

## Estimand

The study separates clean accuracy, remaining corrupted accuracy, and the
corruption-associated change in the gap between two recorded executable
ModelOpt/TensorRT treatments. It does not propose a new quantizer or identify a
universal causal effect of INT8 versus FP8.

```text
clean gap     = AP_FP8,clean - AP_INT8,clean
corrupted gap = AP_FP8,corrupt - AP_INT8,corrupt
DeltaE        = corrupted gap - clean gap
```

All arms in a dataset-by-model block use the same image universe and common
image-bootstrap draws. A negative interaction denotes gap contraction, not
necessarily usable corrupted accuracy or greater robustness. Absolute AP and
clean fidelity are inspected alongside the interaction.

## One active manuscript source

| Path | Purpose |
| --- | --- |
| `paper/` | Active Neural Networks manuscript (`main_nn.tex`, `supplement.tex`), generated tables/figures, build and verify scripts, upload documents |
| `submission_package/` | Current Neural Networks upload bundle; rebuild from `paper/` for each release |
| `src/` | Training, export, inference, corruption and evaluator programs |
| `analysis/` | Scientific analysis, validation and evidence packaging |
| `configs/`, `manifests/` | Experimental settings and retained provenance |
| `outputs/metrics/`, `outputs/nn_*/` | Retained metric records and audit JSONs referenced by run manifests |
| `tests/` | Contract and regression tests |

The Overleaf and flat-source ZIPs under `paper/` are generated deliverables,
not parallel editable sources. Build the manuscript with `paper/build.sh`
(pdfLaTeX; see `paper/README.md`).

## Historical CVIU-era archive

An earlier CVIU-targeted revision of this project (v2.2.0, four-dataset
exploratory plus selection-disjoint holdouts) is preserved unchanged at
[10.5281/zenodo.22664869](https://doi.org/10.5281/zenodo.22664869), including
its own sealed evidence package and verification scripts. It is historical
context only; the current repository and the v3.0.4 release are the
Neural Networks submission package.

## Compile the manuscript

From `paper/`, `./build.sh` compiles `main_nn.tex` and `supplement.tex`
independently with pdfLaTeX/BibTeX and `./verify.sh` regenerates
`SOURCE_MANIFEST.sha256` and checks the audit chain. For the current Neural
Networks upload files, see `submission_package/`; the packaged Overleaf-ready
source is `06_LaTeX_source_NN.zip` there (upload it to Overleaf and compile
`main_nn.tex`; compile `supplement.tex` separately).

The flat source ZIP is a separate journal-upload alternative. The large
research-evidence ZIP is not an Overleaf project. `submission_package/` holds
the current Neural Networks upload files; it is regenerated from the active
sources for each immutable release. The historical CVIU source/evidence
archive remains at [v2.2.0](https://doi.org/10.5281/zenodo.22664869).

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

The Neural Networks v3.0.4 archive provides summary/draw-level compact
evidence and deterministic regeneration of the reported tables; it does not
redistribute datasets, trained checkpoints, TensorRT engines, raw predictions,
or ONNX graphs. Bootstrap uncertainty is conditional on the stated artifacts
and sampling law.

Original software is [MIT licensed](LICENSE). This does **not** relicense
third-party material. KITTI-derived annotations referenced by the evidence
ledgers remain **CC BY-NC-SA 3.0** under their original terms. Elsevier CAS
files retain their original notices. No dataset images, trained checkpoints,
engines or credentials are published in this release.

## Citation, authors and funding

Use [CITATION.cff](CITATION.cff) and version DOI
[10.5281/zenodo.23097634](https://doi.org/10.5281/zenodo.23097634) (v3.0.4, Neural Networks submission); v3.0.3 ([10.5281/zenodo.23082850](https://doi.org/10.5281/zenodo.23082850)) is superseded because it described the KITTI "final" partition as an untouched holdout.
The CVIU-era v2.2.0 is [10.5281/zenodo.22664869](https://doi.org/10.5281/zenodo.22664869).
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
