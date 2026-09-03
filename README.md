# A Paired, Floor-Guarded Evaluation Protocol for INT8 and FP8 Object Detectors under Image Corruptions

[![Manuscript](https://img.shields.io/badge/manuscript-PDF-b31b1b.svg)](paper/preview/main.pdf)
[![Supplement](https://img.shields.io/badge/supplement-PDF-4c6ef5.svg)](paper/preview/supplement.pdf)
[![Reproducibility](https://img.shields.io/badge/package-verified-2f9e44.svg)](paper/README.md)
[![DOI](https://zenodo.org/badge/DOI/10.5281/zenodo.22275640.svg)](https://doi.org/10.5281/zenodo.22275640)

This repository accompanies the manuscript:

> **A Paired, Floor-Guarded Evaluation Protocol for INT8 and FP8 Object
> Detectors under Image Corruptions**

The manuscript is prepared for *Computer Vision and Image Understanding*
(CVIU). Release `v2.1.0` aligns the software, compact evidence, manuscript
sources, and publication metadata with that submission.

The study evaluates recorded post-training-quantization treatments through
executable TensorRT object detectors. Its central measurement is a paired
four-cell interaction contrast that asks whether corruption changes the
FP8–INT8 accuracy gap after accounting for the matched-clean gap. The protocol
uses identical encoded image bytes and common image-resampling draws across the
four clean/corrupted and INT8/FP8 arms.

The repository is intended as research software and a compact audit package.
It does not introduce a new quantization algorithm, and its results should not
be interpreted as a universal ranking of numerical formats.

## Scientific scope

The primary exploratory evaluation covers:

- COCO, Pascal VOC, KITTI, and TT100K;
- YOLO11n, YOLO11m, and YOLO11x;
- TensorRT FP32 reference, INT8-entropy, and FP8 treatments;
- Gaussian noise, motion blur, fog, and JPEG corruption at three severities;
- deterministic JPEG-95 matched-clean controls; and
- 2,000 common image-bootstrap draws for paired interaction analysis.

Additional evidence layers examine untouched VOC/KITTI holdouts, corruption
realizations, training and calibration seeds, a fixed-universe TT100K stress
cell, engine-only runtime, and portability stress cases using RT-DETR-L and
RetinaNet-R50-FPN-v2. These layers have different image universes and
replication axes; they are conditionality checks rather than repeated estimates
of one population parameter.

The direct contrast is

```text
DeltaE = (AP_FP8 - AP_INT8)_corrupt
       - (AP_FP8 - AP_INT8)_matched-clean
```

Positive `DeltaE` means that the FP8-minus-INT8 gap widens under corruption;
negative `DeltaE` means that it contracts. Neither sign alone establishes
absolute corruption robustness. Matched-clean fidelity and absolute corrupted
AP must be inspected before interpreting the interaction.

## Repository layout

| Path | Contents |
|---|---|
| [`paper/`](paper/) | LaTeX sources, verified PDF previews, figures, compact evidence, and submission-package checks |
| [`src/`](src/) | Data preparation, corruption generation, training, export, quantization, inference, evaluation, and bootstrap programs |
| [`configs/`](configs/) | Frozen experiment and analysis configurations |
| [`manifests/`](manifests/) | Ordered image universes, split registries, calibration lists, and provenance records |
| [`analysis/`](analysis/) | Evidence aggregation and manuscript-artifact utilities |
| [`tests/`](tests/) | Unit and integration tests for the experiment contracts |
| [`docs/`](docs/) | Estimand definitions, execution notes, and research-design documentation |

Large datasets, checkpoints, TensorRT engines, raw predictions, and local
runtime outputs are intentionally excluded from Git. Their omission reflects
dataset licensing, artifact size, and hardware-specific engine portability.

## Environment

The recorded experiments used Linux, an NVIDIA RTX 5090, TensorRT, CUDA, and a
Conda environment named `qtsd`. Exact package and framework identities are
retained in the provenance manifests and run records. A lightweight dependency
list for remote experiment utilities is provided in
[`requirements-remote.txt`](requirements-remote.txt).

```bash
conda create -n qtsd python=3.10
conda activate qtsd
python -m pip install -r requirements-remote.txt
```

TensorRT, CUDA, PyTorch, ModelOpt, and detector-specific dependencies must be
installed for the target NVIDIA platform. TensorRT engines are not portable
across arbitrary software and GPU configurations; rebuild them from the frozen
registries on the intended host.

## Reproducing the manuscript package

The self-contained paper package can be checked without retraining models or
rerunning inference. It requires Python 3 and a TeX Live installation providing
`latexmk`, or `pdflatex` and `bibtex` as a fallback.

```bash
cd paper
./verify.sh
```

This command rebuilds the main and supplementary PDFs, validates the active
CVIU source and upload assets, and writes and verifies the source manifest. It
does not silently regenerate results from unavailable external artifacts. See
[`paper/README.md`](paper/README.md) for the package contents and precise
reproducibility boundary.

To create the clean Overleaf and flat Editorial Manager source bundles, run
the two CVIU submission builders documented in `paper/README.md`.

For a PDF-only rebuild:

```bash
cd paper
./build.sh
```

Verified previews are available as the
[`main manuscript`](paper/preview/main.pdf) and
[`Supplementary Information`](paper/preview/supplement.pdf).

## Running tests

The repository contains tests for the experiment registries, paired estimands,
bootstrap schedules, inference/evaluation contracts, and evidence builders.

For example, the self-contained estimand tests can be run with:

```bash
PYTHONPATH=src python -m pytest -q tests/test_bootstrap_estimand.py
```

The complete development suite is `PYTHONPATH=src python -m pytest -q`, but
some integration gates require frozen external artifacts that are not stored in
Git. A missing external completion report is therefore not equivalent to a
failed manuscript-package check. The self-contained `paper/verify.sh` command
is the authoritative release gate for the deposited paper artifact.

## Full experimental reproduction

Full reproduction is deliberately staged rather than exposed as an unsafe
single command. Before running training or inference:

1. acquire each dataset from its official source and preserve its license;
2. resolve the paths referenced by the frozen dataset and calibration
   manifests;
3. record the software environment, hardware identity, and input hashes;
4. train or restore the specified checkpoints;
5. export the reference ONNX models and pass the reference-parity gates;
6. build the recorded INT8 and FP8 Q/DQ graphs and TensorRT engines;
7. materialize and validate corruption manifests before inference; and
8. evaluate the four paired arms with a shared image universe and bootstrap
   schedule.

The current publication scope and reproduction boundary are documented in
[`paper/README.md`](paper/README.md),
[`docs/paired_excess_gap_method.md`](docs/paired_excess_gap_method.md), and the
experiment-specific records under [`docs/`](docs/). Dated IVC planning files
are retained only as historical provenance; the CVIU source and release gates
under `paper/` are authoritative for version 2.1.0.

Do not silently substitute datasets, calibration lists, checkpoints, decoders,
or TensorRT builds while retaining the original result labels. The validators
are designed to fail closed when required identities or hashes are missing.

## Reproducibility boundary

The public repository reproduces the manuscript-level tables, figures, and
internal consistency checks from compact frozen ledgers. It does not currently
redistribute the raw datasets, trained checkpoints, hardware-specific engines,
or all per-image prediction payloads. Consequently, it supports transparent
inspection and deterministic regeneration of the deposited summaries, but it
does not yet constitute independent end-to-end reproduction of every training
and inference result.

All conclusions are conditional on the recorded datasets, partitions,
checkpoints, calibration lists, quantization recipes, corruption realizations,
decoder settings, TensorRT build, and hardware. Image-bootstrap intervals do
not include all of these sources of uncertainty.

## Citation

Citation metadata are provided in [`CITATION.cff`](CITATION.cff). GitHub can
render the corresponding citation through **Cite this repository**. Cite
release `v2.1.0` using its version DOI
[10.5281/zenodo.22275640](https://doi.org/10.5281/zenodo.22275640). The
all-versions concept DOI is
[10.5281/zenodo.22031663](https://doi.org/10.5281/zenodo.22031663).

## Authors and research responsibility

The six human authors are Dinh Thuan Nguyen, Lam Phuong Nguyen, Vinh Huy
Nguyen, Sy Vu Quang, Mohan Rajesh Elara, and Anh Vu Le. They retain full
responsibility for the study, manuscript, code, evidence, and release
metadata. AI-assisted tools are neither authors nor contributors and are not
listed in `CITATION.cff` or `.zenodo.json`; any use of such tools in manuscript
preparation is disclosed separately in the paper in accordance with journal
policy.

## Funding

This research did not receive any specific grant from funding agencies in the
public, commercial, or not-for-profit sectors.

## Data, models, and third-party software

Users must obtain COCO, Pascal VOC, KITTI, and TT100K from their official
distribution channels and comply with the applicable licenses and terms.
Elsevier CAS template files included with the submission source retain their
original notices; see [`CAS_TEMPLATE_NOTICE.txt`](CAS_TEMPLATE_NOTICE.txt) and
[`paper/CAS_TEMPLATE_NOTICE.txt`](paper/CAS_TEMPLATE_NOTICE.txt).

## License and contact

Original software and documentation in this repository are released under the
[`MIT License`](LICENSE). Dataset assets, Elsevier CAS files, and other
third-party components remain under their respective licenses and notices.

For questions about the study or archived evidence, use the corresponding
author details in the manuscript or open a GitHub issue.
