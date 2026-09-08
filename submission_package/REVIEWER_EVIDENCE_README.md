# Reviewer evidence for the revised CVIU manuscript

Active manuscript: **A Paired Evaluation of Clean Accuracy and Corruption Sensitivity in Quantized Object Detection**.

`CVIU_Reviewer_Evidence.zip` is separate from the compact LaTeX/Overleaf source ZIP. It contains the revised manuscript sources and PDFs bound to the retained experimental evidence. It is distributed with **v2.2.0**, DOI [10.5281/zenodo.22664869](https://doi.org/10.5281/zenodo.22664869). The earlier v2.1.0 archive predates these follow-up analyses.

## Verify before running analyses

Extract the evidence ZIP into a new directory. From that directory, with Python 3.11 and NumPy installed:

```bash
export PYTHONDONTWRITEBYTECODE=1
python analysis/build_reviewer_evidence.py --root . --verify --include-v4 --submission-package submission_package
```

This checks the exact SHA-256 inventory, active-publication binding, completed V4 inputs, and numerical regeneration. Keep outputs outside the extraction: subsequent cache/build files are not members of its sealed inventory. Hash consistency establishes internal integrity, not independent execution attestation or timestamp authenticity.

## Recompute the two central V4 tables

From the extracted evidence root:

```bash
python analysis/build_v4_publication_tables.py --output-dir /tmp/cviu-v4-table-check
```

The output tables report all eight controlled clean-input blocks (six primary holdout blocks and two separately selected TT100K diagnostics), and the six holdout four-AP summaries. Clean-control effects and corrupted-gap intervals are computed within common draws; interval endpoints are never added. Historical inputs under `paper/` and the older manuscript directory are provenance records, not alternative editable manuscripts.

## Recompute one full four-arm example from predictions

The directory `outputs/analysis/cviu_v4/holdout_example/package/` contains KITTI final (1,197 images), YOLO11m, fog severity 1, with **original-source clean** inputs. Follow its README in an isolated Python 3.11 environment:

```bash
python -m pip install -r requirements.txt
python reproduce_v4_four_arm.py --manifest manifest.json --out /tmp/cviu-final-example-check.json --workers 4
python package_v4_holdout_example.py --verify-report /tmp/cviu-final-example-check.json --expected expected.json
```

Run these commands from the example directory. The pinned requirements include NumPy 2.4.4 and pycocotools 2.0.11. No GPU, dataset images, checkpoint, engine or AP cache is consumed. The expected all-object interaction is approximately −0.280484 AP, with a 2,000-draw percentile interval of [−1.269237, +0.784413] AP. Four APs and endpoint-specific results are checked against separately retained expected values.

## What can and cannot be reproduced

- Summary/draw-level checks cover the controlled clean-input study, recovered holdout intervals, covariance audit, point-sign diagnostics, full TT100K fixed-universe sensitivity, corrected TIDE summaries, and recipe-portability follow-ups.
- Prediction-level AP/interval recomputation is supplied for the single final-holdout example above. It does not establish frequentist interval coverage.
- The archive does not recreate all detector inference, historical software environments, TensorRT builds or runtime-kernel precision. Where raw artifacts are absent, metadata verification cannot certify their current bytes.
- Software and dataset-derived material have different rights, as detailed below. No images, trained weights, engines or credentials are included.

Do not substitute an old reviewer ZIP after revising the active sources or PDFs. Frozen records and the example's original README retain creation-time local/unpublished wording; those historical statements do not override this versioned release's availability.

## Third-party attribution and licensing

Original project software and documentation are MIT licensed; see the archive's root `LICENSE`. This license does **not** apply to third-party data or template files.

The KITTI Vision Benchmark Suite is a project of Karlsruhe Institute of Technology and Toyota Technological Institute at Chicago. Dataset credit: Andreas Geiger, Philip Lenz, Christoph Stiller and Raquel Urtasun. Reference: Andreas Geiger, Philip Lenz and Raquel Urtasun, *Are we ready for Autonomous Driving? The KITTI Vision Benchmark Suite*, CVPR 2012. Source and copyright notice: https://www.cvlibs.net/datasets/kitti/ .

`outputs/analysis/cviu_v4/holdout_example/package/data/annotations.json` is an adapted, 1,197-image final-partition annotation subset converted from KITTI labels to COCO-style JSON. Conversion and class mapping are documented in the retained manifest, class-map and audit files. No dataset images are included. The KITTI-derived annotations and adaptation are distributed under **Creative Commons Attribution-NonCommercial-ShareAlike 3.0 Unported**, not MIT: https://creativecommons.org/licenses/by-nc-sa/3.0/ (legal code: https://creativecommons.org/licenses/by-nc-sa/3.0/legalcode). Retain attribution, indicate modifications, restrict reuse to noncommercial purposes, and distribute adaptations under the same license. KITTI's creators do not endorse this study. No additional rights, including privacy/publicity rights, are granted by this notice.

Elsevier CAS class/style files retain their original embedded copyright and license notices. Vendored evaluator sources retain their source notices. Metadata integrity does not supersede any third-party terms.
