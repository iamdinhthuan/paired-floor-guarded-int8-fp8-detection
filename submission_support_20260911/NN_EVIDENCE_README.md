# Neural Networks evidence bundle

This compact evidence package accompanies the v3.0.3 manuscript release
([10.5281/zenodo.23082850](https://doi.org/10.5281/zenodo.23082850)).
The active paper is `paper/main_nn.tex` with `paper/supplement.tex`. Older
CVIU-era notes were removed from this directory; the sealed historical package
remains in the v2.2.0 archive (concept DOI 10.5281/zenodo.22031663).

## Rebuild manuscript summaries

From the repository root:

```bash
python3 analysis/nn_final_stats.py
python3 analysis/build_nn_tables.py
cd paper
./verify.sh
```

The paper sources bind generated tables to the retained cell-point ledgers.
Bootstrap intervals use the retained common-image draw caches; post-hoc
max-calibration and KITTI final-holdout results are point estimates without
bootstrap intervals.

## Phase L: KITTI final holdout

`phase_l_holdout_results/` contains 56 per-condition metric records, matching
input-ID and run records, the four engine/ONNX registries and inspector
metadata, the codec-control and calibration manifests, and the aggregate
`cell_points.json`. Its `MANIFEST.sha256` covers every file in that compact
directory except the manifest itself. The records link each AP value to the
input-ID digest, run record, engine registry, prediction digest, and input
manifest marker. The raw prediction payloads, TensorRT engines, ONNX graphs,
and dataset images are not included.

The final partition contains 1,197 images and was excluded from model and
recipe development. This is a post-hoc point-estimate check on a frame-level
split; adjacent frames can cross partitions, so it is not sequence-disjoint
evidence.

## Redistribution boundary

The archive includes analysis code, manifests, metric records, compact
activation summaries, bootstrap draw caches, and hash-bound run metadata.
Raw datasets, trained checkpoints, raw predictions, ONNX model graphs, and
built TensorRT engines are not redistributed. Third-party attribution for the
historical CVIU evidence package is documented in the sealed v2.2.0 archive;
it does not describe the current NN release.
