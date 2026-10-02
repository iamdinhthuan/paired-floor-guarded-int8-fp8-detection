# Neural Networks evidence bundle

This compact evidence package accompanies the v3.0.4 manuscript release
([10.5281/zenodo.23097634](https://doi.org/10.5281/zenodo.23097634)).
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

## Added in v3.0.4

- `phase_m_cluster_bootstrap/`: drive-clustered KITTI bootstrap (108 drives, B=1000) for
  YOLO11, RetinaNet, and FCOS; compared with image-level intervals in Supplement S9.
- `phase_n_kitti_holdout_v2/`: RetinaNet retrained on the 4,788-image resplit list and
  evaluated (six arms x 14 conditions) on the 1,197 final images it never saw, plus the
  selection set; image (B=2000) and drive-clustered (B=1000) bootstrap caches.
- `phase_k_maxreg_results/patch_reproduction.json`: `src/patch_maxreg_scales.py`
  reproduces the archived head-restricted max-calibration graphs.
- `phase_l_holdout_results/` is retained as a seen-image consistency check: its 1,197
  images were in the original checkpoint's training partition (95 also in calibration).
