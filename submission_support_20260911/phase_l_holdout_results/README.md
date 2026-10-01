# Phase L: KITTI final-holdout check

This post-hoc evaluation applies the four RetinaNet regression-head factorial
arms to the untouched 1,197-image KITTI final partition. The images were
excluded from model and recipe development. The check uses the same 14
conditions and existing engines, reports full-sample point estimates only,
and has no bootstrap intervals. The split is frame-level, not sequence-level;
adjacent frames may cross partitions.

`cell_points.json` contains the aggregate AP points used by
`analysis/nn_final_stats.py`. The 56 files in `metrics/`, `inputs/`, and
`runs/` bind every arm-condition result to its image-ID digest, input-manifest
digest, run record, engine registry, and prediction digest. Raw prediction
files are intentionally not included. `engines/` retains the four engine and
ONNX registries plus inspector metadata, but not TensorRT engines or ONNX
graphs. Calibration, annotation, codec-control, and corruption-manifest
records are retained locally or referenced by their repository paths.

`MANIFEST.sha256` inventories every file in this compact evidence directory
except itself.
