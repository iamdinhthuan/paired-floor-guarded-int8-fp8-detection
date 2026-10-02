# Phase N: KITTI holdout with a retrained checkpoint

The original KITTI checkpoints were trained on the full 5,985-image Ultralytics
training partition, which contains the 1,197 "final" images of the later
`kitti_confirmatory_v1` resplit (Phase L is therefore a seen-image check).
Phase N retrains RetinaNet-R50-FPN-v2 with the frozen profile on the
4,788-image resplit training list (`manifests/training/`), selects it on the
1,496-image selection set (epoch 12), rebuilds six arms (fp32, fp8-matched512,
int8-matched512, int8-selective512, int8-wonly512, int8-aonly512) with
calibration list `kitti_confirmatory_train_512_s20260818_v1` (no overlap with
the final partition), and evaluates them on the final partition and the
selection set under the 14-condition schedule.

- `metrics/`, `inputs/`, `runs/`: 168 hash-bound records (6 arms x 14 conditions x 2 partitions).
- `engines/`, `manifests/onnx/`: ONNX and TensorRT engine registries (no engines or graphs).
- `final_image/`, `final_drive/`, `val_image/`: paired bootstrap caches and plug-in cell points
  (image bootstrap B=2000; drive-clustered B=1000 with `kitti_confirmatory_final_v1_drive_clusters.json`).
- Queue: `src/run_nn_kitti_holdout_v2_queue.sh`; bootstrap: `src/run_nn_kitti_holdout_v2_bootstrap.sh`.

The final images share driving sequences with the training list: the holdout
is image-disjoint, not sequence-disjoint.
