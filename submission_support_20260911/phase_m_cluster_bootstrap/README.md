# Phase M: drive-clustered KITTI bootstrap

Every KITTI image is mapped to its raw drive with the official object devkit
(`src/build_kitti_drive_clusters.py`, pinned hashes of `train_rand.txt` and
`train_mapping.txt`). `run_nn_cluster_bootstrap.sh` (run with `N_BOOT=1000`)
reruns the KITTI YOLO11, RetinaNet, and FCOS blocks, resampling whole drives
with one shared schedule per block. `cell_points.json` holds the plug-in values;
`analysis/nn_final_stats.py::cluster_stats` compares image- and drive-level
intervals for the KITTI contrasts reported in the manuscript (Supplement S9).
