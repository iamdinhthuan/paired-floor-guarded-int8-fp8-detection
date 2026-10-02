#!/usr/bin/env bash
# Drive-clustered KITTI bootstrap (post-hoc sensitivity): resamples whole raw
# KITTI drives (official devkit mapping; manifests/clusters/) instead of
# images, with one shared schedule per block so every arm contrast stays
# paired. Same cells and estimands as the image-level caches; only the
# resampling unit changes. Fail-closed: refuses to overwrite draw caches.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/thuan/miniconda3/envs/qtsd/bin/python}
OUT=outputs/analysis/nn_cluster_bootstrap_v1_20261002/bootstrap
N_BOOT=${N_BOOT:-2000}
JOBS=${JOBS:-12}
NS=nn_cluster_bootstrap_v1_20261002
CLUSTERS=manifests/clusters/kitti_val_ultralytics_v1_drive_clusters.json
mkdir -p outputs/logs

run_block() {
  local model=$1 arms=$2 contrasts=$3
  if [ -f "$OUT/kitti__${model}__draws.npz" ]; then echo "SKIP kitti/$model"; return 0; fi
  echo "CLUSTER BOOTSTRAP kitti/$model"
  $PY src/run_nn_paired_bootstrap.py --dataset kitti --model "$model" \
    --arms "$arms" --contrasts "$contrasts" --n-boot "$N_BOOT" --jobs "$JOBS" \
    --seed-namespace "$NS" --cluster-manifest "$CLUSTERS" --out-dir "$OUT" \
    >>"outputs/logs/kitti_${model}_cluster_bootstrap.log" 2>&1
  echo "DONE kitti/$model"
}

YOLO_ARMS='{"fp32":"fp32","fp8":"fp8","int8":"int8-entropy"}'
YOLO_CONTRASTS='[["fp8","int8"]]'
RETINA_ARMS='{"fp32":"fp32","fp8-matched512":"fp8-matched512","int8-matched512":"int8-matched512","int8-selective512":"int8-selective512","int8-wonly512":"int8-wonly512","int8-aonly512":"int8-aonly512","int8-corruptcalib512":"int8-corruptcalib512","int8-sel-corruptcalib512":"int8-sel-corruptcalib512","int8-q95calib512":"int8-q95calib512","int8-sel-q95calib512":"int8-sel-q95calib512","int8-cc2calib512":"int8-cc2calib512","int8-sel-cc2calib512":"int8-sel-cc2calib512"}'
RETINA_CONTRASTS='[["fp8-matched512","int8-matched512"],["int8-selective512","int8-matched512"],["fp8-matched512","int8-selective512"]]'
FCOS_ARMS='{"fp32":"fp32","fp8-matched512":"fp8-matched512","int8-matched512":"int8-matched512","int8-selective512":"int8-selective512"}'
FCOS_CONTRASTS='[["fp8-matched512","int8-matched512"],["int8-selective512","int8-matched512"]]'

for model in yolo11n yolo11m yolo11x; do run_block "$model" "$YOLO_ARMS" "$YOLO_CONTRASTS"; done
run_block retinanet_r50_fpn_v2 "$RETINA_ARMS" "$RETINA_CONTRASTS"
run_block fcos_r50_fpn "$FCOS_ARMS" "$FCOS_CONTRASTS"
echo "NN CLUSTER BOOTSTRAP COMPLETE"
