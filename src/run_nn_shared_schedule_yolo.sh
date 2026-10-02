#!/usr/bin/env bash
# Shared-schedule rerun restricted to the 9 pooled YOLO11 blocks
# (dataset-shared resample schedule enables cross-block covariance).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=/home/thuan/miniconda3/envs/qtsd/bin/python
OUT=outputs/analysis/nn_shared_schedule_v1_20260930/bootstrap
N_BOOT=${N_BOOT:-1000}
JOBS=${JOBS:-12}
NS=nn_shared_schedule_v1_20260930
ARMS='{"fp32":"fp32","fp8":"fp8","int8":"int8-entropy"}'
CONTRASTS='[["fp8","int8"],["fp32","int8"],["fp32","fp8"]]'
for dataset in kitti voc coco; do
  for model in yolo11n yolo11m yolo11x; do
    cache="$OUT/${dataset}__${model}__draws.npz"
    if [ -f "$cache" ]; then echo "SKIP $dataset/$model"; continue; fi
    echo "BOOTSTRAP $dataset/$model (shared schedule)"
    $PY src/run_nn_paired_bootstrap.py --dataset "$dataset" --model "$model" \
      --arms "$ARMS" --contrasts "$CONTRASTS" --n-boot "$N_BOOT" --jobs "$JOBS" \
      --seed-namespace "$NS" --shared-dataset-schedule --out-dir "$OUT" \
      >>"outputs/logs/shared_sched_${dataset}_${model}.log" 2>&1
    echo "DONE $dataset/$model"
  done
done
echo "SHARED-SCHEDULE YOLO RERUN COMPLETE"
