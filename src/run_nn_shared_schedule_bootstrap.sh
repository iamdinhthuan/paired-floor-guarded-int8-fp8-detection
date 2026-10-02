#!/usr/bin/env bash
# Shared-schedule rerun: all blocks within a dataset use the SAME image
# resample schedule (seed keyed on namespace|dataset), enabling cross-block
# covariance estimation between model scales.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/thuan/miniconda3/envs/qtsd/bin/python}
OUT=outputs/analysis/nn_shared_schedule_v1_20260930/bootstrap
N_BOOT=${N_BOOT:-2000}
JOBS=${JOBS:-8}
NS=nn_shared_schedule_v1_20260930
YOLO_ARMS='{"fp32":"fp32","fp8":"fp8","int8":"int8-entropy"}'
YOLO_CONTRASTS='[["fp8","int8"],["fp32","int8"],["fp32","fp8"]]'
RETINA_ARMS='{"fp32":"fp32","fp8-legacy":"fp8","int8-legacy":"int8-entropy","fp8-matched512":"fp8-matched512","int8-matched512":"int8-matched512","int8-selective512":"int8-selective512"}'
RETINA_CONTRASTS='[["fp8-matched512","int8-matched512"],["int8-selective512","int8-matched512"],["fp8-matched512","int8-selective512"]]'
FCOS_ARMS='{"fp32":"fp32","fp8-matched512":"fp8-matched512","int8-matched512":"int8-matched512","int8-selective512":"int8-selective512"}'
FCOS_CONTRASTS='[["fp8-matched512","int8-matched512"],["int8-selective512","int8-matched512"],["fp8-matched512","int8-selective512"]]'
COCO_ARMS='{"fp32":"fp32","fp8-matched512":"fp8-matched512","int8-matched512":"int8-matched512","int8-selective512":"int8-selective512"}'
run_block() {
  local dataset=$1 model=$2 arms=$3 contrasts=$4
  local cache="$OUT/${dataset}__${model}__draws.npz"
  if [ -f "$cache" ]; then echo "SKIP $dataset/$model"; return 0; fi
  echo "BOOTSTRAP $dataset/$model (shared schedule)"
  $PY src/run_nn_paired_bootstrap.py --dataset "$dataset" --model "$model" \
    --arms "$arms" --contrasts "$contrasts" --n-boot "$N_BOOT" --jobs "$JOBS" \
    --seed-namespace "$NS" --shared-dataset-schedule --out-dir "$OUT" \
    >>"outputs/logs/shared_sched_${dataset}_${model}.log" 2>&1
  echo "DONE $dataset/$model"
}
for dataset in kitti voc coco; do
  for model in yolo11n yolo11m yolo11x; do
    run_block "$dataset" "$model" "$YOLO_ARMS" "$YOLO_CONTRASTS"
  done
done
for dataset in kitti voc; do
  run_block "$dataset" "retinanet_r50_fpn_v2" "$RETINA_ARMS" "$RETINA_CONTRASTS"
  run_block "$dataset" "fcos_r50_fpn" "$FCOS_ARMS" "$FCOS_CONTRASTS"
done
run_block coco retinanet_pretrained "$COCO_ARMS" "$FCOS_CONTRASTS"
run_block coco fcos_pretrained "$COCO_ARMS" "$FCOS_CONTRASTS"
echo "SHARED SCHEDULE BOOTSTRAP COMPLETE"
