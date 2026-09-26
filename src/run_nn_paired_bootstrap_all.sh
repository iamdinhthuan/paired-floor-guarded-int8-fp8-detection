#!/usr/bin/env bash
# Phase-B paired bootstrap driver: one npz draw cache + JSON summary per block.
# Idempotent: skips blocks whose draw cache already exists (fail-closed).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/thuan/miniconda3/envs/qtsd/bin/python}
OUT=outputs/analysis/nn_paired_protocol_v1_20260924/bootstrap
N_BOOT=${N_BOOT:-2000}
JOBS=${JOBS:-12}

run_block() {
  local dataset=$1 model=$2 arms=$3 contrasts=$4
  local cache="$OUT/${dataset}__${model}__draws.npz"
  if [ -f "$cache" ]; then
    echo "SKIP $dataset/$model (draw cache exists)"
    return 0
  fi
  echo "BOOTSTRAP $dataset/$model"
  $PY src/run_nn_paired_bootstrap.py \
    --dataset "$dataset" --model "$model" \
    --arms "$arms" --contrasts "$contrasts" \
    --n-boot "$N_BOOT" --jobs "$JOBS" --out-dir "$OUT" \
    >>"outputs/logs/${dataset}_${model}_bootstrap.log" 2>&1
  echo "DONE $dataset/$model"
}

YOLO_ARMS='{"fp32":"fp32","fp8":"fp8","int8":"int8-entropy"}'
YOLO_CONTRASTS='[["fp8","int8"],["fp32","int8"],["fp32","fp8"]]'
RETINA_ARMS='{"fp32":"fp32","fp8-legacy":"fp8","int8-legacy":"int8-entropy","fp8-matched512":"fp8-matched512","int8-matched512":"int8-matched512","int8-selective512":"int8-selective512"}'
RETINA_CONTRASTS='[["fp8-matched512","int8-matched512"],["int8-selective512","int8-matched512"],["fp32","int8-selective512"],["fp8-matched512","int8-selective512"],["fp8-legacy","int8-legacy"],["fp32","int8-legacy"]]'

for dataset in kitti voc coco; do
  for model in yolo11n yolo11m yolo11x; do
    run_block "$dataset" "$model" "$YOLO_ARMS" "$YOLO_CONTRASTS"
  done
done
for dataset in kitti voc; do
  run_block "$dataset" "retinanet_r50_fpn_v2" "$RETINA_ARMS" "$RETINA_CONTRASTS"
done
echo "NN PAIRED BOOTSTRAP COMPLETE"
