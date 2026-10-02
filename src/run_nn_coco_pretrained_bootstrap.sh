#!/usr/bin/env bash
# Pretrained-COCO paired bootstrap: 4-arm caches for retinanet_pretrained and
# fcos_pretrained blocks on the frozen paired-2000 COCO subset.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/thuan/miniconda3/envs/qtsd/bin/python}
OUT=outputs/analysis/nn_coco_pretrained_v1_20260930/bootstrap
N_BOOT=${N_BOOT:-2000}
JOBS=${JOBS:-12}
ARMS='{"fp32":"fp32","fp8-matched512":"fp8-matched512","int8-matched512":"int8-matched512","int8-selective512":"int8-selective512"}'
CONTRASTS='[["fp8-matched512","int8-matched512"],["int8-selective512","int8-matched512"],["fp8-matched512","int8-selective512"],["fp32","int8-selective512"]]'
for model in retinanet_pretrained fcos_pretrained; do
  cache="$OUT/coco__${model}__draws.npz"
  if [ -f "$cache" ]; then echo "SKIP $model"; continue; fi
  echo "BOOTSTRAP coco/$model"
  $PY src/run_nn_paired_bootstrap.py --dataset coco --model "$model" \
    --arms "$ARMS" --contrasts "$CONTRASTS" --n-boot "$N_BOOT" --jobs "$JOBS" --out-dir "$OUT" \
    >>"outputs/logs/coco_pretrained_${model}_bootstrap.log" 2>&1
  echo "DONE $model"
done
echo "NN COCO PRETRAINED BOOTSTRAP COMPLETE"
