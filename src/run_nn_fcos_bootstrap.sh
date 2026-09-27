#!/usr/bin/env bash
# FCOS replication paired bootstrap: 4-arm (fp32/fp8-matched/int8-matched/
# int8-selective) caches for the fcos_r50_fpn block on KITTI and VOC.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/thuan/miniconda3/envs/qtsd/bin/python}
OUT=outputs/analysis/nn_fcos_replication_v1_20260927/bootstrap
N_BOOT=${N_BOOT:-2000}
JOBS=${JOBS:-12}
ARMS='{"fp32":"fp32","fp8-matched512":"fp8-matched512","int8-matched512":"int8-matched512","int8-selective512":"int8-selective512"}'
CONTRASTS='[["fp8-matched512","int8-matched512"],["int8-selective512","int8-matched512"],["fp8-matched512","int8-selective512"],["fp32","int8-selective512"]]'
for dataset in kitti voc; do
  cache="$OUT/${dataset}__fcos_r50_fpn__draws.npz"
  if [ -f "$cache" ]; then echo "SKIP $dataset"; continue; fi
  echo "BOOTSTRAP $dataset/fcos_r50_fpn"
  $PY src/run_nn_paired_bootstrap.py --dataset "$dataset" --model fcos_r50_fpn \
    --arms "$ARMS" --contrasts "$CONTRASTS" --n-boot "$N_BOOT" --jobs "$JOBS" --out-dir "$OUT" \
    >>"outputs/logs/${dataset}_fcos_bootstrap.log" 2>&1
  echo "DONE $dataset"
done
echo "NN FCOS BOOTSTRAP COMPLETE"
