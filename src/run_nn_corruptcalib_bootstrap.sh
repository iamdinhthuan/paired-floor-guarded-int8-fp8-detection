#!/usr/bin/env bash
# Phase C/D paired bootstrap: RetinaNet blocks with all Phase-B arms plus
# the two corruptcalib intervention arms. Same seed namespace => identical
# image-resample schedule as Phase B, fully paired across arms.
# Per-cell draws in the npz enable held-out vs in-family split means.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/thuan/miniconda3/envs/qtsd/bin/python}
OUT=outputs/analysis/nn_corruptcalib_v1_20260924/bootstrap
N_BOOT=${N_BOOT:-2000}
JOBS=${JOBS:-12}

ARMS='{"fp32":"fp32","fp8-legacy":"fp8","int8-legacy":"int8-entropy","fp8-matched512":"fp8-matched512","int8-matched512":"int8-matched512","int8-selective512":"int8-selective512","int8-corruptcalib512":"int8-corruptcalib512","int8-sel-corruptcalib512":"int8-sel-corruptcalib512"}'
CONTRASTS='[["int8-corruptcalib512","int8-matched512"],["int8-sel-corruptcalib512","int8-selective512"],["int8-sel-corruptcalib512","fp8-matched512"],["int8-corruptcalib512","fp8-matched512"],["fp8-matched512","int8-matched512"],["int8-selective512","int8-matched512"],["fp32","int8-corruptcalib512"]]'

for dataset in kitti voc; do
  cache="$OUT/${dataset}__retinanet_r50_fpn_v2__draws.npz"
  if [ -f "$cache" ]; then
    echo "SKIP $dataset/retinanet (draw cache exists)"
    continue
  fi
  echo "BOOTSTRAP $dataset/retinanet"
  $PY src/run_nn_paired_bootstrap.py \
    --dataset "$dataset" --model "retinanet_r50_fpn_v2" \
    --arms "$ARMS" --contrasts "$CONTRASTS" \
    --n-boot "$N_BOOT" --jobs "$JOBS" --out-dir "$OUT" \
    >>"outputs/logs/${dataset}_retinanet_corruptcalib_bootstrap.log" 2>&1
  echo "DONE $dataset/retinanet"
done
echo "NN CORRUPTCALIB BOOTSTRAP COMPLETE"
