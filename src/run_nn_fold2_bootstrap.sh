#!/usr/bin/env bash
# Phase F paired bootstrap: fold-2 intervention arms plus int8-matched512 as a
# pairing anchor. Same seed namespace => identical resample schedule as every
# other RetinaNet block, so contrasts against arms cached in other draw files
# are fully paired (verified by the anchor arm's identical draws).
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/thuan/miniconda3/envs/qtsd/bin/python}
OUT=outputs/analysis/nn_corruptcalib_fold2_v1_20260925/bootstrap
N_BOOT=${N_BOOT:-2000}
JOBS=${JOBS:-12}
ARMS='{"int8-matched512":"int8-matched512","int8-cc2calib512":"int8-cc2calib512","int8-sel-cc2calib512":"int8-sel-cc2calib512"}'
CONTRASTS='[["int8-cc2calib512","int8-matched512"]]'
for dataset in kitti voc; do
  cache="$OUT/${dataset}__retinanet_r50_fpn_v2__draws.npz"
  if [ -f "$cache" ]; then echo "SKIP $dataset"; continue; fi
  echo "BOOTSTRAP $dataset/retinanet fold2"
  $PY src/run_nn_paired_bootstrap.py --dataset "$dataset" --model retinanet_r50_fpn_v2 \
    --arms "$ARMS" --contrasts "$CONTRASTS" --n-boot "$N_BOOT" --jobs "$JOBS" --out-dir "$OUT" \
    >>"outputs/logs/${dataset}_retinanet_fold2_bootstrap.log" 2>&1
  echo "DONE $dataset"
done
echo "NN FOLD2 BOOTSTRAP COMPLETE"
