#!/usr/bin/env bash
# Reduced VOC q95 bootstrap: codec-control arms plus int8-matched512 as a
# pairing anchor. The shared seed namespace gives the same resample schedule as
# the Phase B and Phase C/D caches, so contrasts against those arms are paired;
# nn_contrasts.load_block verifies the anchor's draws are identical.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/thuan/miniconda3/envs/qtsd/bin/python}
OUT=outputs/analysis/nn_q95calib_v1_20260924/bootstrap
ARMS='{"int8-matched512":"int8-matched512","int8-q95calib512":"int8-q95calib512","int8-sel-q95calib512":"int8-sel-q95calib512"}'
CONTRASTS='[["int8-q95calib512","int8-matched512"]]'
$PY src/run_nn_paired_bootstrap.py --dataset voc --model retinanet_r50_fpn_v2 \
  --arms "$ARMS" --contrasts "$CONTRASTS" --n-boot 2000 --jobs "${JOBS:-8}" --out-dir "$OUT" \
  >>outputs/logs/voc_retinanet_q95calib_min_bootstrap.log 2>&1
echo "NN Q95CALIB VOC MIN BOOTSTRAP COMPLETE"
