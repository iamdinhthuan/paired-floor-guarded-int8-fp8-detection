#!/usr/bin/env bash
# Paired bootstrap for the genuine KITTI holdout (nn_kitti_holdout_v2_20261002):
#   * final partition, image-level, B=2000 (same estimator as the frozen layer);
#   * final partition, drive-clustered, B=1000 (devkit drive mapping);
#   * selection partition (retrained checkpoint), image-level, B=2000.
# One shared schedule per block, so every arm contrast is paired.
set -euo pipefail
cd "$(dirname "$0")/.."
PY=${PY:-/home/thuan/miniconda3/envs/qtsd/bin/python}
ATT=nn_kitti_holdout_v2_20261002
OUT=outputs/analysis/$ATT/bootstrap
JOBS=${JOBS:-12}
ARMS='{"fp32":"fp32","fp8-matched512":"fp8-matched512","int8-matched512":"int8-matched512","int8-selective512":"int8-selective512","int8-wonly512":"int8-wonly512","int8-aonly512":"int8-aonly512"}'
CONTRASTS='[["fp8-matched512","int8-matched512"],["int8-selective512","int8-matched512"],["fp8-matched512","int8-selective512"],["int8-wonly512","int8-matched512"],["int8-aonly512","int8-matched512"],["fp32","int8-selective512"]]'
mkdir -p outputs/logs

run() {
  local tag=$1 prefix=$2 ann=$3 nboot=$4 extra=${5:-}
  local out="$OUT/$tag"
  if [ -f "$out/kitti__retinanet_r50_fpn_v2__draws.npz" ]; then echo "SKIP $tag"; return 0; fi
  echo "BOOTSTRAP $tag"
  $PY src/run_nn_paired_bootstrap.py --dataset kitti --model retinanet_r50_fpn_v2 \
    --arms "$ARMS" --contrasts "$CONTRASTS" --n-boot "$nboot" --jobs "$JOBS" \
    --attempts "$ATT" --prefix "$prefix" --annotations "$ann" \
    --seed-namespace "${ATT}_${tag}" --out-dir "$out" $extra \
    >>"outputs/logs/${ATT}_${tag}_bootstrap.log" 2>&1
  echo "DONE $tag"
}

FINAL_ANN=manifests/annotations/kitti_confirmatory_final_v1_coco.json
VAL_ANN=manifests/annotations/kitti_val_ultralytics_v1_coco.json
run final_image kitti_final "$FINAL_ANN" 2000
run final_drive kitti_final "$FINAL_ANN" 1000 "--cluster-manifest manifests/clusters/kitti_confirmatory_final_v1_drive_clusters.json"
run val_image kitti_val "$VAL_ANN" 2000
echo "KITTI HOLDOUT V2 BOOTSTRAP COMPLETE"
