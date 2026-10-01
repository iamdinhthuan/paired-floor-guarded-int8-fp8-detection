#!/usr/bin/env bash
# KITTI final-holdout evaluation queue: 4 RetinaNet arms x 14 conditions on the
# untouched 1,197-image final holdout (confirmatory split, seed 20260818).
# Post-hoc: point estimates only. GPU-gated; idempotent; fails closed on
# partial artifacts.
set -euo pipefail
root=${1:-/home/thuan/topic_c_ivc}
cd "$root"
source /home/thuan/miniconda3/etc/profile.d/conda.sh
conda activate qtsd
export PYTHONPATH=src
PY=/home/thuan/miniconda3/envs/qtsd/bin/python
attempt=nn_holdout_v1_20261001
exec 9>"outputs/logs/holdout_queue.lock"
flock -n 9 || { echo "holdout queue already active"; exit 0; }
mkdir -p "outputs/predictions/$attempt" "outputs/inputs/$attempt" \
  "outputs/metrics/$attempt" "manifests/runs/$attempt" "outputs/logs/$attempt"

wait_gpu() {
  local need=$1
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [[ "$free" -ge "$need" ]]; then return 0; fi
    sleep 60
  done
}

ANN=manifests/annotations/kitti_confirmatory_final_v1_coco.json
CLEAN_ROOT=data/datasets/kitti
CORR_ROOT=data/confirmatory_corruptions/voc_kitti_confirmatory_v1/kitti
CORR_MANIFEST_DIR=manifests/images/voc_kitti_confirmatory_v1
CODEC_MANIFEST=manifests/images/kitti_final_codec_control_q95_p0_v1.json
CODEC_CACHE=data/codec_control/kitti_final

# ---------- PART 1: holdout codec-control (JPEG q95) manifest ----------
if [[ ! -f "${CODEC_MANIFEST}.complete" ]]; then
  $PY src/materialize_codec_control.py \
    --dataset kitti --split test \
    --annotations "$ANN" \
    --clean-root "$CLEAN_ROOT" \
    --cache-root "$CODEC_CACHE" \
    --manifest-out "$CODEC_MANIFEST" \
    --quality 95 --subsampling 0 \
    > "outputs/logs/$attempt/codec_control_materialize.log" 2>&1
  echo "MATERIALIZED kitti_final codec-control"
fi

# ---------- PART 2: 4-arm x 14-condition eval ----------
run_holdout() {
  local arm=$1 corruption=$2 severity=$3
  local registry="outputs/nn_paired_protocol_v1_20260924/engines/kitti_retinanet_${arm}/trt/engine.json"
  local prefix="kitti_holdout__retinanet_r50_fpn_v2__${arm}__${corruption}-s${severity}"
  local prediction="outputs/predictions/$attempt/${prefix}.json"
  local inputs="outputs/inputs/$attempt/${prefix}.json"
  local run="manifests/runs/$attempt/${prefix}.json"
  local metric="outputs/metrics/$attempt/${prefix}.json"
  if [[ -f "$prediction" && -f "$inputs" && -f "$run" && -f "$metric" ]]; then return 0; fi
  if [[ -e "$prediction" || -e "$inputs" || -e "$run" || -e "$metric" ]]; then
    echo "partial condition exists: $prefix" >&2; exit 1
  fi
  local source_args
  case "$corruption" in
    clean)
      source_args=(--image-root "$CLEAN_ROOT") ;;
    codec-control)
      source_args=(--image-manifest "$CODEC_MANIFEST"
                   --manifest-cache-root "$CODEC_CACHE") ;;
    *)
      source_args=(--image-manifest "$CORR_MANIFEST_DIR/kitti_final_${corruption}_s${severity}.json"
                   --manifest-cache-root "$CORR_ROOT") ;;
  esac
  wait_gpu 3072
  $PY src/cross_family_infer_trt.py \
    --engine-registry "$registry" --annotations "$ANN" "${source_args[@]}" \
    --out "$prediction" --input-record "$inputs" --run-record "$run" \
    --condition-id "$prefix" --dataset kitti --split test \
    --corruption "$corruption" --severity "$severity" --confidence 0.05 \
    >"outputs/logs/$attempt/${prefix}.log" 2>&1
  taskset -c 0 $PY src/coco_eval.py --annotations "$ANN" \
    --predictions "$prediction" --input-record "$inputs" \
    --run-record "$run" --out "$metric" \
    >>"outputs/logs/$attempt/${prefix}.log" 2>&1
  echo "DONE $prefix"
}

for arm in int8-matched512 int8-selective512 int8-wonly512 int8-aonly512; do
  run_holdout "$arm" clean 0
  run_holdout "$arm" codec-control 0
  for corruption in fog gaussian_noise jpeg motion_blur; do
    for severity in 1 3 5; do
      run_holdout "$arm" "$corruption" "$severity"
    done
  done
done
echo "HOLDOUT QUEUE COMPLETE"
