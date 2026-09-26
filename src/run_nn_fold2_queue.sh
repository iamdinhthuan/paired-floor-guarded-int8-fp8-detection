#!/usr/bin/env bash
# Phase F: complementary held-out fold (fog+motion_blur calibrated; noise+jpeg held out).
#
# Scope (frozen 2026-09-25, see configs/nn_corruptcalib_fold2_v1_20260925.json):
#   * Build deterministic fold-2 corruptcalib manifests for the same
#     512 clean calibration images (fog + motion_blur @ s1/s3/s5;
#     gaussian_noise + jpeg held out).
#   * Build 4 RetinaNet engines: int8-cc2calib512 and
#     int8-sel-cc2calib512 on kitti + voc.
#   * Run each new arm on the frozen 14-condition protocol
#     (clean, codec-control, 12 corruption cells) on kitti_val/voc_val.
#
# Idempotent per condition; serial GPU execution; fails closed.
set -euo pipefail

root=${1:-/home/thuan/topic_c_ivc}
cd "$root"
source /home/thuan/miniconda3/etc/profile.d/conda.sh
conda activate qtsd
export PYTHONPATH=src
PY=/home/thuan/miniconda3/envs/qtsd/bin/python
attempt=nn_corruptcalib_fold2_v1_20260925
mkdir -p "outputs/predictions/$attempt" "outputs/inputs/$attempt" "outputs/metrics/$attempt" \
  "manifests/runs/$attempt" "outputs/logs/$attempt"
exec 9>"outputs/logs/$attempt/queue.lock"
flock -n 9 || { echo "nn fold2 queue already active"; exit 0; }

# ---------- PART 0: corruptcalib manifests ----------
for dataset in kitti voc; do
  manifest="manifests/calibration/${dataset}_train_corruptcalib_512_nn-corruptcalib-fold2-v1.json"
  if [[ ! -f "$manifest" ]]; then
    $PY src/build_corrupt_calibration.py \
      --calibration-manifest "manifests/calibration/${dataset}_train_clean_512_s20260807_v1.json" \
      --config configs/corruptions.json \
      --schedule-id nn-corruptcalib-fold2-v1 \
      --in-family fog,motion_blur --severities 1,3,5 \
      --held-out gaussian_noise,jpeg \
      --out-root "data/calibration_corrupt/${dataset}_train_corruptcalib_512_fold2_v1" \
      --manifest-out "$manifest"
  fi
done

# ---------- PART 1: engine builds ----------
quantize_arm() {
  local dataset=$1 arm=$2 extra=${3:-}
  local dir="outputs/$attempt/engines/${dataset}_retinanet_${arm}"
  if [[ -f "$dir/onnx.json" && -f "$dir/onnx.json.complete" ]]; then return 0; fi
  mkdir -p "$dir"
  $PY src/quantize_yolo_onnx.py \
    --onnx-registry "manifests/onnx/cross_family_v1/${dataset}_retinanet_r50_fpn_v2_fp32.json" \
    --mode int8-entropy --imgsz 640 \
    --calibration-list "manifests/calibration/${dataset}_train_corruptcalib_512_nn-corruptcalib-fold2-v1.json" \
    --calibration-preprocessing retinanet_normalized $extra \
    --out "$dir/model.onnx" --registry-out "$dir/onnx.json" \
    >"$dir/quantize.log" 2>&1
  echo "BUILT_ONNX ${dataset}_${arm}"
}

build_arm() {
  local dataset=$1 arm=$2
  local dir="outputs/$attempt/engines/${dataset}_retinanet_${arm}"
  local trt="$dir/trt"
  if [[ -f "$trt/engine.json" && -f "$trt/engine.json.complete" ]]; then return 0; fi
  $PY src/build_trt_engine_registry.py \
    --onnx-registry "$dir/onnx.json" --precision int8-entropy --destination "$trt" \
    >"$dir/build.log" 2>&1
  echo "BUILT_ENGINE ${dataset}_${arm}"
}

for dataset in kitti voc; do
  quantize_arm "$dataset" int8-cc2calib512
  build_arm    "$dataset" int8-cc2calib512
  quantize_arm "$dataset" int8-sel-cc2calib512 "--exclude-node-regex ^/head/regression_head/"
  build_arm    "$dataset" int8-sel-cc2calib512
done

# ---------- PART 2: intervention-arm inference ----------
run_retinanet() {
  local dataset=$1 arm=$2 corruption=$3 severity=$4
  local annotations clean_root cache_root
  case "$dataset" in
    kitti) annotations=manifests/annotations/kitti_val_ultralytics_v1_coco.json
           clean_root=data/datasets/kitti; cache_root=data/kitti_c ;;
    voc)   annotations=manifests/annotations/voc_val_ultralytics_v1_coco.json
           clean_root=data/datasets/VOC;   cache_root=data/voc_c ;;
  esac
  local registry="outputs/$attempt/engines/${dataset}_retinanet_${arm}/trt/engine.json"
  local prefix="${dataset}_val__retinanet_r50_fpn_v2__${arm}__${corruption}-s${severity}"
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
      source_args=(--image-root "$clean_root") ;;
    codec-control)
      source_args=(--image-manifest "manifests/images/${dataset}_val_codec_control_q95_p0_v1.json"
                   --manifest-cache-root "data/codec_control/$dataset") ;;
    *)
      source_args=(--image-manifest "manifests/images/${dataset}_val_full_${corruption}_s${severity}.json"
                   --manifest-cache-root "$cache_root") ;;
  esac
  $PY src/cross_family_infer_trt.py \
    --engine-registry "$registry" --annotations "$annotations" "${source_args[@]}" \
    --out "$prediction" --input-record "$inputs" --run-record "$run" \
    --condition-id "$prefix" --dataset "$dataset" --split val \
    --corruption "$corruption" --severity "$severity" --confidence 0.05 \
    >"outputs/logs/$attempt/${prefix}.log" 2>&1
  taskset -c 0 $PY src/coco_eval.py --annotations "$annotations" --predictions "$prediction" \
    --input-record "$inputs" --run-record "$run" --out "$metric" \
    >>"outputs/logs/$attempt/${prefix}.log" 2>&1
  echo "DONE $prefix"
}

# Clean arms first so catastrophic floors are visible early.
for dataset in kitti voc; do
  for arm in int8-cc2calib512 int8-sel-cc2calib512; do
    run_retinanet "$dataset" "$arm" clean 0
    run_retinanet "$dataset" "$arm" codec-control 0
  done
done
for dataset in kitti voc; do
  for arm in int8-cc2calib512 int8-sel-cc2calib512; do
    for corruption in fog gaussian_noise jpeg motion_blur; do
      for severity in 1 3 5; do
        run_retinanet "$dataset" "$arm" "$corruption" "$severity"
      done
    done
  done
done

echo "NN FOLD2 QUEUE COMPLETE"
