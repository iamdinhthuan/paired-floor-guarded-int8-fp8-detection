#!/usr/bin/env bash
# Head-local max-calibration counterfactual (post-hoc, reviewer follow-up):
# ONNX = frozen int8-matched512 graph with ONLY reg-head-consumed activation
# scales replaced by amax_max/127 from the frozen 512-image actstats captures.
set -euo pipefail
cd /home/thuan/topic_c_ivc
PY=/home/thuan/miniconda3/envs/qtsd/bin/python
attempt=nn_maxreg_v1_20261001
mkdir -p "outputs/logs/$attempt"

build_arm() {
  local dataset=$1 arm=$2 precision=$3
  local dir="outputs/$attempt/engines/${dataset}_retinanet_${arm}"
  local trt="$dir/trt"
  if [[ -f "$trt/engine.json" && -f "$trt/engine.json.complete" ]]; then return 0; fi
  $PY src/build_trt_engine_registry.py \
    --onnx-registry "$dir/onnx.json" --precision "$precision" --destination "$trt" \
    >"$dir/build.log" 2>&1
  echo "BUILT_ENGINE ${dataset}_${arm}"
}

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

for dataset in kitti voc; do
  build_arm "$dataset" int8-maxreg512 int8-maxreg
done
for dataset in kitti voc; do
  run_retinanet "$dataset" int8-maxreg512 clean 0
  run_retinanet "$dataset" int8-maxreg512 codec-control 0
done
for dataset in kitti voc; do
  for corruption in fog gaussian_noise jpeg motion_blur; do
    for severity in 1 3 5; do
      run_retinanet "$dataset" int8-maxreg512 "$corruption" "$severity"
    done
  done
done
echo "MAXREG QUEUE COMPLETE"
