#!/usr/bin/env bash
# W/A mechanism-ablation queue: int8-wonly512 / int8-aonly512 RetinaNet arms.
# Waits for >=6 GiB free GPU memory before builds and >=3 GiB before runs,
# because the RTX 5090 is shared with other tenants.
set -euo pipefail
root=${1:-/home/thuan/topic_c_ivc}
cd "$root"
source /home/thuan/miniconda3/etc/profile.d/conda.sh
conda activate qtsd
export PYTHONPATH=src
PY=/home/thuan/miniconda3/envs/qtsd/bin/python
attempt=nn_paired_protocol_v1_20260924
exec 9>"outputs/logs/$attempt/wa_queue.lock"
flock -n 9 || { echo "wa queue already active"; exit 0; }

wait_gpu() {
  local need=$1
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [[ "$free" -ge "$need" ]]; then return 0; fi
    sleep 60
  done
}

for dataset in kitti voc; do
  for arm in int8-wonly512 int8-aonly512; do
    dir="outputs/$attempt/engines/${dataset}_retinanet_${arm}"
    trt="$dir/trt"
    if [[ ! -f "$trt/engine.json.complete" ]]; then
      rm -rf "$trt"
      wait_gpu 8192
      $PY src/build_trt_engine_registry.py \
        --onnx-registry "$dir/onnx.json" --precision int8-entropy --destination "$trt" \
        >"$dir/build.log" 2>&1
      echo "BUILT_ENGINE ${dataset}_${arm}"
    fi
  done
done

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
  wait_gpu 3072
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
  for arm in int8-wonly512 int8-aonly512; do
    run_retinanet "$dataset" "$arm" clean 0
    run_retinanet "$dataset" "$arm" codec-control 0
  done
done
for dataset in kitti voc; do
  for arm in int8-wonly512 int8-aonly512; do
    for corruption in fog gaussian_noise jpeg motion_blur; do
      for severity in 1 3 5; do
        run_retinanet "$dataset" "$arm" "$corruption" "$severity"
      done
    done
  done
done
echo "WA_ABLATION_QUEUE_COMPLETE"
