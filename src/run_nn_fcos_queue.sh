#!/usr/bin/env bash
# FCOS replication queue: train -> export -> quantize -> build -> 14-cond eval.
# GPU-gated; idempotent; fails closed on partial artifacts.
set -euo pipefail
root=${1:-/home/thuan/topic_c_ivc}
cd "$root"
source /home/thuan/miniconda3/etc/profile.d/conda.sh
conda activate qtsd
export PYTHONPATH=src
PY=/home/thuan/miniconda3/envs/qtsd/bin/python
attempt=nn_fcos_replication_v1_20260927
exec 9>"outputs/logs/fcos_queue.lock"
flock -n 9 || { echo "fcos queue already active"; exit 0; }
mkdir -p "outputs/predictions/$attempt" "outputs/inputs/$attempt" "outputs/metrics/$attempt" \
  "manifests/runs/$attempt" "outputs/logs/$attempt" manifests/onnx/fcos_v1 manifests/training

wait_gpu() {
  local need=$1
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [[ "$free" -ge "$need" ]]; then return 0; fi
    sleep 60
  done
}

# ---------- PART 1: training ----------
for dataset in kitti voc; do
  reg="manifests/training/${dataset}_fcos_r50_fpn_s20260807_v1.json"
  run_id="fcos_r50_fpn_${dataset}_s20260807_v1"
  if [[ ! -f "${reg}.complete" ]]; then
    wait_gpu 12288
    resume_arg=()
    last_pt="outputs/training/${dataset}/${run_id}/weights/last.pt"
    [[ -f "$last_pt" ]] && resume_arg=(--resume-from "$last_pt")
    $PY src/train_fcos_dataset.py \
      --project-root . \
      --profile configs/training/cross_family_retinanet_r50_fpn_v2_v1.json \
      --dataset "$dataset" \
      --data-yaml "configs/datasets/${dataset}_ultralytics_v1.yaml" \
      --acquisition-registry "manifests/datasets/${dataset}_acquisition_v1.json" \
      --run-id "$run_id" \
      --registry-out "$reg" \
      "${resume_arg[@]}" \
      >> "outputs/logs/${run_id}.log" 2>&1
    echo "TRAINED $run_id"
  fi
done

# ---------- PART 2: ONNX export ----------
num_classes() { case "$1" in kitti) echo 8;; voc) echo 20;; esac; }
for dataset in kitti voc; do
  reg="manifests/onnx/fcos_v1/${dataset}_fcos_r50_fpn_fp32.json"
  if [[ ! -f "${reg}.complete" ]]; then
    $PY src/export_cross_family_onnx.py \
      --training-registry "manifests/training/${dataset}_fcos_r50_fpn_s20260807_v1.json" \
      --imgsz 640 --num-classes "$(num_classes $dataset)" \
      --out "outputs/$attempt/engines/${dataset}_fcos_fp32/model.onnx" \
      --registry-out "$reg" \
      > "outputs/logs/$attempt/${dataset}_fcos_export.log" 2>&1
    echo "EXPORTED $dataset fcos fp32"
  fi
done

# ---------- PART 3: quantize + build ----------
for dataset in kitti voc; do
  onnx_reg="manifests/onnx/fcos_v1/${dataset}_fcos_r50_fpn_fp32.json"
  for arm in fp8-matched512 int8-matched512 int8-selective512; do
    dir="outputs/$attempt/engines/${dataset}_fcos_${arm}"
    if [[ ! -f "$dir/onnx.json.complete" ]]; then
      extra=""
      mode="${arm%%-*}"
      [[ "$arm" == int8-* ]] && mode="int8-entropy"
      [[ "$arm" == int8-selective512 ]] && extra="--exclude-node-regex ^/head/regression_head/"
      wait_gpu 6144
      $PY src/quantize_yolo_onnx.py \
        --onnx-registry "$onnx_reg" \
        --mode "$mode" --imgsz 640 \
        --calibration-list "manifests/calibration/${dataset}_train_clean_512_s20260807_v1.json" \
        --calibration-preprocessing retinanet_normalized $extra \
        --out "$dir/model.onnx" --registry-out "$dir/onnx.json" \
        >"$dir/quantize.log" 2>&1
      echo "QUANTIZED ${dataset}_fcos_${arm}"
    fi
  done
done
for dataset in kitti voc; do
  for arm in fp32 fp8-matched512 int8-matched512 int8-selective512; do
    dir="outputs/$attempt/engines/${dataset}_fcos_${arm}"
    if [[ "$arm" == fp32 ]]; then
      src_reg="manifests/onnx/fcos_v1/${dataset}_fcos_r50_fpn_fp32.json"
      prec="fp32"
    else
      src_reg="$dir/onnx.json"
      prec="int8-entropy"; [[ "$arm" == fp8-* ]] && prec="fp8"
    fi
    if [[ ! -f "$dir/trt/engine.json.complete" ]]; then
      rm -rf "$dir/trt"
      wait_gpu 8192
      $PY src/build_trt_engine_registry.py \
        --onnx-registry "$src_reg" --precision "$prec" --destination "$dir/trt" \
        >"$dir/build.log" 2>&1
      echo "BUILT ${dataset}_fcos_${arm}"
    fi
  done
done

# ---------- PART 4: 14-condition eval ----------
run_fcos() {
  local dataset=$1 arm=$2 corruption=$3 severity=$4
  local annotations clean_root cache_root
  case "$dataset" in
    kitti) annotations=manifests/annotations/kitti_val_ultralytics_v1_coco.json
           clean_root=data/datasets/kitti; cache_root=data/kitti_c ;;
    voc)   annotations=manifests/annotations/voc_val_ultralytics_v1_coco.json
           clean_root=data/datasets/VOC;   cache_root=data/voc_c ;;
  esac
  local registry="outputs/$attempt/engines/${dataset}_fcos_${arm}/trt/engine.json"
  local prefix="${dataset}_val__fcos_r50_fpn__${arm}__${corruption}-s${severity}"
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
  for arm in fp32 fp8-matched512 int8-matched512 int8-selective512; do
    run_fcos "$dataset" "$arm" clean 0
    run_fcos "$dataset" "$arm" codec-control 0
  done
done
for dataset in kitti voc; do
  for arm in fp32 fp8-matched512 int8-matched512 int8-selective512; do
    for corruption in fog gaussian_noise jpeg motion_blur; do
      for severity in 1 3 5; do
        run_fcos "$dataset" "$arm" "$corruption" "$severity"
      done
    done
  done
done
echo "FCOS_REPLICATION_QUEUE_COMPLETE"
