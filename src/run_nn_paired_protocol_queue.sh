#!/usr/bin/env bash
# Phase B paired-protocol queue (NN upgrade).
#
# Scope (frozen 2026-09-24, see
# submission_support_20260911/review_notes/NN_PHASE_B_FREEZE_DRAFT_20260924.md):
#   * RetinaNet matched-recipe arms on kitti_val and voc_val:
#     fp8-matched512, int8-matched512, int8-selective512 — each on
#     original-clean, codec-control Q95 clean, and 12 corruption cells.
#   * COCO: no new inference; deterministic subset evaluation of the
#     existing 5,000-image paired matrix on the frozen 2,000-image subset.
#   * YOLO11 n/m/x on kitti_val/voc_val and RetinaNet legacy arms reuse
#     existing hash-bound predictions.
#
# Idempotent per condition; serial GPU execution; fails closed on any
# partial/mismatched artifact.
set -euo pipefail

root=${1:-/home/thuan/topic_c_ivc}
cd "$root"
source /home/thuan/miniconda3/etc/profile.d/conda.sh
conda activate qtsd
export PYTHONPATH=src
PY=/home/thuan/miniconda3/envs/qtsd/bin/python
attempt=nn_paired_protocol_v1_20260924
mkdir -p "outputs/predictions/$attempt" "outputs/inputs/$attempt" "outputs/metrics/$attempt" \
  "manifests/runs/$attempt" "outputs/logs/$attempt" "manifests/subsets/$attempt"
exec 9>"outputs/logs/$attempt/queue.lock"
flock -n 9 || { echo "nn paired protocol queue already active"; exit 0; }

# ---------- PART 0: frozen COCO paired subset ----------
SUBSET=manifests/subsets/$attempt/coco_val2017_paired_2000_s20260924.json
if [[ ! -f "$SUBSET" ]]; then
  $PY src/build_paired_subset.py \
    --parent-manifest manifests/images/coco_val2017_clean_p0_v1.json \
    --count 2000 --seed 20260924 --attempt "$attempt" --out "$SUBSET"
fi

# ---------- PART 1: matched-recipe RetinaNet engine builds ----------
# Arms: fp8-matched512, int8-matched512, int8-selective512 per dataset.
quantize_arm() {
  local dataset=$1 arm=$2 mode=$3 extra=${4:-}
  local dir="outputs/$attempt/engines/${dataset}_retinanet_${arm}"
  if [[ -f "$dir/onnx.json" && -f "$dir/onnx.json.complete" ]]; then return 0; fi
  mkdir -p "$dir"
  $PY src/quantize_yolo_onnx.py \
    --onnx-registry "manifests/onnx/cross_family_v1/${dataset}_retinanet_r50_fpn_v2_fp32.json" \
    --mode "$mode" --imgsz 640 \
    --calibration-list "manifests/calibration/${dataset}_train_clean_512_s20260807_v1.json" \
    --calibration-preprocessing retinanet_normalized $extra \
    --out "$dir/model.onnx" --registry-out "$dir/onnx.json" \
    >"$dir/quantize.log" 2>&1
  echo "BUILT_ONNX ${dataset}_${arm}"
}

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

for dataset in kitti voc; do
  quantize_arm "$dataset" fp8-matched512 fp8
  build_arm    "$dataset" fp8-matched512 fp8
  quantize_arm "$dataset" int8-matched512 int8-entropy
  build_arm    "$dataset" int8-matched512 int8-entropy
  quantize_arm "$dataset" int8-selective512 int8-entropy "--exclude-node-regex ^/head/regression_head/"
  build_arm    "$dataset" int8-selective512 int8-entropy
done

# ---------- PART 2: RetinaNet matched-arm inference ----------
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

# Absolute clean AP first so catastrophic floors are visible early.
for dataset in kitti voc; do
  for arm in fp8-matched512 int8-matched512 int8-selective512; do
    run_retinanet "$dataset" "$arm" clean 0
    run_retinanet "$dataset" "$arm" codec-control 0
  done
done
for dataset in kitti voc; do
  for arm in fp8-matched512 int8-matched512 int8-selective512; do
    for corruption in fog gaussian_noise jpeg motion_blur; do
      for severity in 1 3 5; do
        run_retinanet "$dataset" "$arm" "$corruption" "$severity"
      done
    done
  done
done

# ---------- PART 3: COCO subset evaluation (no new inference) ----------
COCO_ANN=data/datasets/coco_pilot_v1_20260923/annotations/instances_val2017.json
subset_eval() {
  local run_record=$1
  local base
  base=$(basename "$run_record" .json)
  local attempt_dir
  attempt_dir=$(basename "$(dirname "$run_record")")
  local prediction="outputs/predictions/$attempt_dir/$base.json"
  local inputs="outputs/inputs/$attempt_dir/$base.json"
  local metric="outputs/metrics/$attempt/subset_${base}.json"
  [[ -f "$metric" ]] && return 0
  if [[ ! -f "$prediction" || ! -f "$inputs" ]]; then
    echo "missing parent artifacts for $base" >&2; exit 1
  fi
  taskset -c 0 $PY src/evaluate_paired_subset.py \
    --annotations "$COCO_ANN" --predictions "$prediction" --input-record "$inputs" \
    --run-record "$run_record" --subset "$SUBSET" --out "$metric" \
    >>"outputs/logs/$attempt/subset_eval.log" 2>&1
  echo "SUBSET $base"
}

for run_record in manifests/runs/coco_uniform_p0_v1/coco_val2017__yolo11*.json \
                  manifests/runs/codec_control_p0_v1/coco_val2017__yolo11*.json; do
  subset_eval "$run_record"
done

echo "NN PAIRED PROTOCOL QUEUE COMPLETE"
