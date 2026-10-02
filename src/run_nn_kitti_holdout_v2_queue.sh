#!/usr/bin/env bash
# Genuine KITTI final-partition holdout (post-hoc, 2026-10-02).
#
# The phase-L "holdout" (nn_holdout_v1_20261001) re-evaluated engines whose
# RetinaNet checkpoint had been trained on the full 5,985-image Ultralytics
# train partition, which contains the 1,197 final images. This queue repeats
# the factorial on a checkpoint that never saw them:
#   * RetinaNet-R50-FPN-v2 retrained with the frozen profile on the 4,788-image
#     kitti_confirmatory_v1 train list, selected by minimum loss on the same
#     1,496-image selection partition (run ..._kitti_confirmatory_s20260807_v1);
#   * calibration on kitti_confirmatory_train_512_s20260818_v1 (drawn from the
#     4,788-image train list; zero overlap with the final partition);
#   * six arms (fp32, fp8-matched512, int8-matched512, int8-selective512,
#     int8-wonly512, int8-aonly512) x 14 conditions on the final partition
#     and, for reference, on the selection partition.
# Idempotent per artifact; fails closed on partial outputs.
set -euo pipefail
root=${1:-/home/thuan/topic_c_ivc}
cd "$root"
source /home/thuan/miniconda3/etc/profile.d/conda.sh
conda activate qtsd
export PYTHONPATH=src
PY=/home/thuan/miniconda3/envs/qtsd/bin/python
attempt=nn_kitti_holdout_v2_20261002
exec 9>"outputs/logs/kitti_holdout_v2_queue.lock"
flock -n 9 || { echo "kitti holdout v2 queue already active"; exit 0; }
mkdir -p "outputs/predictions/$attempt" "outputs/inputs/$attempt" "outputs/metrics/$attempt" \
  "manifests/runs/$attempt" "outputs/logs/$attempt" "outputs/$attempt/engines"

TRAIN_REG=manifests/training/kitti_retinanet_r50_fpn_v2_confirmatory_s20260807_v1.json
FP32_REG=manifests/onnx/$attempt/kitti_retinanet_r50_fpn_v2_confirmatory_fp32.json
CALIB=manifests/calibration/kitti_confirmatory_train_512_s20260818_v1.json
[[ -f "${TRAIN_REG}.complete" ]] || { echo "training registry incomplete: $TRAIN_REG" >&2; exit 1; }

wait_gpu() {
  local need=$1
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [[ "$free" -ge "$need" ]]; then return 0; fi
    sleep 60
  done
}

# ---------- PART 1: FP32 ONNX export ----------
if [[ ! -f "${FP32_REG}.complete" ]]; then
  mkdir -p "$(dirname "$FP32_REG")" "outputs/$attempt/engines/kitti_retinanet_fp32"
  $PY src/export_cross_family_onnx.py --training-registry "$TRAIN_REG" --imgsz 640 --num-classes 8 \
    --out "outputs/$attempt/engines/kitti_retinanet_fp32/model.onnx" --registry-out "$FP32_REG" \
    > "outputs/logs/$attempt/export_fp32.log" 2>&1
  echo "EXPORTED fp32"
fi

# ---------- PART 2: quantized ONNX arms (frozen matched recipe) ----------
quantize_arm() {
  local arm=$1 mode=$2 extra=${3:-}
  local dir="outputs/$attempt/engines/kitti_retinanet_${arm}"
  if [[ -f "$dir/onnx.json" && -f "$dir/onnx.json.complete" ]]; then return 0; fi
  mkdir -p "$dir"
  wait_gpu 6144
  $PY src/quantize_yolo_onnx.py --onnx-registry "$FP32_REG" --mode "$mode" --imgsz 640 \
    --calibration-list "$CALIB" --calibration-preprocessing retinanet_normalized $extra \
    --out "$dir/model.onnx" --registry-out "$dir/onnx.json" > "$dir/quantize.log" 2>&1
  echo "QUANTIZED $arm"
}
quantize_arm fp8-matched512 fp8
quantize_arm int8-matched512 int8-entropy
quantize_arm int8-selective512 int8-entropy "--exclude-node-regex ^/head/regression_head/"
for variant in wonly aonly; do
  dir="outputs/$attempt/engines/kitti_retinanet_int8-${variant}512"
  if [[ ! -f "$dir/onnx.json.complete" ]]; then
    mkdir -p "$dir"
    $PY src/build_reghead_wa_arms.py \
      --source-onnx-registry "outputs/$attempt/engines/kitti_retinanet_int8-matched512/onnx.json" \
      --variant "$variant" \
      --reference-onnx "outputs/$attempt/engines/kitti_retinanet_int8-selective512/model.onnx" \
      --out "$dir/model.onnx" --registry-out "$dir/onnx.json" > "$dir/surgery.log" 2>&1
    echo "REWIRED int8-${variant}512"
  fi
done

# ---------- PART 3: TensorRT engines (Python builder, TF32 disabled) ----------
build_arm() {
  local arm=$1 precision=$2 src_reg=$3
  local dir="outputs/$attempt/engines/kitti_retinanet_${arm}"
  if [[ -f "$dir/trt/engine.json.complete" ]]; then return 0; fi
  rm -rf "$dir/trt"
  wait_gpu 8192
  $PY src/build_trt_engine_registry.py --onnx-registry "$src_reg" --precision "$precision" \
    --destination "$dir/trt" > "$dir/build.log" 2>&1
  echo "BUILT $arm"
}
build_arm fp32 fp32 "$FP32_REG"
build_arm fp8-matched512 fp8 "outputs/$attempt/engines/kitti_retinanet_fp8-matched512/onnx.json"
for arm in int8-matched512 int8-selective512 int8-wonly512 int8-aonly512; do
  build_arm "$arm" int8-entropy "outputs/$attempt/engines/kitti_retinanet_${arm}/onnx.json"
done

# ---------- PART 4: 14-condition evaluation on final + selection ----------
run_eval() {
  local part=$1 arm=$2 corruption=$3 severity=$4
  local ann clean_root codec_manifest codec_cache corr_manifest corr_root split
  case "$part" in
    final) ann=manifests/annotations/kitti_confirmatory_final_v1_coco.json; split=test
           codec_manifest=manifests/images/kitti_final_codec_control_q95_p0_v1.json
           codec_cache=data/codec_control/kitti_final
           corr_manifest=manifests/images/voc_kitti_confirmatory_v1/kitti_final_${corruption}_s${severity}.json
           corr_root=data/confirmatory_corruptions/voc_kitti_confirmatory_v1/kitti ;;
    val)   ann=manifests/annotations/kitti_val_ultralytics_v1_coco.json; split=val
           codec_manifest=manifests/images/kitti_val_codec_control_q95_p0_v1.json
           codec_cache=data/codec_control/kitti
           corr_manifest=manifests/images/kitti_val_full_${corruption}_s${severity}.json
           corr_root=data/kitti_c ;;
  esac
  clean_root=data/datasets/kitti
  local registry="outputs/$attempt/engines/kitti_retinanet_${arm}/trt/engine.json"
  local prefix="kitti_${part}__retinanet_r50_fpn_v2__${arm}__${corruption}-s${severity}"
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
    clean) source_args=(--image-root "$clean_root") ;;
    codec-control) source_args=(--image-manifest "$codec_manifest" --manifest-cache-root "$codec_cache") ;;
    *) source_args=(--image-manifest "$corr_manifest" --manifest-cache-root "$corr_root") ;;
  esac
  wait_gpu 3072
  $PY src/cross_family_infer_trt.py --engine-registry "$registry" --annotations "$ann" "${source_args[@]}" \
    --out "$prediction" --input-record "$inputs" --run-record "$run" \
    --condition-id "$prefix" --dataset kitti --split "$split" \
    --corruption "$corruption" --severity "$severity" --confidence 0.05 \
    > "outputs/logs/$attempt/${prefix}.log" 2>&1
  taskset -c 0 $PY src/coco_eval.py --annotations "$ann" --predictions "$prediction" \
    --input-record "$inputs" --run-record "$run" --out "$metric" >> "outputs/logs/$attempt/${prefix}.log" 2>&1
  echo "DONE $prefix"
}

ARMS="int8-matched512 int8-selective512 int8-wonly512 int8-aonly512 fp8-matched512 fp32"
for part in final val; do
  for arm in $ARMS; do
    run_eval "$part" "$arm" clean 0
    run_eval "$part" "$arm" codec-control 0
  done
done
for part in final val; do
  for arm in $ARMS; do
    for corruption in fog gaussian_noise jpeg motion_blur; do
      for severity in 1 3 5; do
        run_eval "$part" "$arm" "$corruption" "$severity"
      done
    done
  done
done
echo "KITTI HOLDOUT V2 QUEUE COMPLETE"
