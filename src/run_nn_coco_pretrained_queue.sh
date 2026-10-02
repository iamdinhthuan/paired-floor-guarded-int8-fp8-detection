#!/usr/bin/env bash
# COCO-pretrained RetinaNet/FCOS replication queue:
# torchvision COCO_V1 weights -> ONNX -> quantize -> build -> 14-cond eval
# on the frozen paired 2,000-image val2017 subset. GPU-gated; idempotent.
set -euo pipefail
root=${1:-/home/thuan/topic_c_ivc}
cd "$root"
source /home/thuan/miniconda3/etc/profile.d/conda.sh
conda activate qtsd
export PYTHONPATH=src
PY=/home/thuan/miniconda3/envs/qtsd/bin/python
attempt=nn_coco_pretrained_v1_20260930
exec 9>"outputs/logs/coco_pretrained_queue.lock"
flock -n 9 || { echo "coco pretrained queue already active"; exit 0; }
mkdir -p "outputs/predictions/$attempt" "outputs/inputs/$attempt" "outputs/metrics/$attempt" \
  "manifests/runs/$attempt" "outputs/logs/$attempt" manifests/onnx/coco_pretrained_v1

wait_gpu() {
  local need=$1
  while :; do
    free=$(nvidia-smi --query-gpu=memory.free --format=csv,noheader,nounits | head -1)
    if [[ "$free" -ge "$need" ]]; then return 0; fi
    sleep 60
  done
}

ANN_EVAL=manifests/annotations/coco_val2017_paired2000_eval.json
SUBDIR=manifests/images/$attempt
CAL=manifests/calibration/coco_train_clean_512_s20260807_p0_v1_pilotroot.json

# ---------- PART 0: subset assets ----------
if [[ ! -f "$ANN_EVAL" ]]; then
  $PY src/build_coco_pretrained_assets.py --root "$root"
fi

# ---------- PART 1: ONNX export (stock torchvision COCO_V1 heads) ----------
for model in retinanet fcos; do
  reg="manifests/onnx/coco_pretrained_v1/coco_${model}_fp32.json"
  if [[ ! -f "${reg}.complete" ]]; then
    $PY src/export_pretrained_onnx.py --model "$model" --imgsz 640 \
      --out "outputs/$attempt/engines/coco_${model}_fp32/model.onnx" \
      --registry-out "$reg" \
      > "outputs/logs/$attempt/${model}_export.log" 2>&1
    echo "EXPORTED coco_${model}_fp32"
    $PY - "$reg" <<'PY'
import sys, json
from pathlib import Path
sys.path.insert(0, "src")
from topic_c.manifest import sha256_file
reg = Path(sys.argv[1])
Path(str(reg) + ".complete").write_text(sha256_file(reg))
PY
  fi
done

# ---------- PART 2: quantize + build ----------
quant_arm() {
  local model=$1 arm=$2 mode=$3 extra=${4:-}
  local dir="outputs/$attempt/engines/coco_${model}_${arm}"
  if [[ -f "$dir/onnx.json" && -f "$dir/onnx.json.complete" ]]; then return 0; fi
  mkdir -p "$dir"
  wait_gpu 6144
  $PY src/quantize_yolo_onnx.py \
    --onnx-registry "manifests/onnx/coco_pretrained_v1/coco_${model}_fp32.json" \
    --mode "$mode" --imgsz 640 \
    --calibration-list "$CAL" \
    --calibration-preprocessing retinanet_normalized $extra \
    --out "$dir/model.onnx" --registry-out "$dir/onnx.json" \
    >"$dir/quantize.log" 2>&1
  $PY - "$dir/onnx.json" <<'PY'
import sys
from pathlib import Path
sys.path.insert(0, "src")
from topic_c.manifest import sha256_file
reg = Path(sys.argv[1])
Path(str(reg) + ".complete").write_text(sha256_file(reg))
PY
  echo "QUANTIZED coco_${model}_${arm}"
}

build_arm() {
  local model=$1 arm=$2 precision=$3
  local dir="outputs/$attempt/engines/coco_${model}_${arm}"
  local trt="$dir/trt"
  if [[ -f "$trt/engine.json" && -f "$trt/engine.json.complete" ]]; then return 0; fi
  $PY src/build_trt_engine_registry.py \
    --onnx-registry "$dir/onnx.json" --precision "$precision" --destination "$trt" \
    >"$dir/build.log" 2>&1
  echo "BUILT_ENGINE coco_${model}_${arm}"
}

# FP32 build for both models
for model in retinanet fcos; do
  dir="outputs/$attempt/engines/coco_${model}_fp32"
  if [[ ! -f "$dir/trt/engine.json" ]]; then
    onnx_reg="manifests/onnx/coco_pretrained_v1/coco_${model}_fp32.json"
    $PY src/build_trt_engine_registry.py \
      --onnx-registry "$onnx_reg" --precision fp32 --destination "$dir/trt" \
      >"$dir/trt_build.log" 2>&1 || { mkdir -p "$dir"; $PY src/build_trt_engine_registry.py \
      --onnx-registry "$onnx_reg" --precision fp32 --destination "$dir/trt" \
      >"$dir/trt_build.log" 2>&1; }
    echo "BUILT_ENGINE coco_${model}_fp32"
  fi
done

for model in retinanet fcos; do
  quant_arm "$model" fp8-matched512 fp8
  build_arm  "$model" fp8-matched512 fp8
  quant_arm "$model" int8-matched512 int8-entropy
  build_arm  "$model" int8-matched512 int8-entropy
  quant_arm "$model" int8-selective512 int8-entropy "--exclude-node-regex ^/head/regression_head/"
  build_arm  "$model" int8-selective512 int8-entropy
done

# ---------- PART 3: inference + metrics on the 14 frozen conditions ----------
CATMAP_retinanet=manifests/annotations/coco_torchvision_map_retinanet91.json
CATMAP_fcos=manifests/annotations/coco_torchvision_map_fcos91.json

run_cell() {
  local model=$1 arm=$2 corruption=$3 severity=$4 suffix=$5
  local engine="outputs/$attempt/engines/coco_${model}_${arm}/trt/engine.json"
  local prefix="coco_paired2000__${model}_pretrained__${arm}__${corruption}-s${severity}"
  local prediction="outputs/predictions/$attempt/${prefix}.json"
  local inputs="outputs/inputs/$attempt/${prefix}.json"
  local run="manifests/runs/$attempt/${prefix}.json"
  local metric="outputs/metrics/$attempt/${prefix}.json"
  if [[ -f "$prediction" && -f "$inputs" && -f "$run" && -f "$metric" ]]; then return 0; fi
  if [[ -e "$prediction" || -e "$inputs" || -e "$run" || -e "$metric" ]]; then
    echo "partial condition exists: $prefix" >&2; exit 1
  fi
  wait_gpu 2048
  if [[ "$corruption" == "clean" ]]; then
    src_args=(--image-manifest "$SUBDIR/coco_paired2000_clean.json"
              --manifest-cache-root data/datasets/coco_pilot_v1_20260923/images/val2017)
  elif [[ "$corruption" == "codec-control" ]]; then
    src_args=(--image-manifest "$SUBDIR/coco_paired2000_codec_control_q95.json"
              --manifest-cache-root data/codec_control/coco)
  else
    src_args=(--image-manifest "$SUBDIR/coco_paired2000_${corruption}_s${severity}.json"
              --manifest-cache-root data/coco_c)
  fi
  local catmap
  catmap=$(eval "echo \$CATMAP_$model")
  $PY src/cross_family_infer_trt.py \
    --engine-registry "$engine" --annotations "$ANN_EVAL" \
    --categories-json "$catmap" "${src_args[@]}" \
    --out "$prediction" --input-record "$inputs" --run-record "$run" \
    --condition-id "$prefix" --dataset coco --split val \
    --corruption "$corruption" --severity "$severity" --confidence 0.05 \
    > "outputs/logs/$attempt/${prefix}.log" 2>&1
  taskset -c 0 $PY src/coco_eval.py --annotations "$ANN_EVAL" \
    --predictions "$prediction" --input-record "$inputs" \
    --run-record "$run" --out "$metric" \
    >> "outputs/logs/$attempt/${prefix}.log" 2>&1
  echo "DONE $prefix"
}

for model in retinanet fcos; do
  for arm in fp32 fp8-matched512 int8-matched512 int8-selective512; do
    run_cell "$model" "$arm" clean 0 clean
    run_cell "$model" "$arm" codec-control 0 codec-control
    for fam in fog gaussian_noise jpeg motion_blur; do
      for sev in 1 3 5; do
        run_cell "$model" "$arm" "$fam" "$sev" "${fam}-s${sev}"
      done
    done
  done
done
echo "COCO PRETRAINED QUEUE COMPLETE"
