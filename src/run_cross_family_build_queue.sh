#!/usr/bin/env bash
set -euo pipefail

root=${1:-/home/thuan/topic_c_ivc}
cd "$root"
source /home/thuan/miniconda3/etc/profile.d/conda.sh
conda activate qtsd
export PYTHONPATH=src

exec 9>outputs/logs/cross_family_v1/build_queue.lock
flock -n 9 || { echo "cross-family build queue already active"; exit 0; }

valid_registry() {
  local path=$1 marker="${1}.complete"
  [[ -f "$path" && -f "$marker" ]] &&
    [[ "$(sha256sum "$path" | awk '{print $1}')" == "$(tr -d '[:space:]' < "$marker")" ]]
}

wait_existing_quantizer() {
  local output=$1
  while pgrep -af "python src/quantize_yolo_onnx.py" | grep -F -- "--out $output" >/dev/null; do sleep 30; done
}

mkdir -p outputs/onnx/cross_family_v1 outputs/engines/cross_family_v1 \
  manifests/onnx/cross_family_v1 manifests/engines/cross_family_v1 outputs/logs/cross_family_v1

for dataset in voc kitti tt100k; do
  case "$dataset" in
    voc) classes=20; imgsz=640 ;;
    kitti) classes=8; imgsz=640 ;;
    tt100k) classes=221; imgsz=1280 ;;
  esac
  calibration="manifests/calibration/${dataset}_train_clean_512_s20260807_v1.json"
  for model in rtdetr_l retinanet_r50_fpn_v2; do
    stem="${dataset}_${model}"
    train="manifests/training/${stem}_s20260807_v1.json"
    fp32_registry="manifests/onnx/cross_family_v1/${stem}_fp32.json"
    fp32_onnx="outputs/onnx/cross_family_v1/${stem}_fp32.onnx"
    if ! valid_registry "$fp32_registry"; then
      python src/export_cross_family_onnx.py --training-registry "$train" --imgsz "$imgsz" \
        --num-classes "$classes" --out "$fp32_onnx" --registry-out "$fp32_registry" \
        >> "outputs/logs/cross_family_v1/${stem}_export.log" 2>&1
    fi
    for precision in fp32 int8-entropy fp8; do
      onnx_registry="$fp32_registry"
      if [[ "$precision" != fp32 ]]; then
        onnx_registry="manifests/onnx/cross_family_v1/${stem}_${precision}.json"
        output="outputs/onnx/cross_family_v1/${stem}_${precision}.onnx"
        wait_existing_quantizer "$output"
        if ! valid_registry "$onnx_registry"; then
          [[ ! -e "$output" ]] || { echo "uncommitted ONNX exists: $output" >&2; exit 1; }
          python src/quantize_yolo_onnx.py --onnx-registry "$fp32_registry" --mode "$precision" \
            --imgsz "$imgsz" --calibration-list "$calibration" --out "$output" \
            --registry-out "$onnx_registry" >> "outputs/logs/cross_family_v1/${stem}_${precision}_quantize.log" 2>&1
        fi
      fi
      engine_registry="manifests/engines/cross_family_v1/${stem}_${precision}.json"
      if ! valid_registry "$engine_registry"; then
        python src/build_yolo_trt_engine.py --onnx-registry "$onnx_registry" --precision "$precision" \
          --trt-root /home/thuan/traffic/third_party/TensorRT-11.1.0.106 \
          --engine "outputs/engines/cross_family_v1/${stem}_${precision}.engine" \
          --build-log "outputs/logs/cross_family_v1/${stem}_${precision}_build.log" \
          --registry-out "$engine_registry"
      fi
    done
  done
done
echo "CROSS-FAMILY BUILD QUEUE COMPLETE"
