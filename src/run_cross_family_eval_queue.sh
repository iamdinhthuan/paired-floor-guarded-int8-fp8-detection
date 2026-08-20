#!/usr/bin/env bash
set -euo pipefail

root=${1:-/home/thuan/topic_c_ivc}
cd "$root"
source /home/thuan/miniconda3/etc/profile.d/conda.sh
conda activate qtsd
export PYTHONPATH=src
attempt=cross_family_v1
mkdir -p "outputs/predictions/$attempt" "outputs/inputs/$attempt" "outputs/metrics/$attempt" \
  "manifests/runs/$attempt" outputs/logs/cross_family_v1
exec 9>outputs/logs/cross_family_v1/eval_queue.lock
flock -n 9 || { echo "cross-family evaluation queue already active"; exit 0; }

run_condition() {
  local dataset=$1 model=$2 precision=$3 corruption=$4 severity=$5
  local split annotations clean_root cache_root prefix
  case "$dataset" in
    voc) split=val; annotations=manifests/annotations/voc_val_ultralytics_v1_coco.json; clean_root=data/datasets/VOC; cache_root=data/voc_c ;;
    kitti) split=val; annotations=manifests/annotations/kitti_val_ultralytics_v1_coco.json; clean_root=data/datasets/kitti; cache_root=data/kitti_c ;;
    tt100k) split=test; annotations=manifests/annotations/tt100k_test_ultralytics_v1_coco.json; clean_root=data/datasets/TT100K; cache_root=data/tt100k_c ;;
  esac
  prefix="${dataset}_${split}__${model}__${precision}__${corruption}-s${severity}"
  local prediction="outputs/predictions/$attempt/${prefix}.json"
  local inputs="outputs/inputs/$attempt/${prefix}.json"
  local run="manifests/runs/$attempt/${prefix}.json"
  local metric="outputs/metrics/$attempt/${prefix}.json"
  if [[ -f "$prediction" && -f "$inputs" && -f "$run" && -f "$metric" ]]; then return; fi
  if [[ -e "$prediction" || -e "$inputs" || -e "$run" || -e "$metric" ]]; then
    echo "partial condition exists: $prefix" >&2; exit 1
  fi
  local source_args
  if [[ "$corruption" == clean ]]; then
    source_args=(--image-root "$clean_root")
  else
    source_args=(--image-manifest "manifests/images/${dataset}_${split}_full_${corruption}_s${severity}.json" --manifest-cache-root "$cache_root")
  fi
  python src/cross_family_infer_trt.py \
    --engine-registry "manifests/engines/cross_family_v1/${dataset}_${model}_${precision}.json" \
    --annotations "$annotations" "${source_args[@]}" --out "$prediction" --input-record "$inputs" \
    --run-record "$run" --condition-id "$prefix" --dataset "$dataset" --split "$split" \
    --corruption "$corruption" --severity "$severity"
  local evaluator=src/coco_eval.py
  [[ "$dataset" == tt100k ]] && evaluator=src/tt100k_eval.py
  taskset -c 0 python "$evaluator" --annotations "$annotations" --predictions "$prediction" \
    --input-record "$inputs" --run-record "$run" --out "$metric"
}

# Absolute clean AP is generated first so catastrophic quantization floors are
# visible before any corruption-relative contrast is interpreted.
for dataset in voc kitti tt100k; do
  for model in rtdetr_l retinanet_r50_fpn_v2; do
    for precision in fp32 int8-entropy fp8; do
      run_condition "$dataset" "$model" "$precision" clean 0
    done
  done
done
for dataset in voc kitti tt100k; do
  for model in rtdetr_l retinanet_r50_fpn_v2; do
    for precision in fp32 int8-entropy fp8; do
      for corruption in fog gaussian_noise jpeg motion_blur; do
        for severity in 1 3 5; do
          run_condition "$dataset" "$model" "$precision" "$corruption" "$severity"
        done
      done
    done
  done
done
echo "CROSS-FAMILY EVALUATION QUEUE COMPLETE"
