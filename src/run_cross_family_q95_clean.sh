#!/usr/bin/env bash
set -euo pipefail
cd "${1:-/home/thuan/topic_c_ivc}"
source /home/thuan/miniconda3/etc/profile.d/conda.sh
conda activate qtsd
export PYTHONPATH=src
attempt=cross_family_q95_clean_v1
mkdir -p outputs/predictions/$attempt outputs/inputs/$attempt outputs/metrics/$attempt manifests/runs/$attempt
for dataset in voc kitti tt100k; do
  case "$dataset" in
    voc) split=val; annotations=manifests/annotations/voc_val_ultralytics_v1_coco.json ;;
    kitti) split=val; annotations=manifests/annotations/kitti_val_ultralytics_v1_coco.json ;;
    tt100k) split=test; annotations=manifests/annotations/tt100k_test_ultralytics_v1_coco.json ;;
  esac
  manifest="manifests/images/${dataset}_${split}_codec_control_q95_p0_v1.json"
  cache="data/codec_control/${dataset}"
  for model in rtdetr_l retinanet_r50_fpn_v2; do
    for precision in fp32 int8-entropy fp8; do
      id="${dataset}_${split}__${model}__${precision}__q95-clean-s0"
      prediction="outputs/predictions/$attempt/$id.json"; inputs="outputs/inputs/$attempt/$id.json"
      run="manifests/runs/$attempt/$id.json"; metric="outputs/metrics/$attempt/$id.json"
      if [[ -f "$prediction" && -f "$inputs" && -f "$run" && -f "$metric" ]]; then continue; fi
      [[ ! -e "$prediction" && ! -e "$inputs" && ! -e "$run" && ! -e "$metric" ]] || exit 1
      python src/cross_family_infer_trt.py --engine-registry "manifests/engines/cross_family_v1/${dataset}_${model}_${precision}.json" \
        --annotations "$annotations" --image-manifest "$manifest" --manifest-cache-root "$cache" \
        --out "$prediction" --input-record "$inputs" --run-record "$run" --condition-id "$id" \
        --dataset "$dataset" --split "$split" --corruption clean --severity 0
      evaluator=src/coco_eval.py; [[ "$dataset" == tt100k ]] && evaluator=src/tt100k_eval.py
      taskset -c 0 python "$evaluator" --annotations "$annotations" --predictions "$prediction" \
        --input-record "$inputs" --run-record "$run" --out "$metric"
    done
  done
done
echo "CROSS-FAMILY Q95 CLEAN COMPLETE"
