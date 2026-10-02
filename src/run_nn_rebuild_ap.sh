#!/usr/bin/env bash
# AP-spread probe over rebuilt engines: run each rebuild on kitti clean-s0 +
# codec-control-s0, evaluate COCO AP, collect per-rebuild scores.
set -euo pipefail
cd /home/thuan/topic_c_ivc
PY=/home/thuan/miniconda3/envs/qtsd/bin/python
ROOT=outputs/rebuilds/nn_rebuild_variance_v1_20260930
ANN=manifests/annotations/kitti_val_ultralytics_v1_coco.json
CLEAN=data/datasets/kitti
CCMAN=manifests/images/kitti_val_codec_control_q95_p0_v1.json
CCROOT=data/codec_control/kitti
OUTP=outputs/rebuilds/nn_rebuild_variance_v1_20260930/ap_eval

run_one() {
  local arm=$1 rep=$2 cond=$3
  local rdir="$ROOT/${arm}/rebuild-$(printf %02d $rep)"
  local eng="$rdir/model.engine"
  local reg="$rdir/engine_for_infer.json"
  local stem="${arm}__r${rep}__${cond}"
  local pred="$OUTP/predictions/${stem}.json"
  local inp="$OUTP/inputs/${stem}.json"
  local run="$OUTP/runs/${stem}.json"
  local met="$OUTP/metrics/${stem}.json"
  if [[ -f "$met" ]]; then echo "SKIP $stem"; return 0; fi
  mkdir -p "$OUTP/predictions" "$OUTP/inputs" "$OUTP/runs" "$OUTP/metrics"
  python3 - "$rdir" "$reg" <<'PY'
import json, sys
rdir, out = sys.argv[1], sys.argv[2]
rec = json.load(open(f"{rdir}/record.json"))
orig = json.load(open("outputs/nn_paired_protocol_v1_20260924/engines/" + rec["arm"] + "/trt/engine.json"))
manifest = {k: orig[k] for k in ("dataset","model","imgsz","input_names","output_names","decoder")}
manifest.update(engine=rec["engine"], engine_sha256=rec["engine_sha256"],
                precision=orig.get("precision"), build_policy=orig.get("build_policy", {}))
import hashlib
payload = json.dumps(manifest, indent=1) + "\n"
open(out, "w").write(payload)
open(out + ".complete", "w").write(hashlib.sha256(payload.encode()).hexdigest() + "\n")
PY
  local src_args
  if [[ "$cond" == "clean" ]]; then
    src_args=(--image-root "$CLEAN")
  else
    src_args=(--image-manifest "$CCMAN" --manifest-cache-root "$CCROOT")
  fi
  $PY src/cross_family_infer_trt.py \
    --engine-registry "$reg" --annotations "$ANN" "${src_args[@]}" \
    --out "$pred" --input-record "$inp" --run-record "$run" \
    --condition-id "$stem" --dataset kitti --split val \
    --corruption "$cond" --severity 0 --confidence 0.05 \
    >"$OUTP/${stem}.log" 2>&1
  taskset -c 0 $PY src/coco_eval.py --annotations "$ANN" --predictions "$pred" \
    --input-record "$inp" --run-record "$run" --out "$met" \
    >>"$OUTP/${stem}.log" 2>&1
  echo "DONE $stem"
}

for arm in kitti_retinanet_int8-matched512 kitti_retinanet_int8-selective512; do
  for rep in 1 2 3 4 5; do
    run_one "$arm" "$rep" clean
    run_one "$arm" "$rep" codec-control
  done
done
echo "AP-SPREAD COMPLETE"
