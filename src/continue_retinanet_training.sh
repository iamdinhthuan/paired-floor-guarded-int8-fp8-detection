#!/usr/bin/env bash
set -euo pipefail

project_root=${1:-/home/thuan/topic_c_ivc}
cd "$project_root"

exec 9>outputs/logs/cross_family_retinanet_sequence.lock
flock -n 9 || { echo "another RetinaNet sequence watcher is active"; exit 0; }

registry_complete() {
  local registry=$1
  local marker="${registry}.complete"
  [[ -f "$registry" && -f "$marker" ]] || return 1
  [[ "$(sha256sum "$registry" | awk '{print $1}')" == "$(tr -d '[:space:]' < "$marker")" ]]
}

kitti_registry=manifests/training/kitti_retinanet_r50_fpn_v2_s20260807_v1.json
while ! registry_complete "$kitti_registry"; do
  if ! pgrep -f "python src/train_retinanet_dataset.py.*--dataset kitti" >/dev/null; then
    echo "KITTI stopped without a valid completion registry; refusing to start TT100K" >&2
    exit 1
  fi
  sleep 60
done

tt_registry=manifests/training/tt100k_retinanet_r50_fpn_v2_s20260807_v1.json
if registry_complete "$tt_registry"; then
  echo "TT100K already complete"
  exit 0
fi
if pgrep -f "python src/train_retinanet_dataset.py" >/dev/null; then
  echo "another RetinaNet trainer remains active; refusing overlap" >&2
  exit 1
fi
if [[ -d outputs/training/tt100k/cross_family_retinanet_r50_fpn_v2_tt100k_s20260807_v1 ]]; then
  echo "partial TT100K output exists; explicit resume is required" >&2
  exit 1
fi

source /home/thuan/miniconda3/etc/profile.d/conda.sh
conda activate qtsd
exec env PYTHONPATH=src python src/train_retinanet_dataset.py \
  --project-root . \
  --profile configs/training/cross_family_retinanet_r50_fpn_v2_v1.json \
  --dataset tt100k \
  --data-yaml configs/datasets/tt100k_ultralytics_v1.yaml \
  --acquisition-registry manifests/datasets/tt100k_acquisition_v1.json \
  --run-id cross_family_retinanet_r50_fpn_v2_tt100k_s20260807_v1 \
  --registry-out "$tt_registry" \
  >> outputs/logs/cross_family_retinanet_tt100k_training.log 2>&1
