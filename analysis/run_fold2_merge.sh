#!/usr/bin/env bash
# One-shot fold-2 merge: pull VOC bootstrap + collect cell points + regenerate.
set -euo pipefail
REMOTE="thuan@100.111.139.103"
RROOT="/home/thuan/topic_c_ivc"
RPY="/home/thuan/miniconda3/envs/qtsd/bin/python"
ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
SUP="$ROOT/submission_support_20260911"
F="$SUP/phase_f_fold2_results"

# 1) pull bootstrap artifacts
rsync -az "$REMOTE:$RROOT/outputs/analysis/nn_corruptcalib_fold2_v1_20260925/bootstrap/" \
  "$F/bootstrap/"

# 2) collect fold-2 cell points (both datasets) on remote
ssh "$REMOTE" "cd $RROOT && $RPY analysis/collect_cell_points.py \
  --bootstrap-dir outputs/analysis/nn_corruptcalib_fold2_v1_20260925/bootstrap \
  --out /tmp/fold2_cell_points.json"
scp -q "$REMOTE:/tmp/fold2_cell_points.json" "$F/cell_points.json"

# 3) arm-level points for the RetinaNet arms table
ssh "$REMOTE" "cd $RROOT && $RPY analysis/compute_arm_points.py \
  --attempt nn_corruptcalib_fold2_v1_20260925 \
  --arms int8-cc2calib512,int8-sel-cc2calib512 \
  --out /tmp/cc2calib_points.json"
scp -q "$REMOTE:/tmp/cc2calib_points.json" "$SUP/phase_cd_results/cc2calib_points.json"

# 4) regenerate statistics and all derived manuscript artifacts
cd "$ROOT/analysis"
python3 nn_final_stats.py
python3 build_nn_tables.py
echo "fold-2 merge complete"
