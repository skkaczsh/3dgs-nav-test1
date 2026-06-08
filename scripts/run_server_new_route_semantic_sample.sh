#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-scan-vlm}"
REMOTE_SCRIPTS="${REMOTE_SCRIPTS:-/root/epfs/new_route_scripts}"
PY="${PY:-/root/epfs/conda_envs/vlm_seg/bin/python}"
RUN_EVAL_LOCAL="${RUN_EVAL_LOCAL:-/Users/skkac/Work/SCAN/MT20260511-165822/semantic_eval/run_eval.py}"
SSH_OPTS="${SSH_OPTS:-}"
SCP_OPTS="${SCP_OPTS:-$SSH_OPTS}"

ssh $SSH_OPTS "$SERVER" "mkdir -p '$REMOTE_SCRIPTS'"
scp \
  ${SCP_OPTS:+$SCP_OPTS} \
  /Users/skkac/Work/SCAN/scripts/make_new_route_semantic_manifest.py \
  /Users/skkac/Work/SCAN/scripts/qa_new_route_outputs.py \
  /Users/skkac/Work/SCAN/scripts/project_color.py \
  /Users/skkac/Work/SCAN/scripts/project_semantic.py \
  "$RUN_EVAL_LOCAL" \
  "$SERVER:$REMOTE_SCRIPTS/"

ssh $SSH_OPTS "$SERVER" "set -euo pipefail
OUT=/root/epfs/new_route_stage1_skymask/semantic_eval_0000_0500
$PY -m py_compile \
  $REMOTE_SCRIPTS/make_new_route_semantic_manifest.py \
  $REMOTE_SCRIPTS/run_eval.py \
  $REMOTE_SCRIPTS/qa_new_route_outputs.py \
  $REMOTE_SCRIPTS/project_color.py \
  $REMOTE_SCRIPTS/project_semantic.py
$PY $REMOTE_SCRIPTS/make_new_route_semantic_manifest.py \
  --frames-dir /root/epfs/new_route_stage1_skymask/frames \
  --sky-mask-dir /root/epfs/new_route_data/sky_masks_color \
  --output \$OUT/manifest.json \
  --start 0 --end 500 --count 15 --cams 0 1 2 --require-sky-mask
$PY $REMOTE_SCRIPTS/run_eval.py \
  --manifest \$OUT/manifest.json \
  --output-dir \$OUT \
  --sam-masks-dir /root/epfs/new_route_data/sam_masks_missing \
  --combos sky_sam3_rules_qwen_review sam3_sky_rules_qwen_review \
  --vlm-endpoint http://localhost:8001/v1/chat/completions \
  --vlm-model Qwen3.6-35B-A3B \
  --vlm-timeout 180 \
  --vlm-chunk-size 12 \
  --vlm-max-tokens 2048 \
  --min-area 800
export SCAN_DATA_DIR=/root/epfs/new_route_data
export SCAN_IMAGE_DIR=/root/epfs/new_route_data/calib
export SCAN_VIDEO_DIR=/root/epfs/new_route_data/video
export SCAN_EXTRACTED_DIR=/root/epfs/new_route_data/ply
export SCAN_STAGE1_DIR=/root/epfs/new_route_stage1_skymask
$PY $REMOTE_SCRIPTS/project_semantic.py \
  --semantic-eval-dir \$OUT \
  --combo sky_sam3_rules_qwen_review \
  --output-dir /root/epfs/new_route_stage1_skymask/semantic_projection_manifest_sample \
  --frames-from-semantic-dir \
  --max-points 50000 \
  --write-ply \
  --write-merged-ply \
  --merged-name semantic_points_manifest_sample.ply
$PY $REMOTE_SCRIPTS/project_semantic.py \
  --semantic-eval-dir \$OUT \
  --combo sam3_sky_rules_qwen_review \
  --output-dir /root/epfs/new_route_stage1_skymask/semantic_projection_manifest_sample_sam3_sky \
  --frames-from-semantic-dir \
  --max-points 50000 \
  --write-ply \
  --write-merged-ply \
  --merged-name semantic_points_manifest_sample_sam3_sky.ply
$PY $REMOTE_SCRIPTS/qa_new_route_outputs.py \
  --stage-dir /root/epfs/new_route_stage1_skymask \
  --semantic-dir \$OUT \
  --semantic-projection-dir /root/epfs/new_route_stage1_skymask/semantic_projection_manifest_sample_sam3_sky \
  --output /root/epfs/new_route_stage1_skymask/qa_summary_with_semantic.json
"
