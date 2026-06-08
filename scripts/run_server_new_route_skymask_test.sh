#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-scan-vlm}"
REMOTE_SCRIPTS="${REMOTE_SCRIPTS:-/root/epfs/new_route_scripts}"
PY="${PY:-/root/epfs/conda_envs/vlm_seg/bin/python}"
SSH_OPTS="${SSH_OPTS:-}"
SCP_OPTS="${SCP_OPTS:-$SSH_OPTS}"

REMOTE_ENV='
export SCAN_DATA_DIR=/root/epfs/new_route_data
export SCAN_IMAGE_DIR=/root/epfs/new_route_data/calib
export SCAN_VIDEO_DIR=/root/epfs/new_route_data/video
export SCAN_EXTRACTED_DIR=/root/epfs/new_route_data/ply
export SCAN_STAGE1_DIR=/root/epfs/new_route_stage1_skymask
'

ssh $SSH_OPTS "$SERVER" "mkdir -p '$REMOTE_SCRIPTS'"
scp \
  ${SCP_OPTS:+$SCP_OPTS} \
  /Users/skkac/Work/SCAN/scripts/config.py \
  /Users/skkac/Work/SCAN/scripts/extract_frames.py \
  /Users/skkac/Work/SCAN/scripts/project_color.py \
  /Users/skkac/Work/SCAN/scripts/merge_pointcloud.py \
  /Users/skkac/Work/SCAN/scripts/fast_voxel_downsample_ply.py \
  /Users/skkac/Work/SCAN/scripts/qa_new_route_outputs.py \
  "$SERVER:$REMOTE_SCRIPTS/"

ssh $SSH_OPTS "$SERVER" "set -euo pipefail
$REMOTE_ENV
$PY -m py_compile \
  $REMOTE_SCRIPTS/config.py \
  $REMOTE_SCRIPTS/extract_frames.py \
  $REMOTE_SCRIPTS/project_color.py \
  $REMOTE_SCRIPTS/merge_pointcloud.py \
  $REMOTE_SCRIPTS/fast_voxel_downsample_ply.py \
  $REMOTE_SCRIPTS/qa_new_route_outputs.py
$PY $REMOTE_SCRIPTS/extract_frames.py --start 0 --end 500 --workers 16 --skip-existing
$PY $REMOTE_SCRIPTS/project_color.py \
  --start 0 --end 500 \
  --workers 16 \
  --max-points 50000 \
  --sky-mask-dir /root/epfs/new_route_data/sky_masks_color
$PY $REMOTE_SCRIPTS/merge_pointcloud.py \
  --start 0 --end 500 \
  --output /root/epfs/new_route_stage1_skymask/merged_0000_0500_skymask_novoxel.ply \
  --voxel-size 0 \
  --no-cpp
$PY $REMOTE_SCRIPTS/fast_voxel_downsample_ply.py \
  /root/epfs/new_route_stage1_skymask/merged_0000_0500_skymask_novoxel.ply \
  /root/epfs/new_route_stage1_skymask/merged_0000_0500_skymask_v004_fast.ply \
  --voxel-size 0.04
$PY $REMOTE_SCRIPTS/qa_new_route_outputs.py \
  --stage-dir /root/epfs/new_route_stage1_skymask \
  --output /root/epfs/new_route_stage1_skymask/qa_summary.json
"
