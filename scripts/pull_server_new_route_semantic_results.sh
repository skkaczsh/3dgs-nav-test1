#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-scan-vlm}"
REMOTE_STAGE="${REMOTE_STAGE:-/root/epfs/new_route_stage1_skymask}"
LOCAL_OUT="${LOCAL_OUT:-/Users/skkac/Work/SCAN/server_new_route_semantic_manifest_sample}"
SSH_OPTS="${SSH_OPTS:-}"
SCP_OPTS="${SCP_OPTS:-$SSH_OPTS}"

mkdir -p "$LOCAL_OUT"
LOCAL_EVAL="$LOCAL_OUT/semantic_eval_0000_0500"
LOCAL_PROJ="$LOCAL_OUT/semantic_projection_manifest_sample"
LOCAL_PROJ_SAM3_SKY="$LOCAL_OUT/semantic_projection_manifest_sample_sam3_sky"
mkdir -p "$LOCAL_EVAL" "$LOCAL_PROJ" "$LOCAL_PROJ_SAM3_SKY"

scp \
  ${SCP_OPTS:+$SCP_OPTS} \
  "$SERVER:$REMOTE_STAGE/semantic_eval_0000_0500/report.json" \
  "$SERVER:$REMOTE_STAGE/semantic_eval_0000_0500/manifest.json" \
  "$LOCAL_EVAL/"

scp \
  ${SCP_OPTS:+$SCP_OPTS} \
  "$SERVER:$REMOTE_STAGE/semantic_projection_manifest_sample/semantic_projection_report.json" \
  "$SERVER:$REMOTE_STAGE/semantic_projection_manifest_sample/semantic_frame_*.ply" \
  "$SERVER:$REMOTE_STAGE/semantic_projection_manifest_sample/semantic_points_manifest_sample.ply" \
  "$LOCAL_PROJ/"

scp \
  ${SCP_OPTS:+$SCP_OPTS} \
  "$SERVER:$REMOTE_STAGE/semantic_projection_manifest_sample_sam3_sky/semantic_projection_report.json" \
  "$SERVER:$REMOTE_STAGE/semantic_projection_manifest_sample_sam3_sky/semantic_frame_*.ply" \
  "$SERVER:$REMOTE_STAGE/semantic_projection_manifest_sample_sam3_sky/semantic_points_manifest_sample_sam3_sky.ply" \
  "$LOCAL_PROJ_SAM3_SKY/"

scp \
  ${SCP_OPTS:+$SCP_OPTS} \
  "$SERVER:$REMOTE_STAGE/qa_summary_with_semantic.json" \
  "$LOCAL_OUT/"

for ply in "$LOCAL_PROJ"/semantic_frame_*.ply; do
  [ -e "$ply" ] || continue
  python3 /Users/skkac/Work/SCAN/scripts/make_ply_xy_preview.py \
    "$ply" \
    --output "${ply%.ply}.xy_preview.png" \
    --max-points 800000
done

if [ -e "$LOCAL_PROJ/semantic_points_manifest_sample.ply" ]; then
  python3 /Users/skkac/Work/SCAN/scripts/make_ply_xy_preview.py \
    "$LOCAL_PROJ/semantic_points_manifest_sample.ply" \
    --output "$LOCAL_PROJ/semantic_points_manifest_sample.xy_preview.png" \
    --max-points 800000
fi

for ply in "$LOCAL_PROJ_SAM3_SKY"/semantic_frame_*.ply; do
  [ -e "$ply" ] || continue
  python3 /Users/skkac/Work/SCAN/scripts/make_ply_xy_preview.py \
    "$ply" \
    --output "${ply%.ply}.xy_preview.png" \
    --max-points 800000
done

if [ -e "$LOCAL_PROJ_SAM3_SKY/semantic_points_manifest_sample_sam3_sky.ply" ]; then
  python3 /Users/skkac/Work/SCAN/scripts/make_ply_xy_preview.py \
    "$LOCAL_PROJ_SAM3_SKY/semantic_points_manifest_sample_sam3_sky.ply" \
    --output "$LOCAL_PROJ_SAM3_SKY/semantic_points_manifest_sample_sam3_sky.xy_preview.png" \
    --max-points 800000
fi

python3 /Users/skkac/Work/SCAN/scripts/qa_new_route_outputs.py \
  --stage-dir /Users/skkac/Work/SCAN/server_new_route_skymask_0000_0500 \
  --semantic-dir "$LOCAL_EVAL" \
  --semantic-projection-dir "$LOCAL_PROJ_SAM3_SKY" \
  --output "$LOCAL_OUT/qa_summary_local_with_semantic.json"

python3 /Users/skkac/Work/SCAN/scripts/validate_semantic_results.py \
  --qa "$LOCAL_OUT/qa_summary_local_with_semantic.json" \
  --projection "$LOCAL_PROJ/semantic_projection_report.json" \
  --projection "$LOCAL_PROJ_SAM3_SKY/semantic_projection_report.json" \
  --output "$LOCAL_OUT/semantic_acceptance_report.json"

echo "$LOCAL_OUT"
