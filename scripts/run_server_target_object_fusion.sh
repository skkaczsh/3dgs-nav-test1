#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEMANTIC_EVAL_DIR="${SEMANTIC_EVAL_DIR:-/root/epfs/manifold_3dgs_project/processed/semantic_eval_same_section_10x3}"
COMBO="${COMBO:-sam2_prompt_v3_sky_label_merge_completion}"
COLOR_DIR="${COLOR_DIR:-/root/epfs/new_route_stage1_skymask/output}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/epfs/new_route_stage1_skymask/target_object_fusion_0000_0999}"
START_FRAME="${START_FRAME:-0}"
END_FRAME="${END_FRAME:-999}"
VOXEL_SIZE="${VOXEL_SIZE:-0.08}"
MIN_TARGET_POINTS="${MIN_TARGET_POINTS:-20}"
WORK_MODE="${WORK_MODE:-range}"

export SCAN_DATA_DIR="${SCAN_DATA_DIR:-/root/epfs/new_route_data}"
export SCAN_IMAGE_DIR="${SCAN_IMAGE_DIR:-/root/epfs/new_route_data/calib}"
export SCAN_VIDEO_DIR="${SCAN_VIDEO_DIR:-/root/epfs/new_route_data/video}"
export SCAN_EXTRACTED_DIR="${SCAN_EXTRACTED_DIR:-/root/epfs/new_route_data/ply}"
export SCAN_STAGE1_DIR="${SCAN_STAGE1_DIR:-/root/epfs/new_route_stage1_skymask}"
export PYTHONPATH="${SCRIPT_DIR}:${PYTHONPATH:-}"

mkdir -p "${OUTPUT_DIR}"

frame_args=(--start "${START_FRAME}" --end "${END_FRAME}")
if [[ "${WORK_MODE}" == "semantic-dir" ]]; then
  frame_args+=(--frames-from-semantic-dir)
fi

python3 "${SCRIPT_DIR}/build_targets_from_masks.py" \
  --semantic-eval-dir "${SEMANTIC_EVAL_DIR}" \
  --combo "${COMBO}" \
  --color-dir "${COLOR_DIR}" \
  --output-dir "${OUTPUT_DIR}" \
  "${frame_args[@]}" \
  --voxel-size "${VOXEL_SIZE}" \
  --min-target-points "${MIN_TARGET_POINTS}" \
  --resume \
  --write-ply

python3 "${SCRIPT_DIR}/fuse_targets_to_objects.py" \
  --targets "${OUTPUT_DIR}/targets" \
  --output-dir "${OUTPUT_DIR}/objects" \
  --write-ply

python3 "${SCRIPT_DIR}/qa_target_object_fusion.py" \
  --target-report "${OUTPUT_DIR}/reports/target_build_report.json" \
  --objects-jsonl "${OUTPUT_DIR}/objects/objects.jsonl" \
  --fusion-report "${OUTPUT_DIR}/objects/fusion_report.json" \
  --zones-json "${OUTPUT_DIR}/objects/zones.json" \
  --output "${OUTPUT_DIR}/reports/target_object_qa.json"

echo "target/object fusion output: ${OUTPUT_DIR}"
