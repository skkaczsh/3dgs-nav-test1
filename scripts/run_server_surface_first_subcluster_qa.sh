#!/usr/bin/env bash
set -euo pipefail

SCRIPTS_DIR="${SCRIPTS_DIR:-/root/epfs/new_route_scripts}"
STAGE_DIR="${STAGE_DIR:-/root/epfs/new_route_stage1_skymask}"
TARGET_OBJECT_DIR="${TARGET_OBJECT_DIR:-${STAGE_DIR}/target_object_fusion_0000_0999}"
INPUT_PLY="${INPUT_PLY:-${TARGET_OBJECT_DIR}/objects/object_centroids.ply}"
OUTPUT_DIR="${OUTPUT_DIR:-${STAGE_DIR}/surface_first_subcluster_qa_0000_0999}"
OUTPUT_PLY="${OUTPUT_PLY:-${OUTPUT_DIR}/object_points_surface_first_subcluster.ply}"
OUTPUT_REPORT="${OUTPUT_REPORT:-${OUTPUT_DIR}/surface_first_subcluster_report.json}"
OUTPUT_VOXEL_PLY="${OUTPUT_VOXEL_PLY:-${OUTPUT_DIR}/object_points_surface_first_subcluster_voxel004.ply}"
OUTPUT_PREVIEW="${OUTPUT_PREVIEW:-${OUTPUT_DIR}/object_points_surface_first_subcluster_xy.png}"
VOXEL_SIZE="${VOXEL_SIZE:-0.04}"
MAX_PREVIEW_POINTS="${MAX_PREVIEW_POINTS:-900000}"

export PYTHONPATH="${SCRIPTS_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_DIR}"

if [[ ! -f "${INPUT_PLY}" ]]; then
  echo "missing INPUT_PLY: ${INPUT_PLY}" >&2
  echo "set INPUT_PLY to an ASCII PLY with x y z red green blue object semantic frame" >&2
  exit 2
fi

python3 "${SCRIPTS_DIR}/surface_first_subcluster_relabel_object_ply.py" \
  --input-ply "${INPUT_PLY}" \
  --output-ply "${OUTPUT_PLY}" \
  --report "${OUTPUT_REPORT}"

python3 "${SCRIPTS_DIR}/fast_voxel_downsample_ply.py" \
  "${OUTPUT_PLY}" \
  "${OUTPUT_VOXEL_PLY}" \
  --voxel-size "${VOXEL_SIZE}"

python3 "${SCRIPTS_DIR}/make_ply_xy_preview.py" \
  "${OUTPUT_PLY}" \
  --output "${OUTPUT_PREVIEW}" \
  --max-points "${MAX_PREVIEW_POINTS}"

echo "surface-first subcluster PLY: ${OUTPUT_PLY}"
echo "surface-first subcluster voxel PLY: ${OUTPUT_VOXEL_PLY}"
echo "surface-first subcluster report: ${OUTPUT_REPORT}"
echo "surface-first subcluster preview: ${OUTPUT_PREVIEW}"
