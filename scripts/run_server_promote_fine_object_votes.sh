#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

PYTHON_BIN="${PYTHON_BIN:-python3}"
ENRICHED_PLY="${ENRICHED_PLY:-}"
OUTPUT_DIR="${OUTPUT_DIR:-}"
COLOR_DIR="${COLOR_DIR:-/root/epfs/new_route_stage1_skymask/output}"
TARGET_VOXEL_SIZE="${TARGET_VOXEL_SIZE:-0.08}"
MIN_TARGET_POINTS="${MIN_TARGET_POINTS:-5}"
GLOBAL_VOXEL_SIZE="${GLOBAL_VOXEL_SIZE:-0.06}"
OBJECT_VOXEL_SIZE="${OBJECT_VOXEL_SIZE:-0.16}"
MIN_VOXEL_POINTS="${MIN_VOXEL_POINTS:-1}"
MIN_OBJECT_VOXELS="${MIN_OBJECT_VOXELS:-8}"
MIN_LABEL_PURITY="${MIN_LABEL_PURITY:-0.55}"
MIN_VOTE_CONFIDENCE="${MIN_VOTE_CONFIDENCE:-0.2}"
MAX_SIZE_WEIGHT="${MAX_SIZE_WEIGHT:-30.0}"
COLOR_MODE="${COLOR_MODE:-semantic}"

if [[ -z "${ENRICHED_PLY}" || -z "${OUTPUT_DIR}" ]]; then
  echo "ENRICHED_PLY and OUTPUT_DIR are required" >&2
  exit 2
fi

mkdir -p "${OUTPUT_DIR}"
TARGET_DIR="${OUTPUT_DIR}/frame_targets"
GLOBAL_DIR="${OUTPUT_DIR}/global_votes"

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_frame_fine_targets_from_enriched.py" \
  --enriched-ply "${ENRICHED_PLY}" \
  --colored-frame-dir "${COLOR_DIR}" \
  --output-dir "${TARGET_DIR}" \
  --voxel-size "${TARGET_VOXEL_SIZE}" \
  --min-target-points "${MIN_TARGET_POINTS}" \
  --write-ply

"${PYTHON_BIN}" "${SCRIPT_DIR}/build_global_semantic_votes.py" \
  --targets "${TARGET_DIR}" \
  --output-dir "${GLOBAL_DIR}" \
  --voxel-size "${GLOBAL_VOXEL_SIZE}" \
  --object-voxel-size "${OBJECT_VOXEL_SIZE}" \
  --min-voxel-points "${MIN_VOXEL_POINTS}" \
  --min-object-voxels "${MIN_OBJECT_VOXELS}" \
  --min-label-purity "${MIN_LABEL_PURITY}" \
  --min-vote-confidence "${MIN_VOTE_CONFIDENCE}" \
  --max-size-weight "${MAX_SIZE_WEIGHT}" \
  --color-mode "${COLOR_MODE}"

echo "frame targets: ${TARGET_DIR}"
echo "global votes: ${GLOBAL_DIR}"
