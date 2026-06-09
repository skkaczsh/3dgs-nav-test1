#!/usr/bin/env bash
set -euo pipefail

SCRIPTS_DIR="${SCRIPTS_DIR:-/root/epfs/new_route_scripts}"
STAGE_DIR="${STAGE_DIR:-/root/epfs/new_route_stage1_skymask}"
TARGET_OBJECT_DIR="${TARGET_OBJECT_DIR:-${STAGE_DIR}/target_object_fusion_0000_0999}"
RESIDUAL_ASSIGNMENT_DIR="${RESIDUAL_ASSIGNMENT_DIR:-${STAGE_DIR}/residual_surface_assignment_0000_0999}"
OUTPUT_DIR="${OUTPUT_DIR:-${STAGE_DIR}/consolidated_object_qa_0000_0999}"

TARGETS_DIR="${TARGETS_DIR:-${TARGET_OBJECT_DIR}/targets}"
OBJECTS_JSONL="${OBJECTS_JSONL:-${TARGET_OBJECT_DIR}/objects_status_fixed/objects.jsonl}"
RESIDUAL_ASSIGNMENT_PLY="${RESIDUAL_ASSIGNMENT_PLY:-${RESIDUAL_ASSIGNMENT_DIR}/residual_surface_assigned_0000_0999.ply}"
OUTPUT_PLY="${OUTPUT_PLY:-${OUTPUT_DIR}/consolidated_object_qa_0000_0999.ply}"
OUTPUT_REPORT="${OUTPUT_REPORT:-${OUTPUT_DIR}/consolidated_object_qa_0000_0999_report.json}"
OUTPUT_PREVIEW="${OUTPUT_PREVIEW:-${OUTPUT_DIR}/consolidated_object_qa_0000_0999_xy.png}"
OUTPUT_VALIDATION="${OUTPUT_VALIDATION:-${OUTPUT_DIR}/consolidated_object_qa_0000_0999_validation.json}"
MAX_PREVIEW_POINTS="${MAX_PREVIEW_POINTS:-300000}"
PREVIEW_WIDTH="${PREVIEW_WIDTH:-1800}"
PREVIEW_HEIGHT="${PREVIEW_HEIGHT:-1800}"
MIN_TOTAL_VERTICES="${MIN_TOTAL_VERTICES:-8000000}"
MIN_ABSORBED_RESIDUAL="${MIN_ABSORBED_RESIDUAL:-1000000}"

export PYTHONPATH="${SCRIPTS_DIR}:${PYTHONPATH:-}"
mkdir -p "${OUTPUT_DIR}"

python3 "${SCRIPTS_DIR}/build_consolidated_object_ply.py" \
  --targets-dir "${TARGETS_DIR}" \
  --objects-jsonl "${OBJECTS_JSONL}" \
  --residual-assignment-ply "${RESIDUAL_ASSIGNMENT_PLY}" \
  --output-ply "${OUTPUT_PLY}" \
  --output-report "${OUTPUT_REPORT}"

python3 "${SCRIPTS_DIR}/make_ply_xy_preview.py" "${OUTPUT_PLY}" \
  --output "${OUTPUT_PREVIEW}" \
  --max-points "${MAX_PREVIEW_POINTS}" \
  --width "${PREVIEW_WIDTH}" \
  --height "${PREVIEW_HEIGHT}"

python3 "${SCRIPTS_DIR}/qa_consolidated_object_ply.py" \
  --report "${OUTPUT_REPORT}" \
  --ply "${OUTPUT_PLY}" \
  --preview "${OUTPUT_PREVIEW}" \
  --min-total-vertices "${MIN_TOTAL_VERTICES}" \
  --min-absorbed-residual "${MIN_ABSORBED_RESIDUAL}" \
  --output "${OUTPUT_VALIDATION}"

echo "consolidated object QA PLY: ${OUTPUT_PLY}"
echo "consolidated object QA report: ${OUTPUT_REPORT}"
echo "consolidated object QA preview: ${OUTPUT_PREVIEW}"
echo "consolidated object QA validation: ${OUTPUT_VALIDATION}"
