#!/usr/bin/env bash
set -euo pipefail

# Reproduce the current best parking-lot frame-local semantic route.
#
# Default mode is a dry run. Use RUN=1 to execute. Existing output directories
# are protected unless OVERWRITE=1 is set.

REPO_ROOT="${REPO_ROOT:-/home/zsh/Work/SCAN/new_route}"
WORK_DIR="${WORK_DIR:-/home/zsh/Work/SCAN/work_MT20260616-175807}"
PYTHON="${PYTHON:-/home/zsh/Work/SCAN/.venvs/scan-semantic/bin/python}"
RUN="${RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"

BASE_TARGET_DIR="${BASE_TARGET_DIR:-${WORK_DIR}/frame_targets_guarded_v3_full_s10_fine_surface_guard_rtx5070}"
OUT_SUFFIX="${OUT_SUFFIX:-rtx5070_repro}"

ABSORB_DIR="${ABSORB_DIR:-${WORK_DIR}/frame_targets_best_absorbed_demote_${OUT_SUFFIX}}"
REPAIR_DIR="${REPAIR_DIR:-${WORK_DIR}/frame_targets_best_surface_repair_p008_${OUT_SUFFIX}}"
SPLIT_DIR="${SPLIT_DIR:-${WORK_DIR}/frame_targets_best_surface_repair_p008_split_lowplanar_${OUT_SUFFIX}}"
OBJECT_DIR="${OBJECT_DIR:-${WORK_DIR}/frame_objects_best_surface_repair_p008_split_lowplanar_${OUT_SUFFIX}}"
VIEWER_DIR="${VIEWER_DIR:-${WORK_DIR}/frame_object_viewer_best_surface_repair_p008_split_lowplanar_${OUT_SUFFIX}}"
CONSOLIDATED_OBJECT_DIR="${CONSOLIDATED_OBJECT_DIR:-${WORK_DIR}/frame_objects_best_p008_split_lowplanar_surface_consolidated_${OUT_SUFFIX}}"
CONSOLIDATED_VIEWER_DIR="${CONSOLIDATED_VIEWER_DIR:-${WORK_DIR}/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_${OUT_SUFFIX}}"
FINAL_VIEWER_DIR="${FINAL_VIEWER_DIR:-${WORK_DIR}/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_ambresolved_${OUT_SUFFIX}}"
AMBSPLIT_VIEWER_DIR="${AMBSPLIT_VIEWER_DIR:-${WORK_DIR}/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_ambsplit_${OUT_SUFFIX}}"
SPLIT_AMBIGUOUS_SURFACES="${SPLIT_AMBIGUOUS_SURFACES:-1}"

STRIDE="${STRIDE:-10}"

BASE_TARGETS="${BASE_TARGET_DIR}/frame_targets_refined.jsonl"
BASE_TARGET_PLY="${BASE_TARGET_DIR}/frame_targets_refined.ply"

run_cmd() {
  printf '+'
  printf ' %q' "$@"
  printf '\n'
  if [[ "${RUN}" == "1" ]]; then
    "$@"
  fi
}

require_file() {
  if [[ ! -f "$1" ]]; then
    echo "missing required file: $1" >&2
    exit 1
  fi
}

prepare_output_dir() {
  local dir="$1"
  if [[ -e "${dir}" ]]; then
    if [[ "${OVERWRITE}" != "1" ]]; then
      echo "output exists; set OVERWRITE=1 to replace: ${dir}" >&2
      exit 1
    fi
    run_cmd rm -rf "${dir}"
  fi
  run_cmd mkdir -p "${dir}"
}

main() {
  cd "${REPO_ROOT}"

  require_file "${BASE_TARGETS}"
  require_file "${BASE_TARGET_PLY}"
  require_file "${PYTHON}"

  echo "repo=${REPO_ROOT}"
  echo "work_dir=${WORK_DIR}"
  echo "python=${PYTHON}"
  echo "run=${RUN}"
  echo "overwrite=${OVERWRITE}"
  echo "out_suffix=${OUT_SUFFIX}"

  prepare_output_dir "${ABSORB_DIR}"
  prepare_output_dir "${REPAIR_DIR}"
  prepare_output_dir "${SPLIT_DIR}"
  prepare_output_dir "${OBJECT_DIR}"
  prepare_output_dir "${VIEWER_DIR}"
  prepare_output_dir "${CONSOLIDATED_OBJECT_DIR}"
  prepare_output_dir "${CONSOLIDATED_VIEWER_DIR}"
  prepare_output_dir "${FINAL_VIEWER_DIR}"
  if [[ "${SPLIT_AMBIGUOUS_SURFACES}" == "1" ]]; then
    prepare_output_dir "${AMBSPLIT_VIEWER_DIR}"
  fi

  run_cmd "${PYTHON}" scripts/absorb_fine_fragments_into_surfaces.py \
    --targets-jsonl "${BASE_TARGETS}" \
    --output-jsonl "${ABSORB_DIR}/frame_targets_absorbed.jsonl" \
    --report "${ABSORB_DIR}/absorb_report.json" \
    --demote-unabsorbed-weak-label unknown

  run_cmd "${PYTHON}" scripts/repair_surface_target_labels.py \
    --targets-jsonl "${ABSORB_DIR}/frame_targets_absorbed.jsonl" \
    --output-jsonl "${REPAIR_DIR}/frame_targets_repaired.jsonl" \
    --report "${REPAIR_DIR}/surface_repair_report.json" \
    --horizontal-surface-min-planarity 0.08

  run_cmd "${PYTHON}" scripts/refine_frame_targets_by_geometry.py \
    --targets-jsonl "${REPAIR_DIR}/frame_targets_repaired.jsonl" \
    --target-ply "${BASE_TARGET_PLY}" \
    --output-dir "${SPLIT_DIR}" \
    --split-horizontal-wall-by-height \
    --guard-linear-ground-artifacts \
    --guard-fine-surface-artifacts \
    --surface-planarity 0.30 \
    --wall-max-normal-z 0.75 \
    --car-max-centroid-z 2.5 \
    --ceiling-min-z 2.2 \
    --surface-height-split-threshold 0.8 \
    --surface-height-bin 0.45 \
    --surface-min-split-points 800 \
    --surface-split-min-points 100 \
    --surface-split-voxel 0.16 \
    --keep-residual

  run_cmd "${PYTHON}" scripts/fuse_targets_to_objects.py \
    --targets "${SPLIT_DIR}/frame_targets_refined.jsonl" \
    --output-dir "${OBJECT_DIR}" \
    --fallback-zone-scan \
    --write-ply

  run_cmd "${PYTHON}" scripts/export_frame_target_objects_for_viewer.py \
    --targets-jsonl "${SPLIT_DIR}/frame_targets_refined.jsonl" \
    --target-ply "${SPLIT_DIR}/frame_targets_refined.ply" \
    --objects-jsonl "${OBJECT_DIR}/objects.jsonl" \
    --output-dir "${VIEWER_DIR}" \
    --stride "${STRIDE}"

  run_cmd "${PYTHON}" scripts/consolidate_same_label_surface_objects.py \
    --objects-jsonl "${OBJECT_DIR}/objects.jsonl" \
    --output-jsonl "${CONSOLIDATED_OBJECT_DIR}/objects_consolidated.jsonl" \
    --output-report "${CONSOLIDATED_OBJECT_DIR}/consolidation_report.json" \
    --output-mapping "${CONSOLIDATED_OBJECT_DIR}/object_mapping.jsonl" \
    --labels ground wall ceiling \
    --min-points 60 \
    --max-bbox-gap 0.35 \
    --max-centroid-distance 1.0 \
    --max-normal-angle 15 \
    --max-plane-distance 0.22 \
    --max-color-distance 70

  run_cmd "${PYTHON}" scripts/remap_ply_object_ids.py \
    "${VIEWER_DIR}/frame_object_points_stride10.ply" \
    "${CONSOLIDATED_OBJECT_DIR}/object_mapping.jsonl" \
    "${CONSOLIDATED_VIEWER_DIR}/frame_object_points_stride10.ply"

  run_cmd cp "${CONSOLIDATED_OBJECT_DIR}/consolidation_report.json" "${CONSOLIDATED_VIEWER_DIR}/consolidation_report.json"

  run_cmd "${PYTHON}" scripts/prepare_consolidated_viewer_objects.py \
    --objects-jsonl "${CONSOLIDATED_OBJECT_DIR}/objects_consolidated.jsonl" \
    --remap-sidecar "${CONSOLIDATED_VIEWER_DIR}/frame_object_points_stride10.ply.mapping.json" \
    --output-jsonl "${CONSOLIDATED_VIEWER_DIR}/frame_objects_viewer.jsonl" \
    --report "${CONSOLIDATED_VIEWER_DIR}/prepare_viewer_objects_report.json"

  run_cmd "${PYTHON}" scripts/resolve_ambiguous_surface_objects.py \
    --objects-jsonl "${CONSOLIDATED_VIEWER_DIR}/frame_objects_viewer.jsonl" \
    --output-jsonl "${FINAL_VIEWER_DIR}/frame_objects_viewer.jsonl" \
    --report "${FINAL_VIEWER_DIR}/ambiguous_surface_resolve_report.json" \
    --input-ply "${CONSOLIDATED_VIEWER_DIR}/frame_object_points_stride10.ply" \
    --output-ply "${FINAL_VIEWER_DIR}/frame_object_points_stride10.ply"

  run_cmd cp "${CONSOLIDATED_OBJECT_DIR}/consolidation_report.json" "${FINAL_VIEWER_DIR}/consolidation_report.json"
  run_cmd cp "${CONSOLIDATED_VIEWER_DIR}/prepare_viewer_objects_report.json" "${FINAL_VIEWER_DIR}/prepare_viewer_objects_report.json"
  run_cmd cp "${CONSOLIDATED_VIEWER_DIR}/frame_object_points_stride10.ply.mapping.json" "${FINAL_VIEWER_DIR}/frame_object_points_stride10.ply.mapping.json"

  if [[ "${SPLIT_AMBIGUOUS_SURFACES}" == "1" ]]; then
    run_cmd "${PYTHON}" scripts/split_ambiguous_surface_viewer_objects.py \
      --objects-jsonl "${FINAL_VIEWER_DIR}/frame_objects_viewer.jsonl" \
      --targets-jsonl "${SPLIT_DIR}/frame_targets_refined.jsonl" \
      --input-ply "${FINAL_VIEWER_DIR}/frame_object_points_stride10.ply" \
      --output-jsonl "${AMBSPLIT_VIEWER_DIR}/frame_objects_viewer.jsonl" \
      --output-ply "${AMBSPLIT_VIEWER_DIR}/frame_object_points_stride10.ply" \
      --report "${AMBSPLIT_VIEWER_DIR}/surface_ambiguous_split_report.json"
    run_cmd cp "${FINAL_VIEWER_DIR}/consolidation_report.json" "${AMBSPLIT_VIEWER_DIR}/consolidation_report.json"
    run_cmd cp "${FINAL_VIEWER_DIR}/ambiguous_surface_resolve_report.json" "${AMBSPLIT_VIEWER_DIR}/ambiguous_surface_resolve_report.json"
    run_cmd "${PYTHON}" scripts/qa_viewer_candidate.py \
      --ply "${AMBSPLIT_VIEWER_DIR}/frame_object_points_stride10.ply" \
      --objects-jsonl "${AMBSPLIT_VIEWER_DIR}/frame_objects_viewer.jsonl" \
      --ambiguous-report "${AMBSPLIT_VIEWER_DIR}/surface_ambiguous_split_report.json" \
      --consolidation-report "${AMBSPLIT_VIEWER_DIR}/consolidation_report.json" \
      --output-json "${AMBSPLIT_VIEWER_DIR}/viewer_candidate_qa.json" \
      --output-md "${AMBSPLIT_VIEWER_DIR}/viewer_candidate_qa.md"
    echo "final_viewer_dir=${AMBSPLIT_VIEWER_DIR}"
  else
    run_cmd "${PYTHON}" scripts/qa_viewer_candidate.py \
      --ply "${FINAL_VIEWER_DIR}/frame_object_points_stride10.ply" \
      --objects-jsonl "${FINAL_VIEWER_DIR}/frame_objects_viewer.jsonl" \
      --ambiguous-report "${FINAL_VIEWER_DIR}/ambiguous_surface_resolve_report.json" \
      --consolidation-report "${FINAL_VIEWER_DIR}/consolidation_report.json" \
      --output-json "${FINAL_VIEWER_DIR}/viewer_candidate_qa.json" \
      --output-md "${FINAL_VIEWER_DIR}/viewer_candidate_qa.md"
    echo "final_viewer_dir=${FINAL_VIEWER_DIR}"
  fi
}

main "$@"
