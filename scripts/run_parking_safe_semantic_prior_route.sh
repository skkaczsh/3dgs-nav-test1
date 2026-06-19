#!/usr/bin/env bash
set -euo pipefail

# Run the conservative parking-lot semantic-prior route.
#
# This route is intentionally stricter than the earlier global reverse-projection
# probes:
#   1. global raw PLY must carry source-frame metadata;
#   2. geometry guidance keeps only points observed near the image frame;
#   3. semantic priors fill residual surface holes only by default;
#   4. fine labels such as car/railing are not overwritten unless explicitly
#      requested with ALLOW_FINE_SURFACE_OVERRIDE=1.
#
# Default mode is dry-run. Use RUN=1 to execute. Existing output directories are
# protected unless OVERWRITE=1 is set.

REPO_ROOT="${REPO_ROOT:-/home/zsh/Work/SCAN/new_route}"
WORK_DIR="${WORK_DIR:-/home/zsh/Work/SCAN/work_MT20260616-175807}"
PYTHON="${PYTHON:-/home/zsh/Work/SCAN/.venvs/scan-semantic/bin/python}"
DATASET_DIR="${DATASET_DIR:-/home/zsh/Work/SCAN/datasets/MT20260616-175807}"
IMAGE_DIR="${IMAGE_DIR:-${DATASET_DIR}/image}"

RUN="${RUN:-0}"
OVERWRITE="${OVERWRITE:-0}"
BUILD_TARGETS="${BUILD_TARGETS:-0}"
BUILD_OBJECTS="${BUILD_OBJECTS:-${BUILD_TARGETS}}"
RUN_QA="${RUN_QA:-${BUILD_OBJECTS}}"
ALLOW_FINE_SURFACE_OVERRIDE="${ALLOW_FINE_SURFACE_OVERRIDE:-0}"

START="${START:-3400}"
END="${END:-3500}"
STRIDE="${STRIDE:-10}"
CAMS="${CAMS:-0 1 2}"
OUT_SUFFIX="${OUT_SUFFIX:-guarded_semprior_safe}"

LX="${LX:-${DATASET_DIR}/MANIFOLD_MT20260616-175807.lx}"
FRAME_ROOT="${FRAME_ROOT:-${WORK_DIR}/frames_jpeg_s10}"
PRIORITY_DIR="${PRIORITY_DIR:-${WORK_DIR}/priority_surface_mapillary_s10_rtx5070}"
RAW_PLY="${RAW_PLY:-${WORK_DIR}/raw_lx_voxel_full_v001_meta/raw_points_full_voxel001_meta.ply}"
SEMANTIC_PRIOR_PLY="${SEMANTIC_PRIOR_PLY:-${WORK_DIR}/frame_object_viewer_best_p008_split_lowplanar_surface_consolidated_localgeom_rtx5070_default_localgeom_20260619_132858/frame_object_points_stride10.ply}"

GLOBAL_SOURCE_FRAME_WINDOW="${GLOBAL_SOURCE_FRAME_WINDOW:-20}"
GLOBAL_SOURCE_FILTER_MODE="${GLOBAL_SOURCE_FILTER_MODE:-mean}"
PRIOR_VOXEL_SIZE="${PRIOR_VOXEL_SIZE:-0.20}"
PRIOR_NEIGHBOR_RADIUS="${PRIOR_NEIGHBOR_RADIUS:-1}"
EDGE_DEPTH_THRESHOLD="${EDGE_DEPTH_THRESHOLD:-0.35}"
COLOR_EDGE_LAB_THRESHOLD="${COLOR_EDGE_LAB_THRESHOLD:-16.0}"

GEOMETRY_DIR="${GEOMETRY_DIR:-${WORK_DIR}/geometry_guidance_${OUT_SUFFIX}_${START}_${END}}"
REFINE_DIR="${REFINE_DIR:-${WORK_DIR}/geometry_refine_${OUT_SUFFIX}_${START}_${END}}"
PROJECTION_DIR="${PROJECTION_DIR:-${WORK_DIR}/priority_projection_${OUT_SUFFIX}_${START}_${END}}"
TARGET_DIR="${TARGET_DIR:-${WORK_DIR}/frame_targets_${OUT_SUFFIX}_${START}_${END}}"
OBJECT_DIR="${OBJECT_DIR:-${WORK_DIR}/frame_objects_${OUT_SUFFIX}_${START}_${END}}"
VIEWER_DIR="${VIEWER_DIR:-${WORK_DIR}/frame_object_viewer_${OUT_SUFFIX}_${START}_${END}}"

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

require_dir() {
  if [[ ! -d "$1" ]]; then
    echo "missing required directory: $1" >&2
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

cam_args() {
  # shellcheck disable=SC2086
  printf '%s\n' ${CAMS}
}

main() {
  cd "${REPO_ROOT}"
  export PYTHONPATH="${REPO_ROOT}/scripts${PYTHONPATH:+:${PYTHONPATH}}"
  export SCAN_IMAGE_DIR="${SCAN_IMAGE_DIR:-${IMAGE_DIR}}"
  export SCAN_VIDEO_DIR="${SCAN_VIDEO_DIR:-${IMAGE_DIR}}"

  require_file "${PYTHON}"
  require_file "${LX}"
  require_file "${RAW_PLY}"
  require_file "${SEMANTIC_PRIOR_PLY}"
  require_dir "${FRAME_ROOT}"
  require_dir "${PRIORITY_DIR}"

  echo "repo=${REPO_ROOT}"
  echo "work_dir=${WORK_DIR}"
  echo "scan_image_dir=${SCAN_IMAGE_DIR}"
  echo "scan_video_dir=${SCAN_VIDEO_DIR}"
  echo "run=${RUN}"
  echo "overwrite=${OVERWRITE}"
  echo "range=${START}..${END} stride=${STRIDE} cams=${CAMS}"
  echo "out_suffix=${OUT_SUFFIX}"
  echo "global_source_filter=${GLOBAL_SOURCE_FILTER_MODE} window=${GLOBAL_SOURCE_FRAME_WINDOW}"
  echo "allow_fine_surface_override=${ALLOW_FINE_SURFACE_OVERRIDE}"

  prepare_output_dir "${GEOMETRY_DIR}"
  prepare_output_dir "${REFINE_DIR}"
  prepare_output_dir "${PROJECTION_DIR}"
  if [[ "${BUILD_TARGETS}" == "1" ]]; then
    prepare_output_dir "${TARGET_DIR}"
  fi
  if [[ "${BUILD_OBJECTS}" == "1" ]]; then
    prepare_output_dir "${OBJECT_DIR}"
    prepare_output_dir "${VIEWER_DIR}"
  fi

  mapfile -t cams_array < <(cam_args)

  run_cmd "${PYTHON}" scripts/build_geometry_guidance_maps.py \
    --global-colored-ply "${RAW_PLY}" \
    --global-source-filter-mode "${GLOBAL_SOURCE_FILTER_MODE}" \
    --global-source-frame-window "${GLOBAL_SOURCE_FRAME_WINDOW}" \
    --frame-root "${FRAME_ROOT}" \
    --semantic-prior-ply "${SEMANTIC_PRIOR_PLY}" \
    --output-dir "${GEOMETRY_DIR}" \
    --start "${START}" \
    --end "${END}" \
    --stride "${STRIDE}" \
    --cams "${cams_array[@]}" \
    --edge-depth-threshold "${EDGE_DEPTH_THRESHOLD}" \
    --color-edge-lab-threshold "${COLOR_EDGE_LAB_THRESHOLD}" \
    --prior-voxel-size "${PRIOR_VOXEL_SIZE}" \
    --prior-neighbor-radius "${PRIOR_NEIGHBOR_RADIUS}" \
    --mark-invalid-boundary

  refine_args=()
  if [[ "${ALLOW_FINE_SURFACE_OVERRIDE}" == "1" ]]; then
    echo "warning: ALLOW_FINE_SURFACE_OVERRIDE=1 may overwrite car/railing evidence; use only for diagnostics" >&2
    refine_args+=(--guarded-fine-surface-override)
  fi

  run_cmd "${PYTHON}" scripts/refine_priority_masks_with_geometry.py \
    --frame-root "${FRAME_ROOT}" \
    --priority-dir "${PRIORITY_DIR}" \
    --geometry-dir "${GEOMETRY_DIR}" \
    --output-dir "${REFINE_DIR}" \
    --start "${START}" \
    --end "${END}" \
    --stride "${STRIDE}" \
    --cams "${cams_array[@]}" \
    --surface-override-from 0 \
    --min-fine-component-area 24 \
    --component-min-area 80 \
    --max-review-panels 36 \
    "${refine_args[@]}"

  run_cmd "${PYTHON}" scripts/project_priority_masks_to_lx.py \
    --lx "${LX}" \
    --frame-root "${FRAME_ROOT}" \
    --priority-dir "${REFINE_DIR}" \
    --priority-suffix "_priority_refined" \
    --output-dir "${PROJECTION_DIR}" \
    --start "${START}" \
    --end "${END}" \
    --stride "${STRIDE}" \
    --cams "${cams_array[@]}"

  if [[ "${BUILD_TARGETS}" == "1" ]]; then
    run_cmd "${PYTHON}" scripts/build_frame_targets_from_priority.py \
      --lx "${LX}" \
      --frame-root "${FRAME_ROOT}" \
      --priority-dir "${REFINE_DIR}" \
      --priority-suffix "_priority_refined" \
      --output-dir "${TARGET_DIR}" \
      --start "${START}" \
      --end "${END}" \
      --stride "${STRIDE}" \
      --cams "${cams_array[@]}" \
      --labels ground wall grass car railing \
      --split-by-image-components \
      --split-by-depth-support \
      --resume
  fi

  if [[ "${BUILD_OBJECTS}" == "1" ]]; then
    require_file "${TARGET_DIR}/frame_targets.jsonl"
    require_file "${TARGET_DIR}/frame_targets.ply"
    run_cmd "${PYTHON}" scripts/fuse_targets_to_objects.py \
      --targets "${TARGET_DIR}/frame_targets.jsonl" \
      --output-dir "${OBJECT_DIR}" \
      --fallback-zone-scan \
      --write-ply
    run_cmd "${PYTHON}" scripts/export_frame_target_objects_for_viewer.py \
      --targets-jsonl "${TARGET_DIR}/frame_targets.jsonl" \
      --target-ply "${TARGET_DIR}/frame_targets.ply" \
      --objects-jsonl "${OBJECT_DIR}/objects.jsonl" \
      --output-dir "${VIEWER_DIR}" \
      --stride "${STRIDE}"
  fi

  if [[ "${RUN_QA}" == "1" && "${BUILD_OBJECTS}" == "1" ]]; then
    run_cmd "${PYTHON}" scripts/qa_viewer_candidate.py \
      --ply "${VIEWER_DIR}/frame_object_points_stride10.ply" \
      --objects-jsonl "${VIEWER_DIR}/frame_objects_viewer.jsonl" \
      --output-json "${VIEWER_DIR}/viewer_candidate_qa.json" \
      --output-md "${VIEWER_DIR}/viewer_candidate_qa.md"
  fi

  echo "geometry_dir=${GEOMETRY_DIR}"
  echo "refine_dir=${REFINE_DIR}"
  echo "projection_dir=${PROJECTION_DIR}"
  if [[ "${BUILD_TARGETS}" == "1" ]]; then
    echo "target_dir=${TARGET_DIR}"
  fi
  if [[ "${BUILD_OBJECTS}" == "1" ]]; then
    echo "viewer_dir=${VIEWER_DIR}"
  fi
}

main "$@"
