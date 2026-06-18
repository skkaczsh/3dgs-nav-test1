#!/usr/bin/env bash
set -euo pipefail

# Rebuild the current parking candidate surface route on the RTX 5070Ti workspace.
#
# This script intentionally starts from the already validated guarded_v2 frame
# targets. It does not rerun expensive priority segmentation/projection. It
# rebuilds only the cheap surface-object branches:
#
#   baseline strict surface
#   strict surface + object relabel
#   ground artifact target guard + strict surface
#   ground artifact target guard + object relabel
#
# Outputs include viewer PLY/JSONL, frame-local QA packs, and a full-risk
# comparison report.

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
REPO_DIR="$(cd "${SCRIPT_DIR}/.." && pwd)"

WORK="${WORK:-/home/zsh/Work/SCAN/work_MT20260616-175807}"
VENV="${VENV:-/home/zsh/Work/SCAN/.venvs/scan-semantic}"
PY="${PY:-${VENV}/bin/python}"
SOURCE_TARGET_DIR="${SOURCE_TARGET_DIR:-${WORK}/frame_targets_guarded_v2_full_s10_geometry_ceiling_rtx5070}"
SOURCE_TARGETS_JSONL="${SOURCE_TARGETS_JSONL:-${SOURCE_TARGET_DIR}/frame_targets_refined.jsonl}"
SOURCE_TARGET_PLY="${SOURCE_TARGET_PLY:-${SOURCE_TARGET_DIR}/frame_targets_refined.ply}"
VIEWER_STRIDE="${VIEWER_STRIDE:-10}"
QA_CANDIDATE_LIMIT="${QA_CANDIDATE_LIMIT:-160}"
QA_EVIDENCE_PER_OBJECT="${QA_EVIDENCE_PER_OBJECT:-3}"
FORCE="${FORCE:-0}"
CHECK_ONLY="${CHECK_ONLY:-0}"

STRICT_OBJECT_DIR="${STRICT_OBJECT_DIR:-${WORK}/frame_objects_guarded_v2_full_s10_strict_surface_rtx5070}"
STRICT_VIEWER_DIR="${STRICT_VIEWER_DIR:-${WORK}/frame_object_viewer_guarded_v2_full_s10_strict_surface_rtx5070}"
STRICT_QA_DIR="${STRICT_QA_DIR:-${WORK}/frame_local_object_qa_guarded_v2_full_s10_strict_surface_rtx5070}"

STRICT_OBJECT_RELABEL_DIR="${STRICT_OBJECT_RELABEL_DIR:-${WORK}/frame_objects_guarded_v2_full_s10_strict_surface_object_relabel_safe_span_rtx5070}"
STRICT_OBJECT_RELABEL_VIEWER_DIR="${STRICT_OBJECT_RELABEL_VIEWER_DIR:-${WORK}/frame_object_viewer_guarded_v2_full_s10_strict_surface_object_relabel_safe_span_rtx5070}"
STRICT_OBJECT_RELABEL_QA_DIR="${STRICT_OBJECT_RELABEL_QA_DIR:-${WORK}/frame_local_object_qa_guarded_v2_full_s10_strict_surface_object_relabel_safe_span_rtx5070}"

GROUND_GUARD_TARGET_DIR="${GROUND_GUARD_TARGET_DIR:-${WORK}/frame_targets_guarded_v2_full_s10_ground_artifact_guard_rtx5070}"
GROUND_GUARD_OBJECT_DIR="${GROUND_GUARD_OBJECT_DIR:-${WORK}/frame_objects_guarded_v2_full_s10_ground_artifact_guard_strict_rtx5070}"
GROUND_GUARD_VIEWER_DIR="${GROUND_GUARD_VIEWER_DIR:-${WORK}/frame_object_viewer_guarded_v2_full_s10_ground_artifact_guard_strict_rtx5070}"
GROUND_GUARD_QA_DIR="${GROUND_GUARD_QA_DIR:-${WORK}/frame_local_object_qa_guarded_v2_full_s10_ground_artifact_guard_strict_rtx5070}"

GROUND_GUARD_OBJECT_RELABEL_DIR="${GROUND_GUARD_OBJECT_RELABEL_DIR:-${WORK}/frame_objects_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070}"
GROUND_GUARD_OBJECT_RELABEL_VIEWER_DIR="${GROUND_GUARD_OBJECT_RELABEL_VIEWER_DIR:-${WORK}/frame_object_viewer_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070}"
GROUND_GUARD_OBJECT_RELABEL_QA_DIR="${GROUND_GUARD_OBJECT_RELABEL_QA_DIR:-${WORK}/frame_local_object_qa_guarded_v2_full_s10_ground_guard_object_relabel_rtx5070}"

COMPARE_DIR="${COMPARE_DIR:-${WORK}/guarded_v2_surface_refinement_all_risk_compare}"

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

need_file() {
  local path="$1"
  if [[ ! -s "${path}" ]]; then
    echo "missing required file: ${path}" >&2
    return 1
  fi
}

run_step() {
  local output="$1"
  shift
  if [[ "${FORCE}" != "1" && -s "${output}" ]]; then
    log "skip existing: ${output}"
    return 0
  fi
  log "run: $*"
  "$@"
}

build_viewer() {
  local targets_jsonl="$1"
  local target_ply="$2"
  local objects_jsonl="$3"
  local output_dir="$4"
  run_step "${output_dir}/frame_object_viewer_export_report.json" \
    "${PY}" "${SCRIPT_DIR}/export_frame_target_objects_for_viewer.py" \
      --targets-jsonl "${targets_jsonl}" \
      --target-ply "${target_ply}" \
      --objects-jsonl "${objects_jsonl}" \
      --output-dir "${output_dir}" \
      --stride "${VIEWER_STRIDE}" \
      --ply-name "frame_object_points_stride${VIEWER_STRIDE}.ply"
}

build_qa() {
  local targets_jsonl="$1"
  local objects_jsonl="$2"
  local output_dir="$3"
  run_step "${output_dir}/frame_local_object_qa_report.json" \
    "${PY}" "${SCRIPT_DIR}/build_frame_local_object_qa_pack.py" \
      --targets-jsonl "${targets_jsonl}" \
      --objects-jsonl "${objects_jsonl}" \
      --workdir "${WORK}" \
      --output-dir "${output_dir}" \
      --candidate-limit "${QA_CANDIDATE_LIMIT}" \
      --evidence-per-object "${QA_EVIDENCE_PER_OBJECT}"
}

need_file "${PY}"
need_file "${SOURCE_TARGETS_JSONL}"
need_file "${SOURCE_TARGET_PLY}"

if [[ "${CHECK_ONLY}" == "1" ]]; then
  log "check_only ok"
  log "repo=${REPO_DIR}"
  log "work=${WORK}"
  log "python=${PY}"
  log "source_targets=${SOURCE_TARGETS_JSONL}"
  exit 0
fi

mkdir -p "${COMPARE_DIR}"

log "[1/7] strict surface baseline"
run_step "${STRICT_OBJECT_DIR}/fusion_report.json" \
  "${PY}" "${SCRIPT_DIR}/fuse_targets_to_objects.py" \
    --targets "${SOURCE_TARGETS_JSONL}" \
    --output-dir "${STRICT_OBJECT_DIR}" \
    --strict-surface-labels
build_viewer "${SOURCE_TARGETS_JSONL}" "${SOURCE_TARGET_PLY}" "${STRICT_OBJECT_DIR}/objects.jsonl" "${STRICT_VIEWER_DIR}"
build_qa "${SOURCE_TARGETS_JSONL}" "${STRICT_OBJECT_DIR}/objects.jsonl" "${STRICT_QA_DIR}"

log "[2/7] strict surface + object relabel"
run_step "${STRICT_OBJECT_RELABEL_DIR}/object_relabel_report.json" \
  "${PY}" "${SCRIPT_DIR}/refine_target_fusion_objects.py" \
    --objects-jsonl "${STRICT_OBJECT_DIR}/objects.jsonl" \
    --output-jsonl "${STRICT_OBJECT_RELABEL_DIR}/objects.jsonl" \
    --report "${STRICT_OBJECT_RELABEL_DIR}/object_relabel_report.json" \
    --geometry-relabel-flat-wall \
    --horizontal-surface-label ground
build_viewer "${SOURCE_TARGETS_JSONL}" "${SOURCE_TARGET_PLY}" "${STRICT_OBJECT_RELABEL_DIR}/objects.jsonl" "${STRICT_OBJECT_RELABEL_VIEWER_DIR}"
build_qa "${SOURCE_TARGETS_JSONL}" "${STRICT_OBJECT_RELABEL_DIR}/objects.jsonl" "${STRICT_OBJECT_RELABEL_QA_DIR}"

log "[3/7] target-level ground artifact guard"
run_step "${GROUND_GUARD_TARGET_DIR}/geometry_refine_summary.json" \
  "${PY}" "${SCRIPT_DIR}/refine_frame_targets_by_geometry.py" \
    --targets-jsonl "${SOURCE_TARGETS_JSONL}" \
    --target-ply "${SOURCE_TARGET_PLY}" \
    --output-dir "${GROUND_GUARD_TARGET_DIR}" \
    --guard-linear-ground-artifacts

log "[4/7] ground artifact guard + strict surface"
run_step "${GROUND_GUARD_OBJECT_DIR}/fusion_report.json" \
  "${PY}" "${SCRIPT_DIR}/fuse_targets_to_objects.py" \
    --targets "${GROUND_GUARD_TARGET_DIR}/frame_targets_refined.jsonl" \
    --output-dir "${GROUND_GUARD_OBJECT_DIR}" \
    --strict-surface-labels
build_viewer "${GROUND_GUARD_TARGET_DIR}/frame_targets_refined.jsonl" "${GROUND_GUARD_TARGET_DIR}/frame_targets_refined.ply" "${GROUND_GUARD_OBJECT_DIR}/objects.jsonl" "${GROUND_GUARD_VIEWER_DIR}"
build_qa "${GROUND_GUARD_TARGET_DIR}/frame_targets_refined.jsonl" "${GROUND_GUARD_OBJECT_DIR}/objects.jsonl" "${GROUND_GUARD_QA_DIR}"

log "[5/7] ground artifact guard + object relabel"
run_step "${GROUND_GUARD_OBJECT_RELABEL_DIR}/object_relabel_report.json" \
  "${PY}" "${SCRIPT_DIR}/refine_target_fusion_objects.py" \
    --objects-jsonl "${GROUND_GUARD_OBJECT_DIR}/objects.jsonl" \
    --output-jsonl "${GROUND_GUARD_OBJECT_RELABEL_DIR}/objects.jsonl" \
    --report "${GROUND_GUARD_OBJECT_RELABEL_DIR}/object_relabel_report.json" \
    --geometry-relabel-flat-wall \
    --horizontal-surface-label ground
build_viewer "${GROUND_GUARD_TARGET_DIR}/frame_targets_refined.jsonl" "${GROUND_GUARD_TARGET_DIR}/frame_targets_refined.ply" "${GROUND_GUARD_OBJECT_RELABEL_DIR}/objects.jsonl" "${GROUND_GUARD_OBJECT_RELABEL_VIEWER_DIR}"
build_qa "${GROUND_GUARD_TARGET_DIR}/frame_targets_refined.jsonl" "${GROUND_GUARD_OBJECT_RELABEL_DIR}/objects.jsonl" "${GROUND_GUARD_OBJECT_RELABEL_QA_DIR}"

log "[6/7] full-risk comparison"
run_step "${COMPARE_DIR}/qa_compare.json" \
  "${PY}" "${SCRIPT_DIR}/compare_frame_local_object_qa.py" \
    --report "strict_surface=${STRICT_QA_DIR}/frame_local_object_qa_report.json" \
    --report "strict_surface_object_relabel=${STRICT_OBJECT_RELABEL_QA_DIR}/frame_local_object_qa_report.json" \
    --report "ground_artifact_guard_strict=${GROUND_GUARD_QA_DIR}/frame_local_object_qa_report.json" \
    --report "ground_guard_object_relabel=${GROUND_GUARD_OBJECT_RELABEL_QA_DIR}/frame_local_object_qa_report.json" \
    --output-json "${COMPARE_DIR}/qa_compare.json" \
    --output-md "${COMPARE_DIR}/qa_compare.md"

log "[7/7] done"
printf 'candidate_viewer=%s\n' "${GROUND_GUARD_OBJECT_RELABEL_VIEWER_DIR}"
printf 'comparison=%s\n' "${COMPARE_DIR}/qa_compare.md"
