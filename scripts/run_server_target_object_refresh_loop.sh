#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START="${START:-1000}"
END="${END:-1999}"
COMBO="${COMBO:-sam2_prompt_v3_sky_label_merge_completion}"
SEMANTIC_EVAL_DIR="${SEMANTIC_EVAL_DIR:-/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_${START}_${END}}"
TARGET_OUTPUT_DIR="${TARGET_OUTPUT_DIR:-/root/epfs/new_route_stage1_skymask/target_object_fusion_${START}_${END}_surface024_fine012}"
COLOR_DIR="${COLOR_DIR:-/root/epfs/new_route_stage1_skymask/output}"
LOG_DIR="${LOG_DIR:-/root/epfs/new_route_stage1_skymask/logs}"
STATE_FILE="${STATE_FILE:-${LOG_DIR}/target_object_refresh_${START}_${END}.state}"
PID_FILE="${PID_FILE:-${LOG_DIR}/target_object_refresh_${START}_${END}.pid}"
LOCK_DIR="${LOCK_DIR:-${LOG_DIR}/target_object_refresh_${START}_${END}.lock}"
SLEEP_SECONDS="${SLEEP_SECONDS:-900}"
MAX_CYCLES="${MAX_CYCLES:-0}"
MIN_COMPLETION_DELTA="${MIN_COMPLETION_DELTA:-60}"
RUN_ON_FIRST="${RUN_ON_FIRST:-0}"
VOXEL_SIZE="${VOXEL_SIZE:-0.08}"
SURFACE_VOXEL_SIZE="${SURFACE_VOXEL_SIZE:-0.24}"
FINE_VOXEL_SIZE="${FINE_VOXEL_SIZE:-0.12}"
MIN_MERGE_CONFIDENCE="${MIN_MERGE_CONFIDENCE:-0.5}"
STRIDE="${STRIDE:-10}"

mkdir -p "${LOG_DIR}" "${TARGET_OUTPUT_DIR}/reports"

count_label_records() {
  find "${SEMANTIC_EVAL_DIR}/images" -path "*/${COMBO}/label_records.json" 2>/dev/null | wc -l | tr -d ' '
}

write_state() {
  local count="$1"
  printf '%s\n' "${count}" >"${STATE_FILE}"
}

read_state() {
  if [[ -f "${STATE_FILE}" ]]; then
    cat "${STATE_FILE}"
  else
    printf '0\n'
  fi
}

run_refresh() {
  if ! mkdir "${LOCK_DIR}" 2>/dev/null; then
    echo "[refresh] skip: lock held at ${LOCK_DIR}"
    return 75
  fi
  (
    trap 'rm -rf "${LOCK_DIR}"' EXIT
    echo "[refresh] target/object fusion output=${TARGET_OUTPUT_DIR}"
    SEMANTIC_EVAL_DIR="${SEMANTIC_EVAL_DIR}" \
    COMBO="${COMBO}" \
    COLOR_DIR="${COLOR_DIR}" \
    OUTPUT_DIR="${TARGET_OUTPUT_DIR}" \
    START_FRAME="${START}" \
    END_FRAME="${END}" \
    WORK_MODE="semantic-dir" \
    VOXEL_SIZE="${VOXEL_SIZE}" \
    SURFACE_VOXEL_SIZE="${SURFACE_VOXEL_SIZE}" \
    FINE_VOXEL_SIZE="${FINE_VOXEL_SIZE}" \
    MIN_MERGE_CONFIDENCE="${MIN_MERGE_CONFIDENCE}" \
    bash "${SCRIPT_DIR}/run_server_target_object_fusion.sh"

    python3 "${SCRIPT_DIR}/relabel_objects_from_identity.py" \
      --objects-jsonl "${TARGET_OUTPUT_DIR}/objects/objects.jsonl" \
      --output-jsonl "${TARGET_OUTPUT_DIR}/objects/objects_identity_relabel.jsonl" \
      --report "${TARGET_OUTPUT_DIR}/reports/identity_relabel_report.json" \
      --input-ply "${TARGET_OUTPUT_DIR}/objects/object_centroids.ply" \
      --output-ply "${TARGET_OUTPUT_DIR}/objects/object_points_identity_relabel.ply"

    python3 "${SCRIPT_DIR}/stride_ascii_ply.py" \
      "${TARGET_OUTPUT_DIR}/objects/object_points_identity_relabel.ply" \
      "${TARGET_OUTPUT_DIR}/objects/object_points_identity_relabel_stride${STRIDE}.ply" \
      --stride "${STRIDE}"
  )
}

cycle=0
while true; do
  cycle=$((cycle + 1))
  echo "$$" >"${PID_FILE}"
  now="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  current_count="$(count_label_records)"
  last_count="$(read_state)"
  delta=$((current_count - last_count))
  echo "[${now}] cycle=${cycle} label_records=${current_count} last=${last_count} delta=${delta} min_delta=${MIN_COMPLETION_DELTA}"

  if [[ "${last_count}" -eq 0 && ! -f "${STATE_FILE}" && "${RUN_ON_FIRST}" != "1" ]]; then
    write_state "${current_count}"
    echo "[cycle ${cycle}] initialized state without refresh"
  elif [[ "${delta}" -ge "${MIN_COMPLETION_DELTA}" || "${RUN_ON_FIRST}" == "1" ]]; then
    if run_refresh; then
      write_state "${current_count}"
      RUN_ON_FIRST=0
    else
      echo "[cycle ${cycle}] refresh skipped or failed; state remains ${last_count}"
    fi
  else
    echo "[cycle ${cycle}] skip refresh"
  fi

  if [[ "${MAX_CYCLES}" -gt 0 && "${cycle}" -ge "${MAX_CYCLES}" ]]; then
    echo "[stop] reached MAX_CYCLES=${MAX_CYCLES}"
    rm -f "${PID_FILE}"
    break
  fi
  sleep "${SLEEP_SECONDS}"
done
