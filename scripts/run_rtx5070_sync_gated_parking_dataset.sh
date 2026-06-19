#!/usr/bin/env bash
set -euo pipefail

# Run the parking semantic dataset route.
#
# Default is dry-run. Use RUN=1 to start the remote tmux job.
# Parking dataset note, 2026-06-19:
#   The .mkv frame count matches img_pos rows.  Video PTS is a uniform encoding
#   timeline and must not be fitted to nonuniform img_pos timestamps.  The
#   default route therefore uses direct video index == img_pos frame_id.
#   Set SYNC_MODE=frame-map only for explicitly validated external mappings.

SERVER="${SERVER:-scan-rtx5070}"
LOCAL_REPO="${LOCAL_REPO:-/Users/skkac/Work/SCAN/new_route}"
REMOTE_REPO="${REMOTE_REPO:-/home/zsh/Work/SCAN/new_route}"
REMOTE_WORK="${REMOTE_WORK:-/home/zsh/Work/SCAN/work_MT20260616-175807}"
REMOTE_DATASET="${REMOTE_DATASET:-/home/zsh/Work/SCAN/datasets/MT20260616-175807}"
REMOTE_VENV="${REMOTE_VENV:-/home/zsh/Work/SCAN/.venvs/scan-semantic}"
SYNC_RUN_NAME="${SYNC_RUN_NAME:-sync_anchor_constrained_timestamp_absprior_dot3_20260619}"
REMOTE_SYNC_DIR="${REMOTE_SYNC_DIR:-${REMOTE_WORK}/${SYNC_RUN_NAME}}"
BIND_ADDRESS="${BIND_ADDRESS:-}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-8}"
SYNC_MODE="${SYNC_MODE:-opencv-index}"

RUN="${RUN:-0}"
WAIT="${WAIT:-0}"
OVERWRITE="${OVERWRITE:-0}"
SESSION_NAME="${SESSION_NAME:-parking_sync_gated_s10}"

START="${START:-0}"
END="${END:-6180}"
STRIDE="${STRIDE:-10}"
CAMS="${CAMS:-0 1 2}"
OUT_SUFFIX="${OUT_SUFFIX:-direct_index_s10}"

DO_EXTRACT_FRAMES="${DO_EXTRACT_FRAMES:-1}"
DO_COLORIZE="${DO_COLORIZE:-0}"
DO_PRIORITY="${DO_PRIORITY:-1}"
DO_SAFE_ROUTE="${DO_SAFE_ROUTE:-0}"
BUILD_TARGETS="${BUILD_TARGETS:-1}"
BUILD_OBJECTS="${BUILD_OBJECTS:-1}"

FRAME_MAP="${FRAME_MAP:-${REMOTE_SYNC_DIR}/expanded_frame_map.jsonl}"
READINESS_JSON="${READINESS_JSON:-${REMOTE_SYNC_DIR}/sync_frame_map_readiness.json}"
READINESS_EXIT="${READINESS_EXIT:-${REMOTE_SYNC_DIR}/sync_frame_map_readiness.exit_code}"

LX="${LX:-${REMOTE_DATASET}/MANIFOLD_MT20260616-175807.lx}"
FRAME_ROOT="${FRAME_ROOT:-${REMOTE_WORK}/frames_jpeg_${OUT_SUFFIX}}"
PRIORITY_DIR="${PRIORITY_DIR:-${REMOTE_WORK}/priority_surface_mapillary_${OUT_SUFFIX}}"
COLOR_DIR="${COLOR_DIR:-${REMOTE_WORK}/colorized_lx_${OUT_SUFFIX}}"
COLOR_FULL_PLY="${COLOR_FULL_PLY:-${COLOR_DIR}/colorized_points.ply}"
COLOR_VOXEL_PLY="${COLOR_VOXEL_PLY:-${COLOR_DIR}/colorized_points_voxel001.ply}"
COLOR_REPORT="${COLOR_REPORT:-${COLOR_DIR}/colorize_report.json}"

PRIORITY_BATCH_SIZE="${PRIORITY_BATCH_SIZE:-8}"
PRIORITY_MODEL="${PRIORITY_MODEL:-mapillary}"
RUN_PREFLIGHT="${RUN_PREFLIGHT:-1}"
PREFLIGHT_OUTPUT="${PREFLIGHT_OUTPUT:-${LOCAL_REPO}/server_parking_priority_s10/${OUT_SUFFIX}_preflight.json}"
MIN_FREE_VRAM_MIB="${MIN_FREE_VRAM_MIB:-6000}"

ssh_opts=(-o BatchMode=yes -o "ConnectTimeout=${CONNECT_TIMEOUT}")
if [[ -n "${BIND_ADDRESS}" ]]; then
  ssh_opts+=(-o "BindAddress=${BIND_ADDRESS}")
fi

quote() {
  printf '%q' "$1"
}

if [[ "${SYNC_MODE}" == "frame-map" ]]; then
  remote_check=$(cat <<EOF
set -euo pipefail
test -f $(quote "${READINESS_EXIT}") || { echo missing_readiness_exit=$(quote "${READINESS_EXIT}"); exit 2; }
test "\$(cat $(quote "${READINESS_EXIT}"))" = "0" || { echo readiness_not_passing=$(quote "${READINESS_EXIT}"); cat $(quote "${READINESS_JSON}") 2>/dev/null || true; exit 3; }
test -f $(quote "${READINESS_JSON}") || { echo missing_readiness_json=$(quote "${READINESS_JSON}"); exit 2; }
test -f $(quote "${FRAME_MAP}") || { echo missing_frame_map=$(quote "${FRAME_MAP}"); exit 2; }
EOF
)
elif [[ "${SYNC_MODE}" == "opencv-index" ]]; then
  remote_check="set -euo pipefail"
else
  echo "unsupported SYNC_MODE=${SYNC_MODE}; expected opencv-index or frame-map" >&2
  exit 2
fi

extract_sync_args=("--sync-mode" "${SYNC_MODE}")
color_sync_args=()
if [[ "${SYNC_MODE}" == "frame-map" ]]; then
  extract_sync_args+=("--frame-map-jsonl" "${FRAME_MAP}" "--require-frame-map")
  color_sync_args+=("--frame-map-jsonl" "${FRAME_MAP}" "--require-frame-map")
fi
extract_sync_args_text="$(printf '    %q \\\n' "${extract_sync_args[@]}" | sed '$ s/ \\$//')"
if (( ${#color_sync_args[@]} )); then
  color_sync_args_text="$(printf '    %q \\\n' "${color_sync_args[@]}")"
else
  color_sync_args_text=""
fi

remote_job=$(cat <<EOF
set -euo pipefail
cd $(quote "${REMOTE_REPO}")
export SCAN_IMAGE_DIR=$(quote "${REMOTE_DATASET}/image")
export SCAN_VIDEO_DIR=$(quote "${REMOTE_DATASET}/image")
export PYTHONPATH="\$PWD/scripts"

${remote_check}

if [[ $(quote "${OVERWRITE}") == "1" ]]; then
  [[ $(quote "${DO_EXTRACT_FRAMES}") == "1" ]] && rm -rf $(quote "${FRAME_ROOT}")
  [[ $(quote "${DO_PRIORITY}") == "1" ]] && rm -rf $(quote "${PRIORITY_DIR}")
  [[ $(quote "${DO_COLORIZE}") == "1" ]] && rm -rf $(quote "${COLOR_DIR}")
fi

if [[ $(quote "${DO_EXTRACT_FRAMES}") == "1" ]]; then
  $(quote "${REMOTE_VENV}/bin/python") scripts/extract_undistorted_frames_jpeg.py \
    --output-dir $(quote "${FRAME_ROOT}") \
    --start $(quote "${START}") \
    --end $(quote "${END}") \
    --stride $(quote "${STRIDE}") \
    --cams ${CAMS} \
    --workers 3 \
    --quality 92 \
    --skip-existing \
${extract_sync_args_text}
fi

if [[ $(quote "${DO_COLORIZE}") == "1" ]]; then
  mkdir -p $(quote "${COLOR_DIR}")
  $(quote "${REMOTE_VENV}/bin/python") scripts/colorize_lx_stream.py \
    --lx-file $(quote "${LX}") \
    --start $(quote "${START}") \
    --end $(quote "${END}") \
    --frame-step $(quote "${STRIDE}") \
    --cams ${CAMS} \
    --output $(quote "${COLOR_FULL_PLY}") \
    --voxel-output $(quote "${COLOR_VOXEL_PLY}") \
    --voxel-size 0.01 \
    --report $(quote "${COLOR_REPORT}") \
${color_sync_args_text}\
    --sky-filter heuristic
fi

if [[ $(quote "${DO_PRIORITY}") == "1" ]]; then
  $(quote "${REMOTE_VENV}/bin/python") scripts/segment_priority_classes.py \
    --frame-root $(quote "${FRAME_ROOT}") \
    --output-dir $(quote "${PRIORITY_DIR}") \
    --start $(quote "${START}") \
    --end $(quote "${END}") \
    --stride $(quote "${STRIDE}") \
    --cams ${CAMS} \
    --model $(quote "${PRIORITY_MODEL}") \
    --device cuda \
    --batch-size $(quote "${PRIORITY_BATCH_SIZE}") \
    --amp \
    --skip-existing
fi

if [[ $(quote "${DO_SAFE_ROUTE}") == "1" ]]; then
  RUN=1 \
  OVERWRITE=$(quote "${OVERWRITE}") \
  START=$(quote "${START}") \
  END=$(quote "${END}") \
  STRIDE=$(quote "${STRIDE}") \
  CAMS=$(quote "${CAMS}") \
  OUT_SUFFIX=$(quote "${OUT_SUFFIX}") \
  FRAME_ROOT=$(quote "${FRAME_ROOT}") \
  PRIORITY_DIR=$(quote "${PRIORITY_DIR}") \
  BUILD_TARGETS=$(quote "${BUILD_TARGETS}") \
  BUILD_OBJECTS=$(quote "${BUILD_OBJECTS}") \
  scripts/run_parking_safe_semantic_prior_route.sh
fi
EOF
)

cat <<EOF
server=${SERVER}
run=${RUN}
session=${SESSION_NAME}
sync_dir=${REMOTE_SYNC_DIR}
sync_mode=${SYNC_MODE}
frame_map=${FRAME_MAP}
readiness_json=${READINESS_JSON}
range=${START}..${END} stride=${STRIDE} cams=${CAMS}
frame_root=${FRAME_ROOT}
priority_dir=${PRIORITY_DIR}
do_extract_frames=${DO_EXTRACT_FRAMES}
do_colorize=${DO_COLORIZE}
do_priority=${DO_PRIORITY}
do_safe_route=${DO_SAFE_ROUTE}
run_preflight=${RUN_PREFLIGHT}
preflight_output=${PREFLIGHT_OUTPUT}
remote_job:
${remote_job}
EOF

if [[ "${RUN}" != "1" ]]; then
  exit 0
fi

if [[ "${RUN_PREFLIGHT}" == "1" ]]; then
  python3 "${LOCAL_REPO}/scripts/check_rtx5070_parking_runtime.py" \
    --host "${SERVER}" \
    --remote-repo "${REMOTE_REPO}" \
    --remote-work "${REMOTE_WORK}" \
    --venv "${REMOTE_VENV}" \
    --tmux-session "${SESSION_NAME}" \
    --no-require-tmux \
    --no-default-required-files \
    --min-free-vram-mib "${MIN_FREE_VRAM_MIB}" \
    --required-remote-file "${LX}" \
    --required-remote-file "${REMOTE_DATASET}/image/video_cam0.mkv" \
    --required-remote-file "${REMOTE_DATASET}/image/video_cam1.mkv" \
    --required-remote-file "${REMOTE_DATASET}/image/video_cam2.mkv" \
    --required-remote-file "${REMOTE_DATASET}/image/img_pos.txt" \
    --required-remote-file "${REMOTE_DATASET}/image/cam_in_ex.txt" \
    --output "${PREFLIGHT_OUTPUT}"
  if [[ "${SYNC_MODE}" == "frame-map" ]]; then
    python3 "${LOCAL_REPO}/scripts/check_rtx5070_parking_runtime.py" \
      --host "${SERVER}" \
      --remote-repo "${REMOTE_REPO}" \
      --remote-work "${REMOTE_WORK}" \
      --venv "${REMOTE_VENV}" \
      --tmux-session "${SESSION_NAME}_syncmap" \
      --no-require-tmux \
      --no-default-required-files \
      --min-free-vram-mib 0 \
      --required-remote-file "${FRAME_MAP}" \
      --required-remote-file "${READINESS_JSON}" \
      --required-remote-file "${READINESS_EXIT}" \
      --output "${PREFLIGHT_OUTPUT%.json}_syncmap.json"
  fi
fi

ssh "${ssh_opts[@]}" "${SERVER}" "${remote_check}"
remote_job_path="/tmp/${SESSION_NAME}.sh"
printf '%s\n' "${remote_job}" | ssh "${ssh_opts[@]}" "${SERVER}" "cat > $(quote "${remote_job_path}") && chmod +x $(quote "${remote_job_path}")"
ssh "${ssh_opts[@]}" "${SERVER}" "tmux has-session -t $(quote "${SESSION_NAME}") 2>/dev/null && tmux kill-session -t $(quote "${SESSION_NAME}") || true; tmux new-session -d -s $(quote "${SESSION_NAME}") bash $(quote "${remote_job_path}")"

if [[ "${WAIT}" == "1" ]]; then
  while ssh "${ssh_opts[@]}" "${SERVER}" "tmux has-session -t $(quote "${SESSION_NAME}") 2>/dev/null"; do
    ssh "${ssh_opts[@]}" "${SERVER}" "tmux capture-pane -pt $(quote "${SESSION_NAME}") -S -40 || true"
    sleep 30
  done
else
  echo "inspect: ssh ${SERVER} 'tmux attach -t ${SESSION_NAME}'"
fi
