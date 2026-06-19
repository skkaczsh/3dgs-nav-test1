#!/usr/bin/env bash
set -euo pipefail

# Local launcher for the parking LiDAR/video sync anchor loop.
#
# Expected workflow:
#   1. Open the local review page.
#   2. Select reliable candidates and export accepted_sync_anchors.jsonl.
#   3. Run this script to constrain the remote sync solver.
#   4. Review the regenerated constrained review pack before semantic production.

SERVER="${SERVER:-scan-rtx5070}"
REMOTE_REPO="${REMOTE_REPO:-/home/zsh/Work/SCAN/new_route}"
REMOTE_WORK="${REMOTE_WORK:-/home/zsh/Work/SCAN/work_MT20260616-175807}"
LOCAL_REPO="${LOCAL_REPO:-/Users/skkac/Work/SCAN/new_route}"
REMOTE_DATASET="${REMOTE_DATASET:-/home/zsh/Work/SCAN/datasets/MT20260616-175807}"
REMOTE_VENV="${REMOTE_VENV:-/home/zsh/Work/SCAN/.venvs/scan-semantic}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-8}"
BIND_ADDRESS="${BIND_ADDRESS:-}"

REVIEW_NAME="${REVIEW_NAME:-sync_anchor_review_priority_sky_penalty_timestamp_absprior_dot3_20260619}"
RUN_NAME="${RUN_NAME:-sync_anchor_constrained_timestamp_absprior_dot3_20260619}"
LOCAL_ANCHORS="${LOCAL_ANCHORS:-${LOCAL_REPO}/server_parking_priority_s10/${REVIEW_NAME}/accepted_sync_anchors.jsonl}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-${LOCAL_REPO}/server_parking_priority_s10/${RUN_NAME}}"
REMOTE_INPUTS="${REMOTE_INPUTS:-${REMOTE_WORK}/tmp_sync_review_inputs}"
REMOTE_OUTPUT="${REMOTE_OUTPUT:-${REMOTE_WORK}/${RUN_NAME}}"
REMOTE_ANCHORS="${REMOTE_ANCHORS:-${REMOTE_OUTPUT}/accepted_sync_anchors.jsonl}"
REMOTE_CANDIDATES="${REMOTE_CANDIDATES:-${REMOTE_WORK}/sync_calibration_sky_penalty_fullprobe_20260619/sync_candidates.jsonl}"
TOP_N="${TOP_N:-4}"
SHEET_COLS="${SHEET_COLS:-4}"
DOT_PX="${DOT_PX:-3}"
MAP_START="${MAP_START:-0}"
MAP_END="${MAP_END:-6180}"
MAP_STRIDE="${MAP_STRIDE:-10}"
VIDEO_FRAME_COUNT="${VIDEO_FRAME_COUNT:-6181}"
READINESS_CAMS="${READINESS_CAMS:-0 1 2}"
MIN_ACCEPTED_PER_CAM="${MIN_ACCEPTED_PER_CAM:-2}"
SOLVER_TIME_MODE="${SOLVER_TIME_MODE:-timestamp}"
SOLVER_VIDEO_FPS="${SOLVER_VIDEO_FPS:-6.0}"
SOLVER_TIMESTAMP_PHASE_FRACTION="${SOLVER_TIMESTAMP_PHASE_FRACTION:-1.0}"
SOLVER_ABSOLUTE_PRIOR_WEIGHT="${SOLVER_ABSOLUTE_PRIOR_WEIGHT:-1.0}"
SOLVER_ABSOLUTE_PRIOR_TOLERANCE="${SOLVER_ABSOLUTE_PRIOR_TOLERANCE:-200}"
SOLVER_ABSOLUTE_INTERCEPT="${SOLVER_ABSOLUTE_INTERCEPT:-0}"
SOLVER_ABSOLUTE_INTERCEPT_SOURCE="${SOLVER_ABSOLUTE_INTERCEPT_SOURCE:-anchors}"
DRY_RUN="${DRY_RUN:-0}"

ssh_opts=(-o BatchMode=yes -o "ConnectTimeout=${CONNECT_TIMEOUT}")
if [[ -n "${BIND_ADDRESS}" ]]; then
  ssh_opts+=(-o "BindAddress=${BIND_ADDRESS}")
fi

log() {
  printf '[%s] %s\n' "$(date '+%F %T')" "$*"
}

quote() {
  printf '%q' "$1"
}

if [[ ! -f "${LOCAL_ANCHORS}" && "${DRY_RUN}" != "1" ]]; then
  cat >&2 <<EOF
missing_anchors=${LOCAL_ANCHORS}

Open the review page, select accepted anchors, and export the JSONL first:
http://127.0.0.1:8765/server_parking_priority_s10/${REVIEW_NAME}/manual_anchor_review.html
EOF
  exit 2
fi

remote_cmd=$(cat <<EOF
set -euo pipefail
cd $(quote "${REMOTE_REPO}")
export SCAN_IMAGE_DIR=$(quote "${REMOTE_DATASET}/image")
export SCAN_VIDEO_DIR=$(quote "${REMOTE_DATASET}/image")
export PYTHONPATH="\$PWD/scripts"
mkdir -p $(quote "${REMOTE_OUTPUT}")
$(quote "${REMOTE_VENV}/bin/python") scripts/solve_sync_path_from_candidates.py \
  --candidates-jsonl $(quote "${REMOTE_CANDIDATES}") \
  --anchors-jsonl $(quote "${REMOTE_ANCHORS}") \
  --output-dir $(quote "${REMOTE_OUTPUT}/solver") \
  --time-mode $(quote "${SOLVER_TIME_MODE}") \
  --img-pos-file $(quote "${REMOTE_DATASET}/image/img_pos.txt") \
  --video-fps $(quote "${SOLVER_VIDEO_FPS}") \
  --timestamp-phase-fraction $(quote "${SOLVER_TIMESTAMP_PHASE_FRACTION}") \
  --absolute-prior-weight $(quote "${SOLVER_ABSOLUTE_PRIOR_WEIGHT}") \
  --absolute-prior-tolerance $(quote "${SOLVER_ABSOLUTE_PRIOR_TOLERANCE}") \
  --absolute-intercept $(quote "${SOLVER_ABSOLUTE_INTERCEPT}") \
  --absolute-intercept-source $(quote "${SOLVER_ABSOLUTE_INTERCEPT_SOURCE}")
$(quote "${REMOTE_VENV}/bin/python") scripts/expand_sync_frame_map.py \
  --path-jsonl $(quote "${REMOTE_OUTPUT}/solver/sync_smooth_paths.jsonl") \
  --solver-report $(quote "${REMOTE_OUTPUT}/solver/sync_smooth_path_report.json") \
  --img-pos-file $(quote "${REMOTE_DATASET}/image/img_pos.txt") \
  --output-jsonl $(quote "${REMOTE_OUTPUT}/expanded_frame_map.jsonl") \
  --report $(quote "${REMOTE_OUTPUT}/expanded_frame_map_report.json") \
  --start $(quote "${MAP_START}") \
  --end $(quote "${MAP_END}") \
  --stride $(quote "${MAP_STRIDE}") \
  --cams ${READINESS_CAMS} \
  --video-frame-count $(quote "${VIDEO_FRAME_COUNT}")
set +e
$(quote "${REMOTE_VENV}/bin/python") scripts/check_sync_frame_map_readiness.py \
  --anchors-jsonl $(quote "${REMOTE_ANCHORS}") \
  --frame-map-jsonl $(quote "${REMOTE_OUTPUT}/expanded_frame_map.jsonl") \
  --solver-report $(quote "${REMOTE_OUTPUT}/solver/sync_smooth_path_report.json") \
  --start $(quote "${MAP_START}") \
  --end $(quote "${MAP_END}") \
  --stride $(quote "${MAP_STRIDE}") \
  --cams ${READINESS_CAMS} \
  --min-accepted-per-cam $(quote "${MIN_ACCEPTED_PER_CAM}") \
  --output $(quote "${REMOTE_OUTPUT}/sync_frame_map_readiness.json")
readiness_code=\$?
printf '%s\n' "\${readiness_code}" > $(quote "${REMOTE_OUTPUT}/sync_frame_map_readiness.exit_code")
set -e
$(quote "${REMOTE_VENV}/bin/python") scripts/build_sync_anchor_review_pack.py \
  --lx-file $(quote "${REMOTE_DATASET}/MANIFOLD_MT20260616-175807.lx") \
  --candidates-jsonl $(quote "${REMOTE_CANDIDATES}") \
  --smooth-path-jsonl $(quote "${REMOTE_OUTPUT}/solver/sync_smooth_paths.jsonl") \
  --output-dir $(quote "${REMOTE_OUTPUT}/review") \
  --top-n $(quote "${TOP_N}") \
  --sheet-cols $(quote "${SHEET_COLS}") \
  --dot-px $(quote "${DOT_PX}")
EOF
)

if [[ "${DRY_RUN}" == "1" ]]; then
  anchors_status="present"
  if [[ ! -f "${LOCAL_ANCHORS}" ]]; then
    anchors_status="missing"
  fi
  cat <<EOF
dry_run=1
server=${SERVER}
local_anchors=${LOCAL_ANCHORS}
anchors_status=${anchors_status}
remote_anchors=${REMOTE_ANCHORS}
remote_output=${REMOTE_OUTPUT}
local_output=${LOCAL_OUTPUT}
remote_cmd=${remote_cmd}
remote_candidates=${REMOTE_CANDIDATES}
solver_time_mode=${SOLVER_TIME_MODE}
solver_video_fps=${SOLVER_VIDEO_FPS}
solver_timestamp_phase_fraction=${SOLVER_TIMESTAMP_PHASE_FRACTION}
solver_absolute_prior_weight=${SOLVER_ABSOLUTE_PRIOR_WEIGHT}
solver_absolute_prior_tolerance=${SOLVER_ABSOLUTE_PRIOR_TOLERANCE}
solver_absolute_intercept=${SOLVER_ABSOLUTE_INTERCEPT}
solver_absolute_intercept_source=${SOLVER_ABSOLUTE_INTERCEPT_SOURCE}
dot_px=${DOT_PX}
map_start=${MAP_START}
map_end=${MAP_END}
map_stride=${MAP_STRIDE}
video_frame_count=${VIDEO_FRAME_COUNT}
readiness_cams=${READINESS_CAMS}
min_accepted_per_cam=${MIN_ACCEPTED_PER_CAM}
EOF
  exit 0
fi

log "[1/4] prepare remote output"
ssh "${ssh_opts[@]}" "${SERVER}" "mkdir -p $(quote "${REMOTE_OUTPUT}")"

log "[2/4] sync accepted anchors"
rsync -av "${LOCAL_ANCHORS}" "${SERVER}:${REMOTE_ANCHORS}"

log "[3/4] solve constrained path and rebuild review pack"
ssh "${ssh_opts[@]}" "${SERVER}" "${remote_cmd}"

log "[4/4] pull constrained sync artifacts"
mkdir -p "${LOCAL_OUTPUT}"
rsync -av \
  "${SERVER}:${REMOTE_OUTPUT}/accepted_sync_anchors.jsonl" \
  "${SERVER}:${REMOTE_OUTPUT}/sync_frame_map_readiness.json" \
  "${SERVER}:${REMOTE_OUTPUT}/sync_frame_map_readiness.exit_code" \
  "${SERVER}:${REMOTE_OUTPUT}/expanded_frame_map.jsonl" \
  "${SERVER}:${REMOTE_OUTPUT}/expanded_frame_map_report.json" \
  "${SERVER}:${REMOTE_OUTPUT}/solver" \
  "${SERVER}:${REMOTE_OUTPUT}/review" \
  "${LOCAL_OUTPUT}/"

readiness_code="$(cat "${LOCAL_OUTPUT}/sync_frame_map_readiness.exit_code")"
readiness_report="${LOCAL_OUTPUT}/sync_frame_map_readiness.json"

if [[ "${readiness_code}" != "0" ]]; then
  cat >&2 <<EOF
readiness_failed=1
readiness_exit_code=${readiness_code}
readiness_report=${readiness_report}
review_url=http://127.0.0.1:8765/server_parking_priority_s10/${RUN_NAME}/review/manual_anchor_review.html
EOF
  exit 3
fi

cat <<EOF
done=1
local_output=${LOCAL_OUTPUT}
solver_report=${LOCAL_OUTPUT}/solver/sync_smooth_path_report.json
expanded_frame_map=${LOCAL_OUTPUT}/expanded_frame_map.jsonl
readiness_report=${readiness_report}
review_url=http://127.0.0.1:8765/server_parking_priority_s10/${RUN_NAME}/review/manual_anchor_review.html
EOF
