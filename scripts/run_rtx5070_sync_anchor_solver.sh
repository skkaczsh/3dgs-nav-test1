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

REVIEW_NAME="${REVIEW_NAME:-sync_anchor_review_small_20260619_v2}"
RUN_NAME="${RUN_NAME:-sync_anchor_constrained_from_review_v2}"
LOCAL_ANCHORS="${LOCAL_ANCHORS:-${LOCAL_REPO}/server_parking_priority_s10/${REVIEW_NAME}/accepted_sync_anchors.jsonl}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-${LOCAL_REPO}/server_parking_priority_s10/${RUN_NAME}}"
REMOTE_INPUTS="${REMOTE_INPUTS:-${REMOTE_WORK}/tmp_sync_review_inputs}"
REMOTE_OUTPUT="${REMOTE_OUTPUT:-${REMOTE_WORK}/${RUN_NAME}}"
REMOTE_ANCHORS="${REMOTE_ANCHORS:-${REMOTE_OUTPUT}/accepted_sync_anchors.jsonl}"
TOP_N="${TOP_N:-4}"
SHEET_COLS="${SHEET_COLS:-4}"
READINESS_FRAMES="${READINESS_FRAMES:-1000 1600 2200 2800 3400 4000 4600 5200 5800}"
READINESS_CAMS="${READINESS_CAMS:-0 1 2}"
MIN_ACCEPTED_PER_CAM="${MIN_ACCEPTED_PER_CAM:-2}"
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

if [[ ! -f "${LOCAL_ANCHORS}" ]]; then
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
  --candidates-jsonl $(quote "${REMOTE_INPUTS}/sync_candidates.jsonl") \
  --anchors-jsonl $(quote "${REMOTE_ANCHORS}") \
  --output-dir $(quote "${REMOTE_OUTPUT}/solver")
set +e
$(quote "${REMOTE_VENV}/bin/python") scripts/check_sync_frame_map_readiness.py \
  --anchors-jsonl $(quote "${REMOTE_ANCHORS}") \
  --frame-map-jsonl $(quote "${REMOTE_OUTPUT}/solver/sync_smooth_paths.jsonl") \
  --solver-report $(quote "${REMOTE_OUTPUT}/solver/sync_smooth_path_report.json") \
  --frames ${READINESS_FRAMES} \
  --cams ${READINESS_CAMS} \
  --min-accepted-per-cam $(quote "${MIN_ACCEPTED_PER_CAM}") \
  --output $(quote "${REMOTE_OUTPUT}/sync_frame_map_readiness.json")
readiness_code=\$?
printf '%s\n' "\${readiness_code}" > $(quote "${REMOTE_OUTPUT}/sync_frame_map_readiness.exit_code")
set -e
$(quote "${REMOTE_VENV}/bin/python") scripts/build_sync_anchor_review_pack.py \
  --lx-file $(quote "${REMOTE_DATASET}/MANIFOLD_MT20260616-175807.lx") \
  --candidates-jsonl $(quote "${REMOTE_INPUTS}/sync_candidates.jsonl") \
  --smooth-path-jsonl $(quote "${REMOTE_OUTPUT}/solver/sync_smooth_paths.jsonl") \
  --output-dir $(quote "${REMOTE_OUTPUT}/review") \
  --top-n $(quote "${TOP_N}") \
  --sheet-cols $(quote "${SHEET_COLS}")
EOF
)

if [[ "${DRY_RUN}" == "1" ]]; then
  cat <<EOF
dry_run=1
server=${SERVER}
local_anchors=${LOCAL_ANCHORS}
remote_anchors=${REMOTE_ANCHORS}
remote_output=${REMOTE_OUTPUT}
local_output=${LOCAL_OUTPUT}
remote_cmd=${remote_cmd}
readiness_frames=${READINESS_FRAMES}
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
readiness_report=${readiness_report}
review_url=http://127.0.0.1:8765/server_parking_priority_s10/${RUN_NAME}/review/manual_anchor_review.html
EOF
