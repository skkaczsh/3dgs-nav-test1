#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-10.0.8.114}"
SSH_PORT="${SSH_PORT:-31079}"
SSH_USER="${SSH_USER:-root}"
LOCAL_SCRIPT_DIR="${LOCAL_SCRIPT_DIR:-/Users/skkac/Work/SCAN/new_route/scripts}"
REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR:-/root/epfs/new_route_scripts}"
TAR_BIN="${TAR_BIN:-bsdtar}"

START="${START:-1000}"
END="${END:-1999}"
SESSION_NAME="${SESSION_NAME:-vlm_extra_loop_${START}_${END}}"
LOG_DIR="${LOG_DIR:-/root/epfs/new_route_stage1_skymask/logs}"
SLEEP_SECONDS="${SLEEP_SECONDS:-180}"
MAX_CYCLES="${MAX_CYCLES:-0}"
MAX_ITEMS_PER_CYCLE="${MAX_ITEMS_PER_CYCLE:-24}"
MIN_SAM_AGE_SECONDS="${MIN_SAM_AGE_SECONDS:-30}"

ssh_target="${SSH_USER}@${SSH_HOST}"
ssh_opts=(-F /dev/null -p "${SSH_PORT}")

COPYFILE_DISABLE=1 "${TAR_BIN}" --no-xattrs -C "${LOCAL_SCRIPT_DIR}" --exclude='__pycache__' --exclude='._*' -cf - . \
  | ssh "${ssh_opts[@]}" "${ssh_target}" "mkdir -p '${REMOTE_SCRIPT_DIR}' && tar -C '${REMOTE_SCRIPT_DIR}' -xf - && chmod +x '${REMOTE_SCRIPT_DIR}'/*.sh"

ssh "${ssh_opts[@]}" "${ssh_target}" \
  START="${START}" \
  END="${END}" \
  SESSION_NAME="${SESSION_NAME}" \
  LOG_DIR="${LOG_DIR}" \
  SLEEP_SECONDS="${SLEEP_SECONDS}" \
  MAX_CYCLES="${MAX_CYCLES}" \
  MAX_ITEMS_PER_CYCLE="${MAX_ITEMS_PER_CYCLE}" \
  MIN_SAM_AGE_SECONDS="${MIN_SAM_AGE_SECONDS}" \
  REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR}" \
  'bash -s' <<'REMOTE'
set -euo pipefail
mkdir -p "${LOG_DIR}"
if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  echo "already_running_session=${SESSION_NAME}"
  tmux list-sessions | grep "^${SESSION_NAME}:"
  exit 0
fi
tmux new-session -d -s "${SESSION_NAME}" \
  "cd '${REMOTE_SCRIPT_DIR}' && START='${START}' END='${END}' SLEEP_SECONDS='${SLEEP_SECONDS}' MAX_CYCLES='${MAX_CYCLES}' MAX_ITEMS_PER_CYCLE='${MAX_ITEMS_PER_CYCLE}' MIN_SAM_AGE_SECONDS='${MIN_SAM_AGE_SECONDS}' bash ./run_server_vlm_extra_loop.sh >> '${LOG_DIR}/semantic_vlm_extra_loop_${START}_${END}.log' 2>&1"
echo "started_session=${SESSION_NAME}"
REMOTE
