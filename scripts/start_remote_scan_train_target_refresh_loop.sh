#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-10.0.8.114}"
SSH_PORT="${SSH_PORT:-31909}"
SSH_USER="${SSH_USER:-root}"
LOCAL_SCRIPT_DIR="${LOCAL_SCRIPT_DIR:-/Users/skkac/Work/SCAN/new_route/scripts}"
REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR:-/root/epfs/new_route_scripts}"
TAR_BIN="${TAR_BIN:-bsdtar}"

START="${START:-1000}"
END="${END:-1999}"
SESSION_NAME="${SESSION_NAME:-target_refresh_${START}_${END}_surface024_fine012}"
LOG_DIR="${LOG_DIR:-/root/epfs/new_route_stage1_skymask/logs}"
PID_FILE="${PID_FILE:-${LOG_DIR}/${SESSION_NAME}.pid}"
SLEEP_SECONDS="${SLEEP_SECONDS:-900}"
MAX_CYCLES="${MAX_CYCLES:-0}"
MIN_COMPLETION_DELTA="${MIN_COMPLETION_DELTA:-60}"
RUN_ON_FIRST="${RUN_ON_FIRST:-0}"

ssh_target="${SSH_USER}@${SSH_HOST}"
ssh_opts=(-F /dev/null -p "${SSH_PORT}")

COPYFILE_DISABLE=1 "${TAR_BIN}" --no-xattrs -C "${LOCAL_SCRIPT_DIR}" --exclude='__pycache__' --exclude='._*' -cf - . \
  | ssh "${ssh_opts[@]}" "${ssh_target}" "mkdir -p '${REMOTE_SCRIPT_DIR}' && tar -C '${REMOTE_SCRIPT_DIR}' -xf - && chmod +x '${REMOTE_SCRIPT_DIR}'/*.sh"

ssh "${ssh_opts[@]}" "${ssh_target}" \
  START="${START}" \
  END="${END}" \
  SESSION_NAME="${SESSION_NAME}" \
  LOG_DIR="${LOG_DIR}" \
  PID_FILE="${PID_FILE}" \
  SLEEP_SECONDS="${SLEEP_SECONDS}" \
  MAX_CYCLES="${MAX_CYCLES}" \
  MIN_COMPLETION_DELTA="${MIN_COMPLETION_DELTA}" \
  RUN_ON_FIRST="${RUN_ON_FIRST}" \
  REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR}" \
  'bash -s' <<'REMOTE'
set -euo pipefail
mkdir -p "${LOG_DIR}"
if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    echo "already_running_session=${SESSION_NAME}"
    tmux list-sessions | grep "^${SESSION_NAME}:"
    exit 0
  fi
  tmux new-session -d -s "${SESSION_NAME}" \
    "cd '${REMOTE_SCRIPT_DIR}' && START='${START}' END='${END}' SLEEP_SECONDS='${SLEEP_SECONDS}' MAX_CYCLES='${MAX_CYCLES}' MIN_COMPLETION_DELTA='${MIN_COMPLETION_DELTA}' RUN_ON_FIRST='${RUN_ON_FIRST}' bash ./run_server_target_object_refresh_loop.sh >> '${LOG_DIR}/target_object_refresh_loop_${START}_${END}.log' 2>&1"
  echo "started_session=${SESSION_NAME}"
  exit 0
fi

old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
  echo "already_running_pid=${old_pid}"
  exit 0
fi
nohup bash -lc "cd '${REMOTE_SCRIPT_DIR}' && START='${START}' END='${END}' SLEEP_SECONDS='${SLEEP_SECONDS}' MAX_CYCLES='${MAX_CYCLES}' MIN_COMPLETION_DELTA='${MIN_COMPLETION_DELTA}' RUN_ON_FIRST='${RUN_ON_FIRST}' bash ./run_server_target_object_refresh_loop.sh >> '${LOG_DIR}/target_object_refresh_loop_${START}_${END}.log' 2>&1" >/dev/null 2>&1 &
echo "$!" >"${PID_FILE}"
echo "started_pid=$(cat "${PID_FILE}")"
REMOTE
