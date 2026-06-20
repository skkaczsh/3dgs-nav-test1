#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
REPO_ROOT="${REPO_ROOT:-/Users/skkac/Work/SCAN/new_route}"
LOG_DIR="${LOG_DIR:-/Users/skkac/Work/SCAN/.local/logs}"
PID_FILE="${PID_FILE:-/Users/skkac/Work/SCAN/.local/semantic_ply_viewer_${PORT}.pid}"
INDEX_REFRESH_INTERVAL="${INDEX_REFRESH_INTERVAL:-0}"
INDEX_PID_FILE="${INDEX_PID_FILE:-/Users/skkac/Work/SCAN/.local/semantic_viewer_index_refresh_${PORT}.pid}"
URL="http://${HOST}:${PORT}/tools/semantic_ply_viewer.html"
INDEX_URL="http://${HOST}:${PORT}/tools/semantic_viewer_index.html"

mkdir -p "${LOG_DIR}" "$(dirname "${PID_FILE}")" "$(dirname "${INDEX_PID_FILE}")"

refresh_index() {
  if [[ -f "${REPO_ROOT}/scripts/build_semantic_viewer_index.py" ]]; then
    (
      cd "${REPO_ROOT}"
      python3 scripts/build_semantic_viewer_index.py >/dev/null
    )
  fi
}

if lsof -nP -iTCP:"${PORT}" -sTCP:LISTEN >/dev/null 2>&1; then
  echo "already_listening=${HOST}:${PORT}"
else
  echo "starting=${HOST}:${PORT}"
  (
    cd "${REPO_ROOT}"
    nohup python3 -m http.server "${PORT}" --bind "${HOST}" > "${LOG_DIR}/semantic_ply_viewer_${PORT}.log" 2>&1 &
    echo "$!" > "${PID_FILE}"
  )
fi

refresh_index

if [[ "${INDEX_REFRESH_INTERVAL}" != "0" ]]; then
  if [[ -f "${INDEX_PID_FILE}" ]] && kill -0 "$(cat "${INDEX_PID_FILE}")" >/dev/null 2>&1; then
    kill "$(cat "${INDEX_PID_FILE}")" >/dev/null 2>&1 || true
  fi
  (
    while true; do
      refresh_index >> "${LOG_DIR}/semantic_viewer_index_refresh_${PORT}.log" 2>&1 || true
      sleep "${INDEX_REFRESH_INTERVAL}"
    done
  ) &
  echo "$!" > "${INDEX_PID_FILE}"
fi

sleep 1
curl -fsS --max-time 5 -I "${URL}" >/dev/null
echo "viewer_url=${URL}"
echo "index_url=${INDEX_URL}"
echo "pid_file=${PID_FILE}"
if [[ "${INDEX_REFRESH_INTERVAL}" != "0" ]]; then
  echo "index_refresh_interval=${INDEX_REFRESH_INTERVAL}"
  echo "index_pid_file=${INDEX_PID_FILE}"
fi
