#!/usr/bin/env bash
set -euo pipefail

HOST="${HOST:-127.0.0.1}"
PORT="${PORT:-8765}"
REPO_ROOT="${REPO_ROOT:-/Users/skkac/Work/SCAN/new_route}"
LOG_DIR="${LOG_DIR:-/Users/skkac/Work/SCAN/.local/logs}"
PID_FILE="${PID_FILE:-/Users/skkac/Work/SCAN/.local/semantic_ply_viewer_${PORT}.pid}"
URL="http://${HOST}:${PORT}/tools/semantic_ply_viewer.html"

mkdir -p "${LOG_DIR}" "$(dirname "${PID_FILE}")"

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

sleep 1
curl -fsS --max-time 5 -I "${URL}" >/dev/null
echo "viewer_url=${URL}"
echo "pid_file=${PID_FILE}"
