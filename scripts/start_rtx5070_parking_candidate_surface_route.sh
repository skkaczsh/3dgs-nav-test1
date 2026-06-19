#!/usr/bin/env bash
set -euo pipefail

# Local launcher for the RTX 5070Ti parking candidate rebuild.
#
# This is the safe entrypoint for remote execution:
#   1. run the local->remote healthcheck
#   2. run the remote candidate script in CHECK_ONLY mode
#   3. start the real rebuild inside a named tmux session
#
# The rebuild script itself is idempotent unless FORCE=1 is passed through.

SERVER="${SERVER:-scan-rtx5070}"
REMOTE_REPO="${REMOTE_REPO:-/home/zsh/Work/SCAN/new_route}"
REMOTE_WORK="${REMOTE_WORK:-/home/zsh/Work/SCAN/work_MT20260616-175807}"
LOCAL_REPO="${LOCAL_REPO:-/Users/skkac/Work/SCAN/new_route}"
SESSION_NAME="${SESSION_NAME:-rtx5070_parking_candidate}"
LOG_DIR="${LOG_DIR:-${REMOTE_WORK}/logs}"
REMOTE_SCRIPT="${REMOTE_SCRIPT:-scripts/run_rtx5070_parking_candidate_surface_route.sh}"
HEALTHCHECK="${HEALTHCHECK:-${LOCAL_REPO}/scripts/check_rtx5070_parking_runtime.py}"
HEALTHCHECK_OUTPUT="${HEALTHCHECK_OUTPUT:-${LOCAL_REPO}/server_parking_priority_s10/parking_candidate_manifest_rtx5070/rtx5070_runtime_check.json}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-8}"
BIND_ADDRESS="${BIND_ADDRESS:-}"
DRY_RUN="${DRY_RUN:-0}"
RESTART="${RESTART:-0}"
FORCE="${FORCE:-0}"
VIEWER_STRIDE="${VIEWER_STRIDE:-10}"
QA_CANDIDATE_LIMIT="${QA_CANDIDATE_LIMIT:-160}"
QA_EVIDENCE_PER_OBJECT="${QA_EVIDENCE_PER_OBJECT:-3}"

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

remote_env=(
  "WORK=${REMOTE_WORK}"
  "FORCE=${FORCE}"
  "VIEWER_STRIDE=${VIEWER_STRIDE}"
  "QA_CANDIDATE_LIMIT=${QA_CANDIDATE_LIMIT}"
  "QA_EVIDENCE_PER_OBJECT=${QA_EVIDENCE_PER_OBJECT}"
)

remote_env_prefix=""
for item in "${remote_env[@]}"; do
  remote_env_prefix+="$(quote "${item}") "
done

remote_log="${LOG_DIR}/${SESSION_NAME}.log"
remote_cmd="set -euo pipefail; cd $(quote "${REMOTE_REPO}"); mkdir -p $(quote "${LOG_DIR}"); ${remote_env_prefix}bash $(quote "${REMOTE_SCRIPT}") 2>&1 | tee $(quote "${remote_log}")"

log "[1/4] healthcheck: ${SERVER}"
"${HEALTHCHECK}" \
  --host "${SERVER}" \
  --remote-repo "${REMOTE_REPO}" \
  --remote-work "${REMOTE_WORK}" \
  --output "${HEALTHCHECK_OUTPUT}"

log "[2/4] remote check_only"
ssh "${ssh_opts[@]}" "${SERVER}" \
  "cd $(quote "${REMOTE_REPO}") && WORK=$(quote "${REMOTE_WORK}") CHECK_ONLY=1 bash $(quote "${REMOTE_SCRIPT}")"

log "[3/4] tmux session check: ${SESSION_NAME}"
if ssh "${ssh_opts[@]}" "${SERVER}" "tmux has-session -t $(quote "${SESSION_NAME}") 2>/dev/null"; then
  if [[ "${RESTART}" != "1" ]]; then
    echo "session_exists=${SESSION_NAME}" >&2
    echo "inspect: ssh ${SERVER} 'tmux attach -t ${SESSION_NAME}'" >&2
    echo "set RESTART=1 to kill and restart this session" >&2
    exit 2
  fi
  log "restart requested; killing existing session: ${SESSION_NAME}"
  ssh "${ssh_opts[@]}" "${SERVER}" "tmux kill-session -t $(quote "${SESSION_NAME}")"
fi

log "[4/4] start tmux"
if [[ "${DRY_RUN}" == "1" ]]; then
  cat <<EOF
dry_run=1
server=${SERVER}
session=${SESSION_NAME}
remote_log=${remote_log}
remote_cmd=${remote_cmd}
EOF
  exit 0
fi

ssh "${ssh_opts[@]}" "${SERVER}" \
  "tmux new-session -d -s $(quote "${SESSION_NAME}") $(quote "bash -lc $(quote "${remote_cmd}")")"

cat <<EOF
started=1
server=${SERVER}
session=${SESSION_NAME}
remote_log=${remote_log}
inspect=ssh ${SERVER} 'tmux attach -t ${SESSION_NAME}'
tail=ssh ${SERVER} 'tail -f ${remote_log}'
EOF
