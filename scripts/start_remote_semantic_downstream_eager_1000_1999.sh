#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-10.0.8.114}"
SSH_PORT="${SSH_PORT:-31909}"
SSH_USER="${SSH_USER:-root}"
BIND_ADDRESS="${BIND_ADDRESS:-}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-8}"
LOCAL_SCRIPT_DIR="${LOCAL_SCRIPT_DIR:-/Users/skkac/Work/SCAN/new_route/scripts}"
REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR:-/root/epfs/new_route_scripts}"
TAR_BIN="${TAR_BIN:-bsdtar}"

START="${START:-1000}"
END="${END:-1999}"
START_INDEX="${START_INDEX:-1500}"
END_INDEX="${END_INDEX:-3000}"
SESSION_NAME="${SESSION_NAME:-semantic_downstream_eager_${START}_${END}_train}"
WAIT_FOR_SESSION="${WAIT_FOR_SESSION:-}"
FOLLOWUP="${FOLLOWUP:-0}"
LOG_DIR="${LOG_DIR:-/root/epfs/new_route_stage1_skymask/logs}"
MANIFEST="${MANIFEST:-/root/epfs/new_route_stage1_skymask/semantic_manifest_${START}_${END}.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_${START}_${END}}"
SAM_MASKS_DIR="${SAM_MASKS_DIR:-/root/epfs/new_route_stage1_skymask/sam_masks_${START}_${END}_combined}"
SHARDS="${SHARDS:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
PATCH_SCENE_PROMPTS="${PATCH_SCENE_PROMPTS:-1}"
QWEN_PORT="${QWEN_PORT:-8001}"
RESTART_EXISTING="${RESTART_EXISTING:-0}"

ssh_target="${SSH_USER}@${SSH_HOST}"
ssh_opts=(-F /dev/null -o BatchMode=yes -o "ConnectTimeout=${CONNECT_TIMEOUT}" -p "${SSH_PORT}")
if [[ -n "${BIND_ADDRESS}" ]]; then
  ssh_opts+=("-o" "BindAddress=${BIND_ADDRESS}")
fi

echo "[1/2] syncing scripts to scan-train"
COPYFILE_DISABLE=1 "${TAR_BIN}" --no-xattrs -C "${LOCAL_SCRIPT_DIR}" --exclude='__pycache__' --exclude='._*' -cf - . \
  | ssh "${ssh_opts[@]}" "${ssh_target}" "mkdir -p '${REMOTE_SCRIPT_DIR}' && tar -C '${REMOTE_SCRIPT_DIR}' -xf - && chmod +x '${REMOTE_SCRIPT_DIR}'/*.sh"

echo "[2/2] starting downstream-only eager runner"
ssh "${ssh_opts[@]}" "${ssh_target}" \
  SESSION_NAME="${SESSION_NAME}" \
  WAIT_FOR_SESSION="${WAIT_FOR_SESSION}" \
  FOLLOWUP="${FOLLOWUP}" \
  LOG_DIR="${LOG_DIR}" \
  REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR}" \
  MANIFEST="${MANIFEST}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  SAM_MASKS_DIR="${SAM_MASKS_DIR}" \
  START_INDEX="${START_INDEX}" \
  END_INDEX="${END_INDEX}" \
  SHARDS="${SHARDS}" \
  CHUNK_SIZE="${CHUNK_SIZE}" \
  MAX_TOKENS="${MAX_TOKENS}" \
  PATCH_SCENE_PROMPTS="${PATCH_SCENE_PROMPTS}" \
  QWEN_PORT="${QWEN_PORT}" \
  RESTART_EXISTING="${RESTART_EXISTING}" \
  'bash -s' <<'REMOTE'
set -euo pipefail
mkdir -p "${LOG_DIR}"

if [[ "${FOLLOWUP}" == "1" && -z "${WAIT_FOR_SESSION}" ]]; then
  echo "FOLLOWUP=1 requires WAIT_FOR_SESSION" >&2
  exit 2
fi

if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
  if [[ "${RESTART_EXISTING}" == "1" ]]; then
    tmux kill-session -t "${SESSION_NAME}"
  else
    echo "already_running_session=${SESSION_NAME}"
    tmux list-sessions | grep "^${SESSION_NAME}:"
    exit 0
  fi
fi

log_file="${LOG_DIR}/${SESSION_NAME}.log"
runner='cd '"${REMOTE_SCRIPT_DIR}"' && RUN_ID=${SESSION_NAME}_$(date +%Y%m%d_%H%M%S) && MANIFEST='"'"${MANIFEST}"'"' OUTPUT_DIR='"'"${OUTPUT_DIR}"'"' SAM_MASKS_DIR='"'"${SAM_MASKS_DIR}"'"' START_INDEX='"'"${START_INDEX}"'"' END_INDEX='"'"${END_INDEX}"'"' SHARDS='"'"${SHARDS}"'"' CHUNK_SIZE='"'"${CHUNK_SIZE}"'"' MAX_TOKENS='"'"${MAX_TOKENS}"'"' PATCH_SCENE_PROMPTS='"'"${PATCH_SCENE_PROMPTS}"'"' WORK_DIR='"'"${OUTPUT_DIR}"'"'/_sharded_work_${RUN_ID} LOG_DIR='"'"${OUTPUT_DIR}"'"'/_sharded_work_${RUN_ID}/logs VLM_ENDPOINT=http://localhost:'"${QWEN_PORT}"'/v1/chat/completions SKIP_SAM2_QWEN=1 bash ./run_server_semantic_completion_sharded.sh'

if [[ "${FOLLOWUP}" == "1" ]]; then
  cmd="while tmux has-session -t '${WAIT_FOR_SESSION}' 2>/dev/null; do sleep 120; done; ${runner}"
else
  cmd="${runner}"
fi

tmux new-session -d -s "${SESSION_NAME}" "${cmd} >> '${log_file}' 2>&1"
echo "started_session=${SESSION_NAME}"
tmux list-sessions | grep -E "^${SESSION_NAME}:|^${WAIT_FOR_SESSION}:" || true
REMOTE
