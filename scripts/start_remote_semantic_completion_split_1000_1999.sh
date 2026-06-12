#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-10.0.8.114}"
SSH_USER="${SSH_USER:-root}"
TRAIN_PORT="${TRAIN_PORT:-31909}"
VLM_PORT="${VLM_PORT:-31079}"
BIND_ADDRESS="${BIND_ADDRESS:-}"
CONNECT_TIMEOUT="${CONNECT_TIMEOUT:-8}"

LOCAL_SCRIPT_DIR="${LOCAL_SCRIPT_DIR:-/Users/skkac/Work/SCAN/new_route/scripts}"
REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR:-/root/epfs/new_route_scripts}"
TAR_BIN="${TAR_BIN:-bsdtar}"

START="${START:-1000}"
END="${END:-1999}"
SPLIT_FRAME="${SPLIT_FRAME:-1500}"
CAMS_PER_FRAME="${CAMS_PER_FRAME:-3}"
MANIFEST="${MANIFEST:-/root/epfs/new_route_stage1_skymask/semantic_manifest_${START}_${END}.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_${START}_${END}}"
SAM_MASKS_DIR="${SAM_MASKS_DIR:-/root/epfs/new_route_stage1_skymask/sam_masks_${START}_${END}_combined}"
LOG_DIR="${LOG_DIR:-/root/epfs/new_route_stage1_skymask/logs}"

SHARDS="${SHARDS:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
QWEN_PORT="${QWEN_PORT:-8001}"
QWEN_PARALLEL="${QWEN_PARALLEL:-4}"
START_QWEN="${START_QWEN:-0}"
STOP_VLM_EXTRA_LOOP="${STOP_VLM_EXTRA_LOOP:-1}"
PATCH_SCENE_PROMPTS_HEAD="${PATCH_SCENE_PROMPTS_HEAD:-1}"
PATCH_SCENE_PROMPTS_TAIL="${PATCH_SCENE_PROMPTS_TAIL:-0}"

TRAIN_SESSION="${TRAIN_SESSION:-semantic_completion_${START}_${END}_head}"
VLM_SESSION="${VLM_SESSION:-semantic_completion_${START}_${END}_tail_vlm}"
RESTART_EXISTING="${RESTART_EXISTING:-0}"

ssh_target="${SSH_USER}@${SSH_HOST}"

if [[ "${SPLIT_FRAME}" -le "${START}" || "${SPLIT_FRAME}" -gt "${END}" ]]; then
  echo "SPLIT_FRAME must satisfy START < SPLIT_FRAME <= END" >&2
  exit 2
fi

ssh_opts_for_port() {
  local port="$1"
  SSH_OPTS=(-F /dev/null -o BatchMode=yes -o "ConnectTimeout=${CONNECT_TIMEOUT}" -p "${port}")
  if [[ -n "${BIND_ADDRESS}" ]]; then
    SSH_OPTS+=("-o" "BindAddress=${BIND_ADDRESS}")
  fi
}

sync_scripts() {
  local port="$1"
  ssh_opts_for_port "${port}"
  COPYFILE_DISABLE=1 "${TAR_BIN}" --no-xattrs -C "${LOCAL_SCRIPT_DIR}" --exclude='__pycache__' --exclude='._*' -cf - . \
    | ssh "${SSH_OPTS[@]}" "${ssh_target}" "mkdir -p '${REMOTE_SCRIPT_DIR}' && tar -C '${REMOTE_SCRIPT_DIR}' -xf - && chmod +x '${REMOTE_SCRIPT_DIR}'/*.sh"
}

start_qwen() {
  local port="$1"
  local cuda_visible="$2"
  local label="$3"
  ssh_opts_for_port "${port}"
  ssh "${SSH_OPTS[@]}" "${ssh_target}" \
    REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR}" \
    CUDA_VISIBLE_DEVICES="${cuda_visible}" \
    QWEN_PORT="${QWEN_PORT}" \
    QWEN_PARALLEL="${QWEN_PARALLEL}" \
    LABEL="${label}" \
    'bash -s' <<'REMOTE'
set -euo pipefail
cd "${REMOTE_SCRIPT_DIR}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
LLAMA_SERVER="${LLAMA_SERVER:-/root/epfs/llama-server/bin/llama-server}" \
PARALLEL="${QWEN_PARALLEL}" \
PORT="${QWEN_PORT}" \
LOG="/root/epfs/qwen_vl_server_${QWEN_PORT}_${LABEL}.log" \
bash ./restart_qwen_vl_server.sh
REMOTE
}

start_runner() {
  local port="$1"
  local session="$2"
  local start_index="$3"
  local end_index="$4"
  local work_dir="$5"
  local patch_scene_prompts="$6"
  local log_file="$7"
  local pid_file="$8"
  ssh_opts_for_port "${port}"
  ssh "${SSH_OPTS[@]}" "${ssh_target}" \
    REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR}" \
    SESSION_NAME="${session}" \
    LOG_FILE="${log_file}" \
    PID_FILE="${pid_file}" \
    RESTART_EXISTING="${RESTART_EXISTING}" \
    MANIFEST="${MANIFEST}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    SAM_MASKS_DIR="${SAM_MASKS_DIR}" \
    START_INDEX="${start_index}" \
    END_INDEX="${end_index}" \
    SHARDS="${SHARDS}" \
    CHUNK_SIZE="${CHUNK_SIZE}" \
    MAX_TOKENS="${MAX_TOKENS}" \
    PATCH_SCENE_PROMPTS="${patch_scene_prompts}" \
    WORK_DIR="${work_dir}" \
    QWEN_PORT="${QWEN_PORT}" \
    'bash -s' <<'REMOTE'
set -euo pipefail
mkdir -p "$(dirname "${LOG_FILE}")" "$(dirname "${PID_FILE}")"
runner_cmd="cd '${REMOTE_SCRIPT_DIR}' && MANIFEST='${MANIFEST}' OUTPUT_DIR='${OUTPUT_DIR}' SAM_MASKS_DIR='${SAM_MASKS_DIR}' START_INDEX='${START_INDEX}' END_INDEX='${END_INDEX}' SHARDS='${SHARDS}' CHUNK_SIZE='${CHUNK_SIZE}' MAX_TOKENS='${MAX_TOKENS}' PATCH_SCENE_PROMPTS='${PATCH_SCENE_PROMPTS}' WORK_DIR='${WORK_DIR}' VLM_ENDPOINT='http://localhost:${QWEN_PORT}/v1/chat/completions' bash ./run_server_semantic_completion_sharded.sh"

if command -v tmux >/dev/null 2>&1; then
  if tmux has-session -t "${SESSION_NAME}" 2>/dev/null; then
    if [[ "${RESTART_EXISTING}" == "1" ]]; then
      tmux kill-session -t "${SESSION_NAME}"
    else
      echo "already_running_session=${SESSION_NAME}"
      tmux list-sessions | grep "^${SESSION_NAME}:"
      exit 0
    fi
  fi
  tmux new-session -d -s "${SESSION_NAME}" "${runner_cmd} > '${LOG_FILE}' 2>&1"
  echo "started_session=${SESSION_NAME}"
  exit 0
fi

old_pid="$(cat "${PID_FILE}" 2>/dev/null || true)"
if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
  if [[ "${RESTART_EXISTING}" == "1" ]]; then
    kill "${old_pid}" || true
    sleep 3
  else
    echo "already_running_pid=${old_pid}"
    exit 0
  fi
fi
nohup bash -lc "${runner_cmd}" >"${LOG_FILE}" 2>&1 &
echo "$!" >"${PID_FILE}"
echo "started_pid=$(cat "${PID_FILE}")"
REMOTE
}

stop_vlm_extra_loop() {
  ssh_opts_for_port "${VLM_PORT}"
  ssh "${SSH_OPTS[@]}" "${ssh_target}" 'bash -s' <<'REMOTE'
set -euo pipefail
mapfile -t pids < <(pgrep -f '[r]un_server_vlm_extra_loop.sh|[_]sharded_work_vlm_extra' || true)
if [[ "${#pids[@]}" -eq 0 ]]; then
  echo "vlm_extra_loop=not_running"
  exit 0
fi
echo "stopping_vlm_extra_loop=${pids[*]}"
kill "${pids[@]}" 2>/dev/null || true
sleep 5
mapfile -t remaining < <(pgrep -f '[r]un_server_vlm_extra_loop.sh|[_]sharded_work_vlm_extra' || true)
if [[ "${#remaining[@]}" -gt 0 ]]; then
  kill -9 "${remaining[@]}" 2>/dev/null || true
fi
REMOTE
}

echo "[1/4] syncing scripts to scan-train and scan-vlm"
sync_scripts "${TRAIN_PORT}"
sync_scripts "${VLM_PORT}"

head_start_index=0
head_end_index=$(((SPLIT_FRAME - START) * CAMS_PER_FRAME))
tail_start_index="${head_end_index}"
tail_end_index=$(((END - START + 1) * CAMS_PER_FRAME))

if [[ "${START_QWEN}" == "1" ]]; then
  echo "[2/4] starting Qwen endpoints"
  start_qwen "${TRAIN_PORT}" 1 "scan_train"
  start_qwen "${VLM_PORT}" 0 "scan_vlm"
else
  echo "[2/4] leaving existing Qwen endpoints untouched; set START_QWEN=1 for a fresh launch"
fi

if [[ "${STOP_VLM_EXTRA_LOOP}" == "1" ]]; then
  echo "[2/post] stopping old scan-vlm extra loop before split tail starts"
  stop_vlm_extra_loop
fi

echo "[3/4] starting scan-train head runner ${START}-$((SPLIT_FRAME - 1))"
start_runner \
  "${TRAIN_PORT}" \
  "${TRAIN_SESSION}" \
  "${head_start_index}" \
  "${head_end_index}" \
  "${OUTPUT_DIR}/_sharded_work_train_head" \
  "${PATCH_SCENE_PROMPTS_HEAD}" \
  "${LOG_DIR}/${TRAIN_SESSION}.log" \
  "${LOG_DIR}/${TRAIN_SESSION}.pid"

echo "[4/4] starting scan-vlm tail runner ${SPLIT_FRAME}-${END}"
start_runner \
  "${VLM_PORT}" \
  "${VLM_SESSION}" \
  "${tail_start_index}" \
  "${tail_end_index}" \
  "${OUTPUT_DIR}/_sharded_work_vlm_tail" \
  "${PATCH_SCENE_PROMPTS_TAIL}" \
  "${LOG_DIR}/${VLM_SESSION}.log" \
  "${LOG_DIR}/${VLM_SESSION}.pid"

echo "done"
