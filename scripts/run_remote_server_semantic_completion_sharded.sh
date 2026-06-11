#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-scan-train}"
SSH_HOST="${SSH_HOST:-}"
SSH_PORT="${SSH_PORT:-}"
SSH_USER="${SSH_USER:-root}"
BIND_ADDRESS="${BIND_ADDRESS:-}"
REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR:-/root/epfs/new_route_scripts}"
LOCAL_SCRIPT_DIR="${LOCAL_SCRIPT_DIR:-/Users/skkac/Work/SCAN/new_route/scripts}"
TAR_BIN="${TAR_BIN:-bsdtar}"

PATCH_SCENE_PROMPTS="${PATCH_SCENE_PROMPTS:-1}"
SHARDS="${SHARDS:-4}"

ssh_opts=()
server_target="${SERVER}"
if [[ -n "${SSH_HOST}" ]]; then
  ssh_opts+=("-F" "/dev/null")
  if [[ -n "${SSH_PORT}" ]]; then
    ssh_opts+=("-p" "${SSH_PORT}")
  fi
  server_target="${SSH_USER}@${SSH_HOST}"
fi
if [[ -n "${BIND_ADDRESS}" ]]; then
  ssh_opts+=("-o" "BindAddress=${BIND_ADDRESS}")
fi

echo "[1/3] checking SSH connectivity: ${server_target}"
ssh "${ssh_opts[@]}" -o ConnectTimeout=8 "${server_target}" 'hostname; date'

echo "[2/3] syncing scripts to ${server_target}:${REMOTE_SCRIPT_DIR}"
COPYFILE_DISABLE=1 "${TAR_BIN}" --no-xattrs -C "${LOCAL_SCRIPT_DIR}" --exclude='__pycache__' --exclude='._*' -cf - . \
  | ssh "${ssh_opts[@]}" "${server_target}" "mkdir -p '${REMOTE_SCRIPT_DIR}' && tar -C '${REMOTE_SCRIPT_DIR}' -xf -"

echo "[3/3] running scene-aware semantic completion on server"
ssh "${ssh_opts[@]}" "${server_target}" "cd '${REMOTE_SCRIPT_DIR}' && PATCH_SCENE_PROMPTS='${PATCH_SCENE_PROMPTS}' SHARDS='${SHARDS}' bash ./run_server_semantic_completion_sharded.sh"
