#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-scan-train}"
BIND_ADDRESS="${BIND_ADDRESS:-}"
REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR:-/root/epfs/new_route_scripts}"
LOCAL_SCRIPT_DIR="${LOCAL_SCRIPT_DIR:-/Users/skkac/Work/SCAN/new_route/scripts}"

PATCH_SCENE_PROMPTS="${PATCH_SCENE_PROMPTS:-1}"
SHARDS="${SHARDS:-4}"

ssh_opts=()
if [[ -n "${BIND_ADDRESS}" ]]; then
  ssh_opts+=("-o" "BindAddress=${BIND_ADDRESS}")
fi

echo "[1/3] checking SSH connectivity: ${SERVER}"
ssh "${ssh_opts[@]}" -o ConnectTimeout=8 "${SERVER}" 'hostname; date'

echo "[2/3] syncing scripts to ${SERVER}:${REMOTE_SCRIPT_DIR}"
tar -C "${LOCAL_SCRIPT_DIR}" -cf - . | ssh "${ssh_opts[@]}" "${SERVER}" "mkdir -p '${REMOTE_SCRIPT_DIR}' && tar -C '${REMOTE_SCRIPT_DIR}' -xf -"

echo "[3/3] running scene-aware semantic completion on server"
ssh "${ssh_opts[@]}" "${SERVER}" "cd '${REMOTE_SCRIPT_DIR}' && PATCH_SCENE_PROMPTS='${PATCH_SCENE_PROMPTS}' SHARDS='${SHARDS}' bash ./run_server_semantic_completion_sharded.sh"
