#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-scan-train}"
BIND_ADDRESS="${BIND_ADDRESS:-}"
REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR:-/root/epfs/new_route_scripts}"
LOCAL_SCRIPT_DIR="${LOCAL_SCRIPT_DIR:-/Users/skkac/Work/SCAN/new_route/scripts}"

REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-/root/epfs/new_route_stage1_skymask/surface_first_subcluster_qa_0000_0999}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-/Users/skkac/Work/SCAN/server_surface_first_subcluster_qa_0000_0999}"
INPUT_PLY="${INPUT_PLY:-/root/epfs/new_route_stage1_skymask/target_object_fusion_0000_0999/objects/object_centroids.ply}"
TMUX_SESSION="${TMUX_SESSION:-surface_first_subcluster_qa}"
WAIT="${WAIT:-1}"

ssh_opts=()
if [[ -n "${BIND_ADDRESS}" ]]; then
  ssh_opts+=("-o" "BindAddress=${BIND_ADDRESS}")
fi

echo "[1/5] checking SSH connectivity: ${SERVER}"
ssh "${ssh_opts[@]}" -o ConnectTimeout=8 "${SERVER}" 'hostname; date'

echo "[2/5] syncing scripts to ${SERVER}:${REMOTE_SCRIPT_DIR}"
COPYFILE_DISABLE=1 tar --no-xattrs -C "${LOCAL_SCRIPT_DIR}" -cf - . | ssh "${ssh_opts[@]}" "${SERVER}" "mkdir -p '${REMOTE_SCRIPT_DIR}' && tar -C '${REMOTE_SCRIPT_DIR}' -xf -"

remote_cmd="cd '${REMOTE_SCRIPT_DIR}' && INPUT_PLY='${INPUT_PLY}' OUTPUT_DIR='${REMOTE_OUTPUT_DIR}' bash ./run_server_surface_first_subcluster_qa.sh"

echo "[3/5] starting tmux job: ${TMUX_SESSION}"
ssh "${ssh_opts[@]}" "${SERVER}" "tmux has-session -t '${TMUX_SESSION}' 2>/dev/null && tmux kill-session -t '${TMUX_SESSION}' || true; tmux new-session -d -s '${TMUX_SESSION}' \"${remote_cmd}\""

if [[ "${WAIT}" == "1" ]]; then
  echo "[4/5] waiting for tmux job to finish"
  while ssh "${ssh_opts[@]}" "${SERVER}" "tmux has-session -t '${TMUX_SESSION}' 2>/dev/null"; do
    sleep 10
  done
else
  echo "[4/5] not waiting; inspect with: ssh ${SERVER} 'tmux attach -t ${TMUX_SESSION}'"
  exit 0
fi

echo "[5/5] pulling QA artifacts"
mkdir -p "${LOCAL_OUTPUT_DIR}"
scp "${ssh_opts[@]}" "${SERVER}:${REMOTE_OUTPUT_DIR}/surface_first_subcluster_report.json" "${LOCAL_OUTPUT_DIR}/" || true
scp "${ssh_opts[@]}" "${SERVER}:${REMOTE_OUTPUT_DIR}/object_points_surface_first_subcluster_xy.png" "${LOCAL_OUTPUT_DIR}/" || true
scp "${ssh_opts[@]}" "${SERVER}:${REMOTE_OUTPUT_DIR}/object_points_surface_first_subcluster_voxel004.ply" "${LOCAL_OUTPUT_DIR}/" || true

echo "local output: ${LOCAL_OUTPUT_DIR}"
