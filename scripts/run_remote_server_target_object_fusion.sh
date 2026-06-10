#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-scan-train}"
BIND_ADDRESS="${BIND_ADDRESS:-}"
REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR:-/root/epfs/new_route_scripts}"
LOCAL_SCRIPT_DIR="${LOCAL_SCRIPT_DIR:-/Users/skkac/Work/SCAN/new_route/scripts}"

REMOTE_OUTPUT_DIR="${REMOTE_OUTPUT_DIR:-/root/epfs/new_route_stage1_skymask/target_object_fusion_0000_0999}"
LOCAL_OUTPUT_DIR="${LOCAL_OUTPUT_DIR:-/Users/skkac/Work/SCAN/server_resume_target_object_fusion_0000_0999}"

SEMANTIC_EVAL_DIR="${SEMANTIC_EVAL_DIR:-/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999}"
COMBO="${COMBO:-sam2_prompt_v3_sky_label_merge_completion}"
MIN_MERGE_CONFIDENCE="${MIN_MERGE_CONFIDENCE:-0.5}"
WORK_MODE="${WORK_MODE:-semantic-dir}"
START_FRAME="${START_FRAME:-0}"
END_FRAME="${END_FRAME:-999}"

ssh_opts=()
if [[ -n "${BIND_ADDRESS}" ]]; then
  ssh_opts+=("-o" "BindAddress=${BIND_ADDRESS}")
fi

echo "[1/4] checking SSH connectivity: ${SERVER}"
ssh "${ssh_opts[@]}" -o ConnectTimeout=8 "${SERVER}" 'hostname; date'

echo "[2/4] syncing scripts to ${SERVER}:${REMOTE_SCRIPT_DIR}"
tar -C "${LOCAL_SCRIPT_DIR}" -cf - . | ssh "${ssh_opts[@]}" "${SERVER}" "mkdir -p '${REMOTE_SCRIPT_DIR}' && tar -C '${REMOTE_SCRIPT_DIR}' -xf -"

echo "[3/4] running target/object fusion on server"
ssh "${ssh_opts[@]}" "${SERVER}" "cd '${REMOTE_SCRIPT_DIR}' && \
  SEMANTIC_EVAL_DIR='${SEMANTIC_EVAL_DIR}' \
  COMBO='${COMBO}' \
  OUTPUT_DIR='${REMOTE_OUTPUT_DIR}' \
  MIN_MERGE_CONFIDENCE='${MIN_MERGE_CONFIDENCE}' \
  WORK_MODE='${WORK_MODE}' \
  START_FRAME='${START_FRAME}' \
  END_FRAME='${END_FRAME}' \
  bash ./run_server_target_object_fusion.sh"

echo "[4/4] pulling fusion QA artifacts"
mkdir -p "${LOCAL_OUTPUT_DIR}/reports" "${LOCAL_OUTPUT_DIR}/objects"
scp "${ssh_opts[@]}" "${SERVER}:${REMOTE_OUTPUT_DIR}/reports/target_object_qa.json" "${LOCAL_OUTPUT_DIR}/reports/target_object_qa.json"
scp "${ssh_opts[@]}" "${SERVER}:${REMOTE_OUTPUT_DIR}/objects/fusion_report.json" "${LOCAL_OUTPUT_DIR}/objects/fusion_report.json"
scp "${ssh_opts[@]}" "${SERVER}:${REMOTE_OUTPUT_DIR}/objects/objects.jsonl" "${LOCAL_OUTPUT_DIR}/objects/objects.jsonl"

echo "target/object fusion QA: ${LOCAL_OUTPUT_DIR}/reports/target_object_qa.json"
