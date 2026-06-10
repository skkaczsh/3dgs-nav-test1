#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-scan-train}"
BIND_ADDRESS="${BIND_ADDRESS:-}"
QA_SCRIPT="${QA_SCRIPT:-/Users/skkac/Work/SCAN/new_route/scripts/qa_dataset_readiness.py}"
REMOTE_REPORT="${REMOTE_REPORT:-/root/epfs/new_route_stage1_skymask/server_dataset_readiness_0000_0999.json}"
LOCAL_REPORT="${LOCAL_REPORT:-/Users/skkac/Work/SCAN/route_status_20260610/server_dataset_readiness_0000_0999.json}"

START_FRAME="${START_FRAME:-0}"
END_FRAME="${END_FRAME:-999}"
COMBO="${COMBO:-sam2_prompt_v3_sky_label_merge_completion}"

FRAMES_DIR="${FRAMES_DIR:-/root/epfs/new_route_stage1_skymask/frames}"
COLOR_DIR="${COLOR_DIR:-/root/epfs/new_route_stage1_skymask/output}"
SKY_MASK_DIR="${SKY_MASK_DIR:-/root/epfs/new_route_data/sky_masks_color}"
SAM_MASKS_DIR="${SAM_MASKS_DIR:-/root/epfs/new_route_stage1_skymask/sam_masks_0000_0999_combined}"
SEMANTIC_EVAL_DIR="${SEMANTIC_EVAL_DIR:-/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999}"

ssh_opts=()
if [[ -n "${BIND_ADDRESS}" ]]; then
  ssh_opts+=("-o" "BindAddress=${BIND_ADDRESS}")
fi

echo "[1/4] checking SSH connectivity: ${SERVER}"
ssh "${ssh_opts[@]}" -o ConnectTimeout=8 "${SERVER}" 'hostname; date'

echo "[2/4] uploading dataset readiness script"
scp "${ssh_opts[@]}" "${QA_SCRIPT}" "${SERVER}:/tmp/qa_dataset_readiness.py"

echo "[3/4] running dataset readiness QA on server"
ssh "${ssh_opts[@]}" "${SERVER}" "python3 /tmp/qa_dataset_readiness.py \
  --frames-dir '${FRAMES_DIR}' \
  --color-dir '${COLOR_DIR}' \
  --sky-mask-dir '${SKY_MASK_DIR}' \
  --sam-masks-dir '${SAM_MASKS_DIR}' \
  --semantic-eval-dir '${SEMANTIC_EVAL_DIR}' \
  --combo '${COMBO}' \
  --start '${START_FRAME}' \
  --end '${END_FRAME}' \
  --output '${REMOTE_REPORT}'"

echo "[4/4] pulling dataset readiness report"
mkdir -p "$(dirname "${LOCAL_REPORT}")"
scp "${ssh_opts[@]}" "${SERVER}:${REMOTE_REPORT}" "${LOCAL_REPORT}"

echo "dataset readiness report: ${LOCAL_REPORT}"
