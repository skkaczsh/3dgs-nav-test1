#!/usr/bin/env bash
set -euo pipefail

SERVER="${SERVER:-scan-train}"
BIND_ADDRESS="${BIND_ADDRESS:-}"
REMOTE_PACK="${REMOTE_PACK:-/root/epfs/frame_fine_cross_candidate_review_pack_0000_0999_v008_strict_high_v2}"
LOCAL_PACK="${LOCAL_PACK:-/Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_pack_v008_v2/frame_fine_cross_candidate_review_pack_0000_0999_v008_strict_high_v2}"
LOCAL_OUTPUT="${LOCAL_OUTPUT:-/Users/skkac/Work/SCAN/server_frame_fine_cross_candidate_review_pack_v008_v2}"
REVIEW_SCRIPT="${REVIEW_SCRIPT:-/Users/skkac/Work/SCAN/new_route/scripts/review_cross_candidate_merges_vlm.py}"
PROMPT_SCRIPT="${PROMPT_SCRIPT:-/Users/skkac/Work/SCAN/new_route/scripts/vlm_scene_prompt.py}"
RESTART_QWEN="${RESTART_QWEN:-/Users/skkac/Work/SCAN/new_route/scripts/restart_qwen_vl_server.sh}"
APPLY_SCRIPT="${APPLY_SCRIPT:-/Users/skkac/Work/SCAN/new_route/scripts/apply_cross_candidate_merge_reviews.py}"
QA_SCRIPT="${QA_SCRIPT:-/Users/skkac/Work/SCAN/new_route/scripts/qa_reviewed_merge_results.py}"
LONG_OBJECTS="${LONG_OBJECTS:-/Users/skkac/Work/SCAN/server_frame_fine_long_assoc_v008/frame_fine_tracklet_long_assoc_0000_0999_v008_gap60_v2_samecand_loose/long_objects.jsonl}"
REMOTE_LONG_OBJECTS="${REMOTE_LONG_OBJECTS:-/root/epfs/frame_fine_tracklet_long_assoc_0000_0999_v008_gap60_v2_samecand_loose/long_objects.jsonl}"
QWEN_PORT="${QWEN_PORT:-8001}"
CONCURRENCY="${CONCURRENCY:-4}"
IMAGE_LONG_EDGE="${IMAGE_LONG_EDGE:-1280}"
MIN_CONFIDENCE="${MIN_CONFIDENCE:-0.75}"
MAX_TOKENS="${MAX_TOKENS:-1024}"

ssh_opts=()
if [[ -n "${BIND_ADDRESS}" ]]; then
  ssh_opts+=("-o" "BindAddress=${BIND_ADDRESS}")
fi

echo "[1/8] checking SSH connectivity: ${SERVER}"
ssh "${ssh_opts[@]}" -o ConnectTimeout=8 "${SERVER}" 'hostname; date'

echo "[2/8] checking remote GPUs"
ssh "${ssh_opts[@]}" "${SERVER}" 'nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader,nounits'

echo "[3/8] uploading scripts and review pack"
scp "${ssh_opts[@]}" "${REVIEW_SCRIPT}" "${SERVER}:/tmp/review_cross_candidate_merges_vlm.py"
scp "${ssh_opts[@]}" "${PROMPT_SCRIPT}" "${SERVER}:/tmp/vlm_scene_prompt.py"
scp "${ssh_opts[@]}" "${RESTART_QWEN}" "${SERVER}:/tmp/restart_qwen_vl_server.sh"
ssh "${ssh_opts[@]}" "${SERVER}" "rm -rf '${REMOTE_PACK}'"
scp -r "${ssh_opts[@]}" "${LOCAL_PACK}" "${SERVER}:${REMOTE_PACK}"

echo "[4/8] starting/checking Qwen VL server"
ssh "${ssh_opts[@]}" "${SERVER}" "chmod +x /tmp/restart_qwen_vl_server.sh /tmp/review_cross_candidate_merges_vlm.py; CUDA_VISIBLE_DEVICES=\${CUDA_VISIBLE_DEVICES:-1} LLAMA_SERVER=\${LLAMA_SERVER:-/root/epfs/llama-server/bin/llama-server} PARALLEL=${CONCURRENCY} PORT=${QWEN_PORT} LOG=/root/epfs/qwen_vl_server_${QWEN_PORT}_${SERVER}.log bash /tmp/restart_qwen_vl_server.sh"

echo "[5/8] running compact Qwen review"
ssh "${ssh_opts[@]}" "${SERVER}" "rm -rf '${REMOTE_PACK}/vlm_review_qwen_compact'; python3 /tmp/review_cross_candidate_merges_vlm.py --review-jsonl '${REMOTE_PACK}/cross_candidate_review_items.jsonl' --contact-sheet-dir '${REMOTE_PACK}/contact_sheets' --output-dir '${REMOTE_PACK}/vlm_review_qwen_compact' --endpoint 'http://127.0.0.1:${QWEN_PORT}/v1/chat/completions' --model Qwen3.6-35B-A3B-Q4_K_M --concurrency '${CONCURRENCY}' --timeout 240 --max-tokens '${MAX_TOKENS}' --temperature 0 --image-long-edge '${IMAGE_LONG_EDGE}' --resume"

echo "[6/8] pulling Qwen review results"
rm -rf "${LOCAL_OUTPUT}/vlm_review_qwen_compact"
scp -r "${ssh_opts[@]}" "${SERVER}:${REMOTE_PACK}/vlm_review_qwen_compact" "${LOCAL_OUTPUT}/"

echo "[7/8] applying high-confidence merge decisions locally"
LOCAL_REVIEW="${LOCAL_OUTPUT}/vlm_review_qwen_compact/vlm_merge_review_results.jsonl"
LOCAL_APPLIED="${LOCAL_OUTPUT}/vlm_review_qwen_compact_applied"
rm -rf "${LOCAL_APPLIED}"
python3 "${APPLY_SCRIPT}" \
  --objects "${LONG_OBJECTS}" \
  --reviews "${LOCAL_REVIEW}" \
  --output-dir "${LOCAL_APPLIED}" \
  --min-confidence "${MIN_CONFIDENCE}"

echo "[8/8] QA reviewed merge output"
python3 "${QA_SCRIPT}" \
  --input-objects "${LONG_OBJECTS}" \
  --output-objects "${LOCAL_APPLIED}/review_merged_long_objects.jsonl" \
  --decisions "${LOCAL_APPLIED}/review_merge_decisions.jsonl" \
  --output-report "${LOCAL_APPLIED}/qa_reviewed_merge_report.json"

echo "done"
echo "review results: ${LOCAL_REVIEW}"
echo "applied objects: ${LOCAL_APPLIED}/review_merged_long_objects.jsonl"
