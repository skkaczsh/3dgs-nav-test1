#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

SEMANTIC_ROOT="${SEMANTIC_ROOT:-/root/epfs/manifold_3dgs_project/semantic_eval}"
RUN_EVAL="${RUN_EVAL:-${SEMANTIC_ROOT}/run_eval.py}"
MERGE_SCRIPT="${MERGE_SCRIPT:-${SEMANTIC_ROOT}/merge_sam2_adjacency.py}"
REVIEW_SCRIPT="${REVIEW_SCRIPT:-${SEMANTIC_ROOT}/review_merged_labels_prompt_v2.py}"
COMPLETION_SCRIPT="${COMPLETION_SCRIPT:-${SEMANTIC_ROOT}/complete_unknown_regions.py}"
PATCH_SCENE_PROMPTS="${PATCH_SCENE_PROMPTS:-1}"

MANIFEST="${MANIFEST:-/root/epfs/new_route_stage1_skymask/semantic_manifest_0000_0999.json}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_0000_0999}"
SAM_MASKS_DIR="${SAM_MASKS_DIR:-/root/epfs/new_route_stage1_skymask/sam_masks_0000_0999_combined}"
EXISTING_SAM_DIR="${EXISTING_SAM_DIR:-/root/epfs/manifold_3dgs_project/processed/sam_masks}"
PART0="${PART0:-/root/epfs/new_route_stage1_skymask/sam_masks_0000_0999_part0}"
PART1="${PART1:-/root/epfs/new_route_stage1_skymask/sam_masks_0000_0999_part1}"

VLM_ENDPOINT="${VLM_ENDPOINT:-http://localhost:8001/v1/chat/completions}"
VLM_MODEL="${VLM_MODEL:-Qwen3.6-35B-A3B-Q4_K_M}"
START_INDEX="${START_INDEX:-0}"
END_INDEX="${END_INDEX:-3000}"
CHUNK_SIZE="${CHUNK_SIZE:-10}"
MAX_TOKENS="${MAX_TOKENS:-1024}"
SHARDS="${SHARDS:-4}"
WORK_DIR="${WORK_DIR:-${OUTPUT_DIR}/_sharded_work}"
LOG_DIR="${LOG_DIR:-${WORK_DIR}/logs}"

mkdir -p "${SAM_MASKS_DIR}" "${OUTPUT_DIR}" "${WORK_DIR}" "${LOG_DIR}"

if [[ "${PATCH_SCENE_PROMPTS}" == "1" ]]; then
  echo "[0/5] Patching semantic_eval prompts with rooftop scene constraints"
  python3 "${SCRIPT_DIR}/patch_semantic_eval_scene_prompts.py" \
    --semantic-root "${SEMANTIC_ROOT}" \
    --report "${LOG_DIR}/scene_prompt_patch_report.json"
fi

run_shards() {
  local stage="$1"
  local missing_manifest="$2"
  shift 2
  local shard_dir="${WORK_DIR}/${stage}_shards"
  rm -rf "${shard_dir}"
  mkdir -p "${shard_dir}"
  python3 "${SCRIPT_DIR}/split_semantic_manifest.py" \
    --manifest "${missing_manifest}" \
    --output-dir "${shard_dir}" \
    --shards "${SHARDS}" \
    --prefix "${stage}" \
    > "${LOG_DIR}/${stage}_split.json"

  local pids=()
  for shard_manifest in "${shard_dir}"/"${stage}"_*.json; do
    local shard_name
    shard_name="$(basename "${shard_manifest}" .json)"
    python3 "$@" --manifest "${shard_manifest}" > "${LOG_DIR}/${shard_name}.log" 2>&1 &
    pids+=("$!")
  done

  local failed=0
  for pid in "${pids[@]}"; do
    if ! wait "${pid}"; then
      failed=1
    fi
  done
  if [[ "${failed}" -ne 0 ]]; then
    echo "stage ${stage} failed; inspect ${LOG_DIR}" >&2
    exit 1
  fi
}

echo "[1/5] Linking SAM2 masks into ${SAM_MASKS_DIR}"
python3 "${SCRIPT_DIR}/link_sam_masks_by_manifest.py" \
  --manifest "${MANIFEST}" \
  --output-dir "${SAM_MASKS_DIR}" \
  --source-dir "${EXISTING_SAM_DIR}" \
  --source-dir "${PART0}" \
  --source-dir "${PART1}" \
  --start-index "${START_INDEX}" \
  --end-index "${END_INDEX}" \
  --report "${SAM_MASKS_DIR}/link_report_${START_INDEX}_${END_INDEX}.json"

BASE_MANIFEST="${WORK_DIR}/base_manifest_${START_INDEX}_${END_INDEX}.json"
python3 - <<'PY' "${MANIFEST}" "${BASE_MANIFEST}" "${START_INDEX}" "${END_INDEX}"
import json
import sys
from pathlib import Path

src = Path(sys.argv[1])
dst = Path(sys.argv[2])
start = int(sys.argv[3])
end = int(sys.argv[4])
items = json.loads(src.read_text(encoding="utf-8")).get("items", [])[start:end]
dst.parent.mkdir(parents=True, exist_ok=True)
dst.write_text(json.dumps({"items": items}, ensure_ascii=False, indent=2), encoding="utf-8")
print(f"wrote={dst} items={len(items)}")
PY

echo "[2/5] Running missing sam2_qwen with ${SHARDS} shards"
SAM2_MISSING="${WORK_DIR}/missing_sam2_qwen.json"
python3 "${SCRIPT_DIR}/filter_missing_semantic_manifest.py" \
  --manifest "${BASE_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --combo sam2_qwen \
  --output "${SAM2_MISSING}" \
  > "${LOG_DIR}/missing_sam2_qwen.json"
run_shards sam2_qwen "${SAM2_MISSING}" "${RUN_EVAL}" \
  --output-dir "${OUTPUT_DIR}" \
  --sam-masks-dir "${SAM_MASKS_DIR}" \
  --vlm-endpoint "${VLM_ENDPOINT}" \
  --vlm-model "${VLM_MODEL}" \
  --vlm-chunk-size "${CHUNK_SIZE}" \
  --vlm-max-tokens "${MAX_TOKENS}" \
  --combos sam2_qwen

echo "[3/5] Merging missing adjacent SAM2 labels with ${SHARDS} shards"
MERGE_MISSING="${WORK_DIR}/missing_sam2_sky_label_merge_qwen_review.json"
python3 "${SCRIPT_DIR}/filter_missing_semantic_manifest.py" \
  --manifest "${BASE_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --combo sam2_sky_label_merge_qwen_review \
  --require-source-combo sam2_qwen \
  --output "${MERGE_MISSING}" \
  > "${LOG_DIR}/missing_merge.json"
run_shards merge "${MERGE_MISSING}" "${MERGE_SCRIPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --source-combo sam2_qwen \
  --output-combo sam2_sky_label_merge_qwen_review

echo "[4/5] Reviewing missing merged labels with prompt v3 using ${SHARDS} shards"
REVIEW_MISSING="${WORK_DIR}/missing_sam2_prompt_v3_sky_label_merge.json"
python3 "${SCRIPT_DIR}/filter_missing_semantic_manifest.py" \
  --manifest "${BASE_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --combo sam2_prompt_v3_sky_label_merge \
  --require-source-combo sam2_sky_label_merge_qwen_review \
  --output "${REVIEW_MISSING}" \
  > "${LOG_DIR}/missing_review.json"
run_shards review "${REVIEW_MISSING}" "${REVIEW_SCRIPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --source-combo sam2_sky_label_merge_qwen_review \
  --output-combo sam2_prompt_v3_sky_label_merge \
  --vlm-endpoint "${VLM_ENDPOINT}" \
  --vlm-model "${VLM_MODEL}" \
  --vlm-max-tokens "${MAX_TOKENS}"

echo "[5/5] Completing missing unknown non-sky regions with ${SHARDS} shards"
COMPLETION_MISSING="${WORK_DIR}/missing_sam2_prompt_v3_sky_label_merge_completion.json"
python3 "${SCRIPT_DIR}/filter_missing_semantic_manifest.py" \
  --manifest "${BASE_MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --combo sam2_prompt_v3_sky_label_merge_completion \
  --require-source-combo sam2_prompt_v3_sky_label_merge \
  --output "${COMPLETION_MISSING}" \
  > "${LOG_DIR}/missing_completion.json"
run_shards completion "${COMPLETION_MISSING}" "${COMPLETION_SCRIPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --source-combo sam2_prompt_v3_sky_label_merge \
  --output-combo sam2_prompt_v3_sky_label_merge_completion \
  --vlm-endpoint "${VLM_ENDPOINT}" \
  --vlm-model "${VLM_MODEL}" \
  --vlm-max-tokens "${MAX_TOKENS}"

echo "semantic completion output: ${OUTPUT_DIR}"
