#!/usr/bin/env bash
set -euo pipefail

SEMANTIC_ROOT="${SEMANTIC_ROOT:-/root/epfs/manifold_3dgs_project/semantic_eval}"
RUN_EVAL="${RUN_EVAL:-${SEMANTIC_ROOT}/run_eval.py}"
MERGE_SCRIPT="${MERGE_SCRIPT:-${SEMANTIC_ROOT}/merge_sam2_adjacency.py}"
REVIEW_SCRIPT="${REVIEW_SCRIPT:-${SEMANTIC_ROOT}/review_merged_labels_prompt_v2.py}"
COMPLETION_SCRIPT="${COMPLETION_SCRIPT:-${SEMANTIC_ROOT}/complete_unknown_regions.py}"

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
MAX_TOKENS="${MAX_TOKENS:-2048}"

mkdir -p "${SAM_MASKS_DIR}" "${OUTPUT_DIR}"
export MANIFEST SAM_MASKS_DIR EXISTING_SAM_DIR PART0 PART1

echo "[1/5] Combining SAM2 masks into ${SAM_MASKS_DIR}"
python3 - <<'PY'
import os
import shutil
from pathlib import Path

manifest = Path(os.environ["MANIFEST"])
out = Path(os.environ["SAM_MASKS_DIR"])
sources = [Path(os.environ["EXISTING_SAM_DIR"]), Path(os.environ["PART0"]), Path(os.environ["PART1"])]
out.mkdir(parents=True, exist_ok=True)

suffixes = ["_sam_masks.json", "_sam_masks.png", "_numbered.png", "_sam_done.flag"]
copied = 0
for src in sources:
    if not src.exists():
        continue
    for path in src.iterdir():
        if any(path.name.endswith(suffix) for suffix in suffixes):
            dst = out / path.name
            if not dst.exists() or path.stat().st_mtime > dst.stat().st_mtime:
                shutil.copy2(path, dst)
                copied += 1
print({"copied_or_refreshed": copied, "json_count": len(list(out.glob("*_sam_masks.json")))})
PY

echo "[2/5] Running sam2_qwen"
python3 "${RUN_EVAL}" \
  --manifest "${MANIFEST}" \
  --output-dir "${OUTPUT_DIR}" \
  --sam-masks-dir "${SAM_MASKS_DIR}" \
  --vlm-endpoint "${VLM_ENDPOINT}" \
  --vlm-model "${VLM_MODEL}" \
  --vlm-chunk-size "${CHUNK_SIZE}" \
  --vlm-max-tokens "${MAX_TOKENS}" \
  --start-index "${START_INDEX}" \
  --end-index "${END_INDEX}" \
  --combos sam2_qwen

echo "[3/5] Merging adjacent SAM2 labels"
python3 "${MERGE_SCRIPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --source-combo sam2_qwen \
  --output-combo sam2_sky_label_merge_qwen_review \
  --manifest "${MANIFEST}"

echo "[4/5] Reviewing merged labels with prompt v3"
python3 "${REVIEW_SCRIPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --source-combo sam2_sky_label_merge_qwen_review \
  --output-combo sam2_prompt_v3_sky_label_merge \
  --manifest "${MANIFEST}" \
  --vlm-endpoint "${VLM_ENDPOINT}" \
  --vlm-model "${VLM_MODEL}"

echo "[5/5] Completing unknown non-sky regions"
python3 "${COMPLETION_SCRIPT}" \
  --output-dir "${OUTPUT_DIR}" \
  --source-combo sam2_prompt_v3_sky_label_merge \
  --output-combo sam2_prompt_v3_sky_label_merge_completion \
  --manifest "${MANIFEST}" \
  --vlm-endpoint "${VLM_ENDPOINT}" \
  --vlm-model "${VLM_MODEL}"

echo "semantic completion output: ${OUTPUT_DIR}"
