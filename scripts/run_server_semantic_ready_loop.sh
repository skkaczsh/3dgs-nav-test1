#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START="${START:-1000}"
END="${END:-1999}"
COMBO="${COMBO:-sam2_prompt_v3_sky_label_merge_completion}"
FULL_MANIFEST="${FULL_MANIFEST:-/root/epfs/new_route_stage1_skymask/semantic_manifest_${START}_${END}.json}"
READY_MANIFEST="${READY_MANIFEST:-/root/epfs/new_route_stage1_skymask/semantic_manifest_${START}_${END}_ready_current.json}"
SAM_MASKS_DIR="${SAM_MASKS_DIR:-/root/epfs/new_route_stage1_skymask/sam_masks_${START}_${END}_combined}"
LINKED_SAM_DIR="${LINKED_SAM_DIR:-/root/epfs/new_route_stage1_skymask/sam_masks_${START}_${END}_ready_linked}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_${START}_${END}}"
LOG_DIR="${LOG_DIR:-/root/epfs/new_route_stage1_skymask/logs}"
SHARDS="${SHARDS:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
PATCH_SCENE_PROMPTS="${PATCH_SCENE_PROMPTS:-1}"
SLEEP_SECONDS="${SLEEP_SECONDS:-300}"
MAX_CYCLES="${MAX_CYCLES:-0}"
WAIT_SESSION="${WAIT_SESSION:-semantic_${START}_${END}_ready}"

mkdir -p "${LOG_DIR}"

count_items() {
  python3 - "$1" <<'PY'
import json
import sys
from pathlib import Path
path = Path(sys.argv[1])
print(len(json.loads(path.read_text(encoding="utf-8")).get("items", [])) if path.exists() else 0)
PY
}

count_completion() {
  find "${OUTPUT_DIR}/images" -path "*${COMBO}/semantic.png" 2>/dev/null | wc -l | tr -d ' '
}

while tmux has-session -t "${WAIT_SESSION}" 2>/dev/null; do
  echo "[wait] existing semantic session still running: ${WAIT_SESSION}"
  sleep 30
done

cycle=0
while true; do
  cycle=$((cycle + 1))
  python3 "${SCRIPT_DIR}/make_new_route_semantic_manifest.py" \
    --start "${START}" \
    --end "${END}" \
    --count 0 \
    --output "${FULL_MANIFEST}" \
    --require-sky-mask
  python3 "${SCRIPT_DIR}/filter_semantic_manifest_ready.py" \
    --manifest "${FULL_MANIFEST}" \
    --sam-masks-dir "${SAM_MASKS_DIR}" \
    --output "${READY_MANIFEST}" \
    --require-sky

  ready_count="$(count_items "${READY_MANIFEST}")"
  done_count="$(count_completion)"
  echo "[cycle ${cycle}] ready=${ready_count} completed=${done_count}"

  if [[ "${ready_count}" -gt "${done_count}" ]]; then
    MANIFEST="${READY_MANIFEST}" \
    OUTPUT_DIR="${OUTPUT_DIR}" \
    SAM_MASKS_DIR="${LINKED_SAM_DIR}" \
    EXISTING_SAM_DIR="${SAM_MASKS_DIR}" \
    PART0="${SAM_MASKS_DIR}" \
    PART1="${SAM_MASKS_DIR}" \
    START_INDEX=0 \
    END_INDEX="${ready_count}" \
    SHARDS="${SHARDS}" \
    CHUNK_SIZE="${CHUNK_SIZE}" \
    MAX_TOKENS="${MAX_TOKENS}" \
    PATCH_SCENE_PROMPTS="${PATCH_SCENE_PROMPTS}" \
    bash "${SCRIPT_DIR}/run_server_semantic_completion_sharded.sh"
  fi

  done_count="$(count_completion)"
  if [[ "${done_count}" -ge "$(((END - START + 1) * 3))" ]]; then
    echo "[done] all semantic images completed: ${done_count}"
    break
  fi
  if [[ "${MAX_CYCLES}" -gt 0 && "${cycle}" -ge "${MAX_CYCLES}" ]]; then
    echo "[stop] reached MAX_CYCLES=${MAX_CYCLES}"
    break
  fi
  sleep "${SLEEP_SECONDS}"
done
