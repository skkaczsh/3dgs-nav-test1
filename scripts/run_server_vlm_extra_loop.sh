#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

START="${START:-1000}"
END="${END:-1999}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-/root/epfs/new_route_stage1_skymask/semantic_manifest_${START}_${END}_ready_current.json}"
FULL_MANIFEST="${FULL_MANIFEST:-/root/epfs/new_route_stage1_skymask/semantic_manifest_${START}_${END}.json}"
READY_ALL="${READY_ALL:-/root/epfs/new_route_stage1_skymask/semantic_manifest_${START}_${END}_ready_all_for_vlm.json}"
VLM_MANIFEST="${VLM_MANIFEST:-/root/epfs/new_route_stage1_skymask/semantic_manifest_${START}_${END}_ready_vlm_extra.json}"
SAM_MASKS_DIR="${SAM_MASKS_DIR:-/root/epfs/new_route_stage1_skymask/sam_masks_${START}_${END}_combined}"
LINKED_SAM_DIR="${LINKED_SAM_DIR:-/root/epfs/new_route_stage1_skymask/sam_masks_${START}_${END}_vlm_extra_linked}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_${START}_${END}}"
LOG_DIR="${LOG_DIR:-/root/epfs/new_route_stage1_skymask/logs}"
WORK_DIR="${WORK_DIR:-${OUTPUT_DIR}/_sharded_work_vlm_extra}"
RUN_LOG="${RUN_LOG:-${LOG_DIR}/semantic_vlm_extra_loop_${START}_${END}.log}"
PID_FILE="${PID_FILE:-${LOG_DIR}/semantic_vlm_extra_${START}_${END}.pid}"
MIN_SAM_AGE_SECONDS="${MIN_SAM_AGE_SECONDS:-30}"
SLEEP_SECONDS="${SLEEP_SECONDS:-180}"
MAX_CYCLES="${MAX_CYCLES:-0}"
MAX_ITEMS_PER_CYCLE="${MAX_ITEMS_PER_CYCLE:-24}"
SHARDS="${SHARDS:-4}"
CHUNK_SIZE="${CHUNK_SIZE:-4}"
MAX_TOKENS="${MAX_TOKENS:-4096}"
PATCH_SCENE_PROMPTS="${PATCH_SCENE_PROMPTS:-1}"

mkdir -p "${LOG_DIR}" "${WORK_DIR}"

count_completion() {
  find "${OUTPUT_DIR}/images" -path "*sam2_prompt_v3_sky_label_merge_completion/semantic.png" 2>/dev/null | wc -l | tr -d ' '
}

total_images=$(((END - START + 1) * 3))
cycle=0

while true; do
  cycle=$((cycle + 1))
  timestamp="$(date -u +%Y-%m-%dT%H:%M:%SZ)"
  echo "[${timestamp}] cycle=${cycle} start"

  python3 "${SCRIPT_DIR}/make_new_route_semantic_manifest.py" \
    --start "${START}" \
    --end "${END}" \
    --count 0 \
    --output "${FULL_MANIFEST}" \
    --require-sky-mask \
    >"${LOG_DIR}/vlm_make_manifest.log"

  python3 "${SCRIPT_DIR}/filter_semantic_manifest_ready.py" \
    --manifest "${FULL_MANIFEST}" \
    --sam-masks-dir "${SAM_MASKS_DIR}" \
    --output "${READY_ALL}" \
    --require-sky \
    --min-sam-age-seconds "${MIN_SAM_AGE_SECONDS}" \
    >"${LOG_DIR}/vlm_ready_all_filter.log"

  python3 - <<PY
import json
from pathlib import Path

ready = json.loads(Path("${READY_ALL}").read_text(encoding="utf-8"))
train = json.loads(Path("${TRAIN_MANIFEST}").read_text(encoding="utf-8")) if Path("${TRAIN_MANIFEST}").exists() else {"items": []}
train_ids = {x["image_id"] for x in train.get("items", [])}
items = []
invalid_sam = []
for item in ready.get("items", []):
    image_id = item["image_id"]
    if image_id in train_ids:
        continue
    if (Path("${OUTPUT_DIR}") / "images" / image_id / "sam2_prompt_v3_sky_label_merge_completion" / "semantic.png").exists():
        continue
    sam_path = Path("${SAM_MASKS_DIR}") / f"{image_id}_sam_masks.json"
    try:
        json.loads(sam_path.read_text(encoding="utf-8"))
    except Exception as exc:
        invalid_sam.append({"image_id": image_id, "path": str(sam_path), "error": repr(exc)})
        continue
    items.append(item)
    if ${MAX_ITEMS_PER_CYCLE} > 0 and len(items) >= ${MAX_ITEMS_PER_CYCLE}:
        break
ready["items"] = items
ready["filter_report"] = {
    **ready.get("filter_report", {}),
    "excluded_train_manifest": len(train_ids),
    "selected_vlm_extra": len(items),
    "max_items_per_cycle": ${MAX_ITEMS_PER_CYCLE},
    "invalid_sam_extra_candidates": len(invalid_sam),
    "invalid_sam_extra_samples": invalid_sam[:20],
}
Path("${VLM_MANIFEST}").write_text(json.dumps(ready, ensure_ascii=False, indent=2), encoding="utf-8")
print(len(items))
PY

  count="$(python3 - <<PY
import json
from pathlib import Path
print(len(json.loads(Path("${VLM_MANIFEST}").read_text(encoding="utf-8")).get("items", [])))
PY
)"
  done_count="$(count_completion)"
  echo "[cycle ${cycle}] vlm_extra_count=${count} completed=${done_count}/${total_images}"

  if [[ "${done_count}" -ge "${total_images}" ]]; then
    echo "[done] all semantic images completed"
    break
  fi

  if [[ "${count}" -gt 0 ]]; then
    echo "$$" >"${PID_FILE}"
    if ! (
      export MANIFEST="${VLM_MANIFEST}"
      export OUTPUT_DIR="${OUTPUT_DIR}"
      export SAM_MASKS_DIR="${LINKED_SAM_DIR}"
      export EXISTING_SAM_DIR="${SAM_MASKS_DIR}"
      export PART0="${SAM_MASKS_DIR}"
      export PART1="${SAM_MASKS_DIR}"
      export WORK_DIR="${WORK_DIR}"
      export LOG_DIR="${WORK_DIR}/logs"
      export START_INDEX=0
      export END_INDEX="${count}"
      export SHARDS="${SHARDS}"
      export CHUNK_SIZE="${CHUNK_SIZE}"
      export MAX_TOKENS="${MAX_TOKENS}"
      export PATCH_SCENE_PROMPTS="${PATCH_SCENE_PROMPTS}"
      bash "${SCRIPT_DIR}/run_server_semantic_completion_sharded.sh"
    ); then
      echo "[cycle ${cycle}] semantic extra failed; continuing after sleep"
    fi
  fi

  rm -f "${PID_FILE}"
  if [[ "${MAX_CYCLES}" -gt 0 && "${cycle}" -ge "${MAX_CYCLES}" ]]; then
    echo "[stop] reached MAX_CYCLES=${MAX_CYCLES}"
    break
  fi
  sleep "${SLEEP_SECONDS}"
done
