#!/usr/bin/env bash
set -euo pipefail

SSH_HOST="${SSH_HOST:-10.0.8.114}"
SSH_PORT="${SSH_PORT:-31079}"
SSH_USER="${SSH_USER:-root}"
LOCAL_SCRIPT_DIR="${LOCAL_SCRIPT_DIR:-/Users/skkac/Work/SCAN/new_route/scripts}"
REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR:-/root/epfs/new_route_scripts}"
TAR_BIN="${TAR_BIN:-bsdtar}"

START="${START:-1000}"
END="${END:-1999}"
TRAIN_MANIFEST="${TRAIN_MANIFEST:-/root/epfs/new_route_stage1_skymask/semantic_manifest_${START}_${END}_ready_current.json}"
FULL_MANIFEST="${FULL_MANIFEST:-/root/epfs/new_route_stage1_skymask/semantic_manifest_${START}_${END}.json}"
READY_ALL="${READY_ALL:-/root/epfs/new_route_stage1_skymask/semantic_manifest_${START}_${END}_ready_all_for_vlm.json}"
VLM_MANIFEST="${VLM_MANIFEST:-/root/epfs/new_route_stage1_skymask/semantic_manifest_${START}_${END}_ready_vlm_extra.json}"
SAM_MASKS_DIR="${SAM_MASKS_DIR:-/root/epfs/new_route_stage1_skymask/sam_masks_${START}_${END}_combined}"
OUTPUT_DIR="${OUTPUT_DIR:-/root/epfs/manifold_3dgs_project/processed/semantic_eval_new_route_${START}_${END}}"
LOG_DIR="${LOG_DIR:-/root/epfs/new_route_stage1_skymask/logs}"
SESSION_NAME="${SESSION_NAME:-semantic_vlm_extra_${START}_${END}}"
MIN_SAM_AGE_SECONDS="${MIN_SAM_AGE_SECONDS:-30}"

ssh_target="${SSH_USER}@${SSH_HOST}"
ssh_opts=(-F /dev/null -p "${SSH_PORT}")

COPYFILE_DISABLE=1 "${TAR_BIN}" --no-xattrs -C "${LOCAL_SCRIPT_DIR}" --exclude='__pycache__' --exclude='._*' -cf - . \
  | ssh "${ssh_opts[@]}" "${ssh_target}" "mkdir -p '${REMOTE_SCRIPT_DIR}' && tar -C '${REMOTE_SCRIPT_DIR}' -xf - && chmod +x '${REMOTE_SCRIPT_DIR}'/*.sh"

ssh "${ssh_opts[@]}" "${ssh_target}" \
  START="${START}" \
  END="${END}" \
  TRAIN_MANIFEST="${TRAIN_MANIFEST}" \
  FULL_MANIFEST="${FULL_MANIFEST}" \
  READY_ALL="${READY_ALL}" \
  VLM_MANIFEST="${VLM_MANIFEST}" \
  SAM_MASKS_DIR="${SAM_MASKS_DIR}" \
  OUTPUT_DIR="${OUTPUT_DIR}" \
  LOG_DIR="${LOG_DIR}" \
  SESSION_NAME="${SESSION_NAME}" \
  MIN_SAM_AGE_SECONDS="${MIN_SAM_AGE_SECONDS}" \
  REMOTE_SCRIPT_DIR="${REMOTE_SCRIPT_DIR}" \
  'bash -s' <<'REMOTE'
set -euo pipefail
cd "${REMOTE_SCRIPT_DIR}"

python3 make_new_route_semantic_manifest.py --start "${START}" --end "${END}" --count 0 --output "${FULL_MANIFEST}" --require-sky-mask >"${LOG_DIR}/vlm_make_manifest.log"
python3 filter_semantic_manifest_ready.py \
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
ready["items"] = items
ready["filter_report"] = {
    **ready.get("filter_report", {}),
    "excluded_train_manifest": len(train_ids),
    "selected_vlm_extra": len(items),
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
echo "vlm_extra_count=${count}"
if [[ "${count}" -le 0 ]]; then
  exit 0
fi

pid_file="${LOG_DIR}/${SESSION_NAME}.pid"
old_pid="$(cat "${pid_file}" 2>/dev/null || true)"
if [[ -n "${old_pid}" ]] && kill -0 "${old_pid}" 2>/dev/null; then
  echo "already_running_pid=${old_pid}"
  exit 0
fi

nohup bash -lc "cd '${REMOTE_SCRIPT_DIR}' && MANIFEST='${VLM_MANIFEST}' OUTPUT_DIR='${OUTPUT_DIR}' SAM_MASKS_DIR=/root/epfs/new_route_stage1_skymask/sam_masks_${START}_${END}_vlm_extra_linked EXISTING_SAM_DIR='${SAM_MASKS_DIR}' PART0='${SAM_MASKS_DIR}' PART1='${SAM_MASKS_DIR}' WORK_DIR='${OUTPUT_DIR}/_sharded_work_vlm_extra' LOG_DIR='${OUTPUT_DIR}/_sharded_work_vlm_extra/logs' START_INDEX=0 END_INDEX='${count}' SHARDS=4 CHUNK_SIZE=4 MAX_TOKENS=4096 PATCH_SCENE_PROMPTS=1 bash ./run_server_semantic_completion_sharded.sh" >"${LOG_DIR}/${SESSION_NAME}.log" 2>&1 &
echo "$!" >"${pid_file}"
echo "started_pid=$(cat "${pid_file}")"
REMOTE
