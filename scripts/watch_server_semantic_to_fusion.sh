#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"

COMBO="${COMBO:-sam2_prompt_v3_sky_label_merge_completion}"
EXPECTED="${EXPECTED:-3000}"
POLL_SECONDS="${POLL_SECONDS:-300}"

STAGE_DIR="${STAGE_DIR:-/root/epfs/new_route_stage1_skymask}"
PROCESSED_DIR="${PROCESSED_DIR:-/root/epfs/manifold_3dgs_project/processed}"
MERGED_DIR="${MERGED_DIR:-${PROCESSED_DIR}/semantic_eval_new_route_0000_0999}"
FUSION_DIR="${FUSION_DIR:-${STAGE_DIR}/target_object_fusion_0000_0999}"

SPLIT_A_MANIFEST="${SPLIT_A_MANIFEST:-${STAGE_DIR}/semantic_manifest_ready_a_current.json}"
SPLIT_B_MANIFEST="${SPLIT_B_MANIFEST:-${STAGE_DIR}/semantic_manifest_ready_b_current.json}"
SPLIT_C_MANIFEST="${SPLIT_C_MANIFEST:-${STAGE_DIR}/semantic_manifest_ready_c_current.json}"
SPLIT_D_MANIFEST="${SPLIT_D_MANIFEST:-${STAGE_DIR}/semantic_manifest_final_d_missing.json}"

SPLIT_A_OUTPUT="${SPLIT_A_OUTPUT:-${PROCESSED_DIR}/semantic_eval_new_route_0000_0999_a}"
SPLIT_B_OUTPUT="${SPLIT_B_OUTPUT:-${PROCESSED_DIR}/semantic_eval_new_route_0000_0999_b}"
SPLIT_C_OUTPUT="${SPLIT_C_OUTPUT:-${PROCESSED_DIR}/semantic_eval_new_route_0000_0999_c}"
SPLIT_D_OUTPUT="${SPLIT_D_OUTPUT:-${PROCESSED_DIR}/semantic_eval_new_route_0000_0999_d}"

PROGRESS_JSON="${PROGRESS_JSON:-${STAGE_DIR}/semantic_splits_progress_launched.json}"
LOG_DIR="${LOG_DIR:-${STAGE_DIR}/logs}"
mkdir -p "${LOG_DIR}"

progress_count() {
  python3 "${SCRIPT_DIR}/qa_semantic_splits.py" \
    --split a "${SPLIT_A_MANIFEST}" "${SPLIT_A_OUTPUT}" \
    --split b "${SPLIT_B_MANIFEST}" "${SPLIT_B_OUTPUT}" \
    --split c "${SPLIT_C_MANIFEST}" "${SPLIT_C_OUTPUT}" \
    --split d "${SPLIT_D_MANIFEST}" "${SPLIT_D_OUTPUT}" \
    --output "${PROGRESS_JSON}" >/dev/null
  python3 - <<'PY' "${PROGRESS_JSON}" "${COMBO}"
import json
import sys
from pathlib import Path

data = json.loads(Path(sys.argv[1]).read_text(encoding="utf-8"))
combo = sys.argv[2]
row = data["summary"]["combos"].get(combo, {})
print(int(row.get("completed", 0)))
PY
}

while true; do
  count="$(progress_count)"
  echo "[$(date -Is)] ${COMBO} ${count}/${EXPECTED}"
  if [[ "${count}" -ge "${EXPECTED}" ]]; then
    break
  fi
  sleep "${POLL_SECONDS}"
done

echo "[$(date -Is)] merging semantic split outputs into ${MERGED_DIR}"
python3 "${SCRIPT_DIR}/merge_semantic_split_outputs.py" \
  --split a "${SPLIT_A_MANIFEST}" "${SPLIT_A_OUTPUT}" \
  --split b "${SPLIT_B_MANIFEST}" "${SPLIT_B_OUTPUT}" \
  --split c "${SPLIT_C_MANIFEST}" "${SPLIT_C_OUTPUT}" \
  --split d "${SPLIT_D_MANIFEST}" "${SPLIT_D_OUTPUT}" \
  --output-dir "${MERGED_DIR}" \
  --replace \
  --require-combo "${COMBO}" \
  > "${LOG_DIR}/merge_semantic_splits_0000_0999.log" 2>&1

echo "[$(date -Is)] running target/object fusion into ${FUSION_DIR}"
SEMANTIC_EVAL_DIR="${MERGED_DIR}" \
COMBO="${COMBO}" \
OUTPUT_DIR="${FUSION_DIR}" \
WORK_MODE=semantic-dir \
START_FRAME=0 \
END_FRAME=999 \
nohup "${SCRIPT_DIR}/run_server_target_object_fusion.sh" \
  > "${LOG_DIR}/target_object_fusion_0000_0999.log" 2>&1 &

echo "$!" > "${STAGE_DIR}/target_object_fusion_0000_0999.pid"
echo "[$(date -Is)] started target/object fusion pid $(cat "${STAGE_DIR}/target_object_fusion_0000_0999.pid")"
