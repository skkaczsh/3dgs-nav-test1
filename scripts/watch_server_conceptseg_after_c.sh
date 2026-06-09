#!/usr/bin/env bash
set -euo pipefail

STAGE_DIR="${STAGE_DIR:-/root/epfs/new_route_stage1_skymask}"
SCRIPTS_DIR="${SCRIPTS_DIR:-/root/epfs/new_route_scripts}"
PROCESSED_DIR="${PROCESSED_DIR:-/root/epfs/manifold_3dgs_project/processed}"
COMBO="${COMBO:-sam2_prompt_v3_sky_label_merge_completion}"
C_EXPECTED="${C_EXPECTED:-913}"
POLL_SECONDS="${POLL_SECONDS:-300}"
QWEN_PORT="${QWEN_PORT:-8003}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES:-1}"
LIMIT="${LIMIT:-8}"
LOG_DIR="${LOG_DIR:-${STAGE_DIR}/logs}"

mkdir -p "${LOG_DIR}"

C_MANIFEST="${C_MANIFEST:-${STAGE_DIR}/semantic_manifest_ready_c_current.json}"
C_OUTPUT="${C_OUTPUT:-${PROCESSED_DIR}/semantic_eval_new_route_0000_0999_c}"

count_c_completion() {
  python3 - <<'PY' "${C_MANIFEST}" "${C_OUTPUT}" "${COMBO}"
import json
import sys
from pathlib import Path

manifest = Path(sys.argv[1])
output = Path(sys.argv[2])
combo = sys.argv[3]
items = json.loads(manifest.read_text(encoding="utf-8")).get("items", [])
count = 0
for item in items:
    if (output / "images" / item["image_id"] / combo / "semantic.png").exists():
        count += 1
print(count)
PY
}

while [[ ! -f "${STAGE_DIR}/semantic_0000_0999_c.pid" ]]; do
  echo "[$(date -Is)] waiting for C pid file"
  sleep "${POLL_SECONDS}"
done

C_PID="$(cat "${STAGE_DIR}/semantic_0000_0999_c.pid")"
echo "[$(date -Is)] watching C pid ${C_PID}"
while kill -0 "${C_PID}" 2>/dev/null; do
  count="$(count_c_completion)"
  echo "[$(date -Is)] C ${COMBO} ${count}/${C_EXPECTED}"
  sleep "${POLL_SECONDS}"
done

count="$(count_c_completion)"
echo "[$(date -Is)] C pid exited; ${COMBO} ${count}/${C_EXPECTED}"
if [[ "${count}" -lt "${C_EXPECTED}" ]]; then
  echo "[$(date -Is)] C is incomplete; leaving Qwen ${QWEN_PORT} running for retry/debug"
  exit 1
fi

mapfile -t qwen_pids < <(pgrep -f "llama-server.*--port ${QWEN_PORT}" || true)
if [[ "${#qwen_pids[@]}" -gt 0 ]]; then
  echo "[$(date -Is)] stopping Qwen port ${QWEN_PORT}: ${qwen_pids[*]}"
  kill "${qwen_pids[@]}" || true
  sleep 10
fi

echo "[$(date -Is)] starting ConceptSeg-R1 smoke on GPU ${CUDA_VISIBLE_DEVICES}"
CUDA_VISIBLE_DEVICES="${CUDA_VISIBLE_DEVICES}" \
LIMIT="${LIMIT}" \
OUTPUT_DIR="${STAGE_DIR}/conceptseg_smoke_after_c" \
nohup "${SCRIPTS_DIR}/run_server_conceptseg_smoke.sh" \
  > "${LOG_DIR}/conceptseg_smoke_after_c.log" 2>&1 &

echo "$!" > "${STAGE_DIR}/conceptseg_smoke_after_c.pid"
echo "[$(date -Is)] started ConceptSeg smoke pid $(cat "${STAGE_DIR}/conceptseg_smoke_after_c.pid")"
